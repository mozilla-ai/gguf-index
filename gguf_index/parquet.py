"""Parquet import/export utilities for GGUF Index."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from .storage import GGUFEntry

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable


def _get_repos_path(data_path: Path) -> Path:
    """Get the repos cache file path derived from the data file path."""
    return data_path.with_suffix(".repos.parquet")


# Schema for gguf_files parquet export
GGUF_SCHEMA = pa.schema([
    ("sha256", pa.string()),
    ("repo_id", pa.string()),
    ("revision", pa.string()),
    ("filename", pa.string()),
    ("size", pa.int64()),
    ("indexed_at", pa.string()),
])


def export_from_sqlite_streaming(
    conn: "sqlite3.Connection",
    path: Path,
    batch_size: int = 50000,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> int:
    """Stream export from SQLite to parquet without loading all data into memory.

    Uses pandas read_sql with chunking and PyArrow ParquetWriter for
    incremental writing. Maintains Xet CDC optimization.

    Args:
        conn: SQLite connection
        path: Output parquet file path
        batch_size: Number of rows per batch
        progress_callback: Optional callback(exported_count, total_count)

    Returns:
        Number of entries exported
    """
    # Get total count for progress
    total = conn.execute("SELECT COUNT(*) FROM gguf_files").fetchone()[0]

    if total == 0:
        # Create empty parquet with correct schema
        empty_table = pa.Table.from_pydict(
            {col: [] for col in ["sha256", "repo_id", "revision", "filename", "size", "indexed_at"]},
            schema=GGUF_SCHEMA,
        )
        pq.write_table(empty_table, path, use_content_defined_chunking=True, write_page_index=True)
        return 0

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Query for streaming read - ORDER BY primary key for consistent chunk boundaries
    # This ensures Xet CDC deduplication works properly on incremental updates
    query = """
        SELECT sha256, repo_id, revision, filename, size, indexed_at
        FROM gguf_files
        ORDER BY repo_id, revision, filename
    """

    writer = None
    exported = 0

    try:
        for chunk_df in pd.read_sql_query(query, conn, chunksize=batch_size):
            # Convert to PyArrow table
            table = pa.Table.from_pandas(chunk_df, schema=GGUF_SCHEMA, preserve_index=False)

            if writer is None:
                # Initialize writer with first chunk
                writer = pq.ParquetWriter(
                    path,
                    schema=GGUF_SCHEMA,
                    use_content_defined_chunking=True,
                    write_page_index=True,
                )

            writer.write_table(table)
            exported += len(chunk_df)

            if progress_callback:
                progress_callback(exported, total)

    finally:
        if writer is not None:
            writer.close()

    return exported


def export_to_parquet(
    entries: list[GGUFEntry],
    path: Path,
    repos: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Export entries and optionally repos cache to parquet.

    Args:
        entries: List of GGUFEntry objects to export
        path: Output path for the parquet file
        repos: Optional list of repo cache dicts to export

    Returns:
        Path to repos file if exported, None otherwise
    """
    if not entries:
        # Create empty DataFrame with correct schema
        df = pd.DataFrame(columns=["sha256", "repo_id", "revision", "filename", "size", "indexed_at"])
    else:
        df = pd.DataFrame([e.to_dict() for e in entries])

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Export with Xet optimization (content-defined chunking for better deduplication)
    # and page index for fast lookups
    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        use_content_defined_chunking=True,
        write_page_index=True,
    )

    # Export repos cache if provided
    repos_path = None
    if repos is not None:
        repos_path = _get_repos_path(path)
        if repos:
            repos_df = pd.DataFrame(repos)
        else:
            repos_df = pd.DataFrame(columns=["repo_id", "last_indexed_commit", "last_indexed_at", "max_revisions_indexed"])
        repos_df.to_parquet(
            repos_path,
            index=False,
            engine="pyarrow",
        )

    return repos_path


def import_from_parquet(path: Path) -> Iterator[GGUFEntry]:
    """Import entries from parquet file.

    Args:
        path: Path to the parquet file

    Yields:
        GGUFEntry objects
    """
    df = pd.read_parquet(path, engine="pyarrow")

    for _, row in df.iterrows():
        yield GGUFEntry(
            sha256=row["sha256"],
            repo_id=row["repo_id"],
            revision=row["revision"],
            filename=row["filename"],
            size=int(row["size"]),
            indexed_at=row.get("indexed_at"),
        )


def iter_parquet_batches(
    path: Path,
    batch_size: int = 10000,
) -> Iterator[tuple[list[tuple], int]]:
    """Stream parquet file in batches for bulk import.

    Memory-efficient: only one batch is loaded at a time.
    Returns raw tuples ready for SQL insertion (no GGUFEntry overhead).

    Args:
        path: Path to the parquet file
        batch_size: Number of rows per batch

    Yields:
        Tuple of (batch_of_tuples, total_rows) where each tuple is
        (repo_id, revision, filename, sha256, size, indexed_at)
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert to pandas for easier row access
        df = batch.to_pandas()

        rows = [
            (
                row["repo_id"],
                row["revision"],
                row["filename"],
                row["sha256"],
                int(row["size"]),
                row.get("indexed_at"),
            )
            for _, row in df.iterrows()
        ]

        yield rows, total_rows


def import_repos_from_parquet(path: Path) -> list[dict[str, Any]]:
    """Import repos cache from parquet file.

    Args:
        path: Path to the repos parquet file

    Returns:
        List of repo cache dicts
    """
    repos_path = _get_repos_path(path) if not path.name.endswith(".repos.parquet") else path

    if not repos_path.exists():
        return []

    df = pd.read_parquet(repos_path, engine="pyarrow")

    repos = []
    for _, row in df.iterrows():
        repos.append({
            "repo_id": row["repo_id"],
            "last_indexed_commit": row["last_indexed_commit"],
            "last_indexed_at": row["last_indexed_at"],
            "max_revisions_indexed": row.get("max_revisions_indexed"),
        })
    return repos


def download_from_hf(
    repo: str = "mozilla-ai/gguf-index",
    filename: str = "data.parquet",
    token: str | None = None,
) -> tuple[Path, Path | None]:
    """Download parquet files from HF dataset repo.

    Args:
        repo: HuggingFace dataset repository ID
        filename: Name of the parquet file in the repo
        token: Optional HuggingFace token for private repos

    Returns:
        Tuple of (data_path, repos_path) where repos_path may be None if not found
    """
    data_path = Path(hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        token=token,
    ))

    # Try to download repos cache file
    repos_filename = filename.replace(".parquet", ".repos.parquet")
    repos_path = None
    try:
        repos_path = Path(hf_hub_download(
            repo_id=repo,
            filename=repos_filename,
            repo_type="dataset",
            token=token,
        ))
    except Exception:
        pass  # Repos file is optional

    return data_path, repos_path


def push_to_hf(
    path: Path,
    repo: str,
    filename: str = "data.parquet",
    token: str | None = None,
    commit_message: str = "Update gguf-index",
) -> str:
    """Push parquet files to HF dataset repo.

    Uploads both the data file and repos cache file (if exists) in a single commit.

    Args:
        path: Path to the local parquet file
        repo: HuggingFace dataset repository ID
        filename: Name for the file in the repo
        token: HuggingFace token with write access
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded data file
    """
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

    # Build list of operations for a single commit
    operations = [
        CommitOperationAdd(path_in_repo=filename, path_or_fileobj=str(path)),
    ]

    # Add repos cache file if it exists
    repos_path = _get_repos_path(path)
    if repos_path.exists():
        repos_filename = filename.replace(".parquet", ".repos.parquet")
        operations.append(
            CommitOperationAdd(path_in_repo=repos_filename, path_or_fileobj=str(repos_path))
        )

    # Upload all files in a single commit
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
    )

    return f"https://huggingface.co/datasets/{repo}/blob/main/{filename}"
