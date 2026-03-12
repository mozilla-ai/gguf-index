"""Parquet import/export utilities for GGUF Index."""

from pathlib import Path
from typing import Iterator

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from .storage import GGUFEntry


def export_to_parquet(entries: list[GGUFEntry], path: Path) -> None:
    """Export entries to parquet with Xet optimization.

    Args:
        entries: List of GGUFEntry objects to export
        path: Output path for the parquet file
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


def download_from_hf(
    repo: str = "mozilla-ai/gguf-index",
    filename: str = "data.parquet",
    token: str | None = None,
) -> Path:
    """Download parquet from HF dataset repo.

    Args:
        repo: HuggingFace dataset repository ID
        filename: Name of the parquet file in the repo
        token: Optional HuggingFace token for private repos

    Returns:
        Path to the downloaded parquet file
    """
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    return Path(path)


def push_to_hf(
    path: Path,
    repo: str,
    filename: str = "data.parquet",
    token: str | None = None,
    commit_message: str = "Update gguf-index",
) -> str:
    """Push parquet file to HF dataset repo.

    Args:
        path: Path to the local parquet file
        repo: HuggingFace dataset repository ID
        filename: Name for the file in the repo
        token: HuggingFace token with write access
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded file
    """
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

    # Upload the file
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=filename,
        repo_id=repo,
        repo_type="dataset",
        commit_message=commit_message,
    )

    return f"https://huggingface.co/datasets/{repo}/blob/main/{filename}"
