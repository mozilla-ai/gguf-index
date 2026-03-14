"""Storage backends for GGUF index (JSON and SQLite)."""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class GGUFEntry:
    """Represents a single GGUF file entry in the index."""

    def __init__(
        self,
        sha256: str,
        repo_id: str,
        revision: str,
        filename: str,
        size: int,
        indexed_at: str | None = None,
    ):
        self.sha256 = sha256
        self.repo_id = repo_id
        self.revision = revision  # commit hash
        self.filename = filename
        self.size = size
        self.indexed_at = indexed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @property
    def url(self) -> str:
        """Get the HuggingFace blob URL."""
        return f"https://huggingface.co/{self.repo_id}/blob/{self.revision}/{self.filename}"

    @property
    def download_url(self) -> str:
        """Get the HuggingFace download URL."""
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.filename}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sha256": self.sha256,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "filename": self.filename,
            "size": self.size,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GGUFEntry":
        """Create from dictionary representation."""
        return cls(
            sha256=data["sha256"],
            repo_id=data["repo_id"],
            revision=data["revision"],
            filename=data["filename"],
            size=data["size"],
            indexed_at=data.get("indexed_at"),
        )


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def add(self, entry: GGUFEntry) -> None:
        """Add an entry to the index."""
        pass

    @abstractmethod
    def get(self, sha256: str) -> list[GGUFEntry]:
        """Get all entries matching a SHA256 hash."""
        pass

    @abstractmethod
    def get_all(self) -> list[GGUFEntry]:
        """Get all entries in the index."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of entries in the index."""
        pass

    @abstractmethod
    def count_unique_hashes(self) -> int:
        """Get the number of unique SHA256 hashes."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist the index to storage."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the index from storage."""
        pass


class JSONStorage(StorageBackend):
    """JSON file storage backend.

    Storage format (v4 - revision-aware):
    {
        "repo_id/revision/filename": {
            "sha256": "...",
            "repo_id": "...",
            "revision": "...",
            "filename": "...",
            "size": 123456,
            "indexed_at": "..."
        }
    }

    Primary key is (repo_id, revision, filename) - each unique combination
    maps to exactly one SHA256.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # Map of "repo_id/revision/filename" -> entry dict
        self._data: dict[str, dict[str, Any]] = {}

    def _make_key(self, repo_id: str, revision: str, filename: str) -> str:
        """Create a unique key from repo_id, revision, and filename."""
        return f"{repo_id}/{revision}/{filename}"

    def add(self, entry: GGUFEntry) -> None:
        key = self._make_key(entry.repo_id, entry.revision, entry.filename)
        self._data[key] = entry.to_dict()

    def get(self, sha256: str) -> list[GGUFEntry]:
        """Get all entries matching a SHA256 hash."""
        entries = []
        for data in self._data.values():
            if data["sha256"] == sha256:
                entries.append(GGUFEntry.from_dict(data))
        return entries

    def get_all(self) -> list[GGUFEntry]:
        return [GGUFEntry.from_dict(data) for data in self._data.values()]

    def count(self) -> int:
        return len(self._data)

    def count_unique_hashes(self) -> int:
        return len({data["sha256"] for data in self._data.values()})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        with open(self.path) as f:
            raw_data = json.load(f)

        # Only accept v4 format (has revision field)
        self._data = {}
        for key, data in raw_data.items():
            if "revision" in data:
                self._data[key] = data
            # Skip old format entries - they'll be re-indexed

    def export(self) -> dict[str, Any]:
        """Export the full index as a dictionary."""
        return self._data.copy()


class SchemaMigrationNeeded(Exception):
    """Raised when database schema needs migration."""

    def __init__(self, current_version: int, required_version: int, db_path: Path):
        self.current_version = current_version
        self.required_version = required_version
        self.db_path = db_path
        super().__init__(
            f"Database schema migration needed: v{current_version} -> v{required_version}"
        )


class SQLiteStorage(StorageBackend):
    """SQLite storage backend for fast lookups.

    Schema v5: (repo_id, revision, filename) as primary key.
    Each unique (repo_id, revision, filename) maps to exactly one SHA256.
    Added repos table for caching indexed repositories.
    """

    SCHEMA_VERSION = 5

    def __init__(self, path: str | Path, auto_migrate: bool = False):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._auto_migrate = auto_migrate

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def check_schema_version(self) -> tuple[int | None, bool]:
        """Check schema version without modifying the database.

        Returns:
            Tuple of (current_version, needs_migration).
            current_version is None if database doesn't exist or has no schema.
        """
        if not self.path.exists():
            return None, False

        conn = sqlite3.connect(self.path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if not cursor.fetchone():
                return None, True  # Old DB without version table

            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            return version, version != self.SCHEMA_VERSION
        finally:
            conn.close()

    def backup_and_recreate(self) -> Path:
        """Backup the existing database and create a fresh one.

        Returns:
            Path to the backup file.
        """
        if self._conn:
            self._conn.close()
            self._conn = None

        # Find a unique backup name
        version, _ = self.check_schema_version()
        version_str = f".v{version}" if version else ""
        backup_path = self.path.with_suffix(f"{version_str}.backup")
        counter = 1
        while backup_path.exists():
            backup_path = self.path.with_suffix(f"{version_str}.backup.{counter}")
            counter += 1

        # Rename old file to backup
        self.path.rename(backup_path)

        return backup_path

    def _init_schema(self) -> None:
        """Initialize or verify the database schema."""
        conn = self._conn

        # Check if schema_version table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
        if cursor.fetchone():
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            if version == self.SCHEMA_VERSION:
                return  # Already at current version

            # Old version - need migration
            if not self._auto_migrate:
                raise SchemaMigrationNeeded(version, self.SCHEMA_VERSION, self.path)

            # Auto-migrate by dropping and recreating (legacy behavior)
            conn.execute("DROP TABLE IF EXISTS gguf_files")
            conn.execute("DROP TABLE IF EXISTS repos")
            conn.execute("DROP TABLE IF EXISTS schema_version")

        # Create fresh schema
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            )
        """)
        conn.execute("INSERT INTO schema_version VALUES (?)", (self.SCHEMA_VERSION,))

        conn.execute("""
            CREATE TABLE gguf_files (
                repo_id TEXT NOT NULL,
                revision TEXT NOT NULL,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size INTEGER NOT NULL,
                indexed_at TEXT NOT NULL,
                PRIMARY KEY (repo_id, revision, filename)
            )
        """)
        conn.execute("CREATE INDEX idx_sha256 ON gguf_files(sha256)")
        conn.execute("CREATE INDEX idx_repo_id ON gguf_files(repo_id)")

        # Repo cache table for tracking indexed repositories
        conn.execute("""
            CREATE TABLE repos (
                repo_id TEXT PRIMARY KEY,
                last_indexed_commit TEXT NOT NULL,
                last_indexed_at TEXT NOT NULL,
                max_revisions_indexed INTEGER
            )
        """)
        conn.commit()

    def add(self, entry: GGUFEntry) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO gguf_files (repo_id, revision, filename, sha256, size, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entry.repo_id, entry.revision, entry.filename, entry.sha256, entry.size, entry.indexed_at),
        )
        conn.commit()  # Commit immediately so data survives interrupts

    def get(self, sha256: str) -> list[GGUFEntry]:
        rows = self._get_conn().execute(
            "SELECT * FROM gguf_files WHERE sha256 = ?", (sha256,)
        ).fetchall()
        return [
            GGUFEntry(
                sha256=row["sha256"],
                repo_id=row["repo_id"],
                revision=row["revision"],
                filename=row["filename"],
                size=row["size"],
                indexed_at=row["indexed_at"],
            )
            for row in rows
        ]

    def get_all(self) -> list[GGUFEntry]:
        rows = self._get_conn().execute("SELECT * FROM gguf_files").fetchall()
        return [
            GGUFEntry(
                sha256=row["sha256"],
                repo_id=row["repo_id"],
                revision=row["revision"],
                filename=row["filename"],
                size=row["size"],
                indexed_at=row["indexed_at"],
            )
            for row in rows
        ]

    def count(self) -> int:
        row = self._get_conn().execute("SELECT COUNT(*) FROM gguf_files").fetchone()
        return row[0] if row else 0

    def count_unique_hashes(self) -> int:
        row = self._get_conn().execute("SELECT COUNT(DISTINCT sha256) FROM gguf_files").fetchone()
        return row[0] if row else 0

    def save(self) -> None:
        if self._conn:
            self._conn.commit()

    def load(self) -> None:
        # SQLite loads on demand, just ensure connection is ready
        self._get_conn()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_repo_cache(self, repo_id: str) -> dict[str, Any] | None:
        """Get cached repo info.

        Returns:
            Dict with last_indexed_commit, last_indexed_at, max_revisions_indexed
            or None if not cached.
        """
        row = self._get_conn().execute(
            "SELECT last_indexed_commit, last_indexed_at, max_revisions_indexed FROM repos WHERE repo_id = ?",
            (repo_id,),
        ).fetchone()
        if row:
            return {
                "last_indexed_commit": row["last_indexed_commit"],
                "last_indexed_at": row["last_indexed_at"],
                "max_revisions_indexed": row["max_revisions_indexed"],
            }
        return None

    def set_repo_cache(self, repo_id: str, commit: str, max_revisions: int | None) -> None:
        """Update cache after indexing a repository."""
        conn = self._get_conn()
        indexed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        conn.execute(
            """
            INSERT OR REPLACE INTO repos (repo_id, last_indexed_commit, last_indexed_at, max_revisions_indexed)
            VALUES (?, ?, ?, ?)
            """,
            (repo_id, commit, indexed_at, max_revisions),
        )
        conn.commit()

    def clear_repo_cache(self, repo_id: str | None = None) -> None:
        """Clear cache (all or specific repo)."""
        conn = self._get_conn()
        if repo_id:
            conn.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
        else:
            conn.execute("DELETE FROM repos")
        conn.commit()

    def clear_all(self) -> None:
        """Clear all data including cache."""
        conn = self._get_conn()
        conn.execute("DELETE FROM gguf_files")
        conn.execute("DELETE FROM repos")
        conn.commit()

    def get_all_repo_cache(self) -> list[dict[str, Any]]:
        """Get all cached repo info.

        Returns:
            List of dicts with repo_id, last_indexed_commit, last_indexed_at, max_revisions_indexed
        """
        rows = self._get_conn().execute(
            "SELECT repo_id, last_indexed_commit, last_indexed_at, max_revisions_indexed FROM repos"
        ).fetchall()
        return [
            {
                "repo_id": row["repo_id"],
                "last_indexed_commit": row["last_indexed_commit"],
                "last_indexed_at": row["last_indexed_at"],
                "max_revisions_indexed": row["max_revisions_indexed"],
            }
            for row in rows
        ]

    def import_repo_cache(self, repos: list[dict[str, Any]]) -> int:
        """Import repo cache entries.

        Args:
            repos: List of dicts with repo_id, last_indexed_commit, last_indexed_at, max_revisions_indexed

        Returns:
            Number of repos imported
        """
        conn = self._get_conn()
        count = 0
        for repo in repos:
            conn.execute(
                """
                INSERT OR REPLACE INTO repos (repo_id, last_indexed_commit, last_indexed_at, max_revisions_indexed)
                VALUES (?, ?, ?, ?)
                """,
                (repo["repo_id"], repo["last_indexed_commit"], repo["last_indexed_at"], repo.get("max_revisions_indexed")),
            )
            count += 1
        conn.commit()
        return count

    def bulk_import(
        self,
        entries: list[GGUFEntry],
        progress_callback: "Callable[[int], None] | None" = None,
        batch_size: int = 10000,
    ) -> int:
        """Bulk import entries with optimizations for speed.

        Uses batched inserts, deferred indexing, and relaxed durability
        settings for maximum import speed.

        Args:
            entries: List of GGUFEntry objects to import
            progress_callback: Optional callback(count) called after each batch
            batch_size: Number of entries per batch

        Returns:
            Number of entries imported
        """
        conn = self._get_conn()

        # Optimize for bulk import
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")

        # Drop indexes for faster inserts
        conn.execute("DROP INDEX IF EXISTS idx_sha256")
        conn.execute("DROP INDEX IF EXISTS idx_repo_id")

        try:
            # Prepare data as tuples for executemany
            count = 0
            batch = []

            for entry in entries:
                batch.append((
                    entry.repo_id,
                    entry.revision,
                    entry.filename,
                    entry.sha256,
                    entry.size,
                    entry.indexed_at,
                ))

                if len(batch) >= batch_size:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO gguf_files
                        (repo_id, revision, filename, sha256, size, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    conn.commit()
                    count += len(batch)
                    if progress_callback:
                        progress_callback(count)
                    batch = []

            # Insert remaining entries
            if batch:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO gguf_files
                    (repo_id, revision, filename, sha256, size, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                count += len(batch)
                if progress_callback:
                    progress_callback(count)

        finally:
            # Recreate indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sha256 ON gguf_files(sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_id ON gguf_files(repo_id)")

            # Restore safe settings
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.commit()

        return count

    def bulk_import_batches(
        self,
        batch_iterator: "Iterator[tuple[list[tuple], int]]",
        progress_callback: "Callable[[int, int], None] | None" = None,
    ) -> int:
        """Streaming bulk import from pre-batched tuples.

        Memory-efficient: processes one batch at a time without loading
        all data into memory.

        Args:
            batch_iterator: Iterator yielding (batch_of_tuples, total_rows).
                Each tuple is (repo_id, revision, filename, sha256, size, indexed_at)
            progress_callback: Optional callback(imported_count, total_count)

        Returns:
            Number of entries imported
        """
        conn = self._get_conn()

        # Optimize for bulk import
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")

        # Drop indexes for faster inserts
        conn.execute("DROP INDEX IF EXISTS idx_sha256")
        conn.execute("DROP INDEX IF EXISTS idx_repo_id")

        try:
            count = 0

            for batch, total in batch_iterator:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO gguf_files
                    (repo_id, revision, filename, sha256, size, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                count += len(batch)
                if progress_callback:
                    progress_callback(count, total)

        finally:
            # Recreate indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sha256 ON gguf_files(sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_id ON gguf_files(repo_id)")

            # Restore safe settings
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.commit()

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics using efficient SQL aggregation."""
        conn = self._get_conn()

        # Basic counts
        total = conn.execute("SELECT COUNT(*) FROM gguf_files").fetchone()[0]
        unique_files = conn.execute("SELECT COUNT(DISTINCT sha256) FROM gguf_files").fetchone()[0]
        unique_repos = conn.execute("SELECT COUNT(DISTINCT repo_id) FROM gguf_files").fetchone()[0]
        cached_repos = conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0]

        # Total size (sum of sizes for unique hashes only)
        total_size = conn.execute("""
            SELECT COALESCE(SUM(size), 0) FROM (
                SELECT size FROM gguf_files GROUP BY sha256
            )
        """).fetchone()[0]

        # Per-repo counts (sorted by count descending)
        repo_counts = conn.execute("""
            SELECT repo_id, COUNT(*) as count
            FROM gguf_files
            GROUP BY repo_id
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_sources": total,
            "unique_files": unique_files,
            "total_size": total_size,
            "unique_repos": unique_repos,
            "cached_repos": cached_repos,
            "repos": [(row[0], row[1]) for row in repo_counts],
        }
