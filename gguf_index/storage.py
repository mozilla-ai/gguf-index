"""Storage backends for GGUF index (JSON and SQLite)."""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


class SQLiteStorage(StorageBackend):
    """SQLite storage backend for fast lookups.

    Schema v5: (repo_id, revision, filename) as primary key.
    Each unique (repo_id, revision, filename) maps to exactly one SHA256.
    Added repos table for caching indexed repositories.
    """

    SCHEMA_VERSION = 5

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize or verify the database schema."""
        conn = self._conn

        # Check if schema_version table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
        if cursor.fetchone():
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            if version == self.SCHEMA_VERSION:
                return  # Already at current version
            # Old version - drop and recreate
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
