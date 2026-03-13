"""GGUF Index management."""

import hashlib
from pathlib import Path
from typing import Any, Callable

from .api import HuggingFaceAPI
from .storage import GGUFEntry, JSONStorage, SQLiteStorage, StorageBackend


DEFAULT_JSON_PATH = Path.home() / ".gguf-index" / "index.json"
DEFAULT_SQLITE_PATH = Path.home() / ".gguf-index" / "index.db"


class GGUFIndex:
    """
    Main class for building and querying the GGUF file index.

    Supports both JSON and SQLite storage backends.
    """

    def __init__(
        self,
        json_path: str | Path | None = None,
        sqlite_path: str | Path | None = None,
        hf_token: str | None = None,
        use_defaults: bool = True,
        requests_per_second: float | None = None,
        max_workers: int = 4,
        verbose: bool = False,
    ):
        """
        Initialize the GGUF index.

        Args:
            json_path: Path to JSON index file (optional)
            sqlite_path: Path to SQLite database file (optional)
            hf_token: HuggingFace API token for authenticated requests
            use_defaults: If True and no paths provided, use default paths in ~/.gguf-index/
            requests_per_second: Max API requests per second (default: 1.5 anonymous, 3 authenticated)
            max_workers: Number of parallel worker threads (default: 4)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.backends: list[StorageBackend] = []

        # Use default paths if none provided and use_defaults is True
        if not json_path and not sqlite_path and use_defaults:
            sqlite_path = DEFAULT_SQLITE_PATH

        # SQLite is primary backend (added first for stats/queries)
        if sqlite_path:
            self.sqlite_storage = SQLiteStorage(sqlite_path)
            self.backends.append(self.sqlite_storage)
        else:
            self.sqlite_storage = None

        # JSON is secondary/opt-in backend
        if json_path:
            self.json_storage = JSONStorage(json_path)
            self.backends.append(self.json_storage)
        else:
            self.json_storage = None

        if not self.backends:
            # Fallback to SQLite in current directory
            self.sqlite_storage = SQLiteStorage("gguf_index.db")
            self.backends.append(self.sqlite_storage)

        self.api = HuggingFaceAPI(token=hf_token, requests_per_second=requests_per_second, max_workers=max_workers, verbose=verbose)

    def load(self) -> None:
        """Load index from all configured backends."""
        for backend in self.backends:
            backend.load()

    def save(self) -> None:
        """Save index to all configured backends."""
        for backend in self.backends:
            backend.save()

    def add(self, entry: GGUFEntry) -> None:
        """Add an entry to all backends."""
        for backend in self.backends:
            backend.add(entry)

    def lookup(self, sha256: str) -> list[GGUFEntry]:
        """
        Look up a file by its SHA256 hash.

        Returns all known sources for this file hash.
        """
        sha256 = sha256.lower()
        for backend in self.backends:
            result = backend.get(sha256)
            if result:
                return result
        return []

    def count(self) -> int:
        """Get the number of entries (sources) in the primary backend."""
        if self.backends:
            return self.backends[0].count()
        return 0

    def count_unique_hashes(self) -> int:
        """Get the number of unique file hashes in the primary backend."""
        if self.backends:
            return self.backends[0].count_unique_hashes()
        return 0

    def get_all(self) -> list[GGUFEntry]:
        """Get all entries from the primary backend."""
        if self.backends:
            return self.backends[0].get_all()
        return []

    def _get_repo_cache(self, repo_id: str) -> dict[str, Any] | None:
        """Get cached repo info from SQLite backend."""
        if self.sqlite_storage:
            return self.sqlite_storage.get_repo_cache(repo_id)
        return None

    def _set_repo_cache(self, repo_id: str, commit: str, max_revisions: int | None) -> None:
        """Update repo cache in SQLite backend."""
        if self.sqlite_storage:
            self.sqlite_storage.set_repo_cache(repo_id, commit, max_revisions)

    def _revisions_sufficient(self, cached_revisions: int | None, requested_revisions: int | None) -> bool:
        """Check if cached revisions cover the requested amount.

        Args:
            cached_revisions: Number of revisions previously indexed (None = all)
            requested_revisions: Number of revisions now requested (None = all)

        Returns:
            True if we don't need to re-index for more revisions
        """
        # If we cached all revisions, that covers any request
        if cached_revisions is None:
            return True
        # If requesting all revisions but we only cached some, need to re-index
        if requested_revisions is None:
            return False
        # Otherwise, cached >= requested means we're covered
        return cached_revisions >= requested_revisions

    def _should_skip_repo(
        self,
        repo_id: str,
        current_commit: str,
        max_revisions: int | None,
    ) -> bool:
        """Check if we should skip indexing a repo based on cache.

        Args:
            repo_id: Repository ID
            current_commit: Current HEAD commit of the repo
            max_revisions: Requested max revisions to index

        Returns:
            True if the repo should be skipped (already indexed)
        """
        cached = self._get_repo_cache(repo_id)
        if not cached:
            return False

        # Skip if commit unchanged AND we have enough revisions cached
        if (cached["last_indexed_commit"] == current_commit and
            self._revisions_sufficient(cached["max_revisions_indexed"], max_revisions)):
            return True

        return False

    def build_from_search(
        self,
        query: str | None = None,
        limit: int | None = None,
        max_revisions: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        force: bool = False,
        skip_callback: Callable[[str, str], None] | None = None,
    ) -> int:
        """
        Build the index by searching HuggingFace for GGUF repositories.

        Args:
            query: Search query to filter repositories
            limit: Maximum number of repositories to process
            max_revisions: Maximum revisions per file (None = all)
            progress_callback: Optional callback(repo_id, current, total) for progress
            force: Force re-indexing even if cached
            skip_callback: Optional callback(repo_id, commit) when skipping a cached repo

        Returns:
            Number of files indexed
        """
        repos = list(self.api.search_gguf_repos(query=query, limit=limit))
        total_repos = len(repos)
        files_indexed = 0

        for i, repo in enumerate(repos):
            repo_id = repo["repo_id"]
            current_commit = repo["sha"]  # Already available from search results

            if progress_callback:
                progress_callback(repo_id, i + 1, total_repos)

            # Check cache using sha from search results (no extra API call needed)
            if not force and self._should_skip_repo(repo_id, current_commit, max_revisions):
                if skip_callback:
                    skip_callback(repo_id, current_commit)
                continue

            try:
                # Index the repo
                repo_files = 0
                for file_info in self.api.get_repo_gguf_files(repo_id, max_revisions=max_revisions):
                    entry = GGUFEntry(
                        sha256=file_info["sha256"],
                        repo_id=file_info["repo_id"],
                        revision=file_info["revision"],
                        filename=file_info["filename"],
                        size=file_info["size"],
                    )
                    self.add(entry)
                    repo_files += 1
                    files_indexed += 1

                # Update cache after successful indexing (even if no files found)
                self._set_repo_cache(repo_id, current_commit, max_revisions)
            except Exception:
                # Skip repos that fail (access issues, etc.)
                continue

            # Save after each repo so progress survives interrupts
            self.save()
        return files_indexed

    def index_repo(
        self,
        repo_id: str,
        max_revisions: int | None = None,
        force: bool = False,
    ) -> tuple[int, str | None]:
        """
        Index all GGUF files from a specific repository.

        Args:
            repo_id: HuggingFace repository ID
            max_revisions: Maximum number of revisions per file (None = all)
            force: Force re-indexing even if cached

        Returns:
            Tuple of (files_indexed, skipped_commit) where skipped_commit is the
            commit hash if the repo was skipped due to cache, or None if indexed.
        """
        # Get current HEAD commit
        repo_info = self.api.repo_info(repo_id, files_metadata=False)
        current_commit = repo_info.sha

        # Check cache
        if not force and self._should_skip_repo(repo_id, current_commit, max_revisions):
            return 0, current_commit

        files_indexed = 0

        for file_info in self.api.get_repo_gguf_files(repo_id, max_revisions=max_revisions):
            entry = GGUFEntry(
                sha256=file_info["sha256"],
                repo_id=file_info["repo_id"],
                revision=file_info["revision"],
                filename=file_info["filename"],
                size=file_info["size"],
            )
            self.add(entry)
            files_indexed += 1

        # Update cache after successful indexing (even if no files found)
        self._set_repo_cache(repo_id, current_commit, max_revisions)

        self.save()
        return files_indexed, None

    @staticmethod
    def compute_sha256(file_path: str | Path, chunk_size: int = 8192) -> str:
        """
        Compute the SHA256 hash of a local file.

        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read (for memory efficiency)

        Returns:
            Lowercase hex SHA256 hash
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def identify_file(
        self,
        file_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[str, list[GGUFEntry]]:
        """
        Identify a local GGUF file by computing its SHA256 and looking it up.

        Args:
            file_path: Path to the local GGUF file
            progress_callback: Optional callback(bytes_read, total_bytes) for progress

        Returns:
            Tuple of (sha256_hash, list_of_matching_entries)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        total_size = path.stat().st_size
        bytes_read = 0
        sha256_hash = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
                bytes_read += len(chunk)
                if progress_callback:
                    progress_callback(bytes_read, total_size)

        hash_hex = sha256_hash.hexdigest()
        entries = self.lookup(hash_hex)
        return hash_hex, entries

    def export_json(self) -> dict:
        """Export the index as a JSON-serializable dictionary."""
        if self.json_storage:
            return self.json_storage.export()

        # Fall back to converting from other backends
        entries = self.get_all()
        return {entry.sha256: entry.to_dict() for entry in entries}

    def stats(self) -> dict:
        """Get statistics about the index."""
        entries = self.get_all()
        unique_hashes = self.count_unique_hashes()

        if not entries:
            return {
                "total_sources": 0,
                "unique_files": 0,
                "total_size": 0,
                "unique_repos": 0,
                "repos": [],
            }

        repos: dict[str, int] = {}
        seen_hashes: set[str] = set()
        total_size = 0

        for entry in entries:
            repos[entry.repo_id] = repos.get(entry.repo_id, 0) + 1
            # Only count size once per unique hash
            if entry.sha256 not in seen_hashes:
                total_size += entry.size
                seen_hashes.add(entry.sha256)

        return {
            "total_sources": len(entries),
            "unique_files": unique_hashes,
            "total_size": total_size,
            "unique_repos": len(repos),
            "repos": sorted(repos.items(), key=lambda x: x[1], reverse=True),
        }
