"""Tests for GGUFIndex caching logic."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gguf_index.index import GGUFIndex


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def index(temp_dir):
    """Create a GGUFIndex with temp storage."""
    return GGUFIndex(
        json_path=temp_dir / "index.json",
        sqlite_path=temp_dir / "index.db",
        use_defaults=False,
    )


class TestRevisionsSufficient:
    def test_cached_none_covers_all(self, index):
        # If we cached all revisions (None), any request is covered
        assert index._revisions_sufficient(None, None) is True
        assert index._revisions_sufficient(None, 1) is True
        assert index._revisions_sufficient(None, 5) is True
        assert index._revisions_sufficient(None, 100) is True

    def test_cached_some_not_enough_for_all(self, index):
        # If we cached some revisions, requesting all (None) requires re-index
        assert index._revisions_sufficient(1, None) is False
        assert index._revisions_sufficient(5, None) is False

    def test_cached_covers_requested(self, index):
        # Cached >= requested
        assert index._revisions_sufficient(5, 5) is True
        assert index._revisions_sufficient(5, 3) is True
        assert index._revisions_sufficient(5, 1) is True

    def test_cached_not_enough(self, index):
        # Cached < requested
        assert index._revisions_sufficient(1, 5) is False
        assert index._revisions_sufficient(3, 10) is False


class TestShouldSkipRepo:
    def test_skip_when_cached_and_unchanged(self, index):
        index.load()
        index._set_repo_cache("test/repo", "commit123", max_revisions=1)

        # Same commit, same or fewer revisions -> skip
        assert index._should_skip_repo("test/repo", "commit123", 1) is True

    def test_no_skip_when_commit_changed(self, index):
        index.load()
        index._set_repo_cache("test/repo", "commit123", max_revisions=1)

        # Different commit -> don't skip
        assert index._should_skip_repo("test/repo", "commit456", 1) is False

    def test_no_skip_when_more_revisions_needed(self, index):
        index.load()
        index._set_repo_cache("test/repo", "commit123", max_revisions=1)

        # Same commit but need more revisions -> don't skip
        assert index._should_skip_repo("test/repo", "commit123", 5) is False

    def test_no_skip_when_not_cached(self, index):
        index.load()
        # No cache -> don't skip
        assert index._should_skip_repo("test/repo", "commit123", 1) is False

    def test_skip_when_all_revisions_cached(self, index):
        index.load()
        index._set_repo_cache("test/repo", "commit123", max_revisions=None)

        # Cached all revisions -> always skip (if commit unchanged)
        assert index._should_skip_repo("test/repo", "commit123", 1) is True
        assert index._should_skip_repo("test/repo", "commit123", 5) is True
        assert index._should_skip_repo("test/repo", "commit123", None) is True


class TestRepoCacheMethods:
    def test_get_repo_cache_without_sqlite(self, temp_dir):
        # JSON-only index has no cache
        index = GGUFIndex(
            json_path=temp_dir / "index.json",
            sqlite_path=None,
            use_defaults=False,
        )
        index.load()
        assert index._get_repo_cache("any/repo") is None

    def test_set_repo_cache_without_sqlite(self, temp_dir):
        # Should not raise, just no-op
        index = GGUFIndex(
            json_path=temp_dir / "index.json",
            sqlite_path=None,
            use_defaults=False,
        )
        index.load()
        index._set_repo_cache("any/repo", "commit", 1)  # Should not raise

    def test_cache_roundtrip(self, index):
        index.load()

        index._set_repo_cache("test/repo", "commitabc", max_revisions=3)
        cached = index._get_repo_cache("test/repo")

        assert cached is not None
        assert cached["last_indexed_commit"] == "commitabc"
        assert cached["max_revisions_indexed"] == 3


class TestIndexRepoWithCache:
    def test_index_repo_returns_skipped_commit(self, index):
        """Test that index_repo returns skipped commit when cached."""
        index.load()

        # Mock the API
        mock_repo_info = MagicMock()
        mock_repo_info.sha = "commit123"
        index.api.api.repo_info = MagicMock(return_value=mock_repo_info)
        index.api.get_repo_gguf_files = MagicMock(return_value=[])

        # First call - should index
        files, skipped = index.index_repo("test/repo", max_revisions=1)
        assert skipped is None  # Not skipped

        # Second call - should skip
        files, skipped = index.index_repo("test/repo", max_revisions=1)
        assert skipped == "commit123"

    def test_index_repo_force_reindexes(self, index):
        """Test that force=True bypasses cache."""
        index.load()

        # Mock the API
        mock_repo_info = MagicMock()
        mock_repo_info.sha = "commit123"
        index.api.api.repo_info = MagicMock(return_value=mock_repo_info)
        index.api.get_repo_gguf_files = MagicMock(return_value=[])

        # First call - should index
        index.index_repo("test/repo", max_revisions=1)

        # Second call with force - should not skip
        files, skipped = index.index_repo("test/repo", max_revisions=1, force=True)
        assert skipped is None

    def test_index_repo_reindexes_on_commit_change(self, index):
        """Test that new commits trigger re-indexing."""
        index.load()

        # Mock the API - first commit
        mock_repo_info = MagicMock()
        mock_repo_info.sha = "commit1"
        index.api.api.repo_info = MagicMock(return_value=mock_repo_info)
        index.api.get_repo_gguf_files = MagicMock(return_value=[])

        # First call
        index.index_repo("test/repo", max_revisions=1)

        # Change commit
        mock_repo_info.sha = "commit2"

        # Second call - should not skip (commit changed)
        files, skipped = index.index_repo("test/repo", max_revisions=1)
        assert skipped is None
