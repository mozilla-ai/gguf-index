"""Tests for storage backends."""

import tempfile
from pathlib import Path

import pytest

from gguf_index.storage import GGUFEntry, JSONStorage, SQLiteStorage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_entry():
    """Create a sample GGUF entry."""
    return GGUFEntry(
        sha256="abc123def456",
        repo_id="test/repo",
        revision="commit123",
        filename="model.gguf",
        size=1000000,
    )


class TestGGUFEntry:
    def test_to_dict(self, sample_entry):
        d = sample_entry.to_dict()
        assert d["sha256"] == "abc123def456"
        assert d["repo_id"] == "test/repo"
        assert d["revision"] == "commit123"
        assert d["filename"] == "model.gguf"
        assert d["size"] == 1000000
        assert "indexed_at" in d

    def test_from_dict(self):
        data = {
            "sha256": "abc123",
            "repo_id": "test/repo",
            "revision": "commit456",
            "filename": "test.gguf",
            "size": 500,
            "indexed_at": "2024-01-01T00:00:00Z",
        }
        entry = GGUFEntry.from_dict(data)
        assert entry.sha256 == "abc123"
        assert entry.repo_id == "test/repo"
        assert entry.revision == "commit456"
        assert entry.indexed_at == "2024-01-01T00:00:00Z"

    def test_url_properties(self, sample_entry):
        assert "huggingface.co/test/repo/blob" in sample_entry.url
        assert "huggingface.co/test/repo/resolve" in sample_entry.download_url


class TestJSONStorage:
    def test_add_and_get(self, temp_dir, sample_entry):
        storage = JSONStorage(temp_dir / "index.json")
        storage.add(sample_entry)

        results = storage.get("abc123def456")
        assert len(results) == 1
        assert results[0].repo_id == "test/repo"

    def test_count(self, temp_dir, sample_entry):
        storage = JSONStorage(temp_dir / "index.json")
        assert storage.count() == 0
        storage.add(sample_entry)
        assert storage.count() == 1

    def test_save_and_load(self, temp_dir, sample_entry):
        storage = JSONStorage(temp_dir / "index.json")
        storage.add(sample_entry)
        storage.save()

        # Load in new instance
        storage2 = JSONStorage(temp_dir / "index.json")
        storage2.load()
        assert storage2.count() == 1


class TestSQLiteStorage:
    def test_add_and_get(self, temp_dir, sample_entry):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()
        storage.add(sample_entry)

        results = storage.get("abc123def456")
        assert len(results) == 1
        assert results[0].repo_id == "test/repo"

    def test_count(self, temp_dir, sample_entry):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()
        assert storage.count() == 0
        storage.add(sample_entry)
        assert storage.count() == 1

    def test_schema_version(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()
        assert storage.SCHEMA_VERSION == 5


class TestSQLiteRepoCache:
    def test_get_repo_cache_empty(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()
        assert storage.get_repo_cache("nonexistent/repo") is None

    def test_set_and_get_repo_cache(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()

        storage.set_repo_cache("test/repo", "commit123abc", max_revisions=1)
        cached = storage.get_repo_cache("test/repo")

        assert cached is not None
        assert cached["last_indexed_commit"] == "commit123abc"
        assert cached["max_revisions_indexed"] == 1
        assert "last_indexed_at" in cached

    def test_set_repo_cache_updates(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()

        storage.set_repo_cache("test/repo", "commit1", max_revisions=1)
        storage.set_repo_cache("test/repo", "commit2", max_revisions=5)

        cached = storage.get_repo_cache("test/repo")
        assert cached["last_indexed_commit"] == "commit2"
        assert cached["max_revisions_indexed"] == 5

    def test_set_repo_cache_all_revisions(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()

        storage.set_repo_cache("test/repo", "commit123", max_revisions=None)
        cached = storage.get_repo_cache("test/repo")

        assert cached["max_revisions_indexed"] is None

    def test_clear_repo_cache_specific(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()

        storage.set_repo_cache("repo1", "commit1", max_revisions=1)
        storage.set_repo_cache("repo2", "commit2", max_revisions=1)

        storage.clear_repo_cache("repo1")

        assert storage.get_repo_cache("repo1") is None
        assert storage.get_repo_cache("repo2") is not None

    def test_clear_repo_cache_all(self, temp_dir):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()

        storage.set_repo_cache("repo1", "commit1", max_revisions=1)
        storage.set_repo_cache("repo2", "commit2", max_revisions=1)

        storage.clear_repo_cache()

        assert storage.get_repo_cache("repo1") is None
        assert storage.get_repo_cache("repo2") is None

    def test_clear_all(self, temp_dir, sample_entry):
        storage = SQLiteStorage(temp_dir / "index.db")
        storage.load()

        storage.add(sample_entry)
        storage.set_repo_cache("test/repo", "commit123", max_revisions=1)

        storage.clear_all()

        assert storage.count() == 0
        assert storage.get_repo_cache("test/repo") is None
