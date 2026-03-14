"""Tests for parquet import/export utilities."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gguf_index.parquet import export_to_parquet, import_from_parquet
from gguf_index.storage import GGUFEntry


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_entries():
    """Create sample GGUF entries."""
    return [
        GGUFEntry(
            sha256="abc123",
            repo_id="test/repo1",
            revision="commit1",
            filename="model1.gguf",
            size=1000000,
        ),
        GGUFEntry(
            sha256="def456",
            repo_id="test/repo2",
            revision="commit2",
            filename="model2.gguf",
            size=2000000,
        ),
        GGUFEntry(
            sha256="ghi789",
            repo_id="test/repo1",
            revision="commit3",
            filename="model3.gguf",
            size=3000000,
        ),
    ]


class TestExportToParquet:
    def test_export_creates_file(self, temp_dir, sample_entries):
        path = temp_dir / "test.parquet"
        export_to_parquet(sample_entries, path)
        assert path.exists()

    def test_export_contains_all_entries(self, temp_dir, sample_entries):
        path = temp_dir / "test.parquet"
        export_to_parquet(sample_entries, path)

        df = pd.read_parquet(path)
        assert len(df) == 3

    def test_export_preserves_data(self, temp_dir, sample_entries):
        path = temp_dir / "test.parquet"
        export_to_parquet(sample_entries, path)

        df = pd.read_parquet(path)
        assert "abc123" in df["sha256"].values
        assert "test/repo1" in df["repo_id"].values
        assert "model1.gguf" in df["filename"].values

    def test_export_empty_list(self, temp_dir):
        path = temp_dir / "empty.parquet"
        export_to_parquet([], path)

        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 0
        # Check schema is correct even for empty file
        assert "sha256" in df.columns
        assert "repo_id" in df.columns

    def test_export_creates_parent_dirs(self, temp_dir, sample_entries):
        path = temp_dir / "subdir" / "nested" / "test.parquet"
        export_to_parquet(sample_entries, path)
        assert path.exists()


class TestImportFromParquet:
    def test_import_reads_all_entries(self, temp_dir, sample_entries):
        path = temp_dir / "test.parquet"
        export_to_parquet(sample_entries, path)

        imported = list(import_from_parquet(path))
        assert len(imported) == 3

    def test_import_preserves_data(self, temp_dir, sample_entries):
        path = temp_dir / "test.parquet"
        export_to_parquet(sample_entries, path)

        imported = list(import_from_parquet(path))

        sha256s = {e.sha256 for e in imported}
        assert "abc123" in sha256s
        assert "def456" in sha256s
        assert "ghi789" in sha256s

    def test_import_returns_gguf_entries(self, temp_dir, sample_entries):
        path = temp_dir / "test.parquet"
        export_to_parquet(sample_entries, path)

        imported = list(import_from_parquet(path))
        for entry in imported:
            assert isinstance(entry, GGUFEntry)
            assert hasattr(entry, "sha256")
            assert hasattr(entry, "repo_id")
            assert hasattr(entry, "url")

    def test_roundtrip(self, temp_dir, sample_entries):
        """Test that export -> import preserves all data."""
        path = temp_dir / "roundtrip.parquet"
        export_to_parquet(sample_entries, path)

        imported = list(import_from_parquet(path))

        # Create dicts for comparison (without indexed_at which may differ)
        original = {(e.sha256, e.repo_id, e.filename): e.size for e in sample_entries}
        reimported = {(e.sha256, e.repo_id, e.filename): e.size for e in imported}

        assert original == reimported
