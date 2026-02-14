# GGUF Index

A Python tool (CLI + library) that creates an index mapping SHA256 hashes of GGUF files to their HuggingFace URLs, enabling identification of local GGUF files.

## Installation

```bash
pip install -e .
```

## CLI Usage

### Search and Index GGUF Files

Search HuggingFace for GGUF repositories and add them to the index:

```bash
# Search all GGUF repos
gguf-index search

# Search with a query
gguf-index search "llama"

# Limit the number of repos to index
gguf-index search --limit 10
```

### Add a Specific Repository

Index all GGUF files from a specific HuggingFace repository:

```bash
gguf-index add TheBloke/Llama-2-7B-GGUF
```

### Look Up a File by SHA256

```bash
gguf-index lookup abc123def456...
```

### Identify a Local File

Compute the SHA256 of a local file and look it up in the index:

```bash
gguf-index identify /path/to/model.gguf
```

### Export Index to JSON

```bash
# Export to stdout
gguf-index export

# Export to a file
gguf-index export -o my_index.json
```

### View Index Statistics

```bash
gguf-index stats
```

## Library Usage

```python
from gguf_index import GGUFIndex

# Create an index with custom storage paths
index = GGUFIndex(
    json_path="my_index.json",
    sqlite_path="my_index.db",
)

# Load existing index
index.load()

# Build index from HuggingFace search
files_indexed = index.build_from_search(query="llama", limit=10)

# Index a specific repository
index.index_repo("TheBloke/Llama-2-7B-GGUF")

# Look up a file by SHA256 (returns list of all known sources)
entries = index.lookup("abc123def456...")
for entry in entries:
    print(f"Found: {entry.repo_id}/{entry.filename}")
    print(f"Download: {entry.download_url}")

# Identify a local file (returns list of all known sources)
sha256, entries = index.identify_file("/path/to/model.gguf")
print(f"SHA256: {sha256}")
print(f"Found in {len(entries)} repository(ies)")
for entry in entries:
    print(f"  - {entry.repo_id}/{entry.filename}")

# Get statistics
stats = index.stats()
print(f"Unique files: {stats['unique_files']}")
print(f"Total sources: {stats['total_sources']}")
```

## Index Data Structure

The index is **revision-aware** - each unique (repo_id, revision, filename) tuple maps to exactly one SHA256 hash. This enables:

1. **Full history tracking**: All historical versions of a file are indexed, not just the latest
2. **Precise URLs**: Download URLs include the exact commit hash, ensuring you get the exact file
3. **Multiple sources per hash**: The same file content (SHA256) can exist at multiple URLs across repos and revisions

```json
{
  "TheBloke/gpt-oss-GGUF/abc123def.../gpt-oss.Q4_K_M.gguf": {
    "sha256": "...",
    "repo_id": "TheBloke/gpt-oss-GGUF",
    "revision": "abc123def456789...",
    "filename": "gpt-oss.Q4_K_M.gguf",
    "size": 4368438944,
    "indexed_at": "2026-02-13T12:00:00Z"
  }
}
```

**Primary key**: `(repo_id, revision, filename)` - ensures each unique URL maps to exactly one SHA256.

When looking up a file by SHA256, all matching URLs (across repos and historical revisions) are displayed.

## Storage Backends

GGUF Index supports two storage backends:

- **JSON**: Portable and shareable, stored in `~/.gguf-index/index.json`
- **SQLite**: Fast lookups with indexed SHA256 column, stored in `~/.gguf-index/index.db`

By default, both backends are used. You can customize the storage paths or disable either backend via CLI options.

## License

MIT
