# Future Work: Shared Collaborative Index

The goal is to build a **shared, collaboratively contributed index** so users don't need to rebuild it from scratch, and contributions benefit everyone.

## Current State

- Local SQLite database with indexed SHA256 lookups
- JSONL export/import for sharing (417K entries = 138 MB, 21 MB gzipped)
- Optimized HuggingFace API calls (batched requests, fast path for latest revision)

## Planned Steps

### 1. `gguf-index sync` Command

**Rationale:** Users need a simple way to download the shared index without manual steps.

**What it entails:**
- Add `gguf-index sync` command that downloads from a well-known URL
- Support multiple sources (HuggingFace dataset, GitHub release, direct URL)
- Handle gzipped JSONL automatically
- Merge with existing local entries (don't overwrite user's local additions)
- Show progress and stats (X new entries added)

```bash
gguf-index sync                          # Download from default source
gguf-index sync --url https://...        # Custom source
gguf-index sync --force                  # Replace local index entirely
```

### 2. Publish Initial Index to HuggingFace

**Rationale:** HuggingFace datasets are free, versioned, and easy to download programmatically.

**What it entails:**
- Create a HuggingFace dataset repository (e.g., `gguf-index/index`)
- Upload the gzipped JSONL file
- Set up versioning (each upload is a new version)
- Document the dataset format and how to use it
- Configure `gguf-index sync` to use this as the default source

### 3. Contribution Workflow via GitHub

**Rationale:** GitHub PRs provide review, attribution, and history. Anyone can contribute.

**What it entails:**
- Create a GitHub repository for contributions
- Contributors run `gguf-index export` on repos they've indexed
- Submit PR with their JSONL additions
- Automated CI checks:
  - Valid JSONL format
  - Required fields present
  - URLs resolve (spot check)
  - No duplicate entries
- Maintainer merges, triggers new release to HuggingFace

```
contributions/
  2024-02-15-user1.jsonl
  2024-02-15-user2.jsonl
  ...
```

### 4. Signature Verification

**Rationale:** Prevent tampering. Users should be able to verify the index came from trusted sources.

**What it entails:**
- Maintainers sign releases with GPG or SSH key
- Publish public key in well-known location
- `gguf-index sync` verifies signature before importing
- Optional: contributors sign their submissions for attribution

```bash
gguf-index sync --verify                 # Require valid signature
gguf-index sync --trust-key <fingerprint>
```

### 5. Delta Updates

**Rationale:** Downloading 21 MB every time is wasteful when only a few entries changed.

**What it entails:**
- Track "last synced" version/timestamp locally
- Server provides delta files (entries added since version X)
- `gguf-index sync` fetches only the delta
- Fall back to full download if delta unavailable or too old

```
releases/
  full-2024-02-15.jsonl.gz      # Full index
  delta-2024-02-14-to-15.jsonl.gz  # Just new entries
```

### 6. Multiple Mirrors (Decentralization)

**Rationale:** No single point of failure. If HuggingFace is down, other sources work.

**What it entails:**
- Primary: HuggingFace dataset
- Mirrors: GitHub Releases, IPFS, BitTorrent
- `gguf-index sync` tries sources in order, falls back if one fails
- Community can run their own mirrors
- Document how to set up a mirror

## Security Considerations

The index maps SHA256 hashes to URLs. Key properties:

- **Self-verifying:** Anyone can check any entry by downloading and hashing
- **Low-stakes:** It's just a mapping, not secrets or code
- **Main attacks:** Pollution (garbage entries) or misdirection (wrong URLs)

Mitigations:
- Signed releases from trusted maintainers
- Automated verification in CI (spot-check that URLs resolve)
- Community reporting of bad entries
- Reputation tracking for contributors

## Space & Cost Estimates

| Scale | Entries | JSONL Size | Gzipped | Hosting Cost |
|-------|---------|------------|---------|--------------|
| Current | 417K | 138 MB | 21 MB | Free |
| 1M entries | ~1M | ~350 MB | ~50 MB | Free |
| 10M entries | ~10M | ~3.5 GB | ~500 MB | ~$1/month |

HuggingFace datasets and GitHub releases are free for these sizes.
