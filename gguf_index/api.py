"""HuggingFace API interactions for GGUF file discovery."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Iterator

import httpx
from huggingface_hub import HfApi, list_models


def _default_log(msg: str) -> None:
    """Default logging function (prints to stderr with timestamp)."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}", file=sys.stderr)


class HuggingFaceAPI:
    """Wrapper for HuggingFace API with parallel request support and rate limiting."""

    def __init__(
        self,
        token: str | None = None,
        max_workers: int = 4,
        requests_per_second: float | None = None,
        verbose: bool = False,
    ):
        self.api = HfApi(token=token)
        self.token = token
        self.max_workers = max_workers
        # Default: 1.5 req/s for unauthenticated, 3 req/s for authenticated (free tier safe)
        # 0 means no rate limiting
        if requests_per_second is None:
            requests_per_second = 3.0 if token else 1.5
        self.requests_per_second = requests_per_second
        self.min_request_interval = 0.0 if requests_per_second == 0 else 1.0 / requests_per_second
        self._http_client: httpx.Client | None = None
        self._rate_lock = Lock()
        self._last_request_time = 0.0
        self._verbose = verbose
        self._log: Callable[[str], None] = _default_log if verbose else lambda x: None
        if verbose:
            if self.requests_per_second == 0:
                self._log(f"Rate limit: disabled (no throttling), workers: {self.max_workers}")
            else:
                self._log(f"Rate limit: {self.requests_per_second} req/s (interval: {self.min_request_interval:.3f}s), workers: {self.max_workers}")

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client for direct API calls."""
        if self._http_client is None:
            headers = {"User-Agent": "gguf-index/0.1.0"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._http_client = httpx.Client(
                headers=headers,
                timeout=30.0,
                limits=httpx.Limits(max_connections=self.max_workers * 2),
            )
        return self._http_client

    def _rate_limited_request(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs,
    ) -> httpx.Response | None:
        """Make a rate-limited HTTP request with retry logic."""
        client = self._get_http_client()

        for attempt in range(max_retries):
            # Rate limiting - serialize timing check with lock
            with self._rate_lock:
                elapsed = time.time() - self._last_request_time
                wait_time = self.min_request_interval - elapsed
                if wait_time > 0:
                    self._log(f"Rate limit: waiting {wait_time:.3f}s")
                    time.sleep(wait_time)
                self._last_request_time = time.time()

            # Make request outside lock so multiple can be in flight
            try:
                short_url = url.split("huggingface.co")[-1] if "huggingface.co" in url else url
                params = kwargs.get("params")
                json_body = kwargs.get("json")
                if params:
                    param_str = "&".join(f"{k}={v}" for k, v in params.items())
                    short_url = f"{short_url}?{param_str}"
                if json_body and "paths" in json_body:
                    paths = json_body["paths"]
                    if len(paths) == 1:
                        short_url = f"{short_url} [{paths[0]}]"
                    else:
                        short_url = f"{short_url} [{len(paths)} files]"
                self._log(f"HTTP {method} {short_url}")

                if method == "GET":
                    resp = client.get(url, **kwargs)
                else:
                    resp = client.post(url, **kwargs)

                if resp.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    self._log(f"429 Too Many Requests, retry after {retry_after}s")
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                return resp

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    self._log(f"429 Too Many Requests, retry after {2 ** attempt}s")
                    time.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                if attempt < max_retries - 1:
                    self._log(f"Request error: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def search_gguf_repos(
        self,
        query: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Search for repositories containing GGUF files.

        Args:
            query: Optional search query to filter repos
            limit: Maximum number of repos to return

        Yields:
            Repository information dictionaries
        """
        search_kwargs: dict[str, Any] = {
            "filter": "gguf",
            "sort": "downloads",
        }
        if query:
            search_kwargs["search"] = query

        count = 0
        for model in list_models(**search_kwargs):
            if limit and count >= limit:
                break

            yield {
                "repo_id": model.id,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
            }
            count += 1

    def get_repo_gguf_files(
        self,
        repo_id: str,
        max_revisions: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Get all GGUF files from a repository with their history.

        Fetches file history in parallel with rate limiting.

        Args:
            repo_id: HuggingFace repository ID
            max_revisions: Maximum revisions per file (None = all, 1 = latest only)

        Yields:
            File information dictionaries with SHA256 hashes and revision
        """
        self._log(f"Fetching repo info: {repo_id}")

        # First, get the list of GGUF files in the repo (at HEAD)
        try:
            info = self.api.repo_info(repo_id, files_metadata=True)
        except Exception as e:
            raise RuntimeError(f"Failed to get repo info for {repo_id}: {e}") from e

        if not hasattr(info, "siblings") or not info.siblings:
            self._log(f"No files found in {repo_id}")
            return

        # Find all GGUF files with their LFS info
        gguf_siblings = [
            f for f in info.siblings
            if f.rfilename.lower().endswith(".gguf")
        ]

        if not gguf_siblings:
            self._log(f"No GGUF files in {repo_id}")
            return

        self._log(f"Found {len(gguf_siblings)} GGUF files in {repo_id}")

        # Fast path: for max_revisions=1, use data from repo_info directly (no extra API calls)
        if max_revisions == 1:
            revision = info.sha  # Latest commit
            self._log(f"Using repo info for latest revision {revision[:8]} (no extra API calls)")
            for f in gguf_siblings:
                if hasattr(f, "lfs") and f.lfs and f.lfs.sha256:
                    self._log(f"  {f.rfilename}: {f.lfs.sha256[:16]}...")
                    yield {
                        "filename": f.rfilename,
                        "sha256": f.lfs.sha256,
                        "size": f.lfs.size,
                        "repo_id": repo_id,
                        "revision": revision,
                    }
            return

        gguf_files = [f.rfilename for f in gguf_siblings]

        base_url = f"https://huggingface.co/api/models/{repo_id}"

        # Step 1: Get commits for all files in parallel
        self._log(f"Fetching commit history for {len(gguf_files)} files...")
        file_commits: dict[str, list[dict]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._fetch_file_commits, base_url, filename): filename
                for filename in gguf_files
            }

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    commits = future.result()
                    if commits:
                        total_commits = len(commits)
                        # Limit to max_revisions if specified (commits are newest first)
                        if max_revisions is not None and max_revisions > 0:
                            commits = commits[:max_revisions]
                        file_commits[filename] = commits
                        self._log(f"  {filename}: {len(commits)} revision(s) (of {total_commits} total)")
                except Exception as e:
                    self._log(f"  {filename}: error fetching commits - {e}")

        # Step 2: Group files by revision for batched requests
        revision_files: dict[str, list[str]] = {}
        for filename, commits in file_commits.items():
            for commit in commits:
                rev = commit["id"]
                if rev not in revision_files:
                    revision_files[rev] = []
                revision_files[rev].append(filename)

        total_files = sum(len(files) for files in revision_files.values())
        self._log(f"Fetching SHA256 for {total_files} files across {len(revision_files)} revision(s)...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_rev = {
                executor.submit(
                    self._fetch_files_at_revision, base_url, repo_id, revision, filenames
                ): revision
                for revision, filenames in revision_files.items()
            }

            for future in as_completed(future_to_rev):
                revision = future_to_rev[future]
                try:
                    results = future.result()
                    for result in results:
                        self._log(f"  {result['filename']}@{revision[:8]}: {result['sha256'][:16]}...")
                        yield result
                except Exception as e:
                    self._log(f"  revision {revision[:8]}: error - {e}")

    def _fetch_file_commits(
        self,
        base_url: str,
        filename: str,
    ) -> list[dict] | None:
        """Fetch commits that touched a specific file."""
        resp = self._rate_limited_request(
            "GET",
            f"{base_url}/commits/main",
            params={"path": filename},
        )
        if resp:
            return resp.json()
        return None

    def _fetch_files_at_revision(
        self,
        base_url: str,
        repo_id: str,
        revision: str,
        filenames: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch file info for multiple files at a specific revision (batched)."""
        resp = self._rate_limited_request(
            "POST",
            f"{base_url}/paths-info/{revision}",
            json={"paths": filenames, "expand": True},
        )

        if not resp:
            return []

        results = []
        for file_info in resp.json():
            lfs = file_info.get("lfs")
            if not lfs or not lfs.get("oid"):
                continue

            results.append({
                "filename": file_info["path"],
                "sha256": lfs["oid"],
                "size": lfs["size"],
                "repo_id": repo_id,
                "revision": revision,
            })

        return results

    def get_repo_info(self, repo_id: str) -> dict[str, Any]:
        """Get repository metadata."""
        model_info = self.api.model_info(repo_id)
        return {
            "repo_id": model_info.id,
            "author": model_info.author,
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "tags": model_info.tags,
        }

    def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
