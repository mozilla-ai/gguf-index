"""Click-based CLI for GGUF Index."""

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .index import GGUFIndex, DEFAULT_JSON_PATH, DEFAULT_SQLITE_PATH
from .storage import SchemaMigrationNeeded, SQLiteStorage

console = Console()


def get_hf_token(token_arg: str | None) -> str | None:
    """Get HuggingFace token from argument or environment."""
    if token_arg:
        return token_arg
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_index(
    json_path: str | None,
    sqlite_path: str | None,
    use_json: bool,
    use_sqlite: bool,
    hf_token: str | None = None,
    requests_per_second: float | None = None,
    max_workers: int = 4,
    verbose: bool = False,
) -> GGUFIndex:
    """Create a GGUFIndex with the specified storage backends."""
    # JSON is opt-in (only if --json flag or --json-path provided)
    json_p = Path(json_path) if json_path else (DEFAULT_JSON_PATH if use_json else None)
    # SQLite is the default
    sqlite_p = Path(sqlite_path) if sqlite_path else (DEFAULT_SQLITE_PATH if use_sqlite else None)

    # Default to SQLite-only if neither specified
    if not json_p and not sqlite_p:
        sqlite_p = DEFAULT_SQLITE_PATH

    # Check for schema migration before creating index
    if sqlite_p:
        storage = SQLiteStorage(sqlite_p)
        current_version, needs_migration = storage.check_schema_version()
        if needs_migration and current_version is not None:
            console.print(
                f"[yellow]Database schema migration needed: v{current_version} -> v{SQLiteStorage.SCHEMA_VERSION}[/yellow]"
            )
            console.print(f"[dim]Database: {sqlite_p}[/dim]")
            console.print(
                "\nThe existing database will be backed up and a new one created."
            )
            if not click.confirm("Proceed with migration?", default=True):
                console.print("[red]Aborted.[/red]")
                sys.exit(1)

            backup_path = storage.backup_and_recreate()
            console.print(f"[green]Backed up old database to: {backup_path}[/green]")

    # Get token from argument or environment
    token = get_hf_token(hf_token)

    index = GGUFIndex(json_path=json_p, sqlite_path=sqlite_p, hf_token=token, requests_per_second=requests_per_second, max_workers=max_workers, verbose=verbose)
    index.load()
    return index


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """GGUF Index - Index and identify GGUF files by SHA256 hash."""
    pass


@cli.command()
@click.argument("query", required=False)
@click.option("--limit", "-l", type=int, default=None, help="Maximum number of repos to index")
@click.option("--revisions", "-r", type=int, default=1, help="Max revisions per file (default: 1, use 0 for all)")
@click.option("--rate", type=float, default=None, help="Requests per second (default: 1.5 anon, 3 auth; 0 = unlimited)")
@click.option("--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)")
@click.option("--force", "-f", is_flag=True, help="Force re-indexing even if cached")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--token", "-t", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def search(query: str | None, limit: int | None, revisions: int, rate: float | None, workers: int, force: bool, verbose: bool, token: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Search HuggingFace for GGUF files and add them to the index.

    QUERY is an optional search term to filter repositories.
    """
    index = get_index(json_path, sqlite_path, use_json, use_sqlite, hf_token=token, requests_per_second=rate, max_workers=workers, verbose=verbose)

    # Convert 0 to None (no limit)
    max_revisions = revisions if revisions != 0 else None
    skipped_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching repositories...", total=None)

        def update_progress(repo_id: str, current: int, total: int):
            progress.update(task, description=f"Indexing {repo_id}", completed=current, total=total)

        def on_skip(repo_id: str, commit: str):
            nonlocal skipped_count
            skipped_count += 1
            if verbose:
                console.print(f"[dim]Skipping {repo_id} (cached at {commit[:8]})[/dim]")

        files_indexed = index.build_from_search(
            query=query,
            limit=limit,
            max_revisions=max_revisions,
            progress_callback=update_progress,
            force=force,
            skip_callback=on_skip,
        )

    console.print(f"\n[green]Indexed {files_indexed} GGUF files[/green]")
    if skipped_count > 0:
        console.print(f"[dim]Skipped {skipped_count} cached repos (use --force to re-index)[/dim]")


@cli.command()
@click.argument("repo_id")
@click.option("--revisions", "-r", type=int, default=1, help="Max revisions per file (default: 1, use 0 for all)")
@click.option("--rate", type=float, default=None, help="Requests per second (default: 1.5 anon, 3 auth; 0 = unlimited)")
@click.option("--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)")
@click.option("--force", "-f", is_flag=True, help="Force re-indexing even if cached")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--token", "-t", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def add(repo_id: str, revisions: int, rate: float | None, workers: int, force: bool, verbose: bool, token: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Index all GGUF files from a specific repository.

    REPO_ID is the HuggingFace repository ID (e.g., TheBloke/Llama-2-7B-GGUF).
    """
    index = get_index(json_path, sqlite_path, use_json, use_sqlite, hf_token=token, requests_per_second=rate, max_workers=workers, verbose=verbose)

    # Convert 0 to None (no limit)
    max_revisions = revisions if revisions != 0 else None

    with console.status(f"Indexing {repo_id}..."):
        try:
            files_indexed, skipped_commit = index.index_repo(repo_id, max_revisions=max_revisions, force=force)
            if skipped_commit:
                console.print(f"[yellow]Skipping {repo_id} (already indexed at commit {skipped_commit[:12]})[/yellow]")
                console.print("[dim]Use --force to re-index[/dim]")
            else:
                console.print(f"[green]Indexed {files_indexed} GGUF files from {repo_id}[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.argument("hashes", nargs=-1, required=True)
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def lookup(hashes: tuple[str, ...], json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Look up GGUF file(s) by SHA256 hash.

    HASHES is one or more SHA256 hashes to look up.
    """
    index = get_index(json_path, sqlite_path, use_json, use_sqlite)

    any_found = False
    for sha256 in hashes:
        entries = index.lookup(sha256)

        if not entries:
            console.print(f"[yellow]No match found for {sha256}[/yellow]")
            continue

        any_found = True

        # Show file info (same for all entries with same hash)
        first = entries[0]
        console.print(f"\n[cyan]SHA256:[/cyan] {first.sha256}")
        console.print(f"[cyan]Size:[/cyan] {_format_size(first.size)}")

        # Show all sources
        title = f"Found {len(entries)} source(s)" if len(entries) > 1 else "Source"
        table = Table(title=title)
        table.add_column("#", style="dim")
        table.add_column("Repository", style="cyan")
        table.add_column("Revision", style="yellow")
        table.add_column("Filename", style="green")

        for i, entry in enumerate(entries, 1):
            table.add_row(str(i), entry.repo_id, entry.revision[:12], entry.filename)

        console.print(table)

        # Show download URL for first entry
        console.print(f"[blue]Download URL:[/blue] {entries[0].download_url}")

    if not any_found:
        sys.exit(1)


@cli.command()
@click.argument("file_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def identify(file_paths: tuple[str, ...], json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Identify local GGUF file(s) by computing SHA256 and looking up.

    FILE_PATHS is one or more paths to local GGUF files. Supports glob patterns.
    """
    index = get_index(json_path, sqlite_path, use_json, use_sqlite)

    any_found = False
    for file_path in file_paths:
        path = Path(file_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Computing SHA256 of {path.name}...", total=path.stat().st_size)

            def update_progress(bytes_read: int, total_bytes: int):
                progress.update(task, completed=bytes_read)

            sha256, entries = index.identify_file(path, progress_callback=update_progress)

        console.print(f"\n[cyan]File:[/cyan] {path.name}")
        console.print(f"[cyan]SHA256:[/cyan] {sha256}")

        if not entries:
            console.print("[yellow]No match found in index[/yellow]")
            continue

        any_found = True
        first = entries[0]
        console.print(f"[cyan]Size:[/cyan] {_format_size(first.size)}")

        # Show all sources
        title = f"Found {len(entries)} source(s)" if len(entries) > 1 else "Source"
        table = Table(title=title)
        table.add_column("#", style="dim")
        table.add_column("Repository", style="cyan")
        table.add_column("Revision", style="yellow")
        table.add_column("Filename", style="green")

        for i, entry in enumerate(entries, 1):
            table.add_row(str(i), entry.repo_id, entry.revision[:12], entry.filename)

        console.print(table)

        # Show download URL for first entry
        console.print(f"[blue]Download URL:[/blue] {entries[0].download_url}")

    if not any_found:
        sys.exit(1)


@cli.command(name="export")
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: stdout for json/jsonl, required for parquet)")
@click.option("--format", "-f", "fmt", type=click.Choice(["parquet", "json", "jsonl"]), default="parquet", help="Output format (default: parquet)")
@click.option("--push", is_flag=True, help="Push to HuggingFace dataset repo after export")
@click.option("--repo", default="mozilla-ai/gguf-index", help="HuggingFace dataset repo for push")
@click.option("--token", "-t", type=str, default=None, help="HuggingFace API token for push (or set HF_TOKEN env var)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def export_cmd(output: str | None, fmt: str, push: bool, repo: str, token: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Export the index for sharing.

    Default format is Parquet, optimized for HuggingFace datasets with Xet storage.
    Use --push to upload directly to a HuggingFace dataset repository.
    """
    from .parquet import export_to_parquet, push_to_hf

    index = get_index(json_path, sqlite_path, use_json, use_sqlite)
    entries = index.get_all()

    if fmt == "parquet":
        if not output and not push:
            console.print("[red]Error: --output is required for parquet format (or use --push)[/red]")
            sys.exit(1)

        output_path = Path(output) if output else Path("gguf_index.parquet")

        # Get repos cache if SQLite backend is available
        repos = None
        if index.sqlite_storage:
            repos = index.sqlite_storage.get_all_repo_cache()

        with console.status(f"Exporting {len(entries)} entries to parquet..."):
            repos_path = export_to_parquet(entries, output_path, repos=repos)

        console.print(f"[green]Exported {len(entries)} entries to {output_path}[/green]")
        if repos_path and repos:
            console.print(f"[green]Exported {len(repos)} cached repos to {repos_path}[/green]")

        if push:
            hf_token = get_hf_token(token)
            if not hf_token:
                console.print("[red]Error: HF_TOKEN required for push (set env var or use --token)[/red]")
                sys.exit(1)

            with console.status(f"Pushing to {repo}..."):
                url = push_to_hf(output_path, repo, token=hf_token)

            console.print(f"[green]Pushed to {url}[/green]")

    elif fmt == "jsonl":
        lines = [json.dumps(entry.to_dict(), separators=(",", ":")) for entry in entries]
        content = "\n".join(lines)
        if output:
            with open(output, "w") as f:
                f.write(content)
                f.write("\n")
            console.print(f"[green]Exported {len(entries)} entries to {output} (JSONL)[/green]")
        else:
            click.echo(content)

    else:  # json
        data = {f"{e.repo_id}/{e.revision}/{e.filename}": e.to_dict() for e in entries}
        if output:
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]Exported {len(data)} entries to {output} (JSON)[/green]")
        else:
            click.echo(json.dumps(data, indent=2))


@cli.command(name="import")
@click.argument("source", required=False)
@click.option("--repo", default="mozilla-ai/gguf-index", help="HuggingFace dataset repo to download from")
@click.option("--format", "-f", "fmt", type=click.Choice(["parquet", "jsonl"]), default=None, help="Input format (auto-detected from extension)")
@click.option("--merge", is_flag=True, help="Merge with existing data (default: replace)")
@click.option("--token", "-t", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def import_cmd(source: str | None, repo: str, fmt: str | None, merge: bool, token: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Import index from parquet file or HuggingFace dataset.

    SOURCE is a local file path. If not provided, downloads from --repo.
    Format is auto-detected from extension (.parquet or .jsonl).
    """
    from .parquet import download_from_hf, import_from_parquet, import_repos_from_parquet
    from .storage import GGUFEntry

    index = get_index(json_path, sqlite_path, use_json, use_sqlite)
    hf_token = get_hf_token(token)

    # Determine source
    repos_path = None
    if source is None:
        # Download from HuggingFace
        with console.status(f"Downloading from {repo}..."):
            source_path, repos_path = download_from_hf(repo, token=hf_token)
        console.print(f"[dim]Downloaded from {repo}[/dim]")
        detected_fmt = "parquet"
    else:
        source_path = Path(source)
        if not source_path.exists():
            console.print(f"[red]Error: File not found: {source}[/red]")
            sys.exit(1)
        # Auto-detect format
        if fmt:
            detected_fmt = fmt
        elif source_path.suffix == ".parquet":
            detected_fmt = "parquet"
        else:
            detected_fmt = "jsonl"

    # Clear existing data if not merging
    if not merge:
        if index.sqlite_storage:
            index.sqlite_storage.clear_all()
        console.print("[dim]Replacing existing index data[/dim]")

    imported = 0
    errors = 0

    if detected_fmt == "parquet":
        with console.status("Importing from parquet...") as status:
            for entry in import_from_parquet(source_path):
                try:
                    index.add(entry)
                    imported += 1
                    if imported % 10000 == 0:
                        status.update(f"Imported {imported} entries...")
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        console.print(f"[yellow]Error: {e}[/yellow]")

        # Import repos cache if available
        repos_imported = 0
        if index.sqlite_storage:
            repos = import_repos_from_parquet(repos_path if repos_path else source_path)
            if repos:
                repos_imported = index.sqlite_storage.import_repo_cache(repos)
    else:  # jsonl
        repos_imported = 0  # JSONL doesn't support repos cache
        with open(source_path, "r") as f:
            with console.status("Importing from JSONL...") as status:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = GGUFEntry.from_dict(data)
                        index.add(entry)
                        imported += 1
                        if imported % 10000 == 0:
                            status.update(f"Imported {imported} entries...")
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            console.print(f"[yellow]Line {line_num}: {e}[/yellow]")

    index.save()
    console.print(f"[green]Imported {imported} entries[/green]")
    if repos_imported:
        console.print(f"[green]Imported {repos_imported} cached repos[/green]")
    if errors:
        console.print(f"[yellow]Skipped {errors} invalid entries[/yellow]")


@cli.command()
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=False, help="Also use JSON storage (opt-in)")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def stats(json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Show index statistics."""
    index = get_index(json_path, sqlite_path, use_json, use_sqlite)

    statistics = index.stats()

    if statistics["unique_files"] == 0:
        console.print("[yellow]Index is empty. Run 'gguf-index search' to populate it.[/yellow]")
        return

    table = Table(title="GGUF Index Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Unique Files", str(statistics["unique_files"]))
    table.add_row("Total Sources", str(statistics["total_sources"]))
    table.add_row("Total Size", _format_size(statistics["total_size"]))
    table.add_row("Unique Repos", str(statistics["unique_repos"]))

    console.print(table)

    if statistics["repos"]:
        repo_table = Table(title="Top Repositories")
        repo_table.add_column("Repository", style="cyan")
        repo_table.add_column("Sources", style="green")

        for repo_id, count in statistics["repos"][:10]:
            repo_table.add_row(repo_id, str(count))

        console.print(repo_table)


def _format_size(size: int) -> str:
    """Format a size in bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


if __name__ == "__main__":
    cli()
