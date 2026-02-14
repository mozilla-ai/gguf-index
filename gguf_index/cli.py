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
    json_p = Path(json_path) if json_path else (DEFAULT_JSON_PATH if use_json else None)
    sqlite_p = Path(sqlite_path) if sqlite_path else (DEFAULT_SQLITE_PATH if use_sqlite else None)

    # Default to both if neither specified
    if not json_p and not sqlite_p:
        json_p = DEFAULT_JSON_PATH
        sqlite_p = DEFAULT_SQLITE_PATH

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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--token", "-t", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=True, help="Use JSON storage")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def search(query: str | None, limit: int | None, revisions: int, rate: float | None, workers: int, verbose: bool, token: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Search HuggingFace for GGUF files and add them to the index.

    QUERY is an optional search term to filter repositories.
    """
    index = get_index(json_path, sqlite_path, use_json, use_sqlite, hf_token=token, requests_per_second=rate, max_workers=workers, verbose=verbose)

    # Convert 0 to None (no limit)
    max_revisions = revisions if revisions != 0 else None

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

        files_indexed = index.build_from_search(
            query=query,
            limit=limit,
            max_revisions=max_revisions,
            progress_callback=update_progress,
        )

    console.print(f"\n[green]Indexed {files_indexed} GGUF files[/green]")


@cli.command()
@click.argument("repo_id")
@click.option("--revisions", "-r", type=int, default=1, help="Max revisions per file (default: 1, use 0 for all)")
@click.option("--rate", type=float, default=None, help="Requests per second (default: 1.5 anon, 3 auth; 0 = unlimited)")
@click.option("--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--token", "-t", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=True, help="Use JSON storage")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def add(repo_id: str, revisions: int, rate: float | None, workers: int, verbose: bool, token: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Index all GGUF files from a specific repository.

    REPO_ID is the HuggingFace repository ID (e.g., TheBloke/Llama-2-7B-GGUF).
    """
    index = get_index(json_path, sqlite_path, use_json, use_sqlite, hf_token=token, requests_per_second=rate, max_workers=workers, verbose=verbose)

    # Convert 0 to None (no limit)
    max_revisions = revisions if revisions != 0 else None

    with console.status(f"Indexing {repo_id}..."):
        try:
            files_indexed = index.index_repo(repo_id, max_revisions=max_revisions)
            console.print(f"[green]Indexed {files_indexed} GGUF files from {repo_id}[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.argument("hashes", nargs=-1, required=True)
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=True, help="Use JSON storage")
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
@click.option("--json/--no-json", "use_json", default=True, help="Use JSON storage")
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


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: stdout)")
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=True, help="Use JSON storage")
@click.option("--sqlite/--no-sqlite", "use_sqlite", default=True, help="Use SQLite storage")
def export(output: str | None, json_path: str | None, sqlite_path: str | None, use_json: bool, use_sqlite: bool):
    """Export the index to JSON format."""
    index = get_index(json_path, sqlite_path, use_json, use_sqlite)

    data = index.export_json()

    if output:
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Exported {len(data)} entries to {output}[/green]")
    else:
        click.echo(json.dumps(data, indent=2))


@cli.command()
@click.option("--json-path", type=click.Path(), help="Path to JSON index file")
@click.option("--sqlite-path", type=click.Path(), help="Path to SQLite database file")
@click.option("--json/--no-json", "use_json", default=True, help="Use JSON storage")
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
