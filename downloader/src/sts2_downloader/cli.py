"""CLI for downloading Slay the Spire 2 run data from Spire Codex."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from sts2_downloader.client import (
    get_shared_run,
    iter_export_runs,
    iter_list_pages,
)
from sts2_downloader.storage import append_jsonl


@click.group()
def main() -> None:
    """Download Slay the Spire 2 run data from the Spire Codex API."""


@main.command()
@click.option("--character", type=str, help="Filter by character (IRONCLAD, SILENT, DEFECT, NECROBINDER, REGENT).")
@click.option("--win", type=click.Choice(["0", "1"]), help="Filter by win (1) or loss (0).")
@click.option("--username", type=str, help="Filter by submitter username.")
@click.option("--ascension", type=int, help="Exact ascension level.")
@click.option("--ascension-min", type=int, help="Minimum ascension level.")
@click.option("--ascension-max", type=int, help="Maximum ascension level.")
@click.option("--game-mode", type=str, help="Game mode (standard, daily, custom).")
@click.option("--players", type=str, help="Player count filter (single, multi).")
@click.option("--card", type=str, help="Filter runs containing this card (comma-separated for AND).")
@click.option("--relic", type=str, help="Filter runs containing this relic (comma-separated for AND).")
@click.option("--sort", type=str, help="Sort field.")
@click.option("--build-id", type=str, help="Game version (e.g. v0.107.1).")
@click.option("--winrate-min", type=float, help="Minimum submitter win rate (0-100).")
@click.option("--winrate-max", type=float, help="Maximum submitter win rate (0-100).")
@click.option("--limit", type=int, default=50, show_default=True, help="Runs per page.")
@click.option("--max-pages", type=int, help="Maximum number of pages to fetch.")
@click.option("-o", "--output", type=click.Path(), default="data/runs.jsonl", show_default=True, help="Output JSONL file path.")
def list_runs(
    character: str | None,
    win: str | None,
    username: str | None,
    ascension: int | None,
    ascension_min: int | None,
    ascension_max: int | None,
    game_mode: str | None,
    players: str | None,
    card: str | None,
    relic: str | None,
    sort: str | None,
    build_id: str | None,
    winrate_min: float | None,
    winrate_max: float | None,
    limit: int,
    max_pages: int | None,
    output: str,
) -> None:
    """Download run summaries from the list endpoint."""
    out_path = Path(output)
    total = 0

    filters = {
        k: v
        for k, v in {
            "character": character,
            "win": win,
            "username": username,
            "ascension": ascension,
            "ascension_min": ascension_min,
            "ascension_max": ascension_max,
            "game_mode": game_mode,
            "players": players,
            "card": card,
            "relic": relic,
            "sort": sort,
            "build_id": build_id,
            "winrate_min": winrate_min,
            "winrate_max": winrate_max,
            "limit": limit,
        }.items()
        if v is not None
    }

    click.echo(f"Fetching runs → {out_path}")
    for page_runs in iter_list_pages(max_pages=max_pages, **filters):
        n = append_jsonl(out_path, page_runs)
        total += n
        click.echo(f"  +{n} runs (total: {total})")

    click.echo(f"Done. {total} runs saved to {out_path}")


@main.command()
@click.option("--start", type=str, help="Inclusive lower bound on submitted_at (ISO-8601).")
@click.option("--end", type=str, help="Exclusive upper bound on submitted_at (ISO-8601).")
@click.option("--page-size", type=int, default=10000, show_default=True, help="Runs per page (max 50000).")
@click.option("--max-pages", type=int, help="Maximum number of cursor pages to fetch.")
@click.option("-o", "--output", type=click.Path(), default="data/export.jsonl", show_default=True, help="Output JSONL file path.")
def export(
    start: str | None,
    end: str | None,
    page_size: int,
    max_pages: int | None,
    output: str,
) -> None:
    """Bulk-download full run data from the export endpoint (gzipped JSONL)."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    click.echo(f"Exporting runs → {out_path}")
    batch: list[dict] = []
    for run in iter_export_runs(limit=page_size, start=start, end=end, max_pages=max_pages):
        batch.append(run)
        if len(batch) >= 1000:
            n = append_jsonl(out_path, batch)
            total += n
            click.echo(f"  +{n} runs (total: {total})")
            batch.clear()

    if batch:
        n = append_jsonl(out_path, batch)
        total += n

    click.echo(f"Done. {total} runs saved to {out_path}")


@main.command()
@click.argument("run_hash")
@click.option("-o", "--output", type=click.Path(), help="Output file (default: stdout).")
def get_run(run_hash: str, output: str | None) -> None:
    """Fetch a single run's full data by its hash."""
    import json

    data = get_shared_run(run_hash)
    text = json.dumps(data, indent=2)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Saved to {output}")
    else:
        click.echo(text)


if __name__ == "__main__":
    main()
