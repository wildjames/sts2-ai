# sts2-downloader

CLI tool to download Slay the Spire 2 run data from the [Spire Codex API](https://spire-codex.com/docs).

## Setup

```bash
cd downloader
poetry install
```

Create a `.env` file in the repository root with your API key:

```
STS2_CODEX_TOKEN=your_token_here
```

## Usage

### List runs

Download run summaries with optional filters:

```bash
sts2-download list-runs --character IRONCLAD --win 1 --ascension-min 10 --max-pages 5
```

Options include `--character`, `--win`, `--username`, `--ascension`, `--ascension-min`, `--ascension-max`, `--game-mode`, `--players`, `--card`, `--relic`, `--sort`, `--build-id`, `--winrate-min`, `--winrate-max`, `--limit`, `--max-pages`, and `-o/--output`.

### Export (bulk download)

Download full run data (gzipped JSONL with cursor pagination):

```bash
sts2-download export --start 2025-01-01 --end 2025-02-01 --max-pages 10
```

This returns the complete game JSON per run, including `map_point_history`, deck, relics, and card choices.

### Get a single run

```bash
sts2-download get-run <run_hash>
```

## Output format

All commands write [JSONL](https://jsonlines.org/) (one JSON object per line). Default output paths are `data/runs.jsonl` and `data/export.jsonl`.

## API docs

Full endpoint documentation: https://spire-codex.com/docs
