"""HTTP client for the Spire Codex API."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Generator

import httpx
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from this file, or cwd)
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.is_file():
    load_dotenv(_env_path)
else:
    load_dotenv()

BASE_URL = "https://spire-codex.com"
LIST_ENDPOINT = "/api/runs/list"
EXPORT_ENDPOINT = "/api/exports/runs"
SHARED_ENDPOINT = "/api/runs/shared"
CARDS_ENDPOINT = "/api/cards"
RELICS_ENDPOINT = "/api/relics"

DEFAULT_TIMEOUT = 120.0


def _headers() -> dict[str, str]:
    token = os.environ.get("STS2_CODEX_TOKEN")
    if token:
        return {"X-API-Key": token}
    return {}


def list_runs(
    *,
    character: str | None = None,
    win: str | None = None,
    username: str | None = None,
    ascension: int | None = None,
    ascension_min: int | None = None,
    ascension_max: int | None = None,
    game_mode: str | None = None,
    players: str | None = None,
    card: str | None = None,
    relic: str | None = None,
    sort: str | None = None,
    build_id: str | None = None,
    winrate_min: float | None = None,
    winrate_max: float | None = None,
    page: int = 1,
    limit: int = 50,
) -> dict[str, Any]:
    """Fetch a single page from /api/runs/list."""
    params: dict[str, Any] = {"page": page, "limit": limit}
    for key, val in {
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
    }.items():
        if val is not None:
            params[key] = val

    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT, headers=_headers()) as client:
        resp = client.get(LIST_ENDPOINT, params=params)
        resp.raise_for_status()
        return resp.json()


def iter_list_pages(
    *,
    max_pages: int | None = None,
    **filters: Any,
) -> Generator[list[dict[str, Any]], None, None]:
    """Yield pages of run summaries from the list endpoint."""
    page = 1
    while True:
        data = list_runs(page=page, **filters)
        runs = data.get("runs", [])
        if not runs:
            break
        yield runs
        if page >= data.get("total_pages", 1):
            break
        if max_pages and page >= max_pages:
            break
        page += 1


def get_shared_run(run_hash: str) -> dict[str, Any]:
    """Fetch a single run's full data by hash."""
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT, headers=_headers()) as client:
        resp = client.get(f"{SHARED_ENDPOINT}/{run_hash}")
        resp.raise_for_status()
        return resp.json()


def iter_export_runs(
    *,
    limit: int | None = None,
    start: str | None = None,
    end: str | None = None,
    max_pages: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Stream runs from the bulk export endpoint (gzipped JSONL).

    Handles cursor-based pagination via the X-Next-Cursor header.
    """
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    cursor: str | None = None
    pages_fetched = 0
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT, headers=_headers()) as client:
        while True:
            req_params = dict(params)
            if cursor:
                req_params["cursor"] = cursor

            resp = client.get(EXPORT_ENDPOINT, params=req_params)
            resp.raise_for_status()
            pages_fetched += 1

            content = resp.content
            try:
                decompressed = gzip.decompress(content)
            except gzip.BadGzipFile:
                decompressed = content

            for line in decompressed.decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)

            next_cursor = resp.headers.get("X-Next-Cursor")
            if not next_cursor:
                break
            if max_pages and pages_fetched >= max_pages:
                break
            cursor = next_cursor


def get_cards(*, color: str | None = None) -> list[dict[str, Any]]:
    """Fetch all cards from /api/cards."""
    params: dict[str, Any] = {}
    if color is not None:
        params["color"] = color
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT, headers=_headers()) as client:
        resp = client.get(CARDS_ENDPOINT, params=params)
        resp.raise_for_status()
        return resp.json()


def get_relics(*, pool: str | None = None) -> list[dict[str, Any]]:
    """Fetch all relics from /api/relics."""
    params: dict[str, Any] = {}
    if pool is not None:
        params["pool"] = pool
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT, headers=_headers()) as client:
        resp = client.get(RELICS_ENDPOINT, params=params)
        resp.raise_for_status()
        return resp.json()
