"""Write run data to disk as JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Append records to a JSONL file. Returns number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(records)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write records to a JSONL file (overwriting). Returns number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(records)
