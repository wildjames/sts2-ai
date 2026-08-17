import json
from pathlib import Path
from typing import Iterable, Iterator

def load_runs(path: str | Path) -> Iterator[dict]:
    """Yield run dicts from a ``.jsonl`` file or a directory of ``.json`` files."""
    path = Path(path)
    if path.is_file() and path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.is_dir():
        for json_file in sorted(path.glob("*.json")):
            with open(json_file) as f:
                yield json.load(f)
    else:
        raise ValueError(f"Expected a .jsonl file or a directory, got: {path}")
