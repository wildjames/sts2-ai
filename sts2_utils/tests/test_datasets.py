import pytest

import json
from pathlib import Path
import numpy as np

from sts2_utils.datasets import load_runs


def _minimal_run(deck_ids=None, relic_ids=None):
    deck_ids = deck_ids or []
    relic_ids = relic_ids or []
    return {
        "players": [{
            "id": 1,
            "deck": [{"id": card_id} for card_id in deck_ids],
            "relics": [{"id": relic_id} for relic_id in relic_ids],
        }],
        "map_point_history": [],
    }


# ---------- load_runs ----------

class TestLoadRuns:
    def test_load_from_jsonl(self, tmp_path):
        run1 = _minimal_run(deck_ids=["CARD.A"])
        run2 = _minimal_run(deck_ids=["CARD.B"])
        jsonl = tmp_path / "runs.jsonl"
        with open(jsonl, "w") as f:
            f.write(json.dumps(run1) + "\n")
            f.write(json.dumps(run2) + "\n")
        runs = list(load_runs(jsonl))
        assert len(runs) == 2

    def test_load_from_directory(self, tmp_path):
        run1 = _minimal_run(deck_ids=["CARD.A"])
        run2 = _minimal_run(deck_ids=["CARD.B"])
        (tmp_path / "run1.json").write_text(json.dumps(run1))
        (tmp_path / "run2.json").write_text(json.dumps(run2))
        runs = list(load_runs(tmp_path))
        assert len(runs) == 2

    def test_directory_ignores_non_json(self, tmp_path):
        run = _minimal_run()
        (tmp_path / "run.json").write_text(json.dumps(run))
        (tmp_path / "readme.txt").write_text("not a run")
        runs = list(load_runs(tmp_path))
        assert len(runs) == 1

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(ValueError):
            list(load_runs(tmp_path / "nonexistent.csv"))

    def test_skips_blank_lines_in_jsonl(self, tmp_path):
        run = _minimal_run()
        jsonl = tmp_path / "runs.jsonl"
        with open(jsonl, "w") as f:
            f.write(json.dumps(run) + "\n\n\n")
        runs = list(load_runs(jsonl))
        assert len(runs) == 1
