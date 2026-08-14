from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sts2_card_pick.dataset import (
    Dataset,
    _collect_ids_from_run,
    build_dataset,
    build_dataset_from_path,
    build_vocabularies,
    load_runs,
)
from sts2_card_pick.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import load_run

REAL_DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "sts2_utils" / "tests" / "test_data"


def _minimal_run(
    deck_ids=("CARD.STRIKE",),
    relic_ids=("RELIC.ANCHOR",),
    card_choices=None,
):
    """Build a minimal run dict with a single floor."""
    deck = [{"id": cid, "floor_added_to_deck": 1} for cid in deck_ids]
    relics = [{"id": rid, "floor_added_to_deck": 1} for rid in relic_ids]

    floor_stats = {
        "player_id": 1,
        "current_hp": 70,
        "max_hp": 80,
        "current_gold": 100,
        "damage_taken": 0,
        "hp_healed": 0,
        "max_hp_lost": 0,
        "max_hp_gained": 0,
        "gold_gained": 0,
        "gold_spent": 0,
        "gold_lost": 0,
        "card_choices": card_choices or [],
        "cards_transformed": [],
        "cards_removed": [],
        "cards_gained": [],
        "upgraded_cards": [],
        "cards_enchanted": [],
        "relic_choices": [],
        "potion_used": [],
        "potion_discarded": [],
        "potion_choices": [],
    }

    return {
        "map_point_history": [[{
            "map_point_type": "monster",
            "player_stats": [floor_stats],
            "rooms": [],
        }]],
        "players": [{"id": 1, "deck": deck, "relics": relics}],
    }


# ---------- _collect_ids_from_run ----------

class TestCollectIds:
    def test_collects_deck_and_relic_ids(self):
        run = _minimal_run(
            deck_ids=["CARD.A", "CARD.B"],
            relic_ids=["RELIC.X", "RELIC.Y"],
        )
        cards, relics = _collect_ids_from_run(run)
        assert cards == {"CARD.A", "CARD.B"}
        assert relics == {"RELIC.X", "RELIC.Y"}

    def test_collects_card_choice_ids(self):
        choices = [{"card": {"id": "CARD.OFFERED"}, "was_picked": False}]
        run = _minimal_run(deck_ids=["CARD.A"], card_choices=choices)
        cards, _ = _collect_ids_from_run(run)
        assert "CARD.OFFERED" in cards
        assert "CARD.A" in cards


# ---------- build_vocabularies ----------

class TestBuildVocabularies:
    def test_returns_sorted_vocabularies(self):
        run1 = _minimal_run(deck_ids=["CARD.B", "CARD.A"], relic_ids=["RELIC.Y"])
        run2 = _minimal_run(deck_ids=["CARD.C"], relic_ids=["RELIC.X"])
        card_vocab, relic_vocab = build_vocabularies([run1, run2])
        assert card_vocab.ids == ["CARD.A", "CARD.B", "CARD.C"]
        assert relic_vocab.ids == ["RELIC.X", "RELIC.Y"]

    def test_empty_runs(self):
        card_vocab, relic_vocab = build_vocabularies([])
        assert len(card_vocab) == 0
        assert len(relic_vocab) == 0


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


# ---------- build_dataset (integration with real data) ----------

@pytest.fixture
def real_run():
    return load_run(REAL_DATA_DIR / "regent_data_1.json")


class TestBuildDataset:
    def test_basic_shape(self, real_run):
        card_vocab, relic_vocab = build_vocabularies([real_run])
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        assert dataset.X.ndim == 2
        assert dataset.y.ndim == 1
        assert dataset.groups.ndim == 1
        assert dataset.X.shape[0] == dataset.y.shape[0] == dataset.groups.shape[0]

    def test_labels_sum_to_one_per_group(self, real_run):
        card_vocab, relic_vocab = build_vocabularies([real_run])
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        for gid in np.unique(dataset.groups):
            mask = dataset.groups == gid
            assert dataset.y[mask].sum() == pytest.approx(1.0)

    def test_groups_are_sequential(self, real_run):
        card_vocab, relic_vocab = build_vocabularies([real_run])
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        unique = np.unique(dataset.groups)
        assert unique[0] == 0
        assert len(unique) == unique[-1] + 1

    def test_feature_width(self, real_run):
        card_vocab, relic_vocab = build_vocabularies([real_run])
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        expected = 2 * len(card_vocab) + len(relic_vocab) + 2
        assert dataset.X.shape[1] == expected

    def test_empty_runs(self):
        card_vocab = CardVocabulary(["CARD.A"])
        relic_vocab = RelicVocabulary(["RELIC.X"])
        dataset = build_dataset([], card_vocab, relic_vocab)
        assert dataset.X.shape == (0, 2 * 1 + 1 + 2)
        assert dataset.y.shape == (0,)
        assert dataset.groups.shape == (0,)

    def test_multiple_runs(self):
        runs = [
            load_run(REAL_DATA_DIR / "regent_data_1.json"),
            load_run(REAL_DATA_DIR / "necro_data_1.json"),
        ]
        card_vocab, relic_vocab = build_vocabularies(runs)
        dataset = build_dataset(runs, card_vocab, relic_vocab)
        assert dataset.X.shape[0] > 0
        # Every group still sums to 1
        for gid in np.unique(dataset.groups):
            mask = dataset.groups == gid
            assert dataset.y[mask].sum() == pytest.approx(1.0)


# ---------- build_dataset_from_path ----------

class TestBuildDatasetFromPath:
    def test_end_to_end(self):
        dataset, card_vocab, relic_vocab = build_dataset_from_path(REAL_DATA_DIR)
        assert dataset.X.shape[0] > 0
        assert len(card_vocab) > 0
        assert len(relic_vocab) > 0
        for gid in np.unique(dataset.groups):
            mask = dataset.groups == gid
            assert dataset.y[mask].sum() == pytest.approx(1.0)
