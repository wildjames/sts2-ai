from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from logit_model.dataset import (
    Dataset,
    _collect_ids_from_run,
    build_dataset,
    build_dataset_from_path,
    build_vocabularies_from_files,
)
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import load_run

TEST_DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "sts2_utils" / "tests" / "test_data"
DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "data"

RUNS = TEST_DATA_DIR / "runs"

CARDS = TEST_DATA_DIR / "vocabularies" / "test_cards.json"
RELICS = TEST_DATA_DIR / "vocabularies" / "test_relics.json"


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


# ---------- build_dataset (integration with real data) ----------

@pytest.fixture
def real_run():
    return load_run(RUNS / "regent_data_1.json")


class TestBuildDataset:
    def test_basic_shape(self, real_run):
        card_vocab, relic_vocab = build_vocabularies_from_files(CARDS, RELICS)
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        assert len(dataset.X.shape) == 2
        assert dataset.y.ndim == 1
        assert dataset.groups.ndim == 1
        assert dataset.X.shape[0] == dataset.y.shape[0] == dataset.groups.shape[0]

    def test_labels_sum_to_one_per_group(self, real_run):
        card_vocab, relic_vocab = build_vocabularies_from_files(CARDS, RELICS)
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        for gid in np.unique(dataset.groups):
            mask = dataset.groups == gid
            assert dataset.y[mask].sum() == pytest.approx(1.0)

    def test_groups_are_sequential(self, real_run):
        card_vocab, relic_vocab = build_vocabularies_from_files(CARDS, RELICS)
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        unique = np.unique(dataset.groups)
        assert unique[0] == 0
        assert len(unique) == unique[-1] + 1

    def test_feature_width(self, real_run):
        card_vocab, relic_vocab = build_vocabularies_from_files(CARDS, RELICS)
        dataset = build_dataset([real_run], card_vocab, relic_vocab)
        from logit_model.features import feature_dim
        expected = feature_dim(card_vocab, relic_vocab)
        assert dataset.X.shape[1] == expected

    def test_empty_runs(self):
        card_vocab = CardVocabulary(["CARD.A"])
        relic_vocab = RelicVocabulary(["RELIC.X"])
        dataset = build_dataset([], card_vocab, relic_vocab)
        from logit_model.features import feature_dim
        assert dataset.X.shape == (0, feature_dim(card_vocab, relic_vocab))
        assert dataset.y.shape == (0,)
        assert dataset.groups.shape == (0,)

    def test_multiple_runs(self):
        runs = [
            load_run(RUNS / "regent_data_1.json"),
            load_run(RUNS / "necro_data_1.json"),
        ]
        card_vocab, relic_vocab = build_vocabularies_from_files(CARDS, RELICS)
        dataset = build_dataset(runs, card_vocab, relic_vocab)
        assert dataset.X.shape[0] > 0
        # Every group still sums to 1
        for gid in np.unique(dataset.groups):
            mask = dataset.groups == gid
            assert dataset.y[mask].sum() == pytest.approx(1.0)


# ---------- build_dataset_from_path ----------

class TestBuildDatasetFromPath:
    def test_end_to_end(self):
        dataset, card_vocab, relic_vocab = build_dataset_from_path(
            RUNS, CARDS, RELICS,
        )
        assert dataset.X.shape[0] > 0
        assert len(card_vocab) > 0
        assert len(relic_vocab) > 0
        for gid in np.unique(dataset.groups):
            mask = dataset.groups == gid
            assert dataset.y[mask].sum() == pytest.approx(1.0)
