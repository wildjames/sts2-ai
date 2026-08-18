from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gbm_model.dataset import (
  Dataset,
  build_dataset,
  build_dataset_from_path,
  build_vocabularies_from_files
)
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import load_run

TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "sts2_utils" / "tests" / "test_data"
RUNS = TEST_DATA_DIR / "runs"
CARDS = TEST_DATA_DIR / "vocabularies" / "test_cards.json"
RELICS = TEST_DATA_DIR / "vocabularies" / "test_relics.json"


def _minimal_run(
    deck_ids=("CARD.STRIKE",),
    relic_ids=("RELIC.ANCHOR",),
    card_choices=None,
):
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
        "map_point_history": [[{"map_point_type": "monster", "player_stats": [floor_stats], "rooms": []}]],
        "players": [{"id": 1, "deck": deck, "relics": relics}],
    }


class TestBuildDataset:
    def test_build_dataset_labels_and_group_sizes(self):
        card_vocab = CardVocabulary(["CARD.STRIKE", "CARD.DEFEND"])
        relic_vocab = RelicVocabulary(["RELIC.ANCHOR"])
        run = _minimal_run(
            deck_ids=["CARD.STRIKE"],
            relic_ids=["RELIC.ANCHOR"],
            card_choices=[
                {"card": {"id": "CARD.STRIKE"}, "was_picked": True},
                {"card": {"id": "CARD.DEFEND"}, "was_picked": False},
            ],
        )

        dataset = build_dataset([run], card_vocab, relic_vocab)

        assert dataset.X.shape[0] == 3
        assert dataset.y.shape == (3,)
        assert dataset.groups.shape == (3,)
        assert dataset.group_sizes.tolist() == [3]
        assert dataset.y.sum() == 1

    def test_build_dataset_empty_runs(self):
        card_vocab = CardVocabulary(["CARD.STRIKE"])
        relic_vocab = RelicVocabulary(["RELIC.ANCHOR"])

        dataset = build_dataset([], card_vocab, relic_vocab)

        assert dataset.X.shape == (0, 1 + len(card_vocab) + len(relic_vocab) + 2)
        assert dataset.y.shape == (0,)
        assert dataset.groups.shape == (0,)
        assert dataset.group_sizes.shape == (0,)

    def test_save_and_load_round_trip(self, tmp_path):
        card_vocab = CardVocabulary(["CARD.STRIKE"])
        relic_vocab = RelicVocabulary(["RELIC.ANCHOR"])
        dataset = Dataset(
            X=np.array([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            y=np.array([1], dtype=np.int32),
            groups=np.array([0], dtype=np.int64),
            group_sizes=np.array([1], dtype=np.int64),
            card_vocab=card_vocab,
            relic_vocab=relic_vocab,
        )

        dataset.save(tmp_path / "ds")
        loaded = Dataset.load(tmp_path / "ds")

        assert np.array_equal(loaded.X, dataset.X)
        assert np.array_equal(loaded.y, dataset.y)
        assert np.array_equal(loaded.groups, dataset.groups)
        assert np.array_equal(loaded.group_sizes, dataset.group_sizes)
        assert loaded.card_vocab.ids == dataset.card_vocab.ids
        assert loaded.relic_vocab.ids == dataset.relic_vocab.ids

    def test_build_dataset_from_path(self):
        dataset, card_vocab, relic_vocab = build_dataset_from_path(RUNS, CARDS, RELICS)

        assert dataset.X.shape[0] > 0
        assert len(card_vocab) > 0
        assert len(relic_vocab) > 0
        for group_id in np.unique(dataset.groups):
            mask = dataset.groups == group_id
            assert dataset.y[mask].sum() == pytest.approx(1.0)


def test_build_vocabularies_from_files():
    card_vocab, relic_vocab = build_vocabularies_from_files(CARDS, RELICS)

    assert len(card_vocab) > 0
    assert len(relic_vocab) > 0
    assert "CARD.ABRASIVE" in card_vocab.ids
    assert "RELIC.ANCHOR" in relic_vocab.ids
