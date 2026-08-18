from __future__ import annotations

import numpy as np
import pytest

from sts2_utils import Card, CardChoiceResult, GameState, Relic
from sts2_utils import CardVocabulary, RelicVocabulary
from logit_model.features import (
    encode_choice_set,
    encode_state_features,
    feature_dim,
    state_dim,
)

CARD_IDS = ["CARD.STRIKE", "CARD.DEFEND", "CARD.BASH", "CARD.INFLAME", "CARD.SHRUG"]
RELIC_IDS = ["RELIC.ANCHOR", "RELIC.BOOT", "RELIC.LANTERN"]


@pytest.fixture
def card_vocab():
    return CardVocabulary(CARD_IDS)


@pytest.fixture
def relic_vocab():
    return RelicVocabulary(RELIC_IDS)


def _make_state(
    deck_ids: list[str] | None = None,
    relic_ids: list[str] | None = None,
    current_hp: int = 60,
    max_hp: int = 80,
    gold: int = 100,
    floor: int = 5,
) -> GameState:
    deck = [Card(id=cid) for cid in (deck_ids or [])]
    relics = [Relic(id=rid) for rid in (relic_ids or [])]
    return GameState(
        deck=deck, relics=relics, potions=[], current_hp=current_hp,
        max_hp=max_hp, gold=gold, floor=floor,
    )


# ---------- encode_state_features ----------

class TestEncodeStateFeatures:
    def test_output_length(self, card_vocab, relic_vocab):
        state = _make_state()
        features = encode_state_features(state, card_vocab, relic_vocab)
        expected = len(card_vocab) + len(relic_vocab) + 2
        assert features.shape == (expected,)

    def test_deck_counts(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=["CARD.STRIKE", "CARD.STRIKE", "CARD.DEFEND"])
        features = encode_state_features(state, card_vocab, relic_vocab)
        deck_slice = features[: len(card_vocab)]
        assert deck_slice[card_vocab["CARD.STRIKE"]] == 2.0
        assert deck_slice[card_vocab["CARD.DEFEND"]] == 1.0
        assert deck_slice[card_vocab["CARD.BASH"]] == 0.0

    def test_relic_flags(self, card_vocab, relic_vocab):
        state = _make_state(relic_ids=["RELIC.ANCHOR", "RELIC.LANTERN"])
        features = encode_state_features(state, card_vocab, relic_vocab)
        relic_start = len(card_vocab)
        relic_slice = features[relic_start : relic_start + len(relic_vocab)]
        assert relic_slice[relic_vocab["RELIC.ANCHOR"]] == 1.0
        assert relic_slice[relic_vocab["RELIC.BOOT"]] == 0.0
        assert relic_slice[relic_vocab["RELIC.LANTERN"]] == 1.0

    def test_duplicate_relics_still_binary(self, card_vocab, relic_vocab):
        state = _make_state(relic_ids=["RELIC.ANCHOR", "RELIC.ANCHOR"])
        features = encode_state_features(state, card_vocab, relic_vocab)
        relic_start = len(card_vocab)
        assert features[relic_start + relic_vocab["RELIC.ANCHOR"]] == 1.0

    def test_hp_ratio(self, card_vocab, relic_vocab):
        state = _make_state(current_hp=40, max_hp=80)
        features = encode_state_features(state, card_vocab, relic_vocab)
        hp_idx = len(card_vocab) + len(relic_vocab)
        assert features[hp_idx] == pytest.approx(0.5)

    def test_hp_ratio_full_health(self, card_vocab, relic_vocab):
        state = _make_state(current_hp=80, max_hp=80)
        features = encode_state_features(state, card_vocab, relic_vocab)
        hp_idx = len(card_vocab) + len(relic_vocab)
        assert features[hp_idx] == pytest.approx(1.0)

    def test_hp_ratio_zero_max_hp(self, card_vocab, relic_vocab):
        state = _make_state(current_hp=0, max_hp=0)
        features = encode_state_features(state, card_vocab, relic_vocab)
        hp_idx = len(card_vocab) + len(relic_vocab)
        assert features[hp_idx] == 0.0

    def test_floor_value(self, card_vocab, relic_vocab):
        state = _make_state(floor=12)
        features = encode_state_features(state, card_vocab, relic_vocab)
        floor_idx = len(card_vocab) + len(relic_vocab) + 1
        assert features[floor_idx] == 12.0

    def test_unknown_card_ignored(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=["CARD.UNKNOWN_CARD", "CARD.STRIKE"])
        features = encode_state_features(state, card_vocab, relic_vocab)
        deck_slice = features[: len(card_vocab)]
        assert deck_slice.sum() == 1.0  # only STRIKE counted

    def test_unknown_relic_ignored(self, card_vocab, relic_vocab):
        state = _make_state(relic_ids=["RELIC.UNKNOWN"])
        features = encode_state_features(state, card_vocab, relic_vocab)
        relic_start = len(card_vocab)
        relic_slice = features[relic_start : relic_start + len(relic_vocab)]
        assert relic_slice.sum() == 0.0

    def test_empty_deck_and_relics(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=[], relic_ids=[])
        features = encode_state_features(state, card_vocab, relic_vocab)
        deck_slice = features[: len(card_vocab)]
        relic_start = len(card_vocab)
        relic_slice = features[relic_start : relic_start + len(relic_vocab)]
        assert deck_slice.sum() == 0.0
        assert relic_slice.sum() == 0.0

    def test_dtype_is_float32(self, card_vocab, relic_vocab):
        state = _make_state()
        features = encode_state_features(state, card_vocab, relic_vocab)
        assert features.dtype == np.float32


# ---------- feature_dim ----------

class TestFeatureDim:
    def test_value(self, card_vocab, relic_vocab):
        V = len(card_vocab)
        S = state_dim(card_vocab, relic_vocab)
        expected = V + V * S
        assert feature_dim(card_vocab, relic_vocab) == expected

    def test_matches_output_shape(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=["CARD.STRIKE"])
        choices = CardChoiceResult(
            offered=[Card(id="CARD.BASH")],
            picked=Card(id="CARD.BASH"),
        )
        X, _ = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert X.shape[1] == feature_dim(card_vocab, relic_vocab)


# ---------- encode_choice_set ----------

class TestEncodeChoiceSet:
    def test_shape_three_offers(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=["CARD.STRIKE"])
        choices = CardChoiceResult(
            offered=[Card(id="CARD.BASH"), Card(id="CARD.INFLAME"), Card(id="CARD.SHRUG")],
            picked=Card(id="CARD.INFLAME"),
        )
        X, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert X.shape == (4, feature_dim(card_vocab, relic_vocab))
        assert y == 1  # INFLAME is index 1 in offered

    def test_shape_single_offer(self, card_vocab, relic_vocab):
        state = _make_state()
        choices = CardChoiceResult(
            offered=[Card(id="CARD.STRIKE")],
            picked=Card(id="CARD.STRIKE"),
        )
        X, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert X.shape[0] == 2  # 1 card + skip

    def test_picked_index_correct(self, card_vocab, relic_vocab):
        state = _make_state()
        offered = [Card(id="CARD.STRIKE"), Card(id="CARD.DEFEND"), Card(id="CARD.BASH")]
        choices = CardChoiceResult(offered=offered, picked=Card(id="CARD.DEFEND"))
        _, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert y == 1

    def test_skip_is_last_index(self, card_vocab, relic_vocab):
        state = _make_state()
        offered = [Card(id="CARD.STRIKE"), Card(id="CARD.DEFEND")]
        choices = CardChoiceResult(offered=offered, picked=None)
        _, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert y == 2  # skip is at index len(offered)

    def test_skip_row_is_all_zeros(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=["CARD.STRIKE"])
        offered = [Card(id="CARD.BASH")]
        choices = CardChoiceResult(offered=offered, picked=None)
        X, _ = encode_choice_set(state, choices, card_vocab, relic_vocab)
        X_dense = X.toarray()
        skip_row = X_dense[-1]
        assert skip_row.sum() == 0.0

    def test_offered_row_has_correct_card_onehot(self, card_vocab, relic_vocab):
        state = _make_state()
        offered = [Card(id="CARD.BASH"), Card(id="CARD.INFLAME")]
        choices = CardChoiceResult(offered=offered, picked=Card(id="CARD.BASH"))
        X, _ = encode_choice_set(state, choices, card_vocab, relic_vocab)
        X_dense = X.toarray()
        V = len(card_vocab)
        # First row: BASH one-hot in first V columns
        assert X_dense[0, card_vocab["CARD.BASH"]] == 1.0
        assert X_dense[0, :V].sum() == 1.0
        # Second row: INFLAME one-hot in first V columns
        assert X_dense[1, card_vocab["CARD.INFLAME"]] == 1.0
        assert X_dense[1, :V].sum() == 1.0

    def test_interaction_block_contains_state(self, card_vocab, relic_vocab):
        state = _make_state(
            deck_ids=["CARD.STRIKE", "CARD.DEFEND"],
            relic_ids=["RELIC.ANCHOR"],
            current_hp=50, max_hp=75, floor=7,
        )
        offered = [Card(id="CARD.BASH"), Card(id="CARD.INFLAME")]
        choices = CardChoiceResult(offered=offered, picked=Card(id="CARD.BASH"))
        X, _ = encode_choice_set(state, choices, card_vocab, relic_vocab)
        X_dense = X.toarray()
        V = len(card_vocab)
        S = state_dim(card_vocab, relic_vocab)
        expected_state = encode_state_features(state, card_vocab, relic_vocab)
        # BASH row: interaction block at V + BASH_idx * S
        bash_idx = card_vocab["CARD.BASH"]
        bash_block = X_dense[0, V + bash_idx * S : V + (bash_idx + 1) * S]
        np.testing.assert_array_almost_equal(bash_block, expected_state)
        # INFLAME row: interaction block at V + INFLAME_idx * S
        inflame_idx = card_vocab["CARD.INFLAME"]
        inflame_block = X_dense[1, V + inflame_idx * S : V + (inflame_idx + 1) * S]
        np.testing.assert_array_almost_equal(inflame_block, expected_state)

    def test_deck_counts_in_interaction_block(self, card_vocab, relic_vocab):
        state = _make_state(deck_ids=["CARD.STRIKE", "CARD.STRIKE", "CARD.BASH"])
        choices = CardChoiceResult(
            offered=[Card(id="CARD.DEFEND")],
            picked=Card(id="CARD.DEFEND"),
        )
        X, _ = encode_choice_set(state, choices, card_vocab, relic_vocab)
        row = X.toarray()[0]
        V = len(card_vocab)
        S = state_dim(card_vocab, relic_vocab)
        defend_idx = card_vocab["CARD.DEFEND"]
        block_start = V + defend_idx * S
        # Deck counts are at the start of the state vector within the block
        assert row[block_start + card_vocab["CARD.STRIKE"]] == 2.0
        assert row[block_start + card_vocab["CARD.BASH"]] == 1.0

    def test_unknown_offered_card(self, card_vocab, relic_vocab):
        """An offered card not in the vocabulary gets an all-zero row."""
        state = _make_state()
        choices = CardChoiceResult(
            offered=[Card(id="CARD.UNKNOWN")],
            picked=Card(id="CARD.UNKNOWN"),
        )
        X, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert X.toarray()[0].sum() == 0.0
        assert y == 0

    def test_enchanted_card_match(self, card_vocab, relic_vocab):
        """Picked card with enchantment matches the same card in offered."""
        enchant = {"id": "ENCHANTMENT.GLAM", "amount": 1}
        offered = [
            Card(id="CARD.STRIKE", enchantment=enchant),
            Card(id="CARD.DEFEND"),
        ]
        picked = Card(id="CARD.STRIKE", enchantment=enchant)
        state = _make_state()
        choices = CardChoiceResult(offered=offered, picked=picked)
        _, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert y == 0

    def test_picked_matches_by_equality_not_just_id(self, card_vocab, relic_vocab):
        """When two cards share an ID but differ in floor_added, the right one is matched."""
        offered = [
            Card(id="CARD.STRIKE", floor_added=0),
            Card(id="CARD.STRIKE", floor_added=3),
        ]
        picked = Card(id="CARD.STRIKE", floor_added=3)
        state = _make_state()
        choices = CardChoiceResult(offered=offered, picked=picked)
        _, y = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert y == 1  # second STRIKE

    def test_dtype_float32(self, card_vocab, relic_vocab):
        state = _make_state()
        choices = CardChoiceResult(
            offered=[Card(id="CARD.STRIKE")],
            picked=Card(id="CARD.STRIKE"),
        )
        X, _ = encode_choice_set(state, choices, card_vocab, relic_vocab)
        assert X.dtype == np.float32


# ---------- Integration with real-ish game state ----------

class TestEncodeChoiceSetIntegration:
    """Larger test using a more realistic game state."""

    @pytest.fixture
    def big_card_vocab(self):
        ids = [f"CARD.C{i}" for i in range(200)]
        ids.extend(CARD_IDS)
        return CardVocabulary(ids)

    @pytest.fixture
    def big_relic_vocab(self):
        ids = [f"RELIC.R{i}" for i in range(50)]
        ids.extend(RELIC_IDS)
        return RelicVocabulary(ids)

    def test_realistic_dimensions(self, big_card_vocab, big_relic_vocab):
        deck = [Card(id="CARD.STRIKE")] * 5 + [Card(id="CARD.DEFEND")] * 5 + [Card(id="CARD.BASH")]
        relics = [Relic(id="RELIC.ANCHOR"), Relic(id="RELIC.R10")]
        state = GameState(
            deck=deck, relics=relics, potions=[], current_hp=65,
            max_hp=75, gold=120, floor=8,
        )
        offered = [Card(id="CARD.INFLAME"), Card(id="CARD.SHRUG"), Card(id="CARD.C50")]
        choices = CardChoiceResult(offered=offered, picked=Card(id="CARD.SHRUG"))
        X, y = encode_choice_set(state, choices, big_card_vocab, big_relic_vocab)
        assert X.shape == (4, feature_dim(big_card_vocab, big_relic_vocab))
        assert y == 1

    def test_hp_ratio_propagated(self, big_card_vocab, big_relic_vocab):
        state = _make_state(current_hp=30, max_hp=75, floor=3)
        choices = CardChoiceResult(
            offered=[Card(id="CARD.STRIKE")],
            picked=None,
        )
        X, _ = encode_choice_set(state, choices, big_card_vocab, big_relic_vocab)
        V = len(big_card_vocab)
        S = state_dim(big_card_vocab, big_relic_vocab)
        strike_idx = big_card_vocab["CARD.STRIKE"]
        hp_offset_in_state = len(big_card_vocab) + len(big_relic_vocab)
        col = V + strike_idx * S + hp_offset_in_state
        assert X.toarray()[0, col] == pytest.approx(30 / 75)
