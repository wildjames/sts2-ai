from __future__ import annotations

import numpy as np
import pytest

from gbm_model.features import encode_row
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import GameState, Card, Relic


def test_encode_row_has_expected_shape_and_values():
    card_vocab = CardVocabulary(["CARD.STRIKE", "CARD.DEFEND"])
    relic_vocab = RelicVocabulary(["RELIC.ANCHOR", "RELIC.BURNING_BLOOD"])
    state = GameState(
        deck=[Card("CARD.STRIKE", floor_added=1), Card("CARD.STRIKE", floor_added=2)],
        relics=[Relic("RELIC.ANCHOR", floor_added=1)],
        potions=[],
        current_hp=70,
        max_hp=100,
        gold=20,
        floor=8,
    )

    row = encode_row("CARD.DEFEND", state, card_vocab, relic_vocab)

    assert row.shape == (1 + len(card_vocab) + len(relic_vocab) + 2,)
    assert row[0] == card_vocab["CARD.DEFEND"]
    assert row[1 + card_vocab["CARD.STRIKE"]] == 2.0
    assert row[1 + len(card_vocab) + relic_vocab["RELIC.ANCHOR"]] == 1.0
    assert row[-2] == pytest.approx(0.7)
    assert row[-1] == pytest.approx(0.16)


def test_encode_row_skip_uses_minus_one_and_no_card_match():
    card_vocab = CardVocabulary(["CARD.STRIKE"])
    relic_vocab = RelicVocabulary(["RELIC.ANCHOR"])
    state = GameState(
        deck=[], relics=[], potions=[], current_hp=50, max_hp=100, gold=0, floor=3,
    )

    row = encode_row(None, state, card_vocab, relic_vocab)

    assert row[0] == -1.0
    assert row[1] == 0.0
    assert row[1 + len(card_vocab)] == 0.0
    assert row[-2] == pytest.approx(0.5)
    assert row[-1] == pytest.approx(0.06)
