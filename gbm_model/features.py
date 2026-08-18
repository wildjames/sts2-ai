from __future__ import annotations

import numpy as np

# TODO: This should live in the utils module
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import GameState


def encode_row(
    card_id: str | None,
    state: GameState,
    card_vocab: CardVocabulary,
    relic_vocab: RelicVocabulary,
) -> np.ndarray:
    """Encode a single alternative as a dense feature vector.

    Layout:
    [
        card_idx,
        deck_count_0, ..., deck_count_{V-1},
        relic_flag_0, ..., relic_flag_{R-1},
        hp_ratio, floor
    ]
    """
    V = len(card_vocab)
    R = len(relic_vocab)

    deck_counts = np.zeros(V, dtype=np.float32)
    for card in state.deck:
        idx = card_vocab.get(card.id)
        if idx is not None:
            deck_counts[idx] += 1.0

    relic_flags = np.zeros(R, dtype=np.float32)
    for relic in state.relics:
        idx = relic_vocab.get(relic.id)
        if idx is not None:
            relic_flags[idx] = 1.0

    card_idx = card_vocab.get(card_id) if card_id is not None else None
    card_idx_float = float(card_idx) if card_idx is not None else -1.0

    hp_ratio = float(state.current_hp) / 100.0
    floor_value = float(state.floor) / 50.0

    return np.concatenate([
        np.array([card_idx_float], dtype=np.float32),
        deck_counts,
        relic_flags,
        np.array([hp_ratio, floor_value], dtype=np.float32),
    ], dtype=np.float32)
