from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from sts2_card_pick.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import CardChoiceResult, GameState


def encode_state_features(
    state: GameState,
    card_vocab: CardVocabulary,
    relic_vocab: RelicVocabulary,
) -> np.ndarray:
    """Encode state-level features: deck counts, relic flags, HP ratio, floor.

    Layout: [deck_counts (|card_vocab|), relic_flags (|relic_vocab|), hp_ratio, floor]
    """
    deck_counts = np.zeros(len(card_vocab), dtype=np.float32)
    for card in state.deck:
        idx = card_vocab.get(card.id)
        if idx is not None:
            deck_counts[idx] += 1

    relic_flags = np.zeros(len(relic_vocab), dtype=np.float32)
    for relic in state.relics:
        idx = relic_vocab.get(relic.id)
        if idx is not None:
            relic_flags[idx] = 1.0

    hp_ratio = state.current_hp / state.max_hp if state.max_hp > 0 else 0.0

    return np.concatenate([
        deck_counts,
        relic_flags,
        np.array([hp_ratio, state.floor], dtype=np.float32),
    ])


def encode_card_features(
    card_id: str | None,
    card_vocab: CardVocabulary,
) -> np.ndarray:
    """One-hot encode a single card. ``None`` means skip (all zeros)."""
    features = np.zeros(len(card_vocab), dtype=np.float32)
    if card_id is not None:
        idx = card_vocab.get(card_id)
        if idx is not None:
            features[idx] = 1.0
    return features


def feature_dim(card_vocab: CardVocabulary, relic_vocab: RelicVocabulary) -> int:
    """Total width of one feature row."""
    return 2 * len(card_vocab) + len(relic_vocab) + 2


def encode_choice_set(
    state: GameState,
    card_choices: CardChoiceResult,
    card_vocab: CardVocabulary,
    relic_vocab: RelicVocabulary,
) -> tuple[csr_matrix, int]:
    """Encode a card reward screen into a sparse feature matrix and label.

    Each offered card becomes one row; skip is appended as the last row.

    Returns:
        ``(X, picked_idx)`` where ``X`` is a sparse CSR matrix of shape
        ``(n_alternatives, n_features)`` and ``picked_idx`` is the 0-based
        index of the chosen alternative.  If the player skipped,
        ``picked_idx == len(card_choices.offered)`` (the skip row).
    """
    state_features = encode_state_features(state, card_vocab, relic_vocab)
    state_nz = state_features.nonzero()[0]
    state_vals = state_features[state_nz]

    n_alts = len(card_choices.offered) + 1
    n_features = feature_dim(card_vocab, relic_vocab)
    card_offset = len(card_vocab) + len(relic_vocab) + 2

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    # Replicate state features into every row
    for r in range(n_alts):
        row_idx.extend([r] * len(state_nz))
        col_idx.extend(state_nz.tolist())
        data.extend(state_vals.tolist())

    picked_idx = len(card_choices.offered)  # default: skip

    for i, card in enumerate(card_choices.offered):
        idx = card_vocab.get(card.id)
        if idx is not None:
            row_idx.append(i)
            col_idx.append(card_offset + idx)
            data.append(1.0)
        if card_choices.picked is not None and card == card_choices.picked:
            picked_idx = i

    X = csr_matrix(
        (np.array(data, dtype=np.float32), (row_idx, col_idx)),
        shape=(n_alts, n_features),
    )
    return X, picked_idx
