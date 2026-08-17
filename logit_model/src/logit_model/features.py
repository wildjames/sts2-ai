from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from logit_model.vocabulary import CardVocabulary, RelicVocabulary
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

    hp_ratio = state.current_hp / state.max_hp if state.max_hp else 0.0
    floor_value = float(state.floor)

    return np.concatenate([
        deck_counts,
        relic_flags,
        np.array([hp_ratio, floor_value], dtype=np.float32),
    ])


def state_dim(card_vocab: CardVocabulary, relic_vocab: RelicVocabulary) -> int:
    """Width of the state feature vector."""
    return len(card_vocab) + len(relic_vocab) + 2


def feature_dim(card_vocab: CardVocabulary, relic_vocab: RelicVocabulary) -> int:
    """Total width of one feature row: card_onehot + per-card interaction blocks."""
    V = len(card_vocab)
    S = state_dim(card_vocab, relic_vocab)
    return V + V * S


def encode_choice_set(
    state: GameState,
    card_choices: CardChoiceResult,
    card_vocab: CardVocabulary,
    relic_vocab: RelicVocabulary,
) -> tuple[csr_matrix, int]:
    """Encode a card reward screen into a sparse feature matrix and label.

    Layout per row: [card_onehot (V), interaction_block_0 (S), ..., interaction_block_{V-1} (S)]
    For offered card with vocab index k, the card_onehot and interaction_block_k
    are populated; all other blocks are zero. Skip row is all zeros.

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
    V = len(card_vocab)
    S = state_dim(card_vocab, relic_vocab)

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    picked_idx = len(card_choices.offered)  # default: skip

    for i, card in enumerate(card_choices.offered):
        idx = card_vocab.get(card.id)
        if idx is not None:
            # Card one-hot
            row_idx.append(i)
            col_idx.append(idx)
            data.append(1.0)
            # State features in this card's interaction block
            interaction_offset = V + idx * S
            row_idx.extend([i] * len(state_nz))
            col_idx.extend((interaction_offset + state_nz).tolist())
            data.extend(state_vals.tolist())
        if card_choices.picked is not None and card == card_choices.picked:
            picked_idx = i

    # Skip row is all zeros (reference alternative)

    X = csr_matrix(
        (np.array(data, dtype=np.float32), (row_idx, col_idx)),
        shape=(n_alts, n_features),
    )
    return X, picked_idx
