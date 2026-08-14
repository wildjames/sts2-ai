from __future__ import annotations

import numpy as np

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
) -> tuple[np.ndarray, int]:
    """Encode a card reward screen into a feature matrix and label.

    Each offered card becomes one row; skip is appended as the last row.

    Returns:
        ``(X, y)`` where ``X`` has shape ``(n_alternatives, n_features)`` and
        ``y`` is the 0-based index of the chosen alternative.  If the player
        skipped, ``y == len(card_choices.offered)`` (the skip row).
    """
    state_features = encode_state_features(state, card_vocab, relic_vocab)

    rows: list[np.ndarray] = []
    picked_idx = len(card_choices.offered)  # default: skip

    for i, card in enumerate(card_choices.offered):
        card_features = encode_card_features(card.id, card_vocab)
        rows.append(np.concatenate([state_features, card_features]))
        if card_choices.picked is not None and card == card_choices.picked:
            picked_idx = i

    # Skip alternative: zero card features, same state features
    skip_features = encode_card_features(None, card_vocab)
    rows.append(np.concatenate([state_features, skip_features]))

    X = np.stack(rows)
    return X, picked_idx
