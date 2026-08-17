import numpy as np
from sts2_utils import GameState, CardVocabulary, RelicVocabulary

def encode_row(
    card_id: str | None,
    state: GameState,
    card_vocab: CardVocabulary,
    relic_vocab: RelicVocabulary,
) -> np.ndarray:
    """Encode a single alternative as a dense feature vector.

    Returns: array of shape (1 + V + R + 2,)
      [card_idx, deck_count_0, ..., deck_count_{V-1},
       relic_flag_0, ..., relic_flag_{R-1}, hp_ratio, floor]
    """
    card_features = np.zeros(len(card_vocab, dtype=np.float32))
    if card_id is not None:
        idx = card_vocab.get(card_id)
        if idx is not None:
            card_features[idx] = 1

    relic_flags = np.zeros(len(relic_vocab), dtype=np.float32)
    for relic in state.relics:
        idx = relic_vocab.get(relic.id)
        if idx is not None:
            relic_flags[idx] = 1.0


    hp_ratio = state.current_hp / 100.0
    floor_ratio = state.floor / 50.0

    return np.concatenate([
        [card_vocab.get(card_id) if card_id is not None else -1],
        card_features,
        relic_flags,
        np.array([hp_ratio, floor_ratio], dtype=np.float32),
    ])
