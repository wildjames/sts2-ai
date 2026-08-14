# Card Pick Prediction Model

Predicts which card a player should pick from reward screens in Slay the Spire 2, using a **conditional logit** model with L1 regularisation.

## How it works

The model frames card selection as a discrete choice problem. Given a game state (deck, relics, HP, floor) and a set of offered cards (plus skip), it estimates the probability of each alternative being the best pick:

$$P(j \mid g) = \frac{e^{X_j \beta}}{\sum_{k \in g} e^{X_k \beta}}$$

Training uses **McFadden's trick** to reduce the conditional logit to standard binary logistic regression on pairwise feature differences, fitted via scikit-learn's `LogisticRegression` with L1 penalty.

### Feature vector

Each alternative (offered card or skip) is encoded as a flat numpy array:

| Segment | Width | Description |
|---|---|---|
| Deck counts | `\|card_vocab\|` | Count of each card in the current deck |
| Relic flags | `\|relic_vocab\|` | Binary indicator for each held relic |
| HP ratio | 1 | `current_hp / max_hp` |
| Floor | 1 | Current floor number |
| Card one-hot | `\|card_vocab\|` | One-hot for the offered card (all zeros for skip) |

## Installation

```bash
cd card_pick_model
pip install -e ".[test]"
```

Requires Python ≥ 3.12.

## Quick start

### Training

```python
from sts2_card_pick import build_dataset_from_path, CardPickModel

# Build dataset from run JSONs (two-pass: vocabularies then features)
dataset, card_vocab, relic_vocab = build_dataset_from_path("../downloader/data/runs.jsonl")

# Fit model
model = CardPickModel(card_vocab, relic_vocab, C=1.0)
model.fit(dataset)

# Save for later
model.save("trained_model/")
```

### Inference

```python
from sts2_card_pick import CardPickModel
from sts2_utils import GameState, Card, Relic

model = CardPickModel.load("trained_model/")

state = GameState(
    deck=[Card(id="strike"), Card(id="defend")],
    relics=[Relic(id="ring")],
    potions=[],
    current_hp=70,
    max_hp=100,
    gold=150,
    floor=8,
)

probs = model.predict_proba(state, ["fireball", "heal", "shield_bash"])
# {'fireball': 0.62, 'heal': 0.15, 'shield_bash': 0.18, 'skip': 0.05}
```

## Package structure

```
card_pick_model/
├── src/sts2_card_pick/
│   ├── vocabulary.py   # CardVocabulary / RelicVocabulary — ID ↔ index mappings
│   ├── features.py     # Encode (GameState, card) → numpy feature vector
│   ├── dataset.py      # Walk run JSONs → Dataset(X, y, groups)
│   └── model.py        # CardPickModel — fit, predict, save/load
└── tests/
    ├── test_vocabulary.py
    ├── test_features.py
    ├── test_dataset.py
    └── test_model.py
```

### Data flow

```
Run JSONs
  │
  ▼
dataset.py ──► vocabulary.py  (pass 1: build card/relic ID registries)
  │
  ├──────────► features.py    (pass 2: encode game states → numpy arrays)
  │
  ▼
Dataset(X, y, groups)
  │
  ▼
model.py ──► CardPickModel.fit()
  │
  ▼
CardPickModel.predict_proba(state, offered_cards) → {card_id: probability}
```

## API reference

### `CardPickModel(card_vocab, relic_vocab, *, C=1.0, max_iter=1000)`

| Parameter | Description |
|---|---|
| `card_vocab` | `CardVocabulary` mapping card IDs to indices |
| `relic_vocab` | `RelicVocabulary` mapping relic IDs to indices |
| `C` | Inverse regularisation strength (higher = less regularisation) |
| `max_iter` | Maximum solver iterations |

**Methods:**

- **`fit(dataset)`** — Train on a `Dataset`. Builds pairwise difference vectors internally.
- **`predict_proba(state, offered_cards)`** — Returns `dict[str, float]` mapping each offered card ID (plus `"skip"`) to its predicted probability.
- **`save(path)`** — Serialize model coefficients and vocabularies to a directory.
- **`CardPickModel.load(path)`** — Class method to load a saved model.

### `build_dataset_from_path(path, player_id=1)`

End-to-end convenience function. Scans run files at `path` (`.jsonl` or directory of `.json` files), builds vocabularies, encodes features, and returns `(Dataset, CardVocabulary, RelicVocabulary)`.

### `Dataset`

Dataclass holding training data:

- `X` — Feature matrix, shape `(n_rows, n_features)`, `float32`
- `y` — Binary labels, shape `(n_rows,)`, exactly one `1` per group
- `groups` — Choice-set IDs, shape `(n_rows,)`, `int64`

## Running tests

```bash
cd card_pick_model
python -m pytest tests/ -v
```

## Known limitations

- Trains a single model across all characters (no per-character specialisation yet).
- Card upgrades and enchantments are ignored (uses base card ID only).
- No card synergy features (e.g. "has X and Y together").
- No hyperparameter tuning — `C` must be set manually.
- Filters to `player_id=1` only (no multi-player support).
