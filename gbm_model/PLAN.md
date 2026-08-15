# LightGBM LambdaRank Card Pick Model

## Goal

Replace the conditional logit model with a LightGBM ranker that predicts which
card a player picks from a reward screen, given the current game state.

## Why LightGBM over the logit model

The logit model encodes card×state interactions as explicit per-card interaction
blocks (V × S ≈ 500 × 700 = 350,000 parameters).  L1 regularisation zeros most
of them out, and varying C has no measurable effect on accuracy — the model
collapses to learning a flat card ranking with no context sensitivity.

Tree-based models discover interactions automatically via splits, don't require
pre-specified interaction terms, and handle mixed float/categorical features
natively.  LightGBM's `lambdarank` objective directly optimises for ranking the
chosen card above the alternatives.

---

## Architecture

### Data format

LightGBM ranking expects **pointwise** rows grouped by query.  Each row is one
alternative (offered card or skip) in a choice set, with a relevance label.
This is the same logical structure as the existing `Dataset`, but with dense
features instead of sparse interaction-block features.

```
group_id | label | card_id (cat) | deck_count_card_0 | ... | relic_flag_0 | ... | hp_ratio | floor
```

- **label**: 1 for the picked card, 0 for the rest (binary relevance).
- **group**: the choice-set ID (same as `Dataset.groups`).
- **card_id**: categorical feature — the index of the offered card in the
  vocabulary (or a sentinel for skip).

### Feature vector per row

| Section          | Width | Type  | Description                                |
|------------------|-------|-------|--------------------------------------------|
| card_id          | 1     | cat   | Vocab index of offered card (skip = -1)    |
| deck_counts      | V     | float | Count of each card currently in deck       |
| relic_flags      | R     | float | 1.0 if player has relic, else 0.0          |
| hp_ratio         | 1     | float | current_hp / 100                           |
| floor            | 1     | float | Current floor number                       |

Total: 1 + V + R + 2 features per row.  No interaction blocks — LightGBM will
learn card×state interactions through tree splits.

**Why card_id as a categorical**: LightGBM has native categorical split support.
A single categorical feature replaces the V-wide one-hot, letting the trees
partition cards into groups (e.g. "all draw cards") at internal nodes rather
than testing one card at a time.

### Model

```python
lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=...,       # tune
    num_leaves=...,         # tune
    learning_rate=...,      # tune
    min_child_samples=...,  # tune
    categorical_feature=[0],  # card_id column
)
```

Fit with `group` parameter = array of group sizes (number of alternatives per
choice set).

---

## Implementation plan

### Phase 1 — Package scaffolding

Create the package structure mirroring `logit_model`:

```
gbm_model/
  pyproject.toml          # deps: lightgbm, click, numpy, sts2-utils
  README.md
  src/
    gbm_model/
      __init__.py
      __main__.py
      cli.py              # preprocess, train, evaluate, predict
      dataset.py           # dense feature matrix + groups + labels
      features.py          # encode_row() — flat feature vector per alternative
      model.py             # GBMCardPickModel wrapping LGBMRanker
  tests/
    __init__.py
    test_features.py
    test_model.py
```

Register in root `pyproject.toml` as a workspace member.

### Phase 2 — Feature encoding (`features.py`)

Reuse `sts2_utils.GameState`, `CardVocabulary`, `RelicVocabulary` from the
existing packages.

```python
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
```

- `card_id=None` → skip row, card_idx = -1
- Unknown cards → card_idx = -1 (treated same as skip by the tree)

### Phase 3 — Dataset construction (`dataset.py`)

```python
@dataclass
class Dataset:
    X: np.ndarray           # (n_rows, n_features), float32
    y: np.ndarray           # (n_rows,), int32 labels
    groups: np.ndarray      # (n_rows,), choice-set IDs
    group_sizes: np.ndarray # (n_groups,), rows per group (for LGBMRanker)
    card_vocab: CardVocabulary
    relic_vocab: RelicVocabulary
```

- `build_dataset()` iterates runs → floors → card choices, calls `encode_row`
  per alternative, stacks into dense matrix.
- `save()`/`load()` persist as `.npy` + vocab JSON.
- `split()` splits by group (same logic as logit_model).
- `group_sizes` computed from `groups` via `np.bincount` or `np.diff(searchsorted)`.

### Phase 4 — Model wrapper (`model.py`)

```python
class GBMCardPickModel:
    def __init__(self, card_vocab, relic_vocab, **lgb_params): ...

    def fit(self, train: Dataset, eval: Dataset | None = None) -> None:
        """Fit LGBMRanker on training data, optional early stopping on eval."""
        self._model = lgb.LGBMRanker(**self._params)
        self._model.fit(
            train.X, train.y,
            group=train.group_sizes,
            eval_set=[(eval.X, eval.y)] if eval else None,
            eval_group=[eval.group_sizes] if eval else None,
            categorical_feature=[0],
        )

    def predict_scores(self, state, offered_cards) -> dict[str, float]:
        """Score each card + skip. Return raw LightGBM scores."""

    def predict_proba(self, state, offered_cards) -> dict[str, float]:
        """Softmax over scores for probabilities."""

    def save(self, path) -> None:
        """Save model via model.booster_.save_model() + vocab JSON."""

    @classmethod
    def load(cls, path) -> GBMCardPickModel:
        """Load from booster file + vocabs."""
```

### Phase 5 — CLI (`cli.py`)

Same command structure as logit_model for consistency:

| Command      | Description                                           |
|--------------|-------------------------------------------------------|
| `preprocess` | JSONL/dir → Dataset (dense features)                  |
| `split`      | Dataset → train/eval by group                         |
| `train`      | Dataset → fitted model (with optional eval set for early stopping) |
| `evaluate`   | Model + dataset → NDCG@1, top-1 accuracy, mean log-likelihood |
| `predict`    | JSON input → probabilities                            |

New vs logit_model:
- `train` accepts `--eval-dataset` for early stopping.
- `train` accepts LightGBM hyperparameters: `--n-estimators`, `--num-leaves`,
  `--learning-rate`, `--min-child-samples`.
- `evaluate` reports NDCG@1 in addition to top-1 accuracy.

### Phase 6 — Hyperparameter tuning

After the basic pipeline works, sweep over:

| Parameter          | Range to try                |
|--------------------|-----------------------------|
| `n_estimators`     | 100, 300, 500, 1000        |
| `num_leaves`       | 31, 63, 127, 255           |
| `learning_rate`    | 0.01, 0.05, 0.1            |
| `min_child_samples`| 20, 50, 100                |

Use early stopping on the eval set (`callbacks=[lgb.early_stopping(50)]`)
to avoid overfitting — `n_estimators` becomes the upper bound.

---

## What can be reused from `logit_model`

| Component                   | Reuse? | Notes                                    |
|-----------------------------|--------|------------------------------------------|
| `sts2_utils.GameState`      | Yes    | Unchanged                                |
| `logit_model.vocabulary`    | Yes    | Import `CardVocabulary`, `RelicVocabulary` from `logit_model` or copy |
| `logit_model.dataset.load_runs` | Yes | Run loading logic is model-independent  |
| `sts2_utils.build_game_state` | Yes  | State reconstruction is shared           |
| `sts2_utils.get_card_choices` | Yes  | Choice extraction is shared              |
| `logit_model.features`      | No     | New flat encoding replaces interaction blocks |
| `logit_model.model`         | No     | Entirely new model class                 |

Best approach: depend on `logit_model` for vocabulary classes and `sts2_utils`
for game state, write new features and model code.

---

## Success criteria

- Top-1 accuracy **exceeds** the logit model baseline on the same eval set.
- Varying hyperparameters produces **measurably different** results (confirming
  the model is actually learning context-dependent behavior, unlike the logit
  model where all C values converged).
- Feature importance from LightGBM shows non-trivial contribution from state
  features (deck counts, relic flags), not just card identity.
