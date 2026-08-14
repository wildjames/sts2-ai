from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from sts2_card_pick.dataset import Dataset
from sts2_card_pick.features import encode_card_features, encode_state_features
from sts2_card_pick.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import GameState


def _to_numpy(arr: object) -> np.ndarray:
    """Convert cupy/cuML arrays to numpy; pass through numpy arrays."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _make_logistic_regression(
    C: float, max_iter: int, *, gpu: bool
) -> LogisticRegression:
    if gpu:
        from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

        return CuMLLogisticRegression(
            C=C,
            penalty="l1",
            fit_intercept=False,
            max_iter=max_iter,
        )
    return LogisticRegression(
        C=C,
        l1_ratio=1.0,
        solver="saga",
        fit_intercept=False,
        max_iter=max_iter,
    )


class CardPickModel:
    """Conditional logit card-pick model with L1 regularisation.

    Uses McFadden's trick to reduce the conditional logit to standard binary
    logistic regression on pairwise difference vectors.
    """

    def __init__(
        self,
        card_vocab: CardVocabulary,
        relic_vocab: RelicVocabulary,
        *,
        C: float = 1.0,
        max_iter: int = 1000,
        gpu: bool = False,
    ) -> None:
        self.card_vocab = card_vocab
        self.relic_vocab = relic_vocab
        self._C = C
        self._max_iter = max_iter
        self._gpu = gpu
        self._model = _make_logistic_regression(C, max_iter, gpu=gpu)
        self._fitted = False

    def fit(self, dataset: Dataset) -> None:
        """Fit the model on a :class:`Dataset`.

        For each choice set, creates pairwise differences
        ``X_chosen - X_other`` (label 1) and the reverse (label 0) so that
        sklearn sees both classes.  Uses vectorized sparse indexing to avoid
        per-group Python loops.
        """
        chosen_mask = dataset.y == 1
        chosen_indices = np.where(chosen_mask)[0]
        chosen_groups = dataset.groups[chosen_indices]

        # Map group_id -> row index of chosen card
        max_group = int(dataset.groups.max())
        group_to_chosen = np.full(max_group + 1, -1, dtype=np.intp)
        group_to_chosen[chosen_groups] = chosen_indices

        # Non-chosen rows whose group has a valid chosen row
        non_chosen_indices = np.where(~chosen_mask)[0]
        matched_chosen = group_to_chosen[dataset.groups[non_chosen_indices]]
        valid = matched_chosen >= 0
        non_chosen_indices = non_chosen_indices[valid]
        matched_chosen = matched_chosen[valid]

        # Vectorized sparse subtraction: chosen - other
        X_pos = dataset.X[matched_chosen] - dataset.X[non_chosen_indices]

        X_diff = sparse.vstack([X_pos, -X_pos], format="csr")
        y_diff = np.concatenate([
            np.ones(X_pos.shape[0], dtype=np.float32),
            np.zeros(X_pos.shape[0], dtype=np.float32),
        ])

        self._model.fit(X_diff, y_diff)
        self._fitted = True

    def predict_proba(
        self,
        state: GameState,
        offered_cards: list[str],
    ) -> dict[str, float]:
        """Return pick probabilities for each offered card and skip."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")

        beta = _to_numpy(self._model.coef_).ravel()
        state_feat = encode_state_features(state, self.card_vocab, self.relic_vocab)

        keys: list[str] = []
        scores: list[float] = []
        for card_id in offered_cards:
            card_feat = encode_card_features(card_id, self.card_vocab)
            x = np.concatenate([state_feat, card_feat])
            keys.append(card_id)
            scores.append(float(x @ beta))

        # Skip alternative
        skip_feat = encode_card_features(None, self.card_vocab)
        x_skip = np.concatenate([state_feat, skip_feat])
        keys.append("skip")
        scores.append(float(x_skip @ beta))

        # Softmax
        scores_arr = np.array(scores)
        scores_arr -= scores_arr.max()
        exp_scores = np.exp(scores_arr)
        probs = exp_scores / exp_scores.sum()

        return dict(zip(keys, probs.tolist()))

    def save(self, path: str | Path) -> None:
        """Persist fitted model + vocabularies to *path* (a directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.card_vocab.to_json(path / "card_vocab.json")
        self.relic_vocab.to_json(path / "relic_vocab.json")

        model_data = {
            "coef": _to_numpy(self._model.coef_).tolist(),
            "classes": _to_numpy(self._model.classes_).tolist(),
            "C": self._C,
            "max_iter": self._max_iter,
        }
        with open(path / "model.json", "w") as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, path: str | Path, *, gpu: bool = False) -> CardPickModel:
        """Load a previously saved model."""
        path = Path(path)

        card_vocab = CardVocabulary.from_json(path / "card_vocab.json")
        relic_vocab = RelicVocabulary.from_json(path / "relic_vocab.json")

        with open(path / "model.json") as f:
            model_data = json.load(f)

        instance = cls(
            card_vocab=card_vocab,
            relic_vocab=relic_vocab,
            C=model_data["C"],
            max_iter=model_data["max_iter"],
            gpu=gpu,
        )

        instance._model.classes_ = np.array(model_data["classes"])
        instance._model.coef_ = np.array(model_data["coef"])
        instance._model.intercept_ = np.array([0.0])
        instance._fitted = True

        return instance
