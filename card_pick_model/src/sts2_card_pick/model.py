from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from sts2_card_pick.dataset import Dataset
from sts2_card_pick.features import encode_card_features, encode_state_features
from sts2_card_pick.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import GameState


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
    ) -> None:
        self.card_vocab = card_vocab
        self.relic_vocab = relic_vocab
        self._C = C
        self._max_iter = max_iter
        self._model = LogisticRegression(
            C=C,
            l1_ratio=1.0,
            solver="liblinear",
            fit_intercept=False,
            max_iter=max_iter,
        )
        self._fitted = False

    def fit(self, dataset: Dataset) -> None:
        """Fit the model on a :class:`Dataset`.

        For each choice set, creates pairwise differences
        ``X_chosen - X_other`` (label 1) and the reverse (label 0) so that
        sklearn sees both classes.
        """
        pos_rows: list[np.ndarray] = []
        neg_rows: list[np.ndarray] = []

        for g in np.unique(dataset.groups):
            mask = dataset.groups == g
            X_g = dataset.X[mask]
            y_g = dataset.y[mask]

            chosen_mask = y_g == 1
            if not chosen_mask.any():
                continue

            x_chosen = X_g[chosen_mask][0]
            for x_other in X_g[~chosen_mask]:
                diff = x_chosen - x_other
                pos_rows.append(diff)
                neg_rows.append(-diff)

        X_diff = np.vstack(pos_rows + neg_rows)
        y_diff = np.concatenate([
            np.ones(len(pos_rows), dtype=np.float32),
            np.zeros(len(neg_rows), dtype=np.float32),
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

        beta = self._model.coef_.ravel()
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
            "coef": self._model.coef_.tolist(),
            "classes": self._model.classes_.tolist(),
            "C": self._C,
            "max_iter": self._max_iter,
        }
        with open(path / "model.json", "w") as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, path: str | Path) -> CardPickModel:
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
        )

        instance._model.classes_ = np.array(model_data["classes"])
        instance._model.coef_ = np.array(model_data["coef"])
        instance._model.intercept_ = np.array([0.0])
        instance._fitted = True

        return instance
