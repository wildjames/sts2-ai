from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from logit_model.dataset import Dataset
from sts2_utils.features import encode_state_features, state_dim
from sts2_utils import CardVocabulary, RelicVocabulary
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
        penalty="l1",
        solver="liblinear",
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
        self._progress_callback: object | None = None

    def fit(self, dataset: Dataset) -> None:
        """Fit the model on a :class:`Dataset`.

        For each choice set, creates pairwise differences
        ``X_chosen - X_other`` (label 1).  A single dummy row with label 0
        ensures sklearn sees both classes (the symmetric negative is
        mathematically redundant for logistic regression without intercept).
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

        n_pairs = len(non_chosen_indices)
        n_features = dataset.X.shape[1]

        # Pre-allocate output CSR arrays (different cards have non-overlapping
        # columns, so nnz of difference = sum of nnz of the two rows).
        row_nnz = np.diff(dataset.X.indptr)
        pair_nnz = row_nnz[matched_chosen] + row_nnz[non_chosen_indices]
        total_nnz = int(pair_nnz.sum())
        del row_nnz

        out_data = np.empty(total_nnz, dtype=np.float32)
        out_indices = np.empty(total_nnz, dtype=np.int32)
        out_indptr = np.empty(n_pairs + 2, dtype=np.int64)  # +1 for dummy row
        out_indptr[0] = 0
        np.cumsum(pair_nnz, out=out_indptr[1:n_pairs + 1])
        out_indptr[n_pairs + 1] = out_indptr[n_pairs]  # dummy row has 0 nnz
        del pair_nnz

        # Fill in chunks to limit temporary memory from sparse indexing
        chunk_size = 2_000_000
        n_chunks = (n_pairs + chunk_size - 1) // chunk_size
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            chunk = dataset.X[matched_chosen[start:end]] - dataset.X[non_chosen_indices[start:end]]
            chunk = chunk.tocsr()
            nnz_start = int(out_indptr[start])
            nnz_end = nnz_start + chunk.nnz
            out_data[nnz_start:nnz_end] = chunk.data
            out_indices[nnz_start:nnz_end] = chunk.indices
            del chunk
            if self._progress_callback is not None:
                self._progress_callback(1)

        X_train = sparse.csr_matrix(
            (out_data, out_indices, out_indptr),
            shape=(n_pairs + 1, n_features),
        )
        del out_data, out_indices, out_indptr

        y_train = np.ones(n_pairs + 1, dtype=np.float32)
        y_train[-1] = 0.0  # dummy row for class balance

        self._model.fit(X_train, y_train)
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
        V = len(self.card_vocab)
        S = state_dim(self.card_vocab, self.relic_vocab)

        keys: list[str] = []
        scores: list[float] = []
        for card_id in offered_cards:
            idx = self.card_vocab.get(card_id)
            if idx is not None:
                score = float(beta[idx])
                interaction_offset = V + idx * S
                score += float(state_feat @ beta[interaction_offset:interaction_offset + S])
            else:
                score = 0.0
            keys.append(card_id)
            scores.append(score)

        # Skip: all-zero features → score = 0
        keys.append("skip")
        scores.append(0.0)

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
