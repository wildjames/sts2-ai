"""LightGBM-based card pick prediction model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

from gbm_model.dataset import Dataset
from gbm_model.features import encode_row
from sts2_utils import CardVocabulary, GameState, RelicVocabulary


class GBMCardPickModel:
    """Gradient Boosting Model for card pick prediction using LightGBM LambdaRank.

    This model learns to rank cards based on their likelihood of being picked by
    the player in a given game state. It uses LightGBM's LGBMRanker with the
    lambdarank objective to optimize ranking metrics like NDCG.

    Attributes:
        card_vocab: Vocabulary mapping card IDs to indices.
        relic_vocab: Vocabulary mapping relic IDs to indices.
        _params: Dictionary of LightGBM hyperparameters.
        _model: Fitted LightGBM booster, or None if not yet fitted.
    """

    DEFAULT_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    def __init__(self, card_vocab: CardVocabulary, relic_vocab: RelicVocabulary, **lgb_params: Any) -> None:
        """Initialize the GBM card pick model.

        Args:
            card_vocab: Vocabulary for mapping card IDs to indices.
            relic_vocab: Vocabulary for mapping relic IDs to indices.
            lgb_params: Additional LightGBM hyperparameters to override defaults.
        """
        self.card_vocab = card_vocab
        self.relic_vocab = relic_vocab

        # Merge custom params with defaults (custom params take precedence)
        self._params = {**self.DEFAULT_PARAMS, **lgb_params}

        # Fitted model will be stored here
        self._model: Any | None = None

    def fit(self, train: Dataset, eval_set: Dataset | None = None) -> None:
        """Fit the LightGBM Ranker on training data with optional early stopping.

        Args:
            train: Training dataset containing features and labels.
            eval_set: Optional evaluation dataset for early stopping. If provided,
                the model will monitor NDCG on this set and stop training if it
                doesn't improve.

        Raises:
            RuntimeError: If LightGBM is not installed.
        """
        if lgb is None:
            raise RuntimeError("LightGBM is required but not installed")

        self._model = lgb.LGBMRanker(**self._params)

        fit_params: dict[str, Any] = {
            "X": train.X,
            "y": train.y,
            "group": train.group_sizes,
            "categorical_feature": [0],  # card_idx column is categorical
        }

        if eval_set is not None:
            fit_params["eval_set"] = [(eval_set.X, eval_set.y)]
            fit_params["eval_group"] = [eval_set.group_sizes]
            fit_params["eval_names"] = ["eval"]

        self._model.fit(**fit_params)

    def predict_scores(self, state: GameState, offered_cards: list[str]) -> dict[str, float]:
        """Predict raw LightGBM scores for each card and skip option.

        Args:
            state: Current game state at the beginning of the floor.
            offered_cards: List of card IDs offered to the player on this floor.

        Returns:
            Dictionary mapping card IDs (and "skip") to their raw LightGBM scores.
            Higher scores indicate more likely picks.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Build feature matrix for all offered cards plus skip
        rows: list[np.ndarray] = []

        for card_id in offered_cards:
            row = encode_row(card_id, state, self.card_vocab, self.relic_vocab)
            rows.append(row)

        # Add skip option (card_id=None)
        skip_row = encode_row(None, state, self.card_vocab, self.relic_vocab)
        rows.append(skip_row)

        X = np.vstack(rows).astype(np.float32)

        # Get raw scores from the model
        scores = self._model.predict(X)

        # Map scores back to card IDs
        result: dict[str, float] = {}
        for i, card_id in enumerate(offered_cards):
            result[card_id] = float(scores[i])
        result["skip"] = float(scores[len(offered_cards)])

        return result

    def predict_proba(self, state: GameState, offered_cards: list[str]) -> dict[str, float]:
        """Predict probabilities for each card and skip option using softmax.

        Args:
            state: Current game state at the beginning of the floor.
            offered_cards: List of card IDs offered to the player on this floor.

        Returns:
            Dictionary mapping card IDs (and "skip") to their predicted probabilities.
            All probabilities sum to 1.0.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        scores = self.predict_scores(state, offered_cards)

        # Convert scores to probabilities using softmax
        score_array = np.array(list(scores.values()), dtype=np.float64)

        # Numerical stability: subtract max before exp
        score_array -= np.max(score_array)
        exp_scores = np.exp(score_array)
        probs = exp_scores / np.sum(exp_scores)

        # Map probabilities back to card IDs
        result: dict[str, float] = {}
        for i, card_id in enumerate(scores.keys()):
            result[card_id] = float(probs[i])

        return result

    def save(self, path: str | Path) -> None:
        """Save the model and vocabularies to disk.

        Args:
            path: Directory path where model files will be saved.
                Will create the directory if it doesn't exist.

        Saves three files:
            - model.txt: LightGBM booster model file
            - card_vocab.json: Card vocabulary as JSON array
            - relic_vocab.json: Relic vocabulary as JSON array

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before saving")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model via the underlying booster object
        self._model.booster_.save_model(path / "model.txt")

        # Save vocabularies as JSON arrays (just the IDs in order)
        with open(path / "card_vocab.json", "w") as f:
            json.dump(self.card_vocab.ids, f)

        with open(path / "relic_vocab.json", "w") as f:
            json.dump(self.relic_vocab.ids, f)

    @classmethod
    def load(cls, path: str | Path) -> GBMCardPickModel:
        """Load a fitted model from disk.

        Args:
            path: Directory path where model files were saved.

        Returns:
            Loaded GBMCardPickModel instance with fitted model and vocabularies.

        Raises:
            FileNotFoundError: If required model files don't exist.
            RuntimeError: If LightGBM is not installed.
        """
        if lgb is None:
            raise RuntimeError("LightGBM is required but not installed")

        path = Path(path)

        # Load vocabularies
        with open(path / "card_vocab.json", "r") as f:
            card_ids = json.load(f)
        with open(path / "relic_vocab.json", "r") as f:
            relic_ids = json.load(f)

        card_vocab = CardVocabulary(card_ids)
        relic_vocab = RelicVocabulary(relic_ids)

        # Create model instance (without fitting yet)
        model = cls(card_vocab, relic_vocab)

        # Load the fitted booster
        model._model = lgb.Booster(model_file=path / "model.txt")

        return model
