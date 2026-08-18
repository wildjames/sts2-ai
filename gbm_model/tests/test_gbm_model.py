"""Tests for GBM card pick model."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gbm_model.dataset import Dataset
from gbm_model.features import encode_row
from gbm_model.model import GBMCardPickModel
from sts2_utils import CardVocabulary, RelicVocabulary


class TestGBMCardPickModel:
    """Tests for the GBM card pick model."""

    @pytest.fixture
    def sample_card_relic_vocabularies(self) -> tuple[CardVocabulary, RelicVocabulary]:
        """Create sample vocabularies for testing."""
        cards = ["card_1", "card_2", "card_3"]
        relics = ["relic_a", "relic_b"]
        card_vocab = CardVocabulary({card: i for i, card in enumerate(cards)})
        relic_vocab = RelicVocabulary({relic: i for i, relic in enumerate(relics)})
        return card_vocab, relic_vocab

    @pytest.fixture
    def sample_dataset(self, sample_card_relic_vocabularies) -> Dataset:
        """Create a simple dataset for testing."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        # Create 2 groups with 3 alternatives each (2 cards + skip)
        X = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],  # card_1
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],  # card_2
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1], # skip
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.2],  # card_1 (group 2)
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.2],  # card_2 (group 2)
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.2], # skip (group 2)
        ], dtype=np.float32)
        
        y = np.array([0, 1, 0, 1, 0, 0], dtype=np.int32)  # card_2 and first card_1 are picked
        groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        group_sizes = np.array([3, 3], dtype=np.int64)
        
        return Dataset(
            X=X, y=y, groups=groups, group_sizes=group_sizes,
            card_vocab=card_vocab, relic_vocab=relic_vocab
        )

    def test_init_with_default_params(self, sample_card_relic_vocabularies):
        """Test model initialization with default parameters."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        
        assert model.card_vocab is card_vocab
        assert model.relic_vocab is relic_vocab
        assert isinstance(model._params, dict)

    def test_init_with_custom_params(self, sample_card_relic_vocabularies):
        """Test model initialization with custom LightGBM parameters."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        custom_params = {"n_estimators": 100, "num_leaves": 32}
        model = GBMCardPickModel(card_vocab, relic_vocab, **custom_params)
        
        assert model._params["n_estimators"] == 100
        assert model._params["num_leaves"] == 32

    def test_fit_basic(self, sample_dataset, sample_card_relic_vocabularies):
        """Test basic model fitting."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        model.fit(sample_dataset)
        
        assert model._model is not None

    def test_predict_scores_returns_dict(self, sample_dataset, sample_card_relic_vocabularies):
        """Test that predict_scores returns a dictionary."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        model.fit(sample_dataset)
        
        # Mock state for prediction
        from sts2_utils import Card, Relic, GameState
        state = GameState(
            deck=[Card(id="CARD.card_1"), Card(id="CARD.card_2")],
            relics=[Relic(id="RELIC.relic_a")],
            potions=[],
            current_hp=75,
            max_hp=100,
            gold=0,
            floor=10
        )
        
        scores = model.predict_scores(state, ["card_1", "card_2"])
        
        assert isinstance(scores, dict)
        # Should include both cards and skip
        assert "card_1" in scores
        assert "card_2" in scores
        assert "skip" in scores

    def test_predict_proba_returns_valid_probabilities(self, sample_dataset, sample_card_relic_vocabularies):
        """Test that predict_proba returns valid probabilities."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        model.fit(sample_dataset)
        
        from sts2_utils import Card, Relic, GameState
        state = GameState(
            deck=[Card(id="CARD.card_1"), Card(id="CARD.card_2")],
            relics=[Relic(id="RELIC.relic_a")],
            potions=[],
            current_hp=75,
            max_hp=100,
            gold=0,
            floor=10
        )
        
        probs = model.predict_proba(state, ["card_1", "card_2"])
        
        assert isinstance(probs, dict)
        # All probabilities should be between 0 and 1
        for card_id, prob in probs.items():
            assert 0.0 <= prob <= 1.0
        
        # Sum of all probabilities should be approximately 1.0
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.001

    def test_save_and_load(self, sample_dataset, tmp_path: Path, sample_card_relic_vocabularies):
        """Test model serialization and deserialization."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        model.fit(sample_dataset)
        
        # Save model
        save_path = tmp_path / "test_model"
        model.save(save_path)
        
        assert (save_path / "model.txt").exists()
        assert (save_path / "card_vocab.json").exists()
        assert (save_path / "relic_vocab.json").exists()
        
        # Load model
        loaded_model = GBMCardPickModel.load(save_path)
        
        assert isinstance(loaded_model, GBMCardPickModel)
        assert len(loaded_model.card_vocab) == len(card_vocab)
        assert len(loaded_model.relic_vocab) == len(relic_vocab)

    def test_predict_scores_with_unknown_card(self, sample_dataset, sample_card_relic_vocabularies):
        """Test prediction with unknown card."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        model.fit(sample_dataset)
        
        from sts2_utils import Card, GameState
        state = GameState(
            deck=[Card(id="CARD.card_1")],
            relics=[],
            potions=[],
            current_hp=50,
            max_hp=100,
            gold=0,
            floor=25
        )
        
        # Include unknown card in offered cards
        scores = model.predict_scores(state, ["unknown_card", "skip"])
        
        assert isinstance(scores, dict)
        assert "unknown_card" in scores
        assert "skip" in scores

    def test_fit_with_eval_set(self, sample_dataset, sample_card_relic_vocabularies):
        """Test fitting with evaluation set for early stopping."""
        card_vocab, relic_vocab = sample_card_relic_vocabularies
        
        # Create train and eval datasets
        train_X = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],
        ], dtype=np.float32)
        
        eval_X = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1],
        ], dtype=np.float32)
        
        train_y = np.array([0, 1, 0], dtype=np.int32)
        eval_y = np.array([0, 1, 0], dtype=np.int32)
        
        train_dataset = Dataset(
            X=train_X, y=train_y, groups=np.array([0, 0, 0], dtype=np.int64),
            group_sizes=np.array([3], dtype=np.int64),
            card_vocab=card_vocab, relic_vocab=relic_vocab
        )
        
        eval_dataset = Dataset(
            X=eval_X, y=eval_y, groups=np.array([0, 0, 0], dtype=np.int64),
            group_sizes=np.array([3], dtype=np.int64),
            card_vocab=card_vocab, relic_vocab=relic_vocab
        )
        
        model = GBMCardPickModel(card_vocab, relic_vocab)
        model.fit(train_dataset, eval_dataset)
        
        assert model._model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
