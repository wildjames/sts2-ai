from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from logit_model.dataset import Dataset
from logit_model.model import CardPickModel
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import Card, GameState, Relic


@pytest.fixture()
def vocabs():
    card_vocab = CardVocabulary(["strike", "defend", "fireball", "heal"])
    relic_vocab = RelicVocabulary(["ring", "amulet"])
    return card_vocab, relic_vocab


def _make_state() -> GameState:
    return GameState(
        deck=[Card(id="strike"), Card(id="defend")],
        relics=[Relic(id="ring")],
        potions=[],
        current_hp=50,
        max_hp=100,
        gold=100,
        floor=5,
    )


def _synthetic_dataset(card_vocab, relic_vocab, n_groups: int = 40) -> Dataset:
    """Build a dataset where 'fireball' is always picked over 'heal'."""
    from logit_model.features import encode_choice_set
    from sts2_utils import CardChoiceResult

    state = _make_state()
    all_X, all_y, all_groups = [], [], []
    for g in range(n_groups):
        choices = CardChoiceResult(
            offered=[Card(id="fireball"), Card(id="heal")],
            picked=Card(id="fireball"),
        )
        X, picked_idx = encode_choice_set(state, choices, card_vocab, relic_vocab)
        n_alts = X.shape[0]
        y = np.zeros(n_alts, dtype=np.float32)
        y[picked_idx] = 1.0
        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.full(n_alts, g, dtype=np.int64))

    from scipy.sparse import vstack
    return Dataset(
        X=vstack(all_X, format="csr"),
        y=np.concatenate(all_y),
        groups=np.concatenate(all_groups),
    )


class TestCardPickModel:
    def test_fit_runs(self, vocabs):
        card_vocab, relic_vocab = vocabs
        ds = _synthetic_dataset(card_vocab, relic_vocab)
        model = CardPickModel(card_vocab, relic_vocab)
        model.fit(ds)

    def test_predict_proba_sums_to_one(self, vocabs):
        card_vocab, relic_vocab = vocabs
        ds = _synthetic_dataset(card_vocab, relic_vocab)
        model = CardPickModel(card_vocab, relic_vocab)
        model.fit(ds)

        state = _make_state()
        probs = model.predict_proba(state, ["fireball", "heal"])
        assert set(probs.keys()) == {"fireball", "heal", "skip"}
        assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0

    def test_picked_card_has_highest_prob(self, vocabs):
        card_vocab, relic_vocab = vocabs
        ds = _synthetic_dataset(card_vocab, relic_vocab)
        model = CardPickModel(card_vocab, relic_vocab)
        model.fit(ds)

        state = _make_state()
        probs = model.predict_proba(state, ["fireball", "heal"])
        assert probs["fireball"] > probs["heal"]
        assert probs["fireball"] > probs["skip"]

    def test_predict_before_fit_raises(self, vocabs):
        card_vocab, relic_vocab = vocabs
        model = CardPickModel(card_vocab, relic_vocab)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(_make_state(), ["fireball"])

    def test_save_load_roundtrip(self, vocabs, tmp_path):
        card_vocab, relic_vocab = vocabs
        ds = _synthetic_dataset(card_vocab, relic_vocab)
        model = CardPickModel(card_vocab, relic_vocab, C=0.5)
        model.fit(ds)

        state = _make_state()
        probs_before = model.predict_proba(state, ["fireball", "heal"])

        model.save(tmp_path / "model_dir")
        loaded = CardPickModel.load(tmp_path / "model_dir")
        probs_after = loaded.predict_proba(state, ["fireball", "heal"])

        for key in probs_before:
            assert pytest.approx(probs_before[key], abs=1e-6) == probs_after[key]
