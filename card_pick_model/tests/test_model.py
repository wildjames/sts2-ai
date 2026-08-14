from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sts2_card_pick.dataset import Dataset
from sts2_card_pick.model import CardPickModel
from sts2_card_pick.vocabulary import CardVocabulary, RelicVocabulary
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
    from sts2_card_pick.features import encode_card_features, encode_state_features

    state = _make_state()
    state_feat = encode_state_features(state, card_vocab, relic_vocab)

    rows, labels, groups = [], [], []
    for g in range(n_groups):
        for card_id in ["fireball", "heal"]:
            card_feat = encode_card_features(card_id, card_vocab)
            rows.append(np.concatenate([state_feat, card_feat]))
            labels.append(1.0 if card_id == "fireball" else 0.0)
            groups.append(g)
        # skip row
        skip_feat = encode_card_features(None, card_vocab)
        rows.append(np.concatenate([state_feat, skip_feat]))
        labels.append(0.0)
        groups.append(g)

    return Dataset(
        X=csr_matrix(np.stack(rows).astype(np.float32)),
        y=np.array(labels, dtype=np.float32),
        groups=np.array(groups, dtype=np.int64),
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
