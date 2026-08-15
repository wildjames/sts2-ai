from __future__ import annotations

import json
import tempfile

import pytest

from logit_model.vocabulary import CardVocabulary, RelicVocabulary, Vocabulary


class TestVocabulary:
    def test_len(self):
        vocab = Vocabulary(["a", "b", "c"])
        assert len(vocab) == 3

    def test_contains(self):
        vocab = Vocabulary(["a", "b"])
        assert "a" in vocab
        assert "z" not in vocab

    def test_getitem(self):
        vocab = Vocabulary(["x", "y", "z"])
        assert vocab["x"] == 0
        assert vocab["y"] == 1
        assert vocab["z"] == 2

    def test_getitem_missing_raises(self):
        vocab = Vocabulary(["a"])
        with pytest.raises(KeyError):
            vocab["missing"]

    def test_get_returns_default(self):
        vocab = Vocabulary(["a"])
        assert vocab.get("a") == 0
        assert vocab.get("missing") is None
        assert vocab.get("missing", -1) == -1

    def test_deduplicates(self):
        vocab = Vocabulary(["a", "b", "a", "c", "b"])
        assert len(vocab) == 3
        assert vocab["a"] == 0
        assert vocab["b"] == 1
        assert vocab["c"] == 2

    def test_ids_property(self):
        vocab = Vocabulary(["c", "a", "b"])
        assert vocab.ids == ["c", "a", "b"]

    def test_ids_returns_copy(self):
        vocab = Vocabulary(["a", "b"])
        ids = vocab.ids
        ids.append("c")
        assert len(vocab) == 2

    def test_roundtrip_json(self, tmp_path):
        path = tmp_path / "vocab.json"
        original = Vocabulary(["CARD.STRIKE", "CARD.DEFEND", "CARD.BASH"])
        original.to_json(path)
        loaded = Vocabulary.from_json(path)
        assert loaded.ids == original.ids
        assert loaded["CARD.STRIKE"] == 0

    def test_empty_vocabulary(self):
        vocab = Vocabulary([])
        assert len(vocab) == 0
        assert "anything" not in vocab


class TestCardVocabulary:
    def test_inherits_vocabulary(self):
        cv = CardVocabulary(["CARD.STRIKE", "CARD.DEFEND"])
        assert len(cv) == 2
        assert cv["CARD.STRIKE"] == 0

    def test_from_json(self, tmp_path):
        path = tmp_path / "cards.json"
        path.write_text(json.dumps(["CARD.A", "CARD.B"]))
        cv = CardVocabulary.from_json(path)
        assert isinstance(cv, CardVocabulary)
        assert cv["CARD.A"] == 0


class TestRelicVocabulary:
    def test_inherits_vocabulary(self):
        rv = RelicVocabulary(["RELIC.ANCHOR", "RELIC.BOOT"])
        assert len(rv) == 2
        assert rv["RELIC.BOOT"] == 1

    def test_from_json(self, tmp_path):
        path = tmp_path / "relics.json"
        path.write_text(json.dumps(["RELIC.X", "RELIC.Y"]))
        rv = RelicVocabulary.from_json(path)
        assert isinstance(rv, RelicVocabulary)
        assert rv["RELIC.Y"] == 1
