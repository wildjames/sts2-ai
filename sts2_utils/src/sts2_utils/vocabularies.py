from __future__ import annotations

import json
from pathlib import Path


class Vocabulary:
    """Ordered mapping from string IDs to integer indices."""

    def __init__(self, ids: list[str]):
        seen: set[str] = set()
        unique: list[str] = []
        for id_ in ids:
            if id_ not in seen:
                seen.add(id_)
                unique.append(id_)
        self._id_to_idx: dict[str, int] = {id_: i for i, id_ in enumerate(unique)}
        self._ids: list[str] = unique

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._id_to_idx

    def __getitem__(self, id_: str) -> int:
        return self._id_to_idx[id_]

    def get(self, id_: str, default: int | None = None) -> int | None:
        return self._id_to_idx.get(id_, default)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self._ids, f)

    @classmethod
    def from_json(cls, path: str | Path) -> Vocabulary:
        with open(path) as f:
            return cls(json.load(f))


class CardVocabulary(Vocabulary):
    """Card ID → index mapping."""
    pass


class RelicVocabulary(Vocabulary):
    """Relic ID → index mapping."""
    pass


def build_vocabularies_from_files(
    cards_json: str | Path,
    relics_json: str | Path,
) -> tuple["CardVocabulary", "RelicVocabulary"]:
    """Build vocabularies from the static cards.json and relics.json data files."""
    with open(cards_json) as f:
        cards = json.load(f)
    with open(relics_json) as f:
        relics = json.load(f)

    card_ids = sorted({f"CARD.{card['id']}" for card in cards})
    relic_ids = sorted({f"RELIC.{relic['id']}" for relic in relics})

    return CardVocabulary(card_ids), RelicVocabulary(relic_ids)
