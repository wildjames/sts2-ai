from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gbm_model.features import encode_row
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import build_game_state, get_card_choices, load_runs


@dataclass
class Dataset:
    """Dense ranking dataset for the GBM card-pick model."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    group_sizes: np.ndarray
    card_vocab: CardVocabulary
    relic_vocab: RelicVocabulary

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "X.npy", self.X)
        np.save(path / "y.npy", self.y)
        np.save(path / "groups.npy", self.groups)
        np.save(path / "group_sizes.npy", self.group_sizes)
        self.card_vocab.to_json(path / "card_vocab.json")
        self.relic_vocab.to_json(path / "relic_vocab.json")

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        path = Path(path)
        X = np.load(path / "X.npy")
        y = np.load(path / "y.npy")
        groups = np.load(path / "groups.npy")
        group_sizes = np.load(path / "group_sizes.npy")
        card_vocab = CardVocabulary.from_json(path / "card_vocab.json")
        relic_vocab = RelicVocabulary.from_json(path / "relic_vocab.json")
        return cls(
            X=X,
            y=y,
            groups=groups,
            group_sizes=group_sizes,
            card_vocab=card_vocab,
            relic_vocab=relic_vocab,
        )

    def split(self, train_fraction: float = 0.8, seed: int = 42) -> tuple[Dataset, Dataset]:
        rng = np.random.default_rng(seed)
        unique_groups = np.unique(self.groups)
        shuffled = rng.permutation(unique_groups)
        n_train = int(len(shuffled) * train_fraction)
        train_set = set(shuffled[:n_train].tolist())

        train_mask = np.array([group in train_set for group in self.groups])
        eval_mask = ~train_mask

        return (
            Dataset(
                X=self.X[train_mask],
                y=self.y[train_mask],
                groups=self.groups[train_mask],
                group_sizes=np.bincount(self.groups[train_mask]).astype(np.int64),
                card_vocab=self.card_vocab,
                relic_vocab=self.relic_vocab,
            ),
            Dataset(
                X=self.X[eval_mask],
                y=self.y[eval_mask],
                groups=self.groups[eval_mask],
                group_sizes=np.bincount(self.groups[eval_mask]).astype(np.int64),
                card_vocab=self.card_vocab,
                relic_vocab=self.relic_vocab,
            ),
        )


def build_vocabularies_from_files(
    cards_json: str | Path,
    relics_json: str | Path,
) -> tuple[CardVocabulary, RelicVocabulary]:
    with open(cards_json) as f:
        cards = json.load(f)
    with open(relics_json) as f:
        relics = json.load(f)

    card_ids = sorted({f"CARD.{card['id']}" for card in cards})
    relic_ids = sorted({f"RELIC.{relic['id']}" for relic in relics})
    return CardVocabulary(card_ids), RelicVocabulary(relic_ids)


def build_dataset(
    runs: Iterable[dict],
    card_vocab: CardVocabulary,
    relic_vocab: RelicVocabulary,
    player_id: int = 1,
    progress_callback: object | None = None,
) -> Dataset:
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    group_id = 0

    for run_data in runs:
        if progress_callback is not None:
            progress_callback(1)

        all_floors = run_data.get("map_point_history", [])
        n_floors = sum(len(act) for act in all_floors)
        for floor in range(1, n_floors + 1):
            try:
                choices = get_card_choices(run_data, floor, player_id)
            except (ValueError, KeyError, IndexError):
                continue
            if choices is None:
                continue
            try:
                state = build_game_state(run_data, floor, player_id)
            except (ValueError, KeyError, IndexError):
                continue

            n_alts = len(choices.offered) + 1
            rows: list[np.ndarray] = []
            labels = np.zeros(n_alts, dtype=np.int32)
            picked_idx = n_alts - 1

            for i, card in enumerate(choices.offered):
                row = encode_row(card.id, state, card_vocab, relic_vocab)
                rows.append(row)
                if choices.picked is not None and card.id == choices.picked.id:
                    picked_idx = i

            skip_row = encode_row(None, state, card_vocab, relic_vocab)
            rows.append(skip_row)
            labels[picked_idx] = 1

            X = np.vstack(rows).astype(np.float32)
            all_X.append(X)
            all_y.append(labels)
            all_groups.append(np.full(n_alts, group_id, dtype=np.int64))
            group_id += 1

    if not all_X:
        n_features = 1 + len(card_vocab) + len(relic_vocab) + 2
        return Dataset(
            X=np.empty((0, n_features), dtype=np.float32),
            y=np.empty(0, dtype=np.int32),
            groups=np.empty(0, dtype=np.int64),
            group_sizes=np.empty(0, dtype=np.int64),
            card_vocab=card_vocab,
            relic_vocab=relic_vocab,
        )

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int32)
    groups = np.concatenate(all_groups, axis=0).astype(np.int64)
    group_sizes = np.bincount(groups).astype(np.int64)
    return Dataset(
        X=X,
        y=y,
        groups=groups,
        group_sizes=group_sizes,
        card_vocab=card_vocab,
        relic_vocab=relic_vocab,
    )


def build_dataset_from_path(
    path: str | Path,
    cards_json: str | Path,
    relics_json: str | Path,
    player_id: int = 1,
    progress_callback: object | None = None,
) -> tuple[Dataset, CardVocabulary, RelicVocabulary]:
    card_vocab, relic_vocab = build_vocabularies_from_files(cards_json, relics_json)
    dataset = build_dataset(load_runs(path), card_vocab, relic_vocab, player_id, progress_callback=progress_callback)
    return dataset, card_vocab, relic_vocab



