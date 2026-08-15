from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy import sparse

from logit_model.features import encode_choice_set, feature_dim
from logit_model.vocabulary import CardVocabulary, RelicVocabulary
from sts2_utils import build_game_state, get_card_choices

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Training dataset for the card pick model.

    Attributes:
        X: Sparse feature matrix ``(n_rows, n_features)``.  Each row is one
            alternative (offered card or skip) in a choice set.
        y: Binary labels ``(n_rows,)``.  Exactly one row per group is ``1``.
        groups: Integer array ``(n_rows,)`` mapping rows to choice-set IDs.
        card_vocab: Card vocabulary used to encode features.
        relic_vocab: Relic vocabulary used to encode features.
    """

    X: sparse.csr_matrix
    y: np.ndarray
    groups: np.ndarray
    card_vocab: CardVocabulary | None = None
    relic_vocab: RelicVocabulary | None = None

    def save(self, path: str | Path) -> None:
        """Persist dataset arrays and vocabularies to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path / "X.npz", self.X)
        np.save(path / "y.npy", self.y)
        np.save(path / "groups.npy", self.groups)
        if self.card_vocab is not None:
            self.card_vocab.to_json(path / "card_vocab.json")
        if self.relic_vocab is not None:
            self.relic_vocab.to_json(path / "relic_vocab.json")

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Load a dataset from a directory created by :meth:`save`."""
        path = Path(path)
        X = sparse.load_npz(path / "X.npz")
        y = np.load(path / "y.npy")
        groups = np.load(path / "groups.npy")
        card_vocab = (
            CardVocabulary.from_json(path / "card_vocab.json")
            if (path / "card_vocab.json").exists() else None
        )
        relic_vocab = (
            RelicVocabulary.from_json(path / "relic_vocab.json")
            if (path / "relic_vocab.json").exists() else None
        )
        return cls(X=X, y=y, groups=groups, card_vocab=card_vocab, relic_vocab=relic_vocab)

    def split(
        self, train_fraction: float = 0.8, seed: int = 42,
    ) -> tuple[Dataset, Dataset]:
        """Split into train/eval datasets by choice-set group."""
        rng = np.random.default_rng(seed)
        unique_groups = np.unique(self.groups)
        shuffled = rng.permutation(unique_groups)
        n_train = int(len(shuffled) * train_fraction)
        train_set = set(shuffled[:n_train].tolist())

        train_mask = np.array([g in train_set for g in self.groups])
        eval_mask = ~train_mask

        return (
            Dataset(
                X=self.X[train_mask], y=self.y[train_mask],
                groups=self.groups[train_mask],
                card_vocab=self.card_vocab, relic_vocab=self.relic_vocab,
            ),
            Dataset(
                X=self.X[eval_mask], y=self.y[eval_mask],
                groups=self.groups[eval_mask],
                card_vocab=self.card_vocab, relic_vocab=self.relic_vocab,
            ),
        )


def load_runs(path: str | Path) -> Iterator[dict]:
    """Yield run dicts from a ``.jsonl`` file or a directory of ``.json`` files."""
    path = Path(path)
    if path.is_file() and path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.is_dir():
        for json_file in sorted(path.glob("*.json")):
            with open(json_file) as f:
                yield json.load(f)
    else:
        raise ValueError(f"Expected a .jsonl file or a directory, got: {path}")


def _total_floors(run_data: dict) -> int:
    return sum(len(act) for act in run_data["map_point_history"])


def _collect_ids_from_run(
    run_data: dict, player_id: int = 1,
) -> tuple[set[str], set[str]]:
    """Extract all card and relic IDs mentioned anywhere in a single run."""
    card_ids: set[str] = set()
    relic_ids: set[str] = set()

    for player in run_data["players"]:
        if player.get("id") != player_id:
            continue
        for card in player["deck"]:
            card_ids.add(card["id"])
        for relic in player["relics"]:
            relic_ids.add(relic["id"])

    for act in run_data["map_point_history"]:
        for floor_data in act:
            for stats in floor_data.get("player_stats", []):
                for choice in stats.get("card_choices", []):
                    card_ids.add(choice["card"]["id"])
                for transform in stats.get("cards_transformed", []):
                    card_ids.add(transform["original_card"]["id"])
                    card_ids.add(transform["final_card"]["id"])
                for removed in stats.get("cards_removed", []):
                    card_ids.add(removed["id"])
                for gained in stats.get("cards_gained", []):
                    card_ids.add(gained["id"])
                for choice in stats.get("relic_choices", []):
                    relic_ids.add(choice["choice"])

    return card_ids, relic_ids


def build_vocabularies(
    runs: Iterable[dict], player_id: int = 1,
) -> tuple[CardVocabulary, RelicVocabulary]:
    """Pass 1: scan all runs to build card and relic vocabularies."""
    all_card_ids: set[str] = set()
    all_relic_ids: set[str] = set()

    for run_data in runs:
        card_ids, relic_ids = _collect_ids_from_run(run_data, player_id)
        all_card_ids.update(card_ids)
        all_relic_ids.update(relic_ids)

    return (
        CardVocabulary(sorted(all_card_ids)),
        RelicVocabulary(sorted(all_relic_ids)),
    )


def build_vocabularies_from_files(
    cards_json: str | Path,
    relics_json: str | Path,
) -> tuple[CardVocabulary, RelicVocabulary]:
    """Build vocabularies from the static cards.json and relics.json data files."""
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
    """Pass 2: encode every card-choice screen into feature rows."""
    all_X: list[sparse.csr_matrix] = []
    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    group_id = 0

    for run_data in runs:
        if progress_callback is not None:
            progress_callback(1)
        n_floors = _total_floors(run_data)
        for floor in range(1, n_floors + 1):
            try:
                choices = get_card_choices(run_data, floor, player_id)
            except (ValueError, KeyError, IndexError):
                logger.debug("Skipping floor %d: card-choice error", floor, exc_info=True)
                continue

            if choices is None:
                continue

            try:
                state = build_game_state(run_data, floor, player_id)
            except (ValueError, KeyError, IndexError):
                logger.debug("Skipping floor %d: game-state error", floor, exc_info=True)
                continue

            X, picked_idx = encode_choice_set(
                state, choices, card_vocab, relic_vocab,
            )
            n_alts = X.shape[0]

            y = np.zeros(n_alts, dtype=np.float32)
            y[picked_idx] = 1.0

            all_X.append(X)
            all_y.append(y)
            all_groups.append(np.full(n_alts, group_id, dtype=np.int64))
            group_id += 1

    if not all_X:
        n_features = feature_dim(card_vocab, relic_vocab)
        return Dataset(
            X=sparse.csr_matrix((0, n_features), dtype=np.float32),
            y=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=np.int64),
            card_vocab=card_vocab,
            relic_vocab=relic_vocab,
        )

    return Dataset(
        X=sparse.vstack(all_X, format="csr"),
        y=np.concatenate(all_y),
        groups=np.concatenate(all_groups),
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
    """Build a dataset using static card/relic vocabulary files.

    Args:
        path: A ``.jsonl`` file or a directory of ``.json`` run files.
        cards_json: Path to the static ``cards.json`` data file.
        relics_json: Path to the static ``relics.json`` data file.
        player_id: Which player to extract data for (default ``1``).

    Returns:
        ``(dataset, card_vocab, relic_vocab)`` tuple.
    """
    card_vocab, relic_vocab = build_vocabularies_from_files(cards_json, relics_json)
    dataset = build_dataset(load_runs(path), card_vocab, relic_vocab, player_id, progress_callback=progress_callback)
    return dataset, card_vocab, relic_vocab
