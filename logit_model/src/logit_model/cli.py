"""CLI for training and evaluating the card pick prediction model."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

from logit_model.dataset import Dataset, build_dataset_from_path
from logit_model.model import CardPickModel
from sts2_utils import GameState
from sts2_utils.game_state import Card, Relic


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """Card pick prediction model for Slay the Spire 2."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Directory to save the dataset.")
@click.option("--player-id", type=int, default=1, show_default=True, help="Player ID to extract data for.")
@click.option("--cards-json", type=click.Path(exists=True), required=True, help="Path to cards.json data file.")
@click.option("--relics-json", type=click.Path(exists=True), required=True, help="Path to relics.json data file.")
def preprocess(data: str, output: str, player_id: int, cards_json: str, relics_json: str) -> None:
    """Convert run data into a preprocessed dataset.

    DATA is a .jsonl file or directory of .json run files.
    """
    data_path = Path(data)
    if data_path.is_file() and data_path.suffix == ".jsonl":
        n_runs = sum(1 for line in open(data_path) if line.strip())
    elif data_path.is_dir():
        n_runs = len(list(data_path.glob("*.json")))
    else:
        n_runs = 0

    with click.progressbar(length=n_runs, label="Building dataset") as bar:
        dataset, _, _ = build_dataset_from_path(data, cards_json, relics_json, player_id, progress_callback=bar.update)

    n_groups = len(np.unique(dataset.groups))
    click.echo(
        f"Dataset: {dataset.X.shape[0]} rows, {dataset.X.shape[1]} features, "
        f"{n_groups} choice sets"
    )

    if n_groups == 0:
        click.echo("No choice sets found — nothing to save.", err=True)
        sys.exit(1)

    dataset.save(output)
    click.echo(f"Dataset saved to {output}")


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--train-output", type=click.Path(), required=True, help="Directory to save the training split.")
@click.option("--eval-output", type=click.Path(), required=True, help="Directory to save the evaluation split.")
@click.option("--train-fraction", type=float, default=0.8, show_default=True, help="Fraction of choice sets for training.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for the split.")
def split(dataset_path: str, train_output: str, eval_output: str, train_fraction: float, seed: int) -> None:
    """Split a preprocessed dataset into training and evaluation sets."""
    click.echo(f"Loading dataset from {dataset_path} ...")
    dataset = Dataset.load(dataset_path)

    train_ds, eval_ds = dataset.split(train_fraction=train_fraction, seed=seed)

    n_train = len(np.unique(train_ds.groups))
    n_eval = len(np.unique(eval_ds.groups))
    click.echo(f"Split: {n_train} train / {n_eval} eval choice sets")

    train_ds.save(train_output)
    eval_ds.save(eval_output)
    click.echo(f"Train saved to {train_output}")
    click.echo(f"Eval saved to {eval_output}")


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Directory to save the trained model.")
@click.option("-C", "--regularisation", type=float, default=1.0, show_default=True, help="Inverse regularisation strength.")
@click.option("--max-iter", type=int, default=1000, show_default=True, help="Maximum solver iterations.")
@click.option("--gpu", is_flag=True, help="Use GPU-accelerated training via cuML.")
def train(dataset_path: str, output: str, regularisation: float, max_iter: int, gpu: bool) -> None:
    """Train a model from a preprocessed dataset.

    DATASET_PATH is a directory produced by the preprocess or split command.
    """
    click.echo(f"Loading dataset from {dataset_path} ...")
    dataset = Dataset.load(dataset_path)

    if dataset.card_vocab is None or dataset.relic_vocab is None:
        click.echo("Dataset is missing vocabulary files.", err=True)
        sys.exit(1)

    n_groups = len(np.unique(dataset.groups))
    click.echo(
        f"Dataset: {dataset.X.shape[0]} rows, {dataset.X.shape[1]} features, "
        f"{n_groups} choice sets"
    )

    if n_groups == 0:
        click.echo("No choice sets found — nothing to train on.", err=True)
        sys.exit(1)

    backend = "GPU (cuML)" if gpu else "CPU"
    click.echo(f"Training on {backend} (C={regularisation}, max_iter={max_iter}) ...")
    model = CardPickModel(dataset.card_vocab, dataset.relic_vocab, C=regularisation, max_iter=max_iter, gpu=gpu)

    n_pairs = int((dataset.y == 0).sum())
    chunk_size = 2_000_000
    n_chunks = (n_pairs + chunk_size - 1) // chunk_size
    with click.progressbar(length=n_chunks + 1, label="Training") as bar:
        model._progress_callback = bar.update
        model.fit(dataset)
        bar.update(1)  # account for the sklearn/cuML fit step

    model.save(output)
    click.echo(f"Model saved to {output}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--gpu", is_flag=True, help="Use GPU for scoring via cuPy.")
def evaluate(model_path: str, dataset_path: str, gpu: bool) -> None:
    """Evaluate a trained model on a preprocessed dataset.

    Reports top-1 accuracy (how often the model's top pick matches the
    player's actual pick) and mean log-likelihood per choice set.
    """
    click.echo(f"Loading model from {model_path} ...")
    model = CardPickModel.load(model_path)

    click.echo(f"Loading dataset from {dataset_path} ...")
    dataset = Dataset.load(dataset_path)

    n_groups = len(np.unique(dataset.groups))
    if n_groups == 0:
        click.echo("No choice sets found — nothing to evaluate.", err=True)
        sys.exit(1)

    click.echo(f"Evaluating on {n_groups} choice sets ...")

    beta = model._model.coef_.ravel()

    if gpu:
        import cupy as cp
        from cupyx.scipy import sparse as cp_sparse

        all_scores = cp.asnumpy(
            cp_sparse.csr_matrix(dataset.X) @ cp.array(beta)
        ).ravel()
    else:
        all_scores = np.asarray(dataset.X @ beta).ravel()

    # Sort by group once to avoid O(n_rows) scan per group
    order = np.argsort(dataset.groups, kind="mergesort")
    sorted_groups = dataset.groups[order]
    sorted_scores = all_scores[order]
    sorted_y = dataset.y[order]

    unique_groups = np.unique(sorted_groups)
    boundaries = np.searchsorted(sorted_groups, unique_groups, side="left")
    ends = np.append(boundaries[1:], len(sorted_groups))

    correct = 0
    total_ll = 0.0
    with click.progressbar(range(n_groups), label="Scoring") as bar:
        for i in bar:
            sl = slice(boundaries[i], ends[i])
            scores = sorted_scores[sl]
            y_g = sorted_y[sl]

            scores = scores - scores.max()
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum()

            if np.argmax(probs) == np.argmax(y_g):
                correct += 1

            total_ll += np.log(max(float(probs[np.argmax(y_g)]), 1e-12))

    accuracy = correct / n_groups
    mean_ll = total_ll / n_groups

    click.echo(f"Top-1 accuracy: {accuracy:.4f} ({correct}/{n_groups})")
    click.echo(f"Mean log-likelihood: {mean_ll:.4f}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
def inspect(model_path: str) -> None:
    """Show summary information about a saved model."""
    model = CardPickModel.load(model_path)
    beta = model._model.coef_.ravel()
    n_nonzero = int(np.count_nonzero(beta))

    click.echo(f"Model: {model_path}")
    click.echo(f"  Cards in vocabulary:  {len(model.card_vocab)}")
    click.echo(f"  Relics in vocabulary: {len(model.relic_vocab)}")
    click.echo(f"  Feature dimensions:   {len(beta)}")
    click.echo(f"  Non-zero coefficients: {n_nonzero}/{len(beta)}")
    click.echo(f"  C (regularisation):   {model._C}")

    # Show top positive and negative coefficients in the card one-hot section
    V = len(model.card_vocab)
    card_beta = beta[:V]
    indices = np.argsort(card_beta)
    ids = model.card_vocab.ids

    click.echo("\n  Top 10 most-picked cards (by base coefficient):")
    for i in indices[-10:][::-1]:
        if card_beta[i] == 0:
            break
        click.echo(f"    {ids[i]:30s}  {card_beta[i]:+.4f}")

    click.echo("\n  Top 10 most-skipped cards (by base coefficient):")
    for i in indices[:10]:
        if card_beta[i] == 0:
            break
        click.echo(f"    {ids[i]:30s}  {card_beta[i]:+.4f}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("input_json", type=click.Path(exists=True), required=False)
def predict(model_path: str, input_json: str | None) -> None:
    """Predict pick probabilities for a card choice.

    Reads JSON from INPUT_JSON (a file path) or stdin if omitted.

    \b
    Expected JSON format:
    {
      "deck": ["CARD.STRIKE_REGENT", "CARD.DEFEND_REGENT", ...],
      "relics": ["RELIC.DIVINE_RIGHT", ...],
      "offered_cards": ["CARD.FLAME_BARRIER", "CARD.WHIRLWIND", "CARD.IMPERVIOUS"],
      "current_hp": 60,
      "max_hp": 80,
      "gold": 100,
      "floor": 10
    }
    """
    model = CardPickModel.load(model_path)

    if input_json:
        with open(input_json) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    deck = [Card(id=c) if isinstance(c, str) else Card(**c) for c in data.get("deck", [])]
    relics = [Relic(id=r) if isinstance(r, str) else Relic(**r) for r in data.get("relics", [])]
    offered_cards = data["offered_cards"]

    state = GameState(
        deck=deck,
        relics=relics,
        potions=data.get("potions", []),
        current_hp=data.get("current_hp", 50),
        max_hp=data.get("max_hp", 80),
        gold=data.get("gold", 0),
        floor=data.get("floor", 1),
    )

    probs = model.predict_proba(state, offered_cards)

    # Sort by probability descending
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    click.echo(json.dumps({card: round(prob, 6) for card, prob in sorted_probs}, indent=2))
