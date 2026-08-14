"""CLI for training and evaluating the card pick prediction model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

from sts2_card_pick.dataset import Dataset, build_dataset_from_path, load_runs, build_vocabularies, build_dataset
from sts2_card_pick.model import CardPickModel


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """Card pick prediction model for Slay the Spire 2."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Directory to save the trained model.")
@click.option("-C", "--regularisation", type=float, default=1.0, show_default=True, help="Inverse regularisation strength.")
@click.option("--max-iter", type=int, default=1000, show_default=True, help="Maximum solver iterations.")
@click.option("--player-id", type=int, default=1, show_default=True, help="Player ID to extract data for.")
def train(data: str, output: str, regularisation: float, max_iter: int, player_id: int) -> None:
    """Train a model from run data.

    DATA is a .jsonl file or directory of .json run files.
    """
    click.echo(f"Building dataset from {data} ...")
    dataset, card_vocab, relic_vocab = build_dataset_from_path(data, player_id)

    n_groups = len(np.unique(dataset.groups))
    click.echo(
        f"Dataset: {dataset.X.shape[0]} rows, {dataset.X.shape[1]} features, "
        f"{n_groups} choice sets"
    )

    if n_groups == 0:
        click.echo("No choice sets found — nothing to train on.", err=True)
        sys.exit(1)

    click.echo(f"Training (C={regularisation}, max_iter={max_iter}) ...")
    model = CardPickModel(card_vocab, relic_vocab, C=regularisation, max_iter=max_iter)
    model.fit(dataset)

    model.save(output)
    click.echo(f"Model saved to {output}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data", type=click.Path(exists=True))
@click.option("--player-id", type=int, default=1, show_default=True, help="Player ID to extract data for.")
def evaluate(model_path: str, data: str, player_id: int) -> None:
    """Evaluate a trained model on run data.

    Reports top-1 accuracy (how often the model's top pick matches the
    player's actual pick) and mean log-likelihood per choice set.
    """
    click.echo(f"Loading model from {model_path} ...")
    model = CardPickModel.load(model_path)

    click.echo(f"Building dataset from {data} ...")
    dataset = build_dataset(
        load_runs(data), model.card_vocab, model.relic_vocab, player_id,
    )

    n_groups = len(np.unique(dataset.groups))
    if n_groups == 0:
        click.echo("No choice sets found — nothing to evaluate.", err=True)
        sys.exit(1)

    click.echo(f"Evaluating on {n_groups} choice sets ...")

    beta = model._model.coef_.ravel()
    correct = 0
    total_ll = 0.0

    for g in np.unique(dataset.groups):
        mask = dataset.groups == g
        X_g = dataset.X[mask]
        y_g = dataset.y[mask]

        scores = X_g @ beta
        scores -= scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        predicted_idx = int(np.argmax(probs))
        actual_idx = int(np.argmax(y_g))

        if predicted_idx == actual_idx:
            correct += 1

        prob_actual = float(probs[actual_idx])
        total_ll += np.log(max(prob_actual, 1e-12))

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
    card_offset = len(model.card_vocab) + len(model.relic_vocab) + 2
    card_beta = beta[card_offset:]
    if len(card_beta) == len(model.card_vocab):
        indices = np.argsort(card_beta)
        ids = model.card_vocab.ids

        click.echo("\n  Top 10 most-picked cards (by coefficient):")
        for i in indices[-10:][::-1]:
            if card_beta[i] == 0:
                break
            click.echo(f"    {ids[i]:30s}  {card_beta[i]:+.4f}")

        click.echo("\n  Top 10 most-skipped cards (by coefficient):")
        for i in indices[:10]:
            if card_beta[i] == 0:
                break
            click.echo(f"    {ids[i]:30s}  {card_beta[i]:+.4f}")
