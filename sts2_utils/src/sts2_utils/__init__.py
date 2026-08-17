"""Utilities for parsing and reconstructing Slay the Spire 2 run data.

This package provides data models and functions to reconstruct the game state
(deck, relics, potions, HP, gold) at the beginning of any floor from a run
JSON file exported by slay-the-stats.

Example::

    from sts2_utils import load_run, build_game_state

    run = load_run("path/to/run.json")
    state = build_game_state(run, floor=5)
    print(state.deck)       # cards in deck at the start of floor 5
    print(state.current_hp)  # HP entering floor 5
"""

from sts2_utils.game_state import (
    Card,
    CardChoiceResult,
    Relic,
    RelicChoiceResult,
    GameState,
    build_game_state,
    get_card_choices,
    get_relic_choices,
    load_run,
)
from sts2_utils.datasets import load_runs

__all__ = [
    "Card",
    "CardChoiceResult",
    "Relic",
    "RelicChoiceResult",
    "GameState",
    "build_game_state",
    "get_card_choices",
    "get_relic_choices",
    "load_run",
    "loa+d_runs",
]
