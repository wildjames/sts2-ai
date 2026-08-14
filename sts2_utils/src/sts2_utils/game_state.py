from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Card:
    """A single card in the player's deck.

    Attributes:
        id: The card identifier, e.g. ``"CARD.STRIKE_REGENT"``.
        floor_added: The 1-based floor number where this card was added to the
            deck.  A value of ``1`` indicates a starting card or one gained at
            the Neow event.  ``0`` means the floor is unknown.
        upgrade_level: How many times the card has been upgraded.  ``0`` is
            un-upgraded, ``1`` means upgraded once (e.g. via a rest-site smith).
        enchantment: Optional enchantment applied to the card, stored as a dict
            with at least ``"id"`` and ``"amount"`` keys, or ``None`` if the
            card is not enchanted.
    """

    id: str
    floor_added: int = 0
    upgrade_level: int = 0
    enchantment: dict | None = None


@dataclass
class Relic:
    """A relic held by the player.

    Attributes:
        id: The relic identifier, e.g. ``"RELIC.DIVINE_RIGHT"``.
        floor_added: The 1-based floor number where this relic was obtained.
            ``1`` for the character's innate relic or a Neow reward.
    """

    id: str
    floor_added: int = 0


@dataclass
class GameState:
    """Snapshot of the player's state at the beginning of a given floor.

    Attributes:
        deck: All cards currently in the player's deck.
        relics: All relics the player currently holds.
        potions: Potion identifiers currently in the player's potion slots,
            e.g. ``["POTION.FIRE_POTION"]``.
        current_hp: The player's current hit points entering this floor.
        max_hp: The player's maximum hit points entering this floor.
        gold: The player's gold total entering this floor.
        floor: The 1-based floor number this state represents.
    """

    deck: list[Card]
    relics: list[Relic]
    potions: list[str]
    current_hp: int
    max_hp: int
    gold: int
    floor: int


def load_run(path: str | Path) -> dict:
    """Load a slay-the-stats run JSON file and return it as a dict.

    Args:
        path: Filesystem path to the run JSON file.

    Returns:
        The parsed run data dictionary.  The expected top-level keys include
        ``"map_point_history"`` (nested list of floor data per act) and
        ``"players"`` (list of player records with final deck/relics).
    """
    with open(path) as f:
        return json.load(f)


def build_game_state(run_data: dict, floor: int, player_id: int = 1) -> GameState:
    """Build the game state at the beginning of the given floor.

    Reconstructs the player's deck, relics, potions, HP, max HP, and gold as
    they would appear when the player *enters* the requested floor.  The
    reconstruction works by:

    1. Reverse-engineering the starting deck from the final deck recorded in
       the run data, undoing all transforms, removals, and upgrades.
    2. Walking forward through each floor's ``player_stats`` to apply card
       gains, removals, transforms, upgrades, enchantments, relic pickups,
       and potion changes.

    Floors are 1-indexed and correspond to the flattened sequence of map
    points across all acts.  Floor 1 is the state *before* the Neow event.
    Floor 2 is the state *after* Neow but *before* the second map point.

    Args:
        run_data: A parsed run JSON dictionary as returned by :func:`load_run`.
        floor: The 1-based floor number to reconstruct state for.  Must be
            between ``1`` and ``total_floors + 1`` (inclusive), where
            ``total_floors + 1`` represents the state after the final floor.
        player_id: Which player to build state for (default ``1``).  Relevant
            for future multi-player support.

    Returns:
        A :class:`GameState` snapshot for the requested floor.

    Raises:
        ValueError: If *floor* is less than 1 or exceeds the run length.
    """
    all_floors = _flatten_floors(run_data)

    if floor < 1:
        raise ValueError("Floor must be >= 1")
    total_floors = len(all_floors)
    if floor > total_floors + 1:
        raise ValueError(
            f"Floor {floor} is beyond the run's {total_floors} floors"
        )

    initial_deck = _compute_initial_deck(run_data, all_floors, player_id)
    initial_relics = _compute_initial_relics(run_data, all_floors, player_id)
    initial_hp, initial_max_hp, initial_gold = _compute_initial_stats(
        all_floors[0], player_id
    )

    state = GameState(
        deck=list(initial_deck),
        relics=list(initial_relics),
        potions=[],
        current_hp=initial_hp,
        max_hp=initial_max_hp,
        gold=initial_gold,
        floor=floor,
    )

    for i in range(min(floor - 1, total_floors)):
        _apply_floor(state, all_floors[i], i + 1, player_id)

    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_floors(run_data: dict) -> list[dict]:
    """Flatten ``map_point_history`` (nested per act) into a single list.

    Each element is a map-point dict containing ``player_stats`` and ``rooms``.
    The resulting list is ordered by floor number (1-indexed when enumerated
    starting at 1).
    """
    floors: list[dict] = []
    for act in run_data["map_point_history"]:
        floors.extend(act)
    return floors


def _get_player_stats(floor_data: dict, player_id: int = 1) -> dict:
    """Extract the ``player_stats`` entry for *player_id* from a floor dict.

    Falls back to the first entry if no matching ``player_id`` is found, or
    an empty dict if there are no stats at all.
    """
    for stats in floor_data.get("player_stats", []):
        if stats.get("player_id") == player_id:
            return stats
    stats_list = floor_data.get("player_stats", [])
    return stats_list[0] if stats_list else {}


def _card_from_dict(card_data: dict, default_floor: int = 0) -> Card:
    """Create a :class:`Card` from a raw JSON card dict.

    Args:
        card_data: Dict with at least an ``"id"`` key.  May also contain
            ``"floor_added_to_deck"``, ``"current_upgrade_level"``, and
            ``"enchantment"``.
        default_floor: Floor number to use when ``"floor_added_to_deck"`` is
            absent from *card_data*.
    """
    return Card(
        id=card_data["id"],
        floor_added=card_data.get("floor_added_to_deck", default_floor),
        upgrade_level=card_data.get("current_upgrade_level", 0),
        enchantment=card_data.get("enchantment"),
    )


def _remove_card_from_deck(deck: list[Card], card_data: dict) -> bool:
    """Remove the first card matching *card_data* from *deck* in place.

    Matches by ``id`` and ``floor_added_to_deck`` when the floor is present in
    *card_data*; otherwise falls back to matching by ``id`` alone.  Only the
    first match is removed (handles duplicate card IDs correctly).

    Returns:
        ``True`` if a card was removed, ``False`` if no match was found.
    """
    card_id = card_data["id"]
    card_floor = card_data.get("floor_added_to_deck")

    if card_floor is not None:
        for i, card in enumerate(deck):
            if card.id == card_id and card.floor_added == card_floor:
                deck.pop(i)
                return True

    for i, card in enumerate(deck):
        if card.id == card_id:
            deck.pop(i)
            return True

    return False


def _upgrade_card_in_deck(deck: list[Card], card_id: str) -> bool:
    """Increment the upgrade level of the first card with the given *card_id*.

    Returns:
        ``True`` if a card was upgraded, ``False`` if no match was found.
    """
    for card in deck:
        if card.id == card_id:
            card.upgrade_level += 1
            return True
    return False


def _enchant_card_in_deck(deck: list[Card], enchant_data: dict) -> bool:
    """Apply an enchantment to a card in the deck.

    The target card is identified by the ``"card"`` sub-dict inside
    *enchant_data*, matching on ``id`` and optionally ``floor_added_to_deck``.

    Returns:
        ``True`` if the enchantment was applied, ``False`` if no match was
        found.
    """
    card_info = enchant_data["card"]
    card_id = card_info["id"]
    card_floor = card_info.get("floor_added_to_deck")
    enchantment = card_info.get("enchantment")

    for card in deck:
        if card.id == card_id and (
            card_floor is None or card.floor_added == card_floor
        ):
            card.enchantment = enchantment
            return True
    return False


def _compute_initial_deck(
    run_data: dict, all_floors: list[dict], player_id: int
) -> list[Card]:
    """Reconstruct the starting deck before any floor is processed.

    Works backwards from the final deck recorded in *run_data*:

    1. Collects all cards with ``floor_added_to_deck == 1`` from the final
       deck (these survived the entire run).  Upgrades and enchantments are
       stripped to reflect the un-modified starting state.
    2. Walks every floor to undo transforms and removals of starter cards:
       - If a transform's ``original_card`` had ``floor_added_to_deck == 1``,
         that original is added back (it was a starter that was transformed).
       - If a transform's ``final_card`` had ``floor_added_to_deck == 1``,
         it is removed (it replaced a starter at the Neow event).
       - If a ``cards_removed`` entry had ``floor_added_to_deck == 1``, it is
         added back (it was a starter that was removed during the run).
    """
    player = _get_player(run_data, player_id)
    final_deck = player["deck"]

    # Collect floor-1 cards from the final deck (strip upgrades / enchantments)
    initial: list[Card] = []
    for card_data in final_deck:
        if card_data.get("floor_added_to_deck") == 1:
            initial.append(Card(id=card_data["id"], floor_added=1))

    # Walk every floor to undo transforms / removals of starter cards
    for floor_data in all_floors:
        stats = _get_player_stats(floor_data, player_id)

        for transform in stats.get("cards_transformed", []):
            original = transform["original_card"]
            final = transform["final_card"]

            if original.get("floor_added_to_deck") == 1:
                initial.append(Card(id=original["id"], floor_added=1))

            if final.get("floor_added_to_deck") == 1:
                for i, c in enumerate(initial):
                    if c.id == final["id"]:
                        initial.pop(i)
                        break

        for removed in stats.get("cards_removed", []):
            if removed.get("floor_added_to_deck") == 1:
                initial.append(Card(id=removed["id"], floor_added=1))

    return initial


def _compute_initial_relics(
    run_data: dict, all_floors: list[dict], player_id: int
) -> list[Relic]:
    """Determine starting relics (character relic only, before Neow).

    Identifies the character's innate relic by taking all relics with
    ``floor_added_to_deck == 1`` from the final relic list and excluding any
    that appear in the Neow event's ``relic_choices`` with ``was_picked``.
    """
    player = _get_player(run_data, player_id)
    final_relics = player["relics"]

    floor_1_stats = _get_player_stats(all_floors[0], player_id)
    floor_1_gained = {
        c["choice"]
        for c in floor_1_stats.get("relic_choices", [])
        if c.get("was_picked")
    }

    return [
        Relic(id=r["id"], floor_added=1)
        for r in final_relics
        if r.get("floor_added_to_deck") == 1 and r["id"] not in floor_1_gained
    ]


def _compute_initial_stats(
    first_floor: dict, player_id: int
) -> tuple[int, int, int]:
    """Reverse-compute the player's HP, max HP, and gold before floor 1.

    Uses the first floor's end-of-floor stats and reverses the recorded
    changes (damage, healing, gold gained/spent) to derive the values the
    player had before the Neow event.

    Returns:
        A ``(current_hp, max_hp, gold)`` tuple.
    """
    stats = _get_player_stats(first_floor, player_id)
    hp = stats["current_hp"] + stats["damage_taken"] - stats["hp_healed"]
    max_hp = stats["max_hp"] + stats["max_hp_lost"] - stats["max_hp_gained"]
    gold = (
        stats["current_gold"]
        - stats["gold_gained"]
        + stats["gold_spent"]
        + stats["gold_lost"]
    )
    return hp, max_hp, gold


def _apply_floor(
    state: GameState,
    floor_data: dict,
    floor_num: int,
    player_id: int,
) -> None:
    """Mutate *state* to reflect everything that happened at *floor_num*.

    Updates HP, max HP, and gold from the floor's end-of-floor values, then
    applies deck changes (transforms -> removals -> gains -> upgrades ->
    enchantments), relic pickups, and potion changes (used -> discarded ->
    picked up) in the order they logically occur during a floor.
    """
    stats = _get_player_stats(floor_data, player_id)

    state.current_hp = stats["current_hp"]
    state.max_hp = stats["max_hp"]
    state.gold = stats["current_gold"]

    # Deck: transforms -> removals -> gains -> upgrades -> enchantments
    for transform in stats.get("cards_transformed", []):
        _remove_card_from_deck(state.deck, transform["original_card"])
        state.deck.append(
            _card_from_dict(transform["final_card"], default_floor=floor_num)
        )

    for removed in stats.get("cards_removed", []):
        _remove_card_from_deck(state.deck, removed)

    for card_data in stats.get("cards_gained", []):
        state.deck.append(_card_from_dict(card_data, default_floor=floor_num))

    for card_id in stats.get("upgraded_cards", []):
        _upgrade_card_in_deck(state.deck, card_id)

    for enchant_data in stats.get("cards_enchanted", []):
        _enchant_card_in_deck(state.deck, enchant_data)

    # Relics
    for choice in stats.get("relic_choices", []):
        if choice.get("was_picked"):
            state.relics.append(
                Relic(id=choice["choice"], floor_added=floor_num)
            )

    # Potions: use -> discard -> pick up
    for potion in stats.get("potion_used", []):
        try:
            state.potions.remove(potion)
        except ValueError:
            pass

    for potion in stats.get("potion_discarded", []):
        try:
            state.potions.remove(potion)
        except ValueError:
            pass

    for choice in stats.get("potion_choices", []):
        if choice.get("was_picked"):
            state.potions.append(choice["choice"])


def _get_player(run_data: dict, player_id: int) -> dict:
    """Return the player record matching *player_id*, or the first player."""
    for p in run_data["players"]:
        if p["id"] == player_id:
            return p
    return run_data["players"][0]


@dataclass
class CardChoiceResult:
    """The card choices offered at a floor and which card was picked.

    Attributes:
        offered: Cards that were offered as choices.
        picked: The card that was picked, or ``None`` if the player
            skipped the card reward.
    """

    offered: list[Card]
    picked: Card | None


def get_card_choices(
    run_data: dict, floor: int, player_id: int = 1
) -> CardChoiceResult | None:
    """Return the card choices offered at the given floor.

    Args:
        run_data: A parsed run JSON dictionary as returned by :func:`load_run`.
        floor: The 1-based floor number to query.
        player_id: Which player to inspect (default ``1``).

    Returns:
        A :class:`CardChoiceResult` with the offered card IDs and the picked
        card (or ``None`` if skipped), or ``None`` if no card choice was
        presented at this floor.
    """
    all_floors = _flatten_floors(run_data)

    if (floor < 1) or (floor > len(all_floors)):
        raise ValueError(
            f"Floor {floor} is out of range (1-{len(all_floors)})"
        )

    stats = _get_player_stats(all_floors[floor - 1], player_id)
    choices = stats.get("card_choices", [])
    if not choices:
        return None

    offered = [_card_from_dict(c["card"]) for c in choices]
    picked = next(
        (_card_from_dict(c["card"]) for c in choices if c.get("was_picked")),
        None,
    )
    return CardChoiceResult(offered=offered, picked=picked)


@dataclass
class RelicChoiceResult:
    """The relic choices offered at a floor and which was picked.

    Attributes:
        offered: Relic IDs that were offered as choices.
        picked: The relic ID that was picked, or ``None`` if the player
            skipped the relic reward.
    """

    offered: list[str]
    picked: str | None


def get_relic_choices(
    run_data: dict, floor: int, player_id: int = 1
) -> RelicChoiceResult | None:
    """Return the relic choices offered at the given floor.

    Args:
        run_data: A parsed run JSON dictionary as returned by :func:`load_run`.
        floor: The 1-based floor number to query.
        player_id: Which player to inspect (default ``1``).

    Returns:
        A :class:`RelicChoiceResult` with the offered relic IDs and the picked
        relic (or ``None`` if skipped), or ``None`` if no relic choice was
        presented at this floor.
    """
    all_floors = _flatten_floors(run_data)

    if (floor < 1) or (floor > len(all_floors)):
        raise ValueError(
            f"Floor {floor} is out of range (1-{len(all_floors)})"
        )

    stats = _get_player_stats(all_floors[floor - 1], player_id)
    choices = stats.get("relic_choices", [])
    if not choices:
        return None

    offered = [c["choice"] for c in choices]
    picked = next(
        (c["choice"] for c in choices if c.get("was_picked")),
        None,
    )
    return RelicChoiceResult(offered=offered, picked=picked)
