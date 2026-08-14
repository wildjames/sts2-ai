import pytest
from pathlib import Path

from sts2_utils.game_state import build_game_state, get_card_choices, get_relic_choices, load_run

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "test_data"
    / "regent_data_1.json"
)


@pytest.fixture
def run_data():
    return load_run(FIXTURE_PATH)


class TestInitialState:
    """Floor 1 = state before the Neow event."""

    def test_starting_deck_card_ids(self, run_data):
        state = build_game_state(run_data, floor=1)
        ids = [c.id for c in state.deck]
        assert ids.count("CARD.STRIKE_REGENT") == 4
        assert ids.count("CARD.DEFEND_REGENT") == 4
        assert ids.count("CARD.VENERATE") == 1
        assert ids.count("CARD.ASCENDERS_BANE") == 1
        assert ids.count("CARD.FALLING_STAR") == 1
        assert len(state.deck) == 11

    def test_starting_deck_no_upgrades(self, run_data):
        state = build_game_state(run_data, floor=1)
        assert all(c.upgrade_level == 0 for c in state.deck)

    def test_starting_relic(self, run_data):
        state = build_game_state(run_data, floor=1)
        assert len(state.relics) == 1
        assert state.relics[0].id == "RELIC.DIVINE_RIGHT"

    def test_starting_gold(self, run_data):
        state = build_game_state(run_data, floor=1)
        assert state.gold == 99

    def test_starting_potions_empty(self, run_data):
        state = build_game_state(run_data, floor=1)
        assert state.potions == []

    def test_starting_max_hp(self, run_data):
        state = build_game_state(run_data, floor=1)
        assert state.max_hp == 75


class TestAfterNeow:
    """Floor 2 = state after the Neow event."""

    def test_transforms_applied(self, run_data):
        state = build_game_state(run_data, floor=2)
        ids = [c.id for c in state.deck]
        assert ids.count("CARD.STRIKE_REGENT") == 3
        assert ids.count("CARD.DEFEND_REGENT") == 3
        assert ids.count("CARD.BLACK_HOLE") == 1
        assert ids.count("CARD.PILLAR_OF_CREATION") == 1
        assert len(state.deck) == 11

    def test_hp_after_neow(self, run_data):
        state = build_game_state(run_data, floor=2)
        assert state.current_hp == 60
        assert state.max_hp == 63

    def test_relic_gained_at_neow(self, run_data):
        state = build_game_state(run_data, floor=2)
        relic_ids = {r.id for r in state.relics}
        assert "RELIC.DIVINE_RIGHT" in relic_ids
        assert "RELIC.LEAFY_POULTICE" in relic_ids
        assert len(state.relics) == 2


class TestMidRun:
    """State at various mid-run floors."""

    def test_floor_5_deck(self, run_data):
        state = build_game_state(run_data, floor=5)
        ids = [c.id for c in state.deck]
        # Gained at floors 2-4
        assert "CARD.RADIATE" in ids
        assert "CARD.HIDDEN_CACHE" in ids
        assert "CARD.PATTER" in ids
        # Floor 5 gain (second BLACK_HOLE) not yet applied
        assert ids.count("CARD.BLACK_HOLE") == 1

    def test_floor_5_hp_gold(self, run_data):
        state = build_game_state(run_data, floor=5)
        assert state.current_hp == 57
        assert state.max_hp == 63
        assert state.gold == 17

    def test_rest_site_upgrade(self, run_data):
        # Floor 7 (rest site) upgrades VENERATE
        state = build_game_state(run_data, floor=8)
        venerate = next(c for c in state.deck if c.id == "CARD.VENERATE")
        assert venerate.upgrade_level == 1

    def test_rest_site_heal(self, run_data):
        # Floor 12 heals; before it HP was 5, after it HP is 23
        before = build_game_state(run_data, floor=12)
        assert before.current_hp == 5
        after = build_game_state(run_data, floor=13)
        assert after.current_hp == 23


class TestPotions:
    def test_potion_gained_at_shop(self, run_data):
        # Floor 3 (shop): bought VULNERABLE_POTION
        state = build_game_state(run_data, floor=4)
        assert "POTION.VULNERABLE_POTION" in state.potions

    def test_potion_gained_from_combat(self, run_data):
        # Floor 4: picked STAR_POTION as reward
        state = build_game_state(run_data, floor=5)
        assert "POTION.STAR_POTION" in state.potions
        assert "POTION.VULNERABLE_POTION" in state.potions

    def test_potions_used_in_combat(self, run_data):
        # Floor 11: used STAR_POTION and VULNERABLE_POTION
        state = build_game_state(run_data, floor=12)
        assert "POTION.STAR_POTION" not in state.potions
        assert "POTION.VULNERABLE_POTION" not in state.potions


class TestRelics:
    def test_elite_relic(self, run_data):
        state = build_game_state(run_data, floor=9)
        relic_ids = {r.id for r in state.relics}
        assert "RELIC.SPARKLING_ROUGE" in relic_ids

    def test_treasure_relic(self, run_data):
        state = build_game_state(run_data, floor=11)
        relic_ids = {r.id for r in state.relics}
        assert "RELIC.ANCHOR" in relic_ids

    def test_relic_count_grows(self, run_data):
        count_early = len(build_game_state(run_data, floor=2).relics)
        count_late = len(build_game_state(run_data, floor=20).relics)
        assert count_late > count_early


class TestCardRemovalAndEnchantment:
    def test_shop_removal(self, run_data):
        # Floor 22: STRIKE_REGENT removed
        before = build_game_state(run_data, floor=22)
        after = build_game_state(run_data, floor=23)
        strikes_before = sum(
            1 for c in before.deck if c.id == "CARD.STRIKE_REGENT"
        )
        strikes_after = sum(
            1 for c in after.deck if c.id == "CARD.STRIKE_REGENT"
        )
        assert strikes_after == strikes_before - 1

    def test_enchantment_applied(self, run_data):
        # Floor 31: METEOR_SHOWER enchanted with PERFECT_FIT
        state = build_game_state(run_data, floor=32)
        meteor = next(c for c in state.deck if c.id == "CARD.METEOR_SHOWER")
        assert meteor.enchantment is not None
        assert meteor.enchantment["id"] == "ENCHANTMENT.PERFECT_FIT"

    def test_act2_transform(self, run_data):
        # Floor 18: FALLING_STAR transformed to METEOR_SHOWER
        state = build_game_state(run_data, floor=19)
        ids = [c.id for c in state.deck]
        assert "CARD.FALLING_STAR" not in ids
        assert "CARD.METEOR_SHOWER" in ids


class TestFinalState:
    def test_final_deck_ids_match(self, run_data):
        total = sum(len(act) for act in run_data["map_point_history"])
        state = build_game_state(run_data, floor=total + 1)
        final_deck = run_data["players"][0]["deck"]
        assert sorted(c.id for c in state.deck) == sorted(
            c["id"] for c in final_deck
        )

    def test_final_deck_size(self, run_data):
        total = sum(len(act) for act in run_data["map_point_history"])
        state = build_game_state(run_data, floor=total + 1)
        assert len(state.deck) == len(run_data["players"][0]["deck"])

    def test_final_relic_ids_match(self, run_data):
        total = sum(len(act) for act in run_data["map_point_history"])
        state = build_game_state(run_data, floor=total + 1)
        final_relics = run_data["players"][0]["relics"]
        assert sorted(r.id for r in state.relics) == sorted(
            r["id"] for r in final_relics
        )


NECRO_DATA_1 = (
    Path(__file__).resolve().parent
    / "test_data"
    / "necro_data_1.json"
)


@pytest.fixture
def necro_data():
    return load_run(NECRO_DATA_1)


class TestNecrobinderNeowRemoval:
    """Necrobinder run where Neow removed a card (Precise Scissors)."""

    def test_starting_deck(self, necro_data):
        state = build_game_state(necro_data, floor=1)
        ids = [c.id for c in state.deck]
        assert ids.count("CARD.STRIKE_NECROBINDER") == 4
        assert ids.count("CARD.DEFEND_NECROBINDER") == 4
        assert ids.count("CARD.BODYGUARD") == 1
        assert ids.count("CARD.UNLEASH") == 1
        assert ids.count("CARD.ASCENDERS_BANE") == 1
        assert len(state.deck) == 11

    def test_after_neow_card_removed(self, necro_data):
        state = build_game_state(necro_data, floor=2)
        ids = [c.id for c in state.deck]
        # One strike removed by Precise Scissors
        assert ids.count("CARD.STRIKE_NECROBINDER") == 3
        assert len(state.deck) == 10

    def test_starting_relic(self, necro_data):
        state = build_game_state(necro_data, floor=1)
        assert len(state.relics) == 1
        assert state.relics[0].id == "RELIC.BOUND_PHYLACTERY"

    def test_neow_relic_gained(self, necro_data):
        state = build_game_state(necro_data, floor=2)
        relic_ids = {r.id for r in state.relics}
        assert "RELIC.PRECISE_SCISSORS" in relic_ids
        assert len(state.relics) == 2

    def test_final_deck_ids_match(self, necro_data):
        total = sum(len(act) for act in necro_data["map_point_history"])
        state = build_game_state(necro_data, floor=total + 1)
        final_deck = necro_data["players"][0]["deck"]
        assert sorted(c.id for c in state.deck) == sorted(
            c["id"] for c in final_deck
        )

    def test_final_deck_size(self, necro_data):
        total = sum(len(act) for act in necro_data["map_point_history"])
        state = build_game_state(necro_data, floor=total + 1)
        assert len(state.deck) == len(necro_data["players"][0]["deck"])

    def test_final_relic_ids_match(self, necro_data):
        total = sum(len(act) for act in necro_data["map_point_history"])
        state = build_game_state(necro_data, floor=total + 1)
        final_relics = necro_data["players"][0]["relics"]
        assert sorted(r.id for r in state.relics) == sorted(
            r["id"] for r in final_relics
        )

    def test_elite_upgrade_multiple_cards(self, necro_data):
        # Floor 15 (elite): BODYGUARD and DEFY both upgraded
        state = build_game_state(necro_data, floor=16)
        bodyguard = next(c for c in state.deck if c.id == "CARD.BODYGUARD")
        defy = next(c for c in state.deck if c.id == "CARD.DEFY")
        assert bodyguard.upgrade_level == 1
        assert defy.upgrade_level == 1

    def test_potion_discard_on_pickup(self, necro_data):
        # Floor 6: WEAK_POTION discarded, ATTACK_POTION picked up
        state = build_game_state(necro_data, floor=7)
        assert "POTION.ATTACK_POTION" in state.potions
        assert "POTION.WEAK_POTION" not in state.potions

    def test_death_floor_hp_zero(self, necro_data):
        # Floor 17 (boss): player died
        total = sum(len(act) for act in necro_data["map_point_history"])
        state = build_game_state(necro_data, floor=total + 1)
        assert state.current_hp == 0


class TestValidation:
    def test_floor_zero_raises(self, run_data):
        with pytest.raises(ValueError):
            build_game_state(run_data, floor=0)

    def test_floor_too_high_raises(self, run_data):
        with pytest.raises(ValueError):
            build_game_state(run_data, floor=200)

    def test_floor_attribute_set(self, run_data):
        state = build_game_state(run_data, floor=5)
        assert state.floor == 5


class TestCardChoices:
    def test_floor_with_card_picked(self, necro_data):
        result = get_card_choices(necro_data, floor=2)
        assert result is not None
        assert [c.id for c in result.offered] == [
            "CARD.NEGATIVE_PULSE",
            "CARD.RIGHT_HAND_HAND",
            "CARD.DEATH_MARCH",
        ]
        assert result.picked is not None
        assert result.picked.id == "CARD.NEGATIVE_PULSE"

    def test_floor_with_no_pick(self, necro_data):
        # Floor 8 (unknown/monster): all card_choices have was_picked=false
        result = get_card_choices(necro_data, floor=8)
        assert result is not None
        assert len(result.offered) == 3
        assert result.picked is None

    def test_floor_with_no_card_choices(self, necro_data):
        # Floor 7 (rest site): no card_choices
        result = get_card_choices(necro_data, floor=7)
        assert result is None

    def test_neow_floor_no_card_choices(self, necro_data):
        result = get_card_choices(necro_data, floor=1)
        assert result is None

    def test_shop_floor_skipped_all(self, necro_data):
        # Floor 5 (shop): all card_choices have was_picked=false
        result = get_card_choices(necro_data, floor=5)
        assert result is not None
        assert result.picked is None
        assert len(result.offered) == 6

    def test_floor_out_of_range(self, necro_data):
        with pytest.raises(ValueError):
            get_card_choices(necro_data, floor=0)
        with pytest.raises(ValueError):
            get_card_choices(necro_data, floor=200)


FIXTURE_PATH_GLAM = (
    Path(__file__).resolve().parent
    / "test_data"
    / "glam_data_1.json"
)


@pytest.fixture
def glam_data():
    return load_run(FIXTURE_PATH_GLAM)


class TestGlamEnchantment:
    def test_card_choice_preserves_enchantment(self, glam_data):
        result = get_card_choices(glam_data, floor=2)
        assert result is not None
        assert result.picked is not None
        assert result.picked.id == "CARD.RAMPAGE"
        assert result.picked.enchantment == {
            "amount": 1,
            "id": "ENCHANTMENT.GLAM",
        }

    def test_card_choice_all_offered_have_enchantment(self, glam_data):
        result = get_card_choices(glam_data, floor=2)
        assert result is not None
        for card in result.offered:
            assert card.enchantment is not None
            assert card.enchantment["id"] == "ENCHANTMENT.GLAM"

    def test_deck_preserves_enchantment_from_gain(self, glam_data):
        state = build_game_state(glam_data, floor=3)
        rampage = next(c for c in state.deck if c.id == "CARD.RAMPAGE")
        assert rampage.enchantment == {
            "amount": 1,
            "id": "ENCHANTMENT.GLAM",
        }

    def test_non_enchanted_choice_has_none(self, glam_data):
        result = get_card_choices(glam_data, floor=4)
        assert result is not None
        for card in result.offered:
            assert card.enchantment is None


class TestRelicChoices:
    def test_floor_with_relic_picked(self, necro_data):
        result = get_relic_choices(necro_data, floor=1)
        assert result is not None
        assert result.offered == ["RELIC.PRECISE_SCISSORS"]
        assert result.picked == "RELIC.PRECISE_SCISSORS"

    def test_shop_floor_skipped_all(self, necro_data):
        result = get_relic_choices(necro_data, floor=5)
        assert result is not None
        assert result.offered == [
            "RELIC.ODDLY_SMOOTH_STONE",
            "RELIC.STRIKE_DUMMY",
            "RELIC.THE_ABACUS",
        ]
        assert result.picked is None

    def test_floor_with_no_relic_choices(self, necro_data):
        result = get_relic_choices(necro_data, floor=2)
        assert result is None

    def test_floor_out_of_range(self, necro_data):
        with pytest.raises(ValueError):
            get_relic_choices(necro_data, floor=0)
        with pytest.raises(ValueError):
            get_relic_choices(necro_data, floor=200)
