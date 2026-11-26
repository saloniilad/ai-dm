from dm.engine import Engine, Player
from dm.memory import Memory


def eng(seed=1, world=None):
    return Engine(Player(name="T"), world=world, seed=seed)


def test_roll_is_deterministic_with_seed():
    assert eng(3).roll_check("cunning", 12) == eng(3).roll_check("cunning", 12)


def test_roll_outcomes_valid_and_bounded():
    e = eng()
    for _ in range(200):
        r = e.roll_check("might", 12)
        assert 1 <= r["die"] <= 20
        assert r["outcome"] in {"success", "failure", "critical success", "critical failure"}


def test_unknown_stat_rejected():
    assert "error" in eng().roll_check("charisma", 10)


def test_hp_clamped():
    e = eng()
    assert e.change_hp(+100)["hp"] == e.player.max_hp
    r = e.change_hp(-999)
    assert r["hp"] == 0 and r["dead"]


def test_gold_cannot_go_negative():
    e = eng()
    assert "error" in e.change_gold(-999) and e.player.gold == 10


def test_unknown_tool_and_bad_args():
    e = eng()
    assert "error" in e.call("__init__", {})
    assert "error" in e.call("change_hp", {"nope": 1})


def test_travel_costs_time_and_rejects_unknown_places():
    w = Memory().world
    e = eng(world=w)
    r = e.move_to("Naga Ruins")
    assert r["distance_km"] > 10 and r["travel_hours"] > 2 and e.player.hours == r["travel_hours"]
    assert "error" in e.move_to("Moonbase Alpha")
    assert e.move_to("tavern")["location"] == "The Drowned Lotus Tavern"
