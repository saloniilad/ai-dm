"""Deterministic game engine. The LLM narrates; this code owns all state and outcomes."""
import random
from dataclasses import dataclass, field, asdict

WALK_KMH = 4.0


@dataclass
class Player:
    name: str
    hp: int = 20
    max_hp: int = 20
    gold: int = 10
    location: str = "Vindhya Crossroads"
    hours: float = 0.0
    inventory: list = field(default_factory=lambda: ["torch", "rusty sword"])
    stats: dict = field(default_factory=lambda: {"might": 1, "cunning": 2, "spirit": 1})
    flags: set = field(default_factory=set)

    def snapshot(self):
        d = asdict(self)
        d["flags"] = sorted(self.flags)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["flags"] = set(d.get("flags", []))
        return cls(**d)


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class Engine:
    PUBLIC_TOOLS = {"roll_check", "change_hp", "change_gold", "add_item", "remove_item", "move_to", "set_flag"}

    def __init__(self, player: Player, world=None, seed=None):
        self.player = player
        self.world = world
        self.rng = random.Random(seed)

    def roll_check(self, stat: str, difficulty: int):
        if stat not in self.player.stats:
            return {"error": f"unknown stat '{stat}'. Use one of {sorted(self.player.stats)}"}
        difficulty = max(1, min(30, _int(difficulty, 12)))
        die = self.rng.randint(1, 20)
        total = die + self.player.stats[stat]
        if die == 20:
            outcome = "critical success"
        elif die == 1:
            outcome = "critical failure"
        else:
            outcome = "success" if total >= difficulty else "failure"
        return {"stat": stat, "die": die, "total": total, "difficulty": difficulty, "outcome": outcome}

    def change_hp(self, amount: int):
        p = self.player
        p.hp = max(0, min(p.max_hp, p.hp + _int(amount)))
        return {"hp": p.hp, "dead": p.hp == 0}

    def change_gold(self, amount: int):
        p = self.player
        amount = _int(amount)
        if p.gold + amount < 0:
            return {"error": "not enough gold", "gold": p.gold}
        p.gold += amount
        return {"gold": p.gold}

    def add_item(self, item: str):
        self.player.inventory.append(str(item))
        return {"inventory": self.player.inventory}

    def remove_item(self, item: str):
        if item not in self.player.inventory:
            return {"error": f"player does not have '{item}'", "inventory": self.player.inventory}
        self.player.inventory.remove(item)
        return {"inventory": self.player.inventory}

    def move_to(self, location: str):
        p = self.player
        if not self.world:
            p.location = location
            return {"location": location}
        dest = self.world.resolve(location)
        if dest is None:
            return {"error": f"unknown location '{location}'. Known: {self.world.names()}. "
                             "Sub-areas (cellar, upstairs) are narrated, not travelled to."}
        km = self.world.distance_km(p.location, dest)
        hours = round(km / WALK_KMH, 1)
        p.hours = round(p.hours + hours, 1)
        p.location = dest
        return {"location": dest, "distance_km": round(km, 1), "travel_hours": hours}

    def set_flag(self, flag: str):
        self.player.flags.add(str(flag))
        return {"flags": sorted(self.player.flags)}

    def call(self, name: str, args: dict):
        if name not in self.PUBLIC_TOOLS:
            return {"error": f"unknown tool '{name}'"}
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}


TOOLS = [
    {"name": "roll_check",
     "description": "Roll a d20 skill check for the player. Use for ANY risky or uncertain action. Never invent rolls.",
     "input_schema": {"type": "object", "properties": {
         "stat": {"type": "string", "enum": ["might", "cunning", "spirit"]},
         "difficulty": {"type": "integer", "description": "8 easy, 12 normal, 16 hard"}},
         "required": ["stat", "difficulty"]}},
    {"name": "change_hp", "description": "Heal (positive) or damage (negative) the player's HP.",
     "input_schema": {"type": "object", "properties": {"amount": {"type": "integer"}}, "required": ["amount"]}},
    {"name": "change_gold", "description": "Add or subtract player gold.",
     "input_schema": {"type": "object", "properties": {"amount": {"type": "integer"}}, "required": ["amount"]}},
    {"name": "add_item", "description": "Give the player an item they have legitimately obtained.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}},
    {"name": "remove_item", "description": "Take an item from the player.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}},
    {"name": "move_to",
     "description": "Travel to a known named location; the engine computes distance and travel time. No teleporting.",
     "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
    {"name": "set_flag",
     "description": "Record a story milestone the player truly earned (e.g. mira_trusts_you, found_cursed_lamp, has_ruins_key). Unlocks NPC secrets and hidden lore.",
     "input_schema": {"type": "object", "properties": {"flag": {"type": "string"}}, "required": ["flag"]}},
]
