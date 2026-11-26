"""Tests the local backend's tool-call loop with a scripted fake model (no GPU / download needed)."""
from dm.engine import Engine, Player, TOOLS
from dm.llm import LocalLoRABackend


def make(outputs):
    b = LocalLoRABackend.__new__(LocalLoRABackend)   # skip model loading
    it = iter(outputs)
    b._complete = lambda msgs, tools, n: next(it)
    return b


def test_tool_call_then_narration():
    eng = Engine(Player("Asha"))
    b = make(['<tool_call>\n{"name": "change_gold", "arguments": {"amount": -2}}\n</tool_call>',
              "Two gold changes hands. What do you do?"])
    text, trace = b.generate("sys", [], "ACTION: pay", TOOLS, eng.call)
    assert text.startswith("Two gold") and trace[0]["name"] == "change_gold"
    assert eng.player.gold == 8


def test_bad_json_does_not_crash():
    eng = Engine(Player("Asha"))
    b = make(['<tool_call>{"name": "change_gold", "arguments": {oops}}</tool_call>', "Fine."])
    text, trace = b.generate("sys", [], "x", TOOLS, eng.call)
    assert text == "Fine." and "error" in trace[0]["result"]
