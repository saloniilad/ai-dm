import json
import re
from pathlib import Path

from dm.memory import Memory
from dm.retrieval import Retriever

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"


def load_json(name):
    return json.loads((DATA / name).read_text())


def campaign_memory(retriever=None):
    """Lore + scripted 20-turn campaign log, in memory only."""
    m = Memory(retriever=retriever or Retriever())
    for e in load_json("campaign_log.json"):
        m.add_event(e["text"], e["location"], e["turn"], id=e["id"])
    return m


def check_answer(reply, item):
    low = reply.lower()
    if item.get("expect_unknown"):
        return "unknown" in low or "not established" in low or "no record" in low
    return all(any(k in low for k in group) for group in item["any_of"])


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def table(rows, headers):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(headers), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])
