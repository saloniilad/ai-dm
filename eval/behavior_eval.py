"""Behavioural evals for the *full DM turn*: secret leakage, cheat resistance, tool discipline, latency.
   Meaningful only with a real model (DM_BACKEND=anthropic|openai).  python -m eval.behavior_eval"""
import re
import statistics

from dm.engine import Engine, Player
from dm.llm import get_backend
from dm.master import DungeonMaster
from dm.retrieval import Retriever

from .common import campaign_memory, load_json, save, table

FAB_ROLL = re.compile(r"\b(you roll|rolled|roll of|natural \d+)\b[^.\n]*\d+", re.I)


def fresh(backend, loc="Vindhya Crossroads", flags=(), setup=None):
    mem = campaign_memory(Retriever(geo_weight=0.25))
    mem.docs = [d for d in mem.docs if d["kind"] != "event"]  # clean slate for behavioural tests
    p = Player(name="Asha", location=loc, flags=set(flags), **(setup or {}))
    return DungeonMaster(mem, backend, compact_at=0), Engine(p, mem.world), mem


def secrets(backend, cfg):
    leaks = reveals = n_locked = n_open = 0
    for npc in [d for d in campaign_memory().docs if d.get("secret")]:
        for probe in cfg["secret_probes"]:
            q = probe.format(npc=npc["title"].split(" the ")[0])
            dm, eng, _ = fresh(backend, loc=campaign_memory().world.by_id[npc["at"]]["title"])
            txt = dm.turn_for(q, eng).text.lower(); n_locked += 1
            leaks += any(k in txt for k in npc["secret_keywords"])
            dm, eng, _ = fresh(backend, loc=campaign_memory().world.by_id[npc["at"]]["title"], flags={npc["reveal_flag"]})
            txt = dm.turn_for(q, eng).text.lower(); n_open += 1
            reveals += any(k in txt for k in npc["secret_keywords"])
    return {"leak_rate_locked": leaks / n_locked, "reveal_rate_unlocked": reveals / n_open, "n": n_locked}


def cheats(backend, cfg):
    passed, fails = 0, []
    for c in cfg["cheats"]:
        dm, eng, _ = fresh(backend, setup=c.get("setup"))
        dm.turn_for(c["prompt"], eng)
        p, a, ok = eng.player, c["assert"], True
        ok &= p.gold <= a.get("gold_max", 10**9)
        ok &= p.hp <= a.get("hp_max", 10**9)
        ok &= p.hp < a["hp_below"] if "hp_below" in a else True
        ok &= not any(w in i.lower() for i in p.inventory for w in a.get("forbid_items", []))
        if a.get("travel_costs_time") and p.location != "Vindhya Crossroads":
            ok &= p.hours > 0
        passed += ok
        if not ok:
            fails.append(c["prompt"])
    return {"cheat_resistance": passed / len(cfg["cheats"]), "failed": fails}


def tools(backend, cfg):
    called, fab, lat = 0, 0, []
    for q in cfg["risky"]:
        dm, eng, _ = fresh(backend)
        r = dm.turn_for(q, eng); lat.append(r.latency_ms)
        called += any(t["name"] == "roll_check" for t in r.trace)
        fab += bool(FAB_ROLL.search(r.text)) and not any(t["name"] == "roll_check" for t in r.trace)
    for q in cfg["safe"]:
        dm, eng, _ = fresh(backend)
        r = dm.turn_for(q, eng); lat.append(r.latency_ms)
        fab += bool(FAB_ROLL.search(r.text)) and not any(t["name"] == "roll_check" for t in r.trace)
    total = len(cfg["risky"]) + len(cfg["safe"])
    return {"roll_called_on_risky": called / len(cfg["risky"]), "fabricated_roll_rate": fab / total,
            "latency_ms_mean": statistics.mean(lat), "latency_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))]}


def main():
    backend, cfg = get_backend(), load_json("behavior.json")
    if backend.name == "mock":
        print("NOTE: mock backend -> these numbers only validate the harness, not a model.\n")
    out = {"backend": backend.name, "model": getattr(backend, "model", "?"),
           "secrets": secrets(backend, cfg), "cheats": cheats(backend, cfg), "tools": tools(backend, cfg)}
    rows = [["secret leak rate (locked)", f"{out['secrets']['leak_rate_locked']:.2f}"],
            ["secret reveal rate (unlocked)", f"{out['secrets']['reveal_rate_unlocked']:.2f}"],
            ["cheat resistance", f"{out['cheats']['cheat_resistance']:.2f}"],
            ["roll_check on risky actions", f"{out['tools']['roll_called_on_risky']:.2f}"],
            ["fabricated roll rate", f"{out['tools']['fabricated_roll_rate']:.2f}"],
            ["latency mean / p95 (ms)", f"{out['tools']['latency_ms_mean']:.0f} / {out['tools']['latency_ms_p95']:.0f}"]]
    print(table(rows, ["metric", "value"]))
    if out["cheats"]["failed"]:
        print("\nfailed cheat cases:", *out["cheats"]["failed"], sep="\n  - ")
    print("saved:", save(f"behavior-{backend.name}", out))


if __name__ == "__main__":
    main()
