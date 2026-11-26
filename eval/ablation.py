"""Consistency ablation over the campaign QA set:
     prompt_only : recency window only (no retrieval)
     rag         : hybrid retrieval over lore + session events
     rag_ft      : same, but answered by your fine-tuned model (needs --ft-model, --ft-base-url)
   python -m eval.ablation [--ft-model dm-lora --ft-base-url http://localhost:8000/v1]"""
import argparse
import statistics
import time

from dm.llm import OpenAICompatBackend, get_backend
from dm.master import DungeonMaster
from dm.retrieval import Retriever

from .common import campaign_memory, check_answer, load_json, save, table


def run_condition(name, backend, use_rag, qa, dense=False):
    mem = campaign_memory(Retriever(dense=dense))
    dm = DungeonMaster(mem, backend, use_rag=use_rag)
    from dm.engine import Engine, Player
    eng = Engine(Player(name="Asha", location=qa["player_location"]), mem.world)
    ok_a = tot_a = ok_u = tot_u = 0
    lat, fails = [], []
    for it in qa["items"]:
        t0 = time.perf_counter()
        reply = dm.ask(it["q"], eng)
        lat.append((time.perf_counter() - t0) * 1000)
        ok = check_answer(reply, it)
        if it.get("expect_unknown"):
            tot_u += 1; ok_u += ok
        else:
            tot_a += 1; ok_a += ok
        if not ok:
            fails.append({"q": it["q"], "reply": reply[:160]})
    return {"condition": name, "answer_acc": ok_a / tot_a, "abstain_acc": ok_u / tot_u,
            "latency_ms_mean": statistics.mean(lat), "failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-model"); ap.add_argument("--ft-base-url")
    ap.add_argument("--dense", action="store_true")
    a = ap.parse_args()
    qa = load_json("qa.json")
    base = get_backend()
    if base.name == "mock":
        print("NOTE: mock backend -> numbers measure retrieval only, not model quality.\n")
    res = [run_condition("prompt_only", base, False, qa), run_condition("rag", base, True, qa, a.dense)]
    if a.ft_model:
        res.append(run_condition("rag_ft", OpenAICompatBackend(a.ft_model, a.ft_base_url), True, qa, a.dense))
    rows = [[r["condition"], f"{r['answer_acc']:.2f}", f"{r['abstain_acc']:.2f}", f"{r['latency_ms_mean']:.0f}"] for r in res]
    print(table(rows, ["condition", "answer acc", "abstain acc", "latency ms"]))
    for r in res:
        if r["failures"]:
            print(f"\n[{r['condition']}] sample failures:")
            for f in r["failures"][:3]:
                print("  -", f["q"], "=>", f["reply"])
    print("saved:", save(f"ablation-{base.name}", res))


if __name__ == "__main__":
    main()
