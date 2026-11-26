"""Offline retrieval benchmark (no LLM needed): recall@k and MRR over the campaign QA set.
   python -m eval.retrieval_eval [--dense] [--min-recall5 0.85]"""
import argparse
import sys

from dm.retrieval import Retriever

from .common import campaign_memory, load_json, save, table


def run(cfg_name, retriever, qa, use_geo):
    mem = campaign_memory(retriever)
    items = [i for i in qa["items"] if i.get("gold_ids")]
    hits = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    for it in items:
        docs = mem.search(it["q"], set(), near_loc=qa["player_location"] if use_geo else None, k=5)
        ids = [d["id"] for d in docs]
        rank = next((i + 1 for i, d in enumerate(ids) if d in it["gold_ids"]), None)
        for k in hits:
            hits[k] += bool(rank and rank <= k)
        rr += 1 / rank if rank else 0
    n = len(items)
    return {"config": cfg_name, "n": n, "recall@1": hits[1] / n, "recall@3": hits[3] / n,
            "recall@5": hits[5] / n, "mrr": rr / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--min-recall5", type=float, default=None)
    a = ap.parse_args()
    qa = load_json("qa.json")
    results = [run("bm25", Retriever(geo_weight=0), qa, False),
               run("bm25+geo", Retriever(), qa, True)]
    if a.dense:
        results.append(run("hybrid (bm25+dense)", Retriever(dense=True, geo_weight=0), qa, False))
        results.append(run("hybrid+geo", Retriever(dense=True), qa, True))
        results.append(run("hybrid+geo+rerank", Retriever(dense=True, rerank=True), qa, True))
    rows = [[r["config"], r["n"], f"{r['recall@1']:.2f}", f"{r['recall@3']:.2f}", f"{r['recall@5']:.2f}", f"{r['mrr']:.2f}"]
            for r in results]
    print(table(rows, ["config", "n", "recall@1", "recall@3", "recall@5", "MRR"]))
    save("retrieval", results)
    if a.min_recall5 is not None and results[0]["recall@5"] < a.min_recall5:
        print(f"FAIL: bm25 recall@5 {results[0]['recall@5']:.2f} < {a.min_recall5}")
        sys.exit(1)


if __name__ == "__main__":
    main()
