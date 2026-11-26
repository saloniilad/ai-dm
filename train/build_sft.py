"""Turn logged turns into train/val files for SFT.
   python -m train.build_sft data/dataset/*.jsonl --out data/sft [--include-unlabeled]"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--include-unlabeled", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seen, rows = set(), []
    for f in a.files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            q = r.get("quality")
            if q == "bad" or (q is None and not a.include_unlabeled):
                continue
            if not r["messages"][-1]["content"].strip():
                continue
            h = hashlib.md5(json.dumps(r["messages"], sort_keys=True).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            rows.append({"messages": r["messages"], "tools": r["tools"]})
    random.Random(a.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * a.val_frac)) if len(rows) > 1 else 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in part))
    print(f"{len(rows) - n_val} train / {n_val} val -> {out}")


if __name__ == "__main__":
    main()
