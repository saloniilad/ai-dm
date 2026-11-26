"""Label logged turns:  python -m train.review data/dataset/teacher.jsonl
   g = good, b = bad, e = edit the reply (then good), s = skip, q = quit. Saves in place."""
import json
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r.get("quality") is None]
    print(f"{len(todo)} unreviewed of {len(rows)}")
    for r in todo:
        msgs = r["messages"]
        user = next(m for m in reversed(msgs) if m["role"] == "user")["content"]
        print("\n" + "=" * 70 + "\n" + user[-600:])
        for m in msgs:
            for tc in m.get("tool_calls", []):
                print("  tool:", tc["function"]["name"], tc["function"]["arguments"])
        print("\nREPLY:", msgs[-1]["content"])
        c = input("[g]ood / [b]ad / [e]dit / [s]kip / [q]uit > ").strip().lower()
        if c == "q":
            break
        if c == "g":
            r["quality"] = "good"
        elif c == "b":
            r["quality"] = "bad"
        elif c == "e":
            msgs[-1]["content"] = input("new reply: ")
            r["quality"] = "good"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
