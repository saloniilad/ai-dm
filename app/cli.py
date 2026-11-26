"""Terminal client:  python -m app.cli --name Asha"""
import argparse

from dm.session import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--session", default="local")
    a = ap.parse_args()
    name = a.name or input("Name your hero: ").strip() or "Wanderer"
    s = Session(a.session)
    eng = s.join(name)
    print(f"\n[{s.dm.backend.name}] Welcome, {name}. You are at {eng.player.location}."
          " Commands: /state /recap /quit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/state":
            print(s.state(name)); continue
        if line == "/recap":
            print(s.recap()); continue
        try:
            print("\n" + s.act(name, line).text + "\n")
        except RuntimeError as e:
            print(e); break


if __name__ == "__main__":
    main()
