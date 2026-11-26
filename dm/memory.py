"""Campaign memory: static lore + growing session events, with secret gating,
flag-gated hidden lore, geo-aware retrieval and summary compaction."""
import json
from pathlib import Path

from . import config
from .geo import World
from .retrieval import Retriever

HIDDEN_FIELDS = ("secret", "secret_keywords", "reveal_flag", "requires_flag")


class Memory:
    def __init__(self, lore_path=None, session_path=None, retriever=None):
        self.docs = json.loads(Path(lore_path or config.LORE_PATH).read_text())
        self.lore_ids = {d["id"] for d in self.docs}
        self.world = World.from_docs(self.docs)
        self.retriever = retriever or Retriever(dense=config.use_dense(), rerank=config.use_rerank())
        self.session_path = Path(session_path) if session_path else None
        if self.session_path and self.session_path.exists():
            for line in self.session_path.read_text().splitlines():
                if line.strip():
                    self.docs.append(json.loads(line))

    # ---- writing -------------------------------------------------------
    def add_event(self, text, location, turn, id=None):
        doc = {"id": id or f"event_{len(self.docs)}", "kind": "event",
               "title": f"Turn {turn} at {location}", "location": location, "turn": turn, "text": text}
        self.docs.append(doc)
        if self.session_path:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session_path.open("a") as f:
                f.write(json.dumps(doc) + "\n")
        return doc

    def events(self):
        return [d for d in self.docs if d["kind"] in ("event", "summary")]

    def recent_events(self, n):
        return self.events()[-n:] if n > 0 else []

    def compact(self, summarize, threshold=40, keep_recent=15):
        """Fold the oldest events into one summary doc once the log grows past `threshold`."""
        evs = [d for d in self.docs if d["kind"] == "event"]
        if len(evs) <= threshold:
            return False
        old = evs[:-keep_recent]
        summary = summarize("\n".join(d["text"] for d in old))
        old_ids = {d["id"] for d in old}
        n = sum(1 for d in self.docs if d["kind"] == "summary")
        new = {"id": f"summary_{n}", "kind": "summary", "title": "Summary of earlier events", "text": summary}
        first = next(i for i, d in enumerate(self.docs) if d["id"] in old_ids)
        self.docs = [d for d in self.docs if d["id"] not in old_ids]
        self.docs.insert(first, new)
        self._flush()
        return True

    def _flush(self):
        if not self.session_path:
            return
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session_path.open("w") as f:
            for d in self.docs:
                if d["id"] not in self.lore_ids:
                    f.write(json.dumps(d) + "\n")

    # ---- reading -------------------------------------------------------
    def coords_of(self, d):
        if d.get("coords"):
            return tuple(d["coords"])
        if d.get("at"):
            return self.world.coords_by_id(d["at"])
        if d.get("location"):
            return self.world.coords(d["location"])
        return None

    def search(self, query, flags=frozenset(), near_loc=None, k=4):
        visible = [d for d in self.docs if not d.get("requires_flag") or d["requires_flag"] in flags]
        near = self.world.coords(near_loc) if near_loc else None
        hits = self.retriever.search(query, visible, k=k, near=near, coords_of=self.coords_of)
        return [self._gate(d, flags) for d in hits]

    @staticmethod
    def _gate(doc, flags):
        """The model never sees a secret the player has not earned."""
        out = {k: v for k, v in doc.items() if k not in HIDDEN_FIELDS}
        if doc.get("secret") and doc.get("reveal_flag") in flags:
            out["text"] += " SECRET (revealable now): " + doc["secret"]
        return out
