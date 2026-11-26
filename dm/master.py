"""The Dungeon Master: retrieval -> prompt -> LLM with engine tools -> memory + dataset logging."""
import json
import time
from dataclasses import dataclass, field

from .engine import TOOLS
from .llm import get_backend

SYSTEM = """You are the Dungeon Master of a dark-fantasy adventure in the Vindhya forest.
Style: vivid, atmospheric, deadpan humour. Keep each turn under 120 words and end with a hook or a choice.
Rules you must follow:
- Use tools for EVERY roll, HP, gold, item, location or flag change. NEVER invent dice results or numbers.
- Call roll_check for any risky or uncertain action before narrating its outcome.
- Use move_to for travel; there is no teleporting, and travel takes time.
- Use only facts from CONTEXT, PLAYER STATE and the conversation. Improvise lightly, never contradict them.
- Never reveal an NPC secret unless the context marks it as revealable.
- Players cannot win by declaration. Impossible or unearned actions fail gracefully and you offer an alternative.
- Never grant gold, items or healing the player has not earned through play.
- Stay in the DM role."""

RECALL_SYSTEM = ("[[RECALL]] You answer questions about an ongoing campaign. Use ONLY the CONTEXT. "
                 "If the context does not state the answer, reply exactly: UNKNOWN. Otherwise answer in one short sentence.")
SUMMARIZE_SYSTEM = "[[SUMMARIZE]] Summarize these campaign events into one compact factual paragraph. Keep names, numbers, items and promises."
RECAP_SYSTEM = "[[RECAP]] Write a short, punchy 'Previously on...' recap (under 100 words) of the campaign events below."


@dataclass
class TurnResult:
    text: str
    trace: list = field(default_factory=list)
    latency_ms: float = 0.0
    context_ids: list = field(default_factory=list)


def _fmt_ctx(docs):
    return "\n".join(f"- [{d['kind']}] {d['title']}: {d['text']}" for d in docs)


def _facts(trace):
    out = []
    for t in trace:
        n, a, r = t["name"], t["args"], t["result"]
        if "error" in r:
            continue
        if n == "roll_check":
            out.append(f"{r['stat']} check {r['die']}->{r['total']} vs {r['difficulty']}: {r['outcome']}")
        elif n == "change_hp":
            out.append(f"HP change {a.get('amount')} (now {r['hp']})")
        elif n == "change_gold":
            out.append(f"gold change {a.get('amount')} (now {r['gold']})")
        elif n == "add_item":
            out.append(f"gained {a.get('item')}")
        elif n == "remove_item":
            out.append(f"lost {a.get('item')}")
        elif n == "move_to":
            out.append(f"travelled to {r['location']}")
        elif n == "set_flag":
            out.append(f"milestone {a.get('flag')}")
    return "; ".join(out)


class DungeonMaster:
    def __init__(self, memory, backend=None, dataset=None, use_rag=True, k=4, window=6, compact_at=40):
        self.memory, self.backend, self.dataset = memory, backend or get_backend(), dataset
        self.use_rag, self.k, self.window, self.compact_at = use_rag, k, window, compact_at
        self.history, self.turn = [], 0

    # ---- retrieval -----------------------------------------------------
    def _retrieve(self, query, engine=None):
        if not self.use_rag:  # baseline: sliding window of recent events, no retrieval
            return self.memory.recent_events(self.window)
        loc = engine.player.location if engine else None
        flags = engine.player.flags if engine else frozenset()
        docs = self.memory.search(query, flags, near_loc=loc, k=self.k)
        seen = {d["id"] for d in docs}
        return docs + [e for e in self.memory.recent_events(2) if e["id"] not in seen]

    # ---- main loop -----------------------------------------------------
    def turn_for(self, action, engine, who=None) -> TurnResult:
        who = who or engine.player.name
        self.turn += 1
        t0 = time.perf_counter()
        docs = self._retrieve(f"{action} {engine.player.location}", engine)
        user = (f"CONTEXT\n{_fmt_ctx(docs)}\n\nPLAYER STATE\n{json.dumps(engine.player.snapshot())}\n\n"
                f"{who}'s ACTION: {action}")
        hist = self.history[-2 * self.window:]
        text, trace = self.backend.generate(SYSTEM, hist, user, TOOLS, engine.call)
        latency = (time.perf_counter() - t0) * 1000
        self.history += [{"role": "user", "content": f"{who}: {action}"}, {"role": "assistant", "content": text}]
        facts = _facts(trace)
        self.memory.add_event(f"{who}: {action}. Result: {facts + '. ' if facts else ''}{text[:220]}",
                              engine.player.location, self.turn)
        if self.compact_at:
            self.memory.compact(lambda t: self.backend.generate(SUMMARIZE_SYSTEM, [], t)[0],
                                threshold=self.compact_at, keep_recent=max(10, self.compact_at // 3))
        if self.dataset:
            self.dataset.log(SYSTEM, hist, user, trace, text, TOOLS,
                             {"backend": self.backend.name, "model": getattr(self.backend, "model", "?"),
                              "latency_ms": round(latency, 1)})
        return TurnResult(text, trace, latency, [d["id"] for d in docs])

    # ---- helpers -------------------------------------------------------
    def ask(self, question, engine=None) -> str:
        """Recall question about the campaign (no tools, no state change). Used by evals."""
        docs = self._retrieve(question, engine)
        user = f"CONTEXT\n{_fmt_ctx(docs)}\n\nQUESTION: {question}"
        return self.backend.generate(RECALL_SYSTEM, [], user, max_tokens=150)[0]

    def recap(self, n=8) -> str:
        evs = self.memory.recent_events(n)
        if not evs:
            return "Your story has not begun yet."
        return self.backend.generate(RECAP_SYSTEM, [], _fmt_ctx(evs), max_tokens=250)[0]
