"""Logs every DM turn as a fine-tuning example in OpenAI chat format (with tool calls)."""
import json
import time
from pathlib import Path

from .llm import to_openai_tools


class DatasetLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, system, history, user, trace, reply, tools, meta=None):
        messages = [{"role": "system", "content": system}] + list(history) + [{"role": "user", "content": user}]
        if trace:
            calls = [{"id": f"call_{i}", "type": "function",
                      "function": {"name": t["name"], "arguments": json.dumps(t["args"])}} for i, t in enumerate(trace)]
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            for i, t in enumerate(trace):
                messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps(t["result"])})
        messages.append({"role": "assistant", "content": reply})
        row = {"messages": messages, "tools": to_openai_tools(tools), "quality": None,
               "meta": {"ts": time.time(), **(meta or {})}}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
