"""LLM backends behind one interface:
   generate(system, history, user, tools, execute) -> (text, tool_trace)
   - anthropic : Claude via the Anthropic API
   - openai    : any OpenAI-compatible server (vLLM / Ollama / llama.cpp) -> use for your fine-tuned model
   - local     : your fine-tuned LoRA (models/dm-lora) loaded in-process with transformers + peft
   - mock      : offline, deterministic, no network (for tests and demos)"""
import json
import re
import threading
from pathlib import Path

from . import config
from .retrieval import tokenize

MAX_TOOL_ROUNDS = 6


def to_openai_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or config.model()

    def generate(self, system, history, user, tools=None, execute=None, max_tokens=600):
        messages = list(history) + [{"role": "user", "content": user}]
        kw = dict(model=self.model, max_tokens=max_tokens, system=system)
        if tools:
            kw["tools"] = tools
        trace = []
        resp = self.client.messages.create(messages=messages, **kw)
        rounds = 0
        while resp.stop_reason == "tool_use" and execute and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    r = execute(b.name, b.input)
                    trace.append({"name": b.name, "args": dict(b.input), "result": r})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(r)})
            messages.append({"role": "user", "content": results})
            resp = self.client.messages.create(messages=messages, **kw)
        return "".join(b.text for b in resp.content if b.type == "text").strip(), trace


class OpenAICompatBackend:
    name = "openai"

    def __init__(self, model=None, base_url=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or config.openai_base_url(), api_key=api_key or config.openai_api_key())
        self.model = model or config.model()

    def generate(self, system, history, user, tools=None, execute=None, max_tokens=600):
        msgs = [{"role": "system", "content": system}] + list(history) + [{"role": "user", "content": user}]
        oa_tools = to_openai_tools(tools) if tools else None
        trace = []
        for _ in range(MAX_TOOL_ROUNDS + 1):
            kw = {"tools": oa_tools} if oa_tools else {}
            r = self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens, **kw)
            m = r.choices[0].message
            if not (m.tool_calls and execute):
                return (m.content or "").strip(), trace
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in m.tool_calls]})
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                res = execute(tc.function.name, args)
                trace.append({"name": tc.function.name, "args": args, "result": res})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res)})
        return "", trace


_LOCAL_CACHE = {}
_LOAD_LOCK = threading.Lock()
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class LocalLoRABackend:
    """Runs base model + your LoRA adapter directly (no server). Same prompt/tool format as training:
    Qwen chat template, Hermes-style <tool_call> JSON. Model loads once per process and is shared."""
    name = "local"

    def __init__(self, model=None, adapter=None, base=None):
        self.adapter = str(adapter or config.adapter_path())
        self.base = base or config.base_model()
        self.model = model or "dm-lora"
        self._load()

    def _load(self):
        key = (self.base, self.adapter)
        with _LOAD_LOCK:                       # several web requests can arrive at once: load only one copy
            if key not in _LOCAL_CACHE:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
                if not (Path(self.adapter) / "adapter_config.json").exists():
                    raise FileNotFoundError(f"No adapter at {self.adapter}. Put your dm-lora folder there "
                                            "or set DM_ADAPTER_PATH.")
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if dev == "cuda" else torch.float32
                tok = AutoTokenizer.from_pretrained(self.adapter)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                base = AutoModelForCausalLM.from_pretrained(self.base, dtype=dtype)
                base.tie_weights()             # Qwen2.5-1.5B shares lm_head with the embeddings
                m = PeftModel.from_pretrained(base, self.adapter).merge_and_unload().to(dev).eval()
                m.tie_weights()
                if m.config.tie_word_embeddings:
                    assert m.get_output_embeddings().weight.data_ptr() == m.get_input_embeddings().weight.data_ptr(), \
                        "lm_head is not tied to the embeddings - outputs would be garbage"
                _LOCAL_CACHE[key] = (tok, m, dev, threading.Lock())
        self.tok, self.lm, self.dev, self.lock = _LOCAL_CACHE[key]

    def _complete(self, msgs, tools, max_tokens):
        import torch
        prompt = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        inp = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.dev)
        with self.lock, torch.no_grad():
            out = self.lm.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                                   pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
        return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    def generate(self, system, history, user, tools=None, execute=None, max_tokens=600):
        msgs = [{"role": "system", "content": system}] + list(history) + [{"role": "user", "content": user}]
        oa_tools = to_openai_tools(tools) if tools else None
        trace = []
        for _ in range(MAX_TOOL_ROUNDS + 1):
            out = self._complete(msgs, oa_tools, min(max_tokens, 300))
            calls = _TOOL_RE.findall(out) if (oa_tools and execute) else []
            if not calls:
                return _TOOL_RE.sub("", out).strip(), trace
            tcs, results = [], []
            for i, c in enumerate(calls):
                try:
                    d = json.loads(c)
                    name, args = d["name"], d.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                except (json.JSONDecodeError, KeyError, TypeError):
                    name, args = "invalid_call", {}
                res = execute(name, args)
                trace.append({"name": name, "args": args, "result": res})
                tcs.append({"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}})
                results.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps(res)})
            msgs = msgs + [{"role": "assistant", "content": "", "tool_calls": tcs}] + results
        return "", trace


class MockBackend:
    """Offline stand-in. For recall questions it returns the context line that best overlaps the
    question, so eval numbers reflect *retrieval quality only*, not model quality."""
    name = "mock"
    model = "mock"

    STATS = [("might", ("fight", "attack", "strike", "lift", "break", "climb", "leap", "jump", "outrun", "run")),
             ("spirit", ("pray", "ritual", "convince", "persuade", "lie", "calm", "chant")),
             ("cunning", ("sneak", "steal", "pick", "lock", "search", "trick", "hide"))]
    RISKY = ("try", "attempt", "sneak", "steal", "pick", "leap", "jump", "climb", "fight", "attack", "outrun",
             "convince", "lie", "break")

    def generate(self, system, history, user, tools=None, execute=None, max_tokens=600):
        if "[[RECALL]]" in system:
            m = re.search(r"CONTEXT\n(.*?)\n\nQUESTION:(.*)", user, re.S)
            ctx, q = (m.group(1), m.group(2)) if m else ("", user)
            qt, best, best_n = set(tokenize(q)), "", 0
            for line in ctx.splitlines():
                n = len(qt & set(tokenize(line)))
                if n > best_n:
                    best, best_n = line, n
            return (best if best_n >= 2 else "UNKNOWN"), []
        if "[[SUMMARIZE]]" in system:
            return user[:400], []
        if "[[RECAP]]" in system:
            return "Previously: " + user[-400:].replace("\n", " "), []
        action = user.rsplit("ACTION:", 1)[-1].strip()
        low, trace, roll = action.lower(), [], None
        if execute and any(w in low for w in self.RISKY):
            stat = next((s for s, ws in self.STATS if any(w in low for w in ws)), "cunning")
            roll = execute("roll_check", {"stat": stat, "difficulty": 12})
            trace.append({"name": "roll_check", "args": {"stat": stat, "difficulty": 12}, "result": roll})
        out = f"[mock DM] You try: '{action}'."
        if roll:
            out += f" You roll {roll['die']} (total {roll['total']} vs {roll['difficulty']}): {roll['outcome']}."
        return out + " What do you do?", trace


def get_backend(name=None, model=None, base_url=None):
    name = name or config.backend_name()
    if name == "anthropic":
        return AnthropicBackend(model)
    if name == "openai":
        return OpenAICompatBackend(model, base_url)
    if name == "local":
        return LocalLoRABackend(model)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend '{name}'")