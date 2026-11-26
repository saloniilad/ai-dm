"""A Session = one shared campaign (memory + DM) with several players. Used by the API and Discord bot."""
import json
import re
import threading
from pathlib import Path

from . import config
from .dataset import DatasetLogger
from .engine import Engine, Player
from .llm import get_backend
from .master import DungeonMaster
from .memory import Memory

_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class Session:
    def __init__(self, session_id, data_dir=None, backend=None, dataset_path=None):
        if not _ID.match(session_id):
            raise ValueError("session_id must be 1-40 chars of letters, digits, _ or -")
        self.id = session_id
        self.dir = Path(data_dir or config.data_dir()) / "sessions" / session_id
        self.memory = Memory(session_path=self.dir / "events.jsonl")
        path = dataset_path or config.dataset_path()
        self.dm = DungeonMaster(self.memory, backend or get_backend(), DatasetLogger(path) if path else None)
        self.engines = {}
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        f = self.dir / "state.json"
        if f.exists():
            data = json.loads(f.read_text())
            self.dm.turn = data.get("turn", 0)
            for name, snap in data.get("players", {}).items():
                self.engines[name] = Engine(Player.from_dict(snap), self.memory.world)

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "state.json").write_text(json.dumps(
            {"turn": self.dm.turn, "players": {n: e.player.snapshot() for n, e in self.engines.items()}}))

    def join(self, name) -> Engine:
        name = name.strip()[:32] or "Wanderer"
        with self.lock:
            if name not in self.engines:
                self.engines[name] = Engine(Player(name=name), self.memory.world)
                self._save()
            return self.engines[name]

    def act(self, name, action):
        eng = self.engines.get(name) or self.join(name)
        if eng.player.hp == 0:
            raise RuntimeError(f"{name} has fallen.")
        with self.lock:
            result = self.dm.turn_for(action.strip()[:500], eng, who=name)
            self._save()
        return result

    def state(self, name):
        return self.engines[name].player.snapshot()

    def recap(self):
        with self.lock:
            return self.dm.recap()
