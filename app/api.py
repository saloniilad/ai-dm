"""REST API:  uvicorn app.api:app --reload"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dm.session import Session

WEB = Path(__file__).parent / "web"


class NewSession(BaseModel):
    player: str = Field("Wanderer", max_length=32)
    session_id: str | None = None


class Join(BaseModel):
    player: str = Field(..., max_length=32)


class Act(BaseModel):
    player: str = Field(..., max_length=32)
    action: str = Field(..., min_length=1, max_length=500)


def create_app() -> FastAPI:
    api = FastAPI(title="AI Dungeon Master", version="1.0")
    from dm import config
    if config.backend_name() == "local":      # load the model once at startup, not on the first click
        from dm.llm import get_backend
        get_backend()
    sessions: dict[str, Session] = {}

    def get(sid: str) -> Session:
        if sid not in sessions:
            raise HTTPException(404, "unknown session")
        return sessions[sid]

    @api.get("/health")
    def health():
        return {"ok": True, "sessions": len(sessions)}

    @api.post("/sessions")
    def new_session(body: NewSession):
        import secrets
        sid = body.session_id or secrets.token_hex(4)
        try:
            if sid not in sessions:
                sessions[sid] = Session(sid)
        except ValueError as e:
            raise HTTPException(400, str(e))
        eng = sessions[sid].join(body.player)
        return {"session_id": sid, "backend": sessions[sid].dm.backend.name, "state": eng.player.snapshot()}

    @api.post("/sessions/{sid}/players")
    def join(sid: str, body: Join):
        return {"state": get(sid).join(body.player).player.snapshot()}

    @api.post("/sessions/{sid}/act")
    def act(sid: str, body: Act):
        s = get(sid)
        try:
            r = s.act(body.player, body.action)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"narration": r.text, "tools": r.trace, "latency_ms": round(r.latency_ms, 1),
                "context": r.context_ids, "state": s.state(body.player)}

    @api.get("/sessions/{sid}/state")
    def state(sid: str, player: str):
        s = get(sid)
        if player not in s.engines:
            raise HTTPException(404, "unknown player")
        return s.state(player)

    @api.get("/sessions/{sid}/recap")
    def recap(sid: str):
        return {"recap": get(sid).recap()}

    @api.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    return api


app = create_app()