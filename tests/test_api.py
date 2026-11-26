import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DM_BACKEND", "mock")
    monkeypatch.setenv("DM_DATA_DIR", str(tmp_path))
    from app.api import create_app
    return TestClient(create_app())


def test_full_flow(client):
    r = client.post("/sessions", json={"player": "Asha"}).json()
    sid = r["session_id"]
    assert r["state"]["hp"] == 20 and r["backend"] == "mock"
    a = client.post(f"/sessions/{sid}/act", json={"player": "Asha", "action": "I try to leap the gap"}).json()
    assert a["narration"] and a["tools"][0]["name"] == "roll_check"
    assert client.get(f"/sessions/{sid}/state", params={"player": "Asha"}).status_code == 200
    assert "recap" in client.get(f"/sessions/{sid}/recap").json()


def test_multiplayer_and_errors(client):
    sid = client.post("/sessions", json={"player": "A"}).json()["session_id"]
    assert client.post(f"/sessions/{sid}/players", json={"player": "B"}).status_code == 200
    assert client.get("/sessions/nope/recap").status_code == 404
    assert client.post("/sessions", json={"player": "A", "session_id": "../evil"}).status_code == 400
    assert client.post(f"/sessions/{sid}/act", json={"player": "A", "action": ""}).status_code == 422


def test_state_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("DM_BACKEND", "mock")
    monkeypatch.setenv("DM_DATA_DIR", str(tmp_path))
    from app.api import create_app
    c1 = TestClient(create_app())
    c1.post("/sessions", json={"player": "Asha", "session_id": "keep"})
    c1.post("/sessions/keep/act", json={"player": "Asha", "action": "I look around"})
    c2 = TestClient(create_app())
    r = c2.post("/sessions", json={"player": "Asha", "session_id": "keep"}).json()
    assert r["state"]["name"] == "Asha"
    assert len(c2.get("/sessions/keep/recap").json()["recap"]) > 0
