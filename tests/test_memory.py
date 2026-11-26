import json

from dm.memory import Memory
from dm.retrieval import Retriever


def mem(**kw):
    return Memory(retriever=Retriever(**kw))


def test_secret_hidden_until_flag():
    m = mem()
    for d in m.search("Mira barkeep tavern secret key floorboard", flags=set(), k=10):
        assert "floorboard" not in d["text"].lower()
    docs = m.search("Mira barkeep tavern secret key", flags={"mira_trusts_you"}, k=10)
    assert any("floorboard" in d["text"].lower() for d in docs)


def test_hidden_lore_requires_flag():
    m = mem()
    assert all(d["id"] != "lore_vault" for d in m.search("flooded vault lotus petals seal", set(), k=10))
    assert any(d["id"] == "lore_vault" for d in m.search("flooded vault lotus petals seal", {"has_ruins_key"}, k=10))


def test_internal_fields_never_leak():
    m = mem()
    for d in m.search("Kavi smuggler", set(), k=10):
        assert not {"secret", "secret_keywords", "reveal_flag"} & set(d)


def test_basic_retrieval_ranks_gold_first():
    assert mem().search("Who is the ferryman at Mist Ferry", set(), k=3)[0]["id"] in {"npc_boran", "loc_ferry"}


def test_geo_bonus_prefers_nearby_when_ambiguous():
    m = mem()
    m.add_event("A strange lamp flickered.", "Naga Ruins", 1, id="far")
    m.add_event("A strange lamp flickered.", "Vindhya Crossroads", 2, id="near")
    ids = [d["id"] for d in m.search("strange lamp flickered", set(), near_loc="Vindhya Crossroads", k=2)]
    assert ids[0] == "near"


def test_persistence_and_compaction(tmp_path):
    p = tmp_path / "ev.jsonl"
    m = Memory(session_path=p, retriever=Retriever())
    for i in range(12):
        m.add_event(f"event number {i}", "Mist Ferry", i, id=f"e{i}")
    assert len(Memory(session_path=p).events()) == 12
    assert m.compact(lambda t: "SUMMARY", threshold=10, keep_recent=4)
    kinds = [d["kind"] for d in m.events()]
    assert kinds.count("summary") == 1 and kinds.count("event") == 4
    assert len(Memory(session_path=p).events()) == 5
