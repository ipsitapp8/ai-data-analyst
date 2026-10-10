"""Dataset knowledge + lessons. Embeddings are faked so no Gemini call is made."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app import config, knowledge, memory
from app.config import EMBEDDING_DIM
from app.database import SessionLocal
from app.models import KnowledgeNote, Question

KEYWORDS = ["revenue", "churn", "region", "refund"]


def fake_embed(texts, task_type):
    out = []
    for t in texts:
        v = np.zeros(EMBEDDING_DIM)
        v[: len(KEYWORDS)] = [float(k in t.lower()) + 0.01 for k in KEYWORDS]
        out.append((v / np.linalg.norm(v)).tolist())
    return out


@pytest.fixture(autouse=True)
def _fake_embeddings():
    with patch.object(memory, "embed", fake_embed), patch.object(knowledge, "embed", fake_embed):
        yield


def _signup(client, email):
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123", "display_name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _team_headers(client, token, name):
    h = {"Authorization": f"Bearer {token}"}
    cid = client.post("/api/communities", json={"name": name}, headers=h).json()["id"]
    tid = client.post(f"/api/communities/{cid}/teams", json={"name": name}, headers=h).json()["id"]
    return {**h, "X-Team-Id": str(tid)}, tid


@pytest.fixture
def setup(client, unique_email):
    headers, team_id = _team_headers(client, _signup(client, f"k-{unique_email}"), "K")
    files = {"file": ("t.csv", b"a,b\n1,2\n3,4\n", "text/csv")}
    ds = client.post("/api/datasets/upload", files=files, headers=headers).json()["id"]
    return {"h": headers, "team": team_id, "ds": ds}


def test_note_crud_roundtrip(client, setup):
    h, ds = setup["h"], setup["ds"]
    r = client.post(f"/api/datasets/{ds}/notes", json={"kind": "knowledge", "text": "  amount is in cents  "}, headers=h)
    assert r.status_code == 201, r.text
    note = r.json()
    assert note["text"] == "amount is in cents" and note["source"] == "user"

    listed = client.get(f"/api/datasets/{ds}/notes", headers=h).json()
    assert [n["id"] for n in listed] == [note["id"]]
    assert client.get(f"/api/datasets/{ds}/notes?kind=lesson", headers=h).json() == []

    assert client.delete(f"/api/datasets/{ds}/notes/{note['id']}", headers=h).status_code == 204
    assert client.get(f"/api/datasets/{ds}/notes", headers=h).json() == []


def test_note_validation(client, setup):
    h, ds = setup["h"], setup["ds"]
    url = f"/api/datasets/{ds}/notes"
    assert client.post(url, json={"kind": "nope", "text": "x"}, headers=h).status_code == 422
    assert client.post(url, json={"kind": "lesson", "text": "   "}, headers=h).status_code == 422
    too_long = "x" * (config.NOTE_MAX_CHARS + 1)
    assert client.post(url, json={"kind": "lesson", "text": too_long}, headers=h).status_code == 422


def test_notes_are_team_scoped(client, setup, unique_email):
    h, ds = setup["h"], setup["ds"]
    note = client.post(f"/api/datasets/{ds}/notes", json={"kind": "knowledge", "text": "secret"}, headers=h).json()

    other, _ = _team_headers(client, _signup(client, f"other-{unique_email}"), "Other")
    assert client.get(f"/api/datasets/{ds}/notes", headers=other).status_code == 404
    assert client.post(f"/api/datasets/{ds}/notes", json={"kind": "knowledge", "text": "x"}, headers=other).status_code == 404
    assert client.delete(f"/api/datasets/{ds}/notes/{note['id']}", headers=other).status_code == 404
    assert len(client.get(f"/api/datasets/{ds}/notes", headers=h).json()) == 1


def test_lessons_retrieved_by_similarity_knowledge_always_included(setup):
    db = SessionLocal()
    try:
        t, ds = setup["team"], setup["ds"]
        knowledge.add_note(db, team_id=t, dataset_id=ds, kind="knowledge", text="amount is in cents")
        knowledge.add_note(db, team_id=t, dataset_id=ds, kind="lesson", text="refund rows double count revenue")
        knowledge.add_note(db, team_id=t, dataset_id=ds, kind="lesson", text="churn needs a 30 day window")

        lessons = knowledge.retrieve_lessons(db, team_id=t, dataset_id=ds, question_text="total revenue with refund")
        assert [n.text for n in lessons] == ["refund rows double count revenue"]

        prompt = knowledge.format_for_prompt(knowledge.get_knowledge(db, team_id=t, dataset_id=ds), lessons)
        assert "amount is in cents" in prompt and "refund rows double count revenue" in prompt
        assert "30 day window" not in prompt
        assert knowledge.format_for_prompt([], []) == ""
    finally:
        db.close()


def test_never_crosses_teams(setup):
    db = SessionLocal()
    try:
        knowledge.add_note(db, team_id=setup["team"], dataset_id=setup["ds"], kind="lesson", text="refund revenue")
        knowledge.add_note(db, team_id=setup["team"], dataset_id=setup["ds"], kind="knowledge", text="k")
        other = setup["team"] + 10_000
        assert knowledge.retrieve_lessons(db, team_id=other, dataset_id=setup["ds"], question_text="refund revenue") == []
        assert knowledge.get_knowledge(db, team_id=other, dataset_id=setup["ds"]) == []
    finally:
        db.close()


def test_embedding_failure_still_saves_note_and_never_raises(setup):
    db = SessionLocal()
    try:
        with patch.object(knowledge, "embed", side_effect=RuntimeError("quota")):
            note = knowledge.add_note(db, team_id=setup["team"], dataset_id=setup["ds"],
                                      kind="knowledge", text="still saved")
            assert note.embedding is None
            assert knowledge.retrieve_lessons(db, team_id=setup["team"], dataset_id=setup["ds"],
                                              question_text="x") == []
        assert [n.text for n in knowledge.get_knowledge(db, team_id=setup["team"], dataset_id=setup["ds"])] == ["still saved"]
    finally:
        db.close()


def test_critic_rejection_becomes_deduped_lesson(setup):
    db = SessionLocal()
    try:
        q = Question(team_id=setup["team"], dataset_id=setup["ds"], text="revenue by region")
        db.add(q)
        db.commit()
        for _ in range(2):  # same rejection twice -> one lesson
            knowledge.record_critic_lesson(db, question=q, summary="Refunds were counted twice.",
                                           issues=["revenue includes refunds"])
        rows = db.query(KnowledgeNote).filter_by(team_id=setup["team"], kind="lesson", source="critic").all()
        assert len(rows) == 1
        assert "revenue by region" in rows[0].text and "Refunds were counted twice." in rows[0].text
        assert rows[0].question_id == q.id
        # a question with no team (legacy) is skipped, not an error
        knowledge.record_critic_lesson(db, question=Question(text="x", dataset_id=setup["ds"]), summary="s", issues=[])
        knowledge.record_critic_lesson(db, question=None, summary="s", issues=[])
    finally:
        db.close()
