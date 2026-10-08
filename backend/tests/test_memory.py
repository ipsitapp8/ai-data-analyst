"""RAG over past analyses. Embeddings are faked (deterministic keyword vectors)
so no Gemini call is made."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app import memory
from app.config import EMBEDDING_DIM
from app.database import SessionLocal
from app.models import AnalysisMemory, Dashboard, Dataset, Question, Team

KEYWORDS = ["revenue", "churn", "region", "price"]


def fake_embed(texts, task_type):
    out = []
    for t in texts:
        v = np.zeros(EMBEDDING_DIM)
        v[: len(KEYWORDS)] = [float(k in t.lower()) + 0.01 for k in KEYWORDS]
        out.append((v / np.linalg.norm(v)).tolist())
    return out


@pytest.fixture(autouse=True)
def _fake_embeddings():
    with patch.object(memory, "embed", fake_embed):
        yield


def _make_analysis(db, team_id, dataset_id, text, narrative="n"):
    q = Question(team_id=team_id, dataset_id=dataset_id, text=text, status="verified")
    db.add(q)
    db.commit()
    d = Dashboard(team_id=team_id, question_id=q.id, kpis_json=[{"label": "Total", "value": "5"}],
                  narrative=narrative, verified=True, verdict_state="VERIFIED")
    db.add(d)
    db.commit()
    memory.index_dashboard(db, d)
    return q


@pytest.fixture
def ids():
    db = SessionLocal()
    try:
        from app.models import Community
        c = Community(name="C")
        db.add(c)
        db.commit()
        teams = [Team(community_id=c.id, name=n) for n in ("A", "B")]
        db.add_all(teams)
        db.commit()
        ds = Dataset(team_id=teams[0].id, filename="x.csv", filepath="x.csv")
        db.add(ds)
        db.commit()
        return {"a": teams[0].id, "b": teams[1].id, "ds": ds.id}
    finally:
        db.close()


def test_retrieves_similar_and_ignores_unrelated(ids):
    db = SessionLocal()
    try:
        _make_analysis(db, ids["a"], ids["ds"], "revenue by region")
        _make_analysis(db, ids["a"], ids["ds"], "churn rate")
        found = memory.retrieve(db, team_id=ids["a"], dataset_id=ids["ds"],
                                question_text="total revenue per region")
        assert [m.question_text for m in found] == ["revenue by region"]
    finally:
        db.close()


def test_never_crosses_teams(ids):
    db = SessionLocal()
    try:
        _make_analysis(db, ids["a"], ids["ds"], "revenue by region")
        assert memory.retrieve(db, team_id=ids["b"], dataset_id=ids["ds"],
                               question_text="revenue by region") == []
    finally:
        db.close()


def test_excludes_current_question_and_indexes_once(ids):
    db = SessionLocal()
    try:
        q = _make_analysis(db, ids["a"], ids["ds"], "revenue by region")
        d = db.query(Dashboard).filter_by(question_id=q.id).one()
        memory.index_dashboard(db, d)
        assert db.query(AnalysisMemory).filter_by(question_id=q.id).count() == 1
        assert memory.retrieve(db, team_id=ids["a"], dataset_id=ids["ds"],
                               question_text="revenue by region", exclude_question_id=q.id) == []
    finally:
        db.close()


def test_embedding_failure_never_raises(ids):
    db = SessionLocal()
    try:
        with patch.object(memory, "embed", side_effect=RuntimeError("quota")):
            assert memory.retrieve(db, team_id=ids["a"], dataset_id=ids["ds"], question_text="x") == []
            q = Question(team_id=ids["a"], dataset_id=ids["ds"], text="q")
            db.add(q)
            db.commit()
            d = Dashboard(team_id=ids["a"], question_id=q.id)
            db.add(d)
            db.commit()
            memory.index_dashboard(db, d)  # must not raise
    finally:
        db.close()


def test_prompt_block_flags_status_and_forbids_reuse(ids):
    db = SessionLocal()
    try:
        _make_analysis(db, ids["a"], ids["ds"], "price trend")
        found = memory.retrieve(db, team_id=ids["a"], dataset_id=ids["ds"], question_text="price")
        block = memory.format_for_prompt(found)
        assert "price trend" in block and "verified" in block and "do NOT reuse" in block
        assert memory.format_for_prompt([]) == ""
    finally:
        db.close()
