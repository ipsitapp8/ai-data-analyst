"""Dataset knowledge + lessons: short team-owned notes the Planner is given.

- kind="knowledge": what the data means (column definitions, units, quirks).
  Always included in planning, newest first, capped -- a definition is only
  useful if it is *always* there, so it is not left to similarity search.
- kind="lesson": a mistake to avoid. Written by a user, or captured
  automatically when the Critic rejects an analysis (source="critic").
  Retrieved by similarity to the new question, like past analyses.

Notes are context only: the Planner is told to still compute from the data,
and the Critic never sees them, so a note cannot vouch for a result. Every
function degrades to "no notes" on failure -- this must never break a run.
"""
from __future__ import annotations

import logging

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, bindparam

from app import config
from app.memory import embed
from app.models import KnowledgeNote

logger = logging.getLogger(__name__)

KINDS = ("knowledge", "lesson")


def add_note(db, *, team_id: int, dataset_id: int, kind: str, text: str,
             source: str = "user", question_id: int | None = None,
             created_by: int | None = None) -> KnowledgeNote:
    """Store a note (embedding it best-effort). Caller validates kind/length."""
    vector = None
    if config.RAG_ENABLED:
        try:
            vector = embed([text], "RETRIEVAL_DOCUMENT")[0]
        except Exception:  # noqa: BLE001
            logger.warning("Could not embed knowledge note; saving without a vector", exc_info=True)
    note = KnowledgeNote(
        team_id=team_id, dataset_id=dataset_id, kind=kind, source=source, text=text,
        question_id=question_id, created_by=created_by, embedding=vector,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


def record_critic_lesson(db, *, question, summary: str, issues: list[str]) -> None:
    """Turn a Critic rejection into a lesson. Never raises."""
    try:
        if question is None or question.team_id is None:
            return
        detail = " ".join(f"- {i}" for i in issues or [])
        text = (
            f'An earlier analysis asking "{question.text}" was rejected by the Critic. '
            f"Reasoning: {(summary or '').strip()} {detail}"
        ).strip()[: config.NOTE_MAX_CHARS]
        exists = db.query(KnowledgeNote).filter_by(
            team_id=question.team_id, dataset_id=question.dataset_id,
            kind="lesson", source="critic", text=text,
        ).first()
        if exists:
            return
        add_note(db, team_id=question.team_id, dataset_id=question.dataset_id, kind="lesson",
                 text=text, source="critic", question_id=question.id)
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not record Critic lesson", exc_info=True)


def get_knowledge(db, *, team_id: int | None, dataset_id: int) -> list[KnowledgeNote]:
    """Newest-first knowledge notes for this team + dataset, capped."""
    try:
        return (
            db.query(KnowledgeNote)
            .filter_by(team_id=team_id, dataset_id=dataset_id, kind="knowledge")
            .order_by(KnowledgeNote.created_at.desc(), KnowledgeNote.id.desc())
            .limit(config.KNOWLEDGE_MAX_IN_PROMPT)
            .all()
        )
    except Exception:  # noqa: BLE001
        logger.warning("Knowledge lookup failed; continuing without it", exc_info=True)
        return []


def retrieve_lessons(db, *, team_id: int | None, dataset_id: int, question_text: str) -> list[KnowledgeNote]:
    """Top-k lessons for this team + dataset most similar to the question."""
    if not config.RAG_ENABLED:
        return []
    try:
        q = db.query(KnowledgeNote).filter(
            KnowledgeNote.team_id == team_id, KnowledgeNote.dataset_id == dataset_id,
            KnowledgeNote.kind == "lesson", KnowledgeNote.embedding.isnot(None),
        )
        if q.first() is None:
            return []
        query = embed([question_text], "RETRIEVAL_QUERY")[0]
        if db.get_bind().dialect.name == "postgresql":
            dist = KnowledgeNote.embedding.op("<=>", return_type=Float)(
                bindparam("q", query, type_=Vector(config.EMBEDDING_DIM))
            )
            hits = q.add_columns(dist).order_by(dist).limit(config.LESSON_TOP_K).all()
            return [n for n, d in hits if 1 - d >= config.RAG_MIN_SIMILARITY]
        rows = q.all()
        query_vec = np.array(query)
        matrix = np.array([r.embedding for r in rows])
        sims = matrix @ query_vec / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec) + 1e-12)
        ranked = sorted(zip(sims, rows), key=lambda t: -t[0])
        return [r for s, r in ranked[: config.LESSON_TOP_K] if s >= config.RAG_MIN_SIMILARITY]
    except Exception:  # noqa: BLE001
        logger.warning("Lesson retrieval failed; continuing without it", exc_info=True)
        return []


def format_for_prompt(knowledge: list[KnowledgeNote], lessons: list[KnowledgeNote]) -> str:
    out = ""
    if knowledge:
        lines = "\n".join(f"- {n.text}" for n in reversed(knowledge))  # oldest first reads naturally
        out += (
            "\nWhat this team knows about this dataset (written by people who know the data -- "
            "treat these as definitions and caveats, but still compute everything from the data):\n"
            + lines + "\n"
        )
    if lessons:
        lines = "\n".join(f"- {n.text}" for n in lessons)
        out += (
            "\nLessons from earlier mistakes on this dataset (avoid repeating them; they may not "
            "all apply to this question):\n" + lines + "\n"
        )
    return out
