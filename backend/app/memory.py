"""RAG over past analyses: embed each finished dashboard, and let the Planner
retrieve the most similar earlier ones (same team + dataset) as context.

Retrieval is context only -- the Planner is told to recompute from the data, and
the Critic never sees it, so an old number cannot vouch for a new one. Every
function here degrades to "no memory" on failure: RAG must never break a run.
"""
from __future__ import annotations

import logging

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, bindparam

from app import config
from app.models import AnalysisMemory, Dashboard, Question

logger = logging.getLogger(__name__)


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import types

    from app.agents.llm_client import get_client

    resp = get_client().models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=config.EMBEDDING_DIM
        ),
    )
    # Truncated Gemini embeddings aren't unit-length; normalize so cosine
    # distance behaves the same everywhere.
    out = []
    for e in resp.embeddings:
        v = np.array(e.values, dtype=float)
        out.append((v / (np.linalg.norm(v) + 1e-12)).tolist())
    return out


class QueryEmbedding:
    """One question's retrieval embedding, shared by the lookups made for the
    same planning step so the provider is asked once, not once per lookup.

    Computed lazily (no request at all if no lookup needs it) and reused only
    when everything that determines the vector is identical: the embedding
    model, its dimensionality, the task type and the exact text. A failed
    request is not remembered, so it cannot stand in for a vector later."""

    TASK_TYPE = "RETRIEVAL_QUERY"

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._vector: list[float] | None = None

    def get(self, text: str, embed_fn=None) -> list[float]:
        key = (config.EMBEDDING_MODEL, config.EMBEDDING_DIM, self.TASK_TYPE, text)
        if self._key != key or self._vector is None:
            self._vector = (embed_fn or embed)([text], self.TASK_TYPE)[0]     # raises on failure: nothing is stored
            self._key = key
        return self._vector


def _summarize(dashboard: Dashboard) -> str:
    kpis = "; ".join(f"{k.get('label')}: {k.get('value')}" for k in (dashboard.kpis_json or []))
    parts = []
    if kpis:
        parts.append(f"Key figures: {kpis}")
    if dashboard.narrative:
        parts.append(dashboard.narrative.strip())
    return "\n".join(parts)[:2000]


def index_dashboard(db, dashboard: Dashboard) -> None:
    """Embed and store a finished analysis. Never raises."""
    if not config.RAG_ENABLED:
        return
    try:
        question = db.get(Question, dashboard.question_id)
        if question is None or db.query(AnalysisMemory).filter_by(question_id=question.id).first():
            return
        summary = _summarize(dashboard)
        vector = embed([f"{question.text}\n{summary}"], "RETRIEVAL_DOCUMENT")[0]
        db.add(AnalysisMemory(
            team_id=question.team_id,
            dataset_id=question.dataset_id,
            question_id=question.id,
            question_text=question.text,
            summary=summary,
            verdict_state=dashboard.verdict_state or ("VERIFIED" if dashboard.verified else "UNVERIFIED"),
            embedding=vector,
        ))
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not index analysis memory for dashboard %s", dashboard.id, exc_info=True)


def retrieve(db, *, team_id: int | None, dataset_id: int, question_text: str,
             exclude_question_id: int | None = None,
             query_embedding: QueryEmbedding | None = None) -> list[AnalysisMemory]:
    """Top-k similar past analyses for this team + dataset, best first. Never raises.
    Pass a shared `query_embedding` to reuse the question's vector across lookups."""
    if not config.RAG_ENABLED:
        return []
    try:
        q = db.query(AnalysisMemory).filter(
            AnalysisMemory.team_id == team_id, AnalysisMemory.dataset_id == dataset_id
        )
        if exclude_question_id is not None:
            q = q.filter(AnalysisMemory.question_id != exclude_question_id)
        if q.first() is None:
            return []
        query = (query_embedding or QueryEmbedding()).get(question_text, embed)
        if db.get_bind().dialect.name == "postgresql":
            # pgvector: cosine distance in SQL, served by the HNSW index.
            dist = AnalysisMemory.embedding.op("<=>", return_type=Float)(
                bindparam("q", query, type_=Vector(config.EMBEDDING_DIM))
            )
            hits = q.add_columns(dist).order_by(dist).limit(config.RAG_TOP_K).all()
            return [m for m, d in hits if 1 - d >= config.RAG_MIN_SIMILARITY]
        rows = q.all()
        query_vec = np.array(query)
        matrix = np.array([r.embedding for r in rows])
        sims = matrix @ query_vec / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec) + 1e-12)
        ranked = sorted(zip(sims, rows), key=lambda t: -t[0])
        return [r for s, r in ranked[: config.RAG_TOP_K] if s >= config.RAG_MIN_SIMILARITY]
    except Exception:  # noqa: BLE001
        logger.warning("Memory retrieval failed; continuing without it", exc_info=True)
        return []


def format_for_prompt(memories: list[AnalysisMemory]) -> str:
    if not memories:
        return ""
    blocks = []
    for m in memories:
        status = (m.verdict_state or "UNVERIFIED").replace("_", " ").lower()
        blocks.append(
            f"- Question: {m.question_text}\n  Outcome ({status}, {m.created_at:%Y-%m-%d}):\n  "
            + m.summary.replace("\n", "\n  ")
        )
    return (
        "\nEarlier analyses of this same dataset by this team (context only -- the data may "
        "have changed and some were not fully verified, so do NOT reuse their numbers; plan "
        "steps that recompute everything, and use these only to avoid redundant or "
        "contradictory approaches):\n" + "\n".join(blocks) + "\n"
    )
