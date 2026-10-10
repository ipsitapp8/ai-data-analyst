"""Ask Your Dashboard: follow-up questions answered from one finished analysis.

The answer is grounded ONLY in what that analysis already produced -- its KPIs,
narrative, step results and data slices, Critic verdict, plus the dataset profile
and team knowledge notes. The chat does not run new code. If a question needs a
fresh computation it says so and proposes a question to run through the normal,
verified pipeline instead.

Every answer is then checked by an independent model (the Critic's Llama, or
Gemini fallback, via agents.critic._critic_call_tool): if it states a number or
claim the context does not support, the response is marked unverified and lists
the unsupported claims, so the UI never presents an ungrounded answer as solid.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.agents import llm_client
from app.agents.critic import _critic_call_tool
from app.models import Dashboard, Dataset, ExecutionLog, KnowledgeNote, Question
from app.verdict import rejected_reviews, resolve_verdict_state

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 14000
MAX_HISTORY_TURNS = 8
MAX_MESSAGE_CHARS = 600

ANSWER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "Direct answer in plain English, citing the figures it relies on."},
        "sources": {"type": "array", "items": {"type": "string"},
                    "description": "Which parts of the context the answer uses, e.g. 'KPI: Total revenue', 'Step 2'."},
        "needs_new_analysis": {"type": "boolean",
                               "description": "True if the context cannot answer this and new computation is required."},
        "suggested_question": {"type": "string",
                               "description": "If needs_new_analysis, a self-contained question to run as a new analysis; else empty."},
    },
    "required": ["answer", "sources", "needs_new_analysis", "suggested_question"],
}

CHECK_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"},
                               "description": "Numbers or claims in the answer that the context does not support; empty if supported."},
    },
    "required": ["supported", "unsupported_claims"],
}

ANSWER_SYSTEM = (
    "You answer follow-up questions about ONE completed data analysis. Use ONLY the context given: "
    "its KPIs, narrative, step results, data slices, verification verdict, dataset profile and team "
    "notes. Never invent numbers and never compute new statistics the context does not contain. "
    "If the context cannot answer, set needs_new_analysis and propose a precise question to run. "
    "If the analysis was flagged or unverified, say so when it affects your answer. Be concise."
)

CHECK_SYSTEM = (
    "You are a strict fact-checker. Given a context and an answer derived from it, decide whether "
    "EVERY number and factual claim in the answer appears in, or follows directly from, the context. "
    "Rounding and reformatting are fine; new figures, causes or trends the context does not state are not."
)


def build_context(db: Session, question: Question, dash: Dashboard) -> str:
    verdict = resolve_verdict_state(dash, rejected_reviews(db, question.id))
    parts = [f"Original question: {question.text}", f"Verification verdict: {verdict}"]
    if dash.verification_summary:
        parts.append(f"Verification summary: {dash.verification_summary}")
    for r in rejected_reviews(db, question.id):
        parts.append(f"Critic flagged: {r.summary} {list(r.issues_json or [])}")
    kpis = "; ".join(f"{k.get('label')}: {k.get('value')}" for k in dash.kpis_json or [])
    if kpis:
        parts.append(f"KPIs: {kpis}")
    charts = [c.get("title") for c in dash.charts_json or [] if c.get("title")]
    if charts:
        parts.append("Charts: " + "; ".join(charts))
    if dash.narrative:
        parts.append(f"Narrative: {dash.narrative}")

    logs = (db.query(ExecutionLog).filter_by(question_id=question.id, success=True)
            .order_by(ExecutionLog.step_index, ExecutionLog.attempt_number.desc()).all())
    seen: set[int] = set()
    for log in logs:
        if log.step_index in seen or log.step_index < 0:
            continue
        seen.add(log.step_index)
        block = f"Step {log.step_index}: {log.step_description}"
        if log.formula_explanation:
            block += f"\n  Formula: {log.formula_explanation}"
        block += f"\n  Result: {str(log.result_json)[:1200]}"
        sl = log.data_slice_json or {}
        if sl.get("columns") and sl.get("rows"):
            rows = sl["rows"][:12]
            block += f"\n  Data slice columns {sl['columns']}, first rows {rows}"
        parts.append(block)

    ds = db.get(Dataset, question.dataset_id)
    if ds is not None and ds.profile_json:
        cols = [f"{c['name']} ({c.get('kind')})" for c in ds.profile_json.get("columns", [])]
        parts.append(f"Dataset {ds.filename}: {ds.profile_json.get('row_count')} rows; columns: {', '.join(cols)}")
    notes = (db.query(KnowledgeNote).filter_by(team_id=question.team_id, dataset_id=question.dataset_id,
                                               kind="knowledge").limit(10).all())
    if notes:
        parts.append("Team notes about the data:\n" + "\n".join(f"- {n.text}" for n in notes))
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def _history_text(history: list[dict]) -> str:
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        who = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {str(turn.get('content', ''))[:MAX_MESSAGE_CHARS * 2]}")
    return "\n".join(lines)


def answer_question(db: Session, question: Question, dash: Dashboard, message: str,
                    history: list[dict]) -> dict:
    context = build_context(db, question, dash)
    user = f"CONTEXT:\n{context}\n"
    if history:
        user += f"\nConversation so far:\n{_history_text(history)}\n"
    user += f"\nNew question: {message}"
    out = llm_client.call_tool(
        system=ANSWER_SYSTEM, user_content=user, tool_name="submit_answer",
        tool_schema=ANSWER_TOOL_SCHEMA, tool_description="Submit the grounded answer.",
        max_tokens=1500,
    )
    answer = str(out.get("answer", "")).strip()
    result = {
        "answer": answer,
        "sources": [str(s) for s in out.get("sources") or []][:8],
        "needs_new_analysis": bool(out.get("needs_new_analysis")),
        "suggested_question": str(out.get("suggested_question") or "").strip()[:300],
        "verified": None,
        "unsupported_claims": [],
        "checked_by": None,
    }
    # A "can't answer from this" reply states nothing that could be unsupported.
    if answer and not result["needs_new_analysis"]:
        try:
            check, model = _critic_call_tool(
                system=CHECK_SYSTEM,
                user_content=f"CONTEXT:\n{context}\n\nANSWER TO CHECK:\n{answer}",
                tool_name="submit_check", tool_schema=CHECK_TOOL_SCHEMA,
                tool_description="Submit whether the answer is fully supported.",
                max_tokens=800,
            )
            claims = [str(c) for c in check.get("unsupported_claims") or []]
            result["verified"] = bool(check.get("supported")) and not claims
            result["unsupported_claims"] = claims[:6]
            result["checked_by"] = model
        except Exception:  # noqa: BLE001 - an unchecked answer is labelled as such, not dropped
            logger.warning("Chat answer check failed", exc_info=True)
    return result
