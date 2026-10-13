"""What to tell a user when a model-driven investigation fell back to the
deterministic investigator.

Everything returned here is built from structured, already-sanitised records:
the fallback code stored on the investigation and the provider failure events
in the run trace (failure kind, HTTP status, attempt count, model id). No
exception message or provider response text is ever read, so none can be shown.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Investigation, InvestigationAnalysis, RunEvent

# Fallback codes written by app.agents.hypothesis.
AT_FIRST_STEP = "model_unavailable_at_first_step"
BEFORE_HYPOTHESES = "model_unavailable_before_hypotheses"
UNKNOWN = "unknown"

_STAGE = {
    AT_FIRST_STEP: "before it could read the question",
    BEFORE_HYPOTHESES: "after reading the question, before any hypothesis was proposed",
}
# provider failure kind (see llm_client.classify_failure) -> plain wording
_WHY = {
    "overloaded": "the AI model's service was temporarily unavailable",
    "server_error": "the AI model's service returned an error",
    "rate_limited": "the AI model's usage limit was reached",
    "authentication": "the AI model did not accept the configured credentials",
    "model_not_found": "the configured AI model was not found",
    "network": "the AI model could not be reached",
    "malformed_output": "the AI model did not return a usable answer",
}
_WHY_GENERIC = "the AI model was unavailable"
_TRACE_MARKER = "fell back from the hypothesis engine"


def _provider_failure(db: Session, question_id: int | None) -> dict | None:
    """The last provider failure of the run, as recorded in its trace."""
    if question_id is None:
        return None
    events = (db.query(RunEvent).filter(RunEvent.question_id == question_id, RunEvent.kind == "provider")
              .order_by(RunEvent.id).all())
    if not events:
        return None
    last = events[-1].detail_json or {}
    kind = last.get("kind") if last.get("kind") in _WHY else "other"
    status = last.get("status") if isinstance(last.get("status"), int) else None
    return {"kind": kind, "http_status": status, "provider_status": last.get("provider_status"),
            "model": last.get("model"), "failed_attempts": len(events)}


def _recorded_in_trace(db: Session, question_id: int | None) -> bool:
    """Investigations stored before fallback codes existed carry the fallback only as a trace note."""
    if question_id is None:
        return False
    for event in db.query(RunEvent).filter(RunEvent.question_id == question_id, RunEvent.kind == "route").all():
        if any(_TRACE_MARKER in str(r) for r in (event.detail_json or {}).get("reasons") or []):
            return True
    return False


def describe(db: Session, inv: Investigation | None) -> dict | None:
    """None when the investigation did not fall back; otherwise a structured,
    display-safe account of the fallback."""
    if inv is None:
        return None
    stored = (inv.params_json or {}).get("fallback")
    if not stored and not _recorded_in_trace(db, inv.question_id):
        return None
    reason = (stored or {}).get("reason")
    reason = reason if reason in _STAGE else UNKNOWN
    failure = _provider_failure(db, inv.question_id)
    evidence = (inv.params_json or {}).get("evidence") or {}
    computed = db.query(InvestigationAnalysis.id).filter_by(investigation_id=inv.id).first() is not None
    partial = bool(evidence) and computed

    why = _WHY.get((failure or {}).get("kind"), _WHY_GENERIC)
    detail = ""
    if failure and failure["http_status"]:
        tries = failure["failed_attempts"]
        detail = f" (HTTP {failure['http_status']}, {tries} failed attempt{'s' if tries != 1 else ''})"
    message = f"The AI investigation could not complete: {why}{detail}"
    message += f", {_STAGE[reason]}." if reason in _STAGE else "."
    message += (" The result shown is from the deterministic investigator, which uses fixed rules and no AI model; "
                "it is not AI-generated analysis.")
    message += (" Some results were computed before the AI model was lost. They are listed separately as partial evidence "
                "and are not part of this result." if partial else
                " No AI hypotheses or statistical evidence were computed.")
    return {"used": True, "from_engine": "hypothesis", "to_engine": "deterministic investigator", "reason": reason,
            "evidence": "partial" if partial else "none", "provider_failure": failure, "message": message}
