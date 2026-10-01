"""Dashboard-level Critic verdict states.

VERIFIED               Critic approved on the first pass.
VERIFIED_WITH_CAVEATS  Critic rejected, the Planner re-planned, and the second
                       pass was approved -- the result stands, but something
                       was wrong once and the reader should know.
UNVERIFIED             Critic still rejected after the re-plan budget ran out.

The state is decided once, when the dashboard is compiled, and stored on the
dashboard row. `resolve_verdict_state` only exists for rows written before
that column did.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import AuditTrail, CriticReview, Dashboard

VERIFIED = "VERIFIED"
VERIFIED_WITH_CAVEATS = "VERIFIED_WITH_CAVEATS"
UNVERIFIED = "UNVERIFIED"


def compute_verdict_state(critic_verdict: str | None, revision_count: int) -> str:
    if critic_verdict != "verified":
        return UNVERIFIED
    return VERIFIED_WITH_CAVEATS if revision_count > 0 else VERIFIED


def rejected_reviews(db: Session, question_id: int) -> list[CriticReview]:
    return (
        db.query(CriticReview)
        .filter(CriticReview.question_id == question_id, CriticReview.verdict == "rejected")
        .order_by(CriticReview.created_at)
        .all()
    )


def resolve_verdict_state(dashboard: Dashboard, rejections: list[CriticReview]) -> str:
    if dashboard.verdict_state:
        return dashboard.verdict_state
    if not dashboard.verified:
        return UNVERIFIED
    return VERIFIED_WITH_CAVEATS if rejections else VERIFIED


def flagged_element_ids(db: Session, question_id: int) -> set[str]:
    """Elements whose audit_trail row points at a REJECT critic_review."""
    rows = (
        db.query(AuditTrail.element_id)
        .join(CriticReview, AuditTrail.critic_review_id == CriticReview.id)
        .filter(
            AuditTrail.question_id == question_id,
            AuditTrail.element_id.isnot(None),
            CriticReview.verdict == "rejected",
        )
        .all()
    )
    return {r[0] for r in rows}


def rejection_payload(rejections: list[CriticReview]) -> list[dict]:
    return [
        {"summary": r.summary or "", "issues": list(r.issues_json or []), "created_at": r.created_at}
        for r in rejections
    ]


def flagged_item_count(rejections: list[CriticReview]) -> int:
    """N in "Verified with caveats -- N item(s) flagged". Falls back to the
    number of rejections when the Critic gave a verdict but no itemised issues."""
    issues = sum(len(r.issues_json or []) for r in rejections)
    return issues or len(rejections)


def narrative_audit(db: Session, question_id: int) -> tuple[bool, str | None]:
    """(is the narrative flagged, its element_id if it has one). Narratives from
    before element ids existed have none, so they can be flagged but not inspected."""
    row = (
        db.query(AuditTrail.element_id, CriticReview.verdict)
        .outerjoin(CriticReview, AuditTrail.critic_review_id == CriticReview.id)
        .filter(AuditTrail.question_id == question_id, AuditTrail.element_type == "narrative")
        .first()
    )
    if row is None:
        return False, None
    return row[1] == "rejected", row[0]
