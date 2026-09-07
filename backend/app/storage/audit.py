"""Helpers for writing audit_trail rows that link dashboard elements back to
the exact execution log (code + data) and critic review that produced them."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import AuditTrail


def record_audit_entry(
    db: Session,
    question_id: int,
    element_label: str,
    element_type: str,
    reasoning: str,
    execution_log_id: int | None = None,
    critic_review_id: int | None = None,
    team_id: int | None = None,
    element_id: str | None = None,
) -> AuditTrail:
    entry = AuditTrail(
        team_id=team_id,
        question_id=question_id,
        element_label=element_label,
        element_type=element_type,
        element_id=element_id,
        execution_log_id=execution_log_id,
        critic_review_id=critic_review_id,
        reasoning=reasoning,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
