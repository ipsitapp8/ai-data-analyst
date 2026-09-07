"""Click-to-inspect: for one dashboard element, the exact code, plain-English
formula, and data slice that produced it -- the audit trail surfaced inline."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AuditTrail, Dashboard, Team, User
from app.schemas import InspectResponse
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api", tags=["inspect"])


@router.get("/dashboards/{dashboard_id}/elements/{element_id}/inspect", response_model=InspectResponse)
def inspect_element(
    dashboard_id: int,
    element_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dashboard = db.get(Dashboard, dashboard_id)
    if not dashboard or dashboard.team_id != team.id:
        raise HTTPException(404, "Dashboard not found")

    entry = (
        db.query(AuditTrail)
        .filter(AuditTrail.question_id == dashboard.question_id, AuditTrail.element_id == element_id)
        .first()
    )
    if not entry:
        raise HTTPException(404, "Element not found")

    log = entry.execution_log
    if log is None:
        return InspectResponse(
            code=None,
            formula_explanation="Synthesized from all verified step results — no single formula.",
            data_slice={"columns": [], "rows": []},
        )

    try:
        data_slice = json.loads(log.data_slice_json) if log.data_slice_json else {}
    except json.JSONDecodeError:
        data_slice = {}
    if not data_slice.get("columns"):
        data_slice = {"columns": [], "rows": []}

    return InspectResponse(
        code=log.code,
        formula_explanation=log.formula_explanation or "No formula was recorded for this step.",
        data_slice=data_slice,
    )
