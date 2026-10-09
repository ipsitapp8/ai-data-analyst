"""Chart Studio overrides: which chart type / style a dashboard chart is shown
as. Only the presentation is stored -- the verified numbers in charts_json
are never touched, so the audit trail stays exactly as the agent produced it."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dashboard, Team, User
from app.schemas import ChartViewIn
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api", tags=["chart-views"])


def _team_dashboard_and_overrides(db: Session, team: Team, dashboard_id: int, chart_key: str):
    dashboard = db.get(Dashboard, dashboard_id)
    if not dashboard or dashboard.team_id != team.id:
        raise HTTPException(404, "Dashboard not found")

    charts = dashboard.charts_json or []  # native JSON column (see models.JSONVariant)
    if not isinstance(charts, list):
        charts = []
    valid_keys = {c.get("element_id") for c in charts if isinstance(c, dict) and c.get("element_id")}
    valid_keys |= {f"idx{i}" for i in range(len(charts))}
    if chart_key not in valid_keys:
        raise HTTPException(404, "Chart not found")

    try:
        overrides = json.loads(dashboard.view_overrides_json or "{}")
    except json.JSONDecodeError:
        overrides = {}
    if not isinstance(overrides, dict):
        overrides = {}
    return dashboard, overrides


@router.put("/dashboards/{dashboard_id}/charts/{chart_key}/view")
def save_chart_view(
    dashboard_id: int,
    chart_key: str,
    body: ChartViewIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dashboard, overrides = _team_dashboard_and_overrides(db, team, dashboard_id, chart_key)
    overrides[chart_key] = body.model_dump()
    dashboard.view_overrides_json = json.dumps(overrides)
    db.commit()
    return {"chart_key": chart_key, "view": overrides[chart_key]}


@router.delete("/dashboards/{dashboard_id}/charts/{chart_key}/view")
def reset_chart_view(
    dashboard_id: int,
    chart_key: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dashboard, overrides = _team_dashboard_and_overrides(db, team, dashboard_id, chart_key)
    overrides.pop(chart_key, None)
    dashboard.view_overrides_json = json.dumps(overrides)
    db.commit()
    return {"chart_key": chart_key, "view": None}
