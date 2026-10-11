"""Root-cause investigations (see app/investigator.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import jobs
from app import metric_engine as me
from app.database import get_db
from app.models import Investigation, Team, User
from app.routers._common import team_dataset
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/investigations", tags=["investigations"])


class InvestigationCreate(BaseModel):
    dataset_id: int
    formula: str = Field(min_length=1, max_length=me.MAX_FORMULA_CHARS)
    label: str = Field(default="", max_length=120)
    time_column: str = Field(min_length=1, max_length=200)
    filters: list[dict] = Field(default_factory=list, max_length=20)
    period_a: list[str] | None = Field(default=None, min_length=2, max_length=2)
    period_b: list[str] | None = Field(default=None, min_length=2, max_length=2)
    dimensions: list[str] | None = Field(default=None, max_length=12)


def _out(i: Investigation, with_report: bool = True) -> dict:
    out = {"id": i.id, "dataset_id": i.dataset_id, "dataset_version_id": i.dataset_version_id,
           "question_id": i.question_id, "status": i.status, "params": i.params_json or {},
           "summary": (i.report_json or {}).get("summary", ""), "created_at": i.created_at}
    if with_report:
        out["report"] = i.report_json or {}
    return out


@router.post("", status_code=201)
def create_investigation(
    payload: InvestigationCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Queue an investigation with an explicit metric and periods. It runs as a
    normal durable job; the result is both a report and a dashboard."""
    dataset = team_dataset(db, team, payload.dataset_id)
    profile = dataset.profile_json or {}
    columns = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    try:
        formula = me.parse_formula(payload.formula)
        problems = me.validate_against_profile(formula, profile)
        filters = me.normalize_filters(payload.filters, list(columns))
    except me.FormulaError as e:
        raise HTTPException(400, str(e)) from e
    if problems:
        raise HTTPException(400, "; ".join(problems))
    if columns.get(payload.time_column) != "datetime":
        raise HTTPException(400, f"'{payload.time_column}' is not a date column of this dataset")
    if (payload.period_a is None) != (payload.period_b is None):
        raise HTTPException(400, "Give both periods or neither (neither = split the time range at its midpoint)")
    for d in payload.dimensions or []:
        if d not in columns:
            raise HTTPException(400, f"'{d}' is not a column of this dataset")
    label = payload.label.strip() or formula.text
    spec = {"formula": formula.text, "label": label, "filters": filters, "time_column": payload.time_column,
            "period_a": payload.period_a, "period_b": payload.period_b, "dimensions": payload.dimensions}
    try:
        question, job, created = jobs.submit_question(
            db, team_id=team.id, dataset=dataset, text=f"Why did {label} change?", created_by=user.id,
            idempotency_key=idempotency_key, route="root_cause", stage_detail="Investigation queued")
    except jobs.AdmissionRejected as e:
        raise HTTPException(429, str(e)) from e
    if created:
        inv = Investigation(team_id=team.id, dataset_id=dataset.id, dataset_version_id=question.dataset_version_id,
                            question_id=question.id, params_json=spec, report_json={}, status="queued",
                            created_by=user.id)
        db.add(inv)
        db.commit()
        db.refresh(inv)
    else:
        inv = db.query(Investigation).filter_by(question_id=question.id).first()
    return {"investigation_id": inv.id if inv else None, "question_id": question.id, "job_id": job.id,
            "created": created}


@router.get("")
def list_investigations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    q = db.query(Investigation).filter_by(team_id=team.id)
    total = q.count()
    rows = q.order_by(Investigation.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset,
            "investigations": [_out(i, with_report=False) for i in rows]}


@router.get("/{investigation_id}")
def get_investigation(
    investigation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    inv = db.get(Investigation, investigation_id)
    if inv is None or inv.team_id != team.id:
        raise HTTPException(404, "Investigation not found")
    return _out(inv)
