"""Semantic layer: metric definitions, terms, relationships, search."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import metric_engine as me
from app import semantic
from app.database import get_db
from app.models import SemanticMetric, SemanticRelationship, SemanticTerm, Team, User
from app.routers._common import require_admin, team_dataset
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/semantic", tags=["semantic"])


class MetricCreate(BaseModel):
    name: str = Field(min_length=1, max_length=semantic.MAX_NAME)
    formula: str = Field(min_length=1, max_length=me.MAX_FORMULA_CHARS)
    description: str = Field(default="", max_length=1000)
    unit: str = Field(default="", max_length=30)
    dataset_id: int | None = None
    filters: list[dict] = Field(default_factory=list, max_length=20)
    aliases: list[str] = Field(default_factory=list, max_length=20)


class VersionCreate(BaseModel):
    formula: str = Field(min_length=1, max_length=me.MAX_FORMULA_CHARS)
    dataset_id: int | None = None
    filters: list[dict] = Field(default_factory=list, max_length=20)
    note: str = Field(default="", max_length=500)


class TermCreate(BaseModel):
    kind: str
    term: str = Field(min_length=1, max_length=semantic.MAX_NAME)
    metric_id: int | None = None
    dataset_id: int | None = None
    column_name: str | None = Field(default=None, max_length=200)
    unit: str = Field(default="", max_length=30)
    description: str = Field(default="", max_length=500)


class RelationshipCreate(BaseModel):
    src_type: str
    src_ref: str = Field(max_length=200)
    relation: str
    dst_type: str
    dst_ref: str = Field(max_length=200)


class ResolveRequest(BaseModel):
    dataset_id: int
    text: str = Field(min_length=1, max_length=500)


class ValidateRequest(BaseModel):
    formula: str = Field(max_length=me.MAX_FORMULA_CHARS)
    dataset_id: int | None = None
    filters: list[dict] = Field(default_factory=list, max_length=20)


def _metric(db: Session, team: Team, metric_id: int) -> SemanticMetric:
    try:
        return semantic.get_metric(db, team.id, metric_id)
    except semantic.SemanticError as e:
        raise HTTPException(404, str(e)) from e


def _detail(db: Session, metric: SemanticMetric) -> dict:
    return {**semantic.metric_out(db, metric), "versions": [
        {"version": v.version, "formula": v.formula, "filters": list(v.filters_json or []),
         "dataset_id": v.dataset_id, "note": v.note or "", "source": v.source, "created_by": v.created_by,
         "created_at": v.created_at, "approved_by": v.approved_by, "approved_at": v.approved_at,
         "is_current": v.version == metric.current_version}
        for v in semantic.versions_of(db, metric)]}


@router.get("/metrics")
def list_metrics(
    status: str | None = Query(default=None, pattern="^(draft|approved|deprecated)$"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    q = db.query(SemanticMetric).filter_by(team_id=team.id)
    if status:
        q = q.filter_by(status=status)
    return [semantic.metric_out(db, m) for m in q.order_by(SemanticMetric.name).all()]


@router.post("/metrics", status_code=201)
def create_metric(
    payload: MetricCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Create a draft. It has no effect on analyses until an owner or admin approves it."""
    try:
        metric = semantic.create_metric(db, team_id=team.id, user_id=user.id, **payload.model_dump())
    except semantic.SemanticError as e:
        db.rollback()
        raise HTTPException(400, str(e)) from e
    conflicts = []
    for term in semantic.metric_terms(db, metric):
        conflicts += [{**c, "term": term} for c in semantic.conflicts_for(db, team.id, term, metric.id)]
    return {**_detail(db, metric), "conflicts": conflicts}


@router.get("/metrics/{metric_id}")
def get_metric(
    metric_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return _detail(db, _metric(db, team, metric_id))


@router.post("/metrics/{metric_id}/versions", status_code=201)
def add_version(
    metric_id: int,
    payload: VersionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    metric = _metric(db, team, metric_id)
    try:
        semantic.add_version(db, metric, user_id=user.id, **payload.model_dump())
    except semantic.SemanticError as e:
        db.rollback()
        raise HTTPException(400, str(e)) from e
    return _detail(db, metric)


@router.post("/metrics/{metric_id}/versions/{version}/approve")
def approve_version(
    metric_id: int,
    version: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    metric = _metric(db, team, metric_id)
    require_admin(db, team, user.id, "approve a metric definition")
    try:
        semantic.approve(db, metric, version, user.id)
    except semantic.SemanticError as e:
        db.rollback()
        raise HTTPException(400, str(e)) from e
    return _detail(db, metric)


@router.post("/metrics/{metric_id}/deprecate")
def deprecate_metric(
    metric_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    metric = _metric(db, team, metric_id)
    require_admin(db, team, user.id, "deprecate a metric definition")
    return semantic.metric_out(db, semantic.deprecate(db, metric))


@router.post("/validate")
def validate_formula(
    payload: ValidateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Check a formula without saving anything."""
    try:
        parsed, filters = semantic.validate_definition(db, team.id, payload.formula, payload.dataset_id, payload.filters)
    except semantic.SemanticError as e:
        return {"valid": False, "error": str(e)}
    return {"valid": True, "columns": parsed.columns, "additive": parsed.is_additive, "filters": filters}


@router.get("/terms")
def list_terms(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    rows = db.query(SemanticTerm).filter_by(team_id=team.id).order_by(SemanticTerm.term).all()
    return [{"id": t.id, "kind": t.kind, "term": t.term, "metric_id": t.metric_id, "dataset_id": t.dataset_id,
             "column_name": t.column_name, "unit": t.unit or "", "description": t.description or "",
             "created_at": t.created_at} for t in rows]


@router.post("/terms", status_code=201)
def create_term(
    payload: TermCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    try:
        term, conflicts = semantic.add_term(db, team_id=team.id, user_id=user.id, **payload.model_dump())
    except semantic.SemanticError as e:
        db.rollback()
        raise HTTPException(400, str(e)) from e
    return {"id": term.id, "kind": term.kind, "term": term.term, "metric_id": term.metric_id,
            "dataset_id": term.dataset_id, "column_name": term.column_name, "conflicts": conflicts}


@router.delete("/terms/{term_id}", status_code=204)
def delete_term(
    term_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    t = db.get(SemanticTerm, term_id)
    if t is None or t.team_id != team.id:
        raise HTTPException(404, "Term not found")
    require_admin(db, team, user.id, "remove a term")
    db.delete(t)
    db.commit()


@router.post("/relationships", status_code=201)
def create_relationship(
    payload: RelationshipCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    try:
        r = semantic.add_relationship(db, team_id=team.id, user_id=user.id, **payload.model_dump())
    except semantic.SemanticError as e:
        db.rollback()
        raise HTTPException(400, str(e)) from e
    return {"id": r.id, "src_type": r.src_type, "src_ref": r.src_ref, "relation": r.relation,
            "dst_type": r.dst_type, "dst_ref": r.dst_ref}


@router.delete("/relationships/{relationship_id}", status_code=204)
def delete_relationship(
    relationship_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    r = db.get(SemanticRelationship, relationship_id)
    if r is None or r.team_id != team.id:
        raise HTTPException(404, "Relationship not found")
    require_admin(db, team, user.id, "remove a relationship")
    db.delete(r)
    db.commit()


@router.get("/graph")
def knowledge_graph(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return semantic.graph(db, team.id)


@router.post("/resolve")
def resolve_text(
    payload: ResolveRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """How a phrase or question maps onto approved definitions for a dataset."""
    team_dataset(db, team, payload.dataset_id)
    return semantic.resolve(db, team.id, payload.dataset_id, payload.text)


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return {"query": q, "results": semantic.search(db, team.id, q, limit)}
