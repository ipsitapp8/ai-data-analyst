"""Evaluation results (read-only). Runs are produced by `python -m app.evals`.

Evaluation data is synthetic and belongs to no team, so any signed-in user may
read it. Nothing here can start a run: running the suite executes code and, in
live mode, spends money, and that stays an operator action on the command line.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.evals import runner
from app.models import EvalResult, EvalRun, User
from app.security import get_current_user

router = APIRouter(prefix="/api/evals", tags=["evals"])


def _stored(db: Session, run_id: int) -> dict:
    run = db.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(404, "Evaluation run not found")
    return runner.run_out(run, db.query(EvalResult).filter_by(run_id=run_id).order_by(EvalResult.id).all())


@router.get("/runs")
def list_runs(
    limit: int = Query(default=30, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    runs = db.query(EvalRun).order_by(EvalRun.id.desc()).limit(limit).all()
    return [runner.run_out(r) for r in runs]


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return _stored(db, run_id)


@router.get("/compare")
def compare_runs(
    base: int = Query(...),
    new: int = Query(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return runner.compare(_stored(db, base), _stored(db, new))


@router.get("/suite")
def suite_info(user: User = Depends(get_current_user)):
    suite = runner.load_suite()
    by_component: dict[str, int] = {}
    for c in suite["cases"]:
        by_component[c["component"]] = by_component.get(c["component"], 0) + 1
    return {"suite_version": suite["suite_version"], "cases": len(suite["cases"]),
            "live_cases": sum(1 for c in suite["cases"] if c.get("live")), "by_component": by_component}
