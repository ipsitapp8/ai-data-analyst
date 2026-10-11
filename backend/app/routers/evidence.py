"""Evidence records, run manifests, reruns and comparisons."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import data_quality, jobs, provenance, verification
from app.database import get_db
from app.models import DatasetVersion, EvidenceRecord, Team, User
from app.routers._common import dataset_version, team_dataset, team_question
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api", tags=["evidence"])


class RerunRequest(BaseModel):
    dataset_version_id: int | None = None


@router.get("/questions/{question_id}/evidence")
def question_evidence(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    team_question(db, team, question_id)
    rows = db.query(EvidenceRecord).filter_by(question_id=question_id).order_by(EvidenceRecord.id).all()
    records = [verification.record_out(r) for r in rows]
    return {"question_id": question_id, "summary": verification.summarize(
        [{"status": r["status"], "checks": r["checks"]} for r in records]), "records": records}


@router.get("/questions/{question_id}/manifest")
def question_manifest(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return provenance.get_manifest(db, team_question(db, team, question_id))


@router.post("/questions/{question_id}/rerun", status_code=201)
def rerun_question(
    question_id: int,
    body: RerunRequest | None = None,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Run the same question again, pinned to a dataset version (default: the
    one the original ran on). The original run is not modified."""
    question = team_question(db, team, question_id)
    try:
        new_q, job, created = provenance.rerun(
            db, question, user_id=user.id, dataset_version_id=body.dataset_version_id if body else None,
            idempotency_key=idempotency_key)
    except jobs.AdmissionRejected as e:
        raise HTTPException(429, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"question_id": new_q.id, "job_id": job.id, "status": new_q.status, "created": created,
            "rerun_of_question_id": question_id, "dataset_version_id": new_q.dataset_version_id}


@router.get("/questions/{question_id}/compare/{other_id}")
def compare_questions(
    question_id: int,
    other_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    a, b = team_question(db, team, question_id), team_question(db, team, other_id)
    return provenance.compare_runs(db, a, b)


@router.get("/datasets/{dataset_id}/versions")
def list_versions(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dataset = team_dataset(db, team, dataset_id)
    versions = (db.query(DatasetVersion).filter_by(dataset_id=dataset.id)
                .order_by(DatasetVersion.version_number).all())
    return [{"id": v.id, "version": v.version_number, "filename": v.filename, "row_count": v.row_count,
             "col_count": v.col_count, "uploaded_at": v.uploaded_at,
             "content_sha256": provenance.fingerprint(db, v)} for v in versions]


@router.get("/datasets/{dataset_id}/versions/compare")
def compare_dataset_versions(
    dataset_id: int,
    a: int = Query(..., description="Earlier version id"),
    b: int = Query(..., description="Later version id"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dataset = team_dataset(db, team, dataset_id)
    va, vb = dataset_version(db, dataset, a), dataset_version(db, dataset, b)
    provenance.fingerprint(db, va)
    provenance.fingerprint(db, vb)
    try:
        return data_quality.compare_versions(db, dataset, va, vb)
    except OSError as e:
        raise HTTPException(410, "The file of one of these versions is no longer available") from e
