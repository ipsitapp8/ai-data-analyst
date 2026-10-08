"""CRUD for dataset knowledge notes and lessons (see app/knowledge.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import config, knowledge
from app.database import get_db
from app.models import Dataset, KnowledgeNote, Team, User
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/datasets/{dataset_id}/notes", tags=["knowledge"])


class NoteCreate(BaseModel):
    kind: str
    text: str


class NoteOut(BaseModel):
    id: int
    kind: str
    source: str
    text: str
    created_at: str


def _out(n: KnowledgeNote) -> NoteOut:
    return NoteOut(id=n.id, kind=n.kind, source=n.source, text=n.text, created_at=n.created_at.isoformat())


def _team_dataset(db: Session, team: Team, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.get("", response_model=list[NoteOut])
def list_notes(
    dataset_id: int,
    kind: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _team_dataset(db, team, dataset_id)
    q = db.query(KnowledgeNote).filter_by(team_id=team.id, dataset_id=dataset_id)
    if kind:
        q = q.filter_by(kind=kind)
    return [_out(n) for n in q.order_by(KnowledgeNote.created_at.desc(), KnowledgeNote.id.desc()).all()]


@router.post("", response_model=NoteOut, status_code=201)
def create_note(
    dataset_id: int,
    body: NoteCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _team_dataset(db, team, dataset_id)
    if body.kind not in knowledge.KINDS:
        raise HTTPException(422, f"kind must be one of {', '.join(knowledge.KINDS)}")
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Note text is empty")
    if len(text) > config.NOTE_MAX_CHARS:
        raise HTTPException(422, f"Note is too long (max {config.NOTE_MAX_CHARS} characters)")
    count = db.query(KnowledgeNote).filter_by(team_id=team.id, dataset_id=dataset_id).count()
    if count >= config.NOTES_MAX_PER_DATASET:
        raise HTTPException(409, "This dataset has reached its note limit; delete some first")
    note = knowledge.add_note(db, team_id=team.id, dataset_id=dataset_id, kind=body.kind,
                              text=text, created_by=user.id)
    return _out(note)


@router.delete("/{note_id}", status_code=204)
def delete_note(
    dataset_id: int,
    note_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _team_dataset(db, team, dataset_id)
    note = db.get(KnowledgeNote, note_id)
    if not note or note.team_id != team.id or note.dataset_id != dataset_id:
        raise HTTPException(404, "Note not found")
    db.delete(note)
    db.commit()
