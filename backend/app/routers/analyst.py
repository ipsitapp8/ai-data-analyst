"""Analyst Copilot API (see app/analyst). Separate from /api/copilot (the
rules-only session reader) and from the Planner/Executor/Critic questions."""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.analyst import documents, model_info, orchestrator, planner
from app.analyst.data import load_table, schema
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import Team, User
from app.routers._common import team_dataset
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/analyst", tags=["analyst"])
PREVIEW_ROWS = 20


class AskIn(BaseModel):
    dataset_id: int | None = None
    document_id: str | None = Field(default=None, max_length=32)
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str = Field(default="default", max_length=64)
    use_model: bool | None = None      # None: use the model when it is being served


def _frame(db: Session, team: Team, dataset_id: int):
    dataset = team_dataset(db, team, dataset_id)
    version = latest_version(db, dataset)
    path = version.filepath if version is not None else dataset.filepath
    try:
        return dataset, load_table(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"The dataset could not be read ({type(e).__name__}).") from e


def _cell(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    return v.item() if hasattr(v, "item") else v


@router.get("/status")
def status(user: User = Depends(get_current_user)):
    import os
    import urllib.request
    dec = os.getenv("ANALYST_DECOMPOSER_URL")
    dec_ok = None
    if dec:
        try:
            with urllib.request.urlopen(dec.rstrip("/") + "/models", timeout=1.5) as r:
                dec_ok = r.status == 200
        except Exception:  # noqa: BLE001
            dec_ok = False
    return {"model_url": planner.MODEL_URL, "model_name": planner.MODEL_LABEL, "model_available": planner.model_available(),
            "served": model_info.served(planner.MODEL_URL),
            "decomposer_url": dec or planner.MODEL_URL, "decomposer_available": dec_ok if dec else planner.model_available()}


@router.get("/datasets/{dataset_id}/schema")
def dataset_schema(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                   team: Team = Depends(get_current_team)):
    _, df = _frame(db, team, dataset_id)
    return schema(df)


@router.get("/datasets/{dataset_id}/preview")
def dataset_preview(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                    team: Team = Depends(get_current_team)):
    dataset, df = _frame(db, team, dataset_id)
    head = df.head(PREVIEW_ROWS)
    return {"filename": dataset.filename, "rows": int(len(df)), "columns": list(map(str, df.columns)),
            "schema": schema(df)["columns"],
            "preview": [{str(c): _cell(v) for c, v in row.items()} for row in head.to_dict(orient="records")]}


@router.post("/documents")
async def upload_document(file: UploadFile, user: User = Depends(get_current_user),
                          team: Team = Depends(get_current_team)):
    data = await file.read(documents.MAX_BYTES + 1)
    try:
        return documents.save(team.id, file.filename or "document.pdf", data)
    except documents.DocumentError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/documents")
def list_documents(user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    return documents.list_docs(team.id)


@router.post("/ask")
def ask(payload: AskIn, db: Session = Depends(get_db), user: User = Depends(get_current_user),
        team: Team = Depends(get_current_team)):
    if payload.dataset_id is None and payload.document_id is None:
        raise HTTPException(400, "Choose a dataset or a document to ask about.")
    dataset, df = (None, None) if payload.dataset_id is None else _frame(db, team, payload.dataset_id)
    doc = None
    if payload.document_id:
        try:
            doc = documents.load(team.id, payload.document_id)
        except documents.DocumentError as e:
            raise HTTPException(404, str(e)) from e
    key = orchestrator.memory_key(team.id, user.id, payload.dataset_id, payload.document_id, payload.conversation_id)
    # only a conversation the client identified is kept across restarts: a request without one never loads another
    # session's memory from disk
    persist = "conversation_id" in payload.model_fields_set
    with orchestrator.memory_lock(key):                # one turn at a time per conversation: no lost updates
        memory = orchestrator.memory_for(key, persist=persist)
        out = orchestrator.ask(payload.message, df, memory, use_model=payload.use_model,
                               filename=dataset.filename if dataset else None, document=doc)
        if persist:
            orchestrator.save_memory(key, memory)
    return {**out,
            "dataset": {"id": dataset.id, "filename": dataset.filename, "rows": int(len(df))} if dataset else None,
            "document": documents.summary(doc) if doc else None}
