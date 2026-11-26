"""Saved Silt Copilot conversations: list, search, restore, rename, delete.

Every query is scoped to the signed-in user AND their active team, so one person's chats are never returned to
another, including a teammate in the same workspace. The analytical memory a follow-up needs is stored separately
(app/analyst/persist.py) under the same conversation id."""
from __future__ import annotations

import datetime as dt
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CopilotConversation, CopilotMessage, Dataset, Team, User
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/copilot/conversations", tags=["copilot"])

ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
MAX_RESULTS = 60


class ConversationIn(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    dataset_id: int | None = None
    document_id: str | None = Field(default=None, max_length=64)


class MessageIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    response: dict | None = None
    answer_text: str = Field(default="", max_length=20000)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _owned(db: Session, user: User, team: Team, cid: str) -> CopilotConversation:
    conv = db.get(CopilotConversation, cid)
    if conv is None or conv.user_id != user.id or conv.team_id != team.id:
        raise HTTPException(404, "Conversation not found")
    return conv


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _snippet(text: str, needle: str, width: int = 90) -> str:
    text = " ".join((text or "").split())
    i = text.lower().find(needle.lower())
    if i < 0:
        return text[: width * 2]
    start = max(0, i - width // 2)
    return ("…" if start else "") + text[start:start + width * 2] + ("…" if start + width * 2 < len(text) else "")


def _dataset_names(db: Session, team: Team) -> dict[int, str]:
    return {d.id: d.filename for d in db.query(Dataset).filter(Dataset.team_id == team.id).all()}


def _summary(db: Session, conv: CopilotConversation, names: dict[int, str]) -> dict:
    count = db.query(func.count(CopilotMessage.id)).filter_by(conversation_id=conv.id).scalar() or 0
    return {"id": conv.id, "title": conv.title, "dataset_id": conv.dataset_id, "document_id": conv.document_id,
            "dataset_name": names.get(conv.dataset_id), "message_count": count,
            "created_at": conv.created_at.isoformat(), "updated_at": conv.updated_at.isoformat()}


@router.get("")
def list_conversations(
    q: str | None = Query(default=None, max_length=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Recent chats, newest first. With `q`: chats whose title, dataset name, a question or an answer matches,
    each with the snippet that matched."""
    names = _dataset_names(db, team)
    base = db.query(CopilotConversation).filter_by(user_id=user.id, team_id=team.id)
    needle = (q or "").strip()
    if not needle:
        rows = base.order_by(CopilotConversation.updated_at.desc()).limit(200).all()
        return {"conversations": [_summary(db, c, names) for c in rows]}

    pat = _like(needle)
    by_dataset = [i for i, n in names.items() if needle.lower() in (n or "").lower()]
    text_match = or_(CopilotMessage.question.ilike(pat, escape="\\"), CopilotMessage.answer_text.ilike(pat, escape="\\"))
    msg_ids = {cid for (cid,) in db.query(CopilotMessage.conversation_id).join(CopilotConversation).filter(
        CopilotConversation.user_id == user.id, CopilotConversation.team_id == team.id, text_match).all()}
    cond = [CopilotConversation.title.ilike(pat, escape="\\"), CopilotConversation.id.in_(msg_ids or [""])]
    if by_dataset:
        cond.append(CopilotConversation.dataset_id.in_(by_dataset))
    rows = base.filter(or_(*cond)).order_by(CopilotConversation.updated_at.desc()).limit(MAX_RESULTS).all()
    out = []
    for c in rows:
        item = _summary(db, c, names)
        snippet, where = "", ""
        if needle.lower() in (c.title or "").lower():
            snippet, where = c.title, "title"
        else:
            m = (db.query(CopilotMessage).filter(CopilotMessage.conversation_id == c.id, text_match)
                 .order_by(CopilotMessage.position).first())
            if m is not None:
                in_q = needle.lower() in (m.question or "").lower()
                snippet, where = _snippet(m.question if in_q else m.answer_text, needle), ("question" if in_q else "answer")
            elif c.dataset_id in by_dataset:
                snippet, where = names.get(c.dataset_id, ""), "dataset"
        item.update(snippet=snippet, matched_in=where)
        out.append(item)
    return {"conversations": out}


@router.put("/{cid}")
def save_conversation(cid: str, body: ConversationIn, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """Create the conversation, or update its dataset/document context (and title if one is given)."""
    if not ID_RE.match(cid):
        raise HTTPException(400, "Invalid conversation id")
    conv = db.get(CopilotConversation, cid)
    if conv is not None and (conv.user_id != user.id or conv.team_id != team.id):
        raise HTTPException(404, "Conversation not found")
    if conv is None:
        conv = CopilotConversation(id=cid, team_id=team.id, user_id=user.id,
                                   title=(body.title or "New chat").strip() or "New chat")
        db.add(conv)
    elif body.title:
        conv.title = body.title.strip()[:200] or conv.title
    conv.dataset_id, conv.document_id = body.dataset_id, body.document_id
    conv.updated_at = dt.datetime.utcnow()
    db.commit()
    return _summary(db, conv, _dataset_names(db, team))


@router.put("/{cid}/messages/{position}")
def save_message(cid: str, position: int, body: MessageIn, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """Store (or update) the question at `position` and, once known, its answer. Safe to repeat."""
    conv = _owned(db, user, team, cid)
    if position < 0 or position > 5000:
        raise HTTPException(400, "Invalid position")
    msg = db.query(CopilotMessage).filter_by(conversation_id=cid, position=position).first()
    if msg is None:
        msg = CopilotMessage(conversation_id=cid, position=position, question=body.question)
        db.add(msg)
    msg.question = body.question
    msg.response_json = body.response
    msg.answer_text = body.answer_text
    conv.updated_at = dt.datetime.utcnow()
    if conv.title in ("New chat", "New conversation", "") and position == 0:
        conv.title = " ".join(body.question.split())[:80]
    db.commit()
    return {"ok": True}


@router.get("/{cid}")
def get_conversation(cid: str, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    conv = _owned(db, user, team, cid)
    out = _summary(db, conv, _dataset_names(db, team))
    out["messages"] = [{"position": m.position, "question": m.question, "response": m.response_json,
                        "created_at": m.created_at.isoformat()} for m in conv.messages]
    return out


@router.patch("/{cid}")
def rename_conversation(cid: str, body: RenameIn, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    conv = _owned(db, user, team, cid)
    conv.title = body.title.strip()[:200] or conv.title
    db.commit()
    return _summary(db, conv, _dataset_names(db, team))


@router.delete("/{cid}", status_code=204)
def delete_conversation(cid: str, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    conv = _owned(db, user, team, cid)
    db.delete(conv)
    db.commit()
