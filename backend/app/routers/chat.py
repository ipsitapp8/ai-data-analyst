"""POST /api/questions/{id}/chat -- see app/chat.py."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import chat
from app.database import get_db
from app.models import Dashboard, Question, Team, User
from app.rate_limit import enforce as rate_limit
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/questions/{question_id}", tags=["chat"])


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=chat.MAX_MESSAGE_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=40)


@router.post("/chat")
def chat_with_dashboard(
    question_id: int,
    body: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    rate_limit(f"chat:{user.id}", max_attempts=20, window_seconds=60)
    question = db.get(Question, question_id)
    if not question or question.team_id != team.id:
        raise HTTPException(404, "Question not found")
    dash = (
        db.query(Dashboard).filter(Dashboard.question_id == question_id)
        .order_by(Dashboard.created_at.desc()).first()
    )
    if dash is None:
        raise HTTPException(404, "Dashboard not ready yet")
    message = body.message.strip()
    if not message:
        raise HTTPException(422, "Message is empty")
    try:
        return chat.answer_question(db, question, dash, message, [t.model_dump() for t in body.history])
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - provider/quota failures are the common case
        raise HTTPException(502, f"The assistant is unavailable right now ({type(e).__name__}). Try again shortly.")
