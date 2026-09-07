"""Password hashing, JWT session tokens, and the FastAPI auth dependencies
that every team-scoped endpoint depends on.
"""
from __future__ import annotations

import datetime as dt

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.models import Team, TeamMember, User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, email: str) -> str:
    now = dt.datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + dt.timedelta(minutes=config.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except jwt.PyJWTError as e:
        raise HTTPException(401, "Invalid or expired token") from e


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(401, "User no longer exists")
    return user


def get_current_team(
    x_team_id: int | None = Header(default=None, alias="X-Team-Id"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Team:
    if x_team_id is None:
        raise HTTPException(400, "X-Team-Id header is required")
    team = db.get(Team, x_team_id)
    if team is None:
        raise HTTPException(404, "Team not found")
    member = (
        db.query(TeamMember)
        .filter(
            TeamMember.team_id == team.id,
            TeamMember.user_id == user.id,
            TeamMember.status == "active",
        )
        .first()
    )
    if member is None:
        raise HTTPException(403, "Not a member of this team")
    return team
