"""Signup / login / me. Signup also claims any pending invites for the
signing-up email address, and -- only for the very first user ever, so the
pre-multi-tenancy data isn't orphaned -- makes them owner of the Legacy team."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Team, TeamMember, User
from app.schemas import LoginRequest, SignupRequest, TokenResponse, UserOut
from app.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "A valid email is required")
    if len(payload.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "An account with this email already exists")

    is_first_user_ever = db.query(User).count() == 0

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name.strip() or email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Claim any pending invites sent to this email before they had an account.
    pending = db.query(TeamMember).filter(
        TeamMember.invited_email == email, TeamMember.user_id.is_(None)
    ).all()
    for invite in pending:
        invite.user_id = user.id
        invite.status = "active"

    if is_first_user_ever:
        legacy_team = db.query(Team).filter(Team.name == "Legacy").first()
        if legacy_team is not None:
            already_member = db.query(TeamMember).filter(
                TeamMember.team_id == legacy_team.id, TeamMember.user_id == user.id
            ).first()
            if not already_member:
                db.add(TeamMember(team_id=legacy_team.id, user_id=user.id, role="owner", status="active"))

    db.commit()

    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, user=user)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, user=user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
