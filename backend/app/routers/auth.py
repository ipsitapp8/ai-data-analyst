"""Signup / login / me. Signup also claims any pending invites for the
signing-up email address, and -- only for the very first user ever, so the
pre-multi-tenancy data isn't orphaned -- makes them owner of the Legacy team."""
from __future__ import annotations

import hmac
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.authz import ensure_org_member
from app.models import Team, TeamMember, User
from app.rate_limit import enforce as rate_limit
from app.schemas import GoogleExchange, LoginRequest, SignupRequest, TokenResponse, UserOut
from app.security import create_access_token, get_current_user, hash_password, verify_password
from app.validation import is_valid_email

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _claim_pending_invites(db: Session, user: User, email: str) -> None:
    """Claim any pending invites sent to this email before they had an account."""
    pending = db.query(TeamMember).filter(
        TeamMember.invited_email == email, TeamMember.user_id.is_(None)
    ).all()
    for invite in pending:
        invite.user_id = user.id
        invite.status = "active"
        invited_team = db.get(Team, invite.team_id)
        if invited_team is not None:
            ensure_org_member(db, invited_team.community_id, user.id, "member")


@router.post("/signup", response_model=TokenResponse)
def signup(payload: SignupRequest, request: Request, db: Session = Depends(get_db)):
    # Keyed by IP alone (not email too): the point here is slowing down mass
    # account creation from one source, not per-email brute force -- signup
    # already fails fast on a duplicate email regardless.
    rate_limit(f"signup:{request.client.host}", max_attempts=10, window_seconds=60)

    email = payload.email.strip().lower()
    if not is_valid_email(email):
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

    _claim_pending_invites(db, user, email)

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
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    # Keyed by IP+email: caps guessing passwords for one account, and caps
    # one source hammering many accounts, without one user's repeated bad
    # attempts locking out everyone behind the same NAT/proxy IP.
    rate_limit(f"login:{request.client.host}:{email}", max_attempts=8, window_seconds=60)

    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, user=user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.post("/google-exchange", response_model=TokenResponse)
def google_exchange(payload: GoogleExchange, request: Request, db: Session = Depends(get_db),
                    x_internal_secret: str | None = Header(default=None)):
    """Session for an email that Google has already verified.

    Called only by our own Streamlit frontend (after its OIDC login) and gated by
    a shared secret, so it cannot be used to mint a session for an arbitrary
    address from the outside. Disabled (404) unless INTERNAL_AUTH_SECRET is set.
    """
    expected = config.INTERNAL_AUTH_SECRET
    if not expected:
        raise HTTPException(404, "Not found")
    rate_limit(f"google:{request.client.host}", max_attempts=30, window_seconds=60)
    if not x_internal_secret or not hmac.compare_digest(x_internal_secret.encode(), expected.encode()):
        raise HTTPException(403, "Forbidden")
    if not payload.email_verified:
        raise HTTPException(400, "Google has not verified this email address")
    email = payload.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(400, "A valid email is required")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        # No password: this account signs in with Google. A random hash keeps the
        # column populated and unusable for password login.
        user = User(
            email=email,
            password_hash=hash_password(secrets.token_urlsafe(32)),
            display_name=(payload.name or "").strip() or email.split("@")[0],
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        _claim_pending_invites(db, user, email)
        db.commit()
    return TokenResponse(access_token=create_access_token(user.id, user.email), user=user)
