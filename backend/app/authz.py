"""Authorization policy for organizations, teams and workspaces.

Three authorities that must never be conflated:

* Organization owner/admin (CommunityMember.role): manages the company --
  groups, teams, members, invitations, org audit log. Also has implicit
  team-admin access to every team in their organization.
* Team owner/admin (TeamMember.role): legacy team-level administrators
  (the team's creator is its owner).
* Team leader (TeamMember.role == "leader"): coordinates one team -- invites
  plain members, moderates discussion -- but cannot change roles and has no
  organization authority, ever.

Everything here is derived server-side from the database; no role or id is
ever taken from the client.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import secrets

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Community, CommunityMember, OrgAuditEvent, Team, TeamMember, User

ORG_ROLES = ("owner", "admin", "member")
TEAM_ROLES = ("owner", "admin", "leader", "member")
ORG_ADMIN_ROLES = ("owner", "admin")
TEAM_ADMIN_ROLES = ("owner", "admin")
INVITE_TTL = dt.timedelta(days=7)


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


# ------------------------------------------------------------------ lookups --

def org_role(db: Session, community_id: int, user_id: int) -> str | None:
    m = (
        db.query(CommunityMember)
        .filter(CommunityMember.community_id == community_id, CommunityMember.user_id == user_id)
        .first()
    )
    return m.role if m else None


def is_org_admin(db: Session, community_id: int, user_id: int) -> bool:
    return org_role(db, community_id, user_id) in ORG_ADMIN_ROLES


def team_role(db: Session, team: Team, user_id: int) -> str | None:
    """Effective role on a team: the explicit membership, else an implicit
    'admin' for organization admins, else None (no access)."""
    m = (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team.id, TeamMember.user_id == user_id, TeamMember.status == "active")
        .first()
    )
    if m:
        return m.role
    if is_org_admin(db, team.community_id, user_id):
        return "admin"
    return None


def require_org_admin(db: Session, community_id: int, user_id: int, what: str) -> None:
    if not is_org_admin(db, community_id, user_id):
        raise HTTPException(403, f"Only an organization owner or admin can {what}")


def get_org_or_404(db: Session, community_id: int, user_id: int) -> Community:
    """404 (not 403) unless the caller belongs to the organization, so ids of
    other tenants reveal nothing -- not even that they exist."""
    community = db.get(Community, community_id)
    if community is None or org_role(db, community_id, user_id) is None:
        raise HTTPException(404, "Organization not found")
    return community


def get_team_for_manager(db: Session, team_id: int, user_id: int) -> tuple[Team, str]:
    """Team plus the caller's effective role; 404 if the caller has no access."""
    team = db.get(Team, team_id)
    role = team_role(db, team, user_id) if team else None
    if team is None or role is None:
        raise HTTPException(404, "Team not found")
    return team, role


def can_manage_members(role: str | None) -> bool:
    return role in ("owner", "admin", "leader")


def can_assign_team_role(actor_role: str, new_role: str) -> bool:
    """Leaders may only create plain members; owners/admins (team or org) may
    also appoint leaders. Nobody can mint an owner or admin through invitations
    or role changes here -- ownership transfer is a separate, deliberate flow."""
    if new_role == "member":
        return actor_role in ("owner", "admin", "leader")
    if new_role == "leader":
        return actor_role in ("owner", "admin")
    return False


# ------------------------------------------------------------- membership ops --

def ensure_org_member(db: Session, community_id: int, user_id: int, role: str = "member") -> CommunityMember:
    """Idempotent. Never downgrades an existing role."""
    m = (
        db.query(CommunityMember)
        .filter(CommunityMember.community_id == community_id, CommunityMember.user_id == user_id)
        .first()
    )
    if m is None:
        m = CommunityMember(community_id=community_id, user_id=user_id, role=role)
        db.add(m)
        db.flush()
    return m


def audit(db: Session, community_id: int, actor_id: int | None, action: str, *,
          team_id: int | None = None, target_type: str | None = None,
          target_id: int | None = None, **detail) -> None:
    """Record a sensitive action. Added to the caller's transaction (no commit),
    so the event and the change it describes succeed or fail together."""
    db.add(OrgAuditEvent(
        community_id=community_id, team_id=team_id, actor_id=actor_id, action=action,
        target_type=target_type, target_id=target_id, detail_json=detail,
    ))


# ----------------------------------------------------------- invitation tokens --

def new_invite_token() -> tuple[str, str]:
    """(raw token to hand to the invitee, hash to store)."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_token(raw)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ backfill --

def backfill_org_memberships() -> int:
    """Give pre-existing data the organization memberships it never had.

    Additive and idempotent: creators become org owners, every active team
    member becomes an org member, and a community that still has no owner (the
    'Legacy' one) promotes its earliest team owner. Existing rows are never
    modified or removed. Returns the number of rows added."""
    from app.database import SessionLocal

    added = 0
    db = SessionLocal()
    try:
        for c in db.query(Community).all():
            if c.created_by and org_role(db, c.id, c.created_by) is None:
                ensure_org_member(db, c.id, c.created_by, "owner")
                added += 1
        rows = (
            db.query(Team.community_id, TeamMember.user_id)
            .join(TeamMember, TeamMember.team_id == Team.id)
            .filter(TeamMember.status == "active", TeamMember.user_id.isnot(None))
            .distinct()
            .all()
        )
        for community_id, user_id in rows:
            if org_role(db, community_id, user_id) is None:
                ensure_org_member(db, community_id, user_id, "member")
                added += 1
        db.flush()
        for c in db.query(Community).all():
            has_owner = (
                db.query(CommunityMember)
                .filter(CommunityMember.community_id == c.id, CommunityMember.role == "owner")
                .first()
            )
            if has_owner:
                continue
            first_owner = (
                db.query(TeamMember)
                .join(Team, Team.id == TeamMember.team_id)
                .filter(Team.community_id == c.id, TeamMember.role == "owner",
                        TeamMember.status == "active", TeamMember.user_id.isnot(None))
                .order_by(TeamMember.id)
                .first()
            )
            if first_owner:
                cm = ensure_org_member(db, c.id, first_owner.user_id, "owner")
                cm.role = "owner"
                added += 1
        db.commit()
    finally:
        db.close()
    return added


def user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email.strip().lower()).first()
