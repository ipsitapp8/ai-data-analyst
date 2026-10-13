"""Organization structure, role management, expiring invitations and the
workspace discussion feed.

Every route derives the caller's authority from the database (app/authz.py);
no organization id, team id or role in a request body is trusted. Anything
that doesn't belong to the caller's organization answers 404, never 403, so
ids from other tenants reveal nothing.
"""
from __future__ import annotations

import urllib.parse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.orm import Session

from app import authz, config
from app.authz import (
    audit,
    can_assign_team_role,
    can_manage_members,
    ensure_org_member,
    get_org_or_404,
    get_team_for_manager,
    require_org_admin,
    team_role,
    utcnow,
)
from app.database import get_db
from app.email_sender import send_team_invite_email
from app.models import (
    CommunityMember,
    Dataset,
    Group,
    Invitation,
    Investigation,
    OrgAuditEvent,
    Question,
    Team,
    TeamMember,
    User,
    WorkspaceComment,
)
from app.security import get_current_team, get_current_user
from app.validation import is_valid_email

router = APIRouter(prefix="/api", tags=["organization"])


# ===================================================================== groups ==

class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class GroupAssign(BaseModel):
    group_id: int | None = None


@router.get("/communities/{community_id}/groups")
def list_groups(community_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    get_org_or_404(db, community_id, user.id)
    rows = db.query(Group).filter(Group.community_id == community_id).order_by(Group.name).all()
    return [{"id": g.id, "name": g.name} for g in rows]


@router.post("/communities/{community_id}/groups")
def create_group(community_id: int, payload: GroupCreate, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    get_org_or_404(db, community_id, user.id)
    require_org_admin(db, community_id, user.id, "create groups")
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Group name is required")
    if db.query(Group).filter(Group.community_id == community_id, Group.name == name).first():
        raise HTTPException(409, "A group with this name already exists")
    g = Group(community_id=community_id, name=name, created_by=user.id)
    db.add(g)
    db.flush()
    audit(db, community_id, user.id, "group.created", target_type="group", target_id=g.id, name=name)
    db.commit()
    return {"id": g.id, "name": g.name}


@router.put("/teams/{team_id}/group")
def assign_team_group(team_id: int, payload: GroupAssign, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    team, _ = get_team_for_manager(db, team_id, user.id)
    require_org_admin(db, team.community_id, user.id, "move a team between groups")
    if payload.group_id is not None:
        g = db.get(Group, payload.group_id)
        if g is None or g.community_id != team.community_id:
            raise HTTPException(400, "Unknown group for this organization")
    team.group_id = payload.group_id
    audit(db, team.community_id, user.id, "team.group_changed", team_id=team.id,
          target_type="team", target_id=team.id, group_id=payload.group_id)
    db.commit()
    return {"team_id": team.id, "group_id": team.group_id}


# ============================================================ organization members ==

class OrgRoleChange(BaseModel):
    role: str


@router.get("/communities/{community_id}/members")
def list_org_members(community_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    get_org_or_404(db, community_id, user.id)
    require_org_admin(db, community_id, user.id, "view organization members")
    rows = (
        db.query(CommunityMember, User)
        .join(User, User.id == CommunityMember.user_id)
        .filter(CommunityMember.community_id == community_id)
        .order_by(CommunityMember.id)
        .all()
    )
    return [{"user_id": u.id, "email": u.email, "display_name": u.display_name, "role": m.role} for m, u in rows]


@router.put("/communities/{community_id}/members/{user_id}/role")
def change_org_role(community_id: int, user_id: int, payload: OrgRoleChange,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    get_org_or_404(db, community_id, user.id)
    # Only an owner may grant or revoke admin: an admin must not be able to
    # promote themselves or others into the same authority tier.
    if authz.org_role(db, community_id, user.id) != "owner":
        raise HTTPException(403, "Only the organization owner can change organization roles")
    if payload.role not in ("admin", "member"):
        raise HTTPException(400, "Role must be 'admin' or 'member'")
    target = (db.query(CommunityMember)
              .filter(CommunityMember.community_id == community_id, CommunityMember.user_id == user_id).first())
    if target is None:
        raise HTTPException(404, "Member not found")
    if target.role == "owner":
        raise HTTPException(400, "The owner's role cannot be changed here")
    before = target.role
    target.role = payload.role
    audit(db, community_id, user.id, "org.role_changed", target_type="user", target_id=user_id,
          before=before, after=payload.role)
    db.commit()
    return {"user_id": user_id, "role": target.role}


@router.delete("/communities/{community_id}/members/{user_id}")
def remove_org_member(community_id: int, user_id: int, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    get_org_or_404(db, community_id, user.id)
    require_org_admin(db, community_id, user.id, "remove organization members")
    target = (db.query(CommunityMember)
              .filter(CommunityMember.community_id == community_id, CommunityMember.user_id == user_id).first())
    if target is None:
        raise HTTPException(404, "Member not found")
    if target.role == "owner":
        raise HTTPException(400, "The organization owner cannot be removed")
    if target.role == "admin" and authz.org_role(db, community_id, user.id) != "owner":
        raise HTTPException(403, "Only the owner can remove an organization admin")
    team_ids = [t.id for t in db.query(Team.id).filter(Team.community_id == community_id).all()]
    if team_ids:
        db.query(TeamMember).filter(TeamMember.team_id.in_(team_ids), TeamMember.user_id == user_id).delete(
            synchronize_session=False)
    db.delete(target)
    audit(db, community_id, user.id, "org.member_removed", target_type="user", target_id=user_id)
    db.commit()
    return {"status": "removed"}


@router.get("/communities/{community_id}/audit-events")
def list_audit_events(community_id: int, limit: int = 100, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    get_org_or_404(db, community_id, user.id)
    require_org_admin(db, community_id, user.id, "view the organization audit log")
    rows = (db.query(OrgAuditEvent).filter(OrgAuditEvent.community_id == community_id)
            .order_by(OrgAuditEvent.id.desc()).limit(max(1, min(limit, 500))).all())
    return [{
        "id": e.id, "action": e.action, "actor_id": e.actor_id, "team_id": e.team_id,
        "target_type": e.target_type, "target_id": e.target_id, "detail": e.detail_json or {},
        "created_at": e.created_at.isoformat() if e.created_at else None,
    } for e in rows]


# ================================================================== team members ==

class TeamRoleChange(BaseModel):
    role: str


@router.put("/teams/{team_id}/members/{member_id}/role")
def change_team_role(team_id: int, member_id: int, payload: TeamRoleChange,
                     user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    team, actor_role = get_team_for_manager(db, team_id, user.id)
    if payload.role not in ("member", "leader"):
        raise HTTPException(400, "Role must be 'member' or 'leader'")
    if not can_assign_team_role(actor_role, payload.role):
        raise HTTPException(403, "You are not allowed to assign that role")
    target = db.query(TeamMember).filter(TeamMember.id == member_id, TeamMember.team_id == team.id).first()
    if target is None or target.status != "active":
        raise HTTPException(404, "Member not found")
    if target.role in ("owner", "admin"):
        raise HTTPException(403, "Team owners and admins cannot be changed here")
    before = target.role
    target.role = payload.role
    audit(db, team.community_id, user.id, "team.role_changed", team_id=team.id, target_type="user",
          target_id=target.user_id, before=before, after=payload.role)
    db.commit()
    return {"member_id": target.id, "role": target.role}


@router.delete("/teams/{team_id}/members/{member_id}/membership")
def remove_team_member(team_id: int, member_id: int, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    team, actor_role = get_team_for_manager(db, team_id, user.id)
    target = db.query(TeamMember).filter(TeamMember.id == member_id, TeamMember.team_id == team.id).first()
    if target is None or target.status != "active":
        raise HTTPException(404, "Member not found")
    if target.role == "owner":
        raise HTTPException(403, "The team owner cannot be removed")
    if not can_manage_members(actor_role):
        raise HTTPException(403, "Only a team leader, owner or admin can remove members")
    # A leader manages plain members; removing another leader or an admin is
    # reserved to owners/admins so a leader cannot evict their peers or superiors.
    if target.role in ("admin", "leader") and actor_role not in ("owner", "admin"):
        raise HTTPException(403, "Only a team owner or admin can remove a leader or admin")
    audit(db, team.community_id, user.id, "team.member_removed", team_id=team.id,
          target_type="user", target_id=target.user_id, role=target.role)
    db.delete(target)
    db.commit()
    return {"status": "removed"}


# ================================================================== invitations ==

class InvitationCreate(BaseModel):
    email: str
    team_role: str = "member"
    org_role: str = "member"


class InvitationAccept(BaseModel):
    token: str


def _accept_url(raw_token: str) -> str:
    return f"{config.APP_URL.rstrip('/')}/?invite={urllib.parse.quote(raw_token)}"


def _invitation_out(inv: Invitation) -> dict:
    return {
        "id": inv.id, "email": inv.email, "team_id": inv.team_id, "org_role": inv.org_role,
        "team_role": inv.team_role, "expires_at": inv.expires_at.isoformat(),
        "accepted": inv.accepted_at is not None, "revoked": inv.revoked_at is not None,
    }


@router.post("/teams/{team_id}/invitations")
def create_invitation(team_id: int, payload: InvitationCreate, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    team, actor_role = get_team_for_manager(db, team_id, user.id)
    if not can_manage_members(actor_role):
        raise HTTPException(403, "Only a team leader, owner or admin can invite members")
    if payload.team_role not in ("member", "leader") or not can_assign_team_role(actor_role, payload.team_role):
        raise HTTPException(403, "You are not allowed to invite with that team role")
    if payload.org_role not in ("member", "admin"):
        raise HTTPException(400, "Organization role must be 'member' or 'admin'")
    # Granting organization-level authority is reserved to the organization owner.
    if payload.org_role == "admin" and authz.org_role(db, team.community_id, user.id) != "owner":
        raise HTTPException(403, "Only the organization owner can invite someone as an organization admin")
    email = payload.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(400, "That doesn't look like a valid email address")

    existing = authz.user_by_email(db, email)
    if existing and db.query(TeamMember).filter(
        TeamMember.team_id == team.id, TeamMember.user_id == existing.id, TeamMember.status == "active"
    ).first():
        raise HTTPException(409, "This person is already a member of this team")

    # One live invitation per (team, email): a re-invite supersedes the old link.
    now = utcnow()
    for old in db.query(Invitation).filter(
        Invitation.team_id == team.id, Invitation.email == email,
        Invitation.accepted_at.is_(None), Invitation.revoked_at.is_(None),
    ).all():
        old.revoked_at = now

    raw, token_hash = authz.new_invite_token()
    inv = Invitation(
        community_id=team.community_id, team_id=team.id, email=email, org_role=payload.org_role,
        team_role=payload.team_role, token_hash=token_hash, expires_at=now + authz.INVITE_TTL,
        invited_by=user.id,
    )
    db.add(inv)
    db.flush()
    audit(db, team.community_id, user.id, "invitation.created", team_id=team.id,
          target_type="invitation", target_id=inv.id, email=email, org_role=inv.org_role,
          team_role=inv.team_role)
    db.commit()
    db.refresh(inv)

    community = db.get(authz.Community, team.community_id)
    url = _accept_url(raw)
    email_sent = send_team_invite_email(
        to_email=email, inviter_name=user.display_name,
        community_name=community.name if community else "", team_name=team.name, accept_url=url,
    )
    out = _invitation_out(inv)
    out["email_sent"] = email_sent
    # The token is shown to the inviter only when no email carried it, so they
    # can hand it over themselves. Acceptance still requires being signed in
    # with the invited email address, so the link alone is not a credential.
    out["accept_url"] = None if email_sent else url
    return out


@router.get("/teams/{team_id}/invitations")
def list_invitations(team_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    team, actor_role = get_team_for_manager(db, team_id, user.id)
    if not can_manage_members(actor_role):
        raise HTTPException(403, "Only a team leader, owner or admin can view invitations")
    rows = (db.query(Invitation).filter(Invitation.team_id == team.id, Invitation.accepted_at.is_(None),
                                        Invitation.revoked_at.is_(None))
            .order_by(Invitation.id.desc()).all())
    return [_invitation_out(i) for i in rows]


@router.delete("/teams/{team_id}/invitations/{invitation_id}")
def revoke_invitation(team_id: int, invitation_id: int, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    team, actor_role = get_team_for_manager(db, team_id, user.id)
    if not can_manage_members(actor_role):
        raise HTTPException(403, "Only a team leader, owner or admin can revoke invitations")
    inv = db.query(Invitation).filter(Invitation.id == invitation_id, Invitation.team_id == team.id).first()
    if inv is None:
        raise HTTPException(404, "Invitation not found")
    if inv.accepted_at is not None:
        raise HTTPException(400, "This invitation was already accepted")
    inv.revoked_at = inv.revoked_at or utcnow()
    audit(db, team.community_id, user.id, "invitation.revoked", team_id=team.id,
          target_type="invitation", target_id=inv.id, email=inv.email)
    db.commit()
    return {"status": "revoked"}


@router.post("/invitations/accept")
def accept_invitation(payload: InvitationAccept, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    invalid = HTTPException(400, "This invitation is invalid, expired or already used")
    token = (payload.token or "").strip()
    if not token or len(token) > 200:
        raise invalid
    inv = db.query(Invitation).filter(Invitation.token_hash == authz.hash_token(token)).first()
    # Same response for unknown / expired / used / revoked / wrong-account, so
    # the endpoint can't be used to probe which tokens exist.
    if inv is None or inv.email != user.email.strip().lower():
        raise invalid

    # Atomic single-use claim: only one concurrent request can flip accepted_at
    # from NULL, and only while the invitation is still live.
    now = utcnow()
    claimed = db.execute(
        update(Invitation)
        .where(Invitation.id == inv.id, Invitation.accepted_at.is_(None),
               Invitation.revoked_at.is_(None), Invitation.expires_at > now)
        .values(accepted_at=now, accepted_by=user.id)
    ).rowcount
    if claimed != 1:
        db.rollback()
        raise invalid

    team = db.get(Team, inv.team_id)
    cm = ensure_org_member(db, inv.community_id, user.id, "member")
    if inv.org_role == "admin" and cm.role == "member":
        cm.role = "admin"
    tm = db.query(TeamMember).filter(TeamMember.team_id == inv.team_id, TeamMember.user_id == user.id).first()
    if tm is None:
        db.add(TeamMember(team_id=inv.team_id, user_id=user.id, role=inv.team_role, status="active"))
    elif tm.status != "active":
        tm.status = "active"
        tm.role = inv.team_role
    audit(db, inv.community_id, user.id, "invitation.accepted", team_id=inv.team_id,
          target_type="invitation", target_id=inv.id, org_role=inv.org_role, team_role=inv.team_role)
    db.commit()
    return {"status": "joined", "community_id": inv.community_id, "team_id": inv.team_id,
            "team_name": team.name if team else ""}


# =================================================================== discussion ==

COMMENT_TARGETS = {"question": Question, "dataset": Dataset, "investigation": Investigation}


class CommentCreate(BaseModel):
    body: str = Field(min_length=1, max_length=4000)
    target_type: str | None = None
    target_id: int | None = None
    kind: str = "comment"


class CommentUpdate(BaseModel):
    body: str | None = Field(default=None, min_length=1, max_length=4000)
    status: str | None = None


def _comment_out(c: WorkspaceComment, authors: dict[int, User], viewer_id: int, viewer_role: str | None) -> dict:
    a = authors.get(c.author_id)
    return {
        "id": c.id, "author_id": c.author_id, "author_name": a.display_name if a else "Unknown",
        "target_type": c.target_type, "target_id": c.target_id, "kind": c.kind, "body": c.body,
        "status": c.status, "created_at": c.created_at.isoformat() if c.created_at else None,
        "edited_at": c.edited_at.isoformat() if c.edited_at else None,
        "edit_count": len(c.edit_history or []),
        "can_edit": c.author_id == viewer_id,
        "can_moderate": c.author_id == viewer_id or can_manage_members(viewer_role),
    }


def _team_comment(db: Session, team: Team, comment_id: int) -> WorkspaceComment:
    c = db.get(WorkspaceComment, comment_id)
    if c is None or c.team_id != team.id or c.deleted_at is not None:
        raise HTTPException(404, "Comment not found")
    return c


@router.post("/discussion")
def post_comment(payload: CommentCreate, team: Team = Depends(get_current_team),
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    body = payload.body.strip()
    if not body:
        raise HTTPException(400, "Comment cannot be empty")
    if payload.kind not in ("comment", "note"):
        raise HTTPException(400, "kind must be 'comment' or 'note'")
    if (payload.target_type is None) != (payload.target_id is None):
        raise HTTPException(400, "target_type and target_id must be given together")
    if payload.target_type is not None:
        model = COMMENT_TARGETS.get(payload.target_type)
        if model is None:
            raise HTTPException(400, "Unknown target_type")
        target = db.get(model, payload.target_id)
        # Same 404 whether it doesn't exist or belongs to another team.
        if target is None or target.team_id != team.id:
            raise HTTPException(404, "Target not found")
    c = WorkspaceComment(team_id=team.id, author_id=user.id, target_type=payload.target_type,
                         target_id=payload.target_id, kind=payload.kind, body=body, edit_history=[])
    db.add(c)
    db.commit()
    db.refresh(c)
    return _comment_out(c, {user.id: user}, user.id, team_role(db, team, user.id))


@router.get("/discussion")
def list_comments(target_type: str | None = None, target_id: int | None = None, status: str | None = None,
                  limit: int = 100, team: Team = Depends(get_current_team),
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    q = db.query(WorkspaceComment).filter(WorkspaceComment.team_id == team.id,
                                          WorkspaceComment.deleted_at.is_(None))
    if target_type is not None:
        q = q.filter(WorkspaceComment.target_type == target_type)
    if target_id is not None:
        q = q.filter(WorkspaceComment.target_id == target_id)
    if status in ("open", "resolved"):
        q = q.filter(WorkspaceComment.status == status)
    rows = q.order_by(WorkspaceComment.id.desc()).limit(max(1, min(limit, 300))).all()
    ids = {c.author_id for c in rows}
    authors = {u.id: u for u in db.query(User).filter(User.id.in_(ids)).all()} if ids else {}
    role = team_role(db, team, user.id)
    return [_comment_out(c, authors, user.id, role) for c in rows]


@router.patch("/discussion/{comment_id}")
def update_comment(comment_id: int, payload: CommentUpdate, team: Team = Depends(get_current_team),
                   user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    c = _team_comment(db, team, comment_id)
    role = team_role(db, team, user.id)
    if payload.body is None and payload.status is None:
        raise HTTPException(400, "Nothing to update")
    if payload.body is not None:
        # Only the author edits their words; moderators can delete or resolve, not rewrite.
        if c.author_id != user.id:
            raise HTTPException(403, "Only the author can edit a comment")
        new_body = payload.body.strip()
        if not new_body:
            raise HTTPException(400, "Comment cannot be empty")
        if new_body != c.body:
            c.edit_history = list(c.edit_history or []) + [
                {"body": c.body, "edited_at": (c.edited_at or c.created_at).isoformat()}]
            c.body = new_body
            c.edited_at = utcnow()
    if payload.status is not None:
        if payload.status not in ("open", "resolved"):
            raise HTTPException(400, "status must be 'open' or 'resolved'")
        if not (c.author_id == user.id or can_manage_members(role)):
            raise HTTPException(403, "Only the author or a team leader/admin can change review status")
        c.status = payload.status
    db.commit()
    db.refresh(c)
    author = db.get(User, c.author_id)
    return _comment_out(c, {c.author_id: author} if author else {}, user.id, role)


@router.delete("/discussion/{comment_id}")
def delete_comment(comment_id: int, team: Team = Depends(get_current_team),
                   user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    c = _team_comment(db, team, comment_id)
    role = team_role(db, team, user.id)
    moderator = can_manage_members(role)
    if c.author_id != user.id and not moderator:
        raise HTTPException(403, "Only the author or a team leader/admin can delete a comment")
    c.deleted_at = utcnow()
    c.deleted_by = user.id
    if c.author_id != user.id:
        audit(db, team.community_id, user.id, "comment.deleted_by_moderator", team_id=team.id,
              target_type="comment", target_id=c.id, author_id=c.author_id)
    db.commit()
    return {"status": "deleted"}
