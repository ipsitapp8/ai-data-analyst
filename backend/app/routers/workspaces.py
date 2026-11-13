"""Community/Team management: create/list, invite-by-email, accept-invite,
and the one-call nested listing the sidebar workspace switcher uses.

Visibility is deliberately member-only throughout -- a community's teams (and
a team's existence at all) are only visible to people who already belong to
it, not to every member of the parent community. That's a stricter default
than "any community member can browse all its teams", chosen so isolation
holds even at the level of "does this team exist" -- not just its data.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.email_sender import send_team_invite_email
from app import authz
from app.authz import (
    audit,
    can_manage_members,
    ensure_org_member,
    is_org_admin,
    org_role,
    require_org_admin,
    team_role,
)
from app.models import Community, CommunityMember, Group, Team, TeamMember, User
from app.schemas import (
    BulkInviteRequest,
    BulkInviteResponse,
    BulkInviteResult,
    CommunityCreate,
    CommunityOut,
    InviteRequest,
    InviteResponse,
    MemberOut,
    TeamCreate,
    TeamOut,
)
from app.security import get_current_user
from app.validation import is_valid_email

router = APIRouter(prefix="/api", tags=["workspaces"])


def _my_role(db: Session, team_id: int, user_id: int) -> str | None:
    team = db.get(Team, team_id)
    return team_role(db, team, user_id) if team else None


def _team_out(db: Session, t: Team, role: str) -> TeamOut:
    group = db.get(Group, t.group_id) if t.group_id else None
    return TeamOut(id=t.id, community_id=t.community_id, name=t.name, role=role,
                   group_id=t.group_id, group_name=group.name if group else None)


def _my_teams_in_community(db: Session, community_id: int, user_id: int) -> list[TeamOut]:
    """Teams the caller can see: their own memberships, plus every team in the
    organization if they are an organization owner/admin."""
    explicit = {
        t.id: (t, tm.role)
        for t, tm in (
            db.query(Team, TeamMember)
            .join(TeamMember, TeamMember.team_id == Team.id)
            .filter(Team.community_id == community_id, TeamMember.user_id == user_id,
                    TeamMember.status == "active")
            .all()
        )
    }
    if is_org_admin(db, community_id, user_id):
        for t in db.query(Team).filter(Team.community_id == community_id).all():
            explicit.setdefault(t.id, (t, "admin"))
    return [_team_out(db, t, role) for t, role in sorted(explicit.values(), key=lambda x: x[0].id)]


def _my_community_ids(db: Session, user_id: int) -> set[int]:
    ids = {
        row[0] for row in db.query(CommunityMember.community_id)
        .filter(CommunityMember.user_id == user_id).all()
    }
    ids |= {
        row[0] for row in db.query(Team.community_id)
        .join(TeamMember, TeamMember.team_id == Team.id)
        .filter(TeamMember.user_id == user_id, TeamMember.status == "active").all()
    }
    return ids


def _community_out(db: Session, c: Community, user_id: int) -> CommunityOut:
    return CommunityOut(id=c.id, name=c.name, role=org_role(db, c.id, user_id),
                        teams=_my_teams_in_community(db, c.id, user_id))


@router.get("/communities", response_model=list[CommunityOut])
def list_my_communities(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ids = _my_community_ids(db, user.id)
    created = db.query(Community.id).filter(Community.created_by == user.id).all()
    ids |= {row[0] for row in created}
    communities = db.query(Community).filter(Community.id.in_(ids)).all() if ids else []
    return [_community_out(db, c, user.id) for c in communities]


@router.post("/communities", response_model=CommunityOut)
def create_community(payload: CommunityCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Community name is required")
    community = Community(name=name, created_by=user.id)
    db.add(community)
    db.flush()
    ensure_org_member(db, community.id, user.id, "owner")
    audit(db, community.id, user.id, "org.created", target_type="community", target_id=community.id, name=name)
    db.commit()
    db.refresh(community)
    return CommunityOut(id=community.id, name=community.name, role="owner", teams=[])


@router.get("/communities/{community_id}/teams", response_model=list[TeamOut])
def list_teams(community_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    authz.get_org_or_404(db, community_id, user.id)
    return _my_teams_in_community(db, community_id, user.id)


@router.post("/communities/{community_id}/teams", response_model=TeamOut)
def create_team(community_id: int, payload: TeamCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    authz.get_org_or_404(db, community_id, user.id)
    require_org_admin(db, community_id, user.id, "create teams")
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Team name is required")
    group = None
    if payload.group_id is not None:
        group = db.get(Group, payload.group_id)
        if group is None or group.community_id != community_id:
            raise HTTPException(400, "Unknown group for this organization")

    team = Team(community_id=community_id, name=name, created_by=user.id,
                group_id=group.id if group else None)
    db.add(team)
    db.flush()
    db.add(TeamMember(team_id=team.id, user_id=user.id, role="owner", status="active"))
    audit(db, community_id, user.id, "team.created", team_id=team.id, target_type="team",
          target_id=team.id, name=name, group_id=group.id if group else None)
    db.commit()
    db.refresh(team)
    return _team_out(db, team, "owner")


@router.get("/me/workspaces")
def my_workspaces(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ids = _my_community_ids(db, user.id)
    communities = db.query(Community).filter(Community.id.in_(ids)).all() if ids else []
    return {"communities": [_community_out(db, c, user.id) for c in communities]}


@router.get("/me/invites")
def my_pending_invites(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(TeamMember, Team, Community)
        .join(Team, Team.id == TeamMember.team_id)
        .join(Community, Community.id == Team.community_id)
        .filter(TeamMember.user_id == user.id, TeamMember.status == "pending")
        .all()
    )
    return [
        {"team_id": t.id, "team_name": t.name, "community_name": c.name, "role": tm.role}
        for tm, t, c in rows
    ]


@router.post("/teams/{team_id}/accept-invite")
def accept_invite(team_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    invite = (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team_id, TeamMember.user_id == user.id, TeamMember.status == "pending")
        .first()
    )
    if invite is None:
        raise HTTPException(404, "No pending invite for this team")
    team = db.get(Team, team_id)
    invite.status = "active"
    ensure_org_member(db, team.community_id, user.id, "member")
    audit(db, team.community_id, user.id, "team.invite_accepted", team_id=team.id,
          target_type="user", target_id=user.id, via="direct_invite")
    db.commit()
    return {"status": "joined"}


def _invite_or_resend(db: Session, team: Team, community: Community | None, user: User, raw_email: str) -> InviteResponse:
    """Shared by the single and bulk invite endpoints. A second invite to an
    email already `pending` for this team is treated as a resend (refreshes
    and re-emails the same row) rather than a 409 -- that's what makes
    re-invite work without a separate endpoint. An email already `active` on
    the team still raises, since there's nothing useful to resend there."""
    email = raw_email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(400, "That doesn't look like a valid email address")

    existing_user = db.query(User).filter(User.email == email).first()
    already = db.query(TeamMember).filter(
        TeamMember.team_id == team.id,
        (TeamMember.user_id == existing_user.id) if existing_user else (TeamMember.invited_email == email),
    ).first()

    resent = False
    if already:
        if already.status == "active":
            raise HTTPException(409, "This person is already a member of this team")
        # status == "pending" -> resend: reuse the row, just re-send the email.
        invite = already
        resent = True
    else:
        invite = TeamMember(
            team_id=team.id,
            user_id=existing_user.id if existing_user else None,
            invited_email=email,
            role="member",
            status="pending",
        )
        db.add(invite)
        db.commit()
        db.refresh(invite)

    # Best-effort: send_team_invite_email logs and returns False on any
    # failure (including SMTP not being configured at all) rather than
    # raising, so a broken/missing mail setup never breaks invite creation --
    # the pending membership row above is already committed regardless.
    # The frontend surfaces email_sent so "was an email actually sent" is
    # never silently unclear the way it was before this field existed.
    email_sent = send_team_invite_email(
        to_email=email,
        inviter_name=user.display_name,
        community_name=community.name if community else "",
        team_name=team.name,
    )

    return InviteResponse(
        member=MemberOut(
            id=invite.id,
            user_id=invite.user_id,
            email=email,
            display_name=existing_user.display_name if existing_user else None,
            role=invite.role,
            status=invite.status,
        ),
        email_sent=email_sent,
        resent=resent,
    )


@router.post("/teams/{team_id}/invite", response_model=InviteResponse)
def invite_member(team_id: int, payload: InviteRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    role = _my_role(db, team_id, user.id)
    if not can_manage_members(role):
        raise HTTPException(403, "Only a team leader, owner or admin can invite members")

    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(404, "Team not found")
    community = db.get(Community, team.community_id)

    return _invite_or_resend(db, team, community, user, payload.email)


@router.post("/teams/{team_id}/invite/bulk", response_model=BulkInviteResponse)
def invite_members_bulk(team_id: int, payload: BulkInviteRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    role = _my_role(db, team_id, user.id)
    if not can_manage_members(role):
        raise HTTPException(403, "Only a team leader, owner or admin can invite members")

    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(404, "Team not found")
    community = db.get(Community, team.community_id)

    # Dedupe case-insensitively while preserving the order the user typed
    # them in, and cap the batch so one paste can't hammer SMTP or the DB.
    seen: set[str] = set()
    emails: list[str] = []
    for raw in payload.emails:
        e = raw.strip().lower()
        if e and e not in seen:
            seen.add(e)
            emails.append(e)
    if not emails:
        raise HTTPException(400, "No email addresses given")
    if len(emails) > 100:
        raise HTTPException(400, "Too many addresses in one batch (max 100) -- split into smaller batches")

    results: list[BulkInviteResult] = []
    for raw_email in emails:
        try:
            resp = _invite_or_resend(db, team, community, user, raw_email)
            action = "resent" if resp.resent else "invited"
            message = f"{action} -- " + ("email sent" if resp.email_sent else "email not sent (link only)")
            results.append(BulkInviteResult(email=resp.member.email, ok=True, message=message, email_sent=resp.email_sent))
        except HTTPException as e:
            results.append(BulkInviteResult(email=raw_email.strip().lower(), ok=False, message=str(e.detail)))

    return BulkInviteResponse(results=results)


@router.delete("/teams/{team_id}/members/{member_id}")
def revoke_invite(team_id: int, member_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Cancel a pending invite. Deliberately scoped to pending rows only --
    removing an already-active member is a different, higher-stakes action
    (loses them their history/access) that wasn't asked for here."""
    role = _my_role(db, team_id, user.id)
    if not can_manage_members(role):
        raise HTTPException(403, "Only a team leader, owner or admin can revoke invites")

    member = db.query(TeamMember).filter(TeamMember.id == member_id, TeamMember.team_id == team_id).first()
    if member is None:
        raise HTTPException(404, "Invite not found")
    if member.status != "pending":
        raise HTTPException(400, "Only a pending invite can be revoked")

    db.delete(member)
    db.commit()
    return {"status": "revoked"}


@router.get("/teams/{team_id}/members", response_model=list[MemberOut])
def list_members(team_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if _my_role(db, team_id, user.id) is None:
        raise HTTPException(404, "Team not found")

    rows = db.query(TeamMember).filter(TeamMember.team_id == team_id).all()
    out = []
    for m in rows:
        member_user = db.get(User, m.user_id) if m.user_id else None
        out.append(MemberOut(
            id=m.id,
            user_id=m.user_id,
            email=member_user.email if member_user else (m.invited_email or ""),
            display_name=member_user.display_name if member_user else None,
            role=m.role,
            status=m.status,
        ))
    return out
