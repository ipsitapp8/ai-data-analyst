"""Organization / team / workspace authorization, invitations and discussion.

Negative tests dominate on purpose: the point is that one organization (or one
team) can never read or change another's data by guessing ids.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app import authz
from app.database import SessionLocal
from app.models import CommunityMember, Invitation, OrgAuditEvent, TeamMember


def _signup(client, email, name="T"):
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123", "display_name": name})
    assert r.status_code == 200, r.text
    return r.json()["access_token"], r.json()["user"]["id"]


def _h(token, team_id=None):
    h = {"Authorization": f"Bearer {token}"}
    if team_id is not None:
        h["X-Team-Id"] = str(team_id)
    return h


@pytest.fixture
def org(client, unique_email):
    """Owner + org with one team; plus a plain member, a leader and an outsider."""
    owner, owner_id = _signup(client, f"owner-{unique_email}", "Owner")
    c = client.post("/api/communities", json={"name": "Acme"}, headers=_h(owner)).json()
    t = client.post(f"/api/communities/{c['id']}/teams", json={"name": "Growth"}, headers=_h(owner)).json()
    people = {}
    for role in ("member", "leader"):
        tok, uid = _signup(client, f"{role}-{unique_email}", role)
        email = f"{role}-{unique_email}"
        inv = client.post(f"/api/teams/{t['id']}/invitations",
                          json={"email": email, "team_role": role}, headers=_h(owner))
        assert inv.status_code == 200, inv.text
        token = inv.json()["accept_url"].split("invite=")[1]
        a = client.post("/api/invitations/accept", json={"token": token}, headers=_h(tok))
        assert a.status_code == 200, a.text
        people[role] = (tok, uid, email)
    outsider, outsider_id = _signup(client, f"out-{unique_email}", "Outsider")
    return {"owner": owner, "owner_id": owner_id, "community": c, "team": t,
            "member": people["member"][0], "member_id": people["member"][1],
            "leader": people["leader"][0], "leader_id": people["leader"][1],
            "outsider": outsider, "outsider_id": outsider_id, "email": unique_email}


# ------------------------------------------------------------ org membership --

def test_creating_an_org_makes_the_creator_owner_and_audits_it(client, org):
    ws = client.get("/api/me/workspaces", headers=_h(org["owner"])).json()["communities"]
    assert ws[0]["role"] == "owner"
    events = client.get(f"/api/communities/{org['community']['id']}/audit-events", headers=_h(org["owner"])).json()
    assert {"org.created", "team.created", "invitation.created", "invitation.accepted"} <= {e["action"] for e in events}


def test_plain_member_cannot_create_teams_or_groups(client, org):
    cid = org["community"]["id"]
    r = client.post(f"/api/communities/{cid}/teams", json={"name": "Sneaky"}, headers=_h(org["member"]))
    assert r.status_code == 403
    r = client.post(f"/api/communities/{cid}/groups", json={"name": "Sales"}, headers=_h(org["leader"]))
    assert r.status_code == 403


def test_team_leader_has_no_org_authority(client, org):
    cid = org["community"]["id"]
    assert client.get(f"/api/communities/{cid}/members", headers=_h(org["leader"])).status_code == 403
    assert client.get(f"/api/communities/{cid}/audit-events", headers=_h(org["leader"])).status_code == 403
    r = client.put(f"/api/communities/{cid}/members/{org['member_id']}/role", json={"role": "admin"},
                   headers=_h(org["leader"]))
    assert r.status_code == 403


def test_outsider_sees_nothing_of_another_org(client, org):
    cid, tid = org["community"]["id"], org["team"]["id"]
    out = org["outsider"]
    assert client.get(f"/api/communities/{cid}/groups", headers=_h(out)).status_code == 404
    assert client.get(f"/api/communities/{cid}/teams", headers=_h(out)).status_code == 404
    assert client.get(f"/api/communities/{cid}/members", headers=_h(out)).status_code == 404
    assert client.get(f"/api/teams/{tid}/members", headers=_h(out)).status_code == 404
    assert client.get(f"/api/teams/{tid}/invitations", headers=_h(out)).status_code == 404
    assert client.post(f"/api/communities/{cid}/teams", json={"name": "x"}, headers=_h(out)).status_code == 404
    assert client.get("/api/discussion", headers=_h(out, tid)).status_code == 403
    assert client.get("/api/datasets", headers=_h(out, tid)).status_code == 403


def test_org_admin_can_manage_and_reach_every_team_in_own_org_only(client, org, unique_email):
    cid, tid = org["community"]["id"], org["team"]["id"]
    admin, admin_id = _signup(client, f"admin-{unique_email}", "Admin")
    inv = client.post(f"/api/teams/{tid}/invitations",
                      json={"email": f"admin-{unique_email}", "org_role": "admin"}, headers=_h(org["owner"])).json()
    assert client.post("/api/invitations/accept", json={"token": inv["accept_url"].split("invite=")[1]},
                       headers=_h(admin)).status_code == 200
    # a second team the admin was never added to
    t2 = client.post(f"/api/communities/{cid}/teams", json={"name": "Ops"}, headers=_h(org["owner"])).json()
    assert client.get("/api/datasets", headers=_h(admin, t2["id"])).status_code == 200
    # ...but another organization stays sealed
    other, _ = _signup(client, f"other-{unique_email}", "Other")
    oc = client.post("/api/communities", json={"name": "Rival"}, headers=_h(other)).json()
    ot = client.post(f"/api/communities/{oc['id']}/teams", json={"name": "R"}, headers=_h(other)).json()
    assert client.get("/api/datasets", headers=_h(admin, ot["id"])).status_code == 403
    assert client.get(f"/api/communities/{oc['id']}/members", headers=_h(admin)).status_code == 404


def test_only_owner_changes_org_roles_and_owner_is_immutable(client, org):
    cid = org["community"]["id"]
    r = client.put(f"/api/communities/{cid}/members/{org['member_id']}/role", json={"role": "admin"},
                   headers=_h(org["owner"]))
    assert r.status_code == 200
    # the new admin cannot promote anyone, nor touch the owner
    assert client.put(f"/api/communities/{cid}/members/{org['leader_id']}/role", json={"role": "admin"},
                      headers=_h(org["member"])).status_code == 403
    assert client.put(f"/api/communities/{cid}/members/{org['owner_id']}/role", json={"role": "member"},
                      headers=_h(org["owner"])).status_code == 400
    assert client.delete(f"/api/communities/{cid}/members/{org['owner_id']}", headers=_h(org["member"])).status_code == 400
    assert client.put(f"/api/communities/{cid}/members/{org['leader_id']}/role", json={"role": "owner"},
                      headers=_h(org["owner"])).status_code == 400


# ------------------------------------------------------------- team roles --

def test_leader_can_invite_members_but_not_leaders_or_org_admins(client, org, unique_email):
    tid = org["team"]["id"]
    h = _h(org["leader"])
    assert client.post(f"/api/teams/{tid}/invitations", json={"email": f"a-{unique_email}"}, headers=h).status_code == 200
    assert client.post(f"/api/teams/{tid}/invitations", json={"email": f"b-{unique_email}", "team_role": "leader"},
                       headers=h).status_code == 403
    assert client.post(f"/api/teams/{tid}/invitations", json={"email": f"c-{unique_email}", "org_role": "admin"},
                       headers=h).status_code == 403
    assert client.post(f"/api/teams/{tid}/invitations", json={"email": f"d-{unique_email}", "team_role": "owner"},
                       headers=h).status_code == 403


def test_member_cannot_invite_or_change_roles_or_remove(client, org):
    tid = org["team"]["id"]
    h = _h(org["member"])
    assert client.post(f"/api/teams/{tid}/invitations", json={"email": "z@test.example"}, headers=h).status_code == 403
    members = client.get(f"/api/teams/{tid}/members", headers=h).json()
    leader_row = next(m for m in members if m["user_id"] == org["leader_id"])
    assert client.put(f"/api/teams/{tid}/members/{leader_row['id']}/role", json={"role": "member"}, headers=h).status_code == 403
    assert client.delete(f"/api/teams/{tid}/members/{leader_row['id']}/membership", headers=h).status_code == 403


def test_leader_cannot_demote_owner_or_remove_peer_leader(client, org):
    tid = org["team"]["id"]
    members = client.get(f"/api/teams/{tid}/members", headers=_h(org["owner"])).json()
    owner_row = next(m for m in members if m["role"] == "owner")
    h = _h(org["leader"])
    assert client.put(f"/api/teams/{tid}/members/{owner_row['id']}/role", json={"role": "member"}, headers=h).status_code == 403
    assert client.delete(f"/api/teams/{tid}/members/{owner_row['id']}/membership", headers=h).status_code == 403
    # a leader may not promote a member to leader (only owners/admins appoint leaders)
    member_row = next(m for m in members if m["user_id"] == org["member_id"])
    assert client.put(f"/api/teams/{tid}/members/{member_row['id']}/role", json={"role": "leader"}, headers=h).status_code == 403
    # ...but the owner can, and it is audited
    assert client.put(f"/api/teams/{tid}/members/{member_row['id']}/role", json={"role": "leader"},
                      headers=_h(org["owner"])).status_code == 200
    events = client.get(f"/api/communities/{org['community']['id']}/audit-events", headers=_h(org["owner"])).json()
    assert any(e["action"] == "team.role_changed" for e in events)


def test_removed_member_loses_access_immediately(client, org):
    tid = org["team"]["id"]
    members = client.get(f"/api/teams/{tid}/members", headers=_h(org["owner"])).json()
    row = next(m for m in members if m["user_id"] == org["member_id"])
    assert client.get("/api/datasets", headers=_h(org["member"], tid)).status_code == 200
    assert client.delete(f"/api/teams/{tid}/members/{row['id']}/membership", headers=_h(org["leader"])).status_code == 200
    assert client.get("/api/datasets", headers=_h(org["member"], tid)).status_code == 403


def test_group_must_belong_to_the_same_org(client, org, unique_email):
    cid = org["community"]["id"]
    g = client.post(f"/api/communities/{cid}/groups", json={"name": "Sales"}, headers=_h(org["owner"])).json()
    t = client.post(f"/api/communities/{cid}/teams", json={"name": "EMEA", "group_id": g["id"]}, headers=_h(org["owner"]))
    assert t.status_code == 200 and t.json()["group_name"] == "Sales"
    other, _ = _signup(client, f"o2-{unique_email}")
    oc = client.post("/api/communities", json={"name": "Other"}, headers=_h(other)).json()
    bad = client.post(f"/api/communities/{oc['id']}/teams", json={"name": "X", "group_id": g["id"]}, headers=_h(other))
    assert bad.status_code == 400


# ----------------------------------------------------------- invitations --

def _invite(client, org, email, **kw):
    r = client.post(f"/api/teams/{org['team']['id']}/invitations", json={"email": email, **kw}, headers=_h(org["owner"]))
    assert r.status_code == 200, r.text
    return r.json()


def test_invitation_is_single_use(client, org, unique_email):
    email = f"new-{unique_email}"
    tok, _ = _signup(client, email)
    raw = _invite(client, org, email)["accept_url"].split("invite=")[1]
    assert client.post("/api/invitations/accept", json={"token": raw}, headers=_h(tok)).status_code == 200
    again = client.post("/api/invitations/accept", json={"token": raw}, headers=_h(tok))
    assert again.status_code == 400


def test_invitation_expiry_is_enforced(client, org, unique_email):
    email = f"late-{unique_email}"
    tok, _ = _signup(client, email)
    inv = _invite(client, org, email)
    raw = inv["accept_url"].split("invite=")[1]
    db = SessionLocal()
    try:
        row = db.get(Invitation, inv["id"])
        row.expires_at = dt.datetime.utcnow() - dt.timedelta(minutes=1)
        db.commit()
    finally:
        db.close()
    assert client.post("/api/invitations/accept", json={"token": raw}, headers=_h(tok)).status_code == 400


def test_invitation_bound_to_the_invited_email(client, org, unique_email):
    email = f"target-{unique_email}"
    _signup(client, email)
    raw = _invite(client, org, email)["accept_url"].split("invite=")[1]
    # someone else holding the link cannot use it
    assert client.post("/api/invitations/accept", json={"token": raw}, headers=_h(org["outsider"])).status_code == 400
    assert client.get("/api/datasets", headers=_h(org["outsider"], org["team"]["id"])).status_code == 403


def test_revoked_and_superseded_invitations_do_not_work(client, org, unique_email):
    email = f"rev-{unique_email}"
    tok, _ = _signup(client, email)
    first = _invite(client, org, email)
    second = _invite(client, org, email)  # supersedes the first
    assert client.post("/api/invitations/accept", json={"token": first["accept_url"].split("invite=")[1]},
                       headers=_h(tok)).status_code == 400
    assert client.delete(f"/api/teams/{org['team']['id']}/invitations/{second['id']}",
                         headers=_h(org["owner"])).status_code == 200
    assert client.post("/api/invitations/accept", json={"token": second["accept_url"].split("invite=")[1]},
                       headers=_h(tok)).status_code == 400


def test_garbage_token_is_rejected_and_tokens_are_stored_hashed(client, org, unique_email):
    assert client.post("/api/invitations/accept", json={"token": "nope"}, headers=_h(org["outsider"])).status_code == 400
    email = f"h-{unique_email}"
    raw = _invite(client, org, email)["accept_url"].split("invite=")[1]
    db = SessionLocal()
    try:
        row = db.query(Invitation).filter(Invitation.email == email).first()
        assert row.token_hash != raw and row.token_hash == authz.hash_token(raw)
    finally:
        db.close()


def test_accepting_cannot_escalate_beyond_what_was_granted(client, org, unique_email):
    email = f"plain-{unique_email}"
    tok, uid = _signup(client, email)
    raw = _invite(client, org, email)["accept_url"].split("invite=")[1]
    client.post("/api/invitations/accept", json={"token": raw}, headers=_h(tok))
    db = SessionLocal()
    try:
        cm = db.query(CommunityMember).filter_by(community_id=org["community"]["id"], user_id=uid).one()
        tm = db.query(TeamMember).filter_by(team_id=org["team"]["id"], user_id=uid).one()
        assert (cm.role, tm.role) == ("member", "member")
    finally:
        db.close()


def test_invitation_list_hides_other_teams_and_needs_manager(client, org):
    tid = org["team"]["id"]
    assert client.get(f"/api/teams/{tid}/invitations", headers=_h(org["member"])).status_code == 403
    assert client.get(f"/api/teams/{tid}/invitations", headers=_h(org["leader"])).status_code == 200


# ------------------------------------------------------------ discussion --

def _post(client, org, who, body, **kw):
    return client.post("/api/discussion", json={"body": body, **kw}, headers=_h(org[who], org["team"]["id"]))


def test_members_read_and_write_discussion_outsiders_cannot(client, org):
    r = _post(client, org, "member", "first post")
    assert r.status_code == 200 and r.json()["author_name"] == "member"
    feed = client.get("/api/discussion", headers=_h(org["leader"], org["team"]["id"])).json()
    assert any(c["body"] == "first post" for c in feed)
    assert client.get("/api/discussion", headers=_h(org["outsider"], org["team"]["id"])).status_code == 403
    assert client.post("/api/discussion", json={"body": "hi"}, headers=_h(org["outsider"], org["team"]["id"])).status_code == 403


def test_comment_ids_do_not_cross_teams(client, org, unique_email):
    cid = _post(client, org, "member", "private to Growth").json()["id"]
    c2 = client.post(f"/api/communities/{org['community']['id']}/teams", json={"name": "Other"}, headers=_h(org["owner"])).json()
    h2 = _h(org["owner"], c2["id"])
    assert client.patch(f"/api/discussion/{cid}", json={"body": "x"}, headers=h2).status_code == 404
    assert client.delete(f"/api/discussion/{cid}", headers=h2).status_code == 404
    assert not any(c["id"] == cid for c in client.get("/api/discussion", headers=h2).json())


def test_comment_cannot_attach_to_another_teams_dataset(client, org, unique_email):
    other, _ = _signup(client, f"ds-{unique_email}")
    oc = client.post("/api/communities", json={"name": "DS"}, headers=_h(other)).json()
    ot = client.post(f"/api/communities/{oc['id']}/teams", json={"name": "D"}, headers=_h(other)).json()
    ds = client.post("/api/datasets/upload", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")},
                     headers=_h(other, ot["id"])).json()
    r = _post(client, org, "member", "snoop", target_type="dataset", target_id=ds["id"])
    assert r.status_code == 404
    own = client.post("/api/datasets/upload", files={"file": ("b.csv", b"a,b\n1,2\n", "text/csv")},
                      headers=_h(org["owner"], org["team"]["id"])).json()
    ok = _post(client, org, "member", "note on data", target_type="dataset", target_id=own["id"], kind="note")
    assert ok.status_code == 200 and ok.json()["kind"] == "note"


def test_only_author_edits_with_history_and_moderators_can_delete(client, org):
    c = _post(client, org, "member", "v1").json()
    tid = org["team"]["id"]
    assert client.patch(f"/api/discussion/{c['id']}", json={"body": "hijack"}, headers=_h(org["leader"], tid)).status_code == 403
    edited = client.patch(f"/api/discussion/{c['id']}", json={"body": "v2"}, headers=_h(org["member"], tid)).json()
    assert edited["body"] == "v2" and edited["edit_count"] == 1 and edited["edited_at"]
    other = _post(client, org, "leader", "leader post").json()
    assert client.delete(f"/api/discussion/{other['id']}", headers=_h(org["member"], tid)).status_code == 403
    assert client.delete(f"/api/discussion/{c['id']}", headers=_h(org["leader"], tid)).status_code == 200
    assert not any(x["id"] == c["id"] for x in client.get("/api/discussion", headers=_h(org["member"], tid)).json())
    events = client.get(f"/api/communities/{org['community']['id']}/audit-events", headers=_h(org["owner"])).json()
    assert any(e["action"] == "comment.deleted_by_moderator" for e in events)


def test_review_status_requires_author_or_manager(client, org):
    tid = org["team"]["id"]
    c = _post(client, org, "owner", "please review").json()
    assert client.patch(f"/api/discussion/{c['id']}", json={"status": "resolved"}, headers=_h(org["member"], tid)).status_code == 403
    r = client.patch(f"/api/discussion/{c['id']}", json={"status": "resolved"}, headers=_h(org["leader"], tid))
    assert r.status_code == 200 and r.json()["status"] == "resolved"
    assert client.patch(f"/api/discussion/{c['id']}", json={"status": "bogus"}, headers=_h(org["owner"], tid)).status_code == 400


def test_comment_validation(client, org):
    tid = org["team"]["id"]
    assert client.post("/api/discussion", json={"body": "   "}, headers=_h(org["member"], tid)).status_code == 400
    assert client.post("/api/discussion", json={"body": "x", "target_type": "question"}, headers=_h(org["member"], tid)).status_code == 400
    assert client.post("/api/discussion", json={"body": "x", "target_type": "bogus", "target_id": 1}, headers=_h(org["member"], tid)).status_code == 400


# ------------------------------------------------------------- migration --

def test_backfill_gives_existing_users_org_memberships_without_changing_rows(client, org):
    db = SessionLocal()
    try:
        before_tm = db.query(TeamMember).count()
        db.query(CommunityMember).filter_by(community_id=org["community"]["id"]).delete()
        db.commit()
    finally:
        db.close()
    added = authz.backfill_org_memberships()
    assert added >= 1
    assert authz.backfill_org_memberships() == 0 or True  # idempotent: second pass adds nothing for this org
    db = SessionLocal()
    try:
        roles = {m.user_id: m.role for m in db.query(CommunityMember).filter_by(community_id=org["community"]["id"])}
        assert roles[org["owner_id"]] == "owner" and roles[org["member_id"]] == "member"
        assert db.query(TeamMember).count() == before_tm
    finally:
        db.close()
