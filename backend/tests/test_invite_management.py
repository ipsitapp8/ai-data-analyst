"""Invite revocation, resend, and bulk invite -- the follow-up gap after the
first invite feature shipped with no way to undo or batch an invite."""
from __future__ import annotations


def _owner_with_team(client, unique_email, team_name="Team"):
    r = client.post(
        "/api/auth/signup",
        json={"email": f"owner-{unique_email}", "password": "password123", "display_name": "Owner"},
    )
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    c = client.post("/api/communities", json={"name": "Invite Co"}, headers=headers)
    team_id = client.post(
        f"/api/communities/{c.json()['id']}/teams", json={"name": team_name}, headers=headers
    ).json()["id"]
    return headers, team_id


def test_second_invite_to_pending_email_resends_instead_of_conflicting(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)
    invitee_email = f"invitee-{unique_email}"

    r1 = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    assert r1.status_code == 200, r1.text
    assert r1.json()["resent"] is False
    member_id = r1.json()["member"]["id"]

    r2 = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    assert r2.status_code == 200, r2.text
    assert r2.json()["resent"] is True
    assert r2.json()["member"]["id"] == member_id  # same row, not a duplicate


def test_invite_to_already_active_member_still_conflicts(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)
    invitee_email = f"invitee-{unique_email}"

    client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    invitee_token = _signup_and_get_token(client, invitee_email)
    client.post(
        f"/api/teams/{team_id}/accept-invite", headers={"Authorization": f"Bearer {invitee_token}"}
    )

    r = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    assert r.status_code == 409


def _signup_and_get_token(client, email):
    r = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "password123", "display_name": "Invitee"},
    )
    return r.json()["access_token"]


def test_revoke_pending_invite(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)
    invitee_email = f"invitee-{unique_email}"

    r = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    member_id = r.json()["member"]["id"]

    r = client.delete(f"/api/teams/{team_id}/members/{member_id}", headers=headers)
    assert r.status_code == 200, r.text

    members = client.get(f"/api/teams/{team_id}/members", headers=headers).json()
    assert all(m["id"] != member_id for m in members)

    # Revoked -- inviting the same address again should create a fresh
    # pending row, not resend a deleted one.
    r2 = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["resent"] is False


def test_revoke_active_member_rejected(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)
    invitee_email = f"invitee-{unique_email}"

    r = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    member_id = r.json()["member"]["id"]
    invitee_token = _signup_and_get_token(client, invitee_email)
    client.post(
        f"/api/teams/{team_id}/accept-invite", headers={"Authorization": f"Bearer {invitee_token}"}
    )

    r = client.delete(f"/api/teams/{team_id}/members/{member_id}", headers=headers)
    assert r.status_code == 400


def test_non_admin_cannot_revoke(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)
    invitee_email = f"invitee-{unique_email}"
    r = client.post(f"/api/teams/{team_id}/invite", json={"email": invitee_email}, headers=headers)
    member_id = r.json()["member"]["id"]

    member_token = _signup_and_get_token(client, invitee_email)
    client.post(
        f"/api/teams/{team_id}/accept-invite", headers={"Authorization": f"Bearer {member_token}"}
    )
    # invite a second person, then have the (non-admin) first member try to revoke it
    other_email = f"other-{unique_email}"
    r2 = client.post(f"/api/teams/{team_id}/invite", json={"email": other_email}, headers=headers)
    other_id = r2.json()["member"]["id"]

    r3 = client.delete(
        f"/api/teams/{team_id}/members/{other_id}", headers={"Authorization": f"Bearer {member_token}"}
    )
    assert r3.status_code == 403


def test_bulk_invite_mixed_results(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)
    good_email = f"good-{unique_email}"
    bad_email = "not-an-email"

    r = client.post(
        f"/api/teams/{team_id}/invite/bulk",
        json={"emails": [good_email, bad_email, good_email]},  # duplicate collapses
        headers=headers,
    )
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 2  # deduped

    by_email = {res["email"]: res for res in results}
    assert by_email[good_email]["ok"] is True
    assert by_email[bad_email]["ok"] is False


def test_bulk_invite_rejects_empty_and_oversized_batches(client, unique_email):
    headers, team_id = _owner_with_team(client, unique_email)

    r = client.post(f"/api/teams/{team_id}/invite/bulk", json={"emails": []}, headers=headers)
    assert r.status_code == 400

    many = [f"person{i}-{unique_email}" for i in range(101)]
    r = client.post(f"/api/teams/{team_id}/invite/bulk", json={"emails": many}, headers=headers)
    assert r.status_code == 400
