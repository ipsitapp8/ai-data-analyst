"""Cross-team data isolation -- the single most important correctness
requirement in the whole multi-tenancy feature. Every one of these failing
would mean one team can see or touch another team's data.
"""
from __future__ import annotations


def _signup(client, email: str, password: str = "password123", name: str = "Test") -> str:
    r = client.post("/api/auth/signup", json={"email": email, "password": password, "display_name": name})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _create_team(client, token: str, community_name: str, team_name: str) -> int:
    headers = {"Authorization": f"Bearer {token}"}
    r = client.post("/api/communities", json={"name": community_name}, headers=headers)
    assert r.status_code == 200, r.text
    community_id = r.json()["id"]
    r = client.post(f"/api/communities/{community_id}/teams", json={"name": team_name}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_dataset_upload_and_listing_scoped_to_own_team(client, unique_email):
    alice = _signup(client, f"alice-{unique_email}")
    team_a = _create_team(client, alice, "A Co", "Team A")
    headers = {"Authorization": f"Bearer {alice}", "X-Team-Id": str(team_a)}

    files = {"file": ("t.csv", b"a,b\n1,2\n3,4\n", "text/csv")}
    r = client.post("/api/datasets/upload", files=files, headers=headers)
    assert r.status_code == 200, r.text
    dataset_id = r.json()["id"]

    r = client.get("/api/datasets", headers=headers)
    assert r.status_code == 200
    assert any(d["id"] == dataset_id for d in r.json())


def test_cross_team_dataset_access_is_404(client, unique_email):
    alice = _signup(client, f"alice-{unique_email}")
    bob = _signup(client, f"bob-{unique_email}")
    team_a = _create_team(client, alice, "A Co", "Team A")
    team_b = _create_team(client, bob, "B Co", "Team B")

    files = {"file": ("t.csv", b"a,b\n1,2\n", "text/csv")}
    r = client.post(
        "/api/datasets/upload", files=files,
        headers={"Authorization": f"Bearer {alice}", "X-Team-Id": str(team_a)},
    )
    dataset_id = r.json()["id"]

    bob_headers = {"Authorization": f"Bearer {bob}", "X-Team-Id": str(team_b)}

    # Bob's own dataset list never contains Alice's dataset.
    r = client.get("/api/datasets", headers=bob_headers)
    assert r.status_code == 200
    assert all(d["id"] != dataset_id for d in r.json())

    # Guessing Alice's dataset id directly under Bob's own (valid) team
    # header still 404s -- the isolation is enforced on the object, not
    # just by hiding it from list views.
    r = client.get(f"/api/datasets/{dataset_id}", headers=bob_headers)
    assert r.status_code == 404


def test_unauthorized_team_header_is_403(client, unique_email):
    alice = _signup(client, f"alice-{unique_email}")
    bob = _signup(client, f"bob-{unique_email}")
    team_a = _create_team(client, alice, "A Co", "Team A")
    _create_team(client, bob, "B Co", "Team B")

    # Bob has no membership in team_a at all -- using its id as his active
    # team must be rejected outright, not just filtered to empty results.
    r = client.get("/api/datasets", headers={"Authorization": f"Bearer {bob}", "X-Team-Id": str(team_a)})
    assert r.status_code == 403


def test_no_token_is_401_even_with_a_team_header(client):
    r = client.get("/api/datasets", headers={"X-Team-Id": "1"})
    assert r.status_code == 401


def test_invite_of_unregistered_email_auto_activates_on_signup(client, unique_email):
    """Inviting someone who has no account yet: the invite sits pending
    (no user_id to attach to yet), but signing up with that exact email is
    itself the acceptance -- no separate accept-invite call needed."""
    owner = _signup(client, f"owner-{unique_email}")
    team_id = _create_team(client, owner, "Invite Co", "Team")
    invitee_email = f"invitee-{unique_email}"

    r = client.post(
        f"/api/teams/{team_id}/invite", json={"email": invitee_email},
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["member"]["status"] == "pending"
    assert r.json()["email_sent"] is False  # no SMTP configured in tests

    invitee = _signup(client, invitee_email)
    invitee_headers = {"Authorization": f"Bearer {invitee}", "X-Team-Id": str(team_id)}

    # Signup already claimed and activated it -- immediate access, and
    # nothing left pending.
    r = client.get("/api/datasets", headers=invitee_headers)
    assert r.status_code == 200

    r = client.get("/api/me/invites", headers={"Authorization": f"Bearer {invitee}"})
    assert r.status_code == 200
    assert all(i["team_id"] != team_id for i in r.json())


def test_invite_of_existing_user_requires_explicit_accept(client, unique_email):
    """Inviting someone who already has an account: the invite is attached
    to their user_id immediately but stays pending until they explicitly
    accept it -- signing up doesn't come into it since they already have."""
    owner = _signup(client, f"owner2-{unique_email}")
    team_id = _create_team(client, owner, "Invite Co 2", "Team")
    invitee_email = f"invitee2-{unique_email}"

    invitee = _signup(client, invitee_email)
    invitee_headers = {"Authorization": f"Bearer {invitee}"}

    r = client.post(
        f"/api/teams/{team_id}/invite", json={"email": invitee_email},
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["member"]["status"] == "pending"

    r = client.get("/api/me/invites", headers=invitee_headers)
    assert r.status_code == 200
    assert any(i["team_id"] == team_id for i in r.json())

    # Attached but not yet accepted -- no access.
    r = client.get("/api/datasets", headers={**invitee_headers, "X-Team-Id": str(team_id)})
    assert r.status_code == 403

    r = client.post(f"/api/teams/{team_id}/accept-invite", headers=invitee_headers)
    assert r.status_code == 200

    r = client.get("/api/datasets", headers={**invitee_headers, "X-Team-Id": str(team_id)})
    assert r.status_code == 200


def test_only_owner_or_admin_can_invite(client, unique_email):
    owner = _signup(client, f"owner2-{unique_email}")
    team_id = _create_team(client, owner, "Strict Co", "Team")
    outsider = _signup(client, f"outsider-{unique_email}")

    r = client.post(
        f"/api/teams/{team_id}/invite", json={"email": f"nobody-{unique_email}"},
        headers={"Authorization": f"Bearer {outsider}"},
    )
    assert r.status_code == 403
