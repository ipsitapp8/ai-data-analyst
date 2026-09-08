"""Email validation -- signup and invite previously only checked "@" appears
somewhere in the string, which let malformed addresses like a double-@ typo
through. Those got a pending row created for them that could never be
fulfilled, with SMTP's rejection then misreported to the user as a generic
"couldn't send" rather than "that address is invalid".
"""
from __future__ import annotations

from app.validation import is_valid_email


def test_rejects_double_at():
    assert is_valid_email("swetalinrout@006@gmail.com") is False


def test_rejects_no_at():
    assert is_valid_email("not-an-email") is False


def test_rejects_no_domain_dot():
    assert is_valid_email("someone@localhost") is False


def test_rejects_whitespace():
    assert is_valid_email("some one@example.com") is False


def test_rejects_empty():
    assert is_valid_email("") is False


def test_accepts_normal_address():
    assert is_valid_email("someone@example.com") is True


def test_accepts_plus_addressing_and_subdomain():
    assert is_valid_email("someone+tag@mail.example.co.uk") is True


def test_signup_rejects_malformed_email(client, unique_email):
    r = client.post(
        "/api/auth/signup",
        json={"email": "swetalinrout@006@gmail.com", "password": "password123", "display_name": "T"},
    )
    assert r.status_code == 400


def test_invite_rejects_malformed_email(client, unique_email):
    r = client.post(
        "/api/auth/signup",
        json={"email": f"owner-{unique_email}", "password": "password123", "display_name": "Owner"},
    )
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    c = client.post("/api/communities", json={"name": "Validation Co"}, headers=headers)
    team_id = client.post(
        f"/api/communities/{c.json()['id']}/teams", json={"name": "Team"}, headers=headers
    ).json()["id"]

    r = client.post(
        f"/api/teams/{team_id}/invite", json={"email": "swetalinrout@006@gmail.com"}, headers=headers
    )
    assert r.status_code == 400
