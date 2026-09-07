"""Signup/login/me -- the basic account lifecycle."""
from __future__ import annotations


def test_signup_then_login_then_me(client, unique_email):
    r = client.post(
        "/api/auth/signup",
        json={"email": unique_email, "password": "password123", "display_name": "Test User"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["email"] == unique_email
    assert "access_token" in body

    r = client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "password123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]

    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    assert r.json()["email"] == unique_email


def test_signup_duplicate_email_rejected(client, unique_email):
    payload = {"email": unique_email, "password": "password123", "display_name": "T"}
    assert client.post("/api/auth/signup", json=payload).status_code == 200
    r = client.post("/api/auth/signup", json=payload)
    assert r.status_code == 409


def test_signup_rejects_short_password(client, unique_email):
    r = client.post(
        "/api/auth/signup",
        json={"email": unique_email, "password": "short", "display_name": "T"},
    )
    assert r.status_code == 400


def test_login_wrong_password_rejected(client, unique_email):
    client.post(
        "/api/auth/signup",
        json={"email": unique_email, "password": "password123", "display_name": "T"},
    )
    r = client.post("/api/auth/login", json={"email": unique_email, "password": "wrong-password"})
    assert r.status_code == 401


def test_login_unknown_email_rejected(client, unique_email):
    r = client.post("/api/auth/login", json={"email": unique_email, "password": "whatever123"})
    assert r.status_code == 401


def test_me_requires_a_token(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401
