"""Auth rate limiting -- codifies the manual verification from the hardening
pass: 8 login attempts per IP+email per minute get through, the rest 429."""
from __future__ import annotations


def test_login_rate_limited_after_threshold(client, unique_email):
    client.post(
        "/api/auth/signup",
        json={"email": unique_email, "password": "password123", "display_name": "T"},
    )

    codes = [
        client.post("/api/auth/login", json={"email": unique_email, "password": "wrong"}).status_code
        for _ in range(10)
    ]
    assert codes.count(401) == 8
    assert codes.count(429) == 2
    # Rate limiting shouldn't itself ever return anything but 401/429 here.
    assert set(codes) <= {401, 429}


def test_signup_rate_limited_after_threshold(client, unique_email):
    codes = []
    for i in range(12):
        r = client.post(
            "/api/auth/signup",
            json={"email": f"{i}-{unique_email}", "password": "password123", "display_name": "T"},
        )
        codes.append(r.status_code)
    assert codes.count(200) == 10
    assert codes.count(429) == 2
