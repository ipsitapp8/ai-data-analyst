"""Rate limits are per visitor, not shared by everyone who comes through the frontend.

The Streamlit frontend calls the API server-side, so every request arrives from loopback; it forwards the visitor's
address in X-Silt-Client-IP. The API believes that header only on a loopback connection and only when it is an IP."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import rate_limit


def _req(peer, header=None):
    return SimpleNamespace(client=SimpleNamespace(host=peer),
                           headers={rate_limit.CLIENT_IP_HEADER: header} if header is not None else {})


def test_forwarded_address_is_used_from_the_local_frontend():
    assert rate_limit.client_ip(_req("127.0.0.1", "203.0.113.7")) == "203.0.113.7"
    assert rate_limit.client_ip(_req("::1", "2001:db8::1")) == "2001:db8::1"


def test_forwarded_address_is_ignored_from_anywhere_else():
    assert rate_limit.client_ip(_req("198.51.100.4", "203.0.113.7")) == "198.51.100.4"


def test_an_invalid_forwarded_value_is_ignored():
    assert rate_limit.client_ip(_req("127.0.0.1", "not-an-ip; DROP")) == "127.0.0.1"
    assert rate_limit.client_ip(_req("127.0.0.1")) == "127.0.0.1"


def test_one_visitor_hitting_the_signup_limit_does_not_block_another():
    rate_limit._attempts.clear()  # noqa: SLF001
    a, b = rate_limit.client_ip(_req("127.0.0.1", "203.0.113.7")), rate_limit.client_ip(_req("127.0.0.1", "203.0.113.8"))
    for _ in range(3):
        rate_limit.enforce(f"signup:{a}", max_attempts=3, window_seconds=60)
    with pytest.raises(HTTPException):
        rate_limit.enforce(f"signup:{a}", max_attempts=3, window_seconds=60)
    rate_limit.enforce(f"signup:{b}", max_attempts=3, window_seconds=60)      # a different visitor still can


def test_signup_through_the_api_keys_the_limit_on_the_forwarded_address(api_client_factory=None):
    from fastapi.testclient import TestClient

    from app.main import app
    rate_limit._attempts.clear()  # noqa: SLF001
    with TestClient(app) as c:
        h = {rate_limit.CLIENT_IP_HEADER: "203.0.113.9"}
        c.post("/api/auth/signup", json={"email": "x@example.test", "password": "pw-123456789", "display_name": "x"}, headers=h)
    assert any(k == "signup:203.0.113.9" for k in rate_limit._attempts)  # noqa: SLF001
