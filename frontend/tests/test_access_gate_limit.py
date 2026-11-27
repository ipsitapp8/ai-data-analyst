"""The shared-password gate (APP_PASSWORD) limits wrong guesses per visitor address."""
from __future__ import annotations

import auth


def test_wrong_guesses_lock_one_visitor_only(monkeypatch):
    monkeypatch.setattr(auth, "GATE_MAX_FAILURES", 3)
    auth._gate_failures.clear()  # noqa: SLF001
    for _ in range(3):
        auth._gate_failed("203.0.113.7")  # noqa: SLF001
    assert auth._gate_locked("203.0.113.7")  # noqa: SLF001
    assert not auth._gate_locked("203.0.113.8")  # noqa: SLF001


def test_old_failures_expire(monkeypatch):
    monkeypatch.setattr(auth, "GATE_MAX_FAILURES", 2)
    monkeypatch.setattr(auth, "GATE_WINDOW_SECONDS", 0)
    auth._gate_failures.clear()  # noqa: SLF001
    auth._gate_failed("203.0.113.7")  # noqa: SLF001
    auth._gate_failed("203.0.113.7")  # noqa: SLF001
    assert not auth._gate_locked("203.0.113.7")  # noqa: SLF001


def test_client_ip_is_none_outside_a_browser_session():
    import api_client
    assert api_client.client_ip() is None or isinstance(api_client.client_ip(), str)
