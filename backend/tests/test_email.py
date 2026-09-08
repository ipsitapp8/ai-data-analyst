"""Invite emails -- must never break the invite endpoint regardless of SMTP
config or delivery failures, and must actually call smtplib correctly when
configured.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app import config
from app.email_sender import email_configured, send_team_invite_email


def test_unconfigured_smtp_reports_not_configured():
    old_user, old_pass = config.SMTP_USER, config.SMTP_PASSWORD
    config.SMTP_USER, config.SMTP_PASSWORD = "", ""
    try:
        assert email_configured() is False
        assert send_team_invite_email("x@test.example", "Alice", "Acme", "Growth") is False
    finally:
        config.SMTP_USER, config.SMTP_PASSWORD = old_user, old_pass


def test_configured_send_calls_smtp_correctly():
    old_user, old_pass, old_from = config.SMTP_USER, config.SMTP_PASSWORD, config.SMTP_FROM
    config.SMTP_USER, config.SMTP_PASSWORD, config.SMTP_FROM = "bot@example.com", "app-password", "bot@example.com"
    try:
        assert email_configured() is True
        with patch("app.email_sender.smtplib.SMTP") as smtp_cls:
            smtp = MagicMock()
            smtp_cls.return_value.__enter__.return_value = smtp

            ok = send_team_invite_email("invitee@test.example", "Alice", "Acme", "Growth")

            assert ok is True
            smtp.starttls.assert_called_once()
            smtp.login.assert_called_once_with("bot@example.com", "app-password")
            assert smtp.sendmail.call_count == 1
            args, _ = smtp.sendmail.call_args
            assert args[0] == "bot@example.com"
            assert args[1] == ["invitee@test.example"]
            assert "invitee@test.example" not in args[2] or "To:" in args[2]
    finally:
        config.SMTP_USER, config.SMTP_PASSWORD, config.SMTP_FROM = old_user, old_pass, old_from


def test_smtp_failure_is_swallowed_not_raised():
    old_user, old_pass = config.SMTP_USER, config.SMTP_PASSWORD
    config.SMTP_USER, config.SMTP_PASSWORD = "bot@example.com", "app-password"
    try:
        with patch("app.email_sender.smtplib.SMTP", side_effect=OSError("connection refused")):
            ok = send_team_invite_email("invitee@test.example", "Alice", "Acme", "Growth")
        assert ok is False
    finally:
        config.SMTP_USER, config.SMTP_PASSWORD = old_user, old_pass


def test_invite_endpoint_succeeds_even_without_smtp_configured(client, unique_email):
    """End-to-end: the actual invite endpoint (SMTP unset in the test env)
    still creates the pending membership and returns 200."""
    from app import config as app_config

    assert app_config.SMTP_USER == ""  # sanity: test env has no SMTP configured

    r = client.post(
        "/api/auth/signup",
        json={"email": f"owner-{unique_email}", "password": "password123", "display_name": "Owner"},
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    c = client.post("/api/communities", json={"name": "Email Co"}, headers=headers)
    team_id = client.post(
        f"/api/communities/{c.json()['id']}/teams", json={"name": "Team"}, headers=headers
    ).json()["id"]

    r = client.post(
        f"/api/teams/{team_id}/invite", json={"email": f"invitee-{unique_email}"}, headers=headers
    )
    assert r.status_code == 200, r.text
    assert r.json()["member"]["status"] == "pending"
    assert r.json()["email_sent"] is False
