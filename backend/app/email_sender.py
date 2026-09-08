"""Outbound email for team invites, over plain SMTP (Gmail by default).

Deliberately synchronous: invites are a low-frequency, human-triggered
action, not a hot path, so blocking the request for the ~1s an SMTP send
takes is a reasonable tradeoff against the complexity of a background queue.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app import config

logger = logging.getLogger(__name__)


def email_configured() -> bool:
    return bool(config.SMTP_USER and config.SMTP_PASSWORD)


def send_team_invite_email(
    to_email: str,
    inviter_name: str,
    community_name: str,
    team_name: str,
) -> bool:
    """Best-effort: logs and returns False on any failure rather than
    raising, so a broken mail config never breaks the invite API call that
    triggered it -- the pending membership row is already committed by then."""
    if not email_configured():
        logger.info("SMTP not configured -- skipping invite email to %s", to_email)
        return False

    subject = f"{inviter_name} invited you to {team_name} on Silt"
    text_body = (
        f"{inviter_name} invited you to join the \"{team_name}\" team "
        f"in \"{community_name}\" on Silt.\n\n"
        f"Sign in (or create an account with this email address) to accept: {config.APP_URL}\n"
    )
    html_body = f"""
    <div style="font-family:sans-serif;color:#1a1a1a;">
      <p><b>{inviter_name}</b> invited you to join <b>{team_name}</b>
      in <b>{community_name}</b> on Silt.</p>
      <p><a href="{config.APP_URL}" style="color:#4f8fe0;">Open Silt</a> and sign in
      (or sign up with this email address) to accept the invite.</p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_FROM, [to_email], msg.as_string())
        logger.info("Sent invite email to %s", to_email)
        return True
    except Exception:  # noqa: BLE001 - email delivery must never break the invite endpoint
        logger.exception("Failed to send invite email to %s", to_email)
        return False
