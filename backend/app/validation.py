"""Small, shared input-validation helpers.

Deliberately not RFC 5322 compliant -- that grammar is notoriously complex
for what it buys. This just needs to catch the obviously-wrong cases (no @,
more than one @, no domain, whitespace) before they reach the database or,
worse, an external service like Gmail's SMTP server that reports the error
back in a way a caller has to translate into something a user understands.
"""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(_EMAIL_RE.match(email))
