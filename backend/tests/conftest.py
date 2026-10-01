"""Test setup: an isolated scratch SQLite DB (never the real app.db), env vars
set before any `app.*` module is imported since app.config reads them at
import time.
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

_scratch = Path(tempfile.mkdtemp(prefix="silt_pytest_"))
os.environ["DB_PATH"] = str(_scratch / "test.db")
os.environ["DATA_DIR"] = str(_scratch / "data")
os.environ["UPLOADS_DIR"] = str(_scratch / "data" / "uploads")
os.environ["WORKSPACES_DIR"] = str(_scratch / "data" / "workspaces")
os.environ["JWT_SECRET_KEY"] = "test-only-secret-key-not-for-production-32b"
os.environ["SANDBOX_BACKEND"] = "subprocess"
# Explicitly blank, not just "unset": dotenv doesn't override a key that's
# already present in os.environ (even set to ""), so this is what actually
# stops a developer's real SMTP_USER/SMTP_PASSWORD in their local .env from
# leaking into the test run and sending real mail to fake @test.example
# addresses.
# Tests drive the scheduler tick by hand; a live background tick would race them.
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["SMTP_USER"] = ""
os.environ["SMTP_PASSWORD"] = ""
# Same reasoning as SMTP above: a developer's real local .env DATABASE_URL
# must never leak into a test run and point tests at real Postgres by
# accident. TEST_DATABASE_URL is a distinctly-named opt-in for when a
# Postgres-backed test run is actually wanted (e.g. the CI job that verifies
# the Postgres/Alembic path) -- nobody's real .env coincidentally has this
# exact name set, so it's never accidentally picked up.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app as fastapi_app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture
def unique_email() -> str:
    """A fresh email per test -- tests share one DB for the whole session
    (no per-test reset), so collisions would otherwise leak between tests."""
    return f"{uuid.uuid4().hex[:12]}@test.example"


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """The in-memory auth rate limiter is process-global; without this,
    unrelated tests would trip each other's login/signup limits."""
    from app.rate_limit import _attempts
    _attempts.clear()
    yield
