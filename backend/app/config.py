"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
import secrets
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: backend/app/config.py -> root)
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = Path(os.getenv("DATA_DIR") or BACKEND_DIR / "data")
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR") or DATA_DIR / "uploads")
# Overridable so a deployment can put agent workspaces on a separate tree from
# the database and uploads -- the subprocess sandbox drops to an unprivileged
# user, which only buys anything if that user can't reach app.db in the first place.
WORKSPACES_DIR = Path(os.getenv("WORKSPACES_DIR") or DATA_DIR / "workspaces")
DB_PATH = Path(os.getenv("DB_PATH") or DATA_DIR / "app.db")

for d in (DATA_DIR, UPLOADS_DIR, WORKSPACES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Gemini (Planner / Executor / Dashboard) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Flash, not Pro: the Pro models are not available on the free tier (they return
# 429 with "limit: 0"). Flash handles the structured tool-calling this app needs.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Bounded retry-with-backoff for the free tier's per-minute rate limit (429),
# tried before failing over to Llama -- most 429s clear within a retry or two.
GEMINI_RATE_LIMIT_RETRIES = int(os.getenv("GEMINI_RATE_LIMIT_RETRIES", "2"))
GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS = int(os.getenv("GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS", "30"))

# --- Llama (Critic only) ---
# A second, different model family for the Critic's independent verification --
# genuinely independent means a different model, not just a different prompt on
# the same one. Defaults to Meta's official Llama API (OpenAI-compatible); point
# LLAMA_BASE_URL at Groq/Together/OpenRouter/etc instead if that's what your key is for.
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/")

# --- Sandbox ---
# "docker" (recommended, isolated) or "subprocess" (dev-only fallback, NOT isolated)
SANDBOX_BACKEND = os.getenv("SANDBOX_BACKEND", "docker")
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "ai-data-analyst-sandbox:latest")
SANDBOX_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "60"))
SANDBOX_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "512m")
SANDBOX_CPU_LIMIT = os.getenv("SANDBOX_CPU_LIMIT", "1")

# Applied only by the subprocess fallback, which has no container to lean on.
# SANDBOX_RUN_AS_USER: an unprivileged account to drop to (POSIX, requires root).
# SANDBOX_ADDRESS_SPACE_LIMIT: opt-in RLIMIT_AS -- leave empty unless you have
# tested it, since numpy/OpenBLAS reserve enough virtual memory that a tight cap
# breaks `import pandas` outright.
SANDBOX_RUN_AS_USER = os.getenv("SANDBOX_RUN_AS_USER", "")
SANDBOX_MAX_OUTPUT_BYTES = os.getenv("SANDBOX_MAX_OUTPUT_BYTES", "256m")
SANDBOX_ADDRESS_SPACE_LIMIT = os.getenv("SANDBOX_ADDRESS_SPACE_LIMIT", "")

# --- Agent loop ---
MAX_EXECUTOR_RETRIES = int(os.getenv("MAX_EXECUTOR_RETRIES", "3"))
MAX_CRITIC_REVISIONS = int(os.getenv("MAX_CRITIC_REVISIONS", "1"))

# --- Auth (per-user login, JWT sessions) ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not JWT_SECRET_KEY:
    # Ephemeral per-process secret: fine for local dev (tokens just don't
    # survive a restart), but every deployment should set a real one --
    # otherwise a restart invalidates every session, and multiple workers
    # wouldn't agree on tokens at all.
    JWT_SECRET_KEY = secrets.token_hex(32)
    warnings.warn(
        "JWT_SECRET_KEY is not set -- using a random one for this process only. "
        "Set JWT_SECRET_KEY in the environment for any real deployment.",
        stacklevel=2,
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))  # 7 days

# --- API ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
# Comma-separated allowlist for browser-originated requests. Empty means "*"
# (today's default) -- the shipped deployment binds the API to loopback and
# the Streamlit frontend calls it server-side, so no browser ever makes a
# cross-origin request to it; this only matters if the API is exposed
# directly and called from a browser-based client.
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "")

# Max upload size for a single CSV, enforced in main.py before the file is
# read into memory/disk -- otherwise an arbitrarily large upload can exhaust
# either.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200 MB

# --- Outbound email (team invites) ---
# Gmail SMTP by default: SMTP_USER is the full Gmail address, SMTP_PASSWORD is
# a 16-character App Password (Google Account -> Security -> 2-Step
# Verification -> App Passwords) -- NOT the account's login password, and
# only issuable once 2FA is on. Leave SMTP_USER/SMTP_PASSWORD unset to
# disable sending entirely: invites still create the pending membership row,
# they just don't email anyone about it (logged, not raised, so a missing
# mail config never breaks the invite API call itself).
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)
# Public URL of the Streamlit frontend, used only to build the link inside
# invite emails -- BACKEND_BASE_URL points at the API, not the UI, so this
# needs its own setting.
APP_URL = os.getenv("APP_URL", "http://localhost:8501")

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
