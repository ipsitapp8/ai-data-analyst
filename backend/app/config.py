"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
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

# --- API ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
