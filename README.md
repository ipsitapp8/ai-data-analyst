---
title: DataSage
emoji: "⬡"
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Autonomous AI data analyst with a self-verifying Critic
---

# AI Data Analyst

An autonomous AI agent that takes a raw CSV and a plain-English business question, and
independently **plans → writes code → executes it (sandboxed) → self-corrects on errors →
verifies its own conclusions → presents a dashboard with a full audit trail**.

```
Planner  ──▶  Executor  ──▶  Critic  ──▶  Dashboard
   ▲          (retries,          │  (rejected, within
   │        bounded ×3)          │   budget) triggers
   └──────── revision ◀──────────┘   a re-plan (×1)
```

## Architecture

- **LLM**: two models, on purpose. Planner/Executor/Dashboard run on **Gemini** (`google-genai`
  SDK, `backend/app/agents/llm_client.py`). The **Critic** runs on a separate **Llama** model
  (OpenAI-compatible endpoint, `backend/app/agents/llm_client_llama.py`) — a different model family
  catches different mistakes than the same model re-checking its own work, which is the whole point
  of having a Critic at all. Both wrappers expose the same `call_tool`/`call_text` interface, so the
  rest of the agent code doesn't know or care which provider is behind either call. (Originally
  built against Claude/Anthropic; swapped to Gemini+Llama since those are the keys available.)
- **Orchestration**: LangGraph state machine (`backend/app/agents/graph.py`) — Planner, Executor
  (with a bounded self-correction retry loop), Critic, and Dashboard-compile nodes.
- **Backend**: FastAPI (`backend/app/main.py`) — upload/profile, kick off a run in a background
  thread, poll status, fetch the dashboard, fetch the audit trail.
- **Sandbox**: every piece of AI-generated code runs in a network-isolated Docker container
  (`backend/app/sandbox/`) — no host access, capped CPU/memory, non-root user, fresh container per
  attempt. A restricted subprocess fallback exists for local dev without Docker, but it is **not**
  a real sandbox — see the warning below.
- **Storage**: SQLite (`backend/data/app.db`) — `datasets`, `questions`, `plans`,
  `execution_logs`, `critic_reviews`, `dashboards`, `audit_trail`. Every dashboard KPI/chart/
  narrative element has an `audit_trail` row pointing at the exact `execution_log` (code + stdout +
  result) and `critic_review` that produced/verified it.
- **Frontend**: Streamlit, custom dark glassmorphic theme (`frontend/style/theme.py`), 5 pages:
  Home (hero), Upload & Profile, Ask a Question (live Analysis Flow stepper), Dashboard, Audit Trail.

### Why the Critic is a genuinely separate check

The Critic agent does **not** reuse the Executor's conversation or trust its stdout summary, and it
doesn't even run on the same model — it calls Llama instead of Gemini. It writes its **own**
verification Python script that recomputes the key claimed numbers directly from
`/workspace/data/input.csv` (ideally via a different method than the Executor used), runs that in
its own sandbox container, and only then judges the Executor's claims against what it independently
found. See `backend/app/agents/critic.py` and `CRITIC_VERIFY_SYSTEM` / `CRITIC_REVIEW_SYSTEM` in
`backend/app/agents/prompts.py`.

If the Critic rejects the analysis, the graph loops back to the Planner once (bounded by
`MAX_CRITIC_REVISIONS`) with the Critic's specific feedback. If it's still rejected after that, the
dashboard is still generated but marked **unverified**, with the Critic's issues shown as a caveat
— the run always reaches an inspectable end state instead of looping forever or silently failing.

## Setup

### 1. Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running (for the sandbox)
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier available)
- A [Llama API key](https://llama.developer.meta.com/) for the Critic (or any OpenAI-compatible
  Llama provider — Groq/Together/OpenRouter — see the `LLAMA_BASE_URL` comment in `.env.example`)

### 2. Configure

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

### 3. Build the sandbox image

Docker Desktop must be running first.

```bash
# Windows PowerShell
./backend/app/sandbox/build.ps1

# macOS/Linux
./backend/app/sandbox/build.sh
```

This builds `ai-data-analyst-sandbox:latest` — a minimal `python:3.11-slim` image with pinned
pandas/numpy/plotly/matplotlib, running as a non-root user, with no network access at run time.

### 4. Install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 5. Run

Two processes, in separate terminals:

```bash
# Terminal 1 — backend (from repo root)
uvicorn app.main:app --app-dir backend --reload --port 8000

# Terminal 2 — frontend (from repo root)
streamlit run frontend/Home.py
```

Open the Streamlit URL it prints (usually http://localhost:8501). Start on **Upload & Profile**,
then **Ask a Question**.

### No Docker available?

Set `SANDBOX_BACKEND=subprocess` in `.env`. Agent-generated code then runs as a host subprocess
with a scrubbed environment (no API keys), CPU/file-size/process caps, and its own process session,
instead of in a container. That is meaningfully better than nothing, but it shares your kernel and
filesystem — use Docker for anything beyond your own throwaway experiments. On a deployed box also
set `SANDBOX_RUN_AS_USER` so those children drop to an unprivileged account; see
[Deployment](#deployment-hugging-face-spaces).

## How a question flows through the system

1. **Upload & Profile** — CSV is parsed with pandas, profiled (dtypes, row/col counts, missing
   values, per-column stats/top-values), stored as a `datasets` row.
2. **Ask a Question** — POST `/api/questions` creates a `questions` row and starts the LangGraph
   run in a background thread. The page polls `GET /api/questions/{id}/status` every ~2s and
   renders the live **Analysis Flow** stepper from the real `current_stage` on the `questions` row.
3. **Planner** writes 3–6 concrete, executable steps (`plans` row).
4. **Executor** writes one Python script per step, runs it in the sandbox, and on failure feeds the
   stderr back into the next attempt — up to `MAX_EXECUTOR_RETRIES` (default 3). Every attempt is
   logged to `execution_logs`, success or not.
5. **Critic** independently re-derives the key numbers from raw data (see above) and either
   verifies or rejects, logged to `critic_reviews`.
6. **Dashboard** compiles KPIs/charts/narrative from the verified step results, writes a
   `dashboards` row, and writes one `audit_trail` row per KPI/chart/narrative element linking it to
   its source `execution_log` and the `critic_review` that vouches for it.
7. **Dashboard** and **Audit Trail** pages read all of this back through the API — nothing on
   screen is hand-waved; every number is one click from its source code and reasoning.

## Project layout

```
backend/
  app/
    main.py              FastAPI app + all endpoints
    config.py             env-driven settings
    database.py / models.py / schemas.py
    profiling.py           CSV -> profile dict
    agents/
      graph.py              LangGraph wiring (nodes + conditional routing)
      planner.py critic.py executor.py dashboard.py   node implementations
      llm_client.py          Gemini function-calling wrapper (Planner/Executor/Dashboard)
      llm_client_llama.py     Llama function-calling wrapper (Critic only)
      prompts.py              system prompts for Planner/Executor/Critic/Dashboard
      state.py                 shared LangGraph state + DB stage updates
    sandbox/
      Dockerfile               the sandbox image
      runner.py                 Docker + subprocess execution backends
    storage/audit.py             audit_trail helper
  data/                          uploads, per-question sandbox workspaces, app.db (gitignored)
frontend/
  Home.py                        hero landing page
  pages/                          Upload&Profile, Ask a Question, Dashboard, Audit Trail
  style/theme.py                  shared CSS + component helpers
  api_client.py                   requests wrapper around the backend
```

## Deployment (Hugging Face Spaces)

The repo builds as a single container: FastAPI on internal `127.0.0.1:8000`, Streamlit public on
`7860`. One container rather than two services on purpose — the API binds loopback and is never
publicly routable, so the only way to reach it is through the Streamlit app, which sits behind the
password gate.

```bash
# 1. create the Space (sdk: docker) at huggingface.co/new-space, then:
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

Set these as **Space secrets** (Settings → Variables and secrets) — never commit them:

| Secret | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Planner / Executor / Dashboard |
| `LLAMA_API_KEY` | Critic |
| `APP_PASSWORD` | Shared password for the gate. Unset ⇒ gate disabled. |

### What the hosted deployment gives up

Spaces containers can't reach a Docker daemon, so `SANDBOX_BACKEND=subprocess` there — the
per-attempt container is not available. The fallback is hardened rather than merely tolerated:

- **Scrubbed environment.** Generated code gets an env dict built from scratch (`_sandbox_env`),
  never a copy of `os.environ`. `GEMINI_API_KEY` and `LLAMA_API_KEY` are simply not present.
- **Dropped privileges.** Children run as the unprivileged `sandbox` user. `/app/data` is `700`
  and root-owned, so `app.db` and the raw uploads can't even be traversed to; the workspace tree
  lives on a separate path that the sandbox user owns.
- **Resource caps.** `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`, no core dumps, own process
  session so a timeout kills grandchildren too.

That is a genuine boundary against *code reading things it shouldn't*, but it is still one kernel
namespace — there is no network isolation, and a container escape is a container escape. For real
multi-tenant use, run the Docker backend on a VM.

## Known limitations (by design, for this MVP)

- Single-user, local-only: no auth, no multi-tenant isolation beyond the sandbox itself.
- Status polling, not websockets — simple and reliable for a solo local app; swap for SSE/websockets
  if this ever needs multiple concurrent viewers on one run.
- Vision-based ingestion of scanned/handwritten docs, multi-source joins, and the full reasoning-
  trace "Observability" view are stubbed out of scope per the build brief — the schema
  (`execution_logs`, `critic_reviews`) already has everything an Observability view would need to
  render; it just isn't built yet.
