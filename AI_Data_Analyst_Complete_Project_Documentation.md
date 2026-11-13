# AI Data Analyst — Complete Project Documentation

**Project Name:** DataSage / Silt — AI Data Analyst  
**Technology Stack:** Python 3.11 · FastAPI · Streamlit · LangGraph · Gemini · Llama · Docker · SQLite · Pandas · Plotly  
**Date:** September 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Complete Folder & File Structure](#3-complete-folder--file-structure)
4. [Technology Stack Breakdown](#4-technology-stack-breakdown)
5. [How Docker Is Used](#5-how-docker-is-used)
6. [How FastAPI (Backend) Works](#6-how-fastapi-backend-works)
7. [How Streamlit (Frontend) Works](#7-how-streamlit-frontend-works)
8. [How LangGraph Is Used — The AI Agent Pipeline](#8-how-langgraph-is-used--the-ai-agent-pipeline)
9. [Complete Code Flow: From CSV Upload to Dashboard](#9-complete-code-flow-from-csv-upload-to-dashboard)
10. [CSV Analysis & Pandas Calculations](#10-csv-analysis--pandas-calculations)
11. [The Results Pipeline — What Produces Results](#11-the-results-pipeline--what-produces-results)
12. [Code Generation — Which Part Writes Python Code](#12-code-generation--which-part-writes-python-code)
13. [Audit Report System](#13-audit-report-system)
14. [Dashboard Generation — How Dashboards Are Built](#14-dashboard-generation--how-dashboards-are-built)
15. [Database Schema & Models](#15-database-schema--models)
16. [Authentication & Multi-Tenancy](#16-authentication--multi-tenancy)
17. [Security & Sandboxing](#17-security--sandboxing)
18. [Deployment Architecture](#18-deployment-architecture)
19. [API Endpoints Reference](#19-api-endpoints-reference)
20. [Frontend Pages Reference](#20-frontend-pages-reference)
21. [Configuration Reference](#21-configuration-reference)
22. [End-to-End Walkthrough Example](#22-end-to-end-walkthrough-example)

---

## 1. Project Overview

The AI Data Analyst is an **autonomous AI agent system** that takes a raw CSV file and a plain-English business question, and independently:

1. **Plans** the analysis (breaks the question into steps)
2. **Writes Python code** for each step
3. **Executes code** in an isolated sandbox
4. **Self-corrects** on errors (up to 3 retries)
5. **Independently verifies** conclusions using a second, different AI model
6. **Presents a dashboard** with KPI cards, Plotly charts, and a narrative summary
7. **Maintains a complete audit trail** linking every number back to its source code and data

```
User Question → Triage → Planner → Executor (retries ×3) → Critic → Dashboard
                  ▲                      │                    │
                  │                      │   (rejected, within budget)
                  └──────── revision ◀───┘   triggers re-plan (×1)
```

---

## 2. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        SINGLE DOCKER CONTAINER                     │
│                                                                    │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │   STREAMLIT (Frontend)│    │       FASTAPI (Backend)          │  │
│  │   Port: 7860 (public) │───▶│       Port: 8000 (127.0.0.1)    │  │
│  │                        │    │                                  │  │
│  │  • Home.py             │    │  • REST API (main.py)           │  │
│  │  • 9 Streamlit Pages   │    │  • LangGraph Agent Pipeline     │  │
│  │  • Custom Dark Theme   │    │  • SQLite Database              │  │
│  │  • api_client.py       │    │  • Sandbox Runner               │  │
│  └──────────────────────┘    └──────────┬───────────────────────┘  │
│                                          │                         │
│                                          ▼                         │
│                               ┌──────────────────┐                │
│                               │  SANDBOX           │                │
│                               │  Docker / Subprocess│               │
│                               │  (network-isolated) │               │
│                               │  Runs AI-generated  │               │
│                               │  Python code        │               │
│                               └──────────────────┘                │
│                                          │                         │
│           ┌──────────────────────────────┼────────────────────┐    │
│           │                              │                    │    │
│           ▼                              ▼                    ▼    │
│    ┌─────────────┐            ┌────────────────┐    ┌──────────┐  │
│    │ Gemini API   │            │   Llama API     │    │ SQLite   │  │
│    │ (Planner/    │            │   (Critic only) │    │ app.db   │  │
│    │ Executor/    │            │   Cross-model   │    │          │  │
│    │ Dashboard)   │            │   verification  │    │          │  │
│    └─────────────┘            └────────────────┘    └──────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- FastAPI binds to **127.0.0.1:8000** (loopback only) — never publicly accessible
- Streamlit binds to **0.0.0.0:7860** (public) — the only entry point
- The Critic uses a **different model family** (Llama) than the Planner/Executor/Dashboard (Gemini) — a different model catches different mistakes than the same model re-checking its own work
- Every piece of AI-generated code runs in a **sandbox** (Docker container with no network, or hardened subprocess fallback)

---

## 3. Complete Folder & File Structure

```
ai-data-analyst/
│
├── Dockerfile                 # Single-container deployment image
├── start.sh                   # Boots both FastAPI + Streamlit
├── render.yaml                # Render.com deployment blueprint
├── .env.example               # All environment variables with docs
├── .dockerignore              # Files excluded from Docker build
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── LICENSE                    # MIT license
├── pytest.ini                 # Test configuration
├── requirements-dev.txt       # Dev dependencies (pytest, etc.)
│
├── backend/                   # ───── BACKEND (FastAPI + Agents) ─────
│   ├── requirements.txt       # Backend Python dependencies
│   ├── data/                  # Runtime data (gitignored)
│   │   ├── app.db             #   SQLite database (auto-created)
│   │   ├── uploads/           #   Uploaded CSV files
│   │   └── workspaces/        #   Per-question sandbox workspaces
│   │       └── q{id}/         #     Per-question workspace
│   │           ├── data/      #       input.csv is copied here
│   │           └── output/    #       Plotly JSON charts written here
│   │
│   ├── app/                   # Main application package
│   │   ├── __init__.py
│   │   ├── main.py            # ★ FastAPI app + ALL endpoints
│   │   ├── config.py          # ★ Central env-driven configuration
│   │   ├── database.py        # SQLAlchemy engine/session + migrations
│   │   ├── models.py          # ★ All 10 SQLAlchemy ORM models
│   │   ├── schemas.py         # Pydantic request/response schemas
│   │   ├── profiling.py       # ★ CSV profiling with Pandas
│   │   ├── security.py        # Password hashing, JWT, auth deps
│   │   ├── validation.py      # Email validation helper
│   │   ├── rate_limit.py      # In-memory rate limiter
│   │   ├── logging_config.py  # Logging setup
│   │   ├── email_sender.py    # SMTP email for team invites
│   │   │
│   │   ├── agents/            # ─── AI AGENT PIPELINE ───
│   │   │   ├── __init__.py
│   │   │   ├── graph.py       # ★ LangGraph state machine wiring
│   │   │   ├── state.py       # ★ AgentState TypedDict + DB updates
│   │   │   ├── triage.py      # Triage node (question validation)
│   │   │   ├── planner.py     # ★ Planner node (creates analysis plan)
│   │   │   ├── executor.py    # ★ Executor node (writes+runs code)
│   │   │   ├── critic.py      # ★ Critic node (independent verification)
│   │   │   ├── dashboard.py   # ★ Dashboard compile node
│   │   │   ├── prompts.py     # ★ System prompts for all agents
│   │   │   ├── llm_client.py  # ★ Gemini API wrapper + failover
│   │   │   └── llm_client_llama.py  # Llama API wrapper (Critic)
│   │   │
│   │   ├── sandbox/           # ─── CODE SANDBOX ───
│   │   │   ├── __init__.py
│   │   │   ├── Dockerfile     # Sandbox container image definition
│   │   │   ├── runner.py      # ★ Docker + Subprocess execution
│   │   │   ├── build.sh       # Build sandbox image (macOS/Linux)
│   │   │   └── build.ps1      # Build sandbox image (Windows)
│   │   │
│   │   ├── storage/           # ─── DATA STORAGE HELPERS ───
│   │   │   ├── __init__.py
│   │   │   └── audit.py       # Audit trail row writer
│   │   │
│   │   └── routers/           # ─── API ROUTERS ───
│   │       ├── __init__.py
│   │       ├── auth.py        # Signup/Login/Me endpoints
│   │       ├── workspaces.py  # Community/Team/Invite management
│   │       └── inspect.py     # Click-to-inspect element endpoint
│   │
│   └── tests/                 # Backend tests
│
├── frontend/                  # ───── FRONTEND (Streamlit) ─────
│   ├── requirements.txt       # Frontend Python dependencies
│   ├── Home.py                # ★ Landing page with live stats
│   ├── config.py              # Frontend config (backend URL)
│   ├── api_client.py          # ★ HTTP client for backend API
│   ├── auth.py                # Shared password gate + login UI
│   │
│   ├── pages/                 # ─── STREAMLIT PAGES ───
│   │   ├── 1_Overview.py      # Workspace dashboard overview
│   │   ├── 2_Datasets.py      # ★ Upload & profile CSV datasets
│   │   ├── 3_Analyses.py      # ★ Ask questions & live run viewer
│   │   ├── 4_Dashboards.py    # ★ KPIs, charts, narrative display
│   │   ├── 5_Reports.py       # Download text reports
│   │   ├── 6_Audit_Trail.py   # ★ Full traceability viewer
│   │   ├── 7_Settings.py      # Account/system status
│   │   ├── 8_Workspaces.py    # Community/Team management
│   │   └── 9_Team_Overview.py # Team info page
│   │
│   ├── style/                 # ─── UI THEME ───
│   │   ├── __init__.py
│   │   └── theme.py           # ★ Full dark glassmorphic design system
│   │
│   └── tests/                 # Frontend tests
```

---

## 4. Technology Stack Breakdown

### Backend Technologies

| Technology | Version | Purpose |
|---|---|---|
| **FastAPI** | 0.141.1 | REST API framework |
| **Uvicorn** | 0.52.4 | ASGI server for FastAPI |
| **SQLAlchemy** | 2.0.52 | ORM for SQLite database |
| **Pydantic** | 2.13.4 | Data validation & serialization |
| **LangGraph** | 1.2.11 | AI agent state machine orchestration |
| **google-genai** | 2.19.0 | Gemini API SDK (Planner/Executor/Dashboard) |
| **OpenAI** | 3.3.1 | Llama API wrapper (Critic agent) |
| **Pandas** | 2.3.3 | CSV parsing, data profiling, analysis |
| **NumPy** | 2.4.6 | Numerical computation support |
| **Plotly** | 6.9.0 | Interactive chart generation |
| **Matplotlib** | 3.11.1 | Backup charting (in sandbox) |
| **PyJWT** | 2.13.0 | JWT token creation/validation |
| **bcrypt** | 5.0.0 | Password hashing |
| **python-dotenv** | 1.2.3 | .env file loading |

### Frontend Technologies

| Technology | Version | Purpose |
|---|---|---|
| **Streamlit** | 1.62.0 | Web UI framework |
| **Requests** | 2.34.2 | HTTP client for backend API |
| **Plotly** | 6.9.0 | Chart rendering in frontend |
| **Pandas** | 2.3.3 | Data display support |

### Infrastructure

| Technology | Purpose |
|---|---|
| **Docker** | Container for deployment + sandbox for code execution |
| **SQLite** | Application database (datasets, questions, plans, logs, etc.) |
| **SMTP** | Email sending for team invites |

---

## 5. How Docker Is Used

Docker serves **TWO distinct purposes** in this project:

### 5.1 Deployment Container (Root `Dockerfile`)

**File:** `Dockerfile` (project root)

The root Dockerfile packages the entire app into a single container:

```dockerfile
FROM python:3.11-slim

# Creates unprivileged 'sandbox' user for generated code
RUN useradd --create-home --uid 1001 --shell /usr/sbin/nologin sandbox

# Install both backend and frontend dependencies
COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r frontend/requirements.txt

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY start.sh /start.sh

# Environment variables for single-container topology
ENV SANDBOX_BACKEND=subprocess \
    BACKEND_HOST=127.0.0.1 \
    BACKEND_PORT=8000 \
    BACKEND_BASE_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_PORT=7860

# Security: app.db and uploads unreachable by sandbox user
RUN mkdir -p /app/data/uploads /sandbox/workspaces \
 && chmod 700 /app/data \
 && chown -R sandbox:sandbox /sandbox

EXPOSE 7860
CMD ["/start.sh"]
```

### 5.2 Sandbox Container (for Code Execution)

**File:** `backend/app/sandbox/Dockerfile`

A separate, minimal image used to run AI-generated Python code safely:

```dockerfile
FROM python:3.11-slim

# Minimal analysis stack, pinned versions
RUN pip install --no-cache-dir \
    pandas==2.2.2 numpy==1.26.4 plotly==5.22.0 matplotlib==3.8.4

# Non-root user
RUN useradd -m -u 1000 sandboxuser
USER sandboxuser

WORKDIR /workspace
ENTRYPOINT ["python"]
```

**Each sandbox execution** runs `docker run --rm --network none` — meaning:
- ❌ No network access
- ❌ No host filesystem access (only the mounted workspace)
- ❌ No API keys visible
- ✅ Capped CPU and memory
- ✅ Fresh container per attempt

### 5.3 The `start.sh` Script

Boots both processes in one container:

1. Launches `uvicorn` (FastAPI) on `127.0.0.1:8000` in the background
2. Waits up to 90 seconds for the backend to become healthy
3. Launches `streamlit run` on `0.0.0.0:7860` in the background
4. Monitors both — if either exits, kills the other and exits (so the platform restarts the container)

---

## 6. How FastAPI (Backend) Works

### 6.1 Entry Point: `backend/app/main.py`

The FastAPI app is the **central API server**. It:

1. **Initializes** the database on startup (`init_db()`)
2. **Configures CORS** middleware
3. **Includes** three routers: `auth`, `workspaces`, `inspect`
4. **Defines all core endpoints** directly in `main.py`

### 6.2 API Endpoint Groups

#### Health Check
- `GET /api/health` — Reports database status, sandbox configuration, API key readiness

#### Datasets
- `POST /api/datasets/upload` — Accepts CSV file → saves to disk → profiles with Pandas → stores in DB
- `GET /api/datasets` — List all datasets for the active team
- `GET /api/datasets/{id}` — Get one dataset with its profile

#### Questions (Analysis Runs)
- `POST /api/questions` — Creates a question, kicks off `run_question_graph()` in a **background thread**
- `GET /api/questions` — List all questions (recent first, with dashboard info)
- `GET /api/questions/{id}/status` — Poll current stage, step progress, retry counts
- `GET /api/questions/{id}/dashboard` — Fetch the completed dashboard
- `GET /api/questions/{id}/audit-trail` — Fetch the full audit trail
- `GET /api/questions/{id}/critic-reviews` — Fetch independent critic reviews

#### Auth (in `routers/auth.py`)
- `POST /api/auth/signup` — Create account, claim pending invites, issue JWT
- `POST /api/auth/login` — Verify credentials, issue JWT
- `GET /api/auth/me` — Return current user info

#### Workspaces (in `routers/workspaces.py`)
- `GET /api/communities` — List user's communities
- `POST /api/communities` — Create a community
- `GET/POST /api/communities/{id}/teams` — List/create teams
- `GET /api/me/workspaces` — Nested listing for sidebar
- `POST /api/teams/{id}/invite` — Invite by email
- `POST /api/teams/{id}/accept-invite` — Accept a pending invite
- `GET /api/teams/{id}/members` — List team members

#### Inspect (in `routers/inspect.py`)
- `GET /api/dashboards/{id}/elements/{element_id}/inspect` — Returns the exact code, formula explanation, and data slice for one dashboard element

### 6.3 How a Question is Launched

```python
# In main.py POST /api/questions:
question = Question(team_id=team.id, dataset_id=payload.dataset_id,
                    text=payload.question, status="running", current_stage="queued")
db.add(question)
db.commit()

# Launch the LangGraph pipeline in a BACKGROUND THREAD
thread = threading.Thread(target=run_question_graph, args=(question.id,), daemon=True)
thread.start()
```

The frontend then **polls** `GET /api/questions/{id}/status` every ~2 seconds to show live progress.

---

## 7. How Streamlit (Frontend) Works

### 7.1 Architecture

Streamlit is a **multi-page application** with:
- `Home.py` — The root entry point / landing page
- `pages/` — 9 additional pages, numbered for sidebar ordering
- `style/theme.py` — A complete custom dark design system (847 lines of CSS + helpers)
- `api_client.py` — A thin `requests`-based HTTP client that talks to FastAPI
- `auth.py` — Shared-password gate + per-user login/signup flow
- `config.py` — Frontend configuration (backend URL)

### 7.2 Page-by-Page Breakdown

| Page | File | Purpose |
|---|---|---|
| **Home** | `Home.py` | Hero landing page with live workspace snapshot (datasets, analyses, verified counts) |
| **Overview** | `1_Overview.py` | Workspace dashboard with counters, recent analyses list, insights chart, recent dashboards |
| **Datasets** | `2_Datasets.py` | Upload CSVs, browse datasets, view auto-generated profiles (column types, stats, missing values) |
| **Analyses** | `3_Analyses.py` | Ask a new question + **live Analysis Flow viewer** showing plan steps, code execution, progress donut |
| **Dashboards** | `4_Dashboards.py` | Display verified KPI cards, Plotly charts, narrative summary, verification badge, click-to-inspect |
| **Reports** | `5_Reports.py` | Download plain-text analysis reports bundling KPIs + narrative + verification |
| **Audit Trail** | `6_Audit_Trail.py` | Full traceability with 4 tabs: Overview, Code & Logs, Data Lineage, Verifications |
| **Settings** | `7_Settings.py` | Account info, workspace config, model status (Gemini/Llama key configured), system health |
| **Workspaces** | `8_Workspaces.py` | Create communities/teams, switch workspaces, invite members by email |
| **Team Overview** | `9_Team_Overview.py` | Team information page |

### 7.3 Frontend → Backend Communication

All communication happens through `api_client.py`:

```python
# api_client.py — every call includes auth + team context automatically
def _headers(with_team=True):
    headers = {}
    token = st.session_state.get("auth_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if with_team:
        team_id = st.session_state.get("active_team_id")
        if team_id:
            headers["X-Team-Id"] = str(team_id)
    return headers

# Example: upload a dataset
def upload_dataset(filename, file_bytes):
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{BACKEND_BASE_URL}/api/datasets/upload",
                         files=files, headers=_headers(), timeout=30)
    return _handle(resp)
```

### 7.4 Live Analysis Viewer

The **Analyses** page (`3_Analyses.py`) uses Streamlit's `@st.fragment(run_every="2s")` to auto-poll the backend for status updates:

```python
@st.fragment(run_every="2s")
def _render_live_run(qid: int):
    status = get_status(qid)  # polls GET /api/questions/{id}/status

    # Shows: Analysis Plan steps, Code Execution panel, Progress Donut
    # All update in real-time as the agent pipeline progresses
```

This shows:
1. **Plan panel** — list of steps with checkmarks (done/running/pending)
2. **Code panel** — the latest executed Python code
3. **Progress donut** — percentage completion ring
4. **Stage indicator** — "Planning analysis", "Executing code", "Verifying results", etc.

### 7.5 The Design System (`style/theme.py`)

This is an 847-line module providing:
- **`BASE_CSS`** — Complete dark theme CSS with CSS variables for colors, borders, typography
- **`page_setup()`** — Configures Streamlit page, injects CSS, handles auth gates
- **`render_sidebar()`** — Draws the navigation sidebar with workspace switcher
- **`html()`** — Helper to inject raw HTML with `unsafe_allow_html=True`
- **`stat_card()`** — Stat counter card component
- **`badge()`** — Colored status badge (verified/warn/error/running/neutral)
- **`plot()`** — Renders Plotly figures with the app's dark chart theme
- **`figure_from_json()`** — Converts stored Plotly JSON back to interactive figures
- **`open_inspect()` / `render_inspect_dialog_if_open()`** — Click-to-inspect modal for dashboard elements

Color palette:
- Background: `#161616` (deep black)
- Text: `#e6e6e6` (light gray)
- Accent: `#4f8fe0` (blue)
- Chart colorway: `["#4f8fe0", "#3fb87a", "#e0a83e", "#d1596b", "#9575cd", "#41b8c4"]`

---

## 8. How LangGraph Is Used — The AI Agent Pipeline

### 8.1 What is LangGraph?

LangGraph is a framework for building **stateful, multi-step AI agent workflows** as a directed graph (state machine). It manages:
- The flow between nodes (Triage → Planner → Executor → Critic → Dashboard)
- Conditional routing (retry loops, revision loops)
- Shared state across all nodes
- Bounded recursion to prevent infinite loops

### 8.2 The Agent State (`state.py`)

All nodes share one `AgentState` TypedDict:

```python
class AgentState(TypedDict, total=False):
    # Input
    question_id: int
    dataset_id: int
    question_text: str
    csv_path: str
    profile: dict[str, Any]         # Dataset column info

    # Plan
    plan: list[dict[str, Any]]      # [{id, description, goal}, ...]
    plan_reasoning: str
    plan_id: int

    # Execution tracking
    step_index: int                 # Current step being executed
    retry_count: int                # Retries on the current step
    last_error: str | None          # Last stderr for retry context
    step_results: list[StepResult]  # Completed step results

    # Critic
    critic_verdict: str | None      # "verified" or "rejected"
    critic_summary: str
    critic_issues: list[str]
    critic_review_id: int | None

    # Revision (Critic → re-plan)
    revision_count: int
    revision_feedback: str | None

    # Output
    dashboard: dict[str, Any] | None
    failed: bool
    failure_reason: str | None
    rejected: bool                  # Triage rejection
    rejection_reason: str | None
    suggestions: list[str]          # Suggested questions
```

### 8.3 Graph Structure (`graph.py`)

```python
def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("triage", triage_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_revision", prepare_revision_node)
    graph.add_node("dashboard", dashboard_node)

    # Wiring
    graph.set_entry_point("triage")

    graph.add_conditional_edges("triage", route_after_triage,
        {"planner": "planner", "rejected": END})

    graph.add_edge("planner", "executor")

    graph.add_conditional_edges("executor", route_after_executor,
        {"executor": "executor",    # loop: more steps to do
         "critic": "critic",        # all steps done → verify
         "failed": END})            # retries exhausted

    graph.add_conditional_edges("critic", route_after_critic,
        {"dashboard": "dashboard",        # verified OR revisions exhausted
         "revise": "prepare_revision"})   # rejected → re-plan

    graph.add_edge("prepare_revision", "planner")  # back to planner
    graph.add_edge("dashboard", END)                # done!

    return graph.compile()
```

### 8.4 Visual Flow Diagram

```
                    START
                      │
                      ▼
               ┌─────────────┐
               │   TRIAGE     │──── rejected ────▶ END (with guidance)
               │ (is question │
               │  analyzable?)│
               └──────┬──────┘
                      │ accepted
                      ▼
               ┌─────────────┐
         ┌────▶│   PLANNER    │
         │     │ (break into  │
         │     │  3-6 steps)  │
         │     └──────┬──────┘
         │            │
         │            ▼
         │     ┌─────────────┐     failure
         │     │  EXECUTOR    │◀──────┐
         │     │ (write code, │       │ retry (up to 3×)
         │     │  run sandbox)│───────┘
         │     └──────┬──────┘
         │            │ all steps done
         │            ▼
         │     ┌─────────────┐
         │     │   CRITIC     │
         │     │ (independent │
         │     │  re-check on │
         │     │  Llama model)│
         │     └──────┬──────┘
         │            │
         │     ┌──────┴──────┐
         │     │             │
         │  rejected      verified
         │  (budget left)    │
         │     │             │
         │     ▼             ▼
         │  PREPARE_      ┌─────────────┐
         │  REVISION       │  DASHBOARD   │
         │     │           │  (compile    │
         │     │           │  KPIs/charts │
         └─────┘           │  /narrative) │
      (re-plan,            └──────┬──────┘
       max 1×)                    │
                                  ▼
                                 END
```

### 8.5 Routing Logic

```python
def route_after_executor(state):
    if state.get("failed"):         return "failed"   # retries exhausted
    if state["step_index"] >= len(state["plan"]):
        return "critic"              # all steps complete → verify
    return "executor"                # more steps to execute

def route_after_critic(state):
    if state.get("critic_verdict") == "verified":
        return "dashboard"           # verified → compile dashboard
    if state.get("revision_count", 0) < MAX_CRITIC_REVISIONS:
        return "revise"              # rejected → re-plan (1 chance)
    return "dashboard"               # revisions exhausted → show with caveats
```

---

## 9. Complete Code Flow: From CSV Upload to Dashboard

### Step 1: CSV Upload & Profiling

**User action:** Upload a CSV file on the Datasets page

**Flow:**
```
Frontend (2_Datasets.py)
  → upload_dataset(filename, bytes) via api_client.py
    → POST /api/datasets/upload (main.py)
      → Save file to backend/data/uploads/{uuid}_{filename}
      → profiling.load_and_profile_csv(filepath)
        → pd.read_csv(path)
        → Auto-detect datetime columns
        → For each column:
            Numeric → min, max, mean, median, std
            Categorical → top 5 value counts
            Datetime → min/max date range
        → Return {row_count, col_count, columns: [...]}
      → Store Dataset row in SQLite
      → Return DatasetProfile to frontend
```

### Step 2: Ask a Question

**User action:** Type a question and click "Run Analysis"

**Flow:**
```
Frontend (3_Analyses.py)
  → ask_question(dataset_id, question_text) via api_client.py
    → POST /api/questions (main.py)
      → Create Question row (status="running", stage="queued")
      → Launch BACKGROUND THREAD: run_question_graph(question_id)
      → Return {question_id, status} immediately
```

### Step 3: LangGraph Pipeline Executes

**In the background thread:**

```
run_question_graph(question_id)
  │
  ├── Load Question, Dataset, Profile from DB
  │
  ├── Build initial AgentState
  │
  └── graph.invoke(initial_state, config={"recursion_limit": 150})
      │
      ├── NODE: triage_node
      │   ├── Check: is question analyzable against this dataset?
      │   ├── Calls Gemini via call_tool(submit_triage)
      │   └── Rejects gibberish, off-topic, or column-mismatch questions
      │
      ├── NODE: planner_node
      │   ├── Calls Gemini via call_tool(submit_plan)
      │   ├── Input: question + dataset profile + any revision feedback
      │   ├── Output: 2-8 ordered analysis steps, each with {id, description, goal}
      │   └── Stores Plan row in DB
      │
      ├── NODE: executor_node (loops for each step)
      │   ├── Calls Gemini via call_tool(submit_code)
      │   ├── Input: question + profile + prior results + current step
      │   ├── Output: Python script {reasoning, formula_explanation, code}
      │   ├── Runs code in sandbox:
      │   │   ├── Copy CSV → workspace/data/input.csv
      │   │   ├── docker run --rm --network none ... (or subprocess)
      │   │   └── Parse stdout for RESULT_JSON: and DATA_SLICE_JSON:
      │   ├── On SUCCESS: save ExecutionLog, advance to next step
      │   ├── On FAILURE: feed error back into LLM → retry (up to 3×)
      │   └── After all steps: route to critic
      │
      ├── NODE: critic_node
      │   ├── Phase 1 — Verification Code:
      │   │   ├── Calls Llama via call_tool(submit_verification_code)
      │   │   ├── Writes its OWN Python script to recompute key numbers
      │   │   └── Runs in sandbox independently
      │   │
      │   ├── Phase 2 — Review:
      │   │   ├── Calls Llama via call_tool(submit_review)
      │   │   ├── Compares executor's results vs its own recomputed results
      │   │   └── Verdict: "verified" or "rejected" with issues list
      │   │
      │   └── Stores CriticReview row in DB
      │
      ├── (If rejected & revision budget remains)
      │   └── NODE: prepare_revision_node → back to planner_node
      │
      └── NODE: dashboard_node
          ├── Calls Gemini via call_tool(compile_dashboard)
          ├── Extracts: KPIs (2-5), charts (up to 4), narrative (3-6 sentences)
          ├── Resolves chart file references → loads Plotly JSON from disk
          ├── Stores Dashboard row in DB
          ├── Stores AuditTrail rows for every KPI, chart, and narrative
          └── Updates question status to "verified" or "unverified"
```

### Step 4: Frontend Displays Results

```
Frontend polls GET /api/questions/{id}/status every 2 seconds
  → Shows live plan step progress
  → Shows executing code
  → Shows progress percentage

When status = "verified" or "unverified":
  → Navigate to Dashboards page
  → GET /api/questions/{id}/dashboard
  → Render KPI cards, Plotly charts, narrative
  → Each element has click-to-inspect (code + formula + data slice)
```

---

## 10. CSV Analysis & Pandas Calculations

### 10.1 Initial Profiling (`profiling.py`)

When a CSV is uploaded, `load_and_profile_csv()` is called:

```python
def load_and_profile_csv(path):
    df = pd.read_csv(path)

    # Auto-detect datetime columns from string columns
    for col in df.columns:
        if is_string_dtype(df[col]):
            sample = df[col].dropna().head(20)
            if pd.to_datetime(sample, errors="coerce").notna().mean() > 0.8:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df, profile_dataframe(df)
```

**`profile_dataframe(df)`** produces:
```python
{
    "row_count": 1000,
    "col_count": 8,
    "columns": [
        {
            "name": "revenue",
            "dtype": "float64",
            "kind": "numeric",
            "missing_count": 5,
            "missing_pct": 0.5,
            "unique_count": 950,
            "stats": {"min": 100.0, "max": 50000.0, "mean": 12500.0, "median": 11000.0, "std": 8500.0}
        },
        {
            "name": "region",
            "dtype": "object",
            "kind": "categorical",
            "missing_count": 0,
            "missing_pct": 0.0,
            "unique_count": 4,
            "top_values": [
                {"value": "North", "count": 300},
                {"value": "South", "count": 280},
                ...
            ]
        },
        {
            "name": "date",
            "dtype": "datetime64[ns]",
            "kind": "datetime",
            "missing_count": 0,
            "missing_pct": 0.0,
            "unique_count": 365,
            "stats": {"min": "2024-01-01", "max": "2024-12-31"}
        }
    ]
}
```

### 10.2 Agent-Generated Analysis Code (Sandbox Contract)

The AI generates Python scripts that follow a strict contract:

```python
# Every generated script must:
import pandas as pd
import numpy as np
import plotly.express as px
import json

# 1. Load the data
df = pd.read_csv("/workspace/data/input.csv")

# 2. Perform the calculation (example: monthly revenue trend)
df['date'] = pd.to_datetime(df['date'])
monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
growth = monthly.pct_change() * 100

# 3. Output structured result
result = {
    "total_revenue": float(df['revenue'].sum()),
    "avg_monthly_revenue": float(monthly.mean()),
    "worst_month": str(growth.idxmin()),
    "worst_decline_pct": float(growth.min())
}
print("RESULT_JSON:" + json.dumps(result))

# 4. Output the data slice used (for audit trail)
slice_df = monthly.reset_index()
print("DATA_SLICE_JSON:" + json.dumps({
    "columns": list(slice_df.columns),
    "rows": slice_df.values.tolist()[:50]
}))

# 5. Save a chart
fig = px.line(monthly.reset_index(), x='date', y='revenue', title='Monthly Revenue Trend')
fig.write_json("/workspace/output/monthly_revenue.json")
```

---

## 11. The Results Pipeline — What Produces Results

### Who produces what:

| Component | Produces | Stored In |
|---|---|---|
| **profiling.py** | Dataset column stats, types, missing values | `datasets.profile_json` |
| **Planner** (Gemini) | Analysis plan (ordered list of steps with goals) | `plans.steps_json` |
| **Executor** (Gemini → Sandbox) | Python code + execution output (RESULT_JSON, charts) | `execution_logs` (code, stdout, result_json, chart_paths_json) |
| **Critic** (Llama → Sandbox) | Independent verification code + verdict | `critic_reviews` (verdict, issues_json, checks_json) |
| **Dashboard** (Gemini) | KPIs, chart selections, narrative | `dashboards` (kpis_json, charts_json, narrative) |
| **Audit module** | Element-to-source traceability links | `audit_trail` rows |

### The data flow through the pipeline:

```
CSV File
  ↓
[profiling.py] → Profile Dict (column stats)
  ↓ (profile passed to all agents)
[Planner] → Plan (3-6 steps)
  ↓
[Executor Step 1] → Code → Sandbox → Result + Charts
[Executor Step 2] → Code → Sandbox → Result + Charts
[Executor Step 3] → Code → Sandbox → Result + Charts
  ↓ (all step results collected)
[Critic] → Own verification code → Sandbox → Recomputed values → Verdict
  ↓
[Dashboard] → Compile KPIs from step results, pick best charts, write narrative
  ↓
[Audit] → Link each KPI/chart/narrative back to its execution_log + critic_review
```

---

## 12. Code Generation — Which Part Writes Python Code

### 12.1 The Executor Node (`executor.py`)

This is where **AI-generated Python code** is produced:

```python
def executor_node(state):
    step = state["plan"][state["step_index"]]

    # Ask Gemini to write a Python script
    output = llm_client.call_tool(
        system=prompts.EXECUTOR_SYSTEM,   # sandbox contract instructions
        user_content=f"""
        Business question: {state['question_text']}
        Dataset profile: {state['profile']}
        Prior step results: {_prior_results_summary(state['step_results'])}
        Current step: {step['description']}
        Goal: {step['goal']}
        """,
        tool_name="submit_code",
        tool_schema=CODE_TOOL_SCHEMA,     # {reasoning, formula_explanation, code}
    )

    code = output["code"]

    # Execute in sandbox
    runner = get_runner()  # DockerSandboxRunner or SubprocessSandboxRunner
    runner.seed_input_data(question_id, csv_path)  # copy CSV to workspace
    sandbox_result = runner.run(question_id, code, step_index, attempt_number)
    # → SandboxResult {success, stdout, stderr, result, chart_paths, data_slice}
```

### 12.2 The Critic's Verification Code (`critic.py`)

The Critic also writes code, but independently:

```python
def critic_node(state):
    # Phase 1: Write verification code (on Llama, not Gemini!)
    verify_output, verifier_model = _critic_call_tool(
        system=prompts.CRITIC_VERIFY_SYSTEM,
        user_content=f"Executor's claimed results: {_steps_summary(successful_steps)}",
        tool_name="submit_verification_code",
        tool_schema=VERIFY_CODE_TOOL_SCHEMA,
    )

    # Run verification code in its own sandbox
    sandbox_result = runner.run(question_id, verify_output["code"], -1, verify_attempt)

    # Phase 2: Compare executor's results vs critic's recomputed values
    review_output, _ = _critic_call_tool(
        system=prompts.CRITIC_REVIEW_SYSTEM,
        user_content=f"""
        Executor's claims: {_steps_summary(successful_steps)}
        My recomputed values: {sandbox_result.result}
        """,
        tool_name="submit_review",
        tool_schema=REVIEW_TOOL_SCHEMA,  # {verdict, confidence, issues, summary}
    )
```

### 12.3 The Sandbox Contract (in `prompts.py`)

Every generated script must follow this contract:

1. Load data from `/workspace/data/input.csv`
2. Use only `pandas`, `numpy`, `plotly`, `matplotlib`, `json`, `sys`
3. Print `RESULT_JSON:<json>` with structured results
4. Print `DATA_SLICE_JSON:<json>` with the data subset used
5. Save charts with `fig.write_json("/workspace/output/<name>.json")`
6. No network access, no subprocess calls, no file deletion

---

## 13. Audit Report System

### 13.1 How the Audit Trail is Built

The **Dashboard node** creates audit trail entries as the final step:

```python
# In dashboard.py dashboard_node():

# For each KPI:
record_audit_entry(db, question_id,
    element_label="KPI: Revenue Drop %",
    element_type="kpi",
    element_id=kpi["element_id"],       # UUID embedded in dashboard JSON
    reasoning="Value '23.4%' from step 2. Critic: verified",
    execution_log_id=log_id_by_step[2], # links to the exact code that computed it
    critic_review_id=critic_review_id,   # links to the critic's verification
    team_id=team_id,
)

# For each chart:
record_audit_entry(db, question_id,
    element_label="Chart: Monthly Revenue Trend",
    element_type="chart",
    element_id=chart["element_id"],
    reasoning="Generated by step 1 code. Critic: verified",
    execution_log_id=log_id_by_step[1],
    critic_review_id=critic_review_id,
    team_id=team_id,
)

# For the narrative:
record_audit_entry(db, question_id,
    element_label="Narrative summary",
    element_type="narrative",
    reasoning="Synthesized from all successful step results. Critic: verified",
    execution_log_id=None,
    critic_review_id=critic_review_id,
    team_id=team_id,
)
```

### 13.2 What the Audit Trail Contains

Each `audit_trail` row links:

```
Dashboard Element (KPI / Chart / Narrative)
    │
    ├── element_id ──────── matches the UUID in dashboards.kpis_json or charts_json
    │                        (enables click-to-inspect from the dashboard)
    │
    ├── execution_log_id ── links to execution_logs row:
    │                        • code (the exact Python script)
    │                        • stdout (raw output)
    │                        • result_json (structured RESULT_JSON)
    │                        • formula_explanation (plain-English calculation)
    │                        • data_slice_json (the actual data rows used)
    │
    └── critic_review_id ── links to critic_reviews row:
                             • verdict ("verified" / "rejected")
                             • confidence (0.0 to 1.0)
                             • issues_json (specific problems found)
                             • checks_json (independent re-check code + results)
                             • summary (2-4 sentence explanation)
```

### 13.3 Click-to-Inspect

On the Dashboard page, clicking a KPI card triggers:

```
GET /api/dashboards/{id}/elements/{element_id}/inspect
  → Find audit_trail row by element_id
  → Load execution_log → return {code, formula_explanation, data_slice}
  → Frontend displays: Formula, Code, and Data Table in a modal
```

### 13.4 Audit Trail Page (`6_Audit_Trail.py`)

Four tabs:
1. **Overview** — Analysis plan steps + dashboard elements list
2. **Code & Logs** — Browse executed code, stdout, results for each step, with critic verdict
3. **Data Lineage** — How each element traces back through code to the source CSV
4. **Verifications** — Critic reviews with independent re-check code and recomputed values

### 13.5 Reports Page (`5_Reports.py`)

Generates downloadable plain-text reports:

```
REPORT — Why did revenue drop in Q3?
========================================================================
Analysis ID : 42
Status      : verified
Generated   : 2h ago

KPIs
------------------------------------------------------------------------
  Revenue Drop: 23.4%
  Q3 Revenue: $4.2M
  Worst Month: August

NARRATIVE
------------------------------------------------------------------------
Revenue declined 23.4% in Q3, driven primarily by a 31% drop in the
North region during August...

VERIFICATION
------------------------------------------------------------------------
The Critic independently confirmed the 23.4% decline figure...

CHARTS
------------------------------------------------------------------------
  - Monthly Revenue Trend (from step 1)
  - Regional Breakdown (from step 2)
```

---

## 14. Dashboard Generation — How Dashboards Are Built

### 14.1 Dashboard Compile Node (`dashboard.py`)

After the Critic verifies (or revision budget is exhausted), the Dashboard node runs:

```python
def dashboard_node(state):
    # Collect all successful step results
    step_results = [r for r in state["step_results"] if r["success"]]

    # Ask Gemini to compile the dashboard
    output = llm_client.call_tool(
        system=prompts.DASHBOARD_SYSTEM,
        user_content=f"""
        Business question: {state['question_text']}
        Step results: {_steps_for_prompt(step_results)}
        Critic verdict: {state.get('critic_verdict')}
        Critic summary: {state.get('critic_summary')}
        """,
        tool_name="compile_dashboard",
        tool_schema=COMPILE_TOOL_SCHEMA,
    )
```

### 14.2 Dashboard Schema

The LLM returns:

```json
{
    "kpis": [
        {"label": "Revenue Drop", "value": "23.4%", "source_step_index": 2},
        {"label": "Worst Month", "value": "August 2024", "source_step_index": 1}
    ],
    "charts": [
        {"step_index": 1, "file": "monthly_revenue.json", "title": "Monthly Revenue Trend"},
        {"step_index": 3, "file": "regional_breakdown.json", "title": "Revenue by Region"}
    ],
    "narrative": "Revenue declined 23.4% in Q3 compared to Q2, driven primarily by..."
}
```

### 14.3 Chart Resolution

The Dashboard node resolves chart file references to actual Plotly JSON:

```python
# Each chart reference points to a file the Executor wrote in the sandbox
for chart in output["charts"]:
    match = find_chart_file(step_results, chart["step_index"], chart["file"])
    if match and os.path.exists(match):
        with open(match) as f:
            plotly_json = json.load(f)  # Full Plotly figure specification
        resolved_charts.append({
            "title": chart["title"],
            "plotly_json": plotly_json,  # Stored in DB, rendered in frontend
        })
```

### 14.4 Frontend Rendering (`4_Dashboards.py`)

```python
# KPI Cards
kpis = dash.get("kpis", [])
cols = st.columns(min(4, len(kpis)))
for col, kpi in zip(cols, kpis):
    with col:
        # Click to inspect → opens modal with code + formula + data
        if st.button(f"{kpi['label']}\n**{kpi['value']}**"):
            open_inspect(dashboard_id, kpi["element_id"])

# Charts (Plotly interactive)
for chart in dash["charts"]:
    fig = figure_from_json(chart["plotly_json"])  # Reconstruct Plotly figure
    plot(fig, height=300)  # Renders with dark theme applied

# Narrative
st.markdown(dash["narrative"])

# Verification badge
if dash["verified"]:
    badge("Verified", "verified")
else:
    badge("Needs review", "warn")
```

---

## 15. Database Schema & Models

### 15.1 All Tables (`models.py`)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    users      │     │ communities  │     │    teams     │
├──────────────┤     ├──────────────┤     ├──────────────┤
│ id (PK)       │     │ id (PK)      │     │ id (PK)      │
│ email (unique)│     │ name         │     │ community_id │
│ password_hash │     │ created_by   │     │ name         │
│ display_name  │     │ created_at   │     │ created_by   │
│ created_at    │     └──────────────┘     │ created_at   │
└──────────────┘                           └──────────────┘
                                                  │
                       ┌──────────────────────────┘
                       ▼
                ┌──────────────┐
                │ team_members │
                ├──────────────┤
                │ id (PK)      │
                │ team_id (FK) │
                │ user_id (FK) │
                │ invited_email│
                │ role         │  (owner / admin / member)
                │ status       │  (active / pending)
                │ joined_at    │
                └──────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   datasets    │────▶│  questions   │────▶│    plans     │
├──────────────┤     ├──────────────┤     ├──────────────┤
│ id (PK)       │     │ id (PK)      │     │ id (PK)      │
│ team_id (FK)  │     │ team_id (FK) │     │ question_id  │
│ filename      │     │ dataset_id   │     │ steps_json   │
│ filepath      │     │ text         │     │ reasoning    │
│ row_count     │     │ status       │     │ revision     │
│ col_count     │     │ current_stage│     │ created_at   │
│ profile_json  │     │ stage_detail │     └──────────────┘
│ uploaded_at   │     │ error        │
└──────────────┘     │ created_at   │
                      └──────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │execution_logs│  │critic_reviews│  │  dashboards   │
    ├──────────────┤  ├──────────────┤  ├──────────────┤
    │ id (PK)      │  │ id (PK)      │  │ id (PK)      │
    │ question_id  │  │ question_id  │  │ team_id      │
    │ step_index   │  │ verdict      │  │ question_id  │
    │ step_desc    │  │ confidence   │  │ kpis_json    │
    │ attempt_num  │  │ issues_json  │  │ charts_json  │
    │ code         │  │ checks_json  │  │ narrative    │
    │ stdout       │  │ summary      │  │ verified     │
    │ stderr       │  │ created_at   │  │ verif_summary│
    │ success      │  └──────────────┘  │ created_at   │
    │ result_json  │                     └──────────────┘
    │ chart_paths  │
    │ reasoning    │         ┌──────────────┐
    │ formula_expl │         │ audit_trail   │
    │ data_slice   │         ├──────────────┤
    │ created_at   │         │ id (PK)      │
    └──────────────┘         │ team_id      │
                             │ question_id  │
                             │ element_label│
                             │ element_type │
                             │ element_id   │
                             │ exec_log_id  │──▶ execution_logs
                             │ critic_rev_id│──▶ critic_reviews
                             │ reasoning    │
                             │ created_at   │
                             └──────────────┘
```

### 15.2 Question Status Values

| Status | Meaning |
|---|---|
| `pending` | Just created, not yet started |
| `running` | LangGraph pipeline is active |
| `verified` | Critic verified, dashboard ready |
| `unverified` | Dashboard ready but critic had concerns |
| `failed` | Pipeline failed after all retries |
| `rejected` | Triage rejected the question |

### 15.3 Question Stages (current_stage)

| Stage | Meaning |
|---|---|
| `queued` | Waiting to start |
| `triage` | Checking if question is analyzable |
| `planning` | Planner is creating analysis steps |
| `executing` | Executor is running code for each step |
| `critic` | Critic is independently verifying |
| `dashboard` | Dashboard is being compiled |
| `done` | Complete |
| `failed` | Failed |
| `rejected` | Triage rejected |

---

## 16. Authentication & Multi-Tenancy

### 16.1 Two-Layer Auth

1. **Shared Password Gate** (`auth.py` → `require_password()`)
   - Set `APP_PASSWORD` env var for deployed instances
   - Simple "don't let strangers burn my API quota" control
   - Disabled when `APP_PASSWORD` is empty (local dev)

2. **Per-User Login/Signup** (`auth.py` → `require_login()`)
   - Email + password → JWT token (bcrypt hashed, HS256 signed)
   - Token stored in `st.session_state["auth_token"]`
   - Sent as `Authorization: Bearer <token>` header

### 16.2 Multi-Tenancy Model

```
Community (e.g., "Acme Inc.")
  └── Team (e.g., "Growth Team")
        └── Members (owner / admin / member)

- Each team is an isolated workspace
- Datasets, questions, dashboards are scoped to a team via team_id
- Every API call includes X-Team-Id header
- Backend verifies membership before allowing access
```

### 16.3 Team Isolation

```python
# In security.py:
def get_current_team(x_team_id, user, db):
    team = db.get(Team, x_team_id)
    # Verify the user is an active member of this team
    member = db.query(TeamMember).filter(
        TeamMember.team_id == team.id,
        TeamMember.user_id == user.id,
        TeamMember.status == "active"
    ).first()
    if member is None:
        raise HTTPException(403, "Not a member of this team")
    return team
```

---

## 17. Security & Sandboxing

### 17.1 Docker Sandbox Security

```
docker run --rm \
    --network none \             # ❌ No network access
    --memory 512m \              # Memory cap
    --cpus 1 \                   # CPU cap
    -v {workspace}:/workspace \  # Only the workspace is mounted
    ai-data-analyst-sandbox:latest \
    /workspace/step0_attempt1.py
```

- Runs as `sandboxuser` (non-root, uid 1000)
- Fresh container per attempt
- No API keys in environment
- No access to app.db or uploads directory

### 17.2 Subprocess Fallback Security

When Docker is unavailable (e.g., PaaS deployments):

1. **Scrubbed environment** — `_sandbox_env()` builds env from scratch; `GEMINI_API_KEY` and `LLAMA_API_KEY` are never present
2. **Dropped privileges** — Runs as unprivileged `sandbox` user
3. **Resource caps** — `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`, no core dumps
4. **Process isolation** — Own process session, timeout kills grandchildren too
5. **File isolation** — `/app/data` is `chmod 700` root-owned; sandbox user can't traverse to it

### 17.3 API Security

- CORS configurable via `CORS_ALLOWED_ORIGINS`
- Upload size limit: `MAX_UPLOAD_BYTES` (200 MB default)
- Rate limiting on signup (10/min per IP) and login (8/min per IP+email)
- JWT expiry: 7 days (configurable)

---

## 18. Deployment Architecture

### 18.1 Hugging Face Spaces

```yaml
# Metadata in README.md YAML frontmatter
sdk: docker
app_port: 7860
```

- Single Docker container
- `start.sh` boots both processes
- FastAPI on loopback, Streamlit on 7860
- Secrets: `GEMINI_API_KEY`, `LLAMA_API_KEY`, `APP_PASSWORD`

### 18.2 Render.com

```yaml
# render.yaml
services:
  - type: web
    name: ai-data-analyst
    env: docker
    plan: free
    envVars:
      - key: GEMINI_API_KEY
        sync: false    # prompted at deploy time
      - key: JWT_SECRET_KEY
        generateValue: true  # auto-generated
```

### 18.3 Local Development

```bash
# Terminal 1 — Backend
uvicorn app.main:app --app-dir backend --reload --port 8000

# Terminal 2 — Frontend
streamlit run frontend/Home.py
```

---

## 19. API Endpoints Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | System health check |
| `POST` | `/api/auth/signup` | Create account |
| `POST` | `/api/auth/login` | Login, get JWT |
| `GET` | `/api/auth/me` | Current user |
| `POST` | `/api/datasets/upload` | Upload + profile CSV |
| `GET` | `/api/datasets` | List datasets |
| `GET` | `/api/datasets/{id}` | Get dataset + profile |
| `POST` | `/api/questions` | Start analysis run |
| `GET` | `/api/questions` | List all analyses |
| `GET` | `/api/questions/{id}/status` | Poll run progress |
| `GET` | `/api/questions/{id}/dashboard` | Get dashboard data |
| `GET` | `/api/questions/{id}/audit-trail` | Get audit trail |
| `GET` | `/api/questions/{id}/critic-reviews` | Get critic reviews |
| `GET` | `/api/dashboards/{id}/elements/{eid}/inspect` | Click-to-inspect |
| `GET` | `/api/communities` | List communities |
| `POST` | `/api/communities` | Create community |
| `GET` | `/api/communities/{id}/teams` | List teams |
| `POST` | `/api/communities/{id}/teams` | Create team |
| `GET` | `/api/me/workspaces` | Nested community/team listing |
| `GET` | `/api/me/invites` | Pending invites |
| `POST` | `/api/teams/{id}/invite` | Invite member |
| `POST` | `/api/teams/{id}/accept-invite` | Accept invite |
| `GET` | `/api/teams/{id}/members` | List team members |

---

## 20. Frontend Pages Reference

| # | Page | Key Features |
|---|---|---|
| — | **Home** | Hero text, backend status dot, live workspace snapshot (datasets/analyses/verified counts), recent activity |
| 1 | **Overview** | Greeting with time-of-day, 4 stat counters, recent analyses list with badges, insights-by-weekday sparkline chart, recent dashboard cards |
| 2 | **Datasets** | CSV upload with file picker, dataset table (name, rows, cols, quality %), dataset profile inspector (per-column type, missing, stats, top values) |
| 3 | **Analyses** | Dataset picker + question input → "Run Analysis"; Live run viewer: plan steps panel, code execution panel with syntax-highlighted Python, progress donut, stage labels; Previous analyses list |
| 4 | **Dashboards** | Dashboard picker; KPI cards (clickable → inspect modal); Plotly charts with dark theme; Narrative summary; Verification badge + summary |
| 5 | **Reports** | List of completed analyses; "Prepare" button → "Download" button; Plain-text report with KPIs + narrative + verification |
| 6 | **Audit Trail** | 4 tabs: Overview (plan + elements), Code & Logs (step picker → code + result + stdout + critic verdict), Data Lineage (element → source tracing), Verifications (critic reviews with independent code) |
| 7 | **Settings** | Account info, active workspace, workspace config (sandbox type, agent autonomy), model status (Gemini/Llama key configured), system health |
| 8 | **Workspaces** | Community list + create; Team list + switch + create; Team members list; Invite by email form |
| 9 | **Team Overview** | Team information display |

---

## 21. Configuration Reference

All configuration is via environment variables (see `.env.example`):

### LLM Keys
| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Planner / Executor / Dashboard |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Which Gemini model |
| `LLAMA_API_KEY` | (optional) | Critic agent (cross-model verification) |
| `LLAMA_MODEL` | `Llama-4-Maverick-17B-128E-Instruct-FP8` | Which Llama model |
| `LLAMA_BASE_URL` | `https://api.llama.com/compat/v1/` | Llama endpoint |

### Sandbox
| Variable | Default | Purpose |
|---|---|---|
| `SANDBOX_BACKEND` | `docker` | `docker` or `subprocess` |
| `SANDBOX_IMAGE` | `ai-data-analyst-sandbox:latest` | Docker image name |
| `SANDBOX_TIMEOUT_SECONDS` | `60` | Max execution time |
| `SANDBOX_MEMORY_LIMIT` | `512m` | Docker memory cap |
| `SANDBOX_CPU_LIMIT` | `1` | Docker CPU cap |

### Agent Loop
| Variable | Default | Purpose |
|---|---|---|
| `MAX_EXECUTOR_RETRIES` | `3` | Max retry attempts per step |
| `MAX_CRITIC_REVISIONS` | `1` | Max re-plan attempts after critic rejection |

### Auth & Security
| Variable | Default | Purpose |
|---|---|---|
| `APP_PASSWORD` | (empty = disabled) | Shared deployment gate password |
| `JWT_SECRET_KEY` | (auto-generated) | JWT signing key |
| `JWT_EXPIRE_MINUTES` | `10080` (7 days) | Token expiry |
| `CORS_ALLOWED_ORIGINS` | (empty = *) | Allowed browser origins |
| `MAX_UPLOAD_BYTES` | `209715200` (200 MB) | Upload size limit |

---

## 22. End-to-End Walkthrough Example

**Scenario:** A user wants to know "Why did revenue drop in Q3?" from a sales CSV.

### Step 1: Sign Up & Create Workspace
```
User → Frontend (auth.py) → POST /api/auth/signup
  → User created, JWT issued
  → Create community "My Company" → Create team "Analytics"
  → Team ID stored in session_state
```

### Step 2: Upload CSV
```
User uploads "sales_2024.csv" (10,000 rows, 12 columns)
  → POST /api/datasets/upload
    → File saved to backend/data/uploads/abc123_sales_2024.csv
    → profiling.py analyzes:
        • revenue (numeric): min=$50, max=$50K, mean=$12.5K
        • region (categorical): North(3000), South(2800), East(2200), West(2000)
        • date (datetime): 2024-01-01 → 2024-12-31
        • ... 9 more columns
    → Dataset row created in DB
    → Profile displayed on Datasets page
```

### Step 3: Ask the Question
```
User types: "Why did revenue drop in Q3?"
  → POST /api/questions {"dataset_id": 1, "question": "Why did revenue drop in Q3?"}
  → Question row created (status=running, stage=queued)
  → Background thread launched: run_question_graph(question_id=42)
```

### Step 4: Triage (instant)
```
triage_node checks:
  ✅ "Why did revenue drop in Q3?" → genuine analytical question
  ✅ Dataset has "revenue" and "date" columns → analyzable
  → rejected: False
```

### Step 5: Planner (~5 seconds)
```
planner_node calls Gemini:
  "Break down 'Why did revenue drop in Q3?' into analysis steps"

Plan produced:
  1. Compute quarterly revenue totals and Q3 vs Q2 percentage change
  2. Break down Q3 revenue by region to identify underperformers
  3. Analyze monthly trend within Q3 to find the worst month
  4. Compare top products in Q3 vs Q2 for contribution shifts
  5. Summarize root causes and produce a final chart

Plan stored in plans table.
```

### Step 6: Executor (~30 seconds, 5 steps)
```
For each of the 5 steps:
  executor_node calls Gemini → "Write Python code for step N"
  → Returns code like:
      df = pd.read_csv("/workspace/data/input.csv")
      df['date'] = pd.to_datetime(df['date'])
      q3 = df[df['date'].dt.quarter == 3]['revenue'].sum()
      q2 = df[df['date'].dt.quarter == 2]['revenue'].sum()
      change = (q3 - q2) / q2 * 100
      print("RESULT_JSON:" + json.dumps({"q3_revenue": q3, "q2_revenue": q2, "drop_pct": change}))

  → Code runs in Docker sandbox (no network, 512MB RAM, 60s timeout)
  → RESULT_JSON parsed from stdout
  → Chart files collected from /workspace/output/
  → ExecutionLog row stored in DB

  If a step fails (e.g., wrong column name):
    → stderr fed back to Gemini → "Fix the root cause"
    → Retry up to 3 times
```

### Step 7: Critic (~15 seconds)
```
critic_node calls Llama (different model!):
  Phase 1: "Write your OWN code to recompute the key numbers"
    → Llama writes a different script that also calculates Q3 vs Q2
    → Runs in sandbox independently
    → Gets its own RESULT_JSON

  Phase 2: "Compare executor's claims vs your recomputed values"
    → Llama compares: executor said -23.4%, critic found -23.2%
    → Within rounding tolerance → verdict: "verified"
    → confidence: 0.92
    → issues: []
    → summary: "Revenue drop figure confirmed within 0.2pp..."

CriticReview row stored in DB.
```

### Step 8: Dashboard Compile (~5 seconds)
```
dashboard_node calls Gemini:
  Input: all step results + critic verdict
  Output:
    KPIs: [
      {"label": "Q3 Revenue Drop", "value": "-23.4%", "source_step_index": 0},
      {"label": "Worst Region", "value": "North (-31%)", "source_step_index": 1},
      {"label": "Worst Month", "value": "August", "source_step_index": 2},
    ]
    Charts: [
      monthly_revenue.json (step 2),
      regional_breakdown.json (step 1),
    ]
    Narrative: "Revenue declined 23.4% in Q3 compared to Q2, driven by..."

Dashboard row stored in DB.
3 KPI audit_trail rows + 2 chart audit_trail rows + 1 narrative row = 6 audit entries.
Question status → "verified".
```

### Step 9: User Views Results
```
Frontend stops polling, shows:
  ✅ 3 KPI cards (clickable → see code + formula + data)
  ✅ 2 interactive Plotly charts (dark themed)
  ✅ Narrative summary
  ✅ "Verified" badge with critic's explanation
  ✅ Link to Audit Trail page
  ✅ Link to download report
```

---

> **This document covers every folder, every file, every code flow, every technology interaction, and every feature of the AI Data Analyst project.**
