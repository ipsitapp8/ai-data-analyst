# DataSage / SILT — Full Technical Audit and Study Guide

An explanation of every concept used in this project, from basics to advanced, tied to the exact file where it appears.

**How this was made.** I read the README, the three `docs/*.md` design documents, the ORM models, `main.py`, config, database, security/authz, the whole agent graph (router, triage, planner, executor, critic, dashboard compile/verify/publish), the LLM clients, memory/knowledge (RAG), jobs, runtime, verification, verdict, fast path, the sandbox runner, the auth router, schemas, the route list of every router, deployment files, CI, and the frontend entry points. For the big modules I did not read line by line (`scheduler.py`, `copilot.py`, `semantic.py`, `metric_engine.py`, `data_quality.py`, `provenance.py`, `scenarios.py`, `investigator.py`, `hypothesis_engine.py`, `analysis_ops.py`, `stats/*`, the 18 Streamlit pages) I used their module docstrings plus `docs/`. Where that is the case, the text says so. I ran nothing: no tests, no app. Section 19 lists findings from reading, not from running.

**How to read it.** Sections go roughly basics → advanced. Every section starts with "the idea" in plain words, then "where it is in this repo". Section 20 is an A–Z glossary for quick lookup.

---

## 0. Contents

1. The project in one page
2. What happens when you ask a question (end to end)
3. Repo map
4. Tech stack — every library and why
5. Web and API fundamentals used here
6. API design of this project
7. Authentication and authorization
8. Database: concepts, then the full schema
9. Migrations and the two database modes
10. Gen-AI fundamentals (basic terms)
11. Gen-AI as used here (advanced: agents, tool calling, failover, RAG, injection defences)
12. The agent pipeline node by node (LangGraph)
13. The no-model paths: router, fast path, metric engine, semantic layer
14. The sandbox (running AI-written code safely)
15. Verification, evidence and verdicts
16. Durable jobs (the queue inside the database)
17. Hypothesis engine and statistics
18. The other platform features (data quality, monitoring, copilot, what-if, provenance, evals)
19. Frontend (Streamlit)
20. Deployment, configuration, CI, tests
21. Audit findings (strengths, risks, gaps)
22. A–Z glossary

---

## 1. The project in one page

**What it is.** A web app where a team uploads a CSV, asks a business question in English ("why did revenue drop in March?"), and gets back a **dashboard** (KPI cards, charts, a written summary). Its selling point is trust: every number can be traced to the code that made it, and a **second, independent check** tries to catch mistakes before you see them.

**The core trick.** Language models are good at *writing plans and code* and bad at *arithmetic and facts*. So the system lets the model write Python, runs that Python in a locked box, and then checks the numbers with separate code. The model never gets to simply *assert* a number.

**Three layers of defence against wrong answers:**
1. **Critic agent** — a *different* AI model (Llama) writes its own script to recompute the key numbers from the raw CSV and then judges the first model's work.
2. **Deterministic verification** — plain code (no AI) checks that every KPI value really appears in a computed result, that every number in the written summary is supported, that percentages are within 0–100, etc.
3. **Provenance** — the CSV's SHA-256 hash is recorded so you can tell if the data changed after the analysis.

**Cheaper paths.** Simple questions ("total revenue by region") skip the AI entirely and are answered by a small formula engine. "Why did X change" questions go to a *hypothesis engine* where the AI only proposes ideas and code tests them.

**Multi-user.** Organizations ("communities") → teams → members with roles; everything is scoped by team.

---

## 2. What happens when you ask a question (end to end)

This is the single most useful thing to understand. Follow the numbers.

1. **Browser → Streamlit.** You type a question on the *Analyses* page (`frontend/pages/3_Analyses.py`).
2. **Streamlit → FastAPI.** `frontend/api_client.py` sends `POST /api/questions` with `Authorization: Bearer <JWT>` and `X-Team-Id`. (Streamlit's Python code calls the API *server-side* with `requests`; the browser never talks to FastAPI directly.)
3. **API validates and enqueues** (`main.py:ask_question`). It checks the dataset belongs to your team, the text is 1–2000 chars, then `jobs.submit_question` writes two rows in one transaction: a `questions` row (status `running`, stage `queued`) and an `analysis_jobs` row (state `queued`). It returns immediately with `question_id`, `job_id`. If your team already has 5 active jobs → HTTP 429.
4. **A worker thread claims the job** (`jobs.py:claim_next`) with a compare-and-set SQL `UPDATE ... WHERE state='queued'`, sets a **lease**, and starts a **heartbeat** thread that renews it.
5. **The LangGraph runs** (`agents/graph.py:run_question_graph`). It pins the question to a **dataset version**, then enters at the **router**.
6. **Router** (`agents/router.py`) picks a route using regexes: `clarify`, `fast`, `root_cause`, `hypothesis`, `statistical`, or `standard`. It also applies a per-route **budget**.
7. For the AI routes (`standard`/`statistical`):
   - **Triage** — a cheap LLM call that rejects gibberish/off-topic questions before spending money.
   - **Planner** — LLM writes 2–8 analysis steps. Its prompt is enriched with approved metric definitions, team "knowledge notes", past "lessons", and similar past analyses (**RAG**, via embeddings).
   - **Executor** (loops per step) — LLM writes a Python script for one step → runs it in the **sandbox** → if it errors, the error text is fed back and it retries (max 3).
   - **Critic** — Llama writes a verification script, which runs in the sandbox, then Llama gives a verdict `verified`/`rejected`. A rejection triggers *one* re-plan.
   - **Compile** — LLM picks KPIs/charts and writes the narrative from the step results.
   - **Verify** — deterministic checks on every KPI/chart/narrative. A failed required check triggers the same single re-plan budget.
   - **Publish** — writes `dashboards`, `audit_trail`, `evidence_records` rows, and indexes the dashboard as an embedding for future RAG.
8. **Job ends** (`jobs.execute`) → job state `succeeded`/`failed`/…; question status `verified`/`unverified`/`failed`/…; a **run manifest** is written; sandbox files are deleted.
9. **Browser polls** `GET /api/questions/{id}/status` about every 2 s (README says so) and renders a live stepper from `questions.current_stage`.
10. **Dashboard page** calls `GET /api/questions/{id}/dashboard` and shows KPIs/charts/verdict badge; clicking a number calls the `inspect` endpoint to show code, formula, data slice, evidence.

```
Browser ──> Streamlit (UI server, :7860/:8501)
              │  requests (HTTP + JWT)
              ▼
           FastAPI (:8000, loopback only)
              │ writes question + job rows
              ▼
           Database (SQLite or Postgres)  <── worker threads claim jobs
                                                   │
                                           LangGraph run
                                    ┌──────────────┼───────────────┐
                              Gemini API      Llama API        Docker sandbox
                           (plan/code/dash)   (Critic)      (runs generated Python)
```

---

## 3. Repo map

```
backend/
  app/
    main.py              FastAPI app, startup/shutdown, datasets & questions endpoints, /api/health
    config.py            ALL settings, read from environment variables
    database.py          SQLAlchemy engine/session; SQLite hand-migrations vs Postgres Alembic
    models.py            ORM tables (~45 classes)
    schemas.py           Pydantic request/response shapes
    security.py          bcrypt passwords, JWT, get_current_user / get_current_team dependencies
    authz.py             organization/team role policy, invitation tokens
    rate_limit.py        in-memory sliding-window limiter
    routers/             one file per API area (20 files)
    agents/              LangGraph graph + each agent node + LLM clients + prompts
    sandbox/             Dockerfile for the sandbox image + runner.py (Docker & subprocess runners)
    jobs.py              durable job queue in the DB
    runtime.py           per-run context: budgets, deadlines, cancellation, tracing
    verification.py      deterministic claim checking
    verdict.py           VERIFIED / VERIFIED_WITH_CAVEATS / UNVERIFIED rules
    provenance.py        fingerprints, run manifests, reruns, comparisons
    fastpath.py          parse simple questions without AI
    metric_engine.py     safe formula language + independent reference implementation
    semantic.py          approved business metric definitions
    investigator.py      deterministic "where did the change come from" decomposition
    hypothesis_engine.py model proposes hypotheses, code tests them
    analysis_ops.py      the closed set of tests the engine may run
    stats/               statistical methods, selector, result shape
    data_quality.py      schema/quality/drift checks per upload
    scheduler.py         recurring analyses + alerting (APScheduler)
    copilot.py           multi-turn analysis sessions
    scenarios.py         what-if arithmetic
    memory.py / knowledge.py   RAG over past analyses and team notes
    chat.py              "ask your dashboard" follow-up Q&A
    insights.py, correlations.py, change_detection.py, csv_lines.py, email_sender.py, ...
    evals/               the evaluation suite (python -m app.evals)
  alembic/               Postgres migrations (9 versions)
  evals/                 test cases + CSV fixtures
  tests/                 ~9.6k lines of pytest
frontend/
  Home.py                landing page
  pages/                 18 Streamlit pages
  api_client.py          typed wrapper over the API (759 lines)
  auth.py                login / signup / Google / password gate UI
  chart_studio.py        restyle & swap chart types (1003 lines)
  style/                 CSS + theme.py (design system "SILT")
Dockerfile, start.sh, render.yaml, docker-compose.yml, .github/workflows/tests.yml
docs/                    PLATFORM.md, HYPOTHESIS_ENGINE.md, STATISTICAL_REASONING.md, IMPLEMENTATION_ROADMAP.md
```

---

## 4. Tech stack — every library and why

Versions are pinned in `backend/requirements.txt` and `frontend/requirements.txt`.

| Library | Role here | Basic idea |
|---|---|---|
| **Python 3.11** | Everything | The language. Docker images use `python:3.11-slim`. |
| **FastAPI** | HTTP API | A Python web framework. You declare functions with typed parameters; it builds routes, validates input, and auto-generates docs at `/docs`. |
| **Uvicorn** | Runs FastAPI | An ASGI server — the process that actually listens on a port and passes requests to FastAPI. |
| **python-multipart** | File uploads | Lets FastAPI parse `multipart/form-data` (how browsers upload files). |
| **Pydantic v2** | Validation | Classes (`BaseModel`) that validate and serialize data. `schemas.py` uses it for every request/response. |
| **SQLAlchemy 2** | ORM | Lets you work with DB rows as Python objects instead of raw SQL. |
| **Alembic** | Migrations | Versioned scripts that evolve the Postgres schema. |
| **psycopg2-binary** | Postgres driver | How SQLAlchemy talks to Postgres. |
| **pgvector** | Vector search | Postgres extension + Python helper to store embeddings and search by similarity. |
| **SQLite** | Default DB | A single-file DB built into Python. Used for local dev and tests. |
| **PyJWT** | Sessions | Creates/verifies JSON Web Tokens. |
| **bcrypt** | Passwords | Slow, salted password hashing. |
| **python-dotenv** | Config | Loads `.env` into environment variables. |
| **google-genai** | Gemini | Official SDK for Google's Gemini models and embeddings. |
| **openai** (SDK) | Llama | Used only as a client for any *OpenAI-compatible* endpoint (Meta Llama API, Groq, etc.). |
| **LangGraph** | Orchestration | Builds the agent workflow as a graph of nodes with conditional edges. |
| **pandas / numpy** | Data | Dataframes and numeric arrays. |
| **scipy** | Statistics | Hypothesis tests, distributions, resampling helpers. |
| **plotly** | Charts | Interactive charts; the sandbox writes charts as Plotly JSON. |
| **matplotlib** | Charts (sandbox) | Present in the sandbox image; prompts tell the model *not* to use it for dashboard charts. |
| **APScheduler** | Scheduling | In-process timer for scheduled analyses. |
| **Streamlit** | UI | Python-only web UI framework; reruns the script on each interaction. |
| **requests** | HTTP client | Streamlit → FastAPI calls. |
| **Authlib** | OIDC | Used by Streamlit's built-in Google login. |
| **pytest / httpx** | Tests | httpx powers FastAPI's `TestClient`. |
| **Docker** | Sandbox + deployment | Isolation boundary for generated code; also packages the app. |
| **Neon** (referenced in `render.yaml`) | Hosted Postgres | Supplies `DATABASE_URL`. |
| **Render / Hugging Face Spaces** | Hosting | Both run the Docker image; neither can start nested containers. |

---

## 5. Web and API fundamentals used here

**Client / server.** A client sends a *request*, a server returns a *response*. Here the "client" of FastAPI is Streamlit's Python code.

**HTTP request parts.** Method (GET read, POST create/act, PUT replace, PATCH partial update, DELETE remove), URL path, query string (`?limit=20`), headers, optional body.

**Headers this app relies on**
- `Authorization: Bearer <token>` — who you are.
- `X-Team-Id: 7` — which team you're acting for.
- `Idempotency-Key: <string>` — make a retried POST safe (see §6.4).
- `X-Internal-Secret` — shared secret for the Google-exchange endpoint.

**Status codes used.** 200 OK · 201 Created · 204 No Content · 400 bad input · 401 not authenticated · 403 authenticated but not allowed · 404 not found (also used to *hide* other teams' data) · 409 conflict (duplicate email) · 413 payload too large (upload) · 429 too many requests · 503 service unavailable (no model provider for hypothesis route).

**JSON.** The data format of the API. `FastAPI` converts Pydantic models to JSON automatically.

**REST-ish.** Resources are nouns in URLs (`/api/datasets/{id}`), verbs are HTTP methods. Actions that don't fit CRUD are `POST .../cancel`, `POST .../approve`, etc.

**Path vs query vs body parameters.** `/api/questions/{question_id}/status` → path param. `?include_superseded=true` → query. Request JSON → body (validated by Pydantic).

**Dependency injection (`Depends`).** FastAPI's way to share logic. `get_db` yields a DB session and closes it afterward; `get_current_user` reads the bearer token; `get_current_team` reads `X-Team-Id` and checks membership. Endpoints just declare `user: User = Depends(get_current_user)`.

**CORS.** Browsers block cross-origin calls unless the server allows it. Here the browser never calls FastAPI (Streamlit does, server-side), so CORS barely matters; it defaults to `*` and can be locked with `CORS_ALLOWED_ORIGINS`.

**Polling vs websockets.** The UI repeatedly asks "is it done?" (polling). Simpler than websockets; README lists it as a known limitation.

**Multipart streaming upload.** `_save_and_profile_upload` reads the file 1 MB at a time, counts bytes, and aborts with 413 over `MAX_UPLOAD_BYTES` (200 MB). The stored name is `uuid_hex + original name` (`Path(...).name` strips directories) — prevents path tricks and collisions.

**Lifecycle hooks.** `@app.on_event("startup")` initializes logging, DB, job workers and the scheduler; `shutdown` stops them. (This style is deprecated in newer FastAPI in favour of `lifespan`.)

---

## 6. API design of this project

### 6.1 Conventions
- **Everything under `/api`.** Interactive docs at `/docs` (Swagger UI).
- **Auth on everything** except: `/api/health`, signup/login/google-exchange, `/api/public/shared/{token}` (share links), and `/api/evals/*` + `/api/scenarios/models` (no team header).
- **Errors** are FastAPI's `{"detail": "..."}`.
- **Lists** with paging return `{total, limit, offset, ...}`.
- **Response models** (`response_model=...`) make FastAPI filter output to the declared shape — e.g. `UserOut` never leaks `password_hash`.
- **Tenant isolation by 404.** Looking up another team's dataset returns 404 not 403, so ids reveal nothing (`_get_team_question`, `get_org_or_404`).

### 6.2 Endpoint catalog (about 138 routes)

*Core (in `main.py`)*
| Method + path | Purpose |
|---|---|
| `GET /api/health` | DB ok, sandbox readiness, key *presence* flags (not keys), model names, queue depth |
| `POST /api/datasets/upload` | Upload CSV → profile → dataset + version 1 |
| `POST /api/datasets/{id}/versions` | "Replace data": adds a new version to the same dataset |
| `GET /api/datasets`, `GET /api/datasets/{id}` | List/get with profile |
| `GET /api/questions` | Last 50 analyses for the team |
| `POST /api/questions` | Queue an analysis (Idempotency-Key) |
| `GET /api/questions/{id}/status` | Stage, per-step status, job info |
| `GET /api/questions/{id}/dashboard` | KPIs, charts, narrative, verdict, evidence summary |
| `GET /api/questions/{id}/audit-trail` | Plan + each element's code/stdout/result/critic verdict |
| `GET /api/questions/{id}/critic-reviews` | Raw Critic reviews |

*Routers (`backend/app/routers/`)*
| Router | Endpoints |
|---|---|
| `auth` | `POST /signup`, `/login`, `/google-exchange`; `GET /me` |
| `workspaces` | communities, teams, members, `me/workspaces`, `me/invites`, legacy invite & accept-invite, bulk invite |
| `org` | groups, org members & roles, org audit events, team-member roles, expiring token invitations (`/invitations/accept`), team discussion CRUD |
| `jobs` | list/metrics/get/cancel/retry job; `questions/{id}/job`, `/cancel`, `/trace` |
| `evidence` | evidence records, run manifest, **rerun**, **compare** two runs, dataset versions & compare |
| `investigations` | create (explicit), `/ask` (plain words, no model), `/hypothesis` (model-driven), list, get, `/statistics` |
| `quality` | dataset quality snapshot, run now, edit thresholds, acknowledge/resolve incident |
| `alerts` | list, get, read, read-all, acknowledge, resolve |
| `scheduled` | scheduled analyses CRUD (POST/GET/PATCH/DELETE) |
| `copilot` | sessions CRUD + `/messages` |
| `scenarios` | models, baseline, evaluate, save, list, compare, get, delete |
| `semantic` | metrics, versions, approve, deprecate, validate, terms, relationships, graph, resolve, search |
| `sharing` | create/list/revoke share links; public read by token |
| `inspect` | click-to-inspect any dashboard element |
| `chart_views` | save/reset Chart Studio overrides |
| `knowledge` | dataset notes CRUD |
| `insights`, `correlations` | auto starter questions; strongest column relationships |
| `chat` | follow-up Q&A on a finished dashboard |
| `evals` | eval suite/runs/compare |

### 6.3 Design patterns worth knowing
- **Asynchronous job pattern.** Long work returns *202-style* immediately (actually 200 with ids) and the client polls. Never block a request on a multi-minute LLM run.
- **Idempotency.** `Idempotency-Key` + a *unique* DB constraint on `"<team_id>:<key>"` means two identical submissions that race both reach the DB, one wins, the other catches `IntegrityError` and returns the first job. This is the correct, race-safe way (not "check then insert").
- **Compare-and-set (CAS).** Updates like `UPDATE ... WHERE id=? AND state='queued'` succeed for exactly one caller (`rowcount == 1`). Used for job claiming, finishing, recovery, and scheduled-analysis claiming. It is optimistic concurrency without locks.
- **Fail closed.** If the sandbox isn't available, the run fails with `sandbox_unavailable`; nothing silently falls back to something weaker.
- **Safe error messages.** Users see fixed text (`SAFE_ERRORS`); tracebacks go only to server logs. Prevents information disclosure.
- **Rate limiting.** `rate_limit.enforce(key, max, window)` — in-memory sliding window, per process. Login: 8/min per IP+email; signup: 10/min per IP; google: 30/min per IP.
- **Public vs private surface.** Only `/api/public/shared/{token}` is public (read-only, expiring, revocable).

### 6.4 Pydantic details seen in `schemas.py`
- `Field(max_length=200)` on `idempotency_key`; `pattern=r"^[a-z0-9_]+$"` on `chart_type`; a `@field_validator` that rejects a chart `style` dict larger than 4000 chars after JSON-encoding. These are *input hardening* examples.
- `class Config: from_attributes = True` lets Pydantic read ORM objects directly.
- `dict[str, Any] | list[Any] | None` for audit `result` — comment explains the AI's code legitimately prints lists too.

---

## 7. Authentication and authorization

**Authentication = who are you. Authorization = what may you do.**

### 7.1 Passwords — bcrypt (`security.py`)
`bcrypt.hashpw(password, gensalt())`. bcrypt is deliberately slow and adds a random **salt** per password, so identical passwords hash differently and brute-force is expensive. Only the hash (`users.password_hash`) is stored. Minimum length 8 (`signup`).

### 7.2 Sessions — JWT (`security.py`)
A **JSON Web Token** is `header.payload.signature`. The payload here: `sub` (user id), `email`, `iat` (issued at), `exp` (expiry, default 7 days). Signed with **HS256** (HMAC-SHA256) using `JWT_SECRET_KEY`. The server verifies the signature; no DB lookup is needed to trust the claims, but `get_current_user` still loads the user (so deleted users get 401).
- If `JWT_SECRET_KEY` is unset, a random one is generated per process (tokens die on restart; multiple workers would disagree). `render.yaml` auto-generates one.
- JWTs can't be revoked individually before expiry (a standard trade-off).

### 7.3 Google sign-in (OIDC flow)
1. Streamlit's built-in `st.login` handles Google OIDC (needs `.streamlit/secrets.toml`, which `start.sh` writes from env vars at boot — no secrets in the image).
2. After Google verifies the email, Streamlit calls `POST /api/auth/google-exchange` with `X-Internal-Secret`.
3. The API checks the secret using **`hmac.compare_digest`** (constant-time comparison, prevents timing attacks), requires `email_verified`, creates the user if new (with a random unusable password), and returns a normal JWT.
4. Disabled (404) unless `INTERNAL_AUTH_SECRET` is set.

### 7.4 Team scoping (`get_current_team`)
Every team endpoint requires `X-Team-Id`. The server **never trusts it alone**: `team_role()` checks active `TeamMember` or org admin. Then each resource lookup re-checks `resource.team_id == team.id`. This is **application-level multi-tenancy** (no DB row-level security).

### 7.5 Roles (`authz.py`) — three separate authorities
| Authority | Stored in | Can do |
|---|---|---|
| Organization **owner/admin** | `community_members.role` | Manage groups, teams, members, invitations, audit log; implicit *admin* on every team of the org |
| Team **owner/admin** | `team_members.role` | Team administration; approve semantic definitions; edit DQ thresholds; alert delivery settings |
| Team **leader** | `team_members.role = leader` | Invite plain members, moderate discussion; cannot change roles |
| **member** | either | Use the team's data |

`can_assign_team_role`: leaders may only create `member`s; only owner/admin may appoint `leader`s; **nobody** can mint an owner/admin via invitations (ownership transfer is a separate flow).

### 7.6 Invitations
Two mechanisms exist: a legacy "pending member by email" (`TeamMember.invited_email`, claimed at signup by `_claim_pending_invites`) and the newer `Invitation` table: raw token = `secrets.token_urlsafe(32)`; **only its SHA-256 is stored** (`token_hash`), so a DB leak yields no usable links; tokens expire (7 days), are single-use (`accepted_at`) and revocable (`revoked_at`).

### 7.7 Org audit events (`OrgAuditEvent`)
`authz.audit(...)` adds a row *into the caller's transaction without committing*, so the change and its audit record commit or roll back together. Not to be confused with `AuditTrail` (which explains how an analysis result was produced).

### 7.8 First-user special case
The very first user ever becomes owner of the "Legacy" team, so data from before multi-tenancy isn't orphaned.

---

## 8. Database: concepts, then the full schema

### 8.1 Concepts (basic → advanced)
- **Table / row / column / primary key / foreign key.** A FK enforces that a referenced row exists.
- **Index.** Speeds lookups (`index=True`). **Unique constraint** forbids duplicates (e.g. `users.email`, `analysis_jobs.question_id`, `share_links.token`).
- **Nullable.** Whether NULL is allowed. Many `team_id`s are nullable *only* because SQLite can't cheaply add NOT NULL FKs to existing tables; the app treats them as required (`Dataset.team_id` comment).
- **ORM.** `class Dataset(Base)` ↔ table `datasets`. `relationship()` gives Python access to related rows.
- **Session & transaction.** `SessionLocal` creates sessions; `db.commit()` ends a transaction; `db.rollback()` cancels. `autoflush=False` so queries don't silently write.
- **Connection pool (Postgres).** `pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=1800` reuse connections, test them before use, and recycle old ones (managed Postgres closes idle connections).
- **JSON columns.** `JSONVariant = JSON().with_variant(JSONB, "postgresql")` — native binary JSON (indexable, queryable) on Postgres, plain JSON on SQLite. A custom `json_serializer` falls back to `str` for numpy scalars.
- **Vector columns.** `pgvector`'s `Vector(768)` stores an embedding; on SQLite the column is a JSON list and similarity is computed in Python with numpy.
- **HNSW index.** *Hierarchical Navigable Small World* — an approximate nearest-neighbour index for fast similarity search (created in the Alembic migration with `vector_cosine_ops`).
- **Cosine distance operator `<=>`.** Postgres+pgvector operator; similarity = `1 - distance`.
- **Immutability / append-only.** `evidence_records`, `run_manifests`, `run_events` are written once and not updated — good for audits.
- **Soft delete.** `WorkspaceComment.deleted_at` / `deleted_by` rather than removing rows.
- **Versioning.** `DatasetVersion`, `SemanticMetricVersion`, `Plan.revision`.
- **Lease / heartbeat columns.** `worker_id`, `lease_expires_at` on `analysis_jobs`.
- **Deliberate non-FKs.** `questions.scheduled_analysis_id`, `questions.rerun_of_question_id` are plain integers: a real FK would make the FK graph circular (`scheduled_analyses → dashboards → questions → scheduled_analyses`), which SQLite can't create.
- **Pinning.** `questions.dataset_version_id` freezes which data a run used, so old dashboards never silently change.

### 8.2 The schema, grouped (all from `models.py`)

**A. Identity and organization**
| Table | Key columns | Notes |
|---|---|---|
| `users` | id, email (unique), password_hash, display_name | |
| `communities` | name, created_by | An "organization" (customer company) |
| `groups` | community_id, name (unique per community) | Departments grouping teams |
| `teams` | community_id, name, group_id (nullable) | |
| `community_members` | community_id, user_id (unique pair), role owner/admin/member | Org-level authority |
| `team_members` | team_id, user_id (nullable), invited_email, role owner/admin/leader/member, status active/pending | Nullable user_id allows invite-before-signup |
| `invitations` | community_id, team_id, email, org_role, team_role, token_hash (unique), expires_at, accepted_at/by, revoked_at, invited_by | Token hashed |
| `org_audit_events` | community_id, team_id, actor_id, action, target_type/id, detail_json | Who changed structure |
| `workspace_comments` | team_id, author_id, target_type/id, kind comment/note, body, status open/resolved, edit_history (JSON list), edited_at, deleted_at/by | Discussion feed |

**B. Data**
| Table | Notes |
|---|---|
| `datasets` | A stable "slot": team_id, filename, filepath, row/col count, `profile_json`. Mirrors the latest version. |
| `dataset_versions` | dataset_id, version_number, filepath, profile_json, **content_sha256** (fingerprint; lazily filled for old rows) |
| `dataset_insights` | auto-generated starter questions + warnings per version |
| `knowledge_notes` | kind knowledge/lesson, source user/critic, text, optional embedding |

**C. The analysis run (the original "audit-trail-first" core)**
| Table | Notes |
|---|---|
| `questions` | text, status, current_stage, stage_detail, error, dataset_version_id, trigger manual/scheduled, route, rerun_of_question_id, clarification_json |
| `plans` | steps_json `[{id, description, goal}]`, reasoning, revision (0 initial; 1+ after critic) |
| `execution_logs` | **every attempt**: step_index, attempt_number, code, stdout, stderr, success, result_json, chart_paths_json, reasoning, formula_explanation, data_slice_json |
| `critic_reviews` | verdict verified/rejected, confidence, issues_json, checks_json (the verification script + its output + `verifier_model`), summary |
| `dashboards` | kpis_json, charts_json `[{title, plotly_json, step_index, element_id}]`, narrative, verified, verdict_state, verification_summary, view_overrides_json (Text!) |
| `audit_trail` | one row per KPI/chart/narrative: element_label, element_type, **element_id**, execution_log_id (FK), critic_review_id (FK), reasoning |

The link `audit_trail.element_id` ↔ the `element_id` inside `kpis_json`/`charts_json` (a random UUID hex) is what powers click-to-inspect.

**D. Platform infrastructure**
| Table | Notes |
|---|---|
| `analysis_jobs` | The durable queue (see §16). Unique `question_id` and `idempotency_key`. |
| `run_events` | Trace: kind route/node/llm_call/sandbox_run/verification/budget/provider + detail_json + duration_ms |
| `evidence_records` | One per claim: claim_id (unique), claim_type, checks_json, limitations_json, status, reasons_json, validity_json (mathematical/statistical/causal), original/recomputed value, dataset fingerprint |
| `run_manifests` | One per question (unique), manifest_json |
| `analysis_memories` | RAG store of finished analyses: embedding (Vector 768), summary, verdict |
| `share_links` | token (unique), expires_at, revoked_at |

**E. Features**
| Table | Notes |
|---|---|
| `scheduled_analyses` | question_text, interval daily/weekly, thresholds, comparison mode, `suppress_on_dq`, `notify_email`, `last_processed_question_id` (idempotency marker) |
| `alerts` | kind change/anomaly/unverified, status open/ack/resolved, severity, `dedup_key`, occurrences, explanation_json, delivery_state/attempts/next_delivery_at/payload (email retry) |
| `investigations`, `investigation_hypotheses`, `investigation_analyses` | Root-cause / hypothesis engine state. `signature` = hash that stops the same analysis running twice |
| `dq_snapshots`, `dq_configs`, `dq_incidents` | Data quality |
| `copilot_sessions`, `copilot_turns` | Multi-turn sessions: state_json, then per-turn action/result/evidence |
| `scenarios` | Saved what-ifs |
| `semantic_metrics`, `semantic_metric_versions`, `semantic_terms`, `semantic_relationships` | The semantic layer / "knowledge graph" as 4 tables |
| `eval_runs`, `eval_results` | Evaluation history |

### 8.3 Relationships in words
`Community 1─* Team 1─* TeamMember *─1 User`. `Team 1─* Dataset 1─* DatasetVersion`. `Dataset 1─* Question 1─* Plan / ExecutionLog / CriticReview / Dashboard / AuditTrail`. `AuditTrail *─1 ExecutionLog`, `AuditTrail *─1 CriticReview`. `Question 1─1 AnalysisJob`, `1─1 RunManifest`, `1─* RunEvent / EvidenceRecord`. `Investigation 1─* InvestigationHypothesis / InvestigationAnalysis`.

---

## 9. Migrations and the two database modes

**What a migration is.** A script that changes an existing database's structure without losing data.

**Mode 1 — SQLite (default, dev/tests).** `init_db()` runs `Base.metadata.create_all()` (creates missing tables) then `_run_migrations()`: a hand-rolled, idempotent routine that (a) adds any missing columns listed in `_ADDITIVE_COLUMNS` via `ALTER TABLE ... ADD COLUMN` after checking `PRAGMA table_info`, (b) adds `team_id` to legacy tables, and (c) **backfills** orphan rows into a "Legacy" community/team. Additive only — never drops or renames.

**Mode 2 — Postgres (`DATABASE_URL` set).** `init_db()` runs `alembic upgrade head`. Nine versions exist (initial schema → dataset versions/scheduling/verdict → hypothesis investigations → organizations/invitations/discussion → analysis memories (creates `CREATE EXTENSION vector` + HNSW index) → knowledge notes → share/insights/alerts → chart view overrides → platform tables).

**Then both modes:** `backfill_org_memberships()` and `backfill_dataset_versions()` run — additive, idempotent data fixes.

**Also:** `scripts/migrate_sqlite_to_postgres.py` moves existing data across.

**Audit note.** Two migration systems must stay in sync with `models.py`; `tests/test_migration.py` and `test_platform_migrations.py` exist for that, and CI runs the whole suite on both SQLite and Postgres.

---

## 10. Gen-AI fundamentals (basic terms)

- **LLM (Large Language Model).** A neural network trained to predict the next token. Gemini and Llama are LLMs.
- **Token.** A chunk of text (≈ ¾ of a word). Models read and write tokens; you pay per token. `max_tokens` caps the output.
- **Prompt.** The text you send. Split into a **system prompt** (standing instructions, role, rules — see `agents/prompts.py`) and **user content** (the specific job).
- **Context window.** How many tokens the model can consider at once. That's why the app clips cell values to 80 chars and column lists to 200.
- **Temperature / sampling.** Randomness of output. This project does not set it (provider defaults), which is one reason runs differ — see "non-determinism".
- **Non-determinism.** Same input can give different output. The project's manifests therefore *record* a run; they don't promise a rerun equals it (`provenance.py`).
- **Hallucination.** Confident but false output. The whole verification stack exists to catch it.
- **Embedding.** A vector (list of ~768 numbers) representing the meaning of text; similar meanings → nearby vectors.
- **Cosine similarity.** Angle-based closeness of two vectors (1 = same direction). This project requires ≥ 0.6 (`RAG_MIN_SIMILARITY`).
- **Vector database / vector search.** Storage that finds the nearest embeddings (pgvector here).
- **RAG (Retrieval-Augmented Generation).** Before asking the model, *retrieve* relevant stored text and put it in the prompt. Here: similar past analyses and lessons.
- **Function calling / tool use.** Instead of free text, the model is made to return a JSON object matching a schema (a "tool"). Reliable, parseable output.
- **Structured output / JSON schema.** The `*_TOOL_SCHEMA` dicts (e.g. `PLAN_TOOL_SCHEMA`) describe the exact fields the model must return.
- **Agent.** An LLM given a role and (usually) tools/loops to accomplish a task. **Multi-agent**: several specialised roles cooperating.
- **Prompt injection.** Text in the data/user input that tries to hijack the model ("ignore previous instructions…"). See §11.7.
- **Grounding.** Forcing answers to rest on supplied evidence only (the `chat` feature).
- **Guardrails.** Code that constrains or checks model behaviour.
- **Failover / fallback.** Switching to a backup provider when the first fails.
- **Rate limit (429).** Provider says "too many requests".
- **Free tier.** The config notes Pro models return 429 with `limit: 0` on the free tier, hence Flash.
- **Cost estimation.** tokens × price per million (`LLM_PRICE_*`), summed per run.

---

## 11. Gen-AI as used here (advanced)

### 11.1 Two providers, on purpose
- **Gemini** (`llm_client.py`, default model `gemini-3.6-flash`) — Triage, Planner, Executor, Dashboard compile, hypothesis steps, insights, chat, embeddings.
- **Llama** (`llm_client_llama.py`, via the OpenAI-compatible SDK, default `Llama-4-Maverick-17B-128E-Instruct-FP8`) — **Critic only**.
- **Why:** a different model *family* tends to make different mistakes than the one that wrote the work, so cross-model verification catches more than self-review. The code is honest about degradation: if Llama is unavailable, the Critic falls back to Gemini and the review text gets a note that it was *not cross-model*, and `verifier_model` is stored (`gemini-fallback:<model>`).
- Both wrappers expose the same `call_tool` signature so agent code is provider-agnostic (an **adapter pattern**).

### 11.2 Forced tool calling
`call_tool_gemini` builds a `FunctionDeclaration` and sets `FunctionCallingConfig(mode="ANY", allowed_function_names=[tool_name])` — the model *must* answer by calling that one function. Llama does the same via `tool_choice={"type":"function", ...}`. If the model returns prose or a malformed call → `MalformedOutput` with a **diagnosis** (finish reason, part kinds, never the text).

`diagnose_response` categories: `prompt_blocked`, `empty_response`, `output_truncated` (MAX_TOKENS), `malformed_function_call`, `content_filtered`, `wrong_function`, `prose_instead_of_function_call`, `no_function_call`.

### 11.3 Retry, backoff, jitter, failover (`call_tool`)
- **429 (rate limited):** retry up to `GEMINI_RATE_LIMIT_RETRIES` (2), waiting the server's own `RetryInfo.retryDelay` if provided, else 15 s then 30 s, capped at `GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS` (30).
- **Transient 5xx (500/502/503/504):** exponential-ish waits 5 s then 20 s, each ×(0.8–1.2) **jitter** so concurrent runs don't retry in lock-step ("thundering herd"). Honours `Retry-After`; never sleeps past the run's deadline (`runtime.remaining_seconds`).
- **One shared attempt counter** so a call never makes more than `tries` requests regardless of the error mix.
- **Then failover** to Llama if configured and `LLM_FAILOVER_ENABLED`.
- **`RunAborted` is never retried or failed-over** — cancellation/budget isn't a provider failure.
- If everything fails → `ProviderUnavailable` with a *safe* message built only from provider names, error **kinds** and HTTP statuses (never error bodies, keys, or prompts). `classify_failure` maps exceptions to kinds: `rate_limited`, `overloaded`, `server_error`, `authentication` (401/403 or Gemini's 400 with reason `API_KEY_INVALID`), `bad_request`, `model_not_found`, `network`, `malformed_output`, etc.
- Each failure logs one `provider_event ...` line and a `run_events` row.

### 11.4 Budgets and cost control (`runtime.py`)
A `RunContext` (a Python **ContextVar**, so each worker thread's run has its own) tracks LLM calls, sandbox runs, tokens, estimated cost. `checkpoint(about_to="llm")` raises `BudgetExceeded` *before* the (N+1)th use, so a limit of N means exactly N. Limits: `RUN_MAX_LLM_CALLS=40`, `RUN_MAX_SANDBOX_RUNS=30`, `RUN_MAX_COST_USD=0` (off), plus a job deadline (900 s). Per-route budgets: `fast`/`root_cause`/`clarify` = 0 LLM calls and 0 sandbox runs (a `-1` sentinel means "none allowed" because `0` already means "unlimited").

### 11.5 Embeddings and RAG (`memory.py`, `knowledge.py`)
- Model `gemini-embedding-001`, **truncated** to 768 dims via `output_dimensionality` (HNSW indexes cap at 2000 dims; default is 3072). Truncated vectors aren't unit length, so they're **L2-normalised** (`v / (‖v‖ + 1e-12)`) so cosine behaves.
- **Task types:** `RETRIEVAL_DOCUMENT` when storing, `RETRIEVAL_QUERY` when searching (the model embeds them differently for better matching).
- `QueryEmbedding` caches one question's vector so planner lookups (past analyses + lessons) call the API once, keyed by (model, dim, task type, text).
- Retrieval: Postgres → SQL cosine distance via HNSW; SQLite → numpy matrix math. Top-K 3, min similarity 0.6, filtered **by team_id and dataset_id** so nothing crosses tenants.
- **Two kinds of notes:** `knowledge` (always included, newest first, capped at 25 — because a definition only helps if it's *always* there) and `lesson` (retrieved by similarity; auto-created when the Critic rejects, `source="critic"`, deduped by text).
- **Design rule repeated in the prompts:** retrieved context is *context only*; the Planner is told to recompute everything, and the **Critic never sees it**, so an old number can't vouch for a new one.
- Everything in RAG **degrades to "no memory"** on failure and never breaks a run.

### 11.6 Agents as a graph (LangGraph)
- **State machine:** nodes are functions `state -> partial state update`; `AgentState` is a `TypedDict`.
- **Conditional edges:** `route_after_router`, `route_after_executor`, `route_after_critic`, `route_after_verify`, etc. choose the next node.
- **Loops are bounded:** executor retries ≤ `MAX_EXECUTOR_RETRIES` (3) per step; the two edges into `prepare_revision` share one budget `MAX_CRITIC_REVISIONS` (1); `recursion_limit=150` is the last backstop.
- **Tracing wrapper** `_traced` records each node with duration/outcome and checks cancellation before it starts.
- The graph is compiled once and cached (`get_compiled_graph`).

### 11.7 Prompt-injection defences (defence in depth)
1. `UNTRUSTED_DATA_NOTICE` appended to every system prompt: dataset text is data, never instructions.
2. `profile_block()` wraps the profile in `<dataset_profile untrusted="true">`, clips each string to 80 chars, caps columns at 200 and top values at 5.
3. Approved metric definitions come from the **semantic layer** (changed only by an owner/admin via the API) and are labelled separately from dataset content.
4. The model never *executes* anything outside the sandbox and never supplies a number that isn't re-checked.
5. The hypothesis engine sends **no rows** to the model, only schema, low-cardinality sample values, and aggregates.
6. The docs honestly state these reduce, not eliminate, risk — "deterministic checks are what bound the damage."

### 11.8 "The model proposes, code decides"
The recurring architectural principle:
- Hypothesis engine: model picks from a **closed set of operations** with validated parameters; status (`supported/weakened/…`) is set by fixed rules; the report is discarded in favour of a template if it contains a number no analysis produced.
- Copilot: rules first; model only translates when no rule applies, and its output is validated against a whitelist and real columns.
- Fast path: pure parsing, no model.

### 11.9 LLM-as-judge on the chat feature
`chat.py` answers follow-ups from the finished analysis only, then an independent model checks the answer; unsupported numbers/claims mark the response **unverified** in the UI.

### 11.10 Offline evaluation with a scripted model
`runtime.provider_override` lets tests/evals replace *only* the model with a scripted one while the graph, sandbox, verification and publishing run for real. Metrics: pass rate by component, model calls, code runs, retries, tokens, cost, latency. Live evals (`--live`) call real providers and are scored by tolerance.

---

## 12. The agent pipeline node by node

| Node | File | Input → Output | Notes |
|---|---|---|---|
| **router** | `agents/router.py` | question + profile + semantic resolution → route | Regex rules; records reasons; may return `clarify` |
| **triage** | `agents/triage.py` | question + profile → `is_analyzable` | Skips LLM if < 4 chars. On 429 returns a clear quota message instead of letting a doomed run proceed. Other triage errors **allow** the question through. |
| **planner** | `agents/planner.py` | → `Plan` row (2–8 steps) | Adds semantic definitions, hints, knowledge, lessons, past analyses (not for scheduled runs, so re-runs plan fresh) and on revision the rejection feedback |
| **executor** | `agents/executor.py` | step → code → sandbox → `ExecutionLog` | Returns `retry_count+1` with `last_error` to retry; after 3 failures marks the whole run failed. Also asks for a plain-English `formula_explanation`. |
| **critic** | `agents/critic.py` | step results → own verify script → review | Doesn't reuse the executor's conversation. Attempt number `100 + revision` so its files don't collide. Writes `CriticReview`; rejection → `record_critic_lesson`. |
| **prepare_revision** | `graph.py` | feedback (verification and/or critic) → reset state | Increments `revision_count` |
| **compile** | `agents/dashboard.py` | → draft KPIs/charts/narrative | Resolves chart file names back to Plotly JSON; keeps only well-formed KPIs (labels/values clipped to 200 chars); assigns `uuid` element ids |
| **verify** | same | → evidence records | See §15 |
| **publish** | same | → rows | `publish_dashboard` is the **single place** anything becomes visible; shared by every route |
| **fast / investigate / hypothesis** | `agents/deterministic.py`, `agents/hypothesis.py` | no-model routes | Publish through the same function |

**The sandbox contract the model must follow** (`SANDBOX_CONTRACT` in `prompts.py`): read `/workspace/data/input.csv`; print exactly one `RESULT_JSON:{...}` line; write charts with `fig.write_json("/workspace/output/<name>.json")`; also print `DATA_SLICE_JSON:{columns, rows<=50}` (the real subset used, powering click-to-inspect); no network/subprocess/file deletion; leave chart colours to the dashboard theme.

**Verdict logic** (`verdict.py` + `verification.combine_verdict`):
- Critic verified first time → `VERIFIED`; verified after a re-plan → `VERIFIED_WITH_CAVEATS`; rejected → `UNVERIFIED`.
- Any failed **required** deterministic check anywhere → `UNVERIFIED`, whatever the Critic said.
- Any claim not fully `verified` lowers `VERIFIED` to `VERIFIED_WITH_CAVEATS`.
- A run that exhausts its revision budget is **still published**, marked unverified — "the run always reaches an inspectable end state."

**Terminal question statuses:** `verified`, `unverified`, `rejected` (not analyzable — guidance shown, not an error), `needs_clarification`, `cancelled`, `failed`.

---

## 13. The no-model paths

### 13.1 Router rules (`router.py`)
Regex `_ROOT_CAUSE` (why / root cause / what caused / drivers of …) and `_STATISTICAL` (trend / correlat / outlier / compar / vs / seasonal / yoy …). Order of decision: ambiguity (from semantic layer) → fast path parse → root-cause (hypothesis if a model is available + a date column; else deterministic investigator) → statistical → standard. Statistical route adds `STATISTICAL_HINT` to the planner (report n, effect size, small-sample caveat, no causal claims).

### 13.2 Fast path (`fastpath.py`)
A conservative **natural-language → formula** parser. Steps: group-by phrase ("by/per/for each…") → approved metric term → aggregation word + column → **every remaining content word must be filler** (`_STOP`), else it's not "simple" and returns `None` (goes to the full pipeline). Ambiguity (two columns match equally) returns a clarify option list. Result is computed by `metric_engine` and **recomputed by the independent reference implementation**, so even the "no AI" path is verified. Max 60 groups.

### 13.3 Metric engine (`metric_engine.py`)
- A tiny **formula language**: `sum, avg, median, min, max, count, count_distinct`, numbers, `+ - * /`, unary minus; e.g. `(sum(revenue) - sum(cost)) / sum(revenue) * 100`.
- Parsed with Python's `ast` module and **whitelisted** — attribute access, subscripts, other calls, free names are rejected at parse time. It is **never `eval`'d.**
- Filters are `(column, operator, value)` triples validated against real columns.
- `reference_evaluate` is a **second implementation** using only the `csv` module and plain Python. Two independent implementations agreeing is evidence neither has a shared bug (**N-version programming / differential testing**).
- `ENGINE_MAX_ROWS = 2,000,000` guard.

### 13.4 Semantic layer (`semantic.py`)
Business glossary with governance:
- **Metric** (slug, name, unit, status draft/approved/deprecated) + **versions** (formula, dataset, filters) — a new version is a *proposal* until an owner/admin **approves a specific version**.
- **Terms** map words to a metric (alias) or to a dataset column (dimension/entity).
- **Relationships** (`derived_from, measured_by, grouped_by, joins, synonym_of`) form a **knowledge graph** stored relationally — "nothing here needs a graph database."
- **Resolution** prefers the longest matching phrase; if two approved metrics claim a phrase → `ambiguous` → the router asks a clarifying question rather than guessing.
- **Trust model:** nothing from a file, cell, note, or model output can create/change/approve a definition.

---

## 14. The sandbox (running AI-written code safely)

**Why.** Model-written code is *hostile input*: the model read a user's question and a user's CSV, both of which could try to manipulate it.

**Two runners (`sandbox/runner.py`)**

**DockerSandboxRunner — the real boundary.** One fresh container per execution:
| Flag | What it does |
|---|---|
| `--network none` | No internet, no host, no cloud metadata endpoint |
| `--read-only` | Immutable root filesystem |
| `--cap-drop ALL`, `no-new-privileges` | No Linux capabilities; can't gain privileges via setuid |
| `--pids-limit 128` | Prevents fork bombs |
| `--memory 512m`, `--memory-swap` = same | Memory cap with no swap |
| `--cpus 1` | CPU cap |
| `--ulimit fsize/nofile/core` | File-size, open-file and core-dump limits |
| `--ipc none` | No shared memory with others |
| `--tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m` | Writable scratch that can't run binaries |
| `-v data:ro`, output:rw, `main.py:ro` | Dataset read-only; one private output dir |
| `python -I -B` | **Isolated mode**: ignores user site, `PYTHON*` env vars, doesn't put cwd on `sys.path` |
| non-root user (`sandboxuser`, uid 1000) | Image-level |
| optional `--runtime runsc` | **gVisor** user-space kernel (untested per docs) |

**SubprocessSandboxRunner — NOT a boundary.** A child process on the host. Mitigations: environment built from scratch (no API keys — `_sandbox_env`), own process group (killed on timeout), `RLIMIT_CPU/FSIZE/CORE/NPROC`, optional drop to the unprivileged `sandbox` user via `preexec`/`user=` kwargs, per-run directory. Refused unless `SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true`.

**Shared protections (both runners)**
- Per-execution directory, deleted afterward (even read-only files — `_force_rmtree` chmods on Windows).
- Timeout, cancellation and **output flood** kill (`_Pump` threads cap kept bytes; kill if output > 4× limit).
- Code size limit (200 KB), NUL byte rejection.
- **Artifact validation** (`_collect_artifacts`): no symlinks (never followed), filename regex `^[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}\.json$`, ≤ 5 MB each, ≤ 20 files, content must parse as JSON with a `data` list (Plotly figure); the file is **re-serialised** so what is stored is exactly what was validated.
- **Fail closed:** `SandboxUnavailable` (a `RunAborted`) if the boundary is missing. Docker availability is cached 60 s.
- Killing `docker run` doesn't stop the container, so `kill()` calls `docker kill <name>`, then `docker rm -f` in a `finally`.

**Hosted deployment trade-off.** HF Spaces and Render can't start containers → image sets `SANDBOX_BACKEND=subprocess`; generated code is refused unless the operator knowingly sets the unsafe flag. `/app/data` is `chmod 700` root-owned so the sandbox user can't traverse to `app.db`/uploads; workspaces live in `/sandbox/workspaces`.

---

## 15. Verification, evidence and verdicts (`verification.py`)

**Check outcomes:** `pass`, `fail`, `not_run`, `warn`, `not_applicable`. **A check that didn't run is never a pass.**

**Per KPI** (all *required*):
1. `result_present` — the cited step ran and produced parseable JSON.
2. `value_in_result` — the displayed value appears in the step's result. `WARN` if it appears only in a *different* step.
3. `independent_recomputation` — Critic's verification output contains the number (or for deterministic routes: engine vs reference agree and match the display).
4. `provenance` — an execution record exists **and** the dataset file's current hash equals the recorded fingerprint (fail on `dataset_changed`).
5. `invariants` — finite; a "share/proportion" with `%` between 0 and 100; a "count" not negative (unless it's an average/rate).

**Per narrative:** `narrative_numbers_supported` (every number in the text must appear in a result or KPI; years and small integers ≤ 12 are ignored; numbers repeating the question ignored), `independent_review` (Critic accepted), `causal_language` (not required; flags *because/caused/due to/led to/driven by/…*).

**Per chart:** `artifact_validated`, `provenance` (required); `independent_recomputation` recorded as `not_run: chart_values_not_recomputed` (not required — but never shown as a pass).

**Number matching.** `number_matches` handles "$1,234.5", "12.4%", "1.2M". A displayed figure matches a computed one if they differ by **less than one unit of the last displayed digit** (so "1,235" matches 1234.6 and 1235.9, not 1236.9); a percent also matches a fraction (12.4% vs 0.124). `VERIFY_REL_TOLERANCE` widens it (default 0).

**Status:** `unverified` if any required check fails or can't run; `verified_with_caveats` if all pass but a warning/limitation exists; else `verified`.

**Three kinds of validity are reported separately:** *mathematical* (do the numbers follow?), *statistical* (sample adequate? flagged `small_sample` if < 30 rows, otherwise "not assessed"), *causal* (never established).

**Recovery:** a failed required check → back to Planner once; still failing → published `UNVERIFIED`. It never loops, and never passes by running out of retries.

---

## 16. Durable jobs (`jobs.py`)

**The idea.** Instead of a thread per request (lost on restart), work is a **row in a table**; workers *claim* rows. Chosen over Redis/Celery to avoid extra infrastructure.

**State machine:** `queued → running → succeeded | failed | cancelled | timed_out`; `running → queued` if its lease expires and attempts remain.

**Mechanics**
- **Admission control:** ≤ `JOB_TEAM_MAX_ACTIVE` (5) queued+running per team → 429.
- **Claim:** CAS `UPDATE`; sets `worker_id`, increments `attempt`, sets `deadline_at` and `lease_expires_at = now + 60s`.
- **Lease + heartbeat:** `_Heartbeat` thread renews the lease every `max(1, lease/3)` s and reports `cancel_requested`. Every terminal write requires `worker_id == me AND state == running`, so a worker that lost its lease cannot overwrite the new owner's result (**fencing**).
- **Janitor thread** calls `recover_abandoned()`: expired-lease jobs → `succeeded` (if a dashboard already exists — publishing must happen once), `timed_out` (past deadline), `cancelled`, `queued` (attempts left), or `failed/abandoned`. Also times out queued jobs past deadline.
- **Startup recovery** runs before workers start; `fail_orphan_questions` closes questions left `running` with no job (from the old thread model).
- **Cancel:** queued → immediately; running → flag, honoured at the next checkpoint (before each node, model call, code run) and a running container is killed.
- **Retry:** only after abandonment automatically; otherwise `POST /jobs/{id}/retry`, refused if a dashboard already exists.
- **Post-run bookkeeping** (`_after_run`) — each step independent and unable to change the outcome: write manifest, scheduler alerting, delete workspace.
- **Known limit:** workers run *inside* the API process; total concurrency = workers × processes. Claiming is still exclusive.

---

## 17. Hypothesis engine and statistics

*(From `docs/HYPOTHESIS_ENGINE.md`, `docs/STATISTICAL_REASONING.md` and module docstrings; I did not read `hypothesis_engine.py`/`analysis_ops.py`/`stats/*` line by line.)*

### 17.1 Deterministic investigator (`investigator.py`)
Decomposes a metric change between two periods:
- **Additive metrics (sum, count):** per-segment contributions that sum exactly to the change.
- **Ratios:** **rate effect** vs **mix effect** per segment.
- **Price × volume** when a quantity column exists.
- **Unequal period lengths:** sums/counts are compared like-for-like (earlier value scaled to the later length); the "extra days" effect is its own row.
- Period phrases parsed against the **dataset's own calendar**, not today's date.
- Outcomes include `inconclusive` rather than promoting the biggest number. Every report states: *locates a change; does not establish a cause*; lists what wasn't examined and alternatives.

### 17.2 Hypothesis engine (`hypothesis_engine.py`)
Flow: **understand → baseline facts → scan → generate hypotheses → test → adapt → report**.
- Model calls: `submit_understanding`, `submit_hypotheses`, up to 4× `submit_next_step`, `submit_report`.
- Ten-plus validated **operations** (closed set): `metric_change, component_change, share_change, distribution_shift, segment_contribution, mix_shift, driver_decomposition, price_volume, period_coverage, data_quality` + statistical: `level_shift, pre_trend, change_point, paired_change, composition_change, group_difference, correlation, adjusted_change`.
- **Signature** (hash of normalised operation) = same analysis never runs twice.
- Statuses: `supported / weakened / inconclusive / not_testable / failed / proposed`. `not_testable`, `failed`, `proposed` are *absence of evidence* — never count against a hypothesis. No confidence scores.
- Concentration is judged against segment size (a segment with 50% of the metric and 50% of the change is "in proportion").
- If a period is incomplete or DQ finds issues → `data_cannot_support_conclusion`.
- Bounded by LLM calls (8), analyses (24), seconds (240), tokens (80k), hypotheses (6), rounds (4).
- **Persistence & resume:** tables written as it goes; a recovered job reloads and continues. **Evidence provenance:** a stored result is reused only in the same *context* (dataset version/fingerprint, question framing, engine+stats versions); otherwise it's marked *superseded*.
- Falls back to the deterministic investigator if no model answers at the first step.
- Report guard: summary is discarded for a template if it contains a number no analysis produced or asserts a cause.

### 17.3 Statistics glossary (what the stats layer does)
- **Null hypothesis / p-value.** The probability of seeing data at least this extreme if there were truly no effect. Not the probability the hypothesis is true.
- **Significance level α** = 0.05 (`HYP_SIGNIFICANCE_ALPHA`), fixed before tests run.
- **Confidence interval (95%)** — range of plausible values for the effect.
- **Effect size** — how big (vs "significant" = distinguishable from noise). Reported: mean difference, relative difference, **Hedges' g** (standardised mean difference with small-sample correction), d_z (paired), risk difference/ratio/odds ratio, **Cramér's V**, eta squared, r².
- **Welch's t-test** — compares two means without assuming equal variances (Welch–Satterthwaite degrees of freedom).
- **Cochran's rule** — here `n ≥ 25 × skewness²` decides whether a t-based interval is trustworthy; skewness is taken at the *upper end of its own uncertainty* (`|g| + 2√(6/n)`).
- **Permutation test** — shuffle labels many times to build the null distribution; keeps testing the *mean difference* when the t-rule fails.
- **Bootstrap / BCa** — resample the data to estimate an interval; BCa = bias-corrected and accelerated.
- **Mann–Whitney U / Wilcoxon** — rank-based tests, used **only** when the question is about "typical" values (different estimand).
- **Chi-square vs Fisher's exact** — proportion tests; Fisher when expected counts < 5. **Newcombe/Wilson score** intervals (never the naïve normal one).
- **Pearson vs Spearman** — linear vs rank correlation (Spearman when extreme values).
- **Newey–West (HAC) standard errors** — robust to autocorrelation; daily metrics aren't independent rows.
- **HC3** — heteroskedasticity-robust errors for the adjusted regression.
- **Pettitt's test** — detects a single abrupt change-point in a series.
- **Multiplicity correction** — many tests inflate false positives. **Holm** (default) or **Benjamini–Hochberg** (FDR) adjusts p-values across the investigation's family.
- **Attenuation** — an adjusted difference that lost ≥ 50% of the unadjusted one is "largely accounted for by the controls."
- **Seeded resampling** (`STAT_SEED`) → reproducible results.
- **Result model** — every method returns one validated shape; impossible numbers (p outside [0,1], reversed interval, non-finite estimate) are refused; unavailable = `null`, never 0.
- **Decision table** (distinguishable × direction × size) → supported / weakened / inconclusive. Significance alone never supports; non-significance alone never counts against.
- **Honest limits stated in docs:** observational data; rows assumed independent (no clustering correction); < 2 years can't separate seasonality; Newey–West approximate for short series.
- *Not implemented on purpose:* forecasting, seasonal decomposition, logistic/mixed models, causal designs (diff-in-diff, IV, matching).

---

## 18. The other platform features

*(Docstring + docs level; not read line by line.)*

**Data quality (`data_quality.py`).** Per-version snapshot vs a baseline (previous version by default; pinnable for seasonal data). Categories: **schema** (column removed/added/type changed), **quality** (row-count change, more missing, duplicates, out-of-range, stale), **distribution** (drift). False-alarm control: no comparison under `min_sample` (100); numeric drift needs **PSI > threshold AND Kolmogorov–Smirnov rejection at 1%** (two independent signals). **PSI** = Population Stability Index (0.1 warn, 0.25 high). A failing check never fails an upload.

**Scheduling & monitoring (`scheduler.py`).** APScheduler ticks every 60 s; a due analysis is **claimed with a conditional UPDATE** so only one process runs it; the run is a normal durable job. Alert policy: change must exceed `change_threshold_pct` **and** optional `min_effect_abs`; comparison `previous | same_weekday | rolling_mean`; **data-quality gate** (hold KPI alerts and raise one DQ alert if the data broke); **deduplication** by `dedup_key`; **idempotency** via `last_processed_question_id`; lifecycle `open → acknowledged → resolved`; email retried with exponential backoff up to 3 attempts from a stored payload; in-app alerts never depend on email. `change_detection.py` parses KPI display strings ("$1,234.5", "1.2M") and matches KPIs across runs by normalised label.

**Copilot (`copilot.py`).** Multi-turn session whose **state** (metric, filters, group_by, time range) changes through whitelisted operations (`set_metric, add_filter, remove_filter, clear_filters, set_time_range, drill_down, drill_up, rebase, reset`). Rules first, model only as translator; answers recomputed each turn **twice**; ambiguity returns a question; newer dataset version is flagged, not auto-adopted. Private to its creator within the team.

**What-if (`scenarios.py`).** Two explicit models — `unit_economics` (revenue = price×volume; profit = revenue − unit_cost×volume − fixed; price elasticity as a typed assumption) and `funnel` (orders = visitors×conversion; revenue = orders×AOV). Inputs validated against physical limits (rejected, not clamped). Base/optimistic/pessimistic scenarios, sensitivity table, **Shapley-value** attribution of each changed input (sums exactly to the change). Always labelled `kind: "scenario"` — arithmetic, not forecast.

**Provenance (`provenance.py`).** SHA-256 dataset fingerprints; **run manifest** (question, data hash, plans, hash of every script, prompt/app versions, models, token usage, env, config, outcome) written once; `rerun` pinned to original dataset version; `compare_runs` attributes differences to data vs execution/model variation — never claims bit-for-bit reproducibility of an LLM run.

**Insights & correlations.** Starter questions are LLM-suggested (labelled suggestions); data-quality warnings from the profile are deterministic. Correlations: Pearson and Spearman both computed; a pair is reported only if they agree in sign and the sample is large enough.

**Sharing.** Revocable, expiring read-only link to one dashboard; the token is the only credential; Streamlit page `11_Shared.py` uses only the public endpoint.

**Chart Studio.** Swap any chart among ~40 types and restyle; presentation only, stored in `dashboards.view_overrides_json` per team; verified numbers are never recomputed.

**Email (`email_sender.py`).** Plain SMTP (Gmail default, App Password). Synchronous by design; disabled if no creds, and logged not raised so invites still work.

**csv_lines.** Maps a step's data slice back to CSV line numbers (header = line 1) only when every slice column exists in the CSV; aggregates get no line numbers rather than a misleading guess.

**Evaluation (`python -m app.evals`).** Versioned suite (`suite.json`), fixtures with reference answers computed by plain Python (not app code), offline (scripted model, deterministic) and live modes; `compare` exits 1 on regression; wired into CI.

---

## 19. Frontend (Streamlit)

- **Streamlit model:** the script re-runs top to bottom on every interaction; state lives in `st.session_state` (`auth_user`, `auth_token`, `active_community_id`, `active_team_id`).
- **Multipage app:** `Home.py` + `pages/` (numbered filenames define sidebar order): Overview, Datasets, Analyses, Dashboards, Reports, Audit Trail, Settings, Workspaces, Team Overview, Scheduled, Shared, Copilot, Investigations, What-If, Data Quality, Semantic Layer, Evaluation, Discussion.
- **`api_client.py`:** reads token/team from session state so pages call `list_datasets()` with no params. A custom `_Http` wrapper converts any `requests` exception into `BackendUnreachable`, so pages distinguish "server didn't answer" from "server said no"; `SessionExpired` signs the user out and shows a notice (only when a token was actually sent, to prevent redirect loops).
- **`auth.py`:** login/signup UI, optional Google button (only if `[auth]` secrets and `INTERNAL_AUTH_SECRET` exist), optional shared-password gate (`APP_PASSWORD`, constant-time compare) for hosted demos. Injects JS that toggles a class on `<body>` for the sliding sign-in animation.
- **Styling:** `style/theme.py` + CSS files (`glass.css`, `dashboard_glass.css`, `landing.css`, `gate.css`, `silt.css`, `mindset.css`); CSS is flattened to one line (comment explains Markdown ends an HTML block at a blank line). `esc()` helper escapes text (there's a `test_esc.py`) — guarding against HTML injection since content is rendered with `unsafe_allow_html`.
- **Charts:** Plotly JSON from the dashboard is re-themed at render time.
- **Tests:** `frontend/tests` (chart studio, escaping, page smoke tests).

---

## 20. Deployment, configuration, CI, tests

**Single container** (`Dockerfile`, `start.sh`): FastAPI on `127.0.0.1:8000` (never public), Streamlit public on `$PORT` or 7860. `start.sh` (1) writes `secrets.toml` from env vars only if Google is configured (`umask 077`), (2) starts uvicorn, (3) waits up to 90 s for `/api/health`, (4) starts Streamlit, (5) `wait -n` — if either process dies the container exits so the platform restarts it. Rationale: the API is reachable only through Streamlit, which is behind the login/password gate.

**Targets:** Hugging Face Spaces (README front-matter `sdk: docker`, `app_port: 7860`) and Render (`render.yaml` Blueprint: free plan, `healthCheckPath: /`, secrets `sync: false`, `JWT_SECRET_KEY` and `INTERNAL_AUTH_SECRET` auto-generated, `DATABASE_URL` from Neon). On Render's free tier uploads/workspaces are ephemeral but data lives in Postgres.

**Local:** `docker compose up -d postgres` (pgvector/pgvector:pg16 on host port 5433), `run-local.ps1`, uvicorn + `streamlit run frontend/Home.py`.

**Config.** Everything is an environment variable read in `config.py` at import time, with safe defaults. Notables: `_secret()` strips whitespace/quotes pasted around keys; blank model env vars count as unset; booleans parsed via `_bool`. `.env.example` (201 lines) documents each.

**CI (`.github/workflows/tests.yml`).** Postgres+pgvector service container → install deps → **build the sandbox image** (so security tests run instead of skipping) → pytest on SQLite → pytest on Postgres (`TEST_DATABASE_URL`) → offline agent evaluation suite with Docker sandbox → upload report.

**Tests (`backend/tests`, ~9.6k lines).** `conftest.py` sets env **before** importing the app (config reads at import), uses a scratch SQLite DB, `JOB_WORKERS=0` (jobs run explicitly via `run_inline`), blank API keys (no test can reach a paid model), `SCHEDULER_ENABLED=false`, blank SMTP/DATABASE_URL. Themes: auth, team isolation, sandbox security (attacks the real Docker boundary), jobs, verification, statistics, hypothesis engine, migrations, rate limit, reproducibility, evals.

---

## 21. Audit findings (from reading; nothing was executed)

### 21.1 Strong design choices
1. **Trust architecture is layered and honest** — Critic + deterministic checks + provenance; unknown/not-run checks never count as passes; a run always ends in an inspectable state.
2. **Model never owns a number** — closed operation sets, whitelisted formula language parsed with `ast`, independent reference implementation.
3. **Fail-closed sandbox** with a real Docker boundary and explicit, loud opt-out for the unsafe runner.
4. **Concurrency done properly** — CAS updates, leases with fencing, idempotency via unique constraint, recovery for crash windows, "don't publish twice".
5. **Tenant isolation consistently re-checked** per resource, with 404s that leak nothing; invitation tokens stored hashed.
6. **Safe-by-default error handling** — fixed user messages, provider failures recorded as kinds/statuses only.
7. **Everything bounded** — loops, budgets, deadlines, output sizes, artifact counts.
8. **Documentation states its own limits** (HYPOTHESIS_ENGINE "Not verified", STATISTICAL_REASONING "Verification status").
9. **CI runs the security tests and evals** against real Docker and Postgres.

### 21.2 Risks and gaps I can point to
| # | Finding | Where | Impact |
|---|---|---|---|
| 1 | **Documentation drift.** README still says storage is SQLite with 7 tables, 5 frontend pages, "in-process Hugging Face" only; the code has ~45 tables, 18 pages, Postgres/pgvector and Render. | `README.md` vs `models.py`, `frontend/pages/` | Misleads newcomers; `PLATFORM.md` is the accurate reference. |
| 2 | **Hosted deployments run without the isolation boundary.** If an operator sets `SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true` on HF/Render, model-written code runs on the same kernel/network. | `Dockerfile`, README | Documented clearly; still the largest security caveat. Keep the flag off for public deployments. |
| 3 | **Rate limiting is in-memory, per process.** Resets on restart; ineffective across multiple workers. | `rate_limit.py` | Brute-force protection is partial. Also uses `request.client.host`, which behind a proxy may be the proxy's IP unless forwarded headers are configured. |
| 4 | **CORS defaults to `*`.** | `main.py`, `config.py` | Harmless in the loopback topology; set `CORS_ALLOWED_ORIGINS` if the API is ever exposed. |
| 5 | **JWT cannot be revoked; 7-day lifetime; HS256 shared secret.** Ephemeral secret if unset. | `security.py`, `config.py` | Standard trade-off; ensure `JWT_SECRET_KEY` is set in every deployment (render.yaml does). |
| 6 | **bcrypt truncates passwords at 72 bytes** (and bcrypt 5 can raise on longer input). No max-length check in `signup`. | `security.py`, `auth.py` | Edge-case failure/surprise for very long passwords; add a max length or pre-hash. |
| 7 | **Deprecated APIs.** `dt.datetime.utcnow()` (naive UTC) used throughout; FastAPI `@app.on_event`; SQLAlchemy 1.x-style `Column` declarations on 2.0. | many files | Works today; will emit warnings/break in future versions. Timezone-naive datetimes are a latent bug source. |
| 8 | **Nullable `team_id` on core tables** (`datasets`, `questions`, `dashboards`, `audit_trail`) relies on app discipline. | `models.py` | A missed filter would leak across tenants. A DB `NOT NULL` (possible on Postgres) or Row-Level Security would add a safety net. |
| 9 | **Inconsistent JSON typing.** `dashboards.view_overrides_json` is `Text` holding JSON while neighbours use JSON/JSONB; `OrgAuditEvent.detail_json` uses plain `JSON`. | `models.py` | Minor; harder to query, inconsistent. |
| 10 | **Workers inside the API process.** Heavy runs share the process with HTTP handling; concurrency = workers × processes. | `jobs.py`, README | Fine for MVP; split to a separate worker process to scale. |
| 11 | **Polling UI** and no websockets/streaming. | README | Latency/load trade-off, acknowledged. |
| 12 | **Unvalidated with real models in places.** Docs say the Llama key was rejected, the stats feature wasn't validated against a real model (Gemini 429), only one dataset/question tried for the hypothesis engine. | `HYPOTHESIS_ENGINE.md`, `STATISTICAL_REASONING.md` | Offline tests prove plumbing, not model quality on varied schemas. |
| 13 | **`call_text` (Gemini) has no retry/failover**, unlike `call_tool`. Used for narrative text paths. | `llm_client.py:call_text` | Those paths fail on the first 429. (I saw the function; I did not trace every caller.) |
| 14 | **Triage swallows non-quota errors and lets the question through.** | `triage.py` | Intentional ("an outage must not block real questions") but spends budget on junk during outages. |
| 15 | **`status` string checks of 429 via `"429" in str(e)`** are brittle. | `triage.py` | Works, but classification already exists in `llm_client.classify_failure`. |
| 16 | **Uncommitted change** in `frontend/style/mindset.css` at session start. | git status | Not an issue, just something to commit or discard. |
| 17 | **Windows specifics.** `preexec`-based limits and privilege dropping are POSIX-only; on Windows the subprocess runner has fewer controls. | `runner.py` | Matters only for local dev on Windows (the platform you're on); use Docker there. |
| 18 | **Prompt injection is mitigated, not solved** (docs say so). The deterministic checks bound the damage; KPI labels/narrative are model-written text rendered to users. | `prompts.py`, `dashboard.py` | Keep using `esc()` on any model/data text rendered with `unsafe_allow_html`. |

### 21.3 If you want to deepen your understanding next
1. Read `agents/graph.py` + `runtime.py` + `jobs.py` together (control flow).
2. Read `verification.py` against `tests/test_verification.py` (the rules become concrete).
3. Run `python -m app.evals run` (offline, no API key) and read the report in `docs/eval_reports/`.
4. Open `/docs` on a running backend to see every endpoint with its schema.
5. Read `stats/selector.py` last — it is the densest module.

---

## 22. A–Z glossary

- **Adapter pattern** — two clients (Gemini, Llama) hidden behind one `call_tool` interface.
- **Admission control** — refusing new work when a limit (5 active jobs/team) is hit.
- **AgentState** — the shared dictionary passed between LangGraph nodes.
- **Alembic** — migration tool for SQLAlchemy.
- **Alert deduplication** — updating an open alert instead of raising another.
- **ASGI** — async Python web server interface (Uvicorn implements it).
- **AST** — abstract syntax tree; used to parse formulas safely.
- **Audit trail** — rows linking each dashboard element to its code, data and critic review.
- **Autoflush** — SQLAlchemy writing pending changes before queries (disabled here).
- **Backfill** — filling new columns/rows for pre-existing data.
- **bcrypt** — slow salted password hash.
- **Bearer token** — credential carried in the `Authorization` header.
- **BCa bootstrap** — bias-corrected accelerated bootstrap interval.
- **Benjamini–Hochberg** — false-discovery-rate p-value adjustment.
- **Budget** — caps on LLM calls, sandbox runs, cost, time.
- **CAS (compare-and-set)** — conditional UPDATE so only one writer wins.
- **Checkpoint** — function called at safe points to stop cancelled/overdue runs.
- **Clarify route** — the system asks you which of several meanings you intend.
- **Cochran's rule** — sample-size vs skewness rule for trusting a t interval.
- **Community** — an organization (customer company).
- **ContextVar** — per-thread/task variable; holds the current `RunContext`.
- **Cosine similarity** — vector closeness measure.
- **CORS** — browser cross-origin policy.
- **Critic** — second agent (Llama) that independently re-checks results.
- **Dependency injection** — FastAPI `Depends`.
- **Differential testing** — comparing two independent implementations.
- **Dockerfile / image / container** — recipe / packaged filesystem / running instance.
- **DQ** — data quality.
- **Element id** — UUID tying a dashboard KPI/chart to its audit row and evidence.
- **Embedding** — numeric vector for text meaning.
- **Evidence record** — stored list of checks and their results for one claim.
- **Executor** — agent that writes and runs code per step.
- **Failover** — switching providers on failure.
- **Fail closed** — refuse rather than degrade to something unsafe.
- **Fast path** — rule-based answer with no model.
- **Fencing** — rejecting writes from an owner that lost its lease.
- **FK (foreign key)** — column referencing another table's row.
- **Function calling** — model returns a structured call instead of prose.
- **gVisor (runsc)** — user-space kernel for stronger container isolation.
- **HAC / Newey–West** — autocorrelation-robust standard errors.
- **HC3** — heteroskedasticity-robust standard errors.
- **Heartbeat** — periodic lease renewal.
- **HNSW** — approximate nearest-neighbour index.
- **Holm** — step-down multiple-testing correction.
- **Hypothesis engine** — model proposes, code tests, rules decide.
- **Idempotency** — repeating a request has the same effect as once.
- **Invariant** — property that must always hold (share ≤ 100%).
- **JSONB** — Postgres binary JSON type.
- **JWT** — signed token with claims.
- **KS test** — Kolmogorov–Smirnov test comparing distributions.
- **LangGraph** — graph-based agent orchestration library.
- **Lease** — time-limited ownership of a job.
- **Lesson** — stored note about a past mistake (often from a Critic rejection).
- **Manifest** — immutable record of what a run was.
- **Metric engine** — safe formula evaluator.
- **Multi-tenancy** — many teams in one app, isolated by `team_id`.
- **Multiplicity** — inflated false positives from many tests.
- **Newcombe interval** — score-based CI for proportion differences.
- **Non-determinism** — varying outputs for identical input.
- **OIDC** — OpenID Connect (Google sign-in protocol).
- **ORM** — object-relational mapper.
- **Permutation test** — significance by shuffling labels.
- **pgvector** — Postgres vector extension.
- **Planner** — agent that breaks a question into steps.
- **Pettitt test** — single change-point detection.
- **Provenance** — where a result came from (data hash, code, versions).
- **PSI** — Population Stability Index (drift).
- **Pydantic** — validation/serialization library.
- **RAG** — retrieval-augmented generation.
- **Reference implementation** — second, independent computation used to cross-check.
- **Router** — picks the cheapest adequate path.
- **Run event** — one trace row (route/node/llm_call/sandbox_run/...).
- **Salt** — random value added before hashing a password.
- **Sandbox** — restricted environment for untrusted code.
- **Semantic layer** — governed business definitions.
- **Shapley value** — fair attribution of a total change among inputs.
- **SHA-256** — cryptographic hash used for fingerprints and invite tokens.
- **Signature (analysis)** — hash identifying an equivalent analysis.
- **Soft delete** — mark deleted instead of removing.
- **Streamlit** — Python UI framework.
- **Tool schema** — JSON schema of the structured output a model must produce.
- **Triage** — cheap relevance gate before the expensive pipeline.
- **Untrusted data** — anything from uploads; never obeyed as instructions.
- **Verdict** — VERIFIED / VERIFIED_WITH_CAVEATS / UNVERIFIED.
- **Version pinning** — a run records exactly which dataset version it used.
- **Welch's t-test** — unequal-variance two-sample mean test.
- **Worker / janitor** — threads that run jobs / recover abandoned ones.
