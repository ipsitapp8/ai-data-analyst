# Implementation roadmap — autonomous analytics platform

Branch: `feature/autonomous-analytics-platform` (off `feature/postgres-migration` @ `d322b51`).
Nothing on this branch is committed; see "Checkpoint" at the bottom for the current state.

## Baseline (before any change)

| Command | Result |
| --- | --- |
| `.venv313/bin/python -m pytest -q` (SQLite) | 356 passed, 0 failed |

Pre-existing failures: none.

## Feature inventory (what existed before this work)

| # | Track | Already there | Missing |
| --- | --- | --- | --- |
| 1 | Hardened code execution | `sandbox/runner.py`: Docker runner (`--network none`, memory, CPU) and a subprocess fallback used by the hosted image | Read-only root, capability drop, pid/file/output limits, per-run directories, cancellation, artifact validation, fail-closed config, security tests |
| 2 | Durable jobs | Daemon thread per question; `questions.status` polling | Persisted queue, states, leases, recovery, cancellation, admission limits, idempotency |
| 3 | Evaluation lab | Unit tests only; none touch `agents/` or `sandbox/` | Cases, runner, metrics, persistence, comparison, CLI, report view |
| 4 | Evidence and verification | LLM Critic; `audit_trail` linking elements to logs; three-state verdict | Deterministic checks, per-claim evidence records, fingerprints |
| 5 | Reproducibility | Dataset versions pinned on questions | Content hash, run manifest, rerun, comparison, retention |
| 6 | Root-cause investigator | Nothing | Whole track |
| 7 | Data observability | `profiling.py`, `insights.data_warnings` (single version, not persisted as incidents) | Snapshots, baselines, drift metrics, incidents, UI |
| 8 | Stateful copilot | `chat.py`: stateless Q&A over one finished dashboard | Sessions, analytical state, controlled operations |
| 9 | Adaptive routing | Fixed Triage → Planner → Executor → Critic → Dashboard graph | Router, fast path, clarification, budgets, trace |
| 10 | What-if simulator | Nothing | Whole track |
| 11 | Monitoring and alerts | `scheduler.py`, `change_detection.py`, `alerts` table with read flag | Lifecycle, dedup, DQ gates, delivery retry, comparison modes |
| 12 | Semantic layer | `knowledge_notes` free text | Metric definitions, versions, formula validation, resolution, UI |

## Dependency order (why this sequence)

1. Sandbox and jobs first: every later track runs code or work through them, and they are the two high-severity audit findings.
2. Metric engine and semantic layer next, earlier than the spec's Stage E: the fast path (9), verification recomputation (4), investigator (6), copilot (8) and what-if (10) all compute through the same deterministic engine, and routing needs approved definitions to detect ambiguity.
3. Verification, evidence, manifests.
4. Routing, investigator, data quality, monitoring, scenarios, copilot.
5. Evaluation lab last among backend work, because it exercises everything above.
6. UI, docs, Postgres and Docker verification.

## Architecture decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Job queue | Table `analysis_jobs` in the existing database, claimed with a compare-and-set `UPDATE`, in-process worker threads with leases | Durable and recoverable without adding Redis or Celery; same pattern the scheduler already uses |
| Sandbox boundary | Hardened Docker container (optionally under gVisor via `SANDBOX_DOCKER_RUNTIME=runsc`) | Established isolation; verifiable locally |
| Subprocess runner | Kept for development only, refused unless `SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true` | Fail closed; a subprocess is not a security boundary |
| Deterministic computation | One `metric_engine` with a whitelisted formula grammar parsed through `ast` | Shared by fast path, verification, investigator, copilot, scenarios |
| Semantic graph | Relational tables | No demonstrated need for a graph database |
| Migrations | SQLite: `create_all` + additive columns; Postgres: one new Alembic revision | Project's existing mechanism |

## Progress matrix

Status is one of `complete`, `partial`, `blocked`. "Complete" means: backend logic, migrations, API
with authorization, UI, tests for success, failure, boundary and permission cases, docs.

| # | Track | Status | Evidence | What is not verified or not done |
| --- | --- | --- | --- | --- |
| 1 | Hardened code execution | **partial** | `test_sandbox_security.py`: 33 tests, 14 of them attack a real Docker container (network, filesystem, capabilities, fork bomb, memory, disk, output flood, cross-run leakage, symlinks, cancellation). Pipeline eval cases pass inside the Docker sandbox. | Hosts that cannot run Docker (Hugging Face Spaces, Render free tier) have no isolated option: generated code is refused there unless the operator opts into the unsafe runner. A remote sandbox service was not integrated. gVisor (`runsc`) is configurable but untested. Docker tests ran on macOS only, not on a Linux host. |
| 2 | Durable jobs | **complete** | `test_jobs.py`: 30 tests (idempotency, racing submissions, admission, exclusive claim, interruption and recovery, no double publish, lost lease, cancel, deadline, budget, retry, tenancy, worker pool start and stop). Live run with 2 workers. | Exclusive claiming was tested with threads, not with separate API processes. |
| 3 | Evaluation lab | **partial** | 39 offline cases pass; `test_evals.py`: 11 tests; stored run 1; report in `docs/eval_reports/`. | `--live` mode (real Gemini or Llama) is implemented but was never run: no paid call was made. Provider comparison therefore has no real results yet. |
| 4 | Evidence and verification | **complete** | `test_verification.py`: 31 tests; eval cases with planted wrong numbers, hallucinated KPIs and missing results are all caught. | Behaviour with a real model's output was not observed. A KPI the Critic did not recompute is reported as unverified, so real dashboards may show "with caveats" often until prompts are tuned. |
| 5 | Reproducibility | **complete** | `test_reproducibility.py`: 14 tests (identical inputs, changed data, changed prompts, changed model output, pinned rerun, immutable manifest). | — |
| 6 | Root-cause investigator | **complete** | `test_investigator_and_quality.py` (27 tests, shared with track 7); planted change located and reconciled to the cent. | Hypotheses are generated by rules from the data, not by a model. |
| 7 | Data observability | **complete** | Same file as track 6; drift eval cases. | No automatic seasonality model: seasonal data needs a pinned baseline. Snapshots profile at most 60 columns. |
| 8 | Analytical copilot | **complete** | `test_copilot_and_scenarios.py` (29 tests, shared with track 10); live run. | The rule parser understands English only. The model fallback was tested with the scripted provider, not a real model. |
| 9 | Adaptive routing | **complete** | `test_routing_and_semantic.py` (69 tests, shared with track 12; 18 parametrized routing decisions); budgets enforced exactly. | — |
| 10 | What-if simulator | **complete** | Same file as track 8 (incl. 12 rejected inputs). | Two models only (unit economics, funnel). |
| 11 | Monitoring and alerts | **complete** | `test_monitoring.py`: 18 tests (dedup, idempotency, min effect, weekday and rolling comparison, data-quality gate, delivery retry, lifecycle, permissions). | Email delivery was tested with the send function mocked; no real SMTP send. |
| 12 | Semantic layer | **complete** | Same file as track 9 (incl. tenancy and role checks). | Search is word matching over definitions, terms and notes; it does not use embeddings. |

Cross-cutting:

| Area | State |
| --- | --- |
| Migrations | SQLite additive path tested on an old-schema database. Alembic revision `f1a3b5c7d9e2` applied, downgraded and re-applied on Postgres 16 with pgvector. |
| UI | 6 new pages, 7 changed. `frontend/tests/test_pages.py`: 14 headless tests drive the real pages against the real API. Not looked at in a browser by a person or by the agent (the connected browser was on another machine). |
| Real model run | **Attempted, blocked by the providers.** `python -m app.evals run --live --provider gemini` was run four times on 2026-10-10 (stored as evaluation runs 2–4, plus one unstored). Gemini answered single calls but returned 503 (overloaded) and then 429 (quota) during the pipeline; the Llama key in `.env` is rejected with 401 `invalid_api_key`, so there was no working fallback. The app handled each case as designed (bounded retries, then a clear failure or the quota guidance message), but no model-driven analysis has completed end to end on the new code. Model-driven paths are otherwise verified only with the scripted provider. |
| Lint and type checks | The repository configures neither; none were run. |
| CI | `.github/workflows/tests.yml` now builds the sandbox image and runs the offline eval suite. The workflow itself has not been run. |

## Test results (final)

| Command | Result |
| --- | --- |
| `pytest` (SQLite, Docker image present) | 636 passed, 1 skipped |
| `TEST_DATABASE_URL=postgresql://silt:silt@localhost:5433/silt pytest` | 634 passed, 2 skipped |
| `cd backend && SANDBOX_BACKEND=docker python -m app.evals run` | 39 of 39 cases passed; stored as run 1 |
| `alembic upgrade head`, `downgrade -1`, `upgrade head` (Postgres) | all succeeded; vector index intact |
| Live server (uvicorn, 2 workers, Docker sandbox), 11-step script over HTTP | all steps passed |

Skips: on SQLite, the Postgres schema test; on Postgres, the two SQLite migration tests. Baseline
before this work was 356 passed; none of those tests was changed, weakened or removed.
(`backend/tests/conftest.py` gained four environment lines: job workers off, unsafe runner allowed
for pipeline-logic tests, and blank provider keys so no test can reach a paid model.)

## Files

New backend modules: `runtime.py`, `jobs.py`, `metric_engine.py`, `fastpath.py`, `verification.py`,
`provenance.py`, `semantic.py`, `investigator.py`, `data_quality.py`, `scenarios.py`, `copilot.py`,
`agents/router.py`, `agents/deterministic.py`, `evals/` (runner, scripted provider, CLI),
`routers/{_common,jobs,evidence,investigations,quality,copilot,scenarios,semantic,evals}.py`.

Rewritten: `sandbox/runner.py`, `agents/dashboard.py` (compile → verify → publish), `agents/graph.py`.

Changed: `main.py`, `models.py` (17 tables, 24 columns), `database.py`, `schemas.py`, `config.py`,
`scheduler.py`, `agents/{llm_client,llm_client_llama,critic,triage,planner,executor,prompts,state}.py`,
`routers/{alerts,scheduled,inspect}.py`, `sandbox/Dockerfile`.

Frontend: pages 12–17 new; `3_Analyses`, `4_Dashboards`, `10_Scheduled`, `7_Settings`, `1_Overview`,
`6_Audit_Trail`, `style/theme.py`, `api_client.py` changed.

Other: `backend/alembic/versions/f1a3b5c7d9e2_platform_tables.py`, `backend/evals/` (fixtures, cases,
generator), `.env.example`, `Dockerfile`, `README.md`, `.github/workflows/tests.yml`, `docs/`.

## Checkpoint

State on 2026-10-10: all code is in the working tree of `feature/autonomous-analytics-platform`,
**uncommitted** (no commit or push was requested). The tree is coherent: both test runs above are green.

Local environment notes:

- `backend/data/app.db` was upgraded in place by starting the app (additive). Backups:
  `app.db.bak-pre-postgres-branch` and `app.db.bak-pre-platform-<timestamp>`.
- `.env` still says `SANDBOX_BACKEND=subprocess`. With the new fail-closed rule that refuses generated
  code. The dev server was started with `SANDBOX_BACKEND=docker` in its environment instead; change
  `.env` to `docker` to make that permanent.
- A stash on `main` holds earlier uncommitted frontend edits (`git stash list`).
- The Postgres test container is stopped, not removed (`docker compose start postgres`).

Next actions, in order:

0. Fix the provider credentials: the Llama key is invalid (401), and the Gemini key hit its quota. Until one provider works, only the fast path, root-cause, copilot and what-if features produce results.
1. Watch one real model-driven analysis end to end with real keys, and check how often KPIs come back
   "unverified: no matching recomputed value". If often, have the Critic prompt return one recomputed
   value per claimed figure.
2. Run `python -m app.evals run --live --provider gemini` and `--provider llama`, then
   `python -m app.evals compare`.
3. Decide the hosted story for track 1: a Docker-capable host, or integrate a remote sandbox service.
4. Push the branch and let the updated CI workflow run on Linux.
5. Review and commit.
