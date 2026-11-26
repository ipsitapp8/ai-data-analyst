# Silt: Product Requirements Document

## 1. Overview
Silt is an autonomous AI data analyst. A user uploads a CSV and asks a business question in plain English.
The system plans the analysis, runs generated code in a sandbox, verifies its own numbers and returns a dashboard
with a full audit trail. It also includes a what-if decision simulator and team workspaces.

## 2. Problem
- Business users cannot get answers from raw data without an analyst or SQL/Python skills.
- Existing AI tools return confident answers that are hard to check, and can quietly be wrong.
- Decisions ("what if we raise price 10%?") are made on spreadsheets with no record of assumptions.

## 3. Target users
- Business and operations teams who own data but not analytics skills.
- Small teams that need shared, access-controlled analysis workspaces.
- Analysts who want a fast first pass with an auditable trail.

## 4. Goals
1. Answer plain-English questions on uploaded CSVs with correct, verifiable numbers.
2. Make every claim traceable: each KPI, chart and sentence links to the code and result behind it.
3. Run AI-generated code safely.
4. Let users explore decisions (what-if) with transparent, reproducible arithmetic.
5. Support multiple teams with strict data isolation.

Non-goals: forecasting, establishing causation, real-time streaming data, non-CSV sources beyond basic uploads.

## 5. Core features
| Feature | Requirement |
|---|---|
| Upload and profile | CSV upload (streamed, size-capped), automatic profiling, versioned datasets with content hashes |
| Ask a question | Plain-English question, routed to the cheapest adequate path (clarify, fast, root-cause, statistical, standard) |
| Multi-agent analysis | Planner, Executor (bounded self-correction) and an independent Critic on a different model family |
| Sandboxed execution | Network-isolated, read-only, resource-limited Docker container per run; fails closed |
| Verification | Per-claim checks (recomputation, provenance, invariants); dashboard marked VERIFIED, VERIFIED_WITH_CAVEATS or UNVERIFIED |
| Dashboard and audit trail | KPIs, charts, narrative, with click-to-inspect evidence for every element |
| Root-cause investigator | Decomposes a metric's change between periods; reports inconclusive when evidence is thin |
| Analytical copilot | Multi-turn follow-ups as controlled operations (filters, drill-down, metric changes) |
| What-if simulator | Break-even, Monte Carlo risk, goal seek, sensitivity grids, custom formula models, period comparison, saved and comparable scenarios, recalculation on newer data, sealed calculation evidence |
| Monitoring and alerts | Scheduled analyses, KPI change alerts with deduplication and email delivery |
| Data quality | Schema, quality and drift checks on each upload |
| Semantic layer | Team-approved metric definitions and terms |
| Teams and auth | JWT login, team workspaces, roles; another team's data answers as not found |
| Reproducibility | Run manifests; rerun pinned to the original dataset version |

## 6. Principles
- **Verify, don't trust.** A failed or skipped check never counts as a pass.
- **Say what is not known.** Results carry limitations; causal wording is flagged.
- **Deterministic where possible.** Numbers come from code, never from the model alone.
- **Fail closed.** Unavailable isolation means no code runs, not weaker isolation.

## 7. Success metrics
- Verified-dashboard rate: share of runs ending VERIFIED or VERIFIED_WITH_CAVEATS.
- Answer accuracy on the offline evaluation suite (`python -m app.evals run`), with regressions blocking release.
- Time to first answer; fast-path questions need no model call.
- Cost per run within configured budgets (model calls, sandbox runs, USD).
- Zero cross-team data exposure (covered by the security test suite).

## 8. Architecture and tech stack
- **Backend:** Python, FastAPI, SQLAlchemy and Alembic, Postgres (pgvector), APScheduler
- **Agents:** LangGraph orchestrating Gemini (planner, executor, dashboard) and Llama (Critic)
- **Analysis:** pandas, NumPy, SciPy, scikit-learn, Plotly
- **Frontend:** Streamlit
- **Security and infrastructure:** Docker sandbox, JWT and bcrypt, Docker deployment on Render

## 9. Constraints and known limits
- Hosts that cannot start containers (Render's free tier, Hugging Face Spaces) cannot run generated code. There, only the fast path and the root-cause investigator work.
- Prompt-injection defences reduce risk but are not a guarantee; deterministic checks bound the damage.
- Rate limits are in memory and per process.
- Copilot rule parsing supports English only.
- The what-if page has been tested with Streamlit AppTest, not in a real browser.

## 10. Status
Core pipeline, sandbox, durable jobs, evaluation lab, verification, root-cause investigator, data quality, copilot,
semantic layer, monitoring and the what-if simulator (phases 1 to 5) are implemented and merged. See
`docs/PLATFORM.md` for behaviour and configuration, and `docs/IMPLEMENTATION_ROADMAP.md` for track status.
