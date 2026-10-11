# Platform reference

What was added on top of the original agent pipeline, how each part behaves, and how to configure it.
For the status of each track and what is still unverified, see `IMPLEMENTATION_ROADMAP.md`.

## Contents

1. Code execution sandbox
2. Durable jobs
3. Evaluation lab
4. Verification and evidence
5. Reproducibility
6. Root-cause investigator
7. Data quality and drift
8. Analytical copilot
9. Adaptive routing and budgets
10. What-if simulator
11. Monitoring and alerts
12. Semantic layer
13. Security model
14. Observability
15. Configuration reference
16. API reference
17. Running the tests

---

## 1. Code execution sandbox

`backend/app/sandbox/runner.py`

Generated Python is treated as hostile. It runs in one fresh Docker container per execution:

| Control | Setting |
| --- | --- |
| Network | `--network none`: no internet, no host, no cloud metadata endpoint |
| Filesystem | `--read-only` root; dataset mounted read-only; one private output directory; `/tmp` is a size-capped `noexec` tmpfs |
| Privileges | non-root user, `--cap-drop ALL`, `no-new-privileges` |
| Resources | memory (no swap), CPU, pid limit, file-size limit, open-file limit, wall-clock timeout |
| Output | stdout and stderr capped; a process that floods output is killed |
| Lifetime | container removed after the run; killed on timeout and on cancellation |
| Environment | built from scratch; no application secret is passed in |

Each execution gets its own directory that is deleted afterwards, so one run cannot read another's files.

Artifacts are validated before anything leaves that directory: regular files only (symlinks are never
followed), plain `.json` names, a size cap, a count cap, and the content must parse as a Plotly figure.

**Fail closed.** If the configured boundary is unavailable the run fails with `sandbox_unavailable`.
Nothing falls back to a weaker runner.

**The subprocess runner is not a boundary.** `SANDBOX_BACKEND=subprocess` runs code as a child process
on the host: it shares the kernel, the filesystem and the network. It is refused unless
`SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true`, which is meant for development on your own machine.

**gVisor.** Set `SANDBOX_DOCKER_RUNTIME=runsc` to run the container under gVisor. This was not
tested here (gVisor is not installed on the development machine).

**Hosts without Docker.** Hugging Face Spaces and Render's free tier cannot start containers. There,
analyses that need generated code are refused unless the unsafe flag is set knowingly. The fast path
and the root-cause investigator still work, because they run no generated code.

## 2. Durable jobs

`backend/app/jobs.py`

An analysis is a row in `analysis_jobs`, not a thread. Workers claim rows; nothing is lost on restart.

```
queued ──claim──> running ──> succeeded | failed | cancelled | timed_out
   └──cancel──> cancelled
running ──lease expired──> queued        (attempts left, nothing published yet)
```

| Policy | Behaviour |
| --- | --- |
| Admission | at most `JOB_TEAM_MAX_ACTIVE` queued or running jobs per team; beyond that, `429` |
| Concurrency | `JOB_WORKERS` threads per API process, one job each |
| Idempotency | same `Idempotency-Key` from the same team returns the first submission |
| Ownership | a worker holds a lease and renews it; every terminal write requires still owning the job |
| Recovery | at startup and periodically: a running job with an expired lease is re-queued if it has attempts left, otherwise failed as `abandoned` |
| No double publish | if a dashboard already exists for the job, it is marked succeeded instead of re-run |
| Deadline | `JOB_DEADLINE_SECONDS` from first start; then `timed_out` |
| Cancellation | queued: immediate. Running: at the next checkpoint; a running sandbox process is killed |
| Retry | automatic only after abandonment. Otherwise on request: `POST /api/jobs/{id}/retry` |
| Errors | users see a safe message; tracebacks go to the server log only |

`questions.status` keeps its old values and gains `cancelled` and `needs_clarification`.

Limit of the in-process design: with several API processes each runs its own `JOB_WORKERS`, so total
concurrency is workers × processes. Claiming is still exclusive.

## 3. Evaluation lab

`backend/app/evals/`, cases in `backend/evals/cases/suite.json`, fixtures in `backend/evals/fixtures/`

```bash
cd backend
python -m app.evals run                              # offline, deterministic, no API key
python -m app.evals run --component pipeline
python -m app.evals run --unsafe-subprocess          # machine without Docker
python -m app.evals run --live --provider gemini     # real provider; costs money
python -m app.evals run --live --provider llama
python -m app.evals list
python -m app.evals compare <base> <new>             # exit code 1 on regression
python -m app.evals report <run>
```

Offline runs replace only the model, with a scripted provider. The graph, routing, real code
execution in the sandbox, verification and publishing all run for real.

Reference answers are computed by `backend/evals/build_fixtures.py` with plain Python, not by
application code. The suite is versioned (`suite_version`); bump it when a fixture or expectation changes.

Metrics per run: pass rate by component, model calls, code runs, executor retries, plan revisions,
failed verification checks, tokens, estimated cost and latency. Runs are stored in `eval_runs` and
`eval_results` and shown on the Evaluation page.

Live cases are marked `"live": true`, never run offline, and are scored by tolerance and allowed outcomes.

## 4. Verification and evidence

`backend/app/verification.py`

Every KPI, chart and narrative gets an evidence record with explicit checks.

| Check | Applies to | Passes when |
| --- | --- | --- |
| `result_present` | KPI | the cited step ran and produced a parseable result |
| `value_in_result` | KPI | the displayed value appears in the step's result, to display precision |
| `independent_recomputation` | KPI | a second computation produced the same value |
| `provenance` | KPI, chart | an execution record exists and the dataset file still matches its recorded hash |
| `invariants` | KPI | finite; shares within 0–100%; counts non-negative |
| `narrative_numbers_supported` | narrative | every number in the text appears in a computed result |
| `independent_review` | narrative | the Critic accepted the analysis |
| `causal_language` | narrative | informational: flags causal wording |
| `artifact_validated` | chart | the file came from a successful step and passed artifact validation |

Outcomes are `pass`, `fail`, `not_run`, `warn`, `not_applicable`.

**Status rules**

| Status | When |
| --- | --- |
| `unverified` | any required check failed, or any required check could not run |
| `verified_with_caveats` | all required checks passed, and there is a warning or a limitation |
| `verified` | all required checks passed, nothing else to report |

A check that did not run never counts as passed.

**Dashboard verdict.** A failed required check anywhere makes the dashboard `UNVERIFIED`, whatever
the Critic said. Claims that are merely unconfirmed lower `VERIFIED` to `VERIFIED_WITH_CAVEATS`.
Passing checks never upgrade a Critic rejection.

**Recovery.** If verification fails a required check and the revision budget allows, the run goes
back to the Planner once with the failures as feedback. If it still fails it is published as
unverified. It never loops and never passes by exhausting retries.

**Three kinds of validity**, reported separately: mathematical (what the checks test), statistical
(sample adequacy; flagged when under 30 rows, otherwise "not assessed") and causal (never established
here; causal wording is flagged).

Figure matching: a displayed figure matches a computed one if they differ by less than one unit of
the last displayed digit. `VERIFY_REL_TOLERANCE` can widen that; the default is 0.

## 5. Reproducibility

`backend/app/provenance.py`

- Every dataset version has a SHA-256 content hash.
- Every finished run gets a manifest: question, dataset hash, plans, the hash of every script,
  prompt and application versions, models and token usage, environment, configuration, outcome.
  It is written once and never changed.
- `POST /api/questions/{id}/rerun` runs the same question pinned to the original dataset version.
  The original run's records are not touched.
- `GET /api/questions/{a}/compare/{b}` reports which recorded inputs differ and attributes a
  difference in results to data, to execution or model variation, or to both.

Model output is not deterministic. A manifest records what a run was; it does not promise that a
rerun produces the same output, and the comparison says so.

**Retention.** Sandbox workspaces are deleted when a run ends and swept after
`RETENTION_WORKSPACE_DAYS`. Database records are never deleted by retention.

## 6. Root-cause investigator

`backend/app/investigator.py`

Deterministic decomposition of a metric's change between two periods.

- Additive metrics (`sum`, `count`): segment contributions that add up exactly to the change.
- Ratios (`sum/sum`, `avg`): rate effect and mix effect per segment, adding up exactly.
- Price and volume, when the metric is a summed amount and a quantity column exists.
- Each leading contributor gets an evidence-quality rating from row counts and from whether the
  movement holds in both halves of each period.
- Hypotheses with the test applied and its result; one drill-down level into strong contributors.

Bounded by dimensions, segments, drill-downs, engine calls and time. When evidence is thin the
report is `inconclusive`.

It locates a change; it does not establish a cause. Every report carries that caveat, lists what was
not examined and names alternative explanations.

A "why did … change" question is routed here automatically when the dataset has a date column and
one metric can be identified. `POST /api/investigations` takes an explicit metric and periods.

## 7. Data quality and drift

`backend/app/data_quality.py`

Every upload is snapshotted and compared with its baseline. Findings are stored as incidents in one of
three categories, kept apart from business KPI anomalies:

| Category | Examples |
| --- | --- |
| Schema | column removed, column added, type changed |
| Quality | row-count change, more missing values, duplicate rows, out-of-range values, stale data, freshness |
| Distribution | numeric drift, category mix drift |

Avoiding false alarms:

- No distribution comparison under `min_sample` values (default 100); it is listed as skipped.
- Numeric drift needs a PSI over threshold **and** a Kolmogorov–Smirnov rejection at 1%.
- The baseline defaults to the previous version and can be pinned. For seasonal data, pin it to a
  version from the comparable season. There is no automatic seasonality model.

Thresholds are visible and editable per dataset (owner or admin). A failing quality check never
fails an upload.

## 8. Analytical copilot

`backend/app/copilot.py`

A session holds explicit state: metric, filters, grouping. Each message becomes controlled operations:
`set_metric`, `add_filter`, `remove_filter`, `clear_filters`, `set_time_range`, `drill_down`,
`drill_up`, `rebase`, `reset`.

- The answer is recomputed from the dataset on every turn and recomputed again independently.
- Rules interpret the message first. A model is asked only when no rule applies, and its output is
  validated against the same whitelist and the dataset's real columns.
- Ambiguity (a value in two columns, a term with two approved meanings) produces a question back
  and changes nothing.
- A newer dataset version is flagged; the session stays on its version until told to switch, and
  says what it had to drop.
- If nothing can be mapped, it says so and changes nothing.

Sessions are private to their creator within the team.

## 9. Adaptive routing and budgets

`backend/app/agents/router.py`, `backend/app/runtime.py`

| Route | When | Model calls | Code runs |
| --- | --- | --- | --- |
| `clarify` | a term has several approved meanings, or several columns match equally | 0 | 0 |
| `fast` | a simple aggregation the parser fully accounts for | 0 | 0 |
| `root_cause` | "why did … change", with a date column and one metric | 0 | 0 |
| `statistical` | trend, comparison, correlation, outlier wording | configured budget | configured budget |
| `standard` | everything else | configured budget | configured budget |

Routing is rule-based: deterministic, free, and every decision is recorded with its reasons.

Per-run budgets: `RUN_MAX_LLM_CALLS`, `RUN_MAX_SANDBOX_RUNS`, `RUN_MAX_COST_USD` and the job
deadline. A limit of N allows exactly N uses. Exceeding one ends the run as `budget_exceeded`.

The run trace (`GET /api/questions/{id}/trace`) lists the route, each node, each model call with
provider, tokens and estimated cost, and each code run.

## 10. What-if simulator

`backend/app/scenarios.py`

Two explicit models, `unit_economics` and `funnel`, with their formulas returned in every response.

- Inputs are validated against physical limits; impossible values are rejected with the reason.
- Base, optimistic and pessimistic scenarios; sensitivity table; Shapley contribution of each
  changed input, summing exactly to the change.
- Baselines can be derived from dataset columns; each input reports its formula or "assumed".
- Price elasticity is the only behavioural link, and it is a number the user types.

Every result is labelled `kind: "scenario"` with a disclaimer and a list of assumptions the data
does not support. It is arithmetic, not a forecast.

## 11. Monitoring and alerts

`backend/app/scheduler.py`

| Setting (per schedule) | Meaning |
| --- | --- |
| `change_threshold_pct` | percentage a KPI must move |
| `min_effect_abs` | and the minimum move in the KPI's own units |
| `comparison` | `previous`, `same_weekday`, or `rolling_mean` over `window_runs` |
| `suppress_on_dq` | hold KPI alerts while the data has open high-severity quality incidents |
| `notify_email` | send email; in-app alerts are always recorded |

- **Deduplication.** An alert still open for the same schedule, kind and KPI is updated
  (occurrence count, last seen), not raised or emailed again.
- **Idempotency.** Processing the same finished run twice does nothing the second time.
- **Lifecycle.** open → acknowledged → resolved. After resolution the same condition is a new alert.
- **Delivery.** A failed email is retried from its stored payload with backoff, at most
  `ALERT_DELIVERY_MAX_ATTEMPTS` times in total.
- **Explanation.** Each alert stores what it was compared with, the changes, the KPI's history and
  links to the run and its evidence.

Changing `suppress_on_dq` or `notify_email` needs an owner or admin.

## 12. Semantic layer

`backend/app/semantic.py`, `backend/app/metric_engine.py`

- Team-owned metrics with versioned formulas, units, aliases, owner and timestamps.
- Dimensions and entities map a word to a dataset column.
- Relationships between metrics, terms, datasets and columns; the graph view also shows the edges
  the definitions imply.
- A definition takes effect only when an owner or admin approves a specific version. A new version
  is a proposal until approved; history is kept.

Formula language: `sum`, `avg`, `median`, `min`, `max`, `count`, `count_distinct`, numbers and
`+ - * /`. It is parsed with `ast` and never executed as Python.

Resolution prefers the longest matching phrase. A phrase claimed by two approved metrics is reported
as ambiguous and becomes a clarifying question.

Nothing read from a file, a cell, a note or a model can create or change a definition.

## 13. Security model

| Area | Control |
| --- | --- |
| Generated code | section 1 |
| Tenancy | every new table carries `team_id`; every endpoint checks membership; another team's id answers `404` |
| Roles | owner or admin to approve or deprecate definitions, change quality thresholds, change alert delivery |
| Input | formulas through a whitelist parser; filters validated against real columns; ids typed; lengths capped |
| Prompt injection | dataset content is clipped, wrapped in an "untrusted" tag, and every agent prompt says not to follow instructions found in data |
| Model output | validated before use; never executed outside the sandbox; never trusted for numbers without checks |
| Secrets | the sandbox environment is built from scratch; job errors shown to users are fixed messages |
| Uploads | unchanged: CSV only, streamed, size-capped |

**Known limits**

- Prompt-injection defences reduce risk; they are not a guarantee. The deterministic checks are what
  bound the damage a manipulated model can do.
- Rate limits are still in memory and per process.
- The copilot's rule parser understands English phrasing only.
- CORS still defaults to all origins; set `CORS_ALLOWED_ORIGINS` when the API is exposed.

## 14. Observability

- `GET /api/health`: database, sandbox readiness and isolation, queue depth, workers.
- `GET /api/jobs/metrics`: jobs by state, queue depth, age of the oldest queued job.
- `GET /api/questions/{id}/trace`: per-run events with durations, tokens, cost, fallbacks, timeouts.
- `run_events` table: `route`, `node`, `llm_call`, `sandbox_run`, `verification`, `budget`.
- Logs use the existing logger. Labels are low-cardinality; no dataset content is logged.

## 15. Configuration reference

All optional. Defaults in brackets.

| Variable | Meaning |
| --- | --- |
| `SANDBOX_BACKEND` [`docker`] | `docker` or `subprocess` |
| `SANDBOX_ALLOW_UNSAFE_SUBPROCESS` [`false`] | allow the non-isolated development runner |
| `SANDBOX_DOCKER_RUNTIME` [empty] | OCI runtime, for example `runsc` |
| `SANDBOX_PIDS_LIMIT` [`128`] | processes per execution |
| `SANDBOX_TMPFS_SIZE` [`64m`] | size of `/tmp` in the container |
| `SANDBOX_MAX_STDOUT_BYTES` [`1048576`] | captured output per stream |
| `SANDBOX_MAX_ARTIFACT_BYTES` [`5242880`] | largest chart file kept |
| `SANDBOX_MAX_ARTIFACTS` [`20`] | charts kept per execution |
| `SANDBOX_MAX_CODE_BYTES` [`204800`] | largest script accepted |
| `JOB_WORKERS` [`2`] | worker threads per API process; `0` disables workers |
| `JOB_MAX_ATTEMPTS` [`2`] | attempts including recovery after interruption |
| `JOB_DEADLINE_SECONDS` [`900`] | wall-clock limit per job |
| `JOB_LEASE_SECONDS` [`60`] | lease length; recovery happens after it expires |
| `JOB_POLL_SECONDS` [`1.0`] | idle polling interval |
| `JOB_TEAM_MAX_ACTIVE` [`5`] | queued plus running jobs per team |
| `RUN_MAX_LLM_CALLS` [`40`] | model calls per run; `0` = unlimited |
| `RUN_MAX_SANDBOX_RUNS` [`30`] | code runs per run; `0` = unlimited |
| `RUN_MAX_COST_USD` [`0`] | estimated cost per run; `0` = unlimited |
| `LLM_FAILOVER_ENABLED` [`true`] | continue on the second provider when the first fails |
| `LLM_PRICE_GEMINI`, `LLM_PRICE_LLAMA` | USD per million tokens as `input,output`; estimates only |
| `RETENTION_WORKSPACE_DAYS` [`7`] | age after which leftover workspaces are deleted |
| `ALERT_DELIVERY_MAX_ATTEMPTS` [`3`] | total email attempts per alert |
| `ALERT_DELIVERY_RETRY_SECONDS` [`300`] | first retry delay; doubles each time |
| `DQ_MIN_SAMPLE` [`100`] | values needed to compare distributions |
| `DQ_PSI_WARN`, `DQ_PSI_HIGH` [`0.1`, `0.25`] | drift thresholds |
| `VERIFY_REL_TOLERANCE` [`0`] | extra relative tolerance for figure matching |
| `ENGINE_MAX_ROWS` [`2000000`] | largest dataset the deterministic engine accepts |
| `APP_VERSION` [`2.0.0`] | recorded in run manifests |

## 16. API reference

All endpoints need `Authorization: Bearer <token>`. All except `/api/evals/*` and
`/api/scenarios/models` also need `X-Team-Id`. Errors use FastAPI's `{"detail": ...}` shape.
List endpoints with `limit` and `offset` return `{total, limit, offset, ...}`.

**Changed**

| Endpoint | Change |
| --- | --- |
| `POST /api/questions` | queues a job; accepts `Idempotency-Key`; returns `job_id`, `job_state`, `created`; `429` at the team limit |
| `GET /api/questions/{id}/status` | adds `route`, `clarification`, `job` |
| `GET /api/questions/{id}/dashboard` | adds `evidence_summary`, per-element `evidence_status`, `route`, `dataset_fingerprint` |
| `GET /api/dashboards/{id}/elements/{id}/inspect` | adds `evidence` |
| `GET /api/health` | adds `sandbox`, `jobs`, `llm_failover_enabled`, `app_version` |
| `GET /api/alerts` | adds lifecycle fields and a `status` filter |
| `POST`, `PATCH /api/scheduled-analyses` | accept the monitoring settings in section 11 |

**New**

| Area | Endpoints |
| --- | --- |
| Jobs | `GET /api/jobs`, `GET /api/jobs/metrics`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/retry`, `GET /api/questions/{id}/job`, `POST /api/questions/{id}/cancel`, `GET /api/questions/{id}/trace` |
| Evidence | `GET /api/questions/{id}/evidence`, `GET /api/questions/{id}/manifest`, `POST /api/questions/{id}/rerun`, `GET /api/questions/{a}/compare/{b}`, `GET /api/datasets/{id}/versions`, `GET /api/datasets/{id}/versions/compare?a=&b=` |
| Investigations | `POST /api/investigations`, `GET /api/investigations`, `GET /api/investigations/{id}` |
| Data quality | `GET /api/datasets/{id}/quality`, `POST /api/datasets/{id}/quality/run`, `PUT /api/datasets/{id}/quality/config`, `POST /api/quality/incidents/{id}/acknowledge`, `POST /api/quality/incidents/{id}/resolve` |
| Alerts | `GET /api/alerts/{id}`, `POST /api/alerts/{id}/acknowledge`, `POST /api/alerts/{id}/resolve` |
| Copilot | `POST /api/copilot/sessions`, `GET /api/copilot/sessions`, `GET /api/copilot/sessions/{id}`, `POST /api/copilot/sessions/{id}/messages`, `DELETE /api/copilot/sessions/{id}` |
| Scenarios | `GET /api/scenarios/models`, `POST /api/scenarios/baseline`, `POST /api/scenarios/evaluate`, `POST /api/scenarios`, `GET /api/scenarios`, `GET /api/scenarios/compare?ids=`, `GET /api/scenarios/{id}`, `DELETE /api/scenarios/{id}` |
| Semantic | `GET`, `POST /api/semantic/metrics`, `GET /api/semantic/metrics/{id}`, `POST /api/semantic/metrics/{id}/versions`, `POST /api/semantic/metrics/{id}/versions/{v}/approve`, `POST /api/semantic/metrics/{id}/deprecate`, `POST /api/semantic/validate`, `GET`, `POST /api/semantic/terms`, `DELETE /api/semantic/terms/{id}`, `POST /api/semantic/relationships`, `DELETE /api/semantic/relationships/{id}`, `GET /api/semantic/graph`, `POST /api/semantic/resolve`, `GET /api/semantic/search?q=` |
| Evaluation | `GET /api/evals/suite`, `GET /api/evals/runs`, `GET /api/evals/runs/{id}`, `GET /api/evals/compare?base=&new=` |

Interactive documentation is at `/docs` on the running API.

## 17. Running the tests

```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt -r requirements-dev.txt

./backend/app/sandbox/build.sh      # once; without it the Docker boundary tests are skipped
pytest                              # SQLite

docker compose up -d postgres
TEST_DATABASE_URL=postgresql://silt:silt@localhost:5433/silt pytest

cd backend && python -m app.evals run
```

No test calls a real model. Tests that need one use the scripted provider or patch the call.
