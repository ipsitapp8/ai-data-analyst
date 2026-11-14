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

When a model provider is configured, "why" questions go to the hypothesis-driven engine instead, which
builds on these decompositions: a model proposes competing explanations and the same deterministic code
tests them. See [HYPOTHESIS_ENGINE.md](HYPOTHESIS_ENGINE.md); the statistical methods it uses and how they are
selected are in [STATISTICAL_REASONING.md](STATISTICAL_REASONING.md). Without a provider, or with
`HYPOTHESIS_ENGINE_ENABLED=false`, this investigator answers them as before.

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

**Asking in words, without a model.** `POST /api/investigations/ask` takes a dataset and a question. The metric is
the one numeric column the question names (summed); the periods are read from the question against the dataset's own
calendar, not today's date:

| Phrase | Read as |
|---|---|
| last / previous / past month, quarter, year (optionally "last 3 months") | the most recent complete calendar unit(s) in the data, against the same number before |
| this / current month, quarter, year | the unit to date, against the same number of days at the start of the one before |
| last N days, last N weeks, yesterday | the final days of data, against the same number of days before |
| in March, in Sept 2025, in Q2 | that calendar month or quarter, against the one before |

A period the data only partly covers is cut, together with its comparison period, to the same number of days. A period
outside the data is refused with the reason. A question that names no readable period falls back to the midpoint split,
and the report says so.

**Unequal lengths.** A sum over 31 days exceeds one over 28 with nothing having changed. When the two periods differ in
length by more than 2%, a sum or count is compared like for like: the earlier value is scaled to the later period's
length, the part of the change that is only the extra days is reported as its own row, and segment contributions add up
to the like-for-like change. Averages and ratios are not scaled.

**Uncertainty.** The change, and each of the five leading contributors, is tested as a series of time buckets with
Newey-West errors (`app/stats`), never row by row. The contributors' p-values are Holm-adjusted as one family. A change
within the metric's ordinary variation makes the report inconclusive. A trend already under way in the earlier period is
measured and reported as an alternative reading.

**Evidence table.** `report.evidence_table` has one row per candidate explanation. `kind` is `measured` (a location of
the change, an offsetting segment, the period-length effect, a price or volume effect, a pre-existing trend) or
`hypothesis` (not tested, and not testable from the dataset). `support` is `strong`, `moderate`, `weak`, `offset` or
`exact`; it rates the evidence that the change is located there, never that it was caused there. Nothing is rated
`strong` when the periods are not comparable (missing days, a period the data only partly covers), and such a report
is inconclusive.

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

`backend/app/scenarios.py`, `scenario_sim.py`, `scenario_solve.py`, `scenario_verify.py`

Two explicit models, `unit_economics` and `funnel`, with their formulas returned in every response.

- Inputs are validated against physical limits; impossible values are rejected with the reason.
- Base, optimistic and pessimistic scenarios; sensitivity table; Shapley contribution of each
  changed input, summing exactly to the change.
- Baselines can be derived from dataset columns; each input reports its formula or "assumed".
- Price elasticity is the only behavioural link, and it is a number the user types.

Every result is labelled `kind: "scenario"` with a disclaimer and a list of assumptions the data
does not support. It is arithmetic, not a forecast.

### 10.1 Break-even

Returned inside every `evaluate` response (`break_even.baseline` and one per scenario).

- `unit_economics`: break-even quantity = fixed cost / (price − unit cost); break-even revenue; distance from it
  in units; units still needed; safety margin. When a unit does not contribute (price at or below unit cost) there
  is no break-even and the response says `never` or `flat` with the reason, never a number.
- `funnel`: the model has no fixed cost, so profit is proportional to visitors. The result is therefore a
  threshold on conversion rate, order value or cost per visitor, and `profitable_at_any_scale` or `never`.
- A profit curve over the break-even variable is returned for the chart.

### 10.2 Risk simulation (Monte Carlo)

`POST /api/scenarios/simulate`

The caller states, per input, a distribution: `normal` (`sd_pct` or `sd`), `uniform` (`low`/`high`, as `_pct` or
absolute) or `triangular` (`low`, `mode`, `high`). Percentages are of the scenario value. Inputs with no
distribution are held fixed and listed as fixed. A request with no uncertain input is refused: no range is ever
assumed on the caller's behalf, and none is estimated from data in this version.

- Draws use a Gaussian copula. The optional correlation matrix is validated (square, symmetric, unit diagonal,
  entries in [−1, 1], positive semidefinite). Each distribution is truncated to the input's physical range, and the
  probability mass that was cut off is reported.
- Volume follows a sampled price through the stated elasticity unless volume was adjusted or has its own distribution.
- Outputs: mean, median, standard deviation, min, max, 5th/25th/75th/95th percentiles for every output, plus for
  the chosen outcome: probability of being positive, negative, above the baseline and reaching an optional target
  (each with its sampling standard error), a 40-bin histogram, and "low / expected / high" (5th percentile, mean,
  95th percentile). Low and high describe the simulation; they are not bounds.
- 100 to 200,000 iterations (default 10,000). The seed is returned; a fixed seed reproduces the run exactly. Raw
  draws are never returned.

### 10.3 Goal seek

`POST /api/scenarios/goal-seek`

Solves `output(input = x) = target` by numerical root finding on the same deterministic engine (grid search for
sign changes, Brent refinement, then re-evaluation by the original model). Status is one of `solved`,
`multiple_solutions` (all roots returned; the one closest to today's value is recommended), `no_solution`
(with the range the output can reach), `no_feasible_solution` (roots exist but break a constraint),
`no_dependence`, `undefined` (a division by zero everywhere) or `convergence_failure`. No solution is ever
fabricated. Optional bounds and up to five constraints on any input or output. An input with no physical upper limit
is searched up to a stated ceiling, and the result says so. Roots closer together than a grid step, or that only
touch the target, can be missed; the grid size is returned.

### 10.4 Independent verification

Every `evaluate`, `simulate` and `goal-seek` response carries `verification`. A second implementation written in
plain Python (separate formulas, a subset-formula Shapley, a hand-written percentile, an independent bisection
scan) recomputes the results and compares. Status is `unverified` if any required check fails or did not run;
otherwise `verified`. The block states its scope: the calculations were checked, the realism of the inputs and
ranges was not (the check `input_ranges_supported_by_data` is always `not_run`). If a verifier itself fails, the
result is `unverified` with the reason. Tampered results are reported `unverified` (covered by tests).

### 10.5 Historical elasticity estimation (Phase 2)

`backend/app/scenario_elasticity.py`, `POST /api/scenarios/elasticity`

A suggestion, never an override: the slider value the user typed stays authoritative until they press the button that uses
the estimate, and a saved scenario records `assumption_sources` (manual, or the estimate with its interval, engine version,
dataset version and content hash).

- **Model.** `ln(quantity) = a + e x ln(price) + controls`, least squares with the same `stats/series.py` code the
  root-cause analysis uses: Newey-West errors for a single time series, HC3 otherwise, t-based intervals.
- **Observations.** With a date column, rows are aggregated to day/week/month/quarter (quantity summed, price
  quantity-weighted; `auto` picks the coarsest period with at least 24 observations). Without one, each row is an
  observation and the result is cautioned. A segment column adds a per-segment intercept.
- **Controls.** A linear trend (12+ observations); month-of-year or quarter dummies when the data spans two years;
  segment intercepts. A control that cannot be supported, or that is collinear with price, is dropped and reported.
- **Left out and counted:** missing values, non-finite values, price of zero or less, zero or negative quantity.
  Outliers are found by studentized residuals re-checked after each removal; the estimate with and without them is shown,
  and exclusion is the user's choice.
- **Refused (`status: not_estimated`) with reasons:** fewer than 8 usable observations, fewer than 3 distinct prices, a
  price spread under 2%.
- **Returned:** estimate, standard error, interval, p-value, observations, R-squared, residual autocorrelation, outlier
  report, controls used, a reliability level (`usable`, `use_with_caution`, `not_recommended`) with the reasons,
  limitations, the dataset provenance (version, fingerprint) and an independent check (SVD least squares, closed-form
  slope, the interval formula, and the observations rebuilt row by row in plain Python).

### 10.6 Custom business models (Phase 2)

`backend/app/scenario_formula.py`, `scenario_models.py`, `scenario_model_store.py`, `routers/scenario_models.py`;
tables `scenario_models` and `scenario_model_versions` (migration `c4e6a8b0d2f1`).

A model is a validated definition: inputs (with units, limits and an optional data binding), assumptions, outputs (a formula
each, optionally `additive` or a `ratio_of` two outputs), a headline output and constraints.

- **Safe formulas.** Parsed with `ast`, never evaluated as Python: numbers, names, `+ - * /`, a power with a constant
  exponent, brackets and `abs sqrt log exp floor ceil min max`. Everything else is rejected when parsed. Undefined results
  (division by zero, log of a non-positive number) are `None`, never 0.
- **Dependencies.** Outputs are computed in dependency order; a cycle is refused and the loop is named
  (`a → b → a`); an unknown name gets a suggestion. Every problem is reported at once.
- **Units.** Units are checked dimensionally (adding unlike units, a declared unit that does not match, log of a unit). A
  mismatch is a warning, or an error when the model sets `strict_units`. Percent and fraction are both dimensionless, so a
  missing x100 is not caught.
- **Semantic layer.** An input may be bound to an approved metric, a term, or the user's own formula. Only an
  approved metric is accepted (checked at save and again when a baseline is derived); a draft or deprecated metric is refused
  with its status. A term that matches two approved metrics returns 409 with the candidates. A formula binding is allowed
  and labelled "not an approved metric". An output may be reconciled against an approved metric.
- **Same engines.** A compiled model has the shape of the built-in specs, so evaluation, simulation, goal seek, sensitivity
  grids, segments and period comparison run on it unchanged. A custom model matches the built-in `unit_economics` exactly
  when written as one (tested). The engine adds no hidden behaviour: there is no built-in elasticity or break-even for a
  custom model, and constraints at `error` level reject a scenario (or drop that Monte Carlo draw or grid cell).
- **Versions.** A saved definition is immutable. Editing adds a version; a scenario refers to `custom:<id>@<version>` and so
  keeps the version it used. Models are team-scoped (another team's looks like a missing one), can be archived and restored,
  and names are unique per team.
- **Verification** uses a separate evaluator that re-parses the definition's text and shares no code with the engine.

### 10.7 Time-aware baselines and period comparison (Phase 2)

`backend/app/scenario_time.py`; `POST /api/scenarios/baseline` (now with `date_column`, `period`, `dataset_version`),
`GET /api/scenarios/periods`, `POST /api/scenarios/compare-periods`.

- Periods are a year, quarter, month or custom range, taken from the dataset's own calendar (only periods with rows are
  offered). A period with no rows is an error, not a baseline of zeros.
- Every response says what kind of figure it is: `historical_actual` (recorded in a period), `scenario` (arithmetic on
  inputs), `statistical_estimate` (elasticity), and `forecast`, which this feature does not produce.
- A baseline can pin a dataset version and returns its provenance (version, whether a newer one exists, content hash); a saved
  scenario keeps `baseline_source` and the version it used. Baselines are recomputed by the metric engine's independent
  csv path.
- A period comparison gives both periods, input and output changes, and a Shapley attribution of the change in the headline
  output to the inputs that moved. It describes the past; it does not say why inputs moved.

### 10.8 Segments (Phase 2)

`backend/app/scenario_segments.py`; `GET /api/scenarios/segment-values`, `POST /api/scenarios/segments`.

Segments partition the rows by one category column (a missing value is its own segment; at most 50). The total is the sum
of the segments for `additive` outputs, ratios are recomputed from those sums, and other outputs are reported as not
additive. A cost the data does not split is either a total allocated once (equally or in proportion to another input, in shares
that sum to 1) or a value used as entered in every segment. A segment's own adjustment replaces a shared one for the same
input, so no input is changed twice. Verification recounts segments from the file and checks that the segments add up to the
figure computed from all rows at once: a mapped input that is not additive (a simple average of a price column) makes the
result `unverified`.

### 10.9 Two-way sensitivity and attribution at scale (Phase 2)

`backend/app/scenario_sensitivity.py`; `POST /api/scenarios/sensitivity-grid`.

One output over a grid of two inputs (up to 61 x 61 cells) with the exact current position, the best and worst cells (best on
the grid, not a global optimum), and the points where the output crosses a threshold (break-even for profit at 0) by linear
interpolation along grid edges. Ranges that leave an input's allowed range are refused, not clipped. Shapley attribution is exact
for up to 10 changed inputs, computed from the subset formula (2^n evaluations rather than n! orders), and beyond that estimated
from sampled orders, each also reversed, with a standard error per input; the contributions sum exactly to the change either
way, and the response states which method was used.

### 10.10 Persistence

Saved scenarios need no schema change (their settings and results are JSON). Phase 2 adds only the two custom-model tables, in one additive Alembic revision (`c4e6a8b0d2f1`, after `b3d5f7a9c1e2`). `POST /api/scenarios` also keeps `assumption_sources` and `baseline_source` (dataset version, period, filters, mapping, fingerprint) and accepts an optional `simulation`; the server runs it, stores the result in
`results_json["simulation"]` and the settings, including the seed it used, in `adjustments_json["simulation"]`, so
a saved simulation can be reproduced exactly.

### 10.11 One service for the What-If page and the Silt Copilot (Phase 3)

`scenario_service.py` is the only place that calls the engines (`evaluate`, `simulate`, `goal_seek`, `grid`, `run_intent`, `ask`). The
What-If router and the Copilot bridge (`analyst/scenario_bridge.py`) both go through it, and a test fails if either calls an engine
directly. For the same inputs the two doors therefore give the same numbers and the same explanation text.

The bridge answers what-if and goal sentences about a Copilot dataset only when the dataset's column names map cleanly to a built-in
model (`unit_economics` or `funnel`; two candidate columns for one input means it does not guess). It asks for a fixed cost the data
does not hold instead of assuming one, remembers it for that conversation, and returns control to the existing Copilot for anything else
(plain questions, "increase revenue by 10%", a "why" with no scenario). Its notes say which columns were used and that the arithmetic
was independently checked.

### 10.12 Natural-language entry (Phase 3)

`POST /api/scenarios/ask` (`scenario_nl.py`). Rules first: the parser reads wording only (verbs, percent / percentage points / absolute
amounts, ₹, lakh, crore, k, goals such as "what price gives a 25% margin", break-even, comparisons). Every number must come from the user's
message. Optional `use_model` lets a language model translate a sentence the rules could not read (tool `submit_scenario_request`); its
output passes the same validator plus a check that each number appears in the user's message, so a model cannot supply a number. That path
is rate-limited (15 per 60 s). Ambiguity (two inputs match, no input named) returns `status: "clarify"` with options; unreadable input
returns `unsupported` with examples. The response always contains `interpretation`, so the user sees how the sentence was read before
the answer. Conversation context is held by the client, fingerprinted by model + baseline + normalised assumptions, dropped if any of
them change, and carried over only with "also/too/and". Follow-up answers are recomputed from stored adjustments, never from numbers in the context.

### 10.13 Explanations (Phase 3)

`scenario_explain.py`. Template-based, grounded in the result: every number in the text is taken from it, signed levels are
kept apart from unsigned changes, and causal wording is rejected by a test (`CAUSAL`). Sections: what changed, what accounts for it
(exact contributions), assumptions, risks, what this cannot tell you. `polish()` accepts an optional model rewrite only if all its
numbers are grounded and it has no causal wording; otherwise the template text stands.

### 10.14 Calculation evidence (Phase 3)

`POST /api/scenarios/inspect` ("How was this calculated?") returns the formula, each step with the user's numbers substituted, inputs
with their provenance (typed, dataset formula, elasticity estimate), assumptions, constraints and a `matches_engine` flag. With
`persist`, and whenever a scenario is saved, a sealed record is stored in `scenario_evidence` (migration `d5f7b9c1e3a2`, after
`c4e6a8b0d2f1`): request, result, verification, data version/fingerprint, seed, engine versions, and a SHA-256 over the canonical
request and result. Records are server-recomputed, team-scoped and append-only (`POST`, `GET` list/get, `POST /{id}/reproduce`; no
PUT/PATCH/DELETE). `reproduce` re-runs the stored request and reports `reproduced: false` if the record was altered, the engine version
changed, or the result differs, and says which.

Limits: the Copilot bridge covers only the two built-in models; segment results are not sealed; the model translator and the model polish have only been
tested with scripted providers, not a live one.

### 10.15 Saved scenarios you come back to (Phase 4)

No schema change: revision, data status and the rest are derived or kept in the scenario's existing JSON.

- **Reopen and edit.** `GET /api/scenarios/{id}` returns the stored settings, `revision` and `freshness`. In the page, *Open in the builder*
  loads the stored baseline, assumptions and percentage changes into the form and shows what it could not load (non-percentage changes,
  other named scenarios, simulation settings). `PUT /api/scenarios/{id}` recalculates on the server, raises `revision` and seals new
  evidence; evidence of earlier revisions is kept untouched (its `reproduce` still works).
- **Rename and duplicate.** `PATCH /api/scenarios/{id}` changes the name only. `POST /api/scenarios/{id}/duplicate` makes an independent,
  recalculated and sealed copy ("Copy of X", numbered if taken) that stays pinned to the data version the original used.
- **Data status** (`scenario_library.freshness`): `typed_inputs`, `current`, `stale` (a newer dataset version exists), `changed` (the pinned
  version's fingerprint no longer matches what was saved), `unavailable` (dataset or version gone). A revised or archived custom model is
  reported too. Saved numbers are never rewritten and a scenario is never moved onto newer data; re-taking the baseline is a deliberate step.
- **Comparison.** `GET /api/scenarios/compare?ids=` keeps its earlier keys and adds `analysis`: each scenario's change and % change against
  its own baseline, highest/lowest by value only, and warnings when starting inputs, assumptions or data versions differ or a scenario's data is
  stale. `format=csv` downloads it.
- **Export and sharing.** `GET /api/scenarios/{id}/export?format=json|csv|md` is the scenario exactly as saved (not recalculated) with the
  disclaimer, data status, verification, evidence references and a SHA-256 of its own content (also in `X-Content-SHA256`). Sharing means
  sending this file; there are no public links. CSV cells that begin with `= + - @` are defused so a scenario name cannot run as a
  spreadsheet formula.

Limits: segment and period-comparison results are not saved scenarios, so they have no reopen/export; reopening cannot restore a
non-percentage change into the form; the page was not run in a real browser.

### 10.16 Recalculating on newer data, and how the What-If feature is tested (Phase 5)

`POST /api/scenarios/{id}/rebase` takes the baseline again from another version of the scenario's dataset (the newest by default) using the
mapping, date column and period the scenario recorded, applies the same changes and assumptions, and saves the result as a **new** scenario
named "X (data vN)". The original is not touched. An input the data cannot supply (for example a fixed cost you typed) keeps the saved value
and is listed in `kept_from_saved`. The response holds `inputs` (was/now) and the comparison `analysis`. It refuses a scenario with typed
inputs only (400), a version it already uses (400), and a missing version (404). The Saved tab offers it only while the scenario is `stale`.

Test suites for the feature (all under `backend/tests/` unless noted):

| Area | File | What it proves |
|---|---|---|
| Engines, per phase | `test_scenario_phase1.py`, `_elasticity`, `_custom_models`, `_time`, `_segments`, `_sensitivity` | Hand-worked numbers; the independent verifier agrees; bounds and refusals |
| Natural language, explanations, evidence, Copilot bridge | `test_scenario_nl.py`, `_explain`, `_evidence`, `_bridge` | Numbers come only from the user; sealed records detect tampering; the Copilot gives the page's numbers |
| Saved scenarios | `test_scenario_library.py` | Edit/rename/duplicate, data status, comparison, export, isolation |
| End-to-end | `test_scenario_e2e.py` | The seven workflows (basic, Monte Carlo, goal seek, break-even, sentence, new data version, other team), each ending in a check done outside the code under test |
| Performance and bounds | `test_scenario_performance.py` | Time budgets (several times the measured cost) and exact limits: iterations 100-200,000, grid 61, 20 uncertain inputs, formula size/depth/exponent |
| Security | `test_scenario_security.py` | Every route of the three what-if routers (read from the routers, so a new route is covered automatically) refuses anonymous callers and callers with no or a foreign team; another team's id looks the same as a missing one; hostile names; prompt-injection sentences |
| Pages | `frontend/tests/test_pages.py` (What-If cases) | Streamlit AppTest of every control on the page |

Measured on a Windows laptop: 10,000 draws of a 14-input custom model 0.6 s; 200,000 draws 0.7 s; a 61 x 61 grid 0.06 s; attribution over 14
changed inputs (sampled) about 4 s; a baseline from a 100,000-row file a few seconds.

Known limits found while testing: a plain chain of more than about 20 additions in one formula is refused as too deep (split it across
named outputs); sentence entry refuses a percentage change outside -100% to +1000% (the same range as the form) while the API's `evaluate`
accepts any number and relies on the model's own constraints; `GET /api/scenarios/models` needs a login but no team because it returns only the
built-in catalogue.

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
| Investigations | `POST /api/investigations`, `POST /api/investigations/ask`, `GET /api/investigations`, `GET /api/investigations/{id}` |
| Data quality | `GET /api/datasets/{id}/quality`, `POST /api/datasets/{id}/quality/run`, `PUT /api/datasets/{id}/quality/config`, `POST /api/quality/incidents/{id}/acknowledge`, `POST /api/quality/incidents/{id}/resolve` |
| Alerts | `GET /api/alerts/{id}`, `POST /api/alerts/{id}/acknowledge`, `POST /api/alerts/{id}/resolve` |
| Copilot | `POST /api/copilot/sessions`, `GET /api/copilot/sessions`, `GET /api/copilot/sessions/{id}`, `POST /api/copilot/sessions/{id}/messages`, `DELETE /api/copilot/sessions/{id}` |
| Scenario models | `POST /api/scenario-models/validate`, `POST/GET /api/scenario-models`, `GET /api/scenario-models/{id}`, `GET /api/scenario-models/{id}/versions/{v}`, `POST /api/scenario-models/{id}/versions`, `POST /api/scenario-models/{id}/archive`, `POST /api/scenario-models/{id}/restore` |
| Scenario evidence | `POST /api/scenario-evidence`, `GET /api/scenario-evidence?scenario_id=&kind=`, `GET /api/scenario-evidence/{id}`, `POST /api/scenario-evidence/{id}/reproduce` |
| Scenarios | `GET /api/scenarios/models`, `POST /api/scenarios/baseline`, `POST /api/scenarios/evaluate`, `POST /api/scenarios/simulate`, `POST /api/scenarios/goal-seek`, `POST /api/scenarios/inspect`, `POST /api/scenarios/ask`, `POST /api/scenarios/elasticity`, `POST /api/scenarios/sensitivity-grid`, `POST /api/scenarios/segments`, `GET /api/scenarios/segment-values`, `GET /api/scenarios/periods`, `POST /api/scenarios/compare-periods`, `POST /api/scenarios`, `GET /api/scenarios`, `GET /api/scenarios/compare?ids=&format=`, `GET /api/scenarios/{id}`, `PATCH /api/scenarios/{id}`, `PUT /api/scenarios/{id}`, `POST /api/scenarios/{id}/duplicate`, `POST /api/scenarios/{id}/rebase`, `GET /api/scenarios/{id}/export?format=`, `DELETE /api/scenarios/{id}` |
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
