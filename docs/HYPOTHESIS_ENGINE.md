# Hypothesis-driven investigation

Answers "why did X change?" questions by letting a model propose competing explanations and having
deterministic code test each one. The model never computes a number and never runs code.

## What is new compared with the root-cause investigator

| | Root-cause investigator (`investigator.py`) | Hypothesis engine (`hypothesis_engine.py`) |
| --- | --- | --- |
| Who decides what to test | Fixed rules: decompose by every candidate dimension | A model proposes hypotheses from the question and the schema |
| Reads the question | Keyword match for a metric | Structured understanding: metric, population, periods, direction, claimed size, assumptions, missing information |
| Tests | Segment contribution, price/volume | Ten validated operations (below) |
| Follow-up | None | Bounded loop: the model picks the next analysis after seeing results |
| Needs a model | No | Yes; falls back to the investigator if none answers at the first step |

The decomposition maths is shared: the engine calls the investigator's functions rather than copying them.

## Flow

1. **Understand** (`submit_understanding`, 1 model call). The reply is validated: the formula must parse
   and use existing columns; an approved semantic-layer metric named in the question overrides whatever
   formula the model wrote; periods outside the data are replaced by an automatic split; period end dates
   are read as inclusive ("1 Jan to 28 Feb" includes 28 Feb); a claimed size of change is kept only if that
   number is written in the question. No metric or no date column ends the run with a clear reason.
2. **Baseline** (no model). `metric_change`, `period_coverage`, `data_quality`. A change under
   `MATERIAL_CHANGE_PCT` stops here: there is nothing to explain.
3. **Scan** (no model). `segment_contribution` along up to `HYP_SCAN_DIMENSIONS` dimensions. The model
   sees these results before proposing anything, and a concentration that no supported hypothesis covers
   is still listed under observed facts.
4. **Generate** (`submit_hypotheses`, 1 call). Each hypothesis must name one operation, its parameters and
   the outcome it predicts. Rejected with a recorded reason: unknown operation, missing or wrong-typed
   column, a prediction that does not fit the operation, a duplicate statement or duplicate test, empty
   entries. At most `HYP_MAX_HYPOTHESES` are kept, highest priority first.
5. **Test** (no model). Each operation runs through the metric engine on the pinned dataset version.
6. **Adapt** (`submit_next_step`, up to `HYP_MAX_ROUNDS` calls). The model sees statuses and results and
   chooses one more analysis or stops. Invalid or repeated proposals are discarded; two in a row end the loop.
7. **Report** (`submit_report`, 1 call). The model writes the summary. It is discarded in favour of a
   fixed template if it contains a number no executed analysis produced, or asserts a cause when none was
   determined. A suggested explanation or follow-up carrying an unsupported number is dropped. Evidence
   verification then checks the accepted summary a second time, independently.

If no model answers before a hypothesis exists (at the first call, or the model is lost right after it),
the question is handed to the deterministic investigator and still gets an answer. If the model is lost
later, the report is built from the evidence already gathered, with a template summary.

## Operations

`metric_change`, `component_change`, `share_change`, `distribution_shift`, `segment_contribution`
(optionally one named segment, optionally a drill-down inside a sub-population), `mix_shift`,
`driver_decomposition`, `price_volume`, `period_coverage`, `data_quality`. Defined in `analysis_ops.py`.
Formulas go through the same whitelisted grammar as the semantic layer; filters are column/operator/value
triples. There is no SQL and no generated code, so the sandbox is not used (`sandbox_runs` budget is 0).

Equivalent analyses share a signature (operation + normalised parameters) and are executed once; a second
hypothesis asking for the same analysis is judged against the stored result.

## Evidence rules

Statuses are set by code in `analysis_ops.evaluate` and `HypothesisEngine._settle`:

| Status | Meaning |
| --- | --- |
| `supported` | The analysis ran and its result matches the prediction, above the thresholds |
| `weakened` | The analysis ran and the result contradicts the prediction or is immaterial |
| `inconclusive` | Too few rows, not statistically distinguishable, or evidence pointing both ways |
| `not_testable` | The data cannot support the analysis (e.g. no quantity column for price/volume) |
| `failed` | The analysis raised an error |
| `proposed` | Never run (budget ended first) |

- `not_testable`, `failed` and `proposed` are absence of evidence. They never count against a hypothesis.
- There is no confidence score. Each status carries its reasons and an effect size where one exists.
- Concentration is judged against the segment's size: a segment holding 50% of the metric and 50% of the
  change is "in proportion", not a finding (`HYP_STRONG_SHARE`, and 1.5x its base share).
- Significance tests: Welch's test for means and a two-proportion z-test for shares, normal approximation,
  only with at least 30 rows per period; otherwise the test is reported as skipped. Totals get no test.
- A drill-down inside a finding is a refinement: recorded with the hypothesis, does not change its status.
- If a period is incomplete or the data-quality check finds problems, the determination is
  `data_cannot_support_conclusion` even when a hypothesis is supported.
- Supported means "the data is consistent with it". The report says so and never states a cause.

## Limits

| Setting | Default | Bounds |
| --- | --- | --- |
| `HYP_MAX_LLM_CALLS` | 8 | Model calls per investigation |
| `HYP_MAX_ANALYSES` | 24 | Executed analyses |
| `HYP_MAX_SECONDS` | 240 | Wall-clock time |
| `HYP_MAX_TOKENS` | 80000 | Tokens reported by the provider |
| `HYP_MAX_HYPOTHESES` | 6 | Hypotheses kept |
| `HYP_MAX_ROUNDS` | 4 | Follow-up rounds |
| `HYP_SCAN_DIMENSIONS` | 4 | Dimensions decomposed before hypotheses are proposed |

The job's own budget and deadline (`ROUTE_BUDGETS["hypothesis"]`) apply on top. The report states which
limit stopped the run. `HYPOTHESIS_ENGINE_ENABLED=false` sends "why" questions back to the investigator.

## Persistence, recovery, cancellation

Tables `investigation_hypotheses` and `investigation_analyses` (Alembic `a7b9c1d3e5f4`) are written as the
run proceeds, so `GET /api/investigations/{id}` shows live state. A retried or recovered job reloads the
understanding, hypotheses and finished analyses and continues; nothing already done is repeated.
Cancellation is checked before every model call and every analysis; it takes effect within about two
seconds plus any model call already in flight.

## What reaches the model

The question, column names and types, sample values of low-cardinality columns only, and aggregate
results of executed analyses. No rows. Values of columns with more than 50 distinct values are withheld.

## API and UI

- `POST /api/investigations/hypothesis` `{dataset_id, question}` → `201 {question_id, job_id, investigation_id, limits}`.
  `400` no date column, `503` no model provider, `404` dataset not in the team. Honours `Idempotency-Key`.
- `GET /api/investigations/{id}` adds `engine`, `hypotheses`, `analyses`.
- Asking a "why" question on the Analyses page routes here automatically when a provider is configured.
- Investigations page: "Ask why" box, live hypotheses while running, and the report.

## What was verified with a real model

On 2026-10-12 the engine ran end to end through the API and the job runner against Gemini
(`gemini-3.6-flash`) on the synthetic sales fixture, three times to completion (4 model calls, 9 to 12
analyses, about 45 seconds and under one US cent each); other attempts met 503 or 429 responses and took
the fallback path. Those runs found and fixed five defects the scripted tests could not see: a tool schema
Gemini rejects, inclusive period ends dropping a day, a model outage reported as "no testable hypothesis",
a summary stating a fall as a positive number being marked unverified, and the model never hypothesising
the dimension where the change actually was (the reason the scan exists). Each has a test now.

## Not verified

- Only one dataset and one question were tried with a real model. Hypothesis quality on other schemas,
  larger schemas and vaguer questions is unknown.
- The Llama provider was never reached (the configured key is rejected), so schemas and prompts are
  validated against it only by construction.
- The last two changes (claimed-size check, "one claim per hypothesis" prompt line) were made after the
  Gemini key hit its rate limit and are covered by scripted tests only.
- A multi-round adaptive loop on a real model: in the real runs the model stopped after one round.
- The UI was exercised headlessly, not looked at in a browser.
