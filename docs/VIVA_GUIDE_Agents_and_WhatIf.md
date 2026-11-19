# Silt viva guide: the Planner–Executor–Critic loop and the What-if simulator

This guide explains two features of Silt end to end: what they are, how they work under the hood, why they matter, and
how they help a customer. Everything here is taken from the actual code in this repository; file names are given so you
can point at them. Read Part 0 first, then each part, then the Q&A at the end.

---

## Part 0 — The 60-second version (memorise this)

**Silt** is an autonomous AI data analyst. A user uploads a CSV and asks a question in plain English. Silt answers with a
dashboard in which **every number is traceable and checked**.

**Feature 1 — the Planner / Executor / Critic loop.** Three AI roles work in sequence:

- the **Planner** turns the question into a short list of concrete analysis steps;
- the **Executor** writes Python code for each step and runs it in an isolated sandbox, fixing its own errors;
- the **Critic**, deliberately a *different* AI model, writes its *own* independent code to recompute the key numbers from
  the raw data and then says **verified** or **rejected**. A rejection sends the run back to the Planner with feedback.

On top of that, plain deterministic code (no AI) re-checks every number on the dashboard. The result is stamped
`VERIFIED`, `VERIFIED_WITH_CAVEATS` or `UNVERIFIED`. The idea in one line: **the AI proposes, ordinary code and a second
AI verify.**

**Feature 2 — the What-if simulator.** A calculator for "if I change price / volume / cost, what happens to profit?". It is
**deterministic arithmetic, not AI and not a forecast**. It has two built-in models (unit economics and an acquisition
funnel), and on top of them: break-even, sensitivity, Shapley attribution (how much each change accounts for), Monte Carlo
risk simulation, goal seek ("what price gives a 25% margin?"), and a plain-English entry box. Every result is recomputed by
a second, separately written implementation to check it.

**The common philosophy:** *a language model may help you **ask** or **explain**, but it never gets to **invent a number***.
Numbers come from code that can be inspected and re-run, and are independently checked.

---

# PART A — The Planner / Executor / Critic loop

## A1. What problem does it solve?

If you simply ask a chatbot "which region grew fastest?", it can answer confidently and be wrong. Large language models
**hallucinate**: they produce plausible numbers that were never calculated. For a business decision that is dangerous.

Silt's answer is to **never let the model do the arithmetic itself**. The model writes *code*; the code runs on the real
file; the result comes from the data. Then a second model, with a different "way of thinking", tries to *catch* mistakes,
and finally ordinary code checks that every figure shown actually appears in a computed result.

Analogy for the viva: a company with a **planner** (decides what to do), an **analyst** (does the work with a calculator),
and an **auditor** (independent, redoes the key sums and signs off or sends it back). Silt automates all three, and
keeps the paperwork.

## A2. The pipeline at a glance

```
 user question
      │
   ROUTER  (rules, no AI) ── picks the cheapest path that can answer properly
      │
      ├─ clarify ......... "a term has two meanings / two columns match" -> ask the user, stop
      ├─ fast ............ simple total/average -> deterministic engine, NO model, NO code
      ├─ root_cause ...... "why did X change?" -> deterministic investigator, no model
      ├─ comparison ...... "is A different from B?" -> statistics engine
      ├─ hypothesis ...... model proposes explanations, code tests them
      └─ standard / statistical  ->  the FULL AGENT PIPELINE below
                │
             TRIAGE   (AI gate) rejects gibberish / off-topic before any expensive work
                │
             PLANNER  (AI: Gemini) -> ordered list of analysis steps
                │
             EXECUTOR (AI: Gemini) -> for each step: write code -> run in sandbox
                │            ^__ on error: feed the error back, retry (max 3 tries per step)
                │
             CRITIC   (AI: a DIFFERENT model, Llama) -> writes its own check code, runs it,
                │            compares to the Executor's numbers -> "verified" or "rejected"
                │
                ├─ rejected and revision budget left (1) -> back to PLANNER with the Critic's feedback
                │
             COMPILE  (AI: Gemini) -> KPIs, charts, narrative, using only computed numbers
                │
             VERIFY   (plain code, no AI) -> checks every KPI / chart / sentence against computed results
                │            a required check failed and budget left -> back to PLANNER once
                │
             PUBLISH  -> dashboard + audit trail, verdict: VERIFIED / VERIFIED_WITH_CAVEATS / UNVERIFIED
```

Source: the graph is defined in `backend/app/agents/graph.py` using **LangGraph** (a library for building AI workflows as a
state machine: nodes are steps, edges are the allowed transitions).

**Key point to say in the viva:** every loop is **bounded**. The Executor retries at most 3 times per step
(`MAX_EXECUTOR_RETRIES=3`), and the Planner can be sent back at most once (`MAX_CRITIC_REVISIONS=1`). When the budget is
spent the run **still finishes**, honestly marked `UNVERIFIED`. A failed check **never becomes a pass by running out of
retries**.

## A3. Each role in detail

### The Router (`agents/router.py`) — "choose the cheapest path that is still correct"
- **Rule-based on purpose**: regular expressions on the wording ("why", "trend", "compare"...). It is free, deterministic,
  unit-testable, and every decision is written to the run trace with the rule that fired.
- Why it matters: model calls and sandbox runs cost money and time. A simple "total revenue by region" needs **zero**
  model calls (route `fast`, budget: 0 LLM calls, 0 sandbox runs). Only genuinely open questions use the full pipeline.
- Each route has its **own budget ceiling**, so a question judged simple cannot quietly become expensive.

### Triage (`agents/triage.py`) — the gate
- One cheap AI call that rejects gibberish ("asdfgh"), vague input ("hello"), off-topic requests ("write a poem"), and
  questions about things not in the columns. It gives 2–3 suggested questions the dataset *could* answer.
- Why: so a nonsense question never becomes "a plausible plan and a meaningless verified dashboard".
- Fails safe: if triage itself errors (other than quota/abort), the question is allowed through rather than blocked.

### The Planner (`agents/planner.py`) — "what steps will answer this?"
- **Input:** the question + the dataset *profile* (column names, types, missing values, basic stats — not the raw rows) +
  approved business definitions from the **semantic layer** + the team's saved knowledge and *lessons* from past Critic
  rejections + similar past analyses (retrieved by embedding similarity) — except for scheduled re-runs, which plan fresh
  so they don't anchor on old results.
- **Output (forced through a structured "tool" schema):** `reasoning` plus 2–8 `steps`, each with an id, a concrete
  description and a goal. The prompt asks for 3–6 steps, at least one producing a chart, and a last step tying findings back
  to the question. Steps may only reference columns that really exist.
- If it is a **revision**, it also receives the Critic's (or the verifier's) feedback: "a previous attempt was rejected
  for these issues; produce a revised plan that addresses them".
- Saved in the database as a `Plan` row with a revision number.

### The Executor (`agents/executor.py`) — "do one step, in code, safely"
- Runs **once per step**. It is given the question, the profile, the results of earlier steps, and the current step.
- It must return `reasoning`, a **`formula_explanation`** (the arithmetic in plain English, e.g. "growth % = (this month −
  last month) ÷ last month × 100"), and a complete self-contained Python script.
- The script must follow the **sandbox contract** (`agents/prompts.py`): load `/workspace/data/input.csv`; print exactly
  one `RESULT_JSON:` line with the numbers; optionally save Plotly charts as JSON; and print one `DATA_SLICE_JSON:` line with
  up to 50 *real* rows the number came from (so a reviewer can trace a number back to the data). No network, no files outside
  `/workspace`.
- The code is run by the **sandbox runner** (see A5). Everything — code, stdout, stderr, result, charts, reasoning — is
  stored in an `ExecutionLog` row.
- **Self-correction loop:** if the code fails, the error text is fed back into the next attempt ("fix the root cause, don't
  just suppress the exception"). Up to 3 attempts per step. If all fail, the whole run fails with a clear reason.

### The Critic (`agents/critic.py`) — "an independent auditor, not a rubber stamp"
Three design decisions to quote:
1. **Different model family.** The Planner/Executor/compiler use **Gemini**; the Critic prefers **Llama** (now Groq's
   `openai/gpt-oss-20b` by default). A different model catches different mistakes than the same model re-checking its own
   work. If Llama is unavailable it falls back to Gemini, and that fallback is **recorded** (`verifier_model` =
   `gemini-fallback:...`) and a note is appended to the summary saying the check was independent in context and code but
   *not cross-model*. It is never silently presented as cross-model verification.
2. **Its own code, not the Executor's story.** The Critic does **not** reuse the Executor's conversation. It is told "do NOT
   trust the Executor's numbers". It writes **its own Python script** that recomputes the key numbers directly from the raw
   data, "ideally with a slightly different method" (e.g. a manual groupby instead of a helper column) so a real mistake would
   show up. That script is run in its own sandbox run.
3. **Then it judges.** A second Critic call gets the Executor's claims *and* its own recomputed output and returns a
   structured verdict: `verified` or `rejected`, a `confidence` between 0 and 1, specific `issues`, and a 2–4 sentence
   summary. It checks: do independently recomputed numbers match? does the analysis actually answer the question? is the
   chart type appropriate? are there unsupported **causal** claims?
- A rejection is stored as a `CriticReview` and also recorded as a **lesson** (`knowledge.record_critic_lesson`) so the
  Planner can learn from it next time.

### Compile and Verify (`agents/dashboard.py`, `verification.py`) — the deterministic safety net
- **Compile:** an AI call turns the verified step results into 2–5 KPIs, up to 4 charts and a short narrative. Its prompt says
  every value *must* come from a step result and the narrative must use only numbers present in the results.
- **Verify (no AI):** the code builds an **evidence record** for every KPI, chart and narrative with explicit checks:

| Check | Applies to | Passes when |
|---|---|---|
| `result_present` | KPI | the cited step ran and produced a parseable result |
| `value_in_result` | KPI | the displayed value appears in that step's result (to display precision) |
| `independent_recomputation` | KPI | a second computation produced the same value |
| `provenance` | KPI, chart | an execution record exists and the dataset file still matches its recorded hash |
| `invariants` | KPI | finite; shares within 0–100%; counts non-negative |
| `narrative_numbers_supported` | narrative | every number in the text appears in a computed result |
| `independent_review` | narrative | the Critic accepted the analysis |
| `causal_language` | narrative | informational: flags causal wording |
| `artifact_validated` | chart | file came from a successful step and parses as a Plotly figure |

  Outcomes: `pass`, `fail`, `not_run`, `warn`, `not_applicable`. **A check that did not run never counts as passed.**

## A4. The verdict and how it is decided

Dashboard states (`verdict.py`, `verification.py:combine_verdict`):

- **VERIFIED** — the Critic approved on the first pass and every required check passed.
- **VERIFIED_WITH_CAVEATS** — the Critic rejected once, the Planner re-planned and the second pass was approved; *or* some
  claim is merely unconfirmed or has a warning. The result stands, but the reader is told something was wrong once.
- **UNVERIFIED** — the Critic still rejected after the revision budget, **or any required check failed**.

Two rules to remember: **a failed required check makes the dashboard UNVERIFIED whatever the Critic said**, and **passing
checks never upgrade a Critic rejection**. Honesty beats a green tick.

Three kinds of validity are reported separately: *mathematical* (the checks above), *statistical* (sample adequacy, flagged if
under 30 rows) and *causal* (never established; causal wording is flagged).

## A5. Safety: why generated code is safe(ish) to run

AI-written code is treated as **hostile**. `backend/app/sandbox/runner.py` runs each execution in a fresh **Docker** container:
no network (`--network none`), read-only root filesystem, all Linux capabilities dropped, non-root user, memory/CPU/process
limits, a wall-clock timeout (default 60 s), output size caps, and the container is removed afterwards. Charts are validated
(regular files only, size caps, must parse as Plotly). Each run has its own directory, deleted afterwards. The container's
environment is built from scratch, so **no API key is ever visible to generated code**.

**Fail closed:** if Docker isn't available, the run is refused (`sandbox_unavailable`); nothing silently falls back to a weaker
runner.

**Prompt injection defence:** everything from an uploaded file is *untrusted data*. Column names and sample values are clipped
and wrapped in a tag marking them untrusted, and every system prompt tells the agents never to obey instructions found in data
("ignore previous instructions...").

**Honest limitation (say this if asked about deployment):** Render's free tier cannot run Docker. A development-only
*subprocess* runner exists, which scrubs the environment and applies CPU/file/process limits but is **not** an isolation
boundary (it shares the host's kernel, filesystem and network). It is refused unless `SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true`.
The proper production setup is a host with Docker and `SANDBOX_BACKEND=docker`.

## A6. The machinery around the loop (reliability)

- **Durable jobs** (`jobs.py`): an analysis is a database row, not a thread. Workers claim jobs with a lease; if a server
  restarts mid-run, the job is re-queued or marked abandoned; per-team limits (5 active jobs) and idempotency keys prevent
  double-runs; there is a deadline (default 900 s) and cancellation.
- **Provider failover** (`agents/llm_client.py`): rate limits (429) and transient server errors (503) are retried with waits;
  if Gemini keeps failing, the step is handed to Llama. A malformed reply (empty, prose, bad function call) now gets **one**
  retry before failing over. Every failure is classified and logged **without** logging prompts, keys or generated text.
- **Budgets** (`runtime.py`): per run — 40 model calls, 30 sandbox runs, optional cost ceiling; exceeding one ends the run as
  `budget_exceeded`. A run **trace** lists the route, every node, every model call (provider, tokens, estimated cost) and
  every code run — that is the "Run trace" panel you saw.
- **Audit trail + reproducibility** (`provenance.py`): each KPI/chart/sentence links to the exact execution log (code, stdout,
  result) and Critic review behind it. Every dataset version has a SHA-256 hash; every run has a manifest; a rerun can be pinned
  to the original data version. (Model output isn't deterministic, so a rerun isn't promised to match; the comparison says so.)

## A7. Worked example

Question: **"Which quarter has the highest revenue?"** on `sales.csv`.

1. **Router:** wording has "which ... highest" — not "why", not statistical. If the parser fully understands it, route `fast`:
   a deterministic aggregation, **no AI at all**. For a trend-style question it goes to the full pipeline.
2. **Triage:** accepted — real question, matching columns.
3. **Planner:** e.g. (1) load, group by quarter, sum revenue; (2) find the maximum quarter and compare to the others;
   (3) bar chart of revenue by quarter; (4) summarise.
4. **Executor:** step 1 → writes pandas code → sandbox runs it → prints `RESULT_JSON` with quarterly totals and a data slice.
   If a column name was wrong, the error text goes back and it retries.
5. **Critic (Llama):** writes a different script (e.g. a manual groupby on the raw file), recomputes the totals, compares.
   Matches → `verified`.
6. **Compile + Verify:** KPIs such as "Top quarter: Q3" and "Revenue 2.7M"; the verifier confirms each number appears in a
   result and was recomputed. Stamp: **VERIFIED**.
7. The user sees the dashboard, can click any number and see the code, data slice and the Critic's check.

## A8. Why it is important and how it benefits a customer

- **Trust:** every number can be traced to code and data, and independently re-checked. Customers can *defend* the figures.
- **No analyst bottleneck:** plain-English questions; no SQL or Python.
- **Honesty over confidence:** a wrong or unsupported answer is **flagged UNVERIFIED** rather than presented as truth. That is
  the opposite of a chatbot that always sounds sure.
- **Auditability and compliance:** full audit trail, reproducibility manifests, role-based team isolation.
- **Cost and speed control:** cheap routes for simple questions, budgets, and bounded loops, so cost is predictable.
- **Safety:** generated code runs isolated; data in uploads cannot hijack the agents.
- **Learning:** Critic rejections become lessons for future plans; the semantic layer pins one approved definition per metric.

## A9. Honest limitations (examiners like candour)

- The Critic is **another AI**. It is independent in model and method, which greatly reduces correlated errors, but it is
  not a proof. That is why the deterministic verification layer exists on top.
- **Correlation is not causation.** The system does not establish cause; causal wording is flagged.
- It checks that numbers are *consistent with the data and code*, not that the *question was the right one*.
- Model output is non-deterministic; reruns can differ. Providers can be overloaded (we saw 503s) — hence retries and failover.
- Full isolation needs Docker, which some hosts (e.g. Render free tier) can't provide.
- Only one revision pass by default (`MAX_CRITIC_REVISIONS=1`) to bound cost.

---

# PART B — The What-if simulator

## B1. What is it, in one paragraph?

A calculator for decisions. The user states today's numbers (a **baseline**), describes a change (a **scenario**), and Silt
shows the resulting numbers side by side, with the arithmetic, a check, and a plain-English explanation. It answers
"what happens to profit if I raise the price 10%?", "how many units until I break even?", "what price gives me a 25%
margin?", "how likely is a loss if my inputs are uncertain?".

**Crucial framing:** a scenario is **arithmetic on stated inputs**. It is **not a forecast** and **not AI**. Every response is
labelled `kind: "scenario"` with a disclaimer: *"arithmetic on the stated inputs and assumptions. It is not an observed fact and
not a validated forecast."*

## B2. The two built-in models (`backend/app/scenarios.py`)

**Unit economics** (price, volume, unit cost, fixed cost):

```
revenue        = price × volume
variable_cost  = unit_cost × volume
gross_profit   = revenue − variable_cost
profit         = gross_profit − fixed_cost
margin %       = profit ÷ revenue × 100
```

**Acquisition funnel** (visitors, conversion rate, average order value, cost per visitor):

```
orders  = visitors × conversion_rate
revenue = orders × average_order_value
profit  = revenue − visitors × cost_per_visitor
```

**The only behavioural link — price elasticity (an explicit assumption the user types):**

```
volume after a price change = volume × (1 + price_elasticity × price change %)
```
Default elasticity is 0, meaning "volume does not react to price". Example: elasticity −1.5, price +10% → volume
1000 × (1 + (−1.5)(0.10)) = **850**.

**Validation, not clamping:** no negative prices, rates must be within 0–1, no change below −100%. Impossible inputs are
**rejected with the reason**, never silently corrected.

## B3. Worked example (use this in the viva)

Baseline: price 100, volume 1,000, unit cost 60, fixed cost 20,000.

- Profit now = (100 − 60) × 1,000 − 20,000 = **20,000**.
- Scenario "price +10%": price 110 → (110 − 60) × 1,000 − 20,000 = **30,000** (elasticity 0).
- **Break-even quantity** = fixed cost ÷ contribution per unit = 20,000 ÷ (100 − 60) = **500 units**; safety margin =
  (1,000 − 500) ÷ 1,000 = **50%**.
- **Goal seek**: "price for a profit of 30,000?" → (p − 60) × 1,000 − 20,000 = 30,000 → **p = 110** ✓.
- **Attribution** (Shapley) for "price +10% and volume +20%": new price 110, volume 1,200 → profit 40,000, a change of 20,000.
  Order price-first: price adds 10,000, then volume 10,000. Order volume-first: volume adds 8,000, then price 12,000. Averaging
  over both orders: **price accounts for 11,000, volume for 9,000; they sum exactly to 20,000**.

## B4. The workflow, step by step

```
 What-If page (frontend/pages/14_What_If.py)  or  Silt Copilot
              │
   (typed form)  or  (a sentence -> scenario_nl.py parses wording, rules first)
              │
        scenario_service.py   <- the ONE place calculations run, verified and explained
              │
   deterministic engines:  scenarios.py (evaluate, break-even, Shapley)
                           scenario_sim.py (Monte Carlo)   scenario_solve.py (goal seek)
                           scenario_sensitivity.py (grids)
              │
   scenario_verify.py  <- a SECOND, separately written implementation recomputes and compares
              │
   scenario_explain.py <- template-based plain-English explanation (no AI needed)
              │
        response: results + formulas + verification + explanation + assumptions + disclaimer
```

Steps for a user (the "Build a scenario" tab): (1) pick a model, (2) enter or derive the baseline from a dataset's columns,
(3) set assumptions (elasticity), (4) enter percentage changes, (5) read the comparison table, chart, break-even, sensitivity
and explanation, (6) optionally save it.

**Both the What-if page and the Silt Copilot call the same service**, so they give identical numbers and explanations. A test
fails if either calls an engine directly. No step asks a language model for a number.

## B5. The engines and what each is for

- **Scenario comparison:** baseline vs your scenario vs auto-generated *optimistic / pessimistic* (every input moved by a spread
  % in its favourable / unfavourable direction).
- **Sensitivity:** moves each input ±% to show which one the profit is most sensitive to.
- **Shapley attribution:** "how much does each changed input account for?". Exact for up to 10 changed inputs (subset
  formula, 2ⁿ evaluations), sampled beyond that. Contributions **sum exactly** to the change. It says "accounts for", never
  "caused".
- **Break-even:** units needed, break-even revenue, safety margin, and a profit curve. If a unit contributes nothing or less,
  it says there is **no** break-even rather than inventing a number.
- **Monte Carlo risk simulation** (`scenario_sim.py`): the user states, per input, a distribution (normal, uniform or
  triangular) and spread. Silt draws thousands of outcomes (100–200,000; default 10,000) using a **Gaussian copula** (so
  inputs can be correlated), truncated to physical limits (no negative prices), and reports mean, median, 5th/25th/75th/95th
  percentiles, probability of being positive/negative/above baseline/reaching a target (with standard errors) and a histogram.
  A fixed **seed** reproduces a run exactly. "Low/high" are the 5th and 95th percentiles — **not** guaranteed bounds. Nothing is
  assumed: with no uncertain input the request is refused.
- **Goal seek** (`scenario_solve.py`): "what value of one input hits this target?" Done by numerical root-finding on the same
  engine: scan a grid for sign changes, refine each with **Brent's method**, re-evaluate with the original engine, check
  constraints. Statuses include `solved`, `multiple_solutions`, `no_solution` (with the range the output *can* reach),
  `no_feasible_solution`. It never fabricates a solution.
- **Natural-language entry** (`scenario_nl.py`): a sentence becomes a *validated structured request*. **Rules run first**
  (understand verbs, %, ₹, lakh, crore, "k", goals, break-even, comparisons). Every number must come from the user's message
  (checked). Ambiguity returns a clarifying question. An optional language model may translate a sentence the rules couldn't
  read, but its output passes the **same** validator **plus** a check that it did not invent a number. The response shows
  "how I understood it" before the answer.
- **Historical elasticity estimation** (`scenario_elasticity.py`): a *suggestion*, never an override. Fits
  ln(quantity) = a + e·ln(price) + controls with robust errors; refuses when data are inadequate (fewer than 8 observations,
  fewer than 3 distinct prices, price spread under 2%).
- **Custom models** (`scenario_formula.py`, `scenario_models.py`): users define their own inputs and formulas. Formulas are
  parsed with Python's `ast` and a **whitelist** (numbers, names, + − × ÷, constant powers, a few functions) — never executed
  as Python. Cycles are detected, units are checked, definitions are **versioned and immutable**, inputs can bind to approved
  semantic-layer metrics.
- **Time-aware baselines, segments, two-way sensitivity grids, period comparison, rebase on new data** — all built on the same
  engines.

## B6. How results are checked and kept (the trust story)

- **Independent verification** (`scenario_verify.py`): a second implementation in **plain Python only** (its own copy of the
  formulas, subset-formula Shapley, hand-written percentiles, independent bisection scan) recomputes everything and compares.
  Two programs written separately that agree are evidence that neither carries a bug the other lacks. Status is `unverified`
  if any required check fails *or did not run*. It states its scope: **the calculations were checked, the realism of your inputs
  was not.**
- **Explanations** (`scenario_explain.py`): a fixed template, grounded — every number in the text must appear in the result;
  no causal wording; separate sections for assumptions, risks and what cannot be concluded.
- **Evidence sealing:** a saved scenario stores a **sealed record** (request, result, verification, data version, seed, engine
  versions) with a **SHA-256 hash**. "Reproduce" re-runs it and reports if anything was altered or the engine version changed.
  Records are append-only and team-scoped (another team's id looks like "not found").
- **Saved scenarios** can be reopened, edited (new revision, evidence kept), duplicated, compared, exported (JSON/CSV/Markdown
  with a content hash), and **rebased** onto a newer dataset version (creating a *new* scenario, never overwriting).

## B7. Why it is important and how it benefits a customer

- **Decisions, not just reports:** analytics tells you what happened; What-if lets you test what might happen before you commit
  (price rise, cost cut, marketing spend).
- **Transparent and auditable:** the formulas are shown, the inputs' source is recorded, results are independently checked,
  and a saved scenario can be reproduced later.
- **Honest about uncertainty:** Monte Carlo shows a *range and probabilities*, not a false single number; goal seek says when
  no solution exists.
- **Safe from AI hallucination:** no model supplies a number, so nothing can be "made up" (the optional sentence helper is
  boxed in by validation).
- **Accessible:** plain-English questions; works from the customer's own data or typed numbers; one engine shared by the page
  and the Copilot.
- **Collaborative:** team-scoped, versioned, shareable by export.

## B8. Honest limitations

- It is **arithmetic on assumptions**; garbage in, garbage out. The verifier checks the math, not whether your price elasticity
  or distributions are realistic.
- Two built-in models; anything else needs a custom model.
- Elasticity is the only behavioural link and, by default, is a user-typed number.
- Monte Carlo distributions are assumptions, not measurements.
- Goal seek can miss roots closer together than a grid step, or roots that only touch the target.
- Bounds exist (grid ≤ 61×61 cells, iterations 100–200,000, formula size limits) for performance and safety.

---

# PART C — How the two features fit together

| | Planner/Executor/Critic (Analyses) | What-if |
|---|---|---|
| Question type | "What happened and why?" (past data) | "What would happen if?" (forward arithmetic) |
| Uses AI? | Yes: Gemini (plan, code, summary) and a different model (Critic) | No for numbers; optional AI only to translate a sentence, boxed in by validation |
| Where numbers come from | Code the AI wrote, run on your data in a sandbox | Fixed formulas in the engine |
| How it is checked | Critic recomputes independently + deterministic evidence checks | A second plain-Python implementation recomputes |
| Failure mode surfaced as | UNVERIFIED / failed run | `unverified` status, refusals, "no solution" |
| Shared principle | **A model may help you ask or explain, but never invents a number; numbers are traceable and independently checked.** | |

---

# PART D — Likely viva questions and model answers

**1. What is the Planner–Executor–Critic loop?**
Three AI roles in sequence: the Planner breaks a question into analysis steps, the Executor writes and runs code for each step in a sandbox (retrying on error), and the Critic independently recomputes the key numbers with its own code and approves or rejects. A rejection sends the run back to the Planner with feedback. Plain code then verifies every dashboard number.

**2. Why not just ask one LLM for the answer?**
Because LLMs hallucinate numbers. Here the model writes code and the numbers come from running it on the real data; a second model and deterministic checks then verify. The answer is traceable instead of merely plausible.

**3. Why is the Critic a different model?**
A different model family makes different mistakes, so it is more likely to catch the Executor's errors than the same model re-checking itself. If the Critic model is unavailable it falls back to Gemini, and that is clearly recorded and disclosed rather than hidden.

**4. How does the Critic avoid just agreeing with the Executor?**
It does not reuse the Executor's conversation, it is told not to trust its numbers, and it writes its *own* script to recompute from the raw file, ideally by a different method. It then compares recomputed numbers to the claimed ones.

**5. What happens if the Critic rejects?**
The rejection feedback goes back to the Planner for one revision (`MAX_CRITIC_REVISIONS=1`). If it still fails, the run ends and is published as UNVERIFIED with caveats. It never loops forever and never passes by exhausting retries.

**6. What if the generated code crashes?**
The error text is fed back to the Executor for a corrected script, up to 3 attempts per step. If all fail, the run fails with the last error.

**7. Is AI-generated code safe to run?**
It is treated as hostile. It runs in a fresh Docker container with no network, read-only filesystem, dropped capabilities, non-root user, resource limits and a timeout, with no API keys in its environment. If isolation is unavailable the run is refused (fail closed). Uploaded data is marked untrusted in every prompt to resist prompt injection.

**8. What are the verdict states?**
VERIFIED (Critic approved first time, all checks passed), VERIFIED_WITH_CAVEATS (needed a re-plan or has unconfirmed claims), UNVERIFIED (Critic still rejected, or a required check failed). A failed required check always makes it UNVERIFIED.

**9. What does the deterministic verifier check?**
Per KPI: the cited step ran, the displayed value appears in the result, an independent recomputation agrees, the dataset hash still matches, and basic invariants hold. Per narrative: every number in the text appears in a computed result and the Critic approved. A check that couldn't run never counts as passed.

**10. Why a router?**
To use the cheapest correct path: simple aggregations and "why did it change" questions run deterministically with zero model calls. It's rule-based so it is free, testable and every decision is logged. Each route has its own budget.

**11. How do you control cost and runaway runs?**
Bounded loops, per-run budgets (40 model calls, 30 sandbox runs, optional dollar cap), a deadline (900 s), cancellation, and a trace of every call with tokens and estimated cost.

**12. What happens if a provider is down or overloaded?**
Rate limits and 5xx errors are retried with waits; a malformed reply gets one retry; then the step fails over to the second provider. Failures are classified and logged without leaking prompts or keys. We saw Gemini 503s in testing and the trace showed exactly that.

**13. How is a result auditable?**
Every KPI, chart and sentence links to the execution log (code, output, data slice) and Critic review behind it. Datasets are hashed per version; every run has a manifest; a rerun can be pinned to the original data.

**14. What are the limits of the Critic?**
It is another AI: independent but not a proof, and it cannot establish causation. That's why deterministic checks sit on top, and why causal wording is flagged.

**15. What is the What-if simulator?**
A deterministic scenario calculator: you give baseline numbers and a change, it returns the outcome with formulas, break-even, sensitivity, attribution, risk simulation and goal seek, each independently verified. It is explicitly not a forecast.

**16. Which models does it use and why those?**
Unit economics (price, volume, unit cost, fixed cost) and an acquisition funnel (visitors, conversion, order value, cost per visitor): the simplest widely understood profit models, fully specified, with custom models for anything else.

**17. How does price elasticity work?**
volume = volume × (1 + elasticity × price change). It is the only behavioural link, an explicit user-stated assumption (default 0). The app can *suggest* an estimate from history but never overrides what the user typed.

**18. What is Shapley attribution and why use it?**
It splits a change in profit fairly among the inputs that changed by averaging each input's marginal contribution over all orderings; contributions sum exactly to the total. It avoids the arbitrary result of attributing in one fixed order.

**19. How does Monte Carlo work here?**
You state distributions for uncertain inputs; Silt draws thousands of correlated samples (Gaussian copula), truncated to physical limits, computes the output for each and reports percentiles and probabilities. It is seeded, so reproducible. The distributions are your assumptions, and the result says so.

**20. How does goal seek work?**
It scans a grid for sign changes of (output − target), refines each root with Brent's method, re-evaluates with the original engine to reject false roots, and checks constraints. If none exists it says so and shows the reachable range.

**21. Does the AI ever produce a number in What-if?**
No. The parser reads wording only; every number must come from the user's text. An optional model may translate a sentence the rules can't read, but its output passes the same validator and may not use a number the user didn't write. Explanations are templates, not model text.

**22. How do you know the What-if numbers are right?**
A second implementation written separately in plain Python recomputes every result and compares. If a check fails or can't run, status is `unverified`. It states what it did not check: whether your inputs are realistic.

**23. How are scenarios stored and protected?**
Saved with a sealed evidence record and SHA-256 hash; "reproduce" re-runs and detects tampering or an engine change. Records are team-scoped (another team's id looks missing) and append-only; editing creates a new revision.

**24. Why is this valuable to a customer?**
Analysis tells you what happened; What-if lets you test decisions before committing, with transparent, checked numbers and honest uncertainty. The agent loop gives trustworthy answers without an analyst. Both share one rule: no unverifiable numbers.

**25. What would you improve next?**
Isolated execution on hosts without Docker, more built-in models, learning from Critic lessons more aggressively, and measuring Critic accuracy against known-wrong answers (the evaluation lab already runs offline regression suites).

---

## Appendix — Where to find things in the code

| Topic | File |
|---|---|
| Agent graph (loop wiring) | `backend/app/agents/graph.py` |
| Router / Triage | `backend/app/agents/router.py`, `triage.py` |
| Planner / Executor / Critic | `backend/app/agents/planner.py`, `executor.py`, `critic.py` |
| Prompts and the sandbox contract | `backend/app/agents/prompts.py` |
| Compile / verify / publish | `backend/app/agents/dashboard.py`, `backend/app/verification.py`, `verdict.py` |
| Sandbox | `backend/app/sandbox/runner.py` |
| Providers, retries, failover | `backend/app/agents/llm_client.py`, `llm_client_llama.py`, `config.py` |
| Durable jobs, budgets | `backend/app/jobs.py`, `runtime.py` |
| What-if models, break-even, Shapley | `backend/app/scenarios.py` |
| Monte Carlo / goal seek | `backend/app/scenario_sim.py`, `scenario_solve.py` |
| Verification / explanation / NL / service | `scenario_verify.py`, `scenario_explain.py`, `scenario_nl.py`, `scenario_service.py` |
| Custom models / elasticity | `scenario_formula.py`, `scenario_models.py`, `scenario_elasticity.py` |
| What-if page | `frontend/pages/14_What_If.py` |
| Reference docs | `docs/PLATFORM.md` (sections 1, 4, 9, 10) |
