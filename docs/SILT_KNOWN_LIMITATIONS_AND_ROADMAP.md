# Silt — Known Limitations, Bug Audit and Roadmap

This covers Parts 16 and 18 of the audit. Every item is backed by evidence: a run during the audit [RUN], the Phase 5.1 evaluation [EVAL], the source code [CODE], or a training report [REPORT]. Items without direct evidence are labelled **potential risk**. **Bugs** (the code does something wrong) are kept separate from **architectural limitations** (the design doesn't cover something yet).

Severity scale:
- **Critical**: a confident wrong number.
- **High**: a wrong or misleading answer in common cases, or loss of work.
- **Medium**: the request isn't answered, or the answer is degraded.
- **Low**: inconvenience.

---

## Part 16 — Prioritised issue table

| # | Issue | Current behaviour (evidence) | Root cause | User impact | Type | Severity | Suggested fix | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **Relative-period filters silently ignored** | "Total amount for only the last 30 days" → 4,123 over all 40 rows; the true value is 2,938 [RUN] | `scope._time` (`scope.py` l. 83) understands only month names and years; nothing checks that every restriction in the question was applied | A confident wrong total | Bug | **Critical** | Parse relative periods ("last N days/weeks/months", "this/last quarter", "year to date") in `scope.establish`; add a final check that a recognised period was applied, otherwise ask | 1–2 days |
| 2 | **Yes/no adjective filters silently ignored** | "Average resolution hours of only escalated tickets" → 7.92 over all 360 rows; the true value is 12.66 [RUN] | `scope.establish` recognises a yes/no column only after "for/among/of/with" (`scope.py` l. 113); `intent.read` detects the restriction but is only applied to *programs*, not to the deterministic answers | A confident wrong average | Bug | **Critical** | Apply `intent.read(...).restrict` as a post-check in `orchestrator._finish` for every path; extend `scope.establish` to "only/just <flag>" | 1 day |
| 3 | **Wrong metric for an entity's share** | "What fraction of Mei Lin's trips were late?" → Mei Lin's share of *all* trips (9.5%), not her late rate [EVAL `n-fleet-entity-rate`] | the metric catalogue's share logic took the named entity as the event | A confident wrong percentage | Bug | **High** | Run `intent.check` on metric-path specs; when the request names an entity *and* a flag, the entity is the filter and the flag is the event | 1 day |
| 4 | **Inconsistent duplicate policy** | "Total shift hours for only Rahul" = 26 h (duplicate included, flagged) but the feasibility answer counts 18 h (duplicate excluded) [RUN] | the calculator/metric paths keep exact duplicates "as recorded"; the objectives' `select valid_for` excludes them | Two answers about the same hours disagree | Bug | **High** | One policy in `scope.eligibility`, applied everywhere, and stated in every answer (or ask the user once per dataset) | 0.5–1 day |
| 5 | **`hourly_rate` read as a rate (share)** | role "rate" with evidence "named like a rate" [RUN]; the program planner then refuses to sum it and the investigation treats it as a level | `semantics.column_role` checks the rate *name* before the amount *name* (`semantics.py` l. 90 vs l. 94) | Wrong refusals; odd findings for pay-rate columns | Bug | **Medium** | Rate only when the values are in 0–1 or 0–100 *and* the name says share/pct/percent; names with money words (pay, hourly, wage, fee) → amount | 0.5 day |
| 6 | **Uncommitted work** | 20 modified and 20 untracked files, including the whole program/planner/semantics/intent stack [RUN `git status`] | work done across phases without commits | One disk accident loses weeks of work | Process | **High** | Commit to the feature branch (by the owner) and push | minutes |
| 7 | **The program planner's model was not trained for its format** | v1 planned correct metrics in 7/9 (final run) to 10/12 (earlier run) model-planned cases; the base model 12/12 when not memory-starved [EVAL] | the adapters learned the *tool-call* and *calc* JSON, not the program JSON | Composite questions often fall back | Architecture | **High** | See R1 below (training data for programs, or a stronger model) | 1–3 weeks |
| 8 | **Composite answers incomplete** | Only 5/8 composite requests showed every part (v1) [EVAL] | the deterministic paths answer the first part; the rule program needs a grounded measure; synonyms are unknown | "…and why" parts missing on some phrasings (the checklist marks them) | Limitation | **Medium** | Extend `intent.rule_program` coverage; route every composite through `intent.complete`, not only programs | 2–4 days |
| 9 | **Column meaning gaps** | "trip length"→km, "visitors"→footfall, "spend"→bill not linked; such requests fall back [EVAL] | concept groups are small and hand-written (`semantics.CONCEPTS`) | Unanswered questions | Limitation | **Medium** | Learn synonyms from column values and descriptions; optionally embeddings computed locally; always *offer*, never map silently | 3–5 days |
| 10 | **Model latency** | p50/p90 2.4/13 s (v1), 5.5/26 s (base, timeouts on 10/23 calls) [EVAL]; single requests up to ~57 s in earlier runs | a 4B model on a 16 GB laptop; two model processes cause swapping; multi-stage fallbacks | Slow answers for unusual questions | Limitation | **Medium** | Keep the 12 s turn budget; prefer deterministic programs (already done); prompt caching; a dedicated machine with more memory for the base planner | ongoing |
| 11 | **Malformed model output** | Residual failures after repairs (e.g. "Expecting ',' delimiter") [EVAL] | no constrained decoding in mlx_lm 0.32.2 | Fallbacks | Limitation | **Low** | Grammar-constrained decoding (a JSON-schema logits processor) if the server version supports it | 2–3 days |
| 12 | **Conversation memory not persistent** | `_MEMORY` is a process-local `OrderedDict` of 200 conversations [CODE `orchestrator.py` l. 19–35] | design choice for speed | A backend restart loses every conversation's context; multiple workers would not share it | Limitation | **Medium** | Persist `task`, `objectives`, `planned`, `investigation` (JSON) in a table keyed by the memory key | 1–2 days |
| 13 | **Hosted deployment has no model** | Render's Docker image runs only uvicorn and Streamlit; no `mlx` in requirements [CODE `Dockerfile`, `start.sh`] | MLX needs Apple Silicon | The hosted Copilot runs model-off; model-planned features exist only on the local Mac | Limitation | **Medium** | Serve the model on a GPU host (OpenAI-compatible) and set `ANALYST_MODEL_URL`; or demonstrate locally | 1–2 days + hosting cost |
| 14 | **PDF analysis is shallow** | Text only; no OCR (scanned pages reported); tables become plain text; BM25 keyword search (no synonyms) [CODE `documents.py`] | scope of the module | Scanned or table-heavy PDFs answer poorly | Limitation | **Medium** | OCR (e.g. Tesseract), table extraction (e.g. pdfplumber), local embeddings for retrieval | 3–5 days |
| 15 | **Forecasting is basic** | Linear trend vs naive only; ≥ 8 periods; no seasonality; no forecast interval [CODE `followup.forecast`] | deliberate minimum | Seasonal data forecast poorly; no uncertainty band | Limitation | **Medium** | Seasonal-naive and exponential smoothing baselines; backtested intervals | 3–4 days |
| 16 | **"Working days" are calendar days** | Absence of 10 "working days" subtracts 10 calendar days (27 → 17) [RUN] | no working-day calendar | Over- or under-states the absence effect | Bug (wording vs computation) | **Low** | Say "days", or add a weekday calendar (`numpy.busday_count`) | 0.5 day |
| 17 | **"More than 8 hours" needs the model** | "How many employees worked more than 8 hours?" → "couldn't turn this into an analysis" with the model off [RUN] | the deterministic paths cannot combine a derived duration with a numeric condition | Unanswered | Limitation | **Low** | A rule for "<derivable measure> <comparison> <number>" in `intent`/`rule_program` | 1 day |
| 18 | **Prompt privacy differences** | The v1 tool-planner prompt lists category values including person names; the calculator prompt includes the first 3 rows [RUN] | older prompt designs | No external exposure today (the model is local); a risk if the model is ever hosted remotely | Potential risk | **Low** | Use the program planner's grounding rules (names only when the user typed them) in all prompts | 1 day |
| 19 | **Security defaults** | CORS defaults to `*` (`main.py` l. 82); `backend/.env` contains non-empty API key entries (values not inspected); rate limiting of the analyst routes NOT VERIFIED | permissive defaults for local development | Low today (Streamlit calls the API server-side); higher if the API is exposed | Potential risk | **Medium** | Set `CORS_ALLOWED_ORIGINS` in production; keep keys out of files (secrets manager); verify rate limits on `/api/analyst/*` | 0.5 day |
| 20 | **Scaling** | every `/ask` re-reads the dataset file from disk (`routers/analyst.py` `_frame`); memory in-process [CODE] | simplicity | Slow on large files; no horizontal scaling | Potential risk | **Low** (demo) | Cache DataFrames per dataset version; external memory store | 1–2 days |
| 21 | **Excel edge cases** | No merged-cell handling; `.xls` needs `xlrd` (installation NOT VERIFIED) [CODE/INFERRED] | not implemented | Odd columns from merged headers; `.xls` may fail | Limitation | **Low** | Detect merged header rows; check `xlrd`, or convert `.xls` | 0.5–1 day |
| 22 | **Evaluation coverage** | The ten held-out public datasets have **not** been tested; the 49 evaluation cases were written by the same team that wrote the code; the browser scripts live outside the repository [EVAL] | time | Real generalisation is not yet measured independently | Process | **High** (for claims) | Run the ten held-out datasets once, frozen; add the browser scripts to the repo | 1–2 days |
| 23 | **v3 training incomplete** | Stopped at iteration 430 of 800; the final file is the 400 checkpoint [REPORT/RUN] | the run was stopped | v3's evaluation reflects a half-trained adapter | Process | **Low** (v3 inactive) | Note it in reports; retrain only with a new purpose (see R1) | — |

---

## Part 18 — How to fix the system properly (recommendations only — nothing here was changed)

### Quick fixes (days)

**Q1. Never ignore a recognised restriction (issues 1, 2, 3).**
- *Why it exists*: filters are recognised in two places (`scope.establish` for every path, `intent.read` for programs) and only the program path checks them against the answer.
- *Files*: `backend/app/analyst/scope.py` (relative periods; "only <flag>"), `orchestrator.py` `_finish` (post-check), `intent.py` (reuse `read`).
- *Change*: after any path answers, compare the request's recognised restrictions (period, entity, flag) with the records the answer used (`_records_used` and the scope). If one wasn't applied, replace the answer with a clarification or "I couldn't apply X".
- *Needs*: code only.
- *Tests*: the six filter examples in Part 8 of the main report as regression tests; truth from pandas; the assertion that no answer uses more rows than the filter allows.
- *Effort*: 1–2 days. *Risk*: over-strict checks could block valid answers; mitigate with clear clarifications.

**Q2. One duplicate policy (issue 4).**
- *Files*: `scope.eligibility`, `calc.run`/`data_checks`, `objectives.feasibility_program`.
- *Change*: exclude exact duplicates by default everywhere (or keep them everywhere), state it in every answer, and allow "include duplicates".
- *Needs*: code. *Tests*: "total hours for Rahul" equals the feasibility "completed so far" on the same records.
- *Effort*: 0.5–1 day. *Risk*: changes some existing answers' numbers, so update the tests deliberately.

**Q3. Money-named rate columns (issue 5).**
- *File*: `semantics.column_role` (`semantics.py` l. 60).
- *Change*: check amount words before rate words, and require share-like values (0–1, or 0–100 with % context) for "rate".
- *Tests*: `hourly_rate`, `pay_rate` → amount; `conversion_rate` 0–1 → rate; `discount_pct` 0–100 → rate.
- *Effort*: 0.5 day. *Risk*: low.

**Q4. Commit the work (issue 6).** Done by the project owner on the feature branch. *Risk*: none.

**Q5. Wording fixes (issue 16)**: say "days" instead of "working days", or use weekdays. 0.5 day.

### Medium-term improvements (1–2 weeks)

**M1. Persist conversation memory (issue 12).**
- *Files*: a new table, plus `orchestrator.memory_for` reading and writing JSON.
- *Store* the analytical objects only (task, objectives, planned values, findings), never DataFrames or fitted models.
- *Tests*: restart the backend between turns and check "compare with the previous scenario" still works.
- *Risk*: stale memory after the dataset changes, so key by dataset version.

**M2. Broader deterministic composite coverage (issues 8, 17).**
- *Files*: `intent.py` (`read`, `rule_program`, `complete`).
- *Change*: support derived measures with conditions, entity-plus-flag rates, two-group comparisons, and "why" for any measured outcome.
- *Tests*: the 23 new evaluation cases with the model off; the target is ≥ 30/49 model-off without lowering numerical correctness.

**M3. Better column meaning (issue 9).**
- *Files*: `semantics.py`, `intent.py`.
- *Change*: synonyms from a curated list plus column descriptions; local sentence embeddings (computed on the machine, no external calls) to *offer* candidates.
- *Tests*: the metamorphic renaming tests (same answers when columns are renamed to synonyms).

**M4. Forecast quality (issue 15).**
- *File*: `followup.forecast`.
- *Change*: add seasonal-naive and exponential-smoothing candidates, choose by backtest, and report an empirical interval from backtest errors.
- *Tests*: synthetic seasonal series with known structure.

### Long-term improvements

**R1. A planner model that knows the program format (issue 7).**
- *Options*:
  - (a) build program-format SFT data from the evaluation-style generators (new domains, paraphrases, composite requests, clarifications, refusals) and train a LoRA **on the base model**, evaluating against base-without-adapter and v1;
  - (b) a stronger model on a machine with enough memory.
- *Needs*: training data **and** fine-tuning, or a different model.
- *Tests*: `training/scripts/eval_program_planner.py` with frozen held-out cases; never train on the evaluation cases or the ten held-out datasets.
- *Risk*: regressions as seen in v2/v3. Accept a new adapter only if it beats both v1 and the base model on validity, metric correctness and refusals.

**R2. Hosted model serving (issue 13).** An OpenAI-compatible server on a GPU host, with authentication; set `ANALYST_MODEL_URL`. *Risk*: data leaves the machine, so apply the program planner's grounding rules everywhere first (issue 18).

**R3. Document intelligence (issue 14).** OCR, table extraction into DataFrames (then the normal analysis paths), and hybrid BM25 + embedding retrieval.

**R4. Constrained decoding (issue 11).** If the inference server supports grammar or JSON-schema decoding, the parsing repairs become unnecessary. *Tests*: malformed-output rate in the evaluation drops to 0.

**R5. Independent evaluation (issue 22).** Run the ten held-out datasets once, record the results, and keep them frozen. Add the browser scripts to the repository and a CI job for the analyst tests.

### What *not* to do first

Do not train a new adapter before Q1–Q3. The two Critical bugs and the wrong-metric bug happen on **deterministic** paths that never call the model, so no amount of training would fix them.
