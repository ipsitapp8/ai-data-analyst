# Universal What-If

One scenario engine, six domain packs, two doors (the What-If page and the Silt Copilot).

## Audit summary (what already existed and was reused)

| Need | Existing code | Used how |
|---|---|---|
| Deterministic, sensitivity, Shapley attribution, break-even, optimistic/pessimistic | `app/scenarios.py`, `scenario_sensitivity.py`, `scenario_service.evaluate` | Formula-model templates call `scenario_service.evaluate` |
| Monte Carlo (seeded) | `scenario_sim.py` via `scenario_service.simulate` | Healthcare surge, any template with method `monte_carlo` |
| Goal seek / constraints | `scenario_solve.py` via `scenario_service.goal_seek` | Method `goal_seek` |
| Custom formula models | `scenario_models.py`, `scenario_formula.py` | Every domain template that is a formula, and the Custom domain, compile to this |
| Column roles | `analyst/semantics.py` | `whatif/profile.py` builds on it |
| Persistence | `scenarios` table | Universal scenarios stored under model key `universal:<domain>:<template>`; no migration |
| Copilot bridge | `analyst/scenario_bridge.py` | Left untouched; `whatif_bridge.py` sits beside it |

Duplicated before this work: none between page and Copilot (both already used `scenario_service`). Risk managed: the business list/compare routes
now ignore `universal:` rows so the old library never meets a model key it does not know.

## Layout

```
app/whatif/
  domains/     business, education, healthcare, travel, government, custom  (describe: concepts, templates, limits)
  profile.py   DatasetProfile + explainable domain detection + concept->column mapping
  discovery.py suggestions classified ready / needs_assumptions / needs_external_data / unsupported (rules; optional model re-ranks ids only)
  scenario.py  standard scenario record + validation (asks for missing values, never invents)
  operations.py row-level maths: threshold, shift_metric, association (OLS + 95% CI), forecast_trend (prediction interval)
  engine.py    runs a scenario; formula models go to app.scenario_service; compare()
  explain.py   eight-part explanation, evidence kind (exact / estimate / forecast / hypothetical / unsupported)
  service.py   the single entry point used by the router and the Copilot
  parse.py     numbers from a sentence (rules)       llm.py  optional ranker
app/routers/whatif.py        /api/whatif/*  (team-scoped, thin)
app/analyst/whatif_bridge.py Copilot adapter (thin)
frontend/what_if_universal.py  "Explore any scenario" tab
```

## Flows

Standalone: dataset -> `/profile` -> `/suggest` -> pick -> `/run` (validate -> engine -> explanation) -> `/scenarios` save -> `/compare`.
Copilot: sentence -> `whatif_bridge` -> `service.from_question` (template match + numbers from the sentence) -> `service.run` -> structured result -> reply text built from the result.
Neither path needs a language model. `use_model` on `/suggest` only lets a model reorder existing suggestion ids; failure falls back to rules.

## Safety rules enforced in code

* Domain is detected only when several distinct concepts match and it leads the runner-up; otherwise Custom. Manual override is recorded.
* Ambiguous columns are returned as candidates for the user to choose; missing inputs are listed and asked for.
* Association estimates need >=30 complete pairs and p<0.05, else `insufficient_evidence` with no number; forecasts need >=8 periods.
* Monte Carlo seeds are stored and reproduce runs; distributions are labelled assumptions.
* Healthcare: operational only; treatment/outcome questions are unsupported; sensitive-looking columns are never listed.
* The server recomputes every result on save/run; client-sent results are ignored. Datasets and saved scenarios are team-scoped (404 across teams).

## Not done / limits

* Optimisation beyond constrained goal seek is not implemented (a single solve-for input only).
* Copilot replies are templated from the structured result; there is no LLM phrasing layer yet.
* PDF datasets are only usable if the existing ingestion already turns them into a table.
* Domain templates are a starting set; each domain's concept vocabulary is English and keyword-based.
* No live prices (travel) and no external data connectors.
