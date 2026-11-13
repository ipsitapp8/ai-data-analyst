# Analyst planner: base vs v1 vs v2 after the safety fixes

All three models were served the same way: Qwen3-4B-Instruct-2507-4bit (revision `50d42775…`), the same tokenizer, chat template and system prompt, greedy decoding, through `mlx_lm.server`. The only difference is the LoRA adapter (none / v1 / v2). Every run uses the post-fix code: precision-aware grounding, a check on prediction inputs, and a schema contradiction check that allows at most one reconsideration. The frozen v1 reports are unchanged. These runs use new labels.

Sources:
- `training/reports/analyst_finetune_comparison_postfix.md` (held-out)
- `analyst_model_comparison_postfix_summary.md` (25 frozen execution questions)
- `analyst_acceptance_{base_postfix,v1_postfix,v2}.md` (14 scenarios)

## v2 training

The run exited with code 0. All 7 adapter files have 224 tensors each, and none contain NaN or Inf values. `adapters.safetensors` is byte-identical to step 600.

| step | 1 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|
| train loss | – | 0.093 | 0.056 | 0.037 | 0.030 | 0.032 | 0.029 |
| validation loss | 3.500 | 0.064 | 0.038 | 0.039 | **0.031** | 0.032 | 0.036 |

Validation loss is lowest at step 400 and rises slightly by step 600. Training loss says nothing about quality on its own; the tables below are the evidence.

## Held-out test (320 examples, same checker)

| measure | base | v1 | v2 |
|---|---:|---:|---:|
| intent | 0.673 | **0.992** | 0.977 |
| tool selection | 0.531 | 0.850 | **1.000** |
| valid and executable plan | 0.619 | 0.904 | **0.977** |
| exact arguments | 0.423 | 0.812 | **0.908** |
| clarification | 0.000 | **0.900** | 0.850 |
| unsupported-request decline | 0.000 | **1.000** | 0.850 |
| business recommendation planning | 0.000 | 0.500 | **1.000** |
| statistics method | 0.800 | 0.900 | **1.000** |
| Random Forest target and task | 0.025 | 1.000 | 1.000 |
| narration grounded | 0.967 | 0.950 | 0.950 |
| **regression direction (Y on X)** | 0.850 | **0.900** | **0.500** |
| **decline when ad columns absent** | 0.000 | **1.000** | 0.850 |

On the unseen education domain, v2 is worse than v1:
- intent: v1 0.986, v2 0.942;
- unsupported-request decline: v1 1.000, v2 0.769.

## Frozen 25-question execution test (the deterministic planner scores 20/25)

| model | first pass | model plan after one reconsideration | regressions on rule passes (first pass) |
|---|---:|---:|---|
| base | 10/25 | 10/25 | 13 questions |
| v1 | **21/25** | **24/25** | 15, 19, 21 (all recovered) |
| v2 | 20/25 | 23/25 | 18, 19, 21 (all recovered) |

v2 failures:
- #11: malformed JSON (a duplicated fragment).
- #24 "Show ROI and CAC by area": a false decline on its first pass and again after reconsideration.

v1 fails only #8, where it asks which column to predict instead of choosing `sold_quickly`.

## The behaviours checked specifically (execution and acceptance combined)

"After" counts only plans the model wrote, including after its one reconsideration. Answers from the deterministic fallback are never counted.

| behaviour | n | base first / after | v1 first / after | v2 first / after |
|---|---:|---|---|---|
| recommend or ROI when ad columns exist | 7 | 1 / 1 | **4 / 7** | 1 / 6 |
| decline when ad data is absent or unsupported | 3 | 1 / 1 | 3 / 3 | 3 / 3 |
| prediction (RF target and task) | 7 | 0 / 0 | 4 / 6 | **7 / 7** |
| clarify or refuse when no sensible target | 2 | 0 / 0 | 1 / 2 | 1 / 2 |

Regression direction ("Regress Y on X") is measured on the held-out set only, because the execution set has no such question. Scores: base 0.85, v1 0.90, v2 0.50.

## 14 acceptance scenarios

| model | passed | on the model's first pass | model plan only (no fallback) |
|---|---:|---:|---:|
| base | 7/14 | 3 | 3 |
| v1 | 14/14 | 11 | 14 |
| v2 | 14/14 | 11 | 14 |

Reconsiderations:
- v1 needed one on scenarios 5, 7 and 14.
- v2 needed one on scenarios 7, 8 and 14.

On scenario 14 (12 stores), both models first said "there is no churn column" while listing churn as an option. That contradiction was caught. After reconsideration they asked for a Random Forest, and the tool refused because the data is too small, which is the expected outcome.

## What v2 set out to fix and what happened

| target of the v2 data | result in the app |
|---|---|
| "Predict customer churn" false "no churn column" | **fixed**: execution #15 and acceptance #5 now pass on the first pass |
| advertising recommendations when ad data exists | **not fixed**: first pass 1/7 for v2 against 4/7 for v1, despite 1.000 on held-out |
| budget questions | **not fixed**: #21 and acceptance #8 still decline falsely first |
| ROI/CAC comparisons | held-out 0.50 to 1.00, but **execution #24 regressed** (a false decline that survives reconsideration) |
| — side effects | regression direction 0.90 to 0.50; ad-question declines on non-ad data 1.00 to 0.85; a malformed JSON on #11 |

On the held-out set, v2 correctly recommends 40 of 40 times when ad columns exist. On the real campaign demo dataset it still declines falsely. So what v2 learned does not carry over from the training schemas to the demo schema. This should be investigated before any v3: compare how the schema is rendered for the demo `campaigns` data with the generated training schemas.

## Recommendation: use **v1** for the demo

- v1 is better on the frozen execution test: 21 and 24 out of 25, against 20 and 23 for v2. Every v1 first-pass miss on a rule-passed question is recovered by the model itself after one reconsideration. v2 leaves #24 unrecovered and produced one malformed plan.
- Ad recommendations and budget questions are the centrepiece of the demo. v1 gets 4 of 7 right first time; v2 gets 1 of 7, so v2 would depend on the reconsideration step for almost every advertising answer.
- v2 has regressions v1 does not: linear regression direction (0.50) and declines on non-ad data.
- v2's real gains are prediction, especially the churn fix and #8, plus tool selection and arguments on the held-out set. They are not enough to outweigh the points above for this demo.
- Being newer is not a reason to choose v2.

Known v1 weaknesses to disclose in a demo:
- On its first pass it can falsely say "no churn column" or "no advertising history". The contradiction check catches this and asks it once to reconsider, and the answer is labelled "after one reconsideration".
- #8 asks a clarifying question instead of choosing `sold_quickly`.
- Narration can truncate where it should round; the precision-aware checker flags this (0.950 grounded).
