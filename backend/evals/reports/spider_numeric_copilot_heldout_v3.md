# Spider number questions, held out: frozen copilot rules vs baseline (v3)

Single run 7, 2026-11-10T08:22:19 UTC, suite spider-numeric.1. Model calls: 0. Generated code runs: 0. Rules frozen 2026-11-10T08:22:19+00:00 (Git HEAD 7ac5972 plus the working tree; file hashes in the JSON). Baseline: spider_numeric_copilot_baseline_v1.json.

Why this freeze: Harness correction only. run_case in app/evals/spider.py did not pass the dataset file name to the copilot, which the application (copilot.handle_message passes dataset.filename) and the development scorer both do; it now passes '<table>.csv'. Every rule file (copilot.py, fastpath.py, metric_engine.py, profiling.py, runner.py) and the case file have the same SHA-256 as in the v2 freeze; only app/evals/spider.py differs, so the v2 freeze no longer matches the files that produce the run.

Held out: numeric reference and no leakage marker (question_also_in_train / database_family_in_train).

Questions and SQL quoted below are from Spider (CC BY-SA 4.0).

## Accuracy

### Held out (138 questions)

| | Before (v1 baseline) | After (frozen rules) |
| --- | ---: | ---: |
| Exact | 3 (2.2%) | 59 (42.8%) |
| Within tolerance | 3 (2.2%) | 59 (42.8%) |
| Declined (abstained) | 135 (97.8%) | 79 (57.2%) |
| Incorrect answer | 0 (0.0%) | 0 (0.0%) |
| Errors | 0 | 0 |
| Correct when it answers | 1.0 | 1.0 |

### Every numeric reference (160 questions)

| | Before (v1 baseline) | After (frozen rules) |
| --- | ---: | ---: |
| Exact | 4 (2.5%) | 73 (45.6%) |
| Within tolerance | 4 (2.5%) | 73 (45.6%) |
| Declined (abstained) | 155 (96.9%) | 87 (54.4%) |
| Incorrect answer | 1 (0.6%) | 0 (0.0%) |
| Errors | 0 | 0 |
| Correct when it answers | 0.8 | 1.0 |

### All selected (165 questions)

| | Before (v1 baseline) | After (frozen rules) |
| --- | ---: | ---: |
| Exact | 4 (2.4%) | 73 (44.2%) |
| Within tolerance | 4 (2.4%) | 73 (44.2%) |
| Declined (abstained) | 160 (97.0%) | 90 (54.5%) |
| Incorrect answer | 1 (0.6%) | 2 (1.2%) |
| Errors | 0 | 0 |
| Correct when it answers | 0.8 | 0.9733 |

## Compared with the previous held-out run

| Held out (138) | Previous run | This run |
| --- | ---: | ---: |
| Exact | 8 | 59 |
| Within tolerance | 8 | 59 |
| Declined | 130 | 79 |
| Incorrect | 0 | 0 |
| Errors | 0 | 0 |

| Previous -> this run | Questions |
| --- | ---: |
| declined -> declined | 79 |
| declined -> correct | 51 |
| correct -> correct | 8 |

## What changed on the held-out questions (vs the v1 baseline)

| Before -> after | Questions |
| --- | ---: |
| declined -> declined | 79 |
| declined -> correct | 56 |
| correct -> correct | 3 |

Declined, by kind:

| | Before | After |
| --- | ---: | ---: |
| clarification | 12 | 6 |
| no_metric | 35 | 24 |
| not_understood | 88 | 49 |

Still declined, by cause:

| Cause | Questions |
| --- | ---: |
| question_understanding | 76 |
| data_representation | 2 |
| deterministic_limitation | 1 |

| Detail | Questions |
| --- | ---: |
| the aggregation or the column is worded in a way the rules do not recognise | 57 |
| a count of entities ('how many singers') was not read as a row count | 13 |
| the metric wording matched several columns, and the rules asked which one | 6 |
| the engine, reading the CSV as the application does, gets a different number for the same spec | 2 |
| OR across different columns or operators (engine filters are AND-ed) | 1 |

## Incorrect answers (held out)

None.

## Lost (correct before, not now)

None.

Correct answers whose computation matches the reference SQL's: 56 of 59.

Time: 1480 ms for all 165, median 3 ms per question.

The newly correct questions and every per-question detail are in the JSON report beside this file.
