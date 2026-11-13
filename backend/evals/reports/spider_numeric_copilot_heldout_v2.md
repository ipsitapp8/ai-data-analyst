# Spider number questions, held out: frozen copilot rules vs baseline (v2)

Single run 6, 2026-11-10T07:40:51 UTC, suite spider-numeric.1. Model calls: 0. Generated code runs: 0. Rules frozen 2026-11-10T07:40:51+00:00 (Git HEAD 7ac5972 plus the working tree; file hashes in the JSON). Baseline: spider_numeric_copilot_baseline_v1.json.

Held out: numeric reference and no leakage marker (question_also_in_train / database_family_in_train).

Questions and SQL quoted below are from Spider (CC BY-SA 4.0).

## Accuracy

### Held out (138 questions)

| | Before (v1 baseline) | After (frozen rules) |
| --- | ---: | ---: |
| Exact | 3 (2.2%) | 8 (5.8%) |
| Within tolerance | 3 (2.2%) | 8 (5.8%) |
| Declined (abstained) | 135 (97.8%) | 130 (94.2%) |
| Incorrect answer | 0 (0.0%) | 0 (0.0%) |
| Errors | 0 | 0 |
| Correct when it answers | 1.0 | 1.0 |

### Every numeric reference (160 questions)

| | Before (v1 baseline) | After (frozen rules) |
| --- | ---: | ---: |
| Exact | 4 (2.5%) | 13 (8.1%) |
| Within tolerance | 4 (2.5%) | 13 (8.1%) |
| Declined (abstained) | 155 (96.9%) | 146 (91.2%) |
| Incorrect answer | 1 (0.6%) | 1 (0.6%) |
| Errors | 0 | 0 |
| Correct when it answers | 0.8 | 0.9286 |

### All selected (165 questions)

| | Before (v1 baseline) | After (frozen rules) |
| --- | ---: | ---: |
| Exact | 4 (2.4%) | 13 (7.9%) |
| Within tolerance | 4 (2.4%) | 13 (7.9%) |
| Declined (abstained) | 160 (97.0%) | 151 (91.5%) |
| Incorrect answer | 1 (0.6%) | 1 (0.6%) |
| Errors | 0 | 0 |
| Correct when it answers | 0.8 | 0.9286 |

## What changed on the held-out questions

| Before -> after | Questions |
| --- | ---: |
| declined -> declined | 130 |
| declined -> correct | 5 |
| correct -> correct | 3 |

Declined, by kind:

| | Before | After |
| --- | ---: | ---: |
| clarification | 12 | 9 |
| no_metric | 35 | 39 |
| not_understood | 88 | 82 |

Still declined, by cause:

| Cause | Questions |
| --- | ---: |
| question_understanding | 127 |
| data_representation | 2 |
| deterministic_limitation | 1 |

| Detail | Questions |
| --- | ---: |
| the aggregation or the column is worded in a way the rules do not recognise | 68 |
| a count of entities ('how many singers') was not read as a row count | 50 |
| the metric wording matched several columns, and the rules asked which one | 9 |
| the engine, reading the CSV as the application does, gets a different number for the same spec | 2 |
| OR across different columns or operators (engine filters are AND-ed) | 1 |

## Incorrect answers (held out)

None.

## Lost (correct before, not now)

None.

Correct answers whose computation matches the reference SQL's: 3 of 8.

Time: 1317 ms for all 165, median 2 ms per question.

The newly correct questions and every per-question detail are in the JSON report beside this file.
