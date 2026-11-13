# Spider number-answer questions: copilot baseline without a model (spider-numeric.1)

Run 5, 2026-11-10T07:01:00 UTC. Case file SHA-256 `dd147f7b5d159e55…`. Model calls: 0. Generated code runs: 0.

Path measured: copilot.handle_message with allow_model=False, first turn of a new session: rule interpretation, validated operations, deterministic metric engine. No model call, no generated code.

Questions and SQL quoted below are from Spider (CC BY-SA 4.0).

## Accuracy

| Subset | Questions | Exact | Within tolerance |
| --- | ---: | ---: | ---: |
| All selected questions | 165 | 4 (2.4%) | 4 (2.4%) |
| Reference is a number | 160 | 4 (2.5%) | 4 (2.5%) |
| …cleanly held out (no leakage marker) | 138 | 3 (2.2%) | 3 (2.2%) |
| …one number asked for | 142 | 4 (2.8%) | 4 (2.8%) |
| …several numbers asked for | 18 | 0 (0.0%) | 0 (0.0%) |

Distinct reference queries: 87; all paraphrases correct for 1, at least one for 3.

Tolerance: within 0.1% of the reference or 0.005 absolute; includes exact. Several-number questions count only if every number is returned.

Correct number from a different computation than the question asks for: 3 (spider-dev-0087, spider-dev-0088, spider-dev-0195).

## What the copilot did

| Outcome | Questions |
| --- | ---: |
| not_understood | 102 |
| no_metric | 40 |
| clarification | 18 |
| correct_exact | 4 |
| wrong_value | 1 |

Correct 4, declined as not understood or ambiguous 160, answered with a wrong number 1, errors 0.

## Why the others failed

Each failure is put to an oracle: the reference SQL, translated from Spider's own parse into a metric-engine spec and computed by the engine on the CSV the application would receive.

| Cause | Questions | Meaning |
| --- | ---: | --- |
| question_understanding | 135 | the oracle spec gives the right number: the rules misread the question |
| deterministic_limitation | 19 | the engine or the copilot cannot express what the question asks |
| data_representation | 7 | the reference is not a number, or SQLite and the engine read the data differently |

### question_understanding

| Detail | Questions |
| --- | ---: |
| a count of entities ('how many singers') was not read as a row count | 63 |
| the aggregation or the column is worded in a way the rules do not recognise | 56 |
| the metric wording matched several columns, and the rules asked which one | 6 |
| a year in the question was read as a date range, but the table has no date column | 5 |
| a value in the question occurs in several columns, and the rules asked which one | 4 |
| wrong metric: computed count_distinct("state"), the question asks for count() | 1 |

### deterministic_limitation

| Detail | Questions |
| --- | ---: |
| the question asks for several numbers; the copilot answers with one metric | 18 |
| OR across different columns or operators (engine filters are AND-ed) | 1 |

### data_representation

| Detail | Questions |
| --- | ---: |
| the reference answer is not a number | 3 |
| the engine, reading the CSV as the application does, gets a different number for the same spec | 2 |
| the reference SQL returns NULL | 2 |

### Question-understanding issues (a question can have several)

| Issue | Questions |
| --- | ---: |
| a count of entities ('how many singers') was not read as a row count | 63 |
| the aggregation or the column is worded in a way the rules do not recognise | 56 |
| a condition in the question was not turned into a filter | 29 |
| a filter was applied that the question does not state | 19 |
| the metric wording matched several columns, and the rules asked which one | 6 |
| a year in the question was read as a date range, but the table has no date column | 5 |
| a value in the question occurs in several columns, and the rules asked which one | 4 |
| wrong metric | 1 |

## Ceiling with the current engine

139 of 142 single-number questions are expressible and reproduce the reference through the engine. With perfect question reading and no engine change, at most 139 could be correct; 4 are now.

## Time

Total 1254 ms; per case mean 7.6 ms, median 2 ms, 95th percentile 22 ms, max 363 ms (per case, including reading and profiling the CSV (cached after the first case on a table)).

## Answered with a wrong number

- `spider-dev-0687` How many states are there? — computed count_distinct("state") = 51.0, reference [305] (wrong metric: computed count_distinct("state"), the question asks for count())

## Correct

- `spider-dev-0087` How many continents are there? — count_distinct("Continent")  = 5.0
- `spider-dev-0088` What is the number of continents? — count_distinct("Continent")  = 5.0
- `spider-dev-0195` How many airlines are from USA? — count_distinct("Airline") [{'column': 'Country', 'op': '==', 'value': 'USA'}] = 12.0
- `spider-dev-0444` How many different loser names are there? — count_distinct("loser_name")  = 49.0

The full per-question listing is in the JSON report beside this file.
