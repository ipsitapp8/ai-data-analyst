# Analyst planner held-out comparison: base vs v1 vs v2 (same post-fix checker)

Test split sha256 `d0ba13f02e734d09…`, 320 examples, greedy decoding, identical prompts. Base replies identical in both runs: True.

## Standard measures

| measure | n | base | v1 | v2 |
|---|---:|---:|---:|---:|
| 1 intent recognition (plan type) | 260 | 0.673 | 0.992 | 0.977 |
| 2 tool selection | 200 | 0.531 | 0.850 | 1.000 |
| 3 structured plan validity (parses and passes backend validation) | 260 | 0.619 | 0.904 | 0.977 |
| 4 column selection (exact arguments) | 217 | 0.423 | 0.812 | 0.908 |
| 5 clarification behaviour | 20 | 0.000 | 0.900 | 0.850 |
| 6 unsupported-request handling | 20 | 0.000 | 1.000 | 0.850 |
| 7 business recommendation planning | 40 | 0.000 | 0.500 | 1.000 |
| 8 statistical method selection | 100 | 0.800 | 0.900 | 1.000 |
| 9 Random Forest planning (target and task) | 40 | 0.025 | 1.000 | 1.000 |
| 10 general conversation | 20 | 0.250 | 1.000 | 1.000 |
| narration grounded in tool results | 60 | 0.967 | 0.950 | 0.950 |

## Behaviours checked specifically

| behaviour | n | base | v1 | v2 |
|---|---:|---:|---:|---:|
| regression_direction | 20 | 0.850 (17) | 0.900 (18) | 0.500 (10) |
| decline_when_ad_columns_absent | 20 | 0.000 (0) | 1.000 (20) | 0.850 (17) |
| recommend_when_ad_columns_present | 40 | 0.000 (0) | 0.500 (20) | 1.000 (40) |
| prediction_target_and_task | 20 | 0.000 (0) | 1.000 (20) | 1.000 (20) |
| explain_prediction | 20 | 0.050 (1) | 1.000 (20) | 1.000 (20) |
| clarification | 20 | 0.000 (0) | 0.900 (18) | 0.850 (17) |

## Unseen domain: education

| measure | base | v1 | v2 |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 0.696 | 0.986 | 0.942 |
| 2 tool selection | 0.653 | 0.980 | 1.000 |
| 3 structured plan validity (parses and passes backend validation) | 0.768 | 0.971 | 0.942 |
| 4 column selection (exact arguments) | 0.571 | 0.982 | 0.946 |
| 5 clarification behaviour | 0.000 | 0.857 | 0.857 |
| 6 unsupported-request handling | 0.000 | 1.000 | 0.769 |
| 8 statistical method selection | 0.844 | 0.969 | 1.000 |
| 9 Random Forest planning (target and task) | 0.071 | 1.000 | 1.000 |
| narration grounded in tool results | 1.000 | 0.931 | 0.931 |

## Unseen domain: real_estate

| measure | base | v1 | v2 |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 0.742 | 1.000 | 1.000 |
| 2 tool selection | 0.600 | 0.960 | 1.000 |
| 3 structured plan validity (parses and passes backend validation) | 0.697 | 0.985 | 1.000 |
| 4 column selection (exact arguments) | 0.520 | 0.797 | 0.831 |
| 5 clarification behaviour | 0.000 | 1.000 | 1.000 |
| 6 unsupported-request handling | 0.000 | 1.000 | 1.000 |
| 8 statistical method selection | 0.818 | 0.939 | 1.000 |
| 9 Random Forest planning (target and task) | 0.000 | 1.000 | 1.000 |
| narration grounded in tool results | 0.935 | 0.968 | 0.968 |
