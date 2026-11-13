# Analyst model evaluation v1_postfix

Base `mlx-community/Qwen3-4B-Instruct-2507-4bit` vs the same model with LoRA adapter `training/adapters/analyst_qwen3_4b_lora`. 320 test examples (up to 20 per intent), greedy decoding, prompts identical to the application's.

| intent | n | json_valid base → tuned | type_ok base → tuned | tools_ok base → tuned | args_ok base → tuned | executable base → tuned | grounded base → tuned |
|---|---:|---|---|---|---|---|---|
| _all | 260 | 0.98 → 1.00 | 0.67 → 0.99 | 0.53 → 0.85 | 0.42 → 0.81 | 0.58 → 0.89 | 0.97 → 0.95 |
| chat | 20 | 1.00 → 1.00 | 0.25 → 1.00 |  |  |  |  |
| clarify | 20 | 1.00 → 1.00 | 0.00 → 0.90 |  |  |  |  |
| comparison | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |  |
| correlation | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |  |
| decline | 20 | 1.00 → 1.00 | 0.00 → 1.00 |  |  |  |  |
| explain_prediction | 20 | 1.00 → 1.00 | 0.95 → 1.00 | 0.05 → 1.00 | 0.05 → 1.00 | 0.05 → 1.00 |  |
| group_comparison | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.90 → 1.00 |  |
| metric | 20 | 1.00 → 1.00 | 0.75 → 1.00 | 0.70 → 1.00 | 0.25 → 1.00 | 0.30 → 1.00 |  |
| narrate_compare | 20 |  |  |  |  |  | 0.95 → 1.00 |
| narrate_metric | 20 |  |  |  |  |  | 1.00 → 0.95 |
| narrate_which | 20 |  |  |  |  |  | 0.95 → 0.90 |
| performance | 20 | 0.80 → 1.00 | 0.80 → 1.00 | 0.00 → 0.00 | 0.00 → 0.00 | 0.06 → 0.00 |  |
| prediction | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 0.45 → 1.00 | 0.00 → 1.00 | 0.35 → 1.00 |  |
| recommendation | 20 | 1.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 |  |
| regression | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 0.85 → 0.90 | 1.00 → 1.00 |  |
| seen_domains_heldout_phrasing | 125 | 0.97 → 1.00 | 0.62 → 0.99 | 0.43 → 0.73 | 0.30 → 0.73 | 0.43 → 0.79 |  |
| statistics | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 0.50 | 1.00 → 0.50 | 1.00 → 0.85 |  |
| unseen:education | 69 | 1.00 → 1.00 | 0.70 → 0.99 | 0.65 → 0.98 | 0.57 → 0.98 | 0.76 → 0.98 | 1.00 → 0.93 |
| unseen:real_estate | 66 | 1.00 → 1.00 | 0.74 → 1.00 | 0.60 → 0.96 | 0.52 → 0.80 | 0.68 → 0.98 | 0.94 → 0.97 |

## All test examples

| measure | n | base | fine-tuned |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 260 | 0.673 | 0.992 |
| 2 tool selection | 196 | 0.531 | 0.850 |
| 3 structured plan validity (parses and passes backend validation) | 260 | 0.619 | 0.904 |
| 4 column selection (exact arguments) | 196 | 0.423 | 0.812 |
| 5 clarification behaviour | 20 | 0.000 | 0.900 |
| 6 unsupported-request handling | 20 | 0.000 | 1.000 |
| 7 business recommendation planning | 36 | 0.000 | 0.500 |
| 8 statistical method selection | 100 | 0.800 | 0.900 |
| 9 Random Forest planning (target and task) | 40 | 0.025 | 1.000 |
| 10 general conversation | 20 | 0.250 | 1.000 |
| narration grounded in tool results | 60 | 0.967 | 0.950 |

## Unseen domain: education

| measure | n | base | fine-tuned |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 69 | 0.696 | 0.986 |
| 2 tool selection | 49 | 0.653 | 0.980 |
| 3 structured plan validity (parses and passes backend validation) | 69 | 0.768 | 0.971 |
| 4 column selection (exact arguments) | 49 | 0.571 | 0.982 |
| 5 clarification behaviour | 7 | 0.000 | 0.857 |
| 6 unsupported-request handling | 13 | 0.000 | 1.000 |
| 8 statistical method selection | 32 | 0.844 | 0.969 |
| 9 Random Forest planning (target and task) | 14 | 0.071 | 1.000 |
| narration grounded in tool results | 29 | 1.000 | 0.931 |

## Unseen domain: real_estate

| measure | n | base | fine-tuned |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 66 | 0.742 | 1.000 |
| 2 tool selection | 50 | 0.600 | 0.960 |
| 3 structured plan validity (parses and passes backend validation) | 66 | 0.697 | 0.985 |
| 4 column selection (exact arguments) | 50 | 0.520 | 0.797 |
| 5 clarification behaviour | 9 | 0.000 | 1.000 |
| 6 unsupported-request handling | 7 | 0.000 | 1.000 |
| 8 statistical method selection | 33 | 0.818 | 0.939 |
| 9 Random Forest planning (target and task) | 14 | 0.000 | 1.000 |
| narration grounded in tool results | 31 | 0.935 | 0.968 |

Time: base 1.24 s/example, fine-tuned 1.65 s/example.
