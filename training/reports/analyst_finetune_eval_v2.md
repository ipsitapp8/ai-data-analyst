# Analyst model evaluation v2

Base `mlx-community/Qwen3-4B-Instruct-2507-4bit` vs the same model with LoRA adapter `training/adapters/analyst_qwen3_4b_lora_v2`. 320 test examples (up to 20 per intent), greedy decoding, prompts identical to the application's.

| intent | n | json_valid base → tuned | type_ok base → tuned | tools_ok base → tuned | args_ok base → tuned | executable base → tuned | grounded base → tuned |
|---|---:|---|---|---|---|---|---|
| _all | 260 | 0.98 → 1.00 | 0.67 → 0.98 | 0.53 → 1.00 | 0.42 → 0.91 | 0.58 → 1.00 | 0.97 → 0.95 |
| chat | 20 | 1.00 → 1.00 | 0.25 → 1.00 |  |  |  |  |
| clarify | 20 | 1.00 → 1.00 | 0.00 → 0.85 |  |  |  |  |
| comparison | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |  |
| correlation | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |  |
| decline | 20 | 1.00 → 1.00 | 0.00 → 0.85 |  |  |  |  |
| explain_prediction | 20 | 1.00 → 1.00 | 0.95 → 1.00 | 0.05 → 1.00 | 0.05 → 1.00 | 0.05 → 1.00 |  |
| group_comparison | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.90 → 1.00 |  |
| metric | 20 | 1.00 → 1.00 | 0.75 → 1.00 | 0.70 → 1.00 | 0.25 → 1.00 | 0.30 → 1.00 |  |
| narrate_compare | 20 |  |  |  |  |  | 0.95 → 0.95 |
| narrate_metric | 20 |  |  |  |  |  | 1.00 → 1.00 |
| narrate_which | 20 |  |  |  |  |  | 0.95 → 0.90 |
| performance | 20 | 0.80 → 1.00 | 0.80 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.06 → 1.00 |  |
| prediction | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 0.45 → 1.00 | 0.00 → 1.00 | 0.35 → 1.00 |  |
| recommendation | 20 | 1.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 | 0.00 → 1.00 |  |
| regression | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 0.85 → 0.50 | 1.00 → 1.00 |  |
| seen_domains_heldout_phrasing | 125 | 0.97 → 1.00 | 0.62 → 0.98 | 0.43 → 1.00 | 0.30 → 0.93 | 0.43 → 1.00 |  |
| statistics | 20 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |  |
| unseen:education | 69 | 1.00 → 1.00 | 0.70 → 0.94 | 0.65 → 1.00 | 0.57 → 0.95 | 0.76 → 1.00 | 1.00 → 0.93 |
| unseen:real_estate | 66 | 1.00 → 1.00 | 0.74 → 1.00 | 0.60 → 1.00 | 0.52 → 0.83 | 0.68 → 1.00 | 0.94 → 0.97 |

## All test examples

| measure | n | base | fine-tuned |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 260 | 0.673 | 0.977 |
| 2 tool selection | 196 | 0.531 | 1.000 |
| 3 structured plan validity (parses and passes backend validation) | 260 | 0.619 | 0.977 |
| 4 column selection (exact arguments) | 196 | 0.423 | 0.908 |
| 5 clarification behaviour | 20 | 0.000 | 0.850 |
| 6 unsupported-request handling | 20 | 0.000 | 0.850 |
| 7 business recommendation planning | 36 | 0.000 | 1.000 |
| 8 statistical method selection | 100 | 0.800 | 1.000 |
| 9 Random Forest planning (target and task) | 40 | 0.025 | 1.000 |
| 10 general conversation | 20 | 0.250 | 1.000 |
| narration grounded in tool results | 60 | 0.967 | 0.950 |

## Unseen domain: education

| measure | n | base | fine-tuned |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 69 | 0.696 | 0.942 |
| 2 tool selection | 49 | 0.653 | 1.000 |
| 3 structured plan validity (parses and passes backend validation) | 69 | 0.768 | 0.942 |
| 4 column selection (exact arguments) | 49 | 0.571 | 0.946 |
| 5 clarification behaviour | 7 | 0.000 | 0.857 |
| 6 unsupported-request handling | 13 | 0.000 | 0.769 |
| 8 statistical method selection | 32 | 0.844 | 1.000 |
| 9 Random Forest planning (target and task) | 14 | 0.071 | 1.000 |
| narration grounded in tool results | 29 | 1.000 | 0.931 |

## Unseen domain: real_estate

| measure | n | base | fine-tuned |
|---|---:|---:|---:|
| 1 intent recognition (plan type) | 66 | 0.742 | 1.000 |
| 2 tool selection | 50 | 0.600 | 1.000 |
| 3 structured plan validity (parses and passes backend validation) | 66 | 0.697 | 1.000 |
| 4 column selection (exact arguments) | 50 | 0.520 | 0.831 |
| 5 clarification behaviour | 9 | 0.000 | 1.000 |
| 6 unsupported-request handling | 7 | 0.000 | 1.000 |
| 8 statistical method selection | 33 | 0.818 | 1.000 |
| 9 Random Forest planning (target and task) | 14 | 0.000 | 1.000 |
| narration grounded in tool results | 31 | 0.935 | 0.968 |

Time: base 1.23 s/example, fine-tuned 1.51 s/example.
