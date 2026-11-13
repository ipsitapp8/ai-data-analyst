# Analyst acceptance rules_v1 (model off)

Passed 14/14; plans written by the model: 0.

| # | question | expected | planner | tools / plan | result |
|---:|---|---|---|---|---|
| 1 | Hello. | conversational answer, no tools | rules | chat | pass |
| 2 | What is Random Forest? | general explanation | rules | chat | pass |
| 3 | What is the average salary? | correct computed value | rules | compute_metric | pass |
| 4 | Which department pays the most? | grouping and comparison | rules | compute_metric, compare_groups | pass |
| 5 | Predict customer churn. | Random Forest classification | rules | random_forest | pass |
| 6 | Predict salary. | Random Forest regression | rules | random_forest | pass |
| 7 | Which area should receive our next advertising campaign? | historical campaign analysis and recommendation | rules | recommend | pass |
| 8 | Should we increase the advertising budget in Area A? | performance, ROI, comparison, uncertainty, recommendation | rules | recommend | pass |
| 9 | Why do you recommend Area A? | statistics, method and limitations of the previous recommendation | rules | explain_previous | pass |
| 10 | Which factors influenced your prediction? | feature importance from the trained model | rules | feature_importance | pass |
| 11 | What if we increase the budget by 20%? | scenario with disclosed assumptions | rules | budget_scenario | pass |
| 12 | Predict churn. | clarification, no guess | rules | clarify | pass |
| 13 | Which area should receive our next advertising campaign? | explains the data is insufficient | rules | decline | pass |
| 14 | Predict churn. | explains why a prediction cannot responsibly be made | rules | random_forest | pass |
