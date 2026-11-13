# Analyst acceptance fine_tuned_v1 (model on)

Passed 11/14; plans written by the model: 14.

| # | question | expected | planner | tools / plan | result |
|---:|---|---|---|---|---|
| 1 | Hello. | conversational answer, no tools | model | chat | pass |
| 2 | What is Random Forest? | general explanation | model | chat | pass |
| 3 | What is the average salary? | correct computed value | model | compute_metric | pass |
| 4 | Which department pays the most? | grouping and comparison | model | compute_metric, compare_groups | pass |
| 5 | Predict customer churn. | Random Forest classification | model | clarify | **fail** |
| 6 | Predict salary. | Random Forest regression | model | random_forest | pass |
| 7 | Which area should receive our next advertising campaign? | historical campaign analysis and recommendation | model | decline | **fail** |
| 8 | Should we increase the advertising budget in Area A? | performance, ROI, comparison, uncertainty, recommendation | model | recommend | pass |
| 9 | Why do you recommend Area A? | statistics, method and limitations of the previous recommendation | model | explain_previous | pass |
| 10 | Which factors influenced your prediction? | feature importance from the trained model | model | feature_importance | **fail** |
| 11 | What if we increase the budget by 20%? | scenario with disclosed assumptions | model | budget_scenario | pass |
| 12 | Predict churn. | clarification, no guess | model | clarify | pass |
| 13 | Which area should receive our next advertising campaign? | explains the data is insufficient | model | decline | pass |
| 14 | Predict churn. | explains why a prediction cannot responsibly be made | model | clarify | pass |
