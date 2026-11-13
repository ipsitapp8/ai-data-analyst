# Execution comparison: base_postfix, v1_postfix, v2

Deterministic planner (frozen): 20/25.

| model | first pass | after one reconsideration | fixes rule misses (first pass) | regressions on rule passes (first pass / after retry) |
|---|---:|---:|---:|---|
| base_postfix | 10/25 | 10/25 | 3/5 | [2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 18, 19, 21] / [2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 18, 19, 21] |
| v1_postfix | 21/25 | 24/25 | 4/5 | [15, 19, 21] / none |
| v2 | 20/25 | 23/25 | 3/5 | [18, 19, 21] / none |

## Per question (first pass; R = passed only after reconsideration)

| # | question | rules | base_postfix | v1_postfix | v2 |
|---:|---|---|---|---|---|
| 1 | What is the average grade points? | pass | pass | pass | pass |
| 2 | Which campus has the highest average grade points? | pass | **fail** (compute_metric({"aggregation": "avg", "column": "grade_point) | pass | pass |
| 3 | Is there a significant difference in grade points between campus groups? | pass | pass | pass | pass |
| 4 | Predict passed | pass | **fail** (random_forest({"target": "passed", "features": ["study_hours) | pass | pass |
| 5 | Which factors influenced your prediction? | pass | pass | pass | pass |
| 6 | What is the median price in each city? | pass | **fail** (describe_column({"column": "price", "filters": {"city": "Bay) | pass | pass |
| 7 | Predict price | pass | **fail** (random_forest({"target": "price", "features": ["bedrooms", ") | pass | pass |
| 8 | Can you predict which properties will sell within 90 days? | fail | **fail** (predict({"rows": [["Bayview", 3, 2, 100, "Yes", "House"], [") | **fail** (clarify: ['city', 'bedrooms', 'floor_area_sqm', 'price', 'so) | pass |
| 9 | Which area should receive our next advertising campaign? | pass | **fail** (unparseable reply) | pass | pass |
| 10 | What is the average salary in Engineering? | pass | **fail** (compute_metric({"aggregation": "avg", "column": "salary", "f) | pass | pass |
| 11 | How many employees have a salary above 100000? | fail | **fail** (compute_metric({"aggregation": "count", "column": "employee_) | pass | **fail** (unparseable reply) |
| 12 | Show the total salary by department | pass | **fail** (compute_metric({"aggregation": "sum", "column": "salary", "f) | pass | pass |
| 13 | Is years of experience correlated with salary? | fail | pass | pass | pass |
| 14 | What is the standard deviation of salary? | pass | **fail** (compute_metric({"aggregation": "std", "column": "salary"})) | pass | pass |
| 15 | Predict customer churn. | pass | **fail** (random_forest({"target": "churn", "task": "classification", ) | R | pass |
| 16 | Which factors influenced your prediction? | pass | pass | pass | pass |
| 17 | Predict salary. | pass | **fail** (random_forest({"target": "salary", "features": ["department") | pass | pass |
| 18 | Should we advertise in Area A? | pass | **fail** (clarify: ['conversions', 'revenue', 'ROI', 'lead-to-conversi) | pass | R |
| 19 | Which area should receive the next campaign? | pass | **fail** (clarify: ['conversion rate', 'revenue per lead', 'ROI', 'lea) | R | R |
| 20 | Why do you recommend Area A? | pass | pass | pass | pass |
| 21 | Should we increase the budget? | pass | **fail** (clarify: ['Area A', 'Area B', 'Area C', 'Area D', 'Area E', ) | R | R |
| 22 | What happens if the budget increases by 20%? | pass | pass | pass | pass |
| 23 | Is the difference between these areas statistically significant? | fail | pass | pass | pass |
| 24 | Show ROI and CAC by area | fail | pass | pass | **fail** (decline) |
| 25 | Should we advertise in the Tools category? | pass | pass | pass | pass |
