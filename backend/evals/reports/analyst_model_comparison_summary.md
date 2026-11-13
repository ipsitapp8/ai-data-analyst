# Base vs fine-tuned on the 25 execution questions

Deterministic planner (frozen): 20/25. Base model: 10/25. Fine-tuned: 21/25.

## The 5 questions the deterministic planner missed

### 8. Can you predict which properties will sell within 90 days?

| | base | fine-tuned |
|---|---|---|
| 1-3 interpretation / plan | predict({"rows": [["Bayview", 3, 2, 100, "Yes", "House"], ["Elm Park", 4, 3, 150, "No", "Flat"]]}) | clarify: ['city', 'bedrooms', 'floor_area_sqm', 'price', 'sold_quickly'] |
| 4 backend validation | predict: INVALID predict.rows: up to 50 rows, each an object of column -> value | n/a (no tool calls) |
| plan actually used | rules; The model proposed an invalid tool call (predict.rows: up to 50 rows, each an object of column -> value); the deterministic planner was used. | model |
| 5 tool execution | none | none |
| 6 final answer | not correct | not correct |

### 11. How many employees have a salary above 100000?

| | base | fine-tuned |
|---|---|---|
| 1-3 interpretation / plan | compute_metric({"aggregation": "count", "column": "employee_id", "filters": {"salary": {"gt": 100000}}}) | compute_metric({"aggregation": "count", "filters": [{"column": "salary", "op": ">", "value": 100000}]}) |
| 4 backend validation | compute_metric: INVALID compute_metric.filters: filters are a list | compute_metric: valid |
| plan actually used | rules; The model proposed an invalid tool call (compute_metric.filters: filters are a list); the deterministic planner was used. | model |
| 5 tool execution | none | compute_metric: ok {"value": 112.0, "top": null, "rows": 112} |
| 6 final answer | not correct | correct |

### 13. Is years of experience correlated with salary?

| | base | fine-tuned |
|---|---|---|
| 1-3 interpretation / plan | correlation({"x": "years_experience", "y": "salary"}) | correlation({"x": "years_experience", "y": "salary"}) |
| 4 backend validation | correlation: valid | correlation: valid |
| plan actually used | model | model |
| 5 tool execution | correlation: ok {"pearson_r": 0.5385831908740764, "pearson_p": 1.8712571300581867e-19} | correlation: ok {"pearson_r": 0.5385831908740764, "pearson_p": 1.8712571300581867e-19} |
| 6 final answer | correct | correct |

### 23. Is the difference between these areas statistically significant?

| | base | fine-tuned |
|---|---|---|
| 1-3 interpretation / plan | compare_groups({"outcome": "conversions", "group": "area", "filters": null}) | compare_groups({"outcome": "conversions", "group": "area"}) |
| 4 backend validation | compare_groups: valid | compare_groups: valid |
| plan actually used | model | model |
| 5 tool execution | compare_groups: ok {"name": "Welch's one-way test", "statistic": 10.390987685087785, "df": [4.0, 9.072010932500774], "p_value": 0.0019362485762162251, "eta_squared": 0.44781167349 | compare_groups: ok {"name": "Welch's one-way test", "statistic": 10.390987685087785, "df": [4.0, 9.072010932500774], "p_value": 0.0019362485762162251, "eta_squared": 0.44781167349 |
| 6 final answer | correct | correct |

### 24. Show ROI and CAC by area

| | base | fine-tuned |
|---|---|---|
| 1-3 interpretation / plan | business_performance({"group": "area", "focus": "roi"}) | business_performance({"group": "area"}) |
| 4 backend validation | business_performance: valid | business_performance: valid |
| plan actually used | model | model |
| 5 tool execution | business_performance: refused {"ranking": null, "focus": null} | business_performance: ok {"ranking": ["Area A", "Area B", "Area C", "Area D"], "focus": "Area A"} |
| 6 final answer | correct | correct |

## Regression check on the 20 questions the deterministic planner passed

| # | question | base | fine-tuned |
|---:|---|---|---|
| 1 | What is the average grade points? | pass | pass |
| 2 | Which campus has the highest average grade points? | **fail** | pass |
| 3 | Is there a significant difference in grade points between campus groups? | pass | pass |
| 4 | Predict passed | **fail** | pass |
| 5 | Which factors influenced your prediction? | pass | pass |
| 6 | What is the median price in each city? | **fail** | pass |
| 7 | Predict price | **fail** | pass |
| 9 | Which area should receive our next advertising campaign? | **fail** | pass |
| 10 | What is the average salary in Engineering? | **fail** | pass |
| 12 | Show the total salary by department | **fail** | pass |
| 14 | What is the standard deviation of salary? | **fail** | pass |
| 15 | Predict customer churn. | **fail** | **fail** |
| 16 | Which factors influenced your prediction? | pass | pass |
| 17 | Predict salary. | **fail** | pass |
| 18 | Should we advertise in Area A? | **fail** | pass |
| 19 | Which area should receive the next campaign? | **fail** | **fail** |
| 20 | Why do you recommend Area A? | pass | pass |
| 21 | Should we increase the budget? | **fail** | **fail** |
| 22 | What happens if the budget increases by 20%? | pass | pass |
| 25 | Should we advertise in the Tools category? | pass | pass |
