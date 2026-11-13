# Browser end-to-end test: AI Analyst with the v1 adapter

The test ran on 2026-11-11 in Chrome with dark mode on.

**Stack** (all local):
- `mlx_lm.server` on port 8080, serving Qwen3-4B-Instruct-2507-4bit plus `training/adapters/analyst_qwen3_4b_lora` (v1).
- A scratch FastAPI backend on port 8010, using a new SQLite database and data folder in the session scratchpad.
- Streamlit on port 8510.

Production `backend/data/app.db` was not touched. Screenshots are in the session scratchpad (`ui_e2e_v1/screenshots/`).

"Planner" below is what the page's "How this was answered" section reported.

| # | step | result | planner | notes |
|---:|---|---|---|---|
| 1 | sign-up, community and team | pass | – | A new account, the "Demo Community" community and the "Marketing Analytics" team. Opening a page by its URL starts a new Streamlit session and asks for login again; login works. |
| 2 | upload CSV campaign data | pass | – | campaigns.csv: 58 rows, 8 columns, Source "CSV". |
| 3 | upload Excel employee data | pass | – | employees.xlsx: 240 rows, 7 columns, Source **"Excel"**. |
| 4 | conversational question | pass | model, first pass | Describes its capabilities; no tools run. |
| 5 | aggregation ("average salary by department") | pass | model, first pass | All 5 means equal the pandas values (for example Engineering 115,055.77). |
| 6 | significance test | pass, with the model's summary blocked | model, first pass | Welch ANOVA p < 0.001, η² = 0.49, Kruskal–Wallis check. **The model's prose summary cited 1.20 and 23, which are not in the results. The grounding check withheld it and said so; only the computed table was shown.** |
| 7a | train a Random Forest ("Predict salary") | pass | model, first pass | R² 0.82 on 60 held-out rows; MAE 6,752 against a baseline of 16,668; permutation importance; the identifier column was excluded. |
| 7b | use the model with inputs supplied | pass | model, first pass | Predicted 108,319.47 using exactly the values in the message. The inputs are displayed as a raw Python dict (cosmetic). |
| 8 | "Which area should receive our next advertising campaign?" | pass, **after reconsideration** | **model after one reconsideration** | The first pass falsely declined ("no advertising history"). The contradiction check caught it, the model reconsidered once, and the page says so. Every ROI and CAC value equals the pandas values. |
| 9 | "Why do you recommend that area?" | pass | model, first pass | An 8-part explanation: question, data, method, results, statistics, uncertainty, rule applied, limitations. |
| 10 | prediction with no inputs ("salary for a new employee") | **model failure, caught** | **deterministic planner** | **The model invented a prediction row (department "Engineering").** The supplied-input check rejected it and said so. The fallback then asked "Which column should I predict?", which is a poor question because salary had already been named. Nothing was predicted from invented values. |
| 11 | decline when required columns are missing | pass | model, first pass | An advertising question on employees: "The available data is not sufficient…". "Predict churn" on employees: "There is no churn column. Which column should I predict?" with column options. |
| 12 | two consecutive messages | pass, with a limitation | – | Earlier answers are kept: 2–4 turns each, on two datasets. **A message typed while the previous one is still being analysed is not accepted.** The input is disabled and shows "Analysing the previous question…", and what was typed is discarded, not queued. |
| 13 | dark-mode visibility | **3 defects found and fixed during the test** | – | See below. After the fixes, a contrast scan of every visible text element on Home, Datasets, Analyses, Copilot and AI Analyst finds nothing under 3:1 that dark mode causes. The remaining items are the theme's own muted grey labels ("Workspace", 2.56:1, the same in light mode) and the hidden sidebar-collapse icon. |
| 14 | Excel label | pass | – | Shows "Excel", not "CSV". |
| extra | budget scenario +20% | pass | model, first pass | 331,005.04 × 1.2 = 397,206.05 and revenue 506,945.28, both checked; labelled "not a prediction". |
| extra | "Show ROI and CAC by area" (#24, which the rules miss) | pass | model, first pass | The ranking line names Areas A–D and leaves out Area E (2 rows), which is in the table. Minor. |

## Defects found by the browser test and fixed (UI only)

1. **Unreadable code chips** in dark mode (column options in answers): text #1f2330 on background #1a1c24, about 1.1:1. Fixed with a light background for `code` and `pre` inside chat messages and expanders (`frontend/style/theme.py`).
2. **Unreadable open "How this was answered" header** in dark mode (dark text on a dark summary bar). Fixed with a light summary background (`frontend/style/theme.py`).
3. **Unreadable unselected tab labels** ("Lessons" on Datasets: near-white on white). Fixed with a scoped tab-label colour (`frontend/style/theme.py`).
4. **Leaving AI Analyst and returning reset the dataset to the first one and cleared the conversation.** Streamlit discards widget state for pages that aren't shown. The chosen dataset is now remembered in session state (`frontend/pages/18_AI_Analyst.py`). Re-tested: the dataset and the conversation survive navigation.

Frontend tests after the fixes: 290 passed.

## Remaining defects (not fixed)

- The v1 model invents prediction inputs when none are given. The input check blocks it every time, but the fallback clarification ("Which column should I predict?") is the wrong question; it should ask for the input values.
- On its first pass, v1 falsely declines the main advertising question on the campaign data. It is recovered by one transparent reconsideration (about 11 s, against about 5 s normally).
- The model's narrative summaries sometimes cite numbers that are not in the results. They are withheld, and only the deterministic text is shown.
- A message typed while another is being analysed is discarded, not queued.
- Cosmetic: the prediction inputs are shown as a raw Python dict; the ranking line omits Area E; the Datasets upload card says "Upload a CSV" although it accepts Excel; the collapsed JSON viewer bar stays dark (its text is legible).
