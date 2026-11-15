# Silt — Complete Under-the-Hood Technical Report

*An audit of the actual repository, written to teach a first-year B.Tech AI/ML student how every part works.*

Companion documents (nothing is repeated between them):

| File | Contents |
|---|---|
| `docs/SILT_COMPLETE_UNDER_THE_HOOD.md` (this file) | Parts 0–15, 17 and the glossary: how everything works, with source references |
| `docs/SILT_ARCHITECTURE_DIAGRAMS.md` | Part 19: every architecture and execution diagram (Mermaid) |
| `docs/SILT_PROFESSOR_VIVA_GUIDE.md` | Part 20: 60+ viva questions with answers, and the 2/5/10-minute explanations |
| `docs/SILT_KNOWN_LIMITATIONS_AND_ROADMAP.md` | Parts 16 and 18: the bug audit, root causes and recommended fixes |

---

## Part 0 — How to read this report

### Evidence labels

Every claim carries one of these labels, so you always know how sure it is:

| Label | Meaning |
|---|---|
| **[CODE]** | Read in the source code. A file and function (and line numbers) are given. |
| **[TEST]** | Verified by an automated test or a run performed during the audit. |
| **[RUN]** | Observed by running the code read-only during this audit (model off; no file, database or setting changed). |
| **[INFERRED]** | Follows from the architecture or from library behaviour, but not checked line by line. |
| **[THEORY]** | General machine-learning or statistics background — true in general, not specific to Silt. |
| **[GAP]** | Missing, incomplete or broken. |
| **NOT VERIFIED** | Could not be confirmed from the repository. |

### Scope of the audit

The repository contains **two analysis systems** that share one backend, database and frontend:

1. **The Analyst Copilot** (`backend/app/analyst/`, page `frontend/pages/12_Copilot.py`). This is the conversational analyst you demonstrate, with the local Qwen model, the analysis programs, investigation, forecasting and scenarios. **This report concentrates on it.**
2. **The legacy multi-agent platform** (`backend/app/agents/`, routes `/api/questions/...`). This is an earlier LangGraph pipeline (router → triage → Planner → Executor → Critic) that used hosted models (Gemini/Llama) and a code sandbox. Its Planner/Executor/Critic files are *protected*: they were not modified during the Copilot work. It is inventoried in Part 1 but not taught in depth.

Line numbers refer to the working tree on branch `feature/analyst-reliability` at the time of the audit. Much of the Copilot code is **uncommitted** work (see Part 15), so line numbers can shift once it is committed.

### What the audit did not do

No source code, model weights, adapters, databases, environment variables, `.env` files or Git state were changed. The only new files are the four documents in `docs/`. Everything run during the audit was read-only and in memory, with the model off.

---

## Part 1 — Complete project inventory

### 1.1 Directory map (only what matters)

```
ai-data-analyst/
├── backend/                     FastAPI service (Python 3.13 locally; Docker image uses 3.11)
│   ├── app/main.py              API entry point: FastAPI app, dataset upload, legacy question routes
│   ├── app/config.py            environment variables (paths, DB URL, JWT, model keys)
│   ├── app/database.py          SQLAlchemy engine (SQLite by default, Postgres if DATABASE_URL is set)
│   ├── app/models.py            42 SQLAlchemy tables
│   ├── app/security.py          bcrypt passwords + JWT tokens
│   ├── app/ingest.py            reading CSV/Excel correctly (encoding, delimiter, header, numbers, dates)
│   ├── app/routers/analyst.py   the Analyst Copilot API (/api/analyst/...)
│   ├── app/analyst/             THE ANALYST COPILOT ENGINE (≈11,000 lines)
│   │   ├── orchestrator.py      decides which path answers a message (1,905 lines)
│   │   ├── conversation.py      pending clarifications, "yes", corrections, topic changes
│   │   ├── scope.py             filters named in the question (people, groups, months, exclusions)
│   │   ├── semantics.py         what each column *is* (identifier, amount, rate, flag ...)
│   │   ├── intent.py            what a request asks for (quantity, measure, grouping, denominator ...)
│   │   ├── program.py           the analysis-program language and its deterministic executor
│   │   ├── nl_planner.py        model-assisted program planning (Qwen writes JSON; Silt checks and runs it)
│   │   ├── objectives.py        target feasibility, absence scenarios, "compare with the rest"
│   │   ├── investigate.py       autonomous dataset investigation
│   │   ├── associations.py      Spearman / eta² / Cramér's V with Holm correction
│   │   ├── metrics.py           standard business metrics (rates, shares, durations, costs)
│   │   ├── calc.py              a safe calculator language, and the model's calculation planner
│   │   ├── planner.py           the v1 "tool" planner prompt, model client, deterministic rule planner
│   │   ├── tools.py             the tool catalogue (compute_metric, compare_groups, random_forest ...)
│   │   ├── ml.py                Random Forest training and evaluation
│   │   ├── followup.py          forecasting, trends, charts of previous answers, explanations
│   │   ├── documents.py         PDF text extraction and keyword search
│   │   └── data.py, explain.py, present.py, router.py, intake.py, overview.py, plan_checks.py ...
│   ├── app/agents/              legacy LangGraph pipeline (PROTECTED: Planner/Executor/Critic)
│   └── tests/                   51 test files (16 of them test the analyst)
├── frontend/                    Streamlit app
│   ├── Home.py                  frontend entry point (login, navigation)
│   ├── api_client.py            every HTTP call to the backend
│   ├── pages/12_Copilot.py      the Copilot chat page
│   └── style/copilot_ui.py      chart drawing (Altair), typing indicator
├── training/
│   ├── scripts/                 dataset builders, evaluation scripts, serve_analyst_model.sh
│   ├── data/analyst_sft*/       fine-tuning data (JSONL) for v1, v2, v3
│   ├── adapters/                three LoRA adapters (v1 active)
│   └── reports/                 evaluation reports for the adapters
├── Dockerfile, start.sh, render.yaml   hosted deployment (Render, Docker)
├── docs/                        documentation (this report and older design notes)
└── web/                         an untracked Next.js project — NOT VERIFIED, not part of this audit
```

### 1.2 Component inventory

| Component | File | Responsibility | Inputs | Outputs | Dependencies |
|---|---|---|---|---|---|
| Copilot page | `frontend/pages/12_Copilot.py` | Chat UI, question queue, dataset selector, answer display | Typed question, chosen dataset/document | HTTP request; rendered markdown and charts | Streamlit, `api_client`, `style/copilot_ui` |
| API client | `frontend/api_client.py` (`analyst_ask`, l. 770) | Sends requests with the login token | Question, dataset id, conversation id | JSON response | `requests` |
| Analyst API | `backend/app/routers/analyst.py` (`ask`, l. 88) | Validates the request, loads the dataset, picks the conversation memory | `AskIn` JSON | Answer JSON | FastAPI, SQLAlchemy, orchestrator |
| Orchestrator | `backend/app/analyst/orchestrator.py` (`ask`, l. 49; `_dispatch_turn`, l. 125) | Routes a message to the right path; checks the scope; updates memory | Message, DataFrame, memory | Answer dict | Every analyst module |
| Conversation state | `conversation.py` (`resolve` l. 197, `record` l. 108) | Pending clarifications, confirmations, corrections | Message, memory | An "effective message" or a reply | — |
| Scope | `scope.py` (`establish` l. 99, `mask` l. 178, `eligibility` l. 251) | Turns named values/periods into row filters; marks invalid records | Message, DataFrame, memory | Scope dict, boolean row mask | pandas |
| Semantics | `semantics.py` (`column_role` l. 60, `profile` l. 127) | Classifies columns by role | DataFrame | Roles with evidence | pandas |
| Intent | `intent.py` (`read` l. 111, `check` l. 279) | Reads what a request asks for; checks programs against it | Message, DataFrame | `Request`; list of problems | semantics |
| Program engine | `program.py` (`run` l. 450, `op_*`) | Executes typed analysis steps | Step list, DataFrame | `Result` objects with provenance | pandas, numpy, calc, associations |
| Model planner | `nl_planner.py` (`plan` l. 841, `check` l. 356, `execute` l. 971) | Asks Qwen for a program; validates and repairs it | Message, DataFrame, memory | A checked program, clarification or reference | OpenAI-compatible client, program, intent |
| Objectives | `objectives.py` (`handle` l. 460) | Feasibility, absence, comparing scenarios | Message, DataFrame, memory | A program and its presented answer | program, metrics |
| Investigation | `investigate.py` (`plan` l. 38, `investigate` l. 135) | Picks analyses for an unknown dataset; ranks findings | DataFrame | Findings with evidence | program, semantics, associations |
| Calculator | `calc.py` (`evaluate` l. 389, `run` l. 713, `model_plan` l. 1034) | Safe expression language; model-written calculation plans | Expressions / question | Values, tables | Python `ast`, pandas |
| v1 tool planner | `planner.py` (`build_prompt` l. 72, `model_chat` l. 84, `rule_plan` l. 136) | Asks the fine-tuned model to choose a tool; deterministic fallback | Message, schema | Tool-call plan | OpenAI client |
| Tools | `tools.py` | compute_metric, compare_groups (Welch tests), correlation, random_forest, ... | Tool name + args | Results | pandas, scipy, statsmodels-style helpers in `app/stats` |
| ML | `ml.py` (`train` l. 150) | Random Forest classification/regression with honest evaluation | DataFrame, target | Metrics, importances, fitted model | scikit-learn |
| Forecast | `followup.py` (`forecast` l. 501) | Next-period forecast (linear trend vs naive baseline) | Measure over time | Forecast dict | scikit-learn |
| Documents | `documents.py` | PDF text extraction (pypdf) and BM25 search | PDF bytes, question | Passages with page numbers | pypdf |
| Ingestion | `backend/app/ingest.py` (`read_table` l. 252) | Reads CSV/Excel robustly and reports how | File path | DataFrame + read report | pandas, csv, openpyxl |
| Upload endpoint | `backend/app/main.py` (`upload_dataset` l. 257, `_save_and_profile_upload` l. 166) | Saves an upload, validates it, records a Dataset row | Multipart file | Dataset profile | ingest, SQLAlchemy |
| Model server | `training/scripts/serve_analyst_model.sh` | Serves Qwen3-4B + LoRA adapter over HTTP | — | OpenAI-compatible `/v1/...` endpoints on port 8080 | `mlx_lm` (Apple Silicon only) |
| Training | `mlx_lm.lora` with configs in `training/adapters/*/adapter_config.json` | Fine-tuned the adapters (done in the past; not run now) | JSONL chat examples | `adapters.safetensors` | `mlx_lm` |
| Evaluation | `training/scripts/eval_program_planner.py`, `eval_analyst_model.py`, `eval_calc_v3.py` | Measure planners and adapters | Test cases | JSON/Markdown reports | the app itself |

### 1.3 Entry points

| Entry point | Where | Verified how |
|---|---|---|
| Backend application | `backend/app/main.py`, `app = FastAPI(title="AI Data Analyst API")` (l. 75); routers included l. 90–110; `init_db()` on startup (l. 113–116) | [CODE] |
| Backend process | `uvicorn app.main:app --host 127.0.0.1 --port 8000` (`start.sh` l. 23 in Docker; the same command locally) | [CODE] |
| Frontend | `frontend/Home.py` (Streamlit multipage app); the Copilot is `frontend/pages/12_Copilot.py` | [CODE] |
| Model server | `training/scripts/serve_analyst_model.sh` → `python -m mlx_lm server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path training/adapters/analyst_qwen3_4b_lora --port 8080` | [CODE] |
| Data ingestion | `POST /api/datasets/upload` → `_save_and_profile_upload` → `ingest.read_table` | [CODE] |
| Analytical execution | `POST /api/analyst/ask` → `orchestrator.ask` → `_dispatch_turn` → a path → (often) `program.run` | [CODE] |

### 1.4 How the components communicate

1. **Browser ↔ Streamlit**: the student's browser talks only to the Streamlit server (port 8501). Streamlit re-runs the page script on every interaction. [CODE/THEORY]
2. **Streamlit → FastAPI**: plain HTTP with `requests`, server to server, carrying a `Bearer` JWT (`api_client.py` l. 83). Because the call happens server-side, the browser never calls the API directly (the comment above the CORS middleware, `main.py` l. 77–82, says the same). [CODE]
3. **FastAPI → model server**: HTTP to `http://127.0.0.1:8080/v1` (`planner.MODEL_URL`, `planner.py` l. 26). Chat requests use the OpenAI client library (`planner.model_chat` l. 84, `nl_planner._chat` l. 819). The calculator uses the raw `/completions` endpoint with a hand-built chat template (`calc._complete` l. 958). [CODE]
4. **FastAPI → database**: SQLAlchemy sessions (`database.get_db`). Datasets, users and teams are stored there. **The Copilot conversation memory is not** (see Part 13). [CODE]
5. **FastAPI → files**: uploaded datasets are stored under `DATA_DIR/uploads` (`config.py` l. 16–17); PDFs under `DATA_DIR/analyst_documents/<team>/` (`documents.py` l. 22). [CODE]

---

## Part 2 — What exactly is Silt?

### 2.1 What it is

Silt is a **conversational data analyst**. You upload a table (CSV or Excel) or a PDF, then ask questions in plain English. Silt answers with numbers, tables and charts, plus an explanation of how it calculated them.

### 2.2 The problem it solves

Analysing a spreadsheet normally needs either a person who knows pandas/SQL or a dashboard someone built in advance. Silt lets a non-programmer ask a new question — "Can Rahul complete 300 working hours this month?" — and get a correct, explained answer without writing code.

### 2.3 What happens on upload (short version — details in Part 6)

The file is streamed to disk and read by `ingest.read_table`, which detects the encoding, delimiter, decimal mark, header row, numbers stored as text and dates. If the file can't be read reliably it is refused, with the reason. Otherwise a `Dataset` row is saved with a profile of the file. [CODE: `main.py` l. 166–254]

### 2.4 What happens when a question is asked (short version — details in Part 3)

The orchestrator first tries **deterministic** paths, i.e. ordinary Python with fixed rules. If none fits and the model is available, the **local Qwen model** is asked to write a *plan*: a JSON object, never a number. Silt validates the plan, runs it with pandas, and writes the answer from the computed results.

### 2.5 Why it is not an ordinary chatbot

A chatbot such as ChatGPT writes the answer itself, token by token, and can invent numbers ("hallucinate"). In Silt the model **never produces a number that is shown as a result**:
- Numbers come from pandas.
- Any prose the model writes is checked so that it contains only numbers that appear in the computed results (`explain.ungrounded`, `explain.py` l. 92). Where Silt builds the answer from a program, it writes the labels itself (`nl_planner._label` l. 1036).

### 2.6 Why it is not an ordinary dashboard

A dashboard shows charts someone designed in advance. Silt builds the analysis **on demand** from the question: it chooses the filters, measures, groupings and methods, and explains them.

### 2.7 Why combine a language model with deterministic Python

| Job | Done by | Why |
|---|---|---|
| Understanding unfamiliar wording ("turns visitors into takings") | Language model (when rules fail) | Language is too varied for hand-written rules alone |
| Arithmetic, filtering, statistics | Python/pandas/scipy/scikit-learn | Exact, repeatable, testable; a 4-billion-parameter model is unreliable at arithmetic |
| Deciding whether a plan is valid and means what was asked | Python checks (`nl_planner.check`, `intent.check`) | The model can propose anything; only checked plans run |

### 2.8 The six kinds of "intelligence", and what Silt really implements

| Kind | What it means | In Silt | Status |
|---|---|---|---|
| LLM-based reasoning | A neural language model interprets text and proposes an output | Qwen3-4B plans tool calls (`planner.py`), calculations (`calc.py`) and analysis programs (`nl_planner.py`); optionally words summaries | **Implemented**, used only when deterministic paths cannot answer. Planning accuracy is limited (Part 17) |
| Deterministic programming | Fixed rules: the same input always gives the same output | Routing (`orchestrator.py`), filters (`scope.py`), the program executor (`program.py`) | **Implemented**: this is the core |
| Statistical analysis | Summaries and tests with known mathematical properties | Means, medians, Welch t-test/ANOVA, Kruskal–Wallis, Spearman, Cramér's V, eta², Holm correction, IQR outliers | **Implemented** (Part 11) |
| Machine learning | A model learns patterns from examples to predict new cases | Random Forest classification/regression (`ml.py`); linear-trend forecasting (`followup.py`) | **Implemented** with baselines and validation (Part 10) |
| Predictive analytics | Estimating future or unknown values | Next-period forecast; target feasibility (a pace projection, *not* ML); Random Forest predictions for one case | **Partially**: no seasonality, single-series forecasts only |
| Autonomous investigation | The system decides what to analyse | `investigate.py` picks analyses from column roles and ranks findings | **Implemented**, rule-based (no model involved) |

---

## Part 3 — The complete request lifecycle

This is the path of one question, e.g. *"Which department has the highest attrition and why?"*, from the Send button to the rendered answer. Every step names its file and function. The sequence diagram is in `SILT_ARCHITECTURE_DIAGRAMS.md` §2.

### 3.1 Step by step

| # | What happens | Where (file : function, lines) | Evidence |
|---|---|---|---|
| 1 | The input box is Streamlit's `st.chat_input`; its placeholder changes while Silt is busy. | `12_Copilot.py`, l. 297–299 | [CODE] |
| 2 | Pressing Enter/Send makes `st.chat_input` return the text, and **Streamlit re-runs the whole page script** from the top. There is no JavaScript callback: Streamlit's model is "re-run the script on every interaction". | `12_Copilot.py` l. 300; Streamlit behaviour | [CODE]/[THEORY] |
| 3 | `enqueue(conv, q)` appends `{"q": text, "r": None}` to the conversation's log (identical waiting questions are not added twice), then `st.rerun()`. | `12_Copilot.py` `enqueue`, l. 67–77, 300–302 | [CODE] |
| 4 | The message is stored in **Streamlit session state** (`S["cp_convs"][conversation_id]["log"]`). The conversation ID is a random 12-hex-character string from `uuid.uuid4()` created by `new_conv`. | `12_Copilot.py` `new_conv`, l. 59–64 | [CODE] |
| 5 | On the re-run, the first unanswered question is taken, a typing indicator ("Thinking…", or "Training and testing a model" if the question mentions train/forecast/predict) is shown, and `analyst_ask(dataset_id, question, conversation_id, document_id=...)` is called. | `12_Copilot.py` l. 304–320; `copilot_ui.typing_html` l. 142 | [CODE] |
| 6 | `analyst_ask` POSTs JSON `{dataset_id, document_id, message, conversation_id, use_model}` to **`/api/analyst/ask`** with a 300-second timeout. | `api_client.py` l. 770–773 | [CODE] |
| 7 | FastAPI validates the body against the Pydantic model `AskIn`: message 1–1000 characters, conversation_id ≤ 64 characters. `get_current_user` and `get_current_team` check the JWT and the workspace. A request with neither dataset nor document gets HTTP 400. | `routers/analyst.py` `AskIn` l. 23–28, `ask` l. 88–92; `security.py` | [CODE] |
| 8 | The dataset's latest version is read from disk into a pandas DataFrame with `load_table` (Part 6). | `routers/analyst.py` `_frame` l. 31–38; `data.py` `load_table` l. 13 | [CODE] |
| 9 | The **conversation memory** is fetched: `orchestrator.memory_for("team:user:dataset:document:conversation")`, a Python dict kept in an in-process `OrderedDict` (at most 200 conversations, least recently used dropped). | `routers/analyst.py` l. 99; `orchestrator.py` l. 19–20, `memory_for` l. 28 | [CODE] |
| 10 | `orchestrator.ask` runs: it checks whether the model server answers (`planner.model_available()`, a 1.5-second ping to `/v1/models`), then **input check** (`intake.check`: is this readable text?), then the **document** path if a PDF is selected, then `conversation.resolve` (is this a "yes", an answer to Silt's question, a correction, a topic change?). | `orchestrator.py` `ask` l. 49–99; `planner.py` l. 63; `conversation.py` `resolve` l. 197 | [CODE] |
| 11 | `_dispatch` → `_dispatch_turn` tries the **deterministic paths in a fixed order**: charts → **scope** (filters named in the question; asks when a value fits two columns) → prediction inputs → investigation → concept ambiguity ("revenue" fits two columns: ask) → objectives (feasibility, absence, scenario comparison) → model-planner gate → refinement → metric catalogue → associations → comparisons → ranking → forecast → router → calculator → v1 tool planner → late model-planner hook → fallback reply. The first path that returns an answer wins. | `orchestrator.py` l. 125–398 | [CODE] |
| 12 | **When the local Qwen model is called** (only with `model_on`): (a) the program planner when the request is composite, per-unit, restricted or a follow-up on an open analysis (`nl_planner.composite`, `follow_up`), unless a fully grounded deterministic program exists; (b) the v1 calculator (`calc.model_plan`) when the question defines its own calculation; (c) the v1 tool planner (`_ask_planned`); (d) the late program-planner hook when nothing answered; (e) optional prose summaries. A per-turn budget (`MODEL_TURN_BUDGET_MS` = 12 s, l. 400) stops further model paths after a slow planner attempt. | `orchestrator.py` l. 203–212, 280–398, 817–888 | [CODE] |
| 13 | **The prompt**: a system message (the JSON contract and operation list, `nl_planner.SYSTEM` l. 67) plus a user message that is the *grounding* (columns, roles, safe summaries, derivable measures, values named in the request, date tokens, conversation state) followed by the request (`grounding` l. 175). The OpenAI client sends it as chat messages; the server applies Qwen's chat template (`<|im_start|>system … <|im_end|>`). | `nl_planner.py` l. 67–117, 175–258, 819–830 | [CODE]/[RUN] |
| 14 | **Generation**: the server tokenises the prompt, runs the transformer, and generates tokens one at a time at temperature 0 (greedy) up to 700 tokens, 20-second timeout. A repair attempt uses temperature 0.3. | `nl_planner._chat` l. 819; `plan` l. 841–889 | [CODE] |
| 15 | **JSON parsing**: think-tags and code fences are removed and the first `{…}` is decoded. If decoding fails, two one-meaning repairs are tried: a missing `{` before a step, and mismatched brackets (`balance`). An operation written as the whole reply is wrapped as a one-step program. | `nl_planner.parse` l. 306, `balance` l. 273 | [CODE]/[TEST] |
| 16 | **Validation** (`nl_planner.check`, l. 356): every step's operation and arguments are known; columns and values exist; types fit (no average of an identifier, no sum of a rate, no `count` with a column); units agree; only totals and counts are divided; every number is one the user stated or a date token; steps refer only to earlier steps. Then **semantic consistency** (`intent.check`, l. 279): the program must compute the requested quantity, measure, grouping, denominator, entities and restriction. Rejected plans get **one** repair with the reasons. | `nl_planner.py` l. 356–665; `intent.py` l. 279–348 | [CODE]/[TEST] |
| 17 | **Execution**: `nl_planner.execute` prepends a validity `select` (drops impossible records, keeps and flags exact duplicates) and calls `program.run`, which executes each step with pandas and returns `Result` objects with provenance. | `nl_planner.py` `execute` l. 971; `program.py` `run` l. 450 | [CODE] |
| 18 | **Presentation**: `render` writes the answer *from the results*: the main result, highest/lowest, the other steps, the associations table, the caveats, and a "What you asked, and where it is answered" checklist. `details` lists every step for *How this was calculated*. Charts are chart specifications (dicts), not images. | `nl_planner.py` `render` l. 1090, `_checklist`, `details`; `_charts` | [CODE] |
| 19 | **Memory update** (`_finish`, l. 517): scope verification ("records limited to the question's scope"), `memory["scope"]`, `last_analysis`, `last_answer`, `last_charts`, the program's values (`memory["planned"]`); then `conversation.record` stores the task state (done, or awaiting a clarification with its options). | `orchestrator.py` `_finish` l. 517–583; `conversation.py` `record` l. 108 | [CODE] |
| 20 | **Response**: the router returns the answer dict plus `dataset` and `document` summaries; FastAPI serialises it to JSON. | `routers/analyst.py` l. 100–104 | [CODE] |
| 21 | **Rendering**: the page stores the response in the turn, re-runs, and `show_answer` displays the markdown (tables are markdown tables), the charts via `ui.chart(spec)` → Altair, the source label (e.g. "Analysis program planned by the local model …") with the elapsed time, and an expander *How this was calculated* with details, notes and an optional technical trace. | `12_Copilot.py` `show_answer` l. 238–270, `show_charts` l. 230; `copilot_ui.chart` l. 158 | [CODE] |

### 3.2 What happens in special cases

| Situation | What actually happens | Where |
|---|---|---|
| **Model server unavailable** | `model_available()` fails its 1.5 s ping, so `model_on = False` and every model path is skipped. Composite requests can still get a deterministic program (`rule_plan`). If the server dies during a request, the exception is caught, the planner returns `status: failed`, and the deterministic paths answer. The hosted (Render) deployment has no model server at all (Part 15). | `planner.py` l. 63; `nl_planner.plan` l. 851–857; `orchestrator.py` l. 203–212 [CODE][TEST: `test_planning_failure_never_replaces_a_deterministic_answer`] |
| **Malformed JSON** | Repaired if the slip has one meaning (missing `{` before a step, mismatched brackets). Otherwise rejected and the model is shown the parse error **once**; if that also fails, deterministic paths answer. | `nl_planner.parse`; [TEST: `test_a_step_missing_its_opening_brace_is_restored`, `test_mechanical_slips_are_repaired_and_said`] |
| **Nonexistent column** | `check` reports "there is no column 'wait_time'" (with a "did you mean" hint when the name differs only in spaces or case); the plan never reaches the executor. | `nl_planner.check` `need_col` [TEST: `test_invalid_plans_never_reach_the_executor`] |
| **Unsupported operation** | "'regress' is not an operation (use …)". Rejected, then one repair. | same |
| **Ambiguous question** | Three mechanisms: (1) a value occurring in two columns makes `scope.establish` ask "which column?"; (2) a concept two columns could mean ("revenue" with `takings` and `online_sales`) is asked about *before any path guesses* (`orchestrator.py` l. 183–190); (3) the model may itself return a clarification. The options are stored as a pending task, so the reply "takings" continues the original request. | `scope.py` l. 141–145; `nl_planner.ambiguities` l. 260; `conversation.resolve` [TEST] |
| **Follow-up question** | `conversation.resolve` folds "yes"/an option/a correction into the original request. `scope.establish` carries over the previous filters when the follow-up names nothing new (`inherited`). Model-planned follow-ups can use earlier results as inputs (`@r1.total`). | `conversation.py`; `scope.py` l. 146–153; `nl_planner._inputs` l. 832 |
| **New dataset uploaded** | The memory key contains the dataset id, so a new dataset starts with empty memory, and the frontend opens a new conversation for it. | `routers/analyst.py` l. 99; `12_Copilot.py` l. 202 |
| **Fresh conversation** | "＋ New conversation" creates a new random conversation id, so a new memory key and empty memory: no scope, objectives or findings carry over. | `12_Copilot.py` l. 124–126 [TEST: `test_a_fresh_conversation_has_nothing_to_refer_to`; browser run] |

---

## Part 4 — How the model understands human language

### 4.1 What is an LLM? [THEORY, with Silt facts marked]

**A language model** is a program that, given some text, assigns a probability to every possible next piece of text. A *large* language model (LLM) is a very large neural network trained to do this on enormous amounts of text.

**Qwen** is a family of open-weight LLMs released by Alibaba Cloud. Silt uses **Qwen3-4B-Instruct-2507**:
- **Qwen3** is the generation.
- **4B** means about 4 billion **parameters**.
- **Instruct** means it was additionally trained to follow instructions and chat.
- **2507** is the July 2025 release.

Silt's copy is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, a conversion for Apple's MLX framework with weights **quantised to 4 bits** (group size 64). That means each weight is stored in 4 bits instead of 16, making the model about 4× smaller in memory at a small cost in accuracy. [CODE: `config.json` of the model, read during the audit: `"quantization": {"group_size": 64, "bits": 4}`]

**Parameters** are the numbers inside the network that training adjusts. In a tiny example y = w·x + b, the parameters are w and b. Qwen3-4B has about 4 billion of them. The fine-tuning log prints the total as 4,022.468 M (`train.log`, line 13).

**A neural network** is layers of simple computations. Each layer multiplies its input by a matrix of parameters, adds a bias and applies a non-linear function. Stacking many layers lets the network represent very complex functions.

**A transformer** is the neural-network design used by modern LLMs. Facts read from the model's `config.json` [CODE]:

| Setting | Value | Meaning |
|---|---|---|
| `architectures` | `Qwen3ForCausalLM` | a decoder-only transformer predicting the next token |
| `num_hidden_layers` | 36 | 36 transformer blocks stacked |
| `hidden_size` | 2560 | each token is represented by a vector of 2,560 numbers inside the model |
| `num_attention_heads` / `num_key_value_heads` | 32 / 8 | 32 attention heads; keys/values shared in groups (grouped-query attention, which saves memory) |
| `intermediate_size` | 9728 | width of each block's feed-forward layer |
| `vocab_size` | 151,936 | number of token IDs the model can output |
| `max_position_embeddings` | 262,144 | the longest context the architecture supports, in tokens |

**Tokens** are the pieces text is cut into, often parts of words. **Token IDs** are the integers that identify them.

**Embeddings**: each token ID is looked up in a table that turns it into a vector (here 2,560 numbers). Similar meanings end up as similar vectors.

**Attention / self-attention**: in each layer, every token computes how relevant every earlier token is to it, and mixes in information from them. In "Can **Rahul** complete 300 working hours this month? … **he** …", attention lets "he" draw information from "Rahul". Mathematically, each token makes a *query* vector q, every token a *key* k and a *value* v; the weight on token j is softmax(q·kⱼ/√d), and the output is the weighted sum of the values.

**Context window**: the maximum number of tokens the model can consider at once. The architecture allows 262,144. Silt's prompts are far smaller: the planner prompt for the timesheet example is **2,057 tokens**, the v1 tool-planner prompt **792 tokens** [RUN]. The fine-tuning used `max_seq_length` 2,400 (`adapter_config.json`).

**Logits**: at the end, the model outputs one score per vocabulary entry (151,936 scores). Softmax turns them into probabilities. Example: if the logits for the next token are {`{`: 9.1, `Sure`: 6.0, `The`: 5.2}, softmax gives about {0.94, 0.04, 0.02}.

**Next-token prediction**: generation repeats one step — compute the probabilities, pick a token, append it, repeat. **Temperature** controls the picking. At temperature 0 the most likely token is always chosen ("greedy"), which is why Silt's planner calls are repeatable. Higher temperature samples more randomly; Silt uses 0.3 for the single repair attempt so that it doesn't echo the rejected plan.

### 4.2 Tokenisation — real output from Silt's tokenizer [RUN]

The audit loaded **only** `tokenizer.json` from the model folder (no weights) with the Hugging Face `tokenizers` library:

```
"Can Rahul complete 300 working hours this month?"
tokens: ['Can', 'ĠRahul', 'Ġcomplete', 'Ġ', '3', '0', '0', 'Ġworking', 'Ġhours', 'Ġthis', 'Ġmonth', '?']
ids:    [6713, 85136, 4583, 220, 18, 15, 15, 3238, 4115, 419, 2254, 30]
decoded: 'Can Rahul complete 300 working hours this month?'

"hours_between(clock_in, clock_out)"
tokens: ['hours', '_between', '(clock', '_in', ',', 'Ġclock', '_out', ')']

"unbelievably" → ['un', 'belie', 'vably']
vocabulary size: 151,669 entries (the model's output layer has 151,936 slots)
```

What this teaches:
- **`Ġ` marks a leading space.** "Rahul" at the start of a sentence and " Rahul" after a space are different tokens.
- **One word is not one token.** "unbelievably" is three tokens; "Rahul" happens to be one.
- **Numbers are split into digits.** "300" is the tokens `3`, `0`, `0`. The model never sees "300" as a single quantity, which is one reason a language model should not do the arithmetic itself.
- **Decoding reverses the process**: the IDs map back to exactly the original text.

The tokeniser is a byte-level **BPE** (byte-pair encoding) [INFERRED from `merges.txt` in the model folder]: frequent character sequences were merged into single tokens during the tokeniser's own training.

### 4.3 How Silt builds the prompt [CODE + RUN]

Silt has **three** model prompts:

| Prompt | Used by | Format |
|---|---|---|
| v1 tool planner | `planner.build_prompt` (l. 72) → `model_chat` | System: the JSON contract and tool catalogue. User: the schema with **up to 8 values per category** (including person names), any conversation context, the question |
| Calculator | `calc.model_plan` (l. 1034) → `_complete` (raw `/completions`) | A hand-built Qwen chat template: system `CALC_SYSTEM` (l. 930); user: schema **plus the first 3 rows** (`_schema_block` l. 1030), the previous calculations, the question. The answer is generated section by section (type, derive, exclude, group_by, aggregate), each validated |
| Program planner | `nl_planner.plan` (l. 841) | System: `SYSTEM` (l. 67), the JSON contract, operations, rules, three worked examples on fictional schemas. User: `grounding(...)` + `Request: ...` |

The real grounding for the showcase question on the demo timesheet [RUN]:

```
Dataset: 11 records, 7 columns.
Columns:
- employee_id (identifier; 7 distinct values (not listed))
- employee_name (person name; 7 distinct values (not listed))
- department (category; values: Engineering, HR, Finance, Marketing)
- date (datetime; from 2026-09-01 to 2026-12-01)
- clock_in (time of day)
- clock_out (time of day)
- hourly_rate (rate; range -10 to 30, median 25)
Measures that can be derived (use a derive step): shift duration: name shift_hours, formula hours_between(clock_in, clock_out)
Values named in the request: 'Rahul Sharma' is a value of employee_name
Date tokens (today is 2026-11-14): $today=2026-11-14, ..., $month_start=2026-11-01, $month_end=2026-11-30, ..., $days_left_month=16, ...

Request: Can Rahul complete 300 working hours this month?
```

Mapping this to the brief's checklist:

| Prompt element | Present? | Where |
|---|---|---|
| System instructions | yes | `SYSTEM` l. 67 |
| User message | yes | `Request: …` |
| Dataset schema | yes | the Columns block |
| Semantic roles | yes | the word after each column (identifier, person name, rate …) from `semantics.profile` |
| Safe summaries | yes | category values (≤ 15), numeric range and median, date range, missing counts. **Person names, IDs and free text are not listed**; only names the user typed are confirmed |
| Conversation history | partly | `conversation_context` (l. 235): current scope, numbered earlier analyses (marking the open objective), earlier programs' values (`@r1.base`), findings, the last question. **Not** the full chat transcript |
| Previous findings | yes | "finding N: title (columns …)" |
| Named scenarios | yes | "analysis N: feasibility {target, entity …}" |
| Available operations | yes | in `SYSTEM` |
| Output format | yes | the six JSON actions in `SYSTEM` |

After chat templating, the text the model reads starts `<|im_start|>system\n…` and ends `…Request: …<|im_end|>\n<|im_start|>assistant\n`. The model continues from that last line. [RUN, rendered with the model's own chat template]

**A finding from the audit [GAP, privacy]**: the v1 tool-planner prompt lists the category values, so on the timesheet it includes all seven employee names. The calculator prompt includes the first three data rows. Everything stays on the local machine (the model runs locally), but the program planner was deliberately designed to send less.

### 4.4 How "Can Rahul complete 300 working hours this month?" is actually handled

**Important: in the current code this question never reaches Qwen.** [RUN, model off; also with the model on in the browser, `MODEL CALLS 0`]

| Step | Who | What happens |
|---|---|---|
| 1 Tokenisation | — | Not needed: no model call |
| 2 Context | Silt | `scope.establish` finds "Rahul" as a word of exactly one value, "Rahul Sharma", in `employee_name` (`calc.named_values`, l. 1205), so scope = {employee_name: [Rahul Sharma]} |
| 3 Intent | Silt (rules) | `objectives.handle` (l. 460): `_FEAS` matches "can … complete"; `_target` (l. 66) finds 300 with unit "hours" |
| 4 Entity | Silt | from the scope (step 2), via `_entity` (l. 143) |
| 5 Operations | Silt | `_measure` (l. 109): "hours" → the catalogue metric *shift duration*, which isn't a column, so it is derived as `hours_between(clock_in, clock_out)`. `_period` (l. 78): "this month"; the data has no November records, so the latest month with records (Sept 2026) is used and **said** |
| 6 Program | Silt | `feasibility_program` (l. 177) builds select → derive → select period → sum → project → compare |
| 7 Calculation | pandas | see Part 10.3 for the numbers |
| 8 Validation | Silt | `program.validate` plus validity rules (`scope.eligibility`): the exact duplicate (file line 12) is not counted |

**What the model would do if it were asked** (e.g. for an unusual phrasing that no rule matches) [CODE + THEORY]:
1. The server tokenises the 2,057-token prompt.
2. Through attention, the request tokens connect to the grounding: "Rahul" to `'Rahul Sharma' is a value of employee_name`, "hours" to `shift duration … hours_between(clock_in, clock_out)`, "this month" to `$month_start` and `$days_left_month`.
3. Because of the worked example in `SYSTEM` (the "North shop … 5000 … quarter" example, which has the same shape), the most probable continuation is a program: select the entity, derive hours, select month-to-date, sum, project, compare against the target 300.
4. Silt validates it: the target 300 must appear in the request, dates must be tokens, the columns must exist, and the program must actually compute the requested quantity.

**Why the model never calculates the answer**: (a) it sees "300" as three digit tokens and has no calculator inside; (b) it is never shown the data rows by this planner, only summaries; (c) a plan can be checked, but a number written by a model cannot be verified without recomputing it anyway.

---

## Part 5 — Model training: everything explained

### 5.1 Pretraining [THEORY] — and what Silt did

**Pretraining** teaches a network language from scratch by next-token prediction on a huge text collection (Qwen's developers report trillions of tokens; NOT VERIFIED from this repository). For each position, the model predicts the next token and is scored by **cross-entropy loss**: loss = −log(probability the model gave to the correct next token). If it gave the correct token probability 0.9, the loss is 0.105; if 0.01, the loss is 4.6.

Core concepts:
- **Training example**: one piece of text (here, one chat conversation).
- **Epoch**: one full pass over the training set.
- **Batch size**: how many examples are processed before one weight update (Silt used 4).
- **Learning rate**: how big each update step is (Silt used 0.0001).
- **Gradient**: for every parameter, how much the loss would change if that parameter changed slightly.
- **Backpropagation**: the algorithm that computes all gradients efficiently, layer by layer from the output back to the input (the chain rule of calculus).
- **Gradient descent**: move every parameter a little against its gradient: w ← w − learning_rate × gradient. Example: if w = 0.50, gradient = 2.0, learning rate = 0.01, then the new w = 0.48.
- **Optimiser**: the rule that makes those updates. Silt used **Adam**, which keeps running averages of past gradients to adapt the step size per parameter.
- **Training/validation data**: the model learns from the training set. The validation set is held aside and only measured, to see whether learning generalises.
- **Overfitting**: training loss keeps falling while validation loss rises; the model memorises examples instead of learning the pattern. **Underfitting**: both losses stay high, so the model hasn't learned enough.

**Silt did not pretrain anything.** It uses the already-pretrained, instruction-tuned Qwen3-4B-Instruct-2507 (4-bit MLX build) and only fine-tuned small LoRA adapters on top. [CODE: `adapter_config.json` `"model": "mlx-community/Qwen3-4B-Instruct-2507-4bit"`, `"fine_tune_type": "lora"`]

### 5.2 Fine-tuning

**Fine-tuning** continues training a pretrained model on a small, task-specific dataset, so that it learns a behaviour (here: answering with Silt's JSON plans) without relearning language. **Supervised fine-tuning (SFT)** means every example contains the input *and the desired output*.

Silt's training examples are **chat conversations in JSONL** (one JSON object per line). A real v1 example [CODE: `training/data/analyst_sft/train.jsonl`], with the 2,323-character system prompt shortened:

```json
{"messages": [
 {"role": "system", "content": "You are Silt's analyst copilot. You plan; the backend computes. Never write a number that a tool did not return.\nReply with ONE JSON object and nothing else, one of: ... Use only these tools and only column names from the schema: - describe_dataset() ..."},
 {"role": "user", "content": "Dataset schema:\n60 rows\n- order_id (numeric)\n- outlet (categorical, 4 unique) values: Mall, Harbor, Airport, Downtown\n- category (categorical, 4 unique) values: Garden, Toys, Books, Kitchen\n- units (numeric)\n- price (numeric)\n- discount_pct (numeric)\n- total_sales (numeric)\n- returned (categorical, 2 unique) values: No, Yes\nQuestion: How many rows are there?"},
 {"role": "assistant", "content": "{\"type\": \"tools\", \"intent\": \"metric\", \"calls\": [{\"tool\": \"compute_metric\", \"args\": {\"aggregation\": \"count\"}}]}"}
]}
```

**What the model learns**: given *this* system prompt and a schema plus question, produce *this kind* of JSON. Because the configuration sets `"mask_prompt": true`, **the loss is computed only on the assistant's tokens** [CODE]. The model is trained to produce the answer, not to reproduce the prompt.

**How the loss is calculated** [THEORY]: for each assistant token, the cross-entropy is computed (−log p(correct token)) and averaged. Backpropagation computes gradients, and Adam updates the trainable parameters. In LoRA only the adapter's parameters change (5.3).

### 5.3 LoRA from scratch

**The problem**: full fine-tuning would update all 4 billion parameters. That needs gradients and optimiser state for each one (several times the model's memory) and produces a full new copy of the model. That isn't feasible on a 16 GB laptop.

**The idea (Low-Rank Adaptation)**: freeze the original weight matrix W (size d × k) and learn a *small correction* ΔW = B·A, where A is r × k and B is d × r with a small **rank** r. During inference the layer computes y = W·x + (α/r)·B·A·x.

**A matrix example**: let W be 4 × 4 (16 numbers). With rank r = 1, A is 1 × 4 and B is 4 × 1, so 8 numbers in total:

```
A = [1, 0, 2, 0]           B = [0.5, 0, 0, 1]ᵀ
B·A = [[0.5, 0, 1, 0],
       [0,   0, 0, 0],
       [0,   0, 0, 0],
       [1,   0, 2, 0]]      ← a full 4×4 correction made from only 8 numbers
```

For a real 2,560 × 2,560 matrix (6.5 million numbers), rank 8 needs 2 × 2,560 × 8 = 40,960 numbers, which is 0.6%.

**Analogy**: the pretrained model is a printed textbook; LoRA adds a small set of sticky notes. The book (W) never changes, the notes (A, B) are cheap to write, and you can swap sets of notes (v1, v2, v3) without reprinting the book.

**Silt's LoRA settings** [CODE: `training/adapters/analyst_qwen3_4b_lora/adapter_config.json`]:

| Setting | Value | Meaning |
|---|---|---|
| `fine_tune_type` | `lora` | LoRA, not full fine-tuning |
| `lora_parameters.rank` | 8 | r = 8 |
| `lora_parameters.scale` | 20.0 | the multiplier on B·A (MLX's scale setting) |
| `lora_parameters.dropout` | 0.0 | no dropout |
| `num_layers` | 16 | adapters were added to the last 16 of the 36 transformer blocks [INFERRED from mlx_lm's meaning of `num_layers`] |
| Trainable parameters | **7.340 M of 4,022.468 M = 0.182%** | printed in `train.log` (line 13) |
| `optimizer` | adam | |
| `learning_rate` | 1e-4 | constant (`lr_schedule: null`) |
| `batch_size` | 4 | |
| `iters` | 600 (v1, v2), 800 configured (v3) | |
| `max_seq_length` | 2,400 tokens | |
| `mask_prompt` | true | loss on assistant tokens only |
| `grad_checkpoint` | true | recompute activations to save memory |
| `seed` | 0 | |

The adapter file is about 29 MB (`adapters.safetensors`, 29,384,266 bytes), against a model of several gigabytes.

**During inference**, the server is started with `--adapter-path training/adapters/analyst_qwen3_4b_lora`. MLX loads the frozen base weights and the adapter, and every adapted layer adds the B·A correction. Requests for the model name `default_model` get the adapter (`planner.py` l. 27–28; `serve_analyst_model.sh`). [CODE]

### 5.4 Silt's model history (verified from files)

**Training curves** [CODE: `train.log` of each adapter]:

| Adapter | Validation loss at start → end | Notes |
|---|---|---|
| v1 | 3.457 (iter 1) → 0.074 (100) → 0.046 (200) → 0.040 (300) → 0.031 (500) → **0.033 (600)** | Training loss fell to about 0.03. Validation stayed close to training, so there's no strong overfitting on the validation split |
| v2 | 3.500 → 0.064 (100) → 0.031 (400) → 0.036 (600) | |
| v3 | 2.394 → 0.029 (200) → 0.029 (400) | **The log ends at iteration 430 of the 800 configured.** The final `adapters.safetensors` is byte-identical to the 400-iteration checkpoint (SHA-256 `274535f1…`) [RUN]. v3 therefore saw about half its planned training |

**Training data** [CODE: `MANIFEST.json` files]:

| Version | Train / valid / test rows | Content | Format the model learned |
|---|---|---|---|
| v1 | 3,986 / 254 / 1,788 | template-generated examples (2,964), questions adapted from the Spider dataset's "tune" databases (421), narration (453), chat (148). Test domains "education" and "real_estate" held out | Tool-call plans (`{"type":"tools",…}`), clarify, decline, chat |
| v2 | 4,790 / 274 / 1,788 (same test set as v1) | built from v1 plus new examples (`new_examples_meta.jsonl`) | Same as v1 |
| v3 | 3,121 / 299 / 694 | 1,721 new examples **plus 1,400 replayed** v1/v2 examples (60 replayed in validation). 14 training domains, 4 held-out domains (airline, farm, charity, carrental) | Calculator plans (`{"type":"calc",…}`) with the first 3 data rows in the prompt; follow-ups; declines; clarifications |

**Replay** means mixing old training examples into new training so the model doesn't forget earlier skills (catastrophic forgetting). [THEORY; replay counts CODE]

**What the evaluations showed** [CODE: `training/reports/`]:

| Model | Training data | Method | Purpose | Evaluation result | Status |
|---|---|---|---|---|---|
| Base Qwen3-4B-Instruct-2507 (4-bit) | (pretrained by Qwen's developers) | — | starting point | Tool-planner test: intent 0.673, valid plan 0.619, column selection 0.423 (`analyst_finetune_eval_v1.md`) | Used *without* an adapter only in evaluation runs (Phase 5.1 planner comparison) |
| **v1** | 3,986 examples | LoRA r=8, 600 iters | Choose Silt's tools as JSON | Intent 0.992, valid plan 0.904, column selection 0.812; unseen domains 0.97–0.99 valid | **Active**: served by default |
| v2 | 4,790 examples | same | Fix tool selection and recommendations | Tool selection 1.000 and valid plans 0.977, **but** regression direction 0.500 (v1 0.900), declining when advertising columns are absent 0.850 (v1 1.000), unseen-domain unsupported handling 0.769 (v1 1.000) (`analyst_finetune_comparison_postfix.md`) | Kept on disk, not served: regressions on safety behaviours |
| v3 | 3,121 (1,721 new + 1,400 replay) | same, stopped at 430/800 iters | Calculator plans and follow-ups | Refusals 36/36 (v1 23/36), plans on training domains 0.57 (v1 0.33), **but** follow-ups 7/30 (v1 16/30) and valid plans overall 0.63 (v1 0.74, base 0.90) (`adapters/analyst_qwen3_4b_lora_v3/EVAL_COMPARISON.md`) | Kept on disk, not served: worse overall validity and follow-ups |

**Why v1 remains active**: it was the best-balanced. v2 traded safety behaviours for tool accuracy, and v3 traded follow-ups and validity for refusals. The serving script's default is v1 (`ADAPTER="${ANALYST_ADAPTER:-training/adapters/analyst_qwen3_4b_lora}"`). [CODE]

**A finding that matters for the newest code**: the program planner (`nl_planner.py`) uses a JSON format **none of the adapters was trained on**. In the Phase 5.1 evaluation the base model without any adapter wrote better programs (12/12 correct metrics against v1's 7/9–10/12). It could not be used on the 16 GB development machine, though: running two models made it time out on 10 of 23 calls (Part 17). The v1 adapter helps the formats it was trained on (tool calls) and can hurt formats it wasn't.

**Why training a new model immediately is not the right fix** [INFERRED from the evidence above]:
1. Two of three fine-tunes regressed on something important. Every retraining needs a full re-evaluation.
2. Many failures are *not* model failures. They are routing, filtering or semantic rules: silently dropped filters, wrong column roles (see the limitations document). Training cannot fix code that never asks the model.
3. The validation loss was already about 0.03 for every version. A low loss on templated examples does not mean good behaviour on new phrasings. The evaluations, not the loss, show the real quality.
4. A new adapter would need new, varied, verified training data, and the base model's better program planning suggests the cheaper step is to train *for the program format* — or to use a stronger model — not to repeat the old recipe.

### 5.5 Training versus inference

| | Training (done in the past, offline) | Inference (every time you press Send) |
|---|---|---|
| What changes? | The LoRA matrices A and B (7.34 M numbers) | **Nothing.** All weights are fixed |
| What is needed? | Thousands of input/output examples, the loss, gradients, an optimiser | One prompt (system + grounding + question) |
| Does the model update? | Yes, every iteration (600 of them for v1) | **No** |
| When? | Once, with `mlx_lm.lora`, taking hours on the laptop | Every request, taking seconds |
| Output | A new `adapters.safetensors` file | Generated text (a JSON plan) |

**Why pressing Send does not train the model**: the server only runs forward passes. There is no loss, no gradient and no optimiser at inference time, and no code path in Silt writes adapter files. Silt "remembers" a conversation only in a Python dictionary (Part 13), not in the model.

---

## Part 6 — Data ingestion: how files are read

### 6.1 CSV [CODE: `backend/app/ingest.py`]

| Concern | What the code does | Where |
|---|---|---|
| Upload | Streamed to disk in 1 MB chunks under `UPLOADS_DIR/<uuid>_<name>`; files over `MAX_UPLOAD_BYTES` are refused with HTTP 413. Only `.csv`, `.xlsx`, `.xls` are accepted as datasets; PDFs are pointed to the Copilot's document upload | `main.py` `_save_and_profile_upload` l. 166–189 |
| Encoding | UTF-8 with BOM → UTF-16 BOM → UTF-8 → Windows-1252 → Latin-1, reported when not UTF-8 | `_decode` l. 36 |
| Delimiter | Tries `,` `;` tab `|` on the first 400 non-empty lines; picks the one that splits rows into the same number of fields most consistently. Under 60% consistency the file is refused ("may not be a table") | `sniff_delimiter` l. 58; `read_csv_text` l. 188 |
| Decimal comma | With a non-comma delimiter, "3,5"-style values (≥ 10% of cells, no "3.5"-style values) mean decimal commas | `read_csv_text` |
| Header | A first row made only of numbers is data: columns are named `column_1, column_2 …` and this is reported | `read_csv_text` |
| Bad rows | Rows with a different field count are skipped and counted; more than 20% bad → refused. Short rows are listed by line number | `read_csv_text` |
| Column names | Empty unnamed columns are dropped; missing names become `column_N`; duplicate names get `_2`; whitespace is normalised | `_clean_columns` l. 143 |
| Empty rows | Fully empty rows are dropped and counted | `_clean_columns` |
| Numbers as text | "$1,200", "12%", "1 200", "(50)", European "1.234,56" are converted when ≥ 95% of values parse. Columns named like `*_id`, `code`, `zip`, `phone` are never converted | `_to_number` l. 77, `_types` l. 167 |
| Dates | Text columns whose values look like dates are parsed; day/month order is decided from the values, and the ambiguity is *reported* when every value fits both orders | `_looks_like_dates` l. 134, `_parse_dates` l. 118 |
| Missing values | Left as `NaN`. **Not filled in.** | — |
| Duplicates | **Kept in the file.** Handled at analysis time (Part 8) | `scope.eligibility` |
| Invalid values (e.g. negative hours) | **Kept in the file.** Excluded at analysis time with the reason | `scope.eligibility` |
| Storage | A clean UTF-8 comma CSV is stored as uploaded; anything structurally different (other delimiter or encoding, decimal commas, generated headers, skipped lines, Excel) is stored as a normalised CSV. The read report travels with the profile | `main.py` l. 207–219 |
| Schema | `data.schema` (name, kind numeric/categorical/datetime, missing, unique, top values or min/max); semantic roles come later (Part 7) | `data.py` l. 39 |

Example: a file `a;b\n1,5;x\n2,0;y` is read as delimiter "semicolon" with decimal comma, giving a = [1.5, 2.0]. Both facts are reported in `df.attrs["ingestion"]`. [CODE]

### 6.2 Excel

| Concern | Behaviour | Evidence |
|---|---|---|
| Reading | `pd.ExcelFile(path)`, then `pd.read_excel(..., sheet_name=sheet or 0)` | [CODE] `ingest.read_table` l. 252–282 |
| Worksheets | The first sheet by default; the upload form can name a sheet (`sheet` form field), checked to exist. A multi-sheet workbook gets a note naming the sheet used | [CODE] `main.py` l. 192–196; `12_Copilot.py` l. 151 |
| Header row | If more than half the column names are "Unnamed", the first row within the first 20 that is > 60% filled and > 80% text is used as the header, and reported | [CODE] |
| Formulas | pandas (through openpyxl) reads the **stored cached values**, not the formulas. A workbook saved by a program that didn't calculate formulas may show empty cells | [INFERRED: library behaviour, not Silt code] |
| Numeric/date cells | Excel's own types are kept; text numbers and dates go through the same `_types` conversion as CSV | [CODE] |
| Empty rows | Dropped (`_clean_columns`) | [CODE] |
| Merged cells | **No special handling.** The value sits in the top-left cell and the other cells are empty | [GAP] + [INFERRED] |
| `.xls` (old format) | Accepted by the upload check; reading needs the `xlrd` package | **NOT VERIFIED** that `xlrd` is installed |
| Storage | Always stored as a normalised CSV of the chosen sheet | [CODE] `main.py` l. 207–219 |

### 6.3 PDF [CODE: `backend/app/analyst/documents.py`]

| Concern | Behaviour |
|---|---|
| Upload | `POST /api/analyst/documents`, at most 25 MB, stored under `DATA_DIR/analyst_documents/<team>/<id>.pdf` with a `.json` of the extracted pages (no database table) |
| Text extraction | `pypdf.PdfReader(...).pages[i].extract_text()` per page (`extract` l. 42). An encrypted PDF is tried with an empty password, otherwise refused |
| Scanned pages | A page with fewer than 25 characters of text but with images is reported as **scanned**. **OCR is not implemented** — the module docstring says so (l. 1–3) |
| Tables | **Not extracted as tables.** Table text comes out as plain text lines [GAP] |
| Chunking | Pages are split into paragraphs and packed into passages of about 700 characters (`passages` l. 126) |
| Search | **BM25 keyword ranking** (`search` l. 141; k1 = 1.2, b = 0.75), with stop-words removed. **No embeddings, no vector database** |
| Answering | The top 4 passages are shown with page numbers (`orchestrator._ask_document` l. 1441). With the model on, Qwen writes 2–5 sentences from those passages only. If its text contains a number not in the passages (`explain.ungrounded`), it is not shown |
| Limits | Never used for statistics; keyword search misses synonyms ("revenue" vs "turnover"); no OCR; no table structure |

### 6.4 Data cleaning — what is and is not done

**Done automatically at read time**: decoding, delimiter, decimal mark, generated header names, dropping fully empty rows and unnamed empty columns, renaming duplicate column names, converting numbers written as text, parsing dates. [CODE]

**Done at analysis time, not to the file** (`scope.eligibility`, l. 251), for the columns an analysis uses:
- exact duplicate rows are excluded (some paths keep them and flag them — see the limitations document);
- negative values in amount-like columns (rate, price, cost, hours, salary …) are excluded;
- an end time before a start time is excluded;
- missing values in a used column are excluded for that analysis only;
- for historical analyses, rows dated after today are excluded.

Each exclusion is reported with its file line numbers. **Outliers are never removed**; they are reported as "unusual" (Part 11).

**Never done**: filling missing values (no imputation, except inside the Random Forest pipeline, Part 10), correcting typos in categories, merging near-duplicate labels ("HR" vs "H.R."), unit conversion.

Example from the demo timesheet [RUN]: the duplicated Rahul row (file line 12) and the record with `hourly_rate = -10` are excluded whenever the columns they touch are used, and the answer says so ("Not counted: 1 record(s) — exact duplicate of an earlier record (lines 12)").

---

## Part 7 — How Silt understands data columns

### 7.1 Two levels of type

1. **Storage kind** (`data.kind_of`, l. 29): numeric, datetime or categorical (anything else).
2. **Semantic role** (`semantics.column_role`, l. 60): what the column *means*. Roles are judged from **values first**, names second.

### 7.2 The role rules, in the order the code checks them [CODE: `semantics.py` l. 60–124]

| Order | Role | Rule |
|---|---|---|
| 1 | constant | one value or none |
| 2 | datetime | parsed as dates |
| numeric → | year | integers all within 1800–2100 and (named year/yr/fy/season, or ≤ 60 distinct values) |
| | identifier | integers, > 95% unique, not a count-like name, and (named id/code/no/number/ref/key **or** strictly increasing) |
| | flag | exactly two **integer** codes (0/1 → positive "1"; 1/2 → the rarer value is the "event") |
| | rate | named rate/pct/percent/share/ratio/margin/score, **or** all values between 0 and 1 and not integers |
| | amount | named like money (cost, price, revenue, salary, fee, bill …) |
| | count | non-negative integers |
| | measurement | any other numeric |
| text → | time_of_day | > 90% look like "09:30" |
| | flag | exactly two values (a yes/no pair recognised by `metrics.yes_no`, else "two values" with the rarer as the event) |
| | person_name / free_text / identifier | > 90% unique: capitalised multi-word names → person_name; long text → free_text; else identifier |
| | person_name | ≥ 80% capitalised multi-word names, ≥ 3 values |
| | identifier | named id/code/ref/key with > 50% unique |
| | free_text | average length > 60 characters |
| | category | everything else |

`semantics.profile` (l. 127) then lists:
- **measures** (amount, count, measurement, rate);
- **flags**;
- **outcomes**: a flag named like an outcome (churn, attrition, failure, default, late …, including concept synonyms such as "breakdown"), or the only yes/no flag when there are no key amounts;
- **groups** (categories and flags with 2–30 values);
- **dates**, **years**, the **entity** (the first person-name column);
- **nonfactors** (year, datetime, identifier, free text, constant, time of day), which are never used as association factors.

### 7.3 The brief's example columns — real output [RUN]

```
employee_id     identifier   conf=0.85 evidence=['unique per row']
department      category     conf=0.8  evidence=['3 repeated labels']
monthly_salary  amount       conf=0.8  evidence=['named like money']
date            datetime     conf=1.0  evidence=['parsed as dates']
hours_worked    count        conf=0.6  evidence=['non-negative integers']
attrition       flag         conf=0.95 evidence=['yes/no-like values']
is_remote       flag         conf=0.9  evidence=['two numeric codes (0, 1)']
hourly_rate     rate         conf=0.8  evidence=['named like a rate']        ← WRONG: it is money per hour
employee_name   person_name  conf=0.6  evidence=['capitalised multi-word names']
outcomes: ['attrition']   nonfactors: ['employee_id', 'date']
```

What to notice:
- **attrition** is recognised as the outcome because it is a yes/no flag *and* its name is an outcome word.
- **is_remote** (0/1) is a flag, not a number to average blindly.
- **hours_worked** is classed as a *count* (integer values). When Silt compares results it labels the unit "hours" from the column name (`nl_planner.unit_of`, l. 154).
- **hourly_rate is misclassified as a rate** because its name contains "rate". A *rate* in Silt means a share. The program planner therefore refuses to *sum* it, and the investigation treats it as a level. This is a real bug, listed in the limitations document.

### 7.4 Why an ID must not be averaged or used as a category

- The average of `employee_id` (e.g. 1050.5) means nothing. IDs are labels that happen to be numbers. `nl_planner.check` rejects mean/sum/median/min/max of an identifier: "'visit_ref' is a identifier; it cannot be meaned (count or nunique it instead)". [CODE `nl_planner.py` l. 525; TEST]
- Grouping by an ID makes as many groups as rows, so every "group comparison" compares single records. Identifiers are excluded from associations (`associations._id_like`, l. 32; `profile.nonfactors`) and are never offered as concept matches (`semantics.concept_candidates`, l. 164). [CODE/TEST]

### 7.5 Synonyms, ambiguity and encodings

- **Synonyms** are handled by small concept groups (`semantics.CONCEPTS`, l. 38): churn/attrition/quit/leave; failure/fault/breakdown; revenue/sales/income/takings; cost/expense/spend; customer/client; employee/staff; profit/margin; late/delay. Words are matched with simple stemming (`_forms`: quitting → quit, failed → fail). [CODE]
- **A concept is never mapped silently.** If "employees quitting" matches `attrition`, Silt asks "Do you mean `attrition`?" (`orchestrator._ask_associations`, l. 929). If two columns match ("revenue" → `takings` and `online_sales`), Silt asks which (l. 183–190). [CODE/TEST]
- **1/0 and Yes/No**: both become flags. A mean of a yes/no *text* column in a model program is rewritten as the share of "Yes" (`nl_planner._flag_means`, l. 667). A mean of a 0/1 column is already a share. [CODE/TEST: `fleet-late-share (0/1)` evaluation case]

---

## Part 8 — How data is filtered

There are three filtering mechanisms. Which one applies depends on the path that answers.

### 8.1 Question scope (`scope.establish`, `scope.py` l. 99) — deterministic

1. **Values named in the question** (`calc.named_values`, l. 1205): for every categorical column with ≤ 50 values, a value is "named" if it appears as a whole word in the question. A first name ("Rahul") names a full value ("Rahul Sharma") when exactly one value contains that word.
2. **A value in two columns** with nothing saying which: Silt **asks** (`scope["clarify"]`).
3. **Negation** ("excluding", "except", "other than", "without", "but not" …) moves a value to `excluded_entities` (`_negated`, l. 73).
4. **Months and years** ("September 2026", "in 2025") become a time filter on the single date column (`_time`, l. 83). A year with no date column filters a numeric year column (`numeric_eq`).
5. **Rate questions** ("what share … attrition Yes") do not turn the measured value into a filter, which would make every share 100%.
6. **Follow-ups** that name nothing new inherit the previous scope (`inherited`, l. 146–153).

`scope.mask` (l. 178) turns the scope into a boolean row mask:
- values within one column are combined with **OR** (`isin`);
- different columns are combined with **AND**;
- exclusions use **AND NOT**;
- the period becomes `year ==` and `month ==` on the date column.

The filtered DataFrame is `df[mask]`. `_finish` then **verifies** that the answer used only records inside the scope. If it used none, or used records outside it, the numbers are withheld and Silt says so (`orchestrator.py` l. 544–558).

### 8.2 Analysis-program filters (`program.op_select`, `program.py` l. 103)

`select` supports:
- `entities` — values must exist, else ProgramError "X does not occur in 'col'";
- `except`;
- `period` with `start`/`end` dates;
- `where` conditions with `==` `!=` `>` `>=` `<` `<=`, combined with **AND**;
- `valid_for`, which applies `scope.eligibility` and lists the exclusions with file lines.

**OR across different columns is not expressible** in one step [GAP].

### 8.3 Calculator exclusions (`calc.py`)

The v1 calculator writes `exclude` rules as expressions, e.g. `{"where": "line_supervisor == 'Yusuf Okoro'"}` (see the v3 training example, Part 5.4). These are parsed with Python's `ast` module and only whitelisted operations are allowed (`calc._Eval`, l. 262). There is no `eval()` of arbitrary code.

### 8.4 The six example filters — what really happens (model off) [RUN]

| Request | Mechanism | Result | Correct? |
|---|---|---|---|
| "Average monthly income of only Engineering employees" | scope entity `department = Engineering` | 4,663.44 over 174 of 600 records | ✓ (pandas: 4,663.44) |
| "Total shift hours for only Rahul" | scope entity `employee_name = Rahul Sharma` | 26.00 h over 3 records, duplicate **included** and flagged | Inconsistent: the feasibility path counts 18 h (duplicate excluded). See limitations |
| "Total amount for only the last 30 days" | **none**: relative periods are not parsed by `scope._time` | 4,123 over **all** 40 rows | **✗ filter silently ignored** (true value 2,938) |
| "How many employees worked more than 8 hours?" | none deterministically (needs a derived duration plus a condition) | "I couldn't turn this into an analysis …" | Not answered (honest, not wrong). With the model on, the calculator or program planner can express it as `where shift_hours > 8` |
| "Total amount excluding cancelled orders" | scope `excluded_entities status = cancelled` | 2,915 over 28 records | ✓ (pandas: 2,915) |
| "Average resolution hours of only escalated tickets" | **none**: "escalated" is a yes/no column named as an adjective; the rule in `scope.establish` covers "for/among/of/with escalated", not "only escalated" | 7.92 over **all** 360 rows | **✗ filter silently ignored** (true value 12.66) |

**How invalid filters are prevented**:
- named values must exist in the data;
- a value in two columns triggers a question;
- a filter that matches nothing returns "No records match … — I haven't used the rest of the data instead" (`orchestrator.py` l. 157);
- in programs, nonexistent values are rejected before execution.

**What is not prevented**: a filter the parser doesn't recognise is simply absent, as the two ✗ rows show. That is the most serious class of bug found in this audit; see the limitations document.

---

## Part 9 — How calculations work

### 9.1 The program language (`backend/app/analyst/program.py`)

A program is a list of steps. Each step has an `id`, an `op`, its input (`from`: the dataset `"data"` or an earlier step id) and arguments. `validate` (l. 436) checks that operations exist, that ids are unique and that inputs refer to earlier steps. `run` (l. 450) executes the steps in order and stores a `Result(kind, value, unit, provenance, hypothetical)` for each (l. 43). Anything computed from a hypothetical result is marked hypothetical too (l. 450–461).

| Op | What it does | Input → output | pandas core | Small example | Failure cases |
|---|---|---|---|---|---|
| **select** (l. 103) | keeps rows | table → table | boolean masks, `isin`, comparisons, date bounds | 6 rows, `entities {driver: [A]}` → A's 3 rows | value not in column; no date column for a period; bad comparison operator |
| **derive** (l. 156) | adds a per-row column from a safe formula | table → table | `calc.evaluate` (whitelisted `ast`) | `km / trips` per row | formula error (unknown name, disallowed syntax) |
| **aggregate** (l. 172) | sum, mean, median, min, max, count, nunique — overall, `by` a column, or `per` D/W/M/Q/Y | table → scalar or series | `groupby(...)` then the function; `count = len(rows)`; sum over 0 rows = 0 | mean of [10, 20, 30] = 20 | non-numeric column; no date column for `per` |
| **project** (l. 212) | carries a daily pace forward | table → projection | daily totals over the history window (missing days = 0); expected = mean × days; range = ± 1.96 × sd × √days | daily [8, 10, 0], 27 days → 6 × 27 = 162 | no dated values; negative horizon |
| **compare** (l. 251) | difference and ratio of two values, or a value against a **target** with a verdict | 2 results → comparison | arithmetic | target 300, done 18, expected 180 → "unlikely" | different units; missing input |
| **scenario** (l. 298) | removes an entity or scales a column, on a **copy** | table → hypothetical table | `df.copy()`, mask, multiply | scale fee +10% | value missing; non-numeric column |
| **rank** (l. 341) | top or bottom n of a per-group result | series → series | sort | lowest 1 → {East: 110} | input not a series |
| **anomalies** (l. 356) | values outside [Q1 − 1.5 IQR, Q3 + 1.5 IQR] | table → series | `quantile` | [10, 11, 12, 13, 100] → 1 unusual | < 8 values; non-numeric |
| **associate** (l. 381) | each column's association with a target (Part 11) | table → series | scipy | — | target missing |
| **profile** (l. 392) | missing values and impossible records | table → series | `isna`, `eligibility` | — | — |
| **ratio** (l. 404) | a ÷ b for two values or two per-group results | 2 results → scalar/series | per-key division; groups with b = 0 are listed | {N: 200}/{N: 4} = {N: 50} | a series divided by a scalar (refused) |

### 9.2 Formulas

| Quantity | Formula | Example |
|---|---|---|
| Sum | Σxᵢ | 10 + 20 + 30 = 60 |
| Count (rows) | n | 3 |
| Count (non-null) | number of xᵢ that are not missing | [5, NaN, 7] → 2 |
| Mean | Σxᵢ / n | 60 / 3 = 20 |
| Median | middle value of the sorted list | [1, 3, 100] → 3 (robust to the 100) |
| Percentage / share | (number of records meeting the condition / total) × 100 | 2 late of 8 trips = 25% |
| Rate (Silt) | mean of a 0/1 flag | late = [1, 0, 0, 1] → 0.5 |
| Ratio | a / b | 300 km / 4 drivers = 75 km per driver |
| Difference | a − b | 12 − 9 = 3 |
| Growth rate | (new − old) / old | (120 − 100) / 100 = 20% |
| Group-wise aggregation | the aggregate inside each group | mean salary by department |
| Weighted average | Σ wᵢxᵢ / Σ wᵢ | — |
| Per-unit | total / number of units | total km / distinct drivers |

Silt has **no "growth rate" operation**. It can be expressed with `compare` (the difference and ratio of two values), and the trend finding reports first-to-last change. **Weighted averages are not a separate operation**: a ratio of totals *is* the weighted average of the per-record ratios. [CODE/THEORY]

### 9.3 Why denominators matter — a worked example

Delivery data:

| driver | trip | km | late |
|---|---|---:|---|
| A | 1 | 10 | No |
| A | 2 | 10 | No |
| A | 3 | 100 | Yes |
| B | 4 | 40 | No |
| B | 5 | 40 | No |

| Question | Calculation | Value |
|---|---|---:|
| Total kilometres | 10 + 10 + 100 + 40 + 40 | **200** |
| Average kilometres per trip (mean of rows) | 200 / 5 trips | **40** |
| Kilometres per trip for A | 120 / 3 | 40 |
| Kilometres per driver | 200 km / 2 distinct drivers | **100** |
| Count of rows | 5 | 5 |
| Percentage of late deliveries | 1 late / 5 trips | **20%** |
| Average of drivers' own late rates | (A: 1/3 = 33.3% + B: 0/2 = 0%) / 2 | **16.7%** ≠ 20% |
| Late rate as a ratio of totals | (1 + 0) / (3 + 2) | **20%** |

The average of individual ratios (16.7%) gives driver B's 2 trips the same weight as driver A's 3. The **ratio of totals** (20%) weights every trip equally. When a request says "allowing for how far each one drives" or "per trip", the ratio of totals is what's meant. `nl_planner.check` therefore rejects "mean of a per-record ratio" with that explanation, and `intent.check` requires the requested denominator ("per driver" → divide by distinct drivers, not by row count). [CODE/TEST: `test_an_average_of_per_record_ratios_is_rejected_for_a_ratio_of_totals`, `test_the_denominator_must_be_the_one_asked_for`]

### 9.4 Provenance — how every number explains itself

Each `Result` carries `provenance` [CODE `program.py`]:

| Field | Example |
|---|---|
| input columns | `{"op": "aggregate", "fn": "sum", "column": "shift_hours"}` |
| formula | `{"op": "derive", "formula": "hours_between(clock_in, clock_out)"}` |
| records used | `rows_in`, `rows_out`, `rows` |
| records excluded | `excluded: [{"reason": "exact duplicate of an earlier record", "records": 1, "lines": [12]}]` |
| unit | `unit: "hours"` (from the column name, or set by the planner), `"share"` for a flag mean |
| hypothetical status | `hypothetical: True` for scenario tables and everything computed from them |

Example from the showcase answer [RUN]:

```
- `sel` select {"entities": {"employee_name": ["Rahul Sharma"]}, "valid_for": ["clock_in","clock_out"], "historical": true, ...} → 2 rows
- `m0` derive {"name": "shift_hours", "formula": "hours_between(clock_in, clock_out)"} → 2 rows
- `done` aggregate {"fn": "sum", "column": "shift_hours"} → 18.0
- `pace` project {"column": "shift_hours", "days": 27, ...} → expected 162
- `verdict` compare {"a": "pace", "base": "done", "target": 300.0, "days": 27, "unit": "hours", "max_daily": 24} → unlikely on the current pace
```

---

## Part 10 — How predictions work

"Prediction" covers four different things in Silt. Do not mix them up.

| Kind | Uses ML? | Code |
|---|---|---|
| Forecast (next period of a measure) | a linear trend, *only* if it beats a naive baseline | `followup.forecast` l. 501 |
| Supervised ML (predict a column) | Random Forest | `ml.train` l. 150 |
| Target feasibility | **No**: a pace projection | `objectives._feasibility` + `program.op_project`/`op_compare` |
| Hypothetical scenario | **No**: recalculation on changed data | `program.op_scenario`, `objectives._absence` |

### 10.1 Forecasting [CODE: `followup.forecast`, l. 501–630]

1. Builds a time series: the measure per period (day/week/month, `auto_freq`), from eligible historical records in scope. Incomplete periods are dropped and reported.
2. Fewer than 2 periods → refused. Fewer than 8 (`MIN_POINTS_ML`, l. 460) → **no model**: the historical average is reported and labelled an estimate.
3. Otherwise, the last 25% of periods (at least 2) are held out. A `LinearRegression` on the period number is fitted on the rest and compared with a **naive baseline** (repeat the last training value) by mean absolute error.
4. If the trend beats the baseline, it is refitted on all periods and the next period is predicted ("ml_prediction"). If not, the last value is repeated and labelled.
5. If the records end long before today and "next" is ambiguous, Silt asks which period is meant.

**Not implemented**: seasonality, prediction intervals for forecasts, multiple-series models. [GAP]

### 10.2 Supervised machine learning [CODE: `ml.py`]

| Element | Implementation |
|---|---|
| Target | A column the user names; regression if numeric with many values, else classification (`infer_task`, l. 42) |
| Features | All other columns after screening (`screen_features`, l. 57): identifiers, dates, free text, high-cardinality text and **leakage** (a feature with \|correlation\| ≥ 0.98 to the target, `LEAKAGE_CORR`) are removed, with reasons |
| Preprocessing | Inside a `Pipeline`, fitted on the training rows only: median imputation for numbers; most-frequent imputation and one-hot encoding for categories |
| Algorithm | `RandomForestClassifier` (300 trees, `class_weight="balanced"`, `min_samples_leaf=2`) or `RandomForestRegressor` (300 trees) |
| Split | **Time-ordered** (the latest 25% held out) when there is one date column with ≥ 10 distinct dates; otherwise a stratified random split. The user may choose 80/20, 75/25 or 70/30 |
| Cross-validation | 5 folds (`TimeSeriesSplit` when time-ordered, `StratifiedKFold`/`KFold` otherwise), compared fold-by-fold with a baseline |
| Metrics | Classification: accuracy, **balanced accuracy**, macro-F1, ROC-AUC (binary), confusion matrix, against "always predict the majority class". Regression: R², MAE, RMSE, against "always predict the training mean" |
| Explanations | Permutation importance on the held-out rows, labelled "what the model relied on, not what causes the outcome" |
| Refusals | Too few rows (< 40), classes with too few examples, no usable features |

[TEST: `test_ml_on_dated_records_is_validated_forward_in_time`]

**Data leakage** [THEORY]: a feature that wouldn't be known at prediction time, or that is a copy of the target, makes the model look perfect in testing and useless in reality. Silt screens near-copies (|r| ≥ 0.98) and uses forward-in-time splits for dated data.

### 10.3 Target feasibility — "Can Rahul complete 300 working hours this month?" [RUN, numbers verified by hand]

Rahul's valid records: 1 Sep (09:00–17:00 = 8 h) and 2 Sep (09:00–19:00 = 10 h). The 1 Sep duplicate (file line 12) is excluded. The latest month with records is September 2026, as of 3 Sep (the last record date).

| Quantity | Formula | Value |
|---|---|---|
| Completed hours | 8 + 10 | **18 h** |
| History window | 1–3 Sep (3 days; 3 Sep has no record for Rahul → 0) | daily totals [8, 10, 0] |
| Historical daily pace | (8 + 10 + 0) / 3 | **6 h/day** |
| Days left | 30 Sep − 3 Sep | **27** |
| Still needed | 300 − 18 | **282 h** |
| Required daily pace | 282 / 27 | **10.44 h/day** |
| Expected total | 18 + 6 × 27 | **180 h** |
| Range | 180 ± 1.96 × sd × √27, sd of [8, 10, 0] = 5.29 | **126.1 – 233.9 h** |
| Gap | 180 − 300 | **−120 h** |
| Verdict | the whole range is below 300 | **"unlikely on the current pace"** |

Verdict rules (`program.op_compare`, l. 251–296): already reached; not reachable (no days left); **not physically possible** (required pace above `max_daily`, here 24 h for hours); likely (low end ≥ target); unlikely (high end < target); otherwise uncertain.

**Assumptions** (stated in the answer): each future day follows the historical daily average, days without records count as 0, and days vary independently. **Limitations**: three days of history make the range wide; recorded hours say nothing about productivity.

### 10.4 Hypothetical scenarios — "What if Rahul is absent for the next 10 days?" [RUN]

- **Representation in the feasibility path**: `absent_days = 10` is stored in the open objective's parameters and **subtracted from the days left** (27 → 17) for both the projection and the required pace (`objectives._feasibility`, l. 235–239). Result: expected total 18 + 6 × 17 = **120 h**, required pace 282/17 = **16.59 h/day**.
- **Representation in programs**: `scenario` makes a **copy** of the table (`df.copy()`) with an entity removed or a column scaled. The original DataFrame is never modified.
- **Propagation**: the scenario result has `hypothetical=True`, and `program.run` marks everything computed from it as hypothetical (l. 450–461). The answer says "This is a hypothetical scenario: the records were not changed". [TEST: `test_results_computed_from_a_scenario_stay_hypothetical`]
- **Caveat [GAP]**: the absence path subtracts *calendar* days while the wording says "working days"; there is no working-day calendar.

---

## Part 11 — Statistical analysis

| Method | Used for | Formula / idea | Where |
|---|---|---|---|
| **Spearman correlation** | number vs number | the Pearson correlation of the *ranks*; −1 to 1; robust to outliers and to non-linear but monotonic relations | `associations.compute` l. 53 (`scipy.stats.spearmanr`) |
| **Eta squared (η²)** | category vs number | between-group sum of squares / total sum of squares; 0 to 1 = the share of variance explained by the groups. Its p-value comes from **Kruskal–Wallis** | l. 57–67 |
| **Cramér's V** | category vs category (numbers binned into ≤ 5 ranges) | √(χ² / (n·(min(rows, cols) − 1))) from the contingency table; 0 to 1 | l. 68–80 |
| **Holm correction** | many columns tested at once | sort p-values; multiply the k-th smallest by (m − k + 1), keeping them non-decreasing | `_holm` l. 23 |
| **Welch's t-test / Welch ANOVA** | comparing a number between 2 / several groups | tests whether the means differ without assuming equal variances; with Hedges' g (2 groups) or η² (several) as effect size | `tools.py` l. 279–290 |
| **Kruskal–Wallis** | rank-based check alongside Welch ANOVA | | `tools.py` l. 286 |
| **IQR outliers** | unusual values | outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]; "extreme" beyond 3·IQR | `program.op_anomalies` l. 356 |
| **Missing / duplicate analysis** | data quality | `isna().sum()`; `duplicated()` | `op_profile` l. 392, `scope.eligibility` |

**Worked examples** [THEORY]:
- *Spearman*: x = [1, 2, 3, 4], y = [10, 20, 30, 1000] has ranks [1, 2, 3, 4] for both, so ρ = 1.0, even though the Pearson correlation is lower.
- *Cramér's V*: 100 people; overtime Yes: 15 of 30 left; overtime No: 5 of 70 left. χ² ≈ 21.4, so V = √(21.4/100) ≈ 0.46 (a strong association).
- *Holm*: three p-values 0.01, 0.02, 0.04 → adjusted 0.03, 0.04, 0.04, so all stay below 0.05. With 10 tests the same 0.04 would become 0.04 × 10 = 0.4.

**Significance**: an association is reported as "unlikely to be chance" when its Holm-adjusted p < 0.05. The investigation additionally requires strength ≥ 0.1 (`investigate.py` l. 173).

**Association is not causation** [THEORY]: overtime and attrition can be associated because stressful teams cause both (a confounder), because leavers worked overtime to hand over (reverse direction), or by chance. Every association answer says "not causes" (`associations.text` l. 106–108; investigation caution text). Silt never runs a causal analysis. Asked for one, it reports what the data lacks (evaluation case `fleet-unsupported-cause`).

---

## Part 12 — Autonomous investigation [CODE: `investigate.py`; RUN]

Triggered by requests like "Investigate this dataset" (`orchestrator._INVESTIGATE`; `_ask_investigation` l. 770). No model is involved.

| Stage | What happens | Code |
|---|---|---|
| 1 Profiling | `semantics.profile` gives roles, measures, groups, outcome, dates, entity | `plan` l. 38 |
| 2 Choosing analyses | Only analyses the schema supports. Example: no date column means no trend step, reported as a limitation | `plan` |
| 3 Missing values / duplicates / impossible records | a `profile` step, then a `select` with `valid_for` (duplicates kept and flagged) | `plan` |
| 4 Outliers | `anomalies` on up to 4 main measures | `plan` |
| 5 Group comparisons | the outcome rate by group, the main measure by group | `plan` |
| 6 Associations | `associate` with the outcome (or the main measure), excluding nonfactors | `plan` |
| 7 Trends | per month/week/day or year, when there is a date | `plan` |
| 8 Prioritisation | each candidate finding has a `score` and must pass an evidence bar: associations with adjusted p < 0.05 and strength ≥ 0.1 (top 3; ≥ 0.9 flagged "may be calculated from each other"); rate differences ≥ 5 percentage points between groups of ≥ 5 records (`MIN_GROUP`); mean ratios ≥ 1.25; top entity share ≥ 15%; unusual values only if extreme, ≥ 2% of values, or ≥ 2% of the total; category concentration ≥ 70% | `investigate` l. 135–318 |
| 9 Explanation | every finding records title, statement, calculation, columns, scope, n, interpretation, limitations and a follow-up question | `_finding` l. 130 |
| 10 Follow-up | "tell me more about finding N" → `detail`; "dig into finding N" runs its follow-up question | `orchestrator._ask_investigation` |
| 11 Storage | `memory["investigation"] = {"findings": [...], "program": [...]}` | `orchestrator.py` l. 797 |
| 12 Limitations | for example "No date or year column: no trend analysis", "No outcome column" | `investigate` |

**A real end-to-end example** — the demo HR file, 600 rows [RUN, earlier browser run]:

```
I investigated hr_attrition.xlsx [Employees] (600 records) and ran 19 checks chosen for its columns. 6 finding(s):
1. overtime is associated with attrition. Cramér's V 0.28, adjusted p < 0.001.
2. satisfaction is associated with attrition. Cramér's V (number grouped into ranges) 0.22, adjusted p < 0.001.
3. job level is associated with attrition. Cramér's V 0.16, adjusted p < 0.01.
4. monthly income differs by job level. 7,480.80 for Senior (118 records) vs 3,335 for Junior (264 records).
5. attrition rate differs by overtime. Yes in 25% of records with overtime = Yes vs 5% with overtime = No (overall 11%).
6. attrition rate differs by job level. 16% for Junior vs 3% for Senior.
Investigate next: What factors are associated with attrition? ...
```

The numbers are checked independently in `tests/test_analyst_investigate.py`: planted signals are found, a pure-noise column is not reported, and Cramér's V equals scipy's value. [TEST]

---

## Part 13 — Conversation memory

### 13.1 Where and what [CODE]

- **Where**: a Python `OrderedDict` named `_MEMORY` in the backend process (`orchestrator.py` l. 19–20). At most **200** conversations; the least recently used are dropped. **Not stored in SQLite or Postgres.** The `copilot_sessions`/`copilot_turns` tables belong to the separate legacy `/api/copilot` reader (`app/copilot.py`), not to the Analyst Copilot.
- **Key**: `"team:user:dataset:document:conversation"` (`routers/analyst.py` l. 99). Different users, datasets or conversations never share memory.
- **Restarts**: memory is **lost** when the backend restarts. The frontend's message log lives in Streamlit session state, so it survives page interactions but not a server restart or a new browser session. [INFERRED from Streamlit behaviour]

### 13.2 What is stored

| Key | Contents | Written by |
|---|---|---|
| `task` | the open request, status (`awaiting`/`done`), pending clarification (question, options, expected kind of answer), offers | `conversation.record` |
| `scope` | the last answered question's filters | `_finish` |
| `last_answer`, `last_charts`, `last_analysis` | the previous question, answer text and evidence, and charts | `followup.remember`, `_finish` |
| `last_calc`, `calc_history` | calculator results and plans, for refinements | calc paths |
| `model` | a trained Random Forest (fitted pipeline), for "predict for this case" | ML path |
| `objective`, `objectives` | the open target/absence analysis and the history of analyses (the last 6) | `objectives._run_and_store` |
| `program` | the last program's steps and results | objectives, investigation |
| `planned` | model/rule-planned programs (the last 6) with scalar values usable later as `@r1.step` | `nl_planner.execute` |
| `investigation` | findings and program | `_ask_investigation` |
| `_nl_*`, `_model_spent_ms`, `_scope` | per-turn working state, cleared every turn | `_dispatch` |

### 13.3 The brief's conversation, traced [RUN with model off on the timesheet, plus the browser run with the model on]

| Turn | What happens | Memory after the turn |
|---|---|---|
| 1 "Investigate the dataset" | `investigate.investigate` | `investigation` (findings), `program`, `task` done |
| 2 "Explain finding 2" | `_FINDING` regex finds 2 → `investigate.detail` | unchanged findings; `last_answer` |
| 3 "Compare Rahul with others" | `objectives._vs_rest` (when worded "compare Rahul with the others") | `objective`/`objectives` gain a vs_rest entry, `scope` = Rahul |
| 3′ (showcase) "Can Rahul reach 300 hours this month?" | feasibility program | `objective` = {target 300, entity Rahul Sharma, measure shift_hours …}, `objectives` [1] |
| 4 "Change target to 320" / "What if the target changes to 320?" | `objectives.handle` `_CHANGE`/`_CHANGE_IF` → re-run with target 320 | `objectives` [2], `objective` = the latest |
| 5 "Compare with the previous scenario" | `_CMP_SCEN` → `_compare_scenarios(hist[-2], hist[-1])`; nothing recalculated | `last_charts` = the comparison chart |
| 6 "Show the difference as a chart" | `_chart_from_previous_answer` redraws `last_charts` | unchanged analysis state |
| (new conversation) "Compare that scenario with the previous one" | new key, empty memory: no comparison is fabricated | fresh |

Real numbers from turn 5 [RUN]: target 300 → 320; expected total 120 (with the absence) in both; gap −180 → −200; required per day 16.59 → 17.76; "The results differ because the target changed (300 → 320)".

### 13.4 Pending clarifications

When Silt asks a question with options (e.g. "Which column do you mean by revenue? Options: takings, online_sales"), `conversation.record` stores it as `pending`. The next message is read by `conversation.resolve`: "takings" is matched to an option and folded into the original request, "yes" takes the only option, "the other one" takes the alternative, "forget that" drops the task. This is deterministic, so a model outage cannot lose a pending task. [CODE/TEST: `test_a_model_clarification_is_asked_and_its_answer_continues_the_request`]

### 13.5 Limitations

Memory is not persistent, is per backend process (two workers would not share it), keeps the last 6 analyses, and gives the planner a summary rather than the full transcript.

---

## Part 14 — Charts and the frontend

| Concern | How it works | Where |
|---|---|---|
| Results → chart data | Paths build chart specifications as dicts, e.g. `{"type": "bar", "x": "group", "y": "value", "title": ..., "data": [{"group": "East", "value": 110}, ...], "additive": False}`. No images are generated on the server | `nl_planner._charts`, `objectives._charts` l. 558, `investigate` |
| Chart type | Set by the path: per-group results → bar; time series → line; distributions → histogram; "actual vs predicted" → avp. A requested type ("as a pie chart") restyles the previous chart (`followup.restyle`); a pie is refused for non-additive values (`additive: False`) | `followup.py`, `orchestrator._finish` l. 520–539 |
| Labels and axes | From the spec's `x`, `y`, `title`. Program charts use Silt's own labels, not the model's | `copilot_ui.chart` |
| Drawing | `copilot_ui.chart(spec, theme)` builds an **Altair** chart: line, pie (donut), scatter/avp (with a diagonal), histogram, or a horizontal bar chart (optionally grouped by a `series` key), with tooltips | `frontend/style/copilot_ui.py` l. 158–207 |
| Tables | Written as **markdown tables** in the answer text, rendered by `st.markdown` | `show_answer` |
| Messages | `st.chat_message("user"/"assistant")` per turn | `12_Copilot.py` l. 282–289 |
| Loading | A typing indicator (three dots plus a label) in an `st.empty()` slot while the request runs; extra questions are queued and answered in order | l. 304–313; `copilot_ui.typing_html` |
| Errors | An `ApiError` or exception becomes a red `st.error` message with a "Try again" button; the conversation is kept | `show_answer` l. 238–245, l. 321–326 |
| Source labels | The `plan_source` value is mapped to a sentence (`SOURCE` dict, l. 35–50): e.g. "Analysis program planned by the local model; validated and computed by Silt", "Analysis program built by Silt from the request's parts; validated and computed", "Planned by the fine-tuned local model" | `12_Copilot.py` |
| Transparency | The *How this was calculated* expander shows `details` (program steps, checks), `notes`, and optionally the technical trace (plan, tool calls, the model's first pass) | `show_answer` l. 260–270 |

---

## Part 15 — Database and deployment

| Topic | Facts | Evidence |
|---|---|---|
| Database | SQLAlchemy. **SQLite** at `DATA_DIR/app.db` by default; **Postgres** when `DATABASE_URL` is set (connection pooling and Alembic migrations) | `database.py` l. 1–35; `config.py` l. 22 [CODE] |
| Tables | 42 models, e.g. `users`, `teams`, `team_members`, `datasets`, `dataset_versions`, `questions`, `plans`, `execution_logs`, `critic_reviews`, `dashboards`, `investigations`, `scenarios`, `semantic_metrics`, `eval_runs`, `copilot_sessions` … | `models.py` [CODE] |
| Dataset persistence | The `datasets` row stores filename, path, row/column counts and a profile JSON; `dataset_versions` keeps each uploaded version; the file itself is on disk | `main.py` l. 257–281 [CODE] |
| Conversation persistence | **None for the Analyst Copilot** (Part 13) | [CODE] |
| Authentication | Passwords hashed with **bcrypt**; sessions as **JWT** signed with `JWT_SECRET_KEY` (`security.py` l. 19–40); every analyst route depends on `get_current_user` and `get_current_team`; optional Google sign-in through an internal secret | [CODE] |
| API security | CORS allow-list configurable (`CORS_ALLOWED_ORIGINS`, defaulting to `*`); upload size cap; rate limiting module `rate_limit.py`; team checks on dataset access (`team_dataset`) | [CODE]; rate-limit coverage of analyst routes NOT VERIFIED |
| File storage | `DATA_DIR` (uploads, workspaces, analyst_documents) | `config.py` l. 16–24 |
| Local development | Backend `uvicorn` on 127.0.0.1:8000; Streamlit on 8501; **MLX model server on 127.0.0.1:8080 with the v1 adapter** | [CODE/RUN] |
| Hosted deployment | **Render**, Docker (`render.yaml`, `Dockerfile` FROM python:3.11-slim, `start.sh`): uvicorn and Streamlit in one container, Postgres (Neon) through `DATABASE_URL`; uploads on ephemeral disk unless a disk is added | [CODE] |
| Model server when hosted | **Not present.** The Docker image starts only uvicorn and Streamlit, and no requirements file includes `mlx` (MLX runs only on Apple Silicon). `model_available()` is false there, so the hosted Copilot runs only the deterministic paths | [CODE] |
| Backend ↔ model | HTTP to `ANALYST_MODEL_URL` (default `http://127.0.0.1:8080/v1`), model name `default_model` (the adapter); the program planner can be pointed elsewhere with `ANALYST_PLANNER_URL`/`ANALYST_PLANNER_MODEL` | `planner.py` l. 26–30; `nl_planner.py` l. 41–45 |
| Environment variables (names only) | `DATA_DIR`, `UPLOADS_DIR`, `DB_PATH`, `DATABASE_URL`, `JWT_SECRET_KEY`, `INTERNAL_AUTH_SECRET`, `CORS_ALLOWED_ORIGINS`, `MAX_UPLOAD_BYTES`, `GEMINI_API_KEY`, `LLAMA_API_KEY` (legacy pipeline), `ANALYST_MODEL_URL`, `ANALYST_MODEL_NAME`, `ANALYST_MODEL_TIMEOUT`, `ANALYST_PLANNER_URL`, `ANALYST_PLANNER_MODEL`, `ANALYST_PLAN_TIMEOUT`, `ANALYST_PLAN_BUDGET`, `ANALYST_ADAPTER` (serving script), `SANDBOX_*` | [CODE] (values deliberately not shown) |
| Git state at audit time | Branch `feature/analyst-reliability`, HEAD `0655205`; **20 modified and 20 untracked files** hold the recent Copilot work (`intent.py`, `nl_planner.py`, `program.py`, `semantics.py`, `investigate.py`, `objectives.py`, `associations.py`, `ingest.py` and their tests): uncommitted | [RUN] |

---

## Part 17 — Testing and evaluation

### 17.1 Kinds of tests

| Kind | What it checks | In Silt |
|---|---|---|
| Unit tests | one function in isolation | e.g. `test_analyst_program.py` (each operation), `balance()` repairs |
| Integration tests | several parts together through `orchestrator.ask` | most analyst tests ask real questions end to end, with the model off or a **scripted fake model** |
| Numerical correctness tests | answers equal independent pandas/scipy calculations | `test_analyst_objectives.py` (feasibility numbers), `test_analyst_investigate.py` (Cramér's V), `test_analyst_nl_planner.py` (per-head ratios), `test_analyst_intent.py` (rates) |
| Browser tests | the real UI in headless Chromium (Playwright scripts outside the repository) | 10/10 steps on the final code (investigation, findings, target and what-if, scenario comparison and chart, fresh conversation, three HR questions) |
| Model-on vs model-off evaluation | the same cases with deterministic paths only, with the v1 planner, and with the base model as planner | `training/scripts/eval_program_planner.py` |

**Counts at the end of the last phase** [TEST]:
- backend: 1,612 passed, 31 skipped;
- analyst tests: 653 passed;
- frontend: 326 passed.

The test suite blocks every live model call (`tests/conftest.py`, `_no_live_program_planner`), so tests never depend on the model server.

### 17.2 What the evaluation numbers mean

The Phase 5.1 evaluation has 49 cases: 26 original and 23 paraphrased/composite. They cover generated datasets (fleet, retail outlets, support desk) with unit and encoding variants, the public *tips* dataset, ambiguous and impossible requests, and multi-turn conversations. Truth comes from pandas.

| | Model off | v1 planner | Base model planner |
|---|---|---|---|
| Passed | 23/49 | 36/49 | 31/49 |
| Numerically correct answers | 15/40 | 28/40 | 22/40 |
| Model-planned answers with correct numbers | — | 11/12 | 5/6 |
| Clarification asked when needed | 2/2 | 2/2 | 2/2 |
| p50 / p90 end-to-end latency | ~0 s | 2.4 / 13 s | 5.5 / 26 s |

(Results files are in the session's scratch area, not in the repository; the script regenerates them.)

How to read these honestly:
- **36/49 is not "73% accurate on any question."** The cases were written to probe weaknesses. Different questions would give different numbers.
- A **pass** means "the pandas truth numbers appear in the computed evidence" (or the right clarification/refusal). It does not grade the wording.
- **False positives** (a confident but wrong answer) matter more than false negatives (an honest "I couldn't"). The filter bugs in Part 8 are false positives; most other failures are honest fallbacks.
- The base model's lower score here comes from **timeouts on a memory-starved 16 GB machine** (10 of 23 planning calls). In an earlier, less loaded run it reached 39/49 with 12/12 correct metrics.

### 17.3 The ten held-out datasets

An earlier project brief named ten public datasets to be kept for final evaluation. **They were not found on disk during the reliability phases and have not been tested**, so they remain genuinely unseen. Generalisation evidence so far comes from:
- the generated datasets;
- the public tips and gapminder data;
- sklearn's diabetes and breast-cancer data;
- the adapters' held-out domains (education, real_estate for v1; airline, farm, charity, carrental for v3).

Claims of "works on any dataset" should wait for those ten datasets to be run.

---

## Part 21 — Glossary

| Term | Technical definition | Beginner explanation | Example in Silt |
|---|---|---|---|
| LLM | A large neural network trained to predict the next token of text | A program that has read a lot of text and continues text sensibly | Qwen3-4B-Instruct-2507 |
| Transformer | A neural architecture built from self-attention and feed-forward layers | The design inside modern LLMs | 36 blocks in Qwen3-4B |
| Token | A unit of text from the tokenizer's vocabulary | A word piece | "Rahul" = 1 token; "300" = 3 tokens |
| Token ID | The integer index of a token | A token's number | "Rahul" → 85136 |
| Embedding | A learned vector representing a token | A list of numbers capturing meaning | 2,560 numbers per token in Qwen3-4B |
| Attention | A weighted mixing of information from other tokens, weights from query·key similarity | How a word "looks at" related words | "he" attending to "Rahul" |
| Parameters | The learned numbers of a model | The model's adjustable knobs | ~4.02 billion |
| Pretraining | Training from scratch on huge text by next-token prediction | Learning language in general | Done by Qwen's developers, not by Silt |
| Fine-tuning | Further training on a small task dataset | Teaching a specific job | v1/v2/v3 adapters |
| LoRA | Low-rank adaptation: learn ΔW = B·A while W stays frozen | Sticky notes on a textbook | rank 8; 7.34 M trainable parameters |
| Adapter | The file holding the LoRA matrices | The sticky notes | `adapters.safetensors`, ~29 MB |
| Quantisation | Storing weights in fewer bits | Compressing the model | 4-bit, group size 64 |
| Inference | Running a trained model to produce output | Using the model | every model call on Send |
| Prompt | The input text given to the model | The instructions plus the question | `SYSTEM` + grounding + request |
| Context window | The maximum number of tokens the model can read at once | The model's working memory | 262,144 tokens possible; Silt uses ~800–2,100 |
| Temperature | A setting controlling randomness in choosing tokens | Creativity level | 0 for plans; 0.3 for the repair |
| Logits | Raw scores for each possible next token, before softmax | How strongly each word is "voted" for | 151,936 per step |
| Loss | A number measuring how wrong the predictions are | The error score | cross-entropy; v1 validation 3.457 → 0.033 |
| Gradient descent | Updating parameters against the loss gradient | Walking downhill on the error | Adam, learning rate 1e-4 |
| Backpropagation | Computing gradients layer by layer with the chain rule | Finding who is to blame for the error | inside `mlx_lm.lora` |
| Optimiser | The rule that turns gradients into updates | The update strategy | Adam |
| Epoch | One full pass over the training set | One reading of the whole book | v1: 600 iterations × 4 examples ≈ 0.6 epoch of 3,986 [INFERRED] |
| Batch size | Examples per update | How many examples are graded at once | 4 |
| Overfitting | Fitting the training data but not new data | Memorising instead of understanding | watched with the validation loss |
| Dataset | A collection of records | The uploaded table | `hr_attrition.xlsx` |
| Feature | An input column for a model | A clue | `overtime`, `satisfaction` |
| Target | The column a model predicts | The answer column | `attrition` |
| DataFrame | pandas' table object | A spreadsheet in Python | `df` in every analysis |
| Aggregation | Combining many values into one (sum, mean …) | Summarising | `op_aggregate` |
| Correlation | A measure of how two variables move together | "Go up together?" | Spearman in `associations.py` |
| Forecast | An estimate of a future value from history | A projection | `followup.forecast` |
| Prediction | A model's output for an unseen case | A guess from learned patterns | Random Forest for one case |
| Validation | Checking a model or plan against held-out data or rules | Testing before trusting | `nl_planner.check`; CV in `ml.py` |
| Provenance | A record of how a result was produced | The receipt for a number | `Result.provenance` |
| Semantic mapping | Linking words or columns to meanings | Knowing "takings" is money | `semantics.py`, `intent.py` |
| Orchestrator | The component that routes work to other components | The conductor | `orchestrator.py` |
| API | A defined way for programs to talk to each other | A menu of requests | `/api/analyst/ask` |
| Endpoint | One URL+method of an API | One menu item | `POST /api/analyst/ask` |
| Middleware | Code that runs around every request | A security guard at the door | CORS middleware |
| Serialization | Converting objects to a transferable format | Packing for shipping | the answer dict → JSON (`program._jsonable`) |
| JSON | A text format for structured data | Nested lists and labels | the model's plans |
| BM25 | A keyword-ranking formula | Smarter word counting | PDF search |
| IQR | Interquartile range Q3 − Q1 | The spread of the middle half | outlier rule |
| Holm correction | Adjusting p-values for many tests | Being stricter when testing many things | `associations._holm` |
| Cramér's V | Association strength between categories | 0 = unrelated, 1 = fully linked | overtime ↔ attrition 0.28 |
| Hypothetical | Not observed; computed under a stated change | "What if", not "what is" | scenario results |
| Greedy decoding | Always picking the most likely next token | No randomness | temperature 0 |
| Chat template | The special tokens marking roles in a conversation | Speaker labels | `<|im_start|>user … <|im_end|>` |
