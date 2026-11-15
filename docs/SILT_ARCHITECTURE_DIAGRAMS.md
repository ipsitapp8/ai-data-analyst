# Silt — Architecture and Execution Diagrams

All diagrams use Mermaid. Each one corresponds to code that was read during the audit. The source references are under each diagram; the explanations are in `SILT_COMPLETE_UNDER_THE_HOOD.md`.
Labels: **(offline)** means it happened in the past and does not run when the app runs; **(model on)** means it runs only when the local model server answers; **[GAP]** marks something not implemented.

---

## 1. Complete architecture

```mermaid
flowchart LR
  subgraph Browser
    U[Student's browser]
  end
  subgraph Frontend["Streamlit frontend :8501"]
    H[Home.py<br/>login, navigation]
    CP[pages/12_Copilot.py<br/>chat, queue, answers]
    AC[api_client.py<br/>HTTP + JWT]
    UI[style/copilot_ui.py<br/>Altair charts]
  end
  subgraph Backend["FastAPI backend :8000"]
    M[main.py<br/>app, upload, legacy routes]
    R[routers/analyst.py<br/>/api/analyst/ask]
    O[analyst/orchestrator.py<br/>routing + memory]
    D[deterministic paths<br/>scope, objectives, metrics,<br/>investigate, tools, ml, forecast]
    NP[nl_planner.py + intent.py<br/>model plans, checks]
    P[program.py<br/>executor]
    MEM[(in-process memory<br/>OrderedDict, 200 conv.)]
    I[ingest.py<br/>CSV/Excel reader]
    DOC[documents.py<br/>PDF text + BM25]
  end
  subgraph Model["MLX model server :8080 (local Mac only)"]
    Q[Qwen3-4B-Instruct-2507 4-bit<br/>+ LoRA v1 adapter]
  end
  DB[(SQLite app.db<br/>or Postgres)]
  FS[(DATA_DIR<br/>uploads, PDFs)]
  U --> H & CP
  CP --> AC --> R
  AC --> M
  CP --> UI
  M --> I --> FS
  M --> DB
  R --> DB
  R --> O
  O <--> MEM
  O --> D --> P
  O --> NP -->|HTTP /v1/chat/completions| Q
  NP --> P
  D -->|calc / v1 tool planner, model on| Q
  O --> DOC --> FS
```

Sources: `frontend/pages/12_Copilot.py`, `frontend/api_client.py` l. 770, `backend/app/routers/analyst.py` l. 88–104, `backend/app/analyst/orchestrator.py` l. 19–20, 49, 125, `training/scripts/serve_analyst_model.sh`. On Render (hosted), the "Model" box does not exist (`Dockerfile`, `start.sh`).

---

## 2. Send button → response (sequence)

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant Page as 12_Copilot.py (Streamlit)
  participant API as api_client.analyst_ask
  participant Router as routers/analyst.py ask()
  participant Orc as orchestrator.ask / _dispatch_turn
  participant Conv as conversation.resolve/record
  participant Scope as scope.establish
  participant Paths as deterministic paths
  participant Plan as nl_planner.plan (model on)
  participant LLM as MLX server (Qwen + v1)
  participant Exec as program.run
  participant Mem as memory dict

  User->>Page: types question, presses Send (st.chat_input)
  Page->>Page: enqueue() into session_state log, st.rerun()
  Page->>API: analyst_ask(dataset_id, message, conversation_id)
  API->>Router: POST /api/analyst/ask  (JWT Bearer)
  Router->>Router: validate AskIn, auth, load_table(dataset)
  Router->>Mem: memory_for("team:user:dataset:doc:conv")
  Router->>Orc: ask(message, df, memory)
  Orc->>Orc: model_available()? (1.5 s ping)  intake.check
  Orc->>Conv: resolve(message) (yes / option / correction / topic change)
  Conv-->>Orc: effective message or a reply
  Orc->>Scope: establish (named values, exclusions, month/year, inherited)
  alt a value fits two columns, or a concept fits two columns
    Orc-->>Router: clarification (options)
  else deterministic path matches (investigation, objectives, metric, ...)
    Orc->>Paths: run
    Paths->>Exec: program.run(steps) (pandas)
    Exec-->>Paths: Results + provenance
  else composite / per-unit / follow-up and model on
    Orc->>Paths: intent.rule_program (grounded? use it, no model)
    Orc->>Plan: plan(message, df, memory)
    Plan->>LLM: system SYSTEM + grounding + request (temperature 0)
    LLM-->>Plan: JSON text
    Plan->>Plan: parse + repair, check(), intent.check()
    opt rejected (once)
      Plan->>LLM: "Silt rejected that: <reasons>" (temperature 0.3)
      LLM-->>Plan: corrected JSON
    end
    Plan->>Exec: execute (validity select + steps)
    Exec-->>Plan: Results
  end
  Orc->>Orc: _finish: scope verification, labels, charts
  Orc->>Mem: scope, last_answer, objectives / planned / investigation
  Orc->>Conv: record(task state)
  Orc-->>Router: answer dict
  Router-->>API: JSON (answer, charts, details, notes, plan_source, elapsed_ms)
  API-->>Page: response
  Page->>User: markdown + Altair charts + "How this was calculated"
```

Sources: `12_Copilot.py` l. 67–77, 297–327; `api_client.py` l. 770–773; `routers/analyst.py` l. 88–104; `orchestrator.py` l. 49–99, 125–398, 517–583, 817–908; `conversation.py` l. 108, 197; `scope.py` l. 99; `nl_planner.py` l. 841–889, 971.

---

## 3. Dispatch order inside `_dispatch_turn` (simplified)

```mermaid
flowchart TD
  A[message] --> B{conversation about the previous answer?<br/>followup.intent}
  B -- yes --> Z1[explain / advise / why-group]
  B -- no --> C{chart request?}
  C -- yes --> Z2[chart path]
  C -- no --> D[scope.establish]
  D --> D1{ambiguous value?}
  D1 -- yes --> Q1[ask which column]
  D1 -- no --> E{investigation / finding N?}
  E -- yes --> Z3[investigate.py]
  E -- no --> F{concept fits 2+ columns?}
  F -- yes --> Q2[ask which column]
  F -- no --> G{objective?<br/>feasibility, absence,<br/>change target, compare scenarios}
  G -- yes --> Z4[objectives.py program]
  G -- no --> H{composite / per-unit / restricted /<br/>follow-up on open analysis?}
  H -- "yes, model on" --> H1[rule program if grounded,<br/>else model planner]
  H -- "yes, model off" --> H2[rule program if grounded]
  H1 & H2 -- no answer --> I
  H -- no --> I[refine last, metric catalogue,<br/>associations, comparisons, rank, forecast]
  I -- none --> J["router, calculator (model on),<br/>v1 tool planner (model on) / rules"]
  J -- nothing computed --> K["late model-planner hook (model on)"]
  K -- none --> L[specific fallback reply:<br/>what could not be connected]
```

Source: `orchestrator.py` l. 112–398. The 12-second model budget (`MODEL_TURN_BUDGET_MS`, l. 400) switches the later model paths to their rules after a slow planner attempt.

---

## 4. Data upload flow

```mermaid
flowchart TD
  U[Upload in Datasets page] --> E[POST /api/datasets/upload<br/>main.upload_dataset]
  E --> X{.csv / .xlsx / .xls ?}
  X -- no --> R1[HTTP 400<br/>PDFs go to Copilot documents]
  X -- yes --> S[stream to UPLOADS_DIR in 1 MB chunks<br/>size cap -> 413]
  S --> T[ingest.read_table]
  T --> T1[CSV: decode -> sniff delimiter -> decimal mark -><br/>header? -> skip bad lines -> clean columns -><br/>numbers as text -> dates]
  T --> T2[Excel: ExcelFile -> sheet -> find header row -><br/>clean columns -> types]
  T1 & T2 --> V{readable table?}
  V -- no --> R2[IngestError -> HTTP 400 with reason<br/>file deleted]
  V -- yes --> N{structurally non-standard<br/>or Excel?}
  N -- yes --> C[store as normalised UTF-8 comma CSV]
  N -- no --> K[keep file as uploaded]
  C & K --> P[profile + Dataset row + DatasetVersion row in DB]
  P --> Q[data-quality checks, fingerprint]

  subgraph PDF["PDF (Copilot)"]
    D1[POST /api/analyst/documents] --> D2[pypdf text per page<br/>scanned pages reported, no OCR]
    D2 --> D3[DATA_DIR/analyst_documents/team/id.pdf + .json]
  end
```

Sources: `main.py` l. 166–281; `ingest.py` l. 36–282; `documents.py` l. 42–118.

---

## 5. Model training pipeline (offline, historical)

```mermaid
flowchart LR
  subgraph Data["Training data (offline)"]
    T[build_analyst_sft*.py<br/>templates + Spider 'tune' questions<br/>+ narration + chat]
    J[(train.jsonl / valid.jsonl / test.jsonl<br/>chat messages: system, user, assistant)]
    T --> J
  end
  subgraph Train["mlx_lm.lora (offline)"]
    B[Base: Qwen3-4B-Instruct-2507 4-bit<br/>weights FROZEN]
    L[LoRA A,B rank 8 on 16 layers<br/>7.34 M trainable = 0.182%]
    F[forward pass] --> LOSS[cross-entropy on assistant tokens only<br/>mask_prompt = true]
    LOSS --> G[backprop -> gradients for A,B]
    G --> AD[Adam, lr 1e-4, batch 4]
    AD --> L
  end
  J --> F
  B --> F
  L --> F
  AD --> CK[checkpoints every 100 iters<br/>val loss every 100]
  CK --> OUT[adapters.safetensors ~29 MB]
  OUT --> EV[eval scripts -> training/reports]
  EV --> SEL{better than v1 overall?}
  SEL -- "v2: no (safety regressions)" --> KEEP[v1 stays active]
  SEL -- "v3: no (follow-ups, validity)" --> KEEP
```

Sources: `training/adapters/*/adapter_config.json`, `train.log`, `training/data/*/MANIFEST.json`, `training/reports/*.md`, `adapters/analyst_qwen3_4b_lora_v3/EVAL_COMPARISON.md`. v3 stopped at iteration 430 of 800; its final file equals the 400 checkpoint.

---

## 6. Model inference pipeline (model on)

```mermaid
flowchart LR
  Q[question] --> G[grounding: columns + roles + safe summaries<br/>+ derivable measures + named values<br/>+ date tokens + conversation state]
  G --> PR[messages: SYSTEM + grounding + request]
  PR -->|OpenAI client, HTTP| S[mlx_lm server]
  S --> TK[chat template + tokenizer<br/>text -> token IDs]
  TK --> TF[transformer forward pass<br/>base weights + B·A adapter]
  TF --> LG[logits over 151,936 tokens]
  LG --> DEC[greedy pick, temperature 0<br/>repeat up to 700 tokens]
  DEC --> TXT[JSON text]
  TXT --> PA[parse + one-meaning repairs]
  PA --> CH[check: ops, columns, values, types,<br/>units, stated numbers, dependencies]
  CH --> IN[intent.check: quantity, measure, grouping,<br/>denominator, entities, restriction]
  IN -- ok --> EX[program.run]
  CH & IN -- rejected --> RP{budget left?}
  RP -- yes --> S
  RP -- no --> FB[deterministic fallback]
```

Sources: `nl_planner.py` l. 175–258 (grounding), 306 (parse), 356 (check), 819 (`_chat`), 841 (plan); `intent.py` l. 279.

---

## 7. Calculation pipeline (`program.run`)

```mermaid
flowchart TD
  P[program: list of steps] --> V[validate: known ops, unique ids,<br/>inputs are earlier steps or 'data']
  V --> VS[_valid select: drop impossible records,<br/>keep + flag exact duplicates]
  VS --> S1[select: entities / except / period / where]
  S1 --> S2[derive: safe formula via calc.evaluate]
  S2 --> S3[aggregate: sum mean median min max count nunique<br/>overall, by column, or per period]
  S3 --> S4[ratio / rank / compare / project / anomalies / associate]
  S4 --> RS[Result: kind, value, unit, provenance, hypothetical]
  RS --> RD[render from results + Silt labels<br/>+ checklist + caveats]
  RD --> OUT[answer, chart specs, details]
```

Sources: `program.py` l. 103–461; `nl_planner.execute` l. 971, `render` l. 1090.

---

## 8. Prediction pipeline

```mermaid
flowchart TD
  Q[question] --> K{what kind?}
  K -- "forecast next period" --> F1[measure per period, eligible history]
  F1 --> F2{periods >= 8?}
  F2 -- no --> F3[historical average, labelled estimate]
  F2 -- yes --> F4[hold out last 25%:<br/>LinearRegression vs repeat-last-value]
  F4 --> F5{trend beats naive?}
  F5 -- yes --> F6[refit on all -> ml_prediction]
  F5 -- no --> F7[repeat last value, labelled]
  K -- "predict a column" --> M1[screen features: ids, dates, text, leakage r>=0.98]
  M1 --> M2[Pipeline: impute + one-hot + RandomForest 300 trees]
  M2 --> M3{one date column with >=10 dates?}
  M3 -- yes --> M4[time-ordered split + TimeSeriesSplit CV]
  M3 -- no --> M5[stratified random split + KFold CV]
  M4 & M5 --> M6[metrics vs baseline + permutation importance]
  K -- "can X reach target?" --> T1[completed so far] --> T2[project daily pace x days left] --> T3[compare with target -> verdict]
```

Sources: `followup.py` l. 460, 501–630; `ml.py` l. 42–324; `objectives.py` l. 177–252; `program.py` l. 212–296. [GAP] No seasonality and no forecast intervals.

---

## 9. Scenario simulation

```mermaid
flowchart TD
  Q["What if Rahul is absent for 10 days?"] --> O{open objective?}
  O -- "feasibility open" --> A1[params.absent_days = 10]
  A1 --> A2[days left 27 -> 17 for project and compare]
  A2 --> A3[re-run feasibility program]
  O -- "no objective" --> B1[absence program: team expected total<br/>with vs without the person]
  Q2["If fees rose 10%? / if South closed?"] --> S1[scenario step on df.copy:<br/>scale column or remove entity]
  S1 --> S2[Result hypothetical = True]
  S2 --> S3[program.run marks every result<br/>computed from it hypothetical]
  A3 & B1 & S3 --> OUT[answer labelled hypothetical;<br/>original data unchanged]
  OUT --> MEM[stored in memory.objectives / planned<br/>for 'compare with the previous scenario']
```

Sources: `objectives.py` l. 235–239 (absence days), 307 (`_absence`), 460 (`handle`), 593 (`_compare_scenarios`); `program.py` l. 298 (`op_scenario`), 450–461 (propagation). [GAP] "working days" are subtracted as calendar days.
