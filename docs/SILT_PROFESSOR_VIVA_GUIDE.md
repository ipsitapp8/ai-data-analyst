# Silt — Professor / Viva Preparation Guide

Answers rely on facts verified in `SILT_COMPLETE_UNDER_THE_HOOD.md` (source references are there) and `SILT_KNOWN_LIMITATIONS_AND_ROADMAP.md`. **B** = basic, **I** = intermediate, **A** = advanced. When an answer depends on a number, the number is the one measured in the repository; don't round it up in the viva.

---

## Prepared explanations

### 2-minute project explanation

Silt is a conversational data analyst. You upload a spreadsheet or a PDF and ask questions in plain English — "Can Rahul complete 300 working hours this month?" or "Which department has the highest attrition and why?" — and Silt answers with tables, charts and an explanation of exactly how each number was computed.

The key design decision is that **the language model never calculates**. All numbers come from Python and pandas. Most questions are handled by deterministic rules. When the wording is unusual, a local Qwen3-4B model, fine-tuned with LoRA, writes a *plan* as JSON. Silt checks that plan against the real columns and against what was actually asked, runs it, and writes the answer from the computed results. So the model's job is understanding language, and Python's job is being correct.

It also investigates unfamiliar datasets on its own, forecasts, trains Random Forest models with honest baselines, and simulates "what if" scenarios. Every answer shows its records, exclusions and assumptions.

### 5-minute technical explanation

1. **Stack.** Streamlit frontend, FastAPI backend, SQLite/Postgres via SQLAlchemy, and an MLX model server on the Mac serving Qwen3-4B-Instruct (4-bit) plus a LoRA adapter.
2. **Request path.** The Copilot page posts the question to `/api/analyst/ask`. The router loads the dataset with a robust reader (encoding, delimiter, header, numbers as text, dates) and fetches the conversation's memory. The orchestrator routes the message:
   - pending clarifications and confirmations first;
   - then filters named in the question;
   - then specialised deterministic analyses — investigation, target feasibility, metrics, comparisons, forecasts;
   - the model only if those don't fit.
3. **Analysis programs.** A small typed language (select, derive, aggregate, ratio, rank, project, compare, scenario, anomalies, associate, profile), executed with pandas. Every result carries its provenance and a hypothetical flag.
4. **Model planning.** The model gets a grounded prompt: columns, semantic roles, safe summaries, date tokens and conversation state, never row data. It returns JSON. Silt repairs only slips with one meaning, validates columns, types, units and stated numbers, then checks semantically that the plan computes the requested quantity, denominator, grouping and every part of a composite question. One repair is allowed, within a time budget; otherwise a deterministic fallback answers.
5. **Training.** Three LoRA adapters were trained with `mlx_lm.lora` (rank 8, 7.34 M trainable parameters = 0.18%, Adam, learning rate 1e-4, batch 4) on 3–5 thousand chat examples. v1 is active. v2 and v3 regressed on important behaviours in held-out evaluations.
6. **Results.** On 49 evaluation cases the v1 configuration passed 36 against 23 with the model off. The remaining failures — two silently dropped filters among them — are documented with fixes.

### 10-minute architecture explanation (outline to speak from)

1. *Problem and design principle* (1 min): language understanding vs numerical correctness; "the model plans, Python computes".
2. *Components* (2 min): walk through diagram 1 in `SILT_ARCHITECTURE_DIAGRAMS.md` — browser, Streamlit, API client, FastAPI router, orchestrator, deterministic paths, program executor, model planner, MLX server, database, file storage.
3. *One request end to end* (2 min): diagram 2. Name the functions: `st.chat_input` → `analyst_ask` → `routers/analyst.ask` → `orchestrator.ask` → `_dispatch_turn` → path → `program.run` → `_finish` → `conversation.record` → `show_answer`.
4. *Understanding data* (1 min): ingestion detection, semantic roles from values first, and why IDs aren't averaged.
5. *The model* (2 min): tokens (show "300" → 3 tokens), prompt grounding, greedy decoding, JSON contract, validation layers, why the numbers never come from the model. LoRA: rank-8 B·A on frozen weights; v1/v2/v3 history.
6. *Analytics* (1 min): feasibility as a pace projection, scenarios on copies with hypothetical propagation, Random Forest with time-aware validation and baselines, associations with Holm correction.
7. *Honesty* (1 min): evaluation numbers, the known Critical bugs (dropped relative-period and adjective filters), and the roadmap.

### For a non-technical professor

Think of Silt as a careful junior analyst with a calculator. When you ask a question, it first works out which rows and columns you mean, then does the arithmetic with ordinary software that never makes arithmetic mistakes. It shows its working: which records it used, which it left out and why. For unusually worded questions it consults a small AI model, but only to understand the wording. The AI writes a to-do list, Silt checks the list makes sense, and then Silt does the calculation itself. If something is unclear, it asks you instead of guessing.

### For an AI/ML professor

Silt is a **neuro-symbolic** analytics system with a constrained-generation contract enforced after decoding:
- A 4-bit Qwen3-4B with a rank-8 LoRA adapter (16 layers, 7.34 M trainable parameters; masked-prompt SFT on about 4k templated and Spider-derived chat examples; validation loss 3.46 → 0.03) maps natural language to a typed DSL.
- The DSL is validated syntactically, against the schema (types, units, value existence, provenance of numbers) and semantically (quantity, denominator, grouping, entity, restriction, part coverage), then executed deterministically.
- Grounding deliberately excludes row-level data.
- Fallbacks are deterministic. A per-turn inference budget bounds latency.
- Evaluation is model-off vs adapter vs base-model on 49 programmatically scored cases with pandas ground truth. The adapter, trained on a different output format, underperforms the base model in plan validity when resources allow — a format-transfer limitation of narrow SFT.
- Statistical outputs use rank-based association measures with Holm correction. Forecasts are accepted only if they beat a naive baseline on held-out periods. The Random Forest uses forward-in-time validation and permutation importance.

---

## 1. Project overview

**Q1 (B). What is Silt in one sentence?**
A conversational data analyst that answers plain-English questions about an uploaded table or PDF, computing every number with deterministic Python and using a local language model only to understand unusual wording.

**Q2 (B). What problem does it solve?**
Non-programmers can't easily ask new questions of their data; dashboards only answer questions someone anticipated. Silt builds the analysis on demand and explains it.

**Q3 (B). Who are the users and what do they upload?**
Business users and students. They upload CSV or Excel files as datasets, and PDFs as documents. The PDFs are only searched and quoted, never used for statistics.

**Q4 (I). Why not just use ChatGPT on the spreadsheet?**
Three reasons. (1) A general chatbot generates numbers as text and can hallucinate. Silt computes them, and any model-written prose is checked against the computed numbers. (2) Silt shows provenance: the records used, the exclusions with file line numbers, the formulas. (3) the model runs locally (no external AI API), so the data is never sent to an AI provider.

**Q5 (I). What makes Silt more than a set of if-statements?**
The analysis-program language composes operations. Requests the rules don't cover reach the model planner, whose JSON plans are validated and executed. The investigation chooses analyses from what the columns *mean*. And the conversation state lets later questions build on earlier results.

**Q6 (A). What is the single most important design principle, and what does it cost?**
"The model plans, Python computes." It buys correctness and auditability. It costs flexibility: any question the plan language or the rules can't express gets a clarification or a refusal rather than a fluent guess.

## 2. Software architecture

**Q7 (B). Name the main components.**
Streamlit frontend; FastAPI backend; the analyst engine (orchestrator, scope, semantics, intent, program executor, planners, objectives, investigation, ML, forecasting, documents); an SQLAlchemy database; file storage; an MLX model server.

**Q8 (B). What is the orchestrator?**
`backend/app/analyst/orchestrator.py`. It receives each message, resolves it against the conversation, establishes the scope, tries the paths in a fixed order, verifies the answer against the scope, and updates the memory.

**Q9 (I). Why is the order of paths important?**
The first path that returns an answer wins. Deterministic, well-tested paths come first, so they never get replaced by a slower, less reliable model answer. Composite and per-unit questions are routed to the program planner *before* single-purpose handlers that would answer only half of them.

**Q10 (I). How do the frontend and backend communicate?**
The Streamlit server calls FastAPI over HTTP with `requests`, sending a JWT Bearer token (`api_client.py`). The browser only talks to Streamlit.

**Q11 (A). Why keep the model server separate from the backend?**
Isolation and swappability. The backend speaks the OpenAI-compatible HTTP protocol, so the model can be restarted, changed (v1 adapter or base model) or moved to another machine via `ANALYST_MODEL_URL`/`ANALYST_PLANNER_URL` without touching backend code. A model crash can't crash the API.

## 3. Python and backend

**Q12 (B). What does FastAPI do here?**
It defines the HTTP endpoints, validates request bodies with Pydantic (`AskIn`: message 1–1000 characters), injects dependencies (database session, current user, team) and serialises responses to JSON.

**Q13 (B). How are passwords and sessions protected?**
Passwords are hashed with bcrypt; logins return a JWT signed with `JWT_SECRET_KEY`. Every analyst route requires a valid user and team.

**Q14 (I). Where is the conversation memory and what happens on restart?**
In a process-local `OrderedDict` of at most 200 conversations, keyed by team:user:dataset:document:conversation. It is lost on restart. That's a known limitation, and persisting it is on the roadmap.

**Q15 (I). How does the system avoid executing arbitrary code from the model?**
The model never sends code. Programs are JSON with a closed set of operations and arguments. Formulas inside `derive` are parsed with Python's `ast` module and only whitelisted node types and functions are allowed (`calc._Eval`, `FUNCTIONS`). There is no `eval`/`exec` of model text.

**Q16 (A). What happens if a request takes too long?**
The frontend waits up to 300 s. Each model call has a 20 s timeout. Planning has a 30 s budget (the repair is skipped if there isn't time). After 12 s of model time in a turn, the later model paths switch to their deterministic rules.

## 4. Data ingestion

**Q17 (B). Which file types are supported?**
CSV, XLSX and XLS as datasets; PDF as documents.

**Q18 (I). How does Silt detect a CSV's delimiter?**
It tries comma, semicolon, tab and pipe on the first 400 non-empty lines and picks the one that splits rows into the same number of fields most consistently (`ingest.sniff_delimiter`). Below 60% consistency the file is refused as "may not be a table".

**Q19 (I). What if numbers are stored as text like "$1,200" or "12%"?**
`ingest._to_number` converts a column when at least 95% of its values parse: currency symbols, thousands separators and European formats are handled, and it's reported. Columns named like IDs, codes, zip or phone are never converted.

**Q20 (I). Are missing values filled in?**
No. They stay missing (NaN), and each analysis excludes rows missing a value it uses, saying how many. The only imputation is inside the Random Forest pipeline (median or most frequent), fitted on the training rows only.

**Q21 (A). How are duplicates handled, and what is inconsistent?**
They stay in the file. Exact duplicates are excluded by some paths (target feasibility) and kept-and-flagged by others (calculator totals). That's why "Rahul's total hours" (26 h) and the feasibility "completed so far" (18 h) disagree. Unifying the policy is quick fix Q2.

**Q22 (I). What does Silt do with a scanned PDF?**
It reports the scanned pages: OCR is not implemented. Text PDFs are split into passages of about 700 characters and searched with BM25 keyword ranking, with no embeddings.

## 5. pandas and calculations

**Q23 (B). What is a DataFrame?**
pandas' in-memory table: named columns, each with a type, and an index of rows. Every analysis in Silt operates on one.

**Q24 (B). Difference between count of rows and count of non-null values?**
Rows count every record; non-null counts only the records with a value in that column. [5, NaN, 7] has 3 rows but 2 non-null values.

**Q25 (I). Why do denominators matter? Give an example.**
5 trips: driver A 10, 10, 100 km (one late), driver B 40, 40 km. Late rate as a ratio of totals = 1/5 = 20%. The average of the drivers' own rates = (33.3% + 0%)/2 = 16.7%. The first weights every trip equally; the second over-weights B's two trips. "Per driver" means total km / distinct drivers = 200/2 = 100, not the mean per trip (40).

**Q26 (I). Why does Silt refuse to average an ID column?**
An ID is a label that happens to be numeric; its average is meaningless. The planner check rejects mean/sum/min/max of identifiers and suggests count or nunique.

**Q27 (I). What is provenance and why does it matter?**
Each result records its operation, input columns, formula, rows used and excluded (with file lines), unit and hypothetical status. It lets a user, or a professor, audit every number.

**Q28 (A). How is a per-group ratio computed, and what about groups with zero?**
`program.op_ratio` divides two per-group results key by key. Groups missing from either side, or with a zero denominator, get no value and are listed.

**Q29 (A). How are filters combined?**
In the question scope (`scope.mask`), values in one column are ORed; different columns are ANDed; exclusions are AND NOT; months and years are ANDed. Program `where` conditions are ANDed. OR across different columns isn't expressible: a limitation.

## 6. Machine learning

**Q30 (B). Which ML algorithm does Silt use?**
Random Forest (scikit-learn): a classifier for categorical targets, a regressor for numeric ones. 300 trees, `min_samples_leaf=2`, and balanced class weights for classification.

**Q31 (I). How is it evaluated?**
A held-out test split (time-ordered when there is a date column with ≥ 10 dates), plus 5-fold cross-validation, against a naive baseline: the majority class for classification, the training mean for regression. The report says whether the model beats that baseline, so a model no better than guessing is shown as such.

**Q32 (I). What is data leakage and how does Silt prevent it?**
Leakage is a feature that wouldn't be known at prediction time or nearly copies the target, making test scores falsely high. Silt removes identifiers, dates, free text and any feature with |correlation| ≥ 0.98 to the target. It fits preprocessing on the training rows only and uses forward-in-time splits for dated data.

**Q33 (I). What does feature importance mean here?**
Permutation importance on the held-out rows: how much the score drops when a feature's values are shuffled. It shows what the model *relied on*, not what *causes* the outcome, and every answer says so.

**Q34 (A). Why balanced accuracy instead of accuracy?**
With 89% "No" attrition, always predicting "No" scores 89% accuracy but learns nothing. Balanced accuracy averages the recall of each class, so that useless model scores 50%.

## 7. Transformers and LLMs

**Q35 (B). What is a token? Give a Silt example.**
A piece of text from the model's vocabulary. "Can Rahul complete 300 working hours this month?" becomes 12 tokens; "300" is three tokens (3, 0, 0).

**Q36 (B). What are parameters? How many does Qwen3-4B have?**
The learned numbers in the network: about 4.02 billion.

**Q37 (I). What is self-attention?**
Each token builds a query; every token a key and a value. The weights are softmax(q·k/√d), and the token's new representation is the weighted sum of the values. That's how "he" can draw information from "Rahul".

**Q38 (I). What is the context window and how much does Silt use?**
The maximum number of tokens the model reads at once: 262,144 for this architecture. Silt's program-planner prompt for the timesheet is about 2,057 tokens; the v1 tool-planner prompt about 792.

**Q39 (I). What are logits and temperature?**
Logits are the raw scores for each of about 152k possible next tokens. Softmax turns them into probabilities. Temperature 0 always picks the most likely token (greedy), which makes Silt's plans repeatable. The single repair attempt uses 0.3 so it doesn't repeat the rejected plan word for word.

**Q40 (A). Why is the model bad at arithmetic, from a tokenisation point of view?**
Numbers are split into digit tokens and the network predicts text, not values. Exact multi-digit arithmetic over many rows isn't what next-token prediction optimises, so Silt never asks it to compute.

## 8. Qwen

**Q41 (B). What does "Qwen3-4B-Instruct-2507-4bit" mean?**
Qwen family, third generation, about 4 billion parameters, instruction-tuned, July 2025 release, weights quantised to 4 bits (group size 64) for Apple's MLX.

**Q42 (I). Why a 4B model and not a larger one?**
It runs locally on a 16 GB Mac (privacy, no API cost) at a few seconds per plan. Bigger models wouldn't fit alongside the app; even two 4B processes caused swapping in the evaluation.

**Q43 (A). What does quantisation trade off?**
Memory and speed against a small loss in accuracy: 4-bit weights use about a quarter of the memory of 16-bit ones. Silt's validation layer compensates for occasional output mistakes.

## 9. LoRA and fine-tuning

**Q44 (B). Did you train Qwen from scratch?**
No. Silt uses the already-pretrained, instruction-tuned Qwen3-4B and only fine-tuned small LoRA adapters on top.

**Q45 (B). What is fine-tuning?**
Continuing training on a small task-specific dataset, so the model learns a behaviour — here, answering with Silt's JSON plans — without relearning language.

**Q46 (I). Explain LoRA with numbers.**
Freeze the weight matrix W and learn ΔW = B·A with a small rank r. For a 2,560×2,560 matrix (6.5 M numbers), rank 8 needs 2×2,560×8 = 40,960 numbers. Silt's adapter trains 7.34 M parameters out of 4,022 M (0.182%), stored in a 29 MB file.

**Q47 (I). What were Silt's training settings?**
`mlx_lm.lora`; rank 8, scale 20, dropout 0; adapters on 16 layers; Adam with learning rate 1e-4 (constant); batch 4; 600 iterations (v1, v2) and 800 configured for v3; max sequence length 2,400; prompt masked so the loss is only on the assistant's tokens; gradient checkpointing.

**Q48 (I). What did a training example look like?**
A chat with three messages: system (the JSON contract and tool list), user (the dataset schema plus a question), assistant (the correct JSON plan, e.g. `{"type":"tools","intent":"metric","calls":[{"tool":"compute_metric","args":{"aggregation":"count"}}]}`).

**Q49 (A). Why is v1 active rather than v2 or v3?**
- **v2** improved tool selection (1.000) but regressed on safety behaviours: regression direction 0.50 against v1's 0.90; declining when advertising columns are absent 0.85 against 1.00.
- **v3** improved refusals (36/36) but its overall plan validity fell to 0.63 (v1 0.74, base 0.90) and follow-ups to 7/30 (v1 16/30). It also stopped at 430 of 800 iterations; its file is the 400 checkpoint.
- **v1** was the best balance.

**Q50 (A). The validation loss was 0.03. Doesn't that mean it's nearly perfect?**
No. A low loss on templated validation examples shows it fits that distribution. Behaviour on new phrasings and formats is measured by the task evaluations, where v1's program-planning metric correctness was 7/9–10/12, and the base model did better on the newer program format. Loss and task quality are different things.

**Q51 (A). Would training a new adapter fix the remaining failures?**
Mostly not. The Critical bugs (dropped filters) and the wrong-metric bug happen on deterministic paths that never call the model. A new adapter makes sense only for the program format: trained on the base model, with new held-out evaluation, and accepted only if it beats v1 *and* the base model.

**Q52 (B). What is the difference between training and inference? Does pressing Send train the model?**
Training changes weights using examples, a loss, gradients and an optimiser; it was done offline in the past. Inference only runs the fixed model on a prompt. Pressing Send performs inference only: no weights change. Silt's "memory" is a Python dictionary, not the model.

## 10. Inference and prompt engineering

**Q53 (I). What does the planner's prompt contain?**
- **System:** the JSON contract (program / clarify / unsupported / rerun / compare_scenarios / finding), the operation list, rules, and three worked examples on fictional schemas.
- **User:** the columns with roles and safe summaries, derivable measures, values named in the request, date tokens like `$days_left_month`, the conversation state (earlier analyses, values, findings), then the request.

**Q54 (I). Why date tokens instead of dates?**
So the model never does date arithmetic. Silt computes `$month_start`, `$days_left_month` and so on and substitutes them, which removes a whole class of numeric errors.

**Q55 (I). What does Silt do if the model returns invalid JSON?**
It repairs only slips with one meaning (mismatched brackets, a missing `{` before a step) and records them. Otherwise the model sees the parse error once. If that also fails, the deterministic paths answer.

**Q56 (A). How does Silt detect a plan that is valid but answers the wrong question?**
`intent.read` reads the request's requirements from language and the schema — quantity (rate/average/median/total/count), measure, grouping, denominator, entities, restrictions, and the why/unusual/what-if parts. `intent.check` rejects a plan that computes something else, e.g. a count where a rate was asked or a division by records where "per agent" was asked. The parts with one deterministic form (associations for "why", unusual values) are added and stated.

**Q57 (A). How are hallucinated numbers prevented in model prose?**
`explain.ungrounded` extracts every number in the model's text and checks that each matches a computed value (with rounding tolerance) or a number in the question. Prose that fails is not shown. Program answers are written by Silt, with labels built from the operations, not from the model's step names.

## 11. Prediction and forecasting

**Q58 (B). Is "Can Rahul reach 300 hours?" a machine-learning prediction?**
No. It's a pace projection: hours completed so far, plus the historical daily average times the days left, with a ±1.96·sd·√days range, compared with the target. In the demo: 18 h done, 6 h/day, 27 days left, expected 180 h (126–234), required 10.44 h/day → "unlikely on the current pace".

**Q59 (I). How does forecasting work?**
The measure is aggregated per period. With at least 8 periods, the last 25% are held out and a linear trend on the period number competes with "repeat the last value". The trend is used only if it wins. With fewer periods, the historical average is reported as an estimate. There is no seasonality and no forecast interval.

**Q60 (I). How does a "what if" scenario avoid changing the data?**
Scenario steps work on `df.copy()`. The result is flagged hypothetical, and `program.run` flags everything computed from it. The answer says it is hypothetical and not observed.

**Q61 (A). What exactly does "absent for 10 days" do to the feasibility calculation?**
It subtracts 10 from the days left (27 → 17) for both the projection and the required pace: expected 18 + 6×17 = 120 h, required 282/17 = 16.59 h/day. It subtracts calendar days even though the text says "working days", a known wording/computation mismatch.

## 12. Statistical analysis

**Q62 (B). Why does correlation not prove causation?**
Two variables can move together because a third causes both, because the effect runs the other way, or by chance. Silt reports associations and always says they are not causes.

**Q63 (I). Which association measures does Silt use, and when?**
Spearman for number vs number (rank-based, robust to outliers); eta squared with a Kruskal–Wallis p-value for category vs number; Cramér's V from a chi-square table for category vs category (numbers binned into ≤ 5 ranges).

**Q64 (I). What is the Holm correction and why use it?**
Testing many columns at once makes some look significant by chance. Holm sorts the p-values and multiplies the k-th smallest by (m − k + 1), keeping the order, which controls the chance of any false finding.

**Q65 (I). How are outliers found? Are they deleted?**
Values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR], with "extreme" beyond 3·IQR. They are reported with file lines, never deleted: unusual isn't the same as wrong.

**Q66 (A). How does the investigation decide what is worth reporting?**
Evidence bars:
- associations with adjusted p < 0.05 and strength ≥ 0.1 (top 3);
- rate differences ≥ 5 points between groups of at least 5 records;
- mean ratios ≥ 1.25;
- top-entity share ≥ 15%;
- unusual values only if extreme, ≥ 2% of values, or ≥ 2% of the total;
- concentration ≥ 70%.

A pure-noise column is not reported; a test checks this.

## 13. Evaluation

**Q67 (B). How many tests does the project have, and do they pass?**
Backend 1,612 passed (31 skipped); the analyst subset 653; frontend 326; plus browser tests (10/10 steps on the final code).

**Q68 (I). What does "36 of 49 evaluation cases passed" mean?**
On 49 deliberately difficult cases (new domains, paraphrases, composite, ambiguous and impossible requests, conversations), the v1 configuration produced the pandas-verified numbers or the correct clarification/refusal in 36. Model off: 23. Base model as planner: 31 (hurt by timeouts on a 16 GB machine). It is not a general accuracy figure.

**Q69 (I). What is the difference between a false positive and a false negative here, and which is worse?**
A false positive is a confident wrong answer (e.g. a filter silently ignored). A false negative is an honest "I couldn't answer". False positives are worse because users trust them. Silt's design prefers to refuse or ask.

**Q70 (A). Have the ten held-out datasets been tested?**
No. They weren't on disk during the reliability phases, so they remain genuinely unseen. Generalisation evidence so far comes from generated datasets, public datasets (tips, gapminder, sklearn diabetes and breast cancer) and the adapters' held-out domains. Running those ten once, frozen, is roadmap item R5.

## 14. Limitations and future scope

**Q71 (B). What are the main weaknesses?**
- Two filter types are silently ignored on deterministic paths: relative periods ("last 30 days") and "only escalated" style adjectives.
- One wrong-metric case: an entity's share.
- An inconsistent duplicate policy, and a money column misread as a rate.
- Limited synonym understanding, latency for unusual questions, and memory that doesn't survive restarts.
- No model on the hosted deployment; shallow PDF support; basic forecasting.

**Q72 (I). What would you fix first and why?**
The silently ignored filters, because they produce confident wrong numbers. The fix is a post-check that every restriction recognised in the question was applied to the records used, plus parsing relative periods. It's code only, 1–2 days, and needs no model training.

**Q73 (A). How would you improve the architecture long-term?**
- Persist conversation state.
- Unify the restriction and duplicate policies across all paths.
- Train a planner adapter for the program format (on the base model, with a frozen held-out evaluation) or use a stronger model on adequate hardware.
- Add OCR, table extraction and hybrid retrieval for PDFs.
- Add seasonal baselines and backtested intervals to forecasting.
- Use grammar-constrained decoding if the inference server supports it.
- Run the frozen held-out datasets in CI.
