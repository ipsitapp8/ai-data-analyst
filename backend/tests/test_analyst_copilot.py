"""Silt Copilot additions to the analyst: dataset understanding, data quality,
outliers, trends, general scenarios, ML options and metrics, PDF documents,
charts, sheet selection, and the labelling of routed answers."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from platform_helpers import workspace

from app.analyst import documents, ml, orchestrator, overview, planner, present, router, scenario
from app.analyst.business import Insufficient


def sales_frame(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for region, lift in (("North", 1.0), ("South", 0.9), ("West", 1.25)):
        for m in pd.date_range("2025-01-01", periods=12, freq="MS"):
            for _ in range(3):
                spend = 1000 * rng.uniform(0.8, 1.2)
                leads = int(spend / 10)
                rows.append({"region": region, "month": m, "units": int(rng.normal(100, 10) * lift),
                             "sales": float(rng.normal(5000, 400) * lift), "marketing_spend": spend,
                             "leads": leads, "conversions": int(leads * 0.03 * lift) + 1,
                             "monthly_fee": float(rng.normal(50, 5))})
    df = pd.DataFrame(rows)
    df.loc[3, "units"] = np.nan
    df.loc[7, "sales"] = 60000.0
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def attrition_frame(seed: int = 5, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    overtime = rng.choice(["Yes", "No"], n, p=[.35, .65])
    sat = rng.integers(1, 6, n)
    income = rng.normal(4500, 900, n)
    logit = -1.5 + 2.2 * (overtime == "Yes") - 0.9 * (sat - 3) - 0.0005 * (income - 4500)
    left = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"employee_id": [f"E{i}" for i in range(n)], "overtime": overtime, "satisfaction": sat,
                         "monthly_income": income.round(0), "attrition": np.where(left, "Yes", "No")})


S = sales_frame()
A = attrition_frame()


def ask(msg, df, mem=None, **kw):
    return orchestrator.ask(msg, df, {} if mem is None else mem, use_model=False, **kw)


# ------------------------------------------------------------- understanding --

def test_understand_reports_shape_quality_targets_and_suggestions():
    r = overview.understand(S, "sales.csv")
    assert r["rows"] == len(S) and r["columns"] == S.shape[1] and r["duplicates"] == 1
    assert r["kinds"]["datetime"] == 1 and r["missing_cells"] == 1
    assert any("duplicated" in q for q in r["quality_issues"])
    assert any("sales" in q and "unusual" in q for q in r["quality_issues"])
    assert r["suggested_questions"] and "region" in r["suggested_questions"][0]
    assert {t["column"] for t in r["possible_targets"]} >= {"sales"}


@pytest.mark.parametrize("q, tool", [
    ("Here is my sales dataset. Understand it.", "understand"), ("Summarize the dataset", "understand"),
    ("What is in this dataset?", "understand"), ("What should I investigate next?", "understand"),
    ("Are there missing values?", "missing_values"), ("Find outliers in sales", "outliers"),
    ("Find unusual patterns", "patterns"), ("Show monthly trends of sales", "trend"),
    ("Can we increase sales by 20%?", "column_change"),
    ("Assume conversion improves from 3% to 5%. Calculate the expected revenue.", "conversion_change"),
])
def test_router_recognises_requests_the_planner_has_no_tool_for(q, tool):
    plan = router.route(q, S)
    assert plan is not None and plan["type"] == "router" and plan["tool"] == tool


@pytest.mark.parametrize("q", ["What is the average sales by region?", "Which region has the highest sales?",
                               "What if we increase the budget by 20%?", "Predict sales", "Hello."])
def test_planner_requests_are_not_routed(q):
    assert router.route(q, S) is None, "these stay with the fine-tuned planner"


def test_column_names_do_not_trigger_wording_rules():
    df = A.rename(columns={"monthly_income": "monthly_income"})
    plan = router.route("What if monthly income increases by 10%?", df)
    assert plan["tool"] == "column_change" and plan["args"]["column"] == "monthly_income"


def test_routed_answers_are_labelled_and_charted():
    r = ask("Show monthly trends of sales", S)
    assert r["plan_source"] == "router" and "Silt tool router" in r["notes"][0]
    assert r["charts"] and r["charts"][0]["type"] == "line" and len(r["charts"][0]["data"]) == 12
    tr = r["calls"][0]["result"]
    assert tr["first"]["period"] == "2025-01" and tr["last"]["period"] == "2025-12"


def test_trend_without_a_date_column_is_declined_not_invented():
    r = ask("Show the monthly trend of monthly_income", A)
    assert r["plan"]["type"] == "decline" and "no date column" in r["answer"]


def test_outliers_and_missing_values_are_computed():
    o = overview.outliers(S, "sales")
    assert o["columns"][0]["count"] >= 1 and any(e["value"] == 60000.0 for e in o["columns"][0]["examples"])
    m = overview.missing_values(S)
    assert m["missing_cells"] == 1 and m["duplicates"] == 1


# ------------------------------------------------------------------ scenarios --

def test_column_change_is_arithmetic_with_assumptions():
    r = scenario.column_change(S, "sales", 20)
    assert r["hypothetical"]["total"] == pytest.approx(S["sales"].sum() * 1.2)
    assert r["difference"]["total"] == pytest.approx(S["sales"].sum() * 0.2)
    assert r["assumptions"] and "not an observed result" in r["caveat"]
    text = present.scenario(r)
    assert "observed (baseline)" in text and "hypothetical" in text


def test_conversion_scenario_uses_observed_leads_and_revenue_per_conversion():
    r = scenario.conversion_change(S, 0.05, 0.03)
    leads, conv, rev = S["leads"].sum(), S["conversions"].sum(), S["sales"].sum()
    assert r["hypothetical"]["conversions"] == pytest.approx(leads * 0.05)
    assert r["hypothetical"]["revenue"] == pytest.approx(leads * 0.05 * rev / conv)


def test_rate_change_needs_its_unit_and_then_computes():
    plan = router.route("What if churn decreases by 5%?", A)
    assert plan["type"] == "clarify" and "percentage points" in plan["question"]
    plan = router.route("What if attrition decreases by 5 percentage points?", A)
    assert plan["tool"] == "rate_change" and plan["args"]["unit"] == "points"
    r = router.run(plan, A, {})
    assert r["hypothetical"]["rate"] == pytest.approx(r["observed"]["rate"] - 0.05)
    with pytest.raises(Insufficient):
        scenario.rate_change(A, "attrition", "Yes", -5, "unknown")


def test_missing_assumptions_are_asked_for():
    assert router.route("What if we open another store in the West region?", S)["type"] == "clarify"
    assert router.route("What if conversion improves?", S)["type"] == "clarify"


def test_scenario_uses_the_trained_model_when_the_column_is_an_input():
    mem = {}
    ask("Predict attrition", A, mem)
    assert "model" in mem
    r = scenario.column_change(A, "monthly_income", 10, mem)
    assert "model_estimate" in r and r["model_estimate"]["task"] == "classification"
    assert "not causes" in r["model_estimate"]["caution"]


# ------------------------------------------------------------------ ML --

@pytest.mark.parametrize("msg, opts", [("use a 70/30 split", {"test_size": 0.3}), ("80/20 split, seed 7", {"test_size": 0.2, "seed": 7}),
                                       ("predict attrition", {}), ("30% test set", {"test_size": 0.3})])
def test_ml_options_from_words(msg, opts):
    assert router.ml_options(msg) == opts


def test_classification_reports_full_held_out_metrics():
    r = ml.public(ml.train(A, "attrition", test_size=0.3, seed=7))
    m = r["metrics"]
    assert r["test_rows"] == round(len(A) * 0.3) and r["split"] == {"test_size": 0.3, "train_share": 0.7, "seed": 7,
                                                                   "stratified": True}
    for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc",
              "baseline_accuracy", "baseline_balanced_accuracy"):
        assert k in m
    cm = m["confusion_matrix"]
    assert sum(map(sum, cm["matrix"])) == r["test_rows"]
    assert {c["class"] for c in m["per_class"]} == {"Yes", "No"}
    assert "employee_id" in {d["column"] for d in r["dropped"]}, "identifiers are left out"
    assert r["class_balance"]["counts"]["No"] + r["class_balance"]["counts"]["Yes"] == len(A)


def test_split_and_seed_make_training_reproducible():
    a = ml.public(ml.train(A, "attrition", test_size=0.2, seed=3))["metrics"]
    b = ml.public(ml.train(A, "attrition", test_size=0.2, seed=3))["metrics"]
    assert a == b
    with pytest.raises(ml.Unsuitable):
        ml.train(A, "attrition", test_size=0.5)


def test_regression_reports_mse_and_baseline():
    m = ml.public(ml.train(S, "sales"))["metrics"]
    assert {"mae", "mse", "rmse", "r2", "baseline_mae"} <= set(m)
    assert m["mse"] == pytest.approx(m["rmse"] ** 2)


def test_leakage_column_is_left_out():
    df = A.copy()
    df["attrition_flag"] = (df["attrition"] == "Yes").astype(int)
    r = ml.public(ml.train(df, "attrition"))
    assert any(d["column"] == "attrition_flag" and "leakage" in d["reason"] for d in r["dropped"])


def test_requested_split_reaches_the_planned_random_forest():
    mem = {}
    r = ask("Predict attrition with a 70/30 split, seed 7", A, mem)
    res = r["calls"][0]["result"]
    assert res["split"]["test_size"] == 0.3 and res["split"]["seed"] == 7
    assert any("Training options taken from your message" in n for n in r["notes"])
    assert "Confusion matrix" in r["answer"] and "Reading the metrics" in r["answer"]


def test_prediction_answer_is_a_readable_table_not_a_dict():
    mem = {}
    ask("Predict attrition", A, mem)
    r = orchestrator.tools.run({"tool": "predict", "args": {"rows": [{"overtime": "Yes", "satisfaction": 2,
                                                                       "monthly_income": 3000}]}}, A, mem)
    text = orchestrator.explain.render("predict", r)
    assert "{'" not in text and "| input | value |" in text and "held-out" in text


# ------------------------------------------------------------------ documents --

def _pdf_bytes(pages: list[str], scanned_page: bool = False) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    matplotlib.rcParams["pdf.fonttype"] = 42
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for text in pages:
            fig = plt.figure(figsize=(8, 10))
            for i, line in enumerate(text.split("\n")):
                fig.text(0.1, 0.9 - i * 0.04, line)
            pdf.savefig(fig)
            plt.close(fig)
        if scanned_page:
            fig = plt.figure(figsize=(8, 10))
            ax = fig.add_axes([0.1, 0.5, 0.8, 0.3])
            ax.imshow(np.random.default_rng(0).random((40, 120)))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


def test_pdf_pages_search_and_scanned_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    data = _pdf_bytes(["Quarterly review\nWest region sales reached 3.1 million.",
                       "Risks\nSupport staff turnover increased to 21 percent."], scanned_page=True)
    meta = documents.save(1, "review.pdf", data)
    assert meta["page_count"] == 3 and meta["scanned_pages"] == [3] and meta["text_pages"] == [1, 2]
    doc = documents.load(1, meta["id"])
    hits = documents.search(doc, "What happened to staff turnover?")
    assert hits and hits[0]["page"] == 2 and "turnover" in hits[0]["text"].lower()
    r = ask("What does the report say about staff turnover?", None, document=doc)
    assert r["plan_source"] == "document" and "page 2" in r["answer"]
    r = ask("Summarize the document", None, document=doc)
    assert "Page 3 is a scanned image" in r["answer"] and "OCR" in r["answer"]


def test_invalid_pdfs_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    with pytest.raises(documents.DocumentError):
        documents.save(1, "x.pdf", b"plain text pretending")
    with pytest.raises(documents.DocumentError):
        documents.save(1, "x.txt", b"%PDF-1.4")
    with pytest.raises(documents.DocumentError):
        documents.load(1, "../../etc/passwd")


def test_document_answer_drops_model_numbers_not_in_passages(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    doc = documents.load(1, documents.save(1, "r.pdf", _pdf_bytes(["Sales reached 3.1 million in the West."]))["id"])
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "West sales reached 4.7 million (p. 1).")
    r = orchestrator.ask("What were West sales according to the report?", None, {}, use_model=True, document=doc)
    assert "4.7" not in r["answer"] and any("not in the passages" in n for n in r["notes"])
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "West sales reached 3.1 million (p. 1).")
    r = orchestrator.ask("What were West sales according to the report?", None, {}, use_model=True, document=doc)
    assert r["answer"].startswith("West sales reached 3.1 million")


# ------------------------------------------------------------------ API --

def test_api_documents_preview_and_excel_sheet(client, tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    ws = workspace(client, csv=S.head(40).to_csv(index=False).encode(), filename="sales.csv")
    pv = client.get(f"/api/analyst/datasets/{ws['ds']}/preview", headers=ws["h"])
    assert pv.status_code == 200 and len(pv.json()["preview"]) == 20 and "region" in pv.json()["columns"]

    up = client.post("/api/analyst/documents", files={"file": ("r.pdf", _pdf_bytes(["Turnover rose to 21 percent."]),
                                                                "application/pdf")}, headers=ws["h"])
    assert up.status_code == 200, up.text
    bad = client.post("/api/analyst/documents", files={"file": ("x.pdf", b"nope", "application/pdf")}, headers=ws["h"])
    assert bad.status_code == 400 and "PDF" in bad.json()["detail"]
    assert [d["id"] for d in client.get("/api/analyst/documents", headers=ws["h"]).json()] == [up.json()["id"]]
    r = client.post("/api/analyst/ask", json={"document_id": up.json()["id"], "message": "What happened to turnover?",
                                              "use_model": False}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["plan_source"] == "document" and r.json()["dataset"] is None

    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        pd.DataFrame({"note": ["read me"]}).to_excel(xw, sheet_name="Read me", index=False)
        A.to_excel(xw, sheet_name="Employees", index=False)
    x = client.post("/api/datasets/upload", files={"file": ("hr.xlsx", buf.getvalue(), "application/octet-stream")},
                    data={"sheet": "Employees"}, headers=ws["h"])
    assert x.status_code == 200, x.text
    assert x.json()["row_count"] == len(A) and x.json()["filename"] == "hr.xlsx [Employees]"
    missing = client.post("/api/datasets/upload", files={"file": ("hr.xlsx", buf.getvalue(), "application/octet-stream")},
                          data={"sheet": "Nope"}, headers=ws["h"])
    assert missing.status_code == 400 and "no sheet named" in missing.json()["detail"]
    pdf_as_dataset = client.post("/api/datasets/upload", files={"file": ("r.pdf", b"%PDF-1.4", "application/pdf")},
                                 headers=ws["h"])
    assert pdf_as_dataset.status_code == 400 and "document upload" in pdf_as_dataset.json()["detail"]
    assert client.post("/api/analyst/ask", json={"message": "hi"}, headers=ws["h"]).status_code == 400


# ------------------------------------------------------- why a group does better --

def test_why_follow_up_compares_the_leading_group_with_the_rest():
    mem = {}
    r = ask("Which region has the highest sales?", S, mem)
    leader = S.groupby("region")["sales"].mean().idxmax()
    assert mem["last_focus"] == {"group_column": "region", "group": leader, "outcome": "sales"}
    r = ask("Why is that region performing better?", S, mem)
    assert r["plan_source"] == "router" and r["calls"][0]["tool"] == "drivers"
    res = r["calls"][0]["result"]
    assert res["group"] == leader and res["outcome"]["group_mean"] > res["outcome"]["others_mean"]
    units = next(x for x in res["numeric"] if x["column"] == "units")
    a, b = S.loc[S.region == leader, "units"].mean(), S.loc[S.region != leader, "units"].mean()
    assert units["group_mean"] == pytest.approx(a) and units["others_mean"] == pytest.approx(b)
    assert "not proven causes" in r["answer"]
    r = ask("Why is West performing better?", S, mem)
    assert r["calls"][0]["result"]["group"] == "West", "a named group wins over the remembered one"


def test_why_without_a_known_group_asks_which():
    assert router.route("Why is it performing better?", S, {})["type"] == "clarify"
    assert router.route("Why do you recommend Area A?", S, {}) is None, "recommendation reasons stay with the planner"


def test_prediction_answers_get_no_model_summary_and_summaries_are_labelled(monkeypatch):
    mem = {}
    ask("Predict attrition", A, mem)
    calls = []

    def fake_chat(msgs, **kw):
        calls.append(msgs[0]["content"][:20])
        return ('{"type":"tools","intent":"prediction","calls":[{"tool":"predict","args":{"rows":'
                '[{"overtime":"Yes","satisfaction":2,"monthly_income":3000}]}}]}')
    monkeypatch.setattr(planner, "model_chat", fake_chat)
    r = orchestrator.ask("Predict attrition for overtime Yes, satisfaction 2, monthly_income 3000", A, mem, use_model=True)
    assert r["plan_source"] == "model" and r["calls"][0]["status"] == "ok"
    assert len(calls) == 1, "only the planning call; no summary is written for a prediction"
    monkeypatch.setattr(planner, "model_chat", lambda msgs, **kw: (
        '{"type":"tools","intent":"metric","calls":[{"tool":"compute_metric","args":{"aggregation":"avg","column":"monthly_income"}}]}'
        if "planner" not in msgs[0]["content"] and "Write a short" not in msgs[0]["content"] else "The average is shown below."))
    r = orchestrator.ask("What is the average monthly income?", A, {}, use_model=True)
    # the computed answer comes first; the model's reading follows, labelled
    assert r["answer"].startswith("The average monthly_income is") and "**In plain words:** The average is shown below." in r["answer"]
    assert "every number in it was checked" in r["answer"]
