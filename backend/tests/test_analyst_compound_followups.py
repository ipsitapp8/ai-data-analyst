"""Compound questions and follow-ups that were answered by the first fragment a pattern matched.

Failure classes (each found on the combined dashboard file, and checked here on an ordinary table too):
- one loosely matched word overriding the conversation's table ("profit margin" finding `Margin %` of another table,
  "customers" read as naming the Customers table);
- "break it down by X" read as "explain the last answer again" instead of a new grouping;
- a list of several measures per group collapsed to one metric, and "ranked by X" ignored;
- a group picked out by rank ("within the highest-revenue region") never applied;
- two "top N" lists in one question, only the first answered;
- "overview of <table> and reconcile ..." answering only one half.
Every figure is compared with pandas on the same records."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analyst import followup, metrics, orchestrator, planner, scope, tables
from app.analyst.data import load_table

CSV = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "combined" / "Business_Dashboard_Test_Dataset_Combined.csv"


@pytest.fixture(autouse=True)
def model_off(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


@pytest.fixture(scope="module")
def df():
    return load_table(str(CSV))


@pytest.fixture(scope="module")
def sales():
    raw = pd.read_csv(CSV, low_memory=False)
    return raw[raw.Table == "Sales Transactions"]


@pytest.fixture(scope="module")
def shop():
    """An ordinary one-table file: nothing about the combined file's names."""
    rng = np.random.default_rng(7)
    n = 600
    return pd.DataFrame({
        "Store": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "Town": rng.choice([f"T{i}" for i in range(12)], n),
        "Client": rng.choice([f"K{i:03d}" for i in range(80)], n),
        "Sales": rng.integers(100, 5000, n).astype(float),
        "Expenses": rng.integers(50, 3000, n).astype(float),
    })


def ask(q, df, mem=None):
    return orchestrator.ask(q, df, {} if mem is None else mem, use_model=False)


def nums(text):
    return {float(x.replace(",", "")) for x in __import__("re").findall(r"-?\d[\d,]*\.\d+|-?\d{1,3}(?:,\d{3})+", text)}


# ------------------------------------------------------------ table choice --
ON_SALES = {"logical_table": {"table": "Sales Transactions"}}


@pytest.mark.parametrize("q", [
    "Break it down by region: revenue, cost, profit and profit margin for each region.",
    "Profit margin and total revenue per region please",
    "What is the margin by region?",
])
def test_a_loose_column_match_does_not_move_the_conversation_off_its_table(df, q):
    pick = tables.select(q, df, dict(ON_SALES))
    assert pick.get("table") == "Sales Transactions" and not pick.get("clarify")


@pytest.mark.parametrize("q", [
    "Within the East region, what are the top five customers by revenue?",
    "Which customers bring in the most revenue?",
    "List the top 3 customers by profit",
])
def test_a_lowercase_table_name_used_as_a_noun_does_not_switch_tables(df, q):
    assert tables.select(q, df, dict(ON_SALES)).get("table") == "Sales Transactions"
    # a fresh conversation: Sales Transactions, or a question when Revenue is in two tables -- never Customers
    fresh = tables.select(q, df, {})
    assert fresh.get("table") == "Sales Transactions" or "Sales Transactions" in (fresh.get("options") or [])


@pytest.mark.parametrize("q,table", [
    ("What is the total revenue in the Customers table?", "Customers"),      # named: explained, not switched
    ("How many customers are in each region?", "Customers"),                  # the table has every column used
    ("Average margin % by category", "Products"),                             # a full name ending in a symbol
])
def test_a_table_that_is_really_named_is_still_chosen(df, q, table):
    assert tables.select(q, df, dict(ON_SALES)).get("table") == table


# ------------------------------------------------- "break it down by ..." --
@pytest.mark.parametrize("q", ["Now break it down by region.", "Break this down per category", "simplify it for each channel"])
def test_a_new_grouping_is_a_calculation_not_a_rewording(q):
    assert followup.intent(q, {"last_answer": "x"}) is None


@pytest.mark.parametrize("q", ["Break it down for me.", "Explain this in simple terms.", "What does that mean?"])
def test_rewording_requests_are_still_conversational(q):
    assert followup.intent(q, {"last_answer": "x"}) == "explain_previous"


# ------------------------------------------------ several measures per group --
def test_every_requested_measure_per_group_ranked_as_asked(df, sales):
    mem = {}
    ask("Total revenue in Sales Transactions", df, mem)
    out = ask("Now break it down by region: total revenue, cost, profit, number of transactions and profit margin "
              "for each region, ranked by revenue.", df, mem)
    res = out["calls"][0]["result"]
    exp = sales.groupby("Region").agg(r=("Revenue", "sum"), c=("Cost", "sum"), p=("Profit", "sum"), n=("Revenue", "size"))
    exp = exp.sort_values("r", ascending=False)
    assert [g["group"] for g in res["groups"]] == list(exp.index)
    for g in res["groups"]:
        e = exp.loc[g["group"]]
        assert g["total_revenue"] == pytest.approx(e.r) and g["total_cost"] == pytest.approx(e.c)
        assert g["total_profit"] == pytest.approx(e.p) and g["number_of_transactions"] == e.n
        assert g["profit_margin"] == pytest.approx((e.r - e.c) / e.r)
    assert "**East** has the highest total revenue" in out["answer"]


def test_a_list_of_measures_on_an_ordinary_table(shop):
    out = ask("Average sales, total expenses and number of orders by store, sorted by total expenses", shop)
    res = out["calls"][0]["result"]
    exp = shop.groupby("Store").agg(a=("Sales", "mean"), e=("Expenses", "sum"), n=("Sales", "size")).sort_values("e", ascending=False)
    assert [g["group"] for g in res["groups"]] == list(exp.index)
    g = res["groups"][0]
    assert g["average_sales"] == pytest.approx(exp.a.iloc[0]) and g["number_of_orders"] == exp.n.iloc[0]


def test_one_measure_with_a_metric_is_still_the_metric_plan(df):
    sub = tables.subset(df, "Table", "Sales Transactions")
    assert metrics.multi_plan("profit margin by region", sub) is None
    assert metrics.plan("profit margin by region", sub).name == "profit_margin"


# --------------------------------------------------- a group picked by rank --
@pytest.mark.parametrize("phrase,how", [
    ("within the highest-revenue region", "sum"), ("for the region with the highest revenue", "sum"),
    ("in the highest average revenue region", "mean"), ("within the lowest revenue region", "sum"),
])
def test_a_group_named_by_its_rank_is_resolved_from_the_data(df, sales, phrase, how):
    sub = tables.subset(df, "Table", "Sales Transactions")
    got = scope.superlative_group(f"Top cities by profit {phrase}?", sub)
    v = sales.groupby("Region").Revenue.agg(how)
    want = v.idxmin() if "lowest" in phrase else v.idxmax()
    assert got["column"] == "Region" and got["value"] == want


def test_a_rank_without_a_clear_aggregation_is_asked(shop):
    s = shop.assign(Zeta=np.arange(len(shop)) % 7)          # neither a flow (total) nor a level (average) name
    assert "clarify" in scope.superlative_group("top towns in the highest zeta store", s)


def test_ranking_questions_themselves_are_not_filters(df):
    sub = tables.subset(df, "Table", "Sales Transactions")
    assert scope.superlative_group("Which region has the highest revenue?", sub) is None


# ----------------------------------------------------- two top-N lists --
def test_two_top_lists_within_a_ranked_group_fresh_and_follow_up(df, sales):
    east = sales[sales.Region == sales.groupby("Region").Revenue.sum().idxmax()]
    cities = east.groupby("City").Revenue.sum().nlargest(5)
    cust = east.groupby("Customer ID").Revenue.sum().nlargest(5)
    q = "Within the highest-revenue region, what are the top five cities and the top five customers by revenue?"
    for mem in ({}, {"logical_table": {"table": "Sales Transactions"}}):
        out = ask(q if mem else q.replace("region,", "region of Sales Transactions,"), df, mem)
        res = [c["result"] for c in out["calls"] if c["tool"] == "rank_groups"]
        assert [r["group"] for r in res] == ["City", "Customer ID"]
        assert [x["group"] for x in res[0]["rows"]] == list(cities.index)
        assert [x["value"] for x in res[1]["rows"]] == pytest.approx(list(cust.values))
        assert "Region = East" in out["answer"]


def test_two_top_lists_on_an_ordinary_table(shop):
    out = ask("Show the top 3 towns and the top 4 clients by expenses", shop)
    res = [c["result"] for c in out["calls"] if c["tool"] == "rank_groups"]
    assert [(r["group"], r["measure"], len(r["rows"])) for r in res] == [("Town", "Expenses", 3), ("Client", "Expenses", 4)]
    assert [x["value"] for x in res[1]["rows"]] == pytest.approx(list(shop.groupby("Client").Expenses.sum().nlargest(4)))


# ----------------------------------------------- overview + reconciliation --
def test_overview_and_reconciliation_both_answered(df, sales):
    out = ask("Give me an overview of the Sales Transactions table and reconcile the stored Profit against Revenue "
              "minus Cost.", df)
    assert [c["tool"] for c in out["calls"]] == ["understand", "reconcile_profit"]
    assert "12,000 rows" in out["answer"]
    rec = out["calls"][1]["result"]
    assert rec["revenue_minus_cost"] == pytest.approx((sales.Revenue - sales.Cost).sum())
    assert rec["total_profit"] == pytest.approx(sales.Profit.sum())


def test_overview_of_a_named_table_keeps_its_object(df):
    out = ask("Give me an overview of the Sales Transactions table.", df)
    assert [c["tool"] for c in out["calls"]] == ["understand"]


# ------------------------------------------------ the statistics question --
def test_discount_profit_correlation_excludes_returned_and_cancelled(df, sales):
    out = ask("Is there a relationship between Discount and Profit? Give the Pearson and Spearman correlations with "
              "p-values and the sample size, excluding Returned and Cancelled transactions.", df)
    k = sales[~sales.Status.isin(["Returned", "Cancelled"])][["Discount %", "Profit"]].dropna()
    assert f"n = {len(k)}" in out["answer"]
    assert f"r = {stats.pearsonr(k['Discount %'], k['Profit'])[0]:.2f}" in out["answer"]
    assert f"rho = {stats.spearmanr(k['Discount %'], k['Profit'])[0]:.2f}" in out["answer"]


# ------------------------------------------------- model-on failure handling --
def test_a_failed_model_plan_without_an_error_message_keeps_the_tool_answer():
    """Model on: the model's calculation plan failed its checks without an error message, and another tool answered.
    That answer is kept (it used to raise TypeError and show 'Something went wrong')."""
    out = {"calls": [{"tool": "correlation", "status": "ok", "result": {"n": 11196, "r": -0.23}}], "plan": {"type": "tools"},
           "notes": []}
    for fp in ({"error": None}, {}, None):
        kept = orchestrator._after_failed({**out, "notes": []}, {"first_pass": fp, "answer": "withheld"})
        assert kept["calls"][0]["tool"] == "correlation" and "see trace" in kept["notes"][0]


def test_a_count_written_as_noun_count_is_a_record_count(shop):
    p = metrics.multi_plan("total sales, total expenses and order count by store", shop)
    assert p.spec["aggregate"]["number_of_order"] == "count(*)" and p.spec["group_by"] == "Store"


# ------------------------------------- one record policy: model ON == model OFF --
def _program_answer(monkeypatch, q, steps, answer, sub):
    """Model ON: the answer a model-written program gets from Silt's program executor and its verification."""
    import json
    import time
    from app.analyst import nl_planner
    prog = {"action": "program", "objective": q, "answer": answer, "steps": steps}
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(nl_planner, "model_chat", lambda *a, **k: json.dumps(prog))
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)
    mem, t = {}, time.monotonic()
    out = orchestrator._ask_nl_program(q, sub, mem, True, t, "data.csv", False)
    assert out is not None and out["plan_source"] == "model_program", (out or {}).get("answer")
    return orchestrator._verify_answer(out, q, sub, mem, True, t)


PARITY = [
    ("What is the total revenue by status?",
     [{"id": "a", "op": "aggregate", "from": "data", "fn": "sum", "column": "Revenue", "by": "Status"}], "a",
     lambda s: s.groupby("Status").Revenue.sum()),
    ("What is the total revenue by region?",
     [{"id": "a", "op": "aggregate", "from": "data", "fn": "sum", "column": "Revenue", "by": "Region"}], "a",
     lambda s: s.groupby("Region").Revenue.sum()),
    ("What is the average revenue by region?",
     [{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "Revenue", "by": "Region"}], "a",
     lambda s: s.groupby("Region").Revenue.mean()),
    ("What is the total profit by status?",
     [{"id": "a", "op": "aggregate", "from": "data", "fn": "sum", "column": "Profit", "by": "Status"}], "a",
     lambda s: s.groupby("Status").Profit.sum()),
    ("What is the total revenue of Returned orders?",
     [{"id": "s", "op": "select", "from": "data", "entities": {"Status": ["Returned"]}},
      {"id": "a", "op": "aggregate", "from": "s", "fn": "sum", "column": "Revenue"}], "a",
     lambda s: pd.Series({"all": s[s.Status == "Returned"].Revenue.sum()})),
]


@pytest.mark.parametrize("q,steps,answer,expect", PARITY, ids=[p[0] for p in PARITY])
def test_model_on_and_off_give_the_same_figures_with_negative_and_returned_records(monkeypatch, df, sales, q, steps,
                                                                                    answer, expect):
    sub = tables.subset(df, "Table", "Sales Transactions")
    want = expect(sales)
    assert (sales.Revenue < 0).any()                              # the file has negative amounts (Returned orders)
    off = ask(q, sub)
    on = _program_answer(monkeypatch, q, steps, answer, sub)
    for v in want.values:
        cell = f"{v:,.2f}"
        assert cell in off["answer"], (q, "model OFF", cell)
        assert cell in on["answer"], (q, "model ON", cell)
    assert "Left out as impossible" not in on["answer"]          # negative revenue is not dropped by the program
    assert on.get("verify_status") not in ("withheld", "failed")


def test_negative_never_negative_columns_are_still_left_out_in_every_path():
    """The other half of the one policy: a column that can never be negative (a quantity) is left out everywhere."""
    from app.analyst import metrics, scope as sc
    d = pd.DataFrame({"Qty": [2.0, -1.0, 3.0], "Revenue": [10.0, -5.0, 7.0]})
    ok, why = sc.eligibility(d, ["Qty", "Revenue"], historical=False)
    # (Stage 5: the reason now says why the value is impossible -- "negative Qty (a count or quantity cannot be negative)")
    assert list(ok) == [True, False, True] and why[0]["reason"].startswith("negative Qty")
    assert "cannot be negative" in why[0]["reason"]
    ok, why = sc.eligibility(d, ["Revenue"], historical=False)
    assert ok.all() and not why
    assert sc.signed_negatives(d, ["Revenue"])[0]["records"] == 1
    assert metrics.never_negative("Qty") and not metrics.never_negative("Revenue") and not metrics.never_negative("Profit")


# --------------------------------------- model-on acceptance failure classes --
def _model_turn(monkeypatch, planned, program):
    """Model ON with the tool planner's reply `planned` and the program planner's reply `program` simulated."""
    from app.analyst import nl_planner
    monkeypatch.setattr(planner, "model_available", lambda: True)
    real = orchestrator._ask_planned

    def ask_planned(message, df, memory, more, started):
        return planned(message, df, memory, started) if more else real(message, df, memory, False, started)
    monkeypatch.setattr(orchestrator, "_ask_planned", ask_planned)
    monkeypatch.setattr(nl_planner, "plan", lambda *a, **k: program)
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)


_CLARIFY = {"status": "clarify", "question": "Which column should I report?", "options": ["a", "b"], "trace": []}


@pytest.mark.parametrize("q,kind,text", [
    ("Hello.", "chat", "Hello! I'm Silt's analyst."),
    ("What is Random Forest?", "chat", "A Random Forest is an ensemble of decision trees."),
    ("Which area should receive our next advertising campaign?", "decline",
     "**The available data is not sufficient for this.** This dataset has no advertising history."),
])
def test_a_reply_to_a_message_that_names_no_data_is_not_replaced_by_a_program(monkeypatch, shop, q, kind, text):
    _model_turn(monkeypatch, lambda m, d, mem, t: orchestrator._reply(text, "model", True, t, plan={"type": kind}),
                _CLARIFY)
    out = orchestrator.ask(q, shop, {}, use_model=True)
    assert out["answer"].startswith(text[:20]) and out["plan"]["type"] == kind


def test_a_placeholder_reply_is_still_handed_to_the_program_planner(monkeypatch, shop):
    _model_turn(monkeypatch, lambda m, d, mem, t: orchestrator._reply("I'm not sure what to compute.", "model", True, t,
                                                                      plan={"type": "chat"}), _CLARIFY)
    out = orchestrator.ask("Tell me something about expenses", shop, {}, use_model=True)
    assert "Which column should I report?" in out["answer"]


def test_a_model_question_whose_answer_the_request_already_names_is_not_asked(monkeypatch):
    from app.analyst import demo_data as dd
    _model_turn(monkeypatch, lambda m, d, mem, t: orchestrator._reply(
        "Which column should I predict?", "model", True, t,
        plan={"type": "clarify", "options": ["contract", "monthly_charges", "region", "churn"]}), _CLARIFY)
    out = orchestrator.ask("Predict customer churn.", dd.customers(), {}, use_model=True)
    rf = [c for c in out["calls"] if c["tool"] == "random_forest"]
    assert rf and rf[0]["status"] == "ok" and rf[0]["result"]["task"] == "classification"
