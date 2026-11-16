"""A file of several logical tables stacked in one sheet (the exact Business_Dashboard_Test_Dataset_Combined.csv):
no filter the question did not ask for, one table per answer, and the figures pandas computes on that table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator, planner, reconcile, scope, tables
from app.analyst.data import load_table

CSV = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "combined" / "Business_Dashboard_Test_Dataset_Combined.csv"


@pytest.fixture(autouse=True)
def model_off(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


@pytest.fixture(scope="module")
def df():
    return load_table(str(CSV))


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(CSV, low_memory=False)


def sales(raw):
    return raw[raw.Table == "Sales Transactions"]


def ask(q, df, mem=None):
    return orchestrator.ask(q, df, {} if mem is None else mem, use_model=False)


def f2(x):
    return f"{x:,.2f}"


# ------------------------------------------------------------------ the reported failure --
def test_the_word_business_is_not_a_segment_filter(df):
    for q in ("Give me a business overview of Sales Transactions and its status distribution (Completed, Shipped, "
              "Cancelled or Returned).", "Calculate total revenue for the business."):
        assert "Segment" not in scope.establish(q, df, {})["entities"]
    out = ask("Give me a business overview of Sales Transactions and its status distribution (Completed, Shipped, "
              "Cancelled or Returned).", df)
    assert "No records match" not in out["answer"] and "Segment" not in out["answer"]
    for v in (10509, 687, 418, 386):
        assert f"| {v} |" in out["answer"]


def test_several_status_values_stay_separate_values(df):
    sc = scope.establish("Status distribution of Sales Transactions: Completed, Shipped, Cancelled or Returned", df, {})
    assert sorted(sc["entities"]["Status"]) == ["Cancelled", "Completed", "Returned", "Shipped"]


def test_a_named_segment_column_keeps_the_filter(df):
    assert scope.establish("Show customers in the business segment", df, {})["entities"] == {"Segment": ["Business"]}
    assert scope.establish("Customers where segment = business", df, {})["entities"]["Segment"] == ["Business"]


def test_a_filter_on_a_column_empty_in_the_selected_records_is_explained_not_applied(df):
    sc = scope.establish("Filter Sales Transactions by Segment = Business.", df, {})
    assert sc["clarify"] and "only filled in where Table = Customers" in sc["clarify"]
    sc = scope.establish("A Business view of Sales Transactions", df, {})          # the value only, capitalised
    assert "Segment" not in sc["entities"] and sc["dropped"]


# ------------------------------------------------------------------ logical tables --
def test_the_tables_are_read_from_the_data(df):
    assert tables.table_column(df) == "Table"
    assert tables.tables(df, "Table") == {"Sales Transactions": 12000, "Customers": 1500, "Customer Reviews": 1200,
                                          "Website Traffic": 600, "Inventory Snapshots": 480, "Expenses": 120,
                                          "Marketing Campaigns": 72, "Products": 20}
    sub = tables.subset(df, "Table", "Sales Transactions")
    assert len(sub) == 12000 and "Segment" not in sub and "Table" not in sub
    assert str(sub["Date"].dtype).startswith("datetime64")


def test_an_ordinary_table_has_no_table_column():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"department": rng.choice(["Sales", "HR", "IT"], 200), "salary": rng.normal(5e4, 1e4, 200),
                      "age": rng.integers(20, 60, 200)})
    d["commission"] = np.where(d.department == "Sales", 1000.0, np.nan)    # one column empty for two departments
    assert tables.table_column(d) is None


def test_a_measure_in_two_tables_is_asked_about_not_added_up(df):
    out = ask("Calculate total revenue for the business.", df)
    assert not out["calls"] and "Sales Transactions" in out["answer"] and "Marketing Campaigns" in out["answer"]
    assert out["plan"]["options"] == ["Sales Transactions", "Marketing Campaigns"]


def test_record_lines_are_lines_of_the_uploaded_file(df, raw):
    out = ask("Calculate total revenue in Sales Transactions.", df)
    first_negative = int(sales(raw).index[sales(raw).Revenue < 0][0]) + 2
    assert f"lines {first_negative}," in out["answer"]


# ------------------------------------------------------------------ A-F --
def test_a_total_revenue(df, raw):
    out = ask("Calculate total revenue in Sales Transactions.", df)
    assert f2(sales(raw).Revenue.sum()) == "426,249,271.65" and "426,249,271.65" in out["answer"]
    assert out["verify_status"] == "verified" and "Table: **Sales Transactions**" in out["answer"]


def test_b_counts_by_status(df, raw):
    out = ask("Show transaction counts by status in Sales Transactions.", df)
    assert out["verify_status"] == "verified" and "Which column" not in out["answer"]
    for status, n in sales(raw).Status.value_counts().items():
        assert f"| {status} | {n} | {n} |" in out["answer"]


def test_c_financial_summary_and_profit_reconciliation(df, raw):
    st = sales(raw)
    out = ask("Calculate total revenue, total cost, total profit and average transaction revenue in Sales Transactions.", df)
    assert out["verify_status"] == "verified"
    for v in (f2(st.Revenue.sum()), f"{st.Cost.sum():,.0f}", f2(st.Profit.sum()), f2(st.Revenue.mean())):
        assert v in out["answer"]
    gap = st.Profit.sum() - (st.Revenue.sum() - st.Cost.sum())
    assert f2(st.Revenue.sum() - st.Cost.sum()) in out["answer"] and f2(gap) in out["answer"]
    assert "Returned" in out["answer"] and "Cancelled" in out["answer"]


def test_d_highest_revenue_region_ranks_by_total(df, raw):
    out = ask("Identify the highest revenue region in Sales Transactions.", df)
    reg = sales(raw).groupby("Region").Revenue.sum().sort_values(ascending=False)
    assert reg.index[0] == "East" and f"Highest: **East** ({f2(reg.iloc[0])})" in out["answer"]
    assert out["verify_status"] == "verified" and "620,000" not in out["answer"]


def test_e_segment_on_sales_is_explained(df):
    out = ask("Filter Sales Transactions by Segment = Business.", df)
    assert not out["calls"] and "not recorded for **Sales Transactions**" in out["answer"]
    assert "Customers" in out["answer"] and "No records match" not in out["answer"]


def test_f_business_customers(df, raw):
    cu = raw[raw.Table == "Customers"]
    out = ask("Show customers belonging to the Business segment.", df)
    n = int((cu.Segment == "Business").sum())
    assert n == cu.loc[cu.Segment == "Business", "Customer ID"].nunique() == 391
    assert "Table: **Customers**" in out["answer"] and f"**{n}**" in out["answer"]


# ------------------------------------------------------------------ follow-ups --
def test_follow_ups_keep_the_table(df, raw):
    st, mem = sales(raw), {}
    ask("Calculate total revenue in Sales Transactions.", df, mem)
    out = ask("Now show it by region.", df, mem)
    assert "the table this conversation is on" in out["answer"]
    for v in st.groupby("Region").Revenue.sum():
        assert f2(v) in out["answer"]
    out = ask("What percentage of transactions were returned?", df, mem)
    assert f"{(st.Status == 'Returned').mean() * 100:.1f}%" in out["answer"]


def test_its_top_cities_are_the_leading_regions(df, raw):
    st, mem = sales(raw), {}
    ask("Identify the highest revenue region in Sales Transactions.", df, mem)
    out = ask("Show its top five cities.", df, mem)
    east = st[st.Region == "East"].groupby("City").Revenue.sum().sort_values(ascending=False).head(5)
    assert "Within Region = East" in out["answer"]
    for city, v in east.items():
        assert city in out["answer"] and f2(v) in out["answer"]


def test_business_then_vip_customers(df, raw):
    cu, mem = raw[raw.Table == "Customers"], {}
    ask("Show customers belonging to the Business segment.", df, mem)
    out = ask("Show only VIP customers.", df, mem)
    assert f"**{int((cu.Segment == 'VIP').sum())}**" in out["answer"] and "Segment = VIP" in out["answer"]


def test_explain_the_profit_difference(df, raw):
    st, mem = sales(raw), {}
    ask("Calculate total revenue, total cost, total profit and average transaction revenue in Sales Transactions.", df, mem)
    out = ask("Explain the profit difference.", df, mem)
    for s in ("Returned", "Cancelled"):
        x = st[st.Status == s]
        assert f2(x.Profit.sum() - (x.Revenue.sum() - x.Cost.sum())) in out["answer"]
    assert "Profit equals Revenue + Cost" in out["answer"] and "Profit is 0" in out["answer"]


def test_reconcile_agrees_when_profit_is_revenue_minus_cost():
    d = pd.DataFrame({"revenue": [10.0, 20.0, 30.0], "cost": [4.0, 5.0, 6.0]}).assign(profit=lambda x: x.revenue - x.cost)
    r = reconcile.check(d)
    assert r["differ"] == 0 and "agree in all 3 records" in reconcile.text(r)


# ------------------------------------------------------------------ ranking semantics: one aggregate, never "either" --
from app.analyst import intent, plan_checks, verify  # noqa: E402


@pytest.mark.parametrize("q,col,want", [
    ("Identify the highest revenue region.", "Revenue", "sum"),
    ("Identify the highest average revenue region.", "Revenue", "mean"),
    ("Which region has the largest individual transaction?", "Revenue", "max"),
    ("Which region has the highest total unit price?", "Unit Price", "sum"),
    ("Which region has the highest unit price?", "Unit Price", "mean"),
    ("Which region has the most widgets?", "widgets", None),
])
def test_ranking_aggregate_is_explicit(q, col, want):
    assert intent.ranking_aggregate(q, col) == want


def test_highest_average_revenue_region(df, raw):
    out = ask("Identify the highest average revenue region in Sales Transactions.", df)
    avg = sales(raw).groupby("Region").Revenue.mean().sort_values(ascending=False)
    assert avg.index[0] == "West" and f"Highest: **West** ({f2(avg.iloc[0])})" in out["answer"]
    assert out["verify_status"] == "verified" and "89,347,040.55" not in out["answer"]


def test_largest_individual_transaction(df, raw):
    st = sales(raw)
    out = ask("What is the largest individual transaction revenue in Sales Transactions?", df)
    assert f"{st.Revenue.max():,.0f}" in out["answer"]
    out = ask("Which region has the largest individual transaction revenue in Sales Transactions?", df)
    mx = st.groupby("Region").Revenue.max()
    assert "West" in out["answer"] and f"{mx['West']:,.0f}" in out["answer"]
    assert out["verify_status"] != "verified"          # a per-group maximum is not recomputed: never labelled verified


def test_a_ranking_the_measure_does_not_settle_is_asked():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"region": rng.choice(["N", "S", "E"], 90), "widgets": rng.integers(1, 9, 90).astype(float)})
    out = ask("Which region has the highest widgets?", d)
    assert out["plan"]["type"] == "clarify" and not [c for c in out["calls"] if c.get("status") == "ok"]
    assert "total widgets" in out["answer"] and "average widgets" in out["answer"]


def _ranking_steps(fn, values):
    return ([{"id": "r", "op": "aggregate", "from": "data", "fn": fn, "column": "Revenue", "by": "Region"}],
            {"r": {"kind": "series", "value": values, "hypothetical": False}})


def test_the_verifier_holds_a_ranking_to_its_one_aggregate(df):
    st = tables.subset(df, "Table", "Sales Transactions")
    q = "Identify the highest revenue region."
    pol = {"valid": False}
    tot = st.groupby("Region").Revenue.sum().to_dict()
    avg = st.groupby("Region").Revenue.mean().to_dict()
    assert verify.check(q, st, *_ranking_steps("sum", tot), policy=pol)["status"] == "verified"
    # the averages are correct averages, but not what "highest revenue region" compares: not verified
    assert verify.check(q, st, *_ranking_steps("mean", avg), policy=pol)["status"] != "verified"
    q2 = "Identify the highest average revenue region."
    assert verify.check(q2, st, *_ranking_steps("mean", avg), policy=pol)["status"] == "verified"
    assert verify.check(q2, st, *_ranking_steps("sum", tot), policy=pol)["status"] != "verified"


def test_an_open_ranking_is_never_verified_by_matching_either_aggregate():
    rng = np.random.default_rng(4)
    d = pd.DataFrame({"region": rng.choice(["N", "S", "E"], 90), "widgets": rng.integers(1, 9, 90).astype(float)})
    q = "Which region has the highest widgets?"
    for fn in ("sum", "mean"):
        vals = d.groupby("region").widgets.agg(fn).to_dict()
        steps = [{"id": "r", "op": "aggregate", "from": "data", "fn": fn, "column": "widgets", "by": "region"}]
        res = {"r": {"kind": "series", "value": vals, "hypothetical": False}}
        assert verify.check(q, d, steps, res, policy={"valid": False})["status"] != "verified"


@pytest.mark.parametrize("q,agg,bad", [
    ("Identify the highest revenue region.", "avg", True), ("Identify the highest revenue region.", "max", True),
    ("Identify the highest revenue region.", "sum", False),
    ("Identify the highest average revenue region.", "sum", True),
    ("Which region has the largest individual transaction revenue?", "sum", True),
    ("Which region has the largest individual transaction revenue?", "max", False),
])
def test_model_plans_with_another_aggregate_are_rejected(df, q, agg, bad):
    st = tables.subset(df, "Table", "Sales Transactions")
    call = {"tool": "compute_metric", "status": "ok", "args": {"aggregation": agg, "column": "Revenue", "group_by": "Region"}}
    assert bool(plan_checks.metric_mismatch(q, [call], st)) is bad


def test_a_program_ranking_by_another_aggregate_is_rejected(df):
    st = tables.subset(df, "Table", "Sales Transactions")
    q = "Identify the highest revenue region."
    req = intent.read(q, st)
    mean_prog = [{"id": "r", "op": "aggregate", "from": "data", "fn": "mean", "column": "Revenue", "by": "Region"}]
    sum_prog = [{**mean_prog[0], "fn": "sum"}]
    assert any("total Revenue" in p for p in intent.check(mean_prog, req, st, q))
    assert not any("ranks each" in p for p in intent.check(sum_prog, req, st, q))


def test_a_figure_the_ranking_rule_cannot_read_is_unverified_not_failed(df):
    st = tables.subset(df, "Table", "Sales Transactions")
    q = "Identify the highest revenue region."
    per_row = (st.groupby("Region").Revenue.sum() / st.groupby("Region").size()).to_dict()   # numerically = mean
    steps = [{"id": "s", "op": "aggregate", "from": "data", "fn": "sum", "column": "Revenue", "by": "Region"},
             {"id": "n", "op": "aggregate", "from": "data", "fn": "count", "by": "Region"},
             {"id": "r", "op": "ratio", "from": "data", "a": "s", "b": "n"}]
    res = {"r": {"kind": "series", "value": per_row, "hypothetical": False}}
    assert verify.check(q, st, steps, res, policy={"valid": False})["status"] != "verified"


def test_the_reconciliation_keeps_a_named_filter(df, raw):
    out = ask("Explain the profit difference for Returned transactions in Sales Transactions.", df)
    x = sales(raw)[sales(raw).Status == "Returned"]
    assert f2(x.Profit.sum() - (x.Revenue.sum() - x.Cost.sum())) in out["answer"]
    assert "Cancelled" not in out["answer"].split("_This describes")[0]
