"""Stage 1 of the Copilot reliability repair: no confident wrong answers.

Every case here was a reproduced failure (or an unseen rephrasing of one) on data Silt was never written against:
support tickets, shift records, retail orders and a messy creator file -- plus Ambition Box when the file is present.
A test passes only when the answer is the figure pandas gives for the question as asked, or an explicit refusal /
clarification -- never merely because something was returned.

What is held:
  * a filter the question states (a threshold, a yes/no adjective, its negation, a named value) is applied, or the
    answer says it could not be;
  * a name that is not in the data is reported, never answered over all the records;
  * an identifier or code is never averaged, summed, ranked on or trended; a per-record level is never totalled;
  * a share of a measure is a share of that measure's total, not of the records;
  * nothing wrong carries the "verified" mark.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

from app.analyst import intent, orchestrator, semantics
from app.analyst.data import load_table

TODAY = pd.Timestamp("2026-11-17")


@pytest.fixture(autouse=True)
def pinned_clock(monkeypatch):
    """The seeded data is dated around TODAY, so the code under test must see that day as today. Without this
    the 'last 7 days' style questions drift with the real clock and start failing the day after TODAY."""
    monkeypatch.setattr("app.analyst.scope.today", lambda: TODAY)

AMBITION = os.path.join(os.path.dirname(__file__), "..", "..", "Ambition Box.csv")


# ------------------------------------------------------------------ data --
def _tickets(rng):
    n = 360
    d = pd.DataFrame({"ticket_id": np.arange(50001, 50001 + n),
                      "opened": [TODAY - pd.Timedelta(days=int(x)) for x in rng.integers(0, 150, n)],
                      "agent": rng.choice(["Asha Rao", "Ben Cole", "Chen Wei", "Dara Okoye"], n),
                      "priority": rng.choice(["Low", "Medium", "High"], n, p=[.5, .3, .2]),
                      "channel": rng.choice(["Email", "Phone", "Chat"], n),
                      "escalated": rng.choice(["Yes", "No"], n, p=[.25, .75]),
                      "resolution_hours": np.round(rng.gamma(2, 4, n), 1),
                      "csat": rng.integers(1, 6, n).astype(float)})
    d.loc[d.escalated == "Yes", "resolution_hours"] += 6
    d.loc[rng.choice(n, 12, replace=False), "csat"] = np.nan
    return d


def _shifts(rng):
    n = 200
    emp = rng.choice(["Rahul Mehta", "Priya Singh", "Aarav Shah", "Mei Lin", "Tom Baker"], n)
    dept = {"Rahul Mehta": "Engineering", "Priya Singh": "Finance", "Aarav Shah": "Engineering", "Mei Lin": "Support",
            "Tom Baker": "Finance"}
    start, hrs = rng.integers(7, 11, n), rng.integers(4, 10, n)
    return pd.DataFrame({"shift_id": np.arange(1, n + 1), "employee": emp, "department": [dept[e] for e in emp],
                         "shift_date": [TODAY - pd.Timedelta(days=int(x)) for x in rng.integers(0, 90, n)],
                         "clock_in": [f"{s:02d}:00" for s in start],
                         "clock_out": [f"{s + h:02d}:00" for s, h in zip(start, hrs)],
                         "hourly_rate": rng.choice([18.5, 22.0, 27.5, 31.0], n),
                         "overtime": rng.choice([0, 1], n, p=[.8, .2])})


def _orders(rng):
    n = 500
    units, price = rng.integers(1, 12, n), rng.choice([9.99, 24.5, 49.0, 120.0], n)
    return pd.DataFrame({"order_id": rng.permutation(np.arange(700000, 700000 + n)),
                         "order_date": [TODAY - pd.Timedelta(days=int(x)) for x in rng.integers(0, 400, n)],
                         "customer_id": rng.integers(1001, 1121, n),
                         "region": rng.choice(["North", "South", "East", "West"], n),
                         "product_line": rng.choice(["Garden", "Kitchen", "Toys", "Office"], n),
                         "units": units, "unit_price": price, "amount": np.round(units * price, 2),
                         "discount_pct": rng.choice([0, 5, 10, 20], n),
                         "status": rng.choice(["Completed", "Cancelled", "Returned"], n, p=[.8, .12, .08]),
                         "gift_wrapped": rng.choice([0, 1], n, p=[.7, .3])})


def _creators(rng):
    n = 120
    f = rng.integers(500, 90000, n)
    return pd.DataFrame({"Unnamed: 0": np.arange(n), "creator": [f"creator_{i}" for i in range(n)],
                         "niche": rng.choice(["Food", "Travel", "Tech"], n),
                         "followers": [f"{x / 1000:.1f}k" if x >= 1000 else str(x) for x in f],
                         "avg_views": [f"{x:,}" for x in rng.integers(1000, 50000, n)],
                         "zip_code": rng.choice([560001, 560034, 110001, 400050], n),
                         "engagement_rate": np.round(rng.uniform(0.5, 9.5, n), 2)})


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """{name: (frame as the Copilot loads it, the raw frame pandas truth is computed from)}"""
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("stage1")
    out = {}
    for name, make in (("tickets", _tickets), ("shifts", _shifts), ("orders", _orders), ("creators", _creators)):
        raw = make(rng)
        path = root / f"{name}.csv"
        raw.to_csv(path, index=False)
        out[name] = (load_table(str(path)), raw)
    return out


def ask(q, df, memory=None):
    return orchestrator.ask(q, df, {} if memory is None else memory, use_model=False)


def headline(out):
    """The figure the answer rests on, whichever path produced it: a number, or {group: number}."""
    for c in out["calls"]:
        if c.get("status") != "ok":
            continue
        r = c.get("result") or {}
        if c["tool"] == "analysis_program":
            res = r["results"][""]
            v = (res.get("per_unit") or res.get("result"))["value"]
            return v
        if c["tool"] == "compute_metric":
            return {str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r["value"]
        if c["tool"] == "calculator":
            t = r.get("totals") or {}
            return t.get("records_meeting", next(iter(t.values()), None))
    return None


def computed(out) -> bool:
    return any(c.get("status") == "ok" for c in out["calls"])


def refused(out) -> bool:
    """Nothing was calculated and the answer says why."""
    return not computed(out) and out["plan"].get("type") in ("decline", "clarify") and out.get("verify_status") != "verified"


# ----------------------------------------------------- stated filters are applied --
THRESHOLDS = [
    ("tickets", "Total resolution hours for tickets with csat below 3", lambda t: t[t.csat < 3].resolution_hours.sum()),
    ("tickets", "Add up the resolution hours of tickets whose csat is under 3", lambda t: t[t.csat < 3].resolution_hours.sum()),
    ("tickets", "median resolution hours where csat is at least 4", lambda t: t[t.csat >= 4].resolution_hours.median()),
    ("tickets", "how many tickets have a csat of 4 or more", lambda t: float((t.csat >= 4).sum())),
    ("orders", "What is the mean discount pct for orders over 500 in amount?", lambda o: o[o.amount > 500].discount_pct.mean()),
    ("orders", "average discount pct where amount is greater than 500", lambda o: o[o.amount > 500].discount_pct.mean()),
    ("orders", "total amount for orders with more than 5 units in the West",
     lambda o: o[(o.units > 5) & (o.region == "West")].amount.sum()),
    ("orders", "total amount for orders with no more than 2 units", lambda o: o[o.units <= 2].amount.sum()),
]


@pytest.mark.parametrize("ds,q,truth", THRESHOLDS)
def test_a_stated_threshold_filters_the_records(data, ds, q, truth):
    df, raw = data[ds]
    out = ask(q, df)
    assert headline(out) == pytest.approx(truth(raw)), out["answer"][:300]
    assert out["verify_status"] == "verified"


FLAGS = [
    ("tickets", "Average resolution hours of only escalated tickets", lambda t: t[t.escalated == "Yes"].resolution_hours.mean()),
    ("tickets", "What is the mean resolution time for escalated high priority tickets?",
     lambda t: t[(t.escalated == "Yes") & (t.priority == "High")].resolution_hours.mean()),
    ("tickets", "Average resolution hours for non-escalated tickets handled by Ben Cole",
     lambda t: t[(t.escalated == "No") & (t.agent == "Ben Cole")].resolution_hours.mean()),
    ("tickets", "How many tickets were not escalated?", lambda t: float((t.escalated == "No").sum())),
    ("tickets", "How many tickets were escalated?", lambda t: float((t.escalated == "Yes").sum())),
    ("orders", "average amount of gift wrapped orders", lambda o: o[o.gift_wrapped == 1].amount.mean()),
    ("orders", "Average amount for orders that are not gift wrapped", lambda o: o[o.gift_wrapped == 0].amount.mean()),
    ("orders", "total amount of gift wrapped orders in the East",
     lambda o: o[(o.gift_wrapped == 1) & (o.region == "East")].amount.sum()),
]


@pytest.mark.parametrize("ds,q,truth", FLAGS, ids=[q for _, q, _ in FLAGS])
def test_a_yes_no_column_said_about_the_records_filters_them(data, ds, q, truth):
    """... and is never itself the number asked for ("average amount of gift wrapped orders" is not the average of
    gift_wrapped)."""
    df, raw = data[ds]
    out = ask(q, df)
    assert headline(out) == pytest.approx(truth(raw)), out["answer"][:300]
    assert out["verify_status"] == "verified"


def test_the_flag_is_still_the_measure_when_its_rate_is_asked(data):
    df, raw = data["tickets"]
    out = ask("What is the escalation rate for High priority tickets?", df)
    assert headline(out) == pytest.approx((raw[raw.priority == "High"].escalated == "Yes").mean())
    out = ask("Compare resolution hours between escalated and non-escalated tickets", df)
    groups = {x["group"]: x["mean"] for x in next(c for c in out["calls"] if c["tool"] == "compare_groups")["result"]["summary"]}
    assert groups == pytest.approx(raw.groupby("escalated").resolution_hours.mean().to_dict())      # both states: a breakdown


def test_the_request_spec_records_each_stated_restriction(data):
    t, o = data["tickets"][0], data["orders"][0]
    r = intent.read("Total resolution hours for tickets with csat below 3", t)
    assert r.measures == ["resolution_hours"] and [(c["column"], c["op"], c["value"]) for c in r.conditions] == [("csat", "<", 3.0)]
    r = intent.read("average amount of gift wrapped orders", o)
    assert r.measures == ["amount"] and r.restrict == ["gift_wrapped"] and not r.restrict_not
    r = intent.read("How many tickets were not escalated?", t)
    assert r.quantity == "count" and r.restrict_not == ["escalated"] and not r.measures
    r = intent.read("What if unit price increases by more than 10%?", o)
    assert not r.conditions and not r.unresolved, "a number in a scenario is its input, not a filter"
    r = intent.read("Show the top 5 regions by amount over the last 30 days", o)
    assert not r.conditions and not r.unresolved


def test_a_threshold_with_no_column_is_asked_about(data):
    out = ask("How many tickets are above 30?", data["tickets"][0])
    assert refused(out) and "above 30" in out["answer"]


# ----------------------------------------- names that are not in the data --
@pytest.mark.parametrize("ds,q,name", [
    ("tickets", "average resolution hours for Zed Zorro", "Zed Zorro"),
    ("tickets", "How many tickets did Zed Zorro handle?", "Zed Zorro"),
    ("orders", "average discount pct for Atlantis", "Atlantis"),
    ("shifts", "average hourly rate for Rahul Sharma", "Rahul Sharma"),         # Rahul Mehta exists; Sharma does not
])
def test_a_name_that_is_not_in_the_data_is_never_answered_over_all_records(data, ds, q, name):
    out = ask(q, data[ds][0])
    assert refused(out) and name in out["answer"] and "No records match" in out["answer"]
    assert out.get("verify_status") is None


def test_a_near_miss_name_is_offered_not_assumed(data):
    out = ask("average hourly rate for Rahul Sharma", data["shifts"][0])
    assert "Rahul Mehta" in out["answer"]


@pytest.mark.parametrize("ds,q", [
    ("tickets", "Average resolution hours for Asha Rao"),
    ("tickets", "What fraction of Ben Cole's tickets were escalated?"),
    ("orders", "Total amount for Completed orders in the North and South regions"),
    ("shifts", "average hourly rate in Engineering"),
    ("tickets", "How many tickets were opened in October?"),
    ("orders", "What is the Pearson correlation between units and amount?"),
    ("orders", "Show Total Amount By Region"),
])
def test_names_that_are_in_the_data_and_ordinary_capitals_are_not_stopped(data, ds, q):
    assert not intent.read(q, data[ds][0]).unmatched


def test_a_real_name_still_filters(data):
    df, raw = data["tickets"]
    out = ask("Average resolution hours for Asha Rao", df)
    assert headline(out) == pytest.approx(raw[raw.agent == "Asha Rao"].resolution_hours.mean())


# ------------------------------------------------------- column roles are enforced --
def test_identifiers_and_codes_are_recognised_from_name_and_values(data):
    roles = {c: r["role"] for c, r in semantics.profile(data["orders"][0])["roles"].items()}
    assert roles["order_id"] == roles["customer_id"] == "identifier", "customer_id repeats, and is still a label"
    assert roles["units"] == "count" and roles["amount"] == "amount" and roles["gift_wrapped"] == "flag"
    assert semantics.profile(data["creators"][0])["roles"]["zip_code"]["role"] == "identifier"
    assert semantics.profile(data["shifts"][0])["roles"]["hourly_rate"]["role"] == "amount", "money per hour, not a share"
    # a number that merely happens to be unique per row in a small table is not refused
    small = pd.DataFrame({"g": ["x", "y"], "net_sales": [100, 200]})
    assert semantics.aggregation_problem(small, "net_sales", "sum") is None
    assert semantics.aggregation_problem(pd.DataFrame({"number_of_orders": [3, 4, 4, 9]}), "number_of_orders", "sum") is None


IDENTIFIER_QUESTIONS = [
    ("orders", "What is the average customer id?", "customer_id"),
    ("orders", "sum of customer_id by status", "customer_id"),
    ("orders", "average order id by region", "order_id"),
    ("orders", "Top 5 regions by customer id", "customer_id"),
    ("orders", "Which region has the highest customer id?", "customer_id"),
    ("orders", "correlation between customer id and amount", "customer_id"),
    ("orders", "Show the trend of customer id over time", "customer_id"),
    ("orders", "Find outliers in order id", "order_id"),
    ("tickets", "average ticket id", "ticket_id"),
    ("tickets", "describe ticket id", "ticket_id"),
    ("creators", "average zip code", "zip_code"),
]


@pytest.mark.parametrize("ds,q,col", IDENTIFIER_QUESTIONS, ids=[q for _, q, _ in IDENTIFIER_QUESTIONS])
def test_an_identifier_is_never_aggregated_ranked_or_trended(data, ds, q, col):
    out = ask(q, data[ds][0])
    assert not computed(out), f"computed something for an identifier: {out['answer'][:200]}"
    assert out["plan"].get("type") == "decline" and f"`{col}`" in out["answer"] and "identif" in out["answer"]
    assert out.get("verify_status") != "verified"


def test_identifiers_can_still_be_counted_and_filtered(data):
    df, raw = data["shifts"]
    out = ask("How many employees are there?", df)
    assert headline(out) == pytest.approx(raw.employee.nunique())
    from app.analyst import tools
    o = data["orders"][0]
    call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "count_distinct", "column": "customer_id"}}, o)
    assert tools.run(call, o, {})["value"] == data["orders"][1].customer_id.nunique()
    with pytest.raises(tools.RoleRefusal):
        tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "avg", "column": "customer_id"}}, o)


LEVEL_TOTALS = [("orders", "sum of unit price", "unit_price"), ("shifts", "Total hourly rate", "hourly_rate"),
                ("tickets", "total csat by channel", "csat"), ("orders", "total discount pct by region", "discount_pct"),
                ("creators", "total avg_views by niche", "avg_views")]


@pytest.mark.parametrize("ds,q,col", LEVEL_TOTALS, ids=[q for _, q, _ in LEVEL_TOTALS])
def test_a_per_record_level_is_not_added_up(data, ds, q, col):
    out = ask(q, data[ds][0])
    assert not computed(out) and out["plan"].get("type") == "decline" and f"`{col}`" in out["answer"]


def test_levels_are_still_averaged_and_amounts_still_totalled(data):
    df, raw = data["shifts"]
    out = ask("average hourly rate by department", df)
    assert headline(out) == pytest.approx(raw.groupby("department").hourly_rate.mean().to_dict())
    df, raw = data["orders"]
    assert headline(ask("Total amount by region", df)) == pytest.approx(raw.groupby("region").amount.sum().to_dict())
    assert headline(ask("total units by region", df)) == pytest.approx(raw.groupby("region").units.sum().astype(float).to_dict())


def test_the_program_engine_and_the_calculator_follow_the_same_rule(data):
    from app.analyst import calc
    from app.analyst import program as P
    o = data["orders"][0]
    with pytest.raises(P.ProgramError, match="identif"):
        P.run([{"id": "r", "op": "aggregate", "from": "data", "fn": "mean", "column": "customer_id"}], o)
    with pytest.raises(P.ProgramError, match="per-record level"):
        P.run([{"id": "r", "op": "aggregate", "from": "data", "fn": "sum", "column": "unit_price"}], o)
    assert P.run([{"id": "r", "op": "aggregate", "from": "data", "fn": "nunique", "column": "customer_id"}], o)["r"].value \
        == data["orders"][1].customer_id.nunique()
    assert P.run([{"id": "r", "op": "aggregate", "from": "data", "fn": "mean", "column": "unit_price"}], o)["r"].value \
        == pytest.approx(data["orders"][1].unit_price.mean())
    with pytest.raises(calc.CalcError, match="identif"):
        calc.run({"derive": {}, "exclude": [], "aggregate": {"x": "mean(customer_id)"}, "group_by": None}, o)


# ------------------------------------------------- a share of a measure's total --
def test_a_share_of_a_measure_is_a_share_of_its_total_not_of_the_records(data):
    df, raw = data["orders"]
    east_value = raw[raw.region == "East"].amount.sum() / raw.amount.sum()
    east_rows = (raw.region == "East").mean()
    assert abs(east_value - east_rows) > 1e-3, "the two readings must differ for this test to mean anything"
    for q in ("What percentage of amount comes from the East region?", "What share of the amount is from East?"):
        out = ask(q, df)
        assert headline(out) == pytest.approx(east_value), q
        assert out["verify_status"] == "verified" and "%" in out["answer"]
    out = ask("What share of total amount does each region contribute?", df)
    assert headline(out) == pytest.approx((raw.groupby("region").amount.sum() / raw.amount.sum()).to_dict())
    out = ask("What share of units were sold in cancelled orders?", df)
    assert headline(out) == pytest.approx(raw[raw.status == "Cancelled"].units.sum() / raw.units.sum())


def test_a_share_of_revenue_uses_the_one_money_total_and_says_so(data):
    """Stage 2: 'revenue' is read as the one money total whose name says nothing else -- stated in the answer."""
    df, raw = data["orders"]
    out = ask("What percentage of revenue comes from the East region?", df)
    assert headline(out) == pytest.approx(raw[raw.region == "East"].amount.sum() / raw.amount.sum())
    assert "“revenue” read as `amount`" in out["answer"]


def test_a_share_of_records_is_still_a_share_of_records(data):
    df, raw = data["orders"]
    out = ask("What share of orders were returned?", df)
    assert computed(out) and f"{(raw.status == 'Returned').mean() * 100:.1f}%" in out["answer"]
    assert intent.read("What share of orders were returned?", df).share_of is None


# ---------------------------------------------- nothing wrong is marked verified --
def test_the_verifier_rejects_an_answer_that_ignored_a_stated_restriction(data):
    """Independently of which path answered: the executed call is compared with the question's own requirements."""
    from app.analyst import tools, verify
    df, raw = data["tickets"]
    call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "sum", "column": "resolution_hours"}}, df)
    wrong = {"plan_source": "rules", "calls": [{**call, "status": "ok", "result": tools.run(call, df, {})}]}
    for q in ("Total resolution hours for tickets with csat below 3", "Total resolution hours of only escalated tickets",
              "Total resolution hours for tickets that were not escalated"):
        v = verify.check_answer(q, df, wrong, None, {})
        assert v["status"] == "failed", q
    assert verify.check_answer("Total resolution hours", df, wrong, None, {})["status"] == "verified"
    filtered = tools.validate_call({"tool": "compute_metric", "args": {
        "aggregation": "sum", "column": "resolution_hours", "filters": [{"column": "csat", "op": "<", "value": 3}]}}, df)
    right = {"plan_source": "rules", "calls": [{**filtered, "status": "ok", "result": tools.run(filtered, df, {})}]}
    assert verify.check_answer("Total resolution hours for tickets with csat below 3", df, right, None, {})["status"] == "verified"


# ------------------------------------------------------------------ Ambition Box --
@pytest.fixture(scope="module")
def ambition():
    if not os.path.exists(AMBITION):
        pytest.skip("Ambition Box.csv is not in the repository root")
    return load_table(AMBITION)


def test_ambition_box_names_in_a_long_column_filter_or_are_reported(ambition):
    """4,000 company names: a company the question names is selected (not ignored), and one that is not there is
    reported. The general rule under test is 'a named value of a long text column' -- nothing here names a column."""
    name_col = next(c for c in ambition.columns if ambition[c].nunique() > 1000 and ambition[c].dtype == object)
    measure = next(c for c in ambition.columns if pd.api.types.is_float_dtype(ambition[c]))
    company = ambition[name_col].value_counts().index[0]
    truth = pd.to_numeric(ambition.loc[ambition[name_col] == company, measure], errors="coerce").mean()
    overall = pd.to_numeric(ambition[measure], errors="coerce").mean()
    out = ask(f"average {measure} for {company}", ambition)
    if company[:1].isupper() and re.fullmatch(r"[A-Za-z][\w' -]*", company):
        assert headline(out) == pytest.approx(truth) or refused(out)
    assert not (computed(out) and headline(out) == pytest.approx(overall) and abs(truth - overall) > 1e-9)
    out = ask(f"average {measure} for Wakanda Vibranium Corp", ambition)
    assert refused(out) and "No records match" in out["answer"]


def test_ambition_box_existing_answers_are_unchanged(ambition):
    measure = next(c for c in ambition.columns if pd.api.types.is_float_dtype(ambition[c]))
    out = ask(f"What is the average {measure}?", ambition)
    assert headline(out) == pytest.approx(pd.to_numeric(ambition[measure], errors="coerce").mean())
    assert out["verify_status"] == "verified"
    out = ask(f"What is the total {measure}?", ambition)          # a rating: a level, not a quantity to add up
    assert (not computed(out)) == (not semantics.additive(ambition, measure))


