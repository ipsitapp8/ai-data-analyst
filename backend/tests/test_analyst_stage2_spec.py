"""Stage 2 of the Copilot reliability repair: one structured reading of the question (`intent.Request`), consumed by
every path.

The datasets are the four synthetic domains of the Stage 1 tests (support tickets, shifts, retail orders, a messy
creator file), built from a seed -- no column or value here is known to Silt's code. Each test checks either

  * the structured reading itself (measure, aggregation, grouping, filters and their negations, thresholds, periods,
    distinct counts, event rates, weighted shares, sort order), or
  * an answer against pandas, or an explicit refusal / clarification, or
  * where routing a question type to the program engine is Stage 3's work, the deterministic program the reading
    produces, executed and compared with pandas.

Negative cases are included: a word that names no column is never mapped to an unrelated measure; an identifier is
never a measure; a value of a category is a filter, not a measure; an ambiguous date is asked about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import intent, nl_planner, orchestrator, scope, semantics
from app.analyst.data import load_table

from test_analyst_stage1_guards import TODAY, _creators, _orders, _shifts, _tickets, pinned_clock  # noqa: F401 - same seeded data


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("stage2")
    out = {}
    for name, make in (("tickets", _tickets), ("shifts", _shifts), ("orders", _orders), ("creators", _creators)):
        raw = make(rng)
        path = root / f"{name}.csv"
        raw.to_csv(path, index=False)
        out[name] = (load_table(str(path)), raw)
    return out


def ask(q, df, memory=None):
    return orchestrator.ask(q, df, {} if memory is None else memory, use_model=False)


def read(q, df):
    return intent.read(q, df)


def program_value(q, df):
    """The deterministic program built from the structured reading, executed (Stage 3 routes questions to it)."""
    p = nl_planner.rule_plan(q, df, {})
    assert p is not None, f"no program could be built from the reading of {q!r}"
    r = nl_planner.execute(p, df, {}, q)
    return r["results"][p["answer"]]["value"]


def computed(out):
    return any(c.get("status") == "ok" for c in out["calls"])


def headline(out):
    for c in out["calls"]:
        if c.get("status") != "ok":
            continue
        r = c.get("result") or {}
        if c["tool"] == "analysis_program":
            res = r["results"][""]
            return (res.get("per_unit") or res.get("result"))["value"]
        if c["tool"] == "compute_metric":
            return {str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r["value"]
    return None


# --------------------------------------------------------------- aggregation words --
@pytest.mark.parametrize("q,quantity", [
    ("Add up the resolution hours by channel", "total"),
    ("sum up resolution hours for Phone tickets", "total"),
    ("combined resolution hours of Email tickets", "total"),
    ("What is the typical resolution time?", "median"),
    ("On average, how long does resolution take?", "average"),
    ("mean csat by priority", "average"),
    ("What proportion of tickets were escalated?", "rate"),
    ("How many tickets were opened last month?", "count"),
])
def test_quantity_words_are_read_the_same_whatever_the_phrasing(data, q, quantity):
    assert read(q, data["tickets"][0]).quantity == quantity


def test_add_up_with_a_threshold_is_answered_exactly(data):
    df, raw = data["tickets"]
    out = ask("Add up the resolution hours of tickets whose csat is under 3", df)
    assert headline(out) == pytest.approx(raw[raw.csat < 3].resolution_hours.sum())
    assert out["verify_status"] == "verified"


# ---------------------------------------------------------------- distinct counts --
@pytest.mark.parametrize("ds,q,col", [
    ("orders", "How many customers do we have?", "customer_id"),
    ("orders", "number of unique customers by region", "customer_id"),
    ("tickets", "count of distinct agents", "agent"),
    ("tickets", "How many different channels are there?", "channel"),
    ("shifts", "How many employees are there?", "employee"),
])
def test_how_many_of_a_repeating_thing_counts_its_distinct_values(data, ds, q, col):
    r = read(q, data[ds][0])
    assert r.quantity == "count" and r.counted == col and not r.measures
    assert r.group != col, "the thing counted is not also the breakdown"


def test_distinct_counts_are_computed_from_the_reading(data):
    df, raw = data["orders"]
    assert program_value("number of unique customers by region", df) == pytest.approx(
        raw.groupby("region").customer_id.nunique().astype(float).to_dict())
    df, raw = data["tickets"]
    assert program_value("count of distinct agents", df) == raw.agent.nunique()
    df, raw = data["shifts"]
    out = ask("How many employees are there?", df)
    assert headline(out) == pytest.approx(raw.employee.nunique()) and out["verify_status"] == "verified"


def test_records_with_one_identifier_each_are_counted_as_records(data):
    """'how many shifts' / 'how many tickets': each record is one; the records are counted (duplicates are disclosed
    by the duplicate policy), not an identifier's distinct values."""
    for ds, q in (("shifts", "How many shifts are there?"), ("tickets", "How many tickets were escalated?")):
        r = read(q, data[ds][0])
        assert r.quantity == "count" and r.counted is None and not r.measures, q


def test_counts_by_a_column_are_a_breakdown_not_a_distinct_count(data):
    r = read("Show ticket counts by channel", data["tickets"][0])
    assert r.quantity == "count" and r.counted is None and r.group == "channel"


# ------------------------------------------------------------ rates of an event --
@pytest.mark.parametrize("q,event,group", [
    ("Which region has the highest cancellation rate?", ("status", "Cancelled"), "region"),
    ("What is the return rate for Toys?", ("status", "Returned"), None),
    ("cancellation rate by product line", ("status", "Cancelled"), "product_line"),
    ("How often are orders returned, by region?", None, "region"),     # "how often" + a stem: see below
])
def test_a_rate_of_a_category_value_is_an_event_not_a_filter(data, q, event, group):
    r = read(q, data["orders"][0])
    if event is not None:
        assert r.quantity == "rate" and r.event == event and not r.measures, q
        assert event not in r.entities, "the event is what is measured -- filtering to it would make every share 100%"
    assert r.group == group


def test_event_rates_are_computed_from_the_reading(data):
    df, raw = data["orders"]
    want = (raw.status == "Cancelled").groupby(raw.region).mean().to_dict()
    assert program_value("Which region has the highest cancellation rate?", df) == pytest.approx(want)
    toys = raw[raw.product_line == "Toys"]
    assert program_value("What is the return rate for Toys?", df) == pytest.approx((toys.status == "Returned").mean())


def test_the_scope_takes_the_event_out_of_the_record_selection(data):
    df = data["orders"][0]
    sc = scope.establish("What share of orders were Cancelled?", df, {})
    assert "status" not in (sc.get("entities") or {})


def test_a_flag_rate_still_reads_the_flag_as_the_measure(data):
    r = read("What is the escalation rate by priority?", data["tickets"][0])
    assert r.quantity == "rate" and r.measures == ["escalated"] and r.group == "priority" and r.event is None


# ------------------------------------------------------- filters, all in one place --
def test_filters_and_negations_are_represented_together(data):
    df = data["tickets"][0]
    r = read("Average resolution hours for non-escalated Email tickets with csat of 4 or more, excluding Ben Cole", df)
    assert r.measures == ["resolution_hours"] and r.quantity == "average"
    assert r.restrict_not == ["escalated"] and not r.restrict
    assert ("channel", "Email") in r.entities and ("agent", "Ben Cole") in r.excluded
    assert [(c["column"], c["op"], c["value"]) for c in r.conditions] == [("csat", ">=", 4.0)]


def test_the_shared_scope_carries_every_filter_to_every_path(data):
    df, raw = data["tickets"]
    q = "Average resolution hours for non-escalated Email tickets with csat of 4 or more"
    sc = scope.establish(q, df, {})
    kept = df[scope.mask(df, sc)]
    want = raw[(raw.escalated == "No") & (raw.channel == "Email") & (raw.csat >= 4)]
    assert len(kept) == len(want) and kept.resolution_hours.mean() == pytest.approx(want.resolution_hours.mean())
    assert "csat >= 4" in scope.describe(sc)
    # ... and as calculator exclusions, so a calculation held to the scope keeps the same records
    from app.analyst import calc
    rules = scope.as_exclusions(sc)
    drop = pd.Series(False, index=df.index)
    for e in rules:
        drop |= calc.evaluate(e["where"], df, df).fillna(False).astype(bool)
    assert int((~drop).sum()) == len(want)
    out = ask(q, df)
    assert headline(out) == pytest.approx(want.resolution_hours.mean()) and out["verify_status"] == "verified"


def test_a_threshold_is_read_with_the_column_it_is_said_about(data):
    df, raw = data["tickets"]
    r = read("How many tickets took longer than 10 hours to resolve?", df)
    assert r.quantity == "count" and not r.measures
    assert [(c["column"], c["op"], c["value"]) for c in r.conditions] == [("resolution_hours", ">", 10.0)]
    out = ask("How many tickets took longer than 10 hours to resolve?", df)
    assert headline(out) == pytest.approx((raw.resolution_hours > 10).sum())


# --------------------------------------------------------------------- periods --
def test_explicit_date_ranges_are_read_with_both_ends(data):
    df, raw = data["orders"]
    r = read("total amount between 1 August 2026 and 15 August 2026", df)
    assert r.period["start"] == "2026-08-01" and r.period["end"] == "2026-08-15" and r.calendar is None
    out = ask("total amount between 1 August 2026 and 15 August 2026", df)
    want = raw[(raw.order_date >= "2026-08-01") & (raw.order_date <= "2026-08-15")].amount.sum()
    assert headline(out) == pytest.approx(want) and out["verify_status"] == "verified"
    out = ask("total amount from Aug 1 to Aug 15, 2026", df)
    assert headline(out) == pytest.approx(want)
    out = ask("total amount since 1 October 2026", df)
    assert headline(out) == pytest.approx(raw[raw.order_date >= "2026-10-01"].amount.sum())
    out = ask("total amount before 2026-06-01", df)
    assert headline(out) == pytest.approx(raw[raw.order_date < "2026-06-01"].amount.sum())


def test_a_named_month_stays_a_calendar_selection(data):
    r = read("total amount in August 2026", data["orders"][0])
    assert r.period is None and r.calendar == {"column": "order_date", "year": 2026, "month": 8}


def test_an_ambiguous_date_is_asked_about_not_guessed(data):
    out = ask("total amount between 01/02/2026 and 03/04/2026", data["orders"][0])
    assert not computed(out) and out["plan"]["type"] == "clarify" and "day/month" in out["answer"]
    out = ask("total amount between 13/02/2026 and 20/03/2026", data["orders"][0])   # unambiguous: day first
    raw = data["orders"][1]
    assert headline(out) == pytest.approx(raw[(raw.order_date >= "2026-02-13") & (raw.order_date <= "2026-03-20")].amount.sum())


# --------------------------------------------------------- measures and their words --
def test_money_words_map_to_the_one_money_total_and_the_answer_says_so(data):
    df, raw = data["orders"]
    for q in ("total sales by region", "total revenue by region", "Which region brings in the most money?"):
        out = ask(q, df)
        assert headline(out) == pytest.approx(raw.groupby("region").amount.sum().to_dict()), q
        assert "read as `amount`" in out["answer"], q
    cands, why = semantics.money_in_columns(df)
    assert cands == ["amount"], "unit_price is a price, units a quantity: neither is revenue"


def test_money_words_are_never_mapped_without_evidence(data):
    """No money total (shifts, tickets): nothing is read as revenue; two candidates: Silt asks; a column named for
    another kind of money (cost): never revenue; a department called Sales: a filter, not a measure."""
    for ds in ("shifts", "tickets"):
        assert semantics.money_in_columns(data[ds][0])[0] == []
        out = ask("total revenue by department" if ds == "shifts" else "total sales by channel", data[ds][0])
        assert "read as" not in out["answer"] and out.get("verify_status") != "verified"
    rng = np.random.default_rng(1)
    two = pd.DataFrame({"store": list("ab") * 10, "gross_amount": rng.gamma(3, 20, 20).round(2)})
    two["net_amount"] = (two.gross_amount * 0.8).round(2)
    out = ask("total revenue by store", two)
    assert out["plan"]["type"] == "clarify" and set(out["plan"]["options"]) == {"gross_amount", "net_amount"}
    cost = pd.DataFrame({"store": list("ab") * 10, "cost": np.arange(20.0), "units": np.arange(20)})
    assert semantics.money_in_columns(cost)[0] == []
    dept = pd.DataFrame({"department": ["Sales", "Ops"] * 10, "amount": np.arange(20.0)})
    r = read("total amount for Sales", dept)
    assert ("department", "Sales") in r.entities and r.measures == ["amount"]
    assert semantics.measure_words("total amount for Sales", dept)["rewrite"] == []


def test_average_order_value_uses_the_money_total_per_distinct_order(data):
    df, raw = data["orders"]
    out = ask("What is the average order value by region?", df)
    groups = {g["group"]: g["average_order_value"] for g in next(
        c for c in out["calls"] if c["tool"] == "calculator")["result"]["groups"]}
    want = (raw.groupby("region").amount.sum() / raw.groupby("region").order_id.nunique()).to_dict()
    assert groups == pytest.approx(want)
    assert "`amount` is read as the revenue" in out["answer"] or "amount" in out["answer"]


def test_per_customer_divides_by_distinct_customers_never_by_their_ids(data):
    df, raw = data["orders"]
    out = ask("Revenue per customer by region", df)
    want = (raw.groupby("region").amount.sum() / raw.groupby("region").customer_id.nunique()).to_dict()
    assert headline(out) == pytest.approx(want) and out["verify_status"] == "verified"


@pytest.mark.parametrize("ds,q,measure", [
    ("tickets", "average time to resolve by channel", "resolution_hours"),
    ("tickets", "How long does resolution take on average?", "resolution_hours"),
    ("shifts", "average hourly rate in Engineering", "hourly_rate"),
    ("shifts", "What is the mean pay rate per hour by department?", "hourly_rate"),
    ("creators", "Which niche has the highest engagement?", "engagement_rate"),
])
def test_everyday_words_for_a_measure_reach_the_right_column(data, ds, q, measure):
    assert read(q, data[ds][0]).measures == [measure]


def test_the_resolved_measure_is_what_every_path_computes(data):
    df, raw = data["tickets"]
    out = ask("average time to resolve by channel", df)
    assert headline(out) == pytest.approx(raw.groupby("channel").resolution_hours.mean().to_dict())
    assert out["verify_status"] == "verified"


def test_per_a_multi_word_category_is_a_breakdown_not_a_ratio(data):
    df, raw = data["orders"]
    r = read("total units sold per product line, sorted from lowest to highest", df)
    assert r.group == "product_line" and not r.per and r.sort == "asc"
    out = ask("total units sold per product line, sorted from lowest to highest", df)
    shown = list(headline(out))
    assert shown == list(raw.groupby("product_line").units.sum().sort_values().index)


# --------------------------------------------- identifiers stay identifiers (Stage 1) --
def test_an_identifier_is_never_the_measure_of_a_reading(data):
    for ds, q in (("orders", "What is the average customer id?"), ("tickets", "average ticket id"),
                  ("creators", "average zip code")):
        r = read(q, data[ds][0])
        assert not r.measures, q
        assert ask(q, data[ds][0])["plan"]["type"] == "decline"


def test_identifiers_can_be_distinct_counted_and_grouped(data):
    df, raw = data["orders"]
    assert program_value("number of unique customers by region", df) == pytest.approx(
        raw.groupby("region").customer_id.nunique().astype(float).to_dict())


# ------------------------------------------------ the verifier holds every path to it --
def test_an_answer_about_another_measure_is_rejected_whatever_path_produced_it(data):
    from app.analyst import tools, verify
    df = data["tickets"][0]
    call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "avg", "column": "csat"}}, df)
    wrong = {"plan_source": "rules", "calls": [{**call, "status": "ok", "result": tools.run(call, df, {})}]}
    v = verify.check_answer("average time to resolve", df, wrong, None, {})
    assert v["status"] == "failed" and "resolution_hours" in v["checks"][0]["detail"]


def test_records_counted_where_distinct_values_were_asked_is_rejected(data):
    from app.analyst import tools, verify
    df = data["orders"][0]
    call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "count"}}, df)
    wrong = {"plan_source": "rules", "calls": [{**call, "status": "ok", "result": tools.run(call, df, {})}]}
    v = verify.check_answer("How many customers do we have?", df, wrong, None, {})
    assert v["status"] == "failed" and "distinct customer_id" in v["checks"][0]["detail"]


def test_one_reading_per_question_is_shared_and_not_mutated_by_callers(data):
    df = data["orders"][0]
    a = read("total amount by region", df)
    a.group = None                                      # a caller's change must not leak to the next reader
    assert read("total amount by region", df).group == "region"
