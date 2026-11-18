"""Stage 3 of the Copilot reliability repair: questions the structured reading and the program engine understand are
routed to them, end to end.

Same seeded data as Stages 1-2 (tickets, shifts, orders, a messy creator file); nothing here is known to Silt's code.
Every answer is checked against pandas; every question that cannot be answered without guessing must get an honest
clarification or refusal -- the router never turns a placeholder into a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator
from app.analyst.data import load_table

from test_analyst_stage1_guards import TODAY, _creators, _orders, _shifts, _tickets, pinned_clock  # noqa: F401 - same seeded data


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("stage3")
    out = {}
    for name, make in (("tickets", _tickets), ("shifts", _shifts), ("orders", _orders), ("creators", _creators)):
        raw = make(rng)
        path = root / f"{name}.csv"
        raw.to_csv(path, index=False)
        out[name] = (load_table(str(path)), raw)
    return out


def ask(q, df, memory=None):
    return orchestrator.ask(q, df, {} if memory is None else memory, use_model=False)


def computed(out):
    return any(c.get("status") == "ok" for c in out["calls"])


def values(out):
    """Every headline figure of the answer, whichever deterministic tool produced it (one per program / call)."""
    got = []
    for c in out["calls"]:
        if c.get("status") != "ok":
            continue
        r = c.get("result") or {}
        if c["tool"] == "analysis_program":
            res = r["results"][""]
            got.append((res.get("per_unit") or res.get("result"))["value"])
        elif c["tool"] == "compute_metric":
            got.append({str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r["value"])
        elif c["tool"] == "calculator":
            if r.get("groups"):
                key = next(k for k in r["groups"][0] if k not in ("group", "rows"))
                got.append({str(g["group"]): g[key] for g in r["groups"]})
            else:
                got.append(r.get("totals"))
        elif c["tool"] == "correlation":
            got.append(r)
        elif c["tool"] == "compare_groups":
            got.append({x["group"]: x["mean"] for x in r["summary"]})
    return got


def first(out):
    v = values(out)
    assert v, f"nothing was computed: {out['answer'][:300]}"
    return v[0]


def not_a_dead_end(out):
    return "I couldn't turn" not in out["answer"]


# ------------------------------------------------------------- the nine questions --
def test_how_many_customers_counts_distinct_customers(data):
    df, raw = data["orders"]
    for q in ("How many customers?", "How many customers do we have?", "What is the number of customers?",
              "count the unique customers"):
        out = ask(q, df)
        assert first(out) == pytest.approx(raw.customer_id.nunique()), q
        assert out["verify_status"] == "verified", q


def test_count_of_distinct_agents(data):
    df, raw = data["tickets"]
    for q in ("Count of distinct agents.", "How many different agents are there?", "number of unique agents"):
        out = ask(q, df)
        assert first(out) == pytest.approx(raw.agent.nunique()), q
        assert out["verify_status"] == "verified", q


def test_how_many_shifts_counts_the_records_and_discloses_duplicates(data, tmp_path):
    raw = pd.concat([data["shifts"][1], data["shifts"][1].iloc[[3, 4, 5]]], ignore_index=True)   # 3 exact duplicates
    raw.to_csv(tmp_path / "shifts_dup.csv", index=False)
    df = load_table(str(tmp_path / "shifts_dup.csv"))
    for q in ("How many shifts?", "How many shifts are there?", "How many shifts are recorded?"):
        out = ask(q, df)
        assert first(out) == pytest.approx(len(raw)), q
        assert out["verify_status"] == "verified" and "duplicate" in out["answer"], q


def test_escalation_rate_by_priority(data):
    df, raw = data["tickets"]
    want = (raw.escalated == "Yes").groupby(raw.priority).mean().to_dict()
    for q in ("Escalation rate by priority.", "What share of tickets get escalated in each priority?",
              "How often are tickets escalated, by priority?"):
        out = ask(q, df)
        assert first(out) == pytest.approx(want), q
        # the calculator's defined share is honestly "not independently verified"; never a wrong ✓
        assert out["verify_status"] in ("verified", "not independently verified"), q


def test_highest_cancellation_rate(data):
    df, raw = data["orders"]
    want = (raw.status == "Cancelled").groupby(raw.region).mean()
    for q in ("Which region has the highest cancellation rate?", "Which region cancels the most orders, as a rate?",
              "cancellation rate by region"):
        out = ask(q, df)
        assert first(out) == pytest.approx(want.to_dict()), q
        assert out["verify_status"] == "verified", q
    out = ask("Which region has the highest cancellation rate?", df)
    assert f"**Highest: {want.idxmax()}**" in out["answer"]


def test_combined_amount_for_two_named_groups(data):
    df, raw = data["orders"]
    want = raw[raw.region.isin(["East", "West"])].amount.sum()
    for q in ("Combined amount for East and West.", "What is the total amount of the East and West regions together?",
              "add up the amount for East and West"):
        out = ask(q, df)
        assert first(out) == pytest.approx(want), q
        assert out["verify_status"] == "verified", q


def test_orders_per_region_and_typical_order_size(data):
    """Two questions: the first is answered; 'order size' names no column -- asked about, and the answer is marked
    as not complete rather than verified."""
    df, raw = data["orders"]
    out = ask("How many orders per region and what is the typical order size?", df)
    assert first(out) == pytest.approx(raw.region.value_counts().astype(float).to_dict())
    assert out["verify_status"] == "partially verified"
    assert "Which column measures this?" in out["answer"] and "`amount`" in out["answer"] and "`units`" in out["answer"]
    assert "`order_id`" not in out["answer"].split("Which column measures this?")[1]


def test_both_clauses_answered_when_both_are_defined(data):
    df, raw = data["orders"]
    out = ask("How many orders per region and what is the median amount by region?", df)
    v = values(out)
    assert v[0] == pytest.approx(raw.region.value_counts().astype(float).to_dict())
    assert v[1] == pytest.approx(raw.groupby("region").amount.median().to_dict())
    assert out["verify_status"] == "verified"


def test_a_count_and_a_whole_table_figure_are_both_answered_and_verified(data):
    """Regression found during Stage 3: splitting this into clauses must not lose the plain whole-table median."""
    df, raw = data["orders"]
    out = ask("How many orders and what is the median amount?", df)
    v = values(out)
    assert v[0] == pytest.approx(len(raw)) and v[1] == pytest.approx(raw.amount.median())
    assert out["verify_status"] == "verified"


def test_a_list_of_measures_on_one_table_is_not_split_into_questions(data):
    from app.analyst import intent
    df = data["orders"][0]
    assert intent.separate_scopes("Average amount, total units and number of orders by region", df) is None
    assert intent.separate_scopes("Combined amount for East and West", df) is None


def test_correlation_between_two_measures(data):
    df, raw = data["orders"]
    for q in ("Does discount correlate with higher amount?", "Is discount related to amount?",
              "Do discount and amount move together?"):
        out = ask(q, df)
        r = first(out)
        assert {r["x"], r["y"]} == {"discount_pct", "amount"}, q
        assert r["pearson_r"] == pytest.approx(raw.discount_pct.corr(raw.amount), abs=1e-9), q
        assert "association, not causation" in out["answer"], q


def test_a_causal_question_is_answered_as_an_association_and_says_so(data):
    df = data["orders"][0]
    out = ask("Does discount cause higher amount?", df)
    assert computed(out) and "not whether one causes the other" in out["answer"]
    out = ask("Does the region affect amount?", df)
    raw = data["orders"][1]
    assert first(out) == pytest.approx(raw.groupby("region").amount.mean().to_dict())
    assert "not whether region causes the difference" in out["answer"]


def test_labour_cost_for_the_last_7_days(data):
    df, raw = data["shifts"]
    hrs = (pd.to_datetime(raw.clock_out, format="%H:%M") - pd.to_datetime(raw.clock_in, format="%H:%M")).dt.total_seconds() / 3600
    window = (raw.shift_date >= TODAY - pd.Timedelta(days=6)) & (raw.shift_date <= TODAY)
    ok = window & (raw.hourly_rate >= 0)
    want = (hrs * raw.hourly_rate)[ok].sum()
    for q in ("Total labour cost for the last 7 days.", "What did labor cost us over the past 7 days?"):
        out = ask(q, df)
        totals = first(out)
        assert totals["labor_cost"] == pytest.approx(want), q
        assert "last 7 days" in out["answer"] and out["verify_status"] != "withheld", q


# ------------------------------------------------------------ unseen phrasings, all domains --
@pytest.mark.parametrize("ds,q,truth", [
    ("tickets", "rate of escalation for Asha Rao", lambda t: (t[t.agent == "Asha Rao"].escalated == "Yes").mean()),
    ("tickets", "How many tickets?", lambda t: float(len(t))),
    ("tickets", "How many distinct channels?", lambda t: float(t.channel.nunique())),
    ("shifts", "How many distinct departments?", lambda s: float(s.department.nunique())),
    ("shifts", "share of shifts that were overtime in Finance", lambda s: s[s.department == "Finance"].overtime.mean()),
    ("orders", "What share of orders were cancelled in the North?",
     lambda o: (o[o.region == "North"].status == "Cancelled").mean()),
    ("orders", "How often are orders returned by region?", lambda o: (o.status == "Returned").groupby(o.region).mean().to_dict()),
    ("orders", "How many customers ordered more than 5 units?", lambda o: float(o[o.units > 5].customer_id.nunique())),
    ("creators", "How many niches are there?", lambda c: float(c.niche.nunique())),
])
def test_unseen_phrasings_route_to_the_right_program(data, ds, q, truth):
    df, raw = data[ds]
    out = ask(q, df)
    assert first(out) == pytest.approx(truth(raw)), out["answer"][:300]
    assert out["verify_status"] == "verified"


def test_two_questions_about_different_records_are_computed_separately(data):
    df, raw = data["orders"]
    out = ask("total amount for East and the average units for West", df)
    v = values(out)
    assert v[0] == pytest.approx(raw[raw.region == "East"].amount.sum())
    assert v[1] == pytest.approx(raw[raw.region == "West"].units.mean())
    assert not out["answer"].startswith("**Scope:**"), "one scope line for two different scopes would be wrong"


def test_creators_correlation_on_numbers_read_from_text(data):
    df, raw = data["creators"]
    out = ask("Is followers correlated with engagement rate?", df)
    r = first(out)
    assert r["n"] == len(raw) and not_a_dead_end(out)


# ------------------------------------------------------------------- negative cases --
@pytest.mark.parametrize("ds,q", [
    ("orders", "How many unicorns are there?"),
    ("orders", "What is the typical blorp?"),
    ("tickets", "How many widgets were sold?"),
])
def test_a_noun_or_measure_the_data_does_not_have_is_never_routed_to_a_count(data, ds, q):
    out = ask(q, data[ds][0])
    assert not computed(out) and out.get("verify_status") != "verified"


def test_conversation_and_concepts_are_not_turned_into_computations(data):
    df = data["orders"][0]
    for q in ("Hello", "What is Random Forest?", "thanks"):
        out = ask(q, df)
        assert not computed(out), q


def test_an_identifier_part_is_refused_and_the_other_part_is_answered(data):
    df, raw = data["orders"]
    out = ask("What is the average customer id and how many orders are there?", df)
    assert "`customer_id` identifies records" in out["answer"]
    assert first(out) == pytest.approx(len(raw))
    assert out["verify_status"] == "partially verified"


def test_clarifications_from_earlier_stages_still_stand(data):
    """The router acts only on placeholder replies -- a deliberate question to the user is never replaced."""
    df = data["orders"][0]
    out = ask("average discount pct for Atlantis", df)
    assert not computed(out) and "No records match" in out["answer"]
    out = ask("total amount between 01/02/2026 and 03/04/2026", df)
    assert not computed(out) and "day/month" in out["answer"]
    two = pd.DataFrame({"store": list("ab") * 10, "gross_amount": np.random.default_rng(1).gamma(3, 20, 20).round(2)})
    two["net_amount"] = (two.gross_amount * 0.8).round(2)
    out = ask("total revenue by store", two)
    assert not computed(out) and out["plan"]["type"] == "clarify"


def test_routed_answers_are_held_to_the_verifier(data):
    """A routed program goes through the same verification as any answer: the event measured must be the event
    asked about (Stage 3 found the metric path measuring the filter value instead)."""
    from app.analyst import tools, verify
    df = data["orders"][0]
    memory = {"last_calc": {"spec": {"derive": {}, "exclude": [], "group_by": None,
                                     "aggregate": {"share_region_north": "mean(region == 'North')"}},
                            "result": {"totals": {"share_region_north": 0.276}}, "meta": {"primary": "share_region_north"}}}
    out = {"plan_source": "metric", "calls": [{"tool": "calculator", "status": "ok", "result": {}}]}
    v = verify.check_answer("What share of orders were cancelled in the North?", df, out, None, memory)
    assert v["status"] == "failed" and "Cancelled" in v["checks"][0]["detail"]
