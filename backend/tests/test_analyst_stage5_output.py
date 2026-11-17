"""Stage 5 of the Copilot reliability repair: what an answer shows.

  * counts are whole numbers (88, not 88.00); genuine decimals keep their digits;
  * a tie at the top or bottom names every tied group;
  * one record-validity policy on every path (tool catalogue, rankings, column summaries, calculator, programs,
    verifier), with each excluded record and the reason it is impossible stated; amounts that can be negative are kept;
  * identifiers are never offered as measures, forecast targets or association factors -- and can still be counted;
  * a question about something the data does not hold says what is missing and what is available.

Seeded data of Stages 1-4 plus small purpose-built frames; nothing here is known to Silt's code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import explain, orchestrator, scope, semantics
from app.analyst.data import load_table

from test_analyst_stage1_guards import TODAY, _creators, _orders, _shifts, _tickets  # noqa: F401


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("stage5")
    out = {}
    for name, make in (("tickets", _tickets), ("shifts", _shifts), ("orders", _orders), ("creators", _creators)):
        raw = make(rng)
        if name == "shifts":
            raw.loc[5, "hourly_rate"] = -22.0                  # one impossible pay rate, as in the probe data
        path = root / f"{name}.csv"
        raw.to_csv(path, index=False)
        out[name] = (load_table(str(path)), raw)
    return out


def ask(q, df, memory=None):
    return orchestrator.ask(q, df, {} if memory is None else memory, use_model=False)


def computed(out):
    return any(c.get("status") == "ok" for c in out["calls"])


# ---------------------------------------------------------------------- counts --
def test_the_formatter_shows_whole_counts_and_keeps_decimals():
    assert explain.f(88) == "88" and explain.f(np.int64(4)) == "4" and explain.f(1234) == "1,234"
    assert explain.f(4.0) == "4.00" and explain.f(3.02) == "3.02" and explain.f(0.000123) == "0.000123"
    assert explain.f(2304.6) == "2,304.60"


@pytest.mark.parametrize("ds,q,truth", [
    ("tickets", "How many tickets were escalated?", lambda t: (t.escalated == "Yes").sum()),
    ("tickets", "count of distinct agents", lambda t: t.agent.nunique()),
    ("shifts", "How many employees are there?", lambda s: s.employee.nunique()),
    ("orders", "How many customers?", lambda o: o.customer_id.nunique()),
])
def test_counts_are_shown_as_whole_numbers(data, ds, q, truth):
    df, raw = data[ds]
    out = ask(q, df)
    n = int(truth(raw))
    import re as _re
    assert _re.search(rf"(\*\*|: | ){n}(\*\*| |\b)", out["answer"]), out["answer"][:200]
    assert f"{n}.00" not in out["answer"], out["answer"][:200]


def test_count_tables_have_no_decimal_points_but_averages_do(data):
    df, raw = data["orders"]
    out = ask("number of orders by region", df)
    for v in raw.region.value_counts():
        assert f"{v}.00" not in out["answer"]
    df, raw = data["tickets"]
    out = ask("average csat by channel", df)
    assert f"{raw[raw.channel == 'Email'].csat.mean():.2f}" in out["answer"]


# ------------------------------------------------------------------------ ties --
def test_the_tie_helper_names_every_tied_group():
    assert explain.tied_extreme({"a": 3, "b": 5, "c": 5}) == (5, ["b", "c"])
    t = explain.extreme_text({"North": 138, "South": 138, "West": 125, "East": 99})
    assert "North and South" in t and "138 each, tied" in t and "East (99)" in t
    t = explain.extreme_text({"a": 1, "b": 1, "c": 2}, lead_low=True)
    assert t.startswith("**Lowest: a and b**")
    assert explain.extreme_text({"a": 2, "b": 2}) == "**All 2 groups are equal** (2)."
    assert explain.extreme_text({"a": 2.5, "b": 2.5}) == "**All 2 groups are equal** (2.50)."


def test_tied_groups_are_all_named_in_answers(data):
    df, raw = data["orders"]
    vc = raw.region.value_counts()
    tied = sorted(vc[vc == vc.max()].index)
    assert len(tied) >= 2, "the seeded data has a tie at the top"
    for q in ("How many orders per region?", "Which region has the most orders?", "number of orders by region"):
        out = ask(q, df)
        head = out["answer"].split("\n")[0] + out["answer"]
        assert all(f"**{g}**" in head or f"{g}" in head.split("|")[0] for g in tied), q
        assert "tie" in out["answer"].lower(), q


def test_a_program_answer_with_a_tie_names_both(tmp_path):
    d = pd.DataFrame({"team": ["A", "A", "B", "B", "C"], "done": ["Yes", "No", "Yes", "No", "No"],
                      "points": [1.0, 3.0, 2.0, 2.0, 1.0]})
    out = ask("average points by team", d)
    assert "A and B" in out["answer"] or ("**A**" in out["answer"] and "**B**" in out["answer"])


def test_rankings_say_when_groups_tie_for_first(tmp_path):
    from app.analyst import followup
    d = pd.DataFrame({"g": ["x", "y", "z"], "v": [5.0, 5.0, 1.0]})
    r = followup.rank_groups(d, "g", "v", 2, agg="sum")
    assert "**x** and **y** tie for first" in followup.rank_text(r, False)


# ------------------------------------------------------- one record-validity policy --
def test_labour_cost_and_average_hourly_rate_leave_out_the_same_impossible_record(data):
    df, raw = data["shifts"]
    ok = raw[raw.hourly_rate >= 0]
    avg = ask("average hourly rate by department", df)
    groups = {g["group"]: g["value"] for g in next(c for c in avg["calls"] if c["tool"] == "compute_metric")["result"]["groups"]}
    assert groups == pytest.approx(ok.groupby("department").hourly_rate.mean().to_dict())
    assert "negative hourly rate (a rate cannot be negative)" in avg["answer"] and "line 7" in avg["answer"]
    assert avg["verify_status"] == "verified"
    cost = ask("What is the total labor cost by department?", df)
    res = next(c for c in cost["calls"] if c["tool"] == "calculator")["result"]
    rows = {g["group"]: g["rows"] for g in res["groups"]}
    assert rows == ok.groupby("department").size().to_dict(), "both paths use the same records"
    assert "a rate cannot be negative" in cost["details"] or "a rate cannot be negative" in str(res)


def test_rankings_and_column_summaries_follow_the_same_policy(data):
    df, raw = data["shifts"]
    ok = raw[raw.hourly_rate >= 0]
    out = ask("Top 3 departments by average hourly rate", df)
    rows = next(c for c in out["calls"] if c["tool"] == "rank_groups")["result"]["rows"]
    assert {r["group"]: r["value"] for r in rows} == pytest.approx(ok.groupby("department").hourly_rate.mean().nlargest(3).to_dict())
    assert "a rate cannot be negative" in out["answer"]
    out = ask("Give me the mean, median and standard deviation of hourly rate", df)
    r = next(c for c in out["calls"] if c["tool"] == "describe_column")["result"]
    assert r["mean"] == pytest.approx(ok.hourly_rate.mean()) and r["n"] == len(ok)
    assert out["verify_status"] == "verified", "the verifier recomputes under the same policy"


def test_an_amount_that_can_be_negative_is_kept_and_flagged():
    d = pd.DataFrame({"store": ["a", "a", "b", "b"], "revenue": [100.0, -20.0, 50.0, 70.0]})
    kept, notes = scope.valid_records(d, ["revenue"])
    assert len(kept) == 4 and any(n.get("kept") and "negative revenue" in n["reason"] for n in notes)
    out = ask("total revenue by store", d)
    groups = {g["group"]: g["value"] for g in next(c for c in out["calls"] if c["tool"] == "compute_metric")["result"]["groups"]}
    assert groups == {"a": 80.0, "b": 120.0}


def test_every_exclusion_reason_says_why_the_value_is_impossible():
    from app.analyst.metrics import never_negative_reason
    assert never_negative_reason("hourly_rate") == "a rate cannot be negative"
    assert never_negative_reason("unit_price") == "a price cannot be negative"
    assert never_negative_reason("resolution_hours") == "a duration cannot be negative"
    assert never_negative_reason("units") == "a count or quantity cannot be negative"
    d = pd.DataFrame({"resolution_hours": [1.0, -2.0, 3.0]})
    _, why = scope.eligibility(d, ["resolution_hours"], historical=False)
    assert why[0]["reason"] == "negative resolution hours (a duration cannot be negative)"


def test_outliers_are_reported_never_removed(data):
    df, raw = data["tickets"]
    out = ask("average resolution hours", df)
    assert next(c for c in out["calls"] if c.get("status") == "ok")["result"]["value"] == pytest.approx(raw.resolution_hours.mean())


# ------------------------------------------------------------ identifiers in suggestions --
@pytest.mark.parametrize("ds,ids", [("orders", {"order_id", "customer_id"}), ("tickets", {"ticket_id"}),
                                    ("shifts", {"shift_id"}), ("creators", {"zip_code"})])
def test_measure_options_never_offer_an_identifier(data, ds, ids):
    opts = set(semantics.measure_options(data[ds][0], limit=50))
    assert not opts & ids and opts


@pytest.mark.parametrize("ds,q", [("orders", "Top 5 regions"), ("tickets", "What is the mean?"),
                                  ("orders", "Forecast next month"), ("tickets", "average zorbles by channel")])
def test_clarification_options_never_list_identifiers(data, ds, q):
    out = ask(q, data[ds][0])
    for i in ("order_id", "customer_id", "ticket_id"):
        assert f"`{i}`" not in out["answer"], (q, out["answer"][:300])


def test_associations_never_test_an_identifier(data):
    for ds, target in (("tickets", "resolution_hours"), ("orders", "amount")):
        out = ask(f"What drives {target.replace('_', ' ')}?", data[ds][0])
        factors = [f["factor"] for c in out["calls"] if c.get("status") == "ok"
                   for f in (c.get("result") or {}).get("factors", [])]
        assert factors and not {"ticket_id", "order_id", "customer_id"} & set(factors), ds


def test_suggested_questions_never_name_an_identifier(data):
    from app.analyst import overview
    for ds in ("orders", "tickets", "shifts"):
        qs = " ".join(overview.understand(data[ds][0])["suggested_questions"])
        assert "_id" not in qs, (ds, qs)


def test_identifiers_are_still_counted(data):
    df, raw = data["orders"]
    out = ask("How many distinct customer ids are there?", df)
    assert f"{raw.customer_id.nunique()}" in out["answer"] and computed(out)


def test_a_named_forecast_target_is_used_not_asked_about(data):
    out = ask("Forecast next month's amount", data["orders"][0])
    assert "Which value should I forecast?" not in out["answer"]


# --------------------------------------------------------------------- clarifications --
@pytest.mark.parametrize("ds,q,term", [
    ("orders", "How many unicorns are there?", "unicorns"),
    ("orders", "What is the typical blorp?", "blorp"),
    ("tickets", "average zorbles by channel", "zorbles"),
    ("shifts", "total gizmos by department", "gizmos"),
])
def test_a_missing_measure_is_named_and_nothing_is_substituted(data, ds, q, term):
    out = ask(q, data[ds][0])
    assert not computed(out)
    assert f"**{term}**" in out["answer"] and ("nothing called" in out["answer"] or "no column called" in out["answer"])


def test_a_dead_end_lists_what_the_data_can_answer(data):
    df = data["orders"][0]
    out = ask("How many unicorns are there?", df)
    a = out["answer"]
    assert "**Measures**" in a and "`amount`" in a and "**Groups**" in a and "`region`" in a
    assert "`order_id`" not in a.split("**Measures**")[1].split("\n")[0]
    assert "Which value and which groups or period?" not in a, "the generic reply is gone"


def test_a_ranking_with_no_measure_asks_by_which_value(data):
    out = ask("Which region has the highest?", data["orders"][0])
    assert not computed(out) and "by which value" in out["answer"] and "`amount`" in out["answer"]


def test_follow_up_behaviour_from_stage_4_is_unchanged(data):
    df, raw = data["orders"]
    mem = {}
    ask("Total amount by region", df, mem)
    only = ask("What about only completed orders?", df, mem)
    avg = ask("Now show the average instead", df, mem)
    chart = ask("Show that as a chart", df, mem)
    want = raw[raw.status == "Completed"].groupby("region").amount.mean().to_dict()
    ch = chart["charts"][0]
    assert {d[ch["x"]]: d[ch["y"]] for d in ch["data"]} == pytest.approx(want)
    assert only["verify_status"] == avg["verify_status"] == "verified"
