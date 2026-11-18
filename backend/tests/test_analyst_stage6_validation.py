"""Stage 6 of the Copilot reliability repair: final validation.

  * group comparisons are recomputed independently (Welch t / F, Kruskal-Wallis, chi-square, Fisher, effect sizes);
  * a forecast or trend never treats a partial calendar period as a whole one;
  * the latest result survives a restart (the conversation's own file), and is ignored if the dataset changed;
  * distinct counts read naturally;
  * a model that plans badly -- drops a filter, averages an identifier, writes numbers of its own -- cannot get a
    wrong answer past the safeguards (a scripted model stands in for the served one, so these run offline).

Seeded synthetic data from the earlier stages plus purpose-built frames; nothing here is known to Silt's code.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analyst import followup, intent, nl_planner, orchestrator, persist, planner, scope, tools, verify
from app.analyst.data import load_table

from test_analyst_stage1_guards import TODAY, _creators, _orders, _shifts, _tickets, pinned_clock  # noqa: F401


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("stage6")
    out = {}
    for name, make in (("tickets", _tickets), ("shifts", _shifts), ("orders", _orders), ("creators", _creators)):
        raw = make(rng)
        path = root / f"{name}.csv"
        raw.to_csv(path, index=False)
        out[name] = (load_table(str(path)), raw)
    return out


@pytest.fixture
def frozen_today(monkeypatch):
    monkeypatch.setattr(scope, "today", lambda: TODAY)


def ask(q, df, memory=None, model=False):
    return orchestrator.ask(q, df, {} if memory is None else memory, use_model=model)


def comparison(out):
    return next(c["result"] for c in out["calls"] if c["tool"] == "compare_groups")


# ------------------------------------------------------- group comparisons, independently --
@pytest.mark.parametrize("ds,q,y,g", [
    ("tickets", "Is there a significant difference in resolution hours between escalated and non-escalated tickets?",
     "resolution_hours", "escalated"),
    ("orders", "Does the region affect amount?", "amount", "region"),
    ("shifts", "Is there a difference in hourly rate between departments?", "hourly_rate", "department"),
    ("creators", "Compare engagement rate across niches", "engagement_rate", "niche"),
])
def test_numeric_group_comparisons_match_pandas_and_scipy(data, ds, q, y, g):
    df, raw = data[ds]
    raw = raw[~(pd.to_numeric(raw[y], errors="coerce") < 0)] if y == "hourly_rate" else raw   # the validity policy
    out = ask(q, df)
    r = comparison(out)
    assert out["verify_status"] == "verified", [c for c in out["verification"] if not c["ok"]]
    want = raw.groupby(raw[g].astype(str))[y].agg(["mean", "median", "count"])
    for row in r["summary"]:
        assert row["mean"] == pytest.approx(want.loc[row["group"], "mean"])
        assert row["n"] == want.loc[row["group"], "count"]
    groups = [x[y].dropna().to_numpy() for _, x in raw.groupby(raw[g].astype(str))]
    if len(groups) == 2:
        assert r["test"]["p_value"] == pytest.approx(stats.ttest_ind(*groups, equal_var=False).pvalue, rel=1e-6)
    else:
        assert r["test"]["robustness"]["p_value"] == pytest.approx(stats.kruskal(*groups).pvalue, rel=1e-6)


def test_a_categorical_comparison_is_verified_by_its_own_chi_square(data):
    df, raw = data["orders"]
    out = ask("Is status related to region?", df)
    r = comparison(out)
    assert out["verify_status"] == "verified"
    table = pd.crosstab(raw.region, raw.status)
    assert r["test"]["statistic"] == pytest.approx(stats.chi2_contingency(table, correction=False).statistic)


def test_fisher_exact_is_recomputed_for_a_small_two_by_two_table():
    d = pd.DataFrame({"group": ["a"] * 6 + ["b"] * 6, "won": ["Yes"] * 5 + ["No"] + ["Yes"] + ["No"] * 5})
    call = tools.validate_call({"tool": "compare_groups", "args": {"outcome": "won", "group": "group"}}, d)
    r = tools.run(call, d, {})
    assert r["test"]["name"] == "Fisher's exact test"
    v = verify.check_answer("Is winning different between groups?", d,
                            {"plan_source": "rules", "calls": [{**call, "status": "ok", "result": r}]}, None, {})
    assert v["status"] == "verified"


@pytest.mark.parametrize("field", ["mean", "p_value", "n"])
def test_a_tampered_comparison_is_never_verified(data, field):
    df, _ = data["orders"]
    call = tools.validate_call({"tool": "compare_groups", "args": {"outcome": "amount", "group": "region"}}, df)
    r = tools.run(call, df, {})
    if field == "p_value":
        r["test"]["p_value"] = min(1.0, r["test"]["p_value"] * 3 + 0.01)
    elif field == "n":
        r["summary"][1]["n"] += 1
    else:
        r["summary"][2]["mean"] *= 1.01
    v = verify.check_answer("Is there a difference in amount between regions?", df,
                            {"plan_source": "rules", "calls": [{**call, "status": "ok", "result": r}]}, None, {})
    assert v["status"] == "failed"


# ------------------------------------------------------------- forecasts and partial periods --
def _daily(end: str, months: int = 14, per_day: float = 100.0) -> pd.DataFrame:
    days = pd.date_range(pd.Timestamp(end) - pd.DateOffset(months=months) + pd.Timedelta(days=1), end, freq="D")
    rng = np.random.default_rng(3)
    return pd.DataFrame({"day": days, "store": rng.choice(["a", "b"], len(days)),
                         "sales": (per_day + rng.normal(0, 5, len(days))).round(2)})


def test_a_forecast_leaves_the_partial_current_month_out_of_its_history(frozen_today):
    df = _daily("2026-11-17")
    out = ask("Forecast next month's sales", df)
    f = out["calls"][0]["result"]
    assert "2026-11" in f["partial"] and f["partial"]["2026-11"]["covered"] == 17
    assert all(p["period"] != "2026-11" for p in f["points"]), "a 17-day total is not history for a 30-day month"
    assert "**Incomplete period:** 2026-11 is covered only from 2026-11-01 to 2026-11-17 (17 of 30 days)" in out["answer"]
    assert f["next_period"] == "2026-12"
    # the whole months used are whole: each around 30 days x 100
    assert min(p["value"] for p in f["points"]) > 2700


def test_a_forecast_on_complete_months_drops_nothing(frozen_today):
    df = _daily("2026-10-31")
    f = ask("Forecast next month's sales", df)["calls"][0]["result"]
    last_full_start = pd.Timestamp("2026-10-31") - pd.DateOffset(months=14) + pd.Timedelta(days=1)
    assert not {p for p in f["partial"]} - {str(last_full_start.to_period("M"))}     # only a start edge, if any
    assert f["points"][-1]["period"] == "2026-10"


def test_data_recorded_once_a_month_is_never_called_partial(frozen_today):
    df = pd.DataFrame({"month": pd.date_range("2025-01-01", periods=12, freq="MS"),
                       "sales": np.arange(12, dtype=float) * 10 + 100})
    assert followup.partial_periods(df["month"], "M") == {}
    # weekly records (Mondays) ending on May's last Monday: every month is whole
    weeks = pd.Series(pd.date_range("2025-01-06", "2025-05-26", freq="W-MON"))
    assert followup.partial_periods(weeks, "M") == {}
    # ... ending on 19 May, before May's last weekly record (26 May): May is partial, and said to be
    early = pd.Series(pd.date_range("2025-01-06", "2025-05-19", freq="W-MON"))
    assert set(followup.partial_periods(early, "M")) == {"2025-05"}


def test_a_trend_shows_the_partial_month_but_does_not_use_it(frozen_today):
    df = _daily("2026-11-17")
    out = ask("Show the trend of sales over time", df)
    r = out["calls"][0]["result"]
    assert r["last"]["period"] == "2026-10" and any(p.get("partial") for p in r["points"])
    assert "shown but not used" in out["answer"]


# ------------------------------------------------------------------- memory across restarts --
def _simulate_restart(key):
    orchestrator._MEMORY.pop(key, None)          # noqa: SLF001 - the in-process memory is gone after a restart


def test_the_latest_result_survives_a_restart(data, tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYST_MEMORY_DIR", str(tmp_path))
    df, raw = data["orders"]
    key = orchestrator.memory_key(1, 1, 7, None, "conv-restart")
    mem = orchestrator.memory_for(key, persist=True)
    ask("Total amount by region", df, mem)
    orchestrator.save_memory(key, mem)
    _simulate_restart(key)
    mem2 = orchestrator.memory_for(key, persist=True)
    assert mem2["last_result"]["question"] == "Total amount by region" and mem2["_turn"] == 1
    out = ask("What about only completed orders?", df, mem2)
    want = raw[raw.status == "Completed"].groupby("region").amount.sum().to_dict()
    groups = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"][""]["result"]["value"]
    assert groups == pytest.approx(want) and out["verify_status"] == "verified"


def test_a_restored_result_is_ignored_when_the_dataset_changed(data, tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYST_MEMORY_DIR", str(tmp_path))
    df, _ = data["orders"]
    key = orchestrator.memory_key(1, 1, 8, None, "conv-changed")
    mem = orchestrator.memory_for(key, persist=True)
    ask("Total amount by region", df, mem)
    orchestrator.save_memory(key, mem)
    _simulate_restart(key)
    mem2 = orchestrator.memory_for(key, persist=True)
    smaller = df.iloc[:-10]
    out = ask("What about only completed orders?", smaller, mem2)
    assert "Follow-up to" not in out["answer"], "a result computed on other records is never refined"


@pytest.mark.parametrize("payload", [
    "{not json", json.dumps({"version": 1, "key": "OTHER", "saved_at": 0}),
    json.dumps({"version": 1, "key": "K", "saved_at": 9e12, "last_result": {"turn": "x"}}),
])
def test_a_damaged_memory_file_is_recovered_from_gracefully(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("ANALYST_MEMORY_DIR", str(tmp_path))
    p = persist.path_for("K")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload.replace('"K"', '"K"'))
    restored = persist.load("K")
    assert "last_result" not in restored


# ------------------------------------------------------------------------------- wording --
def test_distinct_counts_read_naturally(data):
    df, raw = data["tickets"]
    out = ask("count of distinct agents", df)
    assert f"number of distinct agent values: {raw.agent.nunique()}" in out["answer"]
    assert "distinct agent:" not in out["answer"]


# --------------------------------------------- a badly planning model cannot get past the checks --
def _script(monkeypatch, text):
    calls = []

    def fake(messages, *a, **k):
        calls.append(messages[0]["content"][:30])
        return text
    monkeypatch.setattr(planner, "model_chat", fake)
    monkeypatch.setattr(nl_planner, "model_chat", fake)
    monkeypatch.setattr(planner, "model_available", lambda: True)
    return calls


def test_a_model_plan_that_drops_the_filter_is_not_shown_as_verified(data, monkeypatch):
    df, raw = data["tickets"]
    _script(monkeypatch, json.dumps({"type": "tools", "intent": "metric", "calls": [
        {"tool": "compute_metric", "args": {"aggregation": "sum", "column": "resolution_hours"}}]}))
    out = ask("Total resolution hours for tickets with csat below 3", df, model=True)
    all_rows = raw.resolution_hours.sum()
    shown_wrong = f"{all_rows:,.2f}" in out["answer"]
    assert not (shown_wrong and out["verify_status"] == "verified")
    if out["verify_status"] == "verified":
        assert f"{raw[raw.csat < 3].resolution_hours.sum():,.2f}" in out["answer"]


def test_a_model_plan_that_averages_an_identifier_is_refused(data, monkeypatch):
    df, _ = data["orders"]
    _script(monkeypatch, json.dumps({"type": "tools", "intent": "metric", "calls": [
        {"tool": "compute_metric", "args": {"aggregation": "avg", "column": "customer_id"}}]}))
    out = ask("What is the typical customer number?", df, model=True)
    assert not any(c.get("status") == "ok" and (c.get("args") or {}).get("column") == "customer_id"
                   and (c.get("args") or {}).get("aggregation") == "avg" for c in out["calls"])


def test_a_model_plan_for_the_wrong_measure_is_not_verified(data, monkeypatch):
    df, raw = data["tickets"]
    _script(monkeypatch, json.dumps({"type": "tools", "intent": "metric", "calls": [
        {"tool": "compute_metric", "args": {"aggregation": "avg", "column": "csat", "group_by": "channel"}}]}))
    out = ask("average resolution hours by channel", df, model=True)
    if out["verify_status"] == "verified":
        groups = next(c for c in out["calls"] if c.get("status") == "ok")["result"]
        vals = groups.get("groups") or []
        if vals:
            assert {g["group"]: g["value"] for g in vals} == pytest.approx(
                raw.groupby("channel").resolution_hours.mean().to_dict())
    assert "csat" not in out["answer"].split("\n")[0] or out["verify_status"] != "verified"


def test_invented_numbers_in_model_prose_are_never_shown(data, monkeypatch):
    df, _ = data["orders"]
    _script(monkeypatch, "Revenue grew by 437.5% and the North region made 99,999 last year.")
    out = ask("Total amount by region", df, model=True)
    assert "437.5" not in out["answer"] and "99,999" not in out["answer"]


# --------------------------------------------- failures found on unseen domains in Stage 6 --
@pytest.fixture(scope="module")
def clinic(tmp_path_factory):
    rng = np.random.default_rng(23)
    n = 300
    d = pd.DataFrame({"case_ref": [f"C{70000 + i}" for i in range(n)],
                      "seen_on": [TODAY - pd.Timedelta(days=int(x)) for x in rng.integers(0, 120, n)],
                      "unit": rng.choice(["Renal", "Burns", "Spine"], n), "queue_minutes": rng.integers(3, 200, n).astype(float),
                      "stay_days": rng.integers(1, 12, n).astype(float), "returned_within_30": rng.choice(["Yes", "No"], n),
                      "cohort": rng.choice(["A1", "B2"], n), "cleared": rng.choice(["Yes", "No"], n, p=[.6, .4])})
    d.loc[[4, 8], "queue_minutes"] = -5.0
    path = tmp_path_factory.mktemp("clinic") / "clinic.csv"
    d.to_csv(path, index=False)
    return load_table(str(path)), d


def _value(out):
    c = next(c for c in out["calls"] if c.get("status") == "ok")
    r = c["result"]
    if c["tool"] == "analysis_program":
        return r["results"][""]["result"]["value"]
    return {str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r["value"]


def test_a_negative_duration_in_minutes_is_impossible_on_every_path(clinic, frozen_today):
    df, raw = clinic
    ok = raw[raw.queue_minutes >= 0]
    out = ask("What is the average queue by unit?", df)
    assert _value(out) == pytest.approx(ok.groupby("unit").queue_minutes.mean().to_dict())
    assert "a duration cannot be negative" in out["answer"] and out["verify_status"] == "verified"


def test_a_name_after_the_is_checked_against_the_data(clinic):
    out = ask("average queue minutes for the Cardiology unit", clinic[0])
    assert not any(c.get("status") == "ok" for c in out["calls"]) and "Cardiology" in out["answer"]


def test_a_four_letter_word_finds_its_yes_no_column(clinic):
    df, raw = clinic
    r = intent.read("What is the clear rate for B2?", df)
    assert r.measures == ["cleared"] and ("cohort", "B2") in r.entities
    out = ask("What is the clear rate for B2?", df)
    assert _value(out) == pytest.approx((raw[raw.cohort == "B2"].cleared == "Yes").mean())


def test_a_long_shared_stem_finds_its_yes_no_column(clinic):
    df, raw = clinic
    out = ask("What is the return rate by unit?", df)
    assert _value(out) == pytest.approx((raw.returned_within_30 == "Yes").groupby(raw.unit).mean().to_dict())


def test_the_words_of_a_date_window_never_name_the_measure(clinic, frozen_today):
    df, raw = clinic
    r = intent.read("How many cases in the last 30 days?", df)
    assert r.quantity == "count" and not r.measures and r.period["start"] == "2026-10-19"
    out = ask("How many cases in the last 30 days?", df)
    d = raw.seen_on.dt.normalize()
    assert _value(out) == ((d >= TODAY - pd.Timedelta(days=29)) & (d <= TODAY)).sum()


def test_follow_ups_are_resolved_before_the_model_is_consulted(data, monkeypatch):
    """Found with the served v1 model (Stage 6): its planner answered "What about only completed orders?" as "how
    many orders are completed" and "the average instead" as the average of another column. A follow-up to the latest
    result is resolved from that result's reading first; a model that would plan something else is never asked."""
    df, raw = data["orders"]
    mem = {}
    ask("Total amount by region", df, mem)
    calls = _script(monkeypatch, json.dumps({"type": "tools", "intent": "metric", "calls": [
        {"tool": "compute_metric", "args": {"aggregation": "avg", "column": "units"}}]}))
    only = ask("What about only completed orders?", df, mem, model=True)
    avg = ask("Now show the average instead", df, mem, model=True)
    why = ask("Why?", df, mem, model=True)
    done = raw[raw.status == "Completed"]
    assert _value(only) == pytest.approx(done.groupby("region").amount.sum().to_dict())
    assert _value(avg) == pytest.approx(done.groupby("region").amount.mean().to_dict())
    assert only["verify_status"] == avg["verify_status"] == "verified"
    assert "Associations, not causes" in why["answer"] or "not proven causes" in why["answer"]
    assert not calls, "no model call was needed for a follow-up"
