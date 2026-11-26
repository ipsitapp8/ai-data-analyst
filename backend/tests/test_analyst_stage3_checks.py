"""Checks required before Stage 4: multi-part answer labelling, one date-window convention on every path, and the
recorded limit of group-comparison verification. The convention is documented in intent.relative_period."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import calc, intent, nl_planner, orchestrator, scope, verify
from app.analyst.data import load_table

from test_analyst_stage1_guards import _orders, pinned_clock  # noqa: F401

TODAY = pd.Timestamp("2026-11-17")


@pytest.fixture
def frozen_today(monkeypatch):
    monkeypatch.setattr(scope, "today", lambda: TODAY)


def _visits(end: pd.Timestamp, days: int = 60) -> pd.DataFrame:
    """One record every 6 hours for `days` days up to and including `end` (so the last day has records at 18:00)."""
    ts = pd.date_range(end - pd.Timedelta(days=days - 1), end + pd.Timedelta(hours=18), freq="6h")
    rng = np.random.default_rng(4)
    return pd.DataFrame({"visited_at": ts, "clinic": rng.choice(["A", "B"], len(ts)),
                         "minutes": rng.integers(5, 60, len(ts)).astype(float)})


def _window(df, start, end):
    d = df.visited_at.dt.normalize()
    return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


@pytest.mark.parametrize("phrase,start,end", [
    ("the last 7 days", "2026-11-11", "2026-11-17"),
    ("the last 30 days", "2026-10-19", "2026-11-17"),
    ("the past 2 weeks", "2026-11-04", "2026-11-17"),
    ("this month", "2026-11-01", "2026-11-17"),
    ("last month", "2026-10-01", "2026-10-31"),
])
def test_one_date_window_convention_on_every_path(frozen_today, phrase, start, end):
    """N days = N calendar days ending on and including today; both ends included; compared by calendar day, so the
    records at 18:00 on the end day are inside. Program select, shared scope, calculator rule and verifier agree."""
    df = _visits(TODAY)
    want = _window(df, start, end)
    q = f"Total minutes for {phrase}"
    p = intent.relative_period(q, df)
    assert (p["start"], p["end"]) == (start, end)
    # 1. the program engine
    plan = nl_planner.rule_plan(q, df, {})
    got = nl_planner.execute(plan, df, {}, q)["results"]["result"]["value"]
    assert got == pytest.approx(want.minutes.sum())
    # 2. the shared scope (every tool path filters by it)
    sc = scope.establish(q, df, {})
    assert int(scope.mask(df, sc).sum()) == len(want)
    # 3. the calculator's exclusion rule built from the scope
    drop = pd.Series(False, index=df.index)
    for e in scope.as_exclusions(sc):
        drop |= calc.evaluate(e["where"], df, df).fillna(False).astype(bool)
    assert int((~drop).sum()) == len(want)
    # 4. the verifier's independent recomputation
    v, _ = verify.recompute(intent.read(q, df), df)
    assert v == pytest.approx(want.minutes.sum())
    # ... and the end-to-end answer, which states the window
    out = orchestrator.ask(q, df, {}, use_model=False)
    assert out["verify_status"] == "verified" and start in out["answer"] and end in out["answer"]


def test_today_is_included_and_late_records_on_the_last_day_count(frozen_today):
    df = _visits(TODAY)
    q = "How many visits in the last 7 days?"
    out = orchestrator.ask(q, df, {}, use_model=False)
    late = df[df.visited_at == TODAY + pd.Timedelta(hours=18)]
    assert len(late) == 1
    assert "today is 2026-11-17" in out["answer"]
    assert len(_window(df, "2026-11-11", "2026-11-17")) == 28                  # 7 days x 4 records, today's 18:00 included
    got = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"][""]["result"]["value"]
    assert got == 28 and out["verify_status"] == "verified"


def test_a_window_is_anchored_on_the_last_record_when_the_data_ends_long_before_today(frozen_today):
    df = _visits(pd.Timestamp("2025-03-31"))
    p = intent.relative_period("Total minutes for the last 7 days", df)
    assert (p["start"], p["end"]) == ("2025-03-25", "2025-03-31") and "data ends on 2025-03-31" in p["label"]
    out = orchestrator.ask("Total minutes for the last 7 days", df, {}, use_model=False)
    assert "data ends on 2025-03-31" in out["answer"]


# ------------------------------------------------------------ multi-part labelling --
@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    raw = _orders(np.random.default_rng(7))
    path = tmp_path_factory.mktemp("checks") / "orders.csv"
    raw.to_csv(path, index=False)
    return load_table(str(path)), raw


def test_each_part_of_a_partial_answer_says_what_happened_to_it(orders):
    df, _ = orders
    out = orchestrator.ask("How many orders per region and what is the typical order size?", df, {}, use_model=False)
    a = out["answer"]
    assert out["verify_status"] == "partially verified"
    assert "**1. How many orders per region?** _(✓ answered" in a
    assert "**2. What is the typical order size?** _(not computed — needs your answer)_" in a
    assert "part 1: ✓ recomputed independently with pandas; part 2: not computed (needs your answer)" in a
    assert "This question is not fully answered yet" in a
    assert _VERIFIED_WHOLE not in a, "the whole response must never read as fully verified"


_VERIFIED_WHOLE = "_Verification: ✓ the headline figure was recomputed independently with pandas from your question and matches._"


def test_a_refused_part_is_named_and_the_whole_is_not_verified(orders):
    df, _ = orders
    out = orchestrator.ask("What is the average customer id and how many orders are there?", df, {}, use_model=False)
    assert out["verify_status"] == "partially verified" and "part 1: not computed (refused)" in out["answer"]
    assert _VERIFIED_WHOLE not in out["answer"]


def test_all_parts_computed_is_verified_and_each_part_is_still_named(orders):
    df, raw = orders
    out = orchestrator.ask("How many orders and what is the median amount?", df, {}, use_model=False)
    assert out["verify_status"] == "verified"
    assert "part 1: ✓" in out["answer"] and "part 2: ✓" in out["answer"]


# ------------------------------ group comparisons: independently verified since Stage 6 --
def test_group_comparisons_are_verified_by_an_independent_recomputation(orders):
    """Stage 6 added the independent check that Stage 3 recorded as missing: every figure of a group comparison is
    recomputed from first principles; an answer is verified only when all of them match."""
    df, _ = orders
    for q in ("Does the region affect amount?", "Is there a significant difference in amount between regions?"):
        out = orchestrator.ask(q, df, {}, use_model=False)
        assert any(c["tool"] == "compare_groups" for c in out["calls"]), q
        assert out["verify_status"] == "verified", q
        assert any("Welch F" in c["check"] for c in out["verification"]), q


def test_a_group_comparison_with_a_wrong_figure_is_not_verified(orders):
    from app.analyst import tools
    df, _ = orders
    call = tools.validate_call({"tool": "compare_groups", "args": {"outcome": "amount", "group": "region"}}, df)
    r = tools.run(call, df, {})
    r["summary"][0]["mean"] += 1.0                         # one figure off
    v = verify.check_answer("Is there a difference in amount between regions?", df,
                            {"plan_source": "rules", "calls": [{**call, "status": "ok", "result": r}]}, None, {})
    assert v["status"] == "failed"
