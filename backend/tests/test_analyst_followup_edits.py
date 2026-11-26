"""P1 phase 8: a follow-up edits the previous request's deliverables with ONE explicit change; the rest carries over."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator
from app.analyst.data import load_table


@pytest.fixture()
def shifts(tmp_path):
    rng = np.random.default_rng(9)
    rows = [{"month": m.strftime("%Y-%m-%d"), "nurse": n, "unit": u, "shifts_worked": round(float(rng.normal(b + i, 2)), 1),
             "overtime_hrs": round(float(abs(rng.normal(9, 3))), 1)}
            for n, u, b in (("Ama", "ICU", 20), ("Bo", "ICU", 18), ("Cy", "ER", 22), ("Di", "Ward 3", 17))
            for i, m in enumerate(pd.date_range("2025-01-01", periods=12, freq="MS"))]
    p = tmp_path / "shifts.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return load_table(str(p)), pd.DataFrame(rows)


def turns(df, *qs):
    mem = {}
    for q in qs:
        out = orchestrator.ask(q, df, mem, use_model=False)
    return out


def test_variance_as_well_keeps_the_scope(shifts):
    df, raw = shifts
    out = turns(df, "What's the average overtime hrs for Cy?", "Now calculate the variance as well.")
    assert f"{raw[raw.nurse == 'Cy'].overtime_hrs.var():,.2f}" in out["answer"]


def test_what_about_changes_only_the_subject(shifts):
    df, raw = shifts
    out = turns(df, "Total shifts worked for unit ICU?", "What about ER?")
    a = out["answer"]
    assert f"{raw[raw.unit == 'ER'].shifts_worked.sum():,.2f}" in a and "unit = ER" in a and "570.80" not in a


def test_everyone_together_removes_the_filter(shifts):
    df, raw = shifts
    out = turns(df, "Median overtime hrs for Bo?", "And for everyone together?")
    assert f"{raw.overtime_hrs.median():,.2f}" in out["answer"]


def test_target_change_after_a_compound_answer(shifts):
    df, raw = shifts
    out = turns(df, "Will Ama reach 35 shifts worked a month? Forecast the next month.", "What if the target was 40?")
    assert "Target check: 40" in out["answer"] and "target changed to 40" in out["answer"]
    assert "Forecast of shifts_worked for Ama" in out["answer"]


def test_exclusion_keeps_the_breakdown(shifts):
    df, raw = shifts
    out = turns(df, "Give me the average shifts worked by unit and the total overtime hrs.", "Do the same but exclude ER.")
    a = out["answer"]
    assert "excluded unit = ER" in a and f"{raw[raw.unit != 'ER'].overtime_hrs.sum():,.2f}" in a


def test_an_independent_new_question_is_not_read_as_a_follow_up(shifts):
    df, raw = shifts
    out = turns(df, "Average overtime hrs for Ama?", "What is the maximum shifts worked across the data?")
    assert "Follow-up to" not in out["answer"] and f"{raw.shifts_worked.max():,.2f}".rstrip("0").rstrip(".") in out["answer"]
