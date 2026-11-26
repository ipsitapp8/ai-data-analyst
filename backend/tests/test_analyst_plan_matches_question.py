"""Silt must answer the question asked, not a neighbouring one.

The failure: "Which department has the highest average patient bill, and how does its average length of stay compare
with other departments?" answered part 2 with ONE overall average of length_of_stay -- plus `age`, which no one asked
for ("age" was matched inside "aver-age") -- and the checklist said every part was answered. Safeguards:
  1. a requested metric is never swapped for or joined by another column (whole-word column matching);
  2. before a part is computed, its plan is checked against its clause (metrics, grouping) -- for rule-built and
     model-decomposed plans alike (deliverables.plan_mismatch, called by compound.execute);
  3. a part whose plan does not match is returned for clarification, never shown as answered.
Synthetic fixtures (hospital and retail); expected values come from pandas, never from an answer. No LLM is called.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import compound, deliverables, orchestrator, planner, schema_profile
from app.analyst.semantics import profile


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))


def hospital() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 600
    dept = rng.choice(["Cardiology", "Orthopedics", "Neurology", "Pediatrics", "Oncology"], n)
    base = {"Cardiology": 9000, "Orthopedics": 7000, "Neurology": 8000, "Pediatrics": 3000, "Oncology": 12000}
    los = np.clip(np.round(rng.gamma(2, 2, n) + (dept == "Oncology") * 2), 1, None).astype(int)
    df = pd.DataFrame({"patient_id": [f"P{i:04d}" for i in range(n)], "department": dept, "age": rng.integers(1, 90, n),
                       "length_of_stay": los, "room_charge": np.round(los * rng.normal(900, 80, n), 2),
                       "medication_cost": np.round(rng.gamma(2, 600, n), 2)})
    df["patient_bill"] = np.round([base[d] for d in dept] + df.room_charge + df.medication_cost + rng.normal(0, 500, n), 2)
    return schema_profile.apply(df)


def retail() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame({"order_id": [f"O{i}" for i in range(n)], "region": rng.choice(["North", "South", "East", "West"], n),
                       "units": rng.integers(1, 40, n), "unit_price": rng.normal(25, 5, n).round(2),
                       "discount_rate": rng.uniform(0, .3, n).round(3), "storage_days": rng.integers(1, 30, n)})
    df["revenue"] = (df.units * df.unit_price * (1 - df.discount_rate)).round(2)
    return schema_profile.apply(df)


def _ask(q, df):
    return orchestrator.ask(q, df, {}, use_model=False, filename="data.csv")


def test_a_column_is_named_only_as_whole_words():
    df = hospital()
    prof = profile(df)
    assert deliverables.measures_in("how does its average length of stay compare", df, prof) == ["length_of_stay"]
    assert "age" in deliverables.measures_in("the average age of patients", df, prof)


def test_the_healthcare_question_is_answered_per_department_with_only_the_requested_metrics():
    df = hospital()
    q = ("Which department has the highest average patient bill, and how does its average length of stay compare "
         "with other departments?")
    bill = df.groupby("department")["patient_bill"].mean()
    stay = df.groupby("department")["length_of_stay"].mean()
    out = _ask(q, df)
    a = out["answer"]
    top = bill.idxmax()
    assert f"**Highest: {top}** ({bill.max():,.2f})" in a
    for dep, v in stay.items():                              # part 2 is per department, every department shown
        assert f"| {dep} | {v:,.2f} |" in a
    assert "age" not in [w.strip("`*|") for w in a.split()]  # the unrequested column is not used anywhere
    assert "length_of_stay / age" not in a and "Average of length_of_stay / age" not in a


def test_an_unrelated_compound_question_keeps_its_metrics_and_grouping():
    df = retail()
    q = "Which region has the highest total revenue, and how does its average discount rate compare with all regions?"
    rev = df.groupby("region")["revenue"].sum()
    disc = df.groupby("region")["discount_rate"].mean()
    a = _ask(q, df)["answer"]
    assert f"**Highest: {rev.idxmax()}** ({rev.max():,.2f})" in a
    assert all(f"| {r} | {v:,.2f} |" in a for r, v in disc.items())
    for other in ("units", "unit_price", "storage_days"):
        assert other not in a


def _plan(q, df):
    return deliverables.finalize(deliverables.extract(q, df))


def test_a_plan_that_adds_an_unrequested_metric_is_not_answered():
    """A plan with an extra measure (as a model decomposer could produce) is sent back, nothing computed for it."""
    df = hospital()
    ds = _plan("What is the average patient bill?", df)
    d = next(x for x in ds if x.kind == "stat")
    d.measures = ["patient_bill", "room_charge"]
    ctx = compound.execute(ds, df)
    assert d.status == "needs_input" and "room_charge" in d.detail and d.id not in ctx["results"]
    text = compound.compose(ds, ctx)[0]
    assert "Needs your input" in text and "✓" not in text.split("What you asked")[-1].split("\n")[1]


def test_a_plan_that_drops_the_requested_grouping_is_not_answered():
    df = hospital()
    ds = _plan("How does the average length of stay compare with other departments?", df)
    d = next(x for x in ds if x.kind in ("stat", "breakdown"))
    d.kind, d.group_by = "stat", None                       # the grouping the clause asks for, left out
    ctx = compound.execute(ds, df)
    assert d.status == "needs_input" and "department" in d.detail and d.id not in ctx["results"]


def test_a_plan_that_matches_its_clause_is_computed():
    df = hospital()
    ds = _plan("What is the average patient bill?", df)
    ctx = compound.execute(ds, df)
    d = next(x for x in ds if x.kind == "stat")
    assert d.status == "done"
    assert deliverables.plan_mismatch(d, df, profile(df)) is None


@pytest.mark.parametrize("q", ["Which department charges the most?",
                               "Which department charges the most on average, and how long do its patients stay "
                               "compared to other departments?"])
def test_money_said_by_a_verb_with_several_money_columns_is_asked_not_guessed(q):
    df = hospital()
    out = _ask(q, df)
    assert out["plan"]["type"] == "clarify"
    assert set(out["plan"]["options"]) == {"room_charge", "medication_cost", "patient_bill"}
    assert not out["calls"]                                   # nothing computed on a guessed column


def test_a_money_column_named_in_full_is_used_without_asking():
    df = hospital()
    a = _ask("Which department has the highest average patient bill?", df)["answer"]
    assert df.groupby("department")["patient_bill"].mean().idxmax() in a and "could mean" not in a
