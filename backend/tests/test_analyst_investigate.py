"""Autonomous investigation and semantic understanding on data Silt was not built for.

Planted signals make the right answer known: a failure flag driven by tool wear, a site with a higher failure
rate, a pure-noise column that must NOT be reported, planted outliers, duplicates and missing values. Numbers are
checked against scipy/pandas computed here. Metamorphic variants (renamed columns, shuffled rows, rescaled units)
must give the same findings. The model is offline throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analyst import investigate, orchestrator, planner, semantics


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


def machines(seed=11, n=600):
    rng = np.random.default_rng(seed)
    site = rng.choice(["Leeds", "Turin", "Osaka"], n, p=[0.4, 0.4, 0.2])
    wear = rng.integers(0, 250, n).astype(float)
    noise = rng.normal(50, 5, n)
    p = 0.02 + 0.5 * (wear > 190) + 0.15 * (site == "Osaka")
    fail = np.where(rng.random(n) < p, "Yes", "No")
    df = pd.DataFrame({"unit_code": [f"U{i:05d}" for i in range(n)], "site": site, "tool_wear_min": wear,
                       "spindle_noise_db": noise, "output_units": rng.integers(50, 120, n), "failure": fail,
                       "logged_at": pd.date_range("2025-01-01", periods=n, freq="12h")})
    df.loc[[5, 9, 40], "output_units"] = [900, 950, 1000]               # planted outliers
    df.loc[[3, 4], "spindle_noise_db"] = np.nan                          # planted missing values
    return pd.concat([df, df.iloc[[7]]], ignore_index=True)              # planted duplicate


def ask(q, df, mem=None):
    return orchestrator.ask(q, df, mem if mem is not None else {}, use_model=False, filename="machines.csv")


def findings(df):
    return investigate.investigate(df)["findings"]


def test_planted_signals_are_found_and_noise_is_not():
    df = machines()
    F = findings(df)
    titles = " | ".join(f["title"] for f in F)
    assert "tool wear min is associated with failure" in titles
    assert "failure rate differs by site" in titles
    assert "spindle noise db is associated" not in titles, "a pure-noise column is not a finding"
    q = next(f for f in F if f["kind"] == "quality")
    assert "spindle_noise_db" in q["statement"] and "duplicate" in q["statement"]
    assert any(f["kind"] == "unusual" and "output units" in f["title"] for f in F)
    for f in F:
        assert all(f[k] not in (None, "") for k in ("calculation", "scope", "interpretation", "limitations"))
        assert isinstance(f["n"], int) and f["n"] > 0


def test_finding_numbers_match_independent_calculations():
    df = machines()
    r = investigate.investigate(df)
    site = next(f for f in r["findings"] if f["title"] == "failure rate differs by site")
    rates = (df.failure == "Yes").groupby(df.site).mean()
    assert f"{rates.max():.0%}" in site["statement"] and f"{rates.min():.0%}" in site["statement"]
    assoc = r["results"]["assoc"]["value"]["factors"]
    wear = next(x for x in assoc if x["factor"] == "tool_wear_min")
    used = df.dropna(subset=["spindle_noise_db"])         # the investigation's scope: records without missing measures
    xb = pd.cut(used.tool_wear_min, bins=5).astype(str)
    chi2 = stats.chi2_contingency(pd.crosstab(xb, used.failure))[0]
    assert wear["value"] == pytest.approx(np.sqrt(chi2 / len(used)))


def test_analyses_are_chosen_from_the_schema():
    plain = pd.DataFrame({"group": ["a", "b"] * 20, "x": np.arange(40.0)})
    steps = [s["op"] for s in investigate.plan(plain)[0]]
    assert "project" not in steps and not any(s.get("per") for s in investigate.plan(plain)[0]), "no date: no trend"
    r = investigate.investigate(plain)
    assert any("No date or year column" in x for x in r["limitations"])
    assert any("No outcome column" in x for x in r["limitations"])


@pytest.mark.parametrize("variant", ["renamed", "shuffled", "rescaled"])
def test_metamorphic_variants_give_the_same_findings(variant):
    base = machines()
    if variant == "renamed":
        df = base.rename(columns={"site": "plant_location", "tool_wear_min": "cutter_wear_minutes", "failure": "breakdown"})
    elif variant == "shuffled":
        df = base.sample(frac=1, random_state=3).reset_index(drop=True)
    else:
        df = base.assign(tool_wear_min=base.tool_wear_min * 60)            # minutes -> seconds
    a = {(f["kind"], round(f["score"], 6)) for f in findings(base)}
    b = {(f["kind"], round(f["score"], 6)) for f in findings(df)}
    if variant == "shuffled":
        assert {k for k, _ in a} == {k for k, _ in b}
    else:
        assert a == b


def test_public_dataset_investigation():
    plotly = pytest.importorskip("plotly")
    import os
    tips = pd.read_csv(os.path.join(os.path.dirname(plotly.__file__), "package_data", "datasets", "tips.csv.gz"))
    r = investigate.investigate(tips)
    assert r["findings"] and r["meta"]["main"] == "total_bill"
    assoc = next(x for x in r["results"]["assoc"]["value"]["factors"] if x["factor"] == "tip")
    assert assoc["value"] == pytest.approx(stats.spearmanr(tips.tip, tips.total_bill)[0])


def test_semantic_roles_from_values_not_only_names():
    df = pd.DataFrame({"ref": [f"R{i}" for i in range(40)], "who": [f"Ana Lima{i % 5}" for i in range(40)],
                       "q": [0.1, 0.5] * 20, "yr": [2020 + i % 4 for i in range(40)], "n_items": list(range(40)),
                       "ok": ["Y", "N"] * 20, "when": ["09:30", "17:00"] * 20, "kind": ["a", "b", "c", "d"] * 10})
    roles = {c["column"]: c["role"] for c in semantics.profile(df)["columns"]}
    assert roles == {"ref": "identifier", "who": "category", "q": "rate", "yr": "year", "n_items": "count",
                     "ok": "flag", "when": "time_of_day", "kind": "category"}


def test_a_concept_word_is_offered_not_applied_then_runs_on_yes():
    df = machines().rename(columns={"failure": "breakdown"})
    mem = {}
    out = ask("What factors are associated with machine faults?", df, mem)
    assert out["plan"]["type"] == "clarify" and "breakdown" in out["answer"] and not out["calls"]
    done = ask("yes", df, mem)
    assert any(c["tool"] == "associations" for c in done["calls"])


def test_investigation_follow_ups():
    df, mem = machines(), {}
    out = ask("Analyse this entire dataset and tell me what I should investigate next.", df, mem)
    assert out["plan"]["objective"] == "investigation" and "Investigate next" in out["answer"]
    detail = ask("Tell me more about finding 1", df, mem)
    assert "**Calculation:**" in detail["answer"] and "**Limitations:**" in detail["answer"]
    dug = ask("Dig into finding 1", df, mem)
    assert dug["calls"] and dug["calls"][0]["status"] == "ok"
    assert "There are" in ask("Tell me more about finding 99", df, mem)["answer"]


def test_ml_on_dated_records_is_validated_forward_in_time():
    from app.analyst import ml
    df = machines().drop_duplicates()
    r = ml.train(df, "failure", "classification")
    assert r["split"]["time_column"] == "logged_at" and r["metrics"]["cross_validation"]["time_ordered"]
    order = df.sort_values("logged_at", kind="stable")
    assert r["test_rows"] == len(df) - int(round(len(df) * 0.75))
    assert r["train_rows"] + r["test_rows"] == len(df.dropna(subset=["failure"]))
    undated = ml.train(df.drop(columns=["logged_at"]), "failure", "classification")
    assert undated["split"]["time_column"] is None and undated["split"]["stratified"]


def test_codes_years_and_numeric_outcomes_are_understood():
    rng = np.random.default_rng(5)
    n = 300
    yr = rng.integers(2001, 2011, n)
    sex = rng.choice([1, 2], n)
    bmi = rng.normal(26, 4, n)
    df = pd.DataFrame({"year": yr, "sex": sex, "bmi": bmi, "s4": rng.normal(4, 1, n),
                       "progression": 50 + 8 * bmi + rng.normal(0, 30, n)})
    prof = semantics.profile(df)
    assert prof["roles"]["sex"]["role"] == "flag", "two integer codes are a flag, not a count"
    assert prof["measures"][0] == "progression", "a numeric outcome named like one leads"
    assert "year" in prof["nonfactors"]
    r = investigate.investigate(df)
    assert r["meta"]["main"] == "progression"
    factors = [x["factor"] for x in r["results"]["assoc"]["value"]["factors"]]
    assert "year" not in factors and factors[0] == "bmi"
    rho = stats.spearmanr(df.bmi, df.progression)[0]
    assert r["results"]["assoc"]["value"]["factors"][0]["value"] == pytest.approx(rho)


def test_concept_words_in_other_forms_offer_the_outcome_not_an_identifier():
    df = machines().rename(columns={"failure": "attrition", "unit_code": "employee_id"})
    out = ask("What factors are associated with employees quitting?", df, {})
    assert out["plan"]["type"] == "clarify" and out["plan"]["options"] == ["attrition"]
    assert semantics.concept_candidates("employees", df) == [], "an identifier is never offered"


@pytest.mark.parametrize("q,which", [("Go back to the second thing you found and explain exactly how it was measured.", 2),
                                     ("Tell me more about finding two.", 2), ("Explain the 3rd point.", 3),
                                     ("What about the first finding?", 1), ("Explain the last thing you found.", -1)])
def test_ordinal_references_to_findings(q, which):
    df, mem = machines(), {}
    first = ask("Analyse this entire dataset and identify important patterns.", df, mem)
    stored = mem["investigation"]["findings"]
    i = len(stored) if which == -1 else which
    out = ask(q, df, mem)
    assert out["answer"].startswith(f"**Finding {i}: {stored[i - 1]['title']}") and "**Calculation:**" in out["answer"]
    assert first["calls"]


def test_an_ordinal_without_findings_in_this_conversation_is_not_a_reference():
    df = machines()
    ask("Analyse this entire dataset and identify important patterns.", df, {})           # another conversation
    out = ask("Go back to the second thing you found.", df, {})
    assert not out["answer"].startswith("**Finding")
    assert orchestrator.finding_ref("The second shift ran longer", 5) is None
    assert orchestrator.finding_ref("Show finding 9", 5) == 9                             # asked about, not clamped
