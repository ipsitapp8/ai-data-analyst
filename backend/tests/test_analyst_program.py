"""The analytical program layer: composition, validation, deterministic results, provenance. Expected values are
computed here with pandas; the schemas are made up for the test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import program as P


def visits():
    rng = np.random.default_rng(2)
    rows = []
    for d in pd.date_range("2025-05-01", "2025-05-20"):
        for nurse in ["Asha Varga", "Bo Lund", "Cyd Ito"]:
            if rng.random() < 0.75:
                s = int(rng.choice([7, 8, 9]))
                rows.append({"nurse": nurse, "ward": "East" if nurse != "Cyd Ito" else "West", "day": d,
                             "in": f"{s:02d}:00", "out": f"{s + int(rng.choice([6, 8, 10])):02d}:30", "patients": int(rng.integers(3, 12))})
    return pd.DataFrame(rows)


def hours(df):
    f = lambda t: int(t[:2]) + int(t[3:]) / 60  # noqa: E731
    return df["out"].map(f) - df["in"].map(f)


def test_steps_compose_and_match_pandas():
    df = visits()
    prog = [{"id": "s", "op": "select", "entities": {"nurse": ["Bo Lund"]}},
            {"id": "d", "op": "derive", "from": "s", "name": "h", "formula": "hours_between(`in`, out)"},
            {"id": "t", "op": "aggregate", "from": "d", "fn": "sum", "column": "h"},
            {"id": "w", "op": "aggregate", "from": "d", "fn": "mean", "column": "patients", "per": "W"}]
    r = P.run(prog, df)
    bo = df[df.nurse == "Bo Lund"]
    assert r["t"].value == pytest.approx(hours(bo).sum()) and r["t"].unit == "hours"
    truth = bo.groupby(bo.day.dt.to_period("W").astype(str)).patients.mean()
    assert r["w"].value == pytest.approx(truth.to_dict())
    assert r["s"].provenance["rows_out"] == len(bo)


def test_projection_and_target_comparison():
    df = visits()
    prog = [{"id": "s", "op": "select", "entities": {"nurse": ["Asha Varga"]}},
            {"id": "d", "op": "derive", "from": "s", "name": "h", "formula": "hours_between(`in`, out)"},
            {"id": "done", "op": "aggregate", "from": "d", "fn": "sum", "column": "h"},
            {"id": "p", "op": "project", "from": "d", "column": "h", "days": 11},
            {"id": "c", "op": "compare", "a": "p", "base": "done", "target": 400, "days": 11, "unit": "hours", "max_daily": 24}]
    r = P.run(prog, df)
    a = df[df.nurse == "Asha Varga"].assign(h=lambda x: hours(x))
    daily = a.groupby(a.day).h.sum().reindex(pd.date_range(a.day.min(), a.day.max()), fill_value=0)
    assert r["p"].value["daily_mean"] == pytest.approx(daily.mean())
    assert r["p"].value["expected"] == pytest.approx(daily.mean() * 11)
    c = r["c"].value
    assert c["required_daily"] == pytest.approx((400 - a.h.sum()) / 11)
    assert c["expected_total"] == pytest.approx(a.h.sum() + daily.mean() * 11)
    assert c["verdict"].startswith(("unlikely", "uncertain", "likely", "not physically"))


def test_scenario_is_hypothetical_and_changes_only_the_copy():
    df = visits()
    prog = [{"id": "base", "op": "aggregate", "fn": "sum", "column": "patients"},
            {"id": "sc", "op": "scenario", "name": "without Bo", "changes": [{"remove": {"nurse": "Bo Lund"}}]},
            {"id": "after", "op": "aggregate", "from": "sc", "fn": "sum", "column": "patients"},
            {"id": "cmp", "op": "compare", "a": "after", "b": "base"}]
    r = P.run(prog, df)
    assert r["sc"].hypothetical and len(df) == len(visits())
    assert r["cmp"].value["difference"] == pytest.approx(-df[df.nurse == "Bo Lund"].patients.sum())


@pytest.mark.parametrize("prog,msg", [
    ([{"id": "a", "op": "aggregate", "fn": "sum", "column": "salary"}], "no column 'salary'"),
    ([{"id": "a", "op": "select", "entities": {"nurse": ["Zed"]}}], "Zed does not occur"),
    ([{"id": "a", "op": "aggregate", "from": "b", "fn": "sum", "column": "patients"}], "not an earlier step"),
    ([{"id": "a", "op": "aggregate", "fn": "sum", "column": "ward"}], "not numeric"),
    ([{"id": "a", "op": "aggregate", "fn": "explode", "column": "patients"}], "not an aggregation"),
    ([{"id": "a", "op": "teleport"}], "unknown operation"),
    ([{"id": "d", "op": "derive", "name": "h", "formula": "hours_between(`in`, out)"},
      {"id": "t", "op": "aggregate", "from": "d", "fn": "sum", "column": "h"},
      {"id": "c", "op": "compare", "a": "t", "target": 10, "unit": "dollars"}], "in dollars but the value is in hours"),
    ([{"id": "a", "op": "aggregate", "fn": "sum", "column": "patients"},
      {"id": "b", "op": "aggregate", "from": "a", "fn": "sum", "column": "patients"}], "a table is needed")])
def test_invalid_programs_are_rejected_with_the_reason(prog, msg):
    with pytest.raises(P.ProgramError, match=msg):
        P.run(prog, visits())


def test_select_removes_invalid_records_and_lists_them():
    # validity is judged from what a column is (a count cannot be negative); the name says it is a count
    df = pd.concat([visits().head(5), visits().head(1)], ignore_index=True).rename(columns={"patients": "patient_count"})
    df.loc[2, "patient_count"] = -3
    r = P.run([{"id": "s", "op": "select", "valid_for": ["patient_count"]}], df)
    reasons = {x["reason"] for x in r["s"].provenance["excluded"]}
    assert len(r["s"].value) == 4 and any("duplicate" in x for x in reasons) and any("negative" in x for x in reasons)


def test_results_computed_from_a_scenario_stay_hypothetical():
    import pandas as pd
    from app.analyst import program as P
    df = pd.DataFrame({"team": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]})
    r = P.run([{"id": "s", "op": "scenario", "changes": [{"remove": {"team": "b"}}]},
               {"id": "sel", "op": "select", "from": "s"},
               {"id": "t", "op": "aggregate", "from": "sel", "fn": "sum", "column": "x"},
               {"id": "real", "op": "aggregate", "fn": "sum", "column": "x"},
               {"id": "c", "op": "compare", "a": "t", "b": "real"}], df)
    assert r["s"].hypothetical and r["sel"].hypothetical and r["t"].hypothetical and r["c"].hypothetical
    assert not r["real"].hypothetical and r["t"].value == 4.0
