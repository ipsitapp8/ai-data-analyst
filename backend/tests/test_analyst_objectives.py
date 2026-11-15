"""Stages 3-4: target feasibility and scenarios as analysis programs. A schema unlike the demo timesheet (couriers,
a recorded hours column, a mid-month 'today'); every expected number is computed here with pandas."""
from __future__ import annotations

import pandas as pd
import pytest

from app.analyst import orchestrator, planner, scope

TODAY = pd.Timestamp("2025-06-10")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(scope, "today", lambda: TODAY)


def couriers():
    rows = []
    plan = {"Mei Tan": [9, 0, 10, 8, 0, 9, 11, 0, 7, 9], "Joel Okoro": [6, 6, 6, 0, 6, 6, 0, 6, 6, 6],
            "Rui Costa": [12, 12, 0, 12, 12, 0, 12, 12, 0, 12]}
    for name, hrs in plan.items():
        for i, h in enumerate(hrs):
            if h:
                rows.append({"courier": name, "depot": "North" if name != "Rui Costa" else "South",
                             "shift_day": pd.Timestamp("2025-06-01") + pd.Timedelta(days=i), "hours_driven": float(h),
                             "parcels": int(h * 9)})
    rows.append(rows[0].copy())                                       # an exact duplicate
    return pd.DataFrame(rows)


def ask(q, df, mem):
    return orchestrator.ask(q, df, mem, use_model=False)


def prog_result(out):
    return next(c["result"] for c in out["calls"] if c["tool"] == "analysis_program")


def truth(df, who, target, end="2025-06-30", duplicates="keep"):
    # one duplicate policy everywhere: exact duplicates count as recorded unless the user asks to exclude them
    d = df if duplicates == "keep" else df.drop_duplicates()
    d = d[(d.courier == who) & (d.shift_day <= TODAY)]
    daily = d.groupby("shift_day").hours_driven.sum().reindex(pd.date_range(d.shift_day.min(), TODAY), fill_value=0)
    done = d[d.shift_day >= "2025-06-01"].hours_driven.sum()
    left = (pd.Timestamp(end) - TODAY).days
    return {"done": done, "pace": daily.mean(), "left": left, "required": (target - done) / left,
            "expected": done + daily.mean() * left}


def test_feasibility_for_one_person():
    df, mem = couriers(), {}
    out = ask("Can Mei Tan reach 250 hours driven this month?", df, mem)
    t = truth(df, "Mei Tan", 250)
    v = prog_result(out)["results"]["Mei Tan"]["verdict"]["value"]
    assert v["completed"] == pytest.approx(t["done"]) and v["days_left"] == t["left"]
    assert v["required_daily"] == pytest.approx(t["required"]) and v["current_daily"] == pytest.approx(t["pace"])
    assert v["expected_total"] == pytest.approx(t["expected"])
    assert out["plan"]["type"] == "program" and "not a certainty" in out["answer"]
    assert "hours driven" in out["answer"] and "required pace" in out["answer"] and "current pace" in out["answer"]


@pytest.mark.parametrize("target,word", [(30, "already reached"), (1000, "not physically possible")])
def test_reached_and_impossible_targets(target, word):
    out = ask(f"Can Rui Costa reach {target} hours driven this month?", couriers(), {})
    assert word in out["answer"]


def test_who_is_at_risk_runs_for_everyone_sorted_by_shortfall():
    df = couriers()
    out = ask("Which couriers are at risk of missing a monthly target of 200 hours driven?", df, {})
    res = prog_result(out)["results"]
    assert set(res) == {"Mei Tan", "Joel Okoro", "Rui Costa"}
    for who in res:
        assert res[who]["verdict"]["value"]["expected_total"] == pytest.approx(truth(df, who, 200)["expected"])
    order = [line.split("|")[1].strip() for line in out["answer"].splitlines() if line.startswith("| ") and "|---" not in line][1:]
    gaps = {w: truth(df, w, 200)["expected"] - 200 for w in res}
    assert order == sorted(gaps, key=gaps.get)


def test_absence_change_target_compare_and_why():
    df, mem = couriers(), {}
    ask("Can Joel Okoro complete 150 hours driven this month?", df, mem)
    absent = ask("What if he is absent for 5 days?", df, mem)
    t = truth(df, "Joel Okoro", 150)
    v = prog_result(absent)["results"]["Joel Okoro"]["verdict"]["value"]
    assert v["days_left"] == t["left"] - 5
    assert v["expected_total"] == pytest.approx(t["done"] + t["pace"] * (t["left"] - 5))
    assert "hypothetical" in absent["answer"].lower()
    changed = ask("Change the target to 120 hours.", df, mem)
    v2 = prog_result(changed)["results"]["Joel Okoro"]["verdict"]["value"]
    assert v2["target"] == 120 and v2["days_left"] == t["left"] - 5, "the absence assumption is kept"
    why = ask("Why is the result different?", df, mem)
    assert "target changed (150 → 120)" in why["answer"]


def test_team_absence_scenario_and_rerun():
    df, mem = couriers(), {}
    out = ask("If Rui Costa is absent for the next 10 days, how will the team's workload change?", df, mem)
    d = df                                                            # duplicates counted as recorded
    d = d[d.shift_day <= TODAY]
    window = pd.date_range(d.shift_day.min(), d.shift_day.max())
    team = d.groupby("shift_day").hours_driven.sum().reindex(window, fill_value=0).mean() * 10
    rest = d[d.courier != "Rui Costa"].groupby("shift_day").hours_driven.sum().reindex(window, fill_value=0).mean() * 10
    r = prog_result(out)["results"][""]
    assert r["normal"]["value"]["expected"] == pytest.approx(team)
    assert r["reduced"]["value"]["expected"] == pytest.approx(rest)
    assert "not workload in tasks, output or productivity" in out["answer"]
    again = ask("Make it 20 days.", df, mem)
    assert prog_result(again)["results"][""]["normal"]["value"]["expected"] == pytest.approx(team * 2)
    cmp = ask("Compare this with the previous scenario.", df, mem)
    assert "days changed (10 → 20)" in cmp["answer"]


def test_compare_with_the_others():
    df, mem = couriers(), {}
    ask("Can Mei Tan reach 250 hours driven this month?", df, mem)
    out = ask("Compare her with the others.", df, mem)
    totals = df.groupby("courier").hours_driven.sum()                 # descriptive: duplicates counted as recorded
    r = prog_result(out)["results"][""]["per"]["value"]
    assert r == pytest.approx(totals.to_dict())
    assert f"ranked {1 + int((totals > totals['Mei Tan']).sum())} of 3" in out["answer"]


def test_missing_pieces_are_asked_for_not_invented():
    df = couriers()
    assert "What is the target?" in ask("Which couriers are at risk of missing their targets?", df, {})["answer"]
    out = ask("Can Mei Tan reach 300 dollars this month?", df, {})
    assert out["plan"]["type"] == "clarify" and not out["calls"]
    nodate = df.drop(columns=["shift_day"])
    assert "date column" in ask("Can Mei Tan reach 250 hours driven this month?", nodate, {})["answer"]


def test_descriptive_comparison_keeps_duplicates_like_the_calculator():
    df, mem = couriers(), {}
    calcd = ask("Total hours driven by courier", df, mem)
    out = ask("Compare Mei Tan with the others.", df, mem)
    per = prog_result(out)["results"][""]["per"]["value"]
    assert per == pytest.approx(df.groupby("courier").hours_driven.sum().to_dict()), "duplicate counted, as recorded"
    assert "Counted as recorded" in out["answer"]


def test_one_duplicate_policy_for_totals_targets_reruns_and_scenarios():
    """Exact duplicates are kept and flagged by default on every path, excluded only when asked, and the choice is
    carried into a re-run of the same target. Mei Tan's first record is duplicated in couriers()."""
    df = couriers()
    mei = df[(df.courier == "Mei Tan") & (df.shift_day <= TODAY)]
    total = ask("Total hours driven for Mei Tan", df, {})
    assert f"**{mei.hours_driven.sum():.2f}**" in total["answer"], "the plain total counts the duplicate too"
    assert f"1 exact duplicate record(s) (line {len(df) + 1}) were counted as recorded" in total["answer"], "and says so"
    mem = {}
    kept = prog_result(ask("Can Mei Tan reach 250 hours driven this month?", df, mem))["results"]["Mei Tan"]
    assert kept["verdict"]["value"]["completed"] == pytest.approx(mei.hours_driven.sum())
    rerun = ask("What if the target is 260?", df, mem)
    assert prog_result(rerun)["results"]["Mei Tan"]["verdict"]["value"]["completed"] == pytest.approx(mei.hours_driven.sum())
    assert "Counted as recorded" in rerun["answer"]
    mem = {}
    out = ask("Can Mei Tan reach 250 hours driven this month, excluding duplicates?", df, mem)
    t = truth(df, "Mei Tan", 250, duplicates="drop")
    v = prog_result(out)["results"]["Mei Tan"]["verdict"]["value"]
    assert v["completed"] == pytest.approx(t["done"]) and v["expected_total"] == pytest.approx(t["expected"])
    assert "Not counted: 1 record(s) — exact duplicate" in out["answer"]
    again = ask("What if the target is 260?", df, mem)
    assert prog_result(again)["results"]["Mei Tan"]["verdict"]["value"]["completed"] == pytest.approx(t["done"]), \
        "the exclusion the user asked for is kept on the re-run"
    d = df.drop_duplicates()
    d = d[d.shift_day <= TODAY]
    window = pd.date_range(d.shift_day.min(), d.shift_day.max())
    team = d.groupby("shift_day").hours_driven.sum().reindex(window, fill_value=0).mean() * 10
    sc = ask("If Rui Costa is absent for the next 10 days, excluding duplicates, how will the team's workload change?",
             df, {})
    assert prog_result(sc)["results"][""]["normal"]["value"]["expected"] == pytest.approx(team)


def test_switching_to_another_dataset_is_explained():
    out = ask("Forget that and analyse revenue instead.", couriers(), {})
    assert "Dataset** selector" in out["answer"] and not out["calls"]


def test_the_target_unit_names_the_measure_when_one_column_carries_it():
    df = couriers().assign(km_covered=lambda d: d.hours_driven * 30).drop(columns=["parcels"])
    out = ask("Can Mei Tan reach 6000 km this month?", df, {})
    v = prog_result(out)["results"]["Mei Tan"]["verdict"]["value"]
    d = df
    d = d[(d.courier == "Mei Tan") & (d.shift_day >= "2025-06-01") & (d.shift_day <= TODAY)]
    assert v["target"] == 6000 and v["completed"] == pytest.approx(d.km_covered.sum())


def test_a_closure_is_a_hypothetical_removal_not_a_filter_to_the_closed_group():
    df = couriers()
    out = ask("How many hours driven would we lose if the South depot closed?", df, {})
    r = prog_result(out)["results"][""]
    assert r["normal"]["value"] == pytest.approx(df.hours_driven.sum())
    assert r["closed"]["value"] == pytest.approx(df[df.depot != "South"].hours_driven.sum())
    assert r["closed"]["hypothetical"] and r["cmp"]["hypothetical"] and not r["normal"]["hypothetical"]
    assert "hypothetical scenario" in out["answer"] and "**Scope:** depot = South" not in out["answer"]
    out = ask("Over the next 5 days, how many hours driven would we lose if the South depot were shut?", df, {})
    r = prog_result(out)["results"][""]
    d = df[df.shift_day <= TODAY]
    window = pd.date_range(d.shift_day.min(), d.shift_day.max())
    daily = lambda x: x.groupby("shift_day").hours_driven.sum().reindex(window, fill_value=0).mean() * 5  # noqa: E731
    assert r["normal"]["value"]["expected"] == pytest.approx(daily(d))
    assert r["closed"]["value"]["expected"] == pytest.approx(daily(d[d.depot != "South"]))
    out = ask("If the South depot closed, what happens?", df.drop(columns=["parcels"]).assign(extra=1.0), {})
    assert out["plan"]["type"] == "clarify" and "which value" in out["answer"]
    assert ask("Total hours driven for the South depot", df, {})["answer"].count("Scenario") == 0
