"""P1 phase 6: honest model selection, accuracy metrics and visible data checks for forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analyst import compound, forecasting as F
from app.analyst.data import load_table
from app.analyst.deliverables import Deliverable


def test_a_method_must_beat_the_naive_baseline_by_a_margin():
    # a random walk: the naive forecast is optimal, so any "winner" is luck and must not be preferred
    rng = np.random.default_rng(12)
    picked_other = 0
    for seed in range(25):
        r = np.random.default_rng(seed)
        y = 100 + np.cumsum(r.normal(0, 5, 30))
        fc = F.forecast(pd.Series(y, index=pd.period_range("2023-01", periods=30, freq="M")), 1, "M")
        if fc["method"] != "naive":
            picked_other += 1
            assert fc["backtest_mae"] <= (1 - F.MIN_SKILL) * fc["naive_mae"] + 1e-9
        else:
            assert "naive" in fc["selection"]
    assert picked_other < 25


def test_metrics_are_reported_and_consistent():
    y = 50 + 2 * np.arange(24) + np.random.default_rng(1).normal(0, 1, 24)
    fc = F.forecast(pd.Series(y, index=pd.period_range("2023-01", periods=24, freq="M")), 2, "M")
    row = next(r for r in fc["backtest"] if r["method"] == fc["method"])
    assert fc["backtest_rmse"] >= fc["backtest_mae"] > 0 and fc["mase"] is not None and row["rmse"] == fc["backtest_rmse"]
    assert any(r["method"] == "moving_average" for r in fc["backtest"])


def test_data_checks_report_gaps_combined_records_and_outliers(tmp_path):
    days = list(pd.date_range("2025-01-01", periods=14, freq="MS"))
    rows = [{"month": d.strftime("%Y-%m-%d"), "site": "A", "units": 100.0 + i} for i, d in enumerate(days) if i != 5]
    rows.append({"month": days[2].strftime("%Y-%m-%d"), "site": "A", "units": 7.0})          # a second record in March
    rows[9]["units"] = 900.0                                                                  # an outlying month
    p = tmp_path / "u.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    df = load_table(str(p))
    d = Deliverable(id=1, kind="forecast", clause="q", order=0, measures=["units"], entities={"site": "A"}, horizon=1)
    ctx = compound.execute([d], df)
    res = ctx["results"][1]
    checks = res["forecast"]["data_checks"]
    assert checks["missing_periods"] == ["2025-06"] and checks["multi_record_periods"] == 1 and checks["outliers"]
    assert "with no records (2025-06)" in res["text"] and "combined into one value per month" in res["text"]
    assert "unusual month(s) kept in the history" in res["text"]


def test_daily_pace_on_working_day_data_is_per_recorded_day(tmp_path):
    from app.analyst import decomposer as Z
    days = pd.bdate_range("2025-03-03", periods=40)
    rows = [{"day": d.strftime("%Y-%m-%d"), "rep": "Lina", "calls": 28 + (i % 5)} for i, d in enumerate(days)]
    p = tmp_path / "c.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    df = load_table(str(p))
    ds, errs = Z.ground([{"do": "pace", "col": "calls", "where": {"rep": "Lina"}, "target": 150, "per": "W", "pace_per": "D"}],
                        df, "q")
    a = compound.answer("q", ds, df)["answer"]
    assert "**30.00 per day** (150 ÷ 5.00 recorded days in a week)" in a and "per calendar day that is 21.43" in a
    last3 = pd.DataFrame(rows).calls.iloc[-3:].mean()
    assert f"averaged **{last3:,.2f} per day**" in a


def test_weekly_pace_uses_complete_weeks_of_real_data(tmp_path):
    from app.analyst import decomposer as Z
    days = pd.bdate_range("2025-03-03", periods=37)          # ends on a Tuesday: the last week is incomplete
    rows = [{"day": d.strftime("%Y-%m-%d"), "rep": "Lina", "calls": 28 + i % 5} for i, d in enumerate(days)]   # 150 a full week
    p = tmp_path / "c.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    df = load_table(str(p))
    ds, errs = Z.ground([{"do": "pace", "col": "calls", "where": {"rep": "Lina"}, "target": 640, "per": "M", "pace_per": "W"}], df, "q")
    assert not errs
    a = compound.answer("q", ds, df)["answer"]
    assert "the last 3 complete weeks averaged **150 per week**" in a and "57" not in a.split("Recent pace")[1][:80]
