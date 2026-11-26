"""P2 dev-set regressions (data/training_expansion/p2_hybrid/eval/dev_p2): each test reproduces a failure found there."""
import pandas as pd
import pytest

from app.analyst import intent, orchestrator


def hotels():
    days = pd.date_range("2025-01-01", periods=60, freq="D")
    rows = []
    for prop, rate in (("Harbour Inn", 140.0), ("Summit Lodge", 190.0), ("City Suites", 150.0)):
        for i, d in enumerate(days):
            rows.append({"stay_date": d, "property": prop, "rooms_booked": 40 + i % 7, "adr_usd": rate + (i % 5) + 0.37})
    return pd.DataFrame(rows)


def test_mean_per_a_named_grouping_is_a_mean_for_each_group_not_a_ratio():
    # multi-word capitalised names make 'property' look like a people column; "per property" still groups
    df = hotels()
    r = intent.read("Mean adr_usd per property please", df)
    assert r.group == "property" and not r.denominators and not r.per
    out = orchestrator.ask("Mean adr_usd per property please", df, {}, use_model=False)
    truth = df.groupby("property").adr_usd.mean()
    for name, v in truth.items():
        assert f"{v:,.2f}" in out["answer"], name


def monthly():
    months = pd.date_range("2023-10-01", periods=26, freq="MS")
    return pd.DataFrame([{"order_month": m, "channel": c, "gmv": 1000.0 + 10 * i + (3.5 if c == "Web" else 0)}
                         for i, m in enumerate(months) for c in ("Web", "App")])


@pytest.mark.parametrize("words,months", [("the second half of 2024", range(7, 13)), ("H1 2024", range(1, 7)),
                                          ("Q3 2024", range(7, 10)), ("the first quarter of 2025", range(1, 4))])
def test_half_year_and_quarter_windows_are_applied(words, months):
    df = monthly()
    year = int(words[-4:])
    d = pd.to_datetime(df.order_month)
    truth = df[(d.dt.year == year) & d.dt.month.isin(list(months))].gmv.mean()
    out = orchestrator.ask(f"average gmv in {words}", df, {}, use_model=False)
    assert f"{truth:,.2f}" in out["answer"]
    assert f"{len(df[(d.dt.year == year) & d.dt.month.isin(list(months))])} of {len(df)} records" in out["answer"]


# P2.14 date filters: a partial date means the whole period; a window the question names is owned by the scope reader
def test_an_equality_filter_on_a_month_or_year_means_the_whole_period():
    import pandas as pd
    from app.analyst import tools
    df = pd.DataFrame({"day": pd.date_range("2023-11-01", periods=200, freq="D"), "units": 1.0})
    for value, n in (("2024-02", 29), ("2024", 139), ("2023-12", 31)):
        call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "sum", "column": "units",
                                    "filters": [{"column": "day", "op": "==", "value": value}]}}, df)
        assert tools.run(call, df, {})["value"] == n, value
    full = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "sum", "column": "units",
                                "filters": [{"column": "day", "op": "==", "value": "2024-02-03"}]}}, df)
    assert tools.run(full, df, {})["value"] == 1


def test_the_models_date_filter_never_overrides_the_period_the_question_names():
    from app.analyst import orchestrator
    sc = {"time": {"column": "day", "year": 2024, "month": 2, "label": "February 2024"}}
    calls = [{"tool": "compute_metric", "args": {"aggregation": "sum", "column": "units", "filters": [
        {"column": "day", "op": "==", "value": "2024-02-01"}, {"column": "shop", "op": "==", "value": "North"}]}}]
    notes = []
    orchestrator._window_owned_by_scope(calls, sc, notes)
    assert calls[0]["args"]["filters"] == [{"column": "shop", "op": "==", "value": "North"}]
    assert notes and "2024-02-01" in notes[0]
    untouched = [{"tool": "compute_metric", "args": {"filters": [{"column": "day", "op": ">=", "value": "2024-01-01"}]}}]
    orchestrator._window_owned_by_scope(untouched, {}, [])
    assert untouched[0]["args"]["filters"]
