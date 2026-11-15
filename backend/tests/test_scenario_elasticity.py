"""What-if Phase 2, feature 1: price elasticity estimated from history.

The data are synthetic with a known true elasticity, so what the estimator should find is fixed by the formula that
generated the data, not by the code under test.
"""
from __future__ import annotations

import copy
import io
import time

import numpy as np
import pandas as pd
import pytest
from platform_helpers import workspace

from app import scenario_elasticity as el
from app import scenario_verify


def monthly(n=36, e=-1.5, seed=0, noise=0.05, price_sd=0.12, start="2022-01-01", rows_per_month=1):
    """ln(quantity) = ln(1000) + e x ln(price / 20) + noise, one price per month."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="MS")
    price = 20 * np.exp(rng.normal(0, price_sd, n))
    qty = 1000 * (price / 20) ** e * np.exp(rng.normal(0, noise, n))
    out = []
    for d, p, q in zip(dates, price, qty):
        for k in range(rows_per_month):
            out.append({"date": d + pd.Timedelta(days=k), "price": p, "units": q / rows_per_month})
    return pd.DataFrame(out)


def estimate(df, **kw):
    kw.setdefault("date_column", "date")
    return el.estimate(df, "price", "units", **kw)


# ======================================================== recovering the truth ==

def test_a_known_elasticity_is_recovered_inside_its_interval():
    r = estimate(monthly(36, e=-1.5))
    e = r["elasticity"]
    assert r["status"] == "estimated" and r["data"]["observations"] == 36 and r["data"]["unit"] == "month"
    assert abs(e["estimate"] + 1.5) < 0.15 and e["ci_low"] < -1.5 < e["ci_high"]
    assert r["diagnostics"]["r_squared"] > 0.8 and r["reliability"]["level"] == "usable" and r["reliability"]["can_apply"]
    assert e["standard_error"] > 0 and e["level"] == 0.95 and 0 <= e["p_value"] < 0.001
    assert r["data_kind"] == "statistical_estimate" and r["limitations"]


def test_a_higher_confidence_level_gives_a_wider_interval_and_is_deterministic():
    df = monthly(36)
    narrow, wide = estimate(df, level=0.80)["elasticity"], estimate(df, level=0.99)["elasticity"]
    assert wide["ci_high"] - wide["ci_low"] > narrow["ci_high"] - narrow["ci_low"]
    assert estimate(df) == estimate(df), "no randomness is involved"


def test_several_rows_per_month_are_aggregated_with_a_quantity_weighted_price():
    rows = [{"date": "2022-01-05", "price": 10.0, "units": 100.0}, {"date": "2022-01-20", "price": 20.0, "units": 300.0},
            {"date": "2022-02-03", "price": 12.0, "units": 50.0}]
    prep = el.prepare(pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"])), "price", "units", "date", None, "month", None)
    obs = prep["observations"]
    # January: quantity 400, price (10 x 100 + 20 x 300) / 400 = 17.5
    jan = obs[obs["period"] == pd.Timestamp("2022-01-01")].iloc[0]
    assert jan["qty"] == 400.0 and jan["price"] == 17.5 and jan["rows"] == 2


def test_auto_granularity_takes_the_coarsest_period_with_enough_observations():
    daily = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=60), "price": 1.0, "units": 1.0})
    assert el.choose_granularity(daily["date"]) == "day"
    assert el.choose_granularity(pd.Series(pd.date_range("2018-01-01", periods=40, freq="QS"))) == "quarter"
    assert el.choose_granularity(pd.Series(pd.date_range("2020-01-01", periods=30, freq="MS"))) == "month"


# ============================================================ unsuitable data ==

def test_zero_negative_and_missing_values_are_counted_and_left_out():
    df = monthly(30)
    junk = pd.DataFrame([{"date": df["date"].iloc[0], "price": 0.0, "units": 50.0}, {"date": df["date"].iloc[1], "price": -3.0, "units": 5.0},
                         {"date": df["date"].iloc[2], "price": 20.0, "units": 0.0}, {"date": df["date"].iloc[3], "price": None, "units": 9.0},
                         {"date": df["date"].iloc[4], "price": 20.0, "units": -4.0}])
    r = estimate(pd.concat([df, junk], ignore_index=True))
    d = r["data"]
    assert r["status"] == "estimated" and d["price_not_positive"] == 2 and d["quantity_zero_or_negative"] == 2 and d["missing_value"] == 1
    assert d["rows_in"] == 35 and d["observations"] == 30
    assert abs(r["elasticity"]["estimate"] - estimate(df)["elasticity"]["estimate"]) < 1e-9, "left-out rows change nothing"


@pytest.mark.parametrize("make,fragment", [
    (lambda: monthly(6), "only 6 usable observation"),
    (lambda: monthly(30).assign(price=20.0), "fewer than 3 distinct values"),
    (lambda: monthly(30).assign(price=lambda d: np.where(np.arange(len(d)) % 2 == 0, 20.0, 20.5)), "fewer than 3 distinct values"),
    (lambda: monthly(30, price_sd=0.001), "barely changes"),
    (lambda: monthly(12).assign(price=-1.0), "price of zero or less"),
])
def test_unsuitable_data_is_refused_with_the_reason_and_no_estimate(make, fragment):
    r = estimate(make())
    assert r["status"] == "not_estimated" and r["elasticity"] is None and r["verification"] is None
    assert any(fragment in x for x in r["reasons"]), r["reasons"]
    assert r["reliability"]["level"] == "not_estimated" and "Do not use" in r["reliability"]["recommendation"]


def test_demand_that_rises_with_price_is_not_recommended():
    r = estimate(monthly(36, e=+1.2))
    assert r["status"] == "estimated" and r["elasticity"]["estimate"] > 0
    assert r["reliability"]["level"] == "not_recommended" and r["reliability"]["can_apply"] is False
    assert "not negative" in r["reliability"]["recommendation"]


def test_an_interval_that_includes_zero_is_not_recommended():
    r = estimate(monthly(10, e=-0.05, noise=0.3, price_sd=0.05))
    assert r["status"] == "estimated" and r["elasticity"]["ci_low"] < 0 < r["elasticity"]["ci_high"]
    assert r["reliability"]["level"] == "not_recommended" and "includes zero" in r["reliability"]["recommendation"]


def test_a_small_sample_is_usable_only_with_a_caution():
    r = estimate(monthly(14, e=-1.5, noise=0.03))
    assert r["reliability"]["level"] == "use_with_caution" and any("14 observations" in f for f in r["reliability"]["flags"])


# ================================================================== outliers ==

def test_outliers_are_flagged_reported_and_can_be_excluded_on_request():
    df = monthly(36, e=-1.5, noise=0.03)
    df.loc[5, "units"] *= 6          # a one-off spike that has nothing to do with price
    df.loc[20, "units"] *= 0.1
    kept = estimate(df)
    out = kept["diagnostics"]["outliers"]
    assert out["count"] >= 2 and out["without_outliers"] is not None and out["excluded_from_primary"] is False
    trimmed = estimate(df, exclude_outliers=True)
    assert trimmed["diagnostics"]["outliers"]["excluded_from_primary"] is True and trimmed["diagnostics"]["observations"] < 36
    assert abs(trimmed["elasticity"]["estimate"] + 1.5) < abs(kept["elasticity"]["estimate"] + 1.5)
    assert abs(trimmed["elasticity"]["estimate"] + 1.5) < 0.2
    assert out["without_outliers"]["estimate"] == pytest.approx(trimmed["elasticity"]["estimate"])
    assert any(p["outlier"] for p in kept["scatter"])


# ============================================ seasonality, trend and segments ==

def seasonal(n=48, e=-1.2, seed=1, coupling=0.15, demand_swing=0.6):
    """Price and demand both peak in the same season, so a naive price-only fit reads the season as a price response."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    s = np.cos(2 * np.pi * (dates.month - 1) / 12)
    lnp = np.log(20) + coupling * s + rng.normal(0, 0.05, n)
    lnq = np.log(1000) + e * (lnp - np.log(20)) + demand_swing * s + rng.normal(0, 0.03, n)
    return pd.DataFrame({"date": dates, "price": np.exp(lnp), "units": np.exp(lnq)})


def test_seasonality_that_moves_with_price_is_controlled_when_two_years_of_data_exist():
    df = seasonal(48)
    naive = np.polyfit(np.log(df["price"]), np.log(df["units"]), 1)[0]
    assert naive > 0, "without controls the seasonal pattern makes demand look like it rises with price"
    r = estimate(df)
    assert r["diagnostics"]["controls"]["seasonality"] == "month-of-year dummies"
    assert abs(r["elasticity"]["estimate"] + 1.2) < 0.4 and r["elasticity"]["ci_low"] < -1.2 < r["elasticity"]["ci_high"]


def test_seasonality_that_cannot_be_controlled_is_said_to_be_uncontrolled():
    r = estimate(seasonal(20))
    assert r["diagnostics"]["controls"]["seasonality"].startswith("not controlled: the data spans 20 months")
    assert any("seasonality is not controlled" in f for f in r["reliability"]["flags"])


def test_a_trend_in_demand_is_controlled_for():
    df = monthly(36, e=-1.0, noise=0.02)
    df["units"] *= np.exp(0.03 * np.arange(36))         # demand drifts up 3% a month, unrelated to price
    r = estimate(df)
    assert r["diagnostics"]["controls"]["trend"] == "linear time trend" and abs(r["elasticity"]["estimate"] + 1.0) < 0.25


def test_segment_intercepts_stop_product_differences_being_read_as_a_demand_curve():
    rng = np.random.default_rng(4)
    frames = []
    for name, p0, q0 in (("Basic", 10.0, 5000.0), ("Premium", 100.0, 200.0)):
        d = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=30, freq="MS")})
        d["price"] = p0 * np.exp(rng.normal(0, 0.1, 30))
        d["units"] = q0 * (d["price"] / p0) ** -0.8 * np.exp(rng.normal(0, 0.03, 30))
        d["product"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    rows = df.drop(columns="date")
    pooled = el.estimate(rows, "price", "units")["elasticity"]["estimate"]
    by_product = el.estimate(rows, "price", "units", segment_column="product")
    assert pooled < -1.2, "pooled across products the price gap between them is mistaken for a steeper demand curve"
    assert abs(by_product["elasticity"]["estimate"] + 0.8) < 0.15 and "2 segment intercepts" in by_product["diagnostics"]["controls"]["segments"]
    assert by_product["data"]["observations"] == 60 and "HC3" in by_product["elasticity"]["method"]
    dated = estimate(df, segment_column="product")           # (month, product) observations, each product with its own intercept
    assert dated["data"]["observations"] == 60 and abs(dated["elasticity"]["estimate"] + 0.8) < 0.15
    assert any("combined into one quantity" in x for x in dated["limitations"])


def test_a_single_series_uses_newey_west_errors_and_reports_autocorrelation():
    r = estimate(monthly(36))
    assert "Newey-West" in r["elasticity"]["method"] and r["diagnostics"]["residual_autocorrelation_lag1"] is not None


def test_without_a_date_column_each_row_is_an_observation_and_the_estimate_is_cautioned():
    df = monthly(40, e=-1.3, noise=0.03).drop(columns="date")
    r = el.estimate(df, "price", "units")
    assert r["status"] == "estimated" and r["data"]["unit"] == "row" and abs(r["elasticity"]["estimate"] + 1.3) < 0.2
    assert any("each row is an observation" in f for f in r["reliability"]["flags"]) and r["reliability"]["level"] == "use_with_caution"
    assert r["diagnostics"]["period_start"] is None and "HC3" in r["elasticity"]["method"]


def test_filters_restrict_the_history_used():
    df = monthly(48, e=-1.5)
    r = estimate(df, filters=[{"column": "date", "op": "between", "value": ["2023-01-01", "2024-12-31"]}])
    assert r["data"]["observations"] == 24 and r["diagnostics"]["period_start"] == "2023-01-01"


@pytest.mark.parametrize("kwargs,message", [
    ({"price_column": "ghost"}, "does not exist"), ({"quantity_column": "ghost"}, "does not exist"),
    ({"price_column": "units"}, "must be different"), ({"granularity": "fortnight"}, "granularity must be"),
    ({"level": 1.2}, "level must be"), ({"level": True}, "level must be"), ({"date_column": "ghost"}, "does not exist"),
    ({"segment_column": "ghost"}, "does not exist"), ({"filters": [{"column": "nope", "op": "==", "value": 1}]}, "does not exist"),
])
def test_bad_requests_are_refused_with_the_reason(kwargs, message):
    args = {"price_column": "price", "quantity_column": "units", "date_column": "date"} | kwargs
    with pytest.raises(el.ElasticityError, match=message):
        el.estimate(monthly(20), args.pop("price_column"), args.pop("quantity_column"), **args)


def test_text_columns_are_not_accepted_as_numbers():
    df = monthly(20).assign(label="x", when="not a date")
    with pytest.raises(el.ElasticityError, match="not numeric"):
        el.estimate(df, "label", "units", date_column="date")
    with pytest.raises(el.ElasticityError, match="not made of dates"):
        el.estimate(df, "price", "units", date_column="when")


# =============================================================== verification ==

def test_the_estimate_is_independently_verified_including_the_data_preparation():
    r = estimate(monthly(36, rows_per_month=3))
    v = r["verification"]
    assert v["status"] == "verified", [c for c in v["checks"] if c["status"] != "pass"]
    status = {c["check"]: c["status"] for c in v["checks"]}
    assert all(status.get(k) == "pass" for k in ("coefficient", "r_squared", "interval", "observations_rebuilt")), status
    seg = estimate(seasonal(48))
    assert seg["verification"]["status"] == "verified"


def test_a_tampered_estimate_or_observation_is_not_verified(monkeypatch):
    captured = {}
    original = scenario_verify.verify_elasticity

    def spy(result, design, fit, obs, df):
        captured.update(result=result, design=design, fit=fit, obs=obs, df=df)
        return original(result, design, fit, obs, df)

    monkeypatch.setattr(scenario_verify, "verify_elasticity", spy)
    df = monthly(36)
    estimate(df)
    r, d, f, o = (captured[k] for k in ("result", "design", "fit", "obs"))
    assert original(copy.deepcopy(r), d, f, o, df)["status"] == "verified"
    bad = copy.deepcopy(r)
    bad["elasticity"]["estimate"] += 0.2
    assert original(bad, d, f, o, df)["status"] == "unverified"
    bad = copy.deepcopy(r)
    bad["elasticity"]["ci_high"] += 0.5
    assert {c["check"]: c["status"] for c in original(bad, d, f, o, df)["checks"]}["interval"] == "fail"
    skewed = o.copy()
    skewed.loc[skewed.index[3], "qty"] *= 1.5
    assert {c["check"]: c["status"] for c in original(copy.deepcopy(r), d, f, skewed, df)["checks"]}["observations_rebuilt"] == "fail"


def test_a_large_dataset_is_estimated_quickly():
    rng = np.random.default_rng(2)
    months = pd.date_range("2021-01-01", periods=36, freq="MS")
    month_price = 20 * np.exp(rng.normal(0, 0.1, 36))                 # the price changes between months, not row by row
    month_scale = (month_price / 20) ** -1.4
    n = 60_000
    m = rng.integers(0, 36, n)
    df = pd.DataFrame({"date": months[m] + pd.to_timedelta(rng.integers(0, 27, n), unit="D"), "price": month_price[m],
                       "units": 40 * month_scale[m] * np.exp(rng.normal(0, 0.1, n))})
    started = time.perf_counter()
    r = estimate(df)
    assert r["status"] == "estimated" and r["verification"]["status"] == "verified" and time.perf_counter() - started < 20


# ===================================================================== the API ==

def csv_of(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def demand_ws(client):
    return workspace(client, csv=csv_of(monthly(36, e=-1.5, rows_per_month=2)), filename="demand.csv")


def test_elasticity_endpoint_estimates_and_records_where_the_data_came_from(client, demand_ws):
    body = {"dataset_id": demand_ws["ds"], "price_column": "price", "quantity_column": "units", "date_column": "date"}
    r = client.post("/api/scenarios/elasticity", json=body, headers=demand_ws["h"])
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "estimated" and abs(out["elasticity"]["estimate"] + 1.5) < 0.2 and out["verification"]["status"] == "verified"
    p = out["provenance"]
    assert p["dataset_id"] == demand_ws["ds"] and p["dataset_version"] == 1 and len(p["content_sha256"]) == 64
    assert p["rows_in_dataset"] == 72 and p["latest_version"] == 1


def test_a_chosen_dataset_version_is_the_one_estimated_from(client, demand_ws):
    body = {"dataset_id": demand_ws["ds"], "price_column": "price", "quantity_column": "units", "date_column": "date"}
    v2 = monthly(36, e=-0.5, seed=9, rows_per_month=1)
    up = client.post(f"/api/datasets/{demand_ws['ds']}/versions", files={"file": ("v2.csv", csv_of(v2), "text/csv")}, headers=demand_ws["h"])
    assert up.status_code in (200, 201), up.text
    v1 = client.post("/api/scenarios/elasticity", json={**body, "dataset_version": 1}, headers=demand_ws["h"]).json()
    latest = client.post("/api/scenarios/elasticity", json=body, headers=demand_ws["h"]).json()
    assert v1["provenance"]["dataset_version"] == 1 and latest["provenance"]["dataset_version"] == 2
    assert v1["provenance"]["latest_version"] == 2 and v1["provenance"]["content_sha256"] != latest["provenance"]["content_sha256"]
    assert abs(v1["elasticity"]["estimate"] + 1.5) < 0.25 and abs(latest["elasticity"]["estimate"] + 0.5) < 0.25
    assert client.post("/api/scenarios/elasticity", json={**body, "dataset_version": 9}, headers=demand_ws["h"]).status_code == 404


@pytest.mark.parametrize("patch,code", [
    ({"price_column": "ghost"}, 400), ({"quantity_column": "date"}, 400), ({"granularity": "hourly"}, 400),
    ({"level": 5}, 422), ({"dataset_version": 0}, 422), ({"dataset_id": 987654}, 404),
])
def test_elasticity_endpoint_rejects_bad_requests_safely(client, demand_ws, patch, code):
    body = {"dataset_id": demand_ws["ds"], "price_column": "price", "quantity_column": "units", "date_column": "date", **patch}
    r = client.post("/api/scenarios/elasticity", json=body, headers=demand_ws["h"])
    assert r.status_code == code, r.text
    assert "Traceback" not in r.text


def test_elasticity_endpoint_is_team_scoped_and_needs_a_signed_in_member(client, demand_ws):
    body = {"dataset_id": demand_ws["ds"], "price_column": "price", "quantity_column": "units"}
    assert client.post("/api/scenarios/elasticity", json=body).status_code == 401
    other = workspace(client)
    assert client.post("/api/scenarios/elasticity", json=body, headers=other["h"]).status_code == 404


def test_an_insufficient_dataset_is_a_normal_answer_not_an_error(client):
    ws = workspace(client, csv=csv_of(monthly(5)), filename="tiny.csv")
    r = client.post("/api/scenarios/elasticity", json={"dataset_id": ws["ds"], "price_column": "price", "quantity_column": "units",
                                                       "date_column": "date"}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["status"] == "not_estimated" and r.json()["provenance"]["dataset_version"] == 1


def test_a_chosen_estimate_is_recorded_as_the_source_of_the_assumption_and_a_manual_value_stays_manual(client, demand_ws):
    base = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
    est = client.post("/api/scenarios/elasticity", json={"dataset_id": demand_ws["ds"], "price_column": "price", "quantity_column": "units",
                                                         "date_column": "date"}, headers=demand_ws["h"]).json()
    chosen = round(est["elasticity"]["estimate"], 3)
    source = {"price_elasticity": {"source": "estimate", "estimate": est["elasticity"]["estimate"], "ci_low": est["elasticity"]["ci_low"],
                                   "ci_high": est["elasticity"]["ci_high"], "engine_version": est["engine_version"],
                                   "provenance": est["provenance"]}}
    body = {"model": "unit_economics", "baseline": base, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}}
    manual = client.post("/api/scenarios/evaluate", json={**body, "assumptions": {"price_elasticity": -0.7}}, headers=demand_ws["h"]).json()
    assert manual["assumptions"]["price_elasticity"] == -0.7 and manual["assumption_sources"] == {}
    used = client.post("/api/scenarios/evaluate", json={**body, "assumptions": {"price_elasticity": chosen}, "assumption_sources": source},
                       headers=demand_ws["h"]).json()
    assert used["assumptions"]["price_elasticity"] == chosen and used["assumption_sources"]["price_elasticity"]["source"] == "estimate"
    assert used["assumption_sources"]["price_elasticity"]["provenance"]["dataset_version"] == 1
    ignored = client.post("/api/scenarios/evaluate", json={**body, "assumptions": {"price_elasticity": -1.0},
                                                           "assumption_sources": {"not_an_assumption": {"x": 1}}}, headers=demand_ws["h"]).json()
    assert ignored["assumption_sources"] == {}
    saved = client.post("/api/scenarios", json={**body, "name": "With estimate", "assumptions": {"price_elasticity": chosen},
                                                "assumption_sources": source}, headers=demand_ws["h"])
    got = client.get(f"/api/scenarios/{saved.json()['id']}", headers=demand_ws["h"]).json()
    assert got["request"]["assumption_sources"]["price_elasticity"]["provenance"]["content_sha256"] == est["provenance"]["content_sha256"]
