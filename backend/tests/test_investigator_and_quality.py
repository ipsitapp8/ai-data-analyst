"""Root-cause investigator (decomposition, bounds, honesty about causation) and
data observability (snapshots, drift metrics, incidents, configuration)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from platform_helpers import DRIFTED_CSV, FIXTURES, add_member, ask_and_run, question_row, run_job, workspace

from app import data_quality as dq
from app import investigator as inv
from app import metric_engine as me
from app.database import SessionLocal
from app.models import DQIncident, DQSnapshot
from app.profiling import load_and_profile_csv

SPLIT = "2025-03-01"
PERIODS = {"period_a": ["2025-01-01", SPLIT], "period_b": [SPLIT, "2025-04-29"]}


@pytest.fixture(scope="module")
def sales():
    df, profile = load_and_profile_csv(str(FIXTURES / "sales.csv"))
    return df, profile


def spec(formula="sum(revenue)", **kw):
    return {"formula": formula, "label": "Metric", "time_column": "order_date", **PERIODS, **kw}


# ========================================================= investigator ====

def test_additive_decomposition_reconciles_and_finds_the_planted_change(sales):
    df, profile = sales
    r = inv.investigate(df, profile, spec())
    assert r["status"] == "complete" and r["decomposition_method"] == "additive"
    change = r["totals"]["change"]
    assert math.isclose(change, r["totals"]["b"] - r["totals"]["a"])
    for dim in r["dimensions"]:
        assert dim["reconciles"], dim["dimension"]
        assert math.isclose(sum(s["contribution"] for s in dim["segments"]), change, rel_tol=1e-9)
    top = r["ranked_contributors"][0]
    assert (top["dimension"], top["segment"]) == ("region", "West")
    assert top["share_of_change"] > 1.0, "West fell by more than the total (others rose)"
    assert top["evidence_quality"] == "high" and top["consistent_across_halves"] is True
    assert "West" in r["summary"] and "117%" in r["summary"]


def test_ratio_metric_uses_rate_and_mix_and_still_reconciles(sales):
    df, profile = sales
    for formula in ("sum(revenue) / sum(units)", "avg(unit_price)"):
        r = inv.investigate(df, profile, spec(formula))
        assert r["decomposition_method"] == "rate_and_mix"
        change = r["totals"]["change"]
        for dim in r["dimensions"]:
            assert dim["reconciles"], (formula, dim["dimension"])
            assert math.isclose(dim["rate_effect_total"] + dim["mix_effect_total"], change, rel_tol=1e-9, abs_tol=1e-12)
            for s in dim["segments"]:
                assert math.isclose(s["rate_effect"] + s["mix_effect"], s["contribution"], rel_tol=1e-9, abs_tol=1e-12)


def test_price_volume_split_sums_exactly_to_the_change(sales):
    df, profile = sales
    pvm = inv.investigate(df, profile, spec())["price_volume"]
    assert pvm["reconciles"] and pvm["amount_column"] == "revenue" and pvm["quantity_column"] == "units"
    assert "confirmed on a sample" in pvm["basis"]
    assert pvm["price_effect"] > 0, "Gadget's price rose 10%"
    assert pvm["volume_effect"] < 0, "West's units fell"
    assert "not necessarily a list-price change" in pvm["note"]
    assert inv.investigate(df, profile, spec("sum(units)"))["price_volume"] is None, "not offered when it does not apply"


def test_entry_and_exit_of_segments_is_accounted_for():
    rows = [{"d": "2025-01-10", "seg": "old", "amt": 100.0}, {"d": "2025-01-20", "seg": "stay", "amt": 50.0},
            {"d": "2025-02-10", "seg": "stay", "amt": 80.0}, {"d": "2025-02-20", "seg": "new", "amt": 40.0}]
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"])
    profile = {"row_count": 4, "columns": [{"name": "d", "kind": "datetime"}, {"name": "seg", "kind": "categorical", "unique_count": 3},
                                           {"name": "amt", "kind": "numeric"}]}
    r = inv.investigate(df, profile, {"formula": "sum(amt)", "time_column": "d", "dimensions": ["seg"],
                                      "period_a": ["2025-01-01", "2025-02-01"], "period_b": ["2025-02-01", "2025-02-28"]})
    seg = {s["segment"]: s for s in r["dimensions"][0]["segments"]}
    assert seg["old"]["presence"] == "lost" and seg["new"]["presence"] == "new" and seg["stay"]["presence"] == "both"
    assert r["dimensions"][0]["reconciles"] and r["totals"]["change"] == -30.0
    assert all(c["evidence_quality"] == "low" for c in r["ranked_contributors"]), "two rows per period is thin evidence"
    assert r["status"] == "inconclusive"
    assert any("few rows" in u for u in r["unresolved"])


def test_inconclusive_when_nothing_changed_or_no_time_range(sales):
    df, profile = sales
    flat = inv.investigate(df, profile, spec("count_distinct(product)"))
    assert flat["status"] == "inconclusive" and "no material change" in flat["summary"]
    assert flat["ranked_contributors"] == [] and flat["hypotheses"] == []
    one_day = df[df["order_date"] == df["order_date"].min()]
    auto = inv.investigate(one_day, profile, {"formula": "sum(revenue)", "time_column": "order_date"})
    assert auto["status"] == "inconclusive" and "fewer than two distinct moments" in auto["summary"]


def test_report_never_claims_causation_and_lists_what_it_could_not_test(sales):
    df, profile = sales
    r = inv.investigate(df, profile, spec())
    assert any("not what caused it" in c for c in r["caveats"])
    assert any("data-quality" in a for a in r["alternative_explanations"])
    text = (r["summary"] + " ".join(h["hypothesis"] for h in r["hypotheses"])).lower()
    for word in ("because", "caused by", "due to", "led to"):
        assert word not in text
    assert all(h["status"] in ("supported", "weak", "not_supported") and h["test"] and h["result"] for h in r["hypotheses"])
    assert r["hypotheses"][0]["status"] == "supported"
    assert r["drilldowns"] and r["drilldowns"][0]["within"] == {"dimension": "region", "segment": "West"}
    assert r["drilldowns"][0]["reconciles"]


def test_investigation_is_bounded_and_says_when_it_stopped(sales):
    df, profile = sales
    tight = inv.Budget(max_engine_calls=5, max_dimensions=6)
    r = inv.investigate(df, profile, spec(), tight)
    assert r["budget"]["engine_calls"] <= 5 and r["budget"]["exhausted"] == "engine_calls"
    assert any("limit" in u for u in r["unresolved"]) and any("Not examined" in u for u in r["unresolved"])
    one_dim = inv.investigate(df, profile, spec(), inv.Budget(max_dimensions=1))
    assert len(one_dim["dimensions"]) == 1 and any("Not examined" in u for u in one_dim["unresolved"])
    timed = inv.Budget(max_seconds=0.0)
    assert inv.investigate(df, profile, spec(), timed)["budget"]["exhausted"] == "time"
    full = inv.investigate(df, profile, spec())
    assert full["budget"]["exhausted"] == "" and full["budget"]["engine_calls"] <= full["budget"]["limits"]["engine_calls"]


def test_invalid_specs_are_rejected(sales):
    df, profile = sales
    for bad in (spec("sum(nope)"), spec("os.system('x')"), {**spec(), "time_column": "nope"},
                {**spec(), "period_a": ["not a date", SPLIT]}, {**spec(), "filters": [{"column": "nope", "op": "==", "value": 1}]}):
        with pytest.raises(me.FormulaError):
            inv.investigate(df, profile, bad)


def test_question_routed_to_the_investigator_publishes_a_report_and_dashboard(client):
    ws = workspace(client)
    out = ask_and_run(client, ws, "Why did revenue change?")
    assert out["state"] == "succeeded"
    q = question_row(out["question_id"])
    assert q.route == "root_cause" and q.status == "verified"
    dash = client.get(f"/api/questions/{out['question_id']}/dashboard", headers=ws["h"]).json()
    assert dash["verdict_state"] == "VERIFIED_WITH_CAVEATS", "arithmetic verified; the explanation is descriptive"
    assert [k["evidence_status"] for k in dash["kpis"]] == ["verified"] * 3 and len(dash["charts"]) == 2
    assert "does not establish" in dash["narrative"]
    ev = client.get(f"/api/questions/{out['question_id']}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(r for r in ev if r["claim_type"] == "narrative")
    assert narrative["validity"]["causal"] == "not_established" and "descriptive_not_causal" in narrative["limitations"]
    listing = client.get("/api/investigations", headers=ws["h"]).json()
    assert listing["total"] == 1 and listing["investigations"][0]["question_id"] == out["question_id"]
    report = client.get(f"/api/investigations/{listing['investigations'][0]['id']}", headers=ws["h"]).json()["report"]
    assert report["ranked_contributors"][0]["segment"] == "West" and report["provenance"]["dataset_fingerprint"]
    trace = client.get(f"/api/questions/{out['question_id']}/trace", headers=ws["h"]).json()
    assert trace["usage"]["llm_calls"] == 0 and [e["name"] for e in trace["events"] if e["kind"] == "node"] == ["router", "investigate"]


def test_investigation_api_with_explicit_spec_validation_and_tenancy(client):
    ws = workspace(client)
    body = {"dataset_id": ws["ds"], "formula": "sum(revenue) / sum(units)", "label": "Average selling price",
            "time_column": "order_date", "dimensions": ["product"], **PERIODS}
    r = client.post("/api/investigations", json=body, headers={**ws["h"], "Idempotency-Key": "inv-1"})
    assert r.status_code == 201, r.text
    created = r.json()
    again = client.post("/api/investigations", json=body, headers={**ws["h"], "Idempotency-Key": "inv-1"}).json()
    assert again["investigation_id"] == created["investigation_id"] and again["created"] is False
    assert run_job(created["job_id"], {}) == "succeeded"
    got = client.get(f"/api/investigations/{created['investigation_id']}", headers=ws["h"]).json()
    assert got["status"] == "complete" and got["report"]["decomposition_method"] == "rate_and_mix"
    assert [d["dimension"] for d in got["report"]["dimensions"]] == ["product"]
    assert got["report"]["ranked_contributors"][0]["segment"] == "Gadget"

    for bad, code in (({**body, "formula": "sum(nope)"}, 400), ({**body, "formula": "import os"}, 400),
                      ({**body, "time_column": "region"}, 400), ({**body, "dimensions": ["ghost"]}, 400),
                      ({**body, "period_b": None}, 400), ({**body, "dataset_id": 10**9}, 404),
                      ({**body, "filters": [{"column": "region", "op": "nope", "value": 1}]}, 400)):
        assert client.post("/api/investigations", json=bad, headers=ws["h"]).status_code == code
    other = workspace(client)
    assert client.get(f"/api/investigations/{created['investigation_id']}", headers=other["h"]).status_code == 404
    assert client.get("/api/investigations", headers=other["h"]).json()["total"] == 0
    assert client.post("/api/investigations", json=body, headers=other["h"]).status_code == 404
    assert client.get("/api/investigations").status_code == 401


# ========================================================= data quality ====

def test_psi_and_ks_behave_like_drift_metrics():
    rng = np.random.default_rng(7)
    base = list(rng.normal(100, 10, 400))
    same = list(rng.normal(100, 10, 400))
    shifted = list(rng.normal(130, 10, 400))
    assert dq.numeric_psi(base, same) < 0.1 and dq.ks_statistic(base, same)[1] > 0.01
    assert dq.numeric_psi(base, shifted) > 0.25
    d, p = dq.ks_statistic(base, shifted)
    assert d > 0.5 and p < 1e-6
    assert dq.ks_statistic(base, base) == (0.0, 1.0)
    assert dq.psi([0.5, 0.5], [0.5, 0.5]) == 0.0 and dq.psi([0.9, 0.1], [0.1, 0.9]) > 1.0
    assert dq.numeric_psi([5.0] * 200, [5.0] * 200) == 0.0, "a constant column does not crash the binning"
    assert dq.numeric_psi([5.0] * 200, [6.0] * 200) > 1.0
    value, appeared, vanished = dq.categorical_psi({"categories": {"a": 50, "b": 50}, "other": 0},
                                                   {"categories": {"a": 50, "c": 50}, "other": 0})
    assert value > 0.25 and appeared == ["c"] and vanished == ["b"]


def test_comparison_separates_schema_quality_and_distribution(sales):
    df, _ = sales
    base = dq.build_snapshot(df)
    assert base["rows"] == 720 and base["duplicate_rows"] == 0 and base["columns"]["region"]["kind"] == "categorical"
    assert len(base["columns"]["revenue"]["sample"]) <= dq.SAMPLE_POINTS

    assert dq.compare(base, base, dq.default_thresholds()) == {"findings": [], "skipped": []}

    broken = df.copy()
    broken["units"] = broken["units"].astype(object)
    broken.loc[broken.index[:50], "units"] = "n/a"                       # numeric -> text
    broken.loc[broken.index[100:400], "cost"] = None                     # 42% missing
    broken = pd.concat([broken, broken.head(200)], ignore_index=True)    # duplicates
    broken = broken.drop(columns=["channel"])
    found = {(f["category"], f["check"], f["column"]): f for f in dq.compare(dq.build_snapshot(broken), base, dq.default_thresholds())["findings"]}
    assert found[("schema", "column_removed", "channel")]["severity"] == "high"
    assert found[("schema", "type_changed", "units")]["severity"] == "high"
    assert found[("quality", "missing_increase", "cost")]["severity"] == "high"
    assert ("quality", "duplicate_rows", None) not in found, "duplicates are a single-version check"
    dup = dq.single_version_findings(dq.build_snapshot(broken), dq.default_thresholds())
    assert dup and dup[0]["check"] == "duplicate_rows" and dup[0]["severity"] == "high"
    assert all(f["message"] for f in found.values()), "every finding explains itself"


def test_small_samples_are_skipped_not_alarmed(sales):
    df, _ = sales
    base = dq.build_snapshot(df)
    tiny = df.head(40).copy()
    tiny["unit_price"] = tiny["unit_price"] * 50
    result = dq.compare(dq.build_snapshot(tiny), base, dq.default_thresholds())
    assert not [f for f in result["findings"] if f["category"] == "distribution"]
    assert {s["column"] for s in result["skipped"]} >= {"unit_price", "region"}
    assert "fewer than" in result["skipped"][0]["reason"]
    low_bar = dq.merged_thresholds({"min_sample": 10})
    assert [f for f in dq.compare(dq.build_snapshot(tiny), base, low_bar)["findings"] if f["check"] == "numeric_drift"]


def test_numeric_drift_needs_both_signals_and_range_and_staleness_are_reported(sales):
    df, _ = sales
    base = dq.build_snapshot(df)
    scaled = df.copy()
    scaled["revenue"] = scaled["revenue"] * 100
    f = {(x["check"], x["column"]) for x in dq.compare(dq.build_snapshot(scaled), base, dq.default_thresholds())["findings"]}
    assert ("numeric_drift", "revenue") in f and ("range_violation", "revenue") in f
    older = df[df["order_date"] < "2025-04-01"]
    f = {(x["check"], x["column"]) for x in dq.compare(dq.build_snapshot(older), base, dq.default_thresholds())["findings"]}
    assert ("stale_data", "order_date") in f
    half = df.iloc[:200]
    f = [x for x in dq.compare(dq.build_snapshot(half), base, dq.default_thresholds())["findings"] if x["check"] == "row_count_change"]
    assert f and f[0]["severity"] == "high" and "partial load" in f[0]["message"]


def test_freshness_check_is_off_unless_configured(sales):
    import datetime as dt

    df, _ = sales
    snap = dq.build_snapshot(df)
    now = dt.datetime(2025, 6, 1)
    assert dq.single_version_findings(snap, dq.default_thresholds(), now) == []
    stale = dq.single_version_findings(snap, dq.merged_thresholds({"freshness_max_age_days": 7}), now)
    assert stale[0]["check"] == "freshness" and stale[0]["severity"] == "high" and stale[0]["column"] == "order_date"
    assert dq.single_version_findings(snap, dq.merged_thresholds({"freshness_max_age_days": 60}), now) == []


@pytest.mark.parametrize("bad", [{"nope": 1}, {"psi_warn": -1}, {"psi_warn": "x"}, {"psi_warn": 0.5, "psi_high": 0.2},
                                 {"ks_alpha": 0}, {"ks_alpha": 1.5}, {"min_sample": None}, {"psi_warn": float("nan")},
                                 {"duplicate_pct_warn": True}])
def test_threshold_validation(bad):
    with pytest.raises(ValueError):
        dq.validate_thresholds(bad)


def test_upload_creates_a_snapshot_and_a_new_version_raises_incidents(client):
    ws = workspace(client)
    first = client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()
    assert first["latest_version"] == 1 and first["incidents"] == [] and first["history"][0]["has_snapshot"]
    assert first["baseline"]["version"] is None and first["thresholds"]["psi_warn"] == dq.default_thresholds()["psi_warn"]

    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    body = client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()
    assert body["latest_version"] == 2 and body["baseline"] == {"version_id": body["history"][0]["version_id"], "version": 1,
                                                                 "source": "previous version", "pinned": False}
    kinds = {(i["category"], i["check"], i["column"]) for i in body["incidents"]}
    assert {("schema", "column_removed", "channel"), ("schema", "column_added", "discount"),
            ("distribution", "numeric_drift", "unit_price"), ("distribution", "category_drift", "region")} <= kinds
    assert body["open_incidents"] == len(body["incidents"]) and body["history"][1]["incidents"]["schema"] == 2
    assert body["history"][1]["max_drift_psi"] > 0.25 and body["history"][1]["rows"] == 540
    assert all(i["baseline_version_id"] == body["history"][0]["version_id"] for i in body["incidents"])

    before = len(body["incidents"])
    rerun = client.post(f"/api/datasets/{ws['ds']}/quality/run", headers=ws["h"])
    assert rerun.status_code == 200 and rerun.json()["baseline_version"] == 1
    assert len(client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()["incidents"]) == before, "re-running does not duplicate incidents"
    db = SessionLocal()
    try:
        assert db.query(DQSnapshot).filter_by(dataset_id=ws["ds"]).count() == 2
    finally:
        db.close()


def test_incident_lifecycle_configuration_permissions_and_tenancy(client):
    ws = workspace(client)
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    incidents = client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()["incidents"]
    iid = incidents[0]["id"]
    assert client.post(f"/api/quality/incidents/{iid}/acknowledge", headers=ws["h"]).json()["status"] == "acknowledged"
    resolved = client.post(f"/api/quality/incidents/{iid}/resolve", headers=ws["h"]).json()
    assert resolved["status"] == "resolved" and resolved["resolved_at"]
    assert client.post(f"/api/quality/incidents/{iid}/acknowledge", headers=ws["h"]).json()["status"] == "resolved"

    member = add_member(client, ws, "member")
    cfg = {"baseline_version_id": None, "thresholds": {"psi_warn": 0.05, "min_sample": 50}}
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json=cfg, headers=member).status_code == 403
    assert client.get(f"/api/datasets/{ws['ds']}/quality", headers=member).status_code == 200, "members can read"
    ok = client.put(f"/api/datasets/{ws['ds']}/quality/config", json=cfg, headers=ws["h"])
    assert ok.status_code == 200 and ok.json()["thresholds"]["psi_warn"] == 0.05 and ok.json()["threshold_overrides"] == cfg["thresholds"]
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json={"thresholds": {"bogus": 1}}, headers=ws["h"]).status_code == 400

    versions = client.get(f"/api/datasets/{ws['ds']}/versions", headers=ws["h"]).json()
    pinned = client.put(f"/api/datasets/{ws['ds']}/quality/config", json={"baseline_version_id": versions[0]["id"], "thresholds": {}}, headers=ws["h"]).json()
    assert pinned["baseline"]["pinned"] is True and pinned["baseline"]["version"] == 1
    other = workspace(client)
    foreign = client.get(f"/api/datasets/{other['ds']}/versions", headers=other["h"]).json()[0]["id"]
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json={"baseline_version_id": foreign, "thresholds": {}}, headers=ws["h"]).status_code == 404
    for path, method in ((f"/api/datasets/{ws['ds']}/quality", "get"), (f"/api/datasets/{ws['ds']}/quality/run", "post"),
                         (f"/api/quality/incidents/{iid}/resolve", "post"), (f"/api/quality/incidents/{iid}/acknowledge", "post")):
        assert getattr(client, method)(path, headers=other["h"]).status_code == 404, path
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json=cfg, headers=other["h"]).status_code == 404
    assert client.get(f"/api/datasets/{ws['ds']}/quality").status_code == 401


def test_a_broken_quality_check_never_fails_an_upload(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("profiler exploded")

    monkeypatch.setattr(dq, "build_snapshot", boom)
    ws = workspace(client)  # asserts the upload returned 200
    db = SessionLocal()
    try:
        assert db.query(DQSnapshot).filter_by(dataset_id=ws["ds"]).count() == 0
        assert db.query(DQIncident).filter_by(dataset_id=ws["ds"]).count() == 0
    finally:
        db.close()
