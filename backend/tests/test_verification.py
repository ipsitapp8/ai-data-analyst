"""Evidence-backed verification: deterministic checks, status rules, and the
evidence records exposed through the dashboard, evidence and inspect endpoints."""
from __future__ import annotations

import pytest
from platform_helpers import ask_and_run, workspace

from app import verification as v
from app.database import SessionLocal
from app.models import EvidenceRecord

DATASET = {"version_id": 1, "fingerprint": "a" * 64, "fingerprint_now": "a" * 64, "row_count": 500}
CRITIC_OK = {"ran": True, "success": True, "verdict": "verified", "result": {"recomputed_total": 1234.5}}
STEPS = {0: {"result": {"total": 1234.5, "share": 0.25, "top": "West"}, "execution_log_id": 7, "formula": "sum"}}


def kpi(label="Total revenue", value="1,234.50", step=0):
    return {"label": label, "value": value, "source_step_index": step, "element_id": "e1"}


def outcome(record, name):
    return next(c["outcome"] for c in record["checks"] if c["check"] == name)


# ------------------------------------------------------------------ numbers --

@pytest.mark.parametrize("display,candidate,ok", [
    ("1,234.50", 1234.5, True), ("$1,234.5", 1234.52, True), ("1,235", 1234.6, True), ("1,235", 1236.9, False),
    ("12.4%", 0.124, True), ("12.4%", 12.41, True), ("12.4%", 0.2, False), ("1.2M", 1_204_000, True),
    ("1.2M", 1_300_000, False), ("3 k", 3000, True), ("-5.0", -5.02, True), ("42", 42.0, True),
    ("no number here", 1.0, False),
])
def test_number_matching_respects_displayed_precision(display, candidate, ok):
    assert v.number_matches(display, [candidate])[0] is ok


def test_extract_numbers_walks_nested_results_and_ignores_booleans():
    obj = {"a": 1, "b": [2.5, {"c": "3,000", "d": True, "e": None, "f": "text"}], "g": float("nan")}
    assert sorted(v.extract_numbers(obj)) == [1.0, 2.5, 3000.0]


def test_narrative_number_extraction_skips_years_ordinals_and_question_echo():
    text = "In 2024 the top 3 regions made 1,234.50 (12.4%) across 250 orders; last 30 days."
    assert v.narrative_numbers(text, ignore_text="What happened in the last 30 days?") == ["1,234.50", "12.4%", "250"]


# ------------------------------------------------------------- status rules --

def test_all_required_checks_passing_is_verified():
    r = v.verify_kpi(kpi(), STEPS, CRITIC_OK, DATASET)
    assert r["status"] == "verified" and r["reasons"] == []
    assert {c["check"] for c in r["checks"]} >= {"result_present", "value_in_result", "independent_recomputation",
                                                 "provenance", "invariants"}
    assert r["validity"] == {"mathematical": "established", "statistical": "not_assessed", "causal": "not_applicable"}


def test_a_missing_check_never_counts_as_passed():
    for critic in (None, {"ran": False}, {"ran": True, "success": False, "verdict": "verified", "result": None},
                   {"ran": True, "success": True, "verdict": "verified", "result": {"other": 99}}):
        r = v.verify_kpi(kpi(), STEPS, critic, DATASET)
        assert outcome(r, "independent_recomputation") == "not_run"
        assert r["status"] == "unverified"
        assert any(x.startswith("not_run:independent_recomputation") for x in r["reasons"])
    no_fp = v.verify_kpi(kpi(), STEPS, CRITIC_OK, {**DATASET, "fingerprint": None})
    assert outcome(no_fp, "provenance") == "not_run" and no_fp["status"] == "unverified"


def test_a_failed_required_check_is_never_verified():
    cases = {
        "value_in_result": v.verify_kpi(kpi(value="9,999"), STEPS, CRITIC_OK, DATASET),
        "result_present": v.verify_kpi(kpi(step=5), STEPS, CRITIC_OK, DATASET),
        "independent_recomputation": v.verify_kpi(kpi(), STEPS, {**CRITIC_OK, "verdict": "rejected"}, DATASET),
        "provenance": v.verify_kpi(kpi(), STEPS, CRITIC_OK, {**DATASET, "fingerprint_now": "b" * 64}),
        "invariants": v.verify_kpi(kpi("Share of revenue", "250%"), {0: {"result": {"s": 2.5}, "execution_log_id": 1}},
                                   {**CRITIC_OK, "result": {"s": 2.5}}, DATASET),
    }
    for name, record in cases.items():
        assert outcome(record, name) == "fail", name
        assert record["status"] == "unverified", name
        assert any(x.startswith(f"failed:{name}") for x in record["reasons"]), name


def test_malformed_and_empty_results_are_failures():
    for result in (None, {}, [], {"_unparseable_result": "{oops"}):
        r = v.verify_kpi(kpi(), {0: {"result": result, "execution_log_id": 1}}, CRITIC_OK, DATASET)
        assert outcome(r, "result_present") == "fail" and r["status"] == "unverified"
    no_log = v.verify_kpi(kpi(), {0: {"result": {"total": 1234.5}, "execution_log_id": None}}, CRITIC_OK, DATASET)
    assert outcome(no_log, "provenance") == "fail"


def test_warnings_and_limitations_give_verified_with_caveats():
    steps = {0: {"result": {"x": 1}, "execution_log_id": 1}, 1: {"result": {"total": 1234.5}, "execution_log_id": 2}}
    wrong_step = v.verify_kpi(kpi(step=0), steps, CRITIC_OK, DATASET)
    assert outcome(wrong_step, "value_in_result") == "warn" and wrong_step["status"] == "verified_with_caveats"
    small = v.verify_kpi(kpi(), STEPS, CRITIC_OK, {**DATASET, "row_count": 12})
    assert small["status"] == "verified_with_caveats" and "small_sample" in small["limitations"]
    assert small["validity"]["statistical"].startswith("weak") and small["validity"]["mathematical"] == "established_with_caveats"


def test_text_kpis_are_checked_by_presence_not_arithmetic():
    ok = v.verify_kpi(kpi("Top region", "West"), STEPS, {**CRITIC_OK, "result": {"top": "west"}}, DATASET)
    assert ok["status"] == "verified"
    bad = v.verify_kpi(kpi("Top region", "North"), STEPS, CRITIC_OK, DATASET)
    assert outcome(bad, "value_in_result") == "fail"


def test_structured_recomputation_compares_two_implementations():
    agree = v.verify_kpi(kpi(), STEPS, None, DATASET, {"formula": "sum(x)", "engine_value": 1234.5, "reference_value": 1234.5})
    assert agree["status"] == "verified" and agree["metric"]["formula"] == "sum(x)" and agree["recomputed_value"]
    differ = v.verify_kpi(kpi(), STEPS, None, DATASET, {"formula": "sum(x)", "engine_value": 1234.5, "reference_value": 1200.0})
    assert outcome(differ, "independent_recomputation") == "fail"
    na = v.verify_kpi(kpi(), STEPS, None, DATASET, {"formula": "sum(x)", "engine_value": 1234.5, "reference_value": None,
                                                    "error": "ReferenceNotApplicable"})
    assert outcome(na, "independent_recomputation") == "not_run" and na["status"] == "unverified"


# ---------------------------------------------------------------- narrative --

def test_narrative_numbers_must_be_supported_by_results():
    kpis = [kpi()]
    good = v.verify_narrative("Revenue was 1,234.50 with a 25% share.", "n", kpis, STEPS, CRITIC_OK, DATASET)
    assert good["status"] == "verified"
    bad = v.verify_narrative("Revenue was 1,234.50, up 37.2% year on year.", "n", kpis, STEPS, CRITIC_OK, DATASET)
    assert outcome(bad, "narrative_numbers_supported") == "fail" and "37.2%" in bad["checks"][0]["detail"]
    none = v.verify_narrative("Revenue was healthy.", "n", kpis, STEPS, CRITIC_OK, DATASET)
    assert outcome(none, "narrative_numbers_supported") == "not_applicable" and none["status"] == "verified"


def test_causal_wording_is_flagged_as_not_established():
    r = v.verify_narrative("Revenue fell to 1,234.50 because of the price increase.", "n", [kpi()], STEPS, CRITIC_OK, DATASET)
    assert outcome(r, "causal_language") == "warn" and r["status"] == "verified_with_caveats"
    assert r["validity"]["causal"] == "not_established" and r["validity"]["mathematical"] == "established_with_caveats"
    plain = v.verify_narrative("Revenue was 1,234.50.", "n", [kpi()], STEPS, CRITIC_OK, DATASET)
    assert plain["validity"]["causal"] == "no_causal_claim"


def test_narrative_without_review_is_unverified_and_rejection_fails():
    assert v.verify_narrative("x", "n", [], STEPS, None, DATASET)["status"] == "unverified"
    rejected = v.verify_narrative("x", "n", [], STEPS, {**CRITIC_OK, "verdict": "rejected"}, DATASET)
    assert outcome(rejected, "independent_review") == "fail"


# ------------------------------------------------------------------ verdict --

def test_dashboard_verdict_combines_review_and_evidence():
    ok = [v.verify_kpi(kpi(), STEPS, CRITIC_OK, DATASET)]
    unconfirmed = [v.verify_kpi(kpi(), STEPS, None, DATASET)]
    failed = [v.verify_kpi(kpi(value="9,999"), STEPS, CRITIC_OK, DATASET)]
    assert v.combine_verdict("VERIFIED", ok) == "VERIFIED"
    assert v.combine_verdict("VERIFIED", unconfirmed) == "VERIFIED_WITH_CAVEATS"
    assert v.combine_verdict("VERIFIED", failed) == "UNVERIFIED", "a failed check overrides an approving reviewer"
    assert v.combine_verdict("VERIFIED_WITH_CAVEATS", ok) == "VERIFIED_WITH_CAVEATS"
    assert v.combine_verdict("UNVERIFIED", ok) == "UNVERIFIED", "passing checks never upgrade a rejection"
    assert "value_in_result" in v.failure_feedback(failed)


# -------------------------------------------------------------- integration --

@pytest.fixture
def ran(client):
    ws = workspace(client)
    out = ask_and_run(client, ws, "Total revenue by region")
    assert out["state"] == "succeeded"
    return ws, out["question_id"]


def test_dashboard_carries_evidence_status_per_element(client, ran):
    ws, qid = ran
    dash = client.get(f"/api/questions/{qid}/dashboard", headers=ws["h"]).json()
    assert dash["verdict_state"] == "VERIFIED" and dash["verified"] is True and dash["route"] == "fast"
    assert {k["evidence_status"] for k in dash["kpis"]} == {"verified"}
    assert dash["charts"][0]["evidence_status"] == "verified" and dash["narrative_evidence_status"] == "verified"
    assert dash["evidence_summary"]["claims"] == 5 and dash["evidence_summary"]["verified"] == 5
    assert len(dash["dataset_fingerprint"]) == 64


def test_evidence_endpoint_returns_full_records(client, ran):
    ws, qid = ran
    body = client.get(f"/api/questions/{qid}/evidence", headers=ws["h"]).json()
    assert body["summary"]["claims"] == 5 and body["summary"]["failed_checks"] == 0
    kpi_rec = next(r for r in body["records"] if r["claim_type"] == "kpi")
    assert kpi_rec["metric"]["formula"] == 'sum("revenue")' and kpi_rec["metric"]["group_by"] == ["region"]
    assert kpi_rec["dataset_fingerprint"] and kpi_rec["execution_log_id"] and kpi_rec["recomputed_value"]
    assert {c["check"] for c in kpi_rec["checks"]} >= {"independent_recomputation", "provenance", "value_in_result"}
    assert {"mathematical", "statistical", "causal"} == set(kpi_rec["validity"])
    assert len({r["claim_id"] for r in body["records"]}) == 5


def test_inspect_includes_the_evidence_record(client, ran):
    ws, qid = ran
    dash = client.get(f"/api/questions/{qid}/dashboard", headers=ws["h"]).json()
    el = dash["kpis"][0]["element_id"]
    body = client.get(f"/api/dashboards/{dash['id']}/elements/{el}/inspect", headers=ws["h"]).json()
    assert body["evidence"]["status"] == "verified" and body["evidence"]["element_id"] == el
    assert "sum" in body["formula_explanation"] and body["data_slice"]["columns"] == ["region", "revenue"]
    assert body["csv_lines"][0] == 2, "data-slice rows still map back to CSV lines"


def test_evidence_is_team_scoped_and_requires_auth(client, ran):
    ws, qid = ran
    other = workspace(client)
    assert client.get(f"/api/questions/{qid}/evidence", headers=other["h"]).status_code == 404
    assert client.get(f"/api/questions/{qid}/evidence").status_code == 401
    dash = client.get(f"/api/questions/{qid}/dashboard", headers=ws["h"]).json()
    assert client.get(f"/api/dashboards/{dash['id']}/elements/{dash['kpis'][0]['element_id']}/inspect",
                      headers=other["h"]).status_code == 404


def test_tampered_dataset_file_fails_provenance(client):
    """If the file behind a version changes after upload, results computed from
    it are not verified."""
    ws = workspace(client)
    from app.dataset_versions import latest_version
    from app.models import Dataset

    db = SessionLocal()
    try:
        path = latest_version(db, db.get(Dataset, ws["ds"])).filepath
    finally:
        db.close()
    with open(path, "a", encoding="utf-8") as f:
        f.write("2025-05-01,East,Gadget,Online,1,20.0,20.0,12.0\n")
    out = ask_and_run(client, ws, "What is the total revenue?")
    dash = client.get(f"/api/questions/{out['question_id']}/dashboard", headers=ws["h"]).json()
    assert dash["verdict_state"] == "UNVERIFIED" and dash["verified"] is False
    db = SessionLocal()
    try:
        recs = db.query(EvidenceRecord).filter_by(question_id=out["question_id"], claim_type="kpi").all()
        assert all(any(c["check"] == "provenance" and c["reason_code"] == "dataset_changed" for c in r.checks_json)
                   for r in recs)
    finally:
        db.close()
    assert "provenance" in dash["verification_summary"]
