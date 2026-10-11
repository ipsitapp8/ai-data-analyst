"""Reproducible execution records: fingerprints, immutable manifests, pinned
reruns, run comparison with attribution, dataset version comparison."""
from __future__ import annotations

import json
import copy

import pytest
from platform_helpers import DRIFTED_CSV, SALES_CSV, ask_and_run, job_row, run_job, workspace

from app import provenance
from app.database import SessionLocal
from app.models import Dashboard, DatasetVersion, EvidenceRecord, ExecutionLog, Question, RunManifest

SUITE = json.loads((provenance.config.BACKEND_DIR / "evals" / "cases" / "suite.json").read_text())
PIPELINE = next(c for c in SUITE["cases"] if c["id"] == "pipeline-correct")


@pytest.fixture
def ws(client):
    return workspace(client)


def _manifest(client, ws, qid):
    r = client.get(f"/api/questions/{qid}/manifest", headers=ws["h"])
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_records_a_content_fingerprint(client, ws):
    versions = client.get(f"/api/datasets/{ws['ds']}/versions", headers=ws["h"]).json()
    assert len(versions) == 1 and versions[0]["content_sha256"] == provenance.text_sha256("") != "" or True
    import hashlib

    assert versions[0]["content_sha256"] == hashlib.sha256(SALES_CSV).hexdigest()


def test_manifest_captures_inputs_versions_and_outcome(client, ws):
    out = ask_and_run(client, ws, PIPELINE["question"], PIPELINE["script"])
    m = _manifest(client, ws, out["question_id"])
    assert m["reconstructed"] is False and m["recorded_at"]
    assert m["question"]["normalized"] == provenance.normalize_question(PIPELINE["question"])
    assert len(m["dataset"]["content_sha256"]) == 64 and m["dataset"]["version_number"] == 1
    assert len(m["plans"]) == 1 and len(m["executions"]) == 2
    assert all(len(e["code_sha256"]) == 64 and e["success"] for e in m["executions"])
    assert m["versions"]["prompts"] == provenance.prompt_version() and m["versions"]["app"]
    assert m["environment"]["python"] and m["environment"]["pandas"] and m["environment"]["sandbox_backend"] == "subprocess"
    assert "scripted:scripted" in m["models"] and m["models"]["scripted:scripted"]["calls"] == 7
    assert m["usage"]["llm_calls"] == 7 and m["usage"]["sandbox_runs"] == 3
    assert m["outcome"]["status"] == "verified" and m["outcome"]["job_state"] == "succeeded"
    assert m["outcome"]["kpis"][0]["label"] == "Total revenue"
    assert "not deterministic" in m["reproducibility_note"]


def test_manifest_is_written_once_and_never_rewritten(client, ws):
    out = ask_and_run(client, ws, "Total revenue by region")
    before = _manifest(client, ws, out["question_id"])
    provenance.write_manifest(out["question_id"], {"llm_calls": 999})  # a second writer changes nothing
    db = SessionLocal()
    try:
        assert db.query(RunManifest).filter_by(question_id=out["question_id"]).count() == 1
    finally:
        db.close()
    assert _manifest(client, ws, out["question_id"]) == before


def test_identical_inputs_compare_as_no_difference(client, ws):
    a = ask_and_run(client, ws, "Total revenue by region")
    b = ask_and_run(client, ws, "Total revenue by region")
    cmp = client.get(f"/api/questions/{a['question_id']}/compare/{b['question_id']}", headers=ws["h"]).json()
    assert cmp["results_differ"] is False and cmp["attribution"] == "no_difference"
    assert not any(cmp["changed"].values()), cmp["changed"]
    assert "not deterministic" in cmp["note"], "even identical records carry the reproducibility caveat"


def test_changed_data_is_attributed_to_data(client, ws):
    a = ask_and_run(client, ws, "What is the total revenue?")
    r = client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    assert r.json()["version"] == 2
    b = ask_and_run(client, ws, "What is the total revenue?")
    cmp = client.get(f"/api/questions/{a['question_id']}/compare/{b['question_id']}", headers=ws["h"]).json()
    assert cmp["changed"]["data"] is True and cmp["changed"]["code"] is False and cmp["changed"]["prompts"] is False
    assert cmp["results_differ"] is True and cmp["attribution"] == "data"
    assert cmp["kpi_changes"] and cmp["kpi_changes"][0]["pct"] is not None


def test_rerun_is_pinned_to_the_original_dataset_version(client, ws):
    a = ask_and_run(client, ws, "What is the total revenue?")
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    original = SessionLocal()
    try:
        before = {
            "question": {c.name: getattr(original.get(Question, a["question_id"]), c.name) for c in Question.__table__.columns},
            "dashboards": original.query(Dashboard).filter_by(question_id=a["question_id"]).count(),
            "evidence": original.query(EvidenceRecord).filter_by(question_id=a["question_id"]).count(),
            "logs": original.query(ExecutionLog).filter_by(question_id=a["question_id"]).count(),
        }
    finally:
        original.close()

    r = client.post(f"/api/questions/{a['question_id']}/rerun", headers=ws["h"])
    assert r.status_code == 201, r.text
    rerun = r.json()
    assert rerun["rerun_of_question_id"] == a["question_id"] and rerun["question_id"] != a["question_id"]
    assert run_job(rerun["job_id"], {}) == "succeeded"
    ma, mb = _manifest(client, ws, a["question_id"]), _manifest(client, ws, rerun["question_id"])
    assert mb["dataset"]["version_number"] == 1 == ma["dataset"]["version_number"], "pinned, not latest (which is 2)"
    assert mb["dataset"]["content_sha256"] == ma["dataset"]["content_sha256"]
    assert mb["question"]["rerun_of_question_id"] == a["question_id"]
    cmp = client.get(f"/api/questions/{a['question_id']}/compare/{rerun['question_id']}", headers=ws["h"]).json()
    assert cmp["attribution"] == "no_difference"

    db = SessionLocal()
    try:
        after = {
            "question": {c.name: getattr(db.get(Question, a["question_id"]), c.name) for c in Question.__table__.columns},
            "dashboards": db.query(Dashboard).filter_by(question_id=a["question_id"]).count(),
            "evidence": db.query(EvidenceRecord).filter_by(question_id=a["question_id"]).count(),
            "logs": db.query(ExecutionLog).filter_by(question_id=a["question_id"]).count(),
        }
    finally:
        db.close()
    assert after == before, "a rerun must not touch the original run's records"


def test_rerun_against_an_explicit_version_and_validation(client, ws):
    a = ask_and_run(client, ws, "What is the total revenue?")
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    versions = client.get(f"/api/datasets/{ws['ds']}/versions", headers=ws["h"]).json()
    r = client.post(f"/api/questions/{a['question_id']}/rerun", json={"dataset_version_id": versions[1]["id"]}, headers=ws["h"])
    assert r.status_code == 201 and r.json()["dataset_version_id"] == versions[1]["id"]
    other = workspace(client)
    foreign = client.get(f"/api/datasets/{other['ds']}/versions", headers=other["h"]).json()[0]["id"]
    bad = client.post(f"/api/questions/{a['question_id']}/rerun", json={"dataset_version_id": foreign}, headers=ws["h"])
    assert bad.status_code == 400, "a version of another dataset (another team's) is refused"
    assert client.post(f"/api/questions/{a['question_id']}/rerun", headers=other["h"]).status_code == 404


def test_rerun_is_idempotent_with_a_key(client, ws):
    a = ask_and_run(client, ws, "What is the total revenue?")
    h = {**ws["h"], "Idempotency-Key": "rerun-1"}
    first = client.post(f"/api/questions/{a['question_id']}/rerun", headers=h).json()
    second = client.post(f"/api/questions/{a['question_id']}/rerun", headers=h).json()
    assert first["question_id"] == second["question_id"] and second["created"] is False


def test_changed_prompts_are_reported_as_an_execution_change(client, ws, monkeypatch):
    a = ask_and_run(client, ws, "What is the total revenue?")
    provenance.prompt_version.cache_clear()
    from app.agents import prompts

    monkeypatch.setattr(prompts, "PLANNER_SYSTEM", prompts.PLANNER_SYSTEM + "\nAlways double-check totals.")
    try:
        b = ask_and_run(client, ws, "What is the total revenue?")
    finally:
        provenance.prompt_version.cache_clear()
    cmp = client.get(f"/api/questions/{a['question_id']}/compare/{b['question_id']}", headers=ws["h"]).json()
    assert cmp["changed"]["prompts"] is True and cmp["changed"]["data"] is False
    assert cmp["attribution"] == "no_difference", "inputs changed but the results did not: nothing to attribute"


def test_changed_model_output_on_identical_data_is_attributed_to_execution(client, ws):
    a = ask_and_run(client, ws, PIPELINE["question"], PIPELINE["script"])
    other_script = copy.deepcopy(PIPELINE["script"])
    wrong = next(c for c in SUITE["cases"] if c["id"] == "pipeline-wrong-calculation-is-not-verified")["script"]
    other_script["submit_code"] = wrong["submit_code"]
    other_script["compile_dashboard"] = wrong["compile_dashboard"]
    b = ask_and_run(client, ws, PIPELINE["question"], other_script)
    cmp = client.get(f"/api/questions/{a['question_id']}/compare/{b['question_id']}", headers=ws["h"]).json()
    assert cmp["changed"]["data"] is False and cmp["changed"]["code"] is True
    assert cmp["results_differ"] is True and cmp["attribution"] == "execution_or_model"
    assert "does not explain" in cmp["explanation"]


def test_different_questions_are_not_compared_as_like_for_like(client, ws):
    a = ask_and_run(client, ws, "What is the total revenue?")
    b = ask_and_run(client, ws, "What is the total cost?")
    cmp = client.get(f"/api/questions/{a['question_id']}/compare/{b['question_id']}", headers=ws["h"]).json()
    assert cmp["attribution"] == "different_question"


def test_compare_and_manifest_are_team_scoped(client, ws):
    a = ask_and_run(client, ws, "What is the total revenue?")
    other = workspace(client)
    b = ask_and_run(client, other, "What is the total revenue?")
    assert client.get(f"/api/questions/{a['question_id']}/compare/{b['question_id']}", headers=ws["h"]).status_code == 404
    assert client.get(f"/api/questions/{a['question_id']}/manifest", headers=other["h"]).status_code == 404
    assert client.get(f"/api/datasets/{ws['ds']}/versions", headers=other["h"]).status_code == 404


def test_dataset_version_comparison_summarises_changes(client, ws):
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    v = client.get(f"/api/datasets/{ws['ds']}/versions", headers=ws["h"]).json()
    cmp = client.get(f"/api/datasets/{ws['ds']}/versions/compare", params={"a": v[0]["id"], "b": v[1]["id"]}, headers=ws["h"]).json()
    assert cmp["identical_content"] is False and cmp["row_change"] == 540 - 720
    cols = {c["column"]: c for c in cmp["columns"]}
    assert cols["channel"]["in_a"] and not cols["channel"]["in_b"] and cols["discount"]["in_b"] and not cols["discount"]["in_a"]
    assert cols["unit_price"]["mean_b"] > cols["unit_price"]["mean_a"]
    assert {(f["category"], f["check"]) for f in cmp["findings"]} >= {("schema", "column_removed"), ("distribution", "numeric_drift")}
    same = client.get(f"/api/datasets/{ws['ds']}/versions/compare", params={"a": v[0]["id"], "b": v[0]["id"]}, headers=ws["h"]).json()
    assert same["identical_content"] is True and same["findings"] == []
    other = workspace(client)
    foreign = client.get(f"/api/datasets/{other['ds']}/versions", headers=other["h"]).json()[0]["id"]
    assert client.get(f"/api/datasets/{ws['ds']}/versions/compare", params={"a": v[0]["id"], "b": foreign},
                      headers=ws["h"]).status_code == 404


def test_legacy_runs_get_a_reconstructed_manifest_and_lazy_fingerprint(client, ws):
    db = SessionLocal()
    try:
        version = db.query(DatasetVersion).filter_by(dataset_id=ws["ds"]).first()
        version.content_sha256 = None  # as if uploaded before fingerprints existed
        q = Question(team_id=ws["team"], dataset_id=ws["ds"], dataset_version_id=version.id, text="old run", status="verified")
        db.add(q)
        db.commit()
        qid = q.id
    finally:
        db.close()
    m = _manifest(client, ws, qid)
    assert m["reconstructed"] is True and m["recorded_at"] is None
    assert len(m["dataset"]["content_sha256"]) == 64, "fingerprint computed on first use"
    db = SessionLocal()
    try:
        assert db.query(RunManifest).filter_by(question_id=qid).count() == 0, "a reconstruction is not stored as the record"
    finally:
        db.close()
