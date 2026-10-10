"""Durable job system: submission, idempotency, admission, claiming, recovery
after interruption, cancellation, deadlines, retry and terminal-state guarantees."""
from __future__ import annotations

import datetime as dt
import threading
import time

import pytest
from platform_helpers import ask, job_row, make_team, question_row, run_job, signup, workspace

from app import config, jobs, runtime
from app.agents.state import mark_terminal, update_stage
from app.database import SessionLocal
from app.models import AnalysisJob, Dashboard, Dataset, Question, RunManifest


@pytest.fixture
def ws(client):
    return workspace(client)


def _ok_runner(status="verified"):
    calls = []

    def runner(question_id: int) -> str:
        calls.append(question_id)
        mark_terminal(question_id, status)
        return status

    runner.calls = calls
    return runner


def _expire_lease(job_id: int, seconds: int = 120) -> None:
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        job.lease_expires_at = dt.datetime.utcnow() - dt.timedelta(seconds=seconds)
        db.commit()
    finally:
        db.close()


def _claim(job_id: int, worker: str = "dead-worker") -> AnalysisJob:
    db = SessionLocal()
    try:
        job = jobs.claim_next(db, worker, job_id)
        assert job is not None
        db.expunge(job)
        return job
    finally:
        db.close()


# ---------------------------------------------------------------- submission --

def test_submission_creates_a_queued_job_and_returns_immediately(client, ws):
    created = ask(client, ws, "Total revenue by region")
    assert created["job_state"] == "queued" and created["created"] is True
    job = job_row(created["job_id"])
    assert (job.state, job.attempt, job.team_id) == ("queued", 0, ws["team"])
    status = client.get(f"/api/questions/{created['question_id']}/status", headers=ws["h"]).json()
    assert status["current_stage"] == "queued" and status["job"]["state"] == "queued"


def test_idempotency_key_returns_the_first_submission(client, ws):
    first = ask(client, ws, "Total revenue", **{"Idempotency-Key": "abc-1"})
    again = ask(client, ws, "Total revenue", **{"Idempotency-Key": "abc-1"})
    different_text = ask(client, ws, "Something else entirely", **{"Idempotency-Key": "abc-1"})
    assert again["question_id"] == first["question_id"] and again["created"] is False
    assert different_text["question_id"] == first["question_id"], "the key, not the payload, identifies the submission"
    other_key = ask(client, ws, "Total revenue", **{"Idempotency-Key": "abc-2"})
    assert other_key["question_id"] != first["question_id"]
    db = SessionLocal()
    try:
        assert db.query(AnalysisJob).filter_by(idempotency_key=f"{ws['team']}:abc-1").count() == 1
    finally:
        db.close()


def test_idempotency_keys_are_scoped_to_the_team(client, ws):
    other = workspace(client)
    a = ask(client, ws, "Total revenue", **{"Idempotency-Key": "shared-key"})
    b = ask(client, other, "Total revenue", **{"Idempotency-Key": "shared-key"})
    assert a["question_id"] != b["question_id"]


def test_concurrent_duplicate_submissions_create_one_job(client, ws):
    results, errors = [], []

    def submit():
        db = SessionLocal()
        try:
            ds = db.get(Dataset, ws["ds"])
            q, job, created = jobs.submit_question(db, team_id=ws["team"], dataset=ds, text="race",
                                                   idempotency_key="race-key", enforce_admission=False)
            results.append((q.id, job.id, created))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            db.close()

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len({r[1] for r in results}) == 1, "eight racing submissions must resolve to one job"
    assert sum(1 for r in results if r[2]) == 1
    db = SessionLocal()
    try:
        assert db.query(Question).filter_by(team_id=ws["team"], text="race").count() == 1
    finally:
        db.close()


def test_admission_limit_per_team(client, ws, monkeypatch):
    monkeypatch.setattr(config, "JOB_TEAM_MAX_ACTIVE", 2)
    ask(client, ws, "q1")
    ask(client, ws, "q2")
    r = client.post("/api/questions", json={"dataset_id": ws["ds"], "question": "q3"}, headers=ws["h"])
    assert r.status_code == 429 and "queued or running" in r.json()["detail"]
    other = workspace(client)
    assert client.post("/api/questions", json={"dataset_id": other["ds"], "question": "q"},
                       headers=other["h"]).status_code == 200, "one team's backlog must not block another"


def test_question_text_is_validated(client, ws):
    for bad in ("", "   ", "x" * 2001):
        r = client.post("/api/questions", json={"dataset_id": ws["ds"], "question": bad}, headers=ws["h"])
        assert r.status_code == 400
    r = client.post("/api/questions", json={"dataset_id": 10**9, "question": "ok"}, headers=ws["h"])
    assert r.status_code == 404


# ------------------------------------------------------------------ claiming --

def test_a_job_can_be_claimed_by_exactly_one_worker(client, ws):
    job_id = ask(client, ws, "claim me")["job_id"]
    winners = []
    barrier = threading.Barrier(6)

    def worker(i):
        db = SessionLocal()
        try:
            barrier.wait()
            if jobs.claim_next(db, f"w{i}", job_id) is not None:
                winners.append(i)
        finally:
            db.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1
    job = job_row(job_id)
    assert job.state == "running" and job.attempt == 1 and job.worker_id == f"w{winners[0]}"
    assert job.deadline_at is not None and job.lease_expires_at > dt.datetime.utcnow()


def test_concurrent_execution_runs_the_work_once(client, ws):
    job_id = ask(client, ws, "run me once")["job_id"]
    runner = _ok_runner()
    states = []
    threads = [threading.Thread(target=lambda: states.append(jobs.run_inline(job_id, runner))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(runner.calls) == 1
    assert sorted(s for s in states if s) == ["succeeded"] and states.count(None) == 4
    assert job_row(job_id).state == "succeeded"


def test_successful_run_records_outcome_manifest_and_usage(client, ws):
    created = ask(client, ws, "Total revenue by region")
    assert run_job(created["job_id"], {}) == "succeeded"
    job = job_row(created["job_id"])
    assert job.finished_at is not None and job.failure_class is None and job.lease_expires_at is None
    assert question_row(created["question_id"]).status == "verified"
    db = SessionLocal()
    try:
        assert db.query(RunManifest).filter_by(question_id=created["question_id"]).count() == 1
    finally:
        db.close()
    assert not (config.WORKSPACES_DIR / f"q{created['question_id']}").exists(), "workspace is cleaned at the end"


def test_a_crashing_run_fails_safely_without_leaking_internals(client, ws):
    created = ask(client, ws, "boom")

    def runner(_):
        raise RuntimeError("/secret/path/to/key.pem: connection string postgres://user:pw@host")

    assert jobs.run_inline(created["job_id"], runner) == "failed"
    job = job_row(created["job_id"])
    assert job.failure_class == "internal"
    assert "secret" not in job.error and "postgres" not in job.error
    q = question_row(created["question_id"])
    assert q.status == "failed" and "postgres" not in (q.error or "")
    body = client.get(f"/api/jobs/{job.id}", headers=ws["h"]).json()
    assert body["state"] == "failed" and "postgres" not in str(body)


def test_analysis_failure_keeps_the_user_facing_reason(client, ws):
    created = ask(client, ws, "fails on its own terms")

    def runner(qid):
        mark_terminal(qid, "failed", "Step 2 failed after 3 attempts.")
        return "failed"

    assert jobs.run_inline(created["job_id"], runner) == "failed"
    job = job_row(created["job_id"])
    assert job.failure_class == "analysis_failed" and "Step 2 failed" in job.error


# ------------------------------------------------------------------ recovery --

def test_interrupted_job_is_requeued_and_then_completes(client, ws):
    created = ask(client, ws, "interrupted")
    _claim(created["job_id"])                      # a worker took it...
    _expire_lease(created["job_id"])               # ...and its process died
    counts = jobs.recover_abandoned()
    assert counts["requeued"] >= 1
    job = job_row(created["job_id"])
    assert job.state == "queued" and job.worker_id is None and job.attempt == 1
    assert question_row(created["question_id"]).current_stage == "queued"
    runner = _ok_runner()
    assert jobs.run_inline(created["job_id"], runner) == "succeeded"
    assert job_row(created["job_id"]).attempt == 2 and len(runner.calls) == 1


def test_job_out_of_attempts_is_failed_not_left_running(client, ws, monkeypatch):
    monkeypatch.setattr(config, "JOB_MAX_ATTEMPTS", 1)
    created = ask(client, ws, "one attempt only")
    _claim(created["job_id"])
    _expire_lease(created["job_id"])
    jobs.recover_abandoned()
    job = job_row(created["job_id"])
    assert (job.state, job.failure_class) == ("failed", "abandoned") and job.finished_at is not None
    q = question_row(created["question_id"])
    assert q.status == "failed" and "could not be resumed" in q.error
    assert jobs.run_inline(created["job_id"], _ok_runner()) is None, "a terminal job is never claimed again"


def test_healthy_running_job_is_not_stolen(client, ws):
    created = ask(client, ws, "still alive")
    _claim(created["job_id"], "alive-worker")
    jobs.recover_abandoned()
    job = job_row(created["job_id"])
    assert job.state == "running" and job.worker_id == "alive-worker"


def test_recovery_does_not_publish_twice(client, ws):
    """The worker died after publishing the dashboard but before recording
    success. Recovery must mark it done, never run it again."""
    created = ask(client, ws, "published then died")
    _claim(created["job_id"])
    db = SessionLocal()
    try:
        db.add(Dashboard(team_id=ws["team"], question_id=created["question_id"], kpis_json=[], charts_json=[],
                         narrative="done", verified=True, verdict_state="VERIFIED"))
        db.commit()
    finally:
        db.close()
    _expire_lease(created["job_id"])
    assert jobs.recover_abandoned()["succeeded"] >= 1
    assert job_row(created["job_id"]).state == "succeeded"
    assert question_row(created["question_id"]).status == "verified"
    db = SessionLocal()
    try:
        assert db.query(Dashboard).filter_by(question_id=created["question_id"]).count() == 1
    finally:
        db.close()


def test_requeued_job_with_a_published_dashboard_is_not_rerun(client, ws):
    created = ask(client, ws, "requeued but already published")
    db = SessionLocal()
    try:
        db.add(Dashboard(team_id=ws["team"], question_id=created["question_id"], kpis_json=[], charts_json=[],
                         narrative="done", verified=False, verdict_state="UNVERIFIED"))
        db.commit()
    finally:
        db.close()
    runner = _ok_runner()
    assert jobs.run_inline(created["job_id"], runner) == "succeeded"
    assert runner.calls == [] and question_row(created["question_id"]).status == "unverified"


def test_a_worker_that_lost_its_lease_cannot_overwrite_the_outcome(client, ws):
    created = ask(client, ws, "two owners")
    _claim(created["job_id"], "old-worker")
    _expire_lease(created["job_id"])
    jobs.recover_abandoned()
    _claim(created["job_id"], "new-worker")
    assert jobs.finish(created["job_id"], "old-worker", "failed", "internal", "stale") is False
    assert job_row(created["job_id"]).state == "running"
    assert jobs.finish(created["job_id"], "new-worker", "succeeded") is True
    assert jobs.finish(created["job_id"], "new-worker", "failed") is False, "a terminal state never changes"
    assert job_row(created["job_id"]).state == "succeeded"


def test_startup_fails_orphan_questions_from_the_old_thread_model(client, ws):
    db = SessionLocal()
    try:
        q = Question(team_id=ws["team"], dataset_id=ws["ds"], text="orphan", status="running", current_stage="executing")
        db.add(q)
        db.commit()
        qid = q.id
    finally:
        db.close()
    queued = ask(client, ws, "has a job")
    assert jobs.fail_orphan_questions() >= 1
    assert question_row(qid).status == "failed"
    assert question_row(queued["question_id"]).status == "running", "a question with a job is not an orphan"


def test_startup_and_shutdown_of_the_worker_pool(client, ws, monkeypatch):
    monkeypatch.setattr(config, "JOB_WORKERS", 1)
    monkeypatch.setattr(config, "JOB_POLL_SECONDS", 0.05)
    created = ask(client, ws, "Total revenue by region")
    jobs.start_workers()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and job_row(created["job_id"]).state not in jobs.TERMINAL:
            time.sleep(0.1)
    finally:
        jobs.stop_workers()
    assert job_row(created["job_id"]).state == "succeeded"
    assert question_row(created["question_id"]).status == "verified"


# -------------------------------------------------- cancellation and deadlines --

def test_cancel_a_queued_job(client, ws):
    created = ask(client, ws, "cancel me")
    r = client.post(f"/api/jobs/{created['job_id']}/cancel", headers=ws["h"])
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    assert question_row(created["question_id"]).status == "cancelled"
    assert jobs.run_inline(created["job_id"], _ok_runner()) is None
    again = client.post(f"/api/questions/{created['question_id']}/cancel", headers=ws["h"])
    assert again.json()["state"] == "cancelled", "cancelling twice is harmless"


def test_cancel_a_running_job_stops_it_at_the_next_checkpoint(client, ws):
    created = ask(client, ws, "long running")
    started = threading.Event()

    def runner(qid):
        started.set()
        for _ in range(600):
            update_stage(qid, "executing", "working")  # every node does this; it is the checkpoint
            time.sleep(0.05)
        return "verified"

    result = []
    t = threading.Thread(target=lambda: result.append(jobs.run_inline(created["job_id"], runner)))
    t.start()
    assert started.wait(10)
    r = client.post(f"/api/questions/{created['question_id']}/cancel", headers=ws["h"])
    assert r.json()["cancel_requested"] is True
    t.join(30)
    assert result == ["cancelled"]
    job = job_row(created["job_id"])
    assert (job.state, job.failure_class) == ("cancelled", "cancelled")
    q = question_row(created["question_id"])
    assert q.status == "cancelled" and q.current_stage == "cancelled"


def test_deadline_ends_the_run_as_timed_out(client, ws, monkeypatch):
    monkeypatch.setattr(config, "JOB_DEADLINE_SECONDS", 1)
    created = ask(client, ws, "too slow")

    def runner(qid):
        for _ in range(200):
            update_stage(qid, "executing", "working")
            time.sleep(0.05)
        return "verified"

    assert jobs.run_inline(created["job_id"], runner) == "timed_out"
    job = job_row(created["job_id"])
    assert job.failure_class == "timed_out" and "time limit" in job.error
    assert question_row(created["question_id"]).status == "failed"


def test_budget_exhaustion_stops_the_run(client, ws, monkeypatch):
    monkeypatch.setattr(config, "RUN_MAX_LLM_CALLS", 2)
    created = ask(client, ws, "talks too much")

    def runner(qid):
        for _ in range(10):
            runtime.checkpoint(about_to="llm")
            runtime.note_llm_call("scripted", "scripted", "t", 1, 1, 1)
        return "verified"

    assert jobs.run_inline(created["job_id"], runner) == "failed"
    job = job_row(created["job_id"])
    assert job.failure_class == "budget_exceeded"
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=ws["h"]).json()
    assert trace["usage"]["llm_calls"] == 2, "a limit of N allows exactly N calls"


def test_overdue_queued_job_times_out(client, ws):
    created = ask(client, ws, "nobody picked me up")
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, created["job_id"])
        job.deadline_at = dt.datetime.utcnow() - dt.timedelta(seconds=5)
        db.commit()
    finally:
        db.close()
    assert jobs.recover_abandoned()["timed_out"] >= 1
    assert job_row(created["job_id"]).state == "timed_out"


# --------------------------------------------------------------------- retry --

def test_retry_requeues_a_failed_job_and_refuses_everything_else(client, ws):
    created = ask(client, ws, "retry me")

    def bad(_):
        raise RuntimeError("x")

    jobs.run_inline(created["job_id"], bad)
    r = client.post(f"/api/jobs/{created['job_id']}/retry", headers=ws["h"])
    assert r.status_code == 200 and r.json()["state"] == "queued" and r.json()["attempt"] == 0
    assert question_row(created["question_id"]).status == "running"
    assert client.post(f"/api/jobs/{created['job_id']}/retry", headers=ws["h"]).status_code == 409, "queued: nothing to retry"
    assert jobs.run_inline(created["job_id"], _ok_runner()) == "succeeded"
    assert client.post(f"/api/jobs/{created['job_id']}/retry", headers=ws["h"]).status_code == 409, "succeeded: never re-run"


def test_retry_is_refused_once_a_dashboard_exists(client, ws):
    created = ask(client, ws, "Total revenue by region")
    run_job(created["job_id"], {})
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, created["job_id"])
        job.state = "failed"  # even if someone forces the state...
        db.commit()
        assert jobs.retry(db, job) is False  # ...a published run is not run again
    finally:
        db.close()


# ----------------------------------------------------------- API and tenancy --

def test_job_endpoints_require_auth_and_are_team_scoped(client, ws):
    created = ask(client, ws, "private")
    other = workspace(client)
    token, _ = signup(client)
    outsider = make_team(client, token)
    for headers in (other["h"], outsider):
        assert client.get(f"/api/jobs/{created['job_id']}", headers=headers).status_code == 404
        assert client.post(f"/api/jobs/{created['job_id']}/cancel", headers=headers).status_code == 404
        assert client.post(f"/api/jobs/{created['job_id']}/retry", headers=headers).status_code == 404
        assert client.get(f"/api/questions/{created['question_id']}/job", headers=headers).status_code == 404
        assert client.get(f"/api/questions/{created['question_id']}/trace", headers=headers).status_code == 404
        assert client.post(f"/api/questions/{created['question_id']}/cancel", headers=headers).status_code == 404
        assert all(j["id"] != created["job_id"] for j in client.get("/api/jobs", headers=headers).json()["jobs"])
    assert job_row(created["job_id"]).state == "queued", "another team's cancel must not have landed"
    assert client.get(f"/api/jobs/{created['job_id']}").status_code == 401
    assert client.get("/api/jobs/metrics").status_code == 401


def test_job_listing_pagination_filter_and_metrics(client, ws):
    ids = [ask(client, ws, f"q{i}")["job_id"] for i in range(3)]
    jobs.run_inline(ids[0], _ok_runner())
    page = client.get("/api/jobs?limit=2&offset=0", headers=ws["h"]).json()
    assert page["total"] == 3 and len(page["jobs"]) == 2 and page["jobs"][0]["id"] == ids[2]
    active = client.get("/api/jobs?state=active", headers=ws["h"]).json()
    assert {j["id"] for j in active["jobs"]} == set(ids[1:])
    assert client.get("/api/jobs?state=bogus", headers=ws["h"]).status_code == 422
    m = client.get("/api/jobs/metrics", headers=ws["h"]).json()
    assert m["queue_depth"] == 2 and m["by_state"]["succeeded"] == 1 and m["team_limit"] == config.JOB_TEAM_MAX_ACTIVE


def test_health_reports_sandbox_and_queue(client):
    body = client.get("/api/health").json()
    assert body["sandbox"]["backend"] == "subprocess" and body["sandbox"]["is_security_boundary"] is False
    assert "queue_depth" in body["jobs"] and body["app_version"]


def test_no_model_provider_fails_with_a_clear_reason_not_an_internal_error(client, ws):
    """Keys are blank in the test environment, so a question that needs the
    model-driven pipeline has no provider to call."""
    created = ask(client, ws, "Summarise what this dataset says about the business")
    assert run_job(created["job_id"]) == "failed"
    job = job_row(created["job_id"])
    assert job.failure_class == "provider_unavailable" and "No AI model could be reached" in job.error
    q = question_row(created["question_id"])
    assert q.status == "failed" and "API key" in q.error and "internal error" not in q.error.lower()
    assert "GEMINI_API_KEY" not in q.error, "configuration details stay in the server log"
