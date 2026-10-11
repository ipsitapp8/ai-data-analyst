"""Monitoring: thresholds and minimum effect, comparison windows, data-quality
gating, deduplication, idempotent processing, alert lifecycle, delivery retry."""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest
from platform_helpers import DRIFTED_CSV, add_member, seed_dashboard, workspace

from app import config, scheduler
from app.database import SessionLocal
from app.models import Alert, Dataset, DQIncident, ScheduledAnalysis


@pytest.fixture
def ws(client):
    return workspace(client)


def make_sa(ws, **kw) -> int:
    db = SessionLocal()
    try:
        sa = ScheduledAnalysis(workspace_id=ws["team"], dataset_id=ws["ds"], question_text="Revenue?", interval="daily",
                               change_threshold_pct=kw.pop("change_threshold_pct", 10.0),
                               created_at=dt.datetime.utcnow() - dt.timedelta(days=60), **kw)
        db.add(sa)
        db.commit()
        return sa.id
    finally:
        db.close()


def run(ws, sa_id, value, label="Revenue", **kw) -> int:
    qid, _ = seed_dashboard(ws, [{"label": label, "value": f"${value}"}], trigger="scheduled",
                            scheduled_analysis_id=sa_id, **kw)
    return qid


def alerts(ws, **filters) -> list[Alert]:
    db = SessionLocal()
    try:
        rows = db.query(Alert).filter_by(team_id=ws["team"], **filters).order_by(Alert.id).all()
        for r in rows:
            db.expunge(r)
        return rows
    finally:
        db.close()


def test_processing_the_same_run_twice_is_a_no_op(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        q = run(ws, sa, 150)
        assert scheduler.process_completed_run(q) == "alerted"
        assert scheduler.process_completed_run(q) == "duplicate"   # e.g. the job was retried
        assert scheduler.process_completed_run(q) == "duplicate"
        assert send.call_count == 1, "a retried run must not send a second email"
    assert len(alerts(ws)) == 1 and alerts(ws)[0].occurrences == 1


def test_an_open_incident_is_updated_not_re_raised_or_re_emailed(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        assert scheduler.process_completed_run(run(ws, sa, 150)) == "alerted"
        q3 = run(ws, sa, 220)
        assert scheduler.process_completed_run(q3) == "deduplicated"
        assert scheduler.process_completed_run(run(ws, sa, 330)) == "deduplicated"
        assert send.call_count == 1
    rows = alerts(ws)
    assert len(rows) == 1
    a = rows[0]
    assert a.occurrences == 3 and a.status == "open" and a.delivery_state == "sent"
    assert a.last_seen_at >= a.created_at and "330" in a.title, "the alert shows the latest state"
    assert a.dedup_key == f"{sa}:change:revenue"


def test_resolving_an_alert_lets_the_same_condition_alert_again(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        scheduler.process_completed_run(run(ws, sa, 150))
        first = alerts(ws)[0]
        ack = client.post(f"/api/alerts/{first.id}/acknowledge", headers=ws["h"]).json()
        assert ack["status"] == "acknowledged" and ack["read"] is True and ack["acknowledged_at"]
        assert scheduler.process_completed_run(run(ws, sa, 230)) == "deduplicated", "acknowledged is still the same incident"
        res = client.post(f"/api/alerts/{first.id}/resolve", headers=ws["h"]).json()
        assert res["status"] == "resolved" and res["resolved_at"]
        assert scheduler.process_completed_run(run(ws, sa, 400)) == "alerted"
        assert send.call_count == 2
    rows = alerts(ws)
    assert [r.status for r in rows] == ["resolved", "open"] and rows[1].occurrences == 1
    listing = client.get("/api/alerts?status=open", headers=ws["h"]).json()["alerts"]
    assert [a["id"] for a in listing] == [rows[1].id]
    assert client.get("/api/alerts?status=bogus", headers=ws["h"]).status_code == 422


def test_minimum_effect_size_suppresses_large_percentages_on_tiny_numbers(client, ws):
    sa = make_sa(ws, min_effect_abs=50.0)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 2))
        assert scheduler.process_completed_run(run(ws, sa, 3)) == "silent", "+50% but only +1 in absolute terms"
        send.assert_not_called()
        assert scheduler.process_completed_run(run(ws, sa, 80)) == "alerted", "+77 clears both bars"
    assert len(alerts(ws)) == 1


def test_same_weekday_comparison_ignores_a_weekly_cycle(client, ws):
    """Weekends are always low. Compared with the previous run that would alert
    every Monday; compared with last Monday it does not."""
    monday = dt.datetime(2025, 3, 3, 9, 0)
    pattern = [1000, 1010, 990, 1005, 995, 300, 310]  # Mon..Sun
    for mode, expected_alerts in (("previous", True), ("same_weekday", False)):
        sa = make_sa(ws, comparison=mode, change_threshold_pct=20.0)
        outcomes = []
        with patch.object(scheduler, "send_email", return_value=True):
            for day in range(14):
                q = run(ws, sa, pattern[day % 7], created_at=monday + dt.timedelta(days=day))
                outcomes.append(scheduler.process_completed_run(q))
        second_week = outcomes[7:]
        assert ("alerted" in second_week or "deduplicated" in second_week) is expected_alerts, (mode, outcomes)
    db = SessionLocal()
    try:
        a = db.query(Alert).filter_by(team_id=ws["team"], kind="change").first()
        assert a.explanation_json["compared_with"] == "the previous run"
    finally:
        db.close()


def test_rolling_mean_comparison_smooths_noise(client, ws):
    sa = make_sa(ws, comparison="rolling_mean", window_runs=4, change_threshold_pct=15.0)
    outcomes = []
    with patch.object(scheduler, "send_email", return_value=True):
        for v in (100, 120, 85, 115, 90, 110):      # noisy but centred on ~100; each step is >15% from the last
            outcomes.append(scheduler.process_completed_run(run(ws, sa, v)))
        assert "alerted" not in outcomes[2:], outcomes
        assert scheduler.process_completed_run(run(ws, sa, 160)) == "alerted"
    a = alerts(ws, kind="change")[-1]
    assert a.explanation_json["compared_with"].startswith("the mean of the last 4")


def test_data_quality_incidents_hold_back_kpi_alerts(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        # New data arrives with a column missing -> an open, high-severity schema incident.
        client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
        q = run(ws, sa, 40)
        assert scheduler.process_completed_run(q) == "suppressed"
        rows = alerts(ws)
        assert [r.kind for r in rows] == ["data_quality"] and rows[0].severity == "high"
        assert "channel" in rows[0].detail and rows[0].explanation_json["data_quality"]
        assert send.call_count == 1 and "Data quality" in send.call_args.args[1]
        assert scheduler.process_completed_run(run(ws, sa, 20)) == "suppressed"
        assert send.call_count == 1 and alerts(ws)[0].occurrences == 2, "one data-quality alert, not one per run"

        db = SessionLocal()
        try:
            for i in db.query(DQIncident).filter_by(dataset_id=ws["ds"], status="open").all():
                i.status = "resolved"
            db.commit()
        finally:
            db.close()
        assert scheduler.process_completed_run(run(ws, sa, 200)) == "alerted", "gate lifts once the data is fixed"
    assert {r.kind for r in alerts(ws)} == {"data_quality", "change"}


def test_data_quality_gate_can_be_switched_off_per_schedule(client, ws):
    sa = make_sa(ws, suppress_on_dq=False)
    with patch.object(scheduler, "send_email", return_value=True):
        scheduler.process_completed_run(run(ws, sa, 100))
        client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
        assert scheduler.process_completed_run(run(ws, sa, 40)) == "alerted"


def test_unverified_runs_alert_even_while_data_quality_is_failing(client, ws):
    from app.models import CriticReview

    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True):
        scheduler.process_completed_run(run(ws, sa, 100))
        client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
        q = run(ws, sa, 100, verified=False, verdict_state="UNVERIFIED")
        db = SessionLocal()
        try:
            db.add(CriticReview(question_id=q, verdict="rejected", summary="Totals do not reconcile", issues_json=[]))
            db.commit()
        finally:
            db.close()
        assert scheduler.process_completed_run(q) == "alerted"
    assert [r.kind for r in alerts(ws)] == ["unverified"]


def test_email_can_be_disabled_while_in_app_alerts_continue(client, ws):
    sa = make_sa(ws, notify_email=False)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        assert scheduler.process_completed_run(run(ws, sa, 200)) == "alerted_in_app"
        send.assert_not_called()
    assert alerts(ws)[0].delivery_state == "skipped"


def test_failed_delivery_is_retried_a_bounded_number_of_times(client, ws, monkeypatch):
    monkeypatch.setattr(config, "ALERT_DELIVERY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "ALERT_DELIVERY_RETRY_SECONDS", 60)
    monkeypatch.setattr(scheduler, "email_configured", lambda: True)
    sa = make_sa(ws)
    now = dt.datetime.utcnow()
    with patch.object(scheduler, "send_email", return_value=False) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        assert scheduler.process_completed_run(run(ws, sa, 200)) == "alert_failed"
        a = alerts(ws)[0]
        assert (a.delivery_state, a.delivery_attempts) == ("failed", 1) and a.delivery_payload_json["subject"]
        assert scheduler.retry_failed_deliveries(now) == {"sent": 0, "failed": 0, "gave_up": 0}, "not due yet"
        assert scheduler.retry_failed_deliveries(now + dt.timedelta(minutes=2))["failed"] == 1
        a = alerts(ws)[0]
        assert a.delivery_attempts == 2 and a.next_delivery_at > now + dt.timedelta(minutes=2), "backs off"
        assert scheduler.retry_failed_deliveries(now + dt.timedelta(hours=1))["failed"] == 1
        assert scheduler.retry_failed_deliveries(now + dt.timedelta(hours=5))["gave_up"] == 1
        calls_after_giving_up = send.call_count
        assert scheduler.retry_failed_deliveries(now + dt.timedelta(days=3)) == {"sent": 0, "failed": 0, "gave_up": 0}
        assert send.call_count == calls_after_giving_up == 3, "initial send + 2 retries, then it stops for good"
    a = alerts(ws)[0]
    assert a.delivery_state == "gave_up" and a.status == "open", "the in-app alert is unaffected by email failure"


def test_retry_succeeds_from_the_stored_payload_without_recomputing(client, ws, monkeypatch):
    monkeypatch.setattr(scheduler, "email_configured", lambda: True)
    monkeypatch.setattr(config, "ALERT_DELIVERY_RETRY_SECONDS", 60)
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=False):
        scheduler.process_completed_run(run(ws, sa, 100))
        scheduler.process_completed_run(run(ws, sa, 200))
    with patch.object(scheduler, "send_email", return_value=True) as send, \
            patch.object(scheduler, "diff_kpis", side_effect=AssertionError("must not recompute")):
        assert scheduler.retry_failed_deliveries(dt.datetime.utcnow() + dt.timedelta(minutes=5))["sent"] == 1
        to, subject, text, _html = send.call_args.args
        assert ws["email"] in to and "Revenue" in subject and "question=" in text
        assert scheduler.retry_failed_deliveries(dt.datetime.utcnow() + dt.timedelta(hours=5))["sent"] == 0
        assert send.call_count == 1
    a = alerts(ws)[0]
    assert a.delivery_state == "sent" and a.delivery_payload_json is None


def test_unconfigured_smtp_is_skipped_not_retried(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=False):
        scheduler.process_completed_run(run(ws, sa, 100))
        scheduler.process_completed_run(run(ws, sa, 200))
    a = alerts(ws)[0]
    assert a.delivery_state == "skipped" and a.next_delivery_at is None
    with patch.object(scheduler, "send_email") as send:
        scheduler.retry_failed_deliveries(dt.datetime.utcnow() + dt.timedelta(days=1))
        send.assert_not_called()


def test_noisy_missing_and_text_kpis_do_not_page_anyone(client, ws):
    sa = make_sa(ws, change_threshold_pct=10.0)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        scheduler.process_completed_run(run(ws, sa, 100))
        for v in (104, 97, 103, 99, 101, 96):                      # ±4% noise
            assert scheduler.process_completed_run(run(ws, sa, v)) == "silent"
        q, _ = seed_dashboard(ws, [{"label": "Something else", "value": "5"}], trigger="scheduled", scheduled_analysis_id=sa)
        assert scheduler.process_completed_run(q) == "silent", "a KPI present in only one run is label drift, not a change"
        q, _ = seed_dashboard(ws, [], trigger="scheduled", scheduled_analysis_id=sa)
        assert scheduler.process_completed_run(q) == "silent", "a run with no KPIs at all"
        q, _ = seed_dashboard(ws, [{"label": "Revenue", "value": "n/a"}], trigger="scheduled", scheduled_analysis_id=sa)
        scheduler.process_completed_run(q)
        send.assert_not_called()
    assert alerts(ws) == []


def test_alert_explanation_carries_history_and_evidence_links(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True):
        for v in (100, 102, 98, 101):
            scheduler.process_completed_run(run(ws, sa, v))
        q = run(ws, sa, 150)
        scheduler.process_completed_run(q)
    a = alerts(ws)[0]
    body = client.get(f"/api/alerts/{a.id}", headers=ws["h"]).json()
    ex = body["explanation"]
    assert ex["compared_with"] == "the previous run" and ex["threshold_pct"] == 10.0 and ex["verdict"] == "VERIFIED"
    assert ex["changes"][0]["label"] == "Revenue" and ex["changes"][0]["crossed"] is True and round(ex["changes"][0]["pct"], 1) == 48.5
    assert ex["history"]["Revenue"]["runs"] == 4 and 99 < ex["history"]["Revenue"]["mean"] < 101
    assert ex["evidence"]["question_id"] == q and ex["evidence"]["dashboard_id"]
    assert body["occurrences"] == 1 and body["severity"] == "warn" and body["delivery_state"] == "sent"


def test_monitoring_configuration_validation_and_permissions(client, ws):
    created = client.post("/api/scheduled-analyses", json={
        "dataset_id": ws["ds"], "question": "Revenue?", "comparison": "same_weekday", "min_effect_abs": 25,
        "window_runs": 6}, headers=ws["h"])
    assert created.status_code == 200, created.text
    sa = created.json()
    assert (sa["comparison"], sa["min_effect_abs"], sa["window_runs"], sa["suppress_on_dq"], sa["notify_email"]) == \
           ("same_weekday", 25, 6, True, True)
    for bad in ({"comparison": "vibes"}, {"min_effect_abs": -1}, {"window_runs": 1}, {"window_runs": 500}):
        assert client.patch(f"/api/scheduled-analyses/{sa['id']}", json=bad, headers=ws["h"]).status_code == 400, bad
        assert client.post("/api/scheduled-analyses", json={"dataset_id": ws["ds"], "question": "q", **bad}, headers=ws["h"]).status_code == 400

    member = add_member(client, ws, "member")
    assert client.patch(f"/api/scheduled-analyses/{sa['id']}", json={"change_threshold_pct": 5}, headers=member).status_code == 200
    for restricted in ({"notify_email": False}, {"suppress_on_dq": False}):
        assert client.patch(f"/api/scheduled-analyses/{sa['id']}", json=restricted, headers=member).status_code == 403
    assert client.patch(f"/api/scheduled-analyses/{sa['id']}", json={"notify_email": False, "suppress_on_dq": False},
                        headers=ws["h"]).json()["notify_email"] is False
    listed = client.get("/api/scheduled-analyses", headers=ws["h"]).json()[0]
    assert listed["suppress_on_dq"] is False and listed["change_threshold_pct"] == 5


def test_alert_lifecycle_endpoints_are_team_scoped(client, ws):
    sa = make_sa(ws)
    with patch.object(scheduler, "send_email", return_value=True):
        scheduler.process_completed_run(run(ws, sa, 100))
        scheduler.process_completed_run(run(ws, sa, 200))
    aid = alerts(ws)[0].id
    other = workspace(client)
    for path, method in ((f"/api/alerts/{aid}", "get"), (f"/api/alerts/{aid}/acknowledge", "post"),
                         (f"/api/alerts/{aid}/resolve", "post"), (f"/api/alerts/{aid}/read", "post")):
        assert getattr(client, method)(path, headers=other["h"]).status_code == 404
        assert getattr(client, method)(path).status_code == 401
    assert alerts(ws)[0].status == "open"
    assert client.get("/api/alerts", headers=other["h"]).json()["alerts"] == []


def test_scheduled_runs_go_through_the_durable_queue(client, ws):
    from app.models import AnalysisJob

    sa = make_sa(ws)
    launched = scheduler.run_due_analyses()
    db = SessionLocal()
    try:
        row = db.get(ScheduledAnalysis, sa)
        qids = [q for q in launched if db.get(AnalysisJob, db.query(AnalysisJob).filter_by(question_id=q).first().id).team_id == ws["team"]]
        assert len(qids) == 1
        job = db.query(AnalysisJob).filter_by(question_id=qids[0]).first()
        assert job.state == "queued" and row.last_run_at is not None
    finally:
        db.close()
    assert not any(q == qids[0] for q in scheduler.run_due_analyses()), "claimed: not launched twice"
