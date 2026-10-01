"""Dataset versioning, scheduled re-runs + change alerting, and Critic verdict
states. The LLM pipeline is never invoked: tests seed Question/Dashboard rows
directly and drive the scheduler functions by hand.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest

from app import scheduler
from app.change_detection import diff_kpis, parse_kpi_value, trend_of
from app.database import SessionLocal
from app.models import (
    AuditTrail,
    CriticReview,
    Dashboard,
    DatasetVersion,
    Question,
    ScheduledAnalysis,
    TeamMember,
)
from app.verdict import (
    UNVERIFIED,
    VERIFIED,
    VERIFIED_WITH_CAVEATS,
    compute_verdict_state,
)


def _signup(client, email: str) -> str:
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123", "display_name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _team(client, token: str) -> dict:
    h = {"Authorization": f"Bearer {token}"}
    cid = client.post("/api/communities", json={"name": "Co"}, headers=h).json()["id"]
    tid = client.post(f"/api/communities/{cid}/teams", json={"name": "T"}, headers=h).json()["id"]
    return {**h, "X-Team-Id": str(tid)}


@pytest.fixture
def ws(client, unique_email):
    headers = _team(client, _signup(client, unique_email))
    r = client.post("/api/datasets/upload", files={"file": ("d.csv", b"a,b\n1,2\n", "text/csv")}, headers=headers)
    assert r.status_code == 200, r.text
    return {"headers": headers, "team_id": int(headers["X-Team-Id"]), "dataset": r.json(), "email": unique_email}


def _question(db, ws, **kw) -> Question:
    q = Question(team_id=ws["team_id"], dataset_id=ws["dataset"]["id"], text="Revenue?", status="verified", **kw)
    db.add(q)
    db.commit()
    return q


def _dashboard(db, q, kpis, verified=True, verdict_state=None) -> Dashboard:
    d = Dashboard(
        team_id=q.team_id, question_id=q.id, kpis_json=kpis, charts_json=[], narrative="n",
        verified=verified, verdict_state=verdict_state,
    )
    db.add(d)
    db.commit()
    return d


# ---------------------------------------------------------------- versioning --

def test_replace_data_adds_version_and_keeps_dataset_id(client, ws):
    ds_id = ws["dataset"]["id"]
    assert ws["dataset"]["version"] == 1

    r = client.post(
        f"/api/datasets/{ds_id}/versions",
        files={"file": ("d2.csv", b"a,b\n1,2\n3,4\n5,6\n", "text/csv")},
        headers=ws["headers"],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["id"], body["version"], body["version_count"], body["row_count"]) == (ds_id, 2, 2, 3)

    db = SessionLocal()
    try:
        versions = db.query(DatasetVersion).filter_by(dataset_id=ds_id).order_by(DatasetVersion.version_number).all()
        assert [v.row_count for v in versions] == [1, 3]  # v1 file/profile untouched
    finally:
        db.close()


def test_replace_data_is_team_scoped(client, ws, unique_email):
    other = _team(client, _signup(client, f"other-{unique_email}"))
    r = client.post(
        f"/api/datasets/{ws['dataset']['id']}/versions",
        files={"file": ("x.csv", b"a\n1\n", "text/csv")},
        headers=other,
    )
    assert r.status_code == 404


def test_question_stays_pinned_to_its_version(client, ws):
    ds_id = ws["dataset"]["id"]
    db = SessionLocal()
    try:
        v1 = db.query(DatasetVersion).filter_by(dataset_id=ds_id).one()
        q = _question(db, ws, dataset_version_id=v1.id)
        client.post(f"/api/datasets/{ds_id}/versions",
                    files={"file": ("d2.csv", b"a\n1\n", "text/csv")}, headers=ws["headers"])
        db.refresh(q)
        assert q.dataset_version_id == v1.id
    finally:
        db.close()


# ------------------------------------------------------------ change detection --

@pytest.mark.parametrize("text,expected", [
    ("$1,234.5", 1234.5), ("12.4%", 12.4), ("1.2M", 1.2e6), ("3 k", 3000.0),
    ("12 months", 12.0), ("-8%", -8.0), ("n/a", None), ("", None),
])
def test_parse_kpi_value(text, expected):
    assert parse_kpi_value(text) == expected


def test_diff_kpis_threshold_and_matching():
    old = [{"label": "Revenue", "value": "$100"}, {"label": "Churn", "value": "5%"},
           {"label": "Region", "value": "West"}, {"label": "Gone", "value": "1"}]
    new = [{"label": " revenue ", "value": "$111"}, {"label": "Churn", "value": "5.1%"},
           {"label": "Region", "value": "East"}, {"label": "Brand new", "value": "9"}]
    by = {c.label.strip().lower(): c for c in diff_kpis(old, new, 10)}
    assert by["revenue"].crossed and round(by["revenue"].pct, 1) == 11.0
    assert not by["churn"].crossed  # 2% < 10%
    assert by["region"].crossed and by["region"].pct is None  # text change always counts
    assert "brand new" not in by and "gone" not in by  # unmatched KPIs ignored


def test_diff_kpis_exactly_at_threshold_does_not_cross():
    (c,) = diff_kpis([{"label": "R", "value": "100"}], [{"label": "R", "value": "110"}], 10)
    assert not c.crossed


def test_trend_of():
    up = diff_kpis([{"label": "R", "value": "100"}], [{"label": "R", "value": "120"}], 10)
    down = diff_kpis([{"label": "R", "value": "100"}], [{"label": "R", "value": "80"}], 10)
    flat = diff_kpis([{"label": "R", "value": "100"}], [{"label": "R", "value": "100.1"}], 10)
    assert (trend_of(up), trend_of(down), trend_of(flat), trend_of([])) == ("up", "down", "flat", "flat")


# -------------------------------------------------------------- scheduled API --

def test_scheduled_crud_and_isolation(client, ws, unique_email):
    h = ws["headers"]
    r = client.post("/api/scheduled-analyses",
                    json={"dataset_id": ws["dataset"]["id"], "question": "Revenue?", "interval": "weekly"}, headers=h)
    assert r.status_code == 200, r.text
    sa = r.json()
    assert sa["is_active"] and sa["change_threshold_pct"] == 10.0 and sa["interval"] == "weekly"

    assert client.patch(f"/api/scheduled-analyses/{sa['id']}", json={"is_active": False}, headers=h).json()["is_active"] is False
    assert client.patch(f"/api/scheduled-analyses/{sa['id']}", json={"interval": "hourly"}, headers=h).status_code == 400

    other = _team(client, _signup(client, f"o-{unique_email}"))
    assert client.get("/api/scheduled-analyses", headers=other).json() == []
    assert client.delete(f"/api/scheduled-analyses/{sa['id']}", headers=other).status_code == 404

    assert client.delete(f"/api/scheduled-analyses/{sa['id']}", headers=h).status_code == 200
    assert client.get("/api/scheduled-analyses", headers=h).json() == []


def test_scheduled_rejects_foreign_dataset(client, ws, unique_email):
    other = _team(client, _signup(client, f"o2-{unique_email}"))
    r = client.post("/api/scheduled-analyses",
                    json={"dataset_id": ws["dataset"]["id"], "question": "x"}, headers=other)
    assert r.status_code == 404


# ------------------------------------------------------------------ scheduler --

def _make_sa(db, ws, **kw) -> ScheduledAnalysis:
    sa = ScheduledAnalysis(
        workspace_id=ws["team_id"], dataset_id=ws["dataset"]["id"], question_text="Revenue?",
        interval="daily", created_at=dt.datetime.utcnow() - dt.timedelta(days=2), **kw,
    )
    db.add(sa)
    db.commit()
    return sa


def test_tick_runs_due_analysis_once_against_latest_version(client, ws):
    db = SessionLocal()
    try:
        sa = _make_sa(db, ws)
        client.post(f"/api/datasets/{ws['dataset']['id']}/versions",
                    files={"file": ("d2.csv", b"a\n1\n2\n", "text/csv")}, headers=ws["headers"])
        with patch.object(scheduler, "_run_then_alert"):
            first = scheduler.run_due_analyses()
            second = scheduler.run_due_analyses()  # just claimed -> not due again
        assert len(first) == 1 and second == []
        q = db.get(Question, first[0])
        latest = db.query(DatasetVersion).filter_by(dataset_id=ws["dataset"]["id"]).order_by(
            DatasetVersion.version_number.desc()).first()
        assert (q.trigger, q.scheduled_analysis_id, q.dataset_version_id) == ("scheduled", sa.id, latest.id)
    finally:
        db.close()


def test_tick_skips_paused_and_not_yet_due(client, ws):
    db = SessionLocal()
    try:
        _make_sa(db, ws, is_active=False)
        fresh = _make_sa(db, ws)
        fresh.created_at = dt.datetime.utcnow()
        db.commit()
        with patch.object(scheduler, "_run_then_alert"):
            assert scheduler.run_due_analyses() == []
    finally:
        db.close()


def test_claim_is_exclusive(client, ws):
    db = SessionLocal()
    try:
        sa = _make_sa(db, ws)
        # A second "process" that read the row before the first one claimed it.
        db2 = SessionLocal()
        try:
            stale = db2.get(ScheduledAnalysis, sa.id)
            _ = stale.last_run_at
            now = dt.datetime.utcnow()
            assert scheduler._claim(db, sa, now) is True
            assert scheduler._claim(db2, stale, now) is False
        finally:
            db2.close()
    finally:
        db.close()


# ------------------------------------------------------------------- alerting --

def _run(db, ws, sa, kpis, **dash_kw):
    q = _question(db, ws, trigger="scheduled", scheduled_analysis_id=sa.id)
    d = _dashboard(db, q, kpis, **dash_kw)
    return q, d


def test_alert_only_when_threshold_crossed(client, ws):
    db = SessionLocal()
    try:
        sa = _make_sa(db, ws)
        with patch.object(scheduler, "send_email", return_value=True) as send:
            q1, _ = _run(db, ws, sa, [{"label": "Revenue", "value": "$100"}])
            assert scheduler.process_completed_run(q1.id) == "baseline"

            q2, _ = _run(db, ws, sa, [{"label": "Revenue", "value": "$105"}])
            assert scheduler.process_completed_run(q2.id) == "silent"
            send.assert_not_called()

            q3, d3 = _run(db, ws, sa, [{"label": "Revenue", "value": "$150"}])
            assert scheduler.process_completed_run(q3.id) == "alerted"
            to, subject, text, html = send.call_args.args
            assert ws["email"] in to
            assert "Revenue" in subject and "+42.9%" in subject
            assert f"question={q3.id}" in text

        db.refresh(sa)
        assert sa.last_dashboard_id == d3.id and sa.last_trend == "up"
    finally:
        db.close()


def test_critic_rejection_always_notifies(client, ws):
    db = SessionLocal()
    try:
        sa = _make_sa(db, ws)
        with patch.object(scheduler, "send_email", return_value=True) as send:
            q1, _ = _run(db, ws, sa, [{"label": "Revenue", "value": "$100"}])
            scheduler.process_completed_run(q1.id)

            q2, _ = _run(db, ws, sa, [{"label": "Revenue", "value": "$100"}],
                         verified=False, verdict_state=UNVERIFIED)
            db.add(CriticReview(question_id=q2.id, verdict="rejected", summary="Totals don't reconcile",
                                issues_json=["Revenue overstated"]))
            db.commit()
            assert scheduler.process_completed_run(q2.id) == "alerted"  # no KPI change at all
            subject, text = send.call_args.args[1], send.call_args.args[2]
            assert "Could not verify" in subject
            assert "Totals don't reconcile" in text and "Revenue overstated" in text
    finally:
        db.close()


def test_run_without_dashboard_changes_nothing(client, ws):
    db = SessionLocal()
    try:
        sa = _make_sa(db, ws)
        q = _question(db, ws, trigger="scheduled", scheduled_analysis_id=sa.id)
        q.status = "failed"
        db.commit()
        with patch.object(scheduler, "send_email") as send:
            assert scheduler.process_completed_run(q.id) == "no_dashboard"
            send.assert_not_called()
    finally:
        db.close()


def test_pending_members_are_not_emailed(client, ws):
    db = SessionLocal()
    try:
        db.add(TeamMember(team_id=ws["team_id"], invited_email="pending@test.example", status="pending"))
        db.commit()
        assert ws["email"] in scheduler._recipients(db, ws["team_id"])
        assert "pending@test.example" not in scheduler._recipients(db, ws["team_id"])
    finally:
        db.close()


# -------------------------------------------------------------------- verdicts --

def test_compute_verdict_state():
    assert compute_verdict_state("verified", 0) == VERIFIED
    assert compute_verdict_state("verified", 1) == VERIFIED_WITH_CAVEATS
    assert compute_verdict_state("rejected", 1) == UNVERIFIED
    assert compute_verdict_state(None, 0) == UNVERIFIED


def _flagged_dashboard(db, ws):
    q = _question(db, ws)
    review = CriticReview(question_id=q.id, verdict="rejected", summary="Sums don't match",
                          issues_json=["KPI off by 12%", "Chart double counts"])
    db.add(review)
    db.commit()
    d = _dashboard(
        db, q,
        [{"label": "Revenue", "value": "$1", "element_id": "kpi1"},
         {"label": "Clean", "value": "$2", "element_id": "kpi2"}],
        verified=False, verdict_state=UNVERIFIED,
    )
    db.add(AuditTrail(team_id=q.team_id, question_id=q.id, element_label="KPI: Revenue",
                      element_type="kpi", element_id="kpi1", critic_review_id=review.id))
    db.add(AuditTrail(team_id=q.team_id, question_id=q.id, element_label="Narrative summary",
                      element_type="narrative", critic_review_id=review.id))
    ok = CriticReview(question_id=q.id, verdict="verified", summary="fine")
    db.add(ok)
    db.commit()
    db.add(AuditTrail(team_id=q.team_id, question_id=q.id, element_label="KPI: Clean",
                      element_type="kpi", element_id="kpi2", critic_review_id=ok.id))
    db.commit()
    return q, d


def test_dashboard_api_exposes_verdict_and_per_element_flags(client, ws):
    db = SessionLocal()
    try:
        q, _ = _flagged_dashboard(db, ws)
        qid = q.id
    finally:
        db.close()
    body = client.get(f"/api/questions/{qid}/dashboard", headers=ws["headers"]).json()
    assert body["verdict_state"] == UNVERIFIED
    assert body["flagged_count"] == 2
    assert body["rejections"][0]["summary"] == "Sums don't match"
    assert body["rejections"][0]["issues"] == ["KPI off by 12%", "Chart double counts"]
    assert {k["element_id"]: k["flagged"] for k in body["kpis"]} == {"kpi1": True, "kpi2": False}
    assert body["narrative_flagged"] is True


def test_inspect_returns_critic_reasoning_only_for_flagged_elements(client, ws):
    db = SessionLocal()
    try:
        q, d = _flagged_dashboard(db, ws)
        dash_id = d.id
    finally:
        db.close()
    flagged = client.get(f"/api/dashboards/{dash_id}/elements/kpi1/inspect", headers=ws["headers"]).json()
    assert flagged["flagged"] is True
    assert flagged["critic_reasoning"] == "Sums don't match"
    assert flagged["critic_issues"] == ["KPI off by 12%", "Chart double counts"]
    clean = client.get(f"/api/dashboards/{dash_id}/elements/kpi2/inspect", headers=ws["headers"]).json()
    assert clean["flagged"] is False and clean["critic_reasoning"] is None


def test_legacy_dashboard_without_verdict_state_is_resolved(client, ws):
    db = SessionLocal()
    try:
        q = _question(db, ws)
        _dashboard(db, q, [], verified=True, verdict_state=None)
        q2 = _question(db, ws)
        _dashboard(db, q2, [], verified=False, verdict_state=None)
        q3 = _question(db, ws)
        db.add(CriticReview(question_id=q3.id, verdict="rejected", summary="s"))
        db.commit()
        _dashboard(db, q3, [], verified=True, verdict_state=None)
        ids = (q.id, q2.id, q3.id)
    finally:
        db.close()
    states = [client.get(f"/api/questions/{i}/dashboard", headers=ws["headers"]).json()["verdict_state"] for i in ids]
    assert states == [VERIFIED, UNVERIFIED, VERIFIED_WITH_CAVEATS]


def test_question_list_carries_trigger_and_verdict(client, ws):
    db = SessionLocal()
    try:
        sa = _make_sa(db, ws)
        q = _question(db, ws, trigger="scheduled", scheduled_analysis_id=sa.id)
        _dashboard(db, q, [], verified=True, verdict_state=VERIFIED)
        qid = q.id
    finally:
        db.close()
    rows = client.get("/api/questions", headers=ws["headers"]).json()
    row = next(r for r in rows if r["id"] == qid)
    assert row["trigger"] == "scheduled" and row["verdict_state"] == VERIFIED
