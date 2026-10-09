"""Share links, auto-insights, anomaly alerts + alert feed, and dashboard chat.
No LLM is ever called: tests seed rows directly and patch the model calls."""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest

from app import chat, insights, scheduler
from app.change_detection import detect_anomalies
from app.database import SessionLocal
from app.models import Alert, CriticReview, Dashboard, Question, ScheduledAnalysis, ShareLink
from app.verdict import UNVERIFIED


def _signup(client, email):
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123", "display_name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _team(client, token, name="T"):
    h = {"Authorization": f"Bearer {token}"}
    cid = client.post("/api/communities", json={"name": name}, headers=h).json()["id"]
    tid = client.post(f"/api/communities/{cid}/teams", json={"name": name}, headers=h).json()["id"]
    return {**h, "X-Team-Id": str(tid)}


@pytest.fixture
def ws(client, unique_email):
    headers = _team(client, _signup(client, unique_email))
    r = client.post("/api/datasets/upload", files={"file": ("d.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
                    headers=headers)
    assert r.status_code == 200, r.text
    return {"h": headers, "team": int(headers["X-Team-Id"]), "ds": r.json()["id"], "email": unique_email}


@pytest.fixture
def other(client, unique_email):
    return _team(client, _signup(client, f"other-{unique_email}"), "Other")


def _dash(ws, kpis=None, verified=True, verdict_state=None, **qkw):
    db = SessionLocal()
    try:
        q = Question(team_id=ws["team"], dataset_id=ws["ds"], text="Revenue by region?", status="verified", **qkw)
        db.add(q)
        db.commit()
        d = Dashboard(team_id=ws["team"], question_id=q.id,
                      kpis_json=kpis if kpis is not None else [{"label": "Revenue", "value": "$100", "element_id": "k1"}],
                      charts_json=[{"title": "Chart", "plotly_json": {"data": []}, "element_id": "c1"}],
                      narrative="Revenue grew.", verified=verified, verdict_state=verdict_state)
        db.add(d)
        db.commit()
        return q.id, d.id
    finally:
        db.close()


# ----------------------------------------------------------------- share links --

def test_share_roundtrip_hides_internals_and_revoke_works(client, ws):
    qid, _ = _dash(ws)
    r = client.post(f"/api/questions/{qid}/share", json={"expires_in_days": 3}, headers=ws["h"])
    assert r.status_code == 201, r.text
    link = r.json()
    assert link["url"].endswith(f"/Shared?t={link['token']}") and link["active"]

    pub = client.get(f"/api/public/shared/{link['token']}")  # no auth header
    assert pub.status_code == 200
    body = pub.json()
    assert body["question_text"] == "Revenue by region?" and body["narrative"] == "Revenue grew."
    assert body["kpis"] == [{"label": "Revenue", "value": "$100", "flagged": False}]
    flat = str(body)
    assert "element_id" not in flat and "team" not in flat and "k1" not in flat

    assert [l["id"] for l in client.get(f"/api/questions/{qid}/shares", headers=ws["h"]).json()] == [link["id"]]
    assert client.delete(f"/api/shares/{link['id']}", headers=ws["h"]).status_code == 204
    assert client.get(f"/api/public/shared/{link['token']}").status_code == 404
    assert client.get(f"/api/questions/{qid}/shares", headers=ws["h"]).json() == []


def test_expired_and_unknown_tokens_look_identical(client, ws):
    qid, _ = _dash(ws)
    link = client.post(f"/api/questions/{qid}/share", json={}, headers=ws["h"]).json()
    db = SessionLocal()
    try:
        row = db.get(ShareLink, link["id"])
        row.expires_at = dt.datetime.utcnow() - dt.timedelta(minutes=1)
        db.commit()
    finally:
        db.close()
    expired = client.get(f"/api/public/shared/{link['token']}")
    unknown = client.get("/api/public/shared/not-a-real-token")
    assert expired.status_code == unknown.status_code == 404
    assert expired.json() == unknown.json()


def test_unverified_cannot_be_shared_and_is_rechecked_on_read(client, ws):
    bad_q, _ = _dash(ws, verified=False, verdict_state=UNVERIFIED)
    assert client.post(f"/api/questions/{bad_q}/share", json={}, headers=ws["h"]).status_code == 409

    qid, did = _dash(ws, verified=True, verdict_state="VERIFIED")
    link = client.post(f"/api/questions/{qid}/share", json={}, headers=ws["h"]).json()
    db = SessionLocal()
    try:
        db.get(Dashboard, did).verdict_state = UNVERIFIED
        db.commit()
    finally:
        db.close()
    assert client.get(f"/api/public/shared/{link['token']}").status_code == 404


def test_share_validation_and_team_scoping(client, ws, other):
    qid, _ = _dash(ws)
    assert client.post(f"/api/questions/{qid}/share", json={"expires_in_days": 0}, headers=ws["h"]).status_code == 422
    assert client.post(f"/api/questions/{qid}/share", json={"expires_in_days": 91}, headers=ws["h"]).status_code == 422
    link = client.post(f"/api/questions/{qid}/share", json={}, headers=ws["h"]).json()
    assert client.post(f"/api/questions/{qid}/share", json={}, headers=other).status_code == 404
    assert client.get(f"/api/questions/{qid}/shares", headers=other).status_code == 404
    assert client.delete(f"/api/shares/{link['id']}", headers=other).status_code == 404
    assert client.get(f"/api/public/shared/{link['token']}").status_code == 200  # still alive


# -------------------------------------------------------------- auto-insights --

def test_data_warnings_are_deterministic():
    profile = {"row_count": 10, "columns": [
        {"name": "age", "kind": "numeric", "missing_pct": 60.0, "unique_count": 5,
         "stats": {"min": 1.0, "max": 1000.0, "mean": 5.0, "std": 2.0}},
        {"name": "const", "kind": "categorical", "missing_pct": 0, "unique_count": 1},
        {"name": "ok", "kind": "numeric", "missing_pct": 0, "unique_count": 9,
         "stats": {"min": 1.0, "max": 9.0, "mean": 5.0, "std": 2.5}},
    ]}
    w = insights.data_warnings(profile)
    assert [x["severity"] for x in w] == sorted([x["severity"] for x in w], key={"high": 0, "warn": 1, "info": 2}.get)
    msgs = " ".join(x["message"] for x in w)
    assert "Only 10 rows" in msgs and "60% of values are missing" in msgs
    assert "identical" in msgs and "Extreme values" in msgs
    assert not [x for x in w if x["column"] == "ok"]


def test_insights_generate_cache_and_scope(client, ws, other):
    ds = ws["ds"]
    assert client.get(f"/api/datasets/{ds}/insights", headers=ws["h"]).json()["generated"] is False
    fake = {"questions": [{"question": "  Which b is highest? ", "why": "Find leaders"}, {"question": "  "}]}
    with patch.object(insights.llm_client, "call_tool", return_value=fake):
        r = client.post(f"/api/datasets/{ds}/insights", headers=ws["h"])
    assert r.status_code == 200
    body = r.json()
    assert body["questions"] == [{"question": "Which b is highest?", "why": "Find leaders"}]
    assert any("Only 2 rows" in w["message"] for w in body["warnings"])
    assert client.get(f"/api/datasets/{ds}/insights", headers=ws["h"]).json()["questions"] == body["questions"]
    assert client.get(f"/api/datasets/{ds}/insights", headers=other).status_code == 404
    assert client.post(f"/api/datasets/{ds}/insights", headers=other).status_code == 404


def test_insights_survive_llm_failure(client, ws):
    with patch.object(insights.llm_client, "call_tool", side_effect=RuntimeError("quota")):
        r = client.post(f"/api/datasets/{ws['ds']}/insights", headers=ws["h"])
    assert r.status_code == 200
    assert r.json()["questions"] == [] and r.json()["warnings"]


# ------------------------------------------------------------ anomaly + alerts --

def _hist(*vals):
    return [[{"label": "Revenue", "value": f"${v}"}] for v in vals]


def test_detect_anomalies():
    steady = _hist(100, 102, 98, 101, 99, 100)
    assert detect_anomalies(steady, [{"label": "Revenue", "value": "$101"}]) == []
    hit = detect_anomalies(steady, [{"label": "Revenue", "value": "$150"}])
    assert len(hit) == 1 and hit[0].z > 3 and "above" in hit[0].describe()
    # too little history never fires, and a normally volatile KPI tolerates big moves
    assert detect_anomalies(_hist(100, 100), [{"label": "Revenue", "value": "$900"}]) == []
    assert detect_anomalies(_hist(10, 200, 30, 180, 20, 190), [{"label": "Revenue", "value": "$150"}]) == []
    # perfectly flat history: only a >5% move counts
    flat = _hist(100, 100, 100, 100, 100)
    assert detect_anomalies(flat, [{"label": "Revenue", "value": "$103"}]) == []
    assert detect_anomalies(flat, [{"label": "Revenue", "value": "$120"}])[0].z is None
    # non-numeric KPIs are ignored
    assert detect_anomalies(_hist(1, 2, 3, 4, 5), [{"label": "Revenue", "value": "n/a"}]) == []


def _sa(ws):
    db = SessionLocal()
    try:
        sa = ScheduledAnalysis(workspace_id=ws["team"], dataset_id=ws["ds"], question_text="Revenue?",
                               interval="daily", change_threshold_pct=50.0,
                               is_active=False,  # paused: the shared test DB tick must not pick it up
                               created_at=dt.datetime.utcnow() - dt.timedelta(days=30))
        db.add(sa)
        db.commit()
        return sa.id
    finally:
        db.close()


def _run(ws, sa_id, value, verified=True, verdict_state=None):
    q, _ = _dash(ws, kpis=[{"label": "Revenue", "value": f"${value}"}], verified=verified,
                 verdict_state=verdict_state, trigger="scheduled", scheduled_analysis_id=sa_id)
    return q


def test_anomaly_raises_alert_without_crossing_threshold(client, ws):
    sa_id = _sa(ws)
    with patch.object(scheduler, "send_email", return_value=True) as send:
        for v in (100, 102, 98, 101, 99, 100):
            scheduler.process_completed_run(_run(ws, sa_id, v))
        send.assert_not_called()
        # +30% is under the 50% threshold but ~15 sigma from history
        q = _run(ws, sa_id, 130)
        assert scheduler.process_completed_run(q) == "alerted"
        assert "Unusual" in send.call_args.args[1] and "Unusual compared with history" in send.call_args.args[2]

    feed = client.get("/api/alerts", headers=ws["h"]).json()
    assert feed["unread"] == 1
    a = feed["alerts"][0]
    assert a["kind"] == "anomaly" and a["question_id"] == q and not a["read"]


def test_unverified_and_change_alerts_recorded_even_if_email_fails(client, ws):
    sa_id = _sa(ws)
    with patch.object(scheduler, "send_email", return_value=False):
        scheduler.process_completed_run(_run(ws, sa_id, 100))
        q2 = _run(ws, sa_id, 200, verified=False, verdict_state=UNVERIFIED)
        db = SessionLocal()
        try:
            db.add(CriticReview(question_id=q2, verdict="rejected", summary="Does not reconcile", issues_json=["x"]))
            db.commit()
        finally:
            db.close()
        assert scheduler.process_completed_run(q2) == "alert_failed"
    kinds = {a["kind"] for a in client.get("/api/alerts", headers=ws["h"]).json()["alerts"]}
    assert kinds == {"unverified", "change"}


def test_alert_read_state_and_team_scoping(client, ws, other):
    db = SessionLocal()
    try:
        ids = []
        for i in range(2):
            a = Alert(team_id=ws["team"], kind="change", title=f"t{i}", detail="")
            db.add(a)
            db.commit()
            ids.append(a.id)
    finally:
        db.close()
    assert client.get("/api/alerts", headers=ws["h"]).json()["unread"] == 2
    assert client.get("/api/alerts", headers=other).json() == {"unread": 0, "alerts": []}
    assert client.post(f"/api/alerts/{ids[0]}/read", headers=other).status_code == 404
    assert client.post(f"/api/alerts/{ids[0]}/read", headers=ws["h"]).status_code == 204
    only = client.get("/api/alerts?unread_only=true", headers=ws["h"]).json()
    assert only["unread"] == 1 and [a["id"] for a in only["alerts"]] == [ids[1]]
    assert client.post("/api/alerts/read-all", headers=ws["h"]).status_code == 204
    assert client.get("/api/alerts", headers=ws["h"]).json()["unread"] == 0


# ------------------------------------------------------------------------ chat --

def _patch_chat(answer, check=None, check_error=None):
    ans = patch.object(chat.llm_client, "call_tool", return_value=answer)
    if check_error:
        chk = patch.object(chat, "_critic_call_tool", side_effect=check_error)
    else:
        chk = patch.object(chat, "_critic_call_tool", return_value=(check, "llama:test"))
    return ans, chk


def _ask(client, ws, qid, answer, check=None, check_error=None, **body):
    a, c = _patch_chat(answer, check, check_error)
    with a, c:
        return client.post(f"/api/questions/{qid}/chat", json={"message": "why?", **body}, headers=ws["h"])


GOOD = {"answer": "Revenue is $100.", "sources": ["KPI: Revenue"], "needs_new_analysis": False,
        "suggested_question": ""}


def test_chat_grounded_answer_is_verified(client, ws):
    qid, _ = _dash(ws)
    r = _ask(client, ws, qid, GOOD, {"supported": True, "unsupported_claims": []},
             history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Revenue is $100." and body["verified"] is True
    assert body["checked_by"] == "llama:test" and body["unsupported_claims"] == []


def test_chat_flags_unsupported_claims(client, ws):
    qid, _ = _dash(ws)
    body = _ask(client, ws, qid, GOOD, {"supported": True, "unsupported_claims": ["Revenue is up 40%"]}).json()
    assert body["verified"] is False and body["unsupported_claims"] == ["Revenue is up 40%"]


def test_chat_needs_new_analysis_skips_check_and_failed_check_is_unlabelled(client, ws):
    qid, _ = _dash(ws)
    no_ctx = {**GOOD, "answer": "Not in this analysis.", "needs_new_analysis": True,
              "suggested_question": "What is revenue by month?"}
    a, c = _patch_chat(no_ctx, {"supported": True, "unsupported_claims": []})
    with a, c as check_mock:
        body = client.post(f"/api/questions/{qid}/chat", json={"message": "by month?"}, headers=ws["h"]).json()
        check_mock.assert_not_called()
    assert body["needs_new_analysis"] and body["suggested_question"] == "What is revenue by month?"
    assert body["verified"] is None

    body = _ask(client, ws, qid, GOOD, check_error=RuntimeError("down")).json()
    assert body["answer"] and body["verified"] is None and body["checked_by"] is None


def test_chat_validation_scoping_and_provider_failure(client, ws, other):
    qid, _ = _dash(ws)
    url = f"/api/questions/{qid}/chat"
    assert client.post(url, json={"message": ""}, headers=ws["h"]).status_code == 422
    assert client.post(url, json={"message": "x" * (chat.MAX_MESSAGE_CHARS + 1)}, headers=ws["h"]).status_code == 422
    assert client.post(url, json={"message": "hi", "history": [{"role": "system", "content": "x"}]},
                       headers=ws["h"]).status_code == 422
    assert client.post(url, json={"message": "hi"}, headers=other).status_code == 404
    with patch.object(chat.llm_client, "call_tool", side_effect=RuntimeError("quota")):
        assert client.post(url, json={"message": "hi"}, headers=ws["h"]).status_code == 502


def test_chat_context_includes_dashboard_and_notes(client, ws):
    qid, _ = _dash(ws)
    client.post(f"/api/datasets/{ws['ds']}/notes", json={"kind": "knowledge", "text": "amounts are in USD"},
                headers=ws["h"])
    db = SessionLocal()
    try:
        q = db.get(Question, qid)
        d = db.query(Dashboard).filter_by(question_id=qid).one()
        ctx = chat.build_context(db, q, d)
    finally:
        db.close()
    assert "Revenue: $100" in ctx and "Revenue grew." in ctx and "amounts are in USD" in ctx
    assert "Verification verdict" in ctx


# ---- correlations ---------------------------------------------------------

def test_top_correlations_finds_real_link_and_skips_noise():
    import pandas as pd
    from app.correlations import top_correlations
    df = pd.DataFrame({"x": range(30), "y": [2 * i + 1 for i in range(30)],
                       "z": [(i * 7) % 5 for i in range(30)], "const": [1] * 30})
    pairs = top_correlations(df)
    assert pairs and pairs[0]["a"] == "x" and pairs[0]["b"] == "y"
    assert pairs[0]["direction"] == "positive" and pairs[0]["strength"] == "very strong"
    assert all("const" not in (p["a"], p["b"]) for p in pairs)


def test_top_correlations_needs_enough_rows():
    import pandas as pd
    from app.correlations import top_correlations
    assert top_correlations(pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})) == []


def test_correlations_endpoint_scoped_to_team(client, ws, other):
    r = client.get(f"/api/datasets/{ws['ds']}/correlations", headers=ws["h"])
    assert r.status_code == 200 and "pairs" in r.json()
    assert client.get(f"/api/datasets/{ws['ds']}/correlations", headers=other).status_code == 404
