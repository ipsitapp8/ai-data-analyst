"""Scheduled re-runs: an in-process APScheduler tick finds due
scheduled_analyses, runs each through the normal pipeline exactly as a manual
question would, then diffs the result and emails the team only if it matters.

In-process on purpose (same model as the background-thread question runs): no
new infrastructure. The catch is that every API worker process would run its own
tick, so a due analysis is *claimed* with a conditional UPDATE first -- only the
process that wins the claim runs it.
"""
from __future__ import annotations

import datetime as dt
import html as _html
import logging
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import update
from sqlalchemy.orm import Session

from app import config
from app.change_detection import Anomaly, KpiChange, detect_anomalies, diff_kpis, trend_of
from app.database import SessionLocal
from app.dataset_versions import latest_version
from app.email_sender import send_email
from app.models import Alert, Dashboard, Dataset, Question, ScheduledAnalysis, TeamMember, User
from app.verdict import UNVERIFIED, rejected_reviews, resolve_verdict_state

logger = logging.getLogger(__name__)

INTERVALS = {"daily": dt.timedelta(days=1), "weekly": dt.timedelta(days=7)}

_scheduler: BackgroundScheduler | None = None


def next_run_at(sa: ScheduledAnalysis) -> dt.datetime:
    return (sa.last_run_at or sa.created_at) + INTERVALS[sa.interval]


def _has_run_in_flight(db: Session, sa_id: int) -> bool:
    return (
        db.query(Question.id)
        .filter(Question.scheduled_analysis_id == sa_id, Question.status == "running")
        .first()
        is not None
    )


def _claim(db: Session, sa: ScheduledAnalysis, now: dt.datetime) -> bool:
    """Atomically move last_run_at to `now`; False if another tick got there first."""
    cond = (
        ScheduledAnalysis.last_run_at.is_(None)
        if sa.last_run_at is None
        else ScheduledAnalysis.last_run_at == sa.last_run_at
    )
    result = db.execute(
        update(ScheduledAnalysis)
        .where(ScheduledAnalysis.id == sa.id, ScheduledAnalysis.is_active.is_(True), cond)
        .values(last_run_at=now)
    )
    db.commit()
    return result.rowcount == 1


def launch_scheduled_run(db: Session, sa: ScheduledAnalysis, *, start_thread: bool = True) -> int | None:
    """Create the Question for one scheduled run (pinned to the dataset's latest
    version right now) and start the normal pipeline on a background thread."""
    dataset = db.get(Dataset, sa.dataset_id)
    if dataset is None:
        logger.warning("Scheduled analysis %s: dataset %s is gone; skipping", sa.id, sa.dataset_id)
        return None
    version = latest_version(db, dataset)
    question = Question(
        team_id=sa.workspace_id,
        dataset_id=sa.dataset_id,
        dataset_version_id=version.id if version else None,
        text=sa.question_text,
        status="running",
        current_stage="queued",
        stage_detail="Scheduled run",
        trigger="scheduled",
        scheduled_analysis_id=sa.id,
    )
    db.add(question)
    db.commit()
    db.refresh(question)

    if start_thread:
        threading.Thread(
            target=_run_then_alert, args=(question.id,), daemon=True, name=f"scheduled-{sa.id}"
        ).start()
    return question.id


def _run_then_alert(question_id: int) -> None:
    from app.agents.graph import run_question_graph

    run_question_graph(question_id)
    try:
        process_completed_run(question_id)
    except Exception:  # noqa: BLE001 - alerting must never crash the worker thread silently
        logger.exception("Post-run alerting failed for question %s", question_id)


def run_due_analyses(now: dt.datetime | None = None) -> list[int]:
    """One scheduler tick. Returns the question ids it launched."""
    now = now or dt.datetime.utcnow()
    launched: list[int] = []
    db = SessionLocal()
    try:
        for sa in db.query(ScheduledAnalysis).filter(ScheduledAnalysis.is_active.is_(True)).all():
            if next_run_at(sa) > now or _has_run_in_flight(db, sa.id):
                continue
            if not _claim(db, sa, now):
                continue
            qid = launch_scheduled_run(db, sa)
            if qid:
                launched.append(qid)
                logger.info("Scheduled analysis %s launched as question %s", sa.id, qid)
    finally:
        db.close()
    return launched


# ------------------------------------------------------------ alerting --

def _recipients(db: Session, team_id: int) -> list[str]:
    rows = (
        db.query(User.email)
        .join(TeamMember, TeamMember.user_id == User.id)
        .filter(TeamMember.team_id == team_id, TeamMember.status == "active")
        .all()
    )
    return sorted({r[0] for r in rows})


def _build_email(question: Question, verdict: str, changes: list[KpiChange], rejection_text: str,
                anomalies: list[Anomaly] | None = None):
    link = f"{config.APP_URL.rstrip('/')}/Dashboards?question={question.id}"
    crossed = [c for c in changes if c.crossed]
    q_short = question.text if len(question.text) <= 70 else question.text[:67] + "..."

    if verdict == UNVERIFIED:
        subject = f"[DataSage] Could not verify tracked analysis: {q_short}"
        headline = "The Critic could not confirm this scheduled analysis."
    elif crossed:
        first = crossed[0]
        more = f" (+{len(crossed) - 1} more)" if len(crossed) > 1 else ""
        subject = f"[DataSage] {first.describe()}{more}"
        headline = f"A tracked metric changed by more than your threshold: {q_short}"
    else:
        subject = f"[DataSage] Unusual value: {anomalies[0].label}"
        headline = f"A tracked metric is out of line with its history: {q_short}"

    lines = [headline, ""]
    if rejection_text:
        lines += ["Critic's reasoning:", rejection_text, ""]
    if changes:
        lines += ["Changes since the last run:"] + [f"- {c.describe()}" for c in changes] + [""]
    if anomalies:
        lines += ["Unusual compared with history:"] + [f"- {a.describe()}" for a in anomalies] + [""]
    lines.append(f"View the new dashboard: {link}")
    text_body = "\n".join(lines)

    esc = _html.escape
    html_body = f"<div style='font-family:sans-serif;color:#1a1a1a;'><p><b>{esc(headline)}</b></p>"
    if rejection_text:
        html_body += f"<p style='color:#b42318;'><b>Critic's reasoning:</b><br>{esc(rejection_text)}</p>"
    if changes:
        html_body += "<ul>" + "".join(f"<li>{esc(c.describe())}</li>" for c in changes) + "</ul>"
    if anomalies:
        html_body += "<p><b>Unusual compared with history:</b></p><ul>" + "".join(
            f"<li>{esc(a.describe())}</li>" for a in anomalies) + "</ul>"
    html_body += f"<p><a href='{esc(link)}'>View the new dashboard</a></p></div>"
    return subject, text_body, html_body


def _kpi_history(db: Session, sa_id: int, exclude_question_id: int, limit: int = 20) -> list[list[dict]]:
    """KPI lists of this schedule's earlier runs, oldest first."""
    qids = [
        r[0] for r in db.query(Question.id)
        .filter(Question.scheduled_analysis_id == sa_id, Question.id != exclude_question_id)
        .order_by(Question.id.desc()).limit(limit).all()
    ]
    history = []
    for qid in reversed(qids):
        d = (db.query(Dashboard).filter(Dashboard.question_id == qid)
             .order_by(Dashboard.created_at.desc()).first())
        if d is not None:
            history.append(d.kpis_json or [])
    return history


def _record_alerts(db: Session, sa: ScheduledAnalysis, question: Question, verdict: str,
                   changes: list[KpiChange], anomalies: list[Anomaly], rejection_text: str) -> None:
    """Persist the in-app alert(s) for this run. Independent of email: a team with
    no SMTP configured still sees them on the Scheduled page."""
    def add(kind: str, title: str, detail: str) -> None:
        db.add(Alert(team_id=sa.workspace_id, scheduled_analysis_id=sa.id, question_id=question.id,
                     kind=kind, title=title[:200], detail=detail))

    q_short = question.text if len(question.text) <= 60 else question.text[:57] + "..."
    if verdict == UNVERIFIED:
        add("unverified", f"Could not verify: {q_short}", rejection_text or "The Critic could not confirm this run.")
    crossed = [c for c in changes if c.crossed]
    if crossed:
        more = f" (+{len(crossed) - 1} more)" if len(crossed) > 1 else ""
        add("change", f"{crossed[0].describe()}{more}", "\n".join(c.describe() for c in changes))
    for a in anomalies:
        add("anomaly", f"Unusual {a.label}: {a.value}", a.describe())
    db.commit()


def process_completed_run(question_id: int) -> str:
    """Diff a finished scheduled run against the previous one and notify if
    warranted. Returns what happened: 'no_dashboard', 'baseline', 'silent',
    'alerted' or 'alert_failed' (the last two mean an email was warranted)."""
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        if question is None or question.scheduled_analysis_id is None:
            return "no_dashboard"
        sa = db.get(ScheduledAnalysis, question.scheduled_analysis_id)
        dash = (
            db.query(Dashboard)
            .filter(Dashboard.question_id == question_id)
            .order_by(Dashboard.created_at.desc())
            .first()
        )
        if sa is None or dash is None:
            return "no_dashboard"

        rejections = rejected_reviews(db, question_id)
        verdict = resolve_verdict_state(dash, rejections)
        previous = db.get(Dashboard, sa.last_dashboard_id) if sa.last_dashboard_id else None

        changes = diff_kpis(previous.kpis_json, dash.kpis_json, sa.change_threshold_pct) if previous else []
        crossed = any(c.crossed for c in changes)
        anomalies = detect_anomalies(_kpi_history(db, sa.id, question_id), dash.kpis_json)

        sa.last_dashboard_id = dash.id
        sa.last_trend = trend_of(changes) if previous else None
        sa.last_change_summary = "; ".join(c.describe() for c in changes[:3])
        db.commit()

        # A verification failure is itself the signal, so it notifies regardless
        # of threshold (and even on a first run with nothing to diff against).
        if verdict != UNVERIFIED and not crossed and not anomalies:
            logger.info("Scheduled analysis %s run %s: no KPI crossed %.1f%%; no email",
                        sa.id, question_id, sa.change_threshold_pct)
            return "baseline" if previous is None else "silent"

        rejection_text = ""
        if verdict == UNVERIFIED and rejections:
            last = rejections[-1]
            rejection_text = " ".join([last.summary or ""] + [f"[{i}]" for i in (last.issues_json or [])]).strip()
        try:
            _record_alerts(db, sa, question, verdict, changes, anomalies, rejection_text)
        except Exception:  # noqa: BLE001 - the in-app feed must never block the email
            db.rollback()
            logger.exception("Could not record in-app alerts for question %s", question_id)
        subject, text_body, html_body = _build_email(question, verdict, changes, rejection_text, anomalies)
        ok = send_email(_recipients(db, sa.workspace_id), subject, text_body, html_body)
        return "alerted" if ok else "alert_failed"
    finally:
        db.close()


# ----------------------------------------------------------- lifecycle --

def start_scheduler() -> None:
    global _scheduler
    if not config.SCHEDULER_ENABLED or _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        run_due_analyses, "interval", seconds=config.SCHEDULER_TICK_SECONDS,
        id="scheduled_analyses_tick", max_instances=1, coalesce=True,
    )
    _scheduler.start()
    logger.info("Scheduler started (tick every %ss)", config.SCHEDULER_TICK_SECONDS)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
