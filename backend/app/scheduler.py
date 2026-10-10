"""Scheduled re-runs: an in-process APScheduler tick finds due
scheduled_analyses, runs each through the normal pipeline exactly as a manual
question would, then diffs the result and emails the team only if it matters.

In-process on purpose: no new infrastructure. The catch is that every API
worker process would run its own tick, so a due analysis is *claimed* with a
conditional UPDATE first -- only the process that wins the claim runs it. The
run itself is a durable job (app/jobs.py), so a restart mid-run does not lose it.

Monitoring policy (see docs/PLATFORM.md "Monitoring"):
- Comparison: against the previous run (default), the latest run on the same
  weekday, or the mean of the last N runs -- chosen per schedule, so a weekly
  cycle does not page people every Monday.
- A change must cross the percentage threshold AND, when configured, a minimum
  absolute size; a 40% move from 2 to 3 is rarely worth an alert.
- Data-quality gate: while the dataset version has open, high-severity schema or
  quality incidents, KPI alerts are held back and one data-quality alert is
  raised instead -- the KPI probably moved because the data broke.
- Deduplication: an alert that is still open for the same schedule, kind and
  KPI is updated (occurrence count, last seen), not raised again, and not
  emailed again.
- Idempotency: processing the same finished run twice does nothing the second
  time, so a retried job cannot send a second email.
- Delivery: a failed email is retried a bounded number of times from the stored
  payload. In-app alerts never depend on email.
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
from app.change_detection import Anomaly, KpiChange, _norm, detect_anomalies, diff_kpis, parse_kpi_value, trend_of
from app.database import SessionLocal
from app.dataset_versions import latest_version
from app.email_sender import email_configured, send_email
from app.models import Alert, Dashboard, Dataset, Question, ScheduledAnalysis, TeamMember, User
from app.verdict import UNVERIFIED, rejected_reviews, resolve_verdict_state

logger = logging.getLogger(__name__)

INTERVALS = {"daily": dt.timedelta(days=1), "weekly": dt.timedelta(days=7)}
COMPARISONS = ("previous", "same_weekday", "rolling_mean")
DEFAULT_WINDOW_RUNS = 4

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
    from app import jobs

    dataset = db.get(Dataset, sa.dataset_id)
    if dataset is None:
        logger.warning("Scheduled analysis %s: dataset %s is gone; skipping", sa.id, sa.dataset_id)
        return None
    version = latest_version(db, dataset)
    # A durable job, not a thread: the worker pool runs it, and runs the
    # post-run alerting (process_completed_run) when it finishes. `start_thread`
    # is kept for callers that passed it; there is no thread to start any more.
    # Scheduled runs are not subject to the per-team admission limit -- the
    # schedule itself already bounds how many there can be.
    question, _job, _created = jobs.submit_question(
        db, team_id=sa.workspace_id, dataset=dataset, text=sa.question_text, created_by=sa.created_by,
        trigger="scheduled", scheduled_analysis_id=sa.id,
        dataset_version_id=version.id if version else None, stage_detail="Scheduled run",
        enforce_admission=False,
    )
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
                   changes: list[KpiChange], anomalies: list[Anomaly], rejection_text: str,
                   explanation: dict | None = None, dq_incidents: list | None = None) -> list[Alert]:
    """Persist the in-app alert(s) for this run. Independent of email: a team with
    no SMTP configured still sees them on the Scheduled page.

    Returns the alerts that are NEW. An alert whose incident is already open
    (same schedule, kind and subject) is updated in place and not returned, so
    the caller does not notify anyone about it a second time."""
    now = dt.datetime.utcnow()
    created: list[Alert] = []

    def add(kind: str, title: str, detail: str, subject: str = "", severity: str = "warn") -> None:
        key = f"{sa.id}:{kind}:{_norm(subject)}"
        open_alert = (db.query(Alert)
                      .filter(Alert.team_id == sa.workspace_id, Alert.dedup_key == key,
                              (Alert.status.is_(None)) | (Alert.status.in_(("open", "acknowledged"))))
                      .order_by(Alert.id.desc()).first())
        if open_alert is not None:
            open_alert.occurrences = (open_alert.occurrences or 1) + 1
            open_alert.last_seen_at = now
            open_alert.title, open_alert.detail = title[:200], detail
            open_alert.question_id = question.id
            open_alert.explanation_json = explanation
            return
        alert = Alert(team_id=sa.workspace_id, scheduled_analysis_id=sa.id, question_id=question.id,
                      kind=kind, title=title[:200], detail=detail, status="open", severity=severity,
                      dedup_key=key, occurrences=1, last_seen_at=now, explanation_json=explanation,
                      delivery_state="none", delivery_attempts=0)
        db.add(alert)
        created.append(alert)

    if dq_incidents:
        add("data_quality", f"Data quality problems on the latest data: {len(dq_incidents)} open",
            "KPI alerts for this run were held back because the data itself looks wrong:\n"
            + "\n".join(f"- {i.message}" for i in dq_incidents[:8]), severity="high")
        db.commit()
        return created

    q_short = question.text if len(question.text) <= 60 else question.text[:57] + "..."
    if verdict == UNVERIFIED:
        add("unverified", f"Could not verify: {q_short}", rejection_text or "The Critic could not confirm this run.",
            severity="high")
    crossed = [c for c in changes if c.crossed]
    if crossed:
        more = f" (+{len(crossed) - 1} more)" if len(crossed) > 1 else ""
        add("change", f"{crossed[0].describe()}{more}", "\n".join(c.describe() for c in changes),
            subject=crossed[0].label)
    for a in anomalies:
        add("anomaly", f"Unusual {a.label}: {a.value}", a.describe(), subject=a.label)
    db.commit()
    return created


def _comparison_kpis(db: Session, sa: ScheduledAnalysis, question: Question) -> tuple[list[dict] | None, str]:
    """The KPI list this run is compared against, per the schedule's mode.
    Returns (kpis or None when there is nothing to compare with, description)."""
    mode = sa.comparison if sa.comparison in COMPARISONS else "previous"
    previous = db.get(Dashboard, sa.last_dashboard_id) if sa.last_dashboard_id else None
    if mode == "previous" or previous is None:
        return (previous.kpis_json if previous else None), "the previous run"
    earlier = (db.query(Question)
               .filter(Question.scheduled_analysis_id == sa.id, Question.id != question.id)
               .order_by(Question.id.desc()).limit(60).all())

    def dash_of(q: Question) -> Dashboard | None:
        return (db.query(Dashboard).filter(Dashboard.question_id == q.id)
                .order_by(Dashboard.created_at.desc()).first())

    if mode == "same_weekday":
        weekday = (question.created_at or dt.datetime.utcnow()).weekday()
        for q in earlier:
            if q.created_at is not None and q.created_at.weekday() == weekday:
                d = dash_of(q)
                if d is not None:
                    return d.kpis_json, "the latest run on the same weekday"
        return previous.kpis_json, "the previous run (no earlier run on the same weekday yet)"
    window = sa.window_runs or DEFAULT_WINDOW_RUNS
    values: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    used = 0
    for q in earlier:
        d = dash_of(q)
        if d is None:
            continue
        used += 1
        for k in d.kpis_json or []:
            n = parse_kpi_value(str(k.get("value", "")))
            if n is not None:
                values.setdefault(_norm(k.get("label")), []).append(n)
                labels.setdefault(_norm(k.get("label")), k.get("label", ""))
        if used >= window:
            break
    if not values:
        return previous.kpis_json, "the previous run (no numeric history yet)"
    return ([{"label": labels[key], "value": f"{sum(v) / len(v):.6g}"} for key, v in values.items()],
            f"the mean of the last {used} run(s)")


def _apply_min_effect(changes: list[KpiChange], min_effect_abs: float | None) -> None:
    """A change that crossed the percentage threshold still does not count
    unless it is also at least `min_effect_abs` in the KPI's own units."""
    if not min_effect_abs:
        return
    for c in changes:
        old, new = parse_kpi_value(c.old), parse_kpi_value(c.new)
        if c.crossed and old is not None and new is not None and abs(new - old) < min_effect_abs:
            c.crossed = False


def _history_stats(history: list[list[dict]]) -> dict:
    import statistics

    series: dict[str, list[float]] = {}
    for kpis in history:
        for k in kpis or []:
            n = parse_kpi_value(str(k.get("value", "")))
            if n is not None:
                series.setdefault(str(k.get("label", "")), []).append(n)
    return {label: {"runs": len(v), "mean": statistics.fmean(v),
                    "stdev": statistics.stdev(v) if len(v) > 1 else 0.0, "min": min(v), "max": max(v)}
            for label, v in series.items()}


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

        if sa.last_processed_question_id == question_id:
            # Already handled (a retried job, or a second call). Doing it again
            # would raise the same alerts and send the same email twice.
            return "duplicate"
        rejections = rejected_reviews(db, question_id)
        verdict = resolve_verdict_state(dash, rejections)
        previous = db.get(Dashboard, sa.last_dashboard_id) if sa.last_dashboard_id else None
        compare_kpis, compared_with = _comparison_kpis(db, sa, question)
        changes = diff_kpis(compare_kpis, dash.kpis_json, sa.change_threshold_pct) if compare_kpis else []
        _apply_min_effect(changes, sa.min_effect_abs)
        crossed = any(c.crossed for c in changes)
        history = _kpi_history(db, sa.id, question_id)
        anomalies = detect_anomalies(history, dash.kpis_json)
        from app import data_quality

        dq_blocking = []
        if sa.suppress_on_dq is not False:
            dq_blocking = data_quality.has_blocking_incidents(db, sa.dataset_id, question.dataset_version_id)
        sa.last_dashboard_id = dash.id
        sa.last_trend = trend_of(changes) if previous else None
        sa.last_change_summary = "; ".join(c.describe() for c in changes[:3])
        sa.last_processed_question_id = question_id
        db.commit()
        explanation = {
            "compared_with": compared_with, "threshold_pct": sa.change_threshold_pct,
            "min_effect_abs": sa.min_effect_abs, "verdict": verdict,
            "changes": [{"label": c.label, "old": c.old, "new": c.new, "pct": c.pct, "crossed": c.crossed} for c in changes],
            "anomalies": [{"label": a.label, "value": a.value, "mean": a.mean, "z": a.z, "runs": a.runs} for a in anomalies],
            "history": _history_stats(history),
            "evidence": {"question_id": question_id, "dashboard_id": dash.id,
                         "dataset_version_id": question.dataset_version_id},
            "data_quality": [{"check": i.check_name, "column": i.column_name, "message": i.message} for i in dq_blocking],
        }
        if dq_blocking and (crossed or anomalies) and verdict != UNVERIFIED:
            # The KPI moved, but so did the data's shape or completeness. Say
            # that, once, instead of reporting a business change.
            try:
                new_alerts = _record_alerts(db, sa, question, verdict, [], [], "", explanation, dq_blocking)
            except Exception:  # noqa: BLE001
                db.rollback()
                logger.exception("Could not record the data-quality alert for question %s", question_id)
                new_alerts = []
            logger.info("Scheduled analysis %s run %s: KPI alerts suppressed by %s data-quality incident(s)",
                        sa.id, question_id, len(dq_blocking))
            if new_alerts:
                subject = f"[DataSage] Data quality problem on tracked analysis: {question.text[:60]}"
                body = (new_alerts[0].detail or "") + f"\n\nView: {config.APP_URL.rstrip('/')}/Dashboards?question={question.id}"
                _deliver(db, sa, new_alerts, subject, body, f"<pre>{_html.escape(body)}</pre>")
            return "suppressed"

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
        new_alerts: list[Alert] | None
        try:
            new_alerts = _record_alerts(db, sa, question, verdict, changes, anomalies, rejection_text, explanation)
        except Exception:  # noqa: BLE001 - the in-app feed must never block the email
            db.rollback()
            logger.exception("Could not record in-app alerts for question %s", question_id)
            new_alerts = None
        if new_alerts is not None and not new_alerts:
            # Every incident this run would report is already open and already
            # notified. Counted, not re-sent.
            return "deduplicated"
        subject, text_body, html_body = _build_email(question, verdict, changes, rejection_text, anomalies)
        return _deliver(db, sa, new_alerts or [], subject, text_body, html_body)
    finally:
        db.close()


def _deliver(db: Session, sa: ScheduledAnalysis, alerts: list[Alert], subject: str, text_body: str,
             html_body: str) -> str:
    """Send one email for this run and record the outcome on its alerts."""
    if sa.notify_email is False:
        for a in alerts:
            a.delivery_state = "skipped"
        db.commit()
        return "alerted_in_app"
    ok = send_email(_recipients(db, sa.workspace_id), subject, text_body, html_body)
    now = dt.datetime.utcnow()
    for i, a in enumerate(alerts):
        a.delivery_attempts = 1
        if ok:
            a.delivery_state = "sent"
        elif not email_configured():
            a.delivery_state = "skipped"  # nothing to retry until SMTP is configured
        else:
            a.delivery_state = "failed"
            if i == 0:  # one email per run: keep its payload on the first alert only
                a.delivery_payload_json = {"subject": subject, "text": text_body, "html": html_body}
                a.next_delivery_at = now + dt.timedelta(seconds=config.ALERT_DELIVERY_RETRY_SECONDS)
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
    return "alerted" if ok else "alert_failed"


def retry_failed_deliveries(now: dt.datetime | None = None) -> dict[str, int]:
    """Re-send alert emails that failed, up to ALERT_DELIVERY_MAX_ATTEMPTS, from
    the payload stored at the first attempt. Nothing is recomputed or re-run."""
    now = now or dt.datetime.utcnow()
    counts = {"sent": 0, "failed": 0, "gave_up": 0}
    db = SessionLocal()
    try:
        due = (db.query(Alert)
               .filter(Alert.delivery_state == "failed", Alert.delivery_payload_json.isnot(None),
                       Alert.next_delivery_at.isnot(None), Alert.next_delivery_at <= now).all())
        for alert in due:
            payload = alert.delivery_payload_json or {}
            siblings = (db.query(Alert).filter(Alert.question_id == alert.question_id,
                                               Alert.delivery_state == "failed").all())
            if (alert.delivery_attempts or 0) >= config.ALERT_DELIVERY_MAX_ATTEMPTS:
                for s in siblings:
                    s.delivery_state, s.next_delivery_at = "gave_up", None
                counts["gave_up"] += 1
                db.commit()
                continue
            ok = send_email(_recipients(db, alert.team_id), payload.get("subject", ""), payload.get("text", ""),
                            payload.get("html", ""))
            alert.delivery_attempts = (alert.delivery_attempts or 0) + 1
            if ok:
                for s in siblings:
                    s.delivery_state, s.next_delivery_at = "sent", None
                alert.delivery_payload_json = None
                counts["sent"] += 1
            else:
                backoff = config.ALERT_DELIVERY_RETRY_SECONDS * (2 ** (alert.delivery_attempts - 1))
                alert.next_delivery_at = now + dt.timedelta(seconds=backoff)
                counts["failed"] += 1
            db.commit()
    finally:
        db.close()
    return counts


_last_maintenance = dt.datetime.min


def maintenance(now: dt.datetime | None = None) -> None:
    """Housekeeping on the scheduler tick: delivery retries every tick, the
    workspace retention sweep at most hourly."""
    global _last_maintenance
    now = now or dt.datetime.utcnow()
    try:
        retry_failed_deliveries(now)
    except Exception:  # noqa: BLE001
        logger.exception("Alert delivery retry failed")
    if (now - _last_maintenance).total_seconds() >= 3600:
        _last_maintenance = now
        try:
            from app import provenance

            removed = provenance.retention_sweep()
            if removed:
                logger.info("Retention: removed %s old workspace(s)", removed)
        except Exception:  # noqa: BLE001
            logger.exception("Retention sweep failed")


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
    _scheduler.add_job(
        maintenance, "interval", seconds=config.SCHEDULER_TICK_SECONDS,
        id="maintenance_tick", max_instances=1, coalesce=True,
    )
    _scheduler.start()
    logger.info("Scheduler started (tick every %ss)", config.SCHEDULER_TICK_SECONDS)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
