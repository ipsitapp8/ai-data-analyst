"""Durable analysis jobs: a queue in the application database.

Why a table and not a broker: the app already depends on one database and runs
as one process; a row claimed with a compare-and-set UPDATE gives durability,
recovery and exactly-one-owner without adding Redis or Celery. The same code
works on SQLite and Postgres, and with more than one API process.

State machine (terminal states never change again):

    queued ──claim──> running ──> succeeded
       │                 │──────> failed      (failure_class says why)
       │                 │──────> cancelled
       │                 └──────> timed_out
       └──cancel──> cancelled
    running ──lease expired──> queued   (attempts left, nothing published yet)

Policy, in one place:

- Admission: a team may have at most JOB_TEAM_MAX_ACTIVE queued+running jobs.
- Concurrency: one job per worker thread, JOB_WORKERS threads per process.
- Idempotent submission: the same Idempotency-Key from the same team returns
  the first job; it never starts a second run.
- Ownership: a worker holds a lease it renews every few seconds. Every write
  that ends a job is conditional on still owning it, so a worker that lost its
  lease cannot overwrite the result of the worker that took over.
- Recovery: a running job whose lease expired (the process died) goes back to
  `queued` if it has attempts left, else to `failed/abandoned`. If a dashboard
  was already published for it, it is marked `succeeded` instead of re-run --
  publishing is the one step that must not happen twice.
- Deadline: each job gets JOB_DEADLINE_SECONDS from its first start. Past it,
  the run is stopped at the next checkpoint and ends `timed_out`.
- Cancellation: a queued job is cancelled at once; a running one at its next
  checkpoint (before each node, model call and code run; a running sandbox
  process is killed).
- Automatic retry happens only after abandonment. A run that failed on its own
  terms is not retried silently; POST /api/jobs/{id}/retry does it on request.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
import uuid
from typing import Callable

from sqlalchemy import func, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import config, runtime
from app.database import SessionLocal
from app.models import AnalysisJob, Dashboard, Dataset, DatasetVersion, Question

logger = logging.getLogger(__name__)

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, TIMED_OUT = (
    "queued", "running", "succeeded", "failed", "cancelled", "timed_out")
ACTIVE = (QUEUED, RUNNING)
TERMINAL = (SUCCEEDED, FAILED, CANCELLED, TIMED_OUT)

SAFE_ERRORS = {
    "abandoned": "The server stopped while this analysis was running and it could not be resumed.",
    "internal": "An internal error occurred while running this analysis.",
    "analysis_failed": "The analysis could not be completed.",
}


class AdmissionRejected(Exception):
    """The team already has as many active jobs as it is allowed."""


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


# ------------------------------------------------------------- submission --

def active_count(db: Session, team_id: int) -> int:
    return db.query(AnalysisJob).filter(AnalysisJob.team_id == team_id, AnalysisJob.state.in_(ACTIVE)).count()


def find_by_idempotency_key(db: Session, team_id: int, key: str | None) -> AnalysisJob | None:
    if not key:
        return None
    return db.query(AnalysisJob).filter_by(idempotency_key=f"{team_id}:{key}").first()


def submit_question(
    db: Session, *, team_id: int, dataset: Dataset, text: str, created_by: int | None = None,
    idempotency_key: str | None = None, trigger: str = "manual", scheduled_analysis_id: int | None = None,
    dataset_version_id: int | None = None, rerun_of_question_id: int | None = None,
    route: str | None = None, stage_detail: str = "Waiting for a worker", enforce_admission: bool = True,
) -> tuple[Question, AnalysisJob, bool]:
    """Create a question and its job in one transaction.

    Returns (question, job, created). `created` is False when the idempotency
    key matched an earlier submission, in which case nothing new was written."""
    existing = find_by_idempotency_key(db, team_id, idempotency_key)
    if existing is not None:
        return db.get(Question, existing.question_id), existing, False
    if enforce_admission and active_count(db, team_id) >= config.JOB_TEAM_MAX_ACTIVE:
        raise AdmissionRejected(
            f"This team already has {config.JOB_TEAM_MAX_ACTIVE} analyses queued or running. "
            "Wait for one to finish or cancel one.")

    if dataset_version_id is None:
        from app.dataset_versions import latest_version

        latest = latest_version(db, dataset)
        dataset_version_id = latest.id if latest else None

    question = Question(
        team_id=team_id, dataset_id=dataset.id, dataset_version_id=dataset_version_id, text=text,
        status="running", current_stage="queued", stage_detail=stage_detail, trigger=trigger,
        scheduled_analysis_id=scheduled_analysis_id, rerun_of_question_id=rerun_of_question_id, route=route,
    )
    db.add(question)
    try:
        db.flush()
        job = AnalysisJob(
            team_id=team_id, question_id=question.id, state=QUEUED, max_attempts=config.JOB_MAX_ATTEMPTS,
            idempotency_key=f"{team_id}:{idempotency_key}" if idempotency_key else None,
            created_by=created_by, progress_json={"stage": "queued"},
        )
        db.add(job)
        db.commit()
    except IntegrityError:
        # Two identical submissions raced; the unique key let exactly one win.
        db.rollback()
        existing = find_by_idempotency_key(db, team_id, idempotency_key)
        if existing is None:
            raise
        return db.get(Question, existing.question_id), existing, False
    db.refresh(question)
    db.refresh(job)
    return question, job, True


# ---------------------------------------------------------------- claiming --

def claim_next(db: Session, worker_id: str, job_id: int | None = None) -> AnalysisJob | None:
    """Take ownership of the oldest queued job (or of `job_id`). Returns None
    when there is nothing to take or another worker took it first."""
    q = db.query(AnalysisJob).filter(AnalysisJob.state == QUEUED)
    if job_id is not None:
        q = q.filter(AnalysisJob.id == job_id)
    candidates = q.order_by(AnalysisJob.queued_at, AnalysisJob.id).limit(5).all()
    now = _now()
    for cand in candidates:
        deadline = cand.deadline_at or now + dt.timedelta(seconds=config.JOB_DEADLINE_SECONDS)
        result = db.execute(
            update(AnalysisJob)
            .where(AnalysisJob.id == cand.id, AnalysisJob.state == QUEUED)
            .values(state=RUNNING, worker_id=worker_id, attempt=AnalysisJob.attempt + 1,
                    started_at=cand.started_at or now, deadline_at=deadline,
                    lease_expires_at=now + dt.timedelta(seconds=config.JOB_LEASE_SECONDS))
        )
        db.commit()
        if result.rowcount == 1:
            db.expire_all()
            return db.get(AnalysisJob, cand.id)
    return None


def heartbeat(job_id: int, worker_id: str) -> tuple[bool, bool]:
    """Renew the lease. Returns (still_owner, cancel_requested)."""
    db = SessionLocal()
    try:
        result = db.execute(
            update(AnalysisJob)
            .where(AnalysisJob.id == job_id, AnalysisJob.state == RUNNING, AnalysisJob.worker_id == worker_id)
            .values(lease_expires_at=_now() + dt.timedelta(seconds=config.JOB_LEASE_SECONDS))
        )
        db.commit()
        if result.rowcount != 1:
            return False, True
        job = db.get(AnalysisJob, job_id)
        return True, bool(job.cancel_requested)
    finally:
        db.close()


def finish(job_id: int, worker_id: str, state: str, failure_class: str | None = None,
           error: str | None = None) -> bool:
    """Move a job this worker owns to a terminal state. False if it no longer
    owns it (the job was recovered elsewhere), in which case nothing is written."""
    assert state in TERMINAL
    db = SessionLocal()
    try:
        result = db.execute(
            update(AnalysisJob)
            .where(AnalysisJob.id == job_id, AnalysisJob.state == RUNNING, AnalysisJob.worker_id == worker_id)
            .values(state=state, failure_class=failure_class, error=error, finished_at=_now(),
                    lease_expires_at=None)
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def set_progress(question_id: int, stage: str, detail: str = "") -> None:
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter_by(question_id=question_id).first()
        if job is not None and job.state == RUNNING:
            job.progress_json = {"stage": stage, "detail": detail[:300], "at": _now().isoformat()}
            db.commit()
    except Exception:  # noqa: BLE001 - progress is advisory
        db.rollback()
    finally:
        db.close()


# ------------------------------------------------------- cancel and retry --

def _mark_question(question_id: int, status: str, message: str | None) -> None:
    from app.agents.state import mark_terminal

    mark_terminal(question_id, status, message)


def request_cancel(db: Session, job: AnalysisJob) -> str:
    """Cancel a job. Returns its state afterwards."""
    if job.state in TERMINAL:
        return job.state
    result = db.execute(
        update(AnalysisJob).where(AnalysisJob.id == job.id, AnalysisJob.state == QUEUED)
        .values(state=CANCELLED, finished_at=_now(), failure_class="cancelled",
                error=runtime.RunCancelled.user_message)
    )
    db.commit()
    if result.rowcount == 1:
        _mark_question(job.question_id, "cancelled", runtime.RunCancelled.user_message)
        return CANCELLED
    db.execute(update(AnalysisJob).where(AnalysisJob.id == job.id, AnalysisJob.state == RUNNING)
               .values(cancel_requested=True))
    db.commit()
    db.refresh(job)
    return job.state


def retry(db: Session, job: AnalysisJob) -> bool:
    """Queue a finished-but-unsuccessful job again. Refused when a dashboard
    was already published for it (that would publish twice)."""
    if job.state not in (FAILED, CANCELLED, TIMED_OUT):
        return False
    if db.query(Dashboard.id).filter(Dashboard.question_id == job.question_id).first():
        return False
    result = db.execute(
        update(AnalysisJob).where(AnalysisJob.id == job.id, AnalysisJob.state == job.state)
        .values(state=QUEUED, attempt=0, cancel_requested=False, failure_class=None, error=None,
                worker_id=None, lease_expires_at=None, deadline_at=None, started_at=None,
                finished_at=None, queued_at=_now(), progress_json={"stage": "queued"})
    )
    db.commit()
    if result.rowcount != 1:
        return False
    q = db.get(Question, job.question_id)
    if q is not None:
        q.status, q.current_stage, q.stage_detail, q.error = "running", "queued", "Queued again", None
        db.commit()
    return True


# ----------------------------------------------------------------- recovery --

def recover_abandoned(now: dt.datetime | None = None) -> dict[str, int]:
    """Deal with jobs whose owner died, and with jobs past their deadline.
    Safe to call from any process at any time; every step is compare-and-set."""
    now = now or _now()
    counts = {"requeued": 0, "failed": 0, "succeeded": 0, "timed_out": 0}
    db = SessionLocal()
    try:
        stale = db.query(AnalysisJob).filter(
            AnalysisJob.state == RUNNING, AnalysisJob.lease_expires_at.isnot(None),
            AnalysisJob.lease_expires_at < now).all()
        for job in stale:
            published = db.query(Dashboard.id).filter(Dashboard.question_id == job.question_id).first()
            past_deadline = job.deadline_at is not None and job.deadline_at <= now
            if published:
                values, bucket = {"state": SUCCEEDED, "finished_at": now}, "succeeded"
            elif past_deadline:
                values = {"state": TIMED_OUT, "finished_at": now, "failure_class": "timed_out",
                          "error": runtime.DeadlineExceeded.user_message}
                bucket = "timed_out"
            elif job.cancel_requested:
                values = {"state": CANCELLED, "finished_at": now, "failure_class": "cancelled",
                          "error": runtime.RunCancelled.user_message}
                bucket = "failed"
            elif job.attempt < job.max_attempts:
                values = {"state": QUEUED, "worker_id": None, "queued_at": now,
                          "progress_json": {"stage": "queued", "detail": "Recovered after a restart"}}
                bucket = "requeued"
            else:
                values = {"state": FAILED, "finished_at": now, "failure_class": "abandoned",
                          "error": SAFE_ERRORS["abandoned"]}
                bucket = "failed"
            result = db.execute(
                update(AnalysisJob)
                .where(AnalysisJob.id == job.id, AnalysisJob.state == RUNNING,
                       AnalysisJob.lease_expires_at == job.lease_expires_at)
                .values(lease_expires_at=None, **values)
            )
            db.commit()
            if result.rowcount != 1:
                continue
            counts[bucket] += 1
            state = values["state"]
            if state == QUEUED:
                q = db.get(Question, job.question_id)
                if q is not None:
                    q.status, q.current_stage, q.stage_detail = "running", "queued", "Resuming after a restart"
                    db.commit()
            elif state == SUCCEEDED:
                dash = (db.query(Dashboard).filter(Dashboard.question_id == job.question_id)
                        .order_by(Dashboard.created_at.desc()).first())
                _mark_question(job.question_id, "verified" if dash.verified else "unverified", None)
            elif state == CANCELLED:
                _mark_question(job.question_id, "cancelled", values["error"])
            else:
                _mark_question(job.question_id, "failed", values["error"])
            logger.warning("Recovered job %s (question %s): %s", job.id, job.question_id, bucket)

        # Queued jobs that sat past their deadline (e.g. no worker for a long time).
        overdue = db.query(AnalysisJob).filter(
            AnalysisJob.state == QUEUED, AnalysisJob.deadline_at.isnot(None), AnalysisJob.deadline_at <= now).all()
        for job in overdue:
            result = db.execute(
                update(AnalysisJob).where(AnalysisJob.id == job.id, AnalysisJob.state == QUEUED)
                .values(state=TIMED_OUT, finished_at=now, failure_class="timed_out",
                        error=runtime.DeadlineExceeded.user_message))
            db.commit()
            if result.rowcount == 1:
                counts["timed_out"] += 1
                _mark_question(job.question_id, "failed", runtime.DeadlineExceeded.user_message)
    finally:
        db.close()
    return counts


def fail_orphan_questions() -> int:
    """Questions left `running` with no job at all: runs started by the old
    thread-per-question model and cut off by a restart. Called once at startup."""
    db = SessionLocal()
    try:
        with_job = db.query(AnalysisJob.question_id)
        orphans = (db.query(Question).filter(Question.status.in_(("running", "pending")),
                                             ~Question.id.in_(with_job)).all())
        for q in orphans:
            q.status, q.current_stage = "failed", "failed"
            q.error = SAFE_ERRORS["abandoned"]
        db.commit()
        return len(orphans)
    finally:
        db.close()


# ---------------------------------------------------------------- execution --

def _default_runner(question_id: int) -> str:
    from app.agents.graph import run_question_graph

    return run_question_graph(question_id)


class _Heartbeat(threading.Thread):
    def __init__(self, job_id: int, worker_id: str):
        super().__init__(daemon=True, name=f"job-heartbeat-{job_id}")
        self.job_id, self.worker_id = job_id, worker_id
        self.stop_event = threading.Event()
        self.cancel = False
        self.lost = False
        self._last_poll = 0.0

    def should_stop(self) -> bool:
        """True once the job is cancelled or no longer ours. Checkpoints call
        this often, so it re-reads the row at most every couple of seconds."""
        if not (self.cancel or self.lost) and time.monotonic() - self._last_poll > 2.0:
            self._last_poll = time.monotonic()
            self.beat()
        return self.cancel or self.lost

    def run(self) -> None:
        interval = max(1.0, config.JOB_LEASE_SECONDS / 3)
        while not self.stop_event.wait(interval):
            self.beat()

    def beat(self) -> None:
        try:
            owner, cancel = heartbeat(self.job_id, self.worker_id)
        except Exception:  # noqa: BLE001 - a missed beat is retried on the next tick
            logger.warning("Heartbeat failed for job %s", self.job_id, exc_info=True)
            return
        if not owner:
            self.lost = True
        if cancel:
            self.cancel = True


def execute(job: AnalysisJob, worker_id: str, runner: Callable[[int], str] | None = None) -> str:
    """Run one claimed job to a terminal state. Returns that state."""
    runner = runner or _default_runner
    job_id, question_id, team_id = job.id, job.question_id, job.team_id

    db = SessionLocal()
    try:
        already = db.query(Dashboard).filter(Dashboard.question_id == question_id).first()
    finally:
        db.close()
    if already is not None:
        # A previous attempt published before its worker died. Do not run again.
        finish(job_id, worker_id, SUCCEEDED)
        _mark_question(question_id, "verified" if already.verified else "unverified", None)
        return SUCCEEDED

    hb = _Heartbeat(job_id, worker_id)
    hb.beat()
    hb.start()
    ctx = runtime.RunContext(
        question_id=question_id, team_id=team_id, deadline_at=job.deadline_at,
        should_cancel=hb.should_stop,
        max_llm_calls=config.RUN_MAX_LLM_CALLS, max_sandbox_runs=config.RUN_MAX_SANDBOX_RUNS,
        max_cost_usd=config.RUN_MAX_COST_USD,
    )
    state, failure_class, error, q_status = FAILED, "internal", SAFE_ERRORS["internal"], "failed"
    try:
        with runtime.run_context(ctx):
            outcome = runner(question_id)
        if outcome == "failed":
            state, failure_class, error, q_status = FAILED, "analysis_failed", None, None
        else:
            state, failure_class, error, q_status = SUCCEEDED, None, None, None
    except runtime.RunCancelled as e:
        if hb.lost:
            logger.warning("Job %s: lease lost; leaving the outcome to the new owner", job_id)
            hb.stop_event.set()
            return RUNNING
        state, failure_class, error, q_status = CANCELLED, e.failure_class, e.user_message, "cancelled"
    except runtime.DeadlineExceeded as e:
        state, failure_class, error, q_status = TIMED_OUT, e.failure_class, e.user_message, "failed"
    except runtime.RunAborted as e:
        state, failure_class, error, q_status = FAILED, e.failure_class, e.user_message, "failed"
    except Exception:  # noqa: BLE001 - job boundary: nothing may escape a worker
        logger.exception("Job %s (question %s) crashed", job_id, question_id)
    finally:
        hb.stop_event.set()

    if not finish(job_id, worker_id, state, failure_class, error):
        logger.warning("Job %s finished as %s but is no longer owned by %s; not recorded", job_id, state, worker_id)
        return state
    if q_status is not None:
        _mark_question(question_id, q_status, error)
    if state == FAILED and failure_class == "analysis_failed":
        _copy_question_error(job_id, question_id)
    runtime.record_event("budget", "usage", ctx.usage(), question_id=question_id, team_id=team_id)
    _after_run(question_id, ctx)
    return state


def _copy_question_error(job_id: int, question_id: int) -> None:
    db = SessionLocal()
    try:
        q, job = db.get(Question, question_id), db.get(AnalysisJob, job_id)
        if q is not None and job is not None:
            job.error = (q.error or SAFE_ERRORS["analysis_failed"])[:2000]
            db.commit()
    finally:
        db.close()


def _after_run(question_id: int, ctx: runtime.RunContext) -> None:
    """Bookkeeping once a job is terminal. Each step is independent and none
    can change the job's outcome."""
    try:
        from app import provenance

        provenance.write_manifest(question_id, ctx.usage())
    except Exception:  # noqa: BLE001
        logger.exception("Could not write the run manifest for question %s", question_id)
    try:
        db = SessionLocal()
        try:
            scheduled = db.get(Question, question_id).scheduled_analysis_id
        finally:
            db.close()
        if scheduled is not None:
            from app.scheduler import process_completed_run

            process_completed_run(question_id)
    except Exception:  # noqa: BLE001
        logger.exception("Post-run alerting failed for question %s", question_id)
    try:
        from app.sandbox.runner import SandboxRunner

        SandboxRunner().cleanup_workspace(question_id)
    except Exception:  # noqa: BLE001
        logger.warning("Could not clean the workspace of question %s", question_id, exc_info=True)


def run_inline(job_id: int | None = None, runner: Callable[[int], str] | None = None) -> str | None:
    """Claim and run one job on the calling thread. Used by tests, the
    evaluation runner and the CLI; the worker threads use the same code."""
    worker_id = f"inline-{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        job = claim_next(db, worker_id, job_id)
        if job is None:
            return None
        db.expunge(job)
    finally:
        db.close()
    return execute(job, worker_id, runner)


class _Worker(threading.Thread):
    def __init__(self, index: int, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"job-worker-{index}")
        self.worker_id = f"{os.getpid()}-{index}-{uuid.uuid4().hex[:6]}"
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                db = SessionLocal()
                try:
                    job = claim_next(db, self.worker_id)
                    if job is not None:
                        db.expunge(job)
                finally:
                    db.close()
                if job is None:
                    self.stop_event.wait(config.JOB_POLL_SECONDS)
                    continue
                execute(job, self.worker_id)
            except Exception:  # noqa: BLE001 - a worker must survive anything
                logger.exception("Job worker loop error")
                self.stop_event.wait(config.JOB_POLL_SECONDS)


class _Janitor(threading.Thread):
    def __init__(self, stop_event: threading.Event):
        super().__init__(daemon=True, name="job-janitor")
        self.stop_event = stop_event

    def run(self) -> None:
        interval = max(2.0, config.JOB_LEASE_SECONDS / 2)
        while not self.stop_event.wait(interval):
            try:
                recover_abandoned()
            except Exception:  # noqa: BLE001
                logger.exception("Job recovery pass failed")


_stop_event: threading.Event | None = None
_threads: list[threading.Thread] = []


def start_workers() -> None:
    """Start the worker pool. Recovery runs first, so jobs cut off by the last
    shutdown are back in the queue before any worker looks at it."""
    global _stop_event
    if _stop_event is not None:
        return
    try:
        orphans = fail_orphan_questions()
        recovered = recover_abandoned()
        if orphans or any(recovered.values()):
            logger.info("Startup recovery: %s orphan question(s) failed, jobs %s", orphans, recovered)
    except Exception:  # noqa: BLE001
        logger.exception("Startup job recovery failed")
    if config.JOB_WORKERS <= 0:
        return
    _stop_event = threading.Event()
    for i in range(config.JOB_WORKERS):
        t = _Worker(i, _stop_event)
        t.start()
        _threads.append(t)
    j = _Janitor(_stop_event)
    j.start()
    _threads.append(j)
    logger.info("Started %s job worker(s)", config.JOB_WORKERS)


def stop_workers() -> None:
    """Ask workers to stop. Jobs they are running keep their lease until it
    expires and are then recovered by the next process to start."""
    global _stop_event
    if _stop_event is None:
        return
    _stop_event.set()
    _stop_event = None
    _threads.clear()


# ------------------------------------------------------------------ metrics --

def queue_metrics(db: Session, team_id: int | None = None) -> dict:
    q = db.query(AnalysisJob.state, func.count(AnalysisJob.id))
    if team_id is not None:
        q = q.filter(AnalysisJob.team_id == team_id)
    by_state = {s: 0 for s in (*ACTIVE, *TERMINAL)}
    by_state.update(dict(q.group_by(AnalysisJob.state).all()))
    oldest_q = db.query(func.min(AnalysisJob.queued_at)).filter(AnalysisJob.state == QUEUED)
    if team_id is not None:
        oldest_q = oldest_q.filter(AnalysisJob.team_id == team_id)
    oldest = oldest_q.scalar()
    return {
        "by_state": by_state,
        "queue_depth": by_state[QUEUED],
        "running": by_state[RUNNING],
        "oldest_queued_seconds": int((_now() - oldest).total_seconds()) if oldest else 0,
        "workers": config.JOB_WORKERS,
        "team_limit": config.JOB_TEAM_MAX_ACTIVE,
    }


def job_out(job: AnalysisJob) -> dict:
    return {
        "id": job.id, "question_id": job.question_id, "kind": job.kind, "state": job.state,
        "attempt": job.attempt, "max_attempts": job.max_attempts,
        "cancel_requested": bool(job.cancel_requested), "failure_class": job.failure_class,
        "error": job.error, "progress": job.progress_json or {},
        "queued_at": job.queued_at, "started_at": job.started_at, "finished_at": job.finished_at,
        "deadline_at": job.deadline_at,
    }
