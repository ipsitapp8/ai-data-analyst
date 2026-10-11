"""Reproducibility: dataset fingerprints, run manifests, reruns and comparisons.

A manifest records what a run *was*: the exact data (content hash), the
question, the plan and the hash of every script that ran, the prompt and
application versions, the models that answered, the execution environment and
the outcome. It is written once when the run ends and never changed.

What a manifest is not: a promise that running again gives the same output.
The models are not deterministic, so two runs with identical manifests can
differ. `compare_runs` therefore reports which recorded inputs differ and what
that does and does not let you conclude -- it never claims bit-for-bit
reproducibility of an LLM run.
"""
from __future__ import annotations

import hashlib
import logging
import platform
import re
import subprocess
import sys
from functools import lru_cache
from importlib import metadata

from sqlalchemy.orm import Session

from app import config
from app.change_detection import _norm, diff_kpis
from app.database import SessionLocal
from app.models import (
    AnalysisJob,
    CriticReview,
    Dashboard,
    Dataset,
    DatasetVersion,
    EvidenceRecord,
    ExecutionLog,
    Plan,
    Question,
    RunEvent,
    RunManifest,
)

logger = logging.getLogger(__name__)

REPRODUCIBILITY_NOTE = (
    "This records the inputs and versions of the run. Model output is not deterministic, so a rerun "
    "with the same inputs can produce a different plan, different code and slightly different wording."
)


# -------------------------------------------------------------- fingerprints --

def file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def fingerprint(db: Session, version: DatasetVersion | None) -> str | None:
    """The version's recorded content hash, computed and stored on first use for
    versions uploaded before fingerprints existed."""
    if version is None:
        return None
    if not version.content_sha256:
        digest = file_sha256(version.filepath)
        if digest:
            version.content_sha256 = digest
            db.commit()
    return version.content_sha256


def dataset_evidence(db: Session, version: DatasetVersion | None, dataset: Dataset | None = None) -> dict:
    """The dataset facts verification needs: recorded hash and the file's hash now."""
    recorded = fingerprint(db, version)
    path = version.filepath if version else (dataset.filepath if dataset else None)
    profile = (version.profile_json if version else (dataset.profile_json if dataset else None)) or {}
    return {
        "version_id": version.id if version else None,
        "version_number": version.version_number if version else None,
        "fingerprint": recorded,
        "fingerprint_now": file_sha256(path),
        "row_count": profile.get("row_count"),
    }


# ------------------------------------------------------------------ versions --

@lru_cache(maxsize=1)
def prompt_version() -> str:
    from app.agents import prompts

    blob = "\n".join(str(getattr(prompts, name)) for name in sorted(dir(prompts)) if name.isupper())
    return text_sha256(blob)[:12]


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=str(config.ROOT_DIR))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


@lru_cache(maxsize=1)
def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.system().lower(),
        "pandas": _pkg("pandas"), "numpy": _pkg("numpy"), "plotly": _pkg("plotly"),
        "langgraph": _pkg("langgraph"),
        "sandbox_backend": config.SANDBOX_BACKEND,
        "sandbox_image": config.SANDBOX_IMAGE if config.SANDBOX_BACKEND == "docker" else "",
    }


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip("?.! ")


# ----------------------------------------------------------------- manifests --

def build_manifest(db: Session, question: Question, usage: dict | None = None) -> dict:
    dataset = db.get(Dataset, question.dataset_id)
    version = db.get(DatasetVersion, question.dataset_version_id) if question.dataset_version_id else None
    plans = db.query(Plan).filter_by(question_id=question.id).order_by(Plan.id).all()
    logs = (db.query(ExecutionLog).filter_by(question_id=question.id)
            .order_by(ExecutionLog.step_index, ExecutionLog.attempt_number).all())
    reviews = db.query(CriticReview).filter_by(question_id=question.id).order_by(CriticReview.id).all()
    dash = (db.query(Dashboard).filter_by(question_id=question.id).order_by(Dashboard.created_at.desc()).first())
    evidence = db.query(EvidenceRecord).filter_by(question_id=question.id).all()
    job = db.query(AnalysisJob).filter_by(question_id=question.id).first()
    calls = db.query(RunEvent).filter_by(question_id=question.id, kind="llm_call").all()
    models: dict[str, dict] = {}
    for c in calls:
        d = c.detail_json or {}
        key = f"{d.get('provider')}:{d.get('model')}"
        m = models.setdefault(key, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "fallback_calls": 0})
        m["calls"] += 1
        m["tokens_in"] += int(d.get("tokens_in") or 0)
        m["tokens_out"] += int(d.get("tokens_out") or 0)
        m["fallback_calls"] += 1 if d.get("fallback") else 0
    verifiers = sorted({(chk or {}).get("verifier_model", "") for r in reviews for chk in (r.checks_json or [])} - {""})
    statuses: dict[str, int] = {}
    for e in evidence:
        statuses[e.status] = statuses.get(e.status, 0) + 1
    return {
        "schema": 1,
        "question": {"id": question.id, "text": question.text, "normalized": normalize_question(question.text),
                     "route": question.route, "trigger": question.trigger,
                     "rerun_of_question_id": question.rerun_of_question_id},
        "dataset": {"id": question.dataset_id, "filename": dataset.filename if dataset else None,
                    "version_id": version.id if version else None,
                    "version_number": version.version_number if version else None,
                    "content_sha256": fingerprint(db, version)},
        "plans": [{"id": p.id, "revision": p.revision, "steps": len(p.steps_json or []),
                   "sha256": text_sha256(str(p.steps_json))} for p in plans],
        "executions": [{"execution_log_id": l.id, "step_index": l.step_index, "attempt": l.attempt_number,
                        "success": bool(l.success), "code_sha256": text_sha256(l.code)} for l in logs],
        "reviews": [{"id": r.id, "verdict": r.verdict, "confidence": r.confidence} for r in reviews],
        "verifier_models": verifiers,
        "models": models,
        "usage": usage or {},
        "versions": {"app": config.APP_VERSION, "git_sha": git_sha(), "prompts": prompt_version()},
        "environment": environment(),
        "config": {"max_executor_retries": config.MAX_EXECUTOR_RETRIES,
                   "max_critic_revisions": config.MAX_CRITIC_REVISIONS,
                   "gemini_model": config.GEMINI_MODEL, "llama_model": config.LLAMA_MODEL,
                   "failover_enabled": config.LLM_FAILOVER_ENABLED,
                   "verify_rel_tolerance": config.VERIFY_REL_TOLERANCE},
        "outcome": {"status": question.status, "job_state": job.state if job else None,
                    "failure_class": job.failure_class if job else None,
                    "dashboard_id": dash.id if dash else None,
                    "verdict_state": dash.verdict_state if dash else None,
                    "kpis": [{"label": k.get("label"), "value": k.get("value")} for k in (dash.kpis_json or [])] if dash else [],
                    "evidence": statuses},
        "timestamps": {"created_at": question.created_at.isoformat() if question.created_at else None,
                       "started_at": job.started_at.isoformat() if job and job.started_at else None,
                       "finished_at": job.finished_at.isoformat() if job and job.finished_at else None},
        "reproducibility_note": REPRODUCIBILITY_NOTE,
    }


def write_manifest(question_id: int, usage: dict | None = None) -> None:
    """Write the manifest for a finished run. A second call is a no-op: the
    record of a run is never rewritten."""
    db = SessionLocal()
    try:
        if db.query(RunManifest.id).filter_by(question_id=question_id).first():
            return
        question = db.get(Question, question_id)
        if question is None:
            return
        manifest = build_manifest(db, question, usage)
        db.add(RunManifest(team_id=question.team_id, question_id=question_id,
                           dataset_version_id=question.dataset_version_id,
                           dataset_fingerprint=manifest["dataset"]["content_sha256"], manifest_json=manifest))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_manifest(db: Session, question: Question) -> dict:
    """The stored manifest; for a run that predates manifests, one built now
    from what is still on record (marked as reconstructed, and not stored)."""
    row = db.query(RunManifest).filter_by(question_id=question.id).first()
    if row is not None:
        return {**(row.manifest_json or {}), "recorded_at": row.created_at.isoformat(), "reconstructed": False}
    return {**build_manifest(db, question), "recorded_at": None, "reconstructed": True}


# ---------------------------------------------------------------- comparison --

def _code_set(manifest: dict) -> list[str]:
    return sorted({e["code_sha256"] for e in manifest.get("executions", []) if e.get("success")})


def compare_runs(db: Session, a: Question, b: Question) -> dict:
    """What differs between two runs, and what the difference can be attributed to."""
    ma, mb = get_manifest(db, a), get_manifest(db, b)
    changed = {
        "data": ma["dataset"].get("content_sha256") != mb["dataset"].get("content_sha256"),
        "question": ma["question"]["normalized"] != mb["question"]["normalized"],
        "prompts": ma["versions"].get("prompts") != mb["versions"].get("prompts"),
        "application": (ma["versions"].get("app"), ma["versions"].get("git_sha"))
                       != (mb["versions"].get("app"), mb["versions"].get("git_sha")),
        "models": sorted(ma.get("models", {})) != sorted(mb.get("models", {})),
        "configuration": ma.get("config") != mb.get("config"),
        "code": _code_set(ma) != _code_set(mb),
        "environment": ma.get("environment") != mb.get("environment"),
        "route": ma["question"].get("route") != mb["question"].get("route"),
    }
    kpis_a, kpis_b = ma["outcome"].get("kpis") or [], mb["outcome"].get("kpis") or []
    changes = diff_kpis(kpis_a, kpis_b, threshold_pct=0.0)
    labels_a, labels_b = {_norm(k.get("label")) for k in kpis_a}, {_norm(k.get("label")) for k in kpis_b}
    only_a = [k.get("label") for k in kpis_a if _norm(k.get("label")) not in labels_b]
    only_b = [k.get("label") for k in kpis_b if _norm(k.get("label")) not in labels_a]
    results_differ = bool(changes or only_a or only_b)
    inputs_changed = [k for k in ("prompts", "application", "models", "configuration", "environment") if changed[k]]

    if not results_differ:
        attribution, explanation = "no_difference", "Both runs report the same KPI values."
    elif changed["question"]:
        attribution, explanation = "different_question", "The two runs answer different questions, so their results are not comparable."
    elif changed["data"] and not changed["code"] and not inputs_changed:
        attribution = "data"
        explanation = ("The same code ran on different data, with the same prompts, models and configuration. "
                       "The differences are attributable to the data.")
    elif changed["data"] and not inputs_changed:
        attribution = "data_and_execution"
        explanation = ("The data changed, and the generated code also differs between the runs. The data is the "
                       "likely cause, but part of the difference may come from the model writing different code.")
    elif not changed["data"]:
        attribution = "execution_or_model"
        why = ", ".join(inputs_changed) if inputs_changed else "non-deterministic model output"
        explanation = (f"Both runs used byte-identical data, so the data does not explain the difference. "
                       f"It comes from the execution side: {why}.")
    else:
        attribution = "mixed"
        explanation = ("Both the data and the execution side (" + ", ".join(inputs_changed) + ") changed. "
                       "The difference cannot be attributed to one of them from these two runs alone.")
    return {
        "a": {"question_id": a.id, "manifest": ma}, "b": {"question_id": b.id, "manifest": mb},
        "changed": changed, "results_differ": results_differ,
        "kpi_changes": [{"label": c.label, "a": c.old, "b": c.new, "pct": c.pct} for c in changes],
        "kpis_only_in_a": only_a, "kpis_only_in_b": only_b,
        "attribution": attribution, "explanation": explanation,
        "note": REPRODUCIBILITY_NOTE,
    }


# --------------------------------------------------------------------- rerun --

def rerun(db: Session, question: Question, *, user_id: int | None, dataset_version_id: int | None = None,
          idempotency_key: str | None = None):
    """Submit the same question again, pinned to a dataset version (default: the
    version the original ran on). The original run's records are not touched."""
    from app import jobs

    dataset = db.get(Dataset, question.dataset_id)
    if dataset is None:
        raise ValueError("The dataset of this analysis no longer exists")
    version_id = dataset_version_id or question.dataset_version_id
    if version_id is not None:
        version = db.get(DatasetVersion, version_id)
        if version is None or version.dataset_id != dataset.id:
            raise ValueError("That dataset version does not belong to this analysis's dataset")
    return jobs.submit_question(
        db, team_id=question.team_id, dataset=dataset, text=question.text, created_by=user_id,
        idempotency_key=idempotency_key, dataset_version_id=version_id,
        rerun_of_question_id=question.id, stage_detail="Rerun queued",
    )


# ----------------------------------------------------------------- retention --

def retention_sweep() -> int:
    """Delete old sandbox workspaces. Only files on disk: plans, execution logs,
    evidence, manifests and dashboards are never removed by retention."""
    from app.sandbox.runner import sweep_workspaces

    db = SessionLocal()
    try:
        active = {row[0] for row in db.query(AnalysisJob.question_id)
                  .filter(AnalysisJob.state.in_(("queued", "running"))).all()}
    finally:
        db.close()
    return sweep_workspaces(config.RETENTION_WORKSPACE_DAYS, active)
