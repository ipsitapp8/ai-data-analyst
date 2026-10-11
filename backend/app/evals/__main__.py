"""Evaluation CLI.

    python -m app.evals run                      offline suite (scripted model)
    python -m app.evals run --component pipeline
    python -m app.evals run --live --provider gemini|llama   real providers
    python -m app.evals list                     stored runs
    python -m app.evals compare BASE NEW         regression report (exit 1 on regression)
    python -m app.evals report RUN               markdown report of a stored run

Run from the `backend/` directory. Offline runs execute in a throwaway database
and data directory, so they never touch real datasets; only the run's results
are written to the application database (skip that with --no-store).

Offline pipeline cases really execute the scripted Python. They use the
configured sandbox; on a machine without Docker pass --unsafe-subprocess to run
them with the development runner (the scripts are the fixed ones in the suite).

Exit code: 0 when every case passed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _real_database_url() -> str:
    """The application's own database, resolved before the environment is
    redirected to a scratch one."""
    from dotenv import dotenv_values

    root = Path(__file__).resolve().parents[3]
    env = {**dotenv_values(root / ".env"), **os.environ}
    if env.get("DATABASE_URL"):
        return env["DATABASE_URL"]
    data_dir = Path(env.get("DATA_DIR") or root / "backend" / "data")
    return f"sqlite:///{Path(env.get('DB_PATH') or data_dir / 'app.db').as_posix()}"


def _store_session(url: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kwargs)
    from app.models import EvalResult, EvalRun

    if url.startswith("sqlite"):  # Postgres gets these tables from the Alembic migration
        EvalRun.__table__.create(engine, checkfirst=True)
        EvalResult.__table__.create(engine, checkfirst=True)
    return sessionmaker(bind=engine)()


def _load_stored(db, run_id: int) -> dict:
    from app.evals import runner
    from app.models import EvalResult, EvalRun

    run = db.get(EvalRun, run_id)
    if run is None:
        sys.exit(f"No stored evaluation run with id {run_id}")
    return runner.run_out(run, db.query(EvalResult).filter_by(run_id=run_id).order_by(EvalResult.id).all())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evals", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the suite")
    run_p.add_argument("--live", action="store_true", help="call real providers (costs money; never in CI)")
    run_p.add_argument("--provider", choices=["gemini", "llama"], default="gemini", help="primary provider for --live")
    run_p.add_argument("--component", action="append", help="only this component (repeatable)")
    run_p.add_argument("--case", action="append", help="only this case id (repeatable)")
    run_p.add_argument("--unsafe-subprocess", action="store_true",
                       help="run scripted code with the development runner instead of Docker")
    run_p.add_argument("--no-store", action="store_true", help="do not write the run to the application database")
    run_p.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    run_p.add_argument("--markdown", metavar="PATH", help="also write a markdown report")
    sub.add_parser("list", help="list stored runs")
    cmp_p = sub.add_parser("compare", help="compare two stored runs")
    cmp_p.add_argument("base", type=int)
    cmp_p.add_argument("new", type=int)
    rep_p = sub.add_parser("report", help="markdown report of a stored run")
    rep_p.add_argument("run", type=int)
    args = parser.parse_args(argv)

    real_url = _real_database_url()
    if args.command == "run":
        scratch = Path(tempfile.mkdtemp(prefix="aida_evals_"))
        os.environ.update({"DATABASE_URL": "", "DB_PATH": str(scratch / "evals.db"), "DATA_DIR": str(scratch / "data"),
                           "UPLOADS_DIR": str(scratch / "data" / "uploads"),
                           "WORKSPACES_DIR": str(scratch / "data" / "workspaces"),
                           "SCHEDULER_ENABLED": "false", "JOB_WORKERS": "0", "RAG_ENABLED": "false"})
        os.environ.setdefault("JWT_SECRET_KEY", "evaluation-run-only-secret-key-32-bytes!")
        if args.unsafe_subprocess:
            os.environ.update({"SANDBOX_BACKEND": "subprocess", "SANDBOX_ALLOW_UNSAFE_SUBPROCESS": "true"})
        if args.live and args.provider == "llama":
            os.environ["GEMINI_API_KEY"] = ""  # the provider layer then serves every call from Llama
        from app import config
        from app.database import init_db
        from app.evals import runner
        from app.sandbox.runner import sandbox_status

        init_db()
        if args.live and not (config.GEMINI_API_KEY or config.LLAMA_API_KEY):
            sys.exit("--live needs GEMINI_API_KEY or LLAMA_API_KEY")
        status = sandbox_status()
        print(f"sandbox: {status['backend']} ({status['isolation']})" + ("" if status["ready"] else f" -- NOT READY: {status['reason']}"))
        label = None
        if args.live:
            label = f"llama:{config.LLAMA_MODEL}" if args.provider == "llama" else f"gemini:{config.GEMINI_MODEL}"
        report = runner.run_suite("live" if args.live else "offline", args.component, args.case, label)
        s = report["summary"]
        for r in report["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  {mark}  {r['component']:<24} {r['case_id']}")
            if not r["passed"]:
                for c in r["detail"]["checks"]:
                    if not c["ok"]:
                        print(f"          - {c['check']}: {c['detail']}")
        print(f"\n{s['passed']}/{s['cases']} passed  |  pipeline: {s['pipeline']['llm_calls']} model calls, "
              f"{s['pipeline']['sandbox_runs']} code runs, {s['pipeline']['retries']} retries, "
              f"est. cost ${s['pipeline']['estimated_cost_usd']}")
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        if args.markdown:
            Path(args.markdown).write_text(runner.to_markdown(report), encoding="utf-8")
        if not args.no_store:
            try:
                db = _store_session(real_url)
                try:
                    print(f"stored as evaluation run {runner.store_run(db, report)}")
                finally:
                    db.close()
            except Exception as e:  # noqa: BLE001 - results were already printed
                print(f"could not store the run in the application database: {type(e).__name__}: {e}")
        return 0 if s["failed"] == 0 and s["cases"] > 0 else 1

    from app.evals import runner
    from app.models import EvalRun

    db = _store_session(real_url)
    try:
        if args.command == "list":
            for run in db.query(EvalRun).order_by(EvalRun.id.desc()).limit(30).all():
                s = run.summary_json or {}
                print(f"{run.id:>5}  {run.started_at:%Y-%m-%d %H:%M}  {run.mode:<8} {run.provider:<28} "
                      f"suite {run.suite_version}  {s.get('passed')}/{s.get('cases')}  git {run.git_sha}")
            return 0
        if args.command == "report":
            stored = _load_stored(db, args.run)
            print(runner.to_markdown({**stored, "results": stored["results"]}))
            return 0
        comparison = runner.compare(_load_stored(db, args.base), _load_stored(db, args.new))
        print(json.dumps(comparison, indent=1, default=str))
        return 1 if comparison["has_regression"] else 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
