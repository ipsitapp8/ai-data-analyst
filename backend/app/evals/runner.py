"""Evaluation runner: executes the versioned case suite and scores it.

Components covered
    routing                 the router's decision, including clarification
    fastpath                parse + deterministic answer vs reference numbers
    engine                  formulas and filters vs reference numbers; rejection
                            of anything that is not a formula
    verification            deliberately wrong and malformed inputs must be caught
    verification_narrative  unsupported numbers and causal wording
    investigator            a planted change must be located and must reconcile
    drift                   planted schema and distribution changes; quiet on
                            identical data; skipped on small samples
    scenario                what-if arithmetic and input validation
    pipeline                the whole graph (router, triage, planner, executor,
                            critic, compile, verify, publish) through the job
                            system, with the model scripted and the generated
                            code really executed

Offline runs are deterministic and need no API key. Live runs (`live=True`
cases, `mode="live"`) call the configured providers and are scored with
tolerances and allowed outcomes, never by exact output.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import time
import uuid
from pathlib import Path

from app import config, runtime

SUITE_PATH = Path(__file__).resolve().parents[2] / "evals" / "cases" / "suite.json"
FIXTURES = Path(__file__).resolve().parents[2] / "evals" / "fixtures"
COMPONENTS = ("routing", "fastpath", "engine", "verification", "verification_narrative", "investigator",
              "drift", "scenario", "pipeline")


def load_suite(path: Path | None = None) -> dict:
    suite = json.loads((path or SUITE_PATH).read_text(encoding="utf-8"))
    ids = [c["id"] for c in suite["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case ids in the evaluation suite")
    return suite


def _close(a, b, tol: float, rel: float = 0.0) -> bool:
    """`tol` is an ABSOLUTE tolerance (a case's "tolerance"); `rel` a relative
    one (a case's "rel_tolerance", used only by live cases, where the model
    chooses how to round)."""
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=max(tol, 1e-9))


def _profile(fixture: str) -> tuple[dict, str]:
    from app.profiling import load_and_profile_csv

    path = str(FIXTURES / fixture)
    return load_and_profile_csv(path)[1], path


class _Check:
    """Collects named pass/fail checks for one case."""

    def __init__(self):
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(i["ok"] for i in self.items)


# ------------------------------------------------------------ pure components --

def _run_routing(case: dict, chk: _Check) -> dict:
    from app.agents import router

    profile, _ = _profile(case["fixture"])
    resolution = case.get("resolution") or {"metrics": [], "ambiguous": [], "dimensions": []}
    decision = router.decide(case["question"], profile, resolution)
    chk.add("route", decision["route"] == case["expect"]["route"],
            f"got {decision['route']}, expected {case['expect']['route']}")
    if "options" in case["expect"]:
        n = len((decision.get("clarification") or {}).get("options", []))
        chk.add("clarification_options", n == case["expect"]["options"], f"{n} options")
    return {"route": decision["route"], "reasons": decision["reasons"]}


def _run_fastpath(case: dict, chk: _Check) -> dict:
    from app import fastpath

    profile, path = _profile(case["fixture"])
    spec = fastpath.parse(case["question"], profile)
    chk.add("parsed", spec is not None and spec["kind"] == "spec", f"parse result: {spec and spec.get('kind')}")
    if spec is None or spec["kind"] != "spec":
        return {}
    out = fastpath.compute(path, profile, spec)
    tol, expect = case.get("tolerance", 0.0), case["expect"]
    engine = out["engine"]
    if "value" in expect:
        chk.add("value", _close(engine.get("value"), expect["value"], tol),
                f"engine {engine.get('value')}, reference {expect['value']}")
    if "rows" in expect:
        chk.add("rows", engine.get("rows") == expect["rows"], f"{engine.get('rows')} rows")
    if "groups" in expect:
        dim = spec["group_by"][0]
        got = {str(g["key"][dim]): g["value"] for g in engine.get("groups", [])}
        chk.add("group_keys", set(got) == set(expect["groups"]), f"keys {sorted(got)}")
        for key, value in expect["groups"].items():
            chk.add(f"group:{key}", _close(got.get(key), value, tol), f"engine {got.get(key)}, reference {value}")
    chk.add("independent_reference_ran", out["reference"] is not None, str(out["reference_error"] or ""))
    return {"formula": spec["formula"], "group_by": spec["group_by"]}


def _run_engine(case: dict, chk: _Check) -> dict:
    from app import metric_engine as me

    _, path = _profile(case["fixture"])
    if case["expect"].get("error"):
        try:
            me.parse_formula(case["formula"])
            chk.add("rejected", False, "the formula was accepted")
        except me.FormulaError as e:
            chk.add("rejected", True, str(e))
        return {}
    df = me.load_frame(path)
    value = me.evaluate(df, case["formula"], case.get("filters"))["value"]
    ref = me.reference_evaluate(path, case["formula"], case.get("filters"))["value"]
    tol = case.get("tolerance", 0.0)
    chk.add("value", _close(value, case["expect"]["value"], tol), f"engine {value}, reference {case['expect']['value']}")
    chk.add("second_implementation", _close(ref, case["expect"]["value"], tol), f"plain-Python path {ref}")
    return {"value": value}


def _dataset_ev() -> dict:
    return {"version_id": 1, "fingerprint": "f" * 64, "fingerprint_now": "f" * 64, "row_count": 1000}


def _check_evidence(record: dict, expect: dict, chk: _Check) -> None:
    chk.add("status", record["status"] == expect["status"], f"got {record['status']}, expected {expect['status']}")
    by_name = {c["check"]: c for c in record["checks"]}
    if "failed_check" in expect:
        c = by_name.get(expect["failed_check"], {})
        chk.add("failed_check", c.get("outcome") == "fail", f"{expect['failed_check']}: {c.get('outcome')}")
    if "not_run_check" in expect:
        c = by_name.get(expect["not_run_check"], {})
        chk.add("not_run_is_not_pass", c.get("outcome") == "not_run", f"{expect['not_run_check']}: {c.get('outcome')}")
    if "causal" in expect:
        chk.add("causal_validity", record["validity"]["causal"] == expect["causal"], record["validity"]["causal"])


def _run_verification(case: dict, chk: _Check) -> dict:
    from app import verification

    steps = {int(k): v for k, v in case["steps"].items()}
    record = verification.verify_kpi({**case["kpi"], "element_id": "e1"}, steps, case["critic"], _dataset_ev())
    _check_evidence(record, case["expect"], chk)
    return {"status": record["status"], "reasons": record["reasons"]}


def _run_verification_narrative(case: dict, chk: _Check) -> dict:
    from app import verification

    steps = {int(k): v for k, v in case["steps"].items()}
    record = verification.verify_narrative(case["narrative"], "n1", case["kpis"], steps, case["critic"], _dataset_ev())
    _check_evidence(record, case["expect"], chk)
    return {"status": record["status"], "reasons": record["reasons"]}


def _run_investigator(case: dict, chk: _Check) -> dict:
    from app import investigator
    from app import metric_engine as me

    profile, path = _profile(case["fixture"])
    report = investigator.investigate(me.load_frame(path), profile, case["spec"])
    expect, tol = case["expect"], case.get("tolerance", 0.0)
    if "status" in expect:
        chk.add("status", report["status"] == expect["status"], report["status"])
        return {"status": report["status"]}
    chk.add("total_a", _close(report["totals"]["a"], expect["total_a"], tol), str(report["totals"]["a"]))
    chk.add("total_b", _close(report["totals"]["b"], expect["total_b"], tol), str(report["totals"]["b"]))
    top = report["ranked_contributors"][0] if report["ranked_contributors"] else {}
    chk.add("top_contributor", (top.get("dimension"), top.get("segment")) == (expect["top_dimension"], expect["top_segment"]),
            f"{top.get('dimension')} = {top.get('segment')}")
    chk.add("top_contribution", _close(top.get("contribution"), expect["top_contribution"], tol), str(top.get("contribution")))
    dim = next((d for d in report["dimensions"] if d["dimension"] == expect["top_dimension"]), {})
    chk.add("reconciles", dim.get("reconciles") is expect["reconciles"], f"explained {dim.get('explained')}")
    chk.add("price_volume_reconciles", bool(report["price_volume"] and report["price_volume"]["reconciles"]),
            str(report["price_volume"] and report["price_volume"]["reconciles"]))
    chk.add("states_caveats", bool(report["caveats"]) and bool(report["alternative_explanations"]), "")
    chk.add("bounded", report["budget"]["engine_calls"] <= report["budget"]["limits"]["engine_calls"],
            f"{report['budget']['engine_calls']} engine calls")
    return {"status": report["status"], "summary": report["summary"], "budget": report["budget"]}


def _run_drift(case: dict, chk: _Check) -> dict:
    from app import data_quality
    from app.profiling import load_and_profile_csv

    base_df = load_and_profile_csv(str(FIXTURES / case["base"]))[0]
    new_df = load_and_profile_csv(str(FIXTURES / case["new"]))[0]
    if case.get("head"):
        new_df = new_df.head(case["head"])
    result = data_quality.compare(data_quality.build_snapshot(new_df), data_quality.build_snapshot(base_df),
                                  data_quality.default_thresholds())
    found = {(f["category"], f["check"], f["column"]) for f in result["findings"]}
    expect = case["expect"]
    for cat, check, column in expect.get("findings", []):
        chk.add(f"{check}:{column}", (cat, check, column) in found, "found" if (cat, check, column) in found else "missing")
    if expect.get("no_findings"):
        chk.add("quiet_on_identical_data", not found, f"{len(found)} findings")
    if expect.get("no_distribution_findings"):
        chk.add("no_distribution_findings", not any(c == "distribution" for c, _, _ in found), "")
    if expect.get("skipped"):
        chk.add("skipped_small_sample", bool(result["skipped"]), f"{len(result['skipped'])} skipped")
    return {"findings": sorted(f"{c}/{k}/{col}" for c, k, col in found), "skipped": len(result["skipped"])}


def _run_scenario(case: dict, chk: _Check) -> dict:
    from app import scenarios

    expect, tol = case["expect"], case.get("tolerance", 0.0)
    try:
        out = scenarios.evaluate(case["model"], case["baseline"], case["scenarios"])
    except scenarios.ScenarioError as e:
        chk.add("rejected", bool(expect.get("error")), str(e))
        return {}
    if expect.get("error"):
        chk.add("rejected", False, "an impossible input was accepted")
        return {}
    chk.add("baseline_profit", _close(out["baseline"]["outputs"]["profit"], expect["baseline_profit"], tol), "")
    s = out["scenarios"]["price_up"]
    chk.add("scenario_profit", _close(s["outputs"]["profit"], expect["price_up_profit"], tol), str(s["outputs"]["profit"]))
    chk.add("contributions_sum_to_change", _close(sum(s["contributions"]["by_input"].values()), expect["contribution_sum"], 1e-9), "")
    chk.add("labelled_as_scenario", out["kind"] == "scenario" and bool(out["disclaimer"]), "")
    return {"profit": s["outputs"]["profit"]}


# ---------------------------------------------------------------- pipeline --

def _eval_workspace(db) -> tuple[int, int]:
    """A dedicated team and user that own everything the evaluation creates."""
    from app.models import Community, Team, TeamMember, User
    from app.security import hash_password

    user = db.query(User).filter_by(email="evals@system.invalid").first()
    if user is None:
        user = User(email="evals@system.invalid", password_hash=hash_password(uuid.uuid4().hex),
                    display_name="Evaluation runner")
        db.add(user)
        db.flush()
    team = db.query(Team).filter_by(name="__evals__").first()
    if team is None:
        community = Community(name="__evals__", created_by=user.id)
        db.add(community)
        db.flush()
        team = Team(community_id=community.id, name="__evals__", created_by=user.id)
        db.add(team)
        db.flush()
        db.add(TeamMember(team_id=team.id, user_id=user.id, role="owner", status="active"))
    db.commit()
    return team.id, user.id


def _eval_dataset(db, team_id: int, fixture: str):
    from app import provenance
    from app.dataset_versions import add_version
    from app.models import Dataset
    from app.profiling import load_and_profile_csv

    name = f"eval::{fixture}"
    ds = db.query(Dataset).filter_by(team_id=team_id, filename=name).first()
    if ds is not None:
        return ds
    dest = config.UPLOADS_DIR / f"eval_{uuid.uuid4().hex}_{fixture}"
    shutil.copyfile(FIXTURES / fixture, dest)
    profile = load_and_profile_csv(str(dest))[1]
    ds = Dataset(team_id=team_id, filename=name, filepath=str(dest), row_count=profile["row_count"],
                 col_count=profile["col_count"], profile_json=profile)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    version = add_version(db, ds, filename=name, filepath=str(dest), profile=profile)
    provenance.fingerprint(db, version)
    return ds


def _run_pipeline(case: dict, chk: _Check, live: bool) -> dict:
    from app import jobs
    from app.database import SessionLocal
    from app.evals.scripted import ScriptedProvider
    from app.models import (AnalysisJob, Dashboard, EvidenceRecord, ExecutionLog, Plan, Question, RunEvent)
    from app.change_detection import parse_kpi_value

    db = SessionLocal()
    try:
        team_id, user_id = _eval_workspace(db)
        dataset = _eval_dataset(db, team_id, case["fixture"])
        question, job, _ = jobs.submit_question(db, team_id=team_id, dataset=dataset, text=case["question"],
                                                created_by=user_id, enforce_admission=False,
                                                stage_detail="Evaluation run")
        question_id, job_id = question.id, job.id
    finally:
        db.close()

    provider = None
    if live:
        state = jobs.run_inline(job_id)
    else:
        provider = ScriptedProvider(case.get("script") or {})
        with runtime.provider_override(provider):
            state = jobs.run_inline(job_id)

    db = SessionLocal()
    try:
        q = db.get(Question, question_id)
        job = db.get(AnalysisJob, job_id)
        dash = db.query(Dashboard).filter_by(question_id=question_id).order_by(Dashboard.id.desc()).first()
        evidence = db.query(EvidenceRecord).filter_by(question_id=question_id).all()
        logs = db.query(ExecutionLog).filter(ExecutionLog.question_id == question_id, ExecutionLog.step_index >= 0).all()
        plans = db.query(Plan).filter_by(question_id=question_id).count()
        events = db.query(RunEvent).filter_by(question_id=question_id).all()
        usage = next((e.detail_json for e in events if e.kind == "budget"), {}) or {}
        sandbox_events = [e for e in events if e.kind == "sandbox_run"]
        kpis = {k.get("label"): parse_kpi_value(str(k.get("value"))) for k in (dash.kpis_json if dash else [])}
        kpi_evidence = {}
        for k in (dash.kpis_json if dash else []):
            rec = next((e for e in evidence if e.element_id == k.get("element_id")), None)
            kpi_evidence[k.get("label")] = rec.status if rec else None
        expect, tol = case["expect"], case.get("tolerance", 0.0)
        llm_calls = int(usage.get("llm_calls", provider.count() if provider else 0))
        steps_attempted = len({l.step_index for l in logs})
        retries = max(0, len(logs) - steps_attempted * max(1, plans)) if logs else 0

        chk.add("job_terminal", state in jobs.TERMINAL and job.state == state, f"job state {job.state}")
        if "status" in expect:
            chk.add("status", q.status == expect["status"], f"got {q.status}, expected {expect['status']}")
        if "status_in" in expect:
            chk.add("status_allowed", q.status in expect["status_in"], q.status)
        if "verdict_state" in expect:
            chk.add("verdict_state", bool(dash) and dash.verdict_state == expect["verdict_state"],
                    f"got {dash.verdict_state if dash else None}")
        if "route" in expect:
            chk.add("route", q.route == expect["route"], f"got {q.route}")
        for label, value in (expect.get("kpi_values") or {}).items():
            chk.add(f"kpi:{label}", _close(kpis.get(label), value, tol), f"reported {kpis.get(label)}, reference {value}")
        for label, status in (expect.get("kpi_evidence") or {}).items():
            chk.add(f"evidence:{label}", kpi_evidence.get(label) == status, f"got {kpi_evidence.get(label)}")
        for label, status in (expect.get("kpi_evidence_not") or {}).items():
            chk.add(f"not_{status}:{label}", kpi_evidence.get(label) not in (None, status),
                    f"evidence status {kpi_evidence.get(label)}")
        for label, truth in (expect.get("kpi_wrong") or {}).items():
            chk.add(f"is_actually_wrong:{label}", not _close(kpis.get(label), truth, tol),
                    f"reported {kpis.get(label)}, truth {truth}")
        if "any_kpi_value" in expect:
            rel = case.get("rel_tolerance", 0.0)
            chk.add("a_kpi_matches_reference", any(_close(v, expect["any_kpi_value"], tol, rel) for v in kpis.values()),
                    f"reported {sorted(v for v in kpis.values() if v is not None)}")
        if "charts" in expect:
            chk.add("charts", bool(dash) and len(dash.charts_json or []) == expect["charts"], "")
        if "max_llm_calls" in expect:
            chk.add("bounded_model_calls", llm_calls <= expect["max_llm_calls"], f"{llm_calls} calls")
        if "retries" in expect:
            chk.add("retries", retries == expect["retries"], f"{retries} retries")
        if "revisions" in expect:
            chk.add("revisions", plans - 1 == expect["revisions"], f"{plans - 1} revisions")
        failed_checks = sum(1 for e in evidence for c in (e.checks_json or []) if c.get("outcome") == "fail")
        return {
            "question_id": question_id, "status": q.status, "job_state": job.state, "route": q.route,
            "verdict_state": dash.verdict_state if dash else None, "llm_calls": llm_calls,
            "sandbox_runs": len(sandbox_events), "sandbox_timeouts": sum(1 for e in sandbox_events if (e.detail_json or {}).get("timed_out")),
            "retries": retries, "revisions": max(0, plans - 1), "verification_failed_checks": failed_checks,
            "tokens_in": int(usage.get("tokens_in", 0)), "tokens_out": int(usage.get("tokens_out", 0)),
            "estimated_cost_usd": float(usage.get("estimated_cost_usd", 0.0)),
            "providers": usage.get("providers", []), "execution_success": bool(logs) and all(
                any(l.success for l in logs if l.step_index == s) for s in {l.step_index for l in logs}),
        }
    finally:
        db.close()


_RUNNERS = {
    "routing": _run_routing, "fastpath": _run_fastpath, "engine": _run_engine,
    "verification": _run_verification, "verification_narrative": _run_verification_narrative,
    "investigator": _run_investigator, "drift": _run_drift, "scenario": _run_scenario,
}


def run_case(case: dict, live: bool = False) -> dict:
    chk = _Check()
    started = time.monotonic()
    metrics: dict = {}
    error = None
    try:
        if case["component"] == "pipeline":
            metrics = _run_pipeline(case, chk, live)
        else:
            metrics = _RUNNERS[case["component"]](case, chk)
    except Exception as e:  # noqa: BLE001 - a crashing case is a failed case, with the reason kept
        error = f"{type(e).__name__}: {e}"
        chk.add("ran_without_error", False, error)
    latency = int((time.monotonic() - started) * 1000)
    return {"case_id": case["id"], "component": case["component"], "passed": chk.passed,
            "metrics": {**(metrics or {}), "latency_ms": latency},
            "detail": {"checks": chk.items, "error": error, "note": case.get("note", "")}}


def run_suite(mode: str = "offline", components: list[str] | None = None, case_ids: list[str] | None = None,
              provider_label: str | None = None) -> dict:
    """Run the suite. `mode` is "offline" (scripted model; deterministic) or
    "live" (real providers; only cases marked live)."""
    if mode not in ("offline", "live"):
        raise ValueError("mode must be offline or live")
    suite = load_suite()
    live = mode == "live"
    cases = [c for c in suite["cases"] if bool(c.get("live")) == live]
    if components:
        cases = [c for c in cases if c["component"] in components]
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    started = dt.datetime.utcnow()
    results = [run_case(c, live=live) for c in cases]
    return {"suite_version": suite["suite_version"], "mode": mode,
            "provider": provider_label or ("scripted" if not live else f"gemini:{config.GEMINI_MODEL}"),
            "started_at": started.isoformat(), "finished_at": dt.datetime.utcnow().isoformat(),
            "summary": summarize(results), "results": results}


def summarize(results: list[dict]) -> dict:
    by_component: dict[str, dict] = {}
    for r in results:
        c = by_component.setdefault(r["component"], {"cases": 0, "passed": 0})
        c["cases"] += 1
        c["passed"] += 1 if r["passed"] else 0
    pipe = [r["metrics"] for r in results if r["component"] == "pipeline"]
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "cases": total, "passed": passed, "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "by_component": by_component,
        "pipeline": {
            "runs": len(pipe),
            "execution_success": sum(1 for m in pipe if m.get("execution_success")),
            "llm_calls": sum(m.get("llm_calls", 0) for m in pipe),
            "sandbox_runs": sum(m.get("sandbox_runs", 0) for m in pipe),
            "sandbox_timeouts": sum(m.get("sandbox_timeouts", 0) for m in pipe),
            "retries": sum(m.get("retries", 0) for m in pipe),
            "revisions": sum(m.get("revisions", 0) for m in pipe),
            "verification_failed_checks": sum(m.get("verification_failed_checks", 0) for m in pipe),
            "tokens_in": sum(m.get("tokens_in", 0) for m in pipe),
            "tokens_out": sum(m.get("tokens_out", 0) for m in pipe),
            "estimated_cost_usd": round(sum(m.get("estimated_cost_usd", 0.0) for m in pipe), 6),
            "mean_latency_ms": int(sum(m.get("latency_ms", 0) for m in pipe) / len(pipe)) if pipe else 0,
        },
        "failed_cases": [r["case_id"] for r in results if not r["passed"]],
    }


# -------------------------------------------------------------- persistence --

def store_run(db, report: dict) -> int:
    from app import provenance
    from app.models import EvalResult, EvalRun

    run = EvalRun(suite_version=report["suite_version"], mode=report["mode"], provider=report["provider"],
                  app_version=config.APP_VERSION, git_sha=provenance.git_sha(), summary_json=report["summary"],
                  started_at=dt.datetime.fromisoformat(report["started_at"]),
                  finished_at=dt.datetime.fromisoformat(report["finished_at"]))
    db.add(run)
    db.flush()
    for r in report["results"]:
        db.add(EvalResult(run_id=run.id, case_id=r["case_id"], component=r["component"], passed=r["passed"],
                          metrics_json=r["metrics"], detail_json=r["detail"]))
    db.commit()
    return run.id


def run_out(run, results: list | None = None) -> dict:
    out = {"id": run.id, "suite_version": run.suite_version, "mode": run.mode, "provider": run.provider,
           "app_version": run.app_version, "git_sha": run.git_sha, "summary": run.summary_json or {},
           "started_at": run.started_at, "finished_at": run.finished_at}
    if results is not None:
        out["results"] = [{"case_id": r.case_id, "component": r.component, "passed": r.passed,
                           "metrics": r.metrics_json or {}, "detail": r.detail_json or {}} for r in results]
    return out


def compare(base: dict, new: dict) -> dict:
    """Regression report between two stored runs (dicts from `run_out` with results)."""
    b = {r["case_id"]: r for r in base["results"]}
    n = {r["case_id"]: r for r in new["results"]}
    regressions = sorted(c for c in n if c in b and b[c]["passed"] and not n[c]["passed"])
    fixed = sorted(c for c in n if c in b and not b[c]["passed"] and n[c]["passed"])
    metric_deltas = {}
    for key in ("llm_calls", "retries", "mean_latency_ms", "estimated_cost_usd", "verification_failed_checks"):
        old = (base["summary"].get("pipeline") or {}).get(key)
        cur = (new["summary"].get("pipeline") or {}).get(key)
        if old is not None and cur is not None:
            metric_deltas[key] = {"base": old, "new": cur, "delta": round(cur - old, 6)}
    return {
        "base": {"id": base["id"], "suite_version": base["suite_version"], "git_sha": base["git_sha"],
                 "provider": base["provider"], "pass_rate": base["summary"].get("pass_rate")},
        "new": {"id": new["id"], "suite_version": new["suite_version"], "git_sha": new["git_sha"],
                "provider": new["provider"], "pass_rate": new["summary"].get("pass_rate")},
        "comparable": base["suite_version"] == new["suite_version"] and base["mode"] == new["mode"],
        "regressions": regressions, "fixed": fixed,
        "added_cases": sorted(set(n) - set(b)), "removed_cases": sorted(set(b) - set(n)),
        "metric_deltas": metric_deltas, "has_regression": bool(regressions),
    }


def to_markdown(report: dict, comparison: dict | None = None) -> str:
    s = report["summary"]
    lines = [f"# Evaluation report — suite {report['suite_version']} ({report['mode']}, {report['provider']})", "",
             f"{s['passed']} of {s['cases']} cases passed.", "",
             "| Component | Passed | Cases |", "| --- | --- | --- |"]
    for comp, v in sorted(s["by_component"].items()):
        lines.append(f"| {comp} | {v['passed']} | {v['cases']} |")
    p = s["pipeline"]
    if p["runs"]:
        lines += ["", "## Pipeline metrics", "", "| Metric | Value |", "| --- | --- |",
                  f"| Runs | {p['runs']} |", f"| Model calls | {p['llm_calls']} |",
                  f"| Sandbox runs | {p['sandbox_runs']} |", f"| Executor retries | {p['retries']} |",
                  f"| Plan revisions | {p['revisions']} |",
                  f"| Failed verification checks | {p['verification_failed_checks']} |",
                  f"| Tokens in / out | {p['tokens_in']} / {p['tokens_out']} |",
                  f"| Estimated cost (USD) | {p['estimated_cost_usd']} |",
                  f"| Mean latency (ms) | {p['mean_latency_ms']} |"]
    if s["failed_cases"]:
        lines += ["", "## Failed cases", ""]
        for r in report["results"]:
            if not r["passed"]:
                bad = "; ".join(f"{c['check']} ({c['detail']})" for c in r["detail"]["checks"] if not c["ok"])
                lines.append(f"- `{r['case_id']}`: {bad}")
    if comparison is not None:
        lines += ["", f"## Compared with run {comparison['base']['id']}", "",
                  f"- Regressions: {', '.join(comparison['regressions']) or 'none'}",
                  f"- Fixed: {', '.join(comparison['fixed']) or 'none'}"]
    return "\n".join(lines) + "\n"
