"""Graph node for the hypothesis-driven investigation (app/hypothesis_engine.py).

Its job is everything around the engine: load the pinned dataset version, give
the engine a database-backed store (so progress is visible while it runs and an
interrupted run resumes instead of starting over), and publish the outcome
through the same `publish_dashboard` every other route uses.

If no model can be reached at the very first step, the node hands the question
to the deterministic investigator instead of failing: the user still gets the
decomposition, and the trace records why.
"""
from __future__ import annotations

import logging
import uuid

from app import config, fastpath, hypothesis_engine, investigator, provenance, runtime, verification
from app import metric_engine as me
from app.agents import llm_client
from app.agents.dashboard import dataset_evidence_for, publish_dashboard
from app.agents.deterministic import _filter_text, _kpi, _record_step, _structured
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import DQIncident, Investigation, InvestigationAnalysis, InvestigationHypothesis, Question

logger = logging.getLogger(__name__)


class DbStore(hypothesis_engine.MemoryStore):
    """Engine state in `investigations`, `investigation_hypotheses` and
    `investigation_analyses`, written as the run proceeds."""

    def __init__(self, investigation_id: int, team_id: int):
        super().__init__()
        self.investigation_id, self.team_id = investigation_id, team_id

    def load(self) -> dict:
        db = SessionLocal()
        try:
            inv = db.get(Investigation, self.investigation_id)
            understanding = (inv.params_json or {}).get("understanding")
            hyps = (db.query(InvestigationHypothesis).filter_by(investigation_id=self.investigation_id)
                    .order_by(InvestigationHypothesis.id).all())
            analyses = (db.query(InvestigationAnalysis).filter_by(investigation_id=self.investigation_id)
                        .order_by(InvestigationAnalysis.id).all())
            return {
                "understanding": understanding,
                "hypotheses": [{**(h.spec_json or {}), **(h.assessment_json or {}), "id": h.key, "statement": h.statement,
                                "status": h.status} for h in hyps],
                "analyses": [{"id": a.key, "signature": a.signature, "op": a.op, "params": a.params_json or {},
                              "status": a.status, "result": a.result_json or {}, "error": a.error,
                              "requested_by": a.requested_by, "hypothesis_ids": list(a.hypothesis_keys_json or []),
                              "duration_ms": a.duration_ms} for a in analyses],
            }
        finally:
            db.close()

    def save_understanding(self, understanding: dict) -> None:
        db = SessionLocal()
        try:
            inv = db.get(Investigation, self.investigation_id)
            inv.params_json = {**(inv.params_json or {}), "understanding": understanding}
            db.commit()
        finally:
            db.close()

    def save_hypothesis(self, hypothesis: dict) -> None:
        spec_keys = ("measurable_prediction", "analysis", "prediction", "required_columns", "confounders", "priority", "rationale")
        db = SessionLocal()
        try:
            row = (db.query(InvestigationHypothesis)
                   .filter_by(investigation_id=self.investigation_id, key=hypothesis["id"]).first())
            if row is None:
                row = InvestigationHypothesis(team_id=self.team_id, investigation_id=self.investigation_id,
                                              key=hypothesis["id"], statement=hypothesis["statement"])
                db.add(row)
            row.spec_json = {k: hypothesis.get(k) for k in spec_keys}
            row.status = hypothesis["status"]
            row.assessment_json = {"evidence": hypothesis.get("evidence") or [], "reasons": hypothesis.get("reasons") or [],
                                   "effect": hypothesis.get("effect") or {}, "follow_ups": hypothesis.get("follow_ups") or []}
            db.commit()
        finally:
            db.close()

    def save_analysis(self, analysis: dict) -> None:
        db = SessionLocal()
        try:
            row = (db.query(InvestigationAnalysis)
                   .filter_by(investigation_id=self.investigation_id, key=analysis["id"]).first())
            if row is None:
                row = InvestigationAnalysis(team_id=self.team_id, investigation_id=self.investigation_id, key=analysis["id"],
                                            signature=analysis["signature"], op=str(analysis["op"]), status=analysis["status"])
                db.add(row)
            row.params_json, row.status = analysis["params"], analysis["status"]
            row.result_json, row.error = analysis["result"], analysis["error"]
            row.hypothesis_keys_json = list(analysis["hypothesis_ids"])
            row.requested_by, row.duration_ms = analysis["requested_by"], analysis["duration_ms"]
            db.commit()
        finally:
            db.close()


def _investigation(question_id: int) -> tuple[int, int, int | None, list[str]]:
    """The investigation row for this question (created on first use), plus the
    open data-quality incident messages for its dataset version."""
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        row = db.query(Investigation).filter_by(question_id=question_id).first()
        if row is None:
            row = Investigation(team_id=question.team_id, dataset_id=question.dataset_id,
                                dataset_version_id=question.dataset_version_id, question_id=question_id,
                                params_json={"engine": "hypothesis", "question": question.text}, report_json={},
                                status="running")
            db.add(row)
        else:
            row.params_json = {**(row.params_json or {}), "engine": "hypothesis"}
            row.status = "running"
        db.commit()
        incidents = []
        if question.dataset_version_id is not None:
            incidents = [i.message for i in db.query(DQIncident).filter(
                DQIncident.dataset_id == question.dataset_id, DQIncident.dataset_version_id == question.dataset_version_id,
                DQIncident.status == "open", DQIncident.severity == "high").limit(10).all()]
        return row.id, question.team_id, question.dataset_version_id, incidents
    finally:
        db.close()


def _save(investigation_id: int, report: dict, status: str) -> None:
    db = SessionLocal()
    try:
        row = db.get(Investigation, investigation_id)
        row.report_json, row.status = report, status
        db.commit()
    finally:
        db.close()


def hypothesis_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "investigating", "Reading the question")
    investigation_id, team_id, version_id, dq_messages = _investigation(question_id)
    dataset_ev = dataset_evidence_for(question_id)
    ctx = runtime.current()
    try:
        df = me.load_frame(state["csv_path"])
    except (me.FormulaError, OSError) as e:
        _save(investigation_id, {"engine": "hypothesis", "error": str(e)}, "failed")
        return {"failed": True, "failure_reason": f"The investigation could not read the dataset: {e}"}

    engine = hypothesis_engine.HypothesisEngine(
        question=state["question_text"], df=df, profile=state["profile"], call_model=llm_client.call_tool,
        approved_metrics=state.get("approved_metrics") or [], dq_messages=dq_messages,
        store=DbStore(investigation_id, team_id),
        budget=hypothesis_engine.Budget.from_config(lambda: (ctx.tokens_in + ctx.tokens_out) if ctx else 0),
        progress=lambda message: update_stage(question_id, "investigating", message),
        checkpoint=runtime.checkpoint,
        provenance={"dataset_version_id": version_id, "dataset_version": dataset_ev.get("version_number"),
                    "dataset_fingerprint": dataset_ev.get("fingerprint"), "question_id": question_id,
                    "investigation_id": investigation_id, "prompt_version": provenance.prompt_version(),
                    "app_version": config.APP_VERSION},
    )
    def fall_back() -> dict:
        """No model produced a hypothesis. The deterministic investigator needs none."""
        logger.warning("Hypothesis engine: no model available; using the deterministic investigator")
        runtime.record_event("route", "root_cause", {"reasons": ["no model provider could be reached; fell back from the "
                                                                "hypothesis engine to the deterministic investigator"]})
        db = SessionLocal()
        try:
            q = db.get(Question, question_id)
            q.route = "root_cause"
            inv = db.get(Investigation, investigation_id)
            spec = investigator.infer_spec(state["profile"], state["question_text"], state.get("approved_metrics") or [])
            inv.params_json = spec or {}
            db.commit()
        finally:
            db.close()
        return {"route": "root_cause", "route_spec": spec or {}, "hypothesis_fallback": True}

    try:
        report = engine.run()
    except runtime.ProviderUnavailable:
        if engine.budget.llm_calls <= 1 and not engine.hypotheses:
            return fall_back()
        raise
    except hypothesis_engine.CannotInvestigate as e:
        report = {"engine": "hypothesis", "question": state["question_text"], "status": "inconclusive", "summary": str(e),
                  "determination": {"code": "cannot_investigate", "text": str(e)}, "hypotheses": [], "analyses": [],
                  "limitations": [str(e)], "totals": {}, "stop_reason": "cannot_investigate"}
        _save(investigation_id, report, "inconclusive")
        return {"failed": True, "failure_reason": str(e)}

    if report["stop_reason"] == "model_unavailable" and not report["hypotheses"]:
        return fall_back()      # the model answered at first, then was lost before proposing anything

    report["reproducibility"]["models"] = list(ctx.providers) if ctx else []
    totals, label, u = report["totals"], report["metric"]["label"], report["understanding"]
    kpis, structured = [], {}
    result = {"totals": totals, "determination": report["determination"]["code"],
              "hypotheses": [{"id": h["id"], "status": h["status"], "effect": h.get("effect")} for h in report["hypotheses"]],
              "analyses": {a["id"]: {"op": a["op"], "params": a["params"], "facts": (a["result"] or {}).get("facts"),
                                     "table": (a["result"] or {}).get("table")} for a in report["analyses"] if a["status"] == "ok"}}
    # A summary may state a fall of -9,500 as "a decrease of 9,500", and may count hypotheses.
    result["derived"] = sorted(set(hypothesis_engine.derived_numbers(verification.extract_numbers(result))))
    result["counts"] = {**report["hypothesis_counts"], "hypotheses": len(report["hypotheses"])}
    if totals.get("a") is not None and totals.get("b") is not None:
        fa = investigator._period_filters(u["time_column"], u["period_a"], last=False)
        fb = investigator._period_filters(u["time_column"], u["period_b"], last=True)
        base, spec = u["metric"]["filters"], {"formula": u["metric"]["formula"]}
        refs, ref_error = {}, None
        try:
            refs["a"] = me.reference_evaluate(state["csv_path"], spec["formula"], base + fa)["value"]
            refs["b"] = me.reference_evaluate(state["csv_path"], spec["formula"], base + fb)["value"]
        except (me.ReferenceNotApplicable, me.FormulaError, OSError, UnicodeDecodeError) as e:
            ref_error = f"{type(e).__name__}: {e}"
        k_a = _kpi(f"{label} — earlier period", fastpath.format_value(totals["a"]))
        k_b = _kpi(f"{label} — later period", fastpath.format_value(totals["b"]))
        k_c = _kpi("Change", fastpath.format_value(totals["change"]))
        kpis = [k_a, k_b, k_c]
        structured[k_a["element_id"]] = _structured({**spec, "filters": base + fa}, totals["a"], refs.get("a"), ref_error)
        structured[k_b["element_id"]] = _structured({**spec, "filters": base + fb}, totals["b"], refs.get("b"), ref_error)
        ref_change = refs["b"] - refs["a"] if refs.get("a") is not None and refs.get("b") is not None else None
        structured[k_c["element_id"]] = _structured({**spec, "formula": f"({spec['formula']} later) - ({spec['formula']} earlier)"},
                                                    totals["change"], ref_change, ref_error)

    charts = [{"title": c["title"], "step_index": 0, "element_id": uuid.uuid4().hex, "plotly_json": c["plotly_json"]}
              for c in report.get("charts", [])]
    narrative = report["summary"]
    formula_text = (f"{u['metric']['formula']}  |  earlier {u['period_a']}  vs  later {u['period_b']}"
                    f"  |  filters: {_filter_text(u['metric']['filters'])}")
    log_id = _record_step(
        question_id, f"Investigate the change in {label}", "Test competing explanations against the data",
        "# Hypothesis-driven investigation (analyses are deterministic operations; no generated code was run).\n"
        + "\n".join(f"# {a['id']}  {a['op']}  {a['params']}  -> {a['status']}" for a in report["analyses"]),
        result, formula_text, {"columns": [], "rows": []})
    steps = {0: {"result": result, "execution_log_id": log_id, "formula": formula_text}}
    narrative_id = uuid.uuid4().hex
    records = verification.build_evidence(
        kpis=kpis, charts=charts, narrative=narrative, narrative_element_id=narrative_id, steps=steps, critic=None,
        dataset=dataset_ev, question_text=state["question_text"], structured=structured, deterministic=True)
    for r in records:
        if r["claim_type"] != "narrative":
            continue
        r["limitations"].append("consistent_with_data_not_proof_of_cause")
        if report["executive_summary"]["source"] == "model":
            for c in r["checks"]:
                if c["check"] == "independent_review":
                    c["detail"] = "No second model reviewed the wording; its numbers were checked against the executed analyses."
            r["checks"].append(verification.check(
                "model_summary_grounded", True, verification.PASS, "",
                "Written by a model; every number in it was matched to an executed analysis before it was accepted."))
        if report["status"] != "complete":
            r["limitations"].append("cause_not_determined")
        r["validity"]["causal"] = "not_established"
        r["status"], r["reasons"] = verification.decide_status(r["checks"], r["limitations"])
    report["reproducibility"]["execution_log_id"] = log_id
    _save(investigation_id, report, report["status"])
    dashboard = publish_dashboard(
        question_id=question_id, kpis=kpis, charts=charts, narrative=narrative, narrative_element_id=narrative_id,
        steps=steps, records=records, dataset_ev=dataset_ev, base_state="VERIFIED" if kpis else "UNVERIFIED",
        verification_summary=("Totals were computed deterministically and recomputed independently. Hypotheses were judged "
                              "by fixed evidence rules on executed analyses; a supported hypothesis is consistent with the "
                              "data, not proven to be the cause. " + report["determination"]["text"]),
        critic_note="Hypothesis-driven investigation.",
    )
    return {"dashboard": dashboard, "evidence": records, "investigation_id": investigation_id}
