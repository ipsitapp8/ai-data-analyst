"""Deterministic graph nodes: the fast path and the root-cause investigator.

Neither calls a model nor runs generated code. Both compute through
`metric_engine`, recompute the headline figures with its independent reference
implementation, and publish through the same `publish_dashboard` as the
model-driven route -- so their dashboards, audit trails and evidence records
look the same to every reader.
"""
from __future__ import annotations

import uuid

from app import fastpath, investigator, verification
from app import metric_engine as me
from app.agents.dashboard import dataset_evidence_for, publish_dashboard
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import ExecutionLog, Investigation, Plan, Question


def _record_step(question_id: int, description: str, goal: str, code: str, result: dict, formula: str,
                 data_slice: dict) -> int:
    """A plan row and an execution-log row, so status, the audit trail and
    Inspect work exactly as they do for a model-driven run."""
    db = SessionLocal()
    try:
        db.add(Plan(question_id=question_id, steps_json=[{"id": 1, "description": description, "goal": goal}],
                    reasoning="Answered by the deterministic engine; no model call was needed.", revision=0))
        # Committed before the log on purpose: the status endpoint shows the
        # logs created at or after the latest plan.
        db.commit()
        log = ExecutionLog(
            question_id=question_id, step_index=0, step_description=description, attempt_number=1, code=code,
            stdout="", stderr="", success=True, result_json=result, chart_paths_json=[],
            reasoning="Deterministic metric engine", formula_explanation=formula, data_slice_json=data_slice,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log.id
    finally:
        db.close()


def _kpi(label: str, value: str) -> dict:
    return {"label": label, "value": value, "source_step_index": 0, "element_id": uuid.uuid4().hex}


def _structured(spec: dict, engine_value, reference_value, error=None) -> dict:
    return {"formula": spec["formula"], "filters": spec.get("filters") or [], "group_by": spec.get("group_by") or [],
            "engine_value": engine_value, "reference_value": reference_value, "error": error}


def _filter_text(filters: list[dict]) -> str:
    return "; ".join(f"{f['column']} {f['op']} {f['value']}" for f in filters) if filters else "none"


# ---------------------------------------------------------------- fast path --

def fast_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    spec = state["route_spec"]
    update_stage(question_id, "executing", "Computing with the deterministic engine")
    try:
        out = fastpath.compute(state["csv_path"], state["profile"], spec)
    except (me.FormulaError, OSError) as e:
        return {"failed": True, "failure_reason": f"The deterministic engine could not compute this: {e}"}
    engine, reference, ref_error = out["engine"], out["reference"], out["reference_error"]
    label, unit = spec["label"], spec.get("unit", "")
    kpis: list[dict] = []
    charts: list[dict] = []
    structured: dict[str, dict] = {}

    if not spec["group_by"]:
        value = engine["value"]
        if value is None:
            return {"failed": True,
                    "failure_reason": f"'{spec['formula']}' has no value on this dataset (no usable data in that column)."}
        main = _kpi(label, fastpath.format_value(value, unit))
        rows = _kpi("Rows used", fastpath.format_value(float(engine["rows"])))
        kpis = [main, rows]
        structured[main["element_id"]] = _structured(spec, value, (reference or {}).get("value"), ref_error)
        structured[rows["element_id"]] = _structured(
            {**spec, "formula": "count()"}, float(engine["rows"]),
            float(reference["rows"]) if reference else None, ref_error)
        result = {"value": value, "rows": engine["rows"]}
        narrative = f"{label} is {fastpath.format_value(value, unit)}, computed over {engine['rows']:,} rows."
    else:
        dim = spec["group_by"][0]
        groups = [g for g in engine["groups"] if g["value"] is not None]
        if not groups:
            return {"failed": True, "failure_reason": f"'{spec['formula']}' has no value for any {dim}."}
        ranked = sorted(groups, key=lambda g: -g["value"])
        ref_by_key = {str(g["key"].get(dim)): g["value"] for g in (reference or {}).get("groups", [])}
        hi, lo = ranked[0], ranked[-1]
        k_hi = _kpi(f"Highest: {hi['key'][dim]}", fastpath.format_value(hi["value"], unit))
        k_lo = _kpi(f"Lowest: {lo['key'][dim]}", fastpath.format_value(lo["value"], unit))
        k_n = _kpi(f"Number of {dim} groups", fastpath.format_value(float(len(groups))))
        kpis = [k_hi, k_lo, k_n]
        for kpi, group in ((k_hi, hi), (k_lo, lo)):
            ref_value = ref_by_key.get(str(group["key"][dim]))
            structured[kpi["element_id"]] = _structured(
                spec, group["value"], ref_value,
                ref_error or (None if ref_value is not None else "The reference implementation has no group with this key."))
        structured[k_n["element_id"]] = _structured(
            spec, float(len(groups)),
            float(len([v for v in ref_by_key.values() if v is not None])) if reference else None, ref_error)
        result = {"groups": [{"key": str(g["key"][dim]), "value": g["value"], "rows": g["rows"]} for g in ranked[:200]],
                  "group_count": len(groups), "rows": engine["rows"]}
        shown = ranked[:30]
        charts = [{"title": label, "step_index": 0, "element_id": uuid.uuid4().hex, "plotly_json": {
            "data": [{"type": "bar", "x": [str(g["key"][dim]) for g in shown], "y": [g["value"] for g in shown]}],
            "layout": {"title": {"text": label}, "xaxis": {"title": {"text": dim}}}}}]
        narrative = (f"{label}: the highest is {hi['key'][dim]} at {fastpath.format_value(hi['value'], unit)} and the "
                     f"lowest is {lo['key'][dim]} at {fastpath.format_value(lo['value'], unit)}, across {len(groups)} groups "
                     f"and {engine['rows']:,} rows.")
        if engine.get("truncated"):
            narrative += " Only the first groups are shown; the dataset has more."
    if engine.get("coerced_non_numeric"):
        bad = ", ".join(f"{n} in {c}" for c, n in engine["coerced_non_numeric"].items())
        narrative += f" Non-numeric values were ignored ({bad})."

    formula_text = f"{spec['formula']}  |  filters: {_filter_text(spec['filters'])}"
    if spec["group_by"]:
        formula_text += f"  |  grouped by {', '.join(spec['group_by'])}"
    log_id = _record_step(question_id, f"Compute {label}", "Answer the question directly from the data",
                          fastpath.equivalent_code(spec), result, formula_text, out["data_slice"])
    steps = {0: {"result": result, "execution_log_id": log_id, "formula": formula_text}}
    dataset_ev = dataset_evidence_for(question_id)
    narrative_id = uuid.uuid4().hex
    records = verification.build_evidence(
        kpis=kpis, charts=charts, narrative=narrative, narrative_element_id=narrative_id, steps=steps,
        critic=None, dataset=dataset_ev, question_text=state["question_text"], structured=structured,
        deterministic=True,
    )
    dashboard = publish_dashboard(
        question_id=question_id, kpis=kpis, charts=charts, narrative=narrative,
        narrative_element_id=narrative_id, steps=steps, records=records, dataset_ev=dataset_ev,
        base_state="VERIFIED",
        verification_summary="Computed by the deterministic metric engine and recomputed by an independent "
                             "reference implementation. No model was involved in producing these numbers.",
        critic_note="Deterministic fast path.",
    )
    return {"dashboard": dashboard, "evidence": records}


# ------------------------------------------------------------- investigator --

def _load_or_create_investigation(question_id: int, spec: dict) -> tuple[int, dict]:
    db = SessionLocal()
    try:
        row = db.query(Investigation).filter_by(question_id=question_id).first()
        if row is not None:
            return row.id, dict(row.params_json or {})
        question = db.get(Question, question_id)
        row = Investigation(team_id=question.team_id, dataset_id=question.dataset_id,
                            dataset_version_id=question.dataset_version_id, question_id=question_id,
                            params_json=spec, report_json={}, status="running")
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id, dict(spec)
    finally:
        db.close()


def _save_investigation(investigation_id: int, report: dict, status: str) -> None:
    db = SessionLocal()
    try:
        row = db.get(Investigation, investigation_id)
        row.report_json, row.status = report, status
        db.commit()
    finally:
        db.close()


def investigate_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "investigating", "Decomposing the change across periods and segments")
    investigation_id, spec = _load_or_create_investigation(question_id, state.get("route_spec") or {})
    if not spec.get("formula"):
        _save_investigation(investigation_id, {}, "failed")
        return {"failed": True, "failure_reason": "No metric could be identified to investigate."}
    try:
        df = me.load_frame(state["csv_path"])
        report = investigator.investigate(df, state["profile"], spec)
    except (me.FormulaError, OSError) as e:
        _save_investigation(investigation_id, {"error": str(e)}, "failed")
        return {"failed": True, "failure_reason": f"The investigation could not run: {e}"}

    totals, label = report.get("totals") or {}, report["metric"]["label"]
    kpis: list[dict] = []
    structured: dict[str, dict] = {}
    result = {"status": report["status"], "totals": totals,
              "top_contributors": [{"dimension": r["dimension"], "segment": r["segment"],
                                    "contribution": r["contribution"], "share_of_change": r["share_of_change"]}
                                   for r in report.get("ranked_contributors", [])[:10]],
              "price_volume": {k: (report.get("price_volume") or {}).get(k)
                               for k in ("price_effect", "volume_effect", "entry_exit_effect")}}
    if totals.get("a") is not None and totals.get("b") is not None:
        periods = report["periods"]
        fa = investigator._period_filters(periods["time_column"], periods["a"], last=False)
        fb = investigator._period_filters(periods["time_column"], periods["b"], last=True)
        base_filters = report["metric"]["filters"]
        refs, ref_error = {}, None
        try:
            refs["a"] = me.reference_evaluate(state["csv_path"], spec["formula"], base_filters + fa)["value"]
            refs["b"] = me.reference_evaluate(state["csv_path"], spec["formula"], base_filters + fb)["value"]
        except (me.ReferenceNotApplicable, me.FormulaError, OSError, UnicodeDecodeError) as e:
            ref_error = f"{type(e).__name__}: {e}"
        k_a = _kpi(f"{label} — earlier period", fastpath.format_value(totals["a"]))
        k_b = _kpi(f"{label} — later period", fastpath.format_value(totals["b"]))
        kpis = [k_a, k_b]
        structured[k_a["element_id"]] = _structured({**spec, "filters": base_filters + fa}, totals["a"], refs.get("a"), ref_error)
        structured[k_b["element_id"]] = _structured({**spec, "filters": base_filters + fb}, totals["b"], refs.get("b"), ref_error)
        if totals.get("change") is not None:
            k_c = _kpi("Change", fastpath.format_value(totals["change"]))
            kpis.append(k_c)
            ref_change = refs["b"] - refs["a"] if refs.get("a") is not None and refs.get("b") is not None else None
            structured[k_c["element_id"]] = _structured(
                {**spec, "formula": f"({spec['formula']} in later period) - ({spec['formula']} in earlier period)"},
                totals["change"], ref_change, ref_error)

    charts = [{"title": c["title"], "step_index": 0, "element_id": uuid.uuid4().hex, "plotly_json": c["plotly_json"]}
              for c in report.get("charts", [])]
    narrative = report["summary"] + " This locates the change in the data; it does not establish what brought it about."
    periods_text = report.get("periods") or {}
    formula_text = (f"{spec['formula']}  |  earlier {periods_text.get('a')}  vs  later {periods_text.get('b')}"
                    f"  |  filters: {_filter_text(report['metric']['filters'])}")
    log_id = _record_step(
        question_id, f"Investigate the change in {label}", "Locate where the change came from",
        "# Root-cause investigator (deterministic; no generated code was run).\n"
        f"# metric  : {spec['formula']}\n# periods : {periods_text.get('a')} vs {periods_text.get('b')}\n"
        "# method  : " + report.get("decomposition_method", ""),
        result, formula_text, {"columns": [], "rows": []})
    steps = {0: {"result": result, "execution_log_id": log_id, "formula": formula_text}}
    dataset_ev = dataset_evidence_for(question_id)
    narrative_id = uuid.uuid4().hex
    records = verification.build_evidence(
        kpis=kpis, charts=charts, narrative=narrative, narrative_element_id=narrative_id, steps=steps,
        critic=None, dataset=dataset_ev, question_text=state["question_text"], structured=structured,
        deterministic=True,
    )
    for r in records:
        if r["claim_type"] == "narrative":
            r["limitations"].append("descriptive_not_causal")
            if report["status"] != "complete":
                r["limitations"].append("investigation_inconclusive")
            r["validity"]["causal"] = "not_established"
            r["status"], r["reasons"] = verification.decide_status(r["checks"], r["limitations"])
    report["provenance"] = {"dataset_version_id": dataset_ev.get("version_id"),
                            "dataset_fingerprint": dataset_ev.get("fingerprint"), "question_id": question_id,
                            "execution_log_id": log_id}
    _save_investigation(investigation_id, report, report["status"])
    dashboard = publish_dashboard(
        question_id=question_id, kpis=kpis, charts=charts, narrative=narrative,
        narrative_element_id=narrative_id, steps=steps, records=records, dataset_ev=dataset_ev,
        base_state="VERIFIED" if kpis else "UNVERIFIED",
        verification_summary=("The totals were computed by the deterministic engine and recomputed independently. "
                              "The decomposition is descriptive: it shows where the change is, not what caused it."
                              if kpis else "Inconclusive: " + report["summary"]),
        critic_note="Deterministic root-cause investigator.",
    )
    return {"dashboard": dashboard, "evidence": records, "investigation_id": investigation_id}
