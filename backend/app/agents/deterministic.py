"""Deterministic graph nodes: the fast path, the root-cause investigator and
the comparison workflow.

None calls a model or runs generated code. Each computes through trusted
engines (`metric_engine`, `stats`), recomputes its headline figures with an
independent reference implementation, and publishes through the same
`publish_dashboard` as the model-driven route -- so their dashboards, audit
trails and evidence records look the same to every reader.
"""
from __future__ import annotations

import uuid

from app import comparison, fastpath, investigator, verification
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
                               for k in ("price_effect", "volume_effect", "entry_exit_effect")},
              # The summary states these, so they are recorded with the step it is checked against.
              # "p < 0.001" is how a smaller p-value is stated, so the threshold is recorded too.
              "change_evidence": {**((report.get("change_evidence") or {}).get("overall") or {}), "p_reported_floor": 0.001}}
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


# -------------------------------------------------------------- comparison --

def _reference_pair(engine_value, reference_value, formula: str, plan: dict, group_by: list[str] | None = None) -> dict:
    return {"formula": formula, "filters": plan["filters"], "group_by": group_by or [], "engine_value": engine_value,
            "reference_value": reference_value,
            "error": None if reference_value is not None else "The independent reference produced no value for this figure."}


def compare_node(state: AgentState) -> dict:
    """Carry out a validated comparison plan with the statistics engine and
    publish it. The plan was validated before it reached this node; the data
    can still contradict it (a group that does not exist), and that is
    reported as the reason no analysis was run, not as an internal error."""
    question_id, plan = state["question_id"], state["route_spec"]
    update_stage(question_id, "executing", "Comparing with the statistics engine")
    try:
        plan = comparison.validate({k: v for k, v in plan.items() if k in comparison.PLAN_TOOL_SCHEMA["properties"] or k == "source"},
                                   state["profile"])
        df = me.load_frame(state["csv_path"])
        report = comparison.run(df, plan)
        reference = comparison.reference(state["csv_path"], plan)
    except comparison.PlanError as e:
        return {"failed": True, "failure_reason": f"This comparison could not be run: {e}"}
    except (me.FormulaError, OSError, UnicodeDecodeError) as e:
        return {"failed": True, "failure_reason": f"The statistics engine could not read this dataset: {e}"}

    stat_result, outcome, explanatory = report["statistic"], plan["outcome"], plan["explanatory"]
    kpis: list[dict] = []
    structured: dict[str, dict] = {}
    ref_groups = reference.get("groups") or {}

    def add(label: str, value: float | None, ref: float | None, formula: str, group_by: list[str] | None = None) -> None:
        if value is None:
            return
        kpi = _kpi(label, fastpath.format_value(float(value)))
        kpis.append(kpi)
        structured[kpi["element_id"]] = _reference_pair(float(value), ref, formula, plan, group_by)

    if report["family"] == "correlation":
        if stat_result["status"] == "ok" and stat_result["method"] == "pearson":
            add(f"Correlation: {explanatory} and {outcome}", stat_result["estimate"], reference.get("pearson_r"),
                f"pearson({explanatory}, {outcome})")
        add("Rows used", float(stat_result["sample"].get("n", 0)), float(reference["n"]) if reference.get("n") is not None else None,
            "count()")
    elif plan["design"] == "paired":
        a, b = plan["levels"] or [g["level"] for g in report["groups"]][:2] or ("first", "second")
        if (stat_result["status"] == "ok" or stat_result.get("inference_unavailable")) and plan["estimand"] != "typical":
            add(f"Mean change in {outcome} ({b} − {a})", stat_result["estimate"], reference.get("mean_change"),
                f'avg("{outcome}") change per {plan["unit"]}', [explanatory])
        add(f"Pairs of {plan['unit']} compared", float(stat_result["sample"].get("pairs", 0)),
            float(reference["pairs"]) if reference.get("pairs") is not None else None, "count of matched units")
    elif report["family"] == "groups":
        groups = [g for g in report["groups"] if g["mean"] is not None]
        shown = groups if len(groups) == 2 else sorted(groups, key=lambda g: -g["mean"])[:1] + sorted(groups, key=lambda g: g["mean"])[:1]
        for g in shown:
            add(f"Mean {outcome}: {g['level']}", g["mean"], (ref_groups.get(g["level"]) or {}).get("mean"),
                f'avg("{outcome}")', [explanatory])
        if len(groups) == 2:
            a, b = groups
            ra, rb = (ref_groups.get(a["level"]) or {}).get("mean"), (ref_groups.get(b["level"]) or {}).get("mean")
            add(f"Difference ({b['level']} − {a['level']})", b["mean"] - a["mean"],
                None if ra is None or rb is None else rb - ra, f'avg("{outcome}") difference', [explanatory])
        else:
            add(f"Number of {explanatory} groups", float(len(groups)),
                float(sum(1 for v in ref_groups.values() if v.get("n", 0) >= comparison.MIN_GROUP_ROWS)), "count of groups")
    else:
        for g in report["groups"]:
            if g.get("share") is not None:
                counts = (ref_groups.get(g["level"]) or {})
                level = stat_result["estimand"].split("= ")[-1].split(" (")[0]
                ref = counts["counts"].get(level, 0) / counts["n"] if counts.get("n") else None
                add(f"Share with {outcome} = {level}: {g['level']}", g["share"], ref, "share of rows", [explanatory])
        used = float(sum(g["n"] for g in report["groups"]))
        add("Rows used", used, float(sum(v.get("n", 0) for k, v in ref_groups.items() if any(g["level"] == k for g in report["groups"]))),
            "count()")

    narrative = comparison.narrative(report)
    description = f"Compare {outcome} by {explanatory}" if report["family"] != "correlation" else f"Relate {explanatory} and {outcome}"
    method = stat_result.get("method_label") or "no test"
    formula_text = (f"{report['family']}: {outcome} ~ {explanatory}  |  method: {method}  |  filters: {_filter_text(plan['filters'])}"
                    f"  |  plan from: {plan['source']}")
    data_slice = {"columns": ["group", "n", "mean"], "rows": [[g["level"], g.get("n"), g.get("mean", g.get("share"))]
                                                              for g in report["groups"][:50]]}
    log_id = _record_step(
        question_id, description, "Measure the difference or association, with its uncertainty",
        "# Comparison workflow (deterministic; no generated code was run).\n"
        f"# plan    : {plan['outcome']} ~ {plan['explanatory']} ({plan['design']}, {plan['estimand']})\n"
        f"# method  : {method}, chosen by the statistics engine from the data\n"
        f"# rejected: {'; '.join(r['method'] for r in stat_result.get('rejected_methods') or []) or 'none'}",
        report, formula_text, data_slice)
    steps = {0: {"result": report, "execution_log_id": log_id, "formula": formula_text}}
    dataset_ev = dataset_evidence_for(question_id)
    narrative_id = uuid.uuid4().hex
    records = verification.build_evidence(
        kpis=kpis, charts=[], narrative=narrative, narrative_element_id=narrative_id, steps=steps,
        critic=None, dataset=dataset_ev, question_text=state["question_text"], structured=structured,
        deterministic=True,
    )
    assessed = "assessed" if stat_result["status"] == "ok" else f"not assessed: {stat_result.get('reason')}"
    for r in records:
        r["validity"]["statistical"] = assessed if r["validity"]["statistical"] == "not_assessed" else r["validity"]["statistical"]
        if r["claim_type"] == "narrative":
            # An association is published as an association whatever the question asked.
            r["limitations"].append("association_not_causal")
            if stat_result.get("inference_unavailable"):
                # measured, but no test or interval is supported: not the same as nothing to conclude
                r["limitations"].append("inference_not_available")
            elif stat_result["status"] != "ok":
                r["limitations"].append("no_statistical_conclusion")
            elif stat_result.get("interval") is None and stat_result.get("estimate") is not None:
                r["limitations"].append("interval_not_available")
            r["validity"]["causal"] = "not_established"
            r["status"], r["reasons"] = verification.decide_status(r["checks"], r["limitations"])
    dashboard = publish_dashboard(
        question_id=question_id, kpis=kpis, charts=[], narrative=narrative,
        narrative_element_id=narrative_id, steps=steps, records=records, dataset_ev=dataset_ev,
        base_state="VERIFIED" if kpis else "UNVERIFIED",
        verification_summary=("The group figures were computed by the statistics engine and recomputed by an independent "
                              "reference. The test and interval come from the statistics engine; they describe an "
                              "association and do not establish cause."),
        critic_note="Comparison workflow: validated plan, statistics engine, no generated code.",
    )
    return {"dashboard": dashboard, "evidence": records}
