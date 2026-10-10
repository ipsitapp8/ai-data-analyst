"""Dashboard nodes: compile -> verify -> publish.

compile   the model picks KPIs and charts from the step results and writes the
          narrative. Nothing is stored yet.
verify    the deterministic verification engine checks every claim in that
          draft (app/verification.py). A failed required check sends the run
          back to the Planner once, if the revision budget allows.
publish   the dashboard, its audit trail and its evidence records are written,
          with a verdict that reflects both the reviewer and the checks.

Splitting publish from compile is what lets verification happen *before*
anything reaches a user, and makes publishing a single, final step.
"""
from __future__ import annotations

import json
import os
import uuid

from app import config, memory, provenance, runtime, verification
from app.agents import llm_client, prompts
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import Dashboard, Dataset, DatasetVersion, Question
from app.storage.audit import record_audit_entry
from app.verdict import compute_verdict_state

COMPILE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "kpis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "source_step_index": {"type": "integer"},
                },
                "required": ["label", "value", "source_step_index"],
            },
        },
        "charts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer"},
                    "file": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["step_index", "file", "title"],
            },
        },
        "narrative": {"type": "string"},
    },
    "required": ["kpis", "charts", "narrative"],
}


def _steps_for_prompt(step_results: list[dict]) -> str:
    lines = []
    for r in step_results:
        chart_files = [os.path.basename(p) for p in r.get("chart_paths", [])]
        lines.append(
            f"Step {r['step_index']}: {r['description']}\n"
            f"  result: {r.get('result')}\n"
            f"  available chart files: {chart_files}"
        )
    return "\n".join(lines)


def steps_index(step_results: list[dict]) -> dict[int, dict]:
    """Successful steps keyed by index, in the shape verification expects."""
    return {r["step_index"]: {"result": r.get("result"), "execution_log_id": r.get("execution_log_id"),
                              "formula": r.get("formula_explanation", "")}
            for r in step_results if r.get("success")}


def critic_info(state: AgentState) -> dict | None:
    checks = state.get("critic_checks") or []
    if not checks and state.get("critic_verdict") is None:
        return None
    first = checks[0] if checks else {}
    return {"ran": bool(checks), "success": bool(first.get("success")), "result": first.get("result"),
            "verdict": state.get("critic_verdict"), "verifier_model": first.get("verifier_model", "")}


def dataset_evidence_for(question_id: int) -> dict:
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        version = db.get(DatasetVersion, question.dataset_version_id) if question.dataset_version_id else None
        return provenance.dataset_evidence(db, version, db.get(Dataset, question.dataset_id))
    finally:
        db.close()


# ------------------------------------------------------------------ compile --

def compile_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "dashboard", "Generating dashboard")

    step_results = [r for r in state["step_results"] if r["success"]]
    user_content = f"""Business question: {state['question_text']}

Step results:
{_steps_for_prompt(step_results)}

Critic verdict: {state.get('critic_verdict')}
Critic summary: {state.get('critic_summary')}
Critic issues: {state.get('critic_issues')}
"""
    output = llm_client.call_tool(
        system=prompts.DASHBOARD_SYSTEM,
        user_content=user_content,
        tool_name="compile_dashboard",
        tool_schema=COMPILE_TOOL_SCHEMA,
        tool_description="Submit the compiled dashboard.",
    )

    # Resolve chart file basenames back to full plotly JSON content.
    chart_by_step = {r["step_index"]: r.get("chart_paths", []) for r in step_results}
    resolved_charts = []
    for c in output.get("charts", []) or []:
        candidates = chart_by_step.get(c.get("step_index"), [])
        match = next((p for p in candidates if os.path.basename(p) == c.get("file")), None)
        if not match or not os.path.exists(match):
            continue
        try:
            with open(match, "r", encoding="utf-8") as f:
                plotly_json = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        resolved_charts.append({"title": str(c.get("title", "")), "step_index": c["step_index"],
                                "plotly_json": plotly_json, "element_id": uuid.uuid4().hex})

    # Model output is untrusted: keep only well-formed KPI entries.
    kpis = []
    for k in output.get("kpis", []) or []:
        if isinstance(k, dict) and "label" in k and "value" in k:
            kpis.append({"label": str(k["label"])[:200], "value": str(k["value"])[:200],
                         "source_step_index": k.get("source_step_index"), "element_id": uuid.uuid4().hex})
    return {"draft": {"kpis": kpis, "charts": resolved_charts, "narrative": str(output.get("narrative", "")),
                      "narrative_element_id": uuid.uuid4().hex}}


# ------------------------------------------------------------------- verify --

def verify_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "verifying", "Checking every figure against the computed results")
    draft = state["draft"]
    dataset_ev = dataset_evidence_for(question_id)
    records = verification.build_evidence(
        kpis=draft["kpis"], charts=draft["charts"], narrative=draft["narrative"],
        narrative_element_id=draft["narrative_element_id"], steps=steps_index(state["step_results"]),
        critic=critic_info(state), dataset=dataset_ev, question_text=state["question_text"],
    )
    failed = verification.has_required_failure(records)
    runtime.record_event("verification", "evidence", {**verification.summarize(records), "required_failure": failed})
    return {"evidence": records, "dataset_evidence": dataset_ev, "verification_failed": failed,
            "verification_feedback": verification.failure_feedback(records) if failed else None}


def route_after_verify(state: AgentState) -> str:
    if state.get("verification_failed") and state.get("revision_count", 0) < config.MAX_CRITIC_REVISIONS:
        return "revise"
    return "publish"  # includes "still failing after the recovery pass": published as UNVERIFIED


# ------------------------------------------------------------------ publish --

def publish_dashboard(*, question_id: int, kpis: list[dict], charts: list[dict], narrative: str,
                      narrative_element_id: str, steps: dict[int, dict], records: list[dict],
                      dataset_ev: dict, base_state: str, verification_summary: str,
                      critic_review_id: int | None = None, critic_note: str = "") -> dict:
    """Write the dashboard, its audit trail and its evidence. The single place
    anything becomes visible to a user, shared by every route."""
    verdict_state = verification.combine_verdict(base_state, records)
    verified = verdict_state != "UNVERIFIED"
    evidence_by_element = {r["element_id"]: r["status"] for r in records if r.get("element_id")}
    summary = verification.summarize(records)
    if verification.has_required_failure(records):
        verification_summary = (verification.failure_feedback(records) + "\n\n" + verification_summary).strip()

    db = SessionLocal()
    try:
        team_id = db.get(Question, question_id).team_id
        dash_row = Dashboard(
            team_id=team_id, question_id=question_id, kpis_json=kpis, charts_json=charts, narrative=narrative,
            verified=verified, verdict_state=verdict_state, verification_summary=verification_summary,
        )
        db.add(dash_row)
        db.commit()
        db.refresh(dash_row)

        for kpi in kpis:
            record_audit_entry(
                db, question_id, element_label=f"KPI: {kpi['label']}", element_type="kpi",
                element_id=kpi["element_id"],
                reasoning=(f"Value '{kpi['value']}' taken from step {kpi.get('source_step_index')} result. "
                           f"Evidence: {evidence_by_element.get(kpi['element_id'], 'none')}. {critic_note}").strip(),
                execution_log_id=(steps.get(kpi.get("source_step_index")) or {}).get("execution_log_id"),
                critic_review_id=critic_review_id, team_id=team_id,
            )
        for chart in charts:
            record_audit_entry(
                db, question_id, element_label=f"Chart: {chart['title']}", element_type="chart",
                element_id=chart["element_id"],
                reasoning=f"Generated by step {chart.get('step_index')}. {critic_note}".strip(),
                execution_log_id=(steps.get(chart.get("step_index")) or {}).get("execution_log_id"),
                critic_review_id=critic_review_id, team_id=team_id,
            )
        record_audit_entry(
            db, question_id, element_label="Narrative summary", element_type="narrative",
            element_id=narrative_element_id,
            reasoning=f"Synthesized from all successful step results. {critic_note}".strip(),
            execution_log_id=None, critic_review_id=critic_review_id, team_id=team_id,
        )
        verification.persist(db, records, team_id=team_id, question_id=question_id,
                             dashboard_id=dash_row.id, dataset=dataset_ev)
        memory.index_dashboard(db, dash_row)
        dashboard_id = dash_row.id
    finally:
        db.close()

    update_stage(question_id, "done", "Dashboard ready")
    return {"id": dashboard_id, "kpis": kpis, "charts": charts, "narrative": narrative, "verified": verified,
            "verdict_state": verdict_state, "verification_summary": verification_summary, "evidence": summary}


def publish_node(state: AgentState) -> dict:
    draft = state["draft"]
    verified = state.get("critic_verdict") == "verified"
    base_state = compute_verdict_state(state.get("critic_verdict"), state.get("revision_count", 0))
    verification_summary = state.get("critic_summary") or ("Not independently verified." if not verified else "")
    dashboard = publish_dashboard(
        question_id=state["question_id"], kpis=draft["kpis"], charts=draft["charts"],
        narrative=draft["narrative"], narrative_element_id=draft["narrative_element_id"],
        steps=steps_index(state["step_results"]), records=state["evidence"],
        dataset_ev=state["dataset_evidence"], base_state=base_state,
        verification_summary=verification_summary, critic_review_id=state.get("critic_review_id"),
        critic_note=f"Critic verdict: {state.get('critic_verdict')} — {state.get('critic_summary')}",
    )
    return {"dashboard": dashboard}


def dashboard_node(state: AgentState) -> dict:
    """compile + verify + publish in one call, for callers outside the graph."""
    state = {**state, **compile_node(state)}
    state = {**state, **verify_node(state)}
    return publish_node(state)
