"""LangGraph state machine wiring: Planner -> Executor (bounded retry loop) ->
Critic -> (revision loop, bounded) -> Dashboard-compile."""
from __future__ import annotations

import json
import traceback

from langgraph.graph import END, StateGraph

from app import config
from app.agents.critic import critic_node
from app.agents.dashboard import dashboard_node
from app.agents.executor import executor_node
from app.agents.planner import planner_node
from app.agents.state import AgentState, mark_terminal, update_stage
from app.agents.triage import triage_node
from app.database import SessionLocal
from app.models import Dataset, Question


def prepare_revision_node(state: AgentState) -> dict:
    issues = state.get("critic_issues", [])
    feedback = state.get("critic_summary", "") + (
        ("\nSpecific issues:\n- " + "\n- ".join(issues)) if issues else ""
    )
    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "revision_feedback": feedback,
        "step_results": [],
        "step_index": 0,
        "retry_count": 0,
        "last_error": None,
    }


def route_after_executor(state: AgentState) -> str:
    if state.get("failed"):
        return "failed"
    if state["step_index"] >= len(state["plan"]):
        return "critic"
    return "executor"


def route_after_critic(state: AgentState) -> str:
    if state.get("critic_verdict") == "verified":
        return "dashboard"
    if state.get("revision_count", 0) < config.MAX_CRITIC_REVISIONS:
        return "revise"
    return "dashboard"  # revisions exhausted: finalize as unverified with caveats


def route_after_triage(state: AgentState) -> str:
    return "rejected" if state.get("rejected") else "planner"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_revision", prepare_revision_node)
    graph.add_node("dashboard", dashboard_node)

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage", route_after_triage,
        {"planner": "planner", "rejected": END},
    )
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor", route_after_executor,
        {"executor": "executor", "critic": "critic", "failed": END},
    )
    graph.add_conditional_edges(
        "critic", route_after_critic,
        {"dashboard": "dashboard", "revise": "prepare_revision"},
    )
    graph.add_edge("prepare_revision", "planner")
    graph.add_edge("dashboard", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_question_graph(question_id: int) -> None:
    """Entry point invoked from a FastAPI background task. Synchronous/blocking —
    callers should run this in a background thread/task, not the request handler."""
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        if question is None:
            return
        dataset = db.get(Dataset, question.dataset_id)
        if dataset is None:
            mark_terminal(question_id, "failed", "Dataset not found")
            return

        profile = json.loads(dataset.profile_json)

        initial_state: AgentState = {
            "question_id": question_id,
            "dataset_id": dataset.id,
            "question_text": question.text,
            "csv_path": dataset.filepath,
            "profile": profile,
            "plan": [],
            "step_index": 0,
            "retry_count": 0,
            "step_results": [],
            "revision_count": 0,
            "failed": False,
            "rejected": False,
        }
    finally:
        db.close()

    try:
        graph = get_compiled_graph()
        final_state = graph.invoke(initial_state, config={"recursion_limit": 150})
    except Exception as e:  # noqa: BLE001 - top-level job boundary, must not raise
        mark_terminal(question_id, "failed", f"{e}\n{traceback.format_exc()}")
        return

    if final_state.get("rejected"):
        # Distinct from "failed": nothing broke, the question just wasn't
        # analyzable, so the UI shows guidance instead of an error.
        reason = final_state.get("rejection_reason") or "This question can't be analyzed."
        suggestions = final_state.get("suggestions") or []
        if suggestions:
            reason += "\n\nTry asking:\n" + "\n".join(f"• {s}" for s in suggestions)
        mark_terminal(question_id, "rejected", reason)
        return

    if final_state.get("failed"):
        mark_terminal(question_id, "failed", final_state.get("failure_reason"))
        return

    dashboard = final_state.get("dashboard")
    if dashboard and dashboard.get("verified"):
        mark_terminal(question_id, "verified")
    elif dashboard:
        mark_terminal(question_id, "unverified")
    else:
        mark_terminal(question_id, "failed", "Run ended without producing a dashboard")
