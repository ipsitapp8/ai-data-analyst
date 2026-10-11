"""LangGraph state machine.

    router ──clarify──────────────────────────────────────────> END
       │─────fast────────> fast ──────────────────────────────> END
       │─────root_cause──> investigate ───────────────────────> END
       └─────otherwise───> triage ──rejected──────────────────> END
                              │
                           planner <──────── prepare_revision <──┐
                              │                     ^            │
                           executor (retry loop)    │ rejected   │ a required
                              │                     │ + budget   │ check failed
                            critic ─────────────────┘            │ + budget
                              │                                  │
                           compile ──> verify ───────────────────┘
                                          │
                                       publish ───────────────────> END

Every loop is bounded: the Executor by MAX_EXECUTOR_RETRIES per step, and the
two edges into prepare_revision share one budget, MAX_CRITIC_REVISIONS. When
that budget is spent the run still ends at publish, with the verdict the
evidence supports -- a failed check never turns into a pass by running out of
retries.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app import config, runtime
from app.agents.critic import critic_node
from app.agents.dashboard import compile_node, publish_node, route_after_verify, verify_node
from app.agents.deterministic import fast_node, investigate_node
from app.agents.executor import executor_node
from app.agents.planner import planner_node
from app.agents.router import router_node
from app.agents.state import AgentState, mark_terminal
from app.agents.triage import triage_node
from app.database import SessionLocal
from app.dataset_versions import latest_version
from app.models import Dataset, DatasetVersion, Question

logger = logging.getLogger(__name__)


def prepare_revision_node(state: AgentState) -> dict:
    """Turn a rejection -- by the Critic, or by deterministic verification --
    into feedback for one more planning pass."""
    parts = []
    if state.get("verification_failed") and state.get("verification_feedback"):
        parts.append(state["verification_feedback"])
    if state.get("critic_verdict") != "verified":
        issues = state.get("critic_issues", [])
        parts.append(state.get("critic_summary", "") + (
            ("\nSpecific issues:\n- " + "\n- ".join(issues)) if issues else ""))
    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "revision_feedback": "\n\n".join(p for p in parts if p) or "The previous attempt was rejected.",
        "step_results": [],
        "step_index": 0,
        "retry_count": 0,
        "last_error": None,
        "draft": {},
        "evidence": [],
        "verification_failed": False,
        "verification_feedback": None,
    }


def route_after_router(state: AgentState) -> str:
    route = state.get("route")
    if route == "clarify":
        return "clarify"
    if route == "fast":
        return "fast"
    if route == "root_cause":
        return "investigate"
    return "triage"


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


def _traced(name: str, fn):
    def node(state: AgentState) -> dict:
        with runtime.node_span(name):
            return fn(state)

    node.__name__ = f"{name}_node"
    return node


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in (
        ("router", router_node), ("triage", triage_node), ("planner", planner_node),
        ("executor", executor_node), ("critic", critic_node), ("prepare_revision", prepare_revision_node),
        ("compile", compile_node), ("verify", verify_node), ("publish", publish_node),
        ("fast", fast_node), ("investigate", investigate_node),
    ):
        graph.add_node(name, _traced(name, fn))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", route_after_router,
        {"clarify": END, "fast": "fast", "investigate": "investigate", "triage": "triage"},
    )
    graph.add_edge("fast", END)
    graph.add_edge("investigate", END)
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
        {"dashboard": "compile", "revise": "prepare_revision"},
    )
    graph.add_edge("compile", "verify")
    graph.add_conditional_edges(
        "verify", route_after_verify,
        {"publish": "publish", "revise": "prepare_revision"},
    )
    graph.add_edge("prepare_revision", "planner")
    graph.add_edge("publish", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _aborted(exc: BaseException) -> runtime.RunAborted | None:
    """The RunAborted behind an exception, however the graph wrapped it."""
    seen = 0
    while exc is not None and seen < 10:
        if isinstance(exc, runtime.RunAborted):
            return exc
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return None


def run_question_graph(question_id: int) -> str:
    """Run one question through the graph and record its outcome.

    Returns the question's final status: verified | unverified | rejected |
    needs_clarification | failed. Raises runtime.RunAborted (cancelled, past
    deadline, over budget, sandbox unavailable) for the job layer to record;
    nothing else escapes. Blocking -- call it from a job worker, never from a
    request handler."""
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        if question is None:
            return "failed"
        dataset = db.get(Dataset, question.dataset_id)
        if dataset is None:
            mark_terminal(question_id, "failed", "Dataset not found")
            return "failed"

        # Pin the run to a dataset version: the one the scheduler/caller already
        # chose, else whatever is latest right now. Pinning (rather than reading
        # dataset.filepath at each node) is what keeps an old dashboard tied to
        # the data it was actually computed from.
        version = db.get(DatasetVersion, question.dataset_version_id) if question.dataset_version_id else None
        if version is None:
            version = latest_version(db, dataset)
            if version is not None:
                question.dataset_version_id = version.id
                db.commit()
        csv_path = version.filepath if version else dataset.filepath
        profile = version.profile_json if version else dataset.profile_json

        initial_state: AgentState = {
            "question_id": question_id,
            "dataset_id": dataset.id,
            "question_text": question.text,
            "csv_path": csv_path,
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
    except Exception as e:  # noqa: BLE001 - top-level run boundary
        aborted = _aborted(e)
        if aborted is not None:
            raise aborted from None
        # Full traceback goes to the server log only -- surfacing file paths and
        # library internals to the end user is an information-disclosure risk,
        # and none of it is actionable for them anyway. The question_id ties
        # the two together for whoever's debugging.
        logger.exception("Question %s failed", question_id)
        mark_terminal(
            question_id, "failed",
            "An internal error occurred while running this analysis. "
            f"If it keeps happening, mention analysis #{question_id} to support.",
        )
        return "failed"

    if final_state.get("route") == "clarify":
        clarification = final_state.get("clarification") or {}
        mark_terminal(question_id, "needs_clarification", clarification.get("reason"))
        return "needs_clarification"

    if final_state.get("rejected"):
        # Distinct from "failed": nothing broke, the question just wasn't
        # analyzable, so the UI shows guidance instead of an error.
        reason = final_state.get("rejection_reason") or "This question can't be analyzed."
        suggestions = final_state.get("suggestions") or []
        if suggestions:
            reason += "\n\nTry asking:\n" + "\n".join(f"• {s}" for s in suggestions)
        mark_terminal(question_id, "rejected", reason)
        return "rejected"

    if final_state.get("failed"):
        mark_terminal(question_id, "failed", final_state.get("failure_reason"))
        return "failed"

    dashboard = final_state.get("dashboard")
    if dashboard and dashboard.get("verified"):
        mark_terminal(question_id, "verified")
        return "verified"
    if dashboard:
        mark_terminal(question_id, "unverified")
        return "unverified"
    mark_terminal(question_id, "failed", "Run ended without producing a dashboard")
    return "failed"
