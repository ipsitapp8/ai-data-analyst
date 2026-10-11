"""Router node: choose the cheapest graph path that can answer the question
properly, and say why.

Routes
    clarify       the question names something with more than one approved
                  meaning, or matches several columns equally -- ask, don't guess
    fast          a simple aggregation: deterministic engine, no model, no
                  generated code
    root_cause    "why did X change": the deterministic investigator
    statistical   trends, comparisons, outliers: the full pipeline, with the
                  Planner told what a statistical answer needs
    standard      everything else: the full pipeline

The decision is rule-based on purpose. It is deterministic, costs nothing, can
be unit-tested exhaustively, and every decision is written to the run trace
with the rules that fired. Each route also carries its own budget, so a run
that was judged simple cannot quietly become an expensive one.
"""
from __future__ import annotations

import datetime as dt
import re

from app import config, fastpath, investigator, runtime, semantic
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import Question

ROUTES = ("clarify", "fast", "root_cause", "statistical", "standard")

_ROOT_CAUSE = re.compile(
    r"\bwhy\b|root cause|what (?:caused|drove|is driving|explains|led to)|reasons? (?:for|behind)|"
    r"drivers? of|what(?:'s| is) behind|explain the (?:drop|decline|increase|rise|change|fall|spike|dip)",
    re.IGNORECASE)
_STATISTICAL = re.compile(
    r"\btrend|over time|correlat|outlier|anomal|distribut|compar|versus|\bvs\b|significan|seasonal|"
    r"growth rate|month over month|year over year|\bmom\b|\byoy\b|variance|spread|relationship between",
    re.IGNORECASE)

# Per-route ceilings. 0 = "none allowed" for the deterministic routes; the
# model-driven routes inherit the configured run budget.
ROUTE_BUDGETS = {
    "fast": {"llm_calls": 0, "sandbox_runs": 0, "seconds": 60},
    "root_cause": {"llm_calls": 0, "sandbox_runs": 0, "seconds": 120},
    "clarify": {"llm_calls": 0, "sandbox_runs": 0, "seconds": 30},
}

STATISTICAL_HINT = """
This is a statistical question (trend / comparison / outliers). The plan must:
- report the number of observations behind every comparison;
- report an effect size (difference or ratio), not only a direction;
- say when a sample is too small to support a conclusion;
- describe associations as associations -- do not state or imply causes.
"""


def decide(question_text: str, profile: dict, resolution: dict, forced: str | None = None) -> dict:
    """Pure routing decision. Returns {"route", "reasons", ...route-specific keys}."""
    reasons: list[str] = []
    if resolution.get("ambiguous"):
        amb = resolution["ambiguous"][0]
        return {
            "route": "clarify",
            "reasons": [f"'{amb['term']}' has {len(amb['candidates'])} approved definitions"],
            "clarification": {
                "reason": f"'{amb['term']}' is defined more than one way for this team. Which one do you mean?",
                "options": [{"label": f"{c['name']} = {c['formula']}",
                             "question": f"{question_text.rstrip('?. ')} (using the metric \"{c['name']}\")"}
                            for c in amb["candidates"]],
            },
        }
    metrics = resolution.get("metrics") or []
    if metrics:
        reasons.append("approved metric(s): " + ", ".join(m["name"] for m in metrics))

    if forced in ("root_cause", "standard", "statistical"):
        reasons.append(f"route requested at submission: {forced}")
        if forced != "root_cause":
            return {"route": forced, "reasons": reasons, "metrics": metrics}

    is_root_cause = forced == "root_cause" or bool(_ROOT_CAUSE.search(question_text))
    if not is_root_cause:
        parsed = fastpath.parse(question_text, profile, metrics)
        if parsed is not None and parsed["kind"] == "ambiguous":
            return {"route": "clarify", "reasons": [parsed["reason"]],
                    "clarification": {"reason": parsed["reason"] + " Which one do you mean?",
                                      "options": parsed["options"]}}
        if parsed is not None:
            return {"route": "fast", "reasons": reasons + [f"simple aggregation: {parsed['explanation']}"],
                    "spec": parsed, "metrics": metrics}

    if is_root_cause:
        spec = investigator.infer_spec(profile, question_text, metrics)
        if forced == "root_cause":
            # Submitted through the investigations API: the explicit spec is
            # stored with the investigation and the node loads it from there.
            return {"route": "root_cause", "reasons": reasons, "spec": spec or {}, "metrics": metrics}
        if spec is not None:
            return {"route": "root_cause",
                    "reasons": reasons + ["asks for the cause of a change",
                                          f"metric {spec['formula']} over time column {spec['time_column']}"],
                    "spec": spec, "metrics": metrics}
        reasons.append("asks for the cause of a change, but no time column or no single metric could be identified")
        return {"route": "statistical", "reasons": reasons, "metrics": metrics}

    if _STATISTICAL.search(question_text):
        return {"route": "statistical", "reasons": reasons + ["trend / comparison / outlier wording"], "metrics": metrics}
    return {"route": "standard", "reasons": reasons + ["no cheaper route applies"], "metrics": metrics}


def apply_budget(route: str) -> dict:
    """Tighten the current run's budget to the route's ceiling."""
    limits = ROUTE_BUDGETS.get(route)
    ctx = runtime.current()
    applied = {"llm_calls": config.RUN_MAX_LLM_CALLS, "sandbox_runs": config.RUN_MAX_SANDBOX_RUNS,
               "cost_usd": config.RUN_MAX_COST_USD}
    if limits is None:
        return applied
    applied.update({"llm_calls": limits["llm_calls"], "sandbox_runs": limits["sandbox_runs"],
                    "seconds": limits["seconds"]})
    if ctx is not None:
        # -1 sentinel: "none allowed" (0 already means unlimited in RunContext)
        ctx.max_llm_calls = limits["llm_calls"] if limits["llm_calls"] else -1
        ctx.max_sandbox_runs = limits["sandbox_runs"] if limits["sandbox_runs"] else -1
        cap = dt.datetime.utcnow() + dt.timedelta(seconds=limits["seconds"])
        ctx.deadline_at = min(ctx.deadline_at, cap) if ctx.deadline_at else cap
    return applied


def router_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "routing", "Choosing how to answer this question")
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        forced = question.route if question is not None else None
        try:
            resolution = semantic.resolve(db, question.team_id, question.dataset_id, state["question_text"])
        except Exception:  # noqa: BLE001 - the semantic layer is an aid; routing works without it
            resolution = {"metrics": [], "ambiguous": [], "dimensions": []}
        decision = decide(state["question_text"], state["profile"], resolution, forced)
        if question is not None:
            question.route = decision["route"]
            if decision["route"] == "clarify":
                question.clarification_json = decision["clarification"]
            db.commit()
    finally:
        db.close()
    budget = apply_budget(decision["route"])
    runtime.record_event("route", decision["route"], {"reasons": decision["reasons"], "budget": budget})
    out: dict = {"route": decision["route"], "route_reasons": decision["reasons"],
                 "approved_metrics": decision.get("metrics") or []}
    if decision["route"] == "clarify":
        out["clarification"] = decision["clarification"]
    if "spec" in decision:
        out["route_spec"] = decision["spec"]
    if decision["route"] == "statistical":
        out["planner_hint"] = STATISTICAL_HINT
    return out
