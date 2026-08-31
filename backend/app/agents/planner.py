"""Planner node: turns the question + dataset profile into ordered analysis steps."""
from __future__ import annotations

from app.agents import llm_client, prompts
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import Plan

PLAN_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief rationale for why these steps, in this order, will answer the question.",
        },
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {
                        "type": "string",
                        "description": "Concrete, executable analysis step.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What this step needs to establish for the final answer.",
                    },
                },
                "required": ["id", "description", "goal"],
            },
        },
    },
    "required": ["reasoning", "steps"],
}


def planner_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    revision_feedback = state.get("revision_feedback")
    is_revision = bool(revision_feedback)

    update_stage(question_id, "planning", "Revising plan based on critic feedback" if is_revision else "Understanding data & planning analysis")

    user_content = f"""Business question: {state['question_text']}

Dataset profile:
{llm_client.pretty(state['profile'])}
"""
    if is_revision:
        user_content += f"""
A previous attempt at this analysis was reviewed and REJECTED by the Critic agent with this feedback:
{revision_feedback}

Produce a revised plan that addresses these issues directly.
"""

    output = llm_client.call_tool(
        system=prompts.PLANNER_SYSTEM,
        user_content=user_content,
        tool_name="submit_plan",
        tool_schema=PLAN_TOOL_SCHEMA,
        tool_description="Submit the ordered list of analysis steps.",
    )

    steps = output["steps"]

    db = SessionLocal()
    try:
        plan_row = Plan(
            question_id=question_id,
            steps_json=llm_client.pretty(steps),
            reasoning=output.get("reasoning", ""),
            revision=state.get("revision_count", 0),
        )
        db.add(plan_row)
        db.commit()
        db.refresh(plan_row)
        plan_id = plan_row.id
    finally:
        db.close()

    return {
        "plan": steps,
        "plan_reasoning": output.get("reasoning", ""),
        "plan_id": plan_id,
        "step_index": 0,
        "retry_count": 0,
        "last_error": None,
        "step_results": [],
        "revision_feedback": None,
    }
