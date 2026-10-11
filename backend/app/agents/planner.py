"""Planner node: turns the question + dataset profile into ordered analysis steps."""
from __future__ import annotations

from app import knowledge, memory, semantic
from app.agents import llm_client, prompts
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import Plan, Question

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
{prompts.profile_block(state['profile'])}
"""
    # Approved definitions come from the team's semantic layer (changed only
    # through the API by an owner or admin) -- never from dataset content.
    user_content += semantic.format_for_prompt(state.get("approved_metrics") or [])
    user_content += state.get("planner_hint") or ""
    # Scheduled re-runs watch for change, so they plan fresh instead of
    # anchoring on their own earlier results.
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        if question is not None:
            # Definitions and lessons apply to every run, scheduled or not --
            # unlike past analyses they don't anchor the planner on old results.
            user_content += knowledge.format_for_prompt(
                knowledge.get_knowledge(db, team_id=question.team_id, dataset_id=question.dataset_id),
                knowledge.retrieve_lessons(db, team_id=question.team_id, dataset_id=question.dataset_id,
                                           question_text=question.text),
            )
        if question is not None and question.trigger != "scheduled":
            user_content += memory.format_for_prompt(memory.retrieve(
                db, team_id=question.team_id, dataset_id=question.dataset_id,
                question_text=question.text, exclude_question_id=question_id,
            ))
    finally:
        db.close()
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
            steps_json=steps,
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
