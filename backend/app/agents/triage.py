"""Triage node: cheap relevance gate in front of the expensive pipeline.

A full run costs several LLM calls plus one sandboxed container per step, so
gibberish or off-topic questions are rejected here rather than being turned
into a plausible-looking plan and a meaningless "verified" dashboard.
"""
from __future__ import annotations

from app.agents import llm_client, prompts
from app.agents.state import AgentState, update_stage

TRIAGE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "is_analyzable": {
            "type": "boolean",
            "description": "True only if this question expresses genuine analytical intent the dataset could support.",
        },
        "reason": {
            "type": "string",
            "description": "If rejected, a short user-facing explanation of why. Empty if accepted.",
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "If rejected, 2-3 questions this dataset's columns could actually answer.",
        },
    },
    "required": ["is_analyzable", "reason", "suggestions"],
}

# Below this, it cannot be a real question -- skip the API call entirely.
MIN_QUESTION_CHARS = 4


def _column_summary(profile: dict) -> str:
    cols = profile.get("columns", [])
    return ", ".join(f"{c['name']} ({c['kind']})" for c in cols) or "(no columns)"


def triage_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    question = (state.get("question_text") or "").strip()
    update_stage(question_id, "triage", "Checking the question against this dataset")

    if len(question) < MIN_QUESTION_CHARS:
        return {
            "rejected": True,
            "rejection_reason": "That question is too short to analyze. Try asking something specific about the data.",
            "suggestions": [],
        }

    profile = state["profile"]
    user_content = f"""Dataset columns: {_column_summary(profile)}
Rows: {profile.get('row_count')}

Full profile:
{llm_client.pretty(profile)}

User's question: {question!r}
"""

    try:
        out = llm_client.call_tool(
            system=prompts.TRIAGE_SYSTEM,
            user_content=user_content,
            tool_name="submit_triage",
            tool_schema=TRIAGE_TOOL_SCHEMA,
            tool_description="Decide whether this question is worth analyzing.",
            max_tokens=1024,
        )
    except Exception as e:  # noqa: BLE001 - a triage outage must not block real questions
        text = str(e)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            # Letting this through would spend the remaining quota on a run that
            # is going to die at the planner anyway, with a far uglier error.
            return {
                "rejected": True,
                "rejection_reason": (
                    "The Gemini API quota is exhausted, so no analysis can run right now. "
                    "Free-tier keys are capped per model per day — wait for the quota to "
                    "reset, switch GEMINI_MODEL in .env to a model with remaining quota, "
                    "or enable billing on your Google AI Studio project."
                ),
                "suggestions": [],
            }
        print(f"[triage] check failed ({type(e).__name__}: {e}); allowing the question through.")
        return {"rejected": False}

    if out.get("is_analyzable", True):
        return {"rejected": False}

    return {
        "rejected": True,
        "rejection_reason": out.get("reason", "").strip()
        or "This question doesn't map to anything in this dataset.",
        "suggestions": list(out.get("suggestions") or [])[:3],
    }
