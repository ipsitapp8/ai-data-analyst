"""The optional language-model adapter for discovery. It is never needed: discovery is complete without it.

The model sees the user's objective and the list of grounded suggestions (id, question, reason) and returns the ids in the
order it thinks fits the objective. It cannot add a question, a column or a method, and it computes nothing. Any failure raises,
and discovery then keeps its rules-based order.
"""
from __future__ import annotations

TOOL = "rank_scenarios"
SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]}


def ranker(objective: str, candidates: list[dict]) -> list[str]:
    from app.agents import llm_client
    if not llm_client.provider_available(TOOL):
        raise RuntimeError("no language model available")
    listing = "\n".join(f"{c['id']}: {c['question']} ({c['reason']})" for c in candidates)
    system = ("You order what-if questions by how useful they are to a user's stated objective. Return ONLY ids from the list, most useful first. "
              "Do not invent ids, do not compute anything.")
    raw = llm_client.call_tool(system, f"Objective: {objective}\n\nQuestions:\n{listing}", TOOL, SCHEMA, "Order the question ids by relevance.", 600)
    ids = raw.get("ids") if isinstance(raw, dict) else None
    if not isinstance(ids, list):
        raise RuntimeError("unusable model answer")
    return [str(i) for i in ids]
