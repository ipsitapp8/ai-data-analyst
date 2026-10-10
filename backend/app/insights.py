"""Auto-insights for a freshly uploaded dataset.

Two parts, deliberately different in trust level:
- warnings: computed deterministically from the stored profile (no LLM), so a
  data-quality warning is never a hallucination.
- questions: starter questions proposed by the LLM from the profile. These are
  suggestions only; running one goes through the normal plan/execute/verify
  pipeline, so nothing here is presented as a finding.
"""
from __future__ import annotations

import logging

from app.agents import llm_client

logger = logging.getLogger(__name__)

HIGH_MISSING_PCT = 20.0
SMALL_SAMPLE_ROWS = 30
OUTLIER_SIGMAS = 6.0
MAX_QUESTIONS = 5

QUESTIONS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "A business question in plain English, answerable from this dataset alone."},
                    "why": {"type": "string", "description": "One short sentence on why it is worth asking."},
                },
                "required": ["question", "why"],
            },
        }
    },
    "required": ["questions"],
}

QUESTIONS_SYSTEM = (
    "You are a senior data analyst meeting a new dataset for the first time. Propose the most "
    "useful, specific business questions it can answer. Each must be answerable using ONLY the "
    "columns listed, mention real column names where natural, and differ in kind (a trend, a "
    "comparison across groups, a distribution/outlier check, a relationship). No generic filler."
)


def data_warnings(profile: dict) -> list[dict]:
    """Deterministic data-quality warnings, most severe first."""
    out: list[dict] = []
    rows = profile.get("row_count", 0)
    if 0 < rows < SMALL_SAMPLE_ROWS:
        out.append({"severity": "warn", "column": None,
                    "message": f"Only {rows} rows -- results may not be statistically meaningful."})
    for c in profile.get("columns", []):
        name = c.get("name")
        missing = c.get("missing_pct", 0) or 0
        if missing >= HIGH_MISSING_PCT:
            out.append({"severity": "high" if missing >= 50 else "warn", "column": name,
                        "message": f"{missing:g}% of values are missing."})
        if rows > 1 and c.get("unique_count") == 1 and missing < 100:
            out.append({"severity": "info", "column": name,
                        "message": "Every value is identical, so it cannot explain any variation."})
        stats = c.get("stats") or {}
        if c.get("kind") == "numeric" and stats.get("std"):
            mean, std = stats.get("mean"), stats["std"]
            if mean is not None and std > 0:
                if stats["max"] > mean + OUTLIER_SIGMAS * std or stats["min"] < mean - OUTLIER_SIGMAS * std:
                    out.append({"severity": "warn", "column": name,
                                "message": f"Extreme values (min {stats['min']:g}, max {stats['max']:g}, "
                                           f"mean {mean:g}) -- possible outliers or entry errors."})
    order = {"high": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda w: order[w["severity"]])


def suggest_questions(filename: str, profile: dict, notes: list[str] | None = None) -> list[dict]:
    """LLM-proposed starter questions. Returns [] on any failure."""
    user = f"Dataset: {filename}\n\nProfile:\n{llm_client.pretty(profile)}\n"
    if notes:
        user += "\nWhat the team has said about this data:\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    try:
        result = llm_client.call_tool(
            system=QUESTIONS_SYSTEM,
            user_content=user,
            tool_name="submit_questions",
            tool_schema=QUESTIONS_TOOL_SCHEMA,
            tool_description=f"Submit up to {MAX_QUESTIONS} starter questions.",
        )
        items = result.get("questions") or []
    except Exception:  # noqa: BLE001
        logger.warning("Could not generate starter questions", exc_info=True)
        return []
    cleaned = []
    for item in items:
        q = str(item.get("question", "")).strip()
        if q:
            cleaned.append({"question": q[:300], "why": str(item.get("why", "")).strip()[:300]})
    return cleaned[:MAX_QUESTIONS]
