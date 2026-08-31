"""Critic node: an independent verifier with its own sandbox context.

Deliberately does NOT reuse the Executor's conversation/context. It writes its
own verification code against the raw data, runs it, and only then judges
whether the executor's claims hold up.

It also prefers a DIFFERENT model family (Llama) than the Planner/Executor
(Gemini), since a different model catches different mistakes than the same
model re-checking its own work.
"""
from __future__ import annotations

from app import config
from app.agents import llm_client
from app.agents import llm_client_llama as llama_client
from app.agents import prompts
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import CriticReview
from app.sandbox.runner import get_runner


def _critic_call_tool(**kwargs) -> tuple[dict, str]:
    """Run a Critic tool call on Llama, falling back to Gemini if Llama is
    unavailable (no key, bad key, provider down).

    Returns (output, verifier_label). The label is recorded in the audit trail
    so a fallback run is never silently presented as cross-model verification
    when it wasn't.
    """
    if config.LLAMA_API_KEY:
        try:
            return llama_client.call_tool(**kwargs), f"llama:{config.LLAMA_MODEL}"
        except Exception as e:  # noqa: BLE001 - any provider failure should degrade, not kill the run
            print(f"[critic] Llama call failed ({type(e).__name__}: {e}); "
                  f"falling back to Gemini. Verification will NOT be cross-model.")
    # call_tool_gemini, not call_tool: the latter fails back over to Llama,
    # which we already know just failed here.
    return llm_client.call_tool_gemini(**kwargs), f"gemini-fallback:{config.GEMINI_MODEL}"

VERIFY_CODE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "what_it_checks": {
            "type": "string",
            "description": "Which specific claimed numbers/findings this script independently recomputes.",
        },
        "code": {
            "type": "string",
            "description": "Complete, self-contained Python script following the sandbox contract.",
        },
    },
    "required": ["what_it_checks", "code"],
}

REVIEW_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["verified", "rejected"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific problems found; empty if verdict is verified.",
        },
        "summary": {
            "type": "string",
            "description": "2-4 sentence explanation of the verdict, referencing the independent check.",
        },
    },
    "required": ["verdict", "confidence", "issues", "summary"],
}


def _steps_summary(step_results: list[dict]) -> str:
    lines = []
    for r in step_results:
        lines.append(
            f"--- Step {r['step_index']}: {r['description']} ---\n"
            f"code:\n{r['code']}\n"
            f"claimed result: {r.get('result')}\n"
            f"charts produced: {r.get('chart_paths')}\n"
        )
    return "\n".join(lines)


def critic_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "critic", "Independently re-checking results against raw data")

    step_results = state["step_results"]
    successful_steps = [r for r in step_results if r["success"]]

    verify_user_content = f"""Business question: {state['question_text']}

Dataset profile:
{llm_client.pretty(state['profile'])}

Executor's plan and claimed results (DO NOT TRUST these numbers without verification):
{_steps_summary(successful_steps)}
"""

    verify_output, verifier_model = _critic_call_tool(
        system=prompts.CRITIC_VERIFY_SYSTEM,
        user_content=verify_user_content,
        tool_name="submit_verification_code",
        tool_schema=VERIFY_CODE_TOOL_SCHEMA,
        tool_description="Submit the independent verification script.",
    )

    runner = get_runner()
    runner.seed_input_data(question_id, state["csv_path"])
    # step_index -1, attempt keyed by revision so verification runs don't collide with executor's files
    verify_attempt = 100 + state.get("revision_count", 0)
    sandbox_result = runner.run(question_id, verify_output["code"], -1, verify_attempt)

    review_user_content = f"""Business question: {state['question_text']}

Executor's plan and claimed results:
{_steps_summary(successful_steps)}

Your independent verification check ({verify_output.get('what_it_checks', '')}):
code:
{verify_output['code']}

verification run success: {sandbox_result.success}
verification stdout: {sandbox_result.stdout}
verification stderr: {sandbox_result.stderr}
verification recomputed result: {sandbox_result.result}
"""
    if state.get("failed"):
        review_user_content += f"\nNote: the executor pipeline reported a hard failure before reaching you: {state.get('failure_reason')}\n"

    review_output, _ = _critic_call_tool(
        system=prompts.CRITIC_REVIEW_SYSTEM,
        user_content=review_user_content,
        tool_name="submit_review",
        tool_schema=REVIEW_TOOL_SCHEMA,
        tool_description="Submit the final independent verdict.",
    )

    checks = [{
        "what_it_checks": verify_output.get("what_it_checks", ""),
        "code": verify_output["code"],
        "stdout": sandbox_result.stdout,
        "stderr": sandbox_result.stderr,
        "result": sandbox_result.result,
        "success": sandbox_result.success,
        "verifier_model": verifier_model,
    }]

    summary = review_output.get("summary", "")
    if verifier_model.startswith("gemini-fallback"):
        summary += (
            "\n\n[Note: the Critic ran on the same model family as the Executor "
            "(Llama unavailable), so this verification is independent in context "
            "and code, but not cross-model.]"
        )

    db = SessionLocal()
    try:
        review_row = CriticReview(
            question_id=question_id,
            verdict=review_output["verdict"],
            confidence=review_output.get("confidence", 0.0),
            issues_json=llm_client.pretty(review_output.get("issues", [])),
            checks_json=llm_client.pretty(checks),
            summary=summary,
        )
        db.add(review_row)
        db.commit()
        db.refresh(review_row)
        review_id = review_row.id
    finally:
        db.close()

    return {
        "critic_verdict": review_output["verdict"],
        "critic_summary": summary,
        "critic_issues": review_output.get("issues", []),
        "critic_checks": checks,
        "critic_review_id": review_id,
    }
