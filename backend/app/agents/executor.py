"""Executor node: writes code for the current step, runs it sandboxed, and
implements the bounded self-correction retry loop (feed the error back on failure)."""
from __future__ import annotations

from app import config
from app.agents import llm_client, prompts
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import ExecutionLog
from app.sandbox.runner import get_runner

CODE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the approach this code takes.",
        },
        "code": {
            "type": "string",
            "description": "Complete, self-contained Python script following the sandbox contract.",
        },
    },
    "required": ["reasoning", "code"],
}


def _prior_results_summary(step_results: list[dict]) -> str:
    if not step_results:
        return "(none yet)"
    lines = []
    for r in step_results:
        lines.append(
            f"Step {r['step_index']}: {r['description']}\n  result: {r.get('result')}"
        )
    return "\n".join(lines)


def executor_node(state: AgentState) -> dict:
    question_id = state["question_id"]
    step_index = state["step_index"]
    retry_count = state.get("retry_count", 0)
    attempt_number = retry_count + 1
    step = state["plan"][step_index]

    stage_detail = f"Executing step {step_index + 1}/{len(state['plan'])}: {step['description']}"
    if retry_count > 0:
        stage_detail = f"Retry {retry_count}/{config.MAX_EXECUTOR_RETRIES} on step {step_index + 1}: {step['description']}"
    update_stage(question_id, "executing", stage_detail)

    user_content = f"""Business question: {state['question_text']}

Dataset profile:
{llm_client.pretty(state['profile'])}

Prior step results (already computed, available context — do not recompute them, but you may
reference their numbers in your own analysis if directly relevant):
{_prior_results_summary(state.get('step_results', []))}

Current step ({step_index + 1}/{len(state['plan'])}):
Description: {step['description']}
Goal: {step['goal']}
"""

    if state.get("last_error"):
        user_content += f"""
Your previous attempt at this step failed with this error:
{state['last_error']}

Fix the root cause and provide corrected code.
"""

    output = llm_client.call_tool(
        system=prompts.EXECUTOR_SYSTEM,
        user_content=user_content,
        tool_name="submit_code",
        tool_schema=CODE_TOOL_SCHEMA,
        tool_description="Submit the Python script for this analysis step.",
    )
    code = output["code"]
    reasoning = output.get("reasoning", "")

    runner = get_runner()
    runner.seed_input_data(question_id, state["csv_path"])
    sandbox_result = runner.run(question_id, code, step_index, attempt_number)

    db = SessionLocal()
    try:
        log_row = ExecutionLog(
            question_id=question_id,
            step_index=step_index,
            step_description=step["description"],
            attempt_number=attempt_number,
            code=code,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            success=sandbox_result.success,
            result_json=llm_client.pretty(sandbox_result.result or {}),
            chart_paths_json=llm_client.pretty(sandbox_result.chart_paths),
            reasoning=reasoning,
        )
        db.add(log_row)
        db.commit()
        db.refresh(log_row)
        log_id = log_row.id
    finally:
        db.close()

    if sandbox_result.success:
        step_result = {
            "step_index": step_index,
            "description": step["description"],
            "code": code,
            "stdout": sandbox_result.stdout,
            "stderr": sandbox_result.stderr,
            "result": sandbox_result.result,
            "chart_paths": sandbox_result.chart_paths,
            "success": True,
            "reasoning": reasoning,
            "attempts": attempt_number,
            "execution_log_id": log_id,
        }
        new_results = state.get("step_results", []) + [step_result]
        return {
            "step_results": new_results,
            "step_index": step_index + 1,
            "retry_count": 0,
            "last_error": None,
        }

    # Failure: either retry, or exhaust retries and fail the whole run.
    error_text = sandbox_result.stderr or "(no stderr captured)"
    if retry_count + 1 >= config.MAX_EXECUTOR_RETRIES:
        step_result = {
            "step_index": step_index,
            "description": step["description"],
            "code": code,
            "stdout": sandbox_result.stdout,
            "stderr": sandbox_result.stderr,
            "result": None,
            "chart_paths": [],
            "success": False,
            "reasoning": reasoning,
            "attempts": attempt_number,
            "execution_log_id": log_id,
        }
        new_results = state.get("step_results", []) + [step_result]
        return {
            "step_results": new_results,
            "failed": True,
            "failure_reason": (
                f"Step {step_index + 1} ('{step['description']}') failed after "
                f"{config.MAX_EXECUTOR_RETRIES} attempts. Last error:\n{error_text}"
            ),
        }

    return {
        "retry_count": retry_count + 1,
        "last_error": error_text,
    }
