"""Critic node: an independent verifier with its own sandbox context.

Deliberately does NOT reuse the Executor's conversation/context. It writes its
own verification code against the raw data, runs it, and only then judges
whether the executor's claims hold up.

It also prefers a DIFFERENT model family (Llama) than the Planner/Executor
(Gemini), since a different model catches different mistakes than the same
model re-checking its own work.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar

from app import config, knowledge, llm_gateway, runtime
from app.agents import llm_client
from app.agents import llm_client_llama as llama_client
from app.agents import prompts
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import CriticReview, Question
from app.sandbox.runner import get_runner


def _critic_call_tool(**kwargs) -> tuple[dict, str]:
    """Run a Critic tool call on Llama, falling back to Gemini if Llama is
    unavailable (no key, bad key, provider down).

    Returns (output, verifier_label). The label is recorded in the audit trail
    so a fallback run is never silently presented as cross-model verification
    when it wasn't.
    """
    if runtime.get_provider_override() is not None:
        return llama_client.call_tool(**kwargs), "scripted-critic"
    if config.LLM_HYBRID_ROUTING_ENABLED:
        out, label, info = _hybrid_critic_call(kwargs)
        _last_verifier.set(info)
        return out, label
    # Both calls go through the gateway: each provider's circuit breaker, one deadline for the pair.
    deadline = llm_gateway.Deadline()
    if config.LLAMA_API_KEY:
        try:
            out = llm_gateway.guarded("llama", lambda: llama_client.call_tool(**kwargs), deadline=deadline,
                                      timeout=config.LLAMA_REQUEST_TIMEOUT_SECONDS, wait_ok=not config.GEMINI_API_KEY)
            return out, f"llama:{config.LLAMA_MODEL}"
        except runtime.RunAborted:
            raise
        except Exception as e:  # noqa: BLE001 - any provider failure should degrade, not kill the run
            logger.warning("Critic: Llama call failed (%s); falling back to Gemini. "
                           "Verification will NOT be cross-model.", type(e).__name__)
    # call_tool_gemini, not call_tool: the latter fails back over to Llama,
    # which we already know just failed here.
    out = llm_gateway.guarded("gemini", lambda: llm_client.call_tool_gemini(**kwargs), deadline=deadline,
                              timeout=config.GEMINI_REQUEST_TIMEOUT_SECONDS, fallback=bool(config.LLAMA_API_KEY))
    return out, f"gemini-fallback:{config.GEMINI_MODEL}"

logger = logging.getLogger(__name__)

# What served the most recent hybrid Critic call (families, independence); read by critic_node for the audit record.
_last_verifier: ContextVar[dict | None] = ContextVar("critic_last_verifier", default=None)
# The agents whose models materially shaped the work the Critic checks: the Planner's plan and the Executor's code.
# Outside a run (the chat answer check) the checked work is the chat answer, with these as the fallback.
_RUN_AGENTS = ("planner", "executor")


def checked_families() -> tuple[frozenset[str], str]:
    """(every model family that contributed to the work being verified, where that came from).

    Inside a run: every family that served the Planner or the Executor in THIS run -- each step, not only the latest.
    If this run has no such record, the result is {unknown}: nothing is independent of work whose models are unknown,
    and another run's history is never used instead. Outside a run (chat): the answer's family, else what the
    Executor's chain would use first."""
    from app.llm import families, routing
    if runtime.current() is not None:
        found = routing.families_served(_RUN_AGENTS)
        return (frozenset(found), "served:run") if found else (frozenset({families.UNKNOWN}), "unknown:no_run_history")
    for agents in (("chat",), _RUN_AGENTS):
        found = routing.families_served(agents)
        if found:
            return frozenset(found), f"served:{'+'.join(agents)}"
    for provider, pinned in routing.chain("executor"):
        if routing.usable(provider):
            return frozenset({routing.family_for(provider, pinned)}), "configured:executor"
    return frozenset({families.UNKNOWN}), "unknown"


def _hybrid_critic_call(kwargs: dict) -> tuple[dict, str, dict]:
    """The Critic under hybrid routing. Providers whose model family is known and differs from EVERY family that
    served the Planner or the Executor in this run go first, in the Critic's chain order; if none serves, another
    model is used and the verification is recorded as NOT independent -- or, with LLM_CRITIC_REQUIRE_INDEPENDENT,
    the step fails.

    Labels keep the existing audit vocabulary: "<provider>:<model>" for independent verification,
    "<provider>-fallback:<model>" (e.g. "gemini-fallback:<model>") when it is not."""
    from app.llm import families, routing

    checked, source = checked_families()
    exec_family = ", ".join(sorted(checked))            # for messages and the audit record
    entries = routing.chain("critic")
    independent_first = [(p, m) for p, m in entries if families.independent_of_all(routing.family_for(p, m), checked)]
    rest = [(p, m) for p, m in entries if (p, m) not in independent_first]
    deadline = llm_gateway.Deadline()
    failures: list[str] = []
    reasons: list[dict] = []
    out, info = routing.run_chain("critic", independent_first, kwargs, deadline=deadline, failures=failures,
                                  reasons=reasons)
    if out is routing.NOT_SERVED:
        if config.LLM_CRITIC_REQUIRE_INDEPENDENT:
            llm_gateway.note_request(False)
            raise runtime.ProviderUnavailable(
                "No model of a different family than the Executor's could verify this step.\n  - " + "\n  - ".join(
                    failures or [f"no Critic provider outside the {exec_family} family is configured"]),
                failures=reasons)
        logger.warning("Critic: no provider outside the Executor's model family (%s) served the call; "
                       "verification will NOT be independent.", exec_family)
        same_family = rest + ([("gemini", None)] if not any(p == "gemini" for p, _ in rest) else [])
        out, info = routing.run_chain("critic", same_family, kwargs, deadline=deadline, failures=failures,
                                      reasons=reasons, fallback_from_start=bool(independent_first))
    if out is routing.NOT_SERVED:
        llm_gateway.note_request(False)
        raise runtime.ProviderUnavailable("No model provider could run the Critic.\n  - " + "\n  - ".join(failures),
                                          failures=reasons)
    llm_gateway.note_request(True, fallback=bool(info.get("fallback")))
    independent = families.independent_of_all(info["family"], checked)      # judged on what actually served
    label = f"{info['provider']}{'' if independent else '-fallback'}:{info['model']}"
    return out, label, {"independent": independent, "verifier_family": info["family"], "executor_family": exec_family,
                        "checked_families": sorted(checked), "executor_family_source": source,
                        "requested_model": info["model"], "served_model": info["served_model"],
                        "provider": info["provider"], "label": label}


VERIFY_CODE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "what_it_checks": {
            "type": "string",
            "description": "Which specific claimed numbers/findings this script independently recomputes.",
        },
        "code": {
            "type": "string",
            "minLength": 1,          # an empty script is rejected at the provider, before the sandbox
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
    """Run the independent check. If the verifier itself breaks (provider down,
    malformed reply, sandbox hiccup) the analysis is still delivered, honestly
    labelled UNVERIFIED, rather than the whole run dying after the work is done."""
    try:
        return _critic_run(state)
    except runtime.RunAborted:
        raise
    except Exception as e:  # noqa: BLE001 - the verifier failing must not discard finished work
        logger.exception("Critic failed for question %s", state["question_id"])
        return {
            "critic_verdict": "unverified",
            "critic_summary": (
                "The independent verification step could not be completed "
                f"({type(e).__name__}), so these results were not re-checked against the raw data. "
                "Treat them as unverified, or run the analysis again."
            ),
            "critic_issues": [],
            "critic_checks": [],
            "critic_review_id": None,
        }


def _critic_run(state: AgentState) -> dict:
    question_id = state["question_id"]
    update_stage(question_id, "critic", "Independently re-checking results against raw data")

    step_results = state["step_results"]
    successful_steps = [r for r in step_results if r["success"]]

    verify_user_content = f"""Business question: {state['question_text']}

Dataset profile:
{prompts.profile_block(state['profile'])}

Executor's plan and claimed results (DO NOT TRUST these numbers without verification):
{_steps_summary(successful_steps)}
"""

    _last_verifier.set(None)
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

    verify_info = _last_verifier.get()
    _last_verifier.set(None)
    review_output, review_model = _critic_call_tool(
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
    review_info = _last_verifier.get()
    if verify_info is not None:
        # Hybrid routing: which family checked which, and whether that counts as independent. Both Critic calls
        # must be independent for the verification to be called independent.
        independent = bool(verify_info["independent"]) and bool((review_info or verify_info)["independent"])
        checks[0].update({"verifier_independent": independent, "verifier_family": verify_info["verifier_family"],
                          "executor_family": verify_info["executor_family"],
                          "checked_families": verify_info["checked_families"],
                          "review_verifier_family": (review_info or verify_info)["verifier_family"],
                          "verifier_requested_model": verify_info["requested_model"],
                          "verifier_served_model": verify_info["served_model"], "review_verifier_model": review_model})

    summary = review_output.get("summary", "")
    if verify_info is not None:
        if not checks[0]["verifier_independent"]:
            summary += (
                f"\n\n[Note: the Critic ran on the same model family as the Executor or on a model whose family is "
                f"unknown (verifier: {verify_info['verifier_family']}, executor: {verify_info['executor_family']}), so "
                "this verification is independent in context and code, but not cross-model.]"
            )
    elif verifier_model.startswith("gemini-fallback"):
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
            issues_json=review_output.get("issues", []),
            checks_json=checks,
            summary=summary,
        )
        db.add(review_row)
        db.commit()
        db.refresh(review_row)
        review_id = review_row.id
        if review_output["verdict"] == "rejected":
            knowledge.record_critic_lesson(
                db, question=db.get(Question, question_id), summary=summary,
                issues=review_output.get("issues", []),
            )
    finally:
        db.close()

    return {
        "critic_verdict": review_output["verdict"],
        "critic_summary": summary,
        "critic_issues": review_output.get("issues", []),
        "critic_checks": checks,
        "critic_review_id": review_id,
    }
