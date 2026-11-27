"""Hybrid routing: each agent's ordered provider chain, and the failover engine that walks it.

Inert unless LLM_HYBRID_ROUTING_ENABLED is on. Then:

- an agent is named from the tool it calls (``AGENT_FOR_TOOL``), so no call site or ``call_tool`` signature changes;
- its chain comes from LLM_<AGENT>_PROVIDERS, else LLM_PROVIDER_PRIORITY (the Critic excepted), else the defaults:
  every agent ``gemini,groq,nvidia,openrouter``; the Critic ``groq,nvidia,openrouter``. Entries may pin a model for
  the hybrid providers (``groq=openai/gpt-oss-120b``); Gemini and Llama always use GEMINI_MODEL / LLAMA_MODEL;
- a provider is tried only when it is enabled and has both a key and a model. A key alone changes nothing. Mistral
  stays out until MISTRAL_ENABLED; the legacy Llama endpoint only when a chain names ``llama``;
- the step has one deadline (app/llm_gateway.Deadline) across every provider and retry, at most
  LLM_MAX_PROVIDER_ATTEMPTS providers are tried, and each provider's circuit breaker decides whether it is called.

Failover per failure kind (app/llm_gateway.ErrorKind):
  RATE_LIMITED, PROVIDER_UNAVAILABLE -> bounded retries with jittered exponential backoff (Retry-After first), then
                                        the next provider;
  TIMEOUT                            -> the next provider (the time is spent; the deadline decides what is left);
  QUOTA_EXHAUSTED                    -> next provider; the circuit opens, it is not retried;
  AUTH_ERROR                         -> next provider; the key is held (LLM_AUTH_HOLD_SECONDS), never retried;
  BILLING (402)                      -> next provider; held for LLM_BILLING_HOLD_SECONDS, never retried;
  MODEL_NOT_FOUND (404, invalid model 400), BAD_REQUEST, CONTEXT_LIMIT -> next provider, never retried;
  INVALID_OUTPUT                     -> next provider (Gemini keeps its own bounded malformed-output retry);
  runtime.RunAborted                 -> propagated at once: a cancelled run is not a provider failure.
"""
from __future__ import annotations

import logging
import os
import random
import time
from contextvars import ContextVar
from typing import Any

from app import config, llm_gateway, runtime
from app.llm import families, providers

logger = logging.getLogger(__name__)

DEFAULT_CHAIN = ("gemini", "groq", "nvidia", "openrouter")
DEFAULT_CRITIC_CHAIN = ("groq", "nvidia", "openrouter")

AGENT_FOR_TOOL = {
    "submit_plan": "planner", "submit_code": "executor", "compile_dashboard": "narrator",
    "submit_triage": "triage", "submit_verification_code": "critic", "submit_review": "critic", "submit_check": "critic",
    "submit_understanding": "investigations", "submit_hypotheses": "investigations",
    "submit_next_step": "investigations", "submit_report": "investigations",
    "submit_comparison_plan": "comparison", "submit_scenario_request": "whatif", "rank_scenarios": "whatif",
    "submit_answer": "chat", "submit_questions": "insights", "submit_operations": "copilot",
}
SHOWN_AGENTS = ("planner", "executor", "critic", "narrator", "triage", "investigations", "whatif", "default")

NOT_SERVED = object()
_last_served: ContextVar[dict | None] = ContextVar("llm_last_served", default=None)


def enabled() -> bool:
    return bool(config.LLM_HYBRID_ROUTING_ENABLED)


def agent_for(tool_name: str | None) -> str:
    return AGENT_FOR_TOOL.get(str(tool_name or ""), "default")


def _flag(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def chain(agent: str) -> list[tuple[str, str | None]]:
    """The agent's ordered (provider, pinned model or None) list, before availability is considered."""
    raw = os.getenv(f"LLM_{agent.upper()}_PROVIDERS", "").strip()
    if not raw and agent != "critic":
        raw = config.LLM_PROVIDER_PRIORITY
    if not raw:
        raw = ",".join(DEFAULT_CRITIC_CHAIN if agent == "critic" else DEFAULT_CHAIN)
    out: list[tuple[str, str | None]] = []
    for item in raw.split(","):
        name, _, model = item.strip().partition("=")
        name = name.strip().lower()
        if not name:
            continue
        if name not in providers.ALL_PROVIDERS:
            logger.warning("llm_routing agent=%s unknown provider %r in its route: ignored", agent, name[:40])
            continue
        if any(name == n for n, _ in out):
            continue
        pinned = model.strip() or None
        if pinned and name in ("gemini", "llama"):
            logger.warning("llm_routing agent=%s provider=%s: a pinned model is ignored (set %s_MODEL)", agent, name,
                           name.upper())
            pinned = None
        out.append((name, pinned))
    fallback = _flag(f"LLM_{agent.upper()}_FALLBACK")
    if fallback is None:
        fallback = True if agent == "critic" else config.LLM_FAILOVER_ENABLED
    return out if fallback else out[:1]


def model_for(provider: str, pinned: str | None) -> str:
    return pinned or providers.spec(provider).model


def usable(provider: str) -> bool:
    return providers.spec(provider).state() == "configured"


def family_for(provider: str, pinned: str | None = None) -> str:
    return families.family_of(provider, model_for(provider, pinned))


# ------------------------------------------------------------------------------------------------ served record --

def note_served(agent: str, provider: str, model: str, served_model: str | None) -> dict:
    info = {"provider": provider, "model": model, "served_model": served_model or model,
            "family": families.family_of(provider, model, served_model)}
    ctx = runtime.current()
    if ctx is not None:
        ctx.served[agent] = info
    record = dict(_last_served.get() or {})
    record[agent] = info
    _last_served.set(record)
    return info


def served_for(agent: str) -> dict | None:
    """What served ``agent``'s last call in this run (or, outside a run, in this context)."""
    ctx = runtime.current()
    if ctx is not None and agent in ctx.served:
        return ctx.served[agent]
    return (_last_served.get() or {}).get(agent)


# ------------------------------------------------------------------------------------------ the failover engine --

def _try_hybrid(provider: str, model: str, kwargs: dict, deadline: llm_gateway.Deadline, failures: list[str],
                reasons: list[dict], *, fallback: bool, wait_ok: bool, give_up_action: str) -> Any:
    """One hybrid provider, with at most LLM_PROVIDER_TRANSIENT_RETRIES retries of a rate limit / outage that fit the
    deadline. Returns (arguments, served model) or NOT_SERVED after recording why."""
    from app.agents import llm_client       # the shared failure vocabulary and log line

    s = providers.spec(provider)
    tool_name = kwargs["tool_name"]
    attempts = 1 + max(0, config.LLM_PROVIDER_TRANSIENT_RETRIES)
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            return llm_gateway.guarded(provider, lambda: providers.call_tool(s, model=model, fallback=fallback, **kwargs),
                                       deadline=deadline, timeout=config.LLM_PROVIDER_TIMEOUT_SECONDS,
                                       fallback=fallback, wait_ok=wait_ok)
        except runtime.RunAborted:
            raise
        except llm_gateway.ProviderError as e:      # not admitted: circuit open, rate limited, held, out of time
            why = llm_client.not_called_kind(e)
            failures.append(f"{provider}({model}): not called ({why.replace('_', ' ')})")
            reasons.append({"provider": provider, "kind": why, "status": None, "model": model})
            llm_client.log_provider_event(provider, tool_name, llm_client._skip_failure(why), attempt=attempt - 1,
                                          max_attempts=attempts, action=give_up_action, model=model)
            return NOT_SERVED
        except Exception as e:  # noqa: BLE001 - recorded; retried if transient and it fits, else the next provider
            kind = llm_gateway.classify(e)
            cap = float(config.GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS)
            asked = llm_gateway.retry_after_seconds(e)
            if asked is not None:
                # never sooner than the provider asked (jitter only adds), or the breaker refuses the retry
                wait_s = min(asked * (1.0 + random.uniform(0.0, 0.2)), cap)
            else:
                wait_s = llm_gateway.jittered(min(2.0 * (2 ** (attempt - 1)), cap))   # exponential, +/-20%
            retry = (kind in (llm_gateway.ErrorKind.RATE_LIMITED, llm_gateway.ErrorKind.PROVIDER_UNAVAILABLE)
                     and attempt < attempts and deadline.fits(wait_s + config.LLM_MIN_ATTEMPT_SECONDS))
            failure = llm_client.classify_failure(e)
            llm_client.log_provider_event(provider, tool_name, failure, attempt=attempt, max_attempts=attempts,
                                          action="retry" if retry else give_up_action, wait_s=wait_s if retry else None,
                                          error_type=type(e).__name__, model=model,
                                          elapsed_ms=llm_client._elapsed_ms(started))
            if retry:
                time.sleep(wait_s)
                continue
            failures.append(f"{provider}({model}): {type(e).__name__} ({kind.value})")
            reasons.append(llm_client._failed(provider, model, e))
            return NOT_SERVED
    return NOT_SERVED


def run_chain(agent: str, entries: list[tuple[str, str | None]], kwargs: dict, *,
              deadline: llm_gateway.Deadline | None = None, failures: list[str] | None = None,
              reasons: list[dict] | None = None, fallback_from_start: bool = False) -> tuple[Any, dict | None]:
    """Walk ``entries`` until one provider serves the tool call. Returns (arguments, served info) or (NOT_SERVED,
    None); ``failures`` / ``reasons`` collect what happened for the caller's ProviderUnavailable."""
    from app.agents import llm_client

    deadline = deadline or llm_gateway.Deadline()
    failures = failures if failures is not None else []
    reasons = reasons if reasons is not None else []
    live = []
    for provider, pinned in entries:
        state = providers.spec(provider).state()
        if state != "configured":
            reasons.append(llm_client._not_tried(provider, "disabled" if state == "disabled" else "not_configured"))
            failures.append(f"{provider}: not tried ({state.replace('_', ' ')})")
            continue
        live.append((provider, pinned))
    live = live[:max(1, config.LLM_MAX_PROVIDER_ATTEMPTS)]
    tool_name = kwargs["tool_name"]
    for i, (provider, pinned) in enumerate(live):
        later = i < len(live) - 1
        fallback = fallback_from_start or i > 0
        action = f"failover_to_{live[i + 1][0]}" if later else "give_up"
        model = model_for(provider, pinned)
        if provider == "gemini":
            out = llm_client._try_gemini(kwargs, deadline, failures, reasons, backup=later, next_step=action,
                                         wait_ok=not later, fallback=fallback)
            served = None
        elif provider == "llama":
            out = llm_client._try_llama(kwargs, deadline, failures, reasons, call_kwargs={"_fallback": fallback},
                                        give_up_action=action, fallback=fallback, wait_ok=not later)
            served = None
        if provider in ("gemini", "llama") and out is llm_client._NOT_SERVED:
            out = NOT_SERVED                    # the existing adapters' own "not served" marker
        if provider not in ("gemini", "llama"):
            got = _try_hybrid(provider, model, kwargs, deadline, failures, reasons, fallback=fallback,
                              wait_ok=not later, give_up_action=action)
            out, served = (NOT_SERVED, None) if got is NOT_SERVED else got
        if out is not NOT_SERVED:
            info = dict(note_served(agent, provider, model, served), fallback=fallback)
            logger.info("llm_routing agent=%s tool=%s provider=%s model=%s served_model=%s family=%s attempt=%d/%d "
                        "fallback=%s", agent, tool_name, provider, model, info["served_model"], info["family"], i + 1,
                        len(live), fallback)
            return out, info
    return NOT_SERVED, None


def call_tool(kwargs: dict, agent: str | None = None) -> tuple[dict, dict]:
    """A tool call through the agent's chain. Raises runtime.ProviderUnavailable when no provider serves it."""
    from app.agents import llm_client

    agent = agent or agent_for(kwargs.get("tool_name"))
    failures: list[str] = []
    reasons: list[dict] = []
    out, info = run_chain(agent, chain(agent), kwargs, failures=failures, reasons=reasons)
    if out is NOT_SERVED:
        llm_gateway.note_request(False)
        llm_client._log_not_tried(kwargs["tool_name"], reasons)
        raise runtime.ProviderUnavailable("No configured model provider served this step.\n  - " + "\n  - ".join(failures),
                                          failures=reasons)
    llm_gateway.note_request(True, fallback=bool(info.get("fallback")))
    return out, info


def call_text(system: str, user_content: str, max_tokens: int, agent: str = "narrator") -> str:
    """Plain text through the agent's chain: each provider's plain chat call, one attempt each, one deadline."""
    from app.agents import llm_client, llm_client_llama

    deadline = llm_gateway.Deadline()
    failures: list[str] = []
    reasons: list[dict] = []
    live = [(p, m) for p, m in chain(agent) if usable(p)][:max(1, config.LLM_MAX_PROVIDER_ATTEMPTS)]
    for i, (provider, pinned) in enumerate(live):
        model = model_for(provider, pinned)
        if provider == "gemini":
            fn = lambda: (llm_client._gemini_text(system, user_content, max_tokens), None)  # noqa: E731
            timeout = config.GEMINI_REQUEST_TIMEOUT_SECONDS
        elif provider == "llama":
            fn = lambda: (llm_client_llama.call_text(system, user_content, max_tokens), None)  # noqa: E731
            timeout = config.LLAMA_REQUEST_TIMEOUT_SECONDS
        else:
            s = providers.spec(provider)
            fn = lambda s=s, model=model: providers.call_text(s, system=system, user_content=user_content,  # noqa: E731
                                                              max_tokens=max_tokens, model=model, fallback=i > 0)
            timeout = config.LLM_PROVIDER_TIMEOUT_SECONDS
        try:
            text, served = llm_gateway.guarded(provider, fn, deadline=deadline, timeout=timeout, fallback=i > 0,
                                               wait_ok=i == len(live) - 1)
        except runtime.RunAborted:
            raise
        except Exception as e:  # noqa: BLE001 - the next provider
            failures.append(f"{provider}({model}): {type(e).__name__} ({llm_gateway.classify(e).value})")
            reasons.append(llm_client._failed(provider, model, e) if not isinstance(e, llm_gateway.ProviderError)
                           else {"provider": provider, "kind": llm_client.not_called_kind(e), "status": None,
                                 "model": model})
            continue
        if not text:
            failures.append(f"{provider}({model}): empty text")
            reasons.append({"provider": provider, "kind": "malformed_output", "status": None, "model": model})
            continue
        note_served(agent, provider, model, served)
        llm_gateway.note_request(True, fallback=i > 0)
        return text
    llm_gateway.note_request(False)
    raise runtime.ProviderUnavailable("No configured model provider served this text request.\n  - "
                                      + "\n  - ".join(failures or ["no provider is configured"]), failures=reasons)


def provider_available(tool_name: str | None) -> bool:
    """Some provider in the agent's chain is configured and not refused right now (circuit, key, billing)."""
    from app.agents import llm_client

    return any(usable(p) and llm_client._circuit_closed_enough(p) for p, _ in chain(agent_for(tool_name)))


def status() -> dict:
    """For /api/health: each provider's configuration state, family and circuit; each agent's chain. No keys."""
    return {
        "enabled": enabled(),
        "providers": {p: {**providers.spec(p).public(), "family": family_for(p)} for p in providers.ALL_PROVIDERS},
        "routes": {a: [p for p, _ in chain(a)] for a in SHOWN_AGENTS},
        "critic_require_independent": config.LLM_CRITIC_REQUIRE_INDEPENDENT,
        "max_provider_attempts": config.LLM_MAX_PROVIDER_ATTEMPTS,
    }
