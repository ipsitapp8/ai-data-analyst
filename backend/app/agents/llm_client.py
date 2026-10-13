"""Thin wrapper around the Gemini API for tool-forced structured calls.

Kept as a small, provider-specific module so the rest of the agent code
(planner/executor/critic/dashboard) only ever calls call_tool/call_text/pretty
and doesn't know or care which LLM provider is behind them.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

from google import genai
from google.genai import errors, types

from app import config, runtime
from app.agents import llm_client_llama

logger = logging.getLogger(__name__)

# Fallback backoff schedule when Gemini's 429 response doesn't carry a
# RetryInfo delay (it usually does, but don't depend on that). Indexed by
# attempt number; the last value repeats for any attempt beyond it.
_DEFAULT_BACKOFF_SECONDS = (15.0, 30.0)
# Transient 5xx responses come in bursts lasting several seconds, so the waits
# are spaced well apart (about 5 s, then 20 s), each varied by +/-20% so that
# concurrent runs do not all retry at the same instant. Indexed by attempt;
# the last value repeats. Only these statuses are retried: the rest of the 5xx
# range (501, 505, ...) is not a passing condition.
_SERVER_ERROR_BACKOFF_SECONDS = (5.0, 20.0)
_SERVER_ERROR_JITTER = 0.2
_RETRYABLE_SERVER_STATUS = frozenset({500, 502, 503, 504})


def _jitter() -> float:
    """A multiplier in [1 - jitter, 1 + jitter]."""
    return 1.0 + random.uniform(-_SERVER_ERROR_JITTER, _SERVER_ERROR_JITTER)


def _retry_after_seconds(error: BaseException) -> float | None:
    """The provider's own Retry-After, when it is a plain number of seconds.
    Reads that one response header and nothing else."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    try:
        value = float(str(headers.get("retry-after")).strip()) if headers is not None and headers.get("retry-after") else None
    except (TypeError, ValueError, AttributeError):
        return None                      # an HTTP date or anything unexpected: fall back to the schedule
    return value if value is not None and value >= 0 else None


def _server_retry_delay_seconds(error: BaseException, attempt: int) -> float:
    """How long to wait before retrying a transient 5xx. Never longer than
    GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS, and never past the run's own deadline."""
    wait = _retry_after_seconds(error)
    if wait is None:
        wait = _SERVER_ERROR_BACKOFF_SECONDS[min(attempt, len(_SERVER_ERROR_BACKOFF_SECONDS) - 1)] * _jitter()
    wait = max(0.0, min(wait, float(config.GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS)))
    return min(wait, runtime.remaining_seconds(wait))


def _retry_delay_seconds(error: errors.ClientError, attempt: int) -> float:
    """How long to wait before retrying a 429, preferring the server's own RetryInfo."""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        body = details.get("error", details)
        for item in body.get("details", []) or []:
            if str(item.get("@type", "")).endswith("RetryInfo"):
                match = re.match(r"([\d.]+)", str(item.get("retryDelay", "")))
                if match:
                    return min(float(match.group(1)), config.GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS)
    default = _DEFAULT_BACKOFF_SECONDS[min(attempt, len(_DEFAULT_BACKOFF_SECONDS) - 1)]
    return min(default, config.GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


class MalformedOutput(RuntimeError):
    """The provider answered, but not with the structured output that was
    required. `diagnosis` says what came back instead (never its content)."""

    def __init__(self, provider: str, tool_name: str, diagnosis: dict):
        super().__init__(f"{provider.capitalize()} did not return the expected '{tool_name}' function call "
                         f"({diagnosis.get('category')}; finish reason {diagnosis.get('finish_reasons') or 'unknown'})")
        self.provider, self.tool_name, self.diagnosis = provider, tool_name, diagnosis


_FILTERED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE"}


def diagnose_response(response: Any, tool_name: str) -> dict:
    """Why a response carries no usable function call. Reports the shape of the
    response only -- finish reasons, kinds of parts, names of functions called
    -- and never any generated text or arguments."""
    finish, kinds, called = [], set(), []
    candidates = list(getattr(response, "candidates", None) or [])
    for c in candidates:
        reason = getattr(c, "finish_reason", None)
        finish.append(getattr(reason, "name", None) or (str(reason) if reason else None))
        content = getattr(c, "content", None)
        for part in (getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc is not None:
                kinds.add("function_call")
                called.append(str(getattr(fc, "name", ""))[:80])
            elif getattr(part, "text", None):
                kinds.add("text")
    block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    block = getattr(block, "name", None) or (str(block) if block else None)
    reasons = {r for r in finish if r}
    if block:
        category = "prompt_blocked"
    elif not candidates:
        category = "empty_response"
    elif "MAX_TOKENS" in reasons:
        category = "output_truncated"
    elif reasons & {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}:
        category = "malformed_function_call"
    elif reasons & _FILTERED:
        category = "content_filtered"
    elif called and tool_name not in called:
        category = "wrong_function"
    elif "text" in kinds:
        category = "prose_instead_of_function_call"
    else:
        category = "no_function_call"
    return {"category": category, "finish_reasons": sorted(reasons), "part_kinds": sorted(kinds),
            "functions_called": called[:5], "candidates": len(candidates), "block_reason": block}


# The provider's own status word for an error (the canonical RPC codes). Only
# these exact identifiers are ever recorded; anything else a response carries
# in that field is reduced to a neutral value, so nothing free-form is logged.
PROVIDER_STATUS_WORDS = frozenset({
    "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE",
    "DATA_LOSS", "UNAUTHENTICATED",
})
STATUS_UNSPECIFIED, STATUS_OTHER = "UNSPECIFIED", "OTHER"


def provider_status_word(exc: BaseException) -> str:
    """The provider's status word if it is a known identifier, UNSPECIFIED when
    the failure carries none, OTHER when it carries something unrecognised."""
    word = getattr(exc, "status", None)
    if word is None or word == "":
        return STATUS_UNSPECIFIED
    return word if isinstance(word, str) and word in PROVIDER_STATUS_WORDS else STATUS_OTHER


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


def classify_failure(exc: BaseException) -> dict:
    """A provider failure as a category a person can act on, plus its HTTP
    status when there is one. Built from the exception's type and status code
    only: the message, which could echo request data, is never included."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status = status if isinstance(status, int) else None
    name = type(exc).__name__
    if isinstance(exc, MalformedOutput):
        return {"kind": "malformed_output", "category": exc.diagnosis.get("category"), "status": None,
                "finish_reasons": exc.diagnosis.get("finish_reasons"), "retryable": False,
                "provider_status": STATUS_UNSPECIFIED}
    if status == 429:
        kind = "rate_limited"
    elif status == 503:
        kind = "overloaded"
    elif status is not None and status >= 500:
        kind = "server_error"
    elif status in (401, 403):
        kind = "authentication"
    elif status == 400:
        kind = "bad_request"
    elif status == 404:
        kind = "model_not_found"
    elif status is not None:
        kind = "client_error"
    elif name == "ValidationError":
        kind = "invalid_schema"
    elif any(word in name for word in ("Timeout", "Connect", "Network", "Transport")):
        kind = "network"
    elif isinstance(exc, RuntimeError) and "did not return the expected" in str(exc):
        kind = "malformed_output"
    else:
        kind = "other"
    return {"kind": kind, "category": kind, "status": status, "finish_reasons": None,
            "retryable": kind in ("rate_limited", "overloaded", "server_error"),
            "provider_status": provider_status_word(exc)}


def log_provider_event(provider: str, tool_name: str, failure: dict, *, attempt: int, max_attempts: int, action: str,
                       wait_s: float | None = None, error_type: str = "", model: str = "",
                       elapsed_ms: int | None = None) -> None:
    """One line per provider failure, in the server log and in the run trace.

    Contains: provider, configured model, model step, failure kind, HTTP
    status, the provider's status word (from a fixed list), finish reason, how
    long the failed request took, which attempt this was, and what happens
    next. Never: API keys, headers, prompts, dataset values, error bodies or
    generated text."""
    provider_status = failure.get("provider_status") or STATUS_UNSPECIFIED
    detail = {"provider": provider, "tool": tool_name, "kind": failure["kind"], "category": failure.get("category"),
              "status": failure.get("status"), "finish_reasons": failure.get("finish_reasons"), "attempt": attempt,
              "max_attempts": max_attempts, "action": action, "wait_s": wait_s, "error_type": error_type,
              "model": model, "provider_status": provider_status, "elapsed_ms": elapsed_ms}
    logger.warning("provider_event provider=%s tool=%s kind=%s category=%s status=%s finish_reasons=%s attempt=%d/%d action=%s%s"
                   " model=%s provider_status=%s elapsed_ms=%s",
                   provider, tool_name, failure["kind"], failure.get("category"), failure.get("status"),
                   ",".join(failure.get("finish_reasons") or []) or "-", attempt, max_attempts, action,
                   f" wait_s={wait_s:.0f}" if wait_s is not None else "",
                   model or "-", provider_status, "-" if elapsed_ms is None else elapsed_ms)
    runtime.record_event("provider", f"{provider}:{failure['kind']}", detail)


def call_tool_gemini(
    system: str,
    user_content: str,
    tool_name: str,
    tool_schema: dict[str, Any],
    tool_description: str,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Force Gemini to respond via a single named function call and return its args dict.

    Raw single-provider call with no failover -- use call_tool() for the
    Gemini-then-Llama path.
    """
    runtime.checkpoint(about_to="llm")
    started = time.monotonic()
    override = runtime.get_provider_override()
    if override is not None:
        out = override("gemini", system=system, user_content=user_content, tool_name=tool_name,
                       tool_schema=tool_schema, tool_description=tool_description, max_tokens=max_tokens)
        runtime.note_llm_call("scripted", "scripted", tool_name, 0, 0,
                              int((time.monotonic() - started) * 1000))
        return dict(out)
    client = get_client()
    function_decl = types.FunctionDeclaration(
        name=tool_name,
        description=tool_description,
        parameters=tool_schema,
    )

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=[types.Tool(function_declarations=[function_decl])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
        ),
    )

    usage = getattr(response, "usage_metadata", None)
    tokens_in = getattr(usage, "prompt_token_count", None)
    tokens_out = getattr(usage, "candidates_token_count", None)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    for candidate in response.candidates or []:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            fc = part.function_call
            if fc and fc.name == tool_name:
                runtime.note_llm_call("gemini", config.GEMINI_MODEL, tool_name, tokens_in, tokens_out, elapsed_ms)
                args = dict(fc.args or {})
                if not args:
                    # A call with no arguments is still "an answer"; callers validate it. Say so in the log.
                    logger.warning("provider_event provider=gemini tool=%s kind=malformed_output category=empty_arguments "
                                   "status=- finish_reasons=%s attempt=1/1 action=returned_to_caller", tool_name,
                                   getattr(getattr(candidate, "finish_reason", None), "name", "-"))
                return args

    runtime.note_llm_call("gemini", config.GEMINI_MODEL, tool_name, tokens_in, tokens_out, elapsed_ms, ok=False)
    raise MalformedOutput("gemini", tool_name, diagnose_response(response, tool_name))


# Which provider served the most recent call_tool(); recorded so the audit
# trail can show what actually produced a result rather than assuming Gemini.
LAST_PROVIDER: str = "none"


def call_tool(
    system: str,
    user_content: str,
    tool_name: str,
    tool_schema: dict[str, Any],
    tool_description: str,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Gemini first, automatically failing over to Llama.

    Covers both common Gemini failures: a 429 once the free-tier per-minute cap
    is hit, and the model replying with prose instead of the required function
    call. A 429 gets a bounded, blocking retry first (this call already runs
    on the graph's background thread, so sleeping here doesn't stall the API) --
    the free-tier window is short enough that most 429s clear on their own
    within a retry or two. Once retries are exhausted, or for any other error,
    the run continues on Llama instead of dying mid-pipeline.
    """
    global LAST_PROVIDER
    kwargs = dict(
        system=system,
        user_content=user_content,
        tool_name=tool_name,
        tool_schema=tool_schema,
        tool_description=tool_description,
        max_tokens=max_tokens,
    )
    failures: list[str] = []

    if runtime.get_provider_override() is not None:
        LAST_PROVIDER = "scripted"
        return call_tool_gemini(**kwargs)

    # What follows a Gemini failure that is not retried, and the total number of tries allowed.
    backup = bool(config.LLAMA_API_KEY) and config.LLM_FAILOVER_ENABLED
    next_step = "failover_to_llama" if backup else "give_up_no_backup_provider"
    tries = config.GEMINI_RATE_LIMIT_RETRIES + 1

    if config.GEMINI_API_KEY:
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                out = call_tool_gemini(**kwargs)
                LAST_PROVIDER = f"gemini:{config.GEMINI_MODEL}"
                return out
            except runtime.RunAborted:
                raise  # cancelled / over budget: not a provider failure, never fail over
            except errors.ClientError as e:
                failure = classify_failure(e)
                if e.code == 429 and attempt < config.GEMINI_RATE_LIMIT_RETRIES:
                    wait_s = _retry_delay_seconds(e, attempt)
                    attempt += 1
                    log_provider_event("gemini", tool_name, failure, attempt=attempt, max_attempts=tries, action="retry",
                                       wait_s=wait_s, error_type=type(e).__name__, model=config.GEMINI_MODEL,
                                       elapsed_ms=_elapsed_ms(started))
                    time.sleep(wait_s)
                    continue
                failures.append(f"gemini({config.GEMINI_MODEL}): {e}")
                log_provider_event("gemini", tool_name, failure, attempt=attempt + 1, max_attempts=tries,
                                   action=next_step, error_type=type(e).__name__, model=config.GEMINI_MODEL,
                                   elapsed_ms=_elapsed_ms(started))
                break
            except errors.ServerError as e:
                # Transient 5xx: the provider's side, and usually brief. Retried with spaced,
                # jittered waits inside the same attempt bound as rate limits (one counter for
                # both, so a call never makes more than `tries` requests whatever the mix).
                failure = classify_failure(e)
                if e.code in _RETRYABLE_SERVER_STATUS and attempt < config.GEMINI_RATE_LIMIT_RETRIES:
                    wait_s = _server_retry_delay_seconds(e, attempt)
                    attempt += 1
                    log_provider_event("gemini", tool_name, failure, attempt=attempt, max_attempts=tries, action="retry",
                                       wait_s=wait_s, error_type=type(e).__name__, model=config.GEMINI_MODEL,
                                       elapsed_ms=_elapsed_ms(started))
                    time.sleep(wait_s)
                    continue
                failures.append(f"gemini({config.GEMINI_MODEL}): {e}")
                log_provider_event("gemini", tool_name, failure, attempt=attempt + 1, max_attempts=tries,
                                   action=next_step, error_type=type(e).__name__, model=config.GEMINI_MODEL,
                                   elapsed_ms=_elapsed_ms(started))
                break
            except Exception as e:  # noqa: BLE001 - any other Gemini failure should try the backup
                # Includes MalformedOutput: the model answered without the required structured
                # output. Not retried here (unchanged behaviour): the backup provider gets the step.
                failures.append(f"gemini({config.GEMINI_MODEL}): {e}")
                log_provider_event("gemini", tool_name, classify_failure(e), attempt=attempt + 1, max_attempts=tries,
                                   action=next_step, error_type=type(e).__name__, model=config.GEMINI_MODEL,
                                   elapsed_ms=_elapsed_ms(started))
                break
    else:
        failures.append("gemini: no GEMINI_API_KEY set")

    if not config.LLM_FAILOVER_ENABLED and config.GEMINI_API_KEY:
        failures.append("llama: failover disabled (LLM_FAILOVER_ENABLED=false)")
    elif config.LLAMA_API_KEY:
        started = time.monotonic()
        try:
            out = llm_client_llama.call_tool(**kwargs, _fallback=bool(config.GEMINI_API_KEY))
            LAST_PROVIDER = f"llama:{config.LLAMA_MODEL}"
            logger.info("Served by Llama (%s).", config.LLAMA_MODEL)
            return out
        except runtime.RunAborted:
            raise
        except Exception as e:  # noqa: BLE001 - report both failures together below
            failures.append(f"llama({config.LLAMA_MODEL}): {e}")
            log_provider_event("llama", tool_name, classify_failure(e), attempt=1, max_attempts=1, action="give_up",
                               error_type=type(e).__name__, model=config.LLAMA_MODEL, elapsed_ms=_elapsed_ms(started))
    else:
        failures.append("llama: no LLAMA_API_KEY set")

    LAST_PROVIDER = "none"
    raise runtime.ProviderUnavailable(
        "Both model providers failed for this step.\n  - " + "\n  - ".join(failures)
    )


def call_text(system: str, user_content: str, max_tokens: int = 2048) -> str:
    """Plain text completion, used for the narrative summary."""
    client = get_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    return (response.text or "").strip()


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def provider_available(tool_name: str | None = None) -> bool:
    """Whether some model could serve a call right now, as far as configuration
    can tell. A test double installed with `runtime.provider_override` answers
    for itself when it defines `can_answer(tool_name)`."""
    override = runtime.get_provider_override()
    if override is not None:
        can = getattr(override, "can_answer", None)
        return bool(can(tool_name)) if callable(can) and tool_name else True
    return bool(config.GEMINI_API_KEY or (config.LLAMA_API_KEY and config.LLM_FAILOVER_ENABLED) or
                (config.LLAMA_API_KEY and not config.GEMINI_API_KEY))
