"""Thin wrapper around the Gemini API for tool-forced structured calls.

Kept as a small, provider-specific module so the rest of the agent code
(planner/executor/critic/dashboard) only ever calls call_tool/call_text/pretty
and doesn't know or care which LLM provider is behind them.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import errors, types

from app import config
from app.agents import llm_client_llama

logger = logging.getLogger(__name__)

# Fallback backoff schedule when Gemini's 429 response doesn't carry a
# RetryInfo delay (it usually does, but don't depend on that). Indexed by
# attempt number; the last value repeats for any attempt beyond it.
_DEFAULT_BACKOFF_SECONDS = (15.0, 30.0)


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

    for candidate in response.candidates or []:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            fc = part.function_call
            if fc and fc.name == tool_name:
                return dict(fc.args or {})

    raise RuntimeError(f"Gemini did not return the expected '{tool_name}' function call")


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

    if config.GEMINI_API_KEY:
        attempt = 0
        while True:
            try:
                out = call_tool_gemini(**kwargs)
                LAST_PROVIDER = f"gemini:{config.GEMINI_MODEL}"
                return out
            except errors.ClientError as e:
                if e.code == 429 and attempt < config.GEMINI_RATE_LIMIT_RETRIES:
                    wait_s = _retry_delay_seconds(e, attempt)
                    attempt += 1
                    logger.warning(
                        "Gemini rate-limited (429); retrying in %.0fs (attempt %d/%d)",
                        wait_s, attempt, config.GEMINI_RATE_LIMIT_RETRIES,
                    )
                    time.sleep(wait_s)
                    continue
                failures.append(f"gemini({config.GEMINI_MODEL}): {e}")
                logger.warning("Gemini failed (%s); failing over to Llama.", type(e).__name__)
                break
            except Exception as e:  # noqa: BLE001 - any other Gemini failure should try the backup
                failures.append(f"gemini({config.GEMINI_MODEL}): {e}")
                logger.warning("Gemini failed (%s); failing over to Llama.", type(e).__name__)
                break
    else:
        failures.append("gemini: no GEMINI_API_KEY set")

    if config.LLAMA_API_KEY:
        try:
            out = llm_client_llama.call_tool(**kwargs)
            LAST_PROVIDER = f"llama:{config.LLAMA_MODEL}"
            logger.info("Served by Llama (%s).", config.LLAMA_MODEL)
            return out
        except Exception as e:  # noqa: BLE001 - report both failures together below
            failures.append(f"llama({config.LLAMA_MODEL}): {e}")
    else:
        failures.append("llama: no LLAMA_API_KEY set")

    LAST_PROVIDER = "none"
    raise RuntimeError(
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
