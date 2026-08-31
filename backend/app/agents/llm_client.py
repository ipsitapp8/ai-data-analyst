"""Thin wrapper around the Gemini API for tool-forced structured calls.

Kept as a small, provider-specific module so the rest of the agent code
(planner/executor/critic/dashboard) only ever calls call_tool/call_text/pretty
and doesn't know or care which LLM provider is behind them.
"""
from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from app import config
from app.agents import llm_client_llama

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

    Covers both common Gemini failures: a 429 once the free-tier daily cap is
    hit, and the model replying with prose instead of the required function
    call. Either way the run continues on the other provider instead of dying
    mid-pipeline.
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
        try:
            out = call_tool_gemini(**kwargs)
            LAST_PROVIDER = f"gemini:{config.GEMINI_MODEL}"
            return out
        except Exception as e:  # noqa: BLE001 - any Gemini failure should try the backup
            failures.append(f"gemini({config.GEMINI_MODEL}): {e}")
            print(f"[llm] Gemini failed ({type(e).__name__}); failing over to Llama.")
    else:
        failures.append("gemini: no GEMINI_API_KEY set")

    if config.LLAMA_API_KEY:
        try:
            out = llm_client_llama.call_tool(**kwargs)
            LAST_PROVIDER = f"llama:{config.LLAMA_MODEL}"
            print(f"[llm] Served by Llama ({config.LLAMA_MODEL}).")
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
