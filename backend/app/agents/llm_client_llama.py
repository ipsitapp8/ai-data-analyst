"""Thin wrapper around a Llama model (OpenAI-compatible endpoint) for
tool-forced structured calls.

Used exclusively by the Critic node (see critic.py) so its independent
verification runs on a genuinely different model family than the
Planner/Executor/Dashboard (which use Gemini, see llm_client.py) -- a
different model catches different mistakes than a different prompt on the
same model does.

Same call_tool/call_text/pretty interface as llm_client.py on purpose, so
critic.py can swap between them with a one-line import change.
"""
from __future__ import annotations

import json
from typing import Any

import time

from openai import OpenAI

from app import config, runtime

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLAMA_API_KEY:
            raise RuntimeError(
                "LLAMA_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=config.LLAMA_API_KEY, base_url=config.LLAMA_BASE_URL)
    return _client


def call_tool(
    system: str,
    user_content: str,
    tool_name: str,
    tool_schema: dict[str, Any],
    tool_description: str,
    max_tokens: int = 4096,
    _fallback: bool = False,
) -> dict[str, Any]:
    """Force the Llama model to respond via a single named tool call and return its args dict.

    `_fallback` marks a call made because the primary provider failed, so the
    run trace shows the switch instead of hiding it."""
    runtime.checkpoint(about_to="llm")
    started = time.monotonic()
    override = runtime.get_provider_override()
    if override is not None:
        out = override("llama", system=system, user_content=user_content, tool_name=tool_name,
                       tool_schema=tool_schema, tool_description=tool_description, max_tokens=max_tokens)
        runtime.note_llm_call("scripted", "scripted", tool_name, 0, 0,
                              int((time.monotonic() - started) * 1000))
        return dict(out)
    client = get_client()
    response = client.chat.completions.create(
        model=config.LLAMA_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_schema,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )

    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None)
    tokens_out = getattr(usage, "completion_tokens", None)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    message = response.choices[0].message
    problem = f"Llama did not return the expected '{tool_name}' tool call"
    for call in message.tool_calls or []:
        if call.function.name == tool_name:
            args, bad = _checked_args(call.function.arguments, tool_schema)
            if bad is None:
                runtime.note_llm_call("llama", config.LLAMA_MODEL, tool_name, tokens_in, tokens_out, elapsed_ms,
                                      fallback=_fallback)
                return args
            problem = f"Llama's '{tool_name}' reply was unusable: {bad}"
            break

    runtime.note_llm_call("llama", config.LLAMA_MODEL, tool_name, tokens_in, tokens_out, elapsed_ms,
                          fallback=_fallback, ok=False)
    raise RuntimeError(problem)


def _checked_args(raw: str, schema: dict[str, Any]) -> tuple[dict | None, str | None]:
    """Parse a tool call's arguments and check they hold the schema's required fields.

    Smaller models sometimes send broken JSON or leave out a required field (a step with no `code`).
    Returned as a failure here, the step is handed to the backup provider; returned as-is it would
    crash the run later with a bare KeyError."""
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return None, "the arguments were not valid JSON"
    if not isinstance(args, dict):
        return None, "the arguments were not a JSON object"
    for key in schema.get("required") or []:
        value = args.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, f"the required field '{key}' was missing or empty"
    return args, None


def call_text(system: str, user_content: str, max_tokens: int = 2048) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=config.LLAMA_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
