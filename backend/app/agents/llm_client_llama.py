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

from openai import OpenAI

from app import config

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
) -> dict[str, Any]:
    """Force the Llama model to respond via a single named tool call and return its args dict."""
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

    message = response.choices[0].message
    for call in message.tool_calls or []:
        if call.function.name == tool_name:
            return json.loads(call.function.arguments)

    raise RuntimeError(f"Llama did not return the expected '{tool_name}' tool call")


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
