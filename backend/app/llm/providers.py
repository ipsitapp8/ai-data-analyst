"""The hybrid providers: their settings, and one OpenAI-compatible adapter they share.

Groq, NVIDIA NIM, Mistral and OpenRouter all accept the OpenAI chat-completions API, so they share one adapter that
the existing SDK (``openai``) drives; what differs between them -- endpoint, model, headers, whether they are enabled
-- lives in :class:`ProviderSpec`. Gemini and the legacy Llama endpoint are not here: they keep their own adapters
(app/agents/llm_client.py, app/agents/llm_client_llama.py), unchanged.

The adapter's contract matches the existing ones: ``call_tool`` forces one named tool call and returns its arguments
as a dict, or raises; ``call_text`` returns plain text. A tool call is accepted only when the tool name matches and
the arguments are a JSON object that holds the schema's required fields with the right types; anything else raises
``llm_gateway.InvalidOutput``, never a half-valid "success".
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from openai import BadRequestError, OpenAI

from app import config, llm_gateway, runtime
from app.agents.llm_client_llama import _checked_args

log = logging.getLogger(__name__)

HYBRID_PROVIDERS = ("groq", "nvidia", "mistral", "openrouter")
# Every provider a route may name. "gemini" and "llama" use their existing adapters.
ALL_PROVIDERS = ("gemini", "groq", "nvidia", "openrouter", "mistral", "llama")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    api_key: str
    model: str
    base_url: str
    enabled: bool = True
    headers: dict = field(default_factory=dict)

    def state(self) -> str:
        """configured | disabled | missing_key | missing_model. Never "healthy": that takes a successful call."""
        if not self.enabled:
            return "disabled"
        if not self.api_key:
            return "missing_key"
        if not self.model:
            return "missing_model"
        return "configured"

    def public(self) -> dict:
        """What health and provenance may show: no key, and only the endpoint's host."""
        return {"state": self.state(), "model": self.model or None, "endpoint_host": urlparse(self.base_url).hostname,
                "enabled": self.enabled}


def spec(name: str) -> ProviderSpec:
    """The provider's settings, read from config at call time (so a test or an operator change takes effect)."""
    if name == "gemini":
        return ProviderSpec("gemini", config.GEMINI_API_KEY, config.GEMINI_MODEL, "https://generativelanguage.googleapis.com")
    if name == "llama":
        enabled = config.LLM_FAILOVER_ENABLED or config.LLM_PRIMARY == "llama"
        return ProviderSpec("llama", config.LLAMA_API_KEY, config.LLAMA_MODEL, config.LLAMA_BASE_URL, enabled=enabled)
    if name == "groq":
        return ProviderSpec("groq", config.GROQ_API_KEY, config.GROQ_MODEL, config.GROQ_BASE_URL)
    if name == "nvidia":
        return ProviderSpec("nvidia", config.NVIDIA_API_KEY, config.NVIDIA_MODEL, config.NVIDIA_BASE_URL)
    if name == "mistral":
        return ProviderSpec("mistral", config.MISTRAL_API_KEY, config.MISTRAL_MODEL, config.MISTRAL_BASE_URL,
                            enabled=config.MISTRAL_ENABLED)
    if name == "openrouter":
        headers = {k: v for k, v in (("HTTP-Referer", config.OPENROUTER_HTTP_REFERER),
                                     ("X-Title", config.OPENROUTER_APP_TITLE)) if v}
        return ProviderSpec("openrouter", config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL, config.OPENROUTER_BASE_URL,
                            headers=headers)
    raise KeyError(name)


# ----------------------------------------------------------------------------------------------- the adapter --

_clients: dict[str, tuple[str, OpenAI]] = {}


def _client(s: ProviderSpec) -> OpenAI:
    """One client per provider, rebuilt if its key, endpoint or headers change. Bounded timeout and no SDK retries:
    retries and fallback are the gateway's, so a step never makes hidden duplicate calls past its deadline."""
    fingerprint = hashlib.sha256(f"{s.api_key}|{s.base_url}|{sorted(s.headers.items())}".encode()).hexdigest()
    cached = _clients.get(s.name)
    if cached and cached[0] == fingerprint:
        return cached[1]
    if not s.api_key:
        raise RuntimeError(f"{s.name.upper()}_API_KEY is not set.")
    client = OpenAI(api_key=s.api_key, base_url=s.base_url, timeout=config.LLM_PROVIDER_TIMEOUT_SECONDS, max_retries=0,
                    default_headers=s.headers or None)
    _clients[s.name] = (fingerprint, client)
    return client


def validate_arguments(raw: Any, schema: dict[str, Any]) -> tuple[dict | None, str | None]:
    """Silt's required-field check (shared with the Llama adapter), then the declared top-level types and enums.
    Returns (arguments, None) or (None, why it is unusable)."""
    args, bad = _checked_args(raw if isinstance(raw, str) else json.dumps(raw), schema)
    if bad is not None:
        return None, bad
    checks = {"string": lambda v: isinstance(v, str), "array": lambda v: isinstance(v, list),
              "object": lambda v: isinstance(v, dict), "boolean": lambda v: isinstance(v, bool),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)}
    for name, prop in (schema.get("properties") or {}).items():
        if name not in args or args[name] is None or not isinstance(prop, dict):
            continue
        check = checks.get(prop.get("type"))
        if check is not None and not check(args[name]):
            return None, f"the field '{name}' was not a {prop.get('type')}"
        if "enum" in prop and args[name] not in prop["enum"]:
            return None, f"the field '{name}' was not one of the allowed values"
    return args, None


def _not_model_or_billing(exc: BaseException) -> bool:
    return llm_gateway.classify(exc) not in (llm_gateway.ErrorKind.MODEL_NOT_FOUND, llm_gateway.ErrorKind.BILLING,
                                             llm_gateway.ErrorKind.AUTH_ERROR, llm_gateway.ErrorKind.CONTEXT_LIMIT)


def call_tool(s: ProviderSpec, *, system: str, user_content: str, tool_name: str, tool_schema: dict[str, Any],
              tool_description: str, max_tokens: int = 4096, model: str | None = None, fallback: bool = False,
              ) -> tuple[dict[str, Any], str | None]:
    """Force one named tool call. Returns (arguments, served model id)."""
    runtime.checkpoint(about_to="llm")
    model = model or s.model
    started = time.monotonic()
    client = _client(s)
    per_call = llm_gateway.current_attempt_timeout()
    if per_call:
        client = client.with_options(timeout=per_call)
    tools = [{"type": "function", "function": {"name": tool_name, "description": tool_description,
                                               "parameters": tool_schema}}]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    forced = {"type": "function", "function": {"name": tool_name}}
    # As the Llama adapter does: a provider that refuses a forced tool_choice with a 400 (Groq's tool_use_failed) is
    # asked once more with the looser "required" and a larger budget. Never for a 400 that names the model, billing
    # or the key: those are configuration errors that a retry cannot fix.
    attempts = [(forced, max_tokens), ("required", max_tokens + 2048)]
    response, last_error = None, None
    for i, (choice, budget) in enumerate(attempts):
        if i and time.monotonic() - started + config.LLM_MIN_ATTEMPT_SECONDS > (per_call or float("inf")):
            break
        try:
            response = client.chat.completions.create(model=model, max_tokens=budget, messages=messages, tools=tools,
                                                      tool_choice=choice)
            break
        except BadRequestError as exc:
            last_error = exc
            if not _not_model_or_billing(exc):
                raise
            log.warning("provider_event provider=%s tool=%s kind=bad_request action=retry_tool_choice_required",
                        s.name, tool_name)
    if response is None:
        raise last_error  # type: ignore[misc]

    usage = getattr(response, "usage", None)
    tokens_in, tokens_out = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
    served = getattr(response, "model", None) or None
    elapsed_ms = int((time.monotonic() - started) * 1000)
    choices = getattr(response, "choices", None) or []
    message = choices[0].message if choices else None
    problem = f"{s.name.capitalize()} did not return the expected '{tool_name}' tool call"
    if message is None:
        problem = f"{s.name.capitalize()} returned an empty response"
    for tc in (getattr(message, "tool_calls", None) or []):
        if tc.function.name == tool_name:
            args, bad = validate_arguments(tc.function.arguments, tool_schema)
            if bad is None:
                runtime.note_llm_call(s.name, model, tool_name, tokens_in, tokens_out, elapsed_ms, fallback=fallback,
                                      served_model=served if served and served != model else None)
                return args, served
            problem = f"{s.name.capitalize()}'s '{tool_name}' reply was unusable: {bad}"
            break
    runtime.note_llm_call(s.name, model, tool_name, tokens_in, tokens_out, elapsed_ms, fallback=fallback, ok=False,
                          served_model=served if served and served != model else None)
    raise llm_gateway.InvalidOutput(problem)


def call_text(s: ProviderSpec, *, system: str, user_content: str, max_tokens: int = 2048, model: str | None = None,
              fallback: bool = False) -> tuple[str, str | None]:
    """Plain text, through the plain chat call (never the tool-call path). Returns (text, served model id)."""
    runtime.checkpoint(about_to="llm")
    model = model or s.model
    started = time.monotonic()
    client = _client(s)
    per_call = llm_gateway.current_attempt_timeout()
    if per_call:
        client = client.with_options(timeout=per_call)
    response = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user_content}])
    usage = getattr(response, "usage", None)
    served = getattr(response, "model", None) or None
    choices = getattr(response, "choices", None) or []
    text = ((choices[0].message.content if choices else "") or "").strip()
    runtime.note_llm_call(s.name, model, "text", getattr(usage, "prompt_tokens", None),
                          getattr(usage, "completion_tokens", None), int((time.monotonic() - started) * 1000),
                          fallback=fallback, ok=bool(text), served_model=served if served and served != model else None)
    if not text:
        raise llm_gateway.InvalidOutput(f"{s.name.capitalize()} returned an empty text response")
    return text, served
