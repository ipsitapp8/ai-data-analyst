"""Model family identity, for the Critic's independence check.

Independence is a property of the *model*, never of the provider: OpenRouter serving `nvidia/nemotron-*` is the same
family as NVIDIA serving `nvidia/nemotron-*`, and Groq serving `openai/gpt-oss-*` is the OpenAI gpt-oss family. The
family is read from the model ID (requested, or better, the one the provider reports it served) with conservative
rules; anything unrecognised is UNKNOWN, and UNKNOWN is never independent of anything.

An operator who knows better can pin a provider's family with <PROVIDER>_MODEL_FAMILY (e.g. NVIDIA_MODEL_FAMILY).
"""
from __future__ import annotations

import os
import re

UNKNOWN = "unknown"

# Checked in order; the first match wins. Llama comes before Nemotron on purpose: the older Llama-based Nemotrons
# ("llama-3.1-nemotron-...") are Llama derivatives, so they are not independent of a Llama verifier.
_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("google-gemini", re.compile(r"(^|[/_.-])(gemini|gemma)([/_.:-]|\d|$)", re.I)),
    ("openai-gpt-oss", re.compile(r"(^|/)gpt-oss", re.I)),
    ("openai-gpt", re.compile(r"(^|/)(gpt-|o\d)", re.I)),
    ("meta-llama", re.compile(r"llama", re.I)),
    ("nvidia-nemotron", re.compile(r"nemotron", re.I)),
    ("mistral", re.compile(r"(^|/)(mistral|ministral|magistral|codestral|mixtral|devstral|pixtral|voxtral)", re.I)),
    ("qwen", re.compile(r"(^|/)qwen|qwq", re.I)),
    ("deepseek", re.compile(r"(^|/)deepseek", re.I)),
    ("anthropic-claude", re.compile(r"(^|/)claude", re.I)),
    ("moonshot-kimi", re.compile(r"(^|/)kimi", re.I)),
    ("zhipu-glm", re.compile(r"(^|/)glm", re.I)),
    ("minimax", re.compile(r"(^|/)minimax", re.I)),
)


def family_of(provider: str, model: str | None, served_model: str | None = None) -> str:
    """The family of whatever served (or will serve) a call. The served model wins over the requested one, since a
    router may run something else; a pinned <PROVIDER>_MODEL_FAMILY wins over both."""
    pinned = os.getenv(f"{str(provider).split(':')[0].upper()}_MODEL_FAMILY", "").strip().lower()
    if pinned:
        return pinned
    if str(provider).startswith("gemini"):
        return "google-gemini"           # the Gemini API serves only Google models
    for name in (served_model, model):
        if not name:
            continue
        if str(name).lower() in ("openrouter/auto", "openrouter/free") or str(name).lower().startswith("openrouter/"):
            continue                     # a router alias: the model behind it is not known in advance
        for family, rule in _RULES:
            if rule.search(str(name)):
                return family
    return UNKNOWN


def independent(a: str, b: str) -> bool:
    """True only when both families are known and different."""
    return a != UNKNOWN and b != UNKNOWN and a != b


def independent_of_all(family: str, checked: set[str] | frozenset[str]) -> bool:
    """True only when ``family`` is known and differs from every family in ``checked``, and every family in
    ``checked`` is itself known. An empty set means nothing is known about the work, so nothing is independent of it."""
    return bool(checked) and family != UNKNOWN and UNKNOWN not in checked and family not in checked
