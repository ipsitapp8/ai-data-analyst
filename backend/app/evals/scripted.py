"""A scripted model provider for deterministic evaluation.

It stands in for Gemini and Llama behind the same provider interface the
agents already use (`runtime.provider_override`). Each tool name has a queue of
canned replies; the last reply repeats, so a case can say "the executor always
returns this broken script" with a single entry.

Everything downstream of the model is real: the graph, routing, the sandbox
actually executing the scripted code, verification, publishing. That is the
point -- the model is the only non-deterministic part of the pipeline, so
replacing only the model makes the whole pipeline testable offline.
"""
from __future__ import annotations

import copy
import threading


class ScriptError(RuntimeError):
    """The pipeline asked for a tool the case did not script."""


class ScriptedProvider:
    def __init__(self, script: dict[str, list[dict]]):
        self._script = {tool: list(replies) for tool, replies in script.items()}
        self._index: dict[str, int] = {}
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []  # (provider hint, tool name)

    def __call__(self, provider_hint: str, *, tool_name: str, **_: object) -> dict:
        with self._lock:
            self.calls.append((provider_hint, tool_name))
            replies = self._script.get(tool_name)
            if not replies:
                raise ScriptError(f"no scripted reply for tool '{tool_name}'")
            i = min(self._index.get(tool_name, 0), len(replies) - 1)
            self._index[tool_name] = self._index.get(tool_name, 0) + 1
            return copy.deepcopy(replies[i])

    def count(self, tool_name: str | None = None) -> int:
        return len(self.calls) if tool_name is None else sum(1 for _, t in self.calls if t == tool_name)
