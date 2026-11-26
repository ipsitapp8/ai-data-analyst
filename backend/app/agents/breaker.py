"""Circuit breaker for the model providers.

A provider that has just said "usage limit reached" will say it again a few seconds later, and every further request
spends quota or minutes of a run's time. So after such a failure the provider is left alone for a while: calls go to
the backup, or fail at once so the caller can answer without a model.

State is per process and per provider+model. It is deliberately forgetful: when the cooling period ends the next call
is tried normally, and a success clears everything.
"""
from __future__ import annotations

import threading
import time

from app import config

# Failures that count only when they repeat: one overloaded reply is ordinary, two exhausted calls in a row are not.
_REPEATABLE = frozenset({"overloaded", "server_error", "network"})
_STRIKES_TO_OPEN = 2

_lock = threading.Lock()
_open: dict[str, dict] = {}
_strikes: dict[str, int] = {}


def _now() -> float:
    return time.monotonic()


def blocked(provider: str, model: str) -> dict | None:
    """The failure that put this provider on hold, if it still is: {"kind", "status", "seconds_left"}."""
    key = f"{provider}:{model}"
    with _lock:
        entry = _open.get(key)
        if entry is None:
            return None
        left = entry["until"] - _now()
        if left <= 0:
            del _open[key]
            return None
        return {"kind": entry["kind"], "status": entry["status"], "seconds_left": left}


def failed(provider: str, model: str, failure: dict, *, retry_after: float | None = None,
           quota_exhausted: bool = False) -> float:
    """Record a call that failed after its retries. Returns how long the provider is now on hold (0: not at all)."""
    if config.LLM_COOLDOWN_SECONDS <= 0:
        return 0.0
    kind, key = failure.get("kind"), f"{provider}:{model}"
    with _lock:
        if kind == "rate_limited":
            hold = float(config.LLM_QUOTA_COOLDOWN_SECONDS if quota_exhausted else config.LLM_COOLDOWN_SECONDS)
            # the provider's own waiting time wins when it is longer, but never beyond the quota cooling period
            if retry_after:
                hold = max(hold, min(float(retry_after), float(config.LLM_QUOTA_COOLDOWN_SECONDS)))
        elif kind in _REPEATABLE:
            _strikes[key] = _strikes.get(key, 0) + 1
            if _strikes[key] < _STRIKES_TO_OPEN:
                return 0.0
            hold = float(config.LLM_COOLDOWN_SECONDS)
        else:
            return 0.0
        _open[key] = {"until": _now() + hold, "kind": kind, "status": failure.get("status")}
        return hold


def succeeded(provider: str, model: str) -> None:
    key = f"{provider}:{model}"
    with _lock:
        _open.pop(key, None)
        _strikes.pop(key, None)


def reset() -> None:
    with _lock:
        _open.clear()
        _strikes.clear()
