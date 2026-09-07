"""Minimal in-memory rate limiter for auth endpoints.

Deliberately simple: a per-process dict of (key -> recent timestamps), no
external store. That means limits are per-worker (fine for this app's
single-process deployment) and reset on restart -- not a substitute for a
real rate limiter (e.g. Redis-backed) behind a load balancer with multiple
workers, but meaningfully better than no brute-force protection at all.
"""
from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException

_attempts: dict[str, list[float]] = defaultdict(list)
_lock = Lock()


def enforce(key: str, max_attempts: int, window_seconds: float) -> None:
    """Raise 429 if `key` has made >= max_attempts calls in the trailing window.

    Call this at the top of an endpoint with a key that identifies the
    caller for this limit (e.g. f"login:{client_ip}" or f"login:{email}").
    """
    now = time.monotonic()
    cutoff = now - window_seconds
    with _lock:
        recent = [t for t in _attempts[key] if t > cutoff]
        if len(recent) >= max_attempts:
            _attempts[key] = recent
            raise HTTPException(429, "Too many attempts. Try again in a minute.")
        recent.append(now)
        _attempts[key] = recent
