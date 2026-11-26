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

import ipaddress

from fastapi import HTTPException, Request

# The Streamlit frontend calls this API server-side, so every request arrives from loopback: keyed on the socket
# address, one limit would be shared by every visitor (one person could lock everyone out of signup). The frontend
# forwards the visitor's own address in this header; it is believed only on a loopback connection -- the API listens
# on 127.0.0.1, so that is the frontend on this machine -- and only when it is a valid IP address.
CLIENT_IP_HEADER = "X-Silt-Client-IP"
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def client_ip(request: Request) -> str:
    """The visitor's address for rate limiting: the forwarded one from the local frontend, else the socket's."""
    peer = request.client.host if request.client else "unknown"
    fwd = (request.headers.get(CLIENT_IP_HEADER) or "").strip()
    if fwd and peer in _LOOPBACK:
        try:
            return str(ipaddress.ip_address(fwd))
        except ValueError:
            pass
    return peer

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
