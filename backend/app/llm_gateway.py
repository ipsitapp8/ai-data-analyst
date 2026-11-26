"""LLM gateway: which model provider may be called right now, for how long, and what happened.

Every model call in Silt goes through one of three adapters:

- ``gemini``: the Gemini API (app/agents/llm_client.py), the primary cloud provider;
- ``llama``: an OpenAI-compatible endpoint (app/agents/llm_client_llama.py, Groq by default), used only when
  LLAMA_API_KEY is set;
- ``local``: the local Qwen3 + Silt LoRA server (ANALYST_MODEL_URL), which only the Copilot's own prompts use.

This module does not call any provider itself. It gives those adapters:

- one error vocabulary (:class:`ErrorKind`), built from status codes and identifiers, never from response text;
- an independent circuit breaker per provider (and per local server URL);
- a deadline per request, so retries and fallbacks fit inside one overall time budget;
- a status snapshot for /api/health: provider, circuit state, last failure kind, latency. No keys, no prompts.

Calculations never happen here: Silt's deterministic engine remains the only source of numbers. When no provider can
serve a step, callers take the deterministic paths they already have.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import logging
import random
import re
import threading
import time
from contextvars import ContextVar
from enum import Enum
from typing import Any, Callable, Iterator

from app import config, runtime

logger = logging.getLogger(__name__)


class ErrorKind(str, Enum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_ERROR = "AUTH_ERROR"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    # Not a provider fault: the request itself was refused (400/404/422, a model name that does not exist).
    BAD_REQUEST = "BAD_REQUEST"


# Failures that say "this provider is struggling right now". Only these count towards opening a circuit.
# Authentication errors, bad requests, context limits and malformed output do not: retrying them elsewhere may help,
# but they say nothing about the provider's availability.
TRANSIENT = frozenset({ErrorKind.TIMEOUT, ErrorKind.RATE_LIMITED, ErrorKind.PROVIDER_UNAVAILABLE})


class InvalidOutput(RuntimeError):
    """The provider answered, but not with the structured output that was required."""


class ProviderError(RuntimeError):
    """A provider call that was refused before it was made (circuit open, no time left), or a failure normalized
    by an adapter. ``kind`` is an :class:`ErrorKind`; the message never holds response text."""

    def __init__(self, provider: str, kind: ErrorKind, message: str = "", retry_after: float | None = None):
        super().__init__(message or f"{provider}: {kind.value}")
        self.provider, self.kind, self.retry_after = provider, kind, retry_after


# ------------------------------------------------------------------------------------------- classification --

def _status(exc: BaseException) -> int | None:
    for v in (getattr(exc, "code", None), getattr(exc, "status_code", None)):
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def _error_body(exc: BaseException) -> dict:
    """The structured error body (Gemini's ``details``, OpenAI's ``body``), for identifiers only."""
    for attr in ("details", "body"):
        v = getattr(exc, attr, None)
        if isinstance(v, dict):
            inner = v.get("error", v)
            return inner if isinstance(inner, dict) else {}
    return {}


def _identifiers(exc: BaseException) -> set[str]:
    """Short identifier strings the error carries (codes, types, reasons, quota ids). Free text is skipped."""
    out: set[str] = set()
    body = _error_body(exc)
    for key in ("code", "type", "status"):
        v = body.get(key)
        if isinstance(v, str) and len(v) <= 80:
            out.add(v)
    code = getattr(exc, "code", None)
    if isinstance(code, str) and len(code) <= 80:
        out.add(code)
    for item in body.get("details") or []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("reason"), str):
            out.add(item["reason"])
        for violation in item.get("violations") or []:
            if isinstance(violation, dict):
                for key in ("quotaId", "quotaMetric"):
                    if isinstance(violation.get(key), str):
                        out.add(violation[key])
    return out


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the provider asked us to wait: a Retry-After header in seconds, or Gemini's RetryInfo delay."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        raw = headers.get("retry-after") if headers is not None else None
        if raw:
            value = float(str(raw).strip())
            if value >= 0:
                return value
    except (TypeError, ValueError, AttributeError):
        pass
    for item in _error_body(exc).get("details") or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            match = re.match(r"([\d.]+)", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


_QUOTA_IDS = re.compile(r"insufficient_quota|PerDay|PerMonth|billing", re.I)
_KEY_REJECTED = {"API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_SERVICE_BLOCKED", "API_KEY_HTTP_REFERRER_BLOCKED",
                 "API_KEY_IP_ADDRESS_BLOCKED", "invalid_api_key", "UNAUTHENTICATED", "PERMISSION_DENIED"}
# Matched against the error text without recording it: providers report a too-long prompt only in prose.
_CONTEXT_TEXT = re.compile(r"context[_ ](length|window)|maximum context|too many tokens|prompt is too long|"
                           r"token count .{0,40}exceeds|exceeds the maximum number of tokens", re.I)


def classify(exc: BaseException) -> ErrorKind:
    """One :class:`ErrorKind` for any exception an adapter raised."""
    if isinstance(exc, ProviderError):
        return exc.kind
    if isinstance(exc, InvalidOutput):
        return ErrorKind.INVALID_OUTPUT
    status, ids = _status(exc), _identifiers(exc)
    if "context_length_exceeded" in ids or status == 413:
        return ErrorKind.CONTEXT_LIMIT
    if status == 429 or "rate_limit_exceeded" in ids:
        if any(_QUOTA_IDS.search(i) for i in ids):
            return ErrorKind.QUOTA_EXHAUSTED
        wait = retry_after_seconds(exc)
        if wait is not None and wait > config.LLM_QUOTA_RETRY_AFTER_SECONDS:
            return ErrorKind.QUOTA_EXHAUSTED          # a wait of many minutes is a spent quota, not a burst
        return ErrorKind.RATE_LIMITED
    if status in (401, 403) or ids & _KEY_REJECTED:
        return ErrorKind.AUTH_ERROR
    if status in (408, 504):
        return ErrorKind.TIMEOUT
    if status is not None and status >= 500:
        return ErrorKind.PROVIDER_UNAVAILABLE
    if status is not None and 400 <= status < 500:
        return ErrorKind.CONTEXT_LIMIT if _CONTEXT_TEXT.search(str(exc)) else ErrorKind.BAD_REQUEST
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name or "timed out" in str(exc).lower()[:200]:
        return ErrorKind.TIMEOUT
    if isinstance(exc, (ConnectionError, OSError)) or any(w in name for w in ("Connect", "Network", "Transport", "URLError")):
        return ErrorKind.PROVIDER_UNAVAILABLE
    if isinstance(exc, (ValueError, KeyError, TypeError)) or name in ("ValidationError", "JSONDecodeError"):
        return ErrorKind.INVALID_OUTPUT
    if "did not return the expected" in str(exc)[:300] or "reply was unusable" in str(exc)[:300]:
        return ErrorKind.INVALID_OUTPUT
    return ErrorKind.PROVIDER_UNAVAILABLE


# ------------------------------------------------------------------------------------------- circuit breaker --

def _monotonic() -> float:
    return time.monotonic()          # looked up at call time, so a test can replace this module's clock


class CircuitBreaker:
    """Closed -> open after ``threshold`` transient failures within ``window`` seconds; open for ``cooldown``
    seconds (or longer, if the provider's Retry-After says so); then half-open: exactly one probe call is let
    through. The probe's success closes the circuit, its failure re-opens it."""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, name: str, threshold: int | None = None, window: float | None = None,
                 cooldown: float | None = None, clock: Callable[[], float] | None = None):
        self.name = name
        self.threshold = threshold or config.LLM_BREAKER_FAILURES
        self.window = window or config.LLM_BREAKER_WINDOW_SECONDS
        self.cooldown = cooldown or config.LLM_BREAKER_COOLDOWN_SECONDS
        self.clock = clock or _monotonic
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures: list[float] = []
        self._open_until = 0.0
        self._not_before = 0.0          # Retry-After of a rate limit that did not (yet) open the circuit
        self._auth_until = 0.0          # the provider refused its key: not a transient failure, but not worth calling
        self._probe_out = False
        self.opened_count = 0

    def _advance(self, now: float) -> None:
        if self._state == self.OPEN and now >= self._open_until:
            self._state, self._probe_out = self.HALF_OPEN, False

    def state(self) -> str:
        with self._lock:
            self._advance(self.clock())
            return self._state

    def blocked_for(self) -> float:
        """Seconds until a call is worth making: the open period, or a rate limit's Retry-After. 0 when callable."""
        with self._lock:
            now = self.clock()
            self._advance(now)
            if self._state == self.OPEN:
                return max(0.0, self._open_until - now)
            return max(0.0, self._not_before - now)

    def credentials_rejected(self) -> bool:
        """The provider refused the configured key within the last LLM_AUTH_HOLD_SECONDS. Kept apart from the
        circuit: an authentication error is never counted as a transient failure."""
        with self._lock:
            return self.clock() < self._auth_until

    def allow(self) -> bool:
        """Whether a call may be made now. In half-open state this claims the single probe."""
        with self._lock:
            self._advance(self.clock())
            if self._state == self.CLOSED:
                return True
            if self._state == self.HALF_OPEN and not self._probe_out:
                self._probe_out = True
                return True
            return False

    def _open(self, now: float, seconds: float) -> None:
        self._state, self._probe_out = self.OPEN, False
        self._open_until = now + seconds
        self._failures.clear()
        self.opened_count += 1
        logger.warning("llm_gateway circuit=%s state=open for_s=%.0f", self.name, seconds)

    def record_success(self) -> None:
        with self._lock:
            if self._state != self.CLOSED:
                logger.info("llm_gateway circuit=%s state=closed", self.name)
            self._state, self._probe_out = self.CLOSED, False
            self._failures.clear()
            self._not_before = 0.0
            self._auth_until = 0.0

    def record_failure(self, kind: ErrorKind, retry_after: float | None = None) -> None:
        with self._lock:
            now = self.clock()
            self._advance(now)
            if kind == ErrorKind.QUOTA_EXHAUSTED:
                # never hammered: open at once, for as long as the provider says (or a long default)
                self._open(now, max(self.cooldown, retry_after or config.LLM_QUOTA_COOLDOWN_SECONDS))
                return
            if kind == ErrorKind.AUTH_ERROR:
                self._auth_until = now + config.LLM_AUTH_HOLD_SECONDS
            if kind not in TRANSIENT:
                self._probe_out = False          # the probe told us nothing about availability: let another one try
                return
            if kind == ErrorKind.RATE_LIMITED and retry_after:
                self._not_before = max(self._not_before, now + retry_after)
            if self._state == self.HALF_OPEN:
                self._open(now, max(self.cooldown, retry_after or 0.0))
                return
            self._failures = [t for t in self._failures if now - t < self.window] + [now]
            if len(self._failures) >= self.threshold:
                self._open(now, max(self.cooldown, retry_after or 0.0))

    def release(self) -> None:
        """A probe ended without a verdict (the run was cancelled): free the probe slot."""
        with self._lock:
            self._probe_out = False


# ---------------------------------------------------------------------------------------------------- deadline --

class Deadline:
    """One request's overall time budget, never longer than the run's own deadline."""

    def __init__(self, seconds: float | None = None, clock: Callable[[], float] | None = None):
        budget = float(seconds if seconds is not None else config.LLM_REQUEST_DEADLINE_SECONDS)
        self.clock = clock or _monotonic
        self.budget = runtime.remaining_seconds(budget) if runtime.current() is not None else budget
        self.at = self.clock() + self.budget

    def remaining(self) -> float:
        return max(0.0, self.at - self.clock())

    def fits(self, seconds: float) -> bool:
        return self.remaining() >= seconds

    def attempt_timeout(self, provider_timeout: float) -> float:
        """The timeout for one attempt: the provider's own, cut to what is left of the budget."""
        return max(1.0, min(float(provider_timeout), self.remaining()))


_attempt_timeout: ContextVar[float | None] = ContextVar("llm_attempt_timeout", default=None)


@contextlib.contextmanager
def attempt_timeout(seconds: float | None) -> Iterator[None]:
    """Adapters read this for the per-request timeout of the call they are about to make."""
    token = _attempt_timeout.set(seconds)
    try:
        yield
    finally:
        _attempt_timeout.reset(token)


def current_attempt_timeout() -> float | None:
    return _attempt_timeout.get()


def jittered(seconds: float) -> float:
    return max(0.0, seconds * (1.0 + random.uniform(-0.2, 0.2)))


# ----------------------------------------------------------------------------------------------------- registry --

_lock = threading.Lock()
_breakers: dict[str, CircuitBreaker] = {}
_stats: dict[str, dict[str, Any]] = {}
_totals = {"requests": 0, "succeeded": 0, "failed": 0, "served_by_fallback": 0}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def breaker(provider: str) -> CircuitBreaker:
    with _lock:
        if provider not in _breakers:
            _breakers[provider] = CircuitBreaker(provider)
            _stats[provider] = {"calls": 0, "succeeded": 0, "failed": 0, "skipped": 0, "failures": {},
                                "last_latency_ms": None, "last_failure": None, "last_failure_at": None,
                                "last_success_at": None}
        return _breakers[provider]


def local_provider(url: str) -> str:
    """The breaker name of a local model server: one per URL, so the planner and a separate decomposer server
    fail independently. URLs here are local addresses, never credentials."""
    return "local" if url.rstrip("/") == str(_default_local_url()).rstrip("/") else f"local:{url.rstrip('/')}"


def _default_local_url() -> str:
    from app.analyst import planner
    return planner.MODEL_URL


def reset() -> None:
    """Forget every breaker and counter (tests, and an operator after fixing a key)."""
    with _lock:
        _breakers.clear()
        _stats.clear()
        for k in _totals:
            _totals[k] = 0


def note_success(provider: str, latency_ms: int, *, fallback: bool = False) -> None:
    breaker(provider).record_success()
    with _lock:
        s = _stats[provider]
        s["calls"] += 1
        s["succeeded"] += 1
        s["last_latency_ms"] = latency_ms
        s["last_success_at"] = _now_iso()
    runtime.record_event("gateway", f"{provider}:ok", {"provider": provider, "outcome": "ok", "latency_ms": latency_ms,
                                                        "fallback": fallback, "circuit": breaker(provider).state()})


def note_failure(provider: str, exc: BaseException, latency_ms: int) -> ErrorKind:
    kind = classify(exc)
    wait = getattr(exc, "retry_after", None) if isinstance(exc, ProviderError) else retry_after_seconds(exc)
    breaker(provider).record_failure(kind, wait)
    with _lock:
        s = _stats[provider]
        s["calls"] += 1
        s["failed"] += 1
        s["failures"][kind.value] = s["failures"].get(kind.value, 0) + 1
        s["last_latency_ms"] = latency_ms
        s["last_failure"] = kind.value
        s["last_failure_at"] = _now_iso()
    runtime.record_event("gateway", f"{provider}:{kind.value}",
                         {"provider": provider, "outcome": "failed", "error": kind.value, "latency_ms": latency_ms,
                          "retry_after_s": wait, "circuit": breaker(provider).state()})
    return kind


def note_skipped(provider: str, why: str) -> None:
    breaker(provider)
    with _lock:
        _stats[provider]["skipped"] += 1
    runtime.record_event("gateway", f"{provider}:skipped", {"provider": provider, "outcome": "skipped", "why": why,
                                                             "circuit": breaker(provider).state()})


def note_request(ok: bool, *, fallback: bool = False) -> None:
    with _lock:
        _totals["requests"] += 1
        _totals["succeeded" if ok else "failed"] += 1
        if ok and fallback:
            _totals["served_by_fallback"] += 1


def admit(provider: str, deadline: Deadline | None = None, *, min_seconds: float | None = None,
          wait_ok: bool = True) -> str | None:
    """None when ``provider`` may be called now; otherwise why not ("circuit_open", "rate_limited", "no_time").
    A rate limit's Retry-After still running is waited out only when ``wait_ok`` (no other provider can take the
    step) and the wait fits the deadline; otherwise the provider is skipped."""
    b = breaker(provider)
    need = config.LLM_MIN_ATTEMPT_SECONDS if min_seconds is None else min_seconds
    if deadline is not None and not deadline.fits(need):
        return "no_time"
    if b.state() == CircuitBreaker.OPEN:
        return "circuit_open"
    if b.credentials_rejected():
        return "auth_rejected"
    wait = b.blocked_for()
    if wait > 0:
        if not wait_ok or deadline is None or not deadline.fits(wait + need):
            return "rate_limited"
        time.sleep(wait)                         # a short Retry-After that fits the budget: honour it, then go
    return None if b.allow() else "circuit_open"


def guarded(provider: str, fn: Callable[[], Any], *, deadline: Deadline | None = None, timeout: float | None = None,
            fallback: bool = False, wait_ok: bool = True) -> Any:
    """Make one call to ``provider`` under its circuit breaker and the deadline. Raises :class:`ProviderError` without
    calling when the provider is not admitted; re-raises the adapter's own exception (after recording it) otherwise."""
    why = admit(provider, deadline, wait_ok=wait_ok)
    if why is not None:
        note_skipped(provider, why)
        kind = {"no_time": ErrorKind.TIMEOUT, "rate_limited": ErrorKind.RATE_LIMITED,
                "auth_rejected": ErrorKind.AUTH_ERROR}.get(why, ErrorKind.PROVIDER_UNAVAILABLE)
        raise ProviderError(provider, kind, f"{provider} was not called ({why.replace('_', ' ')})")
    per_call = deadline.attempt_timeout(timeout) if (deadline is not None and timeout) else timeout
    started = time.monotonic()
    try:
        with attempt_timeout(per_call):
            out = fn()
    except runtime.RunAborted:
        breaker(provider).release()
        raise
    except Exception as e:  # noqa: BLE001 - recorded, then the caller decides
        note_failure(provider, e, int((time.monotonic() - started) * 1000))
        raise
    note_success(provider, int((time.monotonic() - started) * 1000), fallback=fallback)
    return out


def local_unavailable(url: str) -> bool:
    """True while the local server's circuit is open: the Copilot then answers with its deterministic tools at once
    instead of waiting for timeouts. Peeks only; never claims the half-open probe."""
    return breaker(local_provider(url)).state() == CircuitBreaker.OPEN


def status() -> dict:
    """Provider health for /api/health. Identifiers and counts only."""
    from app.analyst import planner
    configured = {"gemini": bool(config.GEMINI_API_KEY),
                  "llama": bool(config.LLAMA_API_KEY) and (config.LLM_FAILOVER_ENABLED or config.LLM_PRIMARY == "llama"),
                  "local": bool(planner.MODEL_URL)}
    for name in ("gemini", "llama", "local"):
        breaker(name)
    with _lock:
        names = list(_breakers)
        snap = {n: dict(_stats[n], failures=dict(_stats[n]["failures"])) for n in names}
        totals = dict(_totals)
    providers = {}
    for n in names:
        b = _breakers[n]
        providers[n] = {"configured": configured.get(n.split(":")[0], True), "circuit": b.state(),
                        "credentials_rejected": b.credentials_rejected(),
                        "blocked_for_s": round(b.blocked_for(), 1), "times_opened": b.opened_count, **snap[n]}
    order = ["llama", "gemini"] if config.LLM_PRIMARY == "llama" else ["gemini", "llama"]
    return {"primary": order[0], "fallback": order[1] if configured[order[1]] else None,
            "request_deadline_s": config.LLM_REQUEST_DEADLINE_SECONDS,
            "auth_hold_s": config.LLM_AUTH_HOLD_SECONDS,
            "breaker": {"failures": config.LLM_BREAKER_FAILURES, "window_s": config.LLM_BREAKER_WINDOW_SECONDS,
                        "cooldown_s": config.LLM_BREAKER_COOLDOWN_SECONDS},
            "providers": providers, "requests": totals}
