"""Per-run context: budgets, cancellation, deadlines and the run trace.

A run (one question going through the graph) executes on a worker thread. The
worker opens a RunContext for it; everything expensive the run does -- an LLM
call, a sandbox execution, a graph node -- reports here. That gives three things
from one place:

- bounded runs: `checkpoint()` raises once the run is cancelled, past its
  deadline, or over its LLM-call / sandbox-run / cost budget;
- a trace: every routing decision, node, LLM call and sandbox run is appended
  to `run_events`;
- usage and cost: tokens per provider, with an estimated price.

Code that runs outside a run (insights, chat, tests calling a node directly)
sees no context, and every function here then does nothing.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from app import config

logger = logging.getLogger(__name__)


class RunAborted(Exception):
    """A run stopped on purpose. `failure_class` is what the job records."""

    failure_class = "aborted"
    user_message = "This analysis was stopped."


class RunCancelled(RunAborted):
    failure_class = "cancelled"
    user_message = "This analysis was cancelled."


class DeadlineExceeded(RunAborted):
    failure_class = "timed_out"
    user_message = "This analysis ran past its time limit and was stopped."


class BudgetExceeded(RunAborted):
    failure_class = "budget_exceeded"
    user_message = "This analysis used up its budget of model calls or code runs and was stopped."


class ProviderUnavailable(RunAborted, RuntimeError):
    """No configured model provider could serve a call. Also a RuntimeError, so
    callers outside a run (insights, chat) that already handle provider errors
    keep working unchanged."""

    failure_class = "provider_unavailable"
    user_message = ("No AI model could be reached, so this analysis could not run. The usual causes are a "
                    "missing or invalid API key, an exhausted quota, or a provider outage. Questions that are "
                    "simple aggregations, and root-cause investigations, do not need a model.")


@dataclass
class RunContext:
    question_id: int
    team_id: int | None = None
    deadline_at: dt.datetime | None = None
    should_cancel: Callable[[], bool] | None = None
    max_llm_calls: int = 0
    max_sandbox_runs: int = 0
    max_cost_usd: float = 0.0
    llm_calls: int = 0
    sandbox_runs: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    providers: list[str] = field(default_factory=list)
    fallbacks: int = 0
    record: bool = True

    def usage(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "sandbox_runs": self.sandbox_runs,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "estimated_cost_usd": round(self.cost_usd, 6),
            "providers": list(self.providers),
            "provider_fallbacks": self.fallbacks,
        }


_current: ContextVar[RunContext | None] = ContextVar("run_context", default=None)
# Test/eval hook: when set, every provider call is answered by this callable
# instead of a real model. Signature: (provider_hint, **call_tool kwargs) -> dict.
_provider_override: ContextVar[Callable[..., dict] | None] = ContextVar("provider_override", default=None)


def current() -> RunContext | None:
    return _current.get()


@contextlib.contextmanager
def run_context(ctx: RunContext) -> Iterator[RunContext]:
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


@contextlib.contextmanager
def provider_override(fn: Callable[..., dict]) -> Iterator[None]:
    token = _provider_override.set(fn)
    try:
        yield
    finally:
        _provider_override.reset(token)


def get_provider_override() -> Callable[..., dict] | None:
    return _provider_override.get()


def _spent(count: int, limit: int, about_to_use: bool) -> bool:
    """limit 0 = unlimited; a negative limit = none allowed at all."""
    if limit == 0:
        return False
    allowed = max(limit, 0)
    return count >= allowed if about_to_use else count > allowed


def checkpoint(about_to: str | None = None) -> None:
    """Raise if the current run must stop. Cheap; called before every node,
    LLM call (about_to="llm") and sandbox execution (about_to="sandbox").

    With `about_to`, the budget for that resource is checked *before* it is
    used, so a limit of N means exactly N uses, not N+1."""
    ctx = _current.get()
    if ctx is None:
        return
    if ctx.should_cancel is not None and ctx.should_cancel():
        raise RunCancelled()
    if ctx.deadline_at is not None and dt.datetime.utcnow() >= ctx.deadline_at:
        raise DeadlineExceeded()
    if _spent(ctx.llm_calls, ctx.max_llm_calls, about_to == "llm"):
        raise BudgetExceeded()
    if _spent(ctx.sandbox_runs, ctx.max_sandbox_runs, about_to == "sandbox"):
        raise BudgetExceeded()
    if ctx.max_cost_usd and ctx.cost_usd > ctx.max_cost_usd:
        raise BudgetExceeded()


def remaining_seconds(default: float) -> float:
    """Seconds until the run's deadline, capped at `default`."""
    ctx = _current.get()
    if ctx is None or ctx.deadline_at is None:
        return default
    left = (ctx.deadline_at - dt.datetime.utcnow()).total_seconds()
    return max(1.0, min(default, left))


def is_cancel_requested() -> bool:
    ctx = _current.get()
    if ctx is None:
        return False
    if ctx.should_cancel is not None and ctx.should_cancel():
        return True
    return ctx.deadline_at is not None and dt.datetime.utcnow() >= ctx.deadline_at


def _price(provider: str) -> tuple[float, float]:
    raw = config.LLM_PRICE_GEMINI if provider.startswith("gemini") else config.LLM_PRICE_LLAMA
    try:
        a, b = (float(x) for x in raw.split(","))
        return a, b
    except ValueError:
        return 0.0, 0.0


def estimate_cost(provider: str, tokens_in: int, tokens_out: int) -> float:
    if provider.startswith("scripted"):
        return 0.0
    p_in, p_out = _price(provider)
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def record_event(kind: str, name: str = "", detail: dict | None = None, duration_ms: int | None = None,
                 question_id: int | None = None, team_id: int | None = None) -> None:
    """Append one row to the run trace. Never raises: tracing must not break a run."""
    ctx = _current.get()
    if question_id is None:
        if ctx is None or not ctx.record:
            return
        question_id, team_id = ctx.question_id, ctx.team_id
    try:
        from app.database import SessionLocal
        from app.models import RunEvent

        db = SessionLocal()
        try:
            db.add(RunEvent(team_id=team_id, question_id=question_id, kind=kind, name=name[:120],
                            detail_json=detail or {}, duration_ms=duration_ms))
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        logger.debug("Could not record run event %s/%s", kind, name, exc_info=True)


def note_llm_call(provider: str, model: str, tool_name: str, tokens_in: int | None, tokens_out: int | None,
                  duration_ms: int, fallback: bool = False, ok: bool = True) -> None:
    """Called by the provider adapters after every model call."""
    ctx = _current.get()
    t_in, t_out = int(tokens_in or 0), int(tokens_out or 0)
    cost = estimate_cost(provider, t_in, t_out)
    if ctx is not None:
        ctx.llm_calls += 1
        ctx.tokens_in += t_in
        ctx.tokens_out += t_out
        ctx.cost_usd += cost
        label = f"{provider}:{model}"
        if label not in ctx.providers:
            ctx.providers.append(label)
        if fallback:
            ctx.fallbacks += 1
    record_event("llm_call", tool_name, {
        "provider": provider, "model": model, "tokens_in": t_in, "tokens_out": t_out,
        "tokens_known": tokens_in is not None, "estimated_cost_usd": round(cost, 6),
        "fallback": fallback, "ok": ok,
    }, duration_ms)


def note_sandbox_run(backend: str, step_index: int, attempt: int, success: bool, timed_out: bool,
                     duration_ms: int, extra: dict | None = None) -> None:
    ctx = _current.get()
    if ctx is not None:
        ctx.sandbox_runs += 1
    record_event("sandbox_run", f"step{step_index}", {
        "backend": backend, "attempt": attempt, "success": success, "timed_out": timed_out, **(extra or {}),
    }, duration_ms)


@contextlib.contextmanager
def node_span(name: str) -> Iterator[None]:
    """Trace one graph node, and stop the run here if it must stop."""
    checkpoint()
    started = time.monotonic()
    outcome = "ok"
    try:
        yield
    except RunAborted as e:
        outcome = e.failure_class
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        record_event("node", name, {"outcome": outcome}, int((time.monotonic() - started) * 1000))
