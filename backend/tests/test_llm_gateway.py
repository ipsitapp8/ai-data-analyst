"""The LLM gateway under provider failure (app/llm_gateway.py).

Every provider is faked at its adapter seam (llm_client.call_tool_gemini, llm_client_llama.call_tool, the local
model server's URL), with the real SDK exception types, so the gateway, the failover in llm_client.call_tool and
the callers' deterministic fallbacks are what is tested. Time is a fake clock: nothing here really sleeps.
"""
from __future__ import annotations

import datetime as dt
import types as pytypes

import httpx
import numpy as np
import openai
import pandas as pd
import pytest
from google.genai import errors

from app import config, llm_gateway, runtime
from app.agents import llm_client, llm_client_llama
from app.analyst import orchestrator, planner

GEMINI_KEY, LLAMA_KEY = "gemini-test-key-SECRET-123", "llama-test-key-SECRET-456"
K = llm_gateway.ErrorKind


class Clock:
    """monotonic() and sleep() for the gateway and the adapter: sleeping advances time instead of waiting."""

    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(round(s, 2))
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    ns = pytypes.SimpleNamespace(monotonic=c.monotonic, sleep=c.sleep)
    monkeypatch.setattr(llm_gateway, "time", ns)
    monkeypatch.setattr(llm_client, "time", ns)
    monkeypatch.setattr(llm_client, "_jitter", lambda: 1.0)
    monkeypatch.setattr(llm_gateway, "jittered", lambda s: s)
    return c


@pytest.fixture
def providers(monkeypatch, clock):
    """Both cloud providers configured; each answers from a script of results / exceptions (the last one repeats)."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setattr(config, "LLAMA_API_KEY", LLAMA_KEY)
    monkeypatch.setattr(config, "LLM_PRIMARY", "gemini")
    monkeypatch.setattr(config, "LLM_FAILOVER_ENABLED", True)
    calls = {"gemini": [], "llama": []}
    script = {"gemini": [{"ok": "gemini"}], "llama": [{"ok": "llama"}]}
    took = {"gemini": 0.0, "llama": 0.0}

    def fake(name):
        def call(**kw):
            calls[name].append({"timeout": llm_gateway.current_attempt_timeout(), "at": clock.t})
            clock.t += took[name]
            step = script[name][min(len(calls[name]) - 1, len(script[name]) - 1)]
            if isinstance(step, BaseException):
                raise step
            return dict(step)
        return call

    monkeypatch.setattr(llm_client, "call_tool_gemini", fake("gemini"))
    monkeypatch.setattr(llm_client_llama, "call_tool", fake("llama"))
    return pytypes.SimpleNamespace(calls=calls, script=script, took=took, clock=clock)


def ask():
    return llm_client.call_tool(system="s", user_content="u", tool_name="submit_x",
                                tool_schema={"type": "object", "properties": {}}, tool_description="d")


def gemini_error(code: int, status: str = "", details: list | None = None, message: str = "") -> Exception:
    body = {"error": {"code": code, "status": status, "message": message, "details": details or []}}
    return (errors.ServerError if code >= 500 else errors.ClientError)(code, body)


def retry_info(seconds: float) -> dict:
    return {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{seconds}s"}


DAILY_QUOTA = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
               "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}


def openai_error(cls, status: int, body: dict | None = None):
    response = httpx.Response(status, request=httpx.Request("POST", "https://llama.invalid/v1/chat/completions"))
    return cls("refused", response=response, body=body)


def gateway(name: str) -> dict:
    return llm_gateway.status()["providers"][name]


# --------------------------------------------------------------------------------------- 1. Gemini timeout --

def test_a_gemini_timeout_fails_over_to_llama_once_without_retrying_gemini(providers):
    providers.script["gemini"] = [httpx.ReadTimeout("timed out")]
    providers.took["gemini"] = 75.0                                 # it used its whole per-request timeout
    assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1, "a timed-out provider is not called again in the same step"
    assert len(providers.calls["llama"]) == 1
    assert llm_client.LAST_PROVIDER == f"llama:{config.LLAMA_MODEL}"
    g, s = gateway("gemini"), llm_gateway.status()
    assert g["last_failure"] == "TIMEOUT" and g["circuit"] == "closed"
    assert s["requests"]["served_by_fallback"] == 1
    # the fallback got only what was left of the step's budget, not a fresh one
    assert providers.calls["llama"][0]["timeout"] <= config.LLM_REQUEST_DEADLINE_SECONDS - 75.0


# ------------------------------------------------------------------------------------------ 2. Gemini 429 --

def test_a_gemini_burst_limit_is_retried_after_its_retry_delay_within_the_budget(providers):
    providers.script["gemini"] = [gemini_error(429, "RESOURCE_EXHAUSTED", [retry_info(4)]), {"ok": "gemini"}]
    assert ask() == {"ok": "gemini"}
    assert len(providers.calls["gemini"]) == 2 and providers.calls["llama"] == []
    assert 4.0 in providers.clock.sleeps, "the provider's RetryInfo delay is honoured"


def test_a_429_whose_wait_does_not_fit_the_budget_goes_straight_to_the_fallback(providers, monkeypatch):
    monkeypatch.setattr(config, "LLM_REQUEST_DEADLINE_SECONDS", 40.0)
    providers.script["gemini"] = [gemini_error(429, "RESOURCE_EXHAUSTED", [retry_info(30)])]
    assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1 and providers.clock.sleeps == []
    assert llm_gateway.breaker("gemini").blocked_for() == pytest.approx(30.0), "Retry-After is remembered"
    # the next step does not call Gemini again while its Retry-After is still running and Llama can answer
    assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1


# ----------------------------------------------------------------------------------- 3. quota exhaustion --

def test_an_exhausted_gemini_quota_is_never_retried_and_opens_its_circuit(providers):
    providers.script["gemini"] = [gemini_error(429, "RESOURCE_EXHAUSTED", [DAILY_QUOTA])]
    assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1 and providers.clock.sleeps == []
    assert gateway("gemini")["circuit"] == "open"
    assert gateway("gemini")["blocked_for_s"] >= config.LLM_QUOTA_COOLDOWN_SECONDS - 1
    for _ in range(5):
        assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1, "no further Gemini calls while the quota is spent"


def test_quota_exhaustion_on_every_provider_is_reported_as_such(providers):
    providers.script["gemini"] = [gemini_error(429, "RESOURCE_EXHAUSTED", [DAILY_QUOTA])]
    providers.script["llama"] = [openai_error(openai.RateLimitError, 429, {"code": "insufficient_quota"})]
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert [f["kind"] for f in e.value.failures] == ["quota_exhausted", "quota_exhausted"]
    assert "quota is used up" in e.value.user_message
    assert len(providers.calls["llama"]) == 1, "a spent quota is not retried on the fallback either"


# --------------------------------------------------------------------------------- 4. authentication error --

def test_a_rejected_gemini_key_fails_over_and_never_counts_towards_the_circuit(providers):
    providers.script["gemini"] = [gemini_error(400, "INVALID_ARGUMENT", [{"reason": "API_KEY_INVALID"}])]
    for _ in range(8):
        assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1, "not retried, and not called again while the key is known bad"
    assert gateway("gemini")["circuit"] == "closed", "an authentication error is not a transient outage"
    assert gateway("gemini")["credentials_rejected"] is True
    assert gateway("gemini")["failures"] == {"AUTH_ERROR": 1}
    providers.clock.t += config.LLM_AUTH_HOLD_SECONDS + 1          # after the hold, the key is tried again
    ask()
    assert len(providers.calls["gemini"]) == 2


def test_with_one_key_rejected_and_the_other_provider_open_routing_goes_deterministic(providers, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 0)
    providers.script["gemini"] = [gemini_error(429, "RESOURCE_EXHAUSTED", [DAILY_QUOTA])]
    providers.script["llama"] = [openai_error(openai.AuthenticationError, 401, {"code": "invalid_api_key"})]
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert [f["kind"] for f in e.value.failures] == ["quota_exhausted", "authentication"]
    assert llm_client.provider_available("submit_understanding") is False
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert [f["kind"] for f in e.value.failures] == ["circuit_open", "auth_rejected"]
    assert providers.calls["gemini"] and len(providers.calls["gemini"]) == 1 and len(providers.calls["llama"]) == 1


# -------------------------------------------------------------------------------- 5. fallback success --

def test_the_fallback_serves_the_step_and_the_metadata_says_so(providers, monkeypatch):
    events = []
    monkeypatch.setattr(runtime, "record_event", lambda kind, name="", detail=None, **kw: events.append(
        {"kind": kind, "name": name, "detail": detail or {}}))
    providers.script["gemini"] = [gemini_error(503, "UNAVAILABLE")]
    assert ask() == {"ok": "llama"}
    gw = [e for e in events if e["kind"] == "gateway"]
    assert gw[-1]["name"] == "llama:ok" and gw[-1]["detail"]["fallback"] is True
    assert {e["detail"]["error"] for e in gw if e["detail"].get("outcome") == "failed"} == {"PROVIDER_UNAVAILABLE"}
    assert all("latency_ms" in e["detail"] for e in gw if e["detail"].get("outcome") != "skipped")
    assert len(providers.calls["gemini"]) == 1 + config.GEMINI_RATE_LIMIT_RETRIES, "503 retried within its bound"


# ----------------------------------------------------------------------- 6. both providers unavailable --

def test_both_providers_down_raises_provider_unavailable_within_the_deadline(providers):
    providers.script["gemini"] = [gemini_error(503, "UNAVAILABLE")]
    providers.script["llama"] = [openai.APIConnectionError(request=httpx.Request("POST", "https://llama.invalid"))]
    start = providers.clock.t
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert providers.clock.t - start <= config.LLM_REQUEST_DEADLINE_SECONDS
    assert {f["provider"] for f in e.value.failures} == {"gemini", "llama"}
    assert GEMINI_KEY not in e.value.user_message and LLAMA_KEY not in e.value.user_message
    assert llm_gateway.status()["requests"]["failed"] == 1


def test_once_both_circuits_are_open_nothing_is_called_and_routing_is_deterministic(providers, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 0)
    monkeypatch.setattr(config, "LLAMA_TRANSIENT_RETRIES", 0)
    providers.script["gemini"] = [gemini_error(503, "UNAVAILABLE")]
    providers.script["llama"] = [openai.APIConnectionError(request=httpx.Request("POST", "https://llama.invalid"))]
    for _ in range(config.LLM_BREAKER_FAILURES):
        with pytest.raises(runtime.ProviderUnavailable):
            ask()
    assert gateway("gemini")["circuit"] == "open" and gateway("llama")["circuit"] == "open"
    made = {k: len(v) for k, v in providers.calls.items()}
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert {k: len(v) for k, v in providers.calls.items()} == made, "open circuits: no calls at all"
    assert {f["kind"] for f in e.value.failures} == {"circuit_open"}
    # callers that check availability first (router, what-if, investigations) now take their deterministic paths
    assert llm_client.provider_available("submit_understanding") is False


def _sales() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({"region": rng.choice(["North", "South", "East"], 60),
                         "revenue": rng.integers(100, 1000, 60).astype(float)})


def test_deterministic_copilot_answers_still_work_when_every_provider_is_down(providers, monkeypatch):
    providers.script["gemini"] = [gemini_error(503, "UNAVAILABLE")]
    providers.script["llama"] = [openai.APIConnectionError(request=httpx.Request("POST", "https://llama.invalid"))]
    monkeypatch.setattr(planner, "MODEL_URL", "http://127.0.0.1:9/v1")      # nothing listens there
    df = _sales()
    out = orchestrator.ask("What is the average revenue by region?", df, {}, filename="sales.csv")
    truth = df.groupby("region")["revenue"].mean()
    text = out["answer"]
    for region, value in truth.items():
        assert region in text and (f"{value:,.2f}" in text or f"{value:.2f}" in text), (region, value, text)
    assert out.get("model_available") in (False, None)
    assert providers.calls == {"gemini": [], "llama": []}, "the Copilot's deterministic tools need no cloud model"


# ------------------------------------------------------------------------------ 7. local model unavailable --

def test_an_unreachable_local_model_opens_its_circuit_and_the_copilot_stays_deterministic(monkeypatch, clock):
    monkeypatch.setattr(planner, "MODEL_URL", "http://127.0.0.1:9/v1")
    url = planner.MODEL_URL
    for _ in range(config.LLM_BREAKER_FAILURES):
        with pytest.raises(Exception) as e:
            planner.model_chat([{"role": "user", "content": "hi"}])
        assert not isinstance(e.value, llm_gateway.ProviderError), "a real call was made and failed"
    assert llm_gateway.local_unavailable(url)
    assert planner.model_available() is False, "the open circuit answers without probing the server"
    with pytest.raises(llm_gateway.ProviderError) as e:
        planner.model_chat([{"role": "user", "content": "hi"}])
    assert e.value.kind == K.PROVIDER_UNAVAILABLE
    assert gateway("local")["failures"] == {"PROVIDER_UNAVAILABLE": config.LLM_BREAKER_FAILURES}

    df = _sales()
    out = orchestrator.ask("What is the total revenue?", df, {}, filename="sales.csv")
    assert f"{df['revenue'].sum():,.0f}" in out["answer"].replace(".00", "")


def test_the_analyses_chat_falls_back_to_the_cloud_when_the_local_model_fails_mid_answer(monkeypatch):
    from app import chat
    monkeypatch.setattr(chat, "_local_url", lambda: "http://127.0.0.1:9/v1")

    def boom(url, user):
        raise openai.APIConnectionError(request=httpx.Request("POST", url))
    monkeypatch.setattr(chat, "_local_answer", boom)
    monkeypatch.setattr(chat.llm_client, "call_tool", lambda **kw: {"answer": "from the cloud", "sources": []})
    monkeypatch.setattr(chat, "build_context", lambda db, question, dash: "Revenue by region: North 410.")
    out = chat.answer_question(None, pytypes.SimpleNamespace(), pytypes.SimpleNamespace(), "why?", [])
    assert out["answer"] == "from the cloud"


# --------------------------------------------------------------------------------- 8. malformed output --

def test_malformed_output_is_retried_once_then_handed_to_the_fallback_without_opening_the_circuit(providers):
    bad = llm_client.MalformedOutput("gemini", "submit_x", {"category": "prose_instead_of_function_call"})
    providers.script["gemini"] = [bad]
    assert ask() == {"ok": "llama"}
    assert len(providers.calls["gemini"]) == 1 + config.GEMINI_MALFORMED_RETRIES
    assert gateway("gemini")["failures"] == {"INVALID_OUTPUT": 1 + config.GEMINI_MALFORMED_RETRIES}
    assert gateway("gemini")["circuit"] == "closed", "an unusable answer is not an outage"


def test_malformed_output_from_every_provider_is_never_presented_as_an_answer(providers):
    providers.script["gemini"] = [llm_client.MalformedOutput("gemini", "submit_x", {"category": "content_filtered"})]
    providers.script["llama"] = [llm_gateway.InvalidOutput("Llama's 'submit_x' reply was unusable: not JSON")]
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert [f["kind"] for f in e.value.failures] == ["malformed_output", "malformed_output"]


def test_the_llama_adapter_rejects_arguments_missing_a_required_field(monkeypatch):
    class Fake:
        def with_options(self, **kw):
            return self

        class chat:  # noqa: N801 - mirrors the SDK's attribute path
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    call = pytypes.SimpleNamespace(function=pytypes.SimpleNamespace(name="submit_x", arguments='{"a": ""}'))
                    return pytypes.SimpleNamespace(usage=None, choices=[pytypes.SimpleNamespace(
                        message=pytypes.SimpleNamespace(tool_calls=[call]))])
    monkeypatch.setattr(llm_client_llama, "get_client", lambda: Fake())
    with pytest.raises(llm_gateway.InvalidOutput):
        llm_client_llama.call_tool(system="s", user_content="u", tool_name="submit_x",
                                   tool_schema={"type": "object", "required": ["a"]}, tool_description="d")


# --------------------------------------------------------------------------- 9. circuit-breaker recovery --

def test_the_circuit_opens_after_five_failures_in_thirty_seconds_and_recovers_through_one_probe(clock):
    b = llm_gateway.CircuitBreaker("p", threshold=5, window=30, cooldown=30, clock=clock.monotonic)
    for _ in range(4):
        b.record_failure(K.TIMEOUT)
        clock.t += 5
    assert b.state() == "closed"
    b.record_failure(K.PROVIDER_UNAVAILABLE)
    assert b.state() == "open" and b.allow() is False
    clock.t += 29.9
    assert b.allow() is False
    clock.t += 0.2
    assert b.state() == "half_open"
    assert b.allow() is True, "one probe"
    assert b.allow() is False, "and only one"
    b.record_success()
    assert b.state() == "closed" and b.allow() and b.allow()


def test_a_failed_probe_reopens_the_circuit_and_slow_failures_never_open_it(clock):
    b = llm_gateway.CircuitBreaker("p", threshold=5, window=30, cooldown=30, clock=clock.monotonic)
    for _ in range(5):
        b.record_failure(K.RATE_LIMITED)
    clock.t += 30
    assert b.allow() is True
    b.record_failure(K.TIMEOUT)
    assert b.state() == "open"
    clock.t += 30
    b.allow()
    b.record_success()
    for _ in range(10):                      # one failure every 8 s: never five within 30 s
        b.record_failure(K.TIMEOUT)
        clock.t += 8
    assert b.state() == "closed"


def test_auth_errors_and_bad_requests_do_not_trip_the_breaker_but_retry_after_extends_it(clock):
    b = llm_gateway.CircuitBreaker("p", threshold=5, window=30, cooldown=30, clock=clock.monotonic)
    for kind in (K.AUTH_ERROR, K.BAD_REQUEST, K.CONTEXT_LIMIT, K.INVALID_OUTPUT) * 3:
        b.record_failure(kind)
    assert b.state() == "closed"
    for _ in range(5):
        b.record_failure(K.RATE_LIMITED, retry_after=90)
    assert b.state() == "open" and b.blocked_for() == pytest.approx(90.0), "the provider's Retry-After wins"


def test_gemini_recovers_end_to_end_after_its_cooldown(providers, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 0)
    providers.script["gemini"] = [gemini_error(503, "UNAVAILABLE")] * 5 + [{"ok": "gemini"}]
    for _ in range(5):
        assert ask() == {"ok": "llama"}
    assert gateway("gemini")["circuit"] == "open"
    made = len(providers.calls["gemini"])
    assert ask() == {"ok": "llama"} and len(providers.calls["gemini"]) == made, "open: skipped"
    providers.clock.t += config.LLM_BREAKER_COOLDOWN_SECONDS + 1
    assert ask() == {"ok": "gemini"}, "the half-open probe succeeds"
    assert gateway("gemini")["circuit"] == "closed"


# -------------------------------------------------------------------------- 10. request deadline exhausted --

def test_no_provider_is_called_when_the_runs_deadline_has_no_room_for_an_attempt(providers):
    ctx = runtime.RunContext(question_id=None)
    ctx.deadline_at = dt.datetime.utcnow() + dt.timedelta(seconds=2)
    with runtime.run_context(ctx), pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert providers.calls == {"gemini": [], "llama": []}
    assert {f["kind"] for f in e.value.failures} == {"no_time"}
    assert "time allowed for this step" in e.value.user_message


def test_a_slow_primary_leaves_the_fallback_only_the_remaining_budget_and_never_overruns(providers, monkeypatch):
    monkeypatch.setattr(config, "LLM_REQUEST_DEADLINE_SECONDS", 60.0)
    providers.script["gemini"] = [httpx.ReadTimeout("timed out")]
    providers.took["gemini"] = 57.0                       # 3 s left: below the minimum for another attempt
    start = providers.clock.t
    with pytest.raises(runtime.ProviderUnavailable) as e:
        ask()
    assert providers.calls["llama"] == [], "no attempt is started without time to finish it"
    assert [f["kind"] for f in e.value.failures] == ["network", "no_time"]
    assert providers.clock.t - start <= 60.0
    assert providers.calls["gemini"][0]["timeout"] == pytest.approx(60.0), "per-attempt timeout cut to the budget"


def test_retries_stop_when_the_budget_is_spent(providers, monkeypatch):
    monkeypatch.setattr(config, "LLM_REQUEST_DEADLINE_SECONDS", 50.0)
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 10)
    providers.script["gemini"] = [gemini_error(503, "UNAVAILABLE")]
    assert ask() == {"ok": "llama"}
    total_wait = sum(providers.clock.sleeps)
    assert len(providers.calls["gemini"]) < 11 and total_wait <= 50.0 - config.LLM_FALLBACK_RESERVE_SECONDS


# ---------------------------------------------------------------------------------- classification --

@pytest.mark.parametrize("exc,kind", [
    (httpx.ReadTimeout("timed out"), K.TIMEOUT),
    (openai.APITimeoutError(request=httpx.Request("POST", "https://x.invalid")), K.TIMEOUT),
    (TimeoutError(), K.TIMEOUT),
    (gemini_error(504, "DEADLINE_EXCEEDED"), K.TIMEOUT),
    (gemini_error(429, "RESOURCE_EXHAUSTED", [retry_info(10)]), K.RATE_LIMITED),
    (gemini_error(429, "RESOURCE_EXHAUSTED", [retry_info(3600)]), K.QUOTA_EXHAUSTED),
    (gemini_error(429, "RESOURCE_EXHAUSTED", [DAILY_QUOTA]), K.QUOTA_EXHAUSTED),
    (openai_error(openai.RateLimitError, 429, {"code": "rate_limit_exceeded"}), K.RATE_LIMITED),
    (openai_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}), K.QUOTA_EXHAUSTED),
    (gemini_error(400, "INVALID_ARGUMENT", [{"reason": "API_KEY_INVALID"}]), K.AUTH_ERROR),
    (openai_error(openai.AuthenticationError, 401, {"code": "invalid_api_key"}), K.AUTH_ERROR),
    (gemini_error(403, "PERMISSION_DENIED"), K.AUTH_ERROR),
    (gemini_error(503, "UNAVAILABLE"), K.PROVIDER_UNAVAILABLE),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://x.invalid")), K.PROVIDER_UNAVAILABLE),
    (ConnectionRefusedError(), K.PROVIDER_UNAVAILABLE),
    (llm_client.MalformedOutput("gemini", "t", {"category": "empty_response"}), K.INVALID_OUTPUT),
    (llm_gateway.InvalidOutput("x"), K.INVALID_OUTPUT),
    (gemini_error(400, "INVALID_ARGUMENT", message="The input token count (2000000) exceeds the maximum number of tokens allowed"),
     K.CONTEXT_LIMIT),
    (openai_error(openai.BadRequestError, 400, {"code": "context_length_exceeded"}), K.CONTEXT_LIMIT),
    (openai_error(openai.APIStatusError, 413, None), K.CONTEXT_LIMIT),
    # A model (or endpoint) that does not exist is a configuration error, not a bad request: never retried, the next
    # provider takes the step. Mistral and OpenRouter say so with a 400 that names the model.
    (gemini_error(404, "NOT_FOUND"), K.MODEL_NOT_FOUND),
    (openai_error(openai.NotFoundError, 404, {"message": "The model `x` does not exist"}), K.MODEL_NOT_FOUND),
    (openai_error(openai.BadRequestError, 400, {"type": "invalid_model", "message": "Invalid model: x"}), K.MODEL_NOT_FOUND),
    (gemini_error(400, "INVALID_ARGUMENT"), K.BAD_REQUEST),
    (openai_error(openai.BadRequestError, 400, {"type": "invalid_request_error"}), K.BAD_REQUEST),
    # 402: no billing on the account
    (openai_error(openai.APIStatusError, 402, {"type": "billing_no_api_budget_configured"}), K.BILLING),
])
def test_every_provider_failure_maps_to_one_normalized_kind(exc, kind):
    assert llm_gateway.classify(exc) == kind


def test_retry_after_headers_are_read_in_seconds():
    response = httpx.Response(429, headers={"retry-after": "7"}, request=httpx.Request("POST", "https://x.invalid"))
    exc = openai.RateLimitError("slow down", response=response, body=None)
    assert llm_gateway.retry_after_seconds(exc) == 7.0
    assert llm_gateway.classify(exc) == K.RATE_LIMITED


# ----------------------------------------------------------------------------------------- adapters --

def test_adapters_bound_their_requests(monkeypatch):
    with llm_gateway.attempt_timeout(12.5):
        assert llm_client._attempt_http_options().timeout == 12500
    assert llm_client._attempt_http_options() is None, "outside the gateway the client's own timeout applies"
    monkeypatch.setattr(config, "LLAMA_API_KEY", LLAMA_KEY)
    monkeypatch.setattr(llm_client_llama, "_client", None)
    c = llm_client_llama.get_client()
    assert c.max_retries == 0, "retries are the gateway's, never hidden in the SDK"
    assert c.timeout == config.LLAMA_REQUEST_TIMEOUT_SECONDS
    monkeypatch.setattr(llm_client_llama, "_client", None)


# --------------------------------------------------------------------------------------- monitoring --

def test_health_reports_each_providers_circuit_without_secrets(client, providers):
    providers.script["gemini"] = [gemini_error(429, "RESOURCE_EXHAUSTED", [DAILY_QUOTA])]
    ask()
    body = client.get("/api/health").json()
    gw = body["llm_gateway"]
    assert gw["primary"] == "gemini" and gw["fallback"] == "llama"
    assert gw["providers"]["gemini"]["circuit"] == "open"
    assert gw["providers"]["gemini"]["last_failure"] == "QUOTA_EXHAUSTED"
    assert gw["providers"]["llama"]["succeeded"] == 1 and gw["providers"]["llama"]["last_latency_ms"] is not None
    assert gw["requests"] == {"requests": 1, "succeeded": 1, "failed": 0, "served_by_fallback": 1}
    assert gw["breaker"] == {"failures": 5, "window_s": 30.0, "cooldown_s": 30.0}
    assert GEMINI_KEY not in str(body) and LLAMA_KEY not in str(body)
