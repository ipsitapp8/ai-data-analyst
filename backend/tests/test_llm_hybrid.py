"""Hybrid multi-provider routing (app/llm/): routing, failover per error kind, structured-output validation, Critic
independence by model family, provenance, and that none of it changes anything while it is switched off.

Every provider API is mocked: the OpenAI-compatible providers at their client (app.llm.providers._client), Gemini at
llm_client.call_tool_gemini. No key, network or database is needed.
"""
from __future__ import annotations

import json
import time
import types

import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from app import config, llm_gateway, runtime
from app.agents import critic, llm_client, llm_client_llama
from app.llm import families, providers, routing

K = llm_gateway.ErrorKind
SCHEMA = {"type": "object",
          "properties": {"verdict": {"type": "string", "enum": ["verified", "rejected"]},
                         "confidence": {"type": "number"}, "issues": {"type": "array", "items": {"type": "string"}}},
          "required": ["verdict", "confidence"]}
GOOD = {"verdict": "verified", "confidence": 0.9, "issues": []}
REAL_GEMINI = llm_client.call_tool_gemini


def kwargs(tool="submit_plan"):
    return dict(system="s", user_content="u", tool_name=tool, tool_schema=SCHEMA, tool_description="d")


def http_error(cls, status, body=None, headers=None):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://p.invalid/v1"))
    return cls("refused", response=response, body=body)


def reply(args=GOOD, tool="submit_plan", model=None, content=None):
    calls = None
    if tool is not None:
        calls = [types.SimpleNamespace(function=types.SimpleNamespace(
            name=tool, arguments=args if isinstance(args, str) else json.dumps(args)))]
    message = types.SimpleNamespace(tool_calls=calls, content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="tool_calls")],
                                 usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7), model=model)


class FakeClient:
    """Answers chat.completions.create from a script: each item is a response or an exception to raise."""

    def __init__(self, name, script):
        self.name, self.script, self.calls = name, list(script), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def with_options(self, **_):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if not self.script:
            raise AssertionError(f"{self.name} called more often than scripted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(**kw)
        return item


@pytest.fixture
def hybrid(monkeypatch):
    """Hybrid routing on, all four hybrid providers configured (Mistral disabled), Gemini configured, no Llama."""
    for name, value in dict(LLM_HYBRID_ROUTING_ENABLED=True, GEMINI_API_KEY="g-key", LLAMA_API_KEY="",
                            GROQ_API_KEY="groq-key", GROQ_MODEL="openai/gpt-oss-120b",
                            NVIDIA_API_KEY="nv-key", NVIDIA_MODEL="nvidia/nemotron-3.5-super-vl-120b-a12b",
                            OPENROUTER_API_KEY="or-key", OPENROUTER_MODEL="nvidia/nemotron-3.5-super-vl:free",
                            MISTRAL_API_KEY="m-key", MISTRAL_MODEL="mistral-medium-2604", MISTRAL_ENABLED=False,
                            LLM_PROVIDER_PRIORITY="", LLM_FAILOVER_ENABLED=True, LLM_PRIMARY="gemini",
                            LLM_PROVIDER_TRANSIENT_RETRIES=1, LLM_MAX_PROVIDER_ATTEMPTS=4,
                            GEMINI_RATE_LIMIT_RETRIES=0, GEMINI_MALFORMED_RETRIES=0,
                            LLM_CRITIC_REQUIRE_INDEPENDENT=False).items():
        monkeypatch.setattr(config, name, value)
    for agent in routing.SHOWN_AGENTS + ("chat",):
        monkeypatch.delenv(f"LLM_{agent.upper()}_PROVIDERS", raising=False)
        monkeypatch.delenv(f"LLM_{agent.upper()}_FALLBACK", raising=False)
    for p in providers.ALL_PROVIDERS:
        monkeypatch.delenv(f"{p.upper()}_MODEL_FAMILY", raising=False)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    routing._last_served.set(None)
    fakes: dict[str, FakeClient] = {}

    def script(name, *items):
        fakes[name] = FakeClient(name, items)
        return fakes[name]

    monkeypatch.setattr(providers, "_client", lambda s: fakes.setdefault(s.name, FakeClient(s.name, [])))
    gemini_calls: list[dict] = []

    def gemini(*items):
        queue = list(items)

        def fake(**kw):
            gemini_calls.append(kw)
            if not queue:
                raise AssertionError("gemini called more often than scripted")
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        monkeypatch.setattr(llm_client, "call_tool_gemini", fake)

    gemini()                                        # by default Gemini must not be called at all
    return types.SimpleNamespace(script=script, fakes=fakes, gemini=gemini, gemini_calls=gemini_calls)


def gemini_down():
    return genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": ""}})


# --------------------------------------------------------------------------------------------- off by default --

def test_hybrid_routing_is_off_by_default_and_new_keys_alone_change_nothing(monkeypatch):
    """With the switch off, configured Groq/NVIDIA/OpenRouter keys and models are never used: the step takes the
    existing Gemini path."""
    monkeypatch.setattr(config, "LLM_HYBRID_ROUTING_ENABLED", False)
    for name, value in dict(GEMINI_API_KEY="g", GROQ_API_KEY="k", GROQ_MODEL="m", NVIDIA_API_KEY="k",
                            NVIDIA_MODEL="m", OPENROUTER_API_KEY="k", OPENROUTER_MODEL="m").items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(providers, "_client", lambda s: (_ for _ in ()).throw(AssertionError(f"{s.name} called")))
    monkeypatch.setattr(llm_client, "call_tool_gemini", lambda **kw: {"via": "gemini"})
    assert llm_client.call_tool(**kwargs()) == {"via": "gemini"}
    assert llm_client.LAST_PROVIDER == f"gemini:{config.GEMINI_MODEL}"
    assert llm_gateway.status()["hybrid"]["enabled"] is False


def test_the_scripted_provider_override_still_answers_every_call_under_hybrid_routing(hybrid, monkeypatch):
    class Scripted:
        calls = []

        def __call__(self, hint, **kw):
            self.calls.append(hint)
            return {"scripted": True}
    s = Scripted()
    monkeypatch.setattr(llm_client, "call_tool_gemini", REAL_GEMINI)   # the override answers inside the adapter
    with runtime.provider_override(s):
        assert llm_client.call_tool(**kwargs()) == {"scripted": True}
        assert critic._critic_call_tool(**kwargs("submit_review")) == ({"scripted": True}, "scripted-critic")
    assert hybrid.fakes == {} and hybrid.gemini_calls == []


# ------------------------------------------------------------------------------------------------- routing --

def test_planner_and_executor_go_gemini_first_then_groq_nvidia_openrouter(hybrid):
    assert [p for p, _ in routing.chain("planner")] == ["gemini", "groq", "nvidia", "openrouter"]
    assert [p for p, _ in routing.chain("executor")] == ["gemini", "groq", "nvidia", "openrouter"]
    assert [p for p, _ in routing.chain("critic")] == ["groq", "nvidia", "openrouter"]
    hybrid.gemini({"via": "gemini"})
    assert llm_client.call_tool(**kwargs("submit_code")) == {"via": "gemini"}
    assert routing.served_for("executor")["family"] == "google-gemini"
    assert hybrid.fakes == {}, "no hybrid provider is called while Gemini serves"


def test_a_gemini_outage_fails_over_to_groq_with_the_model_recorded(hybrid):
    hybrid.gemini(gemini_down())
    hybrid.script("groq", reply(model="openai/gpt-oss-120b"))
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        assert llm_client.call_tool(**kwargs()) == GOOD
    assert llm_client.LAST_PROVIDER == "groq:openai/gpt-oss-120b"
    assert ctx.served["planner"] == {"provider": "groq", "model": "openai/gpt-oss-120b",
                                     "served_model": "openai/gpt-oss-120b", "family": "openai-gpt-oss"}
    assert ctx.llm_calls == 1 and ctx.tokens_in == 11 and ctx.tokens_out == 7 and ctx.fallbacks == 1
    sent = hybrid.fakes["groq"].calls[0]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "submit_plan"}}, "a forced tool call"
    assert sent["tools"][0]["function"]["parameters"] is SCHEMA, "the existing schema is sent unchanged"


def test_an_agent_route_and_a_pinned_model_come_from_configuration(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", "nvidia, groq=openai/gpt-oss-20b, nosuchprovider, groq")
    assert routing.chain("planner") == [("nvidia", None), ("groq", "openai/gpt-oss-20b")]
    hybrid.script("nvidia", http_error(openai.InternalServerError, 500), http_error(openai.InternalServerError, 500))
    hybrid.script("groq", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert hybrid.fakes["groq"].calls[0]["model"] == "openai/gpt-oss-20b"
    monkeypatch.setenv("LLM_PLANNER_FALLBACK", "false")
    assert routing.chain("planner") == [("nvidia", None)]


def test_the_global_priority_applies_to_agents_without_their_own_route_but_not_to_the_critic(hybrid, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER_PRIORITY", "openrouter,gemini")
    assert [p for p, _ in routing.chain("narrator")] == ["openrouter", "gemini"]
    assert [p for p, _ in routing.chain("critic")] == ["groq", "nvidia", "openrouter"]


def test_a_provider_needs_a_key_and_a_model_and_mistral_stays_off_until_enabled(hybrid, monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODEL", "")
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", "mistral,groq,nvidia")
    hybrid.script("nvidia", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert "mistral" not in hybrid.fakes or not hybrid.fakes["mistral"].calls
    assert "groq" not in hybrid.fakes or not hybrid.fakes["groq"].calls
    state = llm_gateway.status()["hybrid"]["providers"]
    assert state["groq"]["state"] == "missing_model" and state["mistral"]["state"] == "disabled"
    monkeypatch.setattr(config, "MISTRAL_ENABLED", True)
    hybrid.script("mistral", reply(model="mistral-medium-2604"))
    assert llm_client.call_tool(**kwargs()) == GOOD and len(hybrid.fakes["mistral"].calls) == 1


def test_mistral_is_not_in_any_default_route(hybrid):
    assert all("mistral" not in [p for p, _ in routing.chain(a)] for a in routing.SHOWN_AGENTS)


def test_at_most_the_configured_number_of_providers_is_tried(hybrid, monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_PROVIDER_ATTEMPTS", 2)
    hybrid.gemini(gemini_down())
    hybrid.script("groq", http_error(openai.AuthenticationError, 401))
    hybrid.script("nvidia", reply())
    with pytest.raises(runtime.ProviderUnavailable):
        llm_client.call_tool(**kwargs())
    assert hybrid.fakes["nvidia"].calls == []


def test_the_legacy_llama_endpoint_is_used_only_when_a_route_names_it_and_keeps_its_own_settings(hybrid, monkeypatch):
    monkeypatch.setattr(config, "LLAMA_API_KEY", "legacy-key")
    monkeypatch.setattr(config, "LLAMA_BASE_URL", "https://api.llama.com/compat/v1/")
    monkeypatch.setattr(config, "LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
    assert providers.spec("groq").base_url == config.GROQ_BASE_URL != config.LLAMA_BASE_URL
    assert providers.spec("groq").api_key == "groq-key", "GROQ_* never reads LLAMA_*"
    called = []
    monkeypatch.setattr(llm_client_llama, "call_tool", lambda **kw: called.append(kw) or {"via": "llama"})
    hybrid.gemini(gemini_down())
    hybrid.script("groq", http_error(openai.AuthenticationError, 401))
    with pytest.raises(runtime.ProviderUnavailable):                   # default route: no llama
        monkeypatch.setenv("LLM_PLANNER_PROVIDERS", "groq")
        llm_client.call_tool(**kwargs())
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", "llama")
    assert llm_client.call_tool(**kwargs()) == {"via": "llama"} and len(called) == 1
    assert routing.family_for("llama") == "meta-llama"


# ------------------------------------------------------------------------------------- failover per error kind --

def only(*names):
    return ",".join(names)


def test_401_moves_on_without_retrying_and_the_key_is_held(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", http_error(openai.AuthenticationError, 401))
    hybrid.script("nvidia", reply(), reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 1, "a refused key is never retried"
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 1, "and not called again while held"


def test_402_holds_the_provider_and_is_never_retried(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", http_error(openai.APIStatusError, 402, {"type": "billing_no_api_budget_configured"}))
    hybrid.script("nvidia", reply(), reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 1
    assert llm_gateway.status()["providers"]["groq"]["billing_held"] is True
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq"))
    assert llm_client.provider_available("submit_plan") is False, "a held provider is not offered"


@pytest.mark.parametrize("exc", [
    http_error(openai.NotFoundError, 404, {"message": "The model `x` does not exist"}),
    http_error(openai.BadRequestError, 400, {"type": "invalid_model", "message": "Invalid model: x"}),
])
def test_a_missing_model_is_a_configuration_error_skipped_without_retry(hybrid, monkeypatch, exc):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", exc)
    hybrid.script("nvidia", reply())
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 1, "neither retried nor re-asked with tool_choice=required"


def test_a_missing_model_is_reported_as_such_when_nothing_serves(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq"))
    hybrid.script("groq", http_error(openai.NotFoundError, 404, {"message": "The model `x` does not exist"}))
    with pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(**kwargs())
    assert e.value.failures == [{"provider": "groq", "kind": "model_not_found", "status": 404,
                                 "model": "openai/gpt-oss-120b"}]
    assert "groq-key" not in str(e.value) and "groq-key" not in e.value.user_message


def test_429_waits_for_retry_after_then_retries_once_before_moving_on(hybrid, monkeypatch):
    waits, clock = [], [1000.0]
    monkeypatch.setattr(llm_gateway, "_monotonic", lambda: clock[0])     # the breaker and the deadline share it

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", http_error(openai.RateLimitError, 429, {"code": "rate_limit_exceeded"}, {"retry-after": "3"}),
                  reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 2 and 3.0 <= waits[0] <= 3.6, "never sooner than Retry-After"
    assert "nvidia" not in hybrid.fakes or not hybrid.fakes["nvidia"].calls


def test_an_exhausted_quota_is_not_retried(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", http_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}))
    hybrid.script("nvidia", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 1
    assert llm_gateway.status()["providers"]["groq"]["circuit"] == "open"


def test_5xx_gets_bounded_retries_then_the_next_provider(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", *[http_error(openai.InternalServerError, 503)] * 2)
    hybrid.script("nvidia", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 2, "1 + LLM_PROVIDER_TRANSIENT_RETRIES"


def test_a_timeout_moves_to_the_next_provider_without_a_retry(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", openai.APITimeoutError(request=httpx.Request("POST", "https://p.invalid")))
    hybrid.script("nvidia", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert len(hybrid.fakes["groq"].calls) == 1


def test_a_refused_forced_tool_choice_is_asked_once_more_with_required(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq"))
    hybrid.script("groq", http_error(openai.BadRequestError, 400, {"code": "tool_use_failed"}), reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    first, second = hybrid.fakes["groq"].calls
    assert first["tool_choice"]["function"]["name"] == "submit_plan" and second["tool_choice"] == "required"
    assert second["max_tokens"] > first["max_tokens"]


def test_cancellation_propagates_and_is_not_a_provider_failure(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", reply())
    hybrid.script("nvidia", reply())
    ctx = runtime.RunContext(question_id=0, record=False, should_cancel=lambda: True)
    with runtime.run_context(ctx), pytest.raises(runtime.RunCancelled):
        llm_client.call_tool(**kwargs())
    assert hybrid.fakes["groq"].calls == [] and hybrid.fakes["nvidia"].calls == []
    assert llm_gateway.status()["providers"]["groq"]["failed"] == 0


def test_no_provider_is_tried_past_the_steps_deadline(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    monkeypatch.setattr(config, "LLM_REQUEST_DEADLINE_SECONDS", 1.0)    # less than LLM_MIN_ATTEMPT_SECONDS
    hybrid.script("groq", reply())
    with pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(**kwargs())
    assert {f["kind"] for f in e.value.failures} == {"no_time"} and hybrid.fakes["groq"].calls == []


# ------------------------------------------------------------------------------------ structured output checks --

@pytest.mark.parametrize("bad", [
    reply(tool="another_tool"),                                         # wrong tool
    reply(tool=None, content="Sure! The verdict is verified."),          # prose, no tool call
    reply(args="{not json"),                                            # malformed JSON
    reply(args="[1, 2]"),                                               # not an object
    reply(args={"verdict": "verified"}),                                # a required field missing
    reply(args={"verdict": "", "confidence": 0.5}),                     # a required field empty
    reply(args={"verdict": "verified", "confidence": "high"}),          # wrong type
    reply(args={"verdict": "maybe", "confidence": 0.5}),                # not an allowed value
    types.SimpleNamespace(choices=[], usage=None, model=None),          # empty response
])
def test_an_unusable_tool_reply_is_never_accepted_and_the_next_provider_takes_the_step(hybrid, monkeypatch, bad):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", bad)
    hybrid.script("nvidia", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert llm_gateway.status()["providers"]["groq"]["last_failure"] == "INVALID_OUTPUT"


def test_validation_accepts_a_reply_that_fills_every_required_field_correctly():
    assert providers.validate_arguments(json.dumps(GOOD), SCHEMA) == (GOOD, None)
    assert providers.validate_arguments({"verdict": "rejected", "confidence": 1}, SCHEMA)[1] is None


# --------------------------------------------------------------------------------------------- text generation --

def test_text_goes_through_the_plain_chat_call_and_fails_over(hybrid, monkeypatch):
    monkeypatch.setattr(llm_client, "_gemini_text", lambda *a: (_ for _ in ()).throw(gemini_down()))
    hybrid.script("groq", reply(tool=None, content="  A short summary.  "))
    assert llm_client.call_text("sys", "user", 300) == "A short summary."
    sent = hybrid.fakes["groq"].calls[0]
    assert "tools" not in sent and "tool_choice" not in sent and sent["max_tokens"] == 300


def test_an_empty_text_reply_is_not_a_success(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_NARRATOR_PROVIDERS", only("groq", "nvidia"))
    hybrid.script("groq", reply(tool=None, content=""))
    hybrid.script("nvidia", reply(tool=None, content="ok"))
    assert llm_client.call_text("s", "u") == "ok"


# ----------------------------------------------------------------------------------- provenance and availability --

def test_nvidia_records_the_requested_and_the_served_model_separately(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("nvidia"))
    hybrid.script("nvidia", reply(model="private/nvidia/nemotron-3.5-super-120b-a12b"))
    events = []
    monkeypatch.setattr(runtime, "record_event", lambda kind, name="", detail=None, *a, **k: events.append((kind, detail)))
    assert llm_client.call_tool(**kwargs()) == GOOD
    call = next(d for k, d in events if k == "llm_call")
    assert call["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert call["served_model"] == "private/nvidia/nemotron-3.5-super-120b-a12b"
    assert routing.served_for("planner")["family"] == "nvidia-nemotron"


def test_provider_availability_follows_the_agents_route(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", only("groq"))
    assert llm_client.provider_available("submit_plan") is True
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert llm_client.provider_available("submit_plan") is False, "a provider without a key is not available"


def test_health_never_shows_a_key_and_never_calls_a_provider_healthy_for_having_one(hybrid):
    status = llm_gateway.status()
    dumped = json.dumps(status)
    for secret in ("g-key", "groq-key", "nv-key", "or-key", "m-key"):
        assert secret not in dumped
    states = {p: v["state"] for p, v in status["hybrid"]["providers"].items()}
    assert states["groq"] == "configured" and "healthy" not in dumped
    assert status["hybrid"]["routes"]["critic"] == ["groq", "nvidia", "openrouter"]
    for field in ("gemini_key_configured", "llama_key_configured"):     # the existing health fields are untouched
        assert field not in status


# --------------------------------------------------------------------------------------------- model families --

@pytest.mark.parametrize("provider,model,served,family", [
    ("gemini", "gemini-3.6-flash", None, "google-gemini"),
    ("groq", "openai/gpt-oss-120b", None, "openai-gpt-oss"),
    ("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b", "private/nvidia/nemotron-3.5-super-120b-a12b", "nvidia-nemotron"),
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free", None, "nvidia-nemotron"),
    ("mistral", "mistral-medium-2604", None, "mistral"),
    ("llama", "Llama-4-Maverick-17B-128E-Instruct-FP8", None, "meta-llama"),
    ("nvidia", "nvidia/llama-3.1-nemotron-70b-instruct", None, "meta-llama"),   # a Llama derivative
    ("openrouter", "google/gemma-4-31b-it:free", None, "google-gemini"),
    ("openrouter", "openrouter/free", None, families.UNKNOWN),                    # a router: model unknown
    ("openrouter", "openrouter/free", "qwen/qwen3.8-27b", "qwen"),                # ...unless it reports what ran
    ("nvidia", "poolside/laguna-xs-2.1", None, families.UNKNOWN),
])
def test_model_families_come_from_the_model_never_the_provider(provider, model, served, family):
    assert families.family_of(provider, model, served) == family


def test_unknown_is_never_independent_and_nemotron_is_one_family_whoever_serves_it():
    assert not families.independent(families.UNKNOWN, "google-gemini")
    assert not families.independent("google-gemini", families.UNKNOWN)
    assert not families.independent(families.family_of("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
                                    families.family_of("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b"))
    assert families.independent("openai-gpt-oss", "google-gemini")


def test_a_pinned_family_overrides_the_rules(monkeypatch):
    monkeypatch.setenv("NVIDIA_MODEL_FAMILY", "Meta-Llama")
    assert families.family_of("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b") == "meta-llama"


# ---------------------------------------------------------------------------------------- Critic independence --

def run_executor(hybrid, monkeypatch, provider_reply):
    """Serve one Executor step, so the Critic can see which family produced the work."""
    provider, item = provider_reply
    if provider == "gemini":
        hybrid.gemini(item)
    else:
        monkeypatch.setenv("LLM_EXECUTOR_PROVIDERS", f"gemini,{provider}")
        hybrid.gemini(gemini_down())
        hybrid.script(provider, item)
    llm_client.call_tool(**kwargs("submit_code"))


def test_the_critic_prefers_groq_when_the_executor_ran_on_gemini(hybrid, monkeypatch):
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("gemini", {"code": "x"}))
        hybrid.script("groq", reply(tool="submit_review", model="openai/gpt-oss-120b"))
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert out == GOOD and label == "groq:openai/gpt-oss-120b", "the existing '<provider>:<model>' label"
    assert info["independent"] is True and info["verifier_family"] == "openai-gpt-oss"
    assert info["checked_families"] == ["google-gemini"] and info["executor_family_source"] == "served:run"


def test_the_critic_skips_the_executors_actual_family(hybrid, monkeypatch):
    """The Executor fell back to Groq (gpt-oss), so the Critic must not verify on gpt-oss: NVIDIA takes it."""
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("groq", reply(tool="submit_code", model="openai/gpt-oss-120b")))
        hybrid.script("groq")                                           # must not be called
        hybrid.script("nvidia", reply(tool="submit_review", model="private/nvidia/nemotron-3.5-super-120b-a12b"))
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert label == "nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b" and info["independent"] is True
    assert info["served_model"] == "private/nvidia/nemotron-3.5-super-120b-a12b"
    assert hybrid.fakes["groq"].calls == []


def test_openrouter_nemotron_never_counts_as_independent_of_nvidia_nemotron(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_CRITIC_PROVIDERS", only("openrouter"))
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("nvidia", reply(tool="submit_code")))
        hybrid.script("openrouter", reply(tool="submit_review"))
        hybrid.gemini(GOOD)                                             # the existing same-family last resort...
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    # ...is not used, since OpenRouter can serve: but it serves as a same-family, NOT independent, verifier
    assert label == "openrouter-fallback:nvidia/nemotron-3.5-super-vl:free" and info["independent"] is False
    assert info["verifier_family"] == info["executor_family"] == "nvidia-nemotron"


def test_with_no_independent_provider_the_critic_falls_back_and_says_so(hybrid, monkeypatch):
    """Groq, NVIDIA and OpenRouter all down: the existing behaviour -- Gemini, labelled gemini-fallback."""
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("gemini", {"code": "x"}))
        for p in ("groq", "nvidia", "openrouter"):
            hybrid.script(p, http_error(openai.AuthenticationError, 401))
        hybrid.gemini(GOOD)
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert label == f"gemini-fallback:{config.GEMINI_MODEL}" and info["independent"] is False


def test_requiring_independence_fails_the_critic_instead_of_verifying_on_the_same_family(hybrid, monkeypatch):
    monkeypatch.setattr(config, "LLM_CRITIC_REQUIRE_INDEPENDENT", True)
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("gemini", {"code": "x"}))
        for p in ("groq", "nvidia", "openrouter"):
            hybrid.script(p, http_error(openai.InternalServerError, 503), http_error(openai.InternalServerError, 503))
        with pytest.raises(runtime.ProviderUnavailable):
            critic._critic_call_tool(**kwargs("submit_review"))
    assert hybrid.gemini_calls[-1]["tool_name"] == "submit_code", "Gemini was not used to verify its own work"


def test_an_unknown_family_critic_model_is_never_called_independent(hybrid, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "openrouter/free")
    monkeypatch.setenv("LLM_CRITIC_PROVIDERS", only("openrouter"))
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("gemini", {"code": "x"}))
        hybrid.script("openrouter", reply(tool="submit_review", model="openrouter/free"))
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert info["independent"] is False and label.startswith("openrouter-fallback:")


def test_the_critic_judges_the_family_that_actually_served_not_the_one_requested(hybrid, monkeypatch):
    """A router asked for an unknown-family model but reports it ran Gemma (Google): not independent of Gemini."""
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "some-vendor/new-model")
    monkeypatch.setenv("LLM_CRITIC_PROVIDERS", only("openrouter"))
    monkeypatch.setenv("OPENROUTER_MODEL_FAMILY", "")
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("gemini", {"code": "x"}))
        hybrid.script("openrouter", reply(tool="submit_review", model="google/gemma-4-31b-it:free"))
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert info["verifier_family"] == "google-gemini" and info["independent"] is False


def test_the_legacy_critic_path_and_its_labels_are_unchanged_when_hybrid_is_off(monkeypatch):
    monkeypatch.setattr(config, "LLM_HYBRID_ROUTING_ENABLED", False)
    monkeypatch.setattr(config, "LLAMA_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "g")
    monkeypatch.setattr(llm_client, "call_tool_gemini", lambda **kw: GOOD)
    critic._last_verifier.set(None)
    assert critic._critic_call_tool(**kwargs("submit_review")) == (GOOD, f"gemini-fallback:{config.GEMINI_MODEL}")
    assert critic._last_verifier.get() is None


# ------------------------------------------------------- a failed attempt is never a success; provenance end to end --

def test_a_failed_gemini_attempt_is_never_recorded_as_a_served_response(hybrid, monkeypatch):
    """Gemini's own adapter signals "not served" with its own marker; the router must treat it as a failure, not as
    an answer -- with Gemini alone in the route the step fails, and nothing is recorded as having served it."""
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", "gemini")
    hybrid.gemini(gemini_down())
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx), pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(**kwargs())
    assert e.value.failures[0]["provider"] == "gemini" and e.value.failures[0]["kind"] == "overloaded"
    assert llm_client.LAST_PROVIDER == "none" and ctx.served == {} and routing.served_for("planner") is None
    g = llm_gateway.status()["providers"]["gemini"]
    assert g["succeeded"] == 0 and g["failed"] == 1 and llm_gateway.status()["requests"]["succeeded"] == 0


def test_a_failed_legacy_llama_attempt_is_never_recorded_as_a_served_response(hybrid, monkeypatch):
    monkeypatch.setattr(config, "LLAMA_API_KEY", "legacy-key")
    monkeypatch.setenv("LLM_PLANNER_PROVIDERS", "llama,groq")
    monkeypatch.setattr(llm_client_llama, "call_tool", lambda **kw: (_ for _ in ()).throw(
        http_error(openai.AuthenticationError, 401)))
    hybrid.script("groq", reply())
    assert llm_client.call_tool(**kwargs()) == GOOD
    assert routing.served_for("planner")["provider"] == "groq"


def test_the_critic_follows_the_executors_family_as_it_changes_within_a_run(hybrid, monkeypatch):
    """Same run, two Executor steps on different families: after the second, the Critic must avoid BOTH."""
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        run_executor(hybrid, monkeypatch, ("gemini", {"code": "x"}))
        hybrid.script("groq", reply(tool="submit_review"))
        assert critic._critic_call_tool(**kwargs("submit_review"))[1].startswith("groq:")
        run_executor(hybrid, monkeypatch, ("groq", reply(tool="submit_code")))
        hybrid.script("nvidia", reply(tool="submit_review"))
        assert critic._critic_call_tool(**kwargs("submit_review"))[1].startswith("nvidia:")
        assert critic._last_verifier.get()["checked_families"] == ["google-gemini", "openai-gpt-oss"]


def test_the_run_manifest_keeps_the_requested_and_the_served_model(client):
    """Provenance end to end: llm_call events carrying a served model different from the requested one show up in
    the manifest under the requested model, with the served model alongside; existing fields are unchanged."""
    from platform_helpers import ask_and_run, workspace

    from app import provenance
    from app.database import SessionLocal
    from app.models import Question, RunEvent

    ws = workspace(client)
    qid = ask_and_run(client, ws, "Total revenue by region")["question_id"]
    db = SessionLocal()
    try:
        q = db.get(Question, qid)
        for served in ("private/nvidia/nemotron-3.5-super-120b-a12b", None):
            db.add(RunEvent(team_id=q.team_id, question_id=qid, kind="llm_call", name="submit_plan", duration_ms=5,
                            detail_json={"provider": "nvidia", "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                                         "tokens_in": 3, "tokens_out": 2, "fallback": True, "ok": True,
                                         **({"served_model": served} if served else {})}))
        db.commit()
        m = provenance.build_manifest(db, q)
    finally:
        db.close()
    entry = m["models"]["nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b"]
    assert entry["calls"] == 2 and entry["fallback_calls"] == 2 and entry["tokens_in"] == 6
    assert entry["served_models"] == ["private/nvidia/nemotron-3.5-super-120b-a12b"]
    others = {k: v for k, v in m["models"].items() if k != "nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b"}
    assert all("served_models" not in v for v in others.values()), "unchanged where nothing differs"
    assert m["config"]["hybrid_routing_enabled"] is False and "failover_enabled" in m["config"]


# ---------------------------------------------- P0: run-scoped history, and every family that shaped the work --

def serve(hybrid, monkeypatch, role, provider, model=None):
    """Serve one Planner ("planner") or Executor ("executor") call on ``provider`` inside the current run."""
    tool = {"planner": "submit_plan", "executor": "submit_code"}[role]
    monkeypatch.setenv(f"LLM_{role.upper()}_PROVIDERS", provider)
    if provider == "gemini":
        hybrid.gemini(GOOD)
    else:
        hybrid.script(provider, reply(tool=tool, model=model))
    assert llm_client.call_tool(**kwargs(tool)) == GOOD


def critic_on(hybrid, expected):
    """Run one Critic call where only ``expected`` may serve (any other provider fails the test if called)."""
    for p in ("groq", "nvidia", "openrouter"):
        hybrid.script(p, *([reply(tool="submit_review")] if p == expected else []))
    hybrid.gemini()
    out, label = critic._critic_call_tool(**kwargs("submit_review"))
    info = critic._last_verifier.get()
    for p in ("groq", "nvidia", "openrouter"):
        if p != expected:
            assert hybrid.fakes[p].calls == [], f"{p} must not be asked to verify"
    return label, info


@pytest.mark.parametrize("planner,executor,critic_provider,checked", [
    ("gemini", "gemini", "groq", ["google-gemini"]),                                  # 1
    ("gemini", "groq", "nvidia", ["google-gemini", "openai-gpt-oss"]),                # 2
    ("groq", "gemini", "nvidia", ["google-gemini", "openai-gpt-oss"]),                # 3: the Planner counts too
    (None, "nvidia", "groq", ["nvidia-nemotron"]),                                    # NVIDIA Executor -> Groq Critic
])
def test_the_critic_avoids_every_family_that_planned_or_executed(hybrid, monkeypatch, planner, executor,
                                                                  critic_provider, checked):
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        if planner:
            serve(hybrid, monkeypatch, "planner", planner)
        serve(hybrid, monkeypatch, "executor", executor)
        label, info = critic_on(hybrid, critic_provider)
    assert label.startswith(f"{critic_provider}:") and info["independent"] is True
    assert info["checked_families"] == checked and info["verifier_family"] not in checked


def test_every_executor_step_counts_not_only_the_latest(hybrid, monkeypatch):
    """Executor steps on Groq, then Gemini (it recovered): Groq wrote part of the analysis, so Groq may not verify
    it, even though the latest step was Gemini's."""
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        serve(hybrid, monkeypatch, "executor", "groq")
        serve(hybrid, monkeypatch, "executor", "gemini")
        label, info = critic_on(hybrid, "nvidia")
    assert label.startswith("nvidia:") and info["independent"] is True
    assert info["checked_families"] == ["google-gemini", "openai-gpt-oss"]
    assert [e["family"] for e in ctx.served_history if e["agent"] == "executor"] == ["openai-gpt-oss", "google-gemini"]


def test_an_unknown_family_in_the_work_makes_no_verifier_independent(hybrid, monkeypatch):
    """The Executor ran on a router alias whose model is unknown: no Critic can be shown to differ from it."""
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "openrouter/free")
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        serve(hybrid, monkeypatch, "executor", "openrouter", model="openrouter/free")
        for p in ("groq", "nvidia"):
            hybrid.script(p, reply(tool="submit_review"))
        hybrid.script("openrouter")
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert info["checked_families"] == ["unknown"] and info["independent"] is False
    assert "-fallback:" in label, "served, but recorded as not independent"


def test_a_previous_run_cannot_lend_its_history_to_the_next(hybrid, monkeypatch):
    """Two jobs on the same worker thread: the first ran its Executor on Groq. The second records no Planner or
    Executor call before its Critic runs -- it must see "unknown", not the first run's Groq family."""
    first = runtime.RunContext(question_id=1, record=False)
    with runtime.run_context(first):
        serve(hybrid, monkeypatch, "executor", "groq")
    second = runtime.RunContext(question_id=2, record=False)
    with runtime.run_context(second):
        assert routing.served_for("executor") is None, "no fallback to another run's record"
        assert critic.checked_families() == (frozenset({"unknown"}), "unknown:no_run_history")
        hybrid.script("nvidia", reply(tool="submit_review"))
        hybrid.script("groq", reply(tool="submit_review"))
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert info["independent"] is False and "-fallback:" in label


def test_a_previous_runs_family_does_not_block_the_next_runs_critic(hybrid, monkeypatch):
    """Run 1: Executor on Groq. Run 2: Executor on Gemini. Run 2's Critic may use Groq: run 1 is not its work."""
    with runtime.run_context(runtime.RunContext(question_id=1, record=False)):
        serve(hybrid, monkeypatch, "executor", "groq")
    with runtime.run_context(runtime.RunContext(question_id=2, record=False)):
        serve(hybrid, monkeypatch, "executor", "gemini")
        label, info = critic_on(hybrid, "groq")
    assert label.startswith("groq:") and info["independent"] is True and info["checked_families"] == ["google-gemini"]


def test_history_recorded_outside_a_run_is_never_read_inside_one(hybrid, monkeypatch):
    """A chat answer served on this thread outside any run leaves a context-wide record; a run must ignore it."""
    monkeypatch.setenv("LLM_CHAT_PROVIDERS", "nvidia")
    hybrid.script("nvidia", reply(tool="submit_answer"))
    llm_client.call_tool(**kwargs("submit_answer"))
    assert routing.served_for("chat")["family"] == "nvidia-nemotron"
    with runtime.run_context(runtime.RunContext(question_id=3, record=False)):
        assert routing.served_for("chat") is None and routing.families_served(("chat", "executor")) == set()


def test_outside_a_run_the_chat_check_is_judged_against_the_answers_family(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_CHAT_PROVIDERS", "groq")
    hybrid.script("groq", reply(tool="submit_answer"))
    llm_client.call_tool(**kwargs("submit_answer"))
    assert critic.checked_families() == (frozenset({"openai-gpt-oss"}), "served:chat")
    label, info = critic_on(hybrid, "nvidia")
    assert label.startswith("nvidia:") and info["independent"] is True


def test_same_family_nemotron_through_openrouter_and_nvidia_is_never_independent(hybrid, monkeypatch):
    monkeypatch.setenv("LLM_CRITIC_PROVIDERS", "openrouter")
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        serve(hybrid, monkeypatch, "planner", "gemini")
        serve(hybrid, monkeypatch, "executor", "nvidia", model="private/nvidia/nemotron-3.5-super-120b-a12b")
        hybrid.script("openrouter", reply(tool="submit_review", model="nvidia/nemotron-3.5-super-vl:free"))
        hybrid.gemini()
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    assert info["independent"] is False and label == "openrouter-fallback:nvidia/nemotron-3.5-super-vl:free"
    assert info["checked_families"] == ["google-gemini", "nvidia-nemotron"]


def test_with_no_independent_provider_the_fallback_is_marked_not_independent(hybrid, monkeypatch):
    """Planner on Gemini, Executor on Groq and NVIDIA: only OpenRouter (Nemotron, same as NVIDIA) and Gemini are
    left, both families in the work. The existing fallback still verifies, labelled and recorded as NOT independent."""
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        serve(hybrid, monkeypatch, "planner", "gemini")
        serve(hybrid, monkeypatch, "executor", "groq")
        serve(hybrid, monkeypatch, "executor", "nvidia")
        for p in ("groq", "nvidia"):
            hybrid.script(p, http_error(openai.InternalServerError, 503), http_error(openai.InternalServerError, 503))
        hybrid.script("openrouter", reply(tool="submit_review"))
        hybrid.gemini()
        out, label = critic._critic_call_tool(**kwargs("submit_review"))
        info = critic._last_verifier.get()
    # every Critic provider is in a family that shaped the work, so none goes first; Groq and NVIDIA are down,
    # OpenRouter serves as the same-family fallback
    assert label == "openrouter-fallback:nvidia/nemotron-3.5-super-vl:free" and info["independent"] is False
    assert len(hybrid.fakes["groq"].calls) == 2 and len(hybrid.fakes["nvidia"].calls) == 2
    assert info["checked_families"] == ["google-gemini", "nvidia-nemotron", "openai-gpt-oss"]


# ------------------------------------------------- critic_node under hybrid routing: the persisted audit record --

VERIFY_ARGS = {"what_it_checks": "total revenue per region", "code": "print({'North': 2700, 'South': 1600})"}
REVIEW_ARGS = {"verdict": "verified", "confidence": 0.9, "issues": [], "summary": "The totals check out."}


class FakeRunner:
    """The sandbox stand-in: the Critic's verification script 'runs' and recomputes the claimed totals."""

    def seed_input_data(self, question_id, csv_path):
        return None

    def run(self, question_id, code, step_index, attempt):
        return types.SimpleNamespace(success=True, stdout="{'North': 2700, 'South': 1600}", stderr="",
                                     result={"North": 2700, "South": 1600})


def critic_state(question_id):
    return {"question_id": question_id, "question_text": "Total revenue by region", "csv_path": "unused.csv",
            "profile": {"row_count": 4, "col_count": 3, "columns": [{"name": "region", "dtype": "object"}]},
            "step_results": [{"step_index": 0, "description": "Total revenue per region", "code": "df.groupby(...)",
                              "result": {"North": 2700, "South": 1600}, "chart_paths": [], "success": True}],
            "revision_count": 0, "failed": False}


def persisted_review(review_id):
    from app.database import SessionLocal
    from app.models import CriticReview
    db = SessionLocal()
    try:
        row = db.get(CriticReview, review_id)
        return row.checks_json[0], row.summary, row.verdict
    finally:
        db.close()


@pytest.fixture
def question_id(client):
    from platform_helpers import ask, workspace
    return ask(client, workspace(client), "Total revenue by region")["question_id"]


def test_critic_node_persists_an_independent_verification(hybrid, monkeypatch, question_id):
    monkeypatch.setattr(critic, "get_runner", FakeRunner)
    with runtime.run_context(runtime.RunContext(question_id=question_id, record=False)):
        serve(hybrid, monkeypatch, "planner", "gemini")
        serve(hybrid, monkeypatch, "executor", "gemini")
        hybrid.script("groq", reply(VERIFY_ARGS, tool="submit_verification_code", model="openai/gpt-oss-120b"),
                      reply(REVIEW_ARGS, tool="submit_review", model="openai/gpt-oss-120b"))
        out = critic.critic_node(critic_state(question_id))
    assert out["critic_verdict"] == "verified" and out["critic_review_id"]
    check, summary, verdict = persisted_review(out["critic_review_id"])
    assert check["verifier_model"] == "groq:openai/gpt-oss-120b"
    assert check["verifier_independent"] is True
    assert check["checked_families"] == ["google-gemini"]
    assert check["verifier_family"] == "openai-gpt-oss" and check["review_verifier_family"] == "openai-gpt-oss"
    assert check["verifier_requested_model"] == "openai/gpt-oss-120b"
    assert check["verifier_served_model"] == "openai/gpt-oss-120b"
    assert check["review_verifier_model"] == "groq:openai/gpt-oss-120b"
    assert summary == "The totals check out.", "no 'not cross-model' note on an independent check"
    assert verdict == "verified"


def test_critic_node_persists_a_non_independent_fallback_with_its_note(hybrid, monkeypatch, question_id):
    """Gemini planned, Groq and NVIDIA executed: no Critic provider is outside the work's families. Groq and NVIDIA
    are down, OpenRouter (Nemotron, same as NVIDIA) verifies -- recorded, labelled and summarised as NOT independent."""
    monkeypatch.setattr(critic, "get_runner", FakeRunner)
    with runtime.run_context(runtime.RunContext(question_id=question_id, record=False)):
        serve(hybrid, monkeypatch, "planner", "gemini")
        serve(hybrid, monkeypatch, "executor", "groq")
        serve(hybrid, monkeypatch, "executor", "nvidia")
        down = [http_error(openai.InternalServerError, 503)] * 4
        hybrid.script("groq", *down)
        hybrid.script("nvidia", *down)
        hybrid.script("openrouter", reply(VERIFY_ARGS, tool="submit_verification_code"),
                      reply(REVIEW_ARGS, tool="submit_review"))
        hybrid.gemini()
        out = critic.critic_node(critic_state(question_id))
    check, summary, verdict = persisted_review(out["critic_review_id"])
    assert check["verifier_model"] == "openrouter-fallback:nvidia/nemotron-3.5-super-vl:free"
    assert check["review_verifier_model"] == "openrouter-fallback:nvidia/nemotron-3.5-super-vl:free"
    assert check["verifier_independent"] is False
    assert check["checked_families"] == ["google-gemini", "nvidia-nemotron", "openai-gpt-oss"]
    assert check["verifier_family"] == "nvidia-nemotron" and check["review_verifier_family"] == "nvidia-nemotron"
    assert check["verifier_requested_model"] == "nvidia/nemotron-3.5-super-vl:free"
    assert summary.startswith("The totals check out.")
    assert "not cross-model" in summary and "verifier: nvidia-nemotron" in summary
    assert "google-gemini, nvidia-nemotron, openai-gpt-oss" in summary
    assert verdict == "verified", "the verdict itself is the Critic's; only its independence is qualified"


def test_critic_node_needs_both_calls_independent(hybrid, monkeypatch, question_id):
    """The verification script comes from Groq (independent of NVIDIA's work), but the review falls to OpenRouter's
    Nemotron -- the Executor's own family. One non-independent call makes the whole check non-independent."""
    monkeypatch.setattr(critic, "get_runner", FakeRunner)
    monkeypatch.setenv("LLM_CRITIC_PROVIDERS", "groq,openrouter")
    with runtime.run_context(runtime.RunContext(question_id=question_id, record=False)):
        serve(hybrid, monkeypatch, "executor", "nvidia")
        hybrid.script("groq", reply(VERIFY_ARGS, tool="submit_verification_code"),
                      http_error(openai.AuthenticationError, 401))
        hybrid.script("openrouter", reply(REVIEW_ARGS, tool="submit_review"))
        out = critic.critic_node(critic_state(question_id))
    check, summary, _ = persisted_review(out["critic_review_id"])
    assert check["verifier_model"] == "groq:openai/gpt-oss-120b", "the verify call alone was independent"
    assert check["review_verifier_model"] == "openrouter-fallback:nvidia/nemotron-3.5-super-vl:free"
    assert check["verifier_family"] == "openai-gpt-oss" and check["review_verifier_family"] == "nvidia-nemotron"
    assert check["verifier_independent"] is False and "not cross-model" in summary


def test_critic_node_writes_no_hybrid_fields_when_hybrid_is_off(monkeypatch, question_id):
    """Existing audit records are unchanged with hybrid routing off: the legacy label, no new fields."""
    monkeypatch.setattr(config, "LLM_HYBRID_ROUTING_ENABLED", False)
    monkeypatch.setattr(config, "LLAMA_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "g")
    monkeypatch.setattr(critic, "get_runner", FakeRunner)
    answers = iter([VERIFY_ARGS, REVIEW_ARGS])
    monkeypatch.setattr(llm_client, "call_tool_gemini", lambda **kw: next(answers))
    out = critic.critic_node(critic_state(question_id))
    check, summary, _ = persisted_review(out["critic_review_id"])
    assert check["verifier_model"] == f"gemini-fallback:{config.GEMINI_MODEL}"
    assert not {"verifier_independent", "checked_families", "review_verifier_family"} & set(check)
    assert "(Llama unavailable)" in summary, "the existing note, unchanged"
