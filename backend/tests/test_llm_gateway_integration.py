"""Phase 3B: offline integration tests of the multi-provider gateway through Silt's real workflows.

What is real: the API (TestClient), the job runner, the LangGraph pipeline, every agent, the router, the hybrid
routing engine, app/llm_gateway.py (breakers, deadlines, classification), the provider adapters, the subprocess
sandbox, the test database, and the trace / manifest / critic-review endpoints.

What is fake, and only this: the provider SDK clients at their construction boundary --
  * Gemini:  llm_client._client (the genai.Client that get_client() hands out) -> FakeGemini;
  * Groq, NVIDIA, OpenRouter, Mistral: app.llm.providers.OpenAI (the class _client() constructs) -> FakeOpenAI;
  * legacy Llama: app.agents.llm_client_llama.OpenAI -> FakeOpenAI.
They answer from per-provider scripts keyed by the requested tool, record every request, and raise real SDK
exception types for injected failures.

Never used here: runtime.provider_override, or any mock of call_tool / call_text / provider_available.

Safety, all local to this module: every provider key is a fake value set on app.config for the test only; outbound
sockets are blocked for the whole module (any connect raises), so no live request can be made even by mistake; the
local analyst model URL is blanked so no local model is consulted.
"""
from __future__ import annotations

import copy
import json
import logging
import socket
import sys
import time
import types

import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from platform_helpers import ask, workspace

from app import config, jobs, llm_gateway, runtime
from app.agents import llm_client, llm_client_llama
from app.llm import providers, routing

SUITE = json.loads((config.BACKEND_DIR / "evals" / "cases" / "suite.json").read_text(encoding="utf-8"))
PIPELINE = next(c for c in SUITE["cases"] if c["id"] == "pipeline-correct")
QUESTION = PIPELINE["question"]
SCRIPT = PIPELINE["script"]               # one reply per tool call of the standard pipeline (code runs in the sandbox)
CRITIC_TOOLS = ("submit_verification_code", "submit_review")
PIPELINE_TOOLS = ("submit_triage", "submit_plan", "submit_code", "compile_dashboard")

MODELS = {"gemini": "gemini-3.6-flash", "groq": "openai/gpt-oss-120b",
          "nvidia": "nvidia/nemotron-3.5-super-vl-120b-a12b", "openrouter": "nvidia/nemotron-3.5-super-vl:free",
          "mistral": "mistral-medium-2604"}
SERVED = {"nvidia": "private/nvidia/nemotron-3.5-super-120b-a12b"}   # what NVIDIA reports it ran (seen live)
FAKE_KEYS = {"GEMINI_API_KEY": "fake-gemini-key", "GROQ_API_KEY": "fake-groq-key", "NVIDIA_API_KEY": "fake-nvidia-key",
             "OPENROUTER_API_KEY": "fake-openrouter-key", "MISTRAL_API_KEY": "fake-mistral-key"}
HOSTS = {"api.groq.com": "groq", "integrate.api.nvidia.com": "nvidia", "openrouter.ai": "openrouter",
         "api.mistral.ai": "mistral", "llama.invalid": "llama"}


# ------------------------------------------------------------------------------------------- the fake SDKs --

class Unscripted(Exception):
    """A provider was asked for a tool its script did not cover (recorded; the test fails at teardown)."""


class World:
    """Per-provider scripts and the log of every request any fake SDK received."""

    def __init__(self):
        self.scripts: dict[str, dict[str, list]] = {}
        self.requests: list[dict] = []
        self.unscripted: list[tuple[str, str]] = []
        self.constructed: list[dict] = []
        self.on_request = None

    def serve(self, provider: str, tool: str, *replies):
        self.scripts.setdefault(provider, {}).setdefault(tool, []).extend(
            r if isinstance(r, BaseException) else copy.deepcopy(r) for r in replies)

    def serve_pipeline(self, provider: str, tools=PIPELINE_TOOLS + CRITIC_TOOLS):
        for tool in tools:
            self.serve(provider, tool, *SCRIPT[tool])

    def next_reply(self, provider: str, tool: str):
        queue = self.scripts.get(provider, {}).get(tool)
        if not queue:
            self.unscripted.append((provider, tool))
            raise Unscripted(f"{provider} has no scripted reply for {tool}")
        return queue.pop(0)

    def calls(self, provider=None, tool=None) -> list[dict]:
        return [r for r in self.requests if (provider is None or r["provider"] == provider)
                and (tool is None or r["tool"] == tool)]

    def tools_by_provider(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.requests:
            out.setdefault(r["provider"], []).append(r["tool"])
        return out


def _raise_or(item):
    if isinstance(item, BaseException):
        raise item
    return item


class FakeGemini:
    """Stands in for google.genai.Client: models.generate_content answers with a function call."""

    def __init__(self, world: World):
        self.world = world
        self.models = self

    def generate_content(self, *, model, contents, config):
        decls = [d for t in (config.tools or []) for d in (t.function_declarations or [])]
        tool = decls[0].name if decls else "text"
        timeout = getattr(getattr(config, "http_options", None), "timeout", None)
        self.world.requests.append({"provider": "gemini", "tool": tool, "model": model, "timeout_ms": timeout,
                                    "forced": bool(config.tool_config)})
        if self.world.on_request:
            self.world.on_request("gemini", tool)
        args = _raise_or(self.world.next_reply("gemini", tool))
        self.world.requests[-1]["answered"] = True
        part = types.SimpleNamespace(function_call=types.SimpleNamespace(name=tool, args=args), text=None)
        cand = types.SimpleNamespace(content=types.SimpleNamespace(parts=[part]),
                                     finish_reason=types.SimpleNamespace(name="STOP"))
        return types.SimpleNamespace(candidates=[cand], prompt_feedback=None, text=None, model_version=model,
                                     usage_metadata=types.SimpleNamespace(prompt_token_count=100,
                                                                          candidates_token_count=40,
                                                                          thoughts_token_count=5))


def make_fake_openai(world: World):
    class FakeOpenAI:
        """Stands in for openai.OpenAI. The provider is told apart by base URL, as the real endpoint would be."""

        def __init__(self, *, api_key, base_url, timeout=None, max_retries=None, default_headers=None, **_):
            self.provider = next((p for h, p in HOSTS.items() if h in str(base_url)), "unknown")
            self._timeout = timeout
            world.constructed.append({"provider": self.provider, "base_url": str(base_url), "api_key": api_key,
                                      "max_retries": max_retries, "headers": dict(default_headers or {})})
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def with_options(self, *, timeout=None, **_):
            clone = object.__new__(FakeOpenAI)
            clone.provider, clone._timeout, clone.chat = self.provider, timeout, types.SimpleNamespace(
                completions=types.SimpleNamespace(create=clone._create))
            return clone

        def _create(self, *, model, messages, max_tokens, tools=None, tool_choice=None, **_):
            tool = tools[0]["function"]["name"] if tools else "text"
            world.requests.append({"provider": self.provider, "tool": tool, "model": model, "timeout": self._timeout,
                                   "tool_choice": tool_choice, "max_tokens": max_tokens})
            if world.on_request:
                world.on_request(self.provider, tool)
            item = _raise_or(world.next_reply(self.provider, tool))
            world.requests[-1]["answered"] = True
            served = SERVED.get(self.provider, model)
            usage = types.SimpleNamespace(prompt_tokens=120, completion_tokens=60)
            if tool == "text":
                msg = types.SimpleNamespace(tool_calls=None, content=item)
            else:
                # A str is a raw (possibly malformed) arguments payload and is sent as-is; a dict may name another tool.
                name = item.pop("_tool", tool) if isinstance(item, dict) else tool
                msg = types.SimpleNamespace(content=None, tool_calls=[types.SimpleNamespace(
                    function=types.SimpleNamespace(name=name,
                                                   arguments=item if isinstance(item, str) else json.dumps(item)))])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="tool_calls")],
                                         usage=usage, model=served)
    return FakeOpenAI


# ------------------------------------------------------------------------------------------- SDK errors --

def oai_error(cls, status, body=None, headers=None):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://fake.invalid/v1"))
    return cls("refused", response=response, body=body)


def gemini_error(code, status, details=None):
    body = {"error": {"code": code, "status": status, "message": "", "details": details or []}}
    return (genai_errors.ServerError if code >= 500 else genai_errors.ClientError)(code, body)


def timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://fake.invalid/v1"))


# ------------------------------------------------------------------------------------------- fixtures --

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any outbound connection from this process fails: a live provider request is impossible."""
    def refuse(*a, **k):
        raise OSError("network blocked by test_llm_gateway_integration")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


_LLM_MODULES = ("app.agents.llm_client", "app.llm.routing", "app.llm_gateway", "app.llm.providers",
                "app.agents.llm_client_llama")


@pytest.fixture
def world(monkeypatch, client):
    """Fake SDKs installed at their boundary, fake keys, no local model, and provider retry waits skipped
    (only waits issued by the LLM modules; the sandbox's own polling sleeps for real)."""
    w = World()
    for name, value in FAKE_KEYS.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "LLAMA_API_KEY", "")
    monkeypatch.setenv("ANALYST_MODEL_URL", "")
    monkeypatch.setattr(llm_client, "_client", FakeGemini(w))
    fake_openai = make_fake_openai(w)
    monkeypatch.setattr(providers, "OpenAI", fake_openai)
    monkeypatch.setattr(providers, "_clients", {})
    monkeypatch.setattr(llm_client_llama, "OpenAI", fake_openai)
    monkeypatch.setattr(llm_client_llama, "_client", None)
    real_sleep, real_monotonic = time.sleep, time.monotonic
    w.waits, w.skipped = [], [0.0]

    def sleep(seconds):
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if caller in _LLM_MODULES:
            w.waits.append((caller, seconds))
            w.skipped[0] += seconds           # the wait "passes" for the gateway's breakers and deadlines
            return
        real_sleep(seconds)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(llm_gateway, "_monotonic", lambda: real_monotonic() + w.skipped[0])
    routing._last_served.set(None)
    yield w
    assert runtime.get_provider_override() is None, "the gateway must not be bypassed"
    assert not w.unscripted, f"providers were asked for unscripted tools: {w.unscripted}"


@pytest.fixture
def hybrid(monkeypatch, world):
    for name, value in dict(LLM_HYBRID_ROUTING_ENABLED=True, GROQ_MODEL=MODELS["groq"], NVIDIA_MODEL=MODELS["nvidia"],
                            OPENROUTER_MODEL=MODELS["openrouter"], MISTRAL_MODEL=MODELS["mistral"],
                            MISTRAL_ENABLED=False, LLM_PROVIDER_PRIORITY="", LLM_FAILOVER_ENABLED=True,
                            LLM_CRITIC_REQUIRE_INDEPENDENT=False).items():
        monkeypatch.setattr(config, name, value)
    for agent in routing.SHOWN_AGENTS + ("chat", "comparison", "insights", "copilot"):
        monkeypatch.delenv(f"LLM_{agent.upper()}_PROVIDERS", raising=False)
        monkeypatch.delenv(f"LLM_{agent.upper()}_FALLBACK", raising=False)
    return world


@pytest.fixture
def ws(client):
    return workspace(client)


def route_log(caplog) -> list[dict]:
    """The routing engine's own record of each served call: agent, tool, provider, models, family, attempt."""
    out = []
    for r in caplog.records:
        if r.name == "app.llm.routing" and r.getMessage().startswith("llm_routing agent="):
            out.append(dict(kv.split("=", 1) for kv in r.getMessage().split(" ")[1:] if "=" in kv))
    return out


def run(client, ws, question=QUESTION):
    created = ask(client, ws, question)
    state = jobs.run_inline(created["job_id"])
    qid = created["question_id"]
    get = lambda path: client.get(f"/api/questions/{qid}/{path}", headers=ws["h"])  # noqa: E731
    return types.SimpleNamespace(qid=qid, job_id=created["job_id"], state=state, trace=get("trace").json(),
                                 manifest=get("manifest").json(), reviews=get("critic-reviews").json(),
                                 dashboard=get("dashboard"), status=get("status").json())


def llm_calls(r) -> list[dict]:
    return [e for e in r.trace["events"] if e["kind"] == "llm_call"]


def gateway_events(r) -> list[dict]:
    return [e for e in r.trace["events"] if e["kind"] == "gateway"]


# ------------------------------------------------------------------------------------------- assertions --

def served_requests(world: World) -> list[tuple[str, str]]:
    """(provider, tool) of every fake request that was answered (not failed), in order."""
    return [(r["provider"], r["tool"]) for r in world.requests if r.get("answered")]


def trace_calls(r) -> list[tuple[str, str]]:
    return [(e["detail"]["provider"], e["name"]) for e in llm_calls(r)]


def review(r) -> dict:
    assert len(r.reviews) == 1, r.reviews
    return r.reviews[0]


def check(r) -> dict:
    return review(r)["checks"][0]


def agents_in(caplog) -> list[tuple[str, str, str]]:
    return [(e["agent"], e["tool"], e["provider"]) for e in route_log(caplog)]


def set_route(monkeypatch, agent: str, providers_csv: str):
    monkeypatch.setenv(f"LLM_{agent.upper()}_PROVIDERS", providers_csv)


def no_gemini_retries(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 0)
    monkeypatch.setattr(config, "GEMINI_MALFORMED_RETRIES", 0)


STANDARD = [("triage", "submit_triage"), ("planner", "submit_plan"), ("executor", "submit_code"),
            ("executor", "submit_code"), ("critic", "submit_verification_code"), ("critic", "submit_review"),
            ("narrator", "compile_dashboard")]


# ============================================================================ A. the main pipeline, hybrid on --

def test_main_pipeline_every_stage_reaches_the_real_gateway_on_its_configured_provider(client, ws, hybrid, caplog):
    """Triage -> Planner -> Executor (2 steps, real sandbox) -> Critic -> Dashboard, every call through the gateway:
    Gemini for the pipeline, Groq (a different family) for the Critic. Mistral is never constructed."""
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded" and r.status["status"] == "verified"
    # each agent identity, as the routing engine itself logged it, on the expected provider
    assert [(a, t) for a, t, _ in agents_in(caplog)] == STANDARD
    assert {p for a, _, p in agents_in(caplog) if a != "critic"} == {"gemini"}
    assert {p for a, _, p in agents_in(caplog) if a == "critic"} == {"groq"}
    # provenance: one llm_call per served request, no duplicates, tokens and latency present
    calls = llm_calls(r)
    assert trace_calls(r) == served_requests(hybrid) and len(calls) == 7
    assert all(e["detail"]["tokens_known"] and e["detail"]["ok"] and e["duration_ms"] is not None for e in calls)
    assert r.trace["usage"]["tokens_in"] == 5 * 100 + 2 * 120 and r.trace["usage"]["provider_fallbacks"] == 0
    assert not [e for e in r.trace["events"] if e["kind"] == "provider"], "no failed or skipped provider requests"
    assert len([e for e in gateway_events(r) if e["detail"]["outcome"] == "ok"]) == 7
    assert r.manifest["models"] == {
        "gemini:gemini-3.6-flash": {"calls": 5, "tokens_in": 500, "tokens_out": 200, "fallback_calls": 0},
        "groq:openai/gpt-oss-120b": {"calls": 2, "tokens_in": 240, "tokens_out": 120, "fallback_calls": 0}}
    assert r.manifest["verifier_models"] == ["groq:openai/gpt-oss-120b"]
    assert r.manifest["config"]["hybrid_routing_enabled"] is True
    # the persisted Critic record
    c = check(r)
    assert c["verifier_model"] == "groq:openai/gpt-oss-120b" and c["verifier_independent"] is True
    assert c["checked_families"] == ["google-gemini"] and c["verifier_family"] == "openai-gpt-oss"
    # the dashboard receives the Critic's verdict and summary
    dash = r.dashboard.json()
    assert dash["verdict_state"] == "VERIFIED" and dash["verification_summary"] == review(r)["summary"]
    # requests were well formed: forced tool calls, fake keys, bounded timeouts; Mistral never built
    assert all(q["forced"] and q["timeout_ms"] for q in hybrid.calls("gemini"))
    assert all(q["tool_choice"] == {"type": "function", "function": {"name": q["tool"]}} and q["timeout"]
               for q in hybrid.calls("groq"))
    assert [(c["provider"], c["api_key"], c["max_retries"]) for c in hybrid.constructed] == [("groq", "fake-groq-key", 0)]


def test_main_pipeline_records_the_requested_and_the_served_model_separately(client, ws, hybrid, monkeypatch, caplog):
    """The Executor runs on NVIDIA, which reports a different model id than the one requested."""
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    set_route(monkeypatch, "executor", "nvidia")
    hybrid.serve_pipeline("gemini", ("submit_triage", "submit_plan", "compile_dashboard"))
    hybrid.serve_pipeline("nvidia", ("submit_code",))
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded"
    code_calls = [e["detail"] for e in llm_calls(r) if e["name"] == "submit_code"]
    assert all(d["provider"] == "nvidia" and d["model"] == MODELS["nvidia"] and d["served_model"] == SERVED["nvidia"]
               for d in code_calls) and len(code_calls) == 2
    assert all("served_model" not in e["detail"] for e in llm_calls(r) if e["detail"]["provider"] != "nvidia")
    entry = r.manifest["models"][f"nvidia:{MODELS['nvidia']}"]
    assert entry["calls"] == 2 and entry["served_models"] == [SERVED["nvidia"]]
    executor = [e for e in route_log(caplog) if e["agent"] == "executor"]
    assert {(e["model"], e["served_model"], e["family"]) for e in executor} == {
        (MODELS["nvidia"], SERVED["nvidia"], "nvidia-nemotron")}
    c = check(r)
    assert c["checked_families"] == ["google-gemini", "nvidia-nemotron"] and c["verifier_independent"] is True


def test_main_pipeline_records_a_planner_failover_and_the_critic_avoids_both_families(client, ws, hybrid, monkeypatch,
                                                                                     caplog):
    """Gemini answers the plan with 503; Groq takes it (recorded as a fallback). The Planner's family is now gpt-oss
    and the Executor's Gemini, so the Critic must use NVIDIA."""
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    no_gemini_retries(monkeypatch)
    hybrid.serve_pipeline("gemini", ("submit_triage", "submit_code", "compile_dashboard"))
    hybrid.serve("gemini", "submit_plan", gemini_error(503, "UNAVAILABLE"))
    hybrid.serve_pipeline("groq", ("submit_plan",))
    hybrid.serve_pipeline("nvidia", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded"
    plan = [e["detail"] for e in llm_calls(r) if e["name"] == "submit_plan"]
    assert plan == [{**plan[0], "provider": "groq", "fallback": True}] and len(plan) == 1, "one record, the served call"
    failed = [e for e in r.trace["events"] if e["kind"] == "provider"]
    assert [(e["detail"]["provider"], e["detail"]["kind"], e["detail"]["status"], e["detail"]["action"])
            for e in failed] == [("gemini", "overloaded", 503, "failover_to_groq")]
    assert r.trace["usage"]["provider_fallbacks"] == 1
    assert r.manifest["models"]["groq:openai/gpt-oss-120b"]["fallback_calls"] == 1
    assert ("planner", "submit_plan", "groq") in agents_in(caplog)
    c = check(r)
    assert c["verifier_model"].startswith("nvidia:") and c["verifier_independent"] is True
    assert c["checked_families"] == ["google-gemini", "openai-gpt-oss"]
    assert hybrid.calls("groq", "submit_verification_code") == []


def test_two_consecutive_jobs_on_one_thread_do_not_share_provider_history(client, ws, hybrid, monkeypatch):
    """Job 1's Executor runs on Groq (so its Critic is NVIDIA). Job 2 runs on Gemini only: its Critic may use Groq,
    and its record lists only Gemini -- nothing carried over from job 1."""
    set_route(monkeypatch, "executor", "groq")
    hybrid.serve_pipeline("gemini", ("submit_triage", "submit_plan", "compile_dashboard"))
    hybrid.serve_pipeline("groq", ("submit_code",))
    hybrid.serve_pipeline("nvidia", CRITIC_TOOLS)
    first = run(client, ws)
    assert check(first)["checked_families"] == ["google-gemini", "openai-gpt-oss"]
    monkeypatch.delenv("LLM_EXECUTOR_PROVIDERS")
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    second = run(client, ws)
    c = check(second)
    assert c["checked_families"] == ["google-gemini"], "job 1's Groq Executor did not leak into job 2"
    assert c["verifier_model"] == "groq:openai/gpt-oss-120b" and c["verifier_independent"] is True


# ============================================================================== B. provider failover --

def test_failover_a_gemini_timeout_hands_the_plan_to_groq_without_a_retry(client, ws, hybrid, monkeypatch):
    hybrid.serve_pipeline("gemini", ("submit_triage", "submit_code", "compile_dashboard"))
    hybrid.serve("gemini", "submit_plan", httpx.ReadTimeout("timed out"))
    hybrid.serve_pipeline("groq", ("submit_plan",))
    hybrid.serve_pipeline("nvidia", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded" and len(hybrid.calls("gemini", "submit_plan")) == 1, "a timeout is not retried"
    failed = [e["detail"] for e in r.trace["events"] if e["kind"] == "provider"]
    assert [(d["provider"], d["kind"], d["action"]) for d in failed] == [("gemini", "network", "failover_to_groq")]
    assert [g["detail"]["error"] for g in gateway_events(r) if g["detail"]["outcome"] == "failed"] == ["TIMEOUT"]


def test_failover_a_gemini_429_burst_is_retried_after_its_retry_info_and_then_succeeds(client, ws, hybrid):
    rate = gemini_error(429, "RESOURCE_EXHAUSTED", [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}])
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.scripts["gemini"]["submit_plan"].insert(0, rate)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded" and len(hybrid.calls("gemini", "submit_plan")) == 2
    assert [s for m, s in hybrid.waits if m == "app.agents.llm_client"] == [2.0], "waited what RetryInfo asked"
    plan = [e["detail"] for e in llm_calls(r) if e["name"] == "submit_plan"]
    assert len(plan) == 1 and plan[0]["provider"] == "gemini" and plan[0]["fallback"] is False
    assert [e["detail"]["action"] for e in r.trace["events"] if e["kind"] == "provider"] == ["retry"]


def test_failover_a_groq_429_retry_after_is_waited_out_before_the_critic_retries(client, ws, hybrid):
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve("groq", "submit_verification_code",
                 oai_error(openai.RateLimitError, 429, {"code": "rate_limit_exceeded"}, {"retry-after": "3"}))
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded" and len(hybrid.calls("groq", "submit_verification_code")) == 2
    waited = [s for m, s in hybrid.waits if m == "app.llm.routing"]
    assert len(waited) == 1 and 3.0 <= waited[0] <= 3.6, "never sooner than Retry-After"
    assert check(r)["verifier_model"] == "groq:openai/gpt-oss-120b" and check(r)["verifier_independent"] is True


def test_failover_5xx_is_retried_a_bounded_number_of_times_then_the_next_provider_serves(client, ws, hybrid):
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve("groq", "submit_verification_code", *[oai_error(openai.InternalServerError, 503)] * 2)
    hybrid.serve_pipeline("groq", ("submit_review",))
    hybrid.serve_pipeline("nvidia", ("submit_verification_code",))
    r = run(client, ws)
    assert r.state == "succeeded"
    assert len(hybrid.calls("groq", "submit_verification_code")) == 1 + config.LLM_PROVIDER_TRANSIENT_RETRIES
    c = check(r)
    assert c["verifier_model"].startswith("nvidia:") and c["review_verifier_model"] == "groq:openai/gpt-oss-120b"
    assert c["verifier_independent"] is True


@pytest.mark.parametrize("error,held", [
    (oai_error(openai.AuthenticationError, 401, {"code": "invalid_api_key"}), "credentials_rejected"),
    (oai_error(openai.APIStatusError, 402, {"type": "billing_no_api_budget_configured"}), "billing_held"),
])
def test_failover_auth_and_billing_errors_move_on_and_hold_the_provider(client, ws, hybrid, error, held):
    """Groq refuses the verification call; NVIDIA takes it. The review call does not go to Groq at all."""
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve("groq", "submit_verification_code", error)
    hybrid.serve_pipeline("nvidia", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded" and len(hybrid.calls("groq")) == 1, "never retried, and not asked again"
    assert llm_gateway.status()["providers"]["groq"][held] is True
    c = check(r)
    assert c["verifier_model"].startswith("nvidia:") and c["review_verifier_model"].startswith("nvidia:")
    skipped = [e["detail"] for e in r.trace["events"] if e["kind"] == "provider" and e["detail"]["provider"] == "groq"]
    assert [d["kind"] for d in skipped] == [{"credentials_rejected": "authentication", "billing_held": "billing"}[held],
                                            {"credentials_rejected": "auth_rejected", "billing_held": "billing_hold"}[held]]


@pytest.mark.parametrize("error", [
    oai_error(openai.NotFoundError, 404, {"message": "The model `x` does not exist"}),
    oai_error(openai.BadRequestError, 400, {"type": "invalid_model", "message": "Invalid model: x"}),
])
def test_failover_an_invalid_model_is_skipped_without_retry_on_each_call(client, ws, hybrid, error):
    """Not held (a known Phase 2 P2 item): each of the two Critic calls asks Groq once, never twice."""
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve("groq", "submit_verification_code", error)
    hybrid.serve("groq", "submit_review", error)
    hybrid.serve_pipeline("nvidia", CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded"
    assert [q["tool"] for q in hybrid.calls("groq")] == ["submit_verification_code", "submit_review"]
    kinds = [e["detail"]["kind"] for e in r.trace["events"] if e["kind"] == "provider"]
    assert kinds == ["model_not_found", "model_not_found"]


def test_failover_a_provider_refusing_tool_calls_is_asked_once_with_required_then_skipped(client, ws, hybrid):
    """Current behaviour, documented (no capability tracking exists): forced tool_choice, then "required", then the
    next provider."""
    refusal = oai_error(openai.BadRequestError, 400, {"type": "invalid_request_error",
                                                      "message": "this model does not support tool use"})
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve("groq", "submit_verification_code", refusal, refusal)
    hybrid.serve_pipeline("groq", ("submit_review",))
    hybrid.serve_pipeline("nvidia", ("submit_verification_code",))
    r = run(client, ws)
    asked = hybrid.calls("groq", "submit_verification_code")
    assert [q["tool_choice"] if isinstance(q["tool_choice"], str) else "forced" for q in asked] == ["forced", "required"]
    assert asked[1]["max_tokens"] > asked[0]["max_tokens"]
    assert r.state == "succeeded" and check(r)["verifier_model"].startswith("nvidia:")


@pytest.mark.parametrize("bad", [
    {"_tool": "some_other_tool", "what_it_checks": "x", "code": "print(1)"},     # wrong tool
    {"what_it_checks": "totals"},                                               # required "code" missing
    "{not json",                                                                # malformed arguments
])
def test_failover_a_malformed_tool_reply_is_never_accepted(client, ws, hybrid, bad):
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve("groq", "submit_verification_code", bad)
    hybrid.serve_pipeline("groq", ("submit_review",))
    hybrid.serve_pipeline("nvidia", ("submit_verification_code",))
    r = run(client, ws)
    assert r.state == "succeeded" and check(r)["verifier_model"].startswith("nvidia:")
    rejected = [e["detail"] for e in llm_calls(r) if e["detail"]["provider"] == "groq" and not e["detail"]["ok"]]
    assert len(rejected) == 1 and rejected[0]["model"] == MODELS["groq"], "recorded as a failed call, not a success"


def test_failover_when_every_provider_fails_the_run_fails_with_each_provider_named(client, ws, hybrid, monkeypatch):
    no_gemini_retries(monkeypatch)
    down = oai_error(openai.InternalServerError, 503)
    hybrid.serve("gemini", "submit_triage", gemini_error(503, "UNAVAILABLE"))
    for p in ("groq", "nvidia", "openrouter"):
        hybrid.serve(p, "submit_triage", down, down)
    r = run(client, ws)
    job = r.status["job"]
    assert r.state == "failed" and job["failure_class"] == "provider_unavailable"
    for name in ("Gemini", "Groq", "Nvidia", "Openrouter"):
        assert name in job["error"], job["error"]
    assert "fake-" not in job["error"], "no key in the message"
    assert llm_calls(r) == [] and hybrid.calls("mistral") == []
    assert [len(hybrid.calls(p)) for p in ("gemini", "groq", "nvidia", "openrouter")] == [1, 2, 2, 2]


def test_failover_cancellation_mid_run_stops_every_further_provider_call(client, ws, hybrid):
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    created = ask(client, ws, QUESTION)

    def cancel_on_first_code_step(provider, tool):
        if tool == "submit_code" and not hybrid.calls(tool="submit_code")[:-1]:
            assert client.post(f"/api/questions/{created['question_id']}/cancel", headers=ws["h"]).status_code == 200
            # A running job notices a cancel at its next checkpoint, but re-reads the job row at most every 2 s
            # (jobs._Heartbeat.should_stop). The fakes answer in milliseconds, so wait out one poll interval;
            # otherwise the run finishes before it could have seen the request.
            time.sleep(2.1)
    hybrid.on_request = cancel_on_first_code_step
    state = jobs.run_inline(created["job_id"])
    hybrid.on_request = None
    assert state == "cancelled"
    assert hybrid.calls(tool="submit_verification_code") == [] and hybrid.calls("groq") == []
    gw = llm_gateway.status()["providers"]
    assert all(v["failed"] == 0 for v in gw.values()), "a cancellation is not a provider failure"
    # leave the unused scripted replies alone: they were never requested
    hybrid.scripts.clear()


def test_mistral_is_never_used_unless_explicitly_enabled(client, ws, hybrid, monkeypatch):
    set_route(monkeypatch, "critic", "mistral,groq")
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = run(client, ws)
    assert check(r)["verifier_model"].startswith("groq:") and "mistral" not in [c["provider"] for c in hybrid.constructed]
    monkeypatch.setattr(config, "MISTRAL_ENABLED", True)
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("mistral", CRITIC_TOOLS)
    r2 = run(client, ws)
    c = check(r2)
    assert c["verifier_model"] == f"mistral:{MODELS['mistral']}" and c["verifier_family"] == "mistral"
    assert c["verifier_independent"] is True


# ============================================================================ C. Critic independence --

def critic_case(client, ws, hybrid, monkeypatch, *, planner="gemini", executor="gemini", critic_route=None,
                executor_replies=None):
    if critic_route:
        set_route(monkeypatch, "critic", critic_route)
    set_route(monkeypatch, "planner", planner)
    if executor_replies is None:
        set_route(monkeypatch, "executor", executor)
        hybrid.serve_pipeline(executor, ("submit_code",))
    else:
        set_route(monkeypatch, "executor", executor)
        for provider, reply in executor_replies:
            hybrid.serve(provider, "submit_code", reply)
    hybrid.serve_pipeline(planner, ("submit_plan",))
    hybrid.serve_pipeline("gemini", ("submit_triage", "compile_dashboard"))
    return run(client, ws)


@pytest.mark.parametrize("planner,executor,critic_provider,checked", [
    ("gemini", "gemini", "groq", ["google-gemini"]),
    ("gemini", "groq", "nvidia", ["google-gemini", "openai-gpt-oss"]),
    ("groq", "gemini", "nvidia", ["google-gemini", "openai-gpt-oss"]),
    ("gemini", "nvidia", "groq", ["google-gemini", "nvidia-nemotron"]),
])
def test_critic_full_run_avoids_every_family_that_planned_or_executed(client, ws, hybrid, monkeypatch, planner,
                                                                       executor, critic_provider, checked):
    hybrid.serve_pipeline(critic_provider, CRITIC_TOOLS)
    r = critic_case(client, ws, hybrid, monkeypatch, planner=planner, executor=executor)
    assert r.state == "succeeded"
    c = check(r)
    assert c["verifier_model"].startswith(f"{critic_provider}:") and c["verifier_independent"] is True
    assert c["checked_families"] == checked and c["verifier_family"] not in checked
    assert "not cross-model" not in review(r)["summary"]
    for other in {"groq", "nvidia", "openrouter"} - {critic_provider}:
        assert hybrid.calls(other, "submit_verification_code") == []


def test_critic_full_run_checks_every_executor_family_in_the_run(client, ws, hybrid, monkeypatch):
    """Executor step 1 on Gemini, step 2 fails over to Groq: both families are checked, NVIDIA verifies."""
    no_gemini_retries(monkeypatch)
    hybrid.serve_pipeline("nvidia", CRITIC_TOOLS)
    r = critic_case(client, ws, hybrid, monkeypatch, executor="gemini,groq",
                    executor_replies=[("gemini", SCRIPT["submit_code"][0]), ("gemini", gemini_error(503, "UNAVAILABLE")),
                                      ("groq", SCRIPT["submit_code"][1])])
    assert r.state == "succeeded"
    assert [e["detail"]["provider"] for e in llm_calls(r) if e["name"] == "submit_code"] == ["gemini", "groq"]
    c = check(r)
    assert c["checked_families"] == ["google-gemini", "openai-gpt-oss"] and c["verifier_model"].startswith("nvidia:")


def test_critic_full_run_nvidia_executor_with_an_openrouter_nemotron_critic_is_not_independent(client, ws, hybrid,
                                                                                              monkeypatch):
    hybrid.serve_pipeline("openrouter", CRITIC_TOOLS)
    r = critic_case(client, ws, hybrid, monkeypatch, executor="nvidia", critic_route="openrouter")
    c = check(r)
    assert c["verifier_model"] == f"openrouter-fallback:{MODELS['openrouter']}" and c["verifier_independent"] is False
    assert c["verifier_family"] == "nvidia-nemotron" and "nvidia-nemotron" in c["checked_families"]
    summary = review(r)["summary"]
    assert "not cross-model" in summary
    assert r.dashboard.json()["verification_summary"] == summary, "the dashboard shows the qualified summary"
    assert r.manifest["verifier_models"] == [f"openrouter-fallback:{MODELS['openrouter']}"]


def test_critic_full_run_with_an_unknown_executor_family_claims_no_independence(client, ws, hybrid, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "openrouter/free")
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = critic_case(client, ws, hybrid, monkeypatch, executor="openrouter")
    c = check(r)
    assert "unknown" in c["checked_families"] and c["verifier_independent"] is False
    assert c["verifier_model"] == "groq-fallback:openai/gpt-oss-120b"


def test_critic_full_run_with_no_independent_provider_marks_the_fallback(client, ws, hybrid, monkeypatch):
    """Gemini planned; the Executor used Groq then NVIDIA. Every Critic provider is in a family that shaped the
    work, so the verification still runs but is recorded as not independent."""
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    down = oai_error(openai.InternalServerError, 503)
    r = critic_case(client, ws, hybrid, monkeypatch, executor="groq,nvidia",
                    executor_replies=[("groq", SCRIPT["submit_code"][0]), ("groq", down), ("groq", down),
                                      ("nvidia", SCRIPT["submit_code"][1])])
    c = check(r)
    assert c["checked_families"] == ["google-gemini", "nvidia-nemotron", "openai-gpt-oss"]
    assert c["verifier_model"] == "groq-fallback:openai/gpt-oss-120b" and c["verifier_independent"] is False
    assert "not cross-model" in review(r)["summary"]


def test_critic_full_run_needs_both_the_verify_and_the_review_call_independent(client, ws, hybrid, monkeypatch):
    hybrid.serve_pipeline("groq", ("submit_verification_code",))
    hybrid.serve("groq", "submit_review", oai_error(openai.AuthenticationError, 401))
    hybrid.serve_pipeline("openrouter", ("submit_review",))
    r = critic_case(client, ws, hybrid, monkeypatch, executor="nvidia", critic_route="groq,openrouter")
    c = check(r)
    assert c["verifier_model"] == "groq:openai/gpt-oss-120b"
    assert c["review_verifier_model"] == f"openrouter-fallback:{MODELS['openrouter']}"
    assert c["verifier_independent"] is False and "not cross-model" in review(r)["summary"]


# ============================================================================== D. other agent workflows --

def test_triage_and_dashboard_are_their_own_agents_in_the_route_log(client, ws, hybrid, monkeypatch, caplog):
    """Triage and the Dashboard (Narrator) follow their own routes: sent to Groq here, while the rest stays on Gemini."""
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    set_route(monkeypatch, "triage", "groq")
    set_route(monkeypatch, "narrator", "groq")
    hybrid.serve_pipeline("gemini", ("submit_plan", "submit_code"))
    hybrid.serve_pipeline("groq", ("submit_triage", "compile_dashboard") + CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded"
    log = agents_in(caplog)
    assert ("triage", "submit_triage", "groq") in log and ("narrator", "compile_dashboard", "groq") in log
    # Triage and the Dashboard do not shape the analysis the Critic checks: only the Planner's and the Executor's
    # families count, so Groq (which served Triage and the Dashboard) may still verify, independently.
    c = check(r)
    assert c["checked_families"] == ["google-gemini"]
    assert c["verifier_model"] == "groq:openai/gpt-oss-120b" and c["verifier_independent"] is True


def test_finding_triage_quota_message_depends_on_error_text_legacy_path(client, ws, world, monkeypatch):
    """FINDING (not fixed in this phase): Triage recognises a usage-limit failure by searching the error text for
    "429" / "RESOURCE_EXHAUSTED" (agents/triage.py). On the legacy path the text carries Gemini's own message, so the
    user gets the usage-limit rejection."""
    monkeypatch.setattr(config, "LLM_HYBRID_ROUTING_ENABLED", False)
    no_gemini_retries(monkeypatch)
    world.serve("gemini", "submit_triage", gemini_error(429, "RESOURCE_EXHAUSTED"))
    r = run(client, ws)
    assert r.state == "succeeded" and r.status["status"] == "rejected" or "usage limit" in json.dumps(r.status)


TRIAGE_USAGE_LIMIT = "try again in a few minutes"      # the wording of Triage's own usage-limit rejection


def triage_hits_the_usage_limit(hybrid):
    """Groq, NVIDIA and OpenRouter all answer Triage with a 429 burst limit (two attempts each: 1 + one retry)."""
    for provider in ("groq", "nvidia", "openrouter"):
        hybrid.serve(provider, "submit_triage", *[oai_error(openai.RateLimitError, 429,
                                                            {"code": "rate_limit_exceeded"}) for _ in range(2)])


def test_finding_triage_shows_its_usage_limit_rejection_while_gemini_is_in_its_chain(client, ws, hybrid, monkeypatch):
    """Control for the finding below. Gemini's failure text carries its own "429 RESOURCE_EXHAUSTED", which Triage's
    string match (agents/triage.py) recognises: the question is rejected with the friendly usage-limit message."""
    no_gemini_retries(monkeypatch)
    hybrid.serve("gemini", "submit_triage", gemini_error(429, "RESOURCE_EXHAUSTED"))
    triage_hits_the_usage_limit(hybrid)
    r = run(client, ws)
    assert r.state == "succeeded" and r.status["status"] == "rejected"
    assert TRIAGE_USAGE_LIMIT in r.status["error"]


def test_triage_rejects_with_its_usage_limit_message_when_gemini_is_absent_from_its_chain(client, ws, hybrid,
                                                                                         monkeypatch):
    """Regression (was a finding): Triage used to find a usage limit by searching str(error) for "429", which only
    Gemini's own error text contained, so a 429 from Groq / NVIDIA / OpenRouter failed the run. It now reads the
    structured failure records, so the question is rejected with the usage-limit message whatever the chain."""
    set_route(monkeypatch, "triage", "groq,nvidia,openrouter")
    triage_hits_the_usage_limit(hybrid)
    r = run(client, ws)
    assert r.state == "succeeded" and r.status["status"] == "rejected" and TRIAGE_USAGE_LIMIT in r.status["error"]
    assert hybrid.calls("gemini") == [] and len(hybrid.calls("groq")) == 2


def test_triage_a_single_non_gemini_429_is_enough(client, ws, hybrid, monkeypatch):
    """One provider's HTTP 429, the others merely unconfigured: still a usage limit."""
    set_route(monkeypatch, "triage", "groq")
    hybrid.serve("groq", "submit_triage", *[oai_error(openai.RateLimitError, 429) for _ in range(2)])
    r = run(client, ws)
    assert r.status["status"] == "rejected" and TRIAGE_USAGE_LIMIT in r.status["error"]


def test_triage_an_ordinary_provider_failure_is_not_a_usage_limit(client, ws, hybrid, monkeypatch):
    """503s are an outage, not quota exhaustion: the run still fails as provider_unavailable, with no rejection."""
    set_route(monkeypatch, "triage", "groq,nvidia,openrouter")
    down = oai_error(openai.InternalServerError, 503)
    for p in ("groq", "nvidia", "openrouter"):
        hybrid.serve(p, "submit_triage", down, down)
    r = run(client, ws)
    job = r.status["job"]
    assert r.state == "failed" and job["failure_class"] == "provider_unavailable"
    assert TRIAGE_USAGE_LIMIT not in job["error"] and "usage limit" not in job["error"]


def test_triage_usage_limit_is_read_from_structured_failures_not_text():
    from app.agents.triage import _hit_usage_limit
    PU = runtime.ProviderUnavailable
    assert _hit_usage_limit(PU("x", failures=[{"provider": "groq", "kind": "rate_limited", "status": 429}]))
    assert _hit_usage_limit(PU("x", failures=[{"provider": "nvidia", "kind": "quota_exhausted", "status": None}]))
    assert _hit_usage_limit(PU("x", failures=[{"provider": "a", "kind": "server_error", "status": 503},
                                              {"provider": "b", "kind": "rate_limited", "status": 429}]))
    # Text containing "429" or the Google status name is not evidence: a model ID, a request ID, a message
    assert not _hit_usage_limit(PU("Groq(vendor/model-429b): ServerError (PROVIDER_UNAVAILABLE) RESOURCE_EXHAUSTED",
                                   failures=[{"provider": "groq", "kind": "server_error", "status": 503,
                                              "model": "vendor/model-429b"}]))
    assert not _hit_usage_limit(RuntimeError("request 429 failed"))
    assert not _hit_usage_limit(PU("no key", failures=[{"provider": "groq", "kind": "not_configured"}]))
    # a raw SDK error carries its own status
    assert _hit_usage_limit(type("E", (Exception,), {"status_code": 429})("x"))
    assert not _hit_usage_limit(type("E", (Exception,), {"status_code": 500})("429"))


def test_triage_the_hybrid_failure_text_has_no_429_but_the_structured_record_does(hybrid, monkeypatch):
    """Why text matching could never work here: str(error) is written from error kinds and holds no "429"; the status
    lives in `failures`."""
    set_route(monkeypatch, "triage", "groq,nvidia,openrouter")
    triage_hits_the_usage_limit(hybrid)
    with pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(system="s", user_content="u", tool_name="submit_triage",
                             tool_schema={"type": "object", "properties": {}}, tool_description="d")
    assert "429" not in str(e.value) and "RESOURCE_EXHAUSTED" not in str(e.value)
    tried = [f for f in e.value.failures if f["kind"] != "not_configured"]
    assert [(f["provider"], f["kind"], f["status"]) for f in tried] == [
        ("groq", "rate_limited", 429), ("nvidia", "rate_limited", 429), ("openrouter", "rate_limited", 429)]


def test_comparison_route_gets_its_plan_from_the_gateway(client, ws, hybrid, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    monkeypatch.setattr(config, "COMPARISON_MODEL_PLANNER", True)
    no_gemini_retries(monkeypatch)
    plan = {"outcome": "revenue", "explanatory": "channel", "levels": ["Online", "Retail"], "method": "t_test",
            "claims_cause": False, "assumptions": ["web orders are the Online channel"]}
    hybrid.serve("gemini", "submit_comparison_plan", gemini_error(503, "UNAVAILABLE"))
    hybrid.serve("groq", "submit_comparison_plan", plan)
    r = run(client, ws, "Do web orders earn more per sale than store orders?")
    assert r.state == "succeeded" and any("a model proposed a comparison plan" in x for x in r.trace["route_reasons"])
    assert agents_in(caplog) == [("comparison", "submit_comparison_plan", "groq")]
    assert trace_calls(r) == [("groq", "submit_comparison_plan")] and r.trace["usage"]["provider_fallbacks"] == 1


def test_investigation_engine_runs_its_whole_loop_through_the_gateway(hybrid, monkeypatch, caplog):
    """The hypothesis engine as agents/hypothesis.py wires it (call_model=llm_client.call_tool): understanding,
    hypotheses, next steps and the report all served through the gateway; the first call fails over to Groq."""
    from test_hypothesis_engine import QUESTION as HQ, script as hyp_script
    from platform_helpers import FIXTURES
    from app import hypothesis_engine as he
    from app.profiling import load_and_profile_csv
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    no_gemini_retries(monkeypatch)
    replies = hyp_script()
    hybrid.serve("gemini", "submit_understanding", gemini_error(503, "UNAVAILABLE"))
    hybrid.serve("groq", "submit_understanding", *replies.pop("submit_understanding"))
    for tool, items in replies.items():
        hybrid.serve("gemini", tool, *items)
    df, profile = load_and_profile_csv(str(FIXTURES / "sales.csv"))
    report = he.HypothesisEngine(question=HQ, df=df, profile=profile, call_model=llm_client.call_tool).run()
    assert report["understanding"]["metric"]["formula"] == "sum(revenue)"
    log = agents_in(caplog)
    assert {a for a, _, _ in log} == {"investigations"}
    assert log[0] == ("investigations", "submit_understanding", "groq")
    assert {t for _, t, p in log if p == "gemini"} == {"submit_hypotheses", "submit_next_step", "submit_report"}
    assert routing.served_for("investigations")["provider"] == "gemini", "outside a run: this context's record"


def test_chat_answer_and_its_check_go_through_the_gateway(client, ws, hybrid, caplog):
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    r = run(client, ws)
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    hybrid.serve("gemini", "submit_answer", {"answer": "Total revenue was 277,110.00.", "sources": ["Total revenue"],
                                            "needs_new_analysis": False, "suggested_question": ""})
    hybrid.serve("groq", "submit_check", {"supported": True, "unsupported_claims": []})
    resp = client.post(f"/api/questions/{r.qid}/chat", json={"message": "What was total revenue?"}, headers=ws["h"])
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["verified"] is True and out["checked_by"] == "groq:openai/gpt-oss-120b"
    assert agents_in(caplog) == [("chat", "submit_answer", "gemini"), ("critic", "submit_check", "groq")]


def test_scenario_nl_translation_goes_through_the_gateway(hybrid, monkeypatch, caplog):
    from app import scenario_nl as nl
    from app import scenarios
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    no_gemini_retries(monkeypatch)
    good = {"action": "evaluate", "scenarios": [{"name": "cheaper units",
                                                 "adjustments": [{"input": "unit_cost", "type": "pct", "value": -8}]}]}
    hybrid.serve("gemini", nl.TOOL_NAME, gemini_error(503, "UNAVAILABLE"))
    hybrid.serve("groq", nl.TOOL_NAME, good)
    out = nl.translate_with_model("if the cost per unit falls by 8% what happens to profit?",
                                  scenarios._model("unit_economics"))
    assert out["status"] == "ok" and out["translated_by"] == "model"
    assert agents_in(caplog) == [("whatif", nl.TOOL_NAME, "groq")]


def test_whatif_ranker_goes_through_the_gateway(hybrid, caplog):
    from app.whatif import llm as whatif_llm
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    hybrid.serve("gemini", whatif_llm.TOOL, {"ids": ["b", "a"]})
    order = whatif_llm.ranker("attendance", [{"id": "a", "question": "q1", "reason": "r1"},
                                             {"id": "b", "question": "q2", "reason": "r2"}])
    assert order == ["b", "a"] and agents_in(caplog) == [("whatif", "rank_scenarios", "gemini")]


def test_with_no_usable_provider_the_optional_model_steps_are_skipped_without_any_request(hybrid, monkeypatch):
    from app import scenario_nl as nl
    from app import scenarios
    from app.whatif import llm as whatif_llm
    for name in FAKE_KEYS:
        monkeypatch.setattr(config, name, "")
    with pytest.raises(RuntimeError, match="no language model available"):
        whatif_llm.ranker("x", [{"id": "a", "question": "q", "reason": "r"}])
    assert nl.translate_with_model("profit if price rises 5%?", scenarios._model("unit_economics"))["status"] == "unsupported"
    assert hybrid.requests == [] and hybrid.constructed == []


# ============================================================================= E. hybrid routing off --

def test_hybrid_off_the_legacy_gemini_path_is_unchanged(client, ws, world, caplog):
    caplog.set_level(logging.INFO, logger="app.llm.routing")
    world.serve_pipeline("gemini")
    r = run(client, ws)
    assert r.state == "succeeded" and config.LLM_HYBRID_ROUTING_ENABLED is False
    assert {p for p, _ in served_requests(world)} == {"gemini"} and route_log(caplog) == []
    c = check(r)
    assert c["verifier_model"] == f"gemini-fallback:{config.GEMINI_MODEL}"
    assert not {"verifier_independent", "checked_families", "review_verifier_family"} & set(c)
    assert "(Llama unavailable)" in review(r)["summary"]
    assert r.manifest["config"]["hybrid_routing_enabled"] is False and world.constructed == []


def test_hybrid_off_the_legacy_llama_critic_and_failover_keep_their_own_settings(client, ws, world, monkeypatch):
    """LLAMA_* drive the legacy provider exactly as before; GROQ_* are never read for it."""
    monkeypatch.setattr(config, "LLAMA_API_KEY", "fake-llama-key")
    monkeypatch.setattr(config, "LLAMA_BASE_URL", "https://llama.invalid/v1")
    monkeypatch.setattr(config, "LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
    no_gemini_retries(monkeypatch)
    world.serve_pipeline("gemini", ("submit_triage", "submit_code", "compile_dashboard"))
    world.serve("gemini", "submit_plan", gemini_error(503, "UNAVAILABLE"))
    world.serve_pipeline("llama", ("submit_plan",) + CRITIC_TOOLS)
    r = run(client, ws)
    assert r.state == "succeeded"
    assert world.constructed == [{"provider": "llama", "base_url": "https://llama.invalid/v1", "api_key": "fake-llama-key",
                                  "max_retries": 0, "headers": {}}], "only the legacy client, with LLAMA_* settings"
    plan = [e["detail"] for e in llm_calls(r) if e["name"] == "submit_plan"]
    assert len(plan) == 1 and plan[0]["provider"] == "llama" and plan[0]["fallback"] is True
    assert check(r)["verifier_model"] == "llama:Llama-4-Maverick-17B-128E-Instruct-FP8"
    assert "verifier_independent" not in check(r)


# ======================================================================================== bypass proof --

def test_the_gateway_is_not_bypassed(client, ws, hybrid):
    """Every answer the pipeline used came from a request a fake SDK received, through the gateway: each served
    request has exactly one llm_call record and one gateway 'ok' event, the gateway's own counters match, no scripted
    override was installed, and nothing in the fakes was left unrequested."""
    hybrid.serve_pipeline("gemini", PIPELINE_TOOLS)
    hybrid.serve_pipeline("groq", CRITIC_TOOLS)
    assert runtime.get_provider_override() is None
    r = run(client, ws)
    assert r.state == "succeeded"
    assert trace_calls(r) == served_requests(hybrid) == [
        ("gemini", "submit_triage"), ("gemini", "submit_plan"), ("gemini", "submit_code"), ("gemini", "submit_code"),
        ("groq", "submit_verification_code"), ("groq", "submit_review"), ("gemini", "compile_dashboard")]
    ok = [(e["detail"]["provider"]) for e in gateway_events(r) if e["detail"]["outcome"] == "ok"]
    assert sorted(ok) == sorted(p for p, _ in served_requests(hybrid))
    gw = llm_gateway.status()
    assert gw["providers"]["gemini"]["succeeded"] == 5 and gw["providers"]["groq"]["succeeded"] == 2
    assert all(not queue for tools in hybrid.scripts.values() for queue in tools.values()), "every reply was used"
