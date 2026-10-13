"""Evidence provenance (which run, dataset and framing produced a statistic, and
whether that run finished) and provider diagnostics (what failed, how, and
what happened next).

No test here reaches a real provider. `FakeGemini` stands in for the SDK
client underneath the real adapter, so the retry loop, failure classification,
logging, fallback and recovery run exactly as they do in production."""
from __future__ import annotations

import copy
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from google.genai import errors
from platform_helpers import ask, job_row, question_row, workspace

import test_statistics as stat_fixtures
from app import analysis_ops as ops
from app import config, hypothesis_engine as he, jobs, runtime
from app.agents import llm_client
from app.agents.hypothesis import PROVENANCE_KEY, SUPERSEDED_MARK, DbStore
from app.database import SessionLocal
from app.models import Investigation, InvestigationAnalysis, InvestigationHypothesis
from app.profiling import load_and_profile_csv
from app.stats import result as stat_result

QUESTION = stat_fixtures.QUESTION
SECRET_KEY, PROMPT_MARKER, OUTPUT_MARKER = "test-key-DO-NOT-LOG-4f9a", "PROMPT-MARKER-71c2", "GENERATED-TEXT-MARKER-88d1"


# ------------------------------------------------------------ the stand-in --

def reply(tool: str, args: dict, finish: str = "STOP"):
    part = NS(function_call=NS(name=tool, args=args), text=None)
    return NS(candidates=[NS(finish_reason=NS(name=finish), content=NS(parts=[part]))], prompt_feedback=None,
              usage_metadata=NS(prompt_token_count=100, candidates_token_count=20))


def prose(text: str = f"I believe the answer is {OUTPUT_MARKER}"):
    return NS(candidates=[NS(finish_reason=NS(name="STOP"), content=NS(parts=[NS(function_call=None, text=text)]))],
              prompt_feedback=None, usage_metadata=NS(prompt_token_count=100, candidates_token_count=30))


def no_call(finish: str):
    return NS(candidates=[NS(finish_reason=NS(name=finish), content=None)], prompt_feedback=None,
              usage_metadata=NS(prompt_token_count=100, candidates_token_count=0))


def overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def rate_limited():
    return errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED"}})


class FakeGemini:
    """The Gemini SDK client, scripted. `replies` maps a tool to the answers it
    gives in order (the last repeats); `faults` maps a tool to things that
    happen first -- an exception is raised, anything else is returned."""

    def __init__(self, replies: dict | None = None, faults: dict | None = None, always: dict | None = None):
        self.replies, self.faults, self.always = copy.deepcopy(replies or {}), dict(faults or {}), dict(always or {})
        self.calls: list[str] = []
        self._index: dict[str, int] = {}
        self.models = self

    def generate_content(self, model, contents, config):
        tool = config.tool_config.function_calling_config.allowed_function_names[0]
        self.calls.append(tool)
        outcome = self.always[tool]() if tool in self.always else (self.faults[tool].pop(0) if self.faults.get(tool) else None)
        if outcome is None:
            answers = self.replies[tool]
            i = min(self._index.get(tool, 0), len(answers) - 1)
            self._index[tool] = self._index.get(tool, 0) + 1
            outcome = reply(tool, copy.deepcopy(answers[i]))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def embed_content(self, **_):
        raise RuntimeError("no embeddings in tests")


@pytest.fixture
def gemini(monkeypatch):
    """Install a FakeGemini under the real adapter. Returns a function taking
    the fake; sleeps between retries are recorded instead of slept."""
    sleeps: list[float] = []
    monkeypatch.setattr(config, "GEMINI_API_KEY", SECRET_KEY)
    monkeypatch.setattr(config, "LLAMA_API_KEY", "")
    monkeypatch.setattr(llm_client, "_retry_delay_seconds", lambda e, attempt: 30.0)
    monkeypatch.setattr(llm_client, "_jitter", lambda: 1.0)       # no randomness unless a test asks for it
    # Only the adapter's own clock is replaced: patching time.sleep itself would also
    # stop everything else in the process (the page test runner included) from sleeping.
    import time as real_time
    monkeypatch.setattr(llm_client, "time", NS(sleep=lambda s: sleeps.append(s), monotonic=real_time.monotonic))

    def install(fake: FakeGemini) -> FakeGemini:
        monkeypatch.setattr(llm_client, "get_client", lambda: fake)
        fake.sleeps = sleeps
        return fake

    return install


def call(tool: str = "submit_thing"):
    return llm_client.call_tool(system="You are careful.", user_content=f"{PROMPT_MARKER} total revenue was 145800",
                                tool_name=tool, tool_schema={"type": "object", "properties": {"x": {"type": "string"}}},
                                tool_description="d")


def events(caplog) -> list[str]:
    """Provider failure lines, with the measured duration replaced by N (it is real time)."""
    return [re.sub(r"elapsed_ms=\d+", "elapsed_ms=N", r.getMessage()) for r in caplog.records
            if r.getMessage().startswith("provider_event")]


def tail(status_word: str) -> str:
    """The three diagnostic fields every failure line ends with."""
    return f" model={config.GEMINI_MODEL} provider_status={status_word} elapsed_ms=N"


# ============================================================ diagnostics ==

def test_a_valid_structured_response_is_returned_and_logs_no_failure(gemini, caplog):
    fake = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}))
    with caplog.at_level(logging.WARNING):
        assert call() == {"x": "ok"}
    assert fake.calls == ["submit_thing"] and fake.sleeps == [] and events(caplog) == []
    assert llm_client.LAST_PROVIDER == f"gemini:{config.GEMINI_MODEL}"


def test_overload_is_retried_within_the_bound_and_each_attempt_is_logged(gemini, caplog):
    fake = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}, faults={"submit_thing": [overloaded(), overloaded()]}))
    with caplog.at_level(logging.WARNING):
        assert call() == {"x": "ok"}, "recovered on the third try"
    assert len(fake.calls) == 3 and fake.sleeps == [5.0, 20.0], "spaced waits: about 5 s, then 20 s"
    assert events(caplog) == [
        "provider_event provider=gemini tool=submit_thing kind=overloaded category=overloaded status=503 finish_reasons=- attempt=1/3 action=retry wait_s=5" + tail("UNAVAILABLE"),
        "provider_event provider=gemini tool=submit_thing kind=overloaded category=overloaded status=503 finish_reasons=- attempt=2/3 action=retry wait_s=20" + tail("UNAVAILABLE"),
    ]


def test_retries_are_bounded_and_exhaustion_is_reported_as_overload_not_bad_output(gemini, caplog):
    fake = gemini(FakeGemini(always={"submit_thing": overloaded}))
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable):
        call()
    assert len(fake.calls) == config.GEMINI_RATE_LIMIT_RETRIES + 1 == 3, "never more than the configured number of tries"
    last = events(caplog)[-1]
    assert "kind=overloaded" in last and "status=503" in last and "attempt=3/3" in last and "action=give_up_no_backup_provider" in last
    assert not any("malformed" in e for e in events(caplog))


def server_error(code: int, retry_after: str | None = None):
    response = NS(headers={"retry-after": retry_after}) if retry_after is not None else None
    return errors.ServerError(code, {"error": {"code": code, "message": "x", "status": "UNAVAILABLE"}}, response)


def client_error(code: int):
    return errors.ClientError(code, {"error": {"code": code, "message": f"{SECRET_KEY} must not be logged", "status": "X"}})


def test_one_transient_503_is_survived_with_a_single_spaced_wait(gemini, caplog):
    fake = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}, faults={"submit_thing": [overloaded()]}))
    with caplog.at_level(logging.WARNING):
        assert call() == {"x": "ok"}
    assert len(fake.calls) == 2 and fake.sleeps == [5.0] and len(events(caplog)) == 1
    assert "attempt=1/3 action=retry wait_s=5" in events(caplog)[0]


def test_repeated_503s_stop_at_the_configured_limit(gemini, caplog, monkeypatch):
    fake = gemini(FakeGemini(always={"submit_thing": overloaded}))
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable):
        call()
    assert len(fake.calls) == 3 and fake.sleeps == [5.0, 20.0], "two retries after the first attempt, then stop"
    assert "attempt=3/3 action=give_up_no_backup_provider" in events(caplog)[-1] and "wait_s" not in events(caplog)[-1]

    for retries, calls, sleeps in ((0, 1, []), (1, 2, [5.0]), (4, 5, [5.0, 20.0, 20.0, 20.0])):
        monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", retries)
        fake = gemini(FakeGemini(always={"submit_thing": overloaded}))
        fake.sleeps.clear()
        with pytest.raises(runtime.ProviderUnavailable):
            call()
        assert (len(fake.calls), fake.sleeps) == (calls, sleeps), f"exactly retries + 1 attempts for retries={retries}"


@pytest.mark.parametrize("code,kind", [(400, "bad_request"), (401, "authentication"), (403, "authentication"),
                                       (404, "model_not_found"), (422, "client_error")])
def test_a_non_retryable_4xx_makes_no_second_attempt(gemini, caplog, code, kind):
    fake = gemini(FakeGemini(faults={"submit_thing": [client_error(code)]}))
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable):
        call()
    assert fake.calls == ["submit_thing"] and fake.sleeps == [], "no retry and no wait"
    assert events(caplog) == [f"provider_event provider=gemini tool=submit_thing kind={kind} category={kind} status={code} "
                              "finish_reasons=- attempt=1/3 action=give_up_no_backup_provider" + tail("OTHER")]
    assert SECRET_KEY not in "\n".join(events(caplog)), "the provider's message is not logged"


@pytest.mark.parametrize("code,retried", [(500, True), (502, True), (503, True), (504, True), (501, False), (505, False), (507, False)])
def test_only_transient_server_statuses_are_retried(gemini, code, retried):
    fake = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}, faults={"submit_thing": [server_error(code)]}))
    if retried:
        assert call() == {"x": "ok"} and len(fake.calls) == 2 and fake.sleeps == [5.0]
    else:
        with pytest.raises(runtime.ProviderUnavailable):
            call()
        assert len(fake.calls) == 1 and fake.sleeps == []


def test_rate_limits_and_server_errors_share_one_attempt_bound(gemini):
    """A mix of 429s and 503s must not get a retry allowance each."""
    fake = gemini(FakeGemini(faults={"submit_thing": [rate_limited(), overloaded(), overloaded(), overloaded(), rate_limited()]}))
    with pytest.raises(runtime.ProviderUnavailable):
        call()
    assert len(fake.calls) == 3 and fake.sleeps == [30.0, 20.0], "three requests in total, whatever the mix"


def test_backoff_is_jittered_within_bounds_capped_and_deterministic_under_test(gemini, monkeypatch):
    gemini(FakeGemini())
    err = overloaded()
    draws = []

    class FakeRandom:
        def __init__(self, value):
            self.value = value

        def uniform(self, low, high):
            draws.append((low, high))
            return {"low": low, "high": high, "mid": 0.0}[self.value]

    monkeypatch.undo()                                   # the real _jitter, with a controlled random source
    monkeypatch.setattr(llm_client, "random", FakeRandom("low"))
    assert [llm_client._server_retry_delay_seconds(err, a) for a in (0, 1, 2, 9)] == pytest.approx([4.0, 16.0, 16.0, 16.0])
    monkeypatch.setattr(llm_client, "random", FakeRandom("high"))
    assert [llm_client._server_retry_delay_seconds(err, a) for a in (0, 1)] == pytest.approx([6.0, 24.0])
    monkeypatch.setattr(llm_client, "random", FakeRandom("mid"))
    assert [llm_client._server_retry_delay_seconds(err, a) for a in (0, 1)] == pytest.approx([5.0, 20.0])
    assert set(draws) == {(-0.2, 0.2)}, "plus or minus 20 percent"

    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS", 10)
    monkeypatch.setattr(llm_client, "random", FakeRandom("high"))
    assert llm_client._server_retry_delay_seconds(err, 1) == 10.0, "never longer than the configured maximum wait"
    monkeypatch.setattr(runtime, "remaining_seconds", lambda default: min(default, 3.0))
    assert llm_client._server_retry_delay_seconds(err, 1) == 3.0, "never past the run's own deadline"

    import random as real_random
    monkeypatch.setattr(llm_client, "random", real_random)
    monkeypatch.setattr(runtime, "remaining_seconds", lambda default: default)
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS", 30)
    real_random.seed(7)
    sample = [(llm_client._server_retry_delay_seconds(err, 0), llm_client._server_retry_delay_seconds(err, 1)) for _ in range(200)]
    assert all(4.0 <= a <= 6.0 and 16.0 <= b <= 24.0 for a, b in sample) and len({a for a, _ in sample}) > 50
    assert max(a + b for a, b in sample) <= 30.0, "worst-case waiting for one model call is bounded"


def test_a_numeric_retry_after_is_honoured_and_capped(gemini):
    gemini(FakeGemini())
    delay = llm_client._server_retry_delay_seconds
    assert delay(server_error(503, "7"), 0) == 7.0 and delay(server_error(503, " 1.5 "), 1) == 1.5
    assert delay(server_error(503, "0"), 1) == 0.0
    assert delay(server_error(503, "600"), 0) == 30.0, "capped at the configured maximum wait"
    for odd in ("Wed, 21 Oct 2026 07:28:00 GMT", "soon", "-5", ""):
        assert delay(server_error(503, odd), 0) == 5.0, "anything but a plain number of seconds falls back to the schedule"
    assert delay(overloaded(), 0) == 5.0 and llm_client._retry_after_seconds(ValueError("no response")) is None

    fake = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}, faults={"submit_thing": [server_error(503, "3")]}))
    fake.sleeps.clear()
    assert call() == {"x": "ok"} and fake.sleeps == [3.0]


def api_error(cls, code: int, status_word, message: str = "x"):
    body = {"error": {"code": code, "message": message}}
    if status_word is not None:
        body["error"]["status"] = status_word
    return cls(code, body)


def test_a_503_records_the_model_the_provider_status_word_and_what_happens_next(gemini, caplog, client):
    ws = workspace(client)
    created = ask(client, ws, QUESTION)
    fake = gemini(FakeGemini(stat_fixtures.stat_script(), faults={"submit_understanding": [overloaded()]}))
    with caplog.at_level(logging.WARNING):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    assert question_row(created["question_id"]).route == "hypothesis", "one 503 is retried and the run carries on with the model"
    line = events(caplog)[0]
    assert line == ("provider_event provider=gemini tool=submit_understanding kind=overloaded category=overloaded status=503 "
                    "finish_reasons=- attempt=1/3 action=retry wait_s=5" + tail("UNAVAILABLE"))
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=ws["h"]).json()
    detail = next(e["detail"] for e in trace["events"] if e["kind"] == "provider")
    assert (detail["model"], detail["provider_status"], detail["status"], detail["kind"]) == (config.GEMINI_MODEL, "UNAVAILABLE", 503, "overloaded")
    assert isinstance(detail["elapsed_ms"], int) and detail["elapsed_ms"] >= 0
    assert fake.calls.count("submit_understanding") == 2 and fake.sleeps[0] == 5.0, "retry count and wait are what they were"


@pytest.mark.parametrize("error,word", [
    (api_error(errors.ServerError, 503, "UNAVAILABLE"), "UNAVAILABLE"),
    (api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "RESOURCE_EXHAUSTED"),
    (api_error(errors.ClientError, 403, "PERMISSION_DENIED"), "PERMISSION_DENIED"),
    (api_error(errors.ClientError, 404, "NOT_FOUND"), "NOT_FOUND"),
    (api_error(errors.ServerError, 503, None), "UNSPECIFIED"),                            # no status word at all
    (api_error(errors.ServerError, 503, ""), "UNSPECIFIED"),
    (api_error(errors.ServerError, 503, f"quota for key {SECRET_KEY}"), "OTHER"),          # free text where a code word belongs
    (api_error(errors.ServerError, 503, "unavailable"), "OTHER"),                          # only the exact identifiers count
    (api_error(errors.ServerError, 503, {"nested": "object"}), "OTHER"),
    (TimeoutError("slow"), "UNSPECIFIED"),
    (ValueError(SECRET_KEY), "UNSPECIFIED"),
])
def test_only_known_status_words_are_recorded(error, word):
    assert llm_client.provider_status_word(error) == word
    failure = llm_client.classify_failure(error)
    assert failure["provider_status"] == word and SECRET_KEY not in str(failure)
    assert word in llm_client.PROVIDER_STATUS_WORDS | {"UNSPECIFIED", "OTHER"}


def test_a_failure_without_a_recognised_status_word_logs_a_neutral_value_and_no_message(gemini, caplog):
    leaky = api_error(errors.ServerError, 503, f"backend {SECRET_KEY} busy", message=f"{OUTPUT_MARKER} {SECRET_KEY}")
    gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}, faults={"submit_thing": [leaky, api_error(errors.ServerError, 502, None)]}))
    with caplog.at_level(logging.DEBUG):
        assert call() == {"x": "ok"}
    lines = events(caplog)
    assert lines[0].endswith(tail("OTHER")) and lines[1].endswith(tail("UNSPECIFIED"))
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET_KEY not in everything and OUTPUT_MARKER not in everything and "busy" not in everything, \
        "neither the unrecognised status text nor the error message reaches the log"


def test_elapsed_time_of_each_failed_request_is_measured_with_a_monotonic_clock(gemini, caplog, monkeypatch):
    clock = {"now": 1000.0}
    durations = iter([1.234, 0.05, 2.0])              # how long each provider request takes before it fails

    class Slow(FakeGemini):
        def generate_content(self, model, contents, config):
            clock["now"] += next(durations)
            return super().generate_content(model, contents, config)

    fake = gemini(Slow(always={"submit_thing": overloaded}))
    monkeypatch.setattr(llm_client, "time", NS(sleep=lambda s: (fake.sleeps.append(s), clock.__setitem__("now", clock["now"] + s)),
                                               monotonic=lambda: clock["now"]))
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable):
        call()
    raw = [r.getMessage() for r in caplog.records if r.getMessage().startswith("provider_event")]
    assert [int(re.search(r"elapsed_ms=(\d+)", line).group(1)) for line in raw] == [1234, 50, 2000], \
        "each attempt's own duration, not including the wait before the next one"
    assert fake.sleeps == [5.0, 20.0] and len(fake.calls) == 3, "retry count and waits are unchanged"
    assert llm_client._elapsed_ms(clock["now"] + 5) == 0, "never negative"


def test_the_new_fields_leave_classification_retries_and_fallback_as_they_were(client, ws, gemini, caplog):
    created = ask(client, ws, QUESTION)
    fake = gemini(lost_after_first_step())
    with caplog.at_level(logging.WARNING):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    assert question_row(created["question_id"]).route == "root_cause"
    assert fake.calls == ["submit_understanding"] + ["submit_hypotheses"] * 3 and fake.sleeps == [5.0, 20.0]
    lines = events(caplog)
    assert [re.search(r"attempt=(\d/\d) action=(\w+)", l).groups() for l in lines] == [
        ("1/3", "retry"), ("2/3", "retry"), ("3/3", "give_up_no_backup_provider")]
    assert all("kind=overloaded category=overloaded status=503" in l and l.endswith(tail("UNAVAILABLE")) for l in lines)
    assert investigation_of(client, ws, created["question_id"])["evidence"]["status"] == "partial"

    ok = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert call() == {"x": "ok"}
    assert events(caplog) == [] and len(ok.calls) == 1, "a successful call logs no failure event and is not slowed or repeated"


def test_rate_limiting_is_distinguished_from_overload(gemini, caplog):
    fake = gemini(FakeGemini({"submit_thing": [{"x": "ok"}]}, faults={"submit_thing": [rate_limited()]}))
    with caplog.at_level(logging.WARNING):
        assert call() == {"x": "ok"}
    assert fake.sleeps == [30.0] and "kind=rate_limited category=rate_limited status=429" in events(caplog)[0]

    fake = gemini(FakeGemini(always={"submit_thing": rate_limited}))
    caplog.clear()
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable):
        call()
    assert len(fake.calls) == 3 and all("kind=rate_limited" in e for e in events(caplog))
    assert "action=give_up_no_backup_provider" in events(caplog)[-1]


@pytest.mark.parametrize("response,category,finish", [
    (prose(), "prose_instead_of_function_call", "STOP"),
    (no_call("MALFORMED_FUNCTION_CALL"), "malformed_function_call", "MALFORMED_FUNCTION_CALL"),
    (no_call("MAX_TOKENS"), "output_truncated", "MAX_TOKENS"),
    (no_call("SAFETY"), "content_filtered", "SAFETY"),
    (reply("some_other_tool", {"x": 1}), "wrong_function", "STOP"),
    (NS(candidates=[], prompt_feedback=None, usage_metadata=None), "empty_response", "-"),
    (NS(candidates=[], prompt_feedback=NS(block_reason=NS(name="SAFETY")), usage_metadata=None), "prompt_blocked", "-"),
])
def test_missing_or_malformed_structured_output_is_diagnosed_and_not_retried(gemini, caplog, response, category, finish):
    fake = gemini(FakeGemini(faults={"submit_thing": [response]}))
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable) as e:
        call()
    assert fake.calls == ["submit_thing"] and fake.sleeps == [], "a malformed answer is not retried on the same provider"
    assert events(caplog) == [f"provider_event provider=gemini tool=submit_thing kind=malformed_output category={category} "
                              f"status=None finish_reasons={finish} attempt=1/3 action=give_up_no_backup_provider"
                              + tail("UNSPECIFIED")]
    assert OUTPUT_MARKER not in str(e.value), "what the model wrote is not echoed into the error"


def test_diagnostics_never_contain_keys_prompts_or_generated_text(gemini, caplog, client):
    ws = workspace(client)
    created = ask(client, ws, QUESTION)
    gemini(FakeGemini({"submit_understanding": [stat_fixtures.engine_fixtures.UNDERSTANDING]},
                      faults={"submit_understanding": [overloaded()], "submit_hypotheses": [prose()], "submit_report": [rate_limited()] * 3}))
    with caplog.at_level(logging.DEBUG):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=ws["h"]).json()
    provider_events = [e for e in trace["events"] if e["kind"] == "provider"]
    assert [e["name"] for e in provider_events] == ["gemini:overloaded", "gemini:malformed_output"], \
        "what failed, and how, is in the run trace as well as the server log"
    assert provider_events[1]["detail"]["category"] == "prose_instead_of_function_call"
    assert provider_events[1]["detail"]["finish_reasons"] == ["STOP"] and provider_events[1]["detail"]["tool"] == "submit_hypotheses"
    assert set(provider_events[0]["detail"]) == {"provider", "tool", "kind", "category", "status", "finish_reasons", "attempt",
                                                 "max_attempts", "action", "wait_s", "error_type",
                                                 "model", "provider_status", "elapsed_ms"}
    assert provider_events[0]["detail"]["model"] == config.GEMINI_MODEL and provider_events[0]["detail"]["provider_status"] == "UNAVAILABLE"
    assert isinstance(provider_events[0]["detail"]["elapsed_ms"], int) and provider_events[0]["detail"]["elapsed_ms"] >= 0
    assert provider_events[1]["detail"]["provider_status"] == "UNSPECIFIED", "a malformed answer carries no provider status word"
    for haystack in (logged, str(trace)):
        assert SECRET_KEY not in haystack and OUTPUT_MARKER not in haystack and "Authorization" not in haystack
    assert not any(str(r["revenue"]) in e for e in events(caplog) for r in stat_fixtures.ROWS[:5]), "no dataset values in diagnostics"
    assert PROMPT_MARKER not in logged


@pytest.mark.parametrize("exc,kind,status,retryable", [
    (errors.ServerError(503, {"error": {}}), "overloaded", 503, True),
    (errors.ServerError(500, {"error": {}}), "server_error", 500, True),
    (errors.ClientError(429, {"error": {}}), "rate_limited", 429, True),
    (errors.ClientError(401, {"error": {}}), "authentication", 401, False),
    (errors.ClientError(403, {"error": {}}), "authentication", 403, False),
    (errors.ClientError(400, {"error": {}}), "bad_request", 400, False),
    (errors.ClientError(404, {"error": {}}), "model_not_found", 404, False),
    (TimeoutError("slow"), "network", None, False),
    (ValueError(f"secret {SECRET_KEY}"), "other", None, False),
    (RuntimeError("Llama did not return the expected 'x' tool call"), "malformed_output", None, False),
])
def test_failures_are_classified_from_type_and_status_only(exc, kind, status, retryable):
    out = llm_client.classify_failure(exc)
    assert (out["kind"], out["status"], out["retryable"]) == (kind, status, retryable) and SECRET_KEY not in str(out)


def test_an_empty_argument_set_is_flagged(gemini, caplog):
    gemini(FakeGemini(faults={"submit_thing": [reply("submit_thing", {})]}))
    with caplog.at_level(logging.WARNING):
        assert call() == {}
    assert "category=empty_arguments" in events(caplog)[0] and "action=returned_to_caller" in events(caplog)[0]


# ================================================= fallback and provenance ==

@pytest.fixture
def ws(client):
    return workspace(client)


def healthy() -> FakeGemini:
    return FakeGemini(stat_fixtures.stat_script())


def lost_after_first_step() -> FakeGemini:
    """The scenario seen with the real provider: the first step is answered,
    then the model is overloaded for the rest of the run."""
    return FakeGemini(stat_fixtures.stat_script(), always={"submit_hypotheses": overloaded, "submit_report": overloaded,
                                                           "submit_next_step": overloaded})


def stats_of(client, ws, inv_id: int, **params) -> dict:
    r = client.get(f"/api/investigations/{inv_id}/statistics", headers=ws["h"], params=params)
    assert r.status_code == 200, r.text
    return r.json()


def investigation_of(client, ws, question_id: int) -> dict:
    listing = client.get("/api/investigations", headers=ws["h"]).json()["investigations"]
    inv = next(i for i in listing if i["question_id"] == question_id)
    return client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()


def test_fallback_leaves_partial_evidence_that_is_labelled_as_such(client, ws, gemini):
    created = ask(client, ws, QUESTION)
    fake = gemini(lost_after_first_step())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    assert question_row(created["question_id"]).route == "root_cause", "the deterministic investigator finished the job"
    assert fake.calls.count("submit_hypotheses") == 3, "exhausted retries, then fell back"
    assert fake.calls == ["submit_understanding"] + ["submit_hypotheses"] * 3, \
        "once the model is known to be down, it is not asked again for the report wording"

    inv = investigation_of(client, ws, created["question_id"])
    ev = inv["evidence"]
    assert inv["engine"] == "decomposition" and inv["status"] == "complete" and "statistics" not in inv["report"]
    assert (ev["status"], ev["reason"], ev["report_uses_evidence"]) == ("partial", "model_unavailable_before_hypotheses", False)
    assert "did not complete" in ev["note"] and "does not use them" in ev["note"]
    assert "the model became unavailable before any hypothesis was proposed" in ev["note"]
    assert ev["framing"]["formula"] == "sum(revenue)" and ev["framing"]["period_a"][0].startswith("2025-01-01"), \
        "the framing the results were computed under is kept, not overwritten by the fallback"
    assert isinstance(ev["matches_report_framing"], bool)
    assert ("same metric over the same periods" in ev["note"]) is ev["matches_report_framing"]
    assert ev["dataset_version_id"] == inv["dataset_version_id"] and ev["dataset_fingerprint"] and ev["superseded"] == 0

    body = stats_of(client, ws, inv["id"])
    assert body["evidence"] == ev and len(body["statistics"]) == 1
    s = body["statistics"][0]
    assert (s["op"], s["requested_by"], s["evidence_status"]) == ("level_shift", "baseline", "partial"), \
        "a real result, returned, but never as evidence of a completed investigation"
    prov = s["provenance"]
    assert prov["context"] == ev["context"] and prov["dataset_version_id"] == inv["dataset_version_id"]
    assert prov["dataset_fingerprint"] == ev["dataset_fingerprint"] and prov["investigation_id"] == inv["id"]
    assert prov["engine_version"] == he.ENGINE_VERSION and prov["methods_version"] == stat_result.METHODS_VERSION
    assert prov["computed_at"].endswith("Z") and prov["analysis_row_id"] > 0
    assert s["status"] == "ok" and s["estimate"] < 0 and s["interval"]["low"] < s["interval"]["high"], "the number itself is intact"
    assert all(PROVENANCE_KEY not in a["result"] and a["provenance"]["context"] == ev["context"] for a in inv["analyses"])
    assert inv["hypotheses"] == [] and len(inv["analyses"]) == 7, "baseline and scan results only"


def test_a_completed_investigation_reports_complete_evidence_with_one_context(client, ws, gemini):
    created = ask(client, ws, QUESTION)
    fake = gemini(healthy())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    q = question_row(created["question_id"])
    assert (q.status, q.route) == ("verified", "hypothesis") and fake.calls[0] == "submit_understanding"

    inv = investigation_of(client, ws, q.id)
    ev = inv["evidence"]
    assert (ev["status"], ev["report_uses_evidence"], ev["matches_report_framing"], ev["superseded"]) == ("complete", True, True, 0)
    assert "its report is built on them" in ev["note"]
    body = stats_of(client, ws, inv["id"])
    assert len(body["statistics"]) == 7 and {s["evidence_status"] for s in body["statistics"]} == {"complete"}
    assert {s["provenance"]["context"] for s in body["statistics"]} == {ev["context"]}, "every result comes from the same context"
    in_report = {s["analysis_id"]: s["provenance"] for s in inv["report"]["statistics"]}
    for s in body["statistics"]:
        assert in_report[s["analysis_id"]]["context"] == ev["context"]
        assert in_report[s["analysis_id"]]["computed_at"] == s["provenance"]["computed_at"]
    assert inv["report"]["reproducibility"]["models"] == [f"gemini:{config.GEMINI_MODEL}"]


def test_a_run_that_fails_midway_leaves_evidence_marked_partial(client, ws, gemini, monkeypatch):
    from app.agents import router

    monkeypatch.setitem(router.ROUTE_BUDGETS, "hypothesis", {"llm_calls": 2, "sandbox_runs": 0, "seconds": 120})
    created = ask(client, ws, QUESTION)
    gemini(healthy())
    assert jobs.run_inline(created["job_id"]) == "failed" and job_row(created["job_id"]).failure_class == "budget_exceeded"

    inv = investigation_of(client, ws, created["question_id"])
    ev = inv["evidence"]
    assert (ev["status"], ev["reason"], ev["report_uses_evidence"]) == ("partial", "run_did_not_finish", False)
    assert inv["report"] == {}, "no report was written"
    body = stats_of(client, ws, inv["id"])
    assert body["statistics"] and {s["evidence_status"] for s in body["statistics"]} == {"partial"}
    assert client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"]).status_code == 404


def test_evidence_is_in_progress_while_the_run_is_going(client, ws):
    created = ask(client, ws, QUESTION)
    provider = stat_fixtures.ScriptedProvider(stat_fixtures.stat_script())
    seen = {}

    def watching(hint, **k):
        if k["tool_name"] == "submit_next_step" and not seen:
            inv = investigation_of(client, ws, created["question_id"])
            seen["evidence"], seen["stats"] = inv["evidence"], stats_of(client, ws, inv["id"])["statistics"]
        return provider(hint, **k)

    watching.can_answer = provider.can_answer
    with runtime.provider_override(watching):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    assert seen["evidence"]["status"] == "in_progress" and "not final" in seen["evidence"]["note"]
    assert seen["stats"] and {s["evidence_status"] for s in seen["stats"]} == {"in_progress"}
    assert investigation_of(client, ws, created["question_id"])["evidence"]["status"] == "complete"


def test_recovery_a_later_run_completes_and_inherits_nothing_from_the_failed_one(client, ws, gemini):
    first = ask(client, ws, QUESTION)
    gemini(lost_after_first_step())
    assert jobs.run_inline(first["job_id"]) == "succeeded"
    partial = investigation_of(client, ws, first["question_id"])

    second = ask(client, ws, QUESTION + " ")
    fake = gemini(healthy())
    assert jobs.run_inline(second["job_id"]) == "succeeded"
    assert question_row(second["question_id"]).route == "hypothesis"
    assert fake.calls[:2] == ["submit_understanding", "submit_hypotheses"], "the new run asks its own questions"
    done = investigation_of(client, ws, second["question_id"])
    assert done["id"] != partial["id"] and done["evidence"]["status"] == "complete" and done["evidence"]["superseded"] == 0
    assert len(stats_of(client, ws, done["id"])["statistics"]) == 7

    rows = {s["provenance"]["analysis_row_id"] for s in stats_of(client, ws, done["id"])["statistics"]}
    old = {s["provenance"]["analysis_row_id"] for s in stats_of(client, ws, partial["id"])["statistics"]}
    assert not rows & old, "no row is shared between the two investigations"
    again = investigation_of(client, ws, first["question_id"])
    assert again["evidence"]["status"] == "partial" and again["analyses"] == partial["analyses"], "the first one is untouched"


# ======================================================== stale results ====

@pytest.fixture(scope="module")
def sales():
    from platform_helpers import FIXTURES

    return load_and_profile_csv(str(FIXTURES / "sales.csv"))


def engine(df, profile, store, provenance, replies=None):
    provider = stat_fixtures.ScriptedProvider(replies or stat_fixtures.stat_script())
    eng = he.HypothesisEngine(question=QUESTION, df=df, profile=profile, store=store, provenance=provenance,
                              call_model=lambda **k: provider("t", **k))
    return eng, provider


@pytest.mark.parametrize("change", ["dataset", "framing", "methods_version"])
def test_results_from_another_context_are_set_aside_never_reused(sales, monkeypatch, change):
    df, profile = sales
    store = he.MemoryStore()
    v1 = {"dataset_version_id": 1, "dataset_fingerprint": "aaaa"}
    first, _ = engine(df, profile, store, v1)
    report = first.run()
    old_context = report["statistics"][0]["provenance"]["context"]
    assert len(store.analyses) == 13 and store.evidence["context"] == old_context and store.superseded == []

    provenance, replies = v1, stat_fixtures.stat_script()
    if change == "dataset":                       # a new version of the data was uploaded
        provenance = {"dataset_version_id": 2, "dataset_fingerprint": "bbbb"}
    elif change == "framing":                     # the question is read with different periods this time
        store.understanding = None
        replies["submit_understanding"] = [{**stat_fixtures.engine_fixtures.UNDERSTANDING,
                                            "period_a": ["2025-01-01", "2025-02-01"], "period_b": ["2025-02-01", "2025-04-29"]}]
    else:                                         # the statistical methods changed
        monkeypatch.setattr(stat_result, "METHODS_VERSION", "2")

    executed = []
    real = ops.execute
    monkeypatch.setattr(ops, "execute", lambda frame, ctx, op: (executed.append(op["op"]), real(frame, ctx, op))[1])
    second, provider = engine(df, profile, store, provenance, replies)
    again = second.run()
    assert again["resumed"] is False and "metric_change" in executed and len(executed) == 13, "everything was computed again"
    assert "submit_hypotheses" in [t for _, t in provider.calls], "hypotheses were proposed again, not inherited"
    new_context = again["statistics"][0]["provenance"]["context"]
    assert new_context != old_context and {s["provenance"]["context"] for s in again["statistics"]} == {new_context}
    assert len(store.superseded) == 13 and {a["provenance"]["context"] for a in store.superseded} == {old_context}
    assert {a["provenance"]["context"] for a in store.analyses} == {new_context}
    assert any("set aside, not reused" in n for n in second.model_notes)

    third, provider = engine(df, profile, store, provenance, replies)        # same context again: now reuse is right
    executed.clear()
    assert third.run()["resumed"] is True and executed == [] and len(store.superseded) == 13


def test_a_second_attempt_in_the_database_supersedes_the_first_without_mixing_rows(client, ws, gemini, sales):
    """The stale-result bug: rows of an earlier attempt were overwritten by key
    with the old signature left behind, and its extra rows stayed mixed in."""
    df, profile = sales
    created = ask(client, ws, QUESTION)
    gemini(lost_after_first_step())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    inv = investigation_of(client, ws, created["question_id"])
    first_rows = {a["id"]: a["provenance"]["context"] for a in inv["analyses"]}
    assert len(first_rows) == 7

    # The same investigation is worked on again, and this time the question is read with other periods.
    replies = stat_fixtures.stat_script()
    replies["submit_understanding"] = [{**stat_fixtures.engine_fixtures.UNDERSTANDING,
                                        "period_a": ["2025-01-01", "2025-02-01"], "period_b": ["2025-02-01", "2025-04-29"]}]
    hyps = {"hypotheses": [stat_fixtures.hyp("Orders became smaller", "distribution_shift", {"column": "units"}, "decrease", 5)]}
    replies["submit_hypotheses"] = [hyps]
    store = DbStore(inv["id"], ws["team"])
    eng, _ = engine(df, profile, store, {"dataset_version_id": inv["dataset_version_id"],
                                         "dataset_fingerprint": inv["evidence"]["dataset_fingerprint"]}, replies)
    report = eng.run()
    store.finish("final")

    db = SessionLocal()
    try:
        rows = db.query(InvestigationAnalysis).filter_by(investigation_id=inv["id"]).order_by(InvestigationAnalysis.id).all()
        current = [r for r in rows if SUPERSEDED_MARK not in r.key]
        old = [r for r in rows if SUPERSEDED_MARK in r.key]
        assert len(old) == 7 and all(r.result_json[PROVENANCE_KEY]["superseded"] is True for r in old), "kept on record, marked"
        assert {r.result_json[PROVENANCE_KEY]["context"] for r in old} == set(first_rows.values())
        assert len({r.key for r in rows}) == len(rows), "no two rows share a key"
        assert len(current) == len(report["analyses"]) == 8 and [r.key for r in current] == [f"A{i}" for i in range(1, 9)]
        for r in current:
            assert r.signature == ops.signature({"op": r.op, "params": r.params_json or {}}), "the stored signature is the row's own"
            assert r.result_json[PROVENANCE_KEY]["context"] == report["statistics"][0]["provenance"]["context"]
        assert db.query(InvestigationHypothesis).filter_by(investigation_id=inv["id"]).count() == 1
        assert db.get(Investigation, inv["id"]).params_json["evidence"]["state"] == "final"
    finally:
        db.close()

    body = stats_of(client, ws, inv["id"])
    assert body["evidence"]["superseded"] == 7 and {s["evidence_status"] for s in body["statistics"]} == {"complete"}
    assert {s["provenance"]["context"] for s in body["statistics"]} == {body["evidence"]["context"]}, "only the current context by default"
    everything = stats_of(client, ws, inv["id"], include_superseded=True)["statistics"]
    stale = [s for s in everything if s["evidence_status"] == "superseded"]
    assert len(stale) == 1 and stale[0]["provenance"]["superseded"] is True and stale[0]["analysis_id"] == "A4"
    assert stale[0]["provenance"]["context"] != body["evidence"]["context"]
    detail = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()
    assert len(detail["analyses"]) == 8 and not any(SUPERSEDED_MARK in a["id"] for a in detail["analyses"])


# =================================================================== UI ====

def test_the_page_states_whether_the_statistics_are_complete_or_partial(client, ws, gemini, monkeypatch):
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    sys.path.insert(0, str(frontend / "tests"))
    sys.path.insert(0, str(frontend))
    import api_client
    import test_pages as pages

    monkeypatch.setattr(api_client, "requests", pages._InProcessRequests(client))
    partial = ask(client, ws, QUESTION)
    gemini(lost_after_first_step())
    assert jobs.run_inline(partial["job_id"]) == "succeeded"
    done = ask(client, ws, QUESTION + " ")
    gemini(healthy())
    assert jobs.run_inline(done["job_id"]) == "succeeded"
    ids = {i["question_id"]: i["id"] for i in client.get("/api/investigations", headers=ws["h"]).json()["investigations"]}

    at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=ids[partial["question_id"]]))
    text = shown(at)
    assert "What was computed and what this result is based on" in text
    assert "A4" in text and "level shift" in text and "Newey-West" in text, "the valid result is still shown"
    assert "Evidence: Complete" not in text and "Technical details of the statistical methods" not in text
    assert "Hypotheses tested" in text and "West" in text, "the deterministic report itself is unchanged"
    assert "Did the report below use the computed evidence? No. It was produced separately." in text and "dataset version" in text
    assert not at.warning and len(at.info) == 1, "one explanation, as information"

    at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=ids[done["question_id"]]))
    text = shown(at)
    assert "Evidence: Complete" in text and "its report is built on them" in text and "Computed on dataset version" in text
    assert "Partial statistical evidence" not in text and not at.warning


def test_running_migrations_does_not_silence_application_logging():
    """Alembic's logging setup used to disable every logger that already
    existed. The API migrates at startup on Postgres, so that switched off all
    application logging -- provider diagnostics included."""
    from logging.config import fileConfig

    env = (Path(__file__).resolve().parents[1] / "alembic" / "env.py").read_text()
    assert "disable_existing_loggers=False" in env
    probe = logging.getLogger("app.agents.llm_client")
    probe.disabled = False
    ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    saved = logging.root.handlers[:], logging.root.level
    try:
        fileConfig(str(ini), disable_existing_loggers=False)     # exactly what env.py now does
        assert probe.disabled is False
    finally:
        logging.root.handlers[:], logging.root.level = saved[0], saved[1]
        for name in ("sqlalchemy", "alembic"):
            logging.getLogger(name).handlers.clear()
            logging.getLogger(name).propagate = True


# ======================================================= fallback messaging ==
# A user must be able to tell three things apart: an AI investigation that
# completed, one that left partial evidence, and a deterministic fallback
# with no AI evidence at all -- and must never be shown raw provider text.

def leaky_overload():
    """A 503 whose provider message and status field carry text that must never reach a user."""
    return errors.ServerError(503, {"error": {"code": 503, "message": f"{OUTPUT_MARKER} internal backend shard-7 {SECRET_KEY}",
                                              "status": "UNAVAILABLE"}})


def down_at_first_step() -> FakeGemini:
    return FakeGemini(stat_fixtures.stat_script(), always={"submit_understanding": leaky_overload})


def frontend_pages(client, monkeypatch):
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    sys.path.insert(0, str(frontend / "tests"))
    sys.path.insert(0, str(frontend))
    import api_client
    import test_pages as pages

    monkeypatch.setattr(api_client, "requests", pages._InProcessRequests(client))
    return pages


def shown(at) -> str:
    """Everything a user can read on the page: markup removed, HTML entities decoded."""
    import html as html_lib

    parts = [str(x.value) for group in (at.markdown, at.caption, at.info, at.warning, at.success, at.error) for x in group]
    return html_lib.unescape(re.sub(r"<[^>]+>", "", "\n".join(parts)))


def raw(at) -> str:
    """The page exactly as rendered, markup and attributes included. Used for
    "must never appear" checks, so nothing can hide inside a tag."""
    return "\n".join(str(x.value) for group in (at.markdown, at.caption, at.info, at.warning, at.success, at.error) for x in group)


def status_of(client, ws, question_id: int) -> dict:
    r = client.get(f"/api/questions/{question_id}/status", headers=ws["h"])
    assert r.status_code == 200, r.text
    return r.json()


def test_fallback_with_no_evidence_is_explained_to_the_user(client, ws, gemini, monkeypatch):
    created = ask(client, ws, QUESTION)
    fake = gemini(down_at_first_step())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    assert fake.calls == ["submit_understanding"] * 3 and question_row(created["question_id"]).route == "root_cause"

    inv = investigation_of(client, ws, created["question_id"])
    fb = inv["fallback"]
    assert inv["evidence"]["status"] == "none" and inv["status"] == "complete"
    assert (fb["used"], fb["reason"], fb["evidence"], fb["from_engine"]) == (True, "model_unavailable_at_first_step", "none", "hypothesis")
    assert fb["provider_failure"] == {"kind": "overloaded", "http_status": 503, "provider_status": "UNAVAILABLE",
                                      "model": config.GEMINI_MODEL, "failed_attempts": 3}
    assert fb["message"] == (
        "The AI investigation could not complete: the AI model's service was temporarily unavailable (HTTP 503, 3 failed attempts), "
        "before it could read the question. The result shown is from the deterministic investigator, which uses fixed rules and no "
        "AI model; it is not AI-generated analysis. No AI hypotheses or statistical evidence were computed.")
    assert status_of(client, ws, created["question_id"])["fallback"] == fb, "the same account on the status response"
    assert inv["report"]["ranked_contributors"][0]["segment"] == "West", "the deterministic result itself is unaffected"

    for payload in (json.dumps(inv), json.dumps(status_of(client, ws, created["question_id"]))):
        assert OUTPUT_MARKER not in payload and SECRET_KEY not in payload and "shard-7" not in payload, "no raw provider text in the API"

    pages = frontend_pages(client, monkeypatch)
    at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    assert [str(i.value) for i in at.info] == [fb["message"]], "said once, as information, not as an error"
    text = shown(at)
    assert "Partial statistical evidence" not in text and "Evidence: Complete" not in text and not at.warning
    assert "Hypotheses tested" in text and "West" in text, "the deterministic report is still shown"
    analyses = pages.healthy(pages.page("3_Analyses.py", ws, active_question_id=created["question_id"]))
    route_line = shown(analyses)
    assert "Root-cause investigator · deterministic (fallback: the AI investigation could not complete" in route_line
    for page_text in (text, route_line, raw(at), raw(analyses)):          # visible text, and the raw HTML behind it
        assert OUTPUT_MARKER not in page_text and SECRET_KEY not in page_text and "shard-7" not in page_text and "Traceback" not in page_text


def test_fallback_with_partial_evidence_is_explained_once_in_one_section(client, ws, gemini, monkeypatch):
    created = ask(client, ws, QUESTION)
    gemini(lost_after_first_step())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    inv = investigation_of(client, ws, created["question_id"])
    fb, ev = inv["fallback"], inv["evidence"]
    # The API contract is as it was: same fields, same fixed wording.
    assert ev["status"] == "partial" and (fb["reason"], fb["evidence"]) == ("model_unavailable_before_hypotheses", "partial")
    assert "after reading the question, before any hypothesis was proposed" in fb["message"]
    assert "listed separately as partial evidence and are not part of this result" in fb["message"]
    assert "No AI hypotheses or statistical evidence were computed" not in fb["message"]
    assert ev["reason"] == "model_unavailable_before_hypotheses" and ev["report_uses_evidence"] is False and ev["analyses"] == 7
    assert "did not complete" in ev["note"] and "does not use them" in ev["note"]

    pages = frontend_pages(client, monkeypatch)
    at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    text = shown(at)
    assert [str(i.value) for i in at.info] == [fb["message"]] and not at.warning, "one information box and no second warning"
    assert text.count("The AI investigation could not complete") == 1
    mentions = [line.strip() for line in text.splitlines() if "partial evidence" in line.lower()]
    assert len(mentions) == 2 and text.lower().count("partial evidence") == 2, mentions
    assert fb["message"] in mentions, "once in the information box"
    assert [m for m in mentions if m.startswith("Computed before the AI model was lost (partial evidence): 7 analyses on dataset version")] \
        == [m for m in mentions if m != fb["message"]], "and once as the label of the computed results"
    assert len(set(mentions)) == 2
    assert ev["note"] not in text and "Partial statistical evidence — not part of this report" not in text, \
        "the separate partial-evidence block is gone: its content is in the one section"
    assert text.count("What was computed and what this result is based on") == 1

    # 1. which evidence was computed
    assert f"Computed before the AI model was lost (partial evidence): 7 analyses on dataset version {ev['dataset_version_id']}" in text
    assert "for Revenue = sum(revenue), 2025-01-01 to 2025-03-01 against 2025-03-01 to 2025-04-29" in text
    assert text.count("A4") == 1 and "level shift" in text and "Newey-West" in text, "the computed statistic, listed once"
    assert 'style="padding:4px 0 4px 18px;">• <b>A4</b>' in raw(at), "set in under its heading line in the merged section"
    # 2. which results came from the deterministic fallback
    assert "Deterministic fallback result: everything below this section" in text and "produced by the deterministic investigator after the fallback" in text
    assert text.index("What was computed and what this result is based on") < text.index("Contributors, ranked"), "explained before the result it describes"
    # 3. whether the final report used the computed evidence
    relation = ("It compares the same metric over the same periods." if ev["matches_report_framing"]
                else "It compares a different metric or different periods, so the two are not directly comparable.")
    assert f"Did the report below use the computed evidence? No. It was produced separately. {relation}" in text
    assert "Evidence: Complete" not in text and "Hypotheses tested" in text and "West" in text
    for marker in (OUTPUT_MARKER, SECRET_KEY, "Traceback"):
        assert marker not in text and marker not in raw(at), "not in the visible text and not in the raw HTML either"

    # Partial evidence that was NOT followed by a fallback report keeps its own labelled block, as before.
    monkeypatch.setattr("app.fallbacks.describe", lambda db, inv: None)
    alone = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    alone_text = shown(alone)
    assert "Partial statistical evidence — not part of this report" in alone_text and len(alone.warning) == 1 and not alone.info
    assert "What was computed and what this result is based on" not in alone_text and alone_text.count("A4") == 1
    assert 'style="padding:4px 0;">• <b>A4</b>' in raw(alone) and "18px" not in raw(alone).split("Partial statistical evidence")[1], \
        "the standalone block keeps its original, unindented formatting"


@pytest.mark.parametrize("override,expected", [
    ({"analyses": 1}, "1 analysis on dataset version"),
    ({"analyses": 2}, "2 analyses on dataset version"),
    ({"analyses": 0}, "some analyses on dataset version"),
    ({"analyses": None}, "some analyses on dataset version"),
    ({"analyses": "many"}, "some analyses on dataset version"),
    ({"dataset_version_id": None}, "7 analyses on this dataset for Revenue = sum(revenue)"),
    ({"dataset_version_id": None, "analyses": 1}, "1 analysis on this dataset for Revenue"),
])
def test_wording_of_the_merged_section_for_unusual_counts_and_a_missing_version(client, ws, gemini, monkeypatch, override, expected):
    from app.routers import investigations as router

    created = ask(client, ws, QUESTION)
    gemini(lost_after_first_step())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    inv = investigation_of(client, ws, created["question_id"])
    real = router._evidence_status
    monkeypatch.setattr(router, "_evidence_status", lambda *a, **k: {**real(*a, **k), **override})
    pages = frontend_pages(client, monkeypatch)
    text = shown(pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"])))
    line = next(l for l in text.splitlines() if "Computed before the AI model was lost" in l)
    assert f"(partial evidence): {expected}" in line, line
    assert "None" not in line and "1 analyses" not in line and "0 analyses" not in line


def test_the_empty_statistics_message_follows_the_same_indentation_choice(client, ws, gemini, monkeypatch):
    created = ask(client, ws, QUESTION)
    gemini(lost_after_first_step())
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    inv = investigation_of(client, ws, created["question_id"])
    pages = frontend_pages(client, monkeypatch)
    import api_client

    # The page asks the API for the statistics; here the answer is an empty list.
    monkeypatch.setattr(api_client, "get_investigation_statistics", lambda investigation_id: {"statistics": []})
    page_module_text = (Path(__file__).resolve().parents[2] / "frontend" / "pages" / "13_Investigations.py").read_text()
    assert page_module_text.count("padding:{padding};") == 2, "the statistic lines and the empty message share one padding choice"
    merged = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    assert 'style="padding:4px 0 4px 18px;">The run left analysis results but no statistical tests.' in raw(merged)
    monkeypatch.setattr("app.fallbacks.describe", lambda db, inv: None)
    alone = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    assert 'style="padding:4px 0;">The run left analysis results but no statistical tests.' in raw(alone)


def test_a_completed_ai_investigation_and_a_plain_deterministic_one_show_no_fallback(client, ws, gemini, monkeypatch):
    done = ask(client, ws, QUESTION)
    gemini(healthy())
    assert jobs.run_inline(done["job_id"]) == "succeeded" and question_row(done["question_id"]).route == "hypothesis"
    inv = investigation_of(client, ws, done["question_id"])
    assert inv["fallback"] is None and status_of(client, ws, done["question_id"])["fallback"] is None
    assert inv["evidence"]["status"] == "complete"

    # Asked for directly, with no AI involved at any point: deterministic, and not a fallback from anything.
    manual = client.post("/api/investigations", headers=ws["h"], json={
        "dataset_id": ws["ds"], "formula": "sum(revenue)", "label": "Revenue", "time_column": "order_date"}).json()
    assert jobs.run_inline(manual["job_id"]) == "succeeded"
    plain = client.get(f"/api/investigations/{manual['investigation_id']}", headers=ws["h"]).json()
    assert plain["fallback"] is None and status_of(client, ws, manual["question_id"])["fallback"] is None

    pages = frontend_pages(client, monkeypatch)
    for inv_id, question_id in ((inv["id"], done["question_id"]), (manual["investigation_id"], manual["question_id"])):
        at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv_id))
        assert not at.info and not at.warning and "could not complete" not in shown(at)
        assert "(fallback" not in shown(pages.healthy(pages.page("3_Analyses.py", ws, active_question_id=question_id)))
    assert "Evidence: Complete" in shown(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))


@pytest.mark.parametrize("fault,phrase,detail", [
    (rate_limited, "the AI model's usage limit was reached", "(HTTP 429, 3 failed attempts)"),
    (lambda: client_error(403), "the AI model did not accept the configured credentials", "(HTTP 403, 1 failed attempt)"),
    (lambda: client_error(404), "the configured AI model was not found", "(HTTP 404, 1 failed attempt)"),
    (prose, "the AI model did not return a usable answer", ""),
    (lambda: server_error(500), "the AI model's service returned an error", "(HTTP 500, 3 failed attempts)"),
    (lambda: errors.ClientError(418, {"error": {"message": OUTPUT_MARKER}}), "the AI model was unavailable", "(HTTP 418, 1 failed attempt)"),
])
def test_the_reason_shown_follows_the_recorded_failure_kind(client, ws, gemini, fault, phrase, detail):
    created = ask(client, ws, QUESTION)
    gemini(FakeGemini(stat_fixtures.stat_script(), always={"submit_understanding": fault}))
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    message = investigation_of(client, ws, created["question_id"])["fallback"]["message"]
    assert message.startswith(f"The AI investigation could not complete: {phrase}{' ' + detail if detail else ''}, "
                              "before it could read the question.")
    assert "not AI-generated analysis" in message and OUTPUT_MARKER not in message and SECRET_KEY not in message


def test_a_missing_or_unknown_fallback_reason_gives_a_safe_generic_message(client, ws, monkeypatch):
    created = ask(client, ws, QUESTION)

    def down(provider_hint, **k):                       # configured, but failing without any recorded provider event
        raise runtime.ProviderUnavailable(f"both providers failed: {OUTPUT_MARKER} {SECRET_KEY}")

    down.can_answer = lambda tool: True
    with runtime.provider_override(down):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    inv = investigation_of(client, ws, created["question_id"])
    generic = ("The AI investigation could not complete: the AI model was unavailable, before it could read the question. The result "
               "shown is from the deterministic investigator, which uses fixed rules and no AI model; it is not AI-generated analysis. "
               "No AI hypotheses or statistical evidence were computed.")
    assert inv["fallback"]["provider_failure"] is None and inv["fallback"]["message"] == generic, "no provider event: generic wording"

    def rewrite(change):
        db = SessionLocal()
        try:
            row = db.get(Investigation, inv["id"])
            row.params_json = change(dict(row.params_json or {}))
            db.commit()
        finally:
            db.close()
        return client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()["fallback"]

    odd = rewrite(lambda p: {**p, "fallback": {"reason": f"<script>alert(1)</script> {OUTPUT_MARKER}"}})
    assert odd["reason"] == "unknown" and OUTPUT_MARKER not in json.dumps(odd) and "<script>" not in json.dumps(odd)
    assert odd["message"].startswith("The AI investigation could not complete: the AI model was unavailable. The result shown is from")
    legacy = rewrite(lambda p: {k: v for k, v in p.items() if k != "fallback"})     # stored before fallback codes existed
    assert legacy["used"] is True and legacy["reason"] == "unknown", "still recognised from the run trace"
    assert legacy["message"] == odd["message"]

    pages = frontend_pages(client, monkeypatch)
    at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    assert [str(i.value) for i in at.info] == [legacy["message"]]
    assert OUTPUT_MARKER not in shown(at) and SECRET_KEY not in shown(at)
    assert OUTPUT_MARKER not in raw(at) and SECRET_KEY not in raw(at) and "<script>" not in raw(at)
    monkeypatch.setattr("app.fallbacks.describe", lambda db, inv: {"used": True})        # a response with no message at all
    at = pages.healthy(pages.page("13_Investigations.py", ws, investigation_id=inv["id"]))
    assert "The AI investigation could not complete. The result shown is from the deterministic investigator" in str(at.info[0].value)
