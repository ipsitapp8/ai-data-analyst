"""A run that cannot reach a model says which provider failed and how.

The production failure these cover: "which city have rating more then 4"
failed at its first model call with only "No AI model could be reached". The
default Gemini model had been retired (HTTP 404 for every request, whatever
the key) and the second provider rejected its key (HTTP 401), but neither
fact reached the user, and the 401 was not even classified in the log.
"""
from __future__ import annotations

import logging

import pytest
from google.genai import errors
from platform_helpers import ask, job_row, question_row, run_job, workspace

from app import config, runtime
from app.agents import llm_client
from app.agents import llm_client_llama as llama

SECRET = "sk-THIS-KEY-MUST-NEVER-APPEAR-0123456789"
QUESTION = "which city have rating more then 4"
CITY_CSV = (b"city,restaurant,rating\n"
            b"Pune,A,4.5\nPune,B,4.3\nPune,C,4.4\n"
            b"Delhi,D,3.9\nDelhi,E,4.2\nDelhi,F,3.6\n"
            b"Mumbai,G,4.1\nMumbai,H,4.6\nMumbai,I,4.2\n"
            b"Jaipur,J,3.2\nJaipur,K,3.8\nJaipur,L,4.1\n"
            b"Kochi,M,4.8\nKochi,N,4.0\nKochi,O,4.1\n")
KW = dict(system="s", user_content="u", tool_name="t", tool_schema={"type": "object"}, tool_description="d")


class OpenAIStyleError(Exception):
    """Shaped like the OpenAI-compatible client's errors: the HTTP status is
    `.status_code`, and `.code` is a string from the response body."""

    def __init__(self, status_code: int, code: str | None):
        super().__init__(f"Error code: {status_code} - key {SECRET} rejected")
        self.status_code, self.code = status_code, code


class AuthenticationError(OpenAIStyleError):
    pass


def gemini_bad_key() -> errors.ClientError:
    return errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": f"API key not valid {SECRET}",
                                              "details": [{"@type": "x/ErrorInfo", "reason": "API_KEY_INVALID"}, {"@type": "y"}]}})


def _raise(exc):
    def call(**_):
        raise exc
    return call


@pytest.fixture
def both_keys(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", SECRET)
    monkeypatch.setattr(config, "LLAMA_API_KEY", SECRET)
    monkeypatch.setattr(config, "LLM_FAILOVER_ENABLED", True)
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 0)


# ------------------------------------------------------------ classification --

@pytest.mark.parametrize("exc, kind, status", [
    (AuthenticationError(401, "invalid_api_key"), "authentication", 401),      # was: "other", status None
    (OpenAIStyleError(404, "model_not_found"), "model_not_found", 404),
    (OpenAIStyleError(429, None), "rate_limited", 429),
    (OpenAIStyleError(503, "overloaded"), "overloaded", 503),
    (gemini_bad_key(), "authentication", 400),                                 # was: "bad_request"
    (errors.ClientError(400, {"error": {"details": [{"reason": "SOMETHING_ELSE"}]}}), "bad_request", 400),
    (errors.ClientError(400, {"error": {"details": "not a list of objects"}}), "bad_request", 400),
    (errors.ClientError(404, {"error": {}}), "model_not_found", 404),
])
def test_both_providers_error_shapes_are_classified(exc, kind, status):
    out = llm_client.classify_failure(exc)
    assert (out["kind"], out["status"]) == (kind, status)
    assert SECRET not in str(out), "only the type, the status and a fixed reason word are read"


def test_a_true_status_code_is_never_mistaken_for_a_number():
    class Odd(Exception):
        code, status_code = True, "401"
    assert llm_client.classify_failure(Odd())["status"] is None


# ------------------------------------------------------------- the message --

def test_the_error_names_each_provider_and_how_it_failed(both_keys, monkeypatch, caplog):
    """The production case: a retired default model and a rejected second key."""
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(llm_client, "call_tool_gemini", _raise(errors.ClientError(404, {"error": {"status": "NOT_FOUND"}})))
    monkeypatch.setattr(llama, "call_tool", _raise(AuthenticationError(401, "invalid_api_key")))
    with caplog.at_level(logging.WARNING), pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(**KW)
    assert e.value.failures == [
        {"provider": "gemini", "kind": "model_not_found", "status": 404, "model": "gemini-2.5-flash"},
        {"provider": "llama", "kind": "authentication", "status": 401, "model": config.LLAMA_MODEL}]
    message = e.value.user_message
    assert "Gemini (gemini-2.5-flash): the configured model was not found (HTTP 404)." in message
    assert f"Llama ({config.LLAMA_MODEL}): the provider did not accept the configured API key (HTTP 401)." in message
    assert message.startswith("No AI model could be reached") and "do not need a model" in message
    assert SECRET not in message and "API_KEY" not in message, "no key, and no variable name, in what a user sees"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "provider=llama" in logged and "kind=authentication" in logged and "status=401" in logged
    assert SECRET not in logged


@pytest.mark.parametrize("gemini_key, llama_key, failover, expected", [
    ("", "", True, ["Gemini: no API key is configured.", "Llama: no API key is configured."]),
    ("k", "", True, ["the provider rejected the request (HTTP 400)", "Llama: no API key is configured."]),
    ("k", "k", False, ["the provider rejected the request (HTTP 400)", "Llama: not tried, because provider failover is turned off."]),
])
def test_a_provider_that_was_never_called_is_explained_too(monkeypatch, gemini_key, llama_key, failover, expected):
    monkeypatch.setattr(config, "GEMINI_API_KEY", gemini_key)
    monkeypatch.setattr(config, "LLAMA_API_KEY", llama_key)
    monkeypatch.setattr(config, "LLM_FAILOVER_ENABLED", failover)
    monkeypatch.setattr(llm_client, "call_tool_gemini", _raise(errors.ClientError(400, {"error": {}})))
    with pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(**KW)
    assert all(part in e.value.user_message for part in expected), e.value.user_message


@pytest.mark.parametrize("exc, wording", [
    (errors.ClientError(429, {"error": {}}), "the usage limit was reached (HTTP 429)"),
    (errors.ServerError(503, {"error": {}}), "the provider's service was temporarily unavailable (HTTP 503)"),
    (TimeoutError("slow"), "the provider could not be reached over the network."),
    (gemini_bad_key(), "the provider did not accept the configured API key (HTTP 400)"),
    (llm_client.MalformedOutput("gemini", "t", {"category": "prose_instead_of_function_call"}), "the model did not return a usable answer."),
    (ValueError(SECRET), "the request failed."),
])
def test_each_kind_of_failure_gets_its_own_wording(both_keys, monkeypatch, exc, wording):
    monkeypatch.setattr(llm_client, "call_tool_gemini", _raise(exc))
    monkeypatch.setattr(llama, "call_tool", _raise(TimeoutError("slow too")))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    with pytest.raises(runtime.ProviderUnavailable) as e:
        llm_client.call_tool(**KW)
    assert wording in e.value.user_message and SECRET not in e.value.user_message


def test_an_unexplained_provider_failure_keeps_the_general_message():
    assert "missing or invalid API key" in runtime.ProviderUnavailable("x").user_message
    assert runtime.ProviderUnavailable.user_message == runtime.ProviderUnavailable().user_message


# ------------------------------------------------------------------ config --

def test_keys_lose_the_whitespace_and_quotes_a_pasted_value_can_carry(monkeypatch):
    for raw in (f"  {SECRET}\n", f'"{SECRET}"', f"'{SECRET}' ", f'" {SECRET} "'):
        monkeypatch.setenv("SOME_PROVIDER_KEY", raw)
        assert config._secret("SOME_PROVIDER_KEY") == SECRET
    monkeypatch.setenv("SOME_PROVIDER_KEY", "   ")
    assert config._secret("SOME_PROVIDER_KEY") == ""
    monkeypatch.delenv("SOME_PROVIDER_KEY")
    assert config._secret("SOME_PROVIDER_KEY") == ""


def test_the_default_model_is_not_the_retired_one_and_health_names_the_models(client):
    source = (config.BACKEND_DIR / "app" / "config.py").read_text(encoding="utf-8")
    assert 'or "gemini-3.6-flash"' in source and '"gemini-2.5-flash")' not in source
    assert config.GEMINI_MODEL and config.LLAMA_MODEL
    body = client.get("/api/health").json()
    assert body["gemini_model"] == config.GEMINI_MODEL and body["llama_model"] == config.LLAMA_MODEL
    assert body["gemini_key_configured"] is False, "keys are blank in the test environment"
    assert SECRET not in str(body)


# ------------------------------------------------- the question, end to end --

@pytest.fixture
def cities(client):
    return workspace(client, csv=CITY_CSV, filename="city_ratings.csv")


def test_the_question_needs_a_model_and_its_failure_is_explained_in_the_job_and_the_trace(client, cities, both_keys, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(llm_client, "call_tool_gemini", _raise(errors.ClientError(404, {"error": {"status": "NOT_FOUND"}})))
    monkeypatch.setattr(llama, "call_tool", _raise(AuthenticationError(401, "invalid_api_key")))
    created = ask(client, cities, QUESTION)
    assert run_job(created["job_id"]) == "failed"
    job, q = job_row(created["job_id"]), question_row(created["question_id"])
    assert q.route not in ("fast", "root_cause"), "not a simple aggregation: it goes through triage and the planner"
    assert job.failure_class == "provider_unavailable"
    for text in (job.error, q.error):
        assert "Gemini (gemini-2.5-flash): the configured model was not found (HTTP 404)." in text
        assert "the provider did not accept the configured API key (HTTP 401)." in text and SECRET not in text
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=cities["h"]).json()
    failed = [(e["name"], e["detail"]["status"], e["detail"]["action"]) for e in trace["events"] if e["kind"] == "provider"]
    assert failed == [("gemini:model_not_found", 404, "failover_to_llama"), ("llama:authentication", 401, "give_up")]
    assert SECRET not in str(trace)


def test_with_no_key_at_all_the_trace_still_says_why(client, cities):
    created = ask(client, cities, QUESTION)
    assert run_job(created["job_id"]) == "failed"
    q = question_row(created["question_id"])
    assert "Gemini: no API key is configured. Llama: no API key is configured." in q.error
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=cities["h"]).json()
    assert [(e["name"], e["detail"]["action"]) for e in trace["events"] if e["kind"] == "provider"] == [
        ("gemini:not_configured", "not_tried"), ("llama:not_configured", "not_tried")]


CITY_CODE = (
    "import json\nimport pandas as pd\n"
    "df = pd.read_csv('/workspace/data/input.csv')\n"
    "avg = df.groupby('city')['rating'].mean().round(2)\n"
    "above = avg[avg > 4].sort_values(ascending=False)\n"
    "print('RESULT_JSON:' + json.dumps({'cities_above_4': {k: float(v) for k, v in above.items()}, "
    "'count_above_4': int(len(above)), 'cities_total': int(len(avg))}))\n"
    "print('DATA_SLICE_JSON:' + json.dumps({'columns': ['city', 'average_rating'], "
    "'rows': [[k, float(v)] for k, v in avg.items()]}))\n")
CITY_CHECK = (
    "import csv, json\nsums, counts = {}, {}\n"
    "with open('/workspace/data/input.csv', newline='') as f:\n"
    "    for row in csv.DictReader(f):\n"
    "        sums[row['city']] = sums.get(row['city'], 0.0) + float(row['rating'])\n"
    "        counts[row['city']] = counts.get(row['city'], 0) + 1\n"
    "avg = {c: round(sums[c] / counts[c], 2) for c in sums}\n"
    "above = {c: v for c, v in avg.items() if v > 4}\n"
    "print('RESULT_JSON:' + json.dumps({'recomputed_above_4': above, 'recomputed_count_above_4': len(above)}))\n"
    "print('DATA_SLICE_JSON:' + json.dumps({'columns': [], 'rows': []}))\n")
CITY_SCRIPT = {
    "submit_triage": [{"is_analyzable": True, "reason": "", "suggestions": []}],
    "submit_plan": [{"reasoning": "Average the rating per city, keep those above 4.",
                     "steps": [{"id": 1, "description": "Average rating per city, filtered to above 4", "goal": "Answer"}]}],
    "submit_code": [{"reasoning": "pandas groupby", "formula_explanation": "Mean of rating per city, kept when above 4.",
                     "code": CITY_CODE}],
    "submit_verification_code": [{"what_it_checks": "per-city averages", "code": CITY_CHECK}],
    "submit_review": [{"verdict": "verified", "confidence": 0.9, "issues": [], "summary": "The recomputed averages match."}],
    "compile_dashboard": [{"kpis": [{"label": "Cities with average rating above 4", "value": "3", "source_step_index": 0},
                                    {"label": "Highest: Pune", "value": "4.4", "source_step_index": 0}],
                           "charts": [],
                           "narrative": "3 of 5 cities have an average rating above 4: Pune (4.4), Mumbai (4.3) and Kochi (4.3)."}],
}


def test_the_question_completes_through_the_sandbox_when_a_model_answers(client, cities):
    """Mocked model, real pipeline: triage, plan, generated code run in the
    sandbox on the fixture, independent verification, dashboard."""
    created = ask(client, cities, QUESTION)
    assert run_job(created["job_id"], CITY_SCRIPT) == "succeeded"
    q = question_row(created["question_id"])
    assert q.status == "verified", q.error
    dash = client.get(f"/api/questions/{created['question_id']}/dashboard", headers=cities["h"]).json()
    assert {k["label"]: k["value"] for k in dash["kpis"]} == {"Cities with average rating above 4": "3", "Highest: Pune": "4.4"}
    records = client.get(f"/api/questions/{created['question_id']}/evidence", headers=cities["h"]).json()["records"]
    assert len(records) == 3
    for r in records:
        failed = [c["name"] for c in r["checks"] if c["outcome"] not in ("pass", "not_applicable")]
        assert not failed, (r["claim_type"], failed)
        assert r["reasons"] == ["limitation:small_sample"], "15 rows: the only caveat is the size of the fixture"
    assert dash["verdict_state"] == "VERIFIED_WITH_CAVEATS"
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=cities["h"]).json()
    assert trace["usage"]["sandbox_runs"] >= 1 and trace["usage"]["llm_calls"] >= 4
    assert not [e for e in trace["events"] if e["kind"] == "provider"], "no provider failure on a healthy run"
    assert dash["narrative"].startswith("3 of 5 cities"), "Pune 4.4, Mumbai 4.3, Kochi 4.3; Delhi 3.9 and Jaipur 3.7 are out"
