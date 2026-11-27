"""The shared tool-argument validator (llm_client_llama._checked_args, used by the legacy Llama adapter and, through
providers.validate_arguments, by Groq / NVIDIA / OpenRouter / Mistral) follows the schema contract: a required field
must be present and not null, but "required" alone does not mean non-empty -- minLength does.

Regression for: a valid Triage acceptance ({"is_analyzable": true, "reason": "", "suggestions": []}) was rejected
off Gemini, so the run failed with ProviderUnavailable.
"""
from __future__ import annotations

import json
import types

import pytest

from app import config
from app.agents import llm_client_llama
from app.agents.triage import TRIAGE_TOOL_SCHEMA
from app.llm import providers

TEXT = {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}
NON_EMPTY = {"type": "object", "properties": {"note": {"type": "string", "minLength": 1}}, "required": ["note"]}
ACCEPTED = {"is_analyzable": True, "reason": "", "suggestions": []}


def both(raw, schema):
    """The legacy check and the hybrid adapters' check (which adds types and enums on top of it)."""
    return llm_client_llama._checked_args(raw, schema), providers.validate_arguments(raw, schema)


def test_a_required_string_may_be_empty_when_the_schema_does_not_forbid_it():
    for result in both('{"note": ""}', TEXT):
        assert result == ({"note": ""}, None)


def test_a_required_non_empty_string_is_accepted():
    for result in both('{"note": "fine"}', TEXT):
        assert result == ({"note": "fine"}, None)


def test_min_length_rejects_an_empty_string():
    for args, bad in both('{"note": ""}', NON_EMPTY):
        assert args is None and "minimum length" in bad
    for result in both('{"note": "x"}', NON_EMPTY):
        assert result == ({"note": "x"}, None)


@pytest.mark.parametrize("raw,fragment", [
    ("{}", "was missing"),                                     # missing required string
    ('{"note": null}', "was missing"),                         # null where a string is required
    ("{not json", "not valid JSON"),                           # malformed JSON
    ("[1, 2]", "not a JSON object"),
])
def test_missing_null_and_malformed_arguments_are_still_rejected(raw, fragment):
    for args, bad in both(raw, TEXT):
        assert args is None and fragment in bad


def test_a_wrong_type_is_still_rejected_by_the_hybrid_adapters():
    args, bad = providers.validate_arguments('{"note": 5}', TEXT)
    assert args is None and "was not a string" in bad


def test_the_triage_acceptance_reply_matches_its_own_schema():
    for result in both(json.dumps(ACCEPTED), TRIAGE_TOOL_SCHEMA):
        assert result == (ACCEPTED, None)


def _fake_openai(reply):
    calls = []

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def with_options(self, **kw):
            return self

        def create(self, **kw):
            calls.append(kw)
            call = types.SimpleNamespace(function=types.SimpleNamespace(name=kw["tools"][0]["function"]["name"],
                                                                        arguments=json.dumps(reply)))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(tool_calls=[call], content=None), finish_reason="tool_calls")],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1), model=kw["model"])
    return FakeOpenAI, calls


TRIAGE_CALL = dict(system="s", user_content="u", tool_name="submit_triage", tool_schema=TRIAGE_TOOL_SCHEMA,
                   tool_description="d", max_tokens=1024)


def test_the_triage_acceptance_is_accepted_through_the_openai_compatible_adapter(monkeypatch):
    fake, calls = _fake_openai(ACCEPTED)
    monkeypatch.setattr(providers, "OpenAI", fake)
    monkeypatch.setattr(providers, "_clients", {})
    spec = providers.ProviderSpec("groq", "fake-groq-key", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1")
    args, served = providers.call_tool(spec, **TRIAGE_CALL)
    assert args == ACCEPTED and served == "openai/gpt-oss-120b" and len(calls) == 1


def test_the_triage_acceptance_is_accepted_through_the_legacy_llama_adapter(monkeypatch):
    fake, calls = _fake_openai(ACCEPTED)
    monkeypatch.setattr(llm_client_llama, "OpenAI", fake)
    monkeypatch.setattr(llm_client_llama, "_client", None)
    monkeypatch.setattr(config, "LLAMA_API_KEY", "fake-llama-key")
    assert llm_client_llama.call_tool(**TRIAGE_CALL) == ACCEPTED and len(calls) == 1


# ------------------------------------------- the code fields: minLength 1, rejected before the sandbox --

from app.agents import critic, executor  # noqa: E402
from app import runtime  # noqa: E402

CODE_SCHEMAS = [("executor", executor.CODE_TOOL_SCHEMA,
                 {"reasoning": "r", "formula_explanation": "f", "code": "print(1)"}),
                ("critic", critic.VERIFY_CODE_TOOL_SCHEMA, {"what_it_checks": "totals", "code": "print(1)"})]


@pytest.mark.parametrize("agent,schema,valid", CODE_SCHEMAS)
def test_an_empty_script_is_rejected_and_a_real_one_accepted(agent, schema, valid):
    assert schema["properties"]["code"]["minLength"] == 1
    for result in both(json.dumps(valid), schema):
        assert result == (valid, None)
    for args, bad in both(json.dumps({**valid, "code": ""}), schema):
        assert args is None and "'code' was shorter than its minimum length" in bad


def test_triage_reason_may_still_be_empty_after_the_code_schema_change():
    assert "minLength" not in TRIAGE_TOOL_SCHEMA["properties"]["reason"]
    for result in both(json.dumps(ACCEPTED), TRIAGE_TOOL_SCHEMA):
        assert result == (ACCEPTED, None)


@pytest.fixture
def groq_only(monkeypatch, client):
    """Hybrid routing with Groq as the only configured provider (a fake SDK), so a rejected reply has no fallback.
    `client` creates the test database the agents record their progress in."""
    for name, value in dict(LLM_HYBRID_ROUTING_ENABLED=True, GEMINI_API_KEY="", LLAMA_API_KEY="",
                            GROQ_API_KEY="fake-groq-key", GROQ_MODEL="openai/gpt-oss-120b", NVIDIA_API_KEY="",
                            OPENROUTER_API_KEY="", MISTRAL_ENABLED=False).items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(providers, "_clients", {})

    def install(reply):
        fake, calls = _fake_openai(reply)
        monkeypatch.setattr(providers, "OpenAI", fake)
        return calls
    return install


class SpyRunner:
    def __init__(self):
        self.runs = []

    def seed_input_data(self, *a):
        self.runs.append(("seed", a))

    def run(self, *a):
        self.runs.append(("run", a))
        raise AssertionError("the sandbox must not be reached with an empty script")


def test_an_empty_executor_script_is_rejected_before_the_sandbox(groq_only, monkeypatch):
    calls = groq_only({"reasoning": "r", "formula_explanation": "f", "code": ""})
    spy = SpyRunner()
    monkeypatch.setattr(executor, "get_runner", lambda: spy)
    state = {"question_id": 999_999, "step_index": 0, "plan": [{"description": "Total revenue", "goal": "sum"}],
             "question_text": "Total revenue?", "profile": {"row_count": 1, "col_count": 1, "columns": []},
             "csv_path": "unused.csv", "step_results": []}
    with pytest.raises(runtime.ProviderUnavailable) as e:
        executor.executor_node(state)
    tried = [f for f in e.value.failures if f["kind"] != "not_configured"]
    assert tried == [{"provider": "groq", "kind": "malformed_output", "status": None, "model": "openai/gpt-oss-120b"}]
    assert len(calls) == 1 and spy.runs == [], "rejected at the provider; nothing was seeded or run"


def test_an_empty_critic_verification_script_is_rejected_before_the_sandbox(groq_only, monkeypatch):
    monkeypatch.setenv("LLM_CRITIC_PROVIDERS", "groq")
    calls = groq_only({"what_it_checks": "totals", "code": ""})
    spy = SpyRunner()
    monkeypatch.setattr(critic, "get_runner", lambda: spy)
    state = {"question_id": 999_999, "question_text": "Total revenue?", "csv_path": "unused.csv",
             "profile": {"row_count": 1, "col_count": 1, "columns": []}, "revision_count": 0, "failed": False,
             "step_results": [{"step_index": 0, "description": "d", "code": "print(1)", "result": {}, "success": True}]}
    # No provider produced a usable script: ProviderUnavailable (a RunAborted) propagates from critic_node, as it
    # always has -- and the sandbox was never reached.
    with pytest.raises(runtime.ProviderUnavailable) as e:
        critic.critic_node(state)
    assert [f["kind"] for f in e.value.failures if f["provider"] == "groq"] == ["malformed_output"]
    assert len(calls) == 1 and spy.runs == [], "the verification script was rejected before the sandbox"
