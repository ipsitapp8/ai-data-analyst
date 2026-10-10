"""Evaluation and regression lab: the suite itself, the scripted provider,
persistence, run comparison and the results API."""
from __future__ import annotations

import copy

import pytest
from platform_helpers import signup

from app import runtime
from app.agents import llm_client
from app.agents import llm_client_llama as llama
from app.agents.critic import _critic_call_tool
from app.database import SessionLocal
from app.evals import runner
from app.evals.scripted import ScriptedProvider, ScriptError
from app.models import EvalResult, EvalRun


@pytest.fixture(scope="module")
def report(client):
    """One offline run of the whole suite, shared by the tests below."""
    return runner.run_suite("offline")


def test_suite_is_versioned_and_covers_every_component():
    suite = runner.load_suite()
    assert suite["suite_version"]
    offline = [c for c in suite["cases"] if not c.get("live")]
    assert {c["component"] for c in offline} == set(runner.COMPONENTS)
    assert any(c.get("live") for c in suite["cases"]), "live cases exist but are kept separate"
    pipeline = [c["id"] for c in offline if c["component"] == "pipeline"]
    for required in ("pipeline-correct", "pipeline-wrong-calculation-is-not-verified",
                     "pipeline-hallucinated-kpi-ends-unverified", "pipeline-missing-result-is-caught",
                     "pipeline-executor-gives-up", "pipeline-triage-rejects"):
        assert required in pipeline


def test_offline_suite_passes_completely_and_deterministically(report):
    s = report["summary"]
    failed = [(r["case_id"], [c for c in r["detail"]["checks"] if not c["ok"]]) for r in report["results"] if not r["passed"]]
    assert not failed, failed
    assert s["cases"] == s["passed"] >= 35 and s["pass_rate"] == 1.0
    assert all(r["component"] != "pipeline" or r["metrics"]["job_state"] in ("succeeded", "failed") for r in report["results"])
    assert report["mode"] == "offline" and report["provider"] == "scripted"
    assert not any(r["case_id"].startswith("live-") for r in report["results"]), "offline runs never include live cases"


def test_pipeline_metrics_are_measured(report):
    by_id = {r["case_id"]: r["metrics"] for r in report["results"]}
    ok = by_id["pipeline-correct"]
    assert ok["llm_calls"] == 7 and ok["sandbox_runs"] == 3 and ok["retries"] == 0 and ok["execution_success"] is True
    assert ok["providers"] == ["scripted:scripted"] and ok["estimated_cost_usd"] == 0.0 and ok["latency_ms"] > 0
    assert by_id["pipeline-executor-self-corrects"]["retries"] == 1
    assert by_id["pipeline-executor-gives-up"]["retries"] == 2 and by_id["pipeline-executor-gives-up"]["execution_success"] is False
    hallucinated = by_id["pipeline-hallucinated-kpi-ends-unverified"]
    assert hallucinated["revisions"] == 1 and hallucinated["verification_failed_checks"] > 0
    assert hallucinated["llm_calls"] <= 14, "bounded: one recovery pass, then it stops"
    assert by_id["pipeline-fast-path-uses-no-model"]["llm_calls"] == 0
    p = report["summary"]["pipeline"]
    assert p["runs"] == 9 and p["llm_calls"] == sum(by_id[k]["llm_calls"] for k in by_id if k.startswith("pipeline-"))
    assert p["retries"] == 3 and p["sandbox_timeouts"] == 0


def test_a_wrong_expectation_is_reported_as_a_failure_with_the_reason():
    suite = runner.load_suite()
    case = copy.deepcopy(next(c for c in suite["cases"] if c["id"] == "fast-total-revenue"))
    case["expect"]["value"] += 1000
    result = runner.run_case(case)
    assert result["passed"] is False
    bad = [c for c in result["detail"]["checks"] if not c["ok"]]
    assert [c["check"] for c in bad] == ["value"] and "reference" in bad[0]["detail"]
    crash = runner.run_case({"id": "x", "component": "engine", "fixture": "missing.csv", "formula": "sum(a)", "expect": {"value": 1}})
    assert crash["passed"] is False and crash["detail"]["error"], "a crashing case is a failed case, not a crashed run"


def test_runs_are_stored_and_compared_for_regressions(report):
    db = SessionLocal()
    try:
        base_id = runner.store_run(db, report)
        regressed = copy.deepcopy(report)
        for r in regressed["results"]:
            if r["case_id"] in ("fast-total-revenue", "pipeline-correct"):
                r["passed"] = False
        regressed["summary"] = runner.summarize(regressed["results"])
        new_id = runner.store_run(db, regressed)
        assert db.query(EvalResult).filter_by(run_id=base_id).count() == len(report["results"])

        def stored(run_id):
            return runner.run_out(db.get(EvalRun, run_id), db.query(EvalResult).filter_by(run_id=run_id).all())

        cmp = runner.compare(stored(base_id), stored(new_id))
        assert cmp["has_regression"] and cmp["regressions"] == ["fast-total-revenue", "pipeline-correct"]
        assert cmp["fixed"] == [] and cmp["comparable"] is True
        assert cmp["new"]["pass_rate"] < cmp["base"]["pass_rate"] == 1.0
        back = runner.compare(stored(new_id), stored(base_id))
        assert not back["has_regression"] and back["fixed"] == ["fast-total-revenue", "pipeline-correct"]
        md = runner.to_markdown(regressed, cmp)
        assert "Failed cases" in md and "`pipeline-correct`" in md and "Regressions: fast-total-revenue" in md
        assert stored(base_id)["suite_version"] == report["suite_version"] and stored(base_id)["app_version"]
    finally:
        db.close()


def test_results_api_is_read_only_and_requires_sign_in(client, report):
    db = SessionLocal()
    try:
        a = runner.store_run(db, report)
        b = runner.store_run(db, report)
    finally:
        db.close()
    assert client.get("/api/evals/runs").status_code == 401
    token, _ = signup(client)
    h = {"Authorization": f"Bearer {token}"}
    runs = client.get("/api/evals/runs", headers=h).json()
    assert runs[0]["id"] == b and runs[0]["summary"]["pass_rate"] == 1.0 and "results" not in runs[0]
    one = client.get(f"/api/evals/runs/{a}", headers=h).json()
    assert len(one["results"]) == len(report["results"]) and one["results"][0]["detail"]["checks"]
    cmp = client.get("/api/evals/compare", params={"base": a, "new": b}, headers=h).json()
    assert cmp["has_regression"] is False and cmp["regressions"] == []
    assert client.get("/api/evals/runs/99999999", headers=h).status_code == 404
    info = client.get("/api/evals/suite", headers=h).json()
    assert info["cases"] > info["live_cases"] >= 1 and info["by_component"]["pipeline"] >= 9
    for method in ("post", "put", "delete"):
        assert getattr(client, method)("/api/evals/runs", headers=h).status_code == 405, "nothing can start a run over HTTP"


def test_scripted_provider_replays_and_repeats_its_last_reply():
    p = ScriptedProvider({"t": [{"n": 1}, {"n": 2}]})
    assert [p("gemini", tool_name="t")["n"] for _ in range(4)] == [1, 2, 2, 2]
    assert p.count() == 4 and p.count("t") == 4 and p.calls[0] == ("gemini", "t")
    with pytest.raises(ScriptError):
        p("gemini", tool_name="unscripted")
    reply = p("llama", tool_name="t")
    reply["n"] = 99
    assert p("llama", tool_name="t")["n"] == 2, "replies are copies; a caller cannot corrupt the script"


def test_both_providers_run_behind_the_same_interface():
    """The comparison path: the same call through the Gemini adapter and the
    Llama adapter, with usage recorded per provider call."""
    kwargs = dict(system="s", user_content="u", tool_name="t", tool_schema={"type": "object"}, tool_description="d")
    p = ScriptedProvider({"t": [{"ok": True}]})
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx), runtime.provider_override(p):
        assert llm_client.call_tool(**kwargs) == {"ok": True}
        assert llm_client.call_tool_gemini(**kwargs) == {"ok": True}
        assert llama.call_tool(**kwargs) == {"ok": True}
        out, label = _critic_call_tool(**kwargs)
        assert out == {"ok": True} and label == "scripted-critic"
    assert [hint for hint, _ in p.calls] == ["gemini", "gemini", "llama", "llama"], "the Critic goes through the second provider"
    assert ctx.llm_calls == 4 and ctx.providers == ["scripted:scripted"] and ctx.cost_usd == 0.0


def test_cost_estimates_and_provider_failover_are_recorded(monkeypatch):
    from app import config

    assert runtime.estimate_cost("scripted", 10**6, 10**6) == 0.0
    monkeypatch.setattr(config, "LLM_PRICE_GEMINI", "1.0,2.0")
    assert runtime.estimate_cost("gemini", 2_000_000, 500_000) == 3.0
    monkeypatch.setattr(config, "LLM_PRICE_GEMINI", "garbage")
    assert runtime.estimate_cost("gemini", 10, 10) == 0.0
    ctx = runtime.RunContext(question_id=0, record=False)
    with runtime.run_context(ctx):
        runtime.note_llm_call("gemini", "m", "t", None, None, 5, ok=False)
        runtime.note_llm_call("llama", "m2", "t", 100, 50, 5, fallback=True)
    assert ctx.usage()["provider_fallbacks"] == 1 and ctx.providers == ["gemini:m", "llama:m2"]
    assert ctx.tokens_in == 100 and ctx.llm_calls == 2


def test_failover_can_be_disabled_and_missing_keys_fail_clearly(monkeypatch):
    from app import config

    kwargs = dict(system="s", user_content="u", tool_name="t", tool_schema={"type": "object"}, tool_description="d")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "LLAMA_API_KEY", "")
    with pytest.raises(RuntimeError, match="no GEMINI_API_KEY") as e:
        llm_client.call_tool(**kwargs)
    assert "no LLAMA_API_KEY" in str(e.value)

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(config, "LLAMA_API_KEY", "fake")
    monkeypatch.setattr(llm_client, "call_tool_gemini", lambda **k: (_ for _ in ()).throw(RuntimeError("gemini down")))
    served = []
    monkeypatch.setattr(llama, "call_tool", lambda **k: served.append(k.get("_fallback")) or {"from": "llama"})
    assert llm_client.call_tool(**kwargs) == {"from": "llama"} and served == [True], "the switch is marked as a fallback"
    monkeypatch.setattr(config, "LLM_FAILOVER_ENABLED", False)
    with pytest.raises(RuntimeError, match="failover disabled"):
        llm_client.call_tool(**kwargs)
    assert served == [True], "with failover off, the second provider is never called"


def test_a_cancelled_run_is_never_mistaken_for_a_provider_failure(monkeypatch):
    from app import config

    kwargs = dict(system="s", user_content="u", tool_name="t", tool_schema={"type": "object"}, tool_description="d")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(config, "LLAMA_API_KEY", "fake")
    called = []
    monkeypatch.setattr(llama, "call_tool", lambda **k: called.append(1) or {})
    ctx = runtime.RunContext(question_id=0, should_cancel=lambda: True, record=False)
    with runtime.run_context(ctx), pytest.raises(runtime.RunCancelled):
        llm_client.call_tool(**kwargs)
    assert called == [], "cancellation must not trigger failover to the other provider"


def test_transient_gemini_server_errors_are_retried_a_bounded_number_of_times(monkeypatch):
    from google.genai import errors

    from app import config

    kwargs = dict(system="s", user_content="u", tool_name="t", tool_schema={"type": "object"}, tool_description="d")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(config, "LLAMA_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_RATE_LIMIT_RETRIES", 2)
    monkeypatch.setattr(llm_client, "_SERVER_ERROR_BACKOFF_SECONDS", (0.0, 0.0))
    calls = []

    def flaky(**k):
        calls.append(1)
        if len(calls) < 3:
            raise errors.ServerError(503, {"error": {"message": "overloaded"}})
        return {"ok": True}

    monkeypatch.setattr(llm_client, "call_tool_gemini", flaky)
    assert llm_client.call_tool(**kwargs) == {"ok": True} and len(calls) == 3

    calls.clear()
    monkeypatch.setattr(llm_client, "call_tool_gemini",
                        lambda **k: calls.append(1) or (_ for _ in ()).throw(errors.ServerError(503, {"error": {"message": "down"}})))
    with pytest.raises(runtime.ProviderUnavailable):
        llm_client.call_tool(**kwargs)
    assert len(calls) == 3, "one try plus two retries, then it stops"
