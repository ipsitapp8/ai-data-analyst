"""An AI usage limit or outage must not stop an analysis. Every test goes through the Analyses page's own API: submit
a question, run its job, read the status and the dashboard. The providers are faked at the SDK client, so the real
retry, failover, circuit-breaker and graph code runs."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest
from google.genai import errors
from platform_helpers import ask, job_row, question_row, run_job, workspace

from app import config
from app.agents import breaker, llm_client
from app.agents import llm_client_llama as llama

REQUEST = httpx.Request("POST", "https://example.test/v1")


def gemini_429(quota_id: str = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", limit: str = "10"):
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded.",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": quota_id, "quotaValue": limit}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}]}})


def daily_quota():
    return gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "20")


def llama_429():
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=REQUEST), body=None)


def gemini_503():
    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}})


OUTAGES = {
    "rate_limited": (gemini_429, llama_429),
    "daily_quota_exhausted": (daily_quota, llama_429),
    "timeout": (lambda: httpx.ReadTimeout("timed out", request=REQUEST), lambda: openai.APITimeoutError(request=REQUEST)),
    "unavailable": (gemini_503, lambda: openai.APIConnectionError(request=REQUEST)),
}


class Gemini:
    """`answers[tool]` is a dict to reply with, or a callable making the exception to raise; `otherwise` for the rest."""

    def __init__(self, otherwise, answers: dict | None = None):
        self.otherwise, self.answers, self.calls, self.models = otherwise, answers or {}, [], self

    def generate_content(self, model, contents, config):
        tool = config.tool_config.function_calling_config.allowed_function_names[0]
        self.calls.append(tool)
        answer = self.answers.get(tool, self.otherwise)
        if callable(answer):
            raise answer()
        call = NS(function_call=NS(name=tool, args=answer), text=None)
        return NS(candidates=[NS(finish_reason=NS(name="STOP"), content=NS(parts=[call]))], prompt_feedback=None,
                  usage_metadata=NS(prompt_token_count=10, candidates_token_count=5))

    def embed_content(self, **_):
        raise RuntimeError("no embeddings in tests")


class Llama:
    def __init__(self, otherwise, answers: dict | None = None):
        self.otherwise, self.answers, self.calls = otherwise, answers or {}, []
        self.chat = NS(completions=self)

    def create(self, model, max_tokens, messages, tools, tool_choice):
        tool = tools[0]["function"]["name"]
        self.calls.append(tool)
        answer = self.answers.get(tool, self.otherwise)
        if callable(answer):
            raise answer()
        call = NS(function=NS(name=tool, arguments=json.dumps(answer)))
        return NS(choices=[NS(message=NS(tool_calls=[call]))], usage=NS(prompt_tokens=10, completion_tokens=5))


@pytest.fixture
def providers(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(config, "LLAMA_API_KEY", "test-llama-key")
    monkeypatch.setattr(config, "LLM_FAILOVER_ENABLED", True)
    monkeypatch.setattr(config, "HYPOTHESIS_ENGINE_ENABLED", False)
    import time as real_time
    sleeps: list[float] = []
    monkeypatch.setattr(llm_client, "time", NS(sleep=sleeps.append, monotonic=real_time.monotonic))

    def install(gemini: Gemini, backup: Llama):
        monkeypatch.setattr(llm_client, "get_client", lambda: gemini)
        monkeypatch.setattr(llama, "get_client", lambda: backup)
        return NS(gemini=gemini, llama=backup, sleeps=sleeps)

    return install


@pytest.fixture(scope="module")
def sales(client):
    return workspace(client)


def analyse(client, ws, question: str) -> NS:
    created = ask(client, ws, question)
    run_job(created["job_id"])
    status = client.get(f"/api/questions/{created['question_id']}/status", headers=ws["h"]).json()
    dash = client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"])
    return NS(job=job_row(created["job_id"]), question=question_row(created["question_id"]), status=status,
              dashboard=dash.json() if dash.status_code == 200 else None, ids=created)


def assert_results_available(run, *, mentions: str):
    assert run.job.state == "succeeded", run.question.error
    assert run.question.status in ("verified", "unverified"), "never rejected / not analyzable for a provider failure"
    assert run.status["ai_notice"].startswith("AI narration is temporarily unavailable")
    dash = run.dashboard
    assert dash and mentions.lower() in dash["narrative"].lower()
    assert "AI narration is temporarily unavailable" in dash["verification_summary"]
    assert "Deterministic verification failed" not in dash["verification_summary"], dash["verification_summary"]


# ------------------------------------------- calculations without a model --

QUESTIONS = {
    "Show the monthly revenue trend": "revenue",
    "Compare average revenue across regions": "region",
    "Are there outliers in revenue?": "revenue",
    "Top 5 products by revenue": "Gizmo",
    "How has revenue changed over time?": "revenue",
}


@pytest.mark.parametrize("outage", OUTAGES)
def test_analysis_completes_during_every_kind_of_outage(client, sales, providers, outage):
    first, second = OUTAGES[outage]
    providers(Gemini(first), Llama(second))
    run = analyse(client, sales, "Show the monthly revenue trend")
    assert_results_available(run, mentions="revenue")
    assert any(c["plotly_json"]["data"][0]["type"] == "scatter" for c in run.dashboard["charts"]), "the trend line"


@pytest.mark.parametrize("question", QUESTIONS)
def test_supported_questions_are_computed_when_the_usage_limit_is_reached(client, sales, providers, question):
    providers(Gemini(daily_quota), Llama(llama_429))
    run = analyse(client, sales, question)
    if run.question.route in ("fast", "comparison"):
        assert run.job.state == "succeeded" and run.dashboard, "these routes never needed a model"
        return
    assert_results_available(run, mentions=QUESTIONS[question])


def test_the_figures_are_the_ones_pandas_computes(client, sales, providers):
    import pandas as pd
    from platform_helpers import FIXTURES

    providers(Gemini(daily_quota), Llama(llama_429))
    run = analyse(client, sales, "Top 5 products by revenue")
    expected = pd.read_csv(FIXTURES / "sales.csv").groupby("product")["revenue"].sum().sort_values(ascending=False)
    shown = json.dumps(run.dashboard["charts"])
    for product, total in expected.head(5).items():
        assert product in shown and str(round(float(total), 2)).rstrip("0").rstrip(".") in shown.replace(",", "")


@pytest.mark.parametrize("question", [
    "Write me a poem about this company",
    "Which region has the highest variance in revenue?",   # the rule engine would answer a neighbouring question
    "What does this dataset contain?",                     # a description of the file, not an analysis
])
def test_a_question_that_needs_a_model_fails_honestly_and_can_be_run_again(client, sales, providers, question):
    providers(Gemini(daily_quota), Llama(llama_429))
    run = analyse(client, sales, question)
    assert run.question.status == "failed" and run.question.status != "rejected"
    assert run.job.state == "failed" and run.job.failure_class == "provider_unavailable"
    assert "usage limit was reached (HTTP 429)" in run.question.error and run.dashboard is None
    assert client.post(f"/api/jobs/{run.ids['job_id']}/retry", headers=sales["h"]).status_code == 200


def test_a_scheduled_run_does_not_publish_a_stand_in_dashboard(client, sales, providers):
    from app import jobs
    from app.database import SessionLocal
    from app.models import Dataset

    providers(Gemini(daily_quota), Llama(llama_429))
    db = SessionLocal()
    try:
        question, job, _ = jobs.submit_question(db, team_id=sales["team"], dataset=db.get(Dataset, sales["ds"]),
                                                text="Show the monthly revenue trend", trigger="scheduled")
        job_id, question_id = job.id, question.id
    finally:
        db.close()
    assert run_job(job_id) == "failed"
    assert job_row(job_id).failure_class == "provider_unavailable"
    assert client.get(f"/api/questions/{question_id}/dashboard", headers=sales["h"]).status_code != 200


# ------------------------------------------------ the production failure --

PLAN = {"reasoning": "r", "steps": [{"id": 1, "description": "Total revenue and row count", "goal": "g"},
                                    {"id": 2, "description": "Revenue by region", "goal": "g"}]}
CODE = {"reasoning": "r", "formula_explanation": "sum", "code": (
    'import pandas as pd, json\n'
    'df = pd.read_csv("/workspace/data/input.csv")\n'
    'out = {"total_revenue": float(df.revenue.sum()), "row_count": int(len(df)),\n'
    '       "by_region": df.groupby("region", as_index=False).revenue.sum().to_dict("records")}\n'
    'print("RESULT_JSON:" + json.dumps(out))\n')}
REVIEW = {"verdict": "verified", "confidence": 0.9, "issues": [], "summary": "Recomputed."}
VERIFY = {"what_it_checks": "totals", "code": CODE["code"]}
WRITTEN = {"kpis": [], "charts": [], "narrative": "Revenue was computed per region."}
PIPELINE = {"submit_triage": {"is_analyzable": True, "reason": "", "suggestions": []}, "submit_plan": PLAN,
            "submit_code": CODE, "submit_verification_code": VERIFY, "submit_review": REVIEW,
            "compile_dashboard": WRITTEN}
# the rule engine cannot answer this one, so only the pipeline's own results can be shown
OPEN_QUESTION = "Give me an executive briefing on the shape of the business"


def test_the_backup_model_answers_when_the_first_one_is_rate_limited(client, sales, providers):
    """The production bug: the backup accepted the question with an empty `reason`, which was discarded as unusable,
    and the run was then published as "Not analyzable: usage limit"."""
    fake = providers(Gemini(gemini_429), Llama(llama_429, PIPELINE))
    run = analyse(client, sales, OPEN_QUESTION)
    assert run.question.status in ("verified", "unverified") and run.status["ai_notice"] is None, run.question.error
    assert run.dashboard["narrative"] == WRITTEN["narrative"]
    assert fake.llama.calls[0] == "submit_triage" and "compile_dashboard" in fake.llama.calls
    assert fake.gemini.calls == ["submit_triage"] * (config.GEMINI_RATE_LIMIT_RETRIES + 1), \
        "after the limit is reached once, later steps do not ask again"


def test_a_backup_reply_with_only_the_verdict_is_accepted(providers):
    from app.agents.triage import TRIAGE_TOOL_SCHEMA

    for reply in ('{"is_analyzable": true}', '{"is_analyzable": true, "reason": "", "suggestions": []}',
                  '{"is_analyzable": false, "reason": "no such column"}'):
        assert llama._checked_args(reply, TRIAGE_TOOL_SCHEMA)[1] is None, reply
    assert llama._checked_args('{"reason": "x"}', TRIAGE_TOOL_SCHEMA)[1] is not None


def test_calculations_survive_when_only_the_narration_fails(client, sales, providers):
    answers = {k: v for k, v in PIPELINE.items() if k != "compile_dashboard"}
    providers(Gemini(daily_quota, answers), Llama(llama_429, {k: answers[k] for k in
                                                              ("submit_verification_code", "submit_review")}))
    run = analyse(client, sales, OPEN_QUESTION)
    assert_results_available(run, mentions="Total revenue and row count")
    labels = {k["label"]: k["value"] for k in run.dashboard["kpis"]}
    assert labels["row count"] == "720" and "total revenue" in labels
    tables = [c for c in run.dashboard["charts"] if c["plotly_json"]["data"][0]["type"] == "table"]
    assert tables and "East" in json.dumps(tables)
    trace = client.get(f"/api/questions/{run.ids['question_id']}/trace", headers=sales["h"]).json()
    assert trace["usage"]["sandbox_runs"] >= 2, "the computed results are the ones published"


def test_steps_already_computed_are_kept_when_the_model_is_lost_midway(client, sales, providers):
    calls = {"n": 0}

    class LosesTheModel(Gemini):
        def generate_content(self, model, contents, config):
            tool = config.tool_config.function_calling_config.allowed_function_names[0]
            if tool == "submit_code":
                calls["n"] += 1
                if calls["n"] > 1:
                    raise daily_quota()
            return super().generate_content(model, contents, config)

    providers(LosesTheModel(daily_quota, PIPELINE), Llama(llama_429))
    run = analyse(client, sales, OPEN_QUESTION)
    assert_results_available(run, mentions="Total revenue and row count")
    assert "Only 1 of 2 planned steps were computed" in run.dashboard["verification_summary"]
    assert run.question.status == "unverified"


def test_a_healthy_run_is_unchanged(client, sales, providers):
    fake = providers(Gemini(daily_quota, PIPELINE), Llama(llama_429, PIPELINE))
    run = analyse(client, sales, OPEN_QUESTION)
    assert run.job.state == "succeeded" and run.status["ai_notice"] is None, run.question.error
    assert run.dashboard["narrative"] == WRITTEN["narrative"] and fake.sleeps == []


# ------------------------------------------------------ quota discipline --

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def call():
    return llm_client.call_tool(system="s", user_content="u", tool_name="t", tool_schema=SCHEMA, tool_description="d")


def test_a_daily_quota_is_not_retried_and_the_provider_is_then_left_alone(providers, caplog):
    from app import runtime

    fake = providers(Gemini(daily_quota), Llama(llama_429))
    for _ in range(3):
        with pytest.raises(runtime.ProviderUnavailable) as caught:
            call()
        assert {f["kind"] for f in caught.value.failures} == {"rate_limited"}
    assert len(fake.gemini.calls) == 1 and fake.sleeps == [], "no retries and no repeats of an exhausted daily quota"
    assert len(fake.llama.calls) == 1
    assert "quota_id=GenerateRequestsPerDayPerProjectPerModel-FreeTier limit=20" in caplog.text
    assert "Quota exceeded." not in caplog.text, "the provider's message is not logged"


def test_the_hold_ends_and_a_success_clears_it(providers, monkeypatch):
    from app import runtime

    now = {"t": 1000.0}
    monkeypatch.setattr(breaker, "_now", lambda: now["t"])
    fake = providers(Gemini(gemini_429), Llama(llama_429))
    with pytest.raises(runtime.ProviderUnavailable):
        call()
    tried = len(fake.gemini.calls)
    assert tried == config.GEMINI_RATE_LIMIT_RETRIES + 1, "a per-minute limit still gets its bounded retries"
    now["t"] += 30
    with pytest.raises(runtime.ProviderUnavailable):
        call()
    assert len(fake.gemini.calls) == tried
    now["t"] += 600
    fake.gemini.otherwise = {"x": "ok"}
    assert call() == {"x": "ok"} and breaker.blocked("gemini", config.GEMINI_MODEL) is None


def test_one_overloaded_call_does_not_hold_a_provider_but_two_do(providers):
    providers(Gemini(gemini_503), Llama({"x": "ok"}))
    call()
    assert breaker.blocked("gemini", config.GEMINI_MODEL) is None
    call()
    assert breaker.blocked("gemini", config.GEMINI_MODEL)["kind"] == "overloaded"


def test_the_backup_client_has_a_timeout_and_one_retry(monkeypatch):
    monkeypatch.setattr(config, "LLAMA_API_KEY", "test-llama-key")
    monkeypatch.setattr(llama, "_client", None)
    client = llama.get_client()
    assert client.timeout == config.LLAMA_REQUEST_TIMEOUT_SECONDS and client.max_retries == 1
    monkeypatch.setattr(llama, "_client", None)
