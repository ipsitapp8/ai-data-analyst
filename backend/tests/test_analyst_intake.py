"""Unreadable input is stopped before any planning, calculation or prediction; real questions -- short, misspelled,
unfamiliar wording -- still go through. No LLM is called: the model is stubbed so that any call fails the test."""
from __future__ import annotations

import pandas as pd
import pytest

from app.analyst import calc, intake, orchestrator, planner

SALES = pd.DataFrame({
    "month": pd.date_range("2025-01-01", periods=12, freq="MS").repeat(2),
    "region": ["North", "South"] * 12,
    "units": list(range(10, 34)),
    "sales": [100.0 + 7 * i for i in range(24)],
})


CALLED: list[str] = []


@pytest.fixture
def model_must_not_be_used(monkeypatch):
    """The model is 'available', but the model, the calculator, the ML pipeline and document search all record any
    call and fail: a rejected message must be answered by the input check alone."""
    from app.analyst import documents, ml, tools
    CALLED.clear()

    def boom(name):
        def f(*a, **k):
            CALLED.append(name)
            raise AssertionError(f"{name} was called for input that should have been rejected")
        return f
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(planner, "model_chat", boom("model_chat"))
    monkeypatch.setattr(calc, "_complete", boom("model_completion"))
    monkeypatch.setattr(calc, "run", boom("calculator"))
    monkeypatch.setattr(ml, "train", boom("ml_train"))
    monkeypatch.setattr(ml, "predict", boom("ml_predict"))
    monkeypatch.setattr(tools, "run", boom("tools"))
    monkeypatch.setattr(documents, "search", boom("document_search"))
    yield
    assert CALLED == [], CALLED


@pytest.fixture
def model_off(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))


@pytest.mark.parametrize("text", ["idqjnfiW", "ajsdhakjsdh", "qwertyuiop", "cdfvgbhnjmk,l.", "asdf qwer zxcv",
                                  "sdfghjkl", "lkjhgfd please", "zzzzzz", "?!.,;", "...", "", "   ", "\n\t"])
def test_unreadable_input_asks_to_rephrase_and_computes_nothing(text, model_must_not_be_used):
    mem = {"last_answer": {"question": "earlier", "answer": "x", "evidence": []}, "calc_history": [("q", {})]}
    before = dict(mem)
    out = orchestrator.ask(text, SALES, mem)
    assert out["plan_source"] == "input_check" and out["plan"]["type"] == "rephrase"
    assert out["answer"].startswith("I couldn't understand that. Please ask a valid question about your dataset or document.")
    assert "Nothing was calculated" in out["answer"]
    assert out["calls"] == [] and out["charts"] == []
    assert mem == before, "conversation context is untouched"


def test_an_unknown_non_word_in_a_real_question_is_named_not_turned_into_a_column(model_must_not_be_used):
    out = orchestrator.ask("What is the total of cdfvgbhnjmk?", SALES, {}, filename="sales.csv")
    assert out["plan"]["type"] == "clarify" and out["plan"]["input_check"]["terms"] == ["cdfvgbhnjmk"]
    assert "couldn't find **cdfvgbhnjmk**" in out["answer"] and "`sales`" in out["answer"]
    assert out["calls"] == [] and "haven't calculated anything" in out["answer"]


@pytest.mark.parametrize("text", [
    "strengths", "headcount", "HTTPS", "property", "What is the property value?", "liberty", "fjord", "Qatar",
    "json", "pyjamas", "10%", "2025", "Gracias", "Zusammenfassung", "försäljning", "Dziękuję", "продажи", "売上",
    "whcih regoin has the higest sals?",                 # misspelled but understandable
    "Predict sales.", "What should management prioritize next month?", "Show monthly sales as a line chart.",
    "Explain this in simple terms.", "ok", "Why?", "hmm", "EBITDA by region", "Show HTTPS traffic",
    "What are the strengths and lengths of each region?", "Total sales for North", "rhythm of sales over time",
    "Compare Q1 vs Q2", "¿Cuál región vende más?"])
def test_real_questions_pass_the_check(text):
    assert intake.check(text, SALES) is None


def test_a_value_from_the_data_is_never_noise_even_if_it_looks_like_it():
    df = pd.DataFrame({"code": ["bcdfgh", "qwer"], "amount": [1.0, 2.0]})
    assert intake.check("Total amount for bcdfgh", df) is None
    assert intake.check("Total amount for qwer", df) is None
    assert intake.check("Total amount for bcdfgh", None)["kind"] == "unknown_term"


def test_noise_word_rules_are_about_shape_not_a_dictionary():
    for w in ("idqjnfiW", "ajsdhakjsdh", "qwertyuiop", "cdfvgbhnjmk", "asdf", "qwerty", "zxcvbn", "hjkl", "aaaa",
              "fjdksl", "xqzvbt", "lkqjwer"):
        assert intake.noise_word(w), w
    for w in ("lengths", "strengths", "twelfths", "rhythm", "headcount", "departmnet", "whcih", "kpi", "SKUS", "HTTPS",
              "property", "liberty", "poverty", "typewriter", "fjord", "Qatar", "json", "Mjolnir", "försäljning"):
        assert not intake.noise_word(w), w


def test_valid_questions_still_reach_the_existing_paths(model_off):
    out = orchestrator.ask("Show monthly sales as a line chart.", SALES, {}, use_model=False)
    truth = SALES.groupby(SALES["month"].dt.to_period("M"))["sales"].sum()
    assert out["charts"][0]["type"] == "line"
    assert [d["sales"] for d in out["charts"][0]["data"]] == pytest.approx(list(truth.values))
    out = orchestrator.ask("What should management prioritize next month?", SALES, {}, use_model=False)
    assert out["plan"].get("kind") == "advice"
    out = orchestrator.ask("Predict sales.", SALES, {}, use_model=False)
    assert out["plan_source"] != "input_check"


def test_a_follow_up_after_unreadable_input_still_refers_to_the_real_previous_result(model_off):
    mem = {}
    first = orchestrator.ask("Show the top two regions.", SALES, mem, use_model=False)
    orchestrator.ask("asdf qwer", SALES, mem, use_model=False)
    out = orchestrator.ask("Show that as a pie chart.", SALES, mem, use_model=False)
    assert out["charts"] and out["charts"][0]["type"] == "pie"
    assert out["charts"][0]["data"] == first["charts"][0]["data"]


def test_the_exact_browser_reproduction_is_rejected_before_the_model(model_must_not_be_used):
    """Reported from the browser: 'idqjnfiW' got 'What is the question?' from the model."""
    decision = intake.check("idqjnfiW", SALES)
    assert decision["kind"] == "unintelligible" and decision["terms"] == ["idqjnfiW"]
    out = orchestrator.ask("idqjnfiW", SALES, {})
    assert out["plan_source"] == "input_check"
    assert out["answer"].startswith("I couldn't understand that. Please ask a valid question about your dataset or document.")


def test_an_unknown_random_token_inside_a_real_question_is_named_not_rejected_as_gibberish(model_must_not_be_used):
    out = orchestrator.ask("What is the total of idqjnfiW?", SALES, {})
    assert out["plan"]["input_check"] == {"kind": "unknown_term", "terms": ["idqjnfiW"]}
    assert "couldn't find **idqjnfiW**" in out["answer"]


def test_rejected_input_also_stops_document_questions(model_must_not_be_used):
    doc = {"id": "d", "filename": "r.pdf", "pages": [{"page": 1, "text": "Revenue rose.", "chars": 13, "images": 0}],
           "page_count": 1, "scanned_pages": [], "empty_pages": [], "text_pages": [1], "chars": 13}
    out = orchestrator.ask("idqjnfiW", None, {}, document=doc)
    assert out["plan_source"] == "input_check"


def test_real_dictionary_words_are_almost_never_rejected():
    """The rules were measured against a standard English word list; real words standing alone must pass."""
    words = ["account", "budget", "churn", "customer", "department", "employee", "expenses", "forecast", "growth",
             "headcount", "inventory", "liberty", "margin", "outliers", "payroll", "property", "quarterly", "revenue",
             "salary", "strengths", "summary", "trends", "typewriter", "variance", "workforce", "yearly", "zone"]
    assert [w for w in words if intake.check(w, None)] == []
