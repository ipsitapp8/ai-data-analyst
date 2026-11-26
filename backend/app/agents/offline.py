"""What a run does when no AI model can be reached.

Calculations do not need a model, so a provider outage or an exhausted quota must not throw them away. In order of
preference, a run that loses the model publishes:

1. results the run had already computed (the writer, or a second planning pass, is what failed);
2. an answer from Silt's rule-based analyst (app/analyst, `use_model=False`): pandas and the statistics engine;
3. the steps that did finish, labelled incomplete.

If none of these applies, the run fails with the provider failure, as before. Nothing is ever invented: every figure
shown was computed from the data, the dashboard says no AI wrote it, and a question the rule-based analyst would only
half answer is not answered at all.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid

from app import fastpath, runtime, verification
from app.agents.state import AgentState, update_stage
from app.database import SessionLocal
from app.models import Question

logger = logging.getLogger(__name__)

NOTICE = "AI narration is temporarily unavailable"
_MAX_KPIS, _MAX_TABLES, _MAX_TABLE_ROWS, _MAX_NARRATIVE = 6, 4, 30, 2500
_NOT_AN_ANSWER_SOURCES = {"input_check", "conversation", "conversation_rules", "conversation_state", "withheld"}
_NOT_AN_ANSWER_PLANS = {"clarify", "chat", "rephrase", "withheld", "decline", "error"}
# Fixed thresholds the analyst names when it states a method ("1.5 x IQR rule", "95% CI"): not computed claims.
_METHOD_THRESHOLDS = "1.5 5% 95%"


def reason(error: BaseException) -> str:
    """Which provider failed and how, in the wording the run already uses. Nothing the provider sent is included."""
    failures = getattr(error, "failures", None)
    return runtime.ProviderUnavailable.describe(failures) if failures else "No AI model could be reached."


def notice(error: BaseException, detail: str) -> str:
    return f"{NOTICE}. {reason(error)} {detail}"


def announce(message: str, source: str) -> None:
    """Recorded in the run trace; the status endpoint shows it on the Analyses page."""
    runtime.record_event("ai_fallback", source, {"message": message})


def allowed(question_id: int) -> bool:
    """Not for scheduled runs. Each is compared with the one before it, card by card, so a stand-in dashboard would
    become a baseline with nothing to compare against and hide the next real change. Those runs fail, as before."""
    db = SessionLocal()
    try:
        question = db.get(Question, question_id)
        return question is not None and question.trigger != "scheduled"
    finally:
        db.close()


# --------------------------------------------------------- charts and tables --

def _figure(title: str, traces: list[dict], x: str = "", y: str = "") -> dict:
    layout: dict = {"title": {"text": title}}
    if x:
        layout["xaxis"] = {"title": {"text": x}}
    if y:
        layout["yaxis"] = {"title": {"text": y}}
    return {"data": traces, "layout": layout}


def to_plotly(spec: dict) -> dict | None:
    """The analyst's chart description ({"type", "x", "y", "data"}) as the Plotly figure a dashboard stores."""
    rows, kind = spec.get("data") or [], spec.get("type", "bar")
    x, y, series = spec.get("x"), spec.get("y"), spec.get("series")
    if not rows:
        return None
    title = str(spec.get("title") or "")

    def column(key, part=None):
        return [r.get(key) for r in (rows if part is None else part)]

    if kind == "forecast":
        traces = [{"type": "scatter", "mode": "lines+markers", "name": name,
                   "x": column(x, part), "y": column("value", part)}
                  for name in ("actual", "forecast") if (part := [r for r in rows if r.get("kind") == name])]
        return _figure(title, traces, y=str(spec.get("y_title") or "")) if traces else None
    if kind == "pie":
        return _figure(title, [{"type": "pie", "labels": column(x), "values": column(y)}])
    if kind in ("scatter", "avp"):
        return _figure(title, [{"type": "scatter", "mode": "markers", "x": column(x), "y": column(y)}], x, y)
    shape = {"type": "scatter", "mode": "lines+markers"} if kind == "line" else {"type": "bar"}
    if series and series in rows[0]:
        names = list(dict.fromkeys(str(r.get(series)) for r in rows))
        traces = [{**shape, "name": n, "x": column(x, part), "y": column(y, part)}
                  for n in names if (part := [r for r in rows if str(r.get(series)) == n])]
    else:
        traces = [{**shape, "x": column(x), "y": column(y)}]
    return _figure(title, traces, x, y)


def _table(title: str, header: list[str], rows: list[list]) -> dict:
    cells = [[row[i] if i < len(row) else "" for row in rows[:_MAX_TABLE_ROWS]] for i in range(len(header))]
    return _figure(title, [{"type": "table", "header": {"values": header}, "cells": {"values": cells}}])


def _chart(title: str, figure: dict, step_index: int = 0) -> dict:
    return {"title": title, "step_index": step_index, "plotly_json": figure, "element_id": uuid.uuid4().hex}


# ------------------------------------------------ results already computed --

def _words(name: str) -> str:
    return re.sub(r"[_\s]+", " ", str(name)).strip()


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def draft_from_steps(step_results: list[dict]) -> dict:
    """A dashboard draft built from computed step results by fixed rules: single numbers become KPI cards, lists of
    records become tables, the charts the steps drew are kept, and the narrative only restates those numbers."""
    kpis, charts, sentences, tables = [], [], [], 0
    for step in (r for r in step_results if r.get("success")):
        index, description = step["step_index"], str(step.get("description") or f"Step {step['step_index'] + 1}")
        for path in step.get("chart_paths") or []:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    charts.append(_chart(description[:120], json.load(f), index))
            except (OSError, ValueError):
                continue
        result = step.get("result")
        figures = []
        for key, value in (result.items() if isinstance(result, dict) else []):
            if _number(value):
                shown = fastpath.format_value(float(value))
                figures.append(f"{_words(key)} = {shown}")
                if len(kpis) < _MAX_KPIS:
                    kpis.append({"label": _words(key)[:200], "value": shown, "source_step_index": index,
                                 "element_id": uuid.uuid4().hex})
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value) and tables < _MAX_TABLES:
                header = list(dict.fromkeys(k for v in value[:_MAX_TABLE_ROWS] for k in v))[:8]
                charts.append(_chart(_words(key)[:120], _table(_words(key), [_words(h) for h in header],
                                     [[v.get(h, "") for h in header] for v in value]), index))
                tables += 1
        sentences.append(f"{description.rstrip('. ')}: " + ("; ".join(figures[:8]) if figures else
                         "computed; see the tables, charts and steps") + ".")
    narrative = " ".join(sentences) or "No step produced a result that can be summarised."
    return {"kpis": kpis, "charts": charts, "narrative": narrative[:_MAX_NARRATIVE],
            "narrative_element_id": uuid.uuid4().hex}


# ------------------------------------------------------ rule-based analyst --

def _answers(question: str, columns: list[str], out: dict) -> bool:
    """Whether the analyst computed an answer to this question, not to a neighbouring one."""
    if out.get("verify_status") == "withheld" or out.get("plan_source") in _NOT_AN_ANSWER_SOURCES:
        return False
    if (out.get("plan") or {}).get("type") in _NOT_AN_ANSWER_PLANS:
        return False
    calls = [c for c in out.get("calls") or [] if c.get("status") == "ok"]
    if not calls:
        return False
    if all(c.get("tool") == "understand" for c in calls):
        return False  # a description of the file is not an analysis, and its figures cannot all be traced
    used = json.dumps([[c.get("args"), c.get("result")] for c in calls], default=str).lower()
    asked = _words(question).lower()
    for name in columns:
        # every column the question names must take part in the calculation ("variance in revenue by region" answered
        # with the variance of revenue alone is not an answer)
        phrase = _words(name).lower()
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?:s|es)?(?![a-z0-9])", asked) and str(name).lower() not in used:
            return False
    return True


def _plain(answer: str) -> tuple[str, list[tuple[str, list[str], list[list[str]]]]]:
    """(prose, tables) from the analyst's Markdown: the dashboard shows prose and figures, not Markdown."""
    prose, tables, block, heading = [], [], [], "Results"

    def clean(text: str) -> str:
        return re.sub(r"\*\*|`", "", text).strip().strip("_").strip()

    def flush():
        cells = [[clean(c) for c in line.strip().strip("|").split("|")] for line in block]
        if len(cells) >= 3:
            tables.append((heading, cells[0], cells[2:]))
        block.clear()

    for line in answer.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            block.append(stripped)
            continue
        flush()
        if not stripped or stripped.startswith("_Verification"):
            continue
        text = clean(re.sub(r"^#+\s*|^[-*]\s+", "", stripped))
        if not text:
            continue
        heading = text.rstrip(":.")[:120]
        prose.append(text if text[-1] in ".!?:" else text + ".")
    flush()
    return " ".join(prose)[:_MAX_NARRATIVE], tables[:_MAX_TABLES]


def rule_answer(question: str, csv_path: str, filename: str | None = None) -> dict | None:
    """The analyst's answer without any model, or None when it cannot answer this question."""
    from app.analyst import orchestrator
    from app.analyst.data import load_table

    try:
        df = load_table(csv_path)
        out = orchestrator.ask(question, df, {}, use_model=False, filename=filename)
    except Exception:  # noqa: BLE001 - the fallback must never replace the provider failure with a crash of its own
        logger.exception("The rule-based analyst failed")
        return None
    return out if _answers(question, [str(c) for c in df.columns], out) else None


# ------------------------------------------------------------------- publish --

def _publish(state: AgentState, error: BaseException, out: dict) -> dict:
    from app.agents.dashboard import dataset_evidence_for, publish_dashboard
    from app.agents.deterministic import _record_step

    question_id = state["question_id"]
    calls = [c for c in out["calls"] if c.get("status") == "ok"]
    tools = ", ".join(dict.fromkeys(str(c.get("tool")) for c in calls))
    prose, tables = _plain(out.get("answer", ""))
    figures = [f for spec in out.get("charts") or [] if (f := to_plotly(spec))]
    charts = [_chart(f["layout"]["title"]["text"] or "Chart", f) for f in figures]
    charts += [_chart(title, _table(title, header, rows)) for title, header, rows in tables]
    checked = out.get("verify_status") == "verified"
    result = {"tools": [{"tool": c.get("tool"), "args": c.get("args"), "result": c.get("result")} for c in calls]}
    formula = f"rule-based analyst: {tools}"
    log_id = _record_step(
        question_id, "Compute the answer with the rule-based analyst", "Answer the question without an AI model",
        f"# Rule-based analyst (no AI model was used and no generated code ran).\n# tools: {tools}", result, formula,
        {"columns": [], "rows": []}, reasoning="No AI model could be reached, so the rule-based analyst answered.")
    steps = {0: {"result": result, "execution_log_id": log_id, "formula": formula}}
    dataset_ev = dataset_evidence_for(question_id)
    narrative_id = uuid.uuid4().hex
    records = verification.build_evidence(
        kpis=[], charts=charts, narrative=prose, narrative_element_id=narrative_id, steps=steps, critic=None,
        dataset=dataset_ev, question_text=f"{state['question_text']} {_METHOD_THRESHOLDS}", structured={},
        deterministic=checked)
    message = notice(error, "Silt's built-in engine computed these results from your data. " +
                     ("The headline figure was recomputed independently and matches." if checked else
                      "They were not independently re-checked."))
    dashboard = publish_dashboard(
        question_id=question_id, kpis=[], charts=charts, narrative=prose, narrative_element_id=narrative_id,
        steps=steps, records=records, dataset_ev=dataset_ev,
        base_state="VERIFIED_WITH_CAVEATS" if checked else "UNVERIFIED", verification_summary=message,
        critic_note="Rule-based analyst; no AI model.")
    announce(message, "rule_based_analyst")
    return {"dashboard": dashboard, "evidence": records}


def _publish_steps(state: AgentState, error: BaseException, steps_done: list[dict], detail: str) -> dict:
    from app.agents.dashboard import dataset_evidence_for, publish_dashboard, steps_index

    question_id = state["question_id"]
    draft = draft_from_steps(steps_done)
    dataset_ev = dataset_evidence_for(question_id)
    records = verification.build_evidence(
        kpis=draft["kpis"], charts=draft["charts"], narrative=draft["narrative"],
        narrative_element_id=draft["narrative_element_id"], steps=steps_index(steps_done), critic=None,
        dataset=dataset_ev, question_text=described(state, steps_done))
    message = notice(error, detail)
    dashboard = publish_dashboard(
        question_id=question_id, kpis=draft["kpis"], charts=draft["charts"], narrative=draft["narrative"],
        narrative_element_id=draft["narrative_element_id"], steps=steps_index(steps_done), records=records,
        dataset_ev=dataset_ev, base_state="UNVERIFIED", verification_summary=message,
        critic_note="Built from computed step results; no AI narration.")
    announce(message, "computed_steps")
    return {"dashboard": dashboard, "evidence": records}


def described(state: AgentState, step_results: list[dict]) -> str:
    """The question plus the step descriptions: numbers repeated from either are wording, not computed claims."""
    return " ".join([state["question_text"], *(str(r.get("description", "")) for r in step_results)])


def offline_node(state: AgentState) -> dict:
    error = state["ai_unavailable"]
    question_id = state["question_id"]
    if not allowed(question_id):
        raise error
    update_stage(question_id, "executing", "No AI model could be reached: computing without one")

    earlier = state.get("prior_results") or []
    if earlier:
        return _publish_steps(state, error, earlier,
                              "These results are from the first analysis attempt, which did not pass review; it could "
                              "not be redone. Treat them as unverified.")
    answer = rule_answer(state["question_text"], state["csv_path"])
    if answer is not None:
        return _publish(state, error, answer)
    done = [r for r in state.get("step_results") or [] if r.get("success")]
    if done:
        return _publish_steps(state, error, done,
                              f"Only {len(done)} of {len(state.get('plan') or [])} planned steps were computed, so "
                              "this analysis is incomplete and unverified.")
    raise error  # nothing could be computed without a model: report the provider failure
