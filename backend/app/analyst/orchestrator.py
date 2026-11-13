"""One analyst turn: plan -> validate -> execute -> explain."""
from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict

import pandas as pd

import re

from app.analyst import business, calc, conversation, documents, explain, followup, intake, metrics, ml, plan_checks, planner, present, router, tools
from app.analyst import scope as scoping

logger = logging.getLogger(__name__)
MAX_CONVERSATIONS = 200
_MEMORY: "OrderedDict[str, dict]" = OrderedDict()

SUMMARY_LABEL = ("_Written by the local model; every number in it was checked against the computed results._")
NARRATE_SYSTEM = ("You are Silt's analyst. Write a short plain-language summary (3-6 sentences) of the analysis "
                  "results below for a business user. Use ONLY numbers that appear in the results; do not compute "
                  "new numbers; say when evidence is weak or a difference is not significant; never claim causation.")


def memory_for(conversation_id: str) -> dict:
    """Per-conversation state (trained model, last recommendation). In process
    memory: it does not survive a restart."""
    mem = _MEMORY.get(conversation_id)
    if mem is None:
        mem = _MEMORY[conversation_id] = {}
        while len(_MEMORY) > MAX_CONVERSATIONS:
            _MEMORY.popitem(last=False)
    _MEMORY.move_to_end(conversation_id)
    return mem


DOC_SYSTEM = ("You answer questions about a document using ONLY the passages given. Cite pages as (p. N). If the "
              "passages do not answer the question, say so plainly. Do not add facts or numbers that are not in the "
              "passages. Write 2-5 sentences of plain prose, not JSON.")
_DOC_WORDS = re.compile(r"\b(document|pdf|report|page|pages|paper|policy|contract|section|chapter|according to|"
                        r"the file says|in the text|mentions?)\b", re.I)
_DOC_SUMMARY = re.compile(r"summari[sz]e|what is (this|the) (document|pdf|report|file) about|overview|what('s| is) in "
                          r"(this|the) (document|pdf|report)|key points|main points", re.I)


def ask(message: str, df: pd.DataFrame | None, memory: dict, use_model: bool | None = None,
        filename: str | None = None, document: dict | None = None) -> dict:
    started = time.monotonic()
    model_on = planner.model_available() if use_model is None else use_model
    # Before anything is planned, computed or predicted: is this a question at all? Unreadable input gets a request to
    # rephrase; a question built around a non-word that is not in the data is told so. Conversation memory is untouched.
    unclear = intake.check(message, df)
    if unclear is not None:
        return _reply(intake.reply(unclear, df, filename), "input_check", model_on, started,
                      plan={"type": "rephrase" if unclear["kind"] == "unintelligible" else "clarify",
                            "input_check": unclear},
                      notes=["Silt checked the message before analysis: "
                             + ("it contains no readable question" if unclear["kind"] == "unintelligible" else
                                f"{', '.join(unclear['terms'])} is not a word and not in the data")
                             + ", so the model was not asked to plan anything and nothing was calculated."])
    if document is not None and (df is None or _DOC_WORDS.search(message)):
        out = _ask_document(message, document, model_on, started)
        out.setdefault("details", "")
        return out
    if df is None:
        return {"answer": "Choose a dataset or a document first.", "plan": {"type": "chat"}, "plan_source": "rules",
                "first_pass": None, "calls": [], "notes": [], "charts": [], "grounding": {"checked": False},
                "details": "", "model_available": bool(model_on), "elapsed_ms": int((time.monotonic() - started) * 1000)}
    # The message is read against the conversation's open task first (app/analyst/conversation.py): a "yes" to a
    # pending clarification continues that request, a reply to the question asked is folded into it, a correction
    # changes it, "forget that" drops it. Deterministic, so a model outage cannot lose the task.
    said = message
    res = conversation.resolve(message, memory, df)
    if res and res.get("reply"):
        out = _reply(res["reply"], "conversation_state", model_on, started,
                     plan={"type": "clarify", "conversation": res["kind"]},
                     notes=["Read against the conversation's open request; nothing was calculated."])
        return _finish(out, message, df, memory, None, started)     # Silt's own question opens no new task
    if res:
        message = res["message"]
        if res.get("correction"):
            message = conversation.retarget(message, memory, df) or message
    chart_kind = followup.chart_request(message)
    try:
        out = _dispatch(message, df, memory, model_on, filename, started, chart_kind)
    except Exception as e:  # noqa: BLE001 - one failed path must not lose the conversation
        logger.exception("analyst turn failed")
        out = _reply(f"Something went wrong while answering that ({type(e).__name__}). Your conversation is kept; "
                     "try rephrasing, or ask a narrower question.", "rules", model_on, started,
                     notes=[f"Internal error: {type(e).__name__}: {e}"[:300]])
    out = _finish(out, message, df, memory, chart_kind, started)
    if res:
        out["notes"].insert(0, res["note"] + (f" Run as: “{message}”." if message != said else ""))
        out["understood_as"] = message
        out["plan"] = {**out.get("plan", {}), "conversation": res["kind"]}
    conversation.record(memory, message, out)
    return out


def _reply(answer: str, source: str, model_on: bool, started: float, **extra) -> dict:
    out = {"answer": answer, "plan": extra.pop("plan", {"type": source}), "plan_source": source, "first_pass": None,
           "calls": [], "notes": [], "charts": [], "grounding": {"checked": False}, "details": "",
           "model_available": bool(model_on)}
    out.update(extra)
    out["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return out


def _dispatch(message, df, memory, model_on, filename, started, chart_kind) -> dict:
    opts = router.ml_options(message)
    memory["_ml_options"] = opts
    # 1. conversation about the previous answer, recommendations, "why is X doing badly?"
    kind = followup.intent(message, memory)
    if kind:
        out = _ask_conversation(kind, message, df, memory, model_on, started)
        if out is not None:
            return out
    # 2. charts that are their own analysis (scatter, histogram, actual vs predicted)
    if chart_kind in ("histogram", "avp") or (chart_kind == "scatter" and not re.search(r"correlat|relationship|related",
                                                                                        message, re.I)):
        out = _ask_chart(chart_kind, message, df, memory, model_on, started)
        if out is not None:
            return out
    if chart_kind in ("bar", "line", "pie", "auto"):
        out = _chart_from_last(message, df, memory, model_on, started)
        if out is not None:
            return out
    # The question's scope -- named people/categories/periods, or the previous question's when a follow-up names
    # nothing new -- is established once here. Every path below works on the records inside it, and the answer is
    # checked against it before it is shown. A request to predict ONE case names inputs, not filters, so it has none.
    sc = None
    if not (memory.get("model") and followup._PREDICT_ONE.search(message)):  # noqa: SLF001
        sc = scoping.establish(message, df, memory)
        if sc["clarify"]:
            return _reply(sc["clarify"], "tool", model_on, started, plan={"type": "clarify", "scope": sc},
                          notes=["The question names a value that occurs in more than one column; Silt asks rather "
                                 "than guessing which records you mean."])
    memory["_scope"] = sc
    dfs = df[scoping.mask(df, sc)] if scoping.active(sc) else df
    if scoping.active(sc) and dfs.empty:
        return _reply(f"No records match **{scoping.describe(sc)}**, so there is nothing to analyse — I haven't used "
                      "the rest of the data instead. Check the spelling or the period.", "tool", model_on, started,
                      plan={"type": "clarify", "scope": sc})
    missing = followup.missing_prediction_inputs(message, df, memory)
    if missing is not None:
        return _reply(missing, "tool", model_on, started, plan={"type": "clarify"},
                      notes=["The trained model needs a value for every input it uses; Silt does not fill in values "
                             "you did not give."])
    if chart_kind in ("bar", "line", "pie", "auto"):
        out = _chart_from_previous_answer(message, df, memory, chart_kind, model_on, started)
        if out is not None:
            return out
    out = _refine_last(message, df, memory, model_on, started, sc)
    if out is not None:
        return out
    if _TREND_WORDS.search(message) and not followup.wants_forecast(message):
        m = metrics.recognise(message)
        if m is not None and m.kind == "row" and not metrics.physical(m, df):
            derived, problem = metrics.row_definition(m.label, df)
            if derived is not None:
                t = followup.trend_of_measure(df, {"last_calc": {"spec": derived, "question": m.label}, "_defined": True}, sc)
                if t is not None:
                    return _trend_answer(message, t, model_on, started)
            elif problem:
                return _reply(f"**I can't show a trend of {m.label} here:** {problem}.", "metric", model_on, started,
                              plan={"type": "clarify"})
    elif not (followup.wants_forecast(message) or re.search(r"\bpredict\w*\b", message, re.I) or _SCENARIO_Q.search(message)):
        out = _ask_metric(message, df, memory, model_on, started, sc)
        if out is not None:
            return out
    out = _ask_which_measure(message, df, model_on, started)
    if out is not None:
        return out
    rk = followup.rank_request(message, dfs)
    if rk is not None and not (model_on and calc.wanted(message, df, memory)):
        if rk["measure"] is None:
            from app.analyst.data import kind_of
            nums = [c for c in df.columns if kind_of(df[c]) == "numeric"]
            return _reply(f"Top {rk['n']} {rk['group']} by which value? For example: " + ", ".join(
                f"`{c}`" for c in nums[:8]) + ".", "tool", model_on, started, plan={"type": "clarify"})
        r = followup.rank_groups(dfs, rk["group"], rk["measure"], rk["n"], rk["low"])
        chart = {"type": "bar", "additive": r["additive"], "x": rk["group"], "y": rk["measure"], "title": f"{'Bottom' if rk['low'] else 'Top'} "
                 f"{rk['n']} {rk['group']} by {r['aggregation'] if r['aggregation'] == 'mean' else 'total'} {rk['measure']}",
                 "data": [{rk["group"]: x["group"], rk["measure"]: x["value"]} for x in r["rows"]]}
        return _reply(followup.rank_text(r, rk["assumed_measure"]), "tool", model_on, started, charts=[chart],
                      calls=[{"tool": "rank_groups", "status": "ok", "result": _jsonable(r)}],
                      notes=["Ranked by Silt's deterministic tools over "
                             + (f"the records in scope ({scoping.describe(sc, len(dfs), len(df))})." if scoping.active(sc)
                                else "all records.")])
    out = _ask_next_record(message, df, memory, model_on, started, sc)
    if out is not None:
        return out
    # 3. "predict the next value": a forecast over time, ML only when the history supports it
    if followup.wants_forecast(message) or re.search(r"\bpredict\w*\b.*\bnext\b", message, re.I):
        out = _ask_forecast(message, df, memory, model_on, started, sc)
        if out is not None:
            return out
    routed = router.route(message, dfs, memory)
    # 4. "what if X changes by N%" where X feeds the result just computed: re-run that calculation
    if routed and routed.get("type") == "router" and routed["tool"] == "column_change":
        s = followup.scenario_on_measure(df, memory, routed["args"]["column"], routed["args"]["change_pct"], sc)
        if s is not None:
            out = _scenario_answer(message, s, model_on, started)
            if re.search(r"\b(salar\w*|pay|wages?|earnings)\b", message, re.I) and not re.search(
                    r"salar|pay|wage|earning", s["column"], re.I):
                out["answer"] = (f"_This dataset has no salary column; pay here is **{s['column'].replace('_', ' ')}**, "
                                 "so that is what was changed._\n\n" + out["answer"])
            return out
    if _TREND_WORDS.search(message) and memory.get("last_calc") and not router._numeric_named(df, message.lower()) \
            and not metrics.recognise(message) and \
            followup.forecast_target(message.replace("trend of", "predict"), df, memory, sc)["kind"] == "measure":
        t = followup.trend_of_measure(df, memory, sc)
        if t is not None:
            return _trend_answer(message, t, model_on, started)
    deferred = memory.pop("_deferred_metric", None)
    why_calc = (calc.wanted(message, df, memory) or ("the question sets its own conditions" if deferred else None)) \
        if model_on else None
    failed = None
    if why_calc:
        out = _ask_calc(message, df, memory, why_calc, started, sc=sc)
        if out is not None and out["plan_source"] not in ("model_calc_failed",) and out["plan"].get("type") == "calc" \
                and out.get("calls"):
            return out
        if out is not None and not deferred and out["plan_source"] != "model_calc_failed":
            return out
        failed = out
    if deferred:
        return _reply(f"I can compute **{deferred['label']}** ({deferred['definition']}), but I couldn't apply "
                      + "; ".join(f"*{x}*" for x in deferred["unhandled"]) + " to this data — nothing in its columns "
                      "records that, and the local model could not turn it into a check either. I haven't shown a number "
                      "that ignores it, or a different quantity. Ask without that condition, or say which column and "
                      "value identify it.", "metric", model_on, started,
                      plan={"type": "clarify", "unhandled": deferred["unhandled"]},
                      notes=["The question's condition could not be applied by Silt's deterministic metric layer or by "
                             "the model's validated calculation plan."])
    if routed is not None:
        if routed.get("type") == "router" and routed["tool"] == "trend" and not router._numeric_named(df, message.lower()):  # noqa: SLF001
            t = followup.trend_of_measure(df, memory, sc)
            if t is not None:
                return _trend_answer(message, t, model_on, started)
        out = _ask_routed(message, dfs, memory, routed, model_on, filename, started)
        return _after_failed(out, failed)
    out = _ask_planned(message, dfs, memory, model_on, started)
    mismatch = plan_checks.metric_mismatch(message, out.get("calls", []), dfs)
    if mismatch:
        alt = _ask_calc(message, df, memory, f"the planned tool computed a different quantity: {mismatch}", started,
                        sc=sc) if model_on else None
        if alt is not None and alt["plan_source"] == "model_calc":
            alt["notes"].insert(0, f"Silt rejected the first plan because {mismatch}; it was answered as a calculation.")
            return alt
        return _reply(f"**I couldn't compute what you asked for with the available tools:** {mismatch}. I haven't "
                      "shown the other number instead. Try phrasing it as a calculation, for example *share of records "
                      "where X is Y by group*.", "tool", model_on, started, plan={"type": "clarify"},
                      notes=[f"The planned result did not match the question: {mismatch}."])
    # The trained catalogue could not answer (the model declined, or its plan was invalid and the rules took over
    # with nothing to run): the model gets the calculator instead of the user getting a dead end.
    if model_on and not why_calc and (out["plan"].get("type") in ("decline", "chat") or out["plan_source"] == "rules") \
            and not any(c.get("status") == "ok" for c in out["calls"]):
        alt = _ask_calc(message, df, memory, "the trained tool catalogue had no way to answer it", started,
                        previous=out, sc=sc)
        if alt is not None:
            out = alt
    if opts and any(c.get("tool") == "random_forest" for c in out["calls"]):
        out["notes"].insert(0, "Training options taken from your message: " + ", ".join(
            f"{'test share ' + format(v * 100, 'g') + '%' if k == 'test_size' else 'seed ' + str(v)}" for k, v in opts.items()) + ".")
    out = _after_failed(out, failed)
    if out["plan"].get("type") == "chat" and _VAGUE.search(out["answer"]):
        # a placeholder reply ("I'm not sure what to compute") is not an answer: use the dataset context instead
        out = _dataset_reply(message, df, memory, model_on, started, out)
    return out


_TREND_WORDS = re.compile(r"\b(trends?|over time|by (day|week|month|quarter)|daily|weekly|monthly|month by month)\b",
                          re.I)
_VAGUE = re.compile(r"not sure what to compute|could you rephrase|I'm not sure|cannot help", re.I)


def _after_failed(out: dict, failed: dict | None) -> dict:
    """The calculation could not be verified, but another tool answered: keep that answer and say so."""
    if failed is None:
        return out
    useful = any(c.get("status") == "ok" and not _empty_result(c.get("result")) for c in out.get("calls", [])) \
        or out["plan"].get("type") in ("clarify",)
    if useful:
        out["notes"].insert(0, "The model's calculation plan did not pass Silt's checks ("
                            + (failed.get("first_pass") or {}).get("error", "see trace")[:200]
                            + "), so this was answered with Silt's other tools.")
        return out
    return failed


def _empty_result(r) -> bool:
    """A tool result computed over no records (e.g. a filter that matched nothing) answers nothing."""
    if not isinstance(r, dict):
        return False
    for k in ("rows", "n", "rows_used", "count"):
        if r.get(k) == 0:
            return True
    return "value" in r and r.get("value") is None


def _dataset_reply(message, df, memory, model_on, started, previous) -> dict:
    from app.analyst import overview
    ov = overview.understand(df)
    sugg = ov["suggested_questions"][:4]
    prev = memory.get("last_answer")
    answer = ("I can answer that best with a concrete calculation or comparison on this dataset"
              + (f" ({ov['rows']} records; columns: " + ", ".join(f"`{c}`" for c in df.columns[:12]) + ")") + ". "
              "For example:\n\n" + "\n".join(f"- {q}" for q in sugg))
    if prev:
        answer += f"\n\nOr ask me to explain the previous answer (*{prev['question'][:80]}*) in simple terms."
    out = _reply(answer, "conversation", model_on, started, notes=previous.get("notes", []) + [
        "The question was not specific enough to compute; suggestions are based on this dataset's columns."])
    return out


def _whole_sentences(text: str) -> str:
    """Model text cut off by the token limit ends mid-sentence: keep the complete sentences."""
    text = text.strip()
    if re.search(r"[.!?)_*]\s*$", text):
        return text
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
    return text[:ends[-1]] if ends else text


def _compose(body: str, summary: str | None) -> str:
    """Direct, computed answer first; the model's plain-language reading after it, labelled."""
    if not summary:
        return body
    summary = _whole_sentences(summary)
    return f"{body}\n\n**In plain words:** {summary}\n\n{SUMMARY_LABEL}"


def _finish(out: dict, message: str, df, memory: dict, chart_kind: str | None, started: float) -> dict:
    out.setdefault("details", "")
    out.setdefault("charts", [])
    if chart_kind:
        charts, notes = followup.restyle(out["charts"], chart_kind if chart_kind != "auto" else None, message)
        out["charts"] = charts
        out["notes"] += notes
        out["chart_requested"] = True
        if chart_kind == "scatter" and not any(c["type"] == "scatter" for c in charts):
            from app.analyst.data import find_column, kind_of
            nums = followup.mentioned_order([c for c in find_column(df, message) if kind_of(df[c]) == "numeric"], message)
            if len(nums) >= 2:
                try:
                    charts = charts + [followup.scatter(df, nums[0], nums[1])]
                    out["charts"] = charts
                except ValueError:
                    pass
        if not charts and not out.get("chart_requested") and out.get("plan_source") not in ("conversation",
                                                                                              "conversation_rules"):
            out["answer"] += ("\n\n_No chart could be drawn from this answer: it has no breakdown by group or over "
                              "time. Ask for a value **by** a group (e.g. by department) or **over time**._")
        elif notes:
            out["answer"] += "\n\n_" + " ".join(notes) + "_"
    sc = memory.pop("_scope", None)
    answered = any(c.get("status") == "ok" for c in out.get("calls", []))
    if out.get("plan_source") not in ("conversation", "conversation_rules", "input_check"):
        checks = list(out.get("verification") or [])
        if scoping.active(sc):
            used = _records_used(out)
            n_scope = int(scoping.mask(df, sc).sum()) if df is not None else None
            ok = answered and (used is None or 0 < used <= (n_scope if n_scope is not None else used))
            checks.append({"check": "records limited to the question's scope", "ok": ok,
                           "detail": scoping.describe(sc, n_scope, len(df) if df is not None else None)
                           + (f"; the answer used {used}" if used is not None else "")})
            if answered and not ok:
                # a result computed over no records (or over records outside the scope) is not an answer
                out.update(answer=(f"**I couldn't answer this within the records you asked about "
                                   f"({scoping.describe(sc)}).** The analysis "
                                   + ("used no records in that scope" if not used else "used records outside it")
                                   + ", so I haven't shown its numbers. Try naming the measure and the person or group "
                                     "again."), charts=[], plan_source="tool")
                answered = False
            if answered and "**Scope:**" not in out["answer"] and out.get("plan_source") not in ("model_calc",):
                out["answer"] = f"**Scope:** {checks[-1]['detail']}.\n\n" + out["answer"]
        if checks:
            out["verification"] = checks
            out["details"] = (out.get("details") or "") + ("\n\n" if out.get("details") else "") + \
                "**Checks before answering**\n" + "\n".join(f"- {'✓' if c['ok'] else '✗'} {c['check']}: {c['detail']}"
                                                         for c in checks)
        # a chart of the previous result, or an estimate built from it, is the same analysis: it replaces neither the
        # calculation that follow-ups like "Only Rahul." refine, nor the scope they keep
        redraw = answered and all(c.get("tool") in ("chart_of_previous_result", "chart_of_previous_answer",
                                                    "next_record_estimate")
                                  for c in out.get("calls", []) if c.get("status") == "ok")
        if answered and not redraw:
            memory["scope"] = sc if scoping.active(sc) else None
            memory["last_analysis"] = "calc" if out.get("plan_source") in ("model_calc", "metric", "refine") else "other"
    evidence = [c.get("result") for c in out.get("calls", []) if c.get("status") == "ok"]
    followup.remember(memory, message, out, evidence[:3])
    if out.get("charts") and out.get("plan_source") not in ("conversation", "conversation_rules"):
        memory["last_charts"] = {"question": message, "charts": out["charts"],
                                 "columns": sorted({str(c.get(k)) for c in out["charts"] for k in ("x", "y")})}
    out["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return out


# ------------------------------------------------------------- conversation --
def _ask_conversation(kind, message, df, memory, model_on, started) -> dict | None:
    notes: list[str] = []
    ev = None
    if kind == "explain_previous":
        text, grounding = followup.explain_previous(message, memory, model_on, notes)
        prev_q = memory["last_answer"]["question"]
        notes.insert(0, f"Explains the previous answer (*{prev_q[:100]}*) using only its computed results.")
    elif kind == "advice":
        text, grounding, ev = followup.advice(message, df, memory, model_on, notes)
        notes.insert(0, "Recommendations are built from the computed evidence in this conversation (data checks, "
                        "the last breakdown); they are suggestions, not proven causes.")
    else:
        r = followup.why_group(message, df, memory, model_on, notes)
        if r is None:
            return None
        text, grounding, ev = r
        notes.insert(0, "Explained from the previous calculation's per-group results and data checks.")
    src = "conversation" if grounding.get("used_model") else "conversation_rules"
    if not grounding.get("used_model") and model_on:
        notes.append("The model's wording did not pass the number check, so Silt's own explanation is shown.")
    return _reply(text, src, model_on, started, notes=notes, grounding=grounding,
                  calls=[{"tool": kind, "status": "ok", "result": _jsonable(ev)}] if ev else [],
                  plan={"type": "conversation", "kind": kind})


def _ask_chart(kind, message, df, memory, model_on, started) -> dict | None:
    from app.analyst.data import find_column, kind_of
    low = message.lower()
    nums = followup.mentioned_order([c for c in find_column(df, low) if kind_of(df[c]) == "numeric"], low)
    try:
        if kind == "avp":
            model = memory.get("model")
            if model and model.get("test_predictions"):
                spec = {"type": "avp", "x": "actual", "y": "predicted", "data": model["test_predictions"],
                        "title": f"Actual vs predicted {model['target']} (held-out rows)",
                        "caption": f"{len(model['test_predictions'])} held-out rows the model never saw in training; "
                                   "points on the diagonal are perfect predictions."}
                text = f"Here is how the **{model['target']}** model's predictions compare with the actual values on rows it did not train on."
            elif memory.get("last_forecast", {}).get("fitted"):
                fc = memory["last_forecast"]
                spec = {"type": "avp", "x": "actual", "y": "fitted", "data": fc["fitted"],
                        "title": f"Actual vs fitted {fc['label']}", "caption": "Trend line fitted to each period."}
                text = f"Actual versus fitted values of the trend model for **{fc['label']}**."
            else:
                return _reply("There is no prediction in this conversation yet, so there is nothing to compare with "
                              "actual values. Ask me to predict a numeric column first (for example *Predict "
                              "<column> with an 80/20 split*), then ask for this chart again.", "tool", model_on,
                              started, chart_requested=True)
        elif kind == "scatter":
            if len(nums) < 2:
                return None
            spec = followup.scatter(df, nums[0], nums[1])
            text = (f"**{nums[1].replace('_', ' ')} against {nums[0].replace('_', ' ')}**: "
                    + (f"correlation r = {spec['r']:.2f} over {spec['n']} records" if spec["r"] is not None
                       else f"{spec['n']} records") + ". Correlation shows that two values move together, not that one causes the other.")
        else:
            if not nums:
                nums = [c for c in df.columns if kind_of(df[c]) == "numeric"]
                if len(nums) != 1:
                    return None
            spec = followup.histogram(df, nums[0])
            text = f"**Distribution of {nums[0].replace('_', ' ')}** — {spec['caption']}"
    except (ValueError, KeyError) as e:
        return _reply(f"I could not draw that chart: {e}", "tool", model_on, started)
    return _reply(text, "tool", model_on, started, charts=[spec], chart_requested=True,
                  calls=[{"tool": f"chart_{kind}", "status": "ok", "result": {"rows": len(spec["data"]),
                                                                              "caption": spec.get("caption")}}],
                  notes=["Chart data computed by Silt from the dataset; nothing was estimated."])


def _chart_from_last(message, df, memory, model_on, started) -> dict | None:
    """'Show that as a pie chart', 'chart of labor cost by department' right after computing it: draw the result
    already computed (same records, same exclusions) instead of planning it again."""
    last = memory.get("last_calc")
    if not last:
        return None
    if len(last["result"].get("groups") or []) < 2:
        if memory.get("last_analysis") == "calc" and re.search(r"\b(this|that|it|the result|same|previous)\b", message, re.I):
            per = _per_record(last, df, memory.get("scope"))
            if per is not None and len(per["rows"]) >= 2:
                lab, x = per["label"], per["x"]
                chart = {"type": "line" if per["dated"] else "bar", "x": x, "y": lab, "additive": False,
                         "title": f"{lab} — each record behind the result", "data": per["rows"]}
                return _reply(present.calc_headline(last["result"], last["question"]) + f"\n\nThe result is one value, so "
                              f"the chart shows **each of the {len(per['rows'])} records** it was computed from "
                              f"({lab} per record{' over time' if per['dated'] else ''}) — the same records and "
                              "exclusions, nothing recalculated or estimated.", "tool", model_on, started, charts=[chart],
                              chart_requested=True,
                              calls=[{"tool": "chart_of_previous_result", "status": "ok", "result": {"records": len(per["rows"])}}],
                              notes=["Drawn from the previous calculation's records."])
            res = last["result"]
            return _reply(present.calc_headline(res, last["question"]) + "\n\nThat result is **a single value**, so "
                          "there is nothing to compare in a chart (a pie needs parts of a whole; a bar chart needs at "
                          "least two groups). Ask for it **by** a group — for example *by "
                          + next((c.replace("_", " ") for c in df.columns if df[c].dtype == object
                                  and 2 <= df[c].nunique() <= 30), "category") + "* — and then for the chart.",
                          "tool", model_on, started, chart_requested=True,
                          notes=["No chart was drawn: the previous calculation has no breakdown to show."])
        return None
    res = last["result"]
    low = message.lower()
    g = str(res["group_by"]).lower()
    names = {w for n in list(res["aggregates"]) + list(last["spec"]["derive"]) for w in re.split(r"[_\W]+", n.lower()) if len(w) > 2}
    refers = (g in low or g.replace("_", " ") in low or g + "s" in low or (g.endswith("y") and g[:-1] + "ies" in low)
              or re.search(r"\b(this|that|it|these|them|the result|same)\b", low))
    strong = [m.group(0).lower() for m in calc._STRONG.finditer(message)]
    other = [w for w in strong if not set(re.split(r"\W+", w)) & names]
    if not refers or other or calc.named_values(df, message):
        return None
    charts = present.calc_charts(res)
    if not charts:
        return None
    body = present.calc_headline(res, message) + (
        f"\n\nChart of the result computed for *{last['question'][:120]}* — the same {res['rows_valid']} of "
        f"{res['rows_total']} records" + (f", with {res['rows_excluded']} left out as before" if res["rows_excluded"] else "")
        + ".")
    return _reply(body, "tool", model_on, started, charts=charts, chart_requested=True,
                  details=present.calc_details(res),
                  calls=[{"tool": "chart_of_previous_result", "status": "ok", "result": _jsonable(calc_evidence(res))}],
                  notes=["Drawn from the previous calculation's results; nothing was recomputed or estimated."])


# only a breakdown and nothing else: "show department totals", "totals by team", "averages per region please"
_BARE_BREAKDOWN = re.compile(r"^\W*(?:(?:please|now|ok(?:ay)?|and|then|can you|could you)\s+)*(?:show|give|list|get|"
                             r"what are|display|see)?\s*(?:me\s+)?(?:the\s+)?(?:(?:[a-z_]+)\s+)?(?:totals?|sums?|"
                             r"averages?|breakdown|summary)(?:\s+(?:by|per|for each|for every|across)\s+(?:the\s+)?"
                             r"[a-z_]+)?\s*(?:please)?\W*$", re.I)


def _ask_which_measure(message, df, model_on, started) -> dict | None:
    """'Show department totals': a breakdown with no measure named. Asked as one specific question whose options
    are this data's numeric columns and the standard metrics it can derive -- never generic suggestions."""
    from app.analyst.data import find_column, kind_of
    if not _BARE_BREAKDOWN.match(message) or metrics.recognise(message) or followup.wants_forecast(message):
        return None
    if [c for c in find_column(df, message) if kind_of(df[c]) == "numeric"]:
        return None
    group, _ = metrics.group_column(message, df)
    if group is None:
        return None
    nums = [c for c in map(str, df.columns) if kind_of(df[c]) == "numeric" and not re.search(r"(^|_)id$", c, re.I)]
    opts = metrics.available(df) + nums
    if not opts:
        return None
    word = "averages" if re.search(r"averag", message, re.I) else "totals"
    return _reply(f"{word.capitalize()} of which value by {group.replace('_', ' ')}? Options: "
                  + ", ".join(f"`{o}`" for o in opts[:8]) + ".", "tool", model_on, started,
                  plan={"type": "clarify", "options": opts[:8], "group_by": group},
                  notes=["The question names a breakdown but no measure; Silt asks rather than picking one."])


def _per_record(last: dict, df: pd.DataFrame, sc: dict | None) -> dict | None:
    """The per-record values behind a single-value result (e.g. each shift's hours behind an average)."""
    res, spec = last["result"], last["spec"]
    primary = res.get("primary") or next(iter(res.get("aggregates") or {}), None)
    m = re.fullmatch(r"\s*(?:sum|mean|median|max|min)\((.+)\)\s*", (res.get("aggregates") or {}).get(primary, "") or "")
    if not m:
        return None
    try:
        frame = calc.run_derive(df, spec["derive"])
        keep = kept_mask(spec, df)
        vals = calc.evaluate(m.group(1), frame, df)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(vals, pd.Series):
        return None
    from app.analyst.data import kind_of
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    label = primary.replace("_", " ")
    rows = []
    for i in df.index[keep]:
        v = vals.get(i)
        if v is None or pd.isna(v):
            continue
        key = str(pd.Timestamp(df.at[i, dates[0]]).date()) if dates and pd.notna(df.at[i, dates[0]]) else f"line {int(list(df.index).index(i)) + 2}"
        rows.append({(dates[0] if dates else "record"): key, label: float(v)})
    if dates:
        rows.sort(key=lambda r: r[dates[0]])
    return {"label": label, "x": dates[0] if dates else "record", "rows": rows[:200], "dated": bool(dates),
            "values": [r[label] for r in rows]}


_NEXT_RECORD = re.compile(r"\b(?:next|upcoming|following|coming)\s+(shift|record|entry|order|transaction|visit|session|"
                          r"call|ticket|booking|delivery|trip|job|appointment)\b", re.I)


def _ask_next_record(message, df, memory, model_on, started, sc) -> dict | None:
    """'Estimate his next shift': one future record, estimated from the past records of the same metric and scope.
    A historical estimate (their average), stated as such -- not a forecast over periods and not ML."""
    m = _NEXT_RECORD.search(message)
    if not m or not re.search(r"\b(estimate|predict|forecast|expect|guess|how long|what will|likely)\w*", message, re.I):
        return None
    p = metrics.plan(message, df)
    if p is not None and not p.clarify:
        last = {"spec": p.spec, "result": {"primary": p.name, "aggregates": p.spec["aggregate"]}, "question": message}
        label_src = "defined from your question"
    elif memory.get("last_calc"):
        last, label_src = memory["last_calc"], f"from your previous calculation (*{memory['last_calc']['question']}*)"
    else:
        return None
    spec = json.loads(json.dumps(last["spec"]))
    notes = []
    if scoping.active(sc):
        spec["exclude"] = scoping.replace_conflicting(spec.get("exclude", []), sc)
        if _hold_to_scope(spec, df, sc, notes):
            return None
    per = _per_record({**last, "spec": spec}, df, sc)
    if per is None or not per["values"]:
        return None
    vals = per["values"]
    noun = m.group(1).lower()
    import statistics
    mean, med = statistics.fmean(vals), statistics.median(vals)
    sel = [e["reason"] for e in spec.get("exclude", []) if re.search(r"!=\s*'|not\s+in\b", e.get("where", ""))]
    who = scoping.describe(sc) if scoping.active(sc) else ("; ".join(sel) if sel else "all records")
    from app.analyst.explain import f as fmt
    text = (f"**Estimated {per['label']} for the next {noun}: {fmt(mean)}** — the average of the **{len(vals)}** past "
            f"{noun}s in scope ({who}), {label_src}.\n\nMedian {fmt(med)}; range {fmt(min(vals))} to {fmt(max(vals))}"
            + (f"; most recent {fmt(vals[-1])}" if per["dated"] else "") + ".\n\n_Type of answer: a **historical "
            f"estimate** (the average of past {noun}s) — **not a machine-learning prediction**; with {len(vals)} "
            f"record{'s' if len(vals) != 1 else ''} the next {noun} can easily differ by the spread shown._")
    return _reply(text, "tool", model_on, started,
                  calls=[{"tool": "next_record_estimate", "status": "ok",
                          "result": {"label": per["label"], "estimate": mean, "median": med, "records": len(vals),
                                     "min": min(vals), "max": max(vals), "method": "mean of past records", "scope": who}}],
                  notes=notes + ["Estimated from the past records' values computed by Silt's calculator."])


def _chart_from_previous_answer(message, df, memory, kind, model_on, started) -> dict | None:
    """'Show that as a pie chart' after any answer that had a chart (a comparison, a trend, a ranking): redraw the same
    computed values as the requested type. Only when the question refers back and names nothing new."""
    prev = memory.get("last_charts")
    if not prev or not re.search(r"\b(that|it|this|these|those|them|same|the (result|answer|previous))\b", message, re.I):
        return None
    from app.analyst.data import find_column
    if calc.named_values(df, message) or [m for m in calc._STRONG.finditer(message)] or (  # noqa: SLF001
            find_column(df, message) and not set(find_column(df, message)) <= set(prev.get("columns", []))):
        return None
    charts, notes = followup.restyle(prev["charts"], kind if kind != "auto" else None, message)
    return _reply(f"Here is the previous result (*{prev['question'][:120]}*) as a chart — the same computed values, "
                  "nothing recalculated.", "tool", model_on, started, charts=charts, chart_requested=True, notes=notes
                  + ["Redrawn from the previous answer's computed chart data."],
                  calls=[{"tool": "chart_of_previous_answer", "status": "ok", "result": {"charts": len(charts)}}])


def _ask_forecast(message, df, memory, model_on, started, sc=None) -> dict | None:
    """Forecast exactly what the question names. The target is read from the target phrase alone (an input named in
    "using his hourly rate" is never the target); a metric that is not a column is derived per record by the local
    model through the validated calculator; the result carries the requested name and is checked against it."""
    t = followup.forecast_target(message, df, memory, sc)
    notes: list[str] = []
    derived, column = None, None
    if t["kind"] == "ambiguous":
        return _reply(f"**{t['phrase']}** could mean " + " or ".join(f"`{c}`" for c in t["options"]) + ". Which one "
                      "should I forecast?", "tool", model_on, started, plan={"type": "clarify", "target": t},
                      notes=["Several columns fit the requested target; Silt asks rather than picking one."])
    if t["kind"] == "unknown":
        from app.analyst.data import kind_of
        nums = [c for c in df.columns if kind_of(df[c]) == "numeric"]
        return _reply("Which value should I forecast? Name it — for example " + ", ".join(f"`{c}`" for c in nums[:6])
                      + " — or a metric built from them.", "tool", model_on, started, plan={"type": "clarify"})
    if t["kind"] == "derived":
        # a standard business metric is defined deterministically (no model); anything else needs the model
        derived, problem = metrics.row_definition(t["phrase"], df)
        if derived is not None:
            notes.append(f"'{t['phrase']}' is not a column; Silt derived it with its standard definition (no language "
                         f"model needed): {followup.describe_definition(derived)}.")
        elif problem:
            return _reply(f"**I can't forecast {t['phrase']} here:** {problem.rstrip('.')}. I won't forecast a different quantity "
                          "instead.", "tool", model_on, started, plan={"type": "clarify", "target": t})
        elif not model_on:
            return _reply(f"**{t['phrase']}** is not a column of this dataset and not one of the standard metrics Silt "
                          "can define by itself, so deriving it needs the local model (it is offline). I won't forecast a "
                          "different quantity instead. You could say how it is calculated, for example *cost = hours × "
                          "rate*.", "tool", model_on, started, plan={"type": "clarify", "target": t})
    if t["kind"] == "derived" and derived is None:
        derived, trace, problem = followup.derive_target(t["phrase"], df)
        for x in trace:
            if x.get("repaired"):
                notes.append(f"Silt corrected the model's definition ({x['section']}): " + "; ".join(x["repaired"]) + ".")
        if derived is None:
            return _reply(f"**I couldn't define {t['phrase']} from the columns of this dataset** ({problem}), so I "
                          "haven't forecast anything — and I won't forecast a different quantity instead. You could "
                          "say how it is calculated, for example *labor cost = hours × hourly rate*.", "tool", model_on,
                          started, plan={"type": "clarify", "target": t}, notes=notes)
        notes.append(f"'{t['phrase']}' is not a column; the local model defined it and Silt's calculator checked the "
                     f"definition: {followup.describe_definition(derived)}.")
    elif t["kind"] == "column":
        column = t["column"]
    try:
        f = followup.forecast(df, memory, column, sc, derived=derived, label=t.get("phrase") if derived else None,
                              period=followup.requested_freq(message), target_date=followup.requested_date(message))
    except LookupError:
        return _reply("Which value should I forecast? Name it, or compute it first.", "tool", model_on, started,
                      plan={"type": "clarify"})
    except followup.AmbiguousHorizon as e:
        return _reply(str(e), "tool", model_on, started, plan={"type": "clarify", "target": t}, notes=notes)
    except ValueError as e:
        return _reply(f"**I can't forecast {t.get('phrase') or 'that'} with this data.** {e}", "tool", model_on, started)
    # the answer's metric must be the one asked for
    asked = {w.rstrip("s") for w in re.split(r"[_\W]+", followup._UNIT_WORDS.sub(" ", t.get("phrase") or "").lower())  # noqa: SLF001
             if w}
    got = {w.rstrip("s") for w in re.split(r"[_\W]+", f["label"].lower()) if w}
    target_ok = not asked or asked <= got
    check = {"check": "forecast target is the one asked for", "ok": target_ok,
             "detail": f"asked for '{t.get('phrase') or f['label']}', forecast {f['label']}"
                       + (f" = {f['definition']}" if f.get("definition") else "")}
    if not target_ok:
        return _reply(f"**I couldn't forecast {t['phrase']}** — the only series available was {f['label']}, and I won't "
                      "present a different metric as the answer.", "tool", model_on, started,
                      plan={"type": "clarify", "target": t}, verification=[check])
    memory["last_forecast"] = f
    charts = [{"type": "line", "x": "period", "y": f["label"], "title": f"{f['label']} per {f['frequency']}",
               "data": [{"period": p["period"], f["label"]: p["value"]} for p in f["points"]]}]
    if f.get("prediction") is not None:
        charts[0]["data"].append({"period": f["next_period"] + " (predicted)", f["label"]: f["prediction"]})
    return _reply(followup.forecast_text(f), "tool", model_on, started, charts=charts, verification=[check],
                  calls=[{"tool": "forecast", "status": "ok", "result": _jsonable(f)}],
                  notes=notes + [f"Forecast of {f['source']}: "
                                 + ("a scikit-learn linear trend validated on held-out periods." if f["ml_used"]
                                    else f"no model was trained ({f.get('reason', '')}); the estimate is "
                                         f"{f.get('method', 'from past values')}.")])


def _scenario_answer(message, s, model_on, started) -> dict:
    notes = ["Re-ran the previous calculation with the changed value; every number was computed by Silt's calculator."]
    summary, grounding = (_narrate(message, [_jsonable(s)], notes) if model_on else (None, {"checked": False}))
    charts = []
    if s["groups"]:
        g = s["group_by"]
        charts = [{"type": "bar", "x": g, "y": "value", "title": f"{s['results'][0]['result'].replace('_', ' ')}: now vs with the change",
                   "series": "case",
                   "data": [{g: x["group"], "case": case, "value": round(x[k], 4)} for x in s["groups"]
                            for case, k in (("now", "baseline"), ("with the change", "scenario")) if x[k] is not None]}]
    return _reply(_compose(followup.scenario_text(s), summary), "tool", model_on, started, charts=charts,
                  grounding=grounding, notes=notes,
                  calls=[{"tool": "scenario_on_last_calculation", "status": "ok", "result": _jsonable(s)}])


def _trend_answer(message, t, model_on, started) -> dict:
    notes = [f"Trend of the measure computed in the previous answer ({t['label']})."]
    summary, grounding = (_narrate(message, [_jsonable(t)], notes) if model_on else (None, {"checked": False}))
    chart = {"type": "line", "x": "period", "y": t["label"], "title": f"{t['label']} per {t['frequency']}",
             "data": [{"period": p["period"], t["label"]: p["value"]} for p in t["points"]]}
    return _reply(_compose(followup.trend_text(t), summary), "tool", model_on, started, charts=[chart],
                  grounding=grounding, notes=notes, calls=[{"tool": "trend_of_measure", "status": "ok",
                                                             "result": _jsonable(t)}])


def _ask_planned(message: str, df: pd.DataFrame, memory: dict, model_on: bool, started: float) -> dict:
    notes: list[str] = []
    plan, source, first_pass = None, "rules", None
    if model_on:
        try:
            plan, source, first_pass = _model_plan(message, df, memory, notes)
        except Exception as e:  # noqa: BLE001 - the model is optional; the rules always answer
            notes.append(f"The analyst model was unavailable ({type(e).__name__}); the deterministic planner was used.")
            plan = None
    if plan is None:
        plan, source = planner.rule_plan(message, df, memory), "rules"

    calls_out, answer, results = [], "", []
    if plan["type"] == "tools":
        validated, errors = [], []
        for c in plan["calls"][:4]:
            try:
                validated.append(tools.validate_call(c, df, message))
            except tools.InvalidCall as e:
                errors.append(str(e))
        if errors and first_pass is not None and first_pass.get("valid"):
            first_pass.update(valid=False, error="invalid tool call: " + "; ".join(errors))
        if errors and source in ("model", "model_retry"):
            notes.append("The model proposed an invalid tool call (" + "; ".join(errors) + "); the deterministic planner was used.")
            plan, source = planner.rule_plan(message, df, memory), "rules"
            validated, errors = [], []
            if plan["type"] == "tools":
                for c in plan["calls"]:
                    try:
                        validated.append(tools.validate_call(c, df, message))
                    except tools.InvalidCall as e:
                        errors.append(str(e))
        if plan["type"] == "tools":
            if errors:
                answer = "I could not run that analysis: " + "; ".join(errors)
                calls_out = [{"tool": c.get("tool"), "status": "invalid"} for c in plan["calls"]]
            for call in validated:
                try:
                    r = tools.run(call, df, memory)
                    calls_out.append({**call, "status": "ok", "result": _jsonable(r)})
                    results.append((call["tool"], r))
                except (ml.Unsuitable, business.Insufficient) as e:
                    calls_out.append({**call, "status": "refused", "reason": str(e)})
                    answer += ("\n\n" if answer else "") + f"**I can't responsibly do that with this data.** {e}"
                except Exception as e:  # noqa: BLE001
                    logger.exception("analyst tool failed")
                    calls_out.append({**call, "status": "error", "reason": f"{type(e).__name__}: {e}"})
                    answer += ("\n\n" if answer else "") + f"The {call['tool']} step failed ({type(e).__name__})."
            rendered = "\n\n".join(explain.render(t, r) for t, r in results)
            answer = (answer + "\n\n" + rendered).strip() if rendered else answer
    if plan["type"] == "chat":
        reply = plan["reply"]
        if source in ("model", "model_retry") and explain.ungrounded(reply, {}, message):
            notes.append("The model's reply stated numbers no tool produced; it was replaced.")
            reply = planner.rule_plan(message, df, memory).get("reply") or "Could you rephrase the question?"
        answer = reply
    elif plan["type"] == "clarify":
        opts = plan.get("options") or []
        answer = plan.get("question", "Could you clarify?") + ("\n\nOptions: " + ", ".join(f"`{o}`" for o in opts) if opts else "")
    elif plan["type"] == "decline":
        answer = "**The available data is not sufficient for this.** " + plan.get("reason", "")
    elif plan["type"] == "explain_previous":
        if memory.get("last_recommendation"):
            answer = explain.why(memory["last_recommendation"], message)
            results.append(("why", memory["last_recommendation"]))
        elif memory.get("model"):
            r = tools.run({"tool": "feature_importance", "args": {}}, df, memory)
            answer = explain.render("feature_importance", r)
            results.append(("feature_importance", r))
        else:
            answer = "I haven't made a recommendation or prediction in this conversation yet, so there is nothing to explain."

    summary, grounding = None, {"checked": False}
    # No model summary for predictions: the rendered answer already states the prediction and its context, and a
    # summary there tended to describe it wrongly (e.g. "averaged over the test rows") in ways a number check cannot see.
    if source in ("model", "model_retry") and results and not all(t == "predict" for t, _ in results):
        try:
            evidence = [_jsonable(r) for _, r in results]
            text = planner.model_chat([{"role": "system", "content": NARRATE_SYSTEM},
                                       {"role": "user", "content": f"Question: {message}\nResults: {json.dumps(evidence, default=str)[:6000]}"}],
                                      max_tokens=350)
            text = text.split("</think>")[-1].strip()
            text = _check_significance(_drop_boilerplate(text, evidence, notes), evidence, notes)
            bad = explain.ungrounded(text, evidence, message)
            grounding = {"checked": True, "ungrounded": bad}
            if bad:
                notes.append(f"The model's summary cited numbers not in the results ({', '.join(bad[:5])}); it was not shown.")
            elif text:
                summary = text
        except Exception as e:  # noqa: BLE001
            notes.append(f"No model summary ({type(e).__name__}).")
    if summary:
        answer = _compose(answer, summary)
    for t, r in results:          # the group a ranking put first, for "why is that one better?"
        if t == "compute_metric" and r.get("group_by") and r.get("top"):
            memory["last_focus"] = {"group_column": r["group_by"], "group": r["top"]["group"], "outcome": r.get("column")}
    charts, seen = [], set()
    for t, r in results:
        for c in present.charts(t, r):
            key = (c["x"], tuple(sorted(str(d.get(c["x"])) for d in c["data"])))
            if key not in seen:          # one chart per grouping, not one per tool
                seen.add(key)
                charts.append(c)
    return {"answer": answer.strip(), "plan": plan, "plan_source": source, "first_pass": first_pass,
            "calls": calls_out, "notes": notes, "charts": charts,
            "grounding": grounding, "model_available": bool(model_on), "elapsed_ms": int((time.monotonic() - started) * 1000)}


def _model_plan(message: str, df: pd.DataFrame, memory: dict, notes: list[str]) -> tuple[dict | None, str, dict]:
    """The model's plan, with its first pass recorded as it was. A decline or
    clarification that contradicts the dataset gets one reconsideration, shown
    to the model with the reason; if that also fails, None (the deterministic
    planner answers, and the notes say so). Tool calls are validated later."""
    msgs = planner.build_prompt(message, df, memory)
    raw = planner.model_chat(msgs)
    plan = planner.parse_plan(raw)
    problem = None if plan is None else plan_checks.contradiction(plan, df, message)
    first = {"output": raw[:1500], "plan": plan, "valid": plan is not None and problem is None,
             "error": "not a valid JSON plan" if plan is None else (f"contradicts the dataset: {problem}" if problem else None)}
    if plan is None:
        notes.append("The model's plan was not valid JSON; the deterministic planner was used.")
        return None, "rules", first
    if problem is None:
        return plan, "model", first
    retry_msgs = msgs + [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Your plan was rejected by validation against the dataset: {problem}. "
                                                     "Reconsider using only the schema above and reply with one JSON plan."}]
    raw2 = planner.model_chat(retry_msgs)
    plan2 = planner.parse_plan(raw2)
    problem2 = None if plan2 is None else plan_checks.contradiction(plan2, df, message)
    first["retry"] = {"output": raw2[:1500], "plan": plan2, "valid": plan2 is not None and problem2 is None,
                      "error": "not a valid JSON plan" if plan2 is None else (f"contradicts the dataset: {problem2}" if problem2 else None)}
    if plan2 is not None and problem2 is None:
        notes.append(f"The model's first plan contradicted the dataset ({problem}); it reconsidered once.")
        return plan2, "model_retry", first
    notes.append(f"The model's plan contradicted the dataset ({problem}) and its reconsideration did not resolve it; "
                 "the deterministic planner was used.")
    return None, "rules", first


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


_ROUTED_RENDER = {"drivers": present.drivers, "understand": present.understand, "missing_values": present.missing_values, "outliers": present.outliers,
                  "patterns": present.patterns, "trend": present.trend, "column_change": present.scenario,
                  "conversion_change": present.scenario, "rate_change": present.scenario,
                  "budget_ranked": lambda r: f"**Scope:** {r['chosen_by']}.\n\n" + explain.render("budget_scenario", r)}


def _narrate(message: str, evidence: list, notes: list[str]) -> tuple[str | None, dict]:
    """A short model-written summary of computed results, shown only if every number in it is grounded."""
    try:
        text = planner.model_chat([{"role": "system", "content": NARRATE_SYSTEM},
                                   {"role": "user", "content": f"Question: {message}\nResults: {json.dumps(evidence, default=str)[:6000]}"}],
                                  max_tokens=350)
        text = text.split("</think>")[-1].strip()
        if text.startswith("{") or planner.parse_plan(text) is not None:
            notes.append("The model replied with a plan instead of a summary; no summary is shown.")
            return None, {"checked": True, "ungrounded": [], "discarded": "not prose"}
        text = _drop_boilerplate(text, evidence, notes)
        text = _check_significance(text, evidence, notes)
        bad = explain.ungrounded(text, evidence, message)
        if bad:
            notes.append(f"The model's summary cited numbers not in the results ({', '.join(bad[:5])}); it was not shown.")
            return None, {"checked": True, "ungrounded": bad}
        return (text or None), {"checked": True, "ungrounded": []}
    except Exception as e:  # noqa: BLE001
        notes.append(f"No model summary ({type(e).__name__}).")
        return None, {"checked": False}


def _ask_routed(message, df, memory, routed, model_on, filename, started) -> dict:
    label = routed.get("reason") if routed["type"] == "router" else routed.get("router")
    notes = [f"Answered by the Silt tool router ({label}): the fine-tuned planner "
             "has no tool for this kind of request, so it was not asked."]
    calls, charts, grounding, answer = [], [], {"checked": False}, ""
    if routed["type"] == "router":
        call = {"tool": routed["tool"], "args": routed["args"]}
        try:
            r = router.run(routed, df, memory, filename)
            calls.append({**call, "status": "ok", "result": _jsonable(r)})
            body = (present.suggestions_only(r) if routed["args"].get("focus") == "suggestions"
                    else _ROUTED_RENDER[routed["tool"]](r))
            charts = present.charts("budget_scenario" if routed["tool"] == "budget_ranked" else routed["tool"], r)
            summary = None
            if model_on and routed["tool"] not in ("understand",):
                summary, grounding = _narrate(message, [_jsonable(r)], notes)
            answer = _compose(body, summary)
        except (ml.Unsuitable, business.Insufficient) as e:
            calls.append({**call, "status": "refused", "reason": str(e)})
            answer = f"**I can't do that with this data.** {e}"
        except Exception as e:  # noqa: BLE001
            logger.exception("routed tool failed")
            calls.append({**call, "status": "error", "reason": f"{type(e).__name__}: {e}"})
            answer = f"That analysis failed ({type(e).__name__}). Try rephrasing, or choose another column."
    elif routed["type"] == "clarify":
        opts = routed.get("options") or []
        answer = routed["question"] + ("\n\nOptions: " + ", ".join(f"`{o}`" for o in opts) if opts else "")
    else:
        answer = "**The available data is not sufficient for this.** " + routed.get("reason", "")
    return {"answer": answer.strip(), "plan": routed, "plan_source": "router", "first_pass": None, "calls": calls,
            "notes": notes, "charts": charts, "grounding": grounding, "model_available": bool(model_on),
            "elapsed_ms": int((time.monotonic() - started) * 1000)}


def _ask_document(message: str, meta: dict, model_on: bool, started: float) -> dict:
    notes = ["Answered from the uploaded document by text search; quotes keep their page numbers."]
    grounding = {"checked": False}
    if _DOC_SUMMARY.search(message):
        ov = documents.overview(meta)
        answer, calls = present.doc_overview(ov), [{"tool": "document_overview", "status": "ok", "result": ov}]
        hits = documents.search(meta, " ".join(ov["key_terms"][:8]), k=3)
    else:
        hits = documents.search(meta, message, k=4)
        answer = present.doc_passages(message, hits, meta)
        calls = [{"tool": "document_search", "status": "ok",
                  "result": {"hits": [{"page": h["page"], "score": h["score"], "matched": h["matched"]} for h in hits]}}]
    if model_on and hits:
        try:
            context = "\n\n".join(f"[p. {h['page']}] {h['text']}" for h in hits)
            text = planner.model_chat([{"role": "system", "content": DOC_SYSTEM},
                                       {"role": "user", "content": f"Question: {message}\n\nPassages:\n{context[:6000]}"}],
                                      max_tokens=300)
            text = text.split("</think>")[-1].strip()
            if text.startswith("{") or planner.parse_plan(text) is not None:
                notes.append("The model replied with a plan instead of an answer; only the quoted passages are shown.")
                grounding = {"checked": True, "discarded": "not prose"}
            else:
                bad = explain.ungrounded(text, [h["text"] for h in hits] + [f"p. {h['page']}" for h in hits], message)
                grounding = {"checked": True, "ungrounded": bad}
                if bad:
                    notes.append(f"The model's answer cited numbers not in the passages ({', '.join(bad[:5])}); it was not shown.")
                elif text:
                    answer = f"{text}\n\n{SUMMARY_LABEL}\n\n{answer}"
        except Exception as e:  # noqa: BLE001
            notes.append(f"No model answer ({type(e).__name__}); the passages are shown.")
    return {"answer": answer, "plan": {"type": "document"}, "plan_source": "document", "first_pass": None,
            "calls": calls, "notes": notes, "charts": [], "grounding": grounding, "model_available": bool(model_on),
            "elapsed_ms": int((time.monotonic() - started) * 1000)}


CALC_NARRATE = ("You are Silt Copilot. Explain the calculation results below to a business user in 4-8 sentences: "
                "answer the question first, then how it was calculated (formulas and which records were excluded and "
                "why). For a 'why' question use the explanation given. State the type_of_answer given (a calculation from historical data, not a machine-learning prediction) when the question asks. Use ONLY numbers "
                "that appear in the results; do not compute new numbers; never claim causation. Plain prose, no JSON.")


def kept_mask(spec: dict, df: pd.DataFrame) -> pd.Series:
    """The records a calculation plan keeps (after its exclusions), for checking it against the question's scope."""
    frame = calc.run_derive(df, spec["derive"])
    keep = pd.Series(True, index=df.index)
    for e in spec["exclude"]:
        hit = calc.evaluate(e["where"], frame, df)
        if isinstance(hit, pd.Series):
            keep &= ~hit.fillna(False).astype(bool)
    return keep


def _records_used(out: dict) -> int | None:
    """How many records the answer's computation actually used, when the result says (min over its tools)."""
    counts = []
    for c in out.get("calls", []):
        r = c.get("result")
        if not isinstance(r, dict) or c.get("status") != "ok":
            continue
        for k in ("rows_valid", "rows_used", "records_used", "rows"):
            v = r.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                counts.append(v)
                break
    return min(counts) if counts else None


def _hold_to_scope(spec: dict, df: pd.DataFrame, sc: dict | None, notes: list[str]) -> list[str]:
    """Problems left after holding a calculation plan to the question's scope. A plan that kept records outside the
    scope (a dropped name or period filter) gets the scope's own filters added -- they come from the question, not
    from the model -- and is checked again."""
    if not scoping.active(sc):
        return []
    try:
        probs = scoping.verify_rows(kept_mask(spec, df), df, sc)
    except calc.CalcError as e:
        return [str(e)]
    if not probs:
        return []
    before = list(spec["exclude"])
    kept = scoping.replace_conflicting(before, sc)
    dropped = [e for e in before if e not in kept]
    have = {e["where"].replace(" ", "") for e in kept}
    added = [e for e in scoping.as_exclusions(sc) if e["where"].replace(" ", "") not in have]
    spec["exclude"] = kept + added
    if dropped:
        notes.append("An earlier selection was replaced by this question's scope: "
                     + ", ".join(f"`{e['where']}`" for e in dropped) + ".")
    notes.append("Silt kept the question's scope (" + scoping.describe(sc) + "): the plan "
                 + "; ".join(probs) + ", so " + ", ".join(f"`{e['where']}`" for e in added) + " was added.")
    return scoping.verify_rows(kept_mask(spec, df), df, sc)


def _ask_calc(message: str, df: pd.DataFrame, memory: dict, why: str, started: float,
              previous: dict | None = None, sc: dict | None = None) -> dict | None:
    """The model plans a calculation (app/analyst/calc.py), the calculator runs it, the model explains the result.
    None when the model could not produce a plan that passes the checks and there is an earlier answer to keep."""
    notes = [f"Planned by the local analyst model as a calculation ({why}); every number was computed by Silt's "
             "calculator from the data."]
    try:
        spec, trace = calc.model_plan(message, df, memory)
    except Exception as e:  # noqa: BLE001
        logger.exception("calc planning failed")
        if previous is not None:
            previous["notes"].append(f"The model could not plan a calculation either ({type(e).__name__}).")
            return previous
        return None
    for t in trace:
        if t.get("reused_from_conversation"):
            notes.append("Reused from earlier in this conversation: " + ", ".join(
                f"{k} = {v}" for k, v in t["reused_from_conversation"].items()) + ".")
    rejected = [t for t in trace if t.get("accepted") is False and not t.get("fixed_by_silt")]
    for t in trace:
        if t.get("carried_over"):
            notes.append("Kept from the previous question: " + ", ".join(f"`{w}`" for w in t["carried_over"])
                         + " (the follow-up did not name anyone else).")
        if t.get("repaired"):
            notes.append(f"Silt corrected a small mistake in the model's {t['section']}: " + "; ".join(t["repaired"]) + ".")
    for t in rejected:
        notes.append(f"Validation rejected the model's {t['section']} ({'; '.join(t['problems'])[:300]}); "
                     "it was shown the reason and asked to rewrite it.")
    first_pass = {"output": json.dumps([{k: t.get(k) for k in ("section", "attempt", "output", "accepted", "problems")}
                                        for t in trace])[:4000],
                  "plan": spec, "valid": not rejected, "error": rejected[0]["problems"][0] if rejected else None}
    if spec.get("type") != "calc":
        notes[0] = f"The local analyst model was asked to plan this as a calculation ({why}); no calculation was run."
    base = {"plan": {"type": "calc", **({k: spec.get(k) for k in ("derive", "exclude", "group_by", "aggregate")}
                                         if spec.get("type") == "calc" else spec)},
            "plan_source": "model_calc", "first_pass": first_pass, "model_available": True}
    if spec["type"] == "failed":
        if previous is not None:
            previous["notes"].append(f"The model also tried to plan this as a calculation, but {spec['reason']}.")
            return previous
        base["plan_source"] = "model_calc_failed"
        from app.analyst import overview
        sugg = overview.suggestions(df, overview._target_candidates(df))[:3]  # noqa: SLF001
        answer = ("**I couldn't build a calculation for that which I can verify, so I haven't guessed a number.** "
                  "Could you say it a little more specifically — which value, for whom, and over which records? "
                  "For example:\n\n" + "\n".join(f"- {q}" for q in sugg))
        notes.append(f"Why: the model's plan did not pass Silt's checks after repairs and retries ({spec['reason'][:400]}).")
        return {**base, "answer": answer, "calls": [], "notes": notes, "charts": [], "grounding": {"checked": False},
                "details": "", "elapsed_ms": int((time.monotonic() - started) * 1000)}
    if spec["type"] == "clarify":
        return {**base, "answer": spec.get("question") or "Could you clarify?", "calls": [], "notes": notes,
                "charts": [], "grounding": {"checked": False}, "elapsed_ms": int((time.monotonic() - started) * 1000)}
    if spec["type"] == "decline":
        if previous is not None:
            return previous
        return {**base, "answer": "**The available data is not sufficient for this.** " + str(spec.get("reason", "")),
                "calls": [], "notes": notes, "charts": [], "grounding": {"checked": False},
                "elapsed_ms": int((time.monotonic() - started) * 1000)}
    left = _hold_to_scope(spec, df, sc, notes)
    if left:
        return {**base, "plan_source": "model_calc_failed", "answer": "**I couldn't keep this calculation to the "
                f"records you asked about ({scoping.describe(sc)}), so I haven't shown a number.** " + "; ".join(left)
                + ".", "calls": [], "notes": notes, "charts": [], "grounding": {"checked": False}, "details": "",
                "elapsed_ms": int((time.monotonic() - started) * 1000)}
    added = metrics.validity_rules(spec, df)
    if added:
        spec["exclude"] = list(spec["exclude"]) + added
        base["plan"]["exclude"] = spec["exclude"]
        notes.append("Silt left out records that cannot be valid for this calculation (same rule for every answer): "
                     + "; ".join(e["reason"].split(" — ")[0] for e in added) + ".")
    try:
        result = calc.run(spec, df, question=message)
    except Exception as e:  # noqa: BLE001 - CalcError, or anything unexpected: report it, never crash the turn
        return {**base, "answer": f"**The calculation could not be run:** {e}", "calls": [], "notes": notes,
                "charts": [], "grounding": {"checked": False}, "elapsed_ms": int((time.monotonic() - started) * 1000)}
    return _calc_reply(message, spec, result, base, notes, memory, True, started)


def _calc_reply(message: str, spec: dict, result: dict, base: dict, notes: list[str], memory: dict, model_on: bool,
                started: float, extra_meta: dict | None = None) -> dict:
    """A calculator result as an answer, and as the conversation's latest calculation (for follow-ups)."""
    if extra_meta:
        result.update(extra_meta)
    memory["last_calc"] = {"question": message, "spec": spec, "result": result,
                           "meta": {k: result.get(k) for k in ("primary", "percent", "metric_label", "definition")}}
    memory.setdefault("calc_history", []).append(
        (message, {k: spec[k] for k in ("type", "derive", "exclude", "group_by", "aggregate")}))
    del memory["calc_history"][:-3]
    if result.get("group_by") and result.get("groups"):
        first_agg = result.get("primary") or next(iter(result["aggregates"]), None)
        memory["last_focus"] = {"group_column": result["group_by"], "group": result["groups"][0]["group"],
                                "outcome": first_agg}
    evidence = calc_evidence(result)
    summary, grounding = _narrate_calc(message, evidence, notes) if model_on else (None, {"checked": False})
    answer = _compose(present.calc(result, message), summary)
    return {**base, "answer": answer, "calls": [{"tool": "calculator", "args": base["plan"], "status": "ok",
                                                  "result": _jsonable(result)}],
            "notes": notes, "charts": present.calc_charts(result), "grounding": grounding,
            "details": present.calc_details(result), "elapsed_ms": int((time.monotonic() - started) * 1000)}


_SCENARIO_Q = re.compile(r"\bwhat if\b|\bwhat would\b|\bsuppose\b|\bif (?:the |our |his |her |their )?\w+(?: \w+)? "
                         r"(?:rises?|increases?|decreases?|falls?|drops?|goes up|goes down|doubles?|halves?|changes?)\b|"
                         r"\b(?:increases?|decreases?|raises?|cuts?|rises?|drops?) (?:\w+ ){0,3}by \d", re.I)


def _ask_metric(message: str, df: pd.DataFrame, memory: dict, model_on: bool, started: float,
                sc: dict | None) -> dict | None:
    """A recognised business metric that is not a column (labor cost, conversion rate, ROI, ...): its standard
    definition is resolved against this data's columns and run by the calculator. No language model is needed; when
    an input is missing or ambiguous the user is asked, and no other quantity is substituted."""
    p = metrics.plan(message, df) or metrics.count_plan(message, df) or metrics.formula_plan(message, df) \
        or metrics.event_rate_plan(message, df)
    if p is None:
        return None
    if not p.clarify:
        left = metrics.unhandled_conditions(message, df, p)
        if left:
            if model_on:
                # the question defines its own conditions: the validated calculator plans it -- and if it cannot,
                # no other quantity may be answered instead (see _dispatch)
                memory["_deferred_metric"] = {"label": p.label, "definition": p.definition, "unhandled": left}
                return None
            return _reply(f"I can compute **{p.label}** ({p.definition}), but your question "
                          "also asks for " + "; ".join(f"*{x}*" for x in left) + ", which I can't apply without the local "
                          "model (it is offline). I haven't shown a number that ignores it. Ask without that condition, "
                          "or try again when the model is available.", "metric", model_on, started,
                          plan={"type": "clarify", "metric": p.name, "unhandled": left})
    if p.clarify:
        return _reply(p.clarify, "metric", model_on, started, plan={"type": "clarify", "metric": p.name},
                      notes=[f"Silt recognised the metric **{p.label}** but could not resolve its inputs in this data "
                             "without guessing, so nothing was calculated."])
    spec = p.spec
    notes = ([f"**{p.label}** is not a column of this data, so Silt derived it with its standard definition: "
              f"{p.definition}. The calculation was resolved and run by Silt's deterministic calculator — no language "
              "model was needed."] if metrics.recognise(message) else
             [f"Computed by Silt's deterministic calculator: {p.definition}."]) + p.notes
    base = {"plan": {"type": "calc", **{k: spec.get(k) for k in ("derive", "exclude", "group_by", "aggregate")}},
            "plan_source": "metric", "first_pass": {"plan": spec, "valid": True, "error": None, "output": "deterministic"},
            "model_available": model_on}
    left = _hold_to_scope(spec, df, sc, notes)
    if left:
        return {**base, "plan_source": "tool", "answer": "**I couldn't keep this calculation to the records you asked "
                f"about ({scoping.describe(sc)}), so I haven't shown a number.** " + "; ".join(left) + ".",
                "calls": [], "notes": notes, "charts": [], "grounding": {"checked": False}, "details": "",
                "elapsed_ms": int((time.monotonic() - started) * 1000)}
    try:
        result = calc.run(spec, df, question=message)
    except Exception as e:  # noqa: BLE001 - a tool failure is reported as one, never as a data limitation
        logger.exception("metric calculation failed")
        return _reply(f"**The calculation of {p.label} failed** ({type(e).__name__}: {e}). This is a tool error, not a "
                      "limitation of your data; nothing was estimated instead.", "tool", model_on, started,
                      plan={"type": "error"}, notes=notes)
    return _calc_reply(message, spec, result, base, notes, memory, model_on, started,
                       {"primary": p.name, "percent": p.percent, "metric_label": p.label, "definition": p.definition})


_REF_DUPES_OUT = re.compile(r"^\W*(?:and |now |then |also |please )*(?:exclud\w*|remov\w*|drop\w*|ignor\w*|without|no|"
                            r"de-?dup\w*)\b[^.?!]{0,25}\bduplicat\w*", re.I)
_REF_DUPES_IN = re.compile(r"\b(?:includ\w*|keep\w*|count\w*|with)\b[^.?!]{0,15}\bduplicat\w*", re.I)
_REF_OVERNIGHT = re.compile(r"\b(?:treat|count|consider|they are|these are|those are|as)\b[^.?!]{0,30}\bovernight\b|"
                            r"^\W*overnight shifts?\W*$", re.I)
_REF_SCOPE = re.compile(r"^\W*(?:and |now |then |ok(?:ay)?,? |so )*(?:only|just|what about|how about|same for|and for|for|"
                        r"show (?:only|just)|now for|now only|excluding|except|use|try|switch to|make it|change it to)\b",
                        re.I)
_REF_ALL = re.compile(r"^\W*(?:and |now |then |ok(?:ay)?,? )*(?:(?:for|show|what about|how about|across) )?(?:every(?:one|body)|"
                      r"all (?:the )?(?:employees|staff|people|records|rows|teams|departments|groups|of them)|overall|"
                      r"the whole (?:team|company|dataset))\W*$", re.I)
_REF_GROUP = re.compile(r"^\W*(?:and |now |then |ok(?:ay)?,? )*(?:by|per|break (?:it|that|this) down by|split (?:it )?by|"
                        r"group(?:ed)? by|for each|show (?:it|that) by)\b", re.I)


def _refine_last(message: str, df: pd.DataFrame, memory: dict, model_on: bool, started: float,
                 sc: dict | None) -> dict | None:
    """A short follow-up that changes the previous calculation -- "Only Rahul.", "Exclude duplicates.", "By
    department.", "Treat them as overnight shifts." -- re-runs that same calculation with the change, deterministically.
    Only when the latest analysis was a calculation, and the message asks for nothing else."""
    last = memory.get("last_calc")
    if not last or memory.get("last_analysis") != "calc" or len(message.split()) > 12 or metrics.recognise(message) \
            or _SCENARIO_Q.search(message) or followup.wants_forecast(message) or re.search(r"\bpredict\w*|%", message, re.I) \
            or router._numeric_named(df, message.lower()):  # noqa: SLF001 - a new measure is a new question
        return None
    spec = json.loads(json.dumps(last["spec"]))
    spec.setdefault("exclude", [])
    changes = []
    if _REF_DUPES_OUT.search(message):
        if any("duplicate()" in e["where"] for e in spec["exclude"]):
            changes.append("duplicates were already excluded")
        else:
            spec["exclude"].insert(0, {"reason": "exact duplicate of an earlier record", "where": "duplicate()"})
            changes.append("exact duplicates excluded")
    elif _REF_DUPES_IN.search(message):
        before = len(spec["exclude"])
        spec["exclude"] = [e for e in spec["exclude"] if "duplicate()" not in e["where"]]
        changes.append("duplicates included" if len(spec["exclude"]) < before else "duplicates were already included")
    if _REF_OVERNIGHT.search(message):
        if any("hours_between(" in v for v in spec["derive"].values()):
            spec["derive"] = {k: v.replace("hours_between(", "overnight_hours(") for k, v in spec["derive"].items()}
            spec["exclude"] = [e for e in spec["exclude"] if not re.search(r"hours_between\([^)]*\)\s*<\s*0", e["where"])]
            changes.append("an end time earlier than the start counted as the next day (overnight shift)")
        else:
            return None
    if _REF_GROUP.search(message):
        g, phrase = metrics.group_column(re.sub(r"^\W*(?:and |now |then )*", "", message), df)
        if g is None:
            return _reply(f"'{phrase or message.strip()}' is not a column I can break the previous result down by. "
                          "Columns available: " + ", ".join(f"`{c}`" for c in df.columns if df[c].dtype == object)[:400],
                          "tool", model_on, started, plan={"type": "clarify"})
        spec["group_by"] = g
        changes.append(f"broken down by {g.replace('_', ' ')}")
    named_words = {w.lower() for vals in ((sc or {}).get("entities") or {}).values() for v in vals for w in re.findall(r"[A-Za-z]+", v)}
    other_words = [w for w in re.findall(r"[A-Za-z]+", message) if w.lower() not in named_words]
    scope_change = scoping.active(sc) and not sc.get("inherited") and len(re.findall(r"[A-Za-z]+", message)) <= 8 and \
        (bool(_REF_SCOPE.search(message)) or (not changes and len(other_words) <= 1))
    if scope_change:
        spec["exclude"] = scoping.replace_conflicting(spec["exclude"], sc)
        if spec.get("group_by") in (sc.get("entities") or {}):
            spec["group_by"] = None
        changes.append(f"limited to {scoping.describe(sc)}")
    elif _REF_ALL.search(message) and not changes:
        spec["exclude"] = [e for e in spec["exclude"] if not scoping._SEL.match(e.get("where", ""))]  # noqa: SLF001
        changes.append("for all records (the earlier selection removed)")
    if not changes:
        return None
    meta = dict(last.get("meta") or {})
    notes = [f"Re-ran the previous calculation (*{last['question']}*) with one change: {'; '.join(changes)}. Every "
             "other step — formula, exclusions, grouping — is the same, and it was run by Silt's calculator without a "
             "language model."]
    base = {"plan": {"type": "calc", **{k: spec.get(k) for k in ("derive", "exclude", "group_by", "aggregate")}},
            "plan_source": "refine", "first_pass": {"plan": spec, "valid": True, "error": None, "output": "deterministic"},
            "model_available": model_on}
    if scope_change:
        left = _hold_to_scope(spec, df, sc, notes)
        if left:
            return None
    try:
        result = calc.run(spec, df, question=message)
    except Exception as e:  # noqa: BLE001
        return _reply(f"**Re-running the previous calculation failed** ({type(e).__name__}: {e}). Nothing was "
                      "estimated instead.", "tool", model_on, started, plan={"type": "error"}, notes=notes)
    return _calc_reply(message, spec, result, base, notes, memory, model_on, started,
                       {k: v for k, v in meta.items() if v})


def calc_evidence(result: dict) -> dict:
    groups = []
    for g in result.get("groups", [])[:25]:
        row = {k: v for k, v in g.items() if not k.startswith("share_of")}
        row.update({k.replace("share_of_", "percent_of_total_"): round(v * 100, 1) for k, v in g.items()
                    if k.startswith("share_of") and v is not None})
        groups.append(row)
    return {"type_of_answer": ("deterministic scenario calculation from historical data under the stated assumptions; "
                               "NOT a machine-learning prediction") if result.get("answer_type") == "scenario" else
            "deterministic calculation on the data; not a machine-learning prediction",
            "assumptions": result.get("assumptions"), "data_values_used": result.get("inputs_used"),
            "records_in_file": result["rows_total"], "records_excluded": result["rows_excluded"],
            "records_used": result["rows_valid"], "formulas": result["formulas"], "aggregates": result["aggregates"],
            "exclusions": [{"reason": e["reason"], "rule": e["where"], "records": e["rows"]} for e in result["exclusions"]],
            "excluded_records": [{"line": r["row"], "because": r["excluded_because"]} for r in result["excluded_rows"][:20]],
            "totals": result["totals"], "by_" + str(result.get("group_by") or "group"): groups,
            "data_problems_included": [f"lines {c['lines']}: {c['issue']}" for c in result.get("data_checks") or []],
            **({"explanation": result["why"]["sentence"]} if result.get("why") else {})}


def _narrate_calc(message: str, evidence: dict, notes: list[str]) -> tuple[str | None, dict]:
    try:
        text = planner.model_chat([{"role": "system", "content": CALC_NARRATE},
                                   {"role": "user", "content": f"Question: {message}\nResults: {json.dumps(evidence, default=str)[:6000]}"}],
                                  max_tokens=450)
        text = text.split("</think>")[-1].strip()
        if text.startswith("{") or planner.parse_plan(text) is not None or calc.parse(text) is not None:
            notes.append("The model replied with a plan instead of an explanation; no model explanation is shown.")
            return None, {"checked": True, "ungrounded": [], "discarded": "not prose"}
        boiler = re.compile(r"(?:[^.]|\.(?=\d))*\b(compares averages only|statistical test|why_the_top|explanation\" for)(?:[^.]|\.(?=\d))*\.\s*", re.I)
        if boiler.search(text):
            text = boiler.sub("", text).strip()
            notes.append("A sentence of the model's explanation that referred to a statistical test or to internal "
                         "field names, neither of which applies here, was removed.")
        wrong = rank_contradictions(text, evidence)
        if wrong:
            notes.append("The model's explanation contradicted the computed ranking (" + "; ".join(wrong[:3])
                         + "); it was not shown. The computed results below are complete.")
            return None, {"checked": True, "ungrounded": [], "contradictions": wrong}
        bad = explain.ungrounded(text, evidence, message)
        if bad:
            notes.append(f"The model's explanation cited numbers the calculator did not produce ({', '.join(bad[:5])}); "
                         "it was not shown. The computed results below are complete.")
            return None, {"checked": True, "ungrounded": bad}
        return (text or None), {"checked": True, "ungrounded": []}
    except Exception as e:  # noqa: BLE001
        notes.append(f"No model explanation ({type(e).__name__}).")
        return None, {"checked": False}


def rank_contradictions(text: str, evidence: dict) -> list[str]:
    """Claims that a group has "the highest/most ..." of a measure when the calculator ranked it otherwise."""
    groups = next((v for k, v in evidence.items() if k.startswith("by_") and isinstance(v, list)), None)
    if not groups:
        return []
    measures: dict[str, dict[str, float]] = {}
    for g in groups:
        for k, v in list(g.items()) + list((g.get("per_row") or {}).items()):
            if isinstance(v, (int, float)) and k not in ("group",) and not isinstance(v, bool):
                measures.setdefault(k, {})[str(g["group"])] = v
    out = []
    low = text.lower()
    for k, vals in measures.items():
        if len(vals) < 2:
            continue
        phrase = "[ _]".join(map(re.escape, k.replace("_", " ").lower().split()))
        for word, best in (("highest|largest|most|biggest", max(vals, key=vals.get)),
                           ("lowest|smallest|least|fewest", min(vals, key=vals.get))):
            for m in re.finditer(rf"\b(?:the )?(?:{word})\s+(?:{phrase})", low):
                starts = [x.end() for x in re.finditer(r"[.!?;](?:\s|$)", low[:m.start()])]
                sentence = low[(starts[-1] if starts else 0):m.end()]
                named = [g for g in vals if g.lower() in sentence]
                if named and best not in named:
                    out.append(f"'{m.group(0)}' is {best}, not {named[-1]}")
    return out


_NO_DIFF = re.compile(r"(?:[^.]|\.(?=\d))*\b(do(es)? not show a reliable|no reliable|no (statistically )?significant|not (statistically )?"
                      r"significant|not a reliable)(?:[^.]|\.(?=\d))*\.\s*", re.I)
_SIG = re.compile(r"(?:[^.]|\.(?=\d))*\b(?<!not )(?<!no )(statistically significant|a reliable difference|significantly)(?:[^.]|\.(?=\d))*\.\s*", re.I)


def _p_values(obj, out=None) -> list[float]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("p_value", "p") and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            else:
                _p_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _p_values(v, out)
    return out


def _check_significance(text: str, evidence, notes: list[str]) -> str:
    """The adapter tends to append "the data do not show a reliable difference" whatever the test said. A sentence
    about significance that contradicts the computed p-values (or where no test was run) is removed and noted."""
    ps = _p_values(evidence)
    removed = False
    if (not ps or min(ps) < 0.05) and _NO_DIFF.search(text):
        text, removed = _NO_DIFF.sub("", text).strip(), True
    if (not ps or min(ps) >= 0.05) and _SIG.search(text) and not _NO_DIFF.search(text):
        text, removed = _SIG.sub("", text).strip(), True
    if removed:
        notes.append("A sentence of the model's summary about statistical significance contradicted the computed test "
                     "(or no test was run) and was removed.")
    return text


_BOILER = re.compile(r"(?:[^.]|\.(?=\d))*\b(compares averages only|statistical test)(?:[^.]|\.(?=\d))*\.\s*", re.I)


def _drop_boilerplate(text: str, evidence, notes: list[str]) -> str:
    """The adapter learned to append "This compares averages only; see the statistical test ..." to summaries.
    Where no test was run that sentence is false, so it is removed, and the removal is noted."""
    blob = json.dumps(evidence, default=str).lower()
    if _BOILER.search(text) and not re.search(r'"p(_value)?"|welch|anova|chi|t-test|kruskal|mann', blob):
        notes.append("A sentence of the model's summary referring to a statistical test that was not run here "
                     "was removed.")
        return _BOILER.sub("", text).strip()
    return text
