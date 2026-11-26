"""One analyst turn: plan -> validate -> execute -> explain."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

import re

from app.analyst import business, calc, conversation, documents, explain, followup, intake, metrics, ml, objectives, plan_checks, planner, present, router, scenario_bridge, tools, whatif_bridge
from app.analyst import nl_planner
from app.analyst import program as P
from app.analyst import scope as scoping

logger = logging.getLogger(__name__)
MAX_CONVERSATIONS = 200
_MEMORY: "OrderedDict[str, dict]" = OrderedDict()

SUMMARY_LABEL = ("_Written by the local model; every number in it was checked against the computed results._")
NARRATE_SYSTEM = ("You are Silt's analyst. Write a short plain-language summary (3-6 sentences) of the analysis "
                  "results below for a business user. Use ONLY numbers that appear in the results; do not compute "
                  "new numbers; say when evidence is weak or a difference is not significant; never claim causation.")


def memory_for(conversation_id: str, persist: bool = False) -> dict:
    """Per-conversation state (trained model, last recommendation), in process. With `persist` (the API route), the
    conversation's result register and open objective are restored after a restart from that key's own file
    (app/analyst/persist.py) -- never from another key's."""
    mem = _MEMORY.get(conversation_id)
    if mem is None:
        mem = _MEMORY[conversation_id] = {}
        if persist:
            from app.analyst import persist as _persist
            mem.update(_persist.load(conversation_id))
        while len(_MEMORY) > MAX_CONVERSATIONS:
            _MEMORY.popitem(last=False)
    _MEMORY.move_to_end(conversation_id)
    return mem


DOC_SYSTEM = ("You answer questions about a document using ONLY the passages given. Cite pages as (p. N). If the "
              "passages do not answer the question, say so plainly. Do not add facts or numbers that are not in the "
              "passages, and do not give a reason or cause unless a passage states it; say which facts the passages "
              "give instead. Write 2-5 sentences of plain prose, not JSON.")
DOC_SUMMARY_LABEL = ("_Written by the local model from the passages quoted below; every number in it was checked against "
                     "them. The quotes, with their pages, are the source._")
_DOC_WORDS = re.compile(r"\b(document|pdf|report|page|pages|paper|policy|contract|section|chapter|according to|"
                        r"the file says|in the text|mentions?)\b", re.I)
_DOC_SUMMARY = re.compile(r"summari[sz]e|what is (this|the) (document|pdf|report|file) about|overview|what('s| is) in "
                          r"(this|the) (document|pdf|report)|key points|main points", re.I)


def ask(message: str, df: pd.DataFrame | None, memory: dict, use_model: bool | None = None,
        filename: str | None = None, document: dict | None = None) -> dict:
    started = time.monotonic()
    model_on = planner.model_available() if use_model is None else use_model
    if df is not None:
        message = _comparatives(message, df)
    before = memory.get("_turn")
    out = _ask_turn(message, df, memory, use_model, model_on, filename, document, started)
    if out.get("plan_source") == "input_check":
        # unreadable input leaves the conversation exactly as it was -- not even the turn counter moves
        if before is None:
            memory.pop("_turn", None)
        else:
            memory["_turn"] = before
        return out
    _stamp(memory)
    return out


def _stamp(memory: dict) -> None:
    """Every conversation record carries the turn it was written in, so a later handler can tell whether it is still
    the latest result (memory["last_result"]) or an older one that a newer answer superseded."""
    t = memory.get("_turn", 0)
    for k in ("last_calc", "last_breakdown", "last_charts", "last_result"):
        v = memory.get(k)
        if isinstance(v, dict) and "turn" not in v:
            v["turn"] = t


def _superseded(memory: dict, key: str) -> bool:
    """`key` holds an older result than the latest answer: a follow-up must not refine or redraw it."""
    lr, v = memory.get("last_result"), memory.get(key)
    return bool(lr and isinstance(v, dict) and v.get("turn") is not None and lr.get("turn") is not None
                and v["turn"] < lr["turn"])


def _ask_turn(message, df, memory, use_model, model_on, filename, document, started) -> dict:
    memory["_turn"] = memory.get("_turn", 0) + 1
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
        if out.get("calls"):
            # a document's figures are quoted from its pages, not computed: they cannot be recomputed here
            out["verify_status"] = "not independently verified"
            out["answer"] += "\n\n" + _VERIFIED_LINE["not independently verified"].format(
                why="figures are quoted from the document's pages, not computed from data")
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
    # A file of several tables stacked in one sheet: the question is answered on ONE of them, chosen before any
    # measure, group or filter is read, so no other table's records or empty columns take part (app/analyst/tables.py)
    df, message, table_note = _logical_table(message, df, memory)
    # "within the highest-revenue region": the group is found from the data (shared ranking rule) and the question
    # then reads "within East" for every later step -- scope, planner and verifier see one, named group
    ranked = scoping.superlative_group(message, df) if not isinstance(table_note, dict) else None
    if ranked and ranked.get("clarify"):
        out = _reply(ranked["clarify"], "tool", model_on, started, plan={"type": "clarify"},
                     notes=["The question picks a group by rank, and the measure does not say whether to compare totals "
                            "or averages; nothing was calculated."])
        out = _finish(out, message, df, memory, None, started)
        conversation.record(memory, message, out)
        return out
    if ranked:
        a, b = ranked["span"]
        message = f"{message[:a]}{ranked['prep']} {ranked['value']}{message[b:]}"
        line = f"_“{ranked['phrase'].split(None, 1)[1]}” = **{ranked['column']} {ranked['value']}** ({ranked['text']})._"
        table_note = f"{table_note}\n\n{line}" if table_note else line
    if isinstance(table_note, dict):
        out = _finish(_reply(table_note["clarify"], "tool", model_on, started,
                             plan={"type": "clarify", "options": table_note.get("options") or []},
                             notes=[table_note["why"]]), message, df, memory, None, started)
        conversation.record(memory, message, out)
        return out
    # everyday measure words the columns name differently ("trip length", "visitors", "spend"): one matching column
    # is read in their place, and the answer says so; several are asked about
    from app.analyst import semantics as _sem
    mw = _sem.measure_words(message, df)
    if mw["ambiguous"]:
        w, cs = mw["ambiguous"][0]
        out = _reply(f"Which column do you mean by **{w}**?\n\nOptions: " + ", ".join(f"`{c}`" for c in cs), "tool",
                     model_on, started, plan={"type": "clarify", "options": cs},
                     notes=[f"'{w}' could be measured by {', '.join(cs)}; Silt asks rather than choosing one."])
        out = _finish(out, message, df, memory, None, started)
        conversation.record(memory, message, out)
        return out
    read_as = []
    for w, c in mw["rewrite"]:
        message = re.sub(rf"\b{re.escape(w)}\b", str(c), message, count=1, flags=re.I)
        read_as.append(f"“{w}” read as `{c}` (the only column measuring it)" if w not in _sem.MONEY_IN_WORDS else
                       f"“{w}” read as `{c}` — {_sem.money_in_columns(df)[1]}; say if you mean something else")
    chart_kind = followup.chart_request(message)
    stopped = _money_verb_gate(message, df, model_on, started)
    if stopped is None:
        stopped = _stated_gate(message, df, memory, model_on, started)
    if stopped is not None:
        out = _finish(stopped, message, df, memory, None, started)
        if read_as:
            out["notes"].insert(0, "Measure words: " + "; ".join(read_as) + ".")
        if table_note:
            out["answer"] = table_note + "\n\n" + out["answer"]
        conversation.record(memory, message, out)
        return out
    try:
        if memory.get("_said"):
            memory["_prev_said"] = memory["_said"]   # the previous request (context for the model decomposer)
        memory["_said"] = said               # the user's own words, before any reference was resolved in place
        out = _dispatch(message, df, memory, model_on, filename, started, chart_kind)
    except tools.RoleRefusal as e:
        # the aggregation asked for has no meaning for what the column is (an identifier averaged or ranked on, a
        # price or rate added up): said plainly, by whichever path met it -- never computed with a warning
        out = _reply("**I can't compute that.** " + str(e), "tool", model_on, started,
                     plan={"type": "decline", "role_refusal": True},
                     notes=["Refused by Silt's column-role rule (one rule for every tool and program step); nothing "
                            "was calculated."])
    except Exception as e:  # noqa: BLE001 - one failed path must not lose the conversation
        logger.exception("analyst turn failed")
        out = _reply(f"Something went wrong while answering that ({type(e).__name__}). Your conversation is kept; "
                     "try rephrasing, or ask a narrower question.", "rules", model_on, started,
                     notes=[f"Internal error: {type(e).__name__}: {e}"[:300]])
    if _dead_end(out) or (len(_intent_clauses(message)) >= 2 and _role_failure(out)):
        # No path turned the question into an analysis. The structured reading (app/analyst/intent.py) may still
        # define one exactly: a count, a distinct count, a rate, a combined total, a relation between two measures,
        # or a question in several clauses. Its program runs only if it passes the same verification as any answer.
        routed = _followup_from_last(message, df, memory, model_on, started) or \
            _spec_route(message, df, memory, model_on, started) or \
            _compound_followup(message, df, memory, model_on, started, filename) or \
            _model_followup(message, df, memory, model_on, started, filename)
        if routed is not None:
            out = routed
    if (out.get("plan") or {}).get("type") == "clarify" and (out["plan"].get("options") or []) and \
            not any(c.get("status") == "ok" for c in out.get("calls", [])):
        miss = _missing_terms(message, df)
        if miss and f"**{miss[0][1]}**" not in out["answer"]:
            out["answer"] = (f"This data has no column called **{miss[0][1]}**, so I haven't used one in its place. "
                             + out["answer"])
    out = _verify_answer(out, message, df, memory, model_on, started)
    if out.get("parts_status"):
        # a question in several parts: its status is never more than its weakest part, and each part is named
        ps = out["parts_status"]
        line = "; ".join(f"part {i}: " + {"verified": "✓ recomputed independently with pandas",
                                           "not independently verified": "not independently verified",
                                           "refused": "not computed (refused)",
                                           "needs your answer": "not computed (needs your answer)"}[st] for i, st in ps)
        sts = {st for _, st in ps}
        if out.get("verify_status") not in ("withheld",):
            out["verify_status"] = ("verified" if sts == {"verified"} else
                                    "partially verified" if "verified" in sts else
                                    "not independently verified")
        for old_line in (_VERIFIED_LINE["verified"],):
            out["answer"] = out["answer"].replace(old_line, "")
        out["answer"] = re.sub(r"\n*_Verification: not independently verified — [^\n]*_\s*$", "", out["answer"]).rstrip()
        out["answer"] += f"\n\n_Verification by part — {line}._"
        if out.get("incomplete"):
            out["answer"] += " _This question is not fully answered yet._"
    out = _finish(out, message, df, memory, chart_kind, started)
    # completeness: every statistic or profile item the question names must be in the answer; anything missing is
    # said, so a partial answer is never presented as the whole one (app/analyst/requirements.py)
    if any(c.get("status") == "ok" for c in out.get("calls", [])) and out.get("plan_source") not in (
            "conversation", "conversation_rules", "input_check", "withheld", "compound"):
        from app.analyst import requirements
        gaps = requirements.missing_items(said, out.get("answer", ""))
        if gaps:
            out["answer"] += "\n\n" + requirements.note(gaps)
            out["incomplete"] = gaps
            out["notes"].insert(0, "Not answered: " + ", ".join(gaps) + ".")
    if read_as:
        out["answer"] = "_" + "; ".join(read_as) + "._\n\n" + out["answer"]
        out["notes"].insert(0, "Measure words: " + "; ".join(read_as) + ".")
        out["understood_as"] = message
    if table_note:
        out["answer"] = table_note + "\n\n" + out["answer"]
        out["notes"].insert(0, table_note.strip("_"))
        out["logical_table"] = df.attrs.get("logical_table")
    if res:
        out["notes"].insert(0, res["note"] + (f" Run as: “{message}”." if message != said else ""))
        out["understood_as"] = message
        out["plan"] = {**out.get("plan", {}), "conversation": res["kind"]}
    _remember_deliverables(message, df, memory, out)
    conversation.record(memory, message, out)
    return out


def _intent_clauses(message: str) -> list[str]:
    from app.analyst import intent
    return intent.clauses(message)


def _role_failure(out: dict) -> bool:
    """A refusal of the whole message because one part aggregates an identifier or totals a level."""
    if any(c.get("status") == "ok" for c in out.get("calls", [])):
        return False
    return bool((out.get("plan") or {}).get("role_refusal")) or "identifies records" in out.get("answer", "") \
        or "per-record level" in out.get("answer", "")


# ----------------------------------------------------------- the latest result --
def _answered_request(out: dict, message: str, df):
    """The structured reading the answer actually computed: the reading it was built for (a follow-up), else the
    message's own, completed from what was executed (the measure, aggregate and grouping of the tool call)."""
    from app.analyst import intent
    if out.get("_request") is not None:
        return out["_request"]
    try:
        req = intent.read(message, df)
    except Exception:  # noqa: BLE001
        return None
    ok = [c for c in out.get("calls", []) if c.get("status") == "ok"]
    if len(ok) == 1 and ok[0].get("tool") in ("compute_metric", "rank_groups"):
        a, r = ok[0].get("args") or {}, ok[0].get("result") or {}
        col = a.get("column") or r.get("measure")
        fn = a.get("aggregation") or r.get("aggregation")
        q = {"avg": "average", "mean": "average", "sum": "total", "median": "median", "count": "count"}.get(fn)
        if col and q:
            req.measures, req.quantity = [col], q
            req.counted = None
        elif fn == "count_distinct" and col:
            req.quantity, req.counted, req.measures = "count", col, []
        g = a.get("group_by") or r.get("group")
        if g:
            req.group = g
    return req


_PRONOUN_ONLY = re.compile(r"^\W*(?:so\s+|and\s+|ok(?:ay)?,?\s+)?(?:is|are|was|has|have|did|does|will)\s+(?:it|this|that|they|things?)\b"
                    r"|^\W*(?:which|what)(?:'s| is)?\s+(?:one|the one)\s+(?:is\s+)?(?:better|best|worse|worst|strongest|weakest)\b"
                    r"|^\W*how\s+(?:is|are|did)\s+(?:it|this|that|they)\b", re.I)


def _vague_subject(message: str, df, memory: dict) -> dict | None:
    """'is it dropping?', 'which one is better', 'how did it do?': a question whose only subject is a pronoun, on a
    conversation with nothing earlier for it to refer to, names no column and no value -- Silt asks which measure (and
    what to compare) instead of guessing one."""
    if df is None or memory.get("last_result") or memory.get("_last_ds") or len(message.split()) > 8:
        return None
    if not _PRONOUN_ONLY.search(message) or followup_cols(df, message):
        return None
    from app.analyst import calc
    if calc.named_values(df, message):
        return None
    from app.analyst.semantics import measure_options
    opts = measure_options(df)
    if len(opts) < 2:
        return None
    return _reply("Which measure do you mean — and, if you're comparing, which groups or periods? Options: "
                  + ", ".join(f"`{c}`" for c in opts[:6]) + ".", "tool", True, time.monotonic(),
                  plan={"type": "clarify", "options": opts[:6], "reason": "vague subject"},
                  notes=["The question's only subject is a pronoun and nothing earlier in the conversation says what "
                         "it refers to; Silt asks instead of choosing a measure."])


def _record_last_result(out: dict, message: str, df, memory: dict, sc: dict | None) -> None:
    """One record of the latest computed answer, written for every path: the reading it answered, the records it used,
    its headline values and charts, and the turn. Follow-ups ("what about ...", "compare it with ...", "why?", "show
    that as a chart") read only this record, so an older result can never stand in for the latest one."""
    from app.analyst import intent
    req = _answered_request(out, message, df)
    values, measure, group = None, None, None
    for c in out.get("calls", []):
        if c.get("status") != "ok":
            continue
        r = c.get("result") or {}
        if c.get("tool") == "analysis_program":
            res = (r.get("results") or {}).get("") or {}
            head = res.get("per_unit") or res.get("result")
            values = head.get("value") if head else None
        elif c.get("tool") == "compute_metric":
            values = {str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r.get("value")
        elif c.get("tool") == "rank_groups":
            values = {str(x["group"]): x["value"] for x in r.get("rows") or []}
        elif c.get("tool") == "calculator" and memory.get("last_calc"):
            lc = memory["last_calc"]["result"]
            prim = (memory["last_calc"].get("meta") or {}).get("primary") or next(iter(lc.get("aggregates") or []), None)
            values = ({str(g["group"]): g.get(prim) for g in lc.get("groups") or []} if lc.get("groups") else
                      (lc.get("totals") or {}).get(prim))
        break
    if req is not None:
        measure = req.measures[0] if req.measures else (req.counted or (req.event[0] if req.event else None))
        group = req.group
    memory["last_result"] = {"turn": memory.get("_turn", 0), "question": message,
                             "request": intent.request_to_dict(req) if req is not None else None,
                             "scope": sc if scoping.active(sc) else None, "values": _jsonable(values),
                             "measure": measure, "group": group, "source": out.get("plan_source"),
                             "charts": out.get("charts") or [],
                             "verify_status": out.get("verify_status"),
                             "shape": [int(len(df)), [str(c) for c in df.columns]] if df is not None else None}


_WHAT_ABOUT = re.compile(r"^\W*(?:and\s+|now\s+|ok(?:ay)?,?\s+|so\s+)*(?:what|how)\s+about\b|^\W*(?:and\s+|now\s+)?"
                         r"(?:same|the same)\s+(?:for|but)\b|^\W*(?:and|now|then)\s+(?:for|in|only|just)\b|"
                         r"^\W*(?:only|just)\b", re.I)
_BACKREF = re.compile(r"\b(it|that|this|them|those|these|the same|same|previous|last one|the result|"
                      r"that figure|that number|instead)\b", re.I)


def _last_result(memory: dict, df):
    """The latest result, if it was computed on this table as it is now (same columns and number of records). A result
    restored after a restart for a dataset that has since changed is not used -- follow-ups then start fresh."""
    lr = memory.get("last_result")
    if not lr or df is None:
        return None
    shape = lr.get("shape")
    if shape and (shape[0] != len(df) or list(shape[1]) != [str(c) for c in df.columns]):
        return None
    return lr


def _followup_from_last(message: str, df, memory: dict, model_on: bool, started: float,
                        early: bool = False) -> dict | None:
    """A follow-up to the latest result -- "What about Ben Cole?", "What about only completed orders?", "Compare it
    with Chen Wei", "Now the average instead", "Why?" -- applied to that result's structured reading: only what the
    follow-up changes is changed, the rest is kept and said, and the new program passes the same verification. None
    when there is no latest result, or the message asks something of its own (a new question)."""
    from app.analyst import intent
    import copy
    lr = _last_result(memory, df)
    if not lr or not lr.get("request") or df is None:
        return None
    try:
        f = intent.read(message, df)
        base = intent.request_from_dict(lr["request"])
    except Exception:  # noqa: BLE001
        logger.exception("follow-up could not be read")
        return None
    prof_ = _sem_profile(df)
    if f.measures and all(prof_["roles"][c]["role"] == "flag" for c in f.measures) and _WHAT_ABOUT.search(message) \
            and not re.search(r"\b(rate|share|percent\w*|proportion|fraction|how often|likel\w*)\b", message, re.I):
        # "what about only overtime shifts?": the yes/no column says which records -- it is not the new measure
        for c in f.measures:
            w = re.escape(re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()).replace(r"\ ", r"[\s_-]+")
            neg = re.search(rf"\b(?:not|non|never|no|un)[\s-]*{w}|n't\s+(?:been\s+)?{w}", message, re.I)
            (f.restrict_not if neg else f.restrict).append(c)
        f.measures, f.quantity = [], None
    if f.measures or f.counted or f.event or f.relation or f.share_of or f.record_noun or f.unmatched:
        return None                     # it names its own measure or count: a new question, answered as one
    words = len(re.findall(r"[A-Za-z]+", message))
    elliptic = bool(_WHAT_ABOUT.search(message) or _BACKREF.search(message) or words <= 3
                    or re.match(r"^\W*(?:now|then|and|also|ok(?:ay)?,?\s+now)\b", message, re.I))
    named_metric = metrics.recognise(message)
    if not elliptic or (named_metric is not None and not metrics.physical(named_metric, df)):
        return None                     # (a defined metric -- "labor cost by department" -- is a question of its own;
                                        # one that is simply a column here, like an overtime flag, is not)
    if not (f.restrict or f.restrict_not):
        # "what about only overtime shifts?" / "and the non-escalated ones?": a yes/no column named in a follow-up
        # selects records (the follow-up has no measure of its own for it to be)
        prof = _sem_profile(df)
        for c in followup_cols(df, message):
            if prof["roles"][c]["role"] != "flag" or intent.flag_values(df, c, prof["roles"][c]) is None or c == base.group:
                continue
            w = re.escape(re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()).replace(r"\ ", r"[\s_-]+")
            if re.search(rf"\b(?:not|non|never|no|un)[\s-]*{w}|n't\s+(?:been\s+)?{w}", message, re.I):
                f.restrict_not.append(c)
            else:
                f.restrict.append(c)
    if f.why and not (f.entities or f.excluded or f.restrict or f.restrict_not or f.conditions or f.period):
        if early or lr.get("source") in ("model",) or "random_forest" in json.dumps(lr.get("request") or {}):
            return None                 # "why" is answered by its own handlers first (a model's feature importance ...)
        return _why_last(message, df, memory, lr, base, model_on, started)
    new = copy.deepcopy(base)
    changed = []
    if f.compare and f.entities:
        col = f.entities[0][0]
        prev = [v for c, v in base.entities if c == col]
        add = [v for c, v in f.entities if c == col and v not in prev]
        if not prev:
            prev = [v for c, v in (lr.get("scope") or {}).get("entities", {}).items() for v in v if c == col]
        vals = list(dict.fromkeys(prev + add))
        if len(vals) < 2:
            if col == base.group and isinstance(lr.get("values"), dict):
                others = [k for k in lr["values"] if str(k).lower() not in {str(v).lower() for v in add}]
                return _reply(f"The latest result (*{lr['question'][:120]}*) already shows every {col}. Compare "
                              f"**{', '.join(map(str, add))}** with which {col}? Options: "
                              + ", ".join(f"`{o}`" for o in others[:8]), "tool", model_on, started,
                              plan={"type": "clarify", "options": others[:8]},
                              notes=["A comparison needs two sides; the follow-up named one and the latest result "
                                     "already shows them all, so Silt asks which one."])
            return None
        new.entities = [(c, v) for c, v in base.entities if c != col] + [(col, v) for v in vals]
        new.group, new.compared = col, True
        changed.append(f"compared across {col} = {', '.join(map(str, vals))}")
    else:
        sel = False
        for col in dict.fromkeys(c for c, _ in f.entities):
            vals = [v for c, v in f.entities if c == col]
            if col == base.group and not re.search(r"\b(only|just|limit|restrict|nothing but)\b", message, re.I):
                continue                # "what about East" on a breakdown by region: shown, not filtered
                                        # ("only East" narrows the breakdown to it)
            new.entities = [(c, v) for c, v in new.entities if c != col] + [(col, v) for v in vals]
            changed.append(f"{col} = {' or '.join(map(str, vals))}")
            sel = True
        for c, v in f.excluded:
            if (c, v) not in new.excluded:
                new.excluded.append((c, v))
                changed.append(f"excluding {c} = {v}")
                sel = True
        for c in f.restrict:
            new.restrict_not = [x for x in new.restrict_not if x != c]
            if c not in new.restrict:
                new.restrict.append(c)
            changed.append(f"only records where {c} applies")
            sel = True
        for c in f.restrict_not:
            new.restrict = [x for x in new.restrict if x != c]
            if c not in new.restrict_not:
                new.restrict_not.append(c)
            changed.append(f"only records where {c} does not apply")
            sel = True
        for cd in f.conditions:
            new.conditions = [x for x in new.conditions if x["column"] != cd["column"]] + [cd]
            changed.append(f"{cd['column']} {cd['op']} {cd['value']:g}")
            sel = True
        if f.period:
            new.period = f.period
            changed.append(f.period.get("label") or "the period named")
            sel = True
        if f.quantity and f.quantity != base.quantity and f.quantity in ("average", "median", "total") and \
                base.measures:
            new.quantity = f.quantity
            changed.append(f"{f.quantity} instead of {base.quantity or 'the previous figure'}")
            sel = True
        if f.group and f.group != base.group:
            new.group, new.compared = f.group, False
            changed.append(f"broken down by {f.group}")
            sel = True
        if not sel:
            return None
    plan = nl_planner.plan_for_request(new, df, message)
    if plan is None:
        return None
    prior_scope = memory.get("_scope")
    memory["_scope"] = scoping.from_request(new, df)
    kept = []
    if new.measures:
        kept.append(", ".join(map(str, new.measures)) if new.quantity != base.quantity else
                    f"{new.quantity or 'the figure'} of {', '.join(map(str, new.measures))}")
    elif new.event:
        kept.append(f"the rate of {new.event[0]} = {new.event[1]}")
    elif new.counted:
        kept.append(f"the count of distinct {new.counted}")
    if new.group and not new.compared:
        kept.append(f"by {new.group}")
    out = _run_program(plan, message, df, memory, model_on, started, "rule_program", {"status": "followup"},
                       [f"Follow-up to *{lr['question'][:120]}*: changed {'; '.join(changed)}; kept "
                        f"{', '.join(kept) or 'the rest'}."], request=new)
    if out is None:
        memory["_scope"] = prior_scope      # nothing answered: the turn's own scope stands, no follow-up scope leaks
        return None
    out["answer"] = (f"_Follow-up to *{lr['question'][:120]}* — changed: {'; '.join(changed)}; kept: "
                     f"{', '.join(kept) or 'the rest'}._\n\n" + out["answer"])
    out["understood_as"] = f"{lr['question']} → {'; '.join(changed)}"
    return out


def _why_last(message, df, memory, lr, base, model_on, started) -> dict | None:
    """'Why?' after a result: what the data shows alongside it -- for a named person or group, how its records differ
    from the others'; otherwise which columns are associated with the measure. Associations, never causes."""
    from app.analyst import associations
    m = base.measures[0] if base.measures else None
    one = [(c, v) for c, v in base.entities if c != base.group]
    sc = scoping.from_request(base, df)

    def records(without: str | None = None):
        """The records the latest result used -- minus the selection of the group being explained, which is
        compared with the rest of the same records."""
        s_ = {**sc, "entities": {c: v for c, v in (sc.get("entities") or {}).items() if c != without}}
        return df[scoping.mask(df, s_)] if scoping.active(s_) else df
    if base.compared and base.group:
        vals = lr.get("values") if isinstance(lr.get("values"), dict) else {}
        if vals:
            top = max(vals, key=lambda k: vals[k] if vals[k] is not None else float("-inf"))
            one = [(base.group, top)]
    if len(one) == 1 and m and m in df.columns:
        col, val = one[0]
        d = records(without=col)
        try:
            r = router.run({"type": "router", "tool": "drivers", "args": {"group_column": col, "group": val,
                                                                           "outcome": m}}, d, memory, None)
        except (ml.Unsuitable, business.Insufficient) as e:
            return _reply(f"**I can't explain that from this data.** {e}", "tool", model_on, started,
                          plan={"type": "decline"})
        text = (f"_“Why?” — about *{lr['question'][:120]}*: how the {val} records differ from the other {col} "
                "records. These are differences that go along with the result, not proven causes._\n\n"
                + _ROUTED_RENDER["drivers"](r))
        return _reply(text, "tool", model_on, started, plan={"type": "tools", "intent": "why_previous"},
                      calls=[{"tool": "drivers", "status": "ok", "args": {"group_column": col, "group": val,
                                                                          "outcome": m}, "result": _jsonable(r)}],
                      notes=["Answered from the latest result's reading; associations, not causes."])
    if m and m in df.columns:
        prof = _sem_profile(df)
        d = records()
        try:
            res = P.run([{"id": "why", "op": "associate", "from": "data", "target": m,
                          "exclude": [c for c in prof["nonfactors"] if c != m]}], d)
        except P.ProgramError as e:
            return _reply(f"**I can't explain that from this data.** {e}", "tool", model_on, started,
                          plan={"type": "decline"})
        text = (f"_“Why?” — about *{lr['question'][:120]}*: the columns associated with {m}, over the same "
                f"{len(d):,} records the result used. Associations, not causes._\n\n" + associations.text(res["why"].value))
        return _reply(text, "tool", model_on, started, plan={"type": "tools", "intent": "why_previous"},
                      calls=[{"tool": "associations", "status": "ok", "result": _jsonable(res["why"].value)}],
                      notes=["Answered from the latest result's reading; associations, not causes."])
    return None


def _sem_profile(df):
    from app.analyst import semantics
    return semantics.profile(df)


def _chart_from_last_result(message, df, memory, kind, model_on, started) -> dict | None:
    """'Show that as a chart' redraws the LATEST result (memory["last_result"]) -- after "average instead", the
    averages, never the totals computed before them. Only when the message refers back and names nothing new."""
    lr = _last_result(memory, df)
    if not lr or not re.search(r"\b(that|it|this|these|those|them|same|the (result|answer|previous))\b", message, re.I):
        return None
    if not isinstance(lr.get("values"), dict) or len(lr["values"]) < 2:
        if lr.get("values") is None or isinstance(lr.get("values"), dict):
            return None
        # the latest result is one number: there is nothing to compare in a chart -- said about THAT result
        return _reply(f"The latest result (*{lr['question'][:120]}*) is **a single value**, so there is nothing to "
                      "compare in a chart. Ask for it **by** a group — for example by "
                      + next((c.replace("_", " ") for c in df.columns if df[c].dtype == object
                              and 2 <= df[c].nunique() <= 30), "category") + " — and then for the chart.",
                      "tool", model_on, started, chart_requested=True,
                      notes=["No chart was drawn: the latest result has no breakdown to show."])
    from app.analyst.data import find_column
    if calc.named_values(df, message) or [x for x in calc._STRONG.finditer(message)] or [  # noqa: SLF001
            c for c in find_column(df, message) if c not in (lr.get("measure"), lr.get("group"))]:
        return None
    req = lr.get("request") or {}
    q = req.get("quantity")
    x = lr.get("group") or "group"
    y = lr.get("measure") or "value"
    label = {"total": "total", "average": "average", "median": "median", "rate": "share", "count": "count"}.get(q, "")
    base = {"type": "bar", "x": x, "y": y, "additive": q in ("total", "count"),
            "title": f"{label + ' ' if label else ''}{y} by {x}".strip(),
            "data": [{x: k, y: v} for k, v in lr["values"].items() if v is not None]}
    charts, notes = followup.restyle([base], kind if kind != "auto" else None, message)
    return _reply(f"Here is the latest result (*{lr['question'][:120]}*) as a chart — {label + ' ' if label else ''}{y} "
                  f"by {x}, the same computed values; nothing recalculated.", "tool", model_on, started, charts=charts,
                  chart_requested=True, notes=notes + ["Redrawn from the latest result's computed values."],
                  calls=[{"tool": "chart_of_previous_result", "status": "ok",
                          "result": {"values": lr["values"], "from_turn": lr.get("turn")}}])



def _remember_deliverables(message: str, df, memory: dict, out: dict) -> None:
    """What this turn asked for, as deliverables: a later "same for X" / "variance as well" / "what if the target
    was N" / "plot those" edits exactly this, with one explicit change."""
    if df is None or _dead_end(out):
        return
    try:
        from app.analyst import deliverables as _d
        if out.get("plan_source") == "compound" and out.get("deliverables"):
            items = [{k: v for k, v in d.items() if k in _d.Deliverable.__dataclass_fields__} for d in out["deliverables"]]
            memory["_last_ds"] = {"route": "compound", "question": message, "items": items}
            return
        ds = [d for d in _d.extract(message, df) if d.kind not in ("unparsed",)]
        if ds:
            memory["_last_ds"] = {"route": out.get("plan_source"), "question": message, "items": _d.to_items(ds)}
    except Exception:  # noqa: BLE001 - remembering is best effort
        logger.exception("could not record the turn's deliverables")


def _compound_followup(message: str, df, memory: dict, model_on: bool, started: float, filename=None) -> dict | None:
    last = memory.get("_last_ds")
    if df is None or not last or len(message.split()) > 16:
        return None
    from app.analyst import compound as _compound
    from app.analyst import deliverables as _d
    try:
        edited = _d.followup_edit(message, _d.from_items(last["items"]), df)
    except Exception:  # noqa: BLE001
        logger.exception("follow-up edit failed")
        return None
    if not edited:
        return None
    ds, change = edited
    res = _compound.answer(message, ds, df, filename)
    note = f"_Follow-up to *{last['question']}* — {change}._\n\n"
    out = _reply(note + res["answer"], "compound", model_on, started,
                 plan={"type": "compound", "followup": True, "deliverables": [d["label"] for d in res["deliverables"]]},
                 calls=res["calls"], charts=res["charts"], verification=res["verification"],
                 notes=[f"Follow-up: {change}."])
    out["deliverables"] = res["deliverables"]
    out["decomposition"] = {"source": "followup_edit", "items": len(ds)}
    if res["incomplete"]:
        out["incomplete"] = res["incomplete"]
    return out


def _model_followup(message: str, df, memory: dict, model_on: bool, started: float, filename=None) -> dict | None:
    """Last resort for a follow-up nothing deterministic could read: the model decomposes it with the previous
    request as context; every field is grounded before use."""
    if df is None or not model_on or not memory.get("_prev_said"):
        return None
    from app.analyst import compound as _compound
    from app.analyst import decomposer as _decomposer
    if not _decomposer.enabled():
        return None
    trace: dict = {}
    ds = _decomposer.decompose(message, df, previous=memory["_prev_said"], trace=trace)
    if not ds or any(d.kind in ("unparsed", "clarify") for d in ds):
        return None
    res = _compound.answer(message, ds, df, filename)
    out = _reply(f"_Follow-up to *{memory['_prev_said']}*, read with the previous request as context._\n\n" + res["answer"],
                 "compound", model_on, started,
                 plan={"type": "compound", "followup": True, "deliverables": [d["label"] for d in res["deliverables"]]},
                 calls=res["calls"], charts=res["charts"], verification=res["verification"],
                 notes=["Follow-up read by the model with the previous request; every field grounded in the data."])
    out["deliverables"] = res["deliverables"]
    out["decomposition"] = {"source": trace.get("source"), "items": len(ds), "followup": True}
    if res["incomplete"]:
        out["incomplete"] = res["incomplete"]
    return out

def _dead_end(out: dict) -> bool:
    """An answer that computed nothing and is no deliberate clarification or refusal: a placeholder reply."""
    if any(c.get("status") == "ok" for c in out.get("calls", [])):
        return False
    t = (out.get("plan") or {}).get("type")
    return (t == "chat" and out.get("plan_source") in ("conversation", "rules")) or "I couldn't turn" in out.get("answer", "")


def _spec_route(message: str, df: pd.DataFrame, memory: dict, model_on: bool, started: float) -> dict | None:
    """The program the structured reading defines, for a question no other path answered. None when the reading
    is not complete enough to compute without guessing -- the placeholder reply then stands."""
    from app.analyst import intent
    try:
        req = intent.read(message, df)
    except Exception:  # noqa: BLE001
        logger.exception("could not read the message for routing")
        return None
    if req.relation:
        return _relation_answer(message, df, memory, req, model_on, started)
    parts = intent.clauses(message)
    if len(parts) >= 2:
        return _clause_answers(message, parts, df, memory, model_on, started)
    try:
        rp = nl_planner.rule_plan(message, df, memory)
    except Exception:  # noqa: BLE001
        logger.exception("rule program could not be built")
        rp = None
    if not rp:
        return None
    return _run_program(rp, message, df, memory, model_on, started, "rule_program", {"status": "rule"},
                        ["Built from the question's structured reading (measure, count, rate, grouping and filters) "
                         "and run by Silt's program engine; no model was used."])


def _relation_answer(message, df, memory, req, model_on, started) -> dict | None:
    """'Does X correlate with / cause Y': the association between two measures, from the correlation tool, over the
    records in the question's scope. Asked as cause and effect, the answer says plainly what the data cannot show."""
    sc = scoping.establish(message, df, memory)
    if sc.get("clarify"):
        return None
    d = df[scoping.mask(df, sc)] if scoping.active(sc) else df
    x, y = req.relation
    tool, args = (("compare_groups", {"outcome": y, "group": x}) if req.relation_kind == "groups"
                  else ("correlation", {"x": x, "y": y}))
    try:
        call = tools.validate_call({"tool": tool, "args": args}, d, message)
    except tools.InvalidCall as e:
        return _reply(f"**I can't compute that.** {e}", "tool", model_on, started, plan={"type": "decline"})
    r = tools.run(call, d, memory)
    text = explain.render(tool, r)
    if req.causal:
        what = (f"whether {y} differs between the {x} groups, not whether {x} causes the difference"
                if req.relation_kind == "groups" else f"whether {x} and {y} move together, not whether one causes the other")
        text = (f"**This data can show {what}.** "
                "An association can come from a third factor, from the direction running the other way, or from how "
                "the records were collected; testing a cause needs an experiment or a design that rules those out.\n\n"
                + text)
    memory["_scope"] = sc
    return _reply(text, "tool", model_on, started, plan={"type": "tools", "intent": tool},
                  calls=[{**call, "status": "ok", "result": _jsonable(r)}],
                  notes=[f"Read as an association between {x} and {y} (the structured reading of the question)"
                         + ("; asked as cause and effect, which correlation cannot establish." if req.causal else ".")])


def _clause_answers(message, parts, df, memory, model_on, started) -> dict | None:
    """A question in several clauses: each clause read and answered on its own; a clause whose reading is not
    complete (no column for 'order size') is asked about, and the answer says it is not complete."""
    from app.analyst import intent
    from app.analyst import semantics as _sem
    blocks, calls, checks, charts, missing, status = [], [], [], [], [], []
    checklist_lines: list[str] = []
    for i, part in enumerate(parts, 1):
        rp = None
        try:
            rp = nl_planner.rule_plan(part, df, memory, plain=True)     # one clause may be a plain whole-table figure
        except Exception:  # noqa: BLE001
            rp = None
        sub = _run_program(rp, part, df, memory, model_on, started, "rule_program", {"status": "rule"}, []) if rp else None
        if sub is not None:
            own = sub.get("verification") or []
            done = [c for c in own if str(c.get("check", "")).startswith("independent recomputation")]
            st = "not independently verified" if not done or any("not independently verified" in str(c.get("detail"))
                                                                  for c in done) else "verified"
            status.append((i, st))
            tag = "✓ answered — recomputed independently with pandas" if st == "verified" else \
                "answered — not independently verified"
            body, _, own_list = sub["answer"].partition("\n\n**What you asked, and where it is answered**\n")
            if own_list:
                # one checklist for the whole question (below), not one per part
                lines_ = [ln for ln in own_list.split("\n") if ln.startswith("- ")]
                checklist_lines.extend(f"- part {i}: " + ln[2:] for ln in lines_)
                rest = "\n".join(ln for ln in own_list.split("\n") if ln.strip() and not ln.startswith("- "))
                body = body + ("\n\n" + rest if rest else "")
            blocks.append(f"**{i}. {part[:1].upper() + part[1:]}?** _({tag})_\n\n{body}")
            calls += sub["calls"]
            checks += own
            charts += sub.get("charts") or []
            continue
        r = intent.read(part, df)
        prof = _sem.profile(df)
        fn = {"average": "mean", "median": "median", "total": "sum"}.get(r.quantity or "", "mean")
        refused = next((_sem.aggregation_problem(df, c, fn) for c in followup_cols(df, part)
                        if _sem.aggregation_problem(df, c, fn)), None)
        if refused:
            blocks.append(f"**{i}. {part[:1].upper() + part[1:]}?** _(not computed — refused)_\n\n"
                          f"**I can't compute that.** {refused}")
            missing.append(part)
            status.append((i, "refused"))
            continue
        opts = [c for c in prof["measures"] if prof["roles"][c]["role"] in ("amount", "count", "measurement")
                and (r.quantity not in ("total",) or _sem.additive(df, c, prof["roles"][c]))]
        if r.quantity in ("median", "average", "total") and len(parts) >= 2:
            # a "size" / "value" of a record: quantities that add up per record, not per-record levels (a price)
            opts = [c for c in opts if _sem.additive(df, c, prof["roles"][c])] or opts
        opts = opts[:6]
        if r.quantity and not r.measures and opts:
            ask_ = (f"Which column measures this? Options: " + ", ".join(f"`{c}`" for c in opts) + ". I haven't picked "
                    "one for you.")
        else:
            ask_ = "I couldn't read this part as a calculation on this data; please rephrase it on its own."
        blocks.append(f"**{i}. {part[:1].upper() + part[1:]}?** _(not computed — needs your answer)_\n\n{ask_}")
        missing.append(part)
        status.append((i, "needs your answer"))
    if not calls:
        return None
    if checklist_lines:
        blocks.append("**What you asked, and where it is answered**\n" + "\n".join(checklist_lines))
    out = _reply("\n\n".join(blocks), "rule_program", model_on, started, calls=calls, charts=charts,
                 verification=checks, plan={"type": "program", "objective": "rule_planned", "parts": len(parts)},
                 notes=[f"Answered clause by clause ({len(parts)} parts); "
                        + (f"{len(missing)} part(s) need your answer before they can be computed." if missing else
                           "every part was computed.")])
    out["parts_status"] = status
    if missing:
        out["incomplete"] = missing
    return out


def _stated_gate(message: str, df: pd.DataFrame, memory: dict, model_on: bool, started: float) -> dict | None:
    """Before any path computes: a question about something the data does not contain is answered by saying so --
    never by analysing all the records instead. One gate, read from the structured request (app/analyst/intent.py):
      - a name that occurs nowhere in the data ("for Zed Zorro");
      - a share of a measure no column is named after ("percentage of revenue" with amount and unit_price);
      - a threshold whose column cannot be found ("orders over 500")."""
    from app.analyst import intent, semantics
    from app.analyst.data import kind_of
    try:
        req = intent.read(message, df)
    except Exception:  # noqa: BLE001 - the gate never blocks a question it cannot read
        logger.exception("stated-requirements gate could not read the message")
        return None
    if req.unmatched:
        import difflib
        text_cols = [c for c in df.columns if kind_of(df[c]) == "categorical" and df[c].nunique() <= 5000]
        near = []
        for name in req.unmatched:
            for c in text_cols:
                vals = df[c].dropna().astype(str).unique().tolist()
                close = difflib.get_close_matches(name, vals, n=2, cutoff=0.72)
                if not close and len(vals) <= 5000:
                    # ... or values that share one of its words ("Rahul Sharma": there is a Rahul Mehta)
                    words = [w for w in re.findall(r"[A-Za-z][\w'-]{2,}", name)]
                    close = [v for v in vals if any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", v, re.I)
                                                    for w in words)][:3]
                near += [f"{v} ({c})" for v in close]
        names = ", ".join(f"**{n}**" for n in req.unmatched)
        ans = (f"No records match {names} — {'it does' if len(req.unmatched) == 1 else 'they do'} not occur in any "
               f"column of this dataset, so I haven't analysed the other records instead.")
        if near:
            ans += " Did you mean " + " or ".join(dict.fromkeys(near)) + "?"
        elif text_cols:
            ans += " Columns I searched: " + ", ".join(f"`{c}`" for c in text_cols[:8]) + "."
        return _reply(ans, "tool", model_on, started, plan={"type": "clarify", "unmatched": req.unmatched},
                      notes=["The question names something that is not in the data; nothing was calculated, and the "
                             "full dataset was not used in its place."])
    if (req.period or {}).get("unreadable"):
        return _reply(f"**Which dates do you mean?** {req.period['unreadable'][:1].upper() + req.period['unreadable'][1:]}"
                      " — write it as, for example, *1 August 2026* or *2026-08-01*. I haven't computed anything over "
                      "all the dates instead.", "tool", model_on, started, plan={"type": "clarify"},
                      notes=["The question names a period Silt cannot read without guessing; nothing was calculated."])
    if req.ambiguous_names:
        name, cands = req.ambiguous_names[0]
        return _reply(f"**{name}** matches more than one record name: " + ", ".join(f"`{c}`" for c in cands)
                      + ". Which one do you mean? I haven't analysed all the records instead.", "tool", model_on,
                      started, plan={"type": "clarify", "options": cands},
                      notes=["The name is part of several values; Silt asks rather than picking one or ignoring it."])
    if req.share_word:
        prof = semantics.profile(df)
        opts = [c for c in prof["measures"] if prof["roles"][c]["role"] in ("amount", "count", "measurement")
                and semantics.additive(df, c, prof["roles"][c])][:6]
        if opts:
            return _reply(f"Which column measures **{req.share_word}**? A share of {req.share_word} is a part of that "
                          f"column's total — not a share of the records. Options: " + ", ".join(f"`{c}`" for c in opts),
                          "tool", model_on, started, plan={"type": "clarify", "options": opts},
                          notes=[f"'{req.share_word}' names no column; Silt asks rather than reporting the share of "
                                 "records instead."])
    if req.unresolved:
        prof = semantics.profile(df)
        opts = [c for c in prof["measures"]][:8]
        return _reply(f"Which column does **{req.unresolved[0]}** apply to? I haven't computed anything without that "
                      "limit." + (" Options: " + ", ".join(f"`{c}`" for c in opts) if opts else ""), "tool", model_on,
                      started, plan={"type": "clarify", "options": opts},
                      notes=["The question states a threshold with no column Silt could tie it to; nothing was "
                             "calculated over all the records instead."])
    return None


def _logical_table(message: str, df: pd.DataFrame, memory: dict):
    """(frame, message, note) for a file of several stacked tables: the one table the question is about, with the
    question's own mention of it ("... in Sales Transactions") taken out so its words are not read as columns, and a
    line saying which table answered. The note is a dict {"clarify", "options", "why"} when Silt has to ask instead.
    An ordinary file comes back unchanged with no note."""
    from app.analyst import tables
    pick = tables.select(message, df, memory)
    if pick is None:
        return df, message, None
    if pick.get("clarify"):
        return df, message, {"clarify": pick["clarify"], "options": pick.get("options"),
                             "why": "This file holds several separate tables; Silt analyses one at a time and asks "
                                    "rather than adding different kinds of records together. Nothing was calculated."}
    table = pick.get("table")
    if not table:
        memory.pop("logical_table", None)
        return df, message, None
    named_out = [o for o in pick.get("outside") or [] if o["named"]]
    if named_out:
        o = named_out[0]
        memory["logical_table"] = {"column": pick["column"], "table": table}
        return df, message, {"clarify": f"**{o['column']}** is not recorded for **{table}** — it is empty in all of "
                                        f"its records, so it can't be used to filter, group or measure them. "
                                        f"{o['column']} is only filled in for **{', '.join(o['tables'])}**"
                                        + (f" — for example, ask about {o['tables'][0]} in the "
                                           f"{' or '.join(o['values'])} {o['column'].lower()}." if o["values"] else ".")
                                        + " Nothing was filtered or calculated.",
                             "options": [],
                             "why": f"The question names {o['column']}, which belongs to another table of this file."}
    sub = tables.subset(df, pick["column"], table)
    memory["logical_table"] = {"column": pick["column"], "table": table}
    how = {"named": "named in your question", "previous": "the table this conversation is on",
           "columns": "the only table with every column your question uses"}[pick["how"]]
    note = f"_Table: **{table}** ({len(sub):,} of {len(df):,} records in this file; {how})."
    dropped = [o for o in pick.get("outside") or [] if not o["named"]]
    if dropped:
        note += " Not applied as filters: " + "; ".join(f"“{' or '.join(o['values'])}” ({o['column']} is empty for "
                                                         f"{table})" for o in dropped) + "."
    if pick.get("lacking"):
        note += (" " + ", ".join(f"`{c}`" for c in pick["lacking"]) + f" {'is' if len(pick['lacking']) == 1 else 'are'} "
                 f"not recorded for {table}, so {'it was' if len(pick['lacking']) == 1 else 'they were'} not used.")
    note += "_"
    if pick["how"] == "named":
        # "... in Sales Transactions": the table is chosen; its name is not a measure or a group ("sales")
        forms = {table, table[:-1] if table.endswith("s") else table}       # "Sales Transactions" / "Sales Transaction"
        name = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in sorted(forms, key=len, reverse=True))
        # ... except after "overview / summary / description of", where the table is the object: "this table"
        stripped = re.sub(rf"\b(overview|summary|description|profile|structure|snapshot)\s+of\s+(?:the\s+)?(?:{name})"
                          rf"(?:\s+table)?\b", r"\1 of this table", message, flags=re.I)
        stripped = re.sub(rf"[\s,]*\b(?:in|from|within|for|across|of|on)\s+(?:the\s+)?(?:{name})(?:\s+table)?\b", "",
                          stripped, flags=re.I)
        if stripped.strip(" ?.!") and stripped != message:
            message = stripped
    return sub, message, note


_VERIFIED_LINE = {"verified": "_Verification: ✓ the headline figure was recomputed independently with pandas from your "
                               "question and matches._",
                  "not independently verified": "_Verification: not independently verified — {why}. The steps are "
                                                "under *How this was calculated*._"}


def _verify_answer(out, message, df, memory, model_on, started) -> dict:
    """Every answer that computed something passes app/analyst/verify.py, whichever path produced it. A figure that
    fails is withheld: Silt's own program for the request answers instead if it verifies, otherwise nothing is shown."""
    from app.analyst import verify
    if out.get("plan_source") == "compound":
        # each deliverable was validated and recomputed on its own scope by app/analyst/compound.py; the single-
        # question verifier would read the whole message as one calculation
        items = out.get("deliverables") or []
        ok = [d for d in items if d.get("status") == "done" and d.get("validation") == "recomputed independently"]
        out["verify_status"] = "verified" if items and len(ok) == len(items) else (
            "partially verified" if ok else "not independently verified")
        out["answer"] += ("\n\n_Verification: " + (f"✓ each of the {len(items)} items was computed by Silt's tools and "
                          "recomputed independently" if out["verify_status"] == "verified" else
                          f"{len(ok)} of {len(items)} items computed and recomputed independently; the others are marked "
                          "above") + "._")
        return out
    opts = (out.get("plan") or {}).get("options") or []
    if (out.get("plan") or {}).get("type") == "clarify" and len(opts) == 1 and df is not None:
        # "Which numeric column do you mean? Options: count": one option is no choice. Silt's own program for the
        # request answers if it uses exactly that column and passes verification; otherwise the question stands.
        rp = nl_planner.rule_plan(message, df, memory)
        if rp and any(st.get("column") == opts[0] for st in rp.get("steps", [])):
            alt = _run_program(rp, message, df, memory, model_on, started, "rule_program", {"status": "rule"},
                               [f"The only column that fits is `{opts[0]}`, so Silt used it rather than asking."])
            v2 = verify.check_answer(message, df, alt, memory.get("_scope"), memory) if alt else None
            if alt is not None and (v2 or {}).get("status") == "verified":
                alt["verify_status"] = "verified"
                alt["answer"] = f"_Only one column fits: `{opts[0]}`._\n\n" + alt["answer"] + "\n\n" + _VERIFIED_LINE["verified"]
                return alt
        return out
    try:
        v = verify.check_answer(message, df, out, memory.get("_scope"), memory)
    except Exception:  # noqa: BLE001 - a verifier error is reported as unverified, never as passed
        logger.exception("answer verification failed to run")
        v = {"status": "not independently verified", "checks": [{"check": "independent recomputation", "ok": True,
             "detail": "not independently verified (the check could not run)"}]}
    if v is None:
        return out
    if v["status"] == "failed":
        bad = "; ".join(c["detail"] for c in v["checks"] if not c["ok"])
        logger.info("answer withheld by verification (%s) calls=%s", out.get("plan_source"),
                    [c.get("tool") for c in out.get("calls", [])])          # the question itself is not logged
        said = _failure_reasons(v["checks"])       # what was wrong, without any figure of the rejected answer
        if out.get("plan_source") in ("metric", "model_calc", "refine"):
            memory.pop("last_calc", None)           # a follow-up ("Only Rahul.") must not refine a rejected figure
        alt = None
        if out.get("plan_source") not in ("rule_program",):
            alt = _run_program(nl_planner.rule_plan(message, df, memory), message, df, memory, model_on, started,
                               "rule_program", {"status": "rule"}, [
                                   f"Silt's first answer (from its {out.get('plan_source')} path) did not pass answer "
                                   f"verification ({said}); it was not shown, and Silt built the program from the "
                                   "request itself."])
        if alt is not None:
            v2 = verify.check_answer(message, df, alt, memory.get("_scope"), memory) or \
                {"status": "not independently verified", "checks": []}
            if v2["status"] != "failed":
                return _label(alt, v2, message, df, memory)
        return _reply("**I computed an answer, but it did not pass Silt's verification, so I haven't shown it.** "
                      f"{said[:1].upper() + said[1:]}. Try naming the measure, the grouping and any filter explicitly.",
                      "withheld", model_on, started, plan={"type": "withheld"}, verification=v["checks"],
                      verify_status="withheld",
                      notes=["Withheld by answer verification: the computed figure did not match what the question "
                             "asks for, and no deterministic program for the request passed verification either."])
    return _label(out, v, message, df, memory)


def _failure_reasons(checks) -> str:
    out = []
    for c in checks:
        if c["ok"]:
            continue
        if str(c["check"]).startswith("independent recomputation"):
            out.append("the figure did not match an independent recomputation from your question")
        else:
            out.append(c["detail"])
    return "; ".join(dict.fromkeys(out))


def _label(out, v, message, df, memory) -> dict:
    """The verification status on the answer itself: verified, or not independently verified with the reason; plus
    the duplicate note and any part of the question the answer does not cover."""
    from app.analyst import verify
    out["_post_checks"] = (out.get("_post_checks") or []) + v["checks"]
    out["verify_status"] = v["status"]
    dn = verify.duplicate_note(message, df, memory.get("_scope"), out)
    if dn:
        out["answer"] += "\n\n" + dn
    vn = verify.value_note(df, memory.get("_scope"), out)
    if vn:
        out["answer"] += "\n\n" + vn
    if v["status"] == "verified":
        line = _VERIFIED_LINE["verified"]
    else:
        why = next((re.sub(r"\)$", "", c["detail"].split("(", 1)[-1]) for c in (out.get("verification") or []) + v["checks"]
                    if "not independently verified" in str(c.get("detail"))), "no independent check covers it")
        line = _VERIFIED_LINE["not independently verified"].format(why=why)
    missing = verify.unanswered_parts(message, df, out)
    if missing:
        out["unanswered_parts"] = missing
        line += " _Not answered here: " + "; ".join(missing) + "._"
    out["answer"] += "\n\n" + line
    return out


def memory_key(team_id, user_id, dataset_id, document_id, conversation_id) -> str:
    """team : user : dataset : document : conversation, each part escaped so no id containing ':' can stand for
    another's parts (ids without ':' or '%' are unchanged)."""
    def part(x):
        return str(x).replace("%", "%25").replace(":", "%3A")
    return ":".join(part(x) for x in (team_id, user_id, dataset_id, document_id, conversation_id))


_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def memory_lock(key: str) -> threading.Lock:
    """The lock for one conversation's memory: concurrent requests in the same conversation take turns, so neither
    turn's result register is lost or mixed into the other's."""
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            if len(_LOCKS) > 4 * MAX_CONVERSATIONS:
                for k in [k for k, v in _LOCKS.items() if not v.locked()][: len(_LOCKS) // 2]:
                    _LOCKS.pop(k, None)
            lk = _LOCKS[key] = threading.Lock()
        return lk


def save_memory(conversation_id: str, memory: dict) -> None:
    """Persist what a later turn may refer to (the result register, the open objective) for this key only."""
    from app.analyst import persist as _persist
    _persist.save(conversation_id, memory)


def _reply(answer: str, source: str, model_on: bool, started: float, **extra) -> dict:
    out = {"answer": answer, "plan": extra.pop("plan", {"type": source}), "plan_source": source, "first_pass": None,
           "calls": [], "notes": [], "charts": [], "grounding": {"checked": False}, "details": "",
           "model_available": bool(model_on)}
    out.update(extra)
    out["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return out


def _dispatch(message, df, memory, model_on, filename, started, chart_kind) -> dict:
    if not memory.get("_in_dispatch"):
        # per-turn planner state (a finding's follow-up re-enters _dispatch within the same turn)
        memory.pop("_nl_tried", None)
        memory.pop("_nl_unsupported", None)
        memory.pop("_model_spent_ms", None)
    memory["_in_dispatch"] = True
    try:
        return _dispatch_turn(message, df, memory, model_on, filename, started, chart_kind)
    finally:
        memory.pop("_in_dispatch", None)


def _dispatch_turn(message, df, memory, model_on, filename, started, chart_kind) -> dict:
    opts = router.ml_options(message)
    memory["_ml_options"] = opts
    # 0. a follow-up to a what-if scenario in this conversation ("why did profit change?", "which input mattered most?") is
    # explained from a fresh run of the shared what-if service, before the Copilot's own follow-up reading can take it.
    if memory.get("scenario_context") and scenario_bridge.FOLLOW_UP.search(message):
        followed = scenario_bridge.try_answer(message, df, memory, filename)
        if followed is not None:
            return _reply(followed["answer"], "tool", model_on, started, plan=followed["plan"], calls=followed["calls"],
                          notes=followed["notes"], details=followed.get("details", ""))
    # A compound request (several deliverables: summary + statistics + target + forecast + chart ...) is answered as
    # one task by app/analyst/compound.py BEFORE any single-intent handler can claim the whole message: each
    # deliverable keeps its own scope, runs on its own (a failure never erases the others) and is checked off.
    if df is not None:
        from app.analyst import compound as _compound
        from app.analyst import decomposer as _decomposer
        from app.analyst import deliverables as _deliverables
        try:
            _ds = _deliverables.extract(message, df)
        except Exception:  # noqa: BLE001 - the reader is best effort; the existing paths still answer
            logger.exception("deliverable extraction failed")
            _ds = []
        _source, _trace = "rules", {}
        _followup = _deliverables.follow_up(memory.get("_said") or message, memory)
        if not memory.get("_compound_legacy") and memory.get("_last_ds") and \
                ((memory["_last_ds"].get("route") == "compound") or
                 (_followup and _deliverables.FU_OTHERS.search(message) and len(message.split()) <= 8)):
            fu = _compound_followup(message, df, memory, model_on, started, filename)
            if fu is not None:
                return fu
        if model_on and not memory.get("_compound_legacy") and _decomposer.enabled() and \
                _decomposer.worth_asking(message, _ds, _followup):
            # the model reads the whole request (and, for a follow-up, the previous one); every field it returns is
            # grounded in the data before use, and the rule reader stands in when it fails
            _m = _decomposer.decompose(message, df, trace=_trace)
            if _m:
                _ds, _source = _m, _trace.get("source", "model")
        memory["_decomposition"] = {"source": _source, "items": len(_ds),
                                    "problems": [a.get("problems") for a in _trace.get("attempts", [])]}
        if _ds and _deliverables.is_compound(_ds) and not memory.get("_compound_legacy") and \
                not _deliverables.hypothetical(message) and not _deliverables.hypothetical(memory.get("_said") or "") \
                and not _followup:
            if _deliverables.legacy_first(_ds, message):
                # statistics / rankings only: Silt's existing multi-part paths answer first; their answer stands only
                # if it states every requested value (checked independently below), otherwise the compound answer
                memory["_compound_legacy"] = True
                try:
                    legacy = _dispatch_turn(message, df, memory, model_on, filename, started, chart_kind)
                finally:
                    memory.pop("_compound_legacy", None)
                ok, missing = _compound.covers(legacy, _ds if _source != "rules" else _deliverables.extract(message, df), df)
                if ok:
                    return legacy
                logger.info("compound: the %s answer missed %s; answered as a compound request",
                            legacy.get("plan_source"), missing)
            res = _compound.answer(message, _ds, df, filename)
            out = _reply(res["answer"], "compound", model_on, started,
                         plan={"type": "compound", "deliverables": [d["label"] for d in res["deliverables"]]},
                         calls=res["calls"], charts=res["charts"], verification=res["verification"],
                         notes=[f"Answered as one request with {len(_ds)} requested items, each computed on its own "
                                "scope and checked off" + (f"; not complete: {', '.join(res['incomplete'])}"
                                                           if res["incomplete"] else "; every item was answered") + "."])
            out["deliverables"] = res["deliverables"]
            out["decomposition"] = memory.get("_decomposition")
            if res["incomplete"]:
                out["incomplete"] = res["incomplete"]
            return out
    # "Explain the profit difference", "does profit equal revenue minus cost?": the stored profit reconciled with
    # revenue - cost record by record (app/analyst/reconcile.py), before it is read as a question about the last answer
    # "how many records, what columns, how many missing values": a description of the table itself
    from app.analyst import requirements
    if requirements.profile_request(message, df):
        ov = router.run({"type": "router", "tool": "understand", "args": {}, "reason": "dataset overview"}, df, memory,
                        filename)
        return _reply(_ROUTED_RENDER["understand"](ov), "router", model_on, started,
                      plan={"type": "router", "tool": "understand", "reason": "dataset profile"},
                      calls=[{"tool": "understand", "args": {}, "status": "ok", "result": _jsonable(ov)}],
                      notes=["The question asks for a description of the table (records, columns, missing values, "
                             "duplicates); answered by Silt's dataset profile."])
    # "how many companies are rated below 3, between 3 and 4, and 4 or above": counts per stated value band
    from app.analyst import bands
    bspec = bands.parse(message, df)
    if bspec is not None:
        bsc = scoping.establish(message, df, memory)
        bdf = df[scoping.mask(df, bsc)] if scoping.active(bsc) and not bsc.get("clarify") else df
        br = bands.run(bspec, bdf)
        return _reply(bands.text(br), "tool", model_on, started, plan={"type": "tools", "intent": "bands"},
                      calls=[{"tool": "bands", "status": "ok", "args": {"column": bspec["column"]},
                              "result": _jsonable(br)}],
                      notes=["Counted per value band by Silt's deterministic tools."])
    from app.analyst import reconcile
    if reconcile.ASK.search(message) and reconcile.columns(df):
        rsc = scoping.establish(message, df, memory)        # "... for Returned orders": only those records
        rec = reconcile.check(df[scoping.mask(df, rsc)] if scoping.active(rsc) and not rsc["clarify"] else df)
        text, calls = reconcile.text(rec), [{"tool": "reconcile_profit", "status": "ok", "args": {}, "result": _jsonable(rec)}]
        if re.search(r"\b(overview|summary|summari[sz]e|describe|profile)\b", message, re.I):
            # "an overview of the table and reconcile the profit": both parts are answered, the overview first
            routed = {"type": "router", "tool": "understand", "args": {}, "reason": "dataset overview"}
            ov = router.run(routed, df, memory, filename)
            text = _ROUTED_RENDER["understand"](ov) + "\n\n### Profit reconciliation\n\n" + text
            calls.insert(0, {"tool": "understand", "args": {}, "status": "ok", "result": _jsonable(ov)})
        return _reply(text, "tool", model_on, started, plan={"type": "tools", "intent": "reconcile"},
                      calls=calls,
                      notes=[f"Computed record by record with pandas: {rec['profit']} against {rec['revenue']} − "
                             f"{rec['cost']} over {rec['records']:,} records."])
    # 1. conversation about the previous answer, recommendations, "why is X doing badly?"
    if not memory.get("investigation") and finding_ref(message, 0) is not None and re.search(
            r"\b(finding|findings|found|point|item|issue|insight|observation)\b", message, re.I):
        # "tell me more about finding 2" with no investigation in this conversation: nothing to refer to -- never a
        # different analysis run in its place
        return _reply("There are no numbered findings in this conversation yet, so I can't tell you more about one. "
                      "Ask me to *analyse this dataset* first; I'll list what I find, and you can then ask about any of "
                      "them by number.", "tool", model_on, started, plan={"type": "chat", "reference": "unresolved"},
                      notes=["The message refers to a finding, and this conversation has no investigation; nothing was "
                             "calculated."])
    if model_on and _last_result(memory, df) is not None:
        # A follow-up to the latest result ("what about ...", "only ...", "compare it with ...", "the average instead",
        # "why?") is resolved deterministically from that result's reading BEFORE any model is consulted: the model's
        # planner was found (Stage 6, model ON) to answer such messages as new, different questions.
        legacy_now = any(memory.get(k) and not _superseded(memory, k) for k in ("last_calc", "last_breakdown"))
        out = _followup_from_last(message, df, memory, model_on, started, early=legacy_now)
        if out is not None:
            return out
    kind = followup.intent(message, memory)
    if kind == "advice" and followup_cols(df, message) and re.search(
            r"\b(compare\w*|across|between|by|per|versus|vs\.?|each|related|differ\w*)\b", message, re.I):
        kind = None             # "compare csat across priorities": an analysis of named columns, not a request for advice
    if kind == "explain_previous" and memory.get("investigation") and finding_ref(
            message, len(memory["investigation"]["findings"])) is not None:
        kind = None                     # "explain the last thing you found": a stored finding, not the last answer
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
        # the calculator's own redraw when its calculation is still the latest result (it knows the metric's label
        # and records); otherwise the latest result itself -- never an older one
        out = _chart_from_last(message, df, memory, model_on, started) or \
            _chart_from_last_result(message, df, memory, chart_kind, model_on, started)
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
                          notes=(["The filter names a column that is empty in the records selected, so it could only "
                                  "return no records; Silt says so instead of reporting an empty result."]
                                 if sc.get("unsupported_filter") else
                                 ["The question names a value that occurs in more than one column; Silt asks rather "
                                  "than guessing which records you mean."]))
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
    # "analyse sales instead" on a dataset with nothing called sales: another dataset is meant
    sw = _SWITCH.search(message)
    from app.analyst.data import find_column as _fc
    if sw and re.search(r"\b(instead|switch|another|different)\b", message, re.I) and not _fc(df, sw.group(1)) \
            and sw.group(1).lower() not in _STOPWORDS and not calc.named_values(df, sw.group(1)):
        return _reply(f"This conversation works on the selected dataset ({len(df):,} records; columns "
                      + ", ".join(f"`{c}`" for c in df.columns[:8]) + f"). It has nothing called **{sw.group(1)}**. To analyse "
                      f"{sw.group(1)}, choose that file in the **Dataset** selector — each dataset keeps its own conversation.",
                      "tool", model_on, started, plan={"type": "clarify", "switch_dataset": sw.group(1)})
    vague = _vague_subject(message, df, memory)
    if vague is not None:
        return vague
    out = _ask_investigation(message, df, memory, model_on, started, filename)
    if out is not None:
        return out
    # "Which department performs worst and why?": an evaluative word with a grouping but no measure -- what
    # "performing" means is the user's choice, so the candidate measures are offered (outcomes first), once
    pq = _undefined_performance(message, df)
    if pq is not None:
        return pq if isinstance(pq, dict) and "answer" in pq else _reply(
            f"Which measure should decide how each **{pq['group']}** performs? Options: "
            + ", ".join(f"`{c}`" for c in pq["options"]), "tool", model_on, started,
            plan={"type": "clarify", "options": pq["options"]},
            notes=["'Performs' names no column; Silt asks which measure defines it rather than choosing one. With the "
                   "measure chosen, the comparison and the 'why' (associations, not causes) are computed."])
    # A concept two or more columns could mean ("revenue" with takings and online_sales), named by none of them: asked
    # before any path -- deterministic or model -- picks one
    amb = nl_planner.ambiguities(message, df)
    if amb:
        w, cs = next(iter(amb.items()))
        return _reply(f"Which column do you mean by **{w}**?\n\nOptions: " + ", ".join(f"`{c}`" for c in cs), "tool",
                      model_on, started, plan={"type": "clarify", "options": cs},
                      notes=[f"'{w}' could mean any of {', '.join(cs)}; Silt asks rather than choosing one. Nothing was "
                             "calculated."])
    # Two questions about different records in one message ("total amount for East and the average units for West"):
    # each clause is computed on its own records -- never all of them on the union
    from app.analyst import intent as _intent_mod
    sep = _intent_mod.separate_scopes(message, df)
    if sep:
        out = _clause_answers(message, sep, df, memory, model_on, started)
        if out is not None:
            return out
    # "compare that with the previous result", "set that next to the original ask": earlier answers of this
    # conversation, resolved from its ordered result register (app/analyst/register.py); two target/scenario
    # objectives are compared as scenarios, anything else side by side
    from app.analyst import register
    tq = register.target_request(message, memory) or None
    if tq is None:
        rq = register.recall_request(message, memory)
        if isinstance(rq, str):
            return _reply(rq, "tool", model_on, started, plan={"type": "chat", "reference": "unresolved"},
                          notes=["The reference names an earlier result this conversation does not have; nothing is "
                                 "left pending, so the next question is read on its own."])
        if rq is not None:
            tq = register.recall(memory["register"][rq])
    if tq is not None:
        return _reply(tq["answer"], "program" if tq["calls"] else "tool", model_on, started, plan=tq["plan"],
                      calls=tq["calls"], charts=tq["charts"], notes=tq["notes"])
    cr = register.compare_request(message, memory)
    if isinstance(cr, str):
        return _reply(cr, "tool", model_on, started, plan={"type": "chat", "reference": "unresolved"},
                      notes=["The reference names an earlier result this conversation does not have; nothing is "
                             "left pending, so the next question is read on its own."])
    if cr is not None:
        reg = memory["register"]
        a, b = reg[cr[0]], reg[cr[1]]
        if a.get("objective") and b.get("objective"):
            obj = objectives._compare_scenarios(a["objective"], b["objective"])  # noqa: SLF001
            obj["notes"].insert(0, f"Resolved from the conversation's result register: result {a['n']} "
                                   f"(*{a['question']}*) and result {b['n']} (*{b['question']}*).")
        else:
            obj = register.compare(a, b)
        return _reply(obj["answer"], "program", model_on, started, plan=obj["plan"], calls=obj["calls"],
                      charts=obj["charts"], notes=obj["notes"])
    # Multi-step objectives (target feasibility, absence scenarios, entity vs the rest, changing an assumption,
    # comparing scenarios) run as validated analysis programs (app/analyst/program.py, objectives.py).
    # A target or required pace on data recorded as weekly / monthly / quarterly / yearly totals: compared with those
    # period totals (and a tested forecast for "will ..."), never projected from a daily pace (objectives.py assumes
    # event records within a period).
    if df is not None:
        from app.analyst import compound as _compound
        from app.analyst import deliverables as _deliverables
        _ds = _deliverables.extract(message, df)
        if any(d.kind in ("target_check", "pace") for d in _ds) and _compound.period_totals(df, _ds):
            res = _compound.answer(message, _ds, df, filename)
            out = _reply(res["answer"], "compound", model_on, started,
                         plan={"type": "compound", "deliverables": [d["label"] for d in res["deliverables"]]},
                         calls=res["calls"], charts=res["charts"], verification=res["verification"],
                         notes=["The records are period totals (one per person per period), so the target was compared "
                                "with those totals and, for a question about the future, with a tested forecast."])
            out["deliverables"] = res["deliverables"]
            if res["incomplete"]:
                out["incomplete"] = res["incomplete"]
            return out
    try:
        obj = objectives.handle(message, df, memory, sc)
    except Exception as e:  # noqa: BLE001 - a program failure is reported, never hidden behind another answer
        logger.exception("analysis program failed")
        obj = {"answer": f"**The analysis program failed** ({type(e).__name__}: {e}). Nothing was estimated instead.",
               "calls": [], "charts": [], "notes": [], "plan": {"type": "error"}}
    if obj is not None:
        return _reply(obj["answer"], "program", model_on, started, plan=obj["plan"], calls=obj["calls"],
                      charts=obj["charts"], notes=obj["notes"], details=obj.get("details", ""),
                      verification=obj.get("verification") or [])
    # A request shaped like several analyses ("which ... is worst and why", "after accounting for", "if ... continues")
    # is planned by the local model as one program before any single-purpose handler answers half of it.
    if model_on and (nl_planner.composite(message, df) or nl_planner.follow_up(message, memory)):
        out = _ask_nl_program(message, df, memory, model_on, started, filename, late=False)
        if out is not None:
            return out
    elif not model_on and nl_planner.composite(message, df):
        out = _run_program(nl_planner.rule_plan(message, df, memory), message, df, memory, model_on, started,
                           "rule_program", {"status": "rule"}, [])
        if out is not None:
            return out
    if nl_planner.composite(message, df):
        # "what do outlets take per visitor": a per-unit figure whose numerator names no column -- ask which value,
        # never answer about the denominator instead
        from app.analyst import intent as _intent
        _r = _intent.read(message, df)
        if _r.per and _r.denominators and not _r.measures and _intent._RECORDS not in _r.denominators \
                and metrics.recognise(message) is None:  # noqa: SLF001 - a catalogue metric ("cost per lead") defines it
            from app.analyst import semantics as _sem
            nums = [c for c in _sem.profile(df)["measures"] if c not in _r.denominators][:6]
            if nums:
                den = " / ".join(map(str, _r.denominators))
                return _reply(f"Which value should be divided by **{den}**? Options: " + ", ".join(f"`{c}`" for c in nums),
                              "tool", model_on, started, plan={"type": "clarify", "options": nums},
                              notes=[f"The request asks for a figure per {den} but names no column to divide; Silt asks "
                                     "rather than reporting the denominator itself."])
        lost = nl_planner.unapplied(message, df)
        if lost and memory.get("last_result"):
            out = _followup_from_last(message, df, memory, model_on, started, early=True)
            if out is not None:
                return out
        if lost and metrics.recognise(message) is not None and scoping.active(sc) and sc.get("period"):
            # a defined metric (labour cost, profit ...) over a period: the calculator holds it to the question's
            # scope, which carries the dates -- and the verifier checks the dates were applied
            out = _ask_metric(message, df, memory, model_on, started, sc)
            if out is not None and any(c.get("status") == "ok" for c in out.get("calls", [])):
                return out
        if lost and scoping.active(sc) and sc.get("where"):
            # "which companies are older than 30 and rated above 4": listing the records that meet the conditions
            # applies them exactly (the scope's own mask), so it is an answer, not an unrestricted figure
            out = _ask_lookup(message, df, model_on, started, sc=sc)
            if out is not None and any(c.get("tool") == "filter_records" for c in out.get("calls", [])):
                return out
        if lost:
            # the request is limited (a relative period, "only escalated") and no program could apply it: an answer
            # over all the records would look right and be wrong
            return _reply(f"**I couldn't apply {', '.join(lost)} to this question**, so I haven't computed it over all "
                          "the records instead. Say which value to measure — for example *total <column> for "
                          f"{lost[0].split(' (')[0]}* — and I'll apply it.", "tool", model_on, started,
                          plan={"type": "clarify"},
                          notes=["The request states a restriction that only an analysis program can apply; none could "
                                 "be built, so nothing was calculated."])
    legacy_current = any(memory.get(k) and not _superseded(memory, k) for k in ("last_calc", "last_breakdown"))
    if memory.get("last_result") and not legacy_current:
        # the latest answer is newer than what the older refinement handlers hold ("what about only completed
        # orders?" then "now the average instead"): the follow-up applies to the latest result's reading
        out = _followup_from_last(message, df, memory, model_on, started, early=True)
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
    # A what-if or goal sentence goes to the shared what-if service further down, which declines (None) anything it
    # cannot read cleanly.
    if not (followup.wants_forecast(message) or re.search(r"\bpredict\w*\b", message, re.I) or _SCENARIO_Q.search(message)
            or scenario_bridge.TRIGGER.search(message)) \
            and not (_TREND_WORDS.search(message) and metrics.recognise(message)):
        out = _ask_metric(message, df, memory, model_on, started, sc)
        if out is not None:
            return out
    out = _ask_missing_target(message, df, memory, model_on, started)
    if out is not None:
        return out
    out = _ask_which_measure(message, df, model_on, started)
    if out is not None:
        return out
    out = _ask_associations(message, df, memory, model_on, started)
    if out is not None:
        return out
    out = _ask_compare_values(message, dfs, model_on, started)
    if out is not None:
        return out
    out = _ask_value_distribution(message, df, model_on, started)
    if out is not None:
        return out
    out = _ask_lookup(message, df, model_on, started, sc=sc)
    if out is not None:
        return out
    out = _ask_compare_columns(message, dfs, model_on, started)
    if out is not None:
        return out
    out = _ask_refinement(message, df, memory, model_on, started, sc)
    if out is not None:
        return out
    rks = followup.rank_requests(message, dfs)
    rk = rks[0] if rks else None
    if rks:
        memory["last_breakdown"] = {"kind": "rank", "rks": [dict(x) for x in rks],
                                    "entities": dict((sc or {}).get("entities") or {})}
    if rk is not None and not (model_on and calc.wanted(message, df, memory)):
        missing = next((x for x in rks if x["measure"] is None), None)
        if missing:
            from app.analyst.semantics import measure_options
            nums = measure_options(df)
            return _reply(f"Top {missing['n']} {missing['group']} by which value? For example: " + ", ".join(
                f"`{c}`" for c in nums[:8]) + ".", "tool", model_on, started, plan={"type": "clarify"})
        within = None
        lf = memory.get("last_focus") or {}
        if re.search(r"\b(its|their)\b", message, re.I) and lf.get("group_column") in dfs.columns \
                and all(lf["group_column"] != x["group"] for x in rks) and not scoping.active(sc):
            # "Show its top five cities" after "the highest revenue region is East": the cities of East
            within = (lf["group_column"], lf["group"])
            dfs = dfs[dfs[within[0]].astype(str) == str(within[1])]
        texts, charts, calls = [], [], []
        for x in rks:
            r = followup.rank_groups(dfs, x["group"], x["measure"], x["n"], x["low"], x.get("agg"), x.get("extras"))
            charts.append({"type": "bar", "additive": r["additive"], "x": x["group"], "y": x["measure"],
                           "title": f"{'Bottom' if x['low'] else 'Top'} {x['n']} {x['group']} by "
                                    f"{r['aggregation'] if r['aggregation'] == 'mean' else 'total'} {x['measure']}",
                           "data": [{x["group"]: g["group"], x["measure"]: g["value"]} for g in r["rows"]]})
            texts.append(followup.rank_text(r, x["assumed_measure"]))
            calls.append({"tool": "rank_groups", "status": "ok", "result": _jsonable(r)})
        text = "\n\n".join(texts)
        if within:
            text = (f"**Within {within[0]} = {within[1]}** (“its” — the leading {within[0]} of the previous answer; "
                    f"{len(dfs):,} records):\n\n" + text)
        return _reply(text, "tool", model_on, started, charts=charts, calls=calls,
                      notes=["Ranked by Silt's deterministic tools over "
                             + (f"the records in scope ({scoping.describe(sc, len(dfs), len(df))})." if scoping.active(sc)
                                else f"the {within[0]} = {within[1]} records." if within else "all records.")])
    out = _ask_next_record(message, df, memory, model_on, started, sc)
    if out is not None:
        return out
    # 3. "predict the next value": a forecast over time, ML only when the history supports it
    if followup.wants_forecast(message) or re.search(r"\bpredict\w*\b.*\bnext\b", message, re.I):
        out = _ask_forecast(message, df, memory, model_on, started, sc)
        if out is not None:
            return out
    # A what-if or goal sentence about a dataset that fits a built-in model goes to the shared what-if service (the same one the
    # What-If page uses). It acts only when the sentence reads cleanly on the model's own inputs; anything else returns None here
    # and is answered exactly as it was before.
    bridged = scenario_bridge.try_answer(message, dfs, memory, filename,
                                         scope_note=(scoping.describe(sc, len(dfs), len(df)) if scoping.active(sc) else None))
    if bridged is not None:
        return _reply(bridged["answer"], "tool", model_on, started, plan=bridged["plan"], calls=bridged["calls"], notes=bridged["notes"],
                      details=bridged.get("details", ""))
    # The universal what-if (education, healthcare, travel, government, marketing, hiring...) is the same engine the What-If page runs. It
    # answers only sentences that match one of its scenario templates and returns None for the rest, so nothing answered before changes.
    universal = whatif_bridge.try_answer(message, dfs, memory, filename,
                                         scope_note=(scoping.describe(sc, len(dfs), len(df)) if scoping.active(sc) else None))
    if universal is not None:
        return _reply(universal["answer"], "tool", model_on, started, plan=universal["plan"], calls=universal["calls"], notes=universal["notes"],
                      details=universal.get("details", ""))
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
    # one turn's model budget: after the program planner has spent most of it, the other model paths use their rules
    more = model_on and memory.get("_model_spent_ms", 0) < MODEL_TURN_BUDGET_MS
    why_calc = (calc.wanted(message, df, memory) or ("the question sets its own conditions" if deferred else None)) \
        if more else None
    failed = None
    if why_calc:
        out = _ask_calc(message, df, memory, why_calc, started, sc=sc)
        if deferred and out is not None and out["plan"].get("type") == "calc":
            # the model was asked because of a condition Silt could not apply; a plan with no rule of its own (beyond
            # validity, duplicates and the question's scope) did not apply it either, so it is not an answer
            own = [e for e in (out["plan"].get("exclude") or []) if not re.search(
                r"duplicate\(\)|<\s*0\b|isnull\(|!=\s*'|not\s+in\b|\byear\(|\bmonth\(|hours_between\([^)]*\)\s*<\s*0",
                e.get("where", ""))]
            # ...and each of its own rules must come from the data or the question: a column or value sharing a word
            # with the condition ("status == 'cancelled'"), or a number/time the user stated ("clock_in < '09:00'").
            # An invented proxy ("cancelled shift = hourly_rate == 0") is not an answer.
            cond_words = {w[:5] for x in deferred["unhandled"] for w in re.findall(r"[a-z]{4,}", x.lower())}
            def values(text):                       # 9:00 and 09:00 are the same time; 8 and 8.0 the same number
                return {(int(h) * 60 + int(m or 0)) if ":" in t else float(t)
                        for t, h, m in ((x, *(x.split(":") + [None])[:2]) for x in re.findall(r"\d+(?:\.\d+)?(?::\d+)?", text))}
            stated = values(" ".join(deferred["unhandled"]))
            def grounded(e):
                words = {w[:5] for w in re.findall(r"[a-z]{4,}", e.get("where", "").lower().replace("_", " "))}
                nums = values(e.get("where", ""))
                return bool(words & cond_words) or bool(nums and nums & stated)
            invented = [e for e in own if not grounded(e)]
            if not own or invented:
                if invented:
                    deferred["invented"] = invented
                out = None
        if out is not None and out["plan_source"] not in ("model_calc_failed",) and out["plan"].get("type") == "calc" \
                and out.get("calls"):
            return out
        if out is not None and not deferred and out["plan_source"] != "model_calc_failed":
            return out
        failed = out
    if deferred:
        guess = "; ".join(f"`{e['where']}`" for e in deferred.get("invented", []))
        return _reply(f"I can compute **{deferred['label']}** ({deferred['definition']}), but I couldn't apply "
                      + "; ".join(f"*{x}*" for x in deferred["unhandled"]) + " to this data — nothing in its columns "
                      "records that" + (f" (the local model proposed {guess}, which the data does not support)" if guess
                                        else ", and the local model could not turn it into a check either") +
                      ". I haven't shown a number that ignores it, or a different quantity. Ask without that condition, "
                      "or say which column and value identify it.", "metric", model_on, started,
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
    more = model_on and memory.get("_model_spent_ms", 0) < MODEL_TURN_BUDGET_MS
    out = _ask_planned(message, dfs, memory, more, started)
    asked = [o for o in (out["plan"].get("options") or []) if o in dfs.columns]
    if more and out["plan"].get("type") == "clarify" and out["plan_source"] == "model" and \
            len([o for o in asked if scoping.column_named(message, o)]) == 1:
        # "Predict customer churn." -> "Which column should I predict? churn, ...": the question already names one of
        # the options, so there is no choice to make; Silt's deterministic plan answers if it can
        rules = _ask_planned(message, dfs, memory, False, started)
        if any(c.get("status") == "ok" for c in rules.get("calls", [])):
            rules["notes"].insert(0, "The model asked which column to use, but the question names it; Silt's "
                                     "deterministic plan answered.")
            out = rules
    from app.analyst import intent
    _req = intent.read(message, df)
    if more and _req.group and any(c.get("status") == "ok" for c in out.get("calls", [])) and not any(
            (c.get("args") or {}).get(k) == _req.group for c in out.get("calls", []) for k in ("group", "group_by")):
        # the request breaks the result down ("each outlet") but the model's plan does not: the deterministic plan,
        # if it does, answers instead
        rules = _ask_planned(message, dfs, memory, False, started)
        if any((c.get("args") or {}).get(k) == _req.group for c in rules.get("calls", []) for k in ("group", "group_by")) \
                and any(c.get("status") == "ok" for c in rules.get("calls", [])):
            rules["notes"].insert(0, f"The model's plan left out the breakdown by {_req.group}; Silt's deterministic "
                                     "plan, which keeps it, answered instead.")
            out = rules
    ug = _req.unknown_group
    if ug and any((c.get("args") or {}).get(k) not in (None, "") and str((c.get("args") or {}).get(k)) not in
                  followup_cols(df, message) for c in out.get("calls", []) for k in ("group", "group_by")):
        # "compare the tips each waiter earns" with no waiter column: grouping by another column is not an answer
        return _dataset_reply(message, df, memory, model_on, started, out)
    mismatch = plan_checks.metric_mismatch(message, out.get("calls", []), dfs)
    if mismatch and model_on:
        # the deterministic plan for the same question, before asking the model again
        rules = _ask_planned(message, dfs, memory, False, started)
        if any(c.get("status") == "ok" for c in rules.get("calls", [])) and \
                not plan_checks.metric_mismatch(message, rules.get("calls", []), dfs):
            rules["notes"].insert(0, f"Silt rejected the model's plan because {mismatch}; its deterministic plan "
                                     "answered instead.")
            return rules
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
            and not any(c.get("status") == "ok" for c in out["calls"]) and not _reply_stands(out, message, df):
        # no deterministic path and no catalogue tool answered: the model plans an analysis program
        alt = _ask_nl_program(message, df, memory, model_on, started, filename, late=True)
        if alt is not None:
            return alt
        alt = _ask_calc(message, df, memory, "the trained tool catalogue had no way to answer it", started,
                        previous=out, sc=sc) if memory.get("_model_spent_ms", 0) < MODEL_TURN_BUDGET_MS else None
        if alt is not None:
            out = alt
    if opts and any(c.get("tool") == "random_forest" for c in out["calls"]):
        out["notes"].insert(0, "Training options taken from your message: " + ", ".join(
            f"{'test share ' + format(v * 100, 'g') + '%' if k == 'test_size' else 'seed ' + str(v)}" for k, v in opts.items()) + ".")
    out = _after_failed(out, failed)
    if model_on and not any(c.get("status") == "ok" for c in out.get("calls", [])) and \
            out["plan"].get("type") not in ("clarify", "decline") and not _reply_stands(out, message, df):
        # nothing computed an answer (e.g. the calculation plan failed its checks): the program planner, if not tried
        alt = _ask_nl_program(message, df, memory, model_on, started, filename, late=True)
        if alt is not None:
            return alt
    if out["plan"].get("type") == "chat" and _VAGUE.search(out["answer"]):
        # a placeholder reply ("I'm not sure what to compute") is not an answer: use the dataset context instead
        out = _dataset_reply(message, df, memory, model_on, started, out)
        why = memory.pop("_nl_unsupported", None)
        if why:
            out["answer"] += f"\n\n_The local model's reading of the request: {why}_"
    return out


def _reply_stands(out: dict, message: str, df) -> bool:
    """A conversational reply or a reasoned decline to a message that names nothing in the data ("Hello.", "What is
    Random Forest?", "Which area should get the next campaign?" on data with no campaigns) is the answer -- not a dead
    end for the analysis-program planner to replace with a computation nobody asked for. A placeholder reply, or a
    message that names a column, is still handed on."""
    from app.analyst.data import find_column
    if out["plan"].get("type") not in ("chat", "decline") or out.get("plan_source") == "rules":
        return False
    if _VAGUE.search(out.get("answer") or "") or df is None:
        return False
    return not find_column(df, message)


MODEL_TURN_BUDGET_MS = 12000          # model time after which a turn stops consulting further model paths
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
                            + str((failed.get("first_pass") or {}).get("error") or "see trace")[:200]
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


_ANALYSIS_WORDS = set("""change changed changes come comes coming doesn don didn isn aren wasn different difference
differences appear appears associated association factors factor identify important entire whole problems problem issues
issue business days day weeks week months month years year hours hour time target targets goal goals risk risks
missing team workload capacity absent absence explain explains why happen happens happened would will could
increase increased decrease decreased costs cost operating trend trends pattern patterns unusual investigate possible
explanations explanation recommend option options feasible better best worse worst others other next highest lowest
largest smallest biggest top bottom something anything everything stuff interesting info information insight insights
have has had there typical distinct unique different separate rate rates share percentage proportion""".split())
_STOPWORDS = set("""a an the and or but if then than that this these those it its he she him her his they them their we
our you your i me my to of in on at by for with from as is are was were be been being do does did can could should would
will may might must shall what which who whom whose when where why how all any each every some no not only just also very
more most less least much many few other others same such own so too up down out over under again further once here there
about against between into through during before after above below off now please show tell give make let lets get got
analyse analyze analysis instead forget compare comparison result results previous scenario scenarios thing things data
dataset file value values number numbers total average mean sum count""".split())
_SWITCH = re.compile(r"\b(?:analy[sz]e|switch to|use|open|look at|work on|move to)\s+(?:the\s+)?([a-z][a-z_]+)"
                     r"(?:\s+(?:data|dataset|file|sheet|table))?\s*(?:instead)?\b", re.I)
_REFER = re.compile(r"\b(him|her|them|that|this|it|those|these|previous|earlier|last|the others|scenario|result|before)\b",
                    re.I)


def _concept_offer(words, df):
    from app.analyst import semantics
    for w in words:
        c = semantics.concept_candidates(w, df)
        if c:
            return w, c[:3]
    return None


def _dataset_reply(message, df, memory, model_on, started, previous) -> dict:
    """Nothing could answer the message. Say specifically what could not be connected and to what -- the open
    request, words that are not in this data, another dataset -- and how to go on; never a generic list."""
    from app.analyst.calc import named_values
    from app.analyst.data import find_column
    words = [w for w in re.findall(r"[a-z]+", message.lower()) if len(w) > 3 and w not in _STOPWORDS
             and w not in _ANALYSIS_WORDS]
    from app.analyst import intent
    # a word is known when it is part of a column's name (plurals too: "companies" for company_type) or of a value the
    # question names -- including values of long text columns ("Accenture" among 4,000 company names)
    named = named_values(df, message) + intent.names_in_long_columns(message, df, [])[0]
    known = {t for c in df.columns for t in re.split(r"[^a-z0-9]+", str(c).lower())} | {
        w for _, v in named for w in re.findall(r"[a-z]+", v.lower())}
    known_lemmas = {intent.lemma(t) for t in known if t}
    unknown = [w for w in dict.fromkeys(words) if w not in known and w.rstrip("s") not in known
               and re.sub(r"ies$", "y", w) not in known and intent.lemma(w) not in known_lemmas and not find_column(df, w)]
    parts = [f"I couldn't turn **“{message.strip()}”** into an analysis of this dataset."]
    sw = _SWITCH.search(message)
    refers = bool(_REFER.search(message)) and bool(memory.get("last_calc") or memory.get("last_answer"))
    if sw and not find_column(df, sw.group(1)) and sw.group(1).lower() not in _STOPWORDS:
        parts.append(f"This conversation works on the selected dataset ({len(df):,} records; columns "
                     + ", ".join(f"`{c}`" for c in df.columns[:8]) + f"). To analyse **{sw.group(1)}**, choose that file in "
                     "the **Dataset** selector — each dataset has its own conversation.")
    elif unknown and not refers and _concept_offer(unknown, df):
        offer = _concept_offer(unknown, df)
        parts.append(f"This data has nothing called **{offer[0]}**, but " + " or ".join(f"`{c}`" for c in offer[1])
                     + " may mean the same thing. Ask again using that column, for example with *" + offer[1][0] + "*.")
    elif not refers and _ranks_without_measure(message, df):
        g = _ranks_without_measure(message, df)
        from app.analyst import semantics as _s
        parts.append(f"Highest or lowest **{g}** by which value? This question names none. Options: "
                     + ", ".join(f"`{c}`" for c in _s.measure_options(df, limit=6)) + ".")
    elif not refers:
        missing = _missing_terms(message, df)
        if missing:
            kind, term = missing[0]
            parts.append(f"This data has nothing called **{term}** — no column or value by that name"
                         + (", so there is nothing to count by it." if kind == "count" else
                            ", so I can't tell which values to use for it.")
                         + " I haven't substituted another column.")
        elif unknown:
            parts.append("This data has nothing called " + ", ".join(f"**{w}**" for w in unknown[:3])
                         + " — no column or value by that name.")
    lr, last = memory.get("last_result"), memory.get("last_calc")
    if _REFER.search(message) and last and not _superseded(memory, "last_calc"):
        res = last["result"]
        parts.append(f"The current result is from *{last['question']}*: {present.calc_headline(res, last['question'])} "
                     "You can narrow it (*only <name>*), break it down (*by <column>*), exclude duplicates, chart it or "
                     "ask me to explain it.")
    elif _REFER.search(message) and lr:
        parts.append(f"The latest result is from *{lr['question'][:120]}*. You can narrow it (*what about <name>?*, "
                     "*only <value>*), compare it (*compare it with <name>*), change the figure (*the average "
                     "instead*), break it down (*by <column>*), chart it, or ask *why?*.")
    elif _REFER.search(message) and last:
        res = last["result"]
        parts.append(f"The current result is from *{last['question']}*: {present.calc_headline(res, last['question'])} "
                     "You can narrow it (*only <name>*), break it down (*by <column>*), exclude duplicates, chart it or "
                     "ask me to explain it.")
    elif _REFER.search(message) and memory.get("last_answer"):
        parts.append(f"The last answer was for *{memory['last_answer']['question'][:100]}*; I couldn't tell how your "
                     "message relates to it. Say what to change, or ask me to explain it.")
    else:
        parts.append(_what_is_available(df))
    return _reply("\n\n".join(parts), "conversation", model_on, started, plan={"type": "chat", "fallback": True},
                  notes=previous.get("notes", []) + ["Nothing could compute this message; Silt said what it could not "
                                                     "connect instead of guessing."])


_ASKED_FOR = re.compile(r"\b(average|mean|median|typical|total|sum of|number of|how many|count of|highest|lowest|most|"
                        r"least|maximum|minimum|forecast|predict|trend of|share of|percentage of|rate of)\s+(?:the\s+|our\s+|"
                        r"all\s+|different\s+|distinct\s+|unique\s+)?([a-z][a-z_]*(?:\s+[a-z][a-z_]*)?)", re.I)


def _missing_terms(message: str, df) -> list[tuple[str, str]]:
    """The thing a question asks about -- the noun after its quantity word ("the typical blorp", "how many unicorns")
    -- when nothing in the data is called that: [(kind, term)], kind "count" or "measure"."""
    from app.analyst import intent
    from app.analyst.data import find_column
    vocab, _ = intent._vocabulary(df)  # noqa: SLF001
    out = []
    for m in _ASKED_FOR.finditer(message):
        words = [w for w in m.group(2).lower().split() if w not in _STOPWORDS and w not in _ANALYSIS_WORDS
                 and w not in ("per", "by", "each", "in", "for")]
        if not words:
            continue
        w = words[0]
        if w in vocab or w.rstrip("s") in vocab or find_column(df, w) or intent.lemma(w) in {intent.lemma(v) for v in vocab
                                                                                           if len(v) > 4}:
            continue
        out.append(("count" if m.group(1).lower() in ("how many", "number of", "count of") else "measure", w))
    return out


def _ranks_without_measure(message: str, df) -> str | None:
    """'Which region has the highest?': a ranking of groups that names nothing to rank them by -- the group."""
    from app.analyst import intent
    try:
        r = intent.read(message, df)
    except Exception:  # noqa: BLE001
        return None
    return r.group if r.extreme and r.group and not r.measures and not r.event and not r.counted else None


def _what_is_available(df) -> str:
    """What this data can answer, from its own columns: the measures (never identifiers), the groups, the dates, and
    three questions built from them."""
    from app.analyst import overview, semantics
    prof = semantics.profile(df)
    meas = semantics.measure_options(df, limit=6)
    groups = [c for c in prof["groups"] if prof["roles"][c]["role"] == "category"][:5]
    flags = [c for c in prof["flags"]][:3]
    lines = ["Here is what this data can answer:"]
    if meas:
        lines.append("- **Measures** (totals, averages, rankings, trends): " + ", ".join(f"`{c}`" for c in meas))
    if groups:
        lines.append("- **Groups** to break results down by: " + ", ".join(f"`{c}`" for c in groups))
    if flags:
        lines.append("- **Yes/no columns** (rates, filters such as *only …*): " + ", ".join(f"`{c}`" for c in flags))
    if prof["dates"]:
        lines.append(f"- **Dates** for periods and trends: `{prof['dates'][0]}`")
    lines.append(f"- **Counts**: the {len(df):,} records, or the distinct values of any column")
    sugg = overview.understand(df)["suggested_questions"][:3]
    if sugg:
        lines += ["", "For example:"] + [f"- {q}" for q in sugg]
    return "\n".join(lines)


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
    if sc and sc.get("dropped") and out.get("plan_source") not in ("conversation", "conversation_rules", "input_check"):
        out["answer"] = "_" + "; ".join(sc["dropped"]) + "._\n\n" + out["answer"]
        out.setdefault("notes", []).insert(0, "Filters not applied: " + "; ".join(sc["dropped"]) + ".")
    if out.get("plan_source") not in ("conversation", "conversation_rules", "input_check"):
        checks = list(out.get("verification") or [])
        if scoping.active(sc) and out.get("plan_source") not in ("model_program", "compound"):
            # (a compound answer scopes each deliverable to its own clause and checks it there; a person named in one
            # clause is not the scope of the others, so the message-wide scope check does not apply to it)
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
            if answered and "**Scope:**" not in out["answer"] and out.get("plan_source") not in ("model_calc",) \
                    and not (out.get("plan") or {}).get("parts"):        # each clause states its own records
                out["answer"] = f"**Scope:** {checks[-1]['detail']}.\n\n" + out["answer"]
        checks += out.pop("_post_checks", [])     # answer verification (verify.py), after the path's own checks
        if checks:
            out["verification"] = checks
            out["details"] = (out.get("details") or "") + ("\n\n" if out.get("details") else "") + \
                "**Checks before answering**\n" + "\n".join(f"- {'✓' if c['ok'] else '✗'} {c['check']}: {c['detail']}"
                                                         for c in checks)
        # a chart of the previous result, or an estimate built from it, is the same analysis: it replaces neither the
        # calculation that follow-ups like "Only Rahul." refine, nor the scope they keep
        redraw = answered and all(c.get("tool") in ("chart_of_previous_result", "chart_of_previous_answer",
                                                    "next_record_estimate", "compare_scenarios")
                                  for c in out.get("calls", []) if c.get("status") == "ok")
        # an analysis-program turn that names nobody ("change the target to 320") works on the same people as before
        keep_scope = out.get("plan_source") == "program" and not scoping.active(sc)
        explains = (out.get("plan") or {}).get("intent") == "why_previous"   # explains the latest result; not a new one
        if answered and not redraw and not explains:
            _record_last_result(out, message, df, memory, sc)
        if answered and not redraw and not keep_scope:
            memory["scope"] = sc if scoping.active(sc) else None
            memory["last_analysis"] = "calc" if out.get("plan_source") in ("model_calc", "metric", "refine") else "other"
            if memory["last_analysis"] == "other" and not scoping.active(sc):
                _metric_as_last_calc(out, message, df, memory)
    evidence = [c.get("result") for c in out.get("calls", []) if c.get("status") == "ok"]
    followup.remember(memory, message, out, evidence[:3])
    from app.analyst import register
    register.record(memory, message, out)
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
    if not last or _superseded(memory, "last_calc"):
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
_TARGET_Q = re.compile(r"\b(targets?|goals?|quotas?|at risk of (?:missing|not)|on track|fall(?:ing)? short|miss(?:ing)? "
                       r"(?:their|the|his|her|its)?\s*(?:target|goal|quota))\b", re.I)


def _ask_missing_target(message, df, memory, model_on, started) -> dict | None:
    """A question about targets with no target to compare against: asked for, never answered with something else
    (a monthly trend of another column is not 'at risk of missing targets')."""
    from app.analyst.data import find_column
    if not _TARGET_Q.search(message):
        return None
    if [c for c in df.columns if re.search(r"target|goal|quota|plan|budget|forecast", str(c), re.I)]:
        return None                    # the data has a target column: other paths can compare against it
    if re.search(r"\b\d+(?:[.,]\d+)?\b", message) or (memory.get("target") or {}).get("value"):
        return None                    # a stated target: the feasibility path handles it
    nums = [c for c in find_column(df, message) if df[c].dtype != object]
    what = nums[0].replace("_", " ") if nums else "the measure"
    return _reply(f"**What is the target?** This data has no target column, so I can't tell who is at risk without one. "
                  f"Tell me the value and the period — for example *a monthly target of 160 hours per employee* or "
                  f"*{what} of 500 per month* — and I'll compare each one's pace with it.", "tool", model_on, started,
                  plan={"type": "clarify", "expects": "target"},
                  notes=["The question needs a target the data does not contain; Silt asks rather than substituting "
                         "another analysis."])


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


_INVESTIGATE = re.compile(r"\b(?:analy[sz]e|investigate|audit|examine|explore|review|study|go through|look through|"
                          r"dig into|scan)\b[^?]{0,30}\b(?:entire|whole|full|all|complete)?\s*(?:this|the|my)?\s*(?:data|dataset|"
                          r"data ?set|file|table|spreadsheet)\b|\b(?:find|identify|discover|spot|surface|show)\b[^?]{0,25}"
                          r"\b(?:important|key|main|biggest|major)?\s*(?:patterns|problems|issues|risks|insights|findings)\b|"
                          r"\bwhat(?:'s| is| are)?\s+(?:wrong|going on|interesting|important)\b[^?]{0,20}\b(?:data|dataset|"
                          r"here)\b", re.I)
_FINDING = re.compile(r"\b(?:finding|point|item|issue|number|#)\s*(\d+)\b", re.I)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
             "ninth": 9, "tenth": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "last": -1, "final": -1}
_REF_NOUN = r"(?:things?|findings?|points?|items?|issues?|results?|patterns?|insights?|observations?|ones?)"
_FINDING_WORD = re.compile(r"\b(?:finding|point|item|issue|number)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b"
                           r"|\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|"
                           r"(\d+)(?:st|nd|rd|th))\s+(?:\w+\s+)?" + _REF_NOUN + r"\b", re.I)


def finding_ref(message: str, n: int) -> int | None:
    """Which stored finding a message points at, 1-based: 'finding 2', 'finding two', 'the second thing you found',
    'the 3rd point', 'the last finding'. None when it names none; an index outside 1..n is returned as is (asked)."""
    m = _FINDING.search(message)
    if m:
        return int(m.group(1))
    m = _FINDING_WORD.search(message)
    if not m:
        return None
    w = (m.group(1) or m.group(2)).lower()
    i = int(m.group(3)) if m.group(3) else _ORDINALS[w]
    return n if i == -1 else i


def _ask_investigation(message, df, memory, model_on, started, filename) -> dict | None:
    """'Analyse this entire dataset and identify important patterns': the investigation (app/analyst/investigate.py)
    picks analyses that fit this data and reports evidence-backed findings. 'Tell me more about finding 2' and
    'dig into finding 2' refer to them afterwards."""
    from app.analyst import investigate
    stored = memory.get("investigation")
    fi = finding_ref(message, len(stored["findings"])) if stored else None
    if stored and fi is not None:
        i = fi
        if not 1 <= i <= len(stored["findings"]):
            return _reply(f"There are {len(stored['findings'])} findings; which one?", "tool", model_on, started,
                          plan={"type": "clarify"})
        x = stored["findings"][i - 1]
        if re.search(r"\b(dig|investigate|analy[sz]e|drill|explore|run|check)\b", message, re.I):
            return _dispatch(x["follow_up"], df, memory, model_on, filename, started, followup.chart_request(x["follow_up"]))
        return _reply(investigate.detail(x, i), "program", model_on, started, plan={"type": "program", "objective": "finding"},
                      calls=[{"tool": "investigation_finding", "status": "ok", "result": _jsonable(x)}])
    if not _INVESTIGATE.search(message) or re.search(r"\b(trend|chart|plot|predict|forecast)\b", message, re.I):
        return None
    if [c for c in followup_cols(df, message)]:
        return None                    # "analyse sales by region" names a column: an ordinary question
    try:
        r = investigate.investigate(df)
    except Exception as e:  # noqa: BLE001
        logger.exception("investigation failed")
        return _reply(f"**The investigation failed** ({type(e).__name__}: {e}). Nothing was estimated instead.", "tool",
                      model_on, started, plan={"type": "error"})
    memory["investigation"] = {"findings": r["findings"], "program": r["program"]}
    memory["program"] = {"investigation": {"steps": r["program"], "results": r["results"]}}
    charts = []
    if r["findings"]:
        charts = [{"type": "bar", "x": "finding", "y": "importance", "additive": False, "title": "Findings by importance",
                   "data": [{"finding": f"{i}. {x['title'][:40]}", "importance": round(x["score"], 2)}
                            for i, x in enumerate(r["findings"][:8], 1)]}]
    details = "**Analysis program**\n" + "\n".join(
        f"- `{st['id']}` **{st['op']}** " + ", ".join(f"{k}={v}" for k, v in st.items() if k not in ("id", "op"))
        for st in r["program"])
    return _reply(investigate.text(r, filename), "program", model_on, started,
                  plan={"type": "program", "objective": "investigation"},
                  calls=[{"tool": "analysis_program", "status": "ok",
                          "result": {"objective": "investigation", "programs": {"": r["program"]},
                                     "results": {"": r["results"]}, "findings": _jsonable(r["findings"])}}],
                  charts=charts, details=details,
                  notes=["Analyses were chosen from what the columns are (semantic profile) and run as one validated "
                         "program; findings below a stated evidence bar are not reported."])


_PERFORM = re.compile(r"\b(perform\w*|underperform\w*|doing (?:best|worst|badly|well|poorly)|struggl\w*|"
                      r"(?:best|worst|weakest|strongest) (?:performer|performing|results?))\b|\b(?:best|worst)\b", re.I)


_AGG_SWITCH = re.compile(r"\b(median|mean|average|total|sum|maximum|max|minimum|min)\b(?:[^.?!]{0,30}\binstead\b)|"
                         r"\b(?:use|show|give|try)\s+(?:me\s+)?(?:the\s+)?(median|mean|average|total|sum|maximum|minimum)\b",
                         re.I)
_RERANK = re.compile(r"\b(?:rank|sort|order|re-?rank)\w*\s+(?:them|these|those|it|the list|the companies|the results)?\s*"
                     r"(?:by|on)\s+([^.?!]+)|\bby\s+([^.?!]+?)\s+instead\b", re.I)
_DIRECTION = re.compile(r"\b(?:show\s+(?:me\s+)?)?(?:the\s+)?(bottom|lowest|worst|top|highest|best)\s*(\d{1,2})?\b"
                        r"(?=[^.?!]{0,20}\binstead\b)|\bshow\s+(?:me\s+)?the\s+(bottom|top)\s*(\d{1,2})\b", re.I)
_ONLY = re.compile(r"^\s*(?:(?:now|and|ok|okay|then)\s*,?\s*)?(?:only|just|but only|limit (?:it|that|this) to|restrict (?:it|that) to|"
                   r"what about|now for|for)\b", re.I)


def _ask_refinement(message, df, memory, model_on, started, sc) -> dict | None:
    """A follow-up that changes one thing about the last ranking or grouped figure -- "rank them by reviews instead",
    "now the median", "the bottom 5 instead", "only Private companies" -- re-run with that one change. The grouping,
    measure, aggregation, size and earlier filters are kept unless the follow-up changes them, and the answer says
    what changed and what was kept. Nothing else is reused."""
    last = memory.get("last_breakdown")
    if not last or followup.rank_requests(message, df) or _superseded(memory, "last_breakdown"):
        return None
    from app.analyst.data import find_column, kind_of
    from app.analyst.intent import ranking_aggregate
    new_ents = dict((sc or {}).get("entities") or {}) if scoping.active(sc) else {}
    agg_m, rr, dm = _AGG_SWITCH.search(message), _RERANK.search(message), _DIRECTION.search(message)
    only = bool(new_ents) and (bool(_ONLY.search(message)) or (len(message.split()) <= 4 and not re.search(
        r"\b(why|what|how|which|who|when|is|are|does|do|explain|compare|versus|vs)\b", message, re.I)))
    if not (agg_m or rr or dm or only):
        return None
    changed = []
    ents = {**(last.get("entities") or {}), **new_ents}
    if new_ents:
        changed.append("records limited to " + "; ".join(f"{c} = {' or '.join(map(str, v))}" for c, v in new_ents.items()))
    d = df
    for c, vals in ents.items():
        if c in df.columns:
            d = d[d[c].astype(str).str.lower().isin([str(v).lower() for v in vals])]
    if d.empty:
        return _reply("No records match " + "; ".join(f"{c} = {' or '.join(map(str, v))}" for c, v in ents.items())
                      + ", so there is nothing to rank.", "tool", model_on, started, plan={"type": "clarify"})
    agg_word = None
    if agg_m:
        agg_word = (agg_m.group(1) or agg_m.group(2)).lower()
    to_agg = {"median": "median", "mean": "mean", "average": "mean", "total": "sum", "sum": "sum", "maximum": "max",
              "max": "max", "minimum": "min", "min": "min"}
    new_measure = None
    if rr:
        txt = rr.group(1) or rr.group(2)
        nums = [c for c in find_column(df, txt) if kind_of(df[c]) == "numeric"]
        if not nums:
            from app.analyst.followup import _stem_column
            nums = [c for c in (_stem_column(w, df) for w in re.findall(r"[a-z]+", txt.lower())) if c]
        if not nums:
            return None
        new_measure = nums[0]
    if last["kind"] == "rank":
        rks = [dict(x) for x in last["rks"]]
        for x in rks:
            if new_measure and new_measure != x["measure"]:
                changed.append(f"ranked by {new_measure} instead of {x['measure']}")
                x["measure"], x["agg"], x["extras"] = new_measure, ranking_aggregate(message, new_measure), []
            if agg_word:
                changed.append(f"{agg_word} instead of {x.get('agg') or 'the earlier aggregate'}")
                x["agg"] = to_agg[agg_word]
            if dm:
                word = (dm.group(1) or dm.group(3) or "").lower()
                num = dm.group(2) or dm.group(4)
                low = word in ("bottom", "lowest", "worst")
                if low != x["low"]:
                    changed.append("bottom instead of top" if low else "top instead of bottom")
                x["low"] = low
                if num:
                    x["n"] = int(num)
        memory["last_breakdown"] = {"kind": "rank", "rks": rks, "entities": ents}
        texts, calls = [], []
        for x in rks:
            r = followup.rank_groups(d, x["group"], x["measure"], x["n"], x["low"], x.get("agg"), x.get("extras"))
            texts.append(followup.rank_text(r, False))
            calls.append({"tool": "rank_groups", "status": "ok", "args": {k: x.get(k) for k in ("group", "measure", "n",
                          "low", "agg")}, "result": _jsonable(r)})
        kept = f"kept: grouping by {rks[0]['group']}" + (", earlier filters " + "; ".join(
            f"{c} = {' or '.join(map(str, v))}" for c, v in (last.get("entities") or {}).items()) if last.get("entities") else "")
        head = f"_Changed from the previous answer: {'; '.join(dict.fromkeys(changed)) or 'nothing'} ({kept})._\n\n"
        if ents and not new_ents:          # a new filter is already stated in the answer's scope line
            head += f"**Records:** {'; '.join(f'{c} = ' + ' or '.join(map(str, v)) for c, v in ents.items())} — {len(d):,} of {len(df):,}.\n\n"
        return _reply(head + "\n\n".join(texts), "tool", model_on, started, calls=calls,
                      plan={"type": "tools", "intent": "refinement"},
                      notes=["A follow-up to the previous ranking: re-run by Silt's deterministic tools with the change "
                             "named in the question."])
    args = dict(last["args"])
    if new_measure:
        changed.append(f"{new_measure} instead of {args.get('column')}")
        args["column"] = new_measure
        args["aggregation"] = {"sum": "sum", "mean": "avg", "median": "median"}.get(ranking_aggregate(message, new_measure)
                                                                                   or "", args["aggregation"])
    if agg_word:
        changed.append(f"{agg_word} instead of {args['aggregation']}")
        args["aggregation"] = {"mean": "avg"}.get(to_agg[agg_word], to_agg[agg_word])
    if dm:
        args["order"] = "asc" if (dm.group(1) or dm.group(3) or "").lower() in ("bottom", "lowest", "worst") else "desc"
    memory["last_breakdown"] = {"kind": "metric", "args": args, "entities": ents}
    try:
        call = tools.validate_call({"tool": "compute_metric", "args": args}, d, message)
    except tools.RoleRefusal as e:
        memory["last_breakdown"] = last                 # the refused change is not kept
        return _reply("**I can't compute that.** " + str(e), "tool", model_on, started,
                      plan={"type": "decline", "role_refusal": True})
    r = tools.run(call, d, memory)
    head = f"_Changed from the previous answer: {'; '.join(dict.fromkeys(changed)) or 'nothing'} (kept: grouping by " \
           f"{args.get('group_by')})._\n\n"
    if ents and not new_ents:
        head += f"**Records:** {'; '.join(f'{c} = ' + ' or '.join(map(str, v)) for c, v in ents.items())} — {len(d):,} of {len(df):,}.\n\n"
    return _reply(head + explain.render("compute_metric", r), "tool", model_on, started,
                  calls=[{**call, "status": "ok", "result": _jsonable(r)}], plan={"type": "tools", "intent": "refinement"},
                  notes=["A follow-up to the previous breakdown: re-run with the change named in the question."])


def _undefined_performance(message, df) -> dict | None:
    """{group, options} when the request ranks groups by an undefined notion of performance and names no measure."""
    from app.analyst import intent, semantics
    if not _PERFORM.search(message):
        return None
    try:
        r = intent.read(message, df)
    except Exception:  # noqa: BLE001
        return None
    if r.measures or not r.group or r.per or r.restrict:
        return None
    prof = semantics.profile(df)
    opts = [c for c in prof["outcomes"] if c != r.group] + [c for c in prof["measures"] if c != r.group]
    opts = list(dict.fromkeys(map(str, opts)))[:6]
    return {"group": r.group, "options": opts} if len(opts) >= 2 else None


def _ask_nl_program(message, df, memory, model_on, started, filename, late) -> dict | None:
    """The local model plans an analysis program (app/analyst/nl_planner.py); Silt checks and runs it. None when the
    model is unavailable, its plan failed the checks, or (before the deterministic handlers) it found the request
    unsupported -- the deterministic paths then answer as before."""
    if memory.get("_nl_tried") == message:
        return None                    # one planning attempt per message (the late hook after the early one)
    memory["_nl_tried"] = message
    # a request whose every part is grounded gets its program without a model: the same checks, no inference wait
    if not nl_planner.follow_up(message, memory):
        out = _run_program(nl_planner.rule_plan(message, df, memory), message, df, memory, model_on, started,
                           "rule_program", {"status": "rule"}, [])
        if out is not None:
            return out
    p = nl_planner.plan(message, df, memory)
    ms = sum(t.get("ms", 0) for t in p["trace"])
    trace = {"attempts": len(p["trace"]), "model_ms": ms, "status": p["status"],
             "rejected": [x for t in p["trace"] for x in t.get("problems", [])][:3],
             "output": p["trace"][-1].get("output", "")[:3000] if p["trace"] else ""}
    note = (f"Planned by the local model in {len(p['trace'])} attempt(s), {ms / 1000:.1f} s"
            + (f"; Silt rejected a first plan ({trace['rejected'][0][:160]}) and the model corrected it"
               if trace["rejected"] and p["status"] == "ok" else "") + ".")
    memory["_nl_last"] = trace
    memory["_model_spent_ms"] = memory.get("_model_spent_ms", 0) + ms
    if p["status"] == "failed":
        logger.info("model program planning failed: %s", p["reason"])
        rp = nl_planner.rule_plan(message, df, memory)
        return _run_program(rp, message, df, memory, model_on, started, "rule_program", trace,
                            [note.rstrip(".") + "; its plan did not pass Silt's checks (" + p["reason"][:200]
                             + "), so Silt built the program from the request itself."]) if rp else None
    if p["status"] == "unsupported":
        # the planner never turns a request into a decline on its own: the deterministic paths (and their checked
        # declines) still answer; the reason is kept for the final fallback reply
        memory["_nl_unsupported"] = p["reason"]
        return None
    if p["status"] == "clarify":
        q = p["question"] + (("\n\nOptions: " + ", ".join(f"`{o}`" for o in p["options"])) if p["options"] else "")
        return _reply(q, "model_program", model_on, started, plan={"type": "clarify", "options": p["options"],
                                                                    "planner": trace},
                      notes=[note, "Silt asks rather than guessing; nothing was calculated."])
    if p["status"] == "reference":
        if p["kind"] == "rerun":
            key = "target" if p["key"] == "target" else "absence"
            cmd = (f"change the target to {p['value']:g}" if key == "target" else f"change the absence to {int(p['value'])} days")
            obj = objectives.handle(cmd, df, memory, None)
            if obj is None:
                return None
            obj["notes"].insert(0, f"The local model read this as a change to the open analysis ({cmd}); Silt re-ran it.")
            return _reply(obj["answer"], "model_program", model_on, started, plan={**obj["plan"], "planner": trace},
                          calls=obj["calls"], charts=obj["charts"], notes=obj["notes"] + [note],
                          details=obj.get("details", ""))
        if p["kind"] == "compare":
            hist = memory["objectives"]
            obj = objectives._compare_scenarios(hist[p["a"] - 1], hist[p["b"] - 1])  # noqa: SLF001
            return _reply(obj["answer"], "model_program", model_on, started, plan={**obj["plan"], "planner": trace},
                          calls=obj["calls"], charts=obj["charts"],
                          notes=[f"The local model read this as comparing analyses {p['a']} and {p['b']}."] + obj["notes"] + [note])
        if p["kind"] == "finding":
            verb = "dig into" if p["mode"] == "dig" else "tell me more about"
            out = _ask_investigation(f"{verb} finding {p['n']}", df, memory, model_on, started, filename)
            if out is not None:
                out["notes"].insert(0, f"The local model read this as being about finding {p['n']}.")
                out["plan"] = {**out.get("plan", {}), "planner": trace}
            return out
        return None
    out = _run_program(p, message, df, memory, model_on, started, "model_program", trace,
                       [note, "The model only wrote the plan; every step was validated and computed by Silt from the "
                              "records. The steps are under *How this was calculated*."])
    if out is None:
        failed_check = memory.pop("_verify_failed", None)
        memory["_nl_last"] = {**trace, "status": "verify_failed" if failed_check else "run_failed"}
        if failed_check:
            rp = nl_planner.rule_plan(message, df, memory)
            out = _run_program(rp, message, df, memory, model_on, started, "rule_program", memory["_nl_last"],
                               [note.rstrip(".") + "; its program did not pass Silt's answer verification ("
                                + "; ".join(c["detail"] for c in failed_check["checks"] if not c["ok"])[:200]
                                + "), so Silt built the program from the request itself."]) if rp else None
    return out


def _run_program(p, message, df, memory, model_on, started, source, trace, notes, request=None) -> dict | None:
    """Run a checked program (planned by the model, or built from the request) and present it. `request` is the
    reading the program was built for when it is not the message's own (a follow-up applied to the last result)."""
    if not p:
        return None
    try:
        r = nl_planner.execute(p, df, memory, message)
    except P.ProgramError as e:
        logger.info("analysis program failed at run time: %s", e)
        return None
    from app.analyst import verify
    # the question's scope (a named value, a year column value, a month) is part of what the figure must be computed
    # over: a program that leaves it out does not answer the question
    v = verify.check(message, df, r["steps"], r["results"], scope=memory.get("_scope"), request=request)
    if v["status"] == "failed":
        # the program ran but does not answer the request, or its figure differs from an independent recomputation:
        # not shown; the next path (Silt's own program, or its other handlers) answers instead
        logger.info("verification failed for %s: %s", source, [c["detail"] for c in v["checks"] if not c["ok"]])
        memory["_verify_failed"] = {"source": source, "checks": v["checks"]}
        memory.get("planned", {}).pop(r["name"], None)
        return None
    if source == "rule_program" and not notes:
        notes = ["Built by Silt from the request's parts (measure, grouping, comparison, why) without a model; every "
                 "step was validated and computed from the records."]
    return _reply(r["answer"], source, model_on, started, verification=v["checks"],
                  **({"_request": request} if request is not None else {}),
                  plan={"type": "program", "objective": "model_planned" if source == "model_program" else "rule_planned",
                        "planner": trace},
                  calls=[{"tool": "analysis_program", "status": "ok",
                          "result": {"objective": "model_planned" if source == "model_program" else "rule_planned",
                                     "programs": {"": r["steps"]}, "results": {"": r["results"]}, "name": r["name"]}}],
                  charts=r["charts"], details=r["details"], notes=notes)


def followup_cols(df, message):
    """Columns the message names as whole words ("analyse sales by region"), not by fuzzy similarity ("dataset" is
    not the column date)."""
    from app.analyst.data import _words
    low = " " + re.sub(r"[^a-z0-9]+", " ", message.lower()) + " "
    out = []
    for c in df.columns:
        w = re.sub(r"[^a-z0-9]+", " ", _words(str(c))).strip()
        forms = [w]
        if w and " " not in w and len(w) >= 5:
            # one-word columns also in their -ed form: "rated" for rating, "reviewed" for review
            forms += [re.sub(r"ing$", "ed", w), w + "ed", w + "d"]
        # the name exactly as written ("by depotCode", "per weekOf"), not only split into words
        forms.append(re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip())
        # a code column of a few repeated values names its entity: "by depot" for depotCode, "district" for DistrictCd
        parts = w.split()
        if len(parts) >= 2 and parts[-1] in ("code", "cd", "id") and df[c].dtype == object \
                and 1 < df[c].nunique() <= 200 and df[c].nunique() < len(df):
            forms.append(" ".join(parts[:-1]))
        if w and any(re.search(rf" {re.escape(f)}s? ", low) for f in dict.fromkeys(forms)):
            out.append(c)
    return out


_ASSOC = re.compile(r"\b(?:factors?|drivers?|variables?|columns?|features?|things?|what)\b[^?]{0,40}?\b(?:associated|"
                    r"related|linked|correlated|connected|tied)\s+(?:with|to)\s+(.+)|\b(?:what|which)\s+(?:\w+\s+){0,2}?(?:best\s+)?"
                    r"(?:affects?|influences?|drives?|explains?|predicts?|determines?|impacts?)\s+(.+)|"
                    r"\b(?:drivers|predictors|determinants|causes) of\s+(.+)", re.I)


def _ask_associations(message, df, memory, model_on, started) -> dict | None:
    """'Which factors are associated with progression?': measured directly (correlation, eta squared, Cramer's V),
    no model needed. When a model was trained in this conversation, its feature importance remains available."""
    from app.analyst import associations
    from app.analyst.data import find_column
    m = _ASSOC.search(message)
    if not m or re.search(r"feature importance|the model|trained", message, re.I):
        return None
    tail = next(g for g in m.groups() if g)
    cols = find_column(df, tail)
    if not cols:
        from app.analyst import semantics
        words = re.findall(r"[a-z]+", tail.lower())
        words = sorted(words, key=lambda w: not semantics.outcome_concept(w))   # outcome words (quitting) first
        offers = [c for w in words for c in semantics.concept_candidates(w, df)]
        offers = list(dict.fromkeys(offers))
        if offers:
            return _reply(f"This data has nothing called *{tail.strip(' ?.')}*. Do you mean " + " or ".join(f"`{c}`" for c in offers[:3])
                          + "? Options: " + ", ".join(f"`{c}`" for c in offers[:3]), "tool", model_on, started,
                          plan={"type": "clarify", "options": offers[:3]},
                          notes=["A word in the question matches a column by meaning, not by name; Silt asks before "
                                 "using it."])
        return None
    # the outcome is the head of the phrase: in "machine failure" it is failure, not machine
    cols = followup.mentioned_order(cols, tail)[-1:] + [c for c in cols if c != followup.mentioned_order(cols, tail)[-1]]
    sc = memory.get("_scope")
    if sc and cols[0] in (sc.get("entities") or {}):
        memory["_scope"] = {**sc, "entities": {k: v for k, v in sc["entities"].items() if k != cols[0]}}
    r = associations.compute(df, cols[0])
    chart = {"type": "bar", "x": "column", "y": "strength", "additive": False,
             "title": f"Association with {cols[0]} (strength)",
             "data": [{"column": x["factor"], "strength": x["strength"]} for x in r["factors"][:12]]}
    return _reply(associations.text(r), "tool", model_on, started, charts=[chart] if r["factors"] else [],
                  calls=[{"tool": "associations", "status": "ok", "result": _jsonable(r)}],
                  notes=["Measured directly by Silt (correlation / eta squared / Cramér's V with Holm-adjusted p-values); "
                         "no model was trained."])


def _ask_compare_values(message, df, model_on, started) -> dict | None:
    """'Compare life expectancy between Europe and Africa': two (or more) named values of one category, compared on a
    measure with the existing statistical comparison, restricted to those groups."""
    from app.analyst.data import find_column, kind_of
    if not re.search(r"\b(compare|comparison|versus|vs\.?|difference between|differ)\b", message, re.I):
        return None
    from app.analyst import intent
    hits = list(dict.fromkeys(calc.named_values(df, message) + intent.names_in_long_columns(message, df, [])[0]))
    by_col: dict[str, list[str]] = {}
    for c, v in hits:
        by_col.setdefault(c, []).append(v)
    pick = [(c, v) for c, v in by_col.items() if len(v) >= 2 and kind_of(df[c]) == "categorical"]
    if len(pick) != 1:
        return None
    col, vals = pick[0]
    low = message.lower()
    vals = sorted(vals, key=lambda v: low.find(v.lower()) if v.lower() in low else len(low))   # in the order asked
    nums = [c for c in find_column(df, message) if kind_of(df[c]) == "numeric" and c != col]
    if not nums:
        return None
    sub = df[df[col].astype(str).isin(vals)]
    if sub[col].astype(str).value_counts().reindex(vals).eq(1).all():
        return _compare_records(col, vals, nums, sub, model_on, started)
    texts, charts, calls = [], [], []
    for m in nums:                                   # every measure the question names, each compared on its own
        try:
            r = tools.run({"tool": "compare_groups", "args": {"outcome": m, "group": col}}, sub, {})
        except Exception:  # noqa: BLE001
            return None                              # one measure cannot be compared: leave the question to the others
        texts.append(explain.render("compare_groups", r))
        charts += present.charts("compare_groups", r)
        calls.append({"tool": "compare_groups", "status": "ok", "args": {"outcome": m, "group": col}, "result": _jsonable(r)})
    return _reply("\n\n".join(texts), "tool", model_on, started, charts=charts, calls=calls,
                  notes=[f"Compared only the {col.replace('_', ' ')} values you named: " + ", ".join(vals)
                         + f"; on {', '.join(nums)}."])


def _compare_records(col, vals, nums, sub, model_on, started) -> dict:
    """Each named value is one record ("Compare TCS and Accenture" on a table of companies): a group comparison has
    nothing to test, so the named measures are set side by side exactly as recorded, with the difference."""
    rows = {v: sub[sub[col].astype(str) == v].iloc[0] for v in vals}
    fmt = lambda x: "—" if pd.isna(x) else f"{x:,.2f}".rstrip("0").rstrip(".")
    head = "| " + col.replace("_", " ") + " | " + " | ".join(n.replace("_", " ") for n in nums) + " |"
    lines = [head, "|---|" + "---:|" * len(nums)]
    lines += ["| **" + v + "** | " + " | ".join(fmt(rows[v][n]) for n in nums) + " |" for v in vals]
    summary = []
    for n in nums:
        got = {v: rows[v][n] for v in vals if pd.notna(rows[v][n])}
        if len(got) < 2:
            summary.append(f"- **{n.replace('_', ' ')}**: not recorded for " + ", ".join(v for v in vals if v not in got) + ".")
            continue
        hi, lo = max(got, key=got.get), min(got, key=got.get)
        summary.append(f"- **{n.replace('_', ' ')}**: " + (f"equal ({fmt(got[hi])})." if got[hi] == got[lo] else
                       f"**{hi}** is higher ({fmt(got[hi])} vs {fmt(got[lo])} for {lo}, a difference of {fmt(got[hi] - got[lo])})."))
    answer = (f"**{' vs '.join(vals)}** — each is one record in this data, so their values are compared directly:\n\n"
              + "\n".join(lines) + "\n\n" + "\n".join(summary))
    result = {"column": col, "values": vals, "measures": nums,
              "records": {v: {n: (None if pd.isna(rows[v][n]) else float(rows[v][n])) for n in nums} for v in vals}}
    charts = [c for n in nums for c in present._bar(f"{n.replace('_', ' ')}: {' vs '.join(vals)}", col, n,  # noqa: SLF001
                                                       [{col: v, n: result["records"][v][n]} for v in vals
                                                        if result["records"][v][n] is not None])]
    return _reply(answer, "tool", model_on, started, charts=charts,
                  calls=[{"tool": "compare_records", "status": "ok", "result": result}],
                  notes=[f"Compared the {col.replace('_', ' ')} values you named, one record each: " + ", ".join(vals) + "."])


_DISTRIBUTION = re.compile(r"\b(distribution|breakdown|break-down|split|composition|mix|proportions?)\b", re.I)


def _ask_value_distribution(message, df, model_on, started) -> dict | None:
    """'Show the distribution of public and private companies': the question names two or more values of one category
    column (company_type: Public, Private) and no measure, so it asks how the records are spread over that column --
    a group-by count. The named values are shown with the rest of the column, so the counts add up to every record."""
    from app.analyst.data import find_column, kind_of
    if not _DISTRIBUTION.search(message):
        return None
    by_col: dict[str, list[str]] = {}
    for c, v in calc.named_values(df, message):
        by_col.setdefault(c, []).append(v)
    pick = [(c, v) for c, v in by_col.items() if len(v) >= 2 and kind_of(df[c]) == "categorical"]
    if len(pick) != 1 or any(kind_of(df[c]) == "numeric" for c in find_column(df, message)):
        return None
    col, vals = pick[0]
    counts = df[col].astype("string").value_counts(dropna=True)
    n, named_total = len(df), int(counts.reindex(vals).fillna(0).sum())
    other = counts.drop(labels=[v for v in vals if v in counts.index])
    missing = int(df[col].isna().sum())
    label = col.replace("_", " ")
    pc = lambda k, d: f"{100 * k / d:.1f}%" if d else "—"
    lines = [f"| {label} | records | share of all {n:,} records | share of {' + '.join(vals)} |", "|---|---:|---:|---:|"]
    lines += [f"| **{v}** | {int(counts.get(v, 0)):,} | {pc(int(counts.get(v, 0)), n)} | {pc(int(counts.get(v, 0)), named_total)} |"
              for v in vals]
    if len(other):
        lines.append(f"| other {label} values ({len(other)}) | {int(other.sum()):,} | {pc(int(other.sum()), n)} | — |")
    if missing:
        lines.append(f"| no {label} recorded | {missing:,} | {pc(missing, n)} | — |")
    top = max(vals, key=lambda v: int(counts.get(v, 0)))
    answer = (f"**Distribution of {label}: {', '.join(vals)}** — records counted by `{col}`.\n\n" + "\n".join(lines)
              + f"\n\n**{top}** is the larger group: {int(counts.get(top, 0)):,} of the {named_total:,} records that are "
              + " or ".join(vals) + f" ({pc(int(counts.get(top, 0)), named_total)}).")
    result = {"column": col, "counts": {v: int(counts.get(v, 0)) for v in vals},
              "other": int(other.sum()), "other_values": len(other), "missing": missing, "records": n}
    return _reply(answer, "tool", model_on, started,
                  charts=present._bar(f"Records by {label}", col, "records",  # noqa: SLF001
                                      [{col: v, "records": int(counts.get(v, 0))} for v in vals]),
                  calls=[{"tool": "count_by", "status": "ok", "args": {"group": col, "values": vals}, "result": result}],
                  notes=[f"Grouped by {col} and counted the records in each value; the values you named are "
                         + ", ".join(vals) + "."])


# ------------------------------------------------- plain look-ups: values of a column, one record, extremes, range --
_LOOKUP_STOP = {"no", "of", "the", "num", "number", "nr", "id"}
_LIST_Q = re.compile(r"\b(different|distinct|unique|types? of|kinds? of|categories of|list|values of|what are the|"
                     r"which are the|show (?:me )?(?:all )?the|all the)\b", re.I)
_MODE_Q = re.compile(r"\b(?:most|least) (common|frequent|popular|usual)\b", re.I)
_COUNT_BY_Q = re.compile(r"\b(count|number of|how many)\b.*\b(by|per|for each|in each)\b", re.I)
_DETAIL_Q = re.compile(r"\b(details?|info(?:rmation)?|profile|record|everything|tell me about|show)\b", re.I)
_AGG_Q = re.compile(r"\b(average|mean|median|total|sum|count|how many|number of|compare|versus|vs|trend|forecast|"
                    r"predict|correlat\w*|distribution|highest|lowest|top|bottom)\b", re.I)
_RANGE_Q = re.compile(r"\b(range of|min(?:imum)? and max(?:imum)?|lowest and highest|smallest and largest|spread of)\b", re.I)
# adjectives that name a direction on a column whose name has the word: "oldest company" -> highest *age*
_ADJ = {"oldest": ("age", False), "youngest": ("age", True), "newest": ("age", True),
        "cheapest": ("price", True), "priciest": ("price", False), "costliest": ("cost", False),
        "longest": ("length", False), "shortest": ("length", True), "heaviest": ("weight", False),
        "lightest": ("weight", True), "tallest": ("height", False)}


_COMPARATIVE = {"older": ("age", "above"), "younger": ("age", "below"), "newer": ("age", "below"),
                "cheaper": ("price", "below"), "pricier": ("price", "above"), "costlier": ("cost", "above"),
                "longer": ("length", "above"), "shorter": ("length", "below"), "heavier": ("weight", "above"),
                "lighter": ("weight", "below"), "taller": ("height", "above")}


def _comparatives(message: str, df) -> str:
    """'older than 30 years' says a threshold on the column whose name holds the adjective's word (Company_Age,
    patient_age): written as that threshold ("Company_Age above 30 years") so every reader applies it. Left as it is
    when no column -- or more than one -- has the word: then the threshold is not guessed."""
    from app.analyst.data import kind_of

    def sub(m):
        word, cmp_ = _COMPARATIVE[m.group(1).lower()]
        cols = [c for c in df.columns if kind_of(df[c]) == "numeric" and word in _col_words(c)]
        return f"{cols[0]} {cmp_} " if len(cols) == 1 else m.group(0)
    return re.sub(rf"\b({'|'.join(_COMPARATIVE)})\s+than\s+", sub, message, flags=re.I)


_MONEY_TOKENS = {"charge", "cost", "bill", "price", "amount", "fee", "payment", "revenue", "spend", "expense", "salary",
                 "sale", "income", "pay", "fare", "premium", "wage"}
# verb forms only ("department charges the most", "which store spent more"): "cost per lead" / "charge amount" are nouns
_MONEY_VERB = re.compile(r"\b(charges|charged|costs|spends|spent|pays|paid|bills|billed|earns|earned|makes|made)\s+"
                         r"(?:the\s+|their\s+|its\s+)?(most|more|least|less|highest|lowest|patients|customers|clients)\b", re.I)


def _money_verb_gate(message, df, model_on, started) -> dict | None:
    """'Which department charges the most?' says money without naming which money column: with a room charge, a
    medication cost and a patient bill, picking the one whose name shares the verb (room_charge) would answer a
    different question. When the data has two or more money columns and the question names none of them in full,
    Silt asks which one to use."""
    from app.analyst.data import kind_of
    if df is None or not _MONEY_VERB.search(message):
        return None
    money = [c for c in df.columns if kind_of(df[c]) == "numeric" and _col_words(c) & _MONEY_TOKENS]
    if len(money) < 2:
        return None
    low = message.lower()
    if any(re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", low)
           for c in money for f in {str(c).lower(), str(c).lower().replace("_", " ")}):
        return None                                   # a money column is named in full: no ambiguity
    vm = _MONEY_VERB.search(message)
    verb = vm.group(1)
    example = (message[:vm.start()] + f"has the highest average {money[0]}"
               + re.sub(r"^\s*(?:on average|on)\b", "", message[vm.end():])).strip()
    return _reply(f"“{verb}” could mean any of these money columns: " + ", ".join(f"`{c}`" for c in money)
                  + f". Which one should I use? For example: *{example}*",
                  "tool", model_on, started, plan={"type": "clarify", "options": money},
                  notes=["The question names money by a verb, not a column, and the data holds several money columns; "
                         "Silt asks rather than choosing one."])


def _col_words(c) -> set:
    from app.analyst.deliverables import _stem
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(c))
    return {_stem(t) for t in re.split(r"[^A-Za-z0-9]+", raw.lower()) if t and t not in _LOOKUP_STOP}


def _cols_named(message: str, df) -> list:
    """Columns the message names by their words, in any order or plural ("types of companies" -> company_type,
    "headquarters" -> Head_Quarters, "No_of_Employee" -> No_of_Employee); most words matched first."""
    from app.analyst.deliverables import _stem
    words = {_stem(w) for w in re.findall(r"[a-z0-9]+", message.lower())}
    hits = []
    for c in df.columns:
        toks = _col_words(c)
        compact = _stem(re.sub(r"[^a-z0-9]+", "", str(c).lower()))
        if (toks and toks <= words) or (len(compact) > 4 and compact in words):
            hits.append((-len(toks), str(c)))
    return [c for _, c in sorted(hits)]


def _label_col(df):
    """The column that names each record (company / product / patient name): the text column with the most values."""
    from app.analyst.data import kind_of
    texts = [c for c in df.columns if kind_of(df[c]) == "categorical"]
    named = [c for c in texts if "name" in _col_words(c)]
    return max(named or texts, key=lambda c: df[c].nunique()) if texts else None


def _value_table(df, col, limit=25) -> tuple[str, dict]:
    vc = df[col].astype("string").value_counts(dropna=True)
    miss = int(df[col].isna().sum())
    rows = [f"| {html_cell(v)} | {int(k):,} | {100 * k / len(df):.1f}% |" for v, k in vc.head(limit).items()]
    text = (f"| {col} | records | share |\n|---|---:|---:|\n" + "\n".join(rows)
            + (f"\n\n…and **{len(vc) - limit:,}** more values (the {limit} most frequent are shown)." if len(vc) > limit else "")
            + (f"\n\n{miss:,} record(s) have no {col}." if miss else ""))
    return text, {"column": col, "distinct": int(len(vc)), "missing": miss,
                  "top": {str(v): int(k) for v, k in vc.head(limit).items()}}


def html_cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def _ask_lookup(message, df, model_on, started, sc=None) -> dict | None:
    """Plain questions about what is IN the data, answered straight from the records: one record ("headquarters of
    TCS", "details of Infosys"), the values of a column ("what are the different types of companies", "most common
    company type", "count companies by company type"), an extreme named by an adjective ("oldest company") and a
    range ("range of ratings"). Every figure is read or counted from the table; nothing is estimated."""
    from app.analyst import intent
    from app.analyst.data import kind_of
    named_cols = _cols_named(message, df)
    cats = [c for c in named_cols if kind_of(df[c]) == "categorical"]
    nums = [c for c in named_cols if kind_of(df[c]) == "numeric"]
    low = message.lower()

    # 0. the table itself: its columns, its first / last rows
    if re.search(r"\bcolumn names?\b|\b(?:list|what are|show|which are|name)(?: me)? (?:the |all )?(?:the )?(?:columns|fields|variables)\b|"
                 r"\bwhat columns\b", low):
        tbl = "| column | type | distinct values | missing |\n|---|---|---:|---:|\n" + "\n".join(
            f"| {c} | {kind_of(df[c])} | {df[c].nunique():,} | {int(df[c].isna().sum()):,} |" for c in df.columns)
        return _reply(f"This dataset has **{len(df.columns)} columns** and {len(df):,} records:\n\n" + tbl, "tool", model_on,
                      started, calls=[{"tool": "schema", "status": "ok", "result": {"columns": [str(c) for c in df.columns]}}])
    hm = re.search(r"\b(first|top|last|bottom)\s+(\d{1,3})\s+(rows|records|entries|lines)\b|\b(preview|sample rows|head of)\b", low)
    if hm:
        k = int(hm.group(2)) if hm.group(2) else 5
        part = df.tail(k) if hm.group(1) in ("last", "bottom") else df.head(k)
        cols = list(df.columns)
        tbl = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
            "| " + " | ".join(html_cell("—" if pd.isna(r[c]) else r[c]) for c in cols) + " |" for _, r in part.iterrows())
        which = "last" if hm.group(1) in ("last", "bottom") else "first"
        return _reply(f"The {which} {len(part)} of {len(df):,} records, as stored:\n\n" + tbl, "tool", model_on, started,
                      calls=[{"tool": "rows", "status": "ok", "args": {"which": which, "n": k}, "result": {"rows": int(len(part))}}])

    # percentiles / quartiles of a named measure
    pm = re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+percentile", low)
    if (pm or re.search(r"\bquartiles?\b|\biqr\b|\binterquartile\b", low)) and nums:
        m = nums[0]
        v = df[m].dropna()
        qs = [int(x) for x in pm] or [25, 50, 75]
        vals = {q: float(np.percentile(v, q)) for q in qs}
        lines = "\n".join(f"| {q}th percentile | {vals[q]:,.4g} |" for q in qs)
        extra = (f"\n\nInterquartile range (75th − 25th): **{vals[75] - vals[25]:,.4g}**." if {25, 75} <= set(vals) else "")
        return _reply(f"**{m}** over {len(v):,} records:\n\n| percentile | value |\n|---|---:|\n" + lines + extra,
                      "tool", model_on, started,
                      calls=[{"tool": "percentiles", "status": "ok", "args": {"column": m, "q": qs}, "result": {str(k): x for k, x in vals.items()}}])

    # records meeting stated conditions ("which companies are older than 30 years and have a rating above 4?")
    where = (sc or {}).get("where") or []
    if where and re.match(r"\s*(which|list|show|find|get|what|give)\b", low) and not _AGG_Q.search(message):
        keep = scoping.mask(df, sc)
        hit = df[keep]
        lc = _label_col(df)
        cond_cols = list(dict.fromkeys(w["column"] for w in where if w.get("column") in df.columns))
        cols = ([lc] if lc and lc not in cond_cols else []) + cond_cols
        top = hit.sort_values(cond_cols[0], ascending=False) if cond_cols else hit
        tbl = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
            "| " + " | ".join(html_cell("—" if pd.isna(r[c]) else (f"{r[c]:,.4g}" if isinstance(r[c], float) else r[c]))
                              for c in cols) + " |" for _, r in top.head(25).iterrows())
        cond = scoping.describe(sc)
        head = f"**{len(hit):,}** of {len(df):,} records meet **{cond}**" + (
            f"; the first 25 (by {cond_cols[0]}, highest first) are shown:" if len(hit) > 25 else ":")
        return _reply(head + "\n\n" + (tbl if len(hit) else "_No record meets every condition._"), "tool", model_on, started,
                      calls=[{"tool": "filter_records", "status": "ok", "args": {"where": [w.get("text") for w in where]},
                              "result": {"matching": int(len(hit)), "records": int(len(df))}}])

    # 1. one record named by its value in a long text column
    ents = intent.names_in_long_columns(message, df, [])[0]
    if len(ents) == 1 and not _AGG_Q.search(message) and (_DETAIL_Q.search(message) or named_cols):
        ecol, val = ents[0]
        rows = df[df[ecol].astype(str).str.strip() == val]
        show = [c for c in named_cols if c != ecol] or [c for c in df.columns]
        if len(rows):
            cols = [ecol] + [c for c in show if c != ecol]
            tbl = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
                "| " + " | ".join(html_cell("—" if pd.isna(r[c]) else r[c]) for c in cols) + " |" for _, r in rows.head(5).iterrows())
            head = (f"**{val}** — {', '.join(f'`{c}`' for c in cols[1:])}:" if len(cols) <= 4 else f"**{val}** — its record:")
            return _reply(head + "\n\n" + tbl + (f"\n\n{len(rows)} records are named {val}; the first 5 are shown." if len(rows) > 5 else ""),
                          "tool", model_on, started,
                          calls=[{"tool": "lookup", "status": "ok", "args": {"column": ecol, "value": val, "columns": cols[1:]},
                                  "result": {"rows": int(len(rows))}}],
                          notes=[f"Read from the record(s) where {ecol} = {val}."])

    # 2. min / max / range of a named measure
    if _RANGE_Q.search(message) and len(nums) >= 1:
        m = nums[0]
        v = df[m].dropna()
        if len(v):
            lc = _label_col(df)
            lo_r, hi_r = df.loc[v.idxmin()], df.loc[v.idxmax()]
            who = (lambda r: f" ({r[lc]})" if lc and lc != m and pd.notna(r[lc]) else "")
            ans = (f"**{m}** ranges from **{v.min():,.4g}**{who(lo_r)} to **{v.max():,.4g}**{who(hi_r)} — a range of "
                   f"**{v.max() - v.min():,.4g}** over {len(v):,} records (median {v.median():,.4g}).")
            return _reply(ans, "tool", model_on, started,
                          calls=[{"tool": "range", "status": "ok", "args": {"column": m},
                                  "result": {"min": float(v.min()), "max": float(v.max()), "n": int(len(v))}}])

    # 3. an extreme named by an adjective ("oldest company")
    adj = next((a for a in _ADJ if re.search(rf"\b{a}\b", low)), None)
    if adj:
        word, asc = _ADJ[adj]
        mcol = next((c for c in df.columns if kind_of(df[c]) == "numeric" and word in _col_words(c)), None)
        lc = _label_col(df)
        if mcol and lc:
            top = df.dropna(subset=[mcol]).sort_values(mcol, ascending=asc).head(5)
            best = top.iloc[0]
            tbl = f"| {lc} | {mcol} |\n|---|---:|\n" + "\n".join(f"| {html_cell(r[lc])} | {r[mcol]:,.4g} |" for _, r in top.iterrows())
            return _reply(f"The {adj} by **{mcol}** is **{best[lc]}** ({mcol} = {best[mcol]:,.4g}).\n\n" + tbl,
                          "tool", model_on, started,
                          calls=[{"tool": "extreme", "status": "ok", "args": {"column": mcol, "ascending": asc, "label": lc},
                                  "result": {"value": str(best[lc]), mcol: float(best[mcol])}}],
                          notes=[f"“{adj}” read as the {'lowest' if asc else 'highest'} {mcol}."])

    # 4. the values of one category column: list, most common, count by
    count_only = re.search(r"\b(how many|number of|count)\b", low) and not _COUNT_BY_Q.search(message)
    ranking = re.search(r"\b(top|bottom|best|worst|highest|lowest|largest|smallest|biggest|rank\w*)\b", low)
    pie = re.search(r"\b(pie|donut|doughnut)\b", low)
    chart_of = re.search(r"\b(pie|donut|doughnut|bar)\b(?:\s+chart|\s+graph)?|\b(chart|graph|plot)\b", low)
    if "column" in low and not chart_of:
        return None                                  # "column names" is the schema (handled above), never a column's values
    if len(cats) >= 1 and not nums and not count_only and not ranking and (_LIST_Q.search(message) or _MODE_Q.search(message)
                                                             or _COUNT_BY_Q.search(message) or chart_of):
        col = cats[0]
        text, res = _value_table(df, col)
        mode = _MODE_Q.search(message)
        if mode and res["top"]:
            least = "least" in mode.group(0).lower()
            vc = df[col].astype("string").value_counts(dropna=True)
            v, k = (vc.index[-1], vc.iloc[-1]) if least else (vc.index[0], vc.iloc[0])
            head = (f"The {'least' if least else 'most'} common **{col}** is **{v}** — {int(k):,} of {len(df):,} records "
                    f"({100 * k / len(df):.1f}%).\n\n")
        else:
            head = f"**{col}** has **{res['distinct']:,}** different values in {len(df):,} records:\n\n"
        charts = present._bar(f"Records by {col}", col, "records",  # noqa: SLF001
                              [{col: k, "records": v} for k, v in list(res["top"].items())[:15]])
        if pie and charts:
            charts[0]["type"] = "pie"
        return _reply(head + text, "tool", model_on, started, charts=charts,
                      calls=[{"tool": "value_counts", "status": "ok", "args": {"column": col}, "result": res}])
    return None


_WHICH_TOP = re.compile(r"\bwhich\s+([a-z]+)\s+(?:has|had|have|is|was|shows?|got|gets)\s+(?:the\s+)?(highest|lowest|"
                        r"largest|smallest|biggest|best|worst|most|least|greatest)\b", re.I)


def _ask_compare_columns(message, df, model_on, started) -> dict | None:
    """Wide data -- one column per stock, product or site -- asked 'which stock had the highest average price?':
    the noun names no column, so the numeric columns themselves are the things compared. Said explicitly."""
    from app.analyst.data import find_column, kind_of
    from app.analyst.scope import year_column
    m = _WHICH_TOP.search(message)
    if not m:
        return None
    noun = m.group(1).lower()
    if metrics.group_column(message, df)[0] or find_column(df, noun) or metrics.recognise(message):
        return None
    yc = year_column(df)
    cols = [c for c in df.columns if kind_of(df[c]) == "numeric" and c != yc and not re.search(r"(^|_)id$", str(c), re.I)
            and df[c].nunique() > 2]
    if len(cols) < 2 or [c for c in find_column(df, message) if c in cols]:
        return None
    agg = "sum" if re.search(r"\b(total|sum|overall)\b", message, re.I) else \
        "median" if re.search(r"median", message, re.I) else "max" if re.search(r"\b(peak|maximum|max)\b", message, re.I) else "mean"
    vals = {c: float(getattr(pd.to_numeric(df[c], errors="coerce"), agg)()) for c in cols}
    low = m.group(2).lower() in ("lowest", "smallest", "worst", "least")
    order = sorted(vals.items(), key=lambda kv: kv[1], reverse=not low)
    word = {"mean": "average", "sum": "total", "median": "median", "max": "maximum"}[agg]
    from app.analyst.explain import f
    from app.analyst.explain import tied_extreme
    bv, bks = tied_extreme(vals, low=low)
    ov, oks = tied_extreme(vals, low=not low)
    who = " and ".join(f"**{k}**" for k in bks)
    lines = [f"{who} {'have' if len(bks) > 1 else 'has'} the {'lowest' if low else 'highest'} {word} value: "
             f"**{f(bv)}**{' each — tied' if len(bks) > 1 else ''} ({' and '.join(oks)} "
             f"{'have' if len(oks) > 1 else 'has'} the {'highest' if low else 'lowest'}: {f(ov)}).", "",
             f"| {noun} (column) | {word} | records |", "|---|---:|---:|"]
    lines += [f"| {c} | {f(v)} | {int(pd.to_numeric(df[c], errors='coerce').notna().sum())} |" for c, v in order]
    lines += ["", f"_No column names the {noun}s, so each numeric column was treated as one {noun} ("
              + ", ".join(cols[:8]) + ("…" if len(cols) > 8 else "") + "). They are compared as recorded — check they "
              "are in the same units._"]
    chart = {"type": "bar", "x": noun, "y": word, "additive": agg == "sum",
             "title": f"{word.capitalize()} by {noun}", "data": [{noun: c, word: v} for c, v in order]}
    return _reply("\n".join(lines), "tool", model_on, started, charts=[chart],
                  calls=[{"tool": "compare_columns", "status": "ok", "result": {"aggregation": agg, "values": vals}}],
                  notes=[f"Compared the numeric columns as {noun}s ({word} of each)."])


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
    from app.analyst.data import file_lines
    line_of = file_lines(df)
    for i in df.index[keep]:
        v = vals.get(i)
        if v is None or pd.isna(v):
            continue
        key = str(pd.Timestamp(df.at[i, dates[0]]).date()) if dates and pd.notna(df.at[i, dates[0]]) else f"line {line_of[i]}"
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
    if prev and _superseded(memory, "last_charts"):
        prev = None                     # the latest answer is newer than these charts: never redraw a stale result
    if not prev or not re.search(r"\b(that|it|this|these|those|them|same|the (result|answer|previous|difference|"
                                 r"differences|comparison|gap|change|changes))\b", message, re.I):
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
        from app.analyst.semantics import measure_options
        nums = measure_options(df)
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


def _window_owned_by_scope(calls: list[dict], sc: dict | None, notes: list[str]) -> None:
    """The question names a period ("in February 2025", "the second half of 2024"): Silt's scope reader has already
    limited the records to it, from the question's own words. A date filter the model adds on that same column can
    only repeat that window or contradict it (a wrong day, a wrong half), so it is removed and the removal is said."""
    tcol = ((sc or {}).get("time") or {}).get("column") if scoping.active(sc) else None
    if not tcol:
        return
    dropped = []
    for c in calls:
        fl = (c.get("args") or {}).get("filters")
        if not fl:
            continue
        keep = [f for f in fl if f.get("column") != tcol]
        if len(keep) != len(fl):
            dropped += [f"{f['column']} {f['op']} {f.get('value')}" for f in fl if f.get("column") == tcol]
            c["args"]["filters"] = keep
    if dropped:
        notes.append("The period comes from the question (" + scoping.describe(sc) + "); the model's own date filter ("
                     + "; ".join(dict.fromkeys(dropped)) + ") was not used.")


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
        validated, errors, refused = [], [], []
        for c in plan["calls"][:4]:
            try:
                validated.append(tools.validate_call(c, df, message))
            except tools.RoleRefusal as e:
                refused.append(str(e))
            except tools.InvalidCall as e:
                errors.append(str(e))
        if source in ("model", "model_retry"):
            _window_owned_by_scope(validated, memory.get("_scope"), notes)
        if refused and not any(c["tool"] in ("compute_metric", "describe_column") for c in validated):
            # the aggregation asked for has no meaning for what the column is: said, never computed with a warning
            return _reply("**I can't compute that.** " + " ".join(dict.fromkeys(refused)), "tool", model_on, started,
                          plan={"type": "decline", "role_refusal": True},
                          notes=notes + ["Refused by Silt's column-role rule (one rule for every tool and program "
                                         "step); nothing was calculated."])
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
                    except tools.RoleRefusal as e:
                        return _reply("**I can't compute that.** " + str(e), "tool", model_on, started,
                                      plan={"type": "decline", "role_refusal": True},
                                      notes=notes + ["Refused by Silt's column-role rule; nothing was calculated."])
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
                    if call["tool"] == "compute_metric" and (call.get("args") or {}).get("group_by"):
                        sc_ = memory.get("_scope") or {}
                        memory["last_breakdown"] = {"kind": "metric", "args": dict(call["args"]),
                                                    "entities": dict(sc_.get("entities") or {})}
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
        if source in ("model", "model_retry") and opts:
            # the model's options name columns as it wrote them: resolve to the dataset's own names; when it asks the
            # user to choose a column, never offer one that does not exist
            opts, dropped = plan_checks.canonical_options(opts, df, plan.get("question", ""))
            plan = {**plan, "options": opts}
            if dropped:
                notes.append("Options the model offered that are not columns of this dataset were left out: "
                             + ", ".join(f"`{o}`" for o in dropped[:5]) + ".")
        answer = plan.get("question", "Could you clarify?") + ("\n\nOptions: " + ", ".join(f"`{o}`" for o in opts) if opts else "")
    elif plan["type"] == "decline":
        reason, head = plan.get("reason", ""), "**The available data is not sufficient for this.** "
        if re.search(r"advertis|campaign|marketing|ad spend|budget", reason, re.I) and \
                not re.search(r"advertis|campaign|marketing|ad spend|budget|promot", message, re.I):
            # a reason about a different request (the model's advertising template): never shown as the explanation
            reason = ("This request is not an analysis of the records in this file — Silt computes totals, averages, "
                      "comparisons, trends and forecasts from the data; it does not take actions, make decisions "
                      "about individual people, or answer from outside knowledge.")
            head = "**I can't do that with this data.** "
            notes.append("The model's decline reason referred to something the request did not mention; a plain "
                         "statement of what Silt can and cannot do was shown instead.")
        answer = ("**I can't compute that.** " if plan.get("role_refusal") else head) + reason
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
            text, fixed = _repair_rounding(text, evidence, message, notes)
            bad = explain.ungrounded(text, evidence, message)
            grounding = {"checked": True, "ungrounded": bad, "rounding_repaired": fixed}
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


def _repair_rounding(text: str, evidence, message: str, notes: list[str]) -> tuple[str, list]:
    """explain.repair_roundings, with a note when a last-digit rounding slip was corrected (the grounding check
    that follows is unchanged and still rejects any number that is not a rounding of a result)."""
    text, fixed = explain.repair_roundings(text, evidence, message)
    if fixed:
        notes.append("Rounding in the model's summary was corrected to match the results: "
                     + ", ".join(f"{a} -> {b}" for a, b in fixed[:5]) + ".")
    return text, fixed


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
        text, fixed = _repair_rounding(text, evidence, message, notes)
        bad = explain.ungrounded(text, evidence, message)
        if bad:
            notes.append(f"The model's summary cited numbers not in the results ({', '.join(bad[:5])}); it was not shown.")
            return None, {"checked": True, "ungrounded": bad}
        return (text or None), {"checked": True, "ungrounded": [], "rounding_repaired": fixed}
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
        except tools.RoleRefusal:
            raise                       # one wording for every path: handled in ask()
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
                    text = _drop_boilerplate(text, [], notes)
                    answer = f"{text}\n\n{DOC_SUMMARY_LABEL}\n\n{answer}"
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
    # a list of several results ("total revenue, cost, profit and profit margin by region") before a single
    # catalogue metric named inside it, which would answer one item of the list
    p = metrics.multi_plan(message, df) or metrics.plan(message, df) or metrics.count_plan(message, df) \
        or metrics.formula_plan(message, df) or metrics.threshold_plan(message, df) or metrics.share_plan(message, df) \
        or metrics.event_rate_plan(message, df) or metrics.record_count_plan(message, df)
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
    g = spec.get("group_by")
    if g and sc and g in (sc.get("entities") or {}) and {str(x).lower() for x in df[g].dropna().astype(str).unique()} \
            <= {str(v).lower() for v in sc["entities"][g]}:
        # "status distribution (Completed, Shipped, Cancelled or Returned)": every value of the breakdown column is
        # listed -- the groups to show, not a filter
        sc = {**sc, "entities": {k: v for k, v in sc["entities"].items() if k != g}}
        memory["_scope"] = sc
    if p.event_col and sc and p.event_col in (sc.get("entities") or {}):
        # "share of visits at Clinic B": Clinic B is what is counted, not a filter on the records
        sc = {**sc, "entities": {k: v for k, v in sc["entities"].items() if k != p.event_col}}
        memory["_scope"] = sc
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
    out = _calc_reply(message, spec, result, base, notes, memory, model_on, started,
                      {"primary": None if p.name == "summary" else p.name, "percent": p.percent,
                       "metric_label": p.label, "definition": p.definition})
    if p.name == "summary":
        # revenue, cost and profit all totalled: say whether the stored profit is revenue - cost in these records
        from app.analyst import reconcile
        cols = reconcile.columns(df)
        summed = {re.sub(r"^sum\((.*)\)$", r"\1", e) for e in spec["aggregate"].values() if e.startswith("sum(")}
        if cols and set(cols) <= summed:
            keep = scoping.mask(df, sc) if scoping.active(sc) else pd.Series(True, index=df.index)
            rec = reconcile.check(df[keep])
            if rec is not None:
                out["answer"] += "\n\n" + reconcile.text(rec)
                out["calls"].append({"tool": "reconcile_profit", "status": "ok", "args": {}, "result": _jsonable(rec)})
    return out


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


_METRIC_FN = {"sum": "sum", "avg": "mean", "mean": "mean", "median": "median", "min": "min", "max": "max", "count": "count"}


def _metric_as_last_calc(out: dict, message: str, df, memory: dict) -> None:
    """A verified single aggregate ("total revenue by department") is also the calculation that short follow-ups such
    as "Only Rahul." or "Exclude duplicates." refine: the same aggregate, as a calculator spec, re-run by Silt."""
    ok = [c for c in out.get("calls", []) if c.get("status") == "ok"]
    if df is None or out.get("verify_status") != "verified" or len(ok) != 1 or ok[0].get("tool") != "compute_metric":
        return
    a = ok[0].get("args") or {}
    fn, col, group = _METRIC_FN.get(str(a.get("aggregation"))), a.get("column"), a.get("group_by")
    if not fn or a.get("filters") or not isinstance(col, str) or not col.isidentifier() or col not in df.columns or \
            (group is not None and (not isinstance(group, str) or not group.isidentifier() or group not in df.columns)):
        return
    spec = {"type": "calc", "derive": {}, "exclude": [], "group_by": group,
            "aggregate": {col if fn == "sum" else f"{'average' if fn == 'mean' else fn}_{col}": f"{fn}({col})"},
            "sort": None, "explain": ""}
    try:
        result = calc.run(spec, df, question=message)
    except Exception:  # noqa: BLE001 - not refinable; the answer itself is unaffected
        return
    memory["last_calc"] = {"question": message, "spec": spec, "result": result,
                           "meta": {k: result.get(k) for k in ("primary", "percent", "metric_label", "definition")}}
    memory["last_analysis"] = "calc"


def _refine_last(message: str, df: pd.DataFrame, memory: dict, model_on: bool, started: float,
                 sc: dict | None) -> dict | None:
    """A short follow-up that changes the previous calculation -- "Only Rahul.", "Exclude duplicates.", "By
    department.", "Treat them as overnight shifts." -- re-runs that same calculation with the change, deterministically.
    Only when the latest analysis was a calculation, and the message asks for nothing else."""
    last = memory.get("last_calc")
    if not last or _superseded(memory, "last_calc") or memory.get("last_analysis") != "calc" or len(message.split()) > 12 \
            or metrics.recognise(message) \
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
        text, fixed = _repair_rounding(text, evidence, message, notes)
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
