"""Conversational follow-ups, recommendations, trends and forecasts of a computed measure, and charts.

Everything here is computed from the data (or from the previous computed answer); the local model only
writes the prose, and that prose is shown only when every number in it appears in the evidence. When the
model is offline or its prose fails the check, a deterministic answer built from the same evidence is shown,
so a question always gets a real answer rather than a promise of one.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from app.analyst import calc, explain, planner
from app.analyst import scope as scoping
from app.analyst.data import kind_of

# ------------------------------------------------------------------ intents --
_EXPLAIN_PREV = re.compile(
    r"\b(explain (this|that|it|the (result|answer|above|previous|last))|in (simple|plain|simpler|layman'?s?) "
    r"(terms|english|words|language)|simplif\w*|eli5|break (it|this|that) down|what does (this|that|it|the result|"
    r"the answer|(this|that|the) (result|answer|number|figure|chart|table)s?) (mean|show|tell)|what do (these|those) (numbers|results) mean|so what|what('s| is) the (takeaway|"
    r"bottom line)|summari[sz]e (this|that|it|the (result|answer))|tl;?dr|i don'?t understand|what does that mean|"
    r"meaning of (this|that|the result)|explain (again|more)|interpret (this|that|it|the result))\b", re.I)
_ADVICE = re.compile(
    r"\b(what should (the company|we|i|management|they|the business|the team) (do|prioriti[sz]e|focus on)|"
    r"(what|where) (should|do) (we|i|management|the company) (prioriti[sz]e|focus)|priorities|what (do|would|should) (you|we|i) "
    r"recommend\b(?! (for|in|to) )|(any|your|some|give( me)?) recommendations?|recommendations?\W*$|suggest\w* (some |any )?"
    r"(actions?|steps|improvements?)|next steps?|how (can|could|do|should) (we|i|the company|they) (improve|reduce|"
    r"increase|fix|cut|lower|boost|optimi[sz]e)|improve (the )?(performance|results|efficiency|costs?)|action items?|"
    r"what actions?|advice|advise)\b", re.I)
_WHY_BAD = re.compile(r"\bwhy\b.*\b(badly|bad|poor\w*|worst|struggl\w*|underperform\w*|lowest|lagging|behind|weak\w*)\b",
                      re.I)
_DATASET_WORDS = re.compile(r"\b(this|the|my) (data|dataset|file|table|spreadsheet)\b", re.I)


def intent(message: str, memory: dict) -> str | None:
    """'explain_previous', 'advice', 'why_group' or None. Only a conversational question about an earlier
    answer, or a request for recommendations, is caught here; anything computable goes to the tools."""
    low = message.lower().strip()
    has_prev = bool(memory.get("last_answer"))
    if _ADVICE.search(low) and not low.startswith("why"):
        return "advice"
    if _WHY_BAD.search(low) and (memory.get("last_calc") or memory.get("last_focus")):
        return "why_group"
    if _EXPLAIN_PREV.search(low) and has_prev and not _DATASET_WORDS.search(low):
        return "explain_previous"
    return None


# --------------------------------------------------------------- evidence --
def _short(obj, limit=5000) -> str:
    return json.dumps(obj, default=str)[:limit]


def remember(memory: dict, question: str, out: dict, evidence) -> None:
    """What the next conversational question can refer to: the question, the deterministic answer text and the
    evidence (computed numbers) behind it."""
    if out.get("plan_source") in ("conversation", "conversation_rules") and memory.get("last_answer"):
        return                      # "explain it again" keeps referring to the analysis, not to the explanation
    memory["last_answer"] = {"question": question, "answer": out.get("answer", "")[:4000], "evidence": evidence,
                             "source": out.get("plan_source")}


def _plain(md: str, sentences: int = 4) -> str:
    """The first non-table sentences of a markdown answer, without markup."""
    keep = []
    for line in md.splitlines():
        line = line.strip()
        if not line or line.startswith(("|", "#", "_Summary", "---", "```")):
            continue
        keep.append(re.sub(r"[*_`]", "", line))
    text = " ".join(keep)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9⚠])", text)
    return " ".join(parts[:sentences]).strip()


def simple_from_calc(res: dict, question: str) -> str:
    """A plain-language explanation of a calculator result, deterministic."""
    from app.analyst.present import _cellv, _label, pct
    out = []
    aggs = list(res.get("aggregates") or {})
    key = next((n for n, e in res["aggregates"].items() if e.startswith("sum(")), aggs[0] if aggs else None)
    groups = [g for g in res.get("groups") or [] if key and g.get(key) is not None]
    if res.get("focus") and key:
        row = res["focus"][0]
        out.append(f"For {row['group']}, the {_label(key)} works out to {_cellv(row.get(key))}.")
    elif len(groups) > 1:
        top, bottom = max(groups, key=lambda g: g[key]), min(groups, key=lambda g: g[key])
        share = top.get("share_of_" + key)
        out.append(f"Put simply: {top['group']} accounts for the most {_label(key)} ({_cellv(top[key])}"
                   + (f", about {pct(share * 100)} of the total {_cellv(res['totals'][key])}" if share else "")
                   + f"), and {bottom['group']} the least ({_cellv(bottom[key])}).")
        if res.get("why"):
            w = res["why"]
            out.append(f"That is mostly because {top['group']} has {w['records']} of the {w['records_of']} records"
                       + (" and higher values per record" if any(x["rank"] == 1 for x in w["per_row"]) else "")
                       + " — more entries and bigger entries add up to a bigger total.")
    elif key:
        out.append(f"Put simply: the {_label(key)} comes to {_cellv(res['totals'].get(key))}.")
    if res.get("assumptions"):
        out.append("It is a 'what if' figure: it uses your assumption ("
                   + ", ".join(f"{_label(a['name'])} = {_cellv(a['value'])}" for a in res["assumptions"]
                               if a.get("value") is not None)
                   + ") together with the rates recorded in the data.")
    out.append(f"It was worked out exactly from {res['rows_valid']} of the {res['rows_total']} records"
               + (f" ({res['rows_excluded']} were left out)" if res["rows_excluded"] else "") + ".")
    checks = res.get("data_checks") or []
    if checks:
        out.append("One caution: " + "; ".join(f"line{'s' if len(c['lines']) != 1 else ''} "
                                               f"{', '.join(map(str, c['lines']))} {'have' if len(c['lines']) != 1 else 'has'} "
                                               f"{c['issue'].replace('exact duplicate', 'an exact duplicate')}"
                                               for c in checks)
                   + ", and these were counted as recorded, so the true figure may differ once they are fixed.")
    out.append("This is a calculation from the records, not a machine-learning prediction.")
    return " ".join(out)


SIMPLE_SYSTEM = ("You are Silt Copilot, a friendly data analyst talking to a business user with no data background. "
                 "Explain the previous answer in simple terms, in 3-6 short sentences of plain English: what it says, "
                 "what it means in practice, and any caution about the data. Explain it now — never say you will "
                 "explain. Use ONLY numbers that appear in the evidence, written the same way; do not calculate new "
                 "numbers. No JSON, no formulas, no code, no column names with underscores, no headings.")
ADVICE_SYSTEM = ("You are Silt Copilot, a data analyst advising a business. Using ONLY the evidence given, write 3-5 "
                 "concrete recommended next steps as a markdown bullet list. Each bullet starts with an action verb "
                 "and names the specific evidence it rests on. Use only numbers that appear in the evidence; do not "
                 "calculate new ones. Fixing data errors comes first when there are any. These are suggestions to "
                 "check, not proven causes: do not claim causation. No JSON, no headings.")
WHY_SYSTEM = ("You are Silt Copilot. Explain, in 3-5 plain sentences, why the group named in the evidence has the "
              "result it has, using ONLY the facts and numbers in the evidence (how many records it has, its values "
              "per record, data problems in its records). Say these are contributing facts, not proven causes. "
              "No JSON, no headings.")

_BOILER = re.compile(r"(?:[^.]|\.(?=\d))*\b(compares (averages|totals) only|statistical test|significance test)(?:[^.]|\.(?=\d))*\.\s*", re.I)
_PROMISE = re.compile(r"^\s*(sure[,!.]?\s*)?(i('ll| will| can| am going to)|let me|here('s| is) (a|an|the) "
                      r"(simple|simpler|plain)? ?explanation:?\s*$)", re.I)


def _model_prose(system: str, question: str, evidence, notes: list[str], max_tokens: int = 380,
                 bullets: bool = False) -> str | None:
    """Model prose grounded in the evidence, with one bounded retry naming what was wrong. None if it fails twice."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\nEvidence: {_short(evidence)}"}]
    for attempt in (1, 2):
        try:
            text = planner.model_chat(msgs, max_tokens=max_tokens).split("</think>")[-1].strip()
        except Exception as e:  # noqa: BLE001 - the model is optional; the deterministic answer stands
            notes.append(f"The local model could not be reached ({type(e).__name__}); a deterministic answer is shown.")
            return None
        text = _BOILER.sub("", text).strip()
        problem = None
        if text.startswith("{") or planner.parse_plan(text) is not None or calc.parse(text) is not None:
            problem = "you replied with JSON or a plan; reply in plain prose"
        elif len(text) < 40 or not re.search(r"[.!?](\s|$)", text) or _PROMISE.search(text) and len(text) < 200:
            problem = "you did not actually explain; give the explanation itself now"
        elif _internal_names(text, evidence):
            problem = (f"do not mention internal field names ({', '.join(_internal_names(text, evidence)[:3])}); "
                       "say it in plain words")
        else:
            bad = explain.ungrounded(text, evidence, question)
            if bad:
                problem = (f"these numbers are not in the evidence: {', '.join(bad[:6])}; use only numbers that "
                           "appear in the evidence, or no numbers")
        if problem is None:
            ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
            if ends and not re.search(r"[.!?)_*]\s*$", text):
                text = text[:ends[-1]]          # cut off by the token limit: keep whole sentences
            if attempt == 2:
                notes.append("The model's first draft was rejected and it rewrote it once.")
            return text
        notes.append(f"The model's {'first' if attempt == 1 else 'second'} draft was not used ({problem[:160]}).")
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"That was not usable: {problem}. Write it again."}]
    return None


def _keys(obj, out=None) -> set:
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def _internal_names(text: str, evidence) -> list[str]:
    return [k for k in _keys(evidence) if "_" in k and re.search(rf"\b{re.escape(k)}\b", text)]


def explain_previous(message: str, memory: dict, model_on: bool, notes: list[str]) -> tuple[str, dict]:
    prev = memory["last_answer"]
    calc_prev = (memory.get("last_calc") or {})
    deterministic = (simple_from_calc(calc_prev["result"], prev["question"])
                     if calc_prev and calc_prev.get("question") == prev["question"] else
                     "In short: " + _plain(prev["answer"], 5))
    evidence = {"previous_question": prev["question"], "previous_answer_numbers": prev.get("evidence"),
                "plain_summary": deterministic}
    text = _model_prose(SIMPLE_SYSTEM, f"{message} (about the answer to: {prev['question']})", evidence, notes) \
        if model_on else None
    grounding = {"checked": bool(model_on), "used_model": text is not None}
    if text:
        return text, grounding
    return deterministic, grounding


def advice(message: str, df: pd.DataFrame, memory: dict, model_on: bool, notes: list[str]) -> tuple[str, dict, dict]:
    """Recommendations from evidence: data problems, the last computed breakdown, and dataset facts."""
    from app.analyst import overview
    from app.analyst.present import _cellv, _label, pct
    ev: dict = {"records": int(len(df))}
    cols = [overview.column_summary(df, c) for c in df.columns]
    # Only genuine data problems count as errors to fix; an identifier column or a statistically unusual value is not one.
    profile = list(overview.quality_issues(df, cols))
    issues = [i for i in profile if not re.search(r"identifier|single value|unusual values outside", i)]
    unusual = [i for i in profile if "unusual values outside" in i]
    last = memory.get("last_calc")
    steps: list[str] = []
    if last:
        res = last["result"]
        ev["last_analysis"] = {"question": last["question"], "totals": res["totals"],
                               "by_" + str(res.get("group_by") or "group"): [
                                   {k: v for k, v in g.items() if k != "per_row"} for g in res.get("groups", [])[:12]],
                               "data_checks": res.get("data_checks")}
        if res.get("data_checks"):       # the records the analysis actually used: these replace the generic profile
            issues = [f"line{'s' if len(c['lines']) != 1 else ''} {', '.join(map(str, c['lines']))}: {c['issue']}"
                      for c in res["data_checks"]]
            issues += [i for i in overview.quality_issues(df, cols) if "missing" in i]
        key = next((n for n, e in res["aggregates"].items() if e.startswith("sum(")), None)
        groups = [g for g in res.get("groups") or [] if key and g.get(key) is not None]
        if len(groups) > 1:
            top, bottom = max(groups, key=lambda g: g[key]), min(groups, key=lambda g: g[key])
            share = top.get("share_of_" + key)
            steps.append(f"**Review {top['group']} first**: it carries the largest {_label(key)} ({_cellv(top[key])}"
                         + (f", {pct(share * 100)} of the total" if share else "") + "), so changes there move the "
                         "total most. Check whether its hours and rates are expected.")
            per = {k: v for k, v in (top.get("per_row") or {}).items() if v is not None}
            if per:
                k, v = next(iter(per.items()))
                steps.append(f"**Compare like with like**: {top['group']}'s {k.replace('_', ' ')} is {_cellv(v)}; compare it with the "
                             "other groups before deciding whether the cost is driven by volume (more records) or "
                             "by price (higher values per record).")
            steps.append(f"**Look into {bottom['group']}**: it has the smallest {_label(key)} ({_cellv(bottom[key])}); "
                         "confirm whether that reflects less work or missing / incorrect records.")
    ev["data_quality_issues"] = issues
    if issues:
        steps.insert(0, "**Fix the data errors before acting on the numbers**: "
                     + "; ".join(i.rstrip(". ") for i in issues[:5]) + ".")
    model = memory.get("model")
    if model and model.get("importance"):
        top = [i for i in model["importance"] if i["importance"] > 0][:3]
        if top:
            m = model["metrics"]
            score = (f"balanced accuracy {m['balanced_accuracy']:.2f} vs {m['baseline_balanced_accuracy']:.2f} for "
                     "always guessing the common class" if "balanced_accuracy" in m else
                     f"average error {m['mae']:,.2f} vs {m['baseline_mae']:,.2f} for always guessing the average")
            ev["prediction_model"] = {"target": model["target"], "strongest_inputs": [i["feature"] for i in top],
                                      "held_out_score": score}
            steps.append(f"**Look first at {', '.join(i['feature'].replace('_', ' ') for i in top)}**: these are the "
                         f"inputs the {model['target'].replace('_', ' ')} model relied on most on held-out records "
                         f"({score}). They are associations to investigate, not proven causes.")
    if not last and not ev.get("prediction_model"):
        ev["columns"] = [{k: c[k] for k in ("name", "kind") if k in c} for c in cols]
        steps.append("**Pick the measure that defines performance** (for example cost, revenue or hours) and ask me "
                     "to rank the groups by it; I can then compare them and look for what differs.")
    if unusual:
        steps.append("**Check the unusual values** before relying on averages: " + "; ".join(u.rstrip(". ") for u in unusual[:3])
                     + ". Unusual is not the same as wrong.")
    if any(kind_of(df[c]) == "datetime" for c in df.columns):
        steps.append("**Track it over time**: ask for the trend of the key measure so a change can be checked against "
                     "the period before it.")
    deterministic = "\n".join(f"{i}. {s}" for i, s in enumerate(steps[:5], 1))
    ev["suggested_steps"] = deterministic
    text = _model_prose(ADVICE_SYSTEM, message, ev, notes, max_tokens=450, bullets=True) if model_on else None
    head = "Here is what I would do next, based on the evidence in this conversation:"
    lead = f"**In short:** {_one_para(text)}\n\n" if text else ""
    tail = ("\n\n_These are suggestions drawn from the data, not proven causes: check them against what you know "
            "about the business before acting._")
    return f"{lead}{head}\n\n{deterministic}{tail}", {"checked": bool(model_on), "used_model": text is not None}, ev


def _one_para(text: str) -> str:
    return re.sub(r"\s*\n+\s*", " ", text.strip())


def why_group(message: str, df: pd.DataFrame, memory: dict, model_on: bool, notes: list[str]) -> tuple[str, dict, dict] | None:
    from app.analyst.present import _cellv, _label
    last = memory.get("last_calc")
    if not last or not last["result"].get("groups"):
        return None
    res = last["result"]
    key = next((n for n, e in res["aggregates"].items() if e.startswith("sum(")), next(iter(res["aggregates"]), None))
    groups = [g for g in res["groups"] if g.get(key) is not None]
    if len(groups) < 2:
        return None
    named = [g for g in groups if re.search(rf"(?<!\w){re.escape(g['group'].lower())}(?!\w)", message.lower())]
    target = named[0] if named else min(groups, key=lambda g: g[key])
    rank = 1 + sum(g[key] > target[key] for g in groups)
    facts = [f"{target['group']} ranks {rank} of {len(groups)} on {_label(key)} with {_cellv(target[key])}, "
             f"from {target['rows']} of {res['rows_valid']} records."]
    for k, v in (target.get("per_row") or {}).items():
        others = [g["per_row"].get(k) for g in groups if g is not target and g.get("per_row", {}).get(k) is not None]
        if v is not None and others:
            facts.append(f"Its {k.replace('_', ' ')} is {_cellv(v)}, against {_cellv(min(others))} to {_cellv(max(others))} for the "
                         "other groups.")
    gcol = res.get("group_by")
    lines_in = []
    if gcol in df.columns:
        mine = {i + 2 for i, v in enumerate(df[gcol].astype(str)) if v == target["group"]}
        for c in res.get("data_checks") or []:
            hit = sorted(mine & set(c["lines"]))
            if hit:
                lines_in.append(f"line{'s' if len(hit) != 1 else ''} {', '.join(map(str, hit))} ({c['issue']})")
    if lines_in:
        facts.append("Its records include data problems that pull the figure down or distort it: "
                     + "; ".join(lines_in) + ".")
    facts.append("These are contributing facts from the data, not proven causes.")
    deterministic = " ".join(facts)
    ev = {"group": target["group"], "measure": key, "facts": deterministic,
          "all_groups": [{k: v for k, v in g.items() if not k.startswith("share")} for g in groups]}
    text = _model_prose(WHY_SYSTEM, message, ev, notes) if model_on else None
    lead = f"**In short:** {_one_para(text)}\n\n" if text else ""
    return lead + "\n".join(f"- {x}" for x in facts), {"checked": bool(model_on), "used_model": text is not None}, ev


# ------------------------------------------------- the measure computed earlier --
def _columns_in(spec: dict, df: pd.DataFrame) -> list[str]:
    text = " ".join(list(spec["derive"].values()) + list(spec["aggregate"].values()))
    return [c for c in map(str, df.columns) if re.search(rf"\b{re.escape(c)}\b", text)]


def last_measure(df: pd.DataFrame, memory: dict, sc: dict | None = None,
                 historical: bool = False) -> tuple[pd.DataFrame, str, str] | None:
    """(records kept by the last calculation with a per-record `measure` column, its name, its per-record formula)
    for the last summed result in the conversation, e.g. labor cost per record. None if there is none."""
    last = memory.get("last_calc")
    if not last:
        return None
    spec = last["spec"]
    name, inner = None, None
    for n, a in spec["aggregate"].items():
        m = re.match(r"^\s*sum\s*\((.*)\)\s*$", a, re.S)
        if m and calc._balanced_inner(m.group(1)):  # noqa: SLF001
            name, inner = n, m.group(1)
            break
    if name is None:
        return None
    frame = calc.run_derive(df, spec["derive"])
    keep = pd.Series(True, index=frame.index)
    for e in scoping.replace_conflicting(spec["exclude"], sc):
        hit = calc.evaluate(e["where"], frame, df)
        if isinstance(hit, pd.Series):
            keep &= ~hit.fillna(False).astype(bool)
    in_scope = scoping.mask(df, sc)
    excluded: list[dict] = []
    if historical:
        valid, excluded = scoping.eligibility(df[in_scope & keep], _columns_in(spec, df), historical=True)
        keep = keep & in_scope & valid.reindex(df.index, fill_value=False)
    else:
        keep = keep & in_scope
    frame = frame[keep].copy()
    v = calc.evaluate(inner, frame, df)
    frame["__measure__"] = pd.to_numeric(v if isinstance(v, pd.Series) else pd.Series(v, index=frame.index),
                                         errors="coerce")
    frame.attrs["silt_excluded"] = excluded
    frame.attrs["silt_scope"] = scoping.describe(sc, int(in_scope.sum()), len(df)) if scoping.active(sc) else ""
    return frame, name, inner


def auto_freq(dates: pd.Series) -> str:
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.empty:
        return "M"
    months = d.dt.to_period("M").nunique()
    if months >= 3:
        return "M"
    return "W" if d.dt.to_period("W").nunique() >= 6 and d.dt.normalize().nunique() > 20 else "D"


RATE_LIKE = re.compile(r"rate|price|percent|pct|ratio|avg|average|score|per_|margin|age|tenure|temperature", re.I)


def series(frame: pd.DataFrame, value_col: str, date_col: str, freq: str, agg: str) -> list[dict]:
    d = frame[[date_col, value_col]].dropna()
    period = pd.to_datetime(d[date_col]).dt.to_period(freq)
    g = d.groupby(period)[value_col].agg("mean" if agg == "avg" else "sum").sort_index()
    return [{"period": str(k), "value": float(v), "records": int(n)}
            for (k, v), n in zip(g.items(), d.groupby(period).size().sort_index())]


def trend_of_measure(df: pd.DataFrame, memory: dict, sc: dict | None = None) -> dict | None:
    lm = last_measure(df, memory, sc, historical=True)
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    if lm is None or not dates:
        return None
    frame, name, inner = lm
    freq = auto_freq(frame[dates[0]])
    pts = series(frame, "__measure__", dates[0], freq, "sum")
    if len(pts) < 2:
        return None
    t = _trend_report(pts, name.replace("_", " "), dates[0], freq, "sum", source=f"total {name.replace('_', ' ')} "
                      f"per {_FREQ[freq]} from the previous calculation ({len(frame)} eligible records)")
    t.update(excluded=frame.attrs.get("silt_excluded", []), scope=frame.attrs.get("silt_scope", ""))
    return t


_FREQ = {"D": "day", "W": "week", "M": "month", "Q": "quarter"}


def _trend_report(pts: list[dict], label: str, date: str, freq: str, agg: str, source: str) -> dict:
    y = np.array([p["value"] for p in pts])
    x = np.arange(len(pts), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(pts) >= 2 else 0.0
    first, last = pts[0], pts[-1]
    gaps = []
    if freq in ("D", "M"):
        per = pd.PeriodIndex([p["period"] for p in pts], freq=freq)
        for a, b in zip(per[:-1], per[1:]):
            if (b - a).n > 1:
                gaps.append(f"no records between {a} and {b}")
    return {"label": label, "date": date, "frequency": _FREQ[freq], "aggregation": agg, "points": pts,
            "first": first, "last": last, "change": last["value"] - first["value"],
            "change_pct": (last["value"] - first["value"]) / abs(first["value"]) * 100 if first["value"] else None,
            "peak": max(pts, key=lambda p: p["value"]), "low": min(pts, key=lambda p: p["value"]),
            "slope_per_period": slope, "gaps": gaps, "source": source,
            "note": "A trend describes what happened; it does not by itself predict what comes next."}


def trend_text(t: dict) -> str:
    from app.analyst.present import _cellv
    lab = t["label"]
    direction = "rose" if t["change"] > 0 else "fell" if t["change"] < 0 else "was flat"
    lines = [f"**{lab.capitalize()} {direction}** from **{_cellv(t['first']['value'])}** ({t['first']['period']}) to "
             f"**{_cellv(t['last']['value'])}** ({t['last']['period']})"
             + (f", a change of {_cellv(t['change'])} ({t['change_pct']:+.1f}%)." if t["change_pct"] is not None else "."),
             "",
             f"- Highest {t['frequency']}: {t['peak']['period']} ({_cellv(t['peak']['value'])}); lowest: "
             f"{t['low']['period']} ({_cellv(t['low']['value'])}).",
             f"- Straight-line change per {t['frequency']}: {_cellv(t['slope_per_period'])}."]
    if t["gaps"]:
        lines.append("- Gaps: " + "; ".join(t["gaps"][:3]) + " — the series is not continuous, so read the change "
                     "with care.")
    small = [p for p in t["points"] if p.get("records", 99) < 3]
    if len(t["points"]) < 6 or small:
        lines.append(f"- Only {len(t['points'])} {t['frequency']}s of data"
                     + (f", some with fewer than 3 records" if small else "") + ": this is a description, too short "
                     "to call a reliable trend.")
    lines += ["", f"| {t['frequency']} | {lab} | records |", "|---|---:|---:|"]
    lines += [f"| {p['period']} | {_cellv(p['value'])} | {p.get('records', '')} |" for p in t["points"]]
    lines += ["", f"_Computed from {t['source']}. {t['note']}_"]
    return _scope_lines(t) + "\n".join(lines)


# ---------------------------------------------------------------- forecast --
MIN_POINTS_ML = 8          # fewer periods than this cannot be both fitted and validated on held-out periods


_NEXT = re.compile(r"\b(next|upcoming|future|coming)\b|\bforecast\w*|\bproject(ion|ed)?\b", re.I)


def wants_forecast(message: str) -> bool:
    return bool(re.search(r"\b(predict\w*|forecast\w*|project\w*|estimate\w*)\b", message, re.I)
                and _NEXT.search(message))


_PERIOD_WORDS = (("D", r"\b(day|daily|tomorrow|date|\d{1,2}(st|nd|rd|th)?,? \d{4})\b"), ("W", r"\bweek(ly)?\b"),
                 ("M", r"\bmonth(ly)?\b"), ("Q", r"\bquarter(ly)?\b"))


_DATE_IN_TEXT = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b|\b(january|february|march|april|may|june|july|august|september|"
                           r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b|\b(\d{1,2})(?:st|nd|rd|th)?\s+"
                           r"(january|february|march|april|may|june|july|august|september|october|november|december),?"
                           r"\s+(\d{4})\b", re.I)


def requested_date(message: str) -> pd.Timestamp | None:
    """A specific day the question asks about ("September 4, 2026", "4 September 2026", "2026-09-04")."""
    m = _DATE_IN_TEXT.search(message or "")
    if not m:
        return None
    if m.group(1):
        text = m.group(1)
    elif m.group(2):
        text = f"{m.group(2)} {m.group(3)} {m.group(4)}"
    else:
        text = f"{m.group(6)} {m.group(5)} {m.group(7)}"
    ts = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(ts) else ts


def requested_freq(message: str) -> str | None:
    hits = [f for f, pat in _PERIOD_WORDS if re.search(pat, message or "", re.I)]
    return hits[0] if len(hits) == 1 else None


def forecast(df: pd.DataFrame, memory: dict, value: str | None = None, sc: dict | None = None,
             derived: dict | None = None, label: str | None = None, period: str | None = None,
             target_date: pd.Timestamp | None = None) -> dict:
    """Next-period forecast of a measure over time. With enough periods: a linear trend (scikit-learn
    LinearRegression on the period index) validated on the last periods against a naive last-value baseline,
    used only if it beats the baseline. With too few: no model; the historical average is reported and
    labelled as an estimate."""
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    if not dates:
        raise ValueError("This dataset has no date column, so there is no 'next' period to forecast.")
    if value:
        in_scope = scoping.mask(df, sc)
        valid, excluded = scoping.eligibility(df[in_scope], [value, dates[0]], historical=True)
        frame = df[in_scope & valid.reindex(df.index, fill_value=False)]
        frame = frame.assign(__measure__=pd.to_numeric(frame[value], errors="coerce"))
        frame.attrs["silt_excluded"] = excluded
        frame.attrs["silt_scope"] = scoping.describe(sc, int(in_scope.sum()), len(df)) if scoping.active(sc) else ""
        label = value.replace("_", " ")
        agg = "avg" if RATE_LIKE.search(value) else "sum"
        source = f"{'average' if agg == 'avg' else 'total'} {label} per period"
    else:
        source_memory = {"last_calc": {"spec": derived, "question": label}} if derived else memory
        lm = last_measure(df, source_memory, sc, historical=True)
        if lm is None:
            raise LookupError("which value")
        frame, name, _ = lm
        label, agg = (label or name.replace("_", " ")), "sum"
        source = (f"total {label} per period ({describe_definition(derived)})" if derived else
                  f"total {label} per period, as computed in the previous answer")
    out_label = label
    if frame.empty:
        raise ValueError("No eligible historical records are left in the requested scope"
                         + (f" ({frame.attrs.get('silt_scope')})" if frame.attrs.get("silt_scope") else "") + ".")
    freq = period or auto_freq(frame[dates[0]])
    pts = series(frame, "__measure__", dates[0], freq, agg)
    coverage = None
    last_recorded = pd.Period(pts[-1]["period"], freq=freq) if pts else None
    if agg == "sum" and len(pts) >= 2:
        pts, coverage = _drop_incomplete_periods(df, frame, pts, dates[0], freq, sc)
    period_note = None
    if period and len(pts) < 2:
        auto = auto_freq(frame[dates[0]])
        if auto != period and len(series(frame, "__measure__", dates[0], auto, agg)) >= 2:
            period_note = (f"there is only {len(pts)} {_FREQ[period]} of history, so the estimate is per {_FREQ[auto]}, "
                           f"not per {_FREQ[period]} as asked")
            freq = auto
            pts = series(frame, "__measure__", dates[0], freq, agg)
    out = {"label": out_label, "frequency": _FREQ[freq], "points": pts, "aggregation": agg, "source": source,
           "periods": len(pts), "excluded": frame.attrs.get("silt_excluded", []), "scope": frame.attrs.get("silt_scope", ""),
           "records_used": int(len(frame)), "definition": describe_definition(derived) if derived else None,
           "period_note": period_note, "coverage": coverage}
    if len(pts) < 2:
        raise ValueError(f"Only {len(pts)} {_FREQ[freq]} of eligible history, so nothing can be projected.")
    per = pd.PeriodIndex([p["period"] for p in pts], freq=freq)
    last = max(per[-1], last_recorded) if last_recorded is not None else per[-1]   # an incomplete last period still happened
    target = pd.Period(target_date, freq=freq) if target_date is not None else last + 1
    steps = (target - last).n
    if steps < 1:
        raise ValueError(f"{target} is not after the last recorded period ({last}), so it is not a forecast; ask for "
                         "that period's actual value instead.")
    nxt = str(target)
    out["next_period"] = nxt
    out["horizon"] = steps
    now = pd.Period(scoping.today(), freq=freq)
    if target_date is None and (now - last).n > 1:
        out["horizon_note"] = (f"the records end in {last}, {(now - last).n} {_FREQ[freq]}s before today "
                               f"({scoping.today().date()}), so 'next' is read as the {_FREQ[freq]} after the last record "
                               f"({nxt}); name a date to forecast a different one")
    y = np.array([p["value"] for p in pts])
    if len(pts) < MIN_POINTS_ML:
        out.update(kind="historical_estimate", ml_used=False, method=f"the average of the {len(pts)} past {_FREQ[freq]}s",
                   estimate=float(y.mean()), last_value=float(y[-1]),
                   reason=f"only {len(pts)} {_FREQ[freq]}s of history; at least {MIN_POINTS_ML} are needed to fit a "
                          "model and still hold some periods back to test it")
        return out
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    x = np.array([(p - per[0]).n for p in per], dtype=float).reshape(-1, 1)
    k = max(2, int(round(len(pts) * 0.25)))
    xtr, ytr, xte, yte = x[:-k], y[:-k], x[-k:], y[-k:]
    lin = LinearRegression().fit(xtr, ytr)
    mae = float(mean_absolute_error(yte, lin.predict(xte)))
    naive = float(mean_absolute_error(yte, np.full(k, ytr[-1])))
    full = LinearRegression().fit(x, y)
    x_next = float((target - per[0]).n)
    pred = float(full.predict([[x_next]])[0])
    out.update(validation={"held_out_periods": k, "model_mae": mae, "naive_mae": naive,
                           "naive": "repeat the last value before the held-out periods"},
               fitted=[{"period": p["period"], "actual": p["value"], "fitted": float(f)}
                       for p, f in zip(pts, full.predict(x))])
    from_today = float(full.predict([[float((now + 1 - per[0]).n)]])[0])
    if mae < naive and out.get("horizon_note") and target_date is None and \
            abs(from_today - pred) > 0.05 * max(abs(pred), 1e-9):
        # a trend model's answer depends on how far ahead "next" is; with the records ending long before today the two
        # readings differ materially (more than 5%), so it is asked rather than assumed
        raise AmbiguousHorizon(f"The records end in {last} but today is {scoping.today().date()}. Do you want "
                               f"{lab_for(out)} for the {_FREQ[freq]} right after the records ({nxt}) or for the next "
                               f"{_FREQ[freq]} from today ({now + 1})? A trend model gives different numbers for the two.")
    if mae < naive:
        out.update(kind="ml_prediction", ml_used=True, prediction=pred,
                   method="linear trend regression (scikit-learn LinearRegression on the period number), fitted on "
                          "all periods after validation")
    else:
        out.update(kind="historical_estimate", ml_used=False, estimate=float(y[-1]), last_value=float(y[-1]),
                   method=f"the most recent {_FREQ[freq]}'s value repeated",
                   reason=f"the trend model did not beat simply repeating the last value on the {k} held-out periods "
                          f"(error {mae:,.2f} vs {naive:,.2f}), so no model prediction is given")
    return out


class AmbiguousHorizon(ValueError):
    """The forecast horizon is ambiguous in a way that changes the answer; the message is the question to ask."""


def lab_for(out: dict) -> str:
    return out.get("label") or "the value"


def unit_column(df: pd.DataFrame, date: str) -> str | None:
    """The column naming the units that make up a period's total (employees, stores, products...): the categorical
    column whose values repeat across periods with the most distinct values; names/ids preferred on a tie."""
    best, key = None, None
    for c in df.columns:
        if c == date or kind_of(df[c]) != "categorical":
            continue
        s = df[c].dropna()
        n = s.nunique()
        if n < 2 or n >= len(df) or calc._looks_like_time(s):  # noqa: SLF001
            continue
        k = (n, bool(re.search(r"name|id$|_id|employee|person|staff|store|customer|user|product|site|branch", str(c), re.I)))
        if key is None or k > key:
            best, key = c, k
    return best


def _drop_incomplete_periods(df, frame, pts, date, freq, sc):
    """For a total over many units (all employees, all stores), a period where many units have no VALID record is not
    a comparable total: it is left out of the history and reported. Coverage = distinct units with a valid record in
    the period, against the usual number of units recording per period (median over periods)."""
    unit = unit_column(df, date)
    if unit is None or frame[unit].nunique() <= 1 or df[scoping.mask(df, sc)][unit].nunique() <= 1:
        return pts, None                  # one employee / one store (however the scope names it): no coverage question
    in_scope = scoping.mask(df, sc)
    allrec = df[in_scope & (pd.to_datetime(df[date], errors="coerce") <= scoping.today())]
    rec_per = allrec.groupby(pd.to_datetime(allrec[date]).dt.to_period(freq))[unit].nunique()
    valid_per = frame.groupby(pd.to_datetime(frame[date]).dt.to_period(freq))[unit].nunique()
    usual = float(rec_per.median()) if len(rec_per) else 0.0
    kept, dropped = [], []
    for p in pts:
        n = int(valid_per.get(pd.Period(p["period"], freq=freq), 0))
        if usual and n < 0.75 * usual:
            dropped.append({"period": p["period"], "units": n, "usual": usual, "total_recorded": p["value"]})
        else:
            kept.append({**p, "units": n})
    info = {"unit": unit, "usual_units_per_period": usual, "units_in_history": int(frame[unit].nunique()),
            "incomplete": dropped}
    if len(kept) < 1:
        return pts, {**info, "incomplete": [], "note": "every period is incomplete; nothing was left out"}
    return kept, info


def _scope_lines(r: dict) -> str:
    out = []
    if r.get("scope"):
        out.append(f"**Scope:** {r['scope']}.")
    if r.get("excluded"):
        out.append("**Not used as history:** " + "; ".join(
            f"{e['records']} record{'s' if e['records'] != 1 else ''} — {e['reason']} (line{'s' if len(e['lines']) != 1 else ''} "
            f"{', '.join(map(str, e['lines']))})" for e in r["excluded"]) + ".")
    return ("\n\n".join(out) + "\n\n") if out else ""


def forecast_text(f: dict) -> str:
    from app.analyst.present import _cellv
    lab = f["label"]
    lines = [_scope_lines(f).strip()] if _scope_lines(f) else []
    if f.get("definition"):
        lines += [f"**{lab.capitalize()}** is not a column, so it was derived for each record: {f['definition']}.", ""]
    cov = f.get("coverage")
    if cov:
        lines += [f"**Aggregation:** total over all {cov['unit'].replace('_', ' ')} values per {f['frequency']} "
                  f"({cov['units_in_history']} with valid records; usually {cov['usual_units_per_period']:g} recording "
                  f"per {f['frequency']}).", ""]
        if cov["incomplete"]:
            lines += ["**Left out as incomplete periods** (too few valid records to be a comparable total): " + "; ".join(
                f"{x['period']} — {x['units']} of the usual {x['usual']:g} {cov['unit'].replace('_', ' ')} values had a "
                f"valid record (recorded total {_cellv(x['total_recorded'])})" for x in cov["incomplete"]) + ".", ""]
    if f.get("period_note"):
        lines += [f"_Note: {f['period_note']}._", ""]
    if f.get("horizon_note"):
        lines += [f"_Note: {f['horizon_note']}._", ""]
    if f["kind"] == "ml_prediction":
        v = f["validation"]
        lines += [f"**Predicted {lab} for {f['next_period']}: {_cellv(f['prediction'])}**", "",
                  f"_Type of answer: a **machine-learning prediction** — {f['method']}._", "",
                  f"- Tested on the last {v['held_out_periods']} {f['frequency']}s held back from training: average "
                  f"error {_cellv(v['model_mae'])}, against {_cellv(v['naive_mae'])} for the naive baseline ({v['naive']}).",
                  f"- Trained on {f['periods']} {f['frequency']}s of {f['source']}.",
                  "- A straight-line trend assumes the recent direction continues; it cannot foresee changes in "
                  "staffing, prices or seasonality."]
    else:
        lines += [f"**Estimated {lab} for {f['next_period']}: {_cellv(f['estimate'])}** — {f.get('method', 'from past values')} "
                  f"of {lab}.", "",
                  f"_Type of answer: a **historical estimate** ({f.get('method', 'from past values')}), **not a "
                  f"machine-learning prediction**: {f['reason']}._"]
        if f["estimate"] != f["last_value"]:
            lines.append(f"\nThe most recent {'comparable ' if (f.get('coverage') or {}).get('incomplete') else ''}"
                         f"{f['frequency']} ({f['points'][-1]['period']}) was {_cellv(f['last_value'])}.")
        if len(f["points"]) < MIN_POINTS_ML:
            lines.append(f"\nTo get a real forecast, add more {f['frequency']}s of records (at least {MIN_POINTS_ML}).")
    lines += ["", f"| {f['frequency']} | {lab} |", "|---|---:|"]
    lines += [f"| {p['period']} | {_cellv(p['value'])} |" for p in f["points"][-24:]]
    return "\n".join(lines)


# ------------------------------------------------------------------ scenario --
def scenario_on_measure(df: pd.DataFrame, memory: dict, column: str, change_pct: float,
                        sc: dict | None = None) -> dict | None:
    """Re-run the previous calculation with `column` changed by change_pct %: what the change does to the result the
    user was just looking at (e.g. labor cost when hourly rates rise 10%)."""
    last = memory.get("last_calc")
    if not last:
        return None
    spec = last["spec"]
    derived = spec["derive"]
    def uses(e: str, depth: int = 0) -> bool:
        if re.search(rf"\b{re.escape(column)}\b", e):
            return True
        return depth < 4 and any(re.search(rf"\b{re.escape(k)}\b", e) and uses(v, depth + 1) for k, v in derived.items())
    if not any(re.match(r"\s*(sum|mean|avg|average)\s*\(", a) and uses(a) for a in spec["aggregate"].values()):
        return None
    if scoping.active(sc):     # the question's scope, not the previous one: add what is missing, keep the rest
        kept = scoping.replace_conflicting(spec["exclude"], sc)
        have = {e["where"].replace(" ", "") for e in kept}
        spec = {**spec, "exclude": kept + [e for e in scoping.as_exclusions(sc) if e["where"].replace(" ", "") not in have]}
    changed = df.copy()
    changed[column] = pd.to_numeric(changed[column], errors="coerce") * (1 + change_pct / 100)
    base = calc.run(spec, df, question=last["question"])
    new = calc.run(spec, changed, question=last["question"])
    rows = []
    for n in base["totals"]:
        b, h = base["totals"][n], new["totals"][n]
        rows.append({"result": n, "baseline": b, "scenario": h,
                     "difference": None if b is None or h is None else h - b})
    groups = []
    if base.get("group_by"):
        hyp = {g["group"]: g for g in new["groups"]}
        key = next(iter(base["totals"]))
        for g in base["groups"]:
            hg = hyp.get(g["group"], {})
            groups.append({"group": g["group"], "baseline": g.get(key), "scenario": hg.get(key),
                           "difference": None if g.get(key) is None or hg.get(key) is None else hg[key] - g[key]})
    return {"column": column, "change_pct": change_pct, "of_question": last["question"], "results": rows,
            "scope": scoping.describe(sc, base["rows_valid"], base["rows_total"]) if scoping.active(sc) else "",
            "group_by": base.get("group_by"), "groups": groups, "rows_used": base["rows_valid"],
            "rows_total": base["rows_total"], "data_checks": base.get("data_checks")}


def scenario_text(s: dict) -> str:
    from app.analyst.present import _cellv, _label
    r0 = s["results"][0]
    sign = "+" if s["change_pct"] >= 0 else ""
    lines = ([f"**Scope:** {s['scope']}.", ""] if s.get("scope") else []) + [f"If **{_label(s['column'])}** changes by **{sign}{s['change_pct']:g}%**, **{_label(r0['result'])}** goes from "
             f"**{_cellv(r0['baseline'])}** to **{_cellv(r0['scenario'])}** ({'+' if (r0['difference'] or 0) >= 0 else ''}"
             f"{_cellv(r0['difference'])}).", ""]
    if s["groups"]:
        g = s["group_by"]
        lines += [f"| {_label(g)} | now | with the change | difference |", "|---|---:|---:|---:|"]
        lines += [f"| {x['group']} | {_cellv(x['baseline'])} | {_cellv(x['scenario'])} | {_cellv(x['difference'])} |"
                  for x in s["groups"]]
        lines.append("")
    lines += [f"Re-ran the previous calculation (*{s['of_question']}*) on the same {s['rows_used']} of "
              f"{s['rows_total']} records with every {_label(s['column'])} value changed by {sign}{s['change_pct']:g}%; "
              "nothing else was changed."]
    if s.get("data_checks"):
        lines.append("\n⚠️ The same data problems as before are still included (" + "; ".join(
            f"line{'s' if len(c['lines']) != 1 else ''} {', '.join(map(str, c['lines']))}: {c['issue']}"
            for c in s["data_checks"]) + ").")
    lines += ["", "_Type of answer: a **what-if calculation** under your assumption — not a forecast and not a "
              "machine-learning prediction._"]
    return "\n".join(lines)


# ------------------------------------------------------------------- charts --
CHART_WORDS = re.compile(r"\b(chart|plot|graph|visuali[sz]\w*|diagram|histogram|distribution|scatter|pie|donut|doughnut|bar ?chart|"
                         r"line ?chart|draw|show (it|this|that|them) (as|in)|visual)\b", re.I)


def chart_request(message: str) -> str | None:
    """The chart type asked for ('bar', 'line', 'pie', 'scatter', 'histogram', 'avp', 'auto') or None."""
    low = message.lower()
    if re.search(r"\b(show|plot|draw|visuali[sz]e)\b.*\b(correlation|relationship)\b", low):
        return "scatter"
    if not CHART_WORDS.search(low) and not re.search(r"\b(actual|real) (vs\.?|versus|against) predicted\b", low):
        return None
    if re.search(r"\b(actual|real)\b.*\b(vs\.?|versus|against|and)\b.*\bpredicted\b|\bpredicted\b.*\b(vs\.?|versus|"
                 r"against)\b.*\bactual\b", low):
        return "avp"
    for word, kind in (("pie", "pie"), ("donut", "pie"), ("doughnut", "pie"), ("histogram", "histogram"),
                       ("distribution", "histogram"), ("scatter", "scatter"), ("line", "line"), ("bar", "bar")):
        if re.search(rf"\b{word}", low):
            return kind
    if re.search(r"\bcorrelation|relationship\b|\bvs\.?\b|\bversus\b|\bagainst\b", low):
        return "scatter"
    if re.search(r"\b(over time|trend|monthly|weekly|daily|by (month|week|day|date|quarter))\b", low):
        return "line"
    return "auto"


def top_n(message: str) -> int | None:
    m = re.search(r"\btop\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b", message, re.I)
    if not m:
        return None
    w = m.group(1).lower()
    return int(w) if w.isdigit() else ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                       "ten"].index(w) + 1


def restyle(charts: list[dict], kind: str | None, message: str) -> tuple[list[dict], list[str]]:
    """Charts as the type the user asked for, where that type fits the data; notes say when it does not."""
    notes = []
    n = top_n(message)
    out = []
    for c in charts:
        c = dict(c)
        if n and c["type"] in ("bar", "pie") and len(c["data"]) > n:
            ordered = sorted(c["data"], key=lambda d: -(d.get(c["y"]) or 0))
            c["data"] = ordered[:n]
            if kind == "pie":      # a pie is parts of a whole: the rest stays visible as one slice
                rest = sum(d.get(c["y"]) or 0 for d in ordered[n:])
                c["data"] = c["data"] + [{c["x"]: f"all other ({len(ordered) - n})", c["y"]: rest}]
            c["title"] = f"Top {n}: {c['title']}"
        if kind == "pie" and c["type"] in ("bar", "pie"):
            vals = [d.get(c["y"]) for d in c["data"]]
            if c.get("additive") is False:
                notes.append("A pie chart shows parts of a whole; these values are averages or rates, which do not add "
                             "up to a total, so they are shown as a bar chart.")
            elif any(v is None or v < 0 for v in vals) or len(vals) > 12:
                notes.append("A pie chart needs a few non-negative parts of a whole; this result has "
                             + ("negative values" if any(v is not None and v < 0 for v in vals) else "too many parts")
                             + ", so it is shown as a bar chart.")
            else:
                c["type"] = "pie"
        elif kind == "line" and c["type"] == "bar":
            c["type"] = "line"
        elif kind == "bar" and c["type"] in ("line", "pie"):
            c["type"] = "bar"
        out.append(c)
    return out, notes


def scatter(df: pd.DataFrame, x: str, y: str, limit: int = 1000) -> dict:
    d = pd.DataFrame({x: pd.to_numeric(df[x], errors="coerce"), y: pd.to_numeric(df[y], errors="coerce")}).dropna()
    if len(d) < 3:
        raise ValueError(f"Fewer than 3 records have numbers in both {x} and {y}.")
    r = float(d[x].corr(d[y])) if d[x].std() > 0 and d[y].std() > 0 else None
    sample = d.sample(limit, random_state=0) if len(d) > limit else d
    return {"type": "scatter", "x": x, "y": y, "title": f"{y.replace('_', ' ')} vs {x.replace('_', ' ')}",
            "data": sample.to_dict(orient="records"), "r": r, "n": int(len(d)),
            "caption": f"{len(d)} records" + (f"; correlation r = {r:.2f}" if r is not None else "")
            + (f" (a random {limit} are plotted)" if len(d) > limit else "")
            + ". Correlation shows association, not cause."}


def histogram(df: pd.DataFrame, col: str) -> dict:
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(v) < 3:
        raise ValueError(f"Fewer than 3 numbers in {col}.")
    bins = int(min(20, max(4, round(np.sqrt(len(v))))))
    counts, edges = np.histogram(v, bins=bins)
    data = [{"range": f"{edges[i]:,.4g} – {edges[i + 1]:,.4g}", "start": float(edges[i]), "count": int(c)}
            for i, c in enumerate(counts)]
    return {"type": "histogram", "x": "range", "y": "count", "title": f"Distribution of {col.replace('_', ' ')}",
            "data": data, "caption": f"{len(v)} values in {bins} equal-width bins; median {float(v.median()):,.4g}."}


# ---------------------------------------------------------------- rankings --
_TOP = re.compile(r"\b(top|bottom|best|worst|highest|lowest)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                  re.I)


def rank_request(message: str, df: pd.DataFrame) -> dict | None:
    """'Show the top five products' -> rank the product column by the dataset's main measure."""
    from app.analyst.data import business_roles, find_column
    m = _TOP.search(message)
    if not m:
        return None
    w = m.group(2).lower()
    n = int(w) if w.isdigit() else ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].index(w) + 1
    rest = message[m.end():]
    # the thing ranked is what follows "top N" ("top 2 doctors by fee in Clinic A"), never a column the question only
    # uses to pick records (Clinic A names a value of `clinic`: that is a filter, not what is ranked)
    from app.analyst.calc import named_values
    filters = {c for c, _ in named_values(df, message)}
    cats = mentioned_order([c for c in find_column(df, rest) if kind_of(df[c]) == "categorical" and c not in filters],
                           rest)
    if not cats:
        return None
    nums = [c for c in find_column(df, message) if kind_of(df[c]) == "numeric"]
    roles = business_roles(df)["roles"]
    measure = nums[0] if nums else roles.get("revenue")
    return {"group": cats[0], "measure": measure, "n": n, "low": m.group(1).lower() in ("bottom", "worst", "lowest"),
            "assumed_measure": not nums}


def rank_groups(df: pd.DataFrame, group: str, measure: str, n: int, low: bool = False) -> dict:
    v = pd.to_numeric(df[measure], errors="coerce")
    agg = "mean" if RATE_LIKE.search(measure) else "sum"
    g = v.groupby(df[group].astype(str)).agg([agg, "count"]).rename(columns={agg: "value", "count": "records"})
    g = g.sort_values("value", ascending=low)
    total = float(v.sum()) if agg == "sum" else None
    rows = [{"group": k, "value": float(r["value"]), "records": int(r["records"]),
             "share": (float(r["value"]) / total if total else None)} for k, r in g.iterrows()]
    return {"group": group, "measure": measure, "aggregation": agg, "n": n, "low": low, "rows": rows[:n],
            "additive": agg == "sum",
            "groups_total": len(rows), "total": total}


def rank_text(r: dict, assumed: bool) -> str:
    from app.analyst.present import _cellv, pct
    lab, g = r["measure"].replace("_", " "), r["group"].replace("_", " ")
    word = "average" if r["aggregation"] == "mean" else "total"
    first = r["rows"][0]
    lines = [f"The {'bottom' if r['low'] else 'top'} {len(r['rows'])} of {r['groups_total']} {g} values by {word} {lab}: "
             f"**{first['group']}** {'is lowest' if r['low'] else 'leads'} with **{_cellv(first['value'])}**"
             + (f" ({pct(first['share'] * 100)} of the total)" if first.get("share") is not None else "") + ".", "",
             f"| rank | {g} | {word} {lab} | records |" + (" share |" if r["total"] else ""),
             "|---:|---|---:|---:|" + ("---:|" if r["total"] else "")]
    for i, x in enumerate(r["rows"], 1):
        lines.append(f"| {i} | {x['group']} | {_cellv(x['value'])} | {x['records']} |"
                     + (f" {pct(x['share'] * 100)} |" if r["total"] else ""))
    if assumed:
        lines.append(f"\n_Ranked by {word} **{lab}**, this dataset's main business measure; name another column to "
                     "rank by it instead._")
    return "\n".join(lines)



# ------------------------------------------------------- prediction inputs --
_PREDICT_ONE = re.compile(r"\bpredict\w*\b.*\b(for|if|with|who|whose|given)\b", re.I)
_TRAINING = re.compile(r"\b(train|retrain|split|seed|accuracy|importan\w*|which (inputs|features)|how (good|accurate))\b",
                       re.I)


def mentioned_order(cols: list[str], message: str) -> list[str]:
    """Columns in the order the message names them ("marketing spend and sales" -> spend first)."""
    low = message.lower()
    def pos(c: str) -> int:
        hits = [low.find(f) for f in {c.lower(), c.lower().replace("_", " ")} if f in low]
        return min(hits) if hits else len(low)
    return sorted(dict.fromkeys(cols), key=pos)


def _feature_named(feature: str, df: pd.DataFrame, message: str) -> str:
    """'given', 'unknown_value' (named, but not with a value the data has, e.g. "job level 2" when levels are
    Junior/Mid/Senior) or 'missing'."""
    low = message.lower()
    words = [w for w in re.split(r"[_\W]+", feature.lower()) if len(w) >= 3 and w not in ("pct", "num", "the")]
    named = any(re.search(rf"\b{re.escape(w)}", low) for w in words)
    if feature in df.columns and kind_of(df[feature]) == "categorical":
        vals = [str(v) for v in df[feature].dropna().unique()[:60]]
        if any(re.search(rf"(?<![\w-]){re.escape(v.lower())}(?![\w-])", low) for v in vals):
            return "given"
        return "unknown_value" if named else "missing"
    return "given" if named else "missing"


def missing_prediction_inputs(message: str, df: pd.DataFrame, memory: dict) -> str | None:
    """A request to predict one case with a trained model that leaves some of the model's inputs out: what is still
    needed, with the values seen in the data. None when the request is complete or is not such a request."""
    model = memory.get("model")
    if not model or not _PREDICT_ONE.search(message) or _TRAINING.search(message):
        return None
    target_words = [w for w in re.split(r"[_\W]+", model["target"].lower()) if len(w) >= 3]
    if not any(w in message.lower() for w in target_words) and not re.search(r"\bpredict\w*\s+(it|this|that)\b",
                                                                            message, re.I):
        return None
    feats = model["features"]
    state = {f: _feature_named(f, df, message) for f in feats}
    given = [f for f in feats if state[f] == "given"]
    unknown = [f for f in feats if state[f] == "unknown_value"]
    missing = [f for f in feats if state[f] == "missing"]
    if not missing and not unknown:
        return None
    def hint(f: str) -> str:
        if f not in df.columns:
            return f
        s = df[f].dropna()
        if kind_of(df[f]) == "numeric" and len(s):
            return f"**{f.replace('_', ' ')}** ({float(s.min()):,.4g} to {float(s.max()):,.4g} in the data)"
        vals = ", ".join(map(str, s.astype(str).value_counts().index[:5]))
        return f"**{f.replace('_', ' ')}** (e.g. {vals})"
    return (f"To predict **{model['target'].replace('_', ' ')}** for one case, the trained model needs a value for each "
            f"of its {len(feats)} inputs, and I won't guess the ones you leave out."
            + (f" You gave: {', '.join(g.replace('_', ' ') for g in given)}." if given else "")
            + ("\n\nNot a value the data has, so the model cannot use it — please use one of the listed values:\n"
               + "\n".join(f"- {hint(f)}" for f in unknown) if unknown else "")
            + ("\n\nStill needed:\n" + "\n".join(f"- {hint(f)}" for f in missing) if missing else "")
            + "\n\nFor example: *Predict " + model["target"].replace("_", " ") + " for " + ", ".join(
                f"{f.replace('_', ' ')} …" for f in feats[:4]) + (" …" if len(feats) > 4 else "") + "*")



# ------------------------------------------------------------ requested target --
_GENERIC_TARGET = {"value", "values", "number", "numbers", "figure", "figures", "result", "it", "this", "that", "the",
                   "next", "future", "amount", "data", "historical", "trend", "one", "his", "her", "their", "its", "my",
                   "our", "your", "a", "an", "s", "what", "will", "be", "expected", "likely", "total", "sum", "of"}
_INPUT_CLAUSE = re.compile(r"\b(?:using|based on|from|given|with|by using|via)\b[^,.;?!]*", re.I)
_UNIT_WORDS = re.compile(r"\bin\s+(minutes?|seconds?|hours?|days?|weeks?|thousands?|millions?)\b", re.I)


_WHEN = re.compile(r"\b(?:the\s+)?(?:next|coming|following|upcoming|this|last)\s+(?:day|week|month|quarter|year|period)(?:'s|s'|s)?\b"
                   r"|\b(?:tomorrow|today|yesterday)(?:'s)?\b|\b(?:daily|weekly|monthly|quarterly|yearly|annual)\b", re.I)
_LEVEL = re.compile(r"\b(?:across|among|over|for|of)\s+(?:all|every|each|the whole|the entire)\b[^,.;?!]*|\b(?:company|team|"
                    r"store|organi[sz]ation)[- ]wide\b|\bin total\b|\boverall\b", re.I)


def aggregation_level(message: str) -> str | None:
    """'all' when the question asks for a total over everyone/everything ("across all employees"), else None."""
    return "all" if _LEVEL.search(message or "") else None


def target_phrase(message: str, sc: dict | None) -> str:
    """The words naming what to predict ("the next day's total labor cost across all employees" -> "labor cost"): the
    noun phrase after the predicting verb, without time phrases (when it is predicted for), the aggregation level
    (across all ...), the scope's names, pronouns, generic words, a unit ("in minutes"), or a clause naming the
    inputs to use ("using his hourly rate" names an input, not the target)."""
    low = message.lower()
    for vals in (sc or {}).get("entities", {}).values():    # names go first: "A. Okafor" must not end the phrase
        for v in sorted(vals, key=len, reverse=True):
            low = re.sub(rf"(?<!\w){re.escape(v.lower())}('s)?(?!\w)", " ", low)
    low = _UNIT_WORDS.sub(" ", _INPUT_CLAUSE.sub(" ", low))
    low = _LEVEL.sub(" ", _WHEN.sub(" ", low))
    low = re.sub(r"\s+", " ", low)
    m = re.search(r"\b(?:predict|forecast|project|estimate)\w*\s+(.*?)(?=\s+(?:for|in|on|over|during|at|by|"
                  r"(?:the\s+)?next\s+(?:day|week|month|quarter|year|period)s?|this|of the|per|if|when|assuming)\b|"
                  r"[?.!,;]|$)", low)
    if not m:
        m = re.search(r"\bwhat will\s+(.*?)\s+be\b", low)
    phrase = m.group(1) if m else ""
    for vals in (sc or {}).get("entities", {}).values():
        for v in vals:
            for w in re.findall(r"[a-z0-9]+", v.lower()):
                phrase = re.sub(rf"\b{re.escape(w)}('s)?\b", " ", phrase)
    words = [w for w in re.findall(r"[a-z]+", phrase) if w not in _GENERIC_TARGET]
    return " ".join(words)


def requested_unit(message: str) -> str | None:
    m = _UNIT_WORDS.search(message)
    return m.group(1).lower() if m else None


def _name_words(name: str) -> set[str]:
    out = set()
    for w in re.split(r"[_\W]+", str(name).lower()):
        if w:
            out |= {w, w.rstrip("s")}
    return out


def forecast_target(message: str, df: pd.DataFrame, memory: dict, sc: dict | None) -> dict:
    """What a prediction question asks for, resolved from its target phrase alone:
      {"kind": "column", "column"}           a column of the data named by the phrase
      {"kind": "ambiguous", "options"}       the phrase fits several columns (ask which)
      {"kind": "measure"}                    the measure computed in the previous answer
      {"kind": "derived", "phrase"}          a metric that is not a column (e.g. cost = rate x hours): derive it
      {"kind": "unknown", "phrase": ""}      nothing named and nothing computed before
    An input named elsewhere in the question is never taken as the target."""
    phrase = target_phrase(message, sc)
    want = {w.rstrip("s") for w in phrase.split()}
    last = memory.get("last_calc")
    measure = None
    if last:
        sums = [n for n, a in last["spec"]["aggregate"].items() if re.match(r"\s*sum\s*\(", a)]
        measure = sums[0] if sums else None
    if not want:
        return {"kind": "measure", "phrase": measure} if measure else {"kind": "unknown", "phrase": ""}
    nums = [c for c in df.columns if kind_of(df[c]) == "numeric"]
    fits = [c for c in nums if want <= _name_words(c)]
    unit = requested_unit(message)
    if len(fits) == 1 and not unit:
        return {"kind": "column", "column": fits[0], "phrase": phrase}
    if len(fits) > 1:
        exact = [c for c in fits if _name_words(c) - want <= {""}]
        if len(exact) == 1 and not unit:
            return {"kind": "column", "column": exact[0], "phrase": phrase}
        return {"kind": "ambiguous", "options": fits, "phrase": phrase}
    if measure and want <= _name_words(measure) and not unit:
        return {"kind": "measure", "phrase": phrase}
    return {"kind": "derived", "phrase": phrase + (f" in {unit}" if unit else "")}


def derive_target(phrase: str, df: pd.DataFrame) -> tuple[dict | None, list[dict], str]:
    """A per-record definition of a metric that is not a column, written by the local model through the calculator's
    validated planner (same checks as any calculation: real columns only, no invented constants, a unit factor only
    when the unit is named). Accepted only if the result is named for the requested metric and uses at least one
    column. Returns (spec, trace, problem)."""
    from app.analyst.calc import model_plan
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    ask = f"Calculate the total {phrase}" + (f" by {dates[0]}" if dates else "") + " for every record."
    spec, trace = model_plan(ask, df, {})
    if spec.get("type") != "calc":
        return None, trace, spec.get("reason") or spec.get("question") or "the model could not define it"
    sums = [(n, a) for n, a in spec["aggregate"].items() if re.match(r"\s*sum\s*\(", a)]
    if not sums:
        return None, trace, "the definition has no total to forecast"
    name, agg = sums[0]
    base = _UNIT_WORDS.sub(" ", phrase).split()
    want = {w.rstrip("s") for w in base}
    names = _name_words(name) | {w for k in spec["derive"] for w in _name_words(k)}
    if not want <= names:
        return None, trace, f"the model defined '{name}', not the requested '{phrase}'"
    used = agg + " " + " ".join(spec["derive"].values())
    if not [c for c in map(str, df.columns) if re.search(rf"\b{re.escape(c)}\b", used)]:
        return None, trace, "the definition does not use any column of the data"
    return {**spec, "exclude": [], "group_by": None, "aggregate": {name: agg}}, trace, ""


def describe_definition(spec: dict) -> str:
    from app.analyst.present import readable
    parts = [f"{k.replace('_', ' ')} = {readable(v)}" for k, v in spec["derive"].items()]
    parts += [f"{k.replace('_', ' ')} = {readable(v)}" for k, v in spec["aggregate"].items()
              if not re.fullmatch(r"\s*sum\s*\(\s*(" + "|".join(map(re.escape, spec["derive"])) + r")\s*\)\s*", v)
              or not spec["derive"]]
    return "; ".join(parts) + " (summed per period)"
