"""Model-based decomposition of a whole request into deliverables, grounded and validated against the dataset.

The model reads the COMPLETE message plus a compact schema card and returns a task list as JSON::

    {"tasks": [{"do": "stat", "stat": "avg", "col": "hours_worked", "text": "the average hours"},
               {"do": "forecast", "col": "hours_worked", "where": {"employee": "Rahul"}, "horizon": 1, ...}]}

The model never produces numbers that answer the question: it only says WHAT to compute. Every field is then
grounded deterministically (columns must exist and have the right kind, filter values must occur in the data,
numbers must be finite, references must point at earlier tasks). An invalid plan gets one corrective retry with
the exact problems listed; a plan that is still invalid is discarded and the rule reader
(`deliverables.extract`) is used instead -- the model is never trusted on an unchecked field.

Execution, validation, completeness and composition are the same as for the rule reader (`compound.py`).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re

import pandas as pd

from app.analyst.deliverables import Deliverable, finalize
from app.analyst.data import kind_of
from app.analyst.semantics import profile

logger = logging.getLogger(__name__)

KINDS = {"summary": "summary", "stat": "stat", "rank": "ranking", "breakdown": "breakdown", "trend": "series",
         "target": "target_check", "pace": "pace", "forecast": "forecast", "chart": "chart",
         "compare": "compare", "growth": "growth", "correlation": "correlation",
         "unsupported": "unparsed", "clarify": "clarify"}
STATS = {"avg", "sum", "median", "var", "std", "min", "max", "count"}
PERIODS = {"D", "W", "M", "Q", "Y"}
FIELDS = ("do", "text", "stat", "col", "col2", "groups", "where", "when", "type", "by", "n", "dir", "every", "target", "per", "unit", "pace_per",
          "horizon", "if", "of", "reason")

SYSTEM = """You turn a user's request about a dataset into a complete list of analysis tasks, as JSON.
Read the WHOLE message. Every thing the user asks for is one task, in the order asked. Never drop a task.

Reply with only: {"tasks": [ ... ]}
Each task has "do" and "text" (the exact words of the message that ask for it), plus only the fields it needs:
- summary: an overview of the whole dataset.
- stat: one statistic of one numeric column. "stat": avg|sum|median|var|std|min|max|count, "col".
- rank: top / bottom groups. "by": category column, "col": numeric column, "stat" (default sum), "n", "dir": desc|asc.
- breakdown: a numeric column per group (or a comparison of groups). "by", "col", "stat".
- compare: a numeric column compared between groups (difference, % difference, whether it is meaningful).
  "col", "by", "groups": the named groups (omit for all groups), "stat" (default avg).
- growth: how a numeric column changed over time (growth rate, greatest decline). "col", optional "by" and "groups".
- correlation: association between two numeric columns. "col", "col2".
- trend: a numeric column over time. "col", "every": D|W|M|Q|Y if the user names a granularity.
- target: is a value on track for a target? "col", "target" (number), "per": D|W|M|Q|Y, "unit".
- pace: the average needed per "pace_per" (D|W|M) to reach a target. "col", "target", "per", "unit".
- forecast: predict future values. "col", "horizon" (number of periods, default 1), and "target"/"per" if one is set.
- chart: a graph. "of": the index (0-based) of the task it draws; "type": bar|line|pie|scatter only if the user names one.
- unsupported: something this data cannot answer (a column that does not exist, no dates for a forecast ...). "reason".
- clarify: something too ambiguous to compute without asking. "reason": the question to ask.
Scope: "where": {"column": "value"} or {"column": ["v1", "v2"]} ONLY for the task whose words name that person,
group or filter. A name in one task never applies to another task. "he", "she", "it", "them" refer to the subject
named before; "for X only: A and B" applies X to A and B. Dataset-wide tasks have no "where".
Time windows: a task about part of the time range gets "when": {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"} (inclusive;
omit "from" or "to" for an open end). Work it out from the words ("in 2024", "in Q1 2025", "since March 2025") and,
for "the last 3 months", count back from the latest date shown in the schema. No window = all dates.
Conditions: a task asked only "if yes" / "if not" after a target task gets "if": "yes" or "no".
Use exact column names and values from the schema. If the user names a measure the schema does not have, use
unsupported -- never substitute another column."""


# ----------------------------------------------------------------------------------------------- the prompt --
def schema_card(df: pd.DataFrame, max_values: int = 8) -> str:
    prof = profile(df)
    lines = []
    for c in df.columns:
        s = df[c]
        if c in prof["dates"]:
            d = pd.to_datetime(s, errors="coerce").dropna()
            info = f"date, {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}" if len(d) else "date"
            lines.append(f"- {c}: {info}")
        elif c in prof["measures"]:
            lines.append(f"- {c}: number")
        elif kind_of(s) == "categorical" or s.dtype == object:
            vals = s.dropna().astype(str).value_counts().index.tolist()
            more = f", ... ({len(vals)} values)" if len(vals) > max_values else ""
            lines.append(f"- {c}: category: " + ", ".join(vals[:max_values]) + more)
        else:
            lines.append(f"- {c}: {kind_of(s)}")
    return f"{len(df):,} rows. Columns:\n" + "\n".join(lines)


def build_messages(message: str, df: pd.DataFrame, previous: str | None = None) -> list[dict]:
    user = f"Dataset:\n{schema_card(df)}\n" + (f"Previous request: {previous}\n" if previous else "") + f"Request: {message}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def render(tasks: list[dict]) -> str:
    """The canonical reply for a task list (training targets use exactly this form)."""
    clean = [{k: t[k] for k in FIELDS if k in t and t[k] is not None} for t in tasks]
    return json.dumps({"tasks": clean}, ensure_ascii=False, separators=(", ", ": "))


# ----------------------------------------------------------------------------------------------- parsing --
def parse(text: str) -> tuple[list | None, str | None]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    start = text.find("{")
    if start < 0:
        return None, "the reply has no JSON object"
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        return None, f"the JSON is malformed ({e.msg})"
    if not isinstance(obj, dict) or not isinstance(obj.get("tasks"), list) or not obj["tasks"]:
        return None, 'the reply must be {"tasks": [...]} with at least one task'
    if not all(isinstance(t, dict) for t in obj["tasks"]):
        return None, "every task must be an object"
    return obj["tasks"], None


# ----------------------------------------------------------------------------------------------- grounding --
def _col(name, df, numeric: bool | None = None, prof=None):
    if not isinstance(name, str):
        return None
    cols = {str(c).lower(): c for c in df.columns}
    c = cols.get(name.strip().lower()) or cols.get(name.strip().lower().replace(" ", "_"))
    if c is None:
        return None
    if numeric is True and c not in prof["measures"]:
        return None
    return c


def _value(col, v, df):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = str(int(v)) if float(v).is_integer() else str(v)
    if not isinstance(v, str):
        return None
    vals = {str(x).strip().lower(): x for x in df[col].dropna().unique()}
    hit = vals.get(v.strip().lower())
    return None if hit is None else (str(hit) if not isinstance(hit, str) else hit)


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        try:
            x = float(x)
        except ValueError:
            return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def ground(tasks: list[dict], df: pd.DataFrame, message: str) -> tuple[list[Deliverable], list[str]]:
    """Deliverables for a model task list, or the list of problems (each one names the task and the field)."""
    prof = profile(df)
    errs, out, idmap = [], [], {}
    low = message.lower()
    for i, t in enumerate(tasks):
        where = f"task {i}"
        do = t.get("do")
        if do not in KINDS:
            errs.append(f"{where}: \"do\" must be one of {', '.join(KINDS)}")
            continue
        kind = KINDS[do]
        text = t.get("text") if isinstance(t.get("text"), str) and t.get("text").strip() else message
        d = Deliverable(id=len(out) + 1, kind=kind, clause=text, order=i)
        pos = low.find(text.lower()[:40]) if text is not message else -1
        d.order = pos if pos >= 0 else 100_000 + i      # the user's order (where the words are), else the model's
        if kind in ("compare", "growth") and isinstance(t.get("where"), dict):
            # "where region in [West, East]" for a comparison names the groups, not a filter
            w0 = dict(t["where"])
            for wc, wv in list(w0.items()):
                if isinstance(wv, list) and (t.get("by") in (None, wc)):
                    t = {**t, "by": wc, "groups": t.get("groups") or wv}
                    w0.pop(wc)
            t = {**t, "where": w0}
        if kind in ("stat", "ranking", "breakdown", "series", "target_check", "pace", "forecast", "compare", "growth",
                    "correlation"):
            col = t.get("col")
            if col is None and kind in ("target_check", "pace", "forecast", "series") and len(prof["measures"]) == 1:
                col = prof["measures"][0]
            c = _col(col, df, True, prof) if col is not None else None
            if col is not None and c is None:
                errs.append(f"{where}: \"col\" {col!r} is not a numeric column of the dataset")
                continue
            if c is None and kind not in ("stat",) and not (kind == "stat" and t.get("stat") == "count"):
                errs.append(f"{where}: \"col\" is required for {do}")
                continue
            if c is None and kind == "stat" and t.get("stat") != "count":
                errs.append(f"{where}: \"col\" is required for stat")
                continue
            d.measures = [c] if c is not None else []
        if kind == "correlation":
            c2 = _col(t.get("col2"), df, True, prof)
            if c2 is None or c2 == d.measures[0]:
                errs.append(f"{where}: \"col2\" must be a second numeric column")
                continue
            d.measures = d.measures + [c2]
        if kind in ("stat", "ranking", "breakdown", "compare"):
            st = t.get("stat") or ("sum" if kind == "ranking" else "avg")
            if st not in STATS:
                errs.append(f"{where}: \"stat\" must be one of {', '.join(sorted(STATS))}")
                continue
            d.stat = st
        if kind in ("ranking", "breakdown", "compare") or (kind == "growth" and t.get("by") is not None):
            by = _col(t.get("by"), df, False, prof)
            if by is None or by in prof["measures"]:
                errs.append(f"{where}: \"by\" must be a category column")
                continue
            d.group_by = by
            if kind in ("compare", "growth") and t.get("groups"):
                gs = t["groups"] if isinstance(t["groups"], list) else [t["groups"]]
                got = [_value(by, g, df) for g in gs]
                if any(g is None for g in got):
                    errs.append(f"{where}: {by} has no value {next(g0 for g0, g in zip(gs, got) if g is None)!r}")
                    continue
                d.groups = got
            if kind == "ranking":
                n = _num(t.get("n", 5))
                d.top = int(n) if n and n >= 1 else 5
                d.order_dir = "asc" if str(t.get("dir", "desc")).lower().startswith("asc") else "desc"
        w = t.get("where") or {}
        if not isinstance(w, dict):
            errs.append(f"{where}: \"where\" must be an object of column: value")
            continue
        ents, bad = {}, False
        for col, val in w.items():
            c = _col(col, df, False, prof)
            if c is None:
                errs.append(f"{where}: filter column {col!r} does not exist")
                bad = True
                break
            vals = val if isinstance(val, list) else [val]
            got = [_value(c, v, df) for v in vals]
            if not vals or any(g is None for g in got):
                errs.append(f"{where}: {c} has no value {next(v for v, g in zip(vals, got) if g is None)!r}" if vals
                            else f"{where}: empty filter")
                bad = True
                break
            ents[c] = got if len(got) > 1 else got[0]
        if bad:
            continue
        d.entities = ents
        if ents:
            named = any(str(v).lower() in text.lower() for vs in ents.values() for v in (vs if isinstance(vs, list) else [vs]))
            d.entity_from_context = not named
        if kind in ("series", "growth") and t.get("every") is not None:
            if t["every"] not in PERIODS:
                errs.append(f"{where}: \"every\" must be one of D W M Q Y")
                continue
            d.period = t["every"]
        if kind in ("target_check", "pace", "forecast"):
            if t.get("target") is not None:
                tv = _num(t["target"])
                if tv is None:
                    errs.append(f"{where}: \"target\" must be a number")
                    continue
                d.target = tv
            elif kind in ("target_check", "pace"):
                errs.append(f"{where}: \"target\" is required for {do}")
                continue
            if t.get("per") is not None:
                if t["per"] not in PERIODS:
                    errs.append(f"{where}: \"per\" must be one of D W M Q Y")
                    continue
                d.period = t["per"]
            d.target_unit = t.get("unit") if isinstance(t.get("unit"), str) else None
        if kind == "pace":
            pu = t.get("pace_per")
            if pu is not None and pu not in ("D", "W", "M"):
                errs.append(f"{where}: \"pace_per\" must be D, W or M")
                continue
            d.pace_unit = {"D": "day", "W": "week", "M": "month"}.get(pu)
        if kind == "forecast":
            h = _num(t.get("horizon", 1))
            if h is None or h < 1 or h > 36:
                errs.append(f"{where}: \"horizon\" must be a number of periods between 1 and 36")
                continue
            d.horizon = int(h)
            d.output = "table"
        if kind == "chart":
            of = t.get("of")
            if not isinstance(of, int) or not (0 <= of < len(tasks)) or of == i:
                errs.append(f"{where}: \"of\" must be the index of another task")
                continue
            d.chart_for = of                         # re-pointed to the deliverable id below
            d.output = "chart"
        if kind in ("unparsed", "clarify"):
            r = t.get("reason")
            d.detail = r if isinstance(r, str) and r.strip() else ("this part cannot be answered from this data"
                                                                   if kind == "unparsed" else "this part is ambiguous")
        if t.get("when") is not None:
            w_ = t["when"]
            if not isinstance(w_, dict) or not (w_.get("from") or w_.get("to")):
                errs.append(f"{where}: \"when\" must be {{\"from\": \"YYYY-MM-DD\", \"to\": \"YYYY-MM-DD\"}}")
                continue
            if not prof["dates"]:
                errs.append(f"{where}: the dataset has no date column for \"when\"")
                continue
            iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            if any(w_.get(k) and not iso.match(str(w_[k])) for k in ("from", "to")):
                errs.append(f"{where}: \"when\" dates must be YYYY-MM-DD")
                continue
            try:
                f_ = pd.Timestamp(w_["from"]) if w_.get("from") else None
                t_ = pd.Timestamp(w_["to"]) if w_.get("to") else None
            except (ValueError, TypeError):
                errs.append(f"{where}: \"when\" dates must be YYYY-MM-DD")
                continue
            if f_ is not None and t_ is not None and f_ > t_:
                errs.append(f"{where}: \"when\" from is after to")
                continue
            dmin = pd.to_datetime(df[prof["dates"][0]], errors="coerce").min()
            dmax = pd.to_datetime(df[prof["dates"][0]], errors="coerce").max()
            if (f_ is not None and f_ > dmax) or (t_ is not None and t_ < dmin):
                errs.append(f"{where}: the window {w_} is outside the data ({dmin:%Y-%m-%d} to {dmax:%Y-%m-%d})")
                continue
            d.date_from = f"{f_:%Y-%m-%d}" if f_ is not None else None
            d.date_to = f"{t_:%Y-%m-%d}" if t_ is not None else None
        if kind == "chart" and t.get("type") is not None:
            if t["type"] not in ("bar", "line", "pie", "scatter"):
                errs.append(f"{where}: chart \"type\" must be bar, line, pie or scatter")
                continue
            d.chart_type = t["type"]
        if t.get("if") is not None:
            if t["if"] not in ("yes", "no"):
                errs.append(f"{where}: \"if\" must be \"yes\" or \"no\"")
                continue
            d.condition = t["if"]
        idmap[i] = d.id
        out.append(d)
    if errs:
        return [], errs
    for d in out:
        if d.kind == "chart":
            subj_id = idmap.get(d.chart_for)
            subj = next((x for x in out if x.id == subj_id), None)
            if subj is not None and subj.kind == "series":
                # the request also forecasts this measure for this scope: its chart already draws the whole history,
                # the forecast and any target line, so "visualise the result" is that chart
                fc = next((x for x in out if x.kind == "forecast" and x.measures == subj.measures
                           and x.entities == subj.entities), None)
                if fc is not None:
                    subj = fc
            if subj is not None and subj.kind not in DRAWABLE:
                # a chart "of" a verdict or a single number (a target check, a stat): what can be drawn is the
                # drawable result of the same measure and scope -- the forecast the verdict rests on, or its series
                alt = _drawable_for(subj, out)
                if alt is not None:
                    subj = alt
            if subj is not None:
                subj = _chart_scope(d, subj, out, df)
            if subj is None or subj.kind in ("chart", "unparsed", "clarify", "summary"):
                errs.append(f"chart task: \"of\" must point at a task with data to draw")
                continue
            d.chart_for, d.depends_on = subj.id, [subj.id]
            d.entities, d.measures = dict(subj.entities), list(subj.measures)
            subj.output = "chart"
    if errs:
        return [], errs
    _colon_scope(out, message)
    # one statistic task whose own words name several ("the mean plus the variance of hours"): the model merged them,
    # every one named is a deliverable of the same measure and scope
    for d in list(out):
        if d.kind != "stat" or not d.stat or not _JOIN.search(d.clause or ""):
            continue
        named = [s for w, s in _STAT_WORDS if re.search(rf"\b{w}\b", d.clause, re.I)]
        if d.stat not in named:
            continue
        have = {x.stat for x in out if x.kind == "stat" and x.measures == d.measures and x.entities == d.entities
                and (x.date_from, x.date_to) == (d.date_from, d.date_to)}
        for s in dict.fromkeys(named):
            if s not in have:
                have.add(s)
                out.append(Deliverable(id=max(x.id for x in out) + 1, kind="stat", clause=d.clause, order=d.order,
                                       stat=s, measures=list(d.measures), entities=dict(d.entities),
                                       date_from=d.date_from, date_to=d.date_to, output=d.output))
    # a drawable task whose own words ask for a picture ("chart Leo's monthly hours") but that no chart task draws:
    # the model left the chart out, the request did not
    drawn = {d.chart_for for d in out if d.kind == "chart"}
    for d in list(out):
        if d.kind in DRAWABLE and d.id not in drawn and _CHART_WORDS.search(d.clause or ""):
            d.output = "chart"
            out.append(Deliverable(id=max(x.id for x in out) + 1, kind="chart", clause=d.clause, order=d.order,
                                   chart_for=d.id, depends_on=[d.id], entities=dict(d.entities),
                                   measures=list(d.measures), output="chart"))
    return finalize(out), []


DRAWABLE = ("forecast", "series", "growth", "compare", "breakdown", "ranking", "correlation")
_JOIN = re.compile(r"\band\b|\bplus\b|&|,|\bas well as\b|/", re.I)
_STAT_WORDS = [("average", "avg"), ("mean", "avg"), ("variance", "var"), ("standard deviation", "std"), ("std", "std"),
               ("median", "median"), ("total", "sum"), ("sum", "sum"), ("maximum", "max"), ("max", "max"),
               ("highest", "max"), ("minimum", "min"), ("min", "min"), ("lowest", "min")]
_CHART_WORDS = re.compile(r"\b(graph|chart|plot|visuali[sz]e|draw)\b", re.I)


_SCOPE_HEAD = re.compile(r"\b(?:for|at|in|within|among)\s+([^:.?!\n]{1,60}?)\s*(?:\bonly\b)?\s*:", re.I)


def _colon_scope(out: list, message: str) -> None:
    """'Then for Lakeside only: the total orders and the maximum gross sales' -- a heading ending in a colon scopes
    every part after it in the same sentence. A part there with no scope of its own takes the scope another part of
    that span carries, when every value of it is named in the heading."""
    for sent in re.finditer(r"[^.?!\n]+[.?!\n]?", message or ""):
        s = sent.group()
        h = _SCOPE_HEAD.search(s)
        if not h:
            continue
        head = h.group(1).lower()
        lo, hi = sent.start() + h.end(), sent.end()
        span = [d for d in out if d.clause and lo <= (message.find(d.clause) if message.find(d.clause) >= 0 else -1) < hi]
        scoped = next((d.entities for d in span if d.entities and all(
            str(v).lower() in head for vals in d.entities.values() for v in (vals if isinstance(vals, list) else [vals]))), None)
        if not scoped:
            continue
        for d in span:
            if not d.entities and d.kind not in ("summary", "unparsed", "clarify"):
                d.entities = dict(scoped)


def _chart_scope(chart: Deliverable, subj: Deliverable, out: list, df: pd.DataFrame) -> Deliverable:
    """A chart's own words name ONE entity ("chart Leo's monthly hours") that the task it points at does not scope to
    (a ranking of all consultants): what is asked to be drawn is that entity's result, so the chart is re-pointed to a
    drawable task of that scope -- one already in the plan, else the rule reader's for the chart's own clause (added to
    the plan). Otherwise the subject is kept."""
    from app.analyst import deliverables as D
    if not chart.clause:
        return subj
    named = D._entities(chart.clause, D.entity_index(df))
    if len(named) != 1:
        return subj
    (col, val), = named.items()
    if isinstance(val, list):
        return subj
    have = subj.entities.get(col)
    if have == val or (isinstance(have, list) and val in have) or (subj.group_by == col and val in (subj.groups or [])):
        return subj
    scoped = {col: val}
    hit = next((x for x in out if x.kind in DRAWABLE and x.entities == scoped and
                (not subj.measures or set(subj.measures) & set(x.measures))), None)
    if hit is not None:
        return hit
    try:
        cand = D.extract(chart.clause, df)
    except Exception:  # noqa: BLE001 - best effort; the model's subject stands
        return subj
    new = next((x for x in cand if x.kind in DRAWABLE and x.entities == scoped and
                (not subj.measures or set(subj.measures) & set(x.measures))), None)
    if new is None:
        return subj
    new.id, new.order, new.clause = max(x.id for x in out) + 1, chart.order, chart.clause
    new.depends_on, new.chart_for = [], None
    out.append(new)
    return new


def _drawable_for(subj: Deliverable, out: list) -> Deliverable | None:
    """The drawable task with the subject's measure and scope: a forecast first (its chart carries the target line),
    then a series, then any other drawable result of that measure."""
    same = [x for x in out if x.kind in DRAWABLE and x.id != subj.id and
            (not subj.measures or set(subj.measures) & set(x.measures)) and x.entities == subj.entities]
    for kind in DRAWABLE:
        hit = next((x for x in same if x.kind == kind), None)
        if hit is not None:
            return hit
    return None


# ----------------------------------------------------------------------------------------------- the model --
def model_chat(messages: list[dict], max_tokens: int = 900) -> str:
    """The decomposition model. By default the planner's server; ANALYST_DECOMPOSER_URL points it at a separate
    server (a decomposition-only adapter), so compound decomposition and ordinary planning use different weights."""
    from app.analyst import planner
    url = os.getenv("ANALYST_DECOMPOSER_URL")
    if not url:
        return planner.model_chat(messages, max_tokens=max_tokens)
    from openai import OpenAI
    client = OpenAI(base_url=url, api_key=os.getenv("ANALYST_MODEL_KEY", "local"), timeout=planner.MODEL_TIMEOUT)
    r = client.chat.completions.create(model=os.getenv("ANALYST_DECOMPOSER_MODEL", "default_model"),
                                       messages=messages, temperature=0.0, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def enabled() -> bool:
    return os.getenv("ANALYST_DECOMPOSER", "model") != "off"


_CLAUSE = re.compile(r"[.;?!\n]+|,|\s+(?:and|also|then|plus)\s+", re.I)
_CUE = re.compile(r"\b(average|mean|median|variance|standard deviation|std|total|sum|max\w*|min\w*|highest|lowest|top|bottom|"
                  r"rank\w*|compar\w*|forecast\w*|predict\w*|project\w*|plot|chart|graph|visuali[sz]e|draw|trend|growth|"
                  r"grew|declin\w*|target|goal|quota|budget|benchmark|summary|summari[sz]e|overview|describe|correlat\w*|"
                  r"related|relationship|how many|how much|which|breakdown|split|per (?:day|week|month)|pace|on track)\b", re.I)


def task_clauses(message: str) -> int:
    """How many separate clauses of the message ask for something analytical."""
    return sum(1 for c in _CLAUSE.split(message or "") if c and _CUE.search(c))


def worth_asking(message: str, rule_ds: list, followup: bool) -> bool:
    """Ask the model only when the request holds at least two separate analytical clauses (a single question --
    including a prediction with several inputs -- keeps Silt's specialised handlers and costs no model call).
    Follow-ups are never sent here first: Silt's own follow-up handling runs before the model is consulted."""
    if followup:
        return False
    if len([d for d in rule_ds if d.kind != "chart"]) >= 2 and task_clauses(message) >= 2:
        return True
    return task_clauses(message) >= 2


_STOPW = {"the", "a", "an", "of", "for", "and", "its", "it", "their", "what", "whats", "what's", "is", "are", "me", "give",
          "please", "then", "also", "finally", "lastly", "first", "after", "that", "this", "to", "in", "on", "by", "with"}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9_']+", (text or "").lower()) if w not in _STOPW}


def cover_omissions(ds: list, message: str, df: pd.DataFrame) -> list[str]:
    """Completeness check of a model decomposition: every clause of the request that asks for something analytical
    must be covered by some task's quoted words. A clause none covers ("Finally, the total billable hours", "... and
    its standard deviation") gets the rule reader's deliverables for that clause, read from the WHOLE message so its
    references resolve. Returns the clauses that were added (for the trace)."""
    from app.analyst import deliverables as D
    msg = message or ""
    low = msg.lower()
    spans, loose = [], []
    for d in ds:
        i = low.find((d.clause or "").strip().lower()) if (d.clause or "").strip() else -1
        if i >= 0:
            spans.append((i, i + len(d.clause.strip())))   # the model quotes the message: where its words are
        else:
            loose.append(_words(d.clause))                 # a paraphrased quote: matched by its words instead
    missing, start = [], 0
    for sep in list(_CLAUSE.finditer(msg)) + [None]:
        end = sep.start() if sep else len(msg)
        c = msg[start:end]
        if sep:
            start = sep.end()
        if not c.strip() or not _CUE.search(c):
            continue
        cs, ce = end - len(c), end
        if any(a < ce and b > cs for a, b in spans):
            continue
        w = _words(c)
        if not w or any(len(w & lw) >= max(1, (2 * len(w) + 2) // 3) for lw in loose):
            continue
        missing.append(c.strip())
    if not missing:
        return []
    try:
        rules = D.extract(message, df)
    except Exception:  # noqa: BLE001 - best effort: the model's own tasks stand
        return []
    added = []
    key = lambda d: (d.kind, d.stat, tuple(d.measures), json.dumps(d.entities, sort_keys=True, default=str),  # noqa: E731
                     d.date_from, d.date_to)
    have = {key(d) for d in ds}
    for c in missing:
        cw = _words(c)
        for r in rules:
            if r.kind in ("chart", "summary") or not r.measures or key(r) in have:
                continue
            if len(_words(r.clause) & cw) >= max(1, (2 * len(cw) + 2) // 3):
                r.id = max([d.id for d in ds] + [0]) + 1
                r.depends_on, r.chart_for = [], None
                pos = (message or "").find(c)
                before = [d.order for d in ds if d.clause and 0 <= (message or "").find(d.clause) < pos]
                r.order = (max(before) if before else -1) + 0.5     # between the tasks around it in the message
                ds.append(r)
                have.add(key(r))
                added.append(c)
    if added:
        ds[:] = finalize(ds)
    return added


def decompose(message: str, df: pd.DataFrame, previous: str | None = None, chat=None, trace: dict | None = None
              ) -> list[Deliverable] | None:
    """Grounded deliverables from the model, or None (the caller falls back to the rule reader). One corrective
    retry: the model sees its own reply and the exact problems."""
    chat = chat or model_chat
    msgs = build_messages(message, df, previous)
    trace = trace if trace is not None else {}
    for attempt in (1, 2):
        try:
            reply = chat(msgs)
        except Exception as e:  # noqa: BLE001 - the model is optional; the rule reader answers without it
            trace["error"] = f"{type(e).__name__}: {e}"
            return None
        tasks, err = parse(reply)
        problems = [err] if err else []
        ds = []
        if tasks is not None:
            ds, problems = ground(tasks, df, message)
        trace.setdefault("attempts", []).append({"reply": reply[:4000], "problems": problems})
        if not problems:
            trace["source"] = "model" if attempt == 1 else "model_retry"
            added = cover_omissions(ds, message, df)
            if added:
                trace["added_from_rules"] = added
            return ds
        msgs = msgs + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": "That plan cannot be used: " + "; ".join(problems[:6]) +
                        ". Reply with the corrected complete JSON task list only."}]
    trace["source"] = "rejected"
    return None
