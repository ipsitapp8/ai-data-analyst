"""Model-assisted analytical planning: the local model writes an analysis PROGRAM, Silt validates and runs it.

    message + conversation state
        -> grounding: the columns, their semantic roles and safe summaries, values the message names, date tokens,
           ambiguities, and the analyses / scenarios / findings already in the conversation
        -> the local model returns ONE JSON object (a program, a clarification, "unsupported", or a reference to an
           earlier analysis)                                          -- the model plans; it never computes
        -> strict parsing and checking: known operations and arguments only, columns and values that exist,
           semantically sensible use (no sum of a rate, no average of an identifier, units that agree), steps that
           refer to earlier steps, ambiguous concepts asked about rather than guessed
        -> at most one repair: the model is shown exactly why its plan was rejected
        -> app/analyst/program.py runs the steps deterministically on the real records
        -> the answer is written from the computed results only, with every step and its provenance

The mlx_lm server has no JSON mode or constrained decoding (checked: mlx_lm 0.32.2 accepts sampling, stop words and
logits settings only), so the contract is enforced after generation. Anything that fails is dropped and the
deterministic paths answer instead: a model outage, malformed output, unknown columns or operations, a timeout or an
incomplete plan never reaches the executor and never replaces a deterministic answer.

Nothing here knows a dataset or a question.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import pandas as pd

from app.analyst import intent
from app.analyst import program as P
from app.analyst import semantics
from app.analyst.data import kind_of

logger = logging.getLogger(__name__)

PLAN_TIMEOUT = float(os.getenv("ANALYST_PLAN_TIMEOUT", "20"))      # one model call
PLAN_BUDGET = float(os.getenv("ANALYST_PLAN_BUDGET", "30"))        # all planning calls for one message
# The planner's model endpoint. By default the app's own server (Qwen3-4B + the active v1 adapter). It can point at
# another local OpenAI-compatible server -- e.g. the same base model without the adapter, which plans programs more
# reliably (see the planning evaluation) -- without changing what the rest of the app uses.
PLANNER_URL = os.getenv("ANALYST_PLANNER_URL")
PLANNER_MODEL = os.getenv("ANALYST_PLANNER_MODEL")
MAX_TOKENS = 700
MAX_STEPS = 14
MAX_ATTEMPTS = 2                     # the first plan and one repair

ACTIONS = {"program", "clarify", "unsupported", "rerun", "compare_scenarios", "finding"}
# every argument an operation accepts; anything else is rejected rather than ignored
OP_ARGS = {
    "select": ({"from"}, {"entities", "except", "period", "where"}),
    "derive": ({"name", "formula"}, {"from", "unit"}),
    "aggregate": ({"fn"}, {"from", "column", "by", "per"}),
    "ratio": ({"a", "b"}, {"broadcast"}),
    "rank": ({"from"}, {"n", "order"}),
    "project": ({"column", "days"}, {"from", "history_start", "history_end"}),
    "compare": ({"a"}, {"b", "target", "base", "days", "a_label", "b_label"}),
    "scenario": ({"changes"}, {"from", "name"}),
    "anomalies": ({"column"}, {"from", "k"}),
    "associate": ({"target"}, {"from", "exclude"}),
    "profile": (set(), {"from", "columns"}),
}
PERIODS = {"D", "W", "M", "Q", "Y"}

SYSTEM = """You are the planning component of Silt, a data analyst. You do NOT calculate anything. You turn the user's request into ONE JSON object that Silt's engine validates and runs on the real data.
Never write result numbers. The only numbers you may write are numbers the user stated (targets, percentages, days, counts) and the date tokens listed in the request.

Reply with exactly one JSON object and nothing else. Actions:
{"action":"program","objective":"<what the user wants, one sentence>","steps":[...],"answer":"<id of the step that answers>"}
{"action":"clarify","question":"<one question>","options":["<column or value>", ...]}   when it is ambiguous which column, person or target is meant
{"action":"unsupported","reason":"<what the data or the operations lack>"}   when it cannot be computed from this data
{"action":"rerun","change":{"target":<number>}}  or  {"action":"rerun","change":{"absent_days":<number>}}   the user changes an assumption of the OPEN OBJECTIVE listed under Conversation
{"action":"compare_scenarios","a":<number>,"b":<number>}   compare two analyses numbered under Conversation
{"action":"finding","n":<number>,"mode":"detail" or "dig"}   about a finding numbered under Conversation

Program steps. Each step: "id" (short, unique: letters, digits, _), "op", optional "label" (a few words, no numbers), and its arguments. "from" is "data" or the id of an earlier step.
- select: from, entities {"<column>":["<value>"]}, except {"<column>":["<value>"]}, period {"start":"<date token>","end":"<date token>"}, where [{"column":"<column>","op":"== != > >= < <=","value":<value>}]  -> table
- derive: from, name (new column name), formula (arithmetic on columns, e.g. "price * quantity", hours_between(start, end), month(date_col))  -> table with the new column
- aggregate: from (a table), fn (sum, mean, median, min, max, count, nunique), column (not needed for count), optional by "<category column>" or per "D"/"W"/"M"/"Q"/"Y"  -> a value, or a value per group / per period
- ratio: a, b (two values, or two per-group results with the same groups)  -> a divided by b (for example hours per person)
- rank: from (a per-group result), n, order ("desc" highest first, "asc" lowest first)  -> top or bottom groups
- project: from (a table), column, days (horizon), history_start, history_end  -> the daily pace carried forward, with a range
- compare: a, b  -> difference and ratio of two results;  or a (a projection or value), target (number), base (optional: step with the amount already done), days (days left)  -> verdict against the target
- scenario: from, changes [{"remove":{"<column>":"<value>"}} or {"scale":{"column":"<column>","pct":<number>}}]  -> a hypothetical table
- anomalies: from (a table), column (numeric)  -> unusual values
- associate: from (a table), target (the outcome column)  -> columns associated with it (associations, not causes)
- profile: from (a table)  -> missing values and impossible records

Rules:
- Use column names and values exactly as listed. Never invent a column or a value.
- Use date tokens such as $month_start, $today, $days_left_month instead of writing dates or counting days.
- A share or rate of a yes/no value: derive a flag (formula "<column> == '<value>'"), then aggregate mean of it.
- Durations from start and end times: use the derive formula listed under "Measures that can be derived".
- "Why" questions: compute the comparison, then associate the outcome with the other columns or compare groups. Associations are not causes.
- If two or more columns could mean the concept the user names, clarify. If a needed column does not exist, say unsupported.
- Keep programs short: only the steps the request needs. Always give "answer".
- rerun, compare_scenarios and finding are only for analyses and findings listed under Conversation; with none listed, write a program.

Example. Columns: picker (person name), zone (category: A, B, C), picked_at (date), lines_picked (count), errors (count).
Request: Which zone gets the least work done for the people it has, and what goes with errors?
{"action":"program","objective":"Lines picked per picker in each zone, the lowest zone, and what is associated with errors","steps":[{"id":"lines","op":"aggregate","from":"data","fn":"sum","column":"lines_picked","by":"zone"},{"id":"staff","op":"aggregate","from":"data","fn":"nunique","column":"picker","by":"zone"},{"id":"per_head","op":"ratio","a":"lines","b":"staff","label":"lines per picker"},{"id":"lowest","op":"rank","from":"per_head","n":1,"order":"asc"},{"id":"links","op":"associate","from":"data","target":"errors"}],"answer":"per_head"}

Example. Columns: shop (category: North, South), sold_on (date), sales (amount).
Request: If North keeps going like this, will it pass 5000 in sales by the end of the quarter?
{"action":"program","objective":"Whether North reaches 5000 sales this quarter at its current daily pace","steps":[{"id":"north","op":"select","from":"data","entities":{"shop":["North"]}},{"id":"qtd","op":"select","from":"north","period":{"start":"$quarter_start","end":"$today"}},{"id":"done","op":"aggregate","from":"qtd","fn":"sum","column":"sales"},{"id":"pace","op":"project","from":"north","column":"sales","days":"$days_left_quarter","history_start":"$quarter_start","history_end":"$today"},{"id":"verdict","op":"compare","a":"pace","target":5000,"base":"done","days":"$days_left_quarter"}],"answer":"verdict"}

Example. Columns: press (category: P1, P2, P3), run_date (date), start_time (time of day), end_time (time of day), parts_made (count).
Measures that can be derived: run duration: name run_hours, formula hours_between(start_time, end_time).
Request: How many parts would we lose if P2 stopped for a week, compared with running normally?
{"action":"program","objective":"Parts expected over 7 days with and without press P2","steps":[{"id":"normal","op":"project","from":"data","column":"parts_made","days":7},{"id":"no_p2","op":"scenario","from":"data","changes":[{"remove":{"press":"P2"}}]},{"id":"without","op":"project","from":"no_p2","column":"parts_made","days":7},{"id":"loss","op":"compare","a":"without","b":"normal","a_label":"without P2","b_label":"normal"}],"answer":"loss"}

Example. Columns: region (category), revenue (amount), net_income (amount). Request: what is the income by region?
{"action":"clarify","question":"Which column do you mean by income?","options":["revenue","net_income"]}
"""


class PlanRejected(ValueError):
    pass


# The JSON type every step argument must have. Checked before any other check reads the step, so a plan whose
# arguments have the wrong shape (an entities list, a column given as an object, ...) is rejected with a reason the
# model can repair, never crashes the turn.
_SCALAR = (str, int, float, bool)
ARG_TYPES: dict[str, tuple] = {
    "id": (str,), "op": (str,), "label": (str,), "from": (str,),
    "entities": (dict,), "except": (dict,), "period": (dict,),
    "where": (list, dict),               # {column: value} is a known slip _normalise repairs
    "name": (str,), "formula": (str,), "unit": (str,),
    "fn": (str,), "column": (str,), "by": (str,), "per": (str,),
    "a": (str,), "b": (str,), "broadcast": (bool,),
    "n": (int, float, str), "order": (str,),
    "days": (int, float, str), "history_start": (str,), "history_end": (str,),
    "target": (int, float, str), "base": (str,), "a_label": (str,), "b_label": (str,),
    "changes": (list,), "k": (int, float), "exclude": (list, str), "columns": (list, str),
}


def shape_problems(steps: list) -> list[str]:
    """Structural problems in a program's steps: wrong JSON types, nested where a value belongs, malformed filters."""
    out: list[str] = []
    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            out.append(f"step {i + 1} must be a JSON object")
            continue
        sid = st.get("id") if isinstance(st.get("id"), str) else f"#{i + 1}"
        for k, v in st.items():
            want = ARG_TYPES.get(k)
            if v is None or want is None:
                continue                      # unknown keys are reported by the argument check
            if isinstance(v, bool) and bool not in want:
                out.append(f"step '{sid}': '{k}' must be {' or '.join(t.__name__ for t in want)}, not true/false")
            elif not isinstance(v, want):
                out.append(f"step '{sid}': '{k}' must be {' or '.join(t.__name__ for t in want)}, not {type(v).__name__}")
        for key in ("entities", "except"):
            for c, vals in (st.get(key) if isinstance(st.get(key), dict) else {}).items():
                vs = vals if isinstance(vals, list) else [vals]
                if not all(isinstance(x, _SCALAR) for x in vs):
                    out.append(f"step '{sid}': {key} values for '{c}' must be plain values")
        wh = st.get("where")
        if isinstance(wh, dict) and not ({"column", "op"} <= set(wh)) and \
                not all(isinstance(k, str) and isinstance(v, _SCALAR) for k, v in wh.items()):
            out.append(f"step '{sid}': a where written as {{column: value}} needs plain values")
        for w in wh if isinstance(wh, list) else []:
            if not isinstance(w, dict) or not isinstance(w.get("column"), str) or not isinstance(w.get("op"), str) \
                    or isinstance(w.get("value"), dict):
                out.append(f"step '{sid}': each where condition must be {{\"column\":..,\"op\":..,\"value\":..}}")
        per = st.get("period") if isinstance(st.get("period"), dict) else {}
        if any(v is not None and not isinstance(v, str) for v in per.values()):
            out.append(f"step '{sid}': period start/end must be date tokens")
        for ch in st.get("changes") if isinstance(st.get("changes"), list) else []:
            if not isinstance(ch, dict):
                out.append(f"step '{sid}': each change must be a JSON object")
    return out


# ------------------------------------------------------------------ grounding --
def anchor_date(df: pd.DataFrame) -> tuple[pd.Timestamp, str]:
    """'Today' for period tokens: the real date, or the last recorded date when the records end well before it (a
    historical extract); said with every answer that uses a token."""
    from app.analyst.scope import today
    t = today().normalize()
    dates = [pd.to_datetime(df[c], errors="coerce") for c in df.columns if kind_of(df[c]) == "datetime"]
    last = max((d.max() for d in dates if d.notna().any()), default=None)
    if last is not None and last.normalize() < t - pd.Timedelta(days=31):
        return last.normalize(), f"the data ends on {last.date()}, so periods are counted from that date"
    return t, f"today is {t.date()}"


def date_tokens(anchor: pd.Timestamp) -> dict[str, object]:
    a = anchor
    ms = a.replace(day=1)
    me = ms + pd.offsets.MonthEnd(0)
    qs = pd.Timestamp(year=a.year, month=3 * ((a.month - 1) // 3) + 1, day=1)
    qe = qs + pd.offsets.QuarterEnd(0)
    ys, ye = pd.Timestamp(year=a.year, month=1, day=1), pd.Timestamp(year=a.year, month=12, day=31)
    ws = a - pd.Timedelta(days=a.weekday())
    we = ws + pd.Timedelta(days=6)
    lms = (ms - pd.Timedelta(days=1)).replace(day=1)
    d = lambda x: str(x.date())  # noqa: E731
    return {"$today": d(a), "$week_start": d(ws), "$week_end": d(we), "$month_start": d(ms), "$month_end": d(me),
            "$quarter_start": d(qs), "$quarter_end": d(qe), "$year_start": d(ys), "$year_end": d(ye),
            "$last_month_start": d(lms), "$last_month_end": d(ms - pd.Timedelta(days=1)),
            "$days_left_week": int((we - a).days), "$days_left_month": int((me - a).days),
            "$days_left_quarter": int((qe - a).days), "$days_left_year": int((ye - a).days)}


def unit_of(df: pd.DataFrame, col, prof: dict | None = None) -> str | None:
    """A unit label for a measure, so results of different kinds are never compared or added."""
    if col is None:
        return "records"
    if col not in df.columns:
        return None
    r = (prof or {}).get("roles", {}).get(col) or semantics.column_role(df, col)
    t = semantics._tokens(col)  # noqa: SLF001
    if t & {"hours", "hour", "hrs", "hr"}:
        return "hours"
    if t & {"minutes", "mins", "min"}:
        return "minutes"
    if r["role"] == "amount":
        return "money"
    if r["role"] == "rate":
        return f"rate ({col})"
    if r["role"] == "count":
        return "count"
    return str(col)


def grounding(message: str, df: pd.DataFrame, memory: dict, prof: dict | None = None) -> str:
    """What the model may know about this data: names, roles, and summaries -- never whole records. Person names
    are not listed; only the ones the message itself names are confirmed."""
    from app.analyst.calc import named_values
    prof = prof or semantics.profile(df)
    anchor, why = anchor_date(df)
    lines = [f"Dataset: {len(df):,} records, {len(df.columns)} columns.", "Columns:"]
    for r in prof["columns"]:
        c, role = r["column"], r["role"]
        s = df[c]
        bits = [role.replace("_", " ")]
        if role in ("category", "flag"):
            vals = [str(v) for v in s.dropna().astype(str).value_counts().index[:15]]
            more = s.nunique() - len(vals)
            bits.append("values: " + ", ".join(vals) + (f" (+{more} more)" if more > 0 else ""))
        elif role in ("person_name", "identifier", "free_text"):
            bits.append(f"{s.nunique():,} distinct values (not listed)")
        elif role == "datetime":
            d = pd.to_datetime(s, errors="coerce")
            bits.append(f"from {d.min().date()} to {d.max().date()}")
        elif kind_of(s) == "numeric":
            x = pd.to_numeric(s, errors="coerce")
            bits.append(f"range {x.min():g} to {x.max():g}, median {x.median():g}")
        miss = int(s.isna().sum())
        if miss:
            bits.append(f"{miss} missing")
        lines.append(f"- {c} ({'; '.join(bits)})")
    der = derivable(df)
    if der:
        lines.append("Measures that can be derived (use a derive step): " + "; ".join(
            f"{label}: name {n}, formula {f}" for label, (n, f) in der.items()))
    named = named_values(df, message)
    if named:
        lines.append("Values named in the request: " + "; ".join(f"'{v}' is a value of {c}" for c, v in named))
    amb = ambiguities(message, df)
    if amb:
        lines.append("Ambiguous: " + "; ".join(f"'{w}' could mean any of {', '.join(cs)}" for w, cs in amb.items()))
    toks = date_tokens(anchor)
    lines.append(f"Date tokens ({why}): " + ", ".join(f"{k}={v}" for k, v in toks.items()))
    conv = conversation_context(memory)
    if conv:
        lines.append("Conversation:\n" + conv)
    return "\n".join(lines)


def derivable(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Standard per-record measures this data's columns can make (app/analyst/metrics.py), as derive formulas the
    calculator accepts -- e.g. a shift duration from clock-in and clock-out times."""
    from app.analyst import metrics
    out = {}
    for label in metrics.available(df):
        try:
            d, _ = metrics.row_definition(label, df)
        except Exception:  # noqa: BLE001
            continue
        if d and len(d.get("derive") or {}) == 1:
            out[label] = next(iter(d["derive"].items()))
    return out


def conversation_context(memory: dict) -> str:
    out = []
    from app.analyst import scope as scoping
    if scoping.active(memory.get("scope")):
        out.append(f"- current scope: {scoping.describe(memory['scope'])}")
    hist = memory.get("objectives") or []
    for i, o in enumerate(hist, 1):
        p = o.get("params", {})
        keys = {k: p.get(k) for k in ("entity", "target", "unit", "absent_days", "days", "period") if p.get(k) is not None}
        out.append(f"- analysis {i}: {o['type']} " + json.dumps(keys, default=str)
                   + (" (OPEN OBJECTIVE)" if o is memory.get("objective") or i == len(hist) else ""))
    for name, r in list((memory.get("planned") or {}).items())[-3:]:
        vals = ", ".join(f"@{name}.{k} ({v['label']}{', hypothetical' if v['hypothetical'] else ''})"
                         for k, v in r["values"].items())
        out.append(f"- earlier program {name} for '{r['question'][:80]}': values you may use as step inputs: {vals}")
    inv = memory.get("investigation")
    if inv:
        for i, x in enumerate(inv["findings"][:8], 1):
            out.append(f"- finding {i}: {x['title']} (columns: {', '.join(map(str, x['columns']))})")
    last = memory.get("last_answer")
    if last:
        out.append(f"- last question: {last['question'][:120]}")
    return "\n".join(out)


def ambiguities(message: str, df: pd.DataFrame) -> dict[str, list[str]]:
    """Concept words in the message that two or more columns could mean, when the message names none of them."""
    from app.analyst.orchestrator import followup_cols
    named = set(map(str, followup_cols(df, message)))
    out = {}
    for w in dict.fromkeys(re.findall(r"[a-z]+", message.lower())):
        cs = semantics.concept_candidates(w, df)
        if len(cs) >= 2 and not named & set(cs):
            out[w] = cs
    return out


# -------------------------------------------------------------------- parsing --
def balance(text: str) -> str:
    """Mismatched brackets -- the small model's commonest slip ("}}]}" for "}]}") -- made consistent: a closer that
    does not match the open bracket is dropped (a stray '}' inside a list) or preceded by the missing closer, and
    unclosed brackets are closed. Strings are left untouched. The result must still pass every check below."""
    out, stack, in_str, esc = [], [], False, False
    pair = {"}": "{", "]": "["}
    for ch in text:
        if in_str:
            out.append(ch)
            esc = (ch == "\\" and not esc)
            if ch == '"' and not esc:
                in_str = False
            if ch != "\\":
                esc = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            while stack and stack[-1] != pair[ch] and ch == "]" and stack[-1] == "{":
                out.append("}")
                stack.pop()
            if not stack or stack[-1] != pair[ch]:
                continue                              # a stray closer
            stack.pop()
        out.append(ch)
        if not stack and ch in "}]":
            break
    out += ["}" if b == "{" else "]" for b in reversed(stack)]
    return "".join(out)


def parse(text: str) -> dict:
    """The model's reply as one JSON object, or PlanRejected with the reason."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise PlanRejected("the reply contains no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        try:
            # a new step always opens with "id": '},"id":' and '},"{id":' are a missing '{' before it
            fixed = re.sub(r'\}\s*,\s*"\{id"\s*:', '},{"id":', text[start:])
            fixed = re.sub(r'\}\s*,\s*"id"\s*:', '},{"id":', fixed)
            obj = json.loads(balance(fixed))
            obj["_repaired"] = ["brackets balanced" if fixed == text[start:] else "a missing '{' before a step restored"]
        except (json.JSONDecodeError, TypeError):
            raise PlanRejected(f"the JSON is malformed ({e.msg} at character {e.pos})") from e
    if isinstance(obj, dict) and obj.get("action") in OP_ARGS:
        # one operation written as the whole reply: wrapped as a one-step program (its arguments are still checked)
        a = obj.pop("action")
        obj.pop("op", None)
        obj = {"action": "program", "steps": [{"id": "s1", "op": a, **obj}], "answer": "s1",
               "_repaired": [f"the single operation '{a}' was wrapped as a program"]}
    if not isinstance(obj, dict) or obj.get("action") not in ACTIONS:
        a = obj.get("action") if isinstance(obj, dict) else None
        if a in OP_ARGS:
            raise PlanRejected(f"'{a}' is an operation, not an action: put it in a program, e.g. {{\"action\":\"program\","
                               f"\"steps\":[{{\"id\":\"s1\",\"op\":\"{a}\", ...}}],\"answer\":\"s1\"}}")
        raise PlanRejected(f"'action' must be one of {', '.join(sorted(ACTIONS))}")
    return obj


# ------------------------------------------------------------------- checking --
def _sub_tokens(x, toks):
    if isinstance(x, str) and x.startswith("$"):
        if x not in toks:
            raise PlanRejected(f"'{x}' is not a date token (use one of {', '.join(toks)})")
        return toks[x]
    if isinstance(x, dict):
        return {k: _sub_tokens(v, toks) for k, v in x.items()}
    if isinstance(x, list):
        return [_sub_tokens(v, toks) for v in x]
    return x


def _values_of(df, col) -> set[str]:
    return {str(v).strip().lower() for v in df[col].dropna().astype(str).unique()}


def check(plan: dict, df: pd.DataFrame, inputs: dict, message: str, prof: dict, trusted: bool = False) -> list[dict]:
    """The program's steps, made concrete (tokens resolved, units attached), or PlanRejected with every problem found."""
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanRejected("a program needs a non-empty 'steps' list")
    if len(steps) > MAX_STEPS:
        raise PlanRejected(f"too many steps ({len(steps)}; at most {MAX_STEPS})")
    shape = shape_problems(steps)
    if shape:
        raise PlanRejected("; ".join(shape[:6]))
    toks = date_tokens(anchor_date(df)[0])
    roles = prof["roles"]
    fixed = plan.setdefault("_repaired", [])
    alias = {"avg": "mean", "average": "mean", "total": "sum", "counts": "count", "distinct": "nunique",
             "count_distinct": "nunique", "maximum": "max", "minimum": "min"}
    for st in steps:
        if isinstance(st, dict) and st.get("op") == "aggregate" and st.get("fn") in alias:
            fixed.append(f"'{st['fn']}' read as '{alias[st['fn']]}'")
            st["fn"] = alias[st["fn"]]
    steps = plan["steps"] = _normalise(steps, df, fixed)
    steps = plan["steps"] = _flag_means(steps, df, prof, fixed)
    steps = plan["steps"] = _hoist_filters(steps, fixed)
    steps = plan["steps"] = _add_derives(steps, df, fixed)
    if not plan.get("answer") and isinstance(steps[-1], dict) and steps[-1].get("id"):
        plan["answer"] = steps[-1]["id"]
        fixed.append("no answer step named: the last step is shown first")
    problems: list[str] = []
    out: list[dict] = []
    cols = {str(c): c for c in df.columns}
    tables: dict[str, set] = {"data": set(cols)}        # step id -> columns its table has
    kinds: dict[str, str] = {k: v.kind for k, v in inputs.items()}
    units: dict[str, str | None] = {k: v.unit for k, v in inputs.items()}
    used_cols: set[str] = set()
    fns: dict[str, str] = {}                            # step id -> aggregation, for the ratio check
    ratio_cols: dict[str, tuple] = {}                   # derived column -> (numerator, denominator) for a / b

    for raw in steps:
        if not isinstance(raw, dict):
            problems.append("every step must be a JSON object")
            continue
        sid, op = raw.get("id"), raw.get("op")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,30}", sid) or sid in kinds or sid == "data":
            problems.append(f"step id {sid!r} must be a new short name (letters, digits, _)")
            continue
        if op not in OP_ARGS:
            problems.append(f"step '{sid}': '{op}' is not an operation (use {', '.join(sorted(OP_ARGS))})")
            continue
        need, opt = OP_ARGS[op]
        extra = set(raw) - need - opt - {"id", "op", "label"}
        if extra:
            problems.append(f"step '{sid}': {op} does not take {', '.join(sorted(extra))}")
        missing = need - set(raw)
        if missing:
            problems.append(f"step '{sid}': {op} needs {', '.join(sorted(missing))}")
            continue
        for k in ("days", "n", "target") if not trusted else ():     # Silt's own programs: numbers are its own
            v = raw.get(k)
            if v is None or (isinstance(v, str) and v.startswith("$")) or (k == "target" and op != "compare") \
                    or (k == "n" and op != "rank") or (k == "days" and op not in ("project", "compare")):
                continue
            if k == "n" and isinstance(v, int) and v in (1, 3, 5, 10):
                continue                                  # a top-n size is presentation, not a claim
            try:
                fv = float(v)
            except (TypeError, ValueError):
                problems.append(f"step '{sid}': {k} must be a number")
                continue
            if not _stated(fv, message):
                problems.append(f"step '{sid}': {k} {fv:g} is not a number in the request"
                                + (" (use a token such as $days_left_month)" if k == "days" else ""))
            raw[k] = fv if k != "n" else int(fv)
        for k in ("history_start", "history_end") if not trusted else ():
            v = raw.get(k)
            if v is not None and not (isinstance(v, str) and v.startswith("$")) and str(v) not in message:
                problems.append(f"step '{sid}': write {k} as a date token (such as $month_start), not a date")
        per0 = raw.get("period") if isinstance(raw.get("period"), dict) else {}
        for k in ("start", "end") if not trusted else ():
            v = per0.get(k)
            if v is not None and not (isinstance(v, str) and v.startswith("$")) and str(v) not in message:
                problems.append(f"step '{sid}': write the period {k} as a date token (such as $month_start), not a date")
        try:
            s = _sub_tokens({k: v for k, v in raw.items()}, toks)
        except PlanRejected as e:
            problems.append(f"step '{sid}': {e}")
            continue
        s.setdefault("from", "data") if op in ("select", "derive", "aggregate", "project", "scenario", "anomalies",
                                                 "associate", "profile") else None
        label = s.pop("label", None)
        if isinstance(label, str):
            s["label"] = label[:60]                     # kept in the program trace; answers use Silt's own labels
        src = s.get("from")
        avail = None
        if op in ("select", "derive", "aggregate", "project", "scenario", "anomalies", "associate", "profile"):
            if src not in tables:
                what = "is not an earlier step" if src not in kinds else f"is a {kinds[src]}, not a table"
                problems.append(f"step '{sid}': input '{src}' {what}")
                continue
            avail = tables[src]

        def need_col(c, why=""):
            if not isinstance(c, str) or c not in avail:
                close = [x for x in avail if str(c).lower().replace(" ", "_") == str(x).lower()]
                problems.append(f"step '{sid}': there is no column '{c}'{why}" + (f" (did you mean '{close[0]}'?)" if close else ""))
                return False
            used_cols.add(c)
            return True

        if op == "select":
            for key in ("entities", "except"):
                for c, vals in (s.get(key) or {}).items():
                    if not need_col(c):
                        continue
                    vals = vals if isinstance(vals, list) else [vals]
                    s[key][c] = [str(v) for v in vals]
                    if c in cols:
                        bad = [v for v in vals if str(v).strip().lower() not in _values_of(df, c)]
                        if bad:
                            problems.append(f"step '{sid}': {', '.join(map(str, bad))} does not occur in '{c}'")
            for w in s.get("where") or []:
                if not isinstance(w, dict) or not {"column", "op", "value"} <= set(w):
                    problems.append(f"step '{sid}': each condition needs column, op and value")
                    continue
                if need_col(w["column"]) and w["op"] not in P.CMP:
                    problems.append(f"step '{sid}': '{w['op']}' is not a comparison")
                elif w["column"] in cols and kind_of(df[w["column"]]) == "categorical" and w["op"] in ("==", "!=") \
                        and str(w["value"]).strip().lower() not in _values_of(df, w["column"]):
                    problems.append(f"step '{sid}': '{w['value']}' does not occur in '{w['column']}'")
            per = s.get("period")
            if per is not None:
                if not isinstance(per, dict) or not (per.get("start") or per.get("end")):
                    problems.append(f"step '{sid}': period needs start and/or end")
                elif not prof["dates"]:
                    problems.append(f"step '{sid}': a period needs a date column and this data has none")
                else:
                    for k in ("start", "end"):
                        if per.get(k) is not None:
                            try:
                                pd.Timestamp(per[k])
                            except Exception:  # noqa: BLE001
                                problems.append(f"step '{sid}': period {k} '{per[k]}' is not a date")
            tables[sid], kinds[sid] = set(avail), "table"
        elif op == "derive":
            if not str(s["name"]).isidentifier() or s["name"] in cols:
                problems.append(f"step '{sid}': '{s['name']}' must be a new column name")
            for c in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(s["formula"])):
                if c in cols:
                    used_cols.add(c)
            tables[sid], kinds[sid] = set(avail) | {s["name"]}, "table"
            units[sid] = s.get("unit")
            rm = re.fullmatch(r"\s*\(?\s*([A-Za-z_]\w*)\s*\)?\s*/\s*\(?\s*([A-Za-z_]\w*)\s*\)?\s*", str(s["formula"]))
            if rm and rm.group(1) in cols and rm.group(2) in cols:
                ratio_cols[s["name"]] = (rm.group(1), rm.group(2))
        elif op == "aggregate":
            fn, col = s["fn"], s.get("column")
            if fn not in P.AGG:
                problems.append(f"step '{sid}': '{fn}' is not an aggregation ({', '.join(sorted(P.AGG))})")
            if col in cols and roles[col]["role"] == "flag" and kind_of(df[col]) != "numeric" and fn in ("mean", "sum"):
                vals = sorted(_values_of(df, col))
                problems.append(f"step '{sid}': '{col}' holds text ({', '.join(vals)}); derive a flag first, e.g. "
                                f"formula \"{col} == '{vals[0]}'\", then aggregate that")
            if fn != "count":
                if col is None:
                    problems.append(f"step '{sid}': {fn} needs a column")
                elif need_col(col) and col not in cols and fn in ("mean", "median") and ratio_cols.get(col):
                    a_, b_ = ratio_cols[col]
                    problems.append(f"step '{sid}': '{col}' is {a_} / {b_} per record; its {fn} weights every record "
                                    f"equally. For {a_} per {b_} across groups, divide the totals: aggregate sum of {a_} "
                                    f"and sum of {b_} (with the same by), then a ratio step")
                elif col in cols:
                    role = roles[col]["role"]
                    if fn not in ("nunique",) and role in ("identifier", "person_name", "free_text", "category", "year",
                                                            "datetime", "time_of_day"):
                        problems.append(f"step '{sid}': '{col}' is a {role.replace('_', ' ')}; it cannot be {fn}med "
                                        "(count or nunique it instead)")
                    if fn == "sum" and role == "rate":
                        problems.append(f"step '{sid}': '{col}' is a rate; adding rates up is meaningless (use mean)")
            by, per = s.get("by"), s.get("per")
            if by is not None and per is not None:
                problems.append(f"step '{sid}': give by or per, not both")
            if by is not None and need_col(by) and by in cols and df[by].nunique() > 60 and roles[by]["role"] != "person_name":
                problems.append(f"step '{sid}': '{by}' has {df[by].nunique()} distinct values; it is not a grouping")
            if per is not None and (per not in PERIODS or not prof["dates"]):
                problems.append(f"step '{sid}': per must be D, W, M, Q or Y and needs a date column")
            if fn == "count" and col is not None:
                # count counts records whatever the column; "count of attrition" meant the records where it is Yes
                problems.append(f"step '{sid}': count counts records and takes no column; to count the records where "
                                f"'{col}' has a value, select them first (where), or derive a flag and take its sum or mean")
            if fn == "mean" and (str(col).startswith("is_") or (col in cols and roles[col]["role"] == "flag")):
                s["unit"] = "share"                         # the mean of a 0/1 flag is a share of records
            elif fn in ("count",):
                s["unit"] = "records"
            elif fn == "nunique":
                s["unit"] = "count"      # (the label already says what was counted: "number of distinct agent values")
            elif "unit" not in s:
                s["unit"] = unit_of(df, col, prof) if col in cols else None
            kinds[sid] = "series" if (by or per) else "scalar"
            units[sid] = s["unit"]
            fns[sid] = fn
        elif op == "ratio":
            for k in ("a", "b"):
                if s[k] not in kinds or kinds[s[k]] not in ("scalar", "series", "projection"):
                    problems.append(f"step '{sid}': '{s[k]}' must be an earlier value or per-group result")
            # each group's part of one whole (series ÷ a single total) only when the step says so and both are totals
            # of the same column: a share of that column's total
            of_whole = bool(s.get("broadcast")) and kinds.get(s["a"]) == "series" and kinds.get(s["b"]) == "scalar" \
                and fns.get(s["a"]) == fns.get(s["b"]) == "sum"
            if s["a"] in kinds and s["b"] in kinds and {kinds[s["a"]], kinds[s["b"]]} == {"series", "scalar"} \
                    and not of_whole:
                problems.append(f"step '{sid}': divide two per-group results or two values, not one of each")
            for k in ("a", "b"):
                fn = fns.get(s[k])
                if fn in ("mean", "median", "min", "max"):
                    problems.append(f"step '{sid}': '{s[k]}' is a {fn}; only totals and counts can be divided "
                                    "(use sum or count/nunique, or compare the means directly)")
            kinds[sid] = "series" if kinds.get(s["a"]) == "series" else "scalar"
            units[sid] = f"{units.get(s['a']) or 'value'} per {units.get(s['b']) or 'value'}"
        elif op == "rank":
            if kinds.get(src) != "series":
                problems.append(f"step '{sid}': rank needs a per-group result, '{src}' is not one")
            if s.get("order", "desc") not in ("asc", "desc"):
                problems.append(f"step '{sid}': order must be asc or desc")
            try:
                s["n"] = int(s.get("n", 5))
            except (TypeError, ValueError):
                problems.append(f"step '{sid}': n must be a whole number")
            kinds[sid], units[sid] = "series", units.get(src)
        elif op == "project":
            if need_col(s["column"]) and s["column"] in cols and kind_of(df[s["column"]]) != "numeric":
                problems.append(f"step '{sid}': '{s['column']}' is not numeric")
            if not prof["dates"]:
                problems.append(f"step '{sid}': a projection needs a date column and this data has none")
            if not isinstance(s["days"], (int, float)) or not 0 <= s["days"] <= 3660:
                problems.append(f"step '{sid}': days must be a number of days (a token like $days_left_month)")
            kinds[sid] = "projection"
            units[sid] = unit_of(df, s["column"], prof) if s["column"] in cols else None
        elif op == "compare":
            if s["a"] not in kinds:
                problems.append(f"step '{sid}': '{s['a']}' is not an earlier step")
            elif kinds[s["a"]] in ("table", "series"):
                problems.append(f"step '{sid}': '{s['a']}' is a {kinds[s['a']]}; compare needs single values "
                                "(aggregate first)")
            if "target" in s:
                if not isinstance(s["target"], (int, float)):
                    problems.append(f"step '{sid}': target must be a number the user stated")
                elif not _stated(s["target"], message):
                    problems.append(f"step '{sid}': the target {s['target']:g} is not a number in the request")
                if s.get("base") is not None and s["base"] not in kinds:
                    problems.append(f"step '{sid}': base '{s['base']}' is not an earlier step")
                if s.get("base") and s["base"] in units and s["a"] in units and units[s["base"]] != units[s["a"]]:
                    problems.append(f"step '{sid}': the base is in {units[s['base']]} but '{s['a']}' is in {units[s['a']]}")
            elif "b" in s:
                if s["b"] not in kinds:
                    problems.append(f"step '{sid}': '{s['b']}' is not an earlier step")
                elif kinds[s["b"]] in ("table", "series"):
                    problems.append(f"step '{sid}': '{s['b']}' is a {kinds[s['b']]}; compare needs single values")
                ua, ub = units.get(s["a"]), units.get(s["b"])
                if ua and ub and ua != ub:
                    problems.append(f"step '{sid}': cannot compare {ua} with {ub}")
            else:
                problems.append(f"step '{sid}': compare needs b or target")
            kinds[sid], units[sid] = "comparison", units.get(s["a"])
        elif op == "scenario":
            for ch in s["changes"] if isinstance(s["changes"], list) else [None]:
                if isinstance(ch, dict) and "remove" in ch and isinstance(ch["remove"], dict) and len(ch["remove"]) == 1:
                    c, v = next(iter(ch["remove"].items()))
                    if need_col(c) and c in cols and str(v).strip().lower() not in _values_of(df, c):
                        problems.append(f"step '{sid}': '{v}' does not occur in '{c}'")
                elif isinstance(ch, dict) and "scale" in ch and isinstance(ch["scale"], dict):
                    c = ch["scale"].get("column")
                    if need_col(c) and c in cols and kind_of(df[c]) != "numeric":
                        problems.append(f"step '{sid}': '{c}' is not numeric, so it cannot be scaled")
                    if not _stated(ch["scale"].get("pct"), message):
                        problems.append(f"step '{sid}': the change {ch['scale'].get('pct')}% is not a number in the request")
                else:
                    problems.append(f"step '{sid}': each change is {{\"remove\":{{column:value}}}} or "
                                    "{\"scale\":{\"column\":...,\"pct\":...}}")
            tables[sid], kinds[sid] = set(avail), "table"
        elif op == "anomalies":
            if need_col(s["column"]) and s["column"] in cols and kind_of(df[s["column"]]) != "numeric":
                problems.append(f"step '{sid}': '{s['column']}' is not numeric")
            kinds[sid] = "anomalies"
        elif op == "associate":
            t = s["target"]
            if need_col(t) and t in cols and roles[t]["role"] in ("identifier", "free_text", "constant", "datetime"):
                problems.append(f"step '{sid}': '{t}' is a {roles[t]['role'].replace('_', ' ')}, not an outcome")
            s["exclude"] = [c for c in prof["nonfactors"] if c != t]
            kinds[sid] = "associations"
        elif op == "profile":
            kinds[sid] = "profile"
        out.append(s)
    used_nums = {float(x) for st in steps if isinstance(st, dict) for x in _numbers_in(st)}
    for n in (_request_numbers(message) if not trusted else ()):
        if n not in used_nums and -n not in used_nums:
            problems.append(f"the request states {n:g}, but no step uses it (a target belongs in compare.target, a "
                            "percentage in a scenario change, a number of days in days)")
    ans = plan.get("answer")
    if not problems and ans not in {s["id"] for s in out}:
        problems.append(f"'answer' must be the id of one of the steps (got {ans!r})")
    elif not problems and kinds.get(ans) == "table" and not any(kinds[s["id"]] != "table" for s in out):
        problems.append("the program only selects records and computes nothing; add the aggregate, associate, "
                        "anomalies, project or compare step that answers the request")
    # a concept two columns could mean is asked about, not guessed
    amb = ambiguities(message, df)
    for w, cs in amb.items():
        if used_cols & set(cs):
            raise Ambiguous(w, cs)
    if problems:
        raise PlanRejected("; ".join(dict.fromkeys(problems)))
    # the executor's own checks (program.validate: ids and references) run too; failures here are plan problems
    try:
        P.validate([{k: v for k, v in s.items() if k != "label"} for s in out], known=inputs)
    except P.ProgramError as e:
        raise PlanRejected(str(e)) from e
    return out


_FILTER_KEYS = ("entities", "except", "period", "where")


def _flag_means(steps: list, df: pd.DataFrame, prof: dict, fixed: list) -> list:
    """The mean (or sum) of a yes/no text column is the share (or number) of 'yes' records: written as a derived
    0/1 flag, so the calculation says exactly what it counts. Two-valued columns without a yes-like value are left
    for the checks to reject with the fix spelled out."""
    out = []
    for st in steps:
        c = st.get("column") if isinstance(st, dict) else None
        r = prof["roles"].get(c) if c in df.columns else None
        if r and r["role"] == "flag" and kind_of(df[c]) != "numeric" and st.get("op") == "aggregate" \
                and st.get("fn") in ("mean", "sum") and r.get("positive"):
            name = f"is_{re.sub(r'[^0-9a-zA-Z_]', '_', str(c))}"[:40]
            pos = str(r["positive"]).replace("'", "")
            d = {"id": f"flag_{st.get('id', 'x')}"[:30], "op": "derive", "from": st.get("from", "data"), "name": name,
                 "formula": f"{c} == '{pos}'"}
            out.append(d)
            st["from"], st["column"] = d["id"], name
            fixed.append(f"the {st['fn']} of {c} is the {'share' if st['fn'] == 'mean' else 'number'} of records where "
                         f"{c} = {pos}")
        out.append(st)
    return out


def _normalise(steps: list, df: pd.DataFrame, fixed: list) -> list:
    """Spelling slips with one meaning: 'group_by' for 'by', a where-dict {col: value} for [{column, ==, value}],
    an arithmetic expression where a column belongs (it becomes a derive step)."""
    out, k = [], 0
    cols = set(map(str, df.columns))
    for st in steps:
        if not isinstance(st, dict):
            out.append(st)
            continue
        if "group_by" in st and "by" not in st:
            st["by"] = st.pop("group_by")
            fixed.append(f"'group_by' read as 'by' in '{st.get('id')}'")
        w = st.get("where")
        if isinstance(w, dict) and w and not {"column", "op"} <= set(w):
            st["where"] = [{"column": c, "op": "==", "value": v} for c, v in w.items()]
            fixed.append(f"the condition on '{st.get('id')}' was written as {{column: value}}")
        elif isinstance(w, dict) and {"column", "op", "value"} <= set(w):
            st["where"] = [w]
        if st.get("op") == "aggregate" and st.get("fn") == "count" and st.get("column"):
            filt = {w.get("column") for w in st.get("where") or [] if isinstance(w, dict)} | set(st.get("entities") or {})
            if st["column"] in filt:
                fixed.append(f"'{st.get('id')}' counts the records where {st.pop('column')} matches (count takes no column)")
        c = st.get("column")
        if isinstance(c, str) and c not in cols and re.search(r"[-+*/]", c) and \
                set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", c)) <= cols:
            k += 1
            name = f"expr_{k}"
            out.append({"id": f"derive_{name}", "op": "derive", "from": st.get("from", "data"), "name": name, "formula": c})
            st["from"], st["column"] = f"derive_{name}", name
            fixed.append(f"the expression '{c}' was computed per record as a derive step")
        out.append(st)
    return out


def _hoist_filters(steps: list, fixed: list) -> list:
    """A filter written on a step that does not take one ("aggregate ... where ...") becomes a select step before it:
    the same records, written the way the program language expresses it."""
    out = []
    for st in steps:
        if isinstance(st, dict) and st.get("op") in OP_ARGS and st.get("op") != "select" and \
                any(k in st for k in _FILTER_KEYS) and st.get("op") in ("aggregate", "project", "anomalies", "associate",
                                                                         "profile", "derive", "scenario"):
            sel = {"id": f"{st.get('id', 'step')}_records", "op": "select", "from": st.get("from", "data")}
            for k in _FILTER_KEYS:
                if k in st:
                    sel[k] = st.pop(k)
            st["from"] = sel["id"]
            out.append(sel)
            fixed.append(f"the filter on '{st.get('id')}' was moved into a select step")
        out.append(st)
    return out


def _add_derives(steps: list, df: pd.DataFrame, fixed: list) -> list:
    """A derivable measure used by name without its derive step (shift_hours from clock-in/out) gets that step, with
    the formula Silt listed; steps reading the data then read the table that has it."""
    der = {n: f for n, f in derivable(df).values()}
    named = {st.get("name") for st in steps if isinstance(st, dict) and st.get("op") == "derive"}
    want = []
    for st in steps:
        if isinstance(st, dict):
            for k in ("column", "target", "by"):
                if st.get(k) in der and st.get(k) not in named and st.get(k) not in df.columns and st[k] not in want:
                    want.append(st[k])
    if not want:
        return steps
    pre, src = [], "data"
    for n in want:
        sid = f"derive_{n}"[:30]
        pre.append({"id": sid, "op": "derive", "from": src, "name": n, "formula": der[n]})
        src = sid
        fixed.append(f"'{n}' was derived as {der[n]} (it is not a column)")
    for st in steps:
        if isinstance(st, dict) and st.get("from", "data") == "data" and st.get("op") not in ("rank", "ratio", "compare"):
            st["from"] = src
    return pre + steps


class Ambiguous(Exception):
    def __init__(self, word, columns):
        super().__init__(word)
        self.word, self.columns = word, columns


def _numbers_in(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield x
    elif isinstance(x, str):
        try:
            yield float(x)
        except ValueError:
            return
    elif isinstance(x, dict):
        for v in x.values():
            yield from _numbers_in(v)
    elif isinstance(x, list):
        for v in x:
            yield from _numbers_in(v)


def _request_numbers(message: str) -> set[float]:
    """Numbers the user stated that an analysis must use: targets, percentages, day counts -- not years, dates,
    times of day or ordinals ("finding 2", "the second")."""
    m = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|\b(?:19|20)\d{2}\b", " ", message)
    m = re.sub(r"\b(?:finding|step|analysis|scenario|option|number|#)\s*\d+\b", " ", m, flags=re.I)
    from app.analyst.objectives import NUM_WORDS
    out = {float(n.replace(",", "")) for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", m)}
    # a spelled-out number is a quantity only with a unit after it ("ten days"), not in "which one is longest"
    unit = r"(?:working\s+)?(?:days?|weeks?|months?|hours?|hrs?|years?|percent|people|staff|shifts?|units?|items?|times)"
    out |= {float(v) for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\s+{unit}\b", m, re.I)}
    out |= {float(k) * 1000 for k in re.findall(r"(\d+(?:\.\d+)?)\s*k\b", m, re.I)}
    return out


def _stated(x, message: str) -> bool:
    """A number the model may write must be one the user wrote (500, 10%, 'ten')."""
    if not isinstance(x, (int, float)):
        return False
    from app.analyst.objectives import NUM_WORDS
    nums = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", message)}
    nums |= {float(v) for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\b", message, re.I)}
    nums |= {-n for n in nums}
    for k in re.findall(r"(\d+(?:\.\d+)?)\s*k\b", message, re.I):
        nums.add(float(k) * 1000)
    return float(x) in nums


# --------------------------------------------------------------------- model --
def _chat(messages: list[dict], temperature: float = 0.0) -> str:
    from app.analyst import planner
    from openai import OpenAI
    from app import llm_gateway
    url = PLANNER_URL or planner.MODEL_URL
    client = OpenAI(base_url=url, api_key=os.getenv("ANALYST_MODEL_KEY", "local"), timeout=PLAN_TIMEOUT, max_retries=0)

    def call() -> str:
        r = client.chat.completions.create(model=PLANNER_MODEL or planner.MODEL_NAME, messages=messages,
                                           temperature=temperature, max_tokens=MAX_TOKENS)
        return r.choices[0].message.content or ""
    return llm_gateway.guarded(llm_gateway.local_provider(url), call)


model_chat = _chat                   # replaced in tests


def _inputs(memory: dict) -> dict[str, P.Result]:
    out = {}
    for name, r in list((memory.get("planned") or {}).items())[-3:]:
        for k, v in r["values"].items():
            out[f"@{name}.{k}"] = P.Result(v["kind"], v["value"], unit=v["unit"], hypothetical=v["hypothetical"],
                                           provenance={"op": "earlier result", "program": name, "step": k})
    return out


def plan(message: str, df: pd.DataFrame, memory: dict) -> dict:
    """{status: ok|clarify|unsupported|reference|failed, ...} with a trace of every attempt and its latency."""
    prof = semantics.profile(df)
    inputs = _inputs(memory)
    user = grounding(message, df, memory, prof) + f"\n\nRequest: {message}"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    trace = []
    started = time.monotonic()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.monotonic()
        try:
            # a repair at temperature 0 tends to repeat the rejected plan word for word
            text = model_chat(msgs) if attempt == 1 else model_chat(msgs, temperature=0.3)
        except Exception as e:  # noqa: BLE001 - outage or timeout: the deterministic paths answer
            trace.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"[:200],
                          "ms": int((time.monotonic() - t0) * 1000)})
            return {"status": "failed", "reason": f"the local model did not answer ({type(e).__name__})", "trace": trace}
        ms = int((time.monotonic() - t0) * 1000)
        try:
            obj = parse(text)
            if obj["action"] == "program":
                try:
                    steps = check(obj, df, inputs, message, prof)
                except (PlanRejected, Ambiguous):
                    raise
                except Exception as e:  # noqa: BLE001 - a model plan Silt cannot read is a rejected plan, not a crash
                    logger.exception("plan check failed on a malformed plan")
                    raise PlanRejected(f"the plan could not be read ({type(e).__name__})") from e
                # executable is not enough: the program must compute what was asked, and every part of it
                req = intent.read(message, df, prof)
                steps = intent.complete(steps, req, df, prof, obj["_repaired"])
                missing = intent.check(steps, req, df, message)
                if missing:
                    raise PlanRejected("the program does not compute what was asked: " + "; ".join(missing))
                trace.append({"attempt": attempt, "ms": ms, "accepted": True, "output": text[:3000],
                              "repaired": obj.get("_repaired") or []})
                return {"status": "ok", "repaired": obj.get("_repaired") or [], "request": req, "objective": _clean(obj.get("objective"), message), "steps": steps,
                        "answer": obj["answer"], "inputs": inputs, "trace": trace}
            trace.append({"attempt": attempt, "ms": ms, "accepted": True, "output": text[:3000]})
            return _other(obj, df, memory, message, trace)
        except Ambiguous as a:
            trace.append({"attempt": attempt, "ms": ms, "accepted": False, "output": text[:3000],
                          "problems": [f"'{a.word}' could mean {', '.join(a.columns)}"]})
            return {"status": "clarify", "question": f"Which column do you mean by **{a.word}**?", "options": a.columns,
                    "trace": trace, "by": "silt"}
        except PlanRejected as e:
            trace.append({"attempt": attempt, "ms": ms, "accepted": False, "output": text[:3000], "problems": [str(e)]})
            if time.monotonic() - started > PLAN_BUDGET - PLAN_TIMEOUT:
                break                                      # no time for a repair within the budget
            msgs = msgs + [{"role": "assistant", "content": text[:3000]},
                           {"role": "user", "content": f"Silt rejected that: {e}. Reply with one corrected JSON object "
                                                       "only, using the listed columns and operations."}]
    return {"status": "failed", "reason": "the model's plan did not pass Silt's checks: " + trace[-1]["problems"][0],
            "trace": trace}


def rule_plan(message: str, df: pd.DataFrame, memory: dict, plain: bool = False) -> dict | None:
    """The deterministic program for a composite request whose parts are all grounded (app/analyst/intent.py), checked
    exactly like a model's plan. None when any part would be a guess."""
    prof = semantics.profile(df)
    req = intent.read(message, df, prof)
    steps = intent.rule_program(req, df, prof, plain=plain)
    if not steps:
        return None
    try:
        steps = check({"action": "program", "steps": steps, "answer": next(s["id"] for s in steps if s["id"] in (
            "per_unit", "result"))}, df, {}, message, prof, trusted=True)
    except (PlanRejected, Ambiguous):
        return None
    return {"status": "ok", "objective": None, "steps": steps, "request": req, "repaired": [], "inputs": {},
            "answer": next(s["id"] for s in steps if s["id"] in ("per_unit", "result")), "trace": [], "rule": True}


def plan_for_request(req, df: pd.DataFrame, message: str) -> dict | None:
    """The deterministic program for an already-built reading (a follow-up applied to the previous answer's reading),
    checked exactly like rule_plan's. None when any part would be a guess."""
    prof = semantics.profile(df)
    steps = intent.rule_program(req, df, prof, plain=True)
    if not steps:
        return None
    ans = next(s["id"] for s in steps if s["id"] in ("per_unit", "result"))
    try:
        steps = check({"action": "program", "steps": steps, "answer": ans}, df, {}, message, prof, trusted=True)
    except (PlanRejected, Ambiguous):
        return None
    return {"status": "ok", "objective": None, "steps": steps, "request": req, "repaired": [], "inputs": {},
            "answer": ans, "trace": [], "rule": True}


def _clean(text, message: str) -> str | None:
    """The model's one-line objective is shown only if every number in it is one the user wrote."""
    if not isinstance(text, str) or not text.strip():
        return None
    for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        if not _stated(float(n.replace(",", "")), message):
            return None
    return text.strip()[:200]


def _other(obj: dict, df, memory: dict, message: str, trace: list) -> dict:
    a = obj["action"]
    if a == "clarify":
        q = obj.get("question")
        opts = [str(o) for o in (obj.get("options") or []) if isinstance(o, (str, int, float))][:6]
        if not isinstance(q, str) or not q.strip():
            raise PlanRejected("a clarification needs a question")
        if _clean(q, message) is None:
            raise PlanRejected("the question contains numbers the user did not give")
        return {"status": "clarify", "question": q.strip()[:300], "options": opts, "trace": trace, "by": "model"}
    if a == "unsupported":
        reason = obj.get("reason")
        if not isinstance(reason, str) or not reason.strip() or _clean(reason, message) is None:
            raise PlanRejected("'unsupported' needs a reason without numbers")
        return {"status": "unsupported", "reason": reason.strip()[:300], "trace": trace}
    if a == "rerun":
        last = memory.get("objective")
        ch = obj.get("change") or {}
        if not last:
            raise PlanRejected("there is no open objective in this conversation to change; write a program for the "
                               "request, or clarify")
        key, val = next(iter(ch.items()), (None, None))
        if key not in ("target", "absent_days") or not _stated(val, message):
            raise PlanRejected("rerun.change must be {\"target\": n} or {\"absent_days\": n} with a number the user gave")
        return {"status": "reference", "kind": "rerun", "key": key, "value": float(val), "trace": trace}
    if a == "compare_scenarios":
        hist = memory.get("objectives") or []
        try:
            i, j = int(obj.get("a")), int(obj.get("b"))
        except (TypeError, ValueError) as e:
            raise PlanRejected("compare_scenarios needs two analysis numbers a and b") from e
        if len(hist) < 2:
            raise PlanRejected(f"the conversation has {len(hist)} earlier analyses, so compare_scenarios is not possible; "
                               "write a program for the request")
        if not (1 <= i <= len(hist) and 1 <= j <= len(hist)) or i == j:
            raise PlanRejected(f"a and b must be two different analysis numbers from 1 to {len(hist)}")
        return {"status": "reference", "kind": "compare", "a": i, "b": j, "trace": trace}
    if a == "finding":
        inv = memory.get("investigation")
        if not inv:
            raise PlanRejected("there are no findings in this conversation; write a program for the request")
        try:
            n = int(obj.get("n"))
        except (TypeError, ValueError) as e:
            raise PlanRejected("finding needs a number n") from e
        if not 1 <= n <= len(inv["findings"]):
            raise PlanRejected(f"n must be from 1 to {len(inv['findings'])}")
        return {"status": "reference", "kind": "finding", "n": n, "mode": "dig" if obj.get("mode") == "dig" else "detail",
                "trace": trace}
    raise PlanRejected(f"unknown action {a}")


# ------------------------------------------------------------------- execution --
def execute(p: dict, df: pd.DataFrame, memory: dict, message: str) -> dict:
    """Run a checked program on the real records; store its values for later turns; present it from the results."""
    from app.analyst.explain import f as fmt
    steps = p["steps"]
    run_steps = [{k: v for k, v in s.items() if k != "label"} for s in steps]
    # impossible records (negative hours, end before start ...) are left out of every step that reads the data, and
    # listed; exact duplicates are kept, as recorded, and flagged
    cols = sorted({c for s in steps for c in _cols_of(s) if c in df.columns}, key=str)
    from app.analyst.objectives import duplicate_policy       # one rule: kept and flagged unless asked to exclude
    valid = {"id": "_valid", "op": "select", "from": "data", "valid_for": cols,
             **({"duplicates": "keep"} if duplicate_policy(message) == "keep" else {})}
    for s in run_steps:
        if s.get("from") == "data":
            s["from"] = "_valid"
    res = P.run([valid] + run_steps, df, inputs=p["inputs"])
    ans = res[p["answer"]]
    labels = {s["id"]: _label(s, steps) for s in steps}     # what was computed, never the model's own wording
    for s in steps:
        if s["op"] == "compare" and "b" in s and s["id"] in res and "a_label" in res[s["id"]].value:
            by = {x["id"]: x for x in steps}
            res[s["id"]].value["a_label"] = labels.get(s["a"]) or ("an earlier result" if s["a"].startswith("@") else s["a"])
            res[s["id"]].value["b_label"] = labels.get(s["b"]) or ("an earlier result" if s["b"].startswith("@") else s["b"])
    name = f"r{len(memory.get('planned') or {}) + 1}"
    values = {s["id"]: {"kind": res[s["id"]].kind, "value": res[s["id"]].value, "unit": res[s["id"]].unit,
                        "hypothetical": res[s["id"]].hypothetical, "label": labels[s["id"]]}
              for s in steps if res[s["id"]].kind in ("scalar", "projection")}
    memory.setdefault("planned", {})[name] = {"question": message, "objective": p.get("objective"), "steps": steps,
                                              "answer": p["answer"], "values": values}
    while len(memory["planned"]) > 6:
        memory["planned"].pop(next(iter(memory["planned"])))
    excluded = res["_valid"].provenance.get("excluded") or []
    _ratio_units(steps, res, message)
    text = render(p, steps, res, labels, df, excluded)
    charts = _charts(ans, labels[p["answer"]], res, steps)
    summaries = {k: v.summary() for k, v in res.items() if not k.startswith("@")}
    return {"answer": text, "charts": charts, "name": name, "results": summaries, "steps": steps,
            "details": details(steps, res, labels, fmt)}


_SHARE_WORDS = re.compile(r"\b(share|slice|portion|proportion|percentage|percent|fraction|what part)\b", re.I)


def _ratio_units(steps: list[dict], res: dict, message: str) -> None:
    """How a ratio reads: a share of a same-unit total ("tip as a share of the bill": 16.0%, share of total_bill) when
    the request asks for a share, otherwise '<numerator column> per <denominator column>' -- never the generic
    'money per count'. Display only; the values are unchanged."""
    by = {s["id"]: s for s in steps}
    for s in steps:
        if s["op"] != "ratio" or s["id"] not in res:
            continue
        a, b = by.get(s["a"]), by.get(s["b"])
        if not (a and b and a["op"] == "aggregate" and b["op"] == "aggregate"):
            continue
        ca, cb = a.get("column") or "records", b.get("column") or "records"
        v = res[s["id"]].value
        vals = [x for x in (v.values() if isinstance(v, dict) else [v]) if x is not None]
        # a part of a whole: asked as a share, two totals, and no group's part exceeds its whole
        if _SHARE_WORDS.search(message) and a["fn"] == b["fn"] == "sum" and vals and all(0 <= x <= 1 for x in vals):
            res[s["id"]].unit = f"share of {cb}"
        else:
            res[s["id"]].unit = f"{ca} per {cb}"


def _cols_of(s: dict) -> list:
    out = [s.get(k) for k in ("column", "by", "target")]
    for k in ("entities", "except"):
        out += list((s.get(k) or {}).keys())
    out += [w.get("column") for w in s.get("where") or [] if isinstance(w, dict)]
    return [c for c in out if c]


def _scope_of(sid: str, steps: list[dict]) -> str:
    by = {s["id"]: s for s in steps}
    bits, cur = [], by.get(sid, {}).get("from")
    while cur in by:
        s = by[cur]
        if s["op"] == "select":
            for c, v in (s.get("entities") or {}).items():
                bits.append(f"{c} = {', '.join(v)}")
            for c, v in (s.get("except") or {}).items():
                bits.append(f"{c} not {', '.join(v)}")
            for w in s.get("where") or []:
                bits.append(f"{w['column']} {w['op']} {w['value']}")
            per = s.get("period") or {}
            if per:
                bits.append(f"{per.get('start') or 'start'} to {per.get('end') or 'end'}")
        elif s["op"] == "scenario":
            bits.append("hypothetical: " + "; ".join(json.dumps(c) for c in s["changes"]))
        cur = s.get("from")
    return ", ".join(reversed(bits))


def _label(s: dict, steps: list[dict], depth: int = 0) -> str:
    """What a step computed, in words built from the operation and its inputs -- never from the model's step ids."""
    op = s["op"]
    by = {x["id"]: x for x in steps}

    def ref(i):
        if i in by and depth < 3:
            return _label(by[i], steps, depth + 1)
        return "an earlier result" if str(i).startswith("@") else str(i)
    sc = _scope_of(s["id"], steps)
    sc = f" ({sc})" if sc else ""
    if op == "aggregate":
        what = "number of records" if s["fn"] == "count" else (f"number of distinct {s['column']} values"
                                                               if s["fn"] == "nunique"
                                                               else f"{s['fn']} of {s['column']}")
        return what + (f" by {s['by']}" if s.get("by") else f" per {s['per']}" if s.get("per") else "") + sc
    if op == "ratio":
        return f"({ref(s['a'])}) ÷ ({ref(s['b'])})"
    if op == "rank":
        return f"{'lowest' if s.get('order') == 'asc' else 'highest'} {s.get('n', 5)}: {ref(s['from'])}"
    if op == "project":
        return f"{s['column']} projected over {s['days']:g} days{sc}"
    if op == "compare":
        return f"{ref(s['a'])} against target {s['target']:g}" if "target" in s else f"{ref(s['a'])} vs {ref(s['b'])}"
    if op == "associate":
        return f"what is associated with {s['target']}{sc}"
    if op == "anomalies":
        return f"unusual values of {s['column']}{sc}"
    if op == "profile":
        return f"data quality{sc}"
    return f"{op}{sc}"


def _short(r: P.Result, fmt) -> str:
    v = r.value
    u = f" {r.unit}" if r.unit and r.unit not in ("records",) and not str(r.unit).startswith("rate") else ""
    if r.kind == "scalar":
        return "no value" if v is None else fmt(v) + u
    if r.kind == "projection":
        return f"expected {fmt(v['expected'])}{u}" + (f" (range {fmt(v['low'])}–{fmt(v['high'])})" if v.get("low") is not None else "")
    if r.kind == "comparison":
        return v.get("verdict") or f"difference {fmt(v['difference'])}{u}"
    if r.kind == "table":
        return f"{len(v):,} records"
    if r.provenance.get("op") == "associate":
        return f"{len(v['factors'])} columns tested"
    if r.provenance.get("op") == "anomalies":
        return f"{v['count']} unusual of {v['checked']}"
    if r.provenance.get("op") == "profile":
        return f"{sum(v['missing'].values())} missing values, {len(v['issues'])} kinds of impossible record"
    if isinstance(v, dict):
        return f"{len(v)} groups"
    return ""


def render(p, steps, res, labels, df, excluded) -> str:
    from app.analyst import associations
    from app.analyst.explain import f as fmt
    ans_id = p["answer"]
    a = res[ans_id]
    out = []
    if p.get("objective"):
        out.append(f"_The local model read the request as: {p['objective']}_")
    out.append(_answer_text(a, labels[ans_id], fmt))
    req = p.get("request")
    ext = _extremes(a, req, fmt) if a.kind == "series" else None
    if ext:
        out.append(ext)
    if a.hypothetical:
        out.append("_This is a **hypothetical scenario**: the records were not changed and nothing here was observed._")
    # the other results the program computed, in order: a composite question is answered by all of them
    others = [s for s in steps if s["id"] != ans_id and res[s["id"]].kind != "table"]
    for s in others:
        r = res[s["id"]]
        if r.provenance.get("op") == "associate":
            out.append(associations.text(r.value))
        elif r.kind == "series" and r.provenance.get("op") not in ("anomalies", "profile"):
            out.append(_answer_text(r, labels[s["id"]], fmt))
        else:
            out.append(f"- **{labels[s['id']]}:** {_short(r, fmt)}" + (" _(hypothetical)_" if r.hypothetical else ""))
    if req is not None:
        out.append(_checklist(req, steps, res, labels, ans_id))
    if any(s["op"] == "project" for s in steps):
        out.append("_A projection carries the historical daily pace forward (days with no record count as 0); it is "
                   "an estimate with a range, not a certainty and not an observed result._")
    if any(s["op"] == "associate" for s in steps) and a.provenance.get("op") != "associate":
        out.append("_Associations are patterns in this data, not causes._")
    if any(s.get("period") for s in steps if s["op"] == "select"):
        out.append(f"_Periods: {anchor_date(df)[1]}._")
    drop = [e for e in excluded if not e.get("kept") and not str(e.get("reason", "")).startswith("exact duplicate")]
    if drop:
        out.append("_Left out as impossible: " + "; ".join(f"{e['records']} record(s) — {e['reason']}" for e in drop) + "._")
    dropped_dups = [e for e in excluded if not e.get("kept") and str(e.get("reason", "")).startswith("exact duplicate")]
    if dropped_dups:
        out.append(f"_Left out, as you asked: {dropped_dups[0]['records']} exact duplicate record(s) (lines "
                   f"{', '.join(map(str, dropped_dups[0].get('lines') or []))})._")
    for e in [e for e in excluded if e.get("kept") and e.get("negative")]:
        out.append(f"_{e['records']} record(s) with a negative {e['negative']} were kept as recorded (lines "
                   f"{', '.join(map(str, e['lines'][:10]))}{' …' if e['records'] > 10 else ''}) — a refund, credit or "
                   f"loss is a real amount; ask to *exclude negative {e['negative']} values* if they should not count._")
    dup = [e for e in excluded if e.get("kept") and not e.get("negative")]
    if dup:
        out.append(f"_{dup[0]['records']} exact duplicate record(s) were kept as recorded._")
    return "\n\n".join(x for x in out if x)


def _extremes(r: P.Result, req, fmt) -> str | None:
    v = {k: x for k, x in r.value.items() if x is not None}
    if len(v) < 2 or r.provenance.get("op") in ("associate", "anomalies", "profile"):
        return None
    from app.analyst.explain import extreme_text
    return extreme_text(v, _fmtv(r, fmt), lead_low=req is not None and req.extreme == "min")


def _checklist(req, steps, res, labels, ans_id) -> str:
    """Every part of the request, and where the answer gives it -- or that it could not be answered."""
    ops = {s["op"]: s["id"] for s in steps}
    parts = req.parts()
    lines = ["**What you asked, and where it is answered**"]
    lines.append(f"- {parts[0]}: ✓ {labels[ans_id]}")
    rest = parts[1:]
    for part in rest:
        if part.startswith("which"):
            ok = res[ans_id].kind == "series" or "rank" in ops
            lines.append(f"- {part}: " + ("✓ stated above from the breakdown" if ok else
                                          "✗ not answered — the result has no breakdown to rank"))
        elif part.startswith("what is associated"):
            lines.append(f"- {part}: " + (f"✓ {labels[ops['associate']]}" if "associate" in ops else
                                          "✗ not answered — no outcome column to test associations against"))
        elif part == "unusual values":
            lines.append(f"- {part}: " + (f"✓ {labels[ops['anomalies']]}" if "anomalies" in ops else
                                          "✗ not answered — no numeric measure to check"))
        elif part == "the hypothetical scenario":
            lines.append(f"- {part}: " + ("✓ computed as a scenario (hypothetical, not observed)" if "scenario" in ops
                                          else "✓ computed as a projection" if "project" in ops else "✗ not answered"))
        elif part.startswith("whether the difference is statistically"):
            # a ratio of totals, a mean or a median per group is descriptive; no step here tests it
            lines.append(f"- {part}: ✗ not tested — the figures above are descriptive and say nothing about whether "
                         "the difference is statistically or practically meaningful. Silt's analysis programs have no "
                         "significance test for this figure; ask to *compare <measure> between the groups* for a "
                         "statistical test of the per-record values.")
    return "\n".join(lines)


def _fmtv(r: P.Result, fmt):
    return (lambda x: f"{x * 100:.1f}%") if r.unit == "share" or str(r.unit or "").startswith("share of") else fmt


def _answer_text(r: P.Result, label: str, fmt) -> str:
    from app.analyst import associations
    v = r.value
    u = f" {r.unit}" if r.unit and r.unit not in ("records", "money", "count") and not str(r.unit).startswith(
        ("rate", "money per", "count per")) else ""
    op = r.provenance.get("op")
    if op == "associate":
        return associations.text(v)
    if op == "anomalies":
        ex = ", ".join(f"{fmt(x['value'])} (line {x['line']})" for x in v["examples"][:5])
        return (f"**{label}:** {v['count']} of {v['checked']} values are outside {fmt(v['low'])}–{fmt(v['high'])} "
                f"(1.5 × IQR rule; {v['extreme']} beyond 3 × IQR)." + (f" Examples: {ex}." if ex else ""))
    if op == "profile":
        miss = ", ".join(f"{c}: {n}" for c, n in v["missing"].items()) or "none"
        iss = "; ".join(f"{e['records']} — {e['reason']}" for e in v["issues"]) or "none found"
        return f"**{label}:** {v['rows']:,} records. Missing values — {miss}. Impossible records — {iss}."
    share_of = str(r.unit or "").startswith("share of ")
    if share_of:
        u = ""
    if r.kind == "scalar":
        if r.unit == "share" and v is not None:
            return f"**{label}: {v * 100:.1f}%** (share of records)"
        if share_of and v is not None:
            return f"**{label}: {v * 100:.1f}%** ({r.unit})"
        return f"**{label}: {'no value (no records)' if v is None else fmt(v) + u}**"
    fmt = _fmtv(r, fmt)
    if r.unit == "share":
        u = ""
    if r.kind == "series":
        rows = list(v.items())[:15]
        lines = [f"**{label}**" + (f" ({r.unit})" if u or share_of else " (share of records)" if r.unit == "share" else ""), "",
                 "| group | value |", "|---|---:|"]
        lines += [f"| {k} | {'—' if x is None else fmt(x)} |" for k, x in rows]
        if len(v) > 15:
            lines.append(f"| … {len(v) - 15} more | |")
        return "\n".join(lines)
    if r.kind == "projection":
        rng = f", likely range {fmt(v['low'])}–{fmt(v['high'])}" if v.get("low") is not None else ""
        return (f"**{label}: expected {fmt(v['expected'])}{u}{rng}** — from a daily average of {fmt(v['daily_mean'])} over "
                f"{v['history_days']} days ({v['history_start']} to {v['history_end']}).")
    if r.kind == "comparison" and "target" in v:
        lines = [f"**Assessment: {v['verdict']}.**", "",
                 f"- target: {fmt(v['target'])}{u}", f"- done so far: {fmt(v['completed'])}{u}",
                 f"- still needed: {fmt(v['remaining_needed'])}{u}"]
        if v.get("days_left") is not None:
            lines.append(f"- days left: {fmt(v['days_left'])}")
        if v.get("required_daily") is not None:
            lines.append(f"- required pace: {fmt(v['required_daily'])}{u} per day")
        if v.get("current_daily") is not None:
            lines.append(f"- current pace: {fmt(v['current_daily'])}{u} per day")
        lines.append(f"- expected total: {fmt(v['expected_total'])}{u}" + (
            f" (likely {fmt(v['expected_low'])}–{fmt(v['expected_high'])})" if v.get("expected_low") is not None else ""))
        return "\n".join(lines)
    if r.kind == "comparison":
        ratio = f", ratio {fmt(v['ratio'])}" if v.get("ratio") is not None else ""
        return (f"**{v['a_label']}: {fmt(v['a'])}{u} vs {v['b_label']}: {fmt(v['b'])}{u}** — difference "
                f"{fmt(v['difference'])}{u}{ratio}.")
    if r.kind == "table":
        return f"**{label}:** {len(v):,} records."
    return f"**{label}**"


def _charts(r: P.Result, label: str, res, steps) -> list[dict]:
    if r.kind == "series" and r.provenance.get("op") not in ("associate", "anomalies", "profile"):
        data = [{"group": str(k), "value": x} for k, x in r.value.items() if x is not None][:30]
        if data:
            return [{"type": "bar", "x": "group", "y": "value", "additive": False, "title": label[:80], "data": data}]
    if r.kind == "comparison" and "a" in r.value:
        return [{"type": "bar", "x": "result", "y": "value", "additive": False, "title": label[:80],
                 "data": [{"result": r.value["a_label"], "value": r.value["a"]},
                          {"result": r.value["b_label"], "value": r.value["b"]}]}]
    if r.provenance.get("op") == "associate" and r.value["factors"]:
        return [{"type": "bar", "x": "column", "y": "strength", "additive": False,
                 "title": f"Association with {r.value['target']} (strength)",
                 "data": [{"column": x["factor"], "strength": x["strength"]} for x in r.value["factors"][:12]]}]
    return []


def details(steps, res, labels, fmt) -> str:
    lines = ["**Analysis program** (written by the local model, checked and run by Silt)"]
    for s in steps:
        args = {k: v for k, v in s.items() if k not in ("id", "op", "label", "exclude")}
        lines.append(f"- `{s['id']}` **{s['op']}** {json.dumps(args, default=str)[:180]} → {_short(res[s['id']], fmt)}")
    return "\n".join(lines)


# ------------------------------------------------------------------------ gate --
# When to consult the model BEFORE the deterministic handlers: the request is shaped like more than one analysis --
# two linked sub-questions, a comparison adjusted for something, a conditional projection, a "why" about a
# comparison. Structural cues, not topics. Everything else reaches the model only after the deterministic paths found
# no answer (orchestrator: the late hook).
_TWO_PARTS = re.compile(r"(?:\band|\bthen|\balso|,)\s+(?:why|explain|what|which|how|suggest|tell|show|compare|identify|"
                        r"find|recommend|where|is|are|does|do|did|has|have|was|were|can|could|would|will)\b", re.I)
_ADJUSTED = re.compile(r"\b(?:after|when|once)\s+(?:you\s+)?(?:account(?:ing)?|adjust(?:ing)?|control(?:ling)?|"
                       r"allow(?:ing)?|correct(?:ing)?)\s+for\b|\ballowing for\b|\bfor each \w+ (?:it|they) (?:has|have|"
                       r"employs?|uses?)\b|\bout of each\b|"
                       r"\brelative to (?:their|its|the)\b|\bper (?:head|person|capita|employee|member|worker)\b|"
                       r"\bnormali[sz]\w*\b|\btaking\b[^?]{0,25}\binto account\b", re.I)
_CONDITIONAL = re.compile(r"\b(?:if|assuming|suppose|provided)\b[^?]{3,80}\b(?:continu\w*|keeps?|stays?|carr(?:y|ies) on|"
                          r"goes on|current (?:rate|pace|trend)|same (?:rate|pace))\b", re.I)
_COUNTERFACTUAL = re.compile(r"\b(?:if|suppose|imagine|assuming|what if)\b[^?]{2,80}\b(?:were|was|had been|closed|"
                             r"shut|stopped|removed|dropped|rose|fell|doubled|halved|went (?:up|down))\b", re.I)
_WHY_CMP = re.compile(r"\bwhy\b[^?]{0,60}\b(?:poor\w*|low\w*|high\w*|worse|better|under\w*|behind|struggl\w*|best|worst)\b|"
                      r"\b(?:poor\w*|worst|underperform\w*|struggl\w*|lagging)\b[^?]{0,60}\bwhy\b", re.I)


def unapplied(message: str, df: pd.DataFrame) -> list[str]:
    """Restrictions the request states that only an analysis program applies (a relative period, "only <yes/no
    column>"): if no program could be built, no other path may answer over all the records instead."""
    try:
        r = intent.read(message, df)
    except Exception:  # noqa: BLE001
        return []
    out = [r.period.get("label") or "the period named"] if r.period and not r.period.get("unreadable") else []
    out += intent.stated_filters_text(r)
    if r.share_of:
        out.append(f"a share of {r.share_of}'s total")
    return list(dict.fromkeys(out))


def composite(message: str, df: pd.DataFrame | None = None) -> bool:
    if df is not None:
        try:
            r = intent.read(message, df)
            # (a stated threshold alone does not route here: the calculator applies thresholds already, and the
            # verifier sends anything that ignored one to Silt's own program)
            if r.per or r.restrict or r.period or (r.compared and r.measures) or r.restrict_not or r.share_of:
                return True             # a per-unit figure, a restricted subset or a relative period: single-purpose
                                        # paths miss these
            roles = semantics.profile(df)["roles"]
            if r.quantity == "rate" and r.entities and r.measures and all(
                    roles[c]["role"] == "flag" for c in r.measures if c in roles):
                return True             # "what fraction of Mei Lin's trips were late": the person is the filter
        except Exception:  # noqa: BLE001
            pass
    return bool(_TWO_PARTS.search(message) or _ADJUSTED.search(message) or _CONDITIONAL.search(message)
                or _COUNTERFACTUAL.search(message) or _WHY_CMP.search(message))


_BACK = re.compile(r"\b(?:what if|suppose|imagine|assume|assuming|instead|that|this|it|those|these|previous|earlier|"
                   r"original|first|second|last|before|again|scenario|finding|goal|target)\b", re.I)


def follow_up(message: str, memory: dict) -> bool:
    """A short message that refers back to an analysis program, scenario or finding in this conversation ("what if the
    target changes to 320?", "set that next to the original ask"): the planner, which sees those objects, reads it
    before single-purpose paths that do not."""
    has = memory.get("objective") or memory.get("planned") or memory.get("investigation")
    return bool(has) and len(message.split()) <= 25 and bool(_BACK.search(message)) \
        and memory.get("last_analysis") != "calc"
