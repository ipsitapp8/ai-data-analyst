"""Silt calculator: exact arithmetic for free-form questions, planned by the model.

The analyst model reads the question and writes a calculation plan (JSON):

    {"type": "calc",
     "derive":    {"labor_cost": "hours_between(clock_in, clock_out) * hourly_rate"},
     "exclude":   [{"reason": "negative hourly rate", "where": "hourly_rate < 0"}],
     "group_by":  "department",
     "aggregate": {"labor_cost": "sum(labor_cost)"},
     "sort":      "-labor_cost",
     "explain":   "..."}

This module checks the plan against the dataset and the question and runs it.
It does not plan, loop or call anything: expressions are parsed with `ast` and
evaluated by a small whitelist (column names, literals, arithmetic, comparisons,
and/or/not and the functions in FUNCTIONS). There is no `eval`, no attribute
access and no imports, so a plan can compute but cannot do anything else.

`check` also refuses constants the user never stated (a 4B model likes to
invent a 0.5-hour break deduction), so the numbers come from the data and the
question only.
"""
from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd

from app.analyst.data import kind_of

MAX_EXPR = 300
MAX_DERIVE = 8
MAX_EXCLUDE = 10
MAX_AGG = 8
MAX_EXCLUDED_SHOWN = 50
AGG_FUNCS = {"sum": "sum", "total": "sum", "mean": "mean", "avg": "mean", "average": "mean", "min": "min",
             "max": "max", "count": "count", "median": "median", "nunique": "nunique"}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
# Constants that are units, not assumptions: they may appear without being stated.
FREE_CONSTANTS = {0.0, 1.0, 100.0}


class CalcError(ValueError):
    """The plan cannot be run as written; the message says why (shown to the model on a retry)."""


# ------------------------------------------------------------------ parsing --

def parse(text: str) -> dict | None:
    """The first JSON object in the model's reply, normalised; None if there is none."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("type") not in ("calc", "decline", "clarify"):
        return None
    if obj["type"] != "calc":
        return obj
    derive = obj.get("derive") or {}
    if isinstance(derive, list):       # [{"name": .., "expression": ..}] is a common variant
        derive = {d.get("name"): d.get("expression") or d.get("expr") or d.get("formula")
                  for d in derive if isinstance(d, dict)}
    exclude = obj.get("exclude") or []
    if isinstance(exclude, dict):
        exclude = [{"reason": k, "where": v} for k, v in exclude.items()]
    agg = obj.get("aggregate") or {}
    if isinstance(agg, list):
        agg = {a.get("name"): a.get("expression") or a.get("expr") for a in agg if isinstance(a, dict)}
    return {"type": "calc", "derive": {str(k): str(v) for k, v in derive.items() if k and v},
            "exclude": [{"reason": str(e.get("reason") or e.get("where")), "where": str(e.get("where"))}
                        for e in exclude if isinstance(e, dict) and e.get("where")],
            "group_by": obj.get("group_by") or None, "aggregate": {str(k): str(v) for k, v in agg.items() if k and v},
            "sort": obj.get("sort") or None, "explain": str(obj.get("explain") or "")}


# --------------------------------------------------------------- expressions --

def _prepare(expr: str, names: list[str]) -> tuple[str, dict]:
    """Backticked or space-containing column names become safe identifiers."""
    alias = {}
    def sub(m):
        key = f"__c{len(alias)}"
        alias[key] = m.group(1)
        return key
    expr = re.sub(r"`([^`]+)`", sub, expr)
    for n in sorted((n for n in names if not n.isidentifier()), key=len, reverse=True):
        if n in expr:
            key = f"__c{len(alias)}"
            alias[key] = n
            expr = expr.replace(n, key)
    expr = re.sub(r"\bAND\b", "and", re.sub(r"\bOR\b", "or", re.sub(r"\bNOT\b", "not", expr)))
    return quote_times(expr), alias


_BARE_TIME = re.compile(r"(?<![\w.'\"])(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?|\d{1,2}\s*[ap]\.?m\.?)(?![\w'\"])", re.I)


def quote_times(expr: str) -> str:
    """09:00 or 9 AM written without quotes is not valid syntax; quote it so it is read as a clock time.
    Text already inside quotes is left alone."""
    parts = re.split(r"('[^']*'|\"[^\"]*\")", expr)
    return "".join(p if i % 2 else _BARE_TIME.sub(lambda m: "'" + m.group(1).strip() + "'", p)
                   for i, p in enumerate(parts))


def balance(expr: str) -> str:
    """Close brackets the model left open at the end ("sum(hours_between(a, b)"): syntax only, never meaning."""
    missing = expr.count("(") - expr.count(")")
    return expr + ")" * missing if 0 < missing <= 3 else expr


def _tree(expr: str, names: list[str]) -> tuple[ast.Expression, dict]:
    expr = balance(expr)
    if len(expr) > MAX_EXPR:
        raise CalcError(f"The expression '{expr[:60]}…' is too long.")
    src, alias = _prepare(expr.strip(), names)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"'{expr}' is not a valid expression.") from e
    return tree, alias


# Clock times: "09:00", "9:30", "17:45:30", "8 AM", "5:30 pm", "noon", "midnight". A bare number is never a time.
_TIME_TEXT = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*([ap])\.?\s*m\.?\s*$|^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$",
                        re.I)


def parse_time(value) -> float | None:
    """Hours after midnight for a clock time, or None if `value` is not one (invalid times such as 25:00 or 9:75
    are None too). 24:00 is accepted as the end of the day."""
    if value is None or isinstance(value, (int, float, np.integer, np.floating, bool)):
        return None
    text = str(value).strip().lower()
    if text in ("noon", "midday"):
        return 12.0
    if text == "midnight":
        return 0.0
    m = _TIME_TEXT.match(text)
    if not m:
        return None
    if m.group(4):                                      # 12-hour clock
        h, mi, se = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if m.group(4) == "p" else 0)
    else:
        h, mi, se = int(m.group(5)), int(m.group(6)), int(m.group(7) or 0)
    if mi > 59 or se > 59 or h > 24 or (h == 24 and (mi or se)):
        return None
    return h + mi / 60 + se / 3600


def _to_hours(s: pd.Series) -> pd.Series:
    """Clock times or datetimes in a column, as hours after midnight (NaN where a value is missing or invalid)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - s.dt.normalize()).dt.total_seconds() / 3600
    return s.map(parse_time).astype(float)


def _as_datetime(s):
    if isinstance(s, pd.Series):
        return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")
    if parse_time(s) is not None:
        raise CalcError(f"'{s}' is a clock time, not a date.")
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise CalcError(f"'{s}' is not a date (use 'YYYY-MM-DD').")
    return ts


def _date_part(d, part: str):
    v = _as_datetime(d)
    return getattr(v.dt, part) if isinstance(v, pd.Series) else getattr(v, part)


def _clock(x, which: str):
    """One side of a duration, as hours after midnight: a time column or one clock time. Numbers are refused:
    9 or 9.5 could mean a time or a duration, and guessing would silently change the answer."""
    if isinstance(x, pd.Series):
        if pd.api.types.is_numeric_dtype(x):
            raise CalcError(f"the {which} column holds numbers, not clock times; hours_between needs times like '09:00'.")
        h = _to_hours(x)
        if len(x) and h.isna().all():
            raise CalcError(f"the {which} column has no clock times like '09:00'.")
        return h
    if isinstance(x, str):
        h = parse_time(x)
        if h is None:
            raise CalcError(f"'{x}' is not a valid clock time (use 'HH:MM' such as '09:00', or '9 AM').")
        return h
    raise CalcError(f"the {which} of hours_between must be a clock time such as '09:00' or a time column, "
                    f"not the number {x!r}.")


def _hours_between(a, b):
    """end - start in hours. Negative when the end is earlier on the clock (a record error, or a shift past
    midnight: use overnight_hours for those)."""
    a_dt = isinstance(a, pd.Series) and pd.api.types.is_datetime64_any_dtype(a)
    b_dt = isinstance(b, pd.Series) and pd.api.types.is_datetime64_any_dtype(b)
    if a_dt and b_dt:
        return (b - a).dt.total_seconds() / 3600
    return _clock(b, "end") - _clock(a, "start")


def _overnight_hours(a, b):
    """Like hours_between, but an end earlier on the clock than the start is taken to be the next day
    ('22:00' to '06:00' is 8 hours). Full date-times already carry their day and are not changed."""
    a_dt = isinstance(a, pd.Series) and pd.api.types.is_datetime64_any_dtype(a)
    b_dt = isinstance(b, pd.Series) and pd.api.types.is_datetime64_any_dtype(b)
    d = _hours_between(a, b)
    if a_dt and b_dt:
        return d
    return d.where(d >= 0, d + 24) if isinstance(d, pd.Series) else (d if d >= 0 else d + 24)


def _looks_like_time(v) -> bool:
    if isinstance(v, str):
        return parse_time(v) is not None
    return isinstance(v, pd.Series) and not pd.api.types.is_numeric_dtype(v) and len(v) > 0 \
        and _to_hours(v).notna().mean() > 0.8


def _num(v):
    """Numbers for arithmetic. Clock times are refused (never read as decimals); so is text with no numbers."""
    if _looks_like_time(v) or (isinstance(v, pd.Series) and pd.api.types.is_datetime64_any_dtype(v)):
        shown = v if isinstance(v, str) else (v.name or "that column")
        raise CalcError(f"'{shown}' holds clock times or dates; use hours_between(start, end) to turn times into hours "
                        "before doing arithmetic.")
    if isinstance(v, pd.Series):
        if pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
            return v
        out = pd.to_numeric(v, errors="coerce")
        if v.notna().any() and out.isna().all():
            raise CalcError(f"'{v.name}' is text, not numbers, so it cannot be used in arithmetic.")
        return out
    return v


FUNCTIONS = {
    "hours_between": (2, _hours_between),
    "overnight_hours": (2, _overnight_hours),
    "year": (1, lambda d: _date_part(d, "year")),
    "month": (1, lambda d: _date_part(d, "month")),
    "day": (1, lambda d: _date_part(d, "day")),
    "abs": (1, lambda x: _num(x).abs() if isinstance(x, pd.Series) else abs(x)),
    "round": (2, lambda x, n: _num(x).round(int(n)) if isinstance(x, pd.Series) else round(x, int(n))),
    "isnull": (1, lambda x: x.isna()),
    "lower": (1, lambda x: x.astype("string").str.lower()),
}
_CMP = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}


class _Eval:
    def __init__(self, frame: pd.DataFrame, original: pd.DataFrame, alias: dict):
        self.f, self.orig, self.alias = frame, original, alias

    def __call__(self, node):
        m = getattr(self, "_" + type(node).__name__, None)
        if m is None:
            raise CalcError(f"'{type(node).__name__}' is not allowed in a calculation.")
        return m(node)

    def _Expression(self, n):
        return self(n.body)

    def _Constant(self, n):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float, str)):
            raise CalcError("Only numbers and quoted text are allowed as values.")
        return n.value

    def _Name(self, n):
        name = self.alias.get(n.id, n.id)
        if name in self.f.columns:
            return self.f[name]
        if n.id in ("true", "True"):
            return True
        if n.id in ("false", "False"):
            return False
        raise CalcError(f"There is no column named '{name}'.")

    def _BinOp(self, n):
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
               ast.Div: lambda a, b: a / b}
        op = ops.get(type(n.op))
        if op is None:
            raise CalcError("Only + - * / are allowed.")
        a, b = _num(self(n.left)), _num(self(n.right))
        if isinstance(a, str) or isinstance(b, str):
            raise CalcError(f"Text ('{a if isinstance(a, str) else b}') cannot be used in arithmetic.")
        if a is None or b is None:
            return np.nan
        if type(n.op) is ast.Div and not isinstance(b, pd.Series) and not isinstance(a, pd.Series) and b == 0:
            return np.nan               # x / 0 has no value (shown as "—"), never an error that loses the other groups
        with np.errstate(divide="ignore", invalid="ignore"):
            r = op(a, b)
        return r.replace([np.inf, -np.inf], np.nan) if isinstance(r, pd.Series) else r

    def _UnaryOp(self, n):
        v = self(n.operand)
        if isinstance(n.op, ast.USub):
            return -_num(v)
        if isinstance(n.op, ast.UAdd):
            return _num(v)
        if isinstance(n.op, ast.Not):
            return ~v.fillna(False).astype(bool) if isinstance(v, pd.Series) else not v
        raise CalcError("Unsupported operator.")

    def _BoolOp(self, n):
        vals = [self(v) for v in n.values]
        vals = [v.fillna(False).astype(bool) if isinstance(v, pd.Series) else bool(v) for v in vals]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(n.op, ast.And) else (out | v)
        return out

    def _Compare(self, n):
        left, result = self(n.left), None
        for op, right_node in zip(n.ops, n.comparators):
            right = self(right_node)
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right, list):
                    raise CalcError("'in' needs a list such as ['a', 'b'].")
                r = left.astype(str).isin([str(x) for x in right])
                r = ~r if isinstance(op, ast.NotIn) else r
            else:
                name = _CMP.get(type(op))
                if name is None:
                    raise CalcError("Unsupported comparison.")
                a, b = _align(left, right)
                r = getattr(a, name)(b) if isinstance(a, pd.Series) else getattr(b, _flip(name))(a)
            result = r if result is None else (result & r)
            left = right
        return result.fillna(False).astype(bool) if isinstance(result, pd.Series) else result

    def _List(self, n):
        return [self(e) for e in n.elts]

    _Tuple = _List

    def _Call(self, n):
        if not isinstance(n.func, ast.Name):
            raise CalcError("Only the listed functions can be called.")
        fname = n.func.id.lower()
        if fname == "duplicate":
            return self.orig.duplicated(keep="first").reindex(self.f.index, fill_value=False)
        if fname not in FUNCTIONS:
            raise CalcError(f"'{n.func.id}' is not an available function (use: {', '.join(FUNCTIONS)}, duplicate).")
        arity, fn = FUNCTIONS[fname]
        if len(n.args) != arity or n.keywords:
            raise CalcError(f"{fname} takes {arity} argument{'s' if arity > 1 else ''}.")
        # hours_between('clock_in', 'clock_out'): a quoted argument that names a column means that column
        args = [self.f[a.value] if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value in self.f.columns
                else self(a) for a in n.args]
        return fn(*args)


def _flip(name: str) -> str:
    return {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}.get(name, name)


def _align(a, b):
    """Make the two sides of a comparison comparable: clock times as hours, dates as dates, text as numbers."""
    if isinstance(a, pd.Series) and _looks_like_time(a) and (isinstance(b, pd.Series) and _looks_like_time(b)
                                                            or isinstance(b, str)):
        return _to_hours(a), (_to_hours(b) if isinstance(b, pd.Series) else _clock(b, "time"))
    if isinstance(b, pd.Series) and _looks_like_time(b) and isinstance(a, str):
        return _clock(a, "time"), _to_hours(b)
    for x, y in ((a, b), (b, a)):
        if isinstance(x, pd.Series) and pd.api.types.is_datetime64_any_dtype(x) and isinstance(y, str):
            ts = pd.to_datetime(y, errors="coerce")
            if pd.notna(ts):
                return (x, ts) if x is a else (ts, x)
    if isinstance(a, pd.Series) and isinstance(b, (int, float)) and not pd.api.types.is_numeric_dtype(a):
        return pd.to_numeric(a, errors="coerce"), b
    if isinstance(a, pd.Series) and isinstance(b, str):
        return a.astype("string").str.strip(), b
    return a, b


def evaluate(expr: str, frame: pd.DataFrame, original: pd.DataFrame):
    tree, alias = _tree(expr, list(map(str, frame.columns)))
    try:
        return _Eval(frame, original, alias)(tree)
    except CalcError:
        raise
    except Exception as e:  # noqa: BLE001 - report, never crash the turn on a malformed expression
        raise CalcError(f"'{expr}' could not be computed ({type(e).__name__}: {e}).") from e


# ------------------------------------------------------------------ checking --

def _constants(expr: str) -> list[float | str]:
    try:
        tree = ast.parse(_prepare(expr, [])[0], mode="eval")
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round":
            node.args = node.args[:1]      # the decimals argument of round() is formatting, not data
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            out.append(float(node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", node.value):
            out.append(node.value)
    return out


def stated_numbers(message: str) -> set[float]:
    nums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", message or "")}
    nums |= {n / 100 for n in nums} | {1 + n / 100 for n in nums} | {1 - n / 100 for n in nums}
    nums |= {float(i) for m, i in MONTHS.items() if re.search(rf"\b{m[:3]}", message or "", re.I)}
    # "a 30-minute break" is 0.5 hours; "half an hour" is 0.5
    nums |= {float(x) / 60 for x in re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*(?:minutes?|mins?)\b", message or "", re.I)}
    if re.search(r"half an? hour", message or "", re.I):
        nums.add(0.5)
    nums |= unit_factors(message)
    return nums


_UNIT_FACTORS = ((r"\bminutes?\b|\bmins?\b", (60.0,)), (r"\bseconds?\b|\bsecs?\b", (60.0, 3600.0)),
                 (r"\bdays?\b", (24.0,)), (r"\bweeks?\b", (7.0,)), (r"\bthousands?\b|\b\d+\s*k\b", (1000.0,)),
                 (r"\bmillions?\b", (1_000_000.0,)), (r"\bper\s?cent|percentage\b", (100.0,)))


def unit_factors(message: str) -> set[float]:
    """Conversion factors a unit named in the question justifies ("labor time in minutes" -> 60)."""
    return {f for pat, fs in _UNIT_FACTORS if re.search(pat, message or "", re.I) for f in fs}


_TIME_IN_TEXT = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?|\d{1,2}\s*[ap]\.?m\.?|noon|midnight)(?![\w])", re.I)


def stated_times(message: str) -> set[float]:
    return {h for h in (parse_time(t) for t in _TIME_IN_TEXT.findall(message or "")) if h is not None}


def _time_literals(expr: str) -> list[tuple[str, float | None]]:
    try:
        tree = ast.parse(_prepare(expr, [])[0], mode="eval")
    except SyntaxError:
        return []
    return [(n.value, parse_time(n.value)) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and (":" in n.value or re.search(r"\d\s*[ap]\.?m", n.value, re.I))]


def literal_durations(exprs: list[str], already: dict) -> list[dict]:
    """Durations written with clock times inside a result formula, e.g. hours_between('09:00', '17:30') = 8.5, so
    the answer (and the check on the model's wording) can state them."""
    out, seen = [], {_norm(v) for v in already.values()}
    for e in exprs:
        for m in re.finditer(r"(hours_between|overnight_hours)\(\s*(['\"][^'\"]+['\"])\s*,\s*(['\"][^'\"]+['\"])\s*\)", e):
            call = m.group(0)
            if _norm(call) in seen:
                continue
            seen.add(_norm(call))
            try:
                v = evaluate(call, pd.DataFrame(index=[0]), pd.DataFrame(index=[0]))
            except CalcError:
                continue
            v = float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
            out.append({"name": f"hours from {m.group(2).strip(chr(39) + chr(34))} to {m.group(3).strip(chr(39) + chr(34))}",
                        "formula": call, "value": v})
    return out


_SEL_NE = re.compile(r"^\s*\(?\s*(\w+)\s*(?:!=\s*['\"][^'\"]+['\"]|not\s+in\s*\[[^\]]*\])\s*\)?\s*$")
_ALL_SCOPE = re.compile(r"\b(every(one|body)?|all (the )?(employees|staff|people|records|rows|departments|teams)|overall|"
                        r"whole|entire|company|each|per|by|which|who|highest|lowest|most|least|top|rank\w*|compare\w*)\b",
                        re.I)
_ELLIPTIC = re.compile(r"\b(he|she|his|her|him|they|them|their|it|its|that|this|same|those|these)\b|^\s*(and|also|now|"
                       r"what about|how about|excluding|without|except|only|but)\b", re.I)


def carried_selections(history: list, exclude: list[dict], frame: pd.DataFrame, message: str) -> list[dict]:
    """The person or group the previous question selected ("only Rahul Sharma"), when the follow-up refers back to
    it ("excluding duplicates, what is it?") without naming anyone else or widening the scope."""
    if not history or not _ELLIPTIC.search(message) or _ALL_SCOPE.search(message):
        return []
    named_cols = {c for c, _ in named_values(frame, message)}
    have = {e["where"].replace(" ", "") for e in exclude}
    out = []
    for e in history[-1][1].get("exclude") or []:
        m = _SEL_NE.match(e.get("where", ""))
        if not m or m.group(1) not in frame.columns or m.group(1) in named_cols:
            continue
        if e["where"].replace(" ", "") not in have and re.search(r"[A-Za-z]", e["where"].split(m.group(1), 1)[1]):
            out.append({"reason": e.get("reason") or "kept from the previous question", "where": e["where"]})
    return out


def _time_problems(exprs: list[str], message: str) -> list[str]:
    """Clock times must be valid and stated by the user; a literal shift that ends before it starts must use
    overnight_hours (with hours_between it would come out negative)."""
    out, said = [], stated_times(message)
    for e in exprs:
        for text, h in _time_literals(e):
            if h is None:
                out.append(f"'{text}' in '{e}' is not a valid clock time")
            elif not any(abs(h - x) < 1e-9 or abs(h - x) == 24 for x in said):
                out.append(f"the time '{text}' in '{e}' was not stated by the user; do not add time rules the user did "
                           "not give (for record clock times use hours_between(start_column, end_column))")
        try:
            tree = ast.parse(_prepare(e, [])[0], mode="eval")
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hours_between"
                    and len(n.args) == 2 and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in n.args)):
                start, end = parse_time(n.args[0].value), parse_time(n.args[1].value)
                if start is not None and end is not None and end < start:
                    out.append(f"the shift from '{n.args[0].value}' to '{n.args[1].value}' ends after midnight: use "
                               f"overnight_hours('{n.args[0].value}', '{n.args[1].value}')")
    return out


def _norm(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def check(spec: dict, df: pd.DataFrame, message: str, earlier: str = "", reused: frozenset = frozenset()) -> list[str]:
    """Problems that stop the plan from running as the user asked; empty if it can run.
    Exclusions may use numbers stated anywhere in the conversation (`earlier` + `message`): a follow-up keeps
    "September 2026". A new formula may use only numbers and times from the current `message`, unless it repeats an
    earlier formula verbatim (`reused`): an earlier question's assumption must not leak into a new scenario."""
    problems = []
    whole = (earlier + " " + message).strip()
    if len(spec["derive"]) > MAX_DERIVE or len(spec["exclude"]) > MAX_EXCLUDE or len(spec["aggregate"]) > MAX_AGG:
        problems.append("The plan is too large.")
    formulas = [e for e in list(spec["derive"].values()) + list(spec["aggregate"].values()) if _norm(e) not in reused]
    wheres = [e["where"] for e in spec["exclude"]]
    problems += _constant_problems(wheres, whole) + _time_problems(wheres, whole)
    problems += _constant_problems(formulas, message) + _time_problems(formulas, message)
    if not spec["aggregate"]:
        asked = asked_quantities(message)
        problems.append("the plan has no 'aggregate', so it reports no result" +
                        (f"; the question asks for: {', '.join(asked)}" if asked else ""))
    names = list(map(str, df.columns)) + list(spec["derive"])
    if spec["group_by"] and spec["group_by"] not in names:
        problems.append(f"group_by '{spec['group_by']}' is not a column")
    try:
        run(spec, df, dry=True)
    except CalcError as e:
        problems.append(str(e))
    return list(dict.fromkeys(problems))


_BREAK = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(minutes?|mins?|hours?|hrs?)\b[^.?!\d]{0,25}?\bbreak|"
                    r"\bbreak\b[^.?!]{0,12}?\b(?:of\s+)?(\d+(?:\.\d+)?)\s*-?\s*(minutes?|mins?|hours?|hrs?)\b|"
                    r"(half an? hour)[^.?!]{0,25}?\bbreak", re.I)


def stated_break_hours(text: str) -> float | None:
    """The unpaid break the user states, in hours ("a 30-minute unpaid break" -> 0.5); None if none is stated."""
    m = _BREAK.search(text or "")
    if not m:
        return None
    if m.group(5):
        return 0.5
    num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    v = float(num)
    return v / 60 if unit.lower().startswith("m") else v


def break_problems(spec: dict, question: str) -> list[str]:
    """A stated break has to change the hours: somewhere the plan must subtract it (as a number or as a derived
    break column). Otherwise the answer would silently ignore part of the question."""
    b = stated_break_hours(question)
    if not b or re.search(r"\bno (unpaid )?break", question or "", re.I):
        return []
    exprs = list(spec["derive"].values()) + list(spec["aggregate"].values())
    break_cols = [k for k, v in spec["derive"].items() if _constants(v) and all(abs(c - b) < 1e-9 for c in _constants(v)
                                                                                if isinstance(c, float))]
    pattern = "|".join([re.escape(f"{b:g}"), re.escape(str(b))] + [rf"\b{re.escape(k)}\b" for k in break_cols])
    if any(re.search(rf"-\s*\(?\s*(?:{pattern})", e) for e in exprs):
        return []
    return [f"the question states an unpaid break of {b:g} hours, but no formula subtracts it; subtract it from the "
            f"working hours (e.g. hours - {b:g}) before multiplying by the rate"]


def asked_quantities(message: str) -> list[str]:
    stop = r"(?:and|across|for|of|per|by|in|on|from|with|over|as|the|all|each|is|are|,|\.)"
    found = re.findall(rf"\b(?:total|sum of|average|mean|overall)\s+((?:(?!{stop}\b)[a-z_]+\s?){{1,3}})", message.lower())
    return list(dict.fromkeys(q.strip() for q in found if q.strip()))


# ------------------------------------------------------------------- running --

_AGG = re.compile(r"^(\w+)\s*\((.*)\)$", re.S)


class _AggEval(_Eval):
    """A result: aggregates over the kept records (sum(x), mean(x) ...) combined with numbers, stated assumptions
    (derived columns that hold one value, e.g. an assumed shift length) and functions of literals, e.g.
    mean(hourly_rate) * hours_between('09:00', '18:00'). A per-record column outside an aggregate is refused:
    it has many values, so the result would be ambiguous."""

    def __init__(self, frame, original, alias, constants):
        super().__init__(frame, original, alias)
        self.constants = constants

    def _Name(self, n):
        name = self.alias.get(n.id, n.id)
        if name in self.constants:
            return self.constants[name]
        if name in self.f.columns:
            raise CalcError(f"'{name}' has a value per record, so it must be inside an aggregate such as "
                            f"mean({name}) or sum({name}).")
        return super()._Name(n)

    def _Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id.lower() in AGG_FUNCS:
            if len(n.args) != 1 or n.keywords:
                raise CalcError(f"{n.func.id}() takes one expression.")
            inner = _Eval(self.f, self.orig, self.alias)(n.args[0])
            series = inner if isinstance(inner, pd.Series) else pd.Series(inner, index=self.f.index)
            r = _aggregate(AGG_FUNCS[n.func.id.lower()], series)
            return np.nan if r is None else r
        return super()._Call(n)


def agg_value(expr: str, frame: pd.DataFrame, original: pd.DataFrame, constants: dict | None = None):
    src = re.sub(r"\(\s*\*\s*\)", "(1)", expr)                 # count(*)
    tree, alias = _tree(src, list(map(str, frame.columns)))
    try:
        v = _AggEval(frame, original, alias, constants or {})(tree)
    except CalcError:
        raise
    except Exception as e:  # noqa: BLE001
        raise CalcError(f"'{expr}' could not be computed ({type(e).__name__}: {e}).") from e
    if isinstance(v, pd.Series):
        raise CalcError(f"'{expr}' gives a value per record, not one result; wrap it in an aggregate such as sum(...).")
    if isinstance(v, (bool, np.bool_)) or isinstance(v, str):
        raise CalcError(f"'{expr}' is not a number.")
    v = float(v)
    return None if np.isnan(v) or np.isinf(v) else (int(v) if v.is_integer() and "count(" in expr.replace(" ", "") else v)


def _names_in(expr: str) -> set[str]:
    try:
        tree = ast.parse(_prepare(re.sub(r"\(\s*\*\s*\)", "(1)", expr), [])[0], mode="eval")
    except SyntaxError:
        return set()
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in called}


def _scenario_columns(derive: dict, frame) -> list[str]:
    """Derived columns built on a stated assumption (a literal clock time, or an assumption column): per record they
    hold 'what this record would give under the assumption', so the history gives one estimate by averaging."""
    base = [c for c in map(str, frame.columns) if c not in derive]
    assumed = assumption_columns(derive, base)
    out = []
    for name, expr in derive.items():
        if name in assumed:
            continue
        if _time_literals(expr) or (_names_in(expr) & (set(assumed) | set(out))):
            out.append(name)
    return out


def assumption_columns(derive: dict, columns) -> dict:
    """Derived columns that refer to no data column: a single stated value (an assumed shift, a new rate)."""
    cols, out = set(map(str, columns)), {}
    for name, expr in derive.items():
        refs = _names_in(expr) - {"true", "false", "True", "False"}
        if all(r in out for r in refs) and not any(r in cols for r in refs):
            out[name] = expr
    return out


def _agg_match(a: str):
    return _AGG.match(balance(a.strip()))


def _balanced_inner(inner: str) -> bool:
    """True if `inner` is one balanced expression, i.e. sum(a) * b is not mistaken for sum of 'a) * b'."""
    depth = 0
    for ch in inner:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


_HYPOTHETICAL = re.compile(r"\b(what if|what would|estimate\w*|assum\w*|suppose|if (he|she|they|we|it|\w+) (works?|worked|"
                           r"were|was|had|increas\w*|decreas\w*)|hypothetical|scenario|would (be|cost))\b", re.I)


def _aggregate(func: str, values: pd.Series):
    v = pd.to_numeric(values, errors="coerce") if func not in ("count", "nunique") else values
    if func == "sum" and len(v) and v.notna().sum() == 0:
        return None                     # a total of nothing but missing values is unknown, not 0
    r = getattr(v, func)()
    return None if r is None or (isinstance(r, float) and np.isnan(r)) else (int(r) if func in ("count", "nunique") else float(r))


def run(spec: dict, df: pd.DataFrame, dry: bool = False, question: str = "") -> dict:
    original = df.copy()
    frame = df.copy()
    formulas = []
    for name, expr in spec["derive"].items():
        v = evaluate(expr, frame, original)
        frame[name] = v if isinstance(v, pd.Series) else pd.Series(v, index=frame.index)
        formulas.append({"column": name, "formula": expr})
    flags = pd.DataFrame(index=frame.index)
    rules = []
    for i, rule in enumerate(spec["exclude"]):
        hit = evaluate(rule["where"], frame, original)
        if not isinstance(hit, pd.Series):
            raise CalcError(f"The exclusion '{rule['where']}' does not test each row.")
        hit = hit.fillna(False).astype(bool)
        flags[i] = hit
        rules.append({"reason": rule["reason"], "where": rule["where"], "rows": int(hit.sum())})
    excluded = flags.any(axis=1) if rules else pd.Series(False, index=frame.index)
    valid = frame[~excluded]
    aggs, full = {}, {}
    for name, a in spec["aggregate"].items():
        m = _agg_match(a)
        simple = m and m.group(1).lower() in AGG_FUNCS and _balanced_inner(m.group(2))
        aggs[name] = (AGG_FUNCS[m.group(1).lower()], m.group(2).strip() or "1") if simple else ("expr", a)
        full[name] = balance(a.strip())
    group = spec.get("group_by")
    if group and group not in frame.columns:
        raise CalcError(f"group_by '{group}' is not a column.")
    assumed = assumption_columns(spec["derive"], df.columns)
    def constants_in(sub):
        return {k: (float(frame[k].iloc[0]) if len(frame) else np.nan) for k in assumed}
    consts = constants_in(frame)
    def values(sub, expr):
        if expr in ("*", "1"):
            return pd.Series(1, index=sub.index)
        v = evaluate(expr, sub, original)
        return v if isinstance(v, pd.Series) else pd.Series(v, index=sub.index)
    totals = {name: agg_value(full[name], valid, original, consts) for name in aggs}
    if dry:
        return {}
    groups = []
    if group:
        for key, sub in valid.groupby(group, dropna=False, sort=False):
            row = {"group": str(key), "rows": int(len(sub))}
            row.update({name: agg_value(full[name], sub, original, consts) for name in aggs})
            groups.append(row)
        sort = (spec.get("sort") or "").strip()
        key = sort.lstrip("-") if sort.lstrip("-") in aggs else (next(iter(aggs)) if aggs else "rows")
        groups.sort(key=lambda r: (r.get(key) is None, -(r.get(key) or 0) if sort.startswith("-") or not sort else (r.get(key) or 0)))
        # Per-row averages of what each total is made of, so "why is X highest?" can be answered from evidence:
        # a total is high because there are more rows, or larger values per row.
        inputs = [c for c in map(str, df.columns) if kind_of(df[c]) == "numeric"
                  and any(re.search(rf"\b{re.escape(c)}\b", e) for e in list(spec["derive"].values()) + [x for _, x in aggs.values()])]
        for r in groups:
            sub = valid[valid[group].astype(str) == r["group"]] if r["group"] != "nan" else valid[valid[group].isna()]
            avg = {f"average {c} per row": _aggregate("mean", sub[c]) for c in inputs}
            avg.update({f"average {n} per row": _aggregate("mean", values(sub, e)) for n, (f, e) in aggs.items()
                        if f == "sum"})
            r["per_row"] = avg
        sort_key = sort.lstrip("-")
        share_of = sort_key if sort_key in aggs and aggs[sort_key][0] == "sum" else \
            next((n for n, (f, _) in aggs.items() if f == "sum"), None)
        if share_of and totals.get(share_of):
            for r in groups:
                r["share_of_" + share_of] = (r[share_of] or 0) / totals[share_of]
    focus = [g for g in groups if any(val == g["group"] for col, val in named_values(df, question) if col == group)]
    why = None
    if group and len(groups) > 1 and re.search(r"\bwhy\b|\breason|\bexplain why|\bdriv", question, re.I):
        named = [n for n in aggs if n.replace("_", " ").lower() in question.lower()]
        why = _why(groups, aggs, len(valid), named[0] if named else spec.get("sort"))
    shown_cols = list(map(str, df.columns)) + list(spec["derive"])
    excluded_rows = []
    for idx in frame.index[excluded][:MAX_EXCLUDED_SHOWN]:
        rec = {c: _cell(frame.at[idx, c]) for c in shown_cols}
        rec["row"] = int(list(frame.index).index(idx)) + 2       # the line in the file (header is line 1)
        rec["excluded_because"] = "; ".join(r["reason"] for i, r in enumerate(rules) if flags.at[idx, i])
        excluded_rows.append(rec)
    # The data values each result was built from, so a reader can see e.g. whether a historical rate was
    # one consistent value or several (and which aggregation combined them).
    used_cols = [c for c in map(str, df.columns) if kind_of(df[c]) == "numeric"
                 and any(re.search(rf"\b{re.escape(c)}\b", e) for e in list(full.values())
                         + [x for k, x in spec["derive"].items() if any(re.search(rf"\b{re.escape(k)}\b", f) for f in full.values())])]
    inputs_used = []
    for c in used_cols:
        vc = pd.to_numeric(valid[c], errors="coerce").dropna().round(6).value_counts().sort_index()
        inputs_used.append({"column": c, "records": int(vc.sum()), "distinct": int(len(vc)),
                            "values": [{"value": float(k), "records": int(v)} for k, v in list(vc.items())[:8]]})
    incomplete = {k: int(frame.loc[~excluded, k].isna().sum()) for k in spec["derive"]
                  if k in frame and int(frame.loc[~excluded, k].isna().sum())}
    scenario = bool(assumed) or any(_time_literals(e) for e in full.values()) or any(
        not (_names_in(e) & set(map(str, frame.columns))) for e in full.values()) or bool(_HYPOTHETICAL.search(question))
    selected = bool(group) or any(val.lower() in e["where"].lower() for e in spec["exclude"]
                                  for _, val in named_values(df, question)) or any(
        re.search(r"(!=|==)\s*'[^']*[A-Za-z][^']*'", e["where"]) for e in spec["exclude"])
    checks = data_checks(spec, frame, valid, original, list(full.values()), used_cols)
    return {"kind": "calc", "answer_type": "scenario" if scenario else "calculation", "selected": selected,
            "data_checks": checks,
            "assumptions": [{"name": k, "formula": v, "value": consts.get(k)} for k, v in assumed.items()]
                           + literal_durations(list(spec["derive"].values()) + list(full.values()), assumed),
            "inputs_used": inputs_used, "incomplete": incomplete,
            "rows_total": int(len(frame)), "rows_excluded": int(excluded.sum()),
            "rows_valid": int(len(valid)), "formulas": formulas, "exclusions": rules, "excluded_rows": excluded_rows,
            "group_by": group, "aggregates": {n: (f"{f}({e})" if f != "expr" else e) for n, (f, e) in aggs.items()},
            "totals": totals,
            "groups": groups, "focus": focus, "why": why, "explain": spec.get("explain") or "",
            "overlap_note": ("Some rows meet more than one exclusion; each is excluded once."
                             if rules and int(flags.sum(axis=1).gt(1).sum()) else None)}


_AMOUNT_COL = re.compile(r"rate|price|cost|amount|salary|pay|wage|hours?|qty|quantity|revenue|sales|spend", re.I)


def data_checks(spec: dict, frame: pd.DataFrame, valid: pd.DataFrame, original: pd.DataFrame,
                aggregates: list[str], used_cols: list[str]) -> list[dict]:
    """Suspicious records that were INCLUDED in the result as they are, so the answer can say so instead of
    silently presenting a number built on them: exact duplicates, clock-out before clock-in (a negative
    duration), negative amounts. Lines are file lines (the header is line 1)."""
    pos = {idx: i + 2 for i, idx in enumerate(frame.index)}
    out = []
    exprs = list(spec["derive"].values()) + [e["where"] for e in spec["exclude"]] + aggregates
    if not any("duplicate(" in e for e in exprs):
        dup = original.duplicated(keep="first").reindex(valid.index, fill_value=False)
        if dup.any():
            lines = [pos[i] for i in valid.index[dup]]
            out.append({"issue": "exact duplicate of an earlier record", "records": len(lines), "lines": lines[:20],
                        "fix": "exclude duplicates"})
    seen = set()
    for e in list(spec["derive"].values()) + aggregates:
        for call in re.findall(r"hours_between\([^()]*\)", e):
            if call in seen or not any(re.search(rf"\b{re.escape(c)}\b", call) for c in map(str, original.columns)):
                continue
            seen.add(call)
            try:
                hrs = evaluate(call, valid, original)
            except CalcError:
                continue
            if isinstance(hrs, pd.Series) and (hrs < 0).any():
                lines = [pos[i] for i in valid.index[hrs.lt(0).fillna(False)]]
                cols = [c for c in map(str, original.columns) if re.search(rf"\b{re.escape(c)}\b", call)]
                out.append({"issue": (f"{cols[-1].replace('_', ' ')} earlier than {cols[0].replace('_', ' ')}"
                                      if len(cols) == 2 else "end time before start time") + " (negative hours)", "records": len(lines),
                            "lines": lines[:20], "fix": "exclude records where the end is before the start, or say "
                                                        "if these shifts run past midnight"})
    dates = [c for c in map(str, original.columns) if kind_of(original[c]) == "datetime"]
    if len(dates) == 1:
        from app.analyst.scope import today  # noqa: PLC0415
        future = pd.to_datetime(valid[dates[0]], errors="coerce") > today()
        if future.any():
            lines = [pos[i] for i in valid.index[future]]
            out.append({"issue": f"{dates[0].replace('_', ' ')} after today ({today().date()}): a planned or future "
                                 "record, not an observation", "records": len(lines), "lines": lines[:20],
                        "fix": "exclude records dated after today"})
    for c in used_cols:
        if not _AMOUNT_COL.search(c):
            continue
        v = pd.to_numeric(valid[c], errors="coerce")
        if (v < 0).any():
            lines = [pos[i] for i in valid.index[v.lt(0).fillna(False)]]
            out.append({"issue": f"negative {c.replace('_', ' ')}", "records": len(lines), "lines": lines[:20],
                        "fix": f"exclude negative {c.replace('_', ' ')} values"})
    return out


def _rank_word(rank: int, n: int, high: bool = True) -> str:
    if rank == 1:
        return "the highest" if high else "the lowest"
    if rank == n:
        return "the lowest" if high else "the highest"
    return {2: "the second highest", 3: "the third highest"}.get(rank, f"number {rank} of {n}")


def _why(groups: list[dict], aggs: dict, valid_rows: int, sort: str | None) -> dict | None:
    """Why the first group leads, in terms of its parts: how many records it has and how large they are."""
    key = (sort or "").lstrip("-")
    key = key if key in aggs else next((n for n, (f, _) in aggs.items() if f == "sum"), None)
    if not key or aggs[key][0] != "sum":
        return None
    top, n = groups[0], len(groups)
    facts = {"group": top["group"], "result": key, "value": top.get(key), "records": top["rows"],
             "records_of": valid_rows, "records_rank": 1 + sum(g["rows"] > top["rows"] for g in groups),
             "groups": n, "per_row": []}
    for k, v in (top.get("per_row") or {}).items():
        if v is None:
            continue
        rank = 1 + sum((g.get("per_row") or {}).get(k) is not None and g["per_row"][k] > v for g in groups)
        facts["per_row"].append({"measure": k, "value": v, "rank": rank})
    parts = [f"{top['group']} has {_rank_word(facts['records_rank'], n)} number of records ({top['rows']} of {valid_rows})"]
    parts += [f"{_rank_word(x['rank'], n)} {x['measure']} ({x['value']:,.2f})" for x in facts["per_row"]]
    facts["sentence"] = (f"{top['group']} has the highest {key.replace('_', ' ')}. A total is the number of records "
                         f"times their average size: " + "; ".join(parts) + ".")
    return facts


def _cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()[:10] if v == v.normalize() else v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


# ------------------------------------------------------------ model planning --
# The fine-tuned model reliably writes one short JSON section at a time but stops
# early on a long plan. So the plan is built in sections: the model chooses
# calc / decline / clarify, then writes `derive`, `exclude`, `group_by` and
# `aggregate`, each continuing its own reply so far (raw completion in the
# model's chat template). Every expression is the model's; Silt only opens each
# section and checks it before the next. A rejected section is shown back to the
# model once with the reason (visible in the answer's trace); if it is rejected
# again the calculation stops and says so rather than guessing.

CALC_SYSTEM = """You are Silt Copilot, a data analyst. You do not do arithmetic yourself: you write a calculation plan and Silt's calculator runs it exactly on the user's data.
Reply with ONE JSON object and nothing else, with these keys (omit keys you do not need):
"type": "calc"
"derive": new columns, {"name": "expression"}, computed for every row
"exclude": rows to leave out, [{"reason": "plain words", "where": "condition that is true for the rows to exclude"}]
"group_by": a column to break results down by, or null
"aggregate": the results, {"name": "sum(column)"}; functions sum mean min max count median
Expressions use column names, numbers, 'text' or 'YYYY-MM-DD' literals, + - * / ( ), < <= > >= == !=, and, or, not, and the functions
hours_between(start, end) (already in HOURS, never divide it), overnight_hours(start, end) (only for a shift the user says ends after midnight), year(d), month(d), day(d), abs(x), round(x, n), isnull(x), duplicate() (true for a row that exactly repeats an earlier row).
Clock times the user gives are quoted text: hours_between('09:00', '18:00') is 9; for a stated night shift overnight_hours('22:00', '06:00') is 8.
For an estimate under assumptions, derive each assumption as its own value and combine it with an average of the history in aggregate, e.g. "estimated_cost": "mean(hourly_rate) * shift_hours". Do not filter on the date of a future event.
Use only columns from the schema or ones you derive. Never add a number, deduction or rule that the user did not state.
"Which group has the highest X" means the total, sum(X), per group; use max only for the single largest record.
For a "why" question also aggregate the inputs (row counts, totals and averages) that explain the difference.
If the data cannot answer the question reply {"type":"decline","reason":"..."}; if it is ambiguous reply {"type":"clarify","question":"..."}.

Example. Columns: order_id, region, order_date, quantity, unit_price, status.
Question: Total revenue per region for 2025 as quantity times unit price, excluding cancelled orders and repeated rows.
{"type":"calc","derive":{"revenue":"quantity * unit_price"},"exclude":[{"reason":"exact duplicate row","where":"duplicate()"},{"reason":"cancelled order","where":"status == 'cancelled'"},{"reason":"not in 2025","where":"year(order_date) != 2025"}],"group_by":"region","aggregate":{"orders":"count(revenue)","revenue":"sum(revenue)"}}

Example. Columns: driver, trip_date, km, cost_per_km.
Question: Using Ana Lopez's past trips, estimate the cost of a 120 km trip.
{"type":"calc","derive":{"trip_km":"120"},"exclude":[{"reason":"not Ana Lopez","where":"driver != 'Ana Lopez'"}],"group_by":null,"aggregate":{"historical_cost_per_km":"mean(cost_per_km)","trip_km":"trip_km","estimated_cost":"mean(cost_per_km) * trip_km"}}"""

CHAT_TEMPLATE = ("<|im_start|>{role}\n{content}<|im_end|>\n")
SECTION_TOKENS = {"type": 12, "derive": 300, "exclude": 500, "group_by": 20, "aggregate": 300}


def _complete(prompt: str, max_tokens: int) -> str:
    import os
    import urllib.request
    from app.analyst import planner
    body = json.dumps({"model": planner.MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(planner.MODEL_URL.rstrip("/") + "/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer " + os.getenv("ANALYST_MODEL_KEY", "local")})
    with urllib.request.urlopen(req, timeout=planner.MODEL_TIMEOUT) as r:
        return json.load(r)["choices"][0]["text"]


class _Unparsed:
    def __repr__(self):
        return "<not JSON>"


UNPARSED = _Unparsed()


def _first_value(text: str):
    """The first JSON value at the start of `text` (the rest of the model's reply is ignored);
    UNPARSED if there is none (None is a valid value: "group_by": null)."""
    text = text.lstrip()
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except json.JSONDecodeError:
        # A missing closing quote before } or ] is the commonest slip ("hourly_rate < 0}"): one targeted repair.
        fixed = re.sub(r'(:\s*"[^"\n]*?)(\s*[}\]])', r'\1"\2', text, count=1)
        # A string closed without its quote before the next key: "sum(f(a, b)), "next": -> "sum(f(a, b))", "next":
        fixed = re.sub(r'(\))\s*,\s*"(\w+)"\s*:', r'\1", "\2":', fixed)
        # An unquoted expression that itself contains quotes: {"h": hours_between("a", "b") }
        fixed = re.sub(r'("[\w ]+"\s*:\s*)([a-z_][\w]*\(.*?\)[^,}\n]*?)(\s*(?:,\s*"[\w ]+"\s*:|\}))',
                       lambda m: m.group(1) + json.dumps(m.group(2).strip().replace('"', "'")) + m.group(3), fixed, flags=re.I)
        # An unquoted expression as a value: {"hours": hours_between(a, b) - 1 }
        fixed = re.sub(r'("[\w ]+"\s*:\s*)(?!["\[{\d-]|null|true|false)([^"\n]+?)(\s*(?:,\s*"[\w ]+"\s*:|[}\]]))',
                       lambda m: m.group(1) + json.dumps(m.group(2).strip()) + m.group(3), fixed)
        try:
            return json.JSONDecoder().raw_decode(fixed)[0]
        except json.JSONDecodeError:
            return UNPARSED


def context_text(memory: dict) -> str:
    prev = (memory or {}).get("last_calc")
    if not prev:
        return ""
    res = prev["result"]
    groups = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items() if not k.startswith("share_of")}
              for g in res.get("groups", [])[:12]]
    return ("Earlier in this conversation Silt ran this calculation for \"" + prev["question"][:300] + "\":\n"
            + json.dumps(prev["spec"], default=str)[:1500] + "\nIts results: totals "
            + json.dumps(res.get("totals"), default=str) + (f"; by {res['group_by']}: " + json.dumps(groups, default=str)
                                                            if res.get("group_by") else "")
            + "\nReuse its derived columns and exclusions when the new question refers to them.\n")


def _results_note(memory: dict) -> str:
    prev = memory.get("last_calc")
    if not prev:
        return ""
    res = prev["result"]
    return ("(Silt ran your previous plan: totals " + json.dumps(res.get("totals"), default=str)
            + (f"; by {res['group_by']}: " + json.dumps([{k: v for k, v in g.items() if not k.startswith("share_of")}
                                                        for g in res.get("groups", [])[:12]], default=str)
               if res.get("group_by") else "")
            + ". If the new question refers to it, keep its derived columns and exclusions.)\n")


def _schema_block(df: pd.DataFrame, memory: dict) -> str:
    from app.analyst.data import schema, schema_text
    sample = df.head(3).to_csv(index=False)
    return f"Dataset schema:\n{schema_text(schema(df))}\nFirst rows:\n{sample}" + context_text(memory)


def model_plan(message: str, df: pd.DataFrame, memory: dict | None = None) -> tuple[dict, list[dict]]:
    """(plan, trace). plan is a checked calc spec, or a decline / clarify, or {"type": "failed", "reason"}.
    trace records every section the model wrote, including rejected ones."""
    memory = memory or {}
    head = CHAT_TEMPLATE.format(role="system", content=CALC_SYSTEM)
    history = memory.get("calc_history") or []
    for i, (q, plan) in enumerate(history[-2:]):
        head += (CHAT_TEMPLATE.format(role="user", content=(_schema_block(df, {}) if i == 0 else "") + f"Question: {q}")
                 + CHAT_TEMPLATE.format(role="assistant", content=json.dumps(plan)))
    head += (CHAT_TEMPLATE.format(role="user", content=("" if history else _schema_block(df, {}))
                                  + (_results_note(memory) if history else "") + f"Question: {message}")
             + "<|im_start|>assistant\n")
    trace: list[dict] = []
    # Numbers the user stated earlier in this conversation (e.g. "September 2026") still count as stated.
    stated_text = " ".join([q for q, _ in history] + [message])
    first = _complete(head + '{"type": "', 60)
    trace.append({"section": "type", "output": first[:600]})
    if not first.startswith("calc"):
        obj = parse('{"type": "' + first)
        if obj and obj["type"] in ("decline", "clarify"):
            return obj, trace
        return {"type": "failed", "reason": "the model did not choose to calculate, decline or ask"}, trace
    built = '{"type": "calc"'
    # Formulas from earlier plans in this conversation, reusable verbatim (their numbers were checked then).
    reused = frozenset(_norm(x) for _, plan in history
                       for x in list((plan.get("derive") or {}).values()) + list((plan.get("aggregate") or {}).values()))
    spec = {"type": "calc", "derive": {}, "exclude": [], "group_by": None, "aggregate": {}, "sort": None, "explain": "",
            "_history": history}
    inherited: dict = {}
    for _, plan in history:
        for k, v in (plan.get("derive") or {}).items():
            if k not in df.columns:
                inherited[k] = v
    try:
        frame = run_derive(df, inherited) if inherited else df
    except CalcError:
        inherited, frame = {}, df
    done: set[str] = set()
    for section in ("derive", "exclude", "group_by", "aggregate"):
        opener = {"derive": "{", "exclude": "[", "aggregate": "{"}.get(section, "")
        feedback = None
        for attempt in (1, 2, 3):
            if section == "group_by" and attempt >= 2:
                opener = '"'      # the rejection was "nothing selects the named group": a column name is required
            if section == "aggregate" and attempt >= 2 and feedback and any("at least one result" in p for p in feedback["problems"]):
                opener = '{"'     # an empty result was rejected: the rewrite must name at least one result
            prompt = head + built + f', "{section}": ' + opener
            if feedback:
                prompt = (head + built + f', "{section}": ' + feedback["shown"] + "}<|im_end|>\n"
                          + CHAT_TEMPLATE.format(role="user", content="Silt's calculator rejected that " + section
                                                 + ": " + "; ".join(feedback["problems"]) + ". Write it again.")
                          + "<|im_start|>assistant\n" + built + f', "{section}": ' + opener)
            out = _complete(prompt, SECTION_TOKENS[section])
            value = _first_value(opener + out)
            problems = _check_section(section, value, spec, frame, stated_text if section == "exclude" else message,
                                      message, reused)
            trace.append({"section": section, "attempt": attempt, "output": (opener + out)[:800],
                          "accepted": not problems, "problems": problems})
            if problems:
                # Harmless, recognisable mistakes are repaired deterministically (and the repair is reported)
                # instead of costing the user another model round or the whole answer.
                fixed = repair_section(section, value, problems, spec, frame, message)
                if fixed is not None:
                    new_value, other, why = fixed
                    trial = {**spec, **(other or {})}
                    left = _check_section(section, new_value, trial, frame,
                                          stated_text if section == "exclude" else message, message, reused)
                    if not left:
                        trace[-1]["fixed_by_silt"] = True
                        spec.update(other or {})
                        value, problems = new_value, []
                        trace.append({"section": section, "attempt": attempt, "repaired": why, "accepted": True,
                                      "problems": [], "output": json.dumps(new_value, default=str)[:800]})
            if not problems:
                break
            feedback = {"problems": problems,
                        "shown": (opener + out).strip()[:600] if value is UNPARSED else json.dumps(value, default=str)}
        if problems:
            return {"type": "failed", "reason": f"the model's {section} was rejected three times ({'; '.join(problems)})"}, trace
        value = _normalise_section(section, value)
        if section == "exclude":
            carried = carried_selections(history, value, frame, message)
            if carried:
                value = value + carried
                trace.append({"section": "exclude", "carried_over": [c["where"] for c in carried]})
        spec[section] = value
        built = '{"type": "calc"' + "".join(f', "{s}": ' + json.dumps(spec[s])
                                            for s in ("derive", "exclude", "group_by", "aggregate")
                                            if s in done or s == section)
        done.add(section)
        if section == "derive":
            frame = run_derive(df, {**inherited, **value})
    # Columns derived earlier in this conversation stay defined; keep the ones this plan uses.
    used = " ".join(list(spec["derive"].values()) + [e["where"] for e in spec["exclude"]] + list(spec["aggregate"].values())
                    + [spec["group_by"] or ""])
    reused = {k: v for k, v in inherited.items() if k not in spec["derive"] and re.search(rf"\b{re.escape(k)}\b", used)}
    if reused:
        spec["derive"] = {**reused, **spec["derive"]}
        trace.append({"section": "derive", "reused_from_conversation": reused})
    spec.pop("_history", None)
    final = check(spec, df, message, earlier=" ".join(q for q, _ in history), reused=reused) + break_problems(spec, message)
    if final:
        return {"type": "failed", "reason": "; ".join(final)}, trace
    return spec, trace


def run_derive(df: pd.DataFrame, derive: dict) -> pd.DataFrame:
    frame = df.copy()
    for name, expr in derive.items():
        v = evaluate(expr, frame, df)
        frame[name] = v if isinstance(v, pd.Series) else pd.Series(v, index=frame.index)
    return frame


def _normalise_section(section: str, value):
    if section == "derive":
        if isinstance(value, list):
            value = {d.get("name"): d.get("expression") or d.get("expr") for d in value if isinstance(d, dict)}
        return {str(k): str(v) for k, v in (value or {}).items() if k and v}
    if section == "exclude":
        if isinstance(value, dict):
            value = [{"reason": k, "where": v} for k, v in value.items()]
        return [{"reason": str(e.get("reason") or e.get("where")), "where": str(e["where"])}
                for e in value or [] if isinstance(e, dict) and e.get("where")]
    if section == "aggregate":
        if isinstance(value, list):
            value = {a.get("name"): a.get("expression") or a.get("expr") for a in value if isinstance(a, dict)}
        return {str(k): str(v) for k, v in (value or {}).items() if k and v}
    return value if isinstance(value, str) and value else None


def _constant_problems(exprs: list[str], message: str) -> list[str]:
    stated = stated_numbers(message)
    ints = {int(x) for x in stated if x == int(x)}
    out = []
    for e in exprs:
        for c in _constants(e):
            if isinstance(c, str):
                if int(c[:4]) not in ints or (int(c[5:7]) not in ints and int(c[5:7]) not in (1, 12)):
                    out.append(f"the date '{c}' in '{e}' is not one the question mentions")
            elif c not in FREE_CONSTANTS and not any(abs(c - s) < 1e-9 for s in stated):
                times = sorted(_TIME_IN_TEXT.findall(message or ""), key=lambda t: (message or "").find(t))
                if "hours_between" in e and c in (60.0, 3600.0, 1440.0, 24.0) and c not in unit_factors(message):
                    out.append(f"hours_between() already returns hours, so remove the {c:g} in '{e}' "
                               "(write e.g. sum(hours_between(clock_in, clock_out) * rate))")
                elif re.search(rf"(hours_between|overnight_hours)\([^)]*\)\s*\)?\s*-\s*{re.escape(format(c, 'g'))}\b", e):
                    out.append(f"'{e}' subtracts {c:g} hours, a break the user never mentioned; use the duration as "
                               "it is, e.g. hours_between(clock_in, clock_out), with nothing subtracted")
                elif len(times) >= 2:
                    out.append(f"the number {c:g} in '{e}' was not stated; the user gave clock times, so write the "
                               f"duration as hours_between('{times[-2]}', '{times[-1]}') (or overnight_hours if it "
                               "ends after midnight) instead of a number")
                elif len(stated) and any(x == int(x) and 0 < x < 100 for x in stated):
                    out.append(f"the number {c:g} in '{e}' was not stated by the user; write the arithmetic with the "
                               "numbers the user gave (e.g. 10 - 0.5 for 10 hours less a 30-minute break), "
                               "not a result you worked out")
                else:
                    out.append(f"the number {c:g} in '{e}' was not stated by the user; do not add deductions or "
                               "assumptions the user did not give")
    return out


_EXCLUDE_INTENT = re.compile(r"\b(exclud\w*|except|remove|drop|without|ignore|skip|filter\w*|only|valid|invalid|outside|"
                             r"within|between|after|before|during|negative|duplicat\w*|where|in \d{4}|for \w+ \d{4}|"
                             r"for (the )?(month|year|quarter|week)|not )\b", re.I)
_RATE_Q = re.compile(r"\b(rate|share|percent\w*|proportion|fraction)\b", re.I)
_EVENT_VALUES = ("yes", "true", "1", "y", "t")
_RANKING = re.compile(r"\b(highest|lowest|most|least|largest|smallest|biggest|top|bottom)\b", re.I)
_SINGLE = re.compile(r"\b(single|individual|one record|per record|per row|per shift|per entry|each record|maximum|minimum)\b", re.I)


def named_values(df: pd.DataFrame, question: str) -> list[tuple[str, str]]:
    """(column, value) for category values the question names, e.g. ("department", "HR")."""
    out = []
    for c in df.columns:
        if kind_of(df[c]) != "categorical" or df[c].nunique() > 50 or _looks_like_time(df[c].dropna()):
            continue
        values = [v for v in df[c].dropna().astype(str).unique() if re.search(r"[A-Za-z]", v)]
        full = [v for v in values if len(v) >= 2 and re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", question, re.I)]
        out += [(str(c), v) for v in full]
        if full:
            continue
        # "Rahul" names "Rahul Sharma" when exactly one value has that word (case as written: names are capitalised)
        for v in values:
            for w in re.findall(r"[A-Za-z][\w'-]{3,}", v):
                if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", question) and \
                        sum(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", x) is not None for x in values) == 1:
                    out.append((str(c), v))
                    break
    return list(dict.fromkeys(out))


_NEGATE = re.compile(r"\b(exclud\w*|except|without|other than|besides|apart from|ignor\w*|remov\w*|drop\w*|leave out|"
                     r"not|n't|everyone but|all but|rest of|others?)\b", re.I)
_COMPARE = re.compile(r"\b(compar\w*|vs\.?|versus|each|every|all (the )?(employees|people|staff|departments|groups)|"
                      r"by (employee|person|name|department|team)|per (employee|person|department|team)|rank\w*|"
                      r"others?|rest)\b", re.I)
_SEL_EQ = re.compile(r"^\s*\(?\s*(\w+)\s*==\s*['\"]([^'\"]+)['\"]\s*\)?\s*$")


def inverted_selection(where: str, frame: pd.DataFrame, question: str) -> tuple[str, str] | None:
    """(column, value) when an exclusion `col == 'Value'` removes the very person or group the question is
    about, and the question gives no sign of wanting them left out ("excluding Rahul", "everyone but Rahul")."""
    m = _SEL_EQ.match(where or "")
    if not m or m.group(1) not in frame.columns:
        return None
    named = {(c, v.lower()) for c, v in named_values(frame, question)}
    if (m.group(1), m.group(2).lower()) not in named:
        return None
    return None if _NEGATE.search(question) else (m.group(1), m.group(2))


def _selection_for(col: str, vals: list[str]) -> dict:
    if len(vals) == 1:
        return {"reason": f"only {vals[0]}", "where": f"{col} != '{vals[0]}'"}
    return {"reason": "only " + " and ".join(vals),
            "where": f"{col} not in [" + ", ".join(f"'{v}'" for v in vals) + "]"}


def repair_section(section: str, value, problems: list[str], spec: dict, frame: pd.DataFrame,
                   question: str) -> tuple[object, dict | None, list[str]] | None:
    """Deterministic fixes for harmless, recognisable mistakes in a section the validator rejected:
    (new value, changes to other sections, descriptions), or None when nothing safe applies. A repair
    only removes what the user did not ask for or rewrites an expression into the form the validator
    names (an inverted selection, a unit division, an invented break, max/min for a ranking); it never
    adds a number, an assumption or a record filter of its own, except the one that selects the
    person or group the question names."""
    if value is UNPARSED:
        return None
    try:
        v = _normalise_section(section, value)
    except (AttributeError, KeyError, TypeError):
        return None
    notes: list[str] = []
    text = " ".join(problems)
    if section == "exclude":
        out = []
        for e in v:
            inv = inverted_selection(e["where"], frame, question)
            if inv:
                out.append({"reason": f"only {inv[1]}", "where": f"{inv[0]} != '{inv[1]}'"})
                notes.append(f"the exclusion `{e['where']}` would have removed {inv[1]}, the subject of the question; "
                             f"it was inverted to keep only {inv[1]}")
            elif f"('{e['reason']}')" in text and ("did not ask to exclude" in text):
                notes.append(f"the exclusion '{e['reason']}' (`{e['where']}`) was not asked for and was removed")
            elif f"'{e['where']}' does not test each row" in text:
                notes.append(f"the exclusion `{e['where']}` did not test each record and was removed")
            else:
                out.append(e)
        return (out, None, notes) if notes else None
    if section == "group_by":
        extra, group = [], v
        by_col: dict[str, list[str]] = {}
        for col, val in named_values(frame, question):
            if not any(val.lower() in e["where"].lower() for e in spec.get("exclude") or []):
                by_col.setdefault(col, []).append(val)
        for col, vals in by_col.items():
            extra.append(_selection_for(col, vals))
            if len(vals) > 1:
                group = col
                notes.append(f"the question names {' and '.join(vals)}, so only their records are compared by {col}")
            else:
                if group == col:
                    group = None
                notes.append(f"the question is about {vals[0]}, so only {vals[0]}'s records are used "
                             f"(rather than every {col})")
        if not by_col and group and "grouping by every" in text:
            vals = [val for c, val in named_values(frame, question) if c == group]
            notes.append(f"the question is about {vals[0] if vals else 'one ' + group}, so the per-{group} breakdown "
                         "was dropped")
            group = None
        if not notes:
            return None
        return group, ({"exclude": list(spec.get("exclude") or []) + extra} if extra else None), notes
    if section in ("derive", "aggregate"):
        out = {}
        copied = _copied_data_values(text, frame, question)
        for name, expr in v.items():
            new = expr
            for num, col in copied:
                pat = rf"(?<![\w.]){re.escape(num)}(?:\.0+)?(?![\w.])"
                if re.search(pat, new):
                    repl = col if section == "derive" or _inside_aggregate(new, re.search(pat, new).start()) else f"mean({col})"
                    new = re.sub(pat, repl, new, count=1)
                    notes.append(f"`{expr}` used {num}, a value copied from the data ({col}); it was replaced by "
                                 f"{'the column' if repl == col else 'the average of'} {col} so it is read from the records")
            for c in ("3600", "1440", "60", "24"):
                new2 = re.sub(rf"((?:hours_between|overnight_hours)\([^()]*\))\s*/\s*{c}(?:\.0)?\b", r"\1", new)
                if new2 != new:
                    notes.append(f"`{expr}` divided hours by {c}; hours_between already gives hours, so the division "
                                 "was removed")
                    new = new2
            m = re.search(r"subtracts ([\d.]+) hours, a break the user never mentioned", text)
            if m and f"'{expr}'" in text:
                c = re.escape(m.group(1))
                new2 = re.sub(rf"\(\s*((?:hours_between|overnight_hours)\([^()]*\))\s*-\s*{c}\s*\)", r"\1", new)
                new2 = re.sub(rf"((?:hours_between|overnight_hours)\([^()]*\))\s*-\s*{c}\b", r"\1", new2)
                if new2 != new:
                    notes.append(f"`{expr}` subtracted a {m.group(1)}-hour break you did not mention; it was removed")
                    new = new2
            if section == "aggregate" and "counts every record, so the rate is always 100%" in text:
                mm = re.search(r"\bcount\s*\(\s*(\w+)\s*\)", new)
                col = mm.group(1) if mm and mm.group(1) in frame.columns else None
                if col is not None:
                    vals = list(frame[col].dropna().astype(str).unique())
                    named = [v for c, v in named_values(frame, question) if c == col]
                    event = named[0] if len(named) == 1 else next((v for v in vals if v.lower() in _EVENT_VALUES), None)
                    if event is not None and (len(vals) <= 2 or len(named) == 1):
                        new2 = f"mean({col} == '{event}')"
                        notes.append(f"`{expr}` counted every record (always 100%); a rate counts the records where "
                                     f"{col} is '{event}', so it became `{new2}`")
                        new = new2
            if section == "aggregate" and f"'{name}' uses " in text and "use sum()" in text:
                new2 = re.sub(r"^\s*(max|min)\s*\(", "sum(", new, flags=re.I)
                if new2 != new:
                    notes.append(f"`{expr}` took the single largest record per group; a ranking compares group "
                                 "totals, so it was changed to sum()")
                    new = new2
            out[name] = new
        return (out, None, notes) if notes else None
    return None


def _copied_data_values(problems: str, frame: pd.DataFrame, question: str) -> list[tuple[str, str]]:
    """(number, column) for unstated numbers the model copied out of the data, e.g. Priya's rate 22 written as a
    constant: the number occurs in a numeric column, among the named person's records when one is named."""
    rows = frame
    for col, val in named_values(frame, question):
        rows = rows[rows[col].astype(str) == val] if col in rows else rows
    out = []
    for m in re.finditer(r"the number ([\d.]+) in '", problems):
        x = float(m.group(1))
        cols = [c for c in frame.columns if kind_of(frame[c]) == "numeric"
                and (pd.to_numeric(rows[c], errors="coerce") - x).abs().lt(1e-9).any()]
        if len(cols) == 1 and (m.group(1), cols[0]) not in out:
            out.append((format(x, "g"), cols[0]))
    return out


def _inside_aggregate(expr: str, pos: int) -> bool:
    depth_fn = []
    for m in re.finditer(r"\b(sum|mean|avg|average|min|max|count|total)\s*\(|[()]", expr[:pos]):
        tok = m.group(0)
        if tok == ")":
            if depth_fn:
                depth_fn.pop()
        else:
            depth_fn.append(tok != "(")
    return any(depth_fn)


def _check_section(section: str, value, spec: dict, frame: pd.DataFrame, message: str,
                   question: str = "", reused: frozenset = frozenset()) -> list[str]:
    if value is UNPARSED:
        return [f"the {section} was not valid JSON"]
    try:
        v = _normalise_section(section, value)
    except (AttributeError, KeyError, TypeError):
        return [f"the {section} does not have the expected shape"]
    try:
        if section == "derive":
            new = [x for x in v.values() if _norm(x) not in reused]
            probs = _constant_problems(new, message) + _time_problems(new, message)
            if len(v) > MAX_DERIVE:
                probs.append("too many derived columns")
            if not probs:
                run_derive(frame, v)
            return probs
        if section == "exclude":
            probs = _constant_problems([e["where"] for e in v], message) + _time_problems([e["where"] for e in v], message)
            named = [val.lower() for _, val in named_values(frame, question or message)]
            earlier = {e["where"].replace(" ", "") for _, plan in (spec.get("_history") or []) for e in plan.get("exclude", [])}
            for e in v:
                if e["where"].replace(" ", "") in earlier:
                    continue
                if "duplicate()" in e["where"] and not re.search(r"duplicat|repeat|identical", message, re.I):
                    probs.append(f"the user did not ask to exclude duplicates ('{e['reason']}'); remove it")
                elif not _EXCLUDE_INTENT.search(message) and not any(val in e["where"].lower() for val in named):
                    probs.append(f"the user did not ask to exclude any records ('{e['reason']}'); use []")
            if len(v) > MAX_EXCLUDE:
                probs.append("too many exclusions")
            for e in v:
                inv = inverted_selection(e["where"], frame, question or message)
                if inv:
                    probs.append(f"'{e['where']}' removes the records of {inv[1]}, which the question is about; to keep "
                                 f"only {inv[1]} write {inv[0]} != '{inv[1]}'")
            kept = pd.Series(True, index=frame.index)
            for e in v:
                hit = evaluate(e["where"], frame, frame[[c for c in frame.columns if c not in spec["derive"]]])
                if not isinstance(hit, pd.Series):
                    probs.append(f"'{e['where']}' does not test each row")
                else:
                    kept &= ~hit.fillna(False).astype(bool)
            if v and len(frame) and not kept.any() and not probs:
                probs.append("these exclusions leave no records at all; check the values used (for example the exact "
                             "spelling of a name) and do not filter on the date of a hypothetical future event")
            return probs
        if section == "group_by":
            if v is not None and v not in frame.columns:
                return [f"'{v}' is not a column; use one of {', '.join(map(str, frame.columns))} or null"]
            probs = []
            for col, val in named_values(frame, question):
                if v != col and not any(val.lower() in e["where"].lower() for e in spec["exclude"]):
                    probs.append(f"the question is about {col} '{val}', but nothing selects it: group_by \"{col}\" "
                                 f"so its result is shown")
            named_cols = {c for c, _ in named_values(frame, question)}
            if v in named_cols and len([1 for c, _ in named_values(frame, question) if c == v]) == 1 \
                    and not _COMPARE.search(question) and not any(
                        val.lower() in e["where"].lower() for c, val in named_values(frame, question) for e in spec["exclude"]):
                probs.append(f"the question is about one {v}, so grouping by every {v} shows people it did not ask about")
            return probs
        if section == "aggregate":
            if not v:
                asked = asked_quantities(question or message)
                derived = list(spec.get("derive") or {})
                scen = _scenario_columns(spec.get("derive") or {}, frame)
                hint = ""
                if derived:
                    how = "mean" if (scen or _HYPOTHETICAL.search(question or message)) else "sum"
                    hint = ("; report what the question asks for from the derived columns, e.g. {"
                            + ", ".join(f'"{d}": "{how}({d})"' for d in derived[:4]) + "}"
                            + (" (mean: one estimate from the historical records, not a total over them)" if how == "mean" else ""))
                return ["there must be at least one result" + (f"; the question asks for: {', '.join(asked)}" if asked else "")
                        + hint]
            new = [x for x in v.values() if _norm(x) not in reused]
            probs = _constant_problems(new, message) + _time_problems(new, message)
            probs += break_problems({"derive": spec.get("derive") or {}, "aggregate": v}, question or message)
            assumed = assumption_columns(spec.get("derive") or {}, [c for c in frame.columns if c not in (spec.get("derive") or {})])
            kept = frame
            for e in spec.get("exclude") or []:
                try:
                    hit = evaluate(e["where"], frame, frame)
                    kept = kept[~hit.reindex(kept.index).fillna(False).astype(bool)]
                except CalcError:
                    pass
            consts = {k: float(frame[k].iloc[0]) for k in assumed if k in frame and len(frame)}
            for name, a in v.items():
                if _RATE_Q.search(question or message) and re.search(r"\bcount\s*\(", a, re.I) \
                        and not re.search(r"==|!=|<|>|\bin\b", a):
                    col = next((c for c in map(str, frame.columns) if re.search(rf"\bcount\s*\(\s*{re.escape(c)}\s*\)", a)),
                               None)
                    vals = (", ".join(f"'{x}'" for x in frame[col].dropna().astype(str).value_counts().index[:6])
                            if col else "")
                    probs.append(f"'{name}' = {a} counts every record, so the rate is always 100%; a rate counts only "
                                 f"the records where the event happened, e.g. mean({col or 'column'} == 'value')"
                                 + (f" (values of {col}: {vals})" if vals else ""))
                m = _agg_match(a)
                if (m and m.group(1).lower() in ("max", "min") and _balanced_inner(m.group(2)) and spec.get("group_by")
                        and _RANKING.search(question) and not _SINGLE.search(question)):
                    probs.append(f"'{name}' uses {m.group(1)}(), the single largest/smallest record in each "
                                 f"{spec['group_by']}; 'which {spec['group_by']} has the highest/lowest ...' compares "
                                 "each group's total, so use sum() unless the user asks about a single record")
                # An assumption is one value; summing it over the history counts it once per record.
                scen_cols = _scenario_columns(spec.get("derive") or {}, frame)
                for call in re.finditer(r"\b(sum|total)\s*\(", a, re.I):
                    inner = a[call.end():]
                    used = [k for k in list(assumed) + scen_cols if re.search(rf"\b{re.escape(k)}\b", inner)] + \
                           (["a clock-time duration"] if _time_literals(inner) else [])
                    if used and len(kept) > 1:
                        probs.append(f"'{name}' sums {used[0]} — one assumed value — over {len(kept)} historical records, "
                                     f"counting it {len(kept)} times; combine it with an average instead, e.g. "
                                     f"mean(rate_column) * {used[0] if used[0] in assumed else 'hours'}")
                try:
                    agg_value(a, kept if len(kept) else frame, frame, consts)
                except CalcError as e:
                    probs.append(f"{e} (in '{name}')")
            return list(dict.fromkeys(probs))
    except CalcError as e:
        return [str(e)]
    except Exception as e:  # noqa: BLE001 - a malformed expression must not crash the turn
        return [f"{type(e).__name__}: {e}"]
    return []


# ------------------------------------------------------------------ routing --
# When a question goes to the calculator rather than the trained tool catalogue.
_STRONG = re.compile(r"\b(multipl\w*|times|divided by|minus|exclud\w*|except|valid (records|rows|entries)|"
                     r"filter(ed)? out|leave out|drop (the )?(duplicate|rows|records)|only (count|include|use)|"
                     r"(calculated|computed|defined) as|formula|labou?r cost|working hours|hours worked|hourly rate|"
                     r"unpaid break|\d+(?:\.\d+)?[- ]hours?( shift)?|shift|overnight|\d{1,2}:\d{2}|\d{1,2}\s*[ap]\.?m\.?)(?!\w)", re.I)
_CALC_VERB = re.compile(r"\b(calculate|compute|work out|add up|sum up|how much|tally|total)\b", re.I)
_NOT_CALC = re.compile(r"\b(correlat\w*|significan\w*|statistical(ly)? test|t-?test|anova|chi|predict\w*|forecast\w*|"
                       r"train\w*|machine learning|random forest|classif\w*|regress\w*|cluster\w*|outlier\w*|unusual|"
                       r"anomal\w*|trend\w*|churn|recommend\w*|feature importance)\b", re.I)


_GENERIC = {"total", "count", "rows", "records", "average", "mean", "number", "value", "values", "amount", "result"}


def wanted(message: str, df: pd.DataFrame, memory: dict | None = None) -> str | None:
    """Why this question should be calculated (a short reason), or None."""
    from app.analyst.data import find_column
    memory = memory or {}
    if _NOT_CALC.search(message) and not _STRONG.search(message):
        return None
    if _STRONG.search(message):
        return "the question defines its own calculation or exclusions"
    history = memory.get("calc_history") or []
    if history:
        cols = {w for c in df.columns for w in re.split(r"[^a-z0-9]+", str(c).lower())}
        words = {w for _, plan in history for n in list(plan.get("derive", {})) + list(plan.get("aggregate", {}))
                 for w in re.split(r"[^a-z0-9]+", n.lower()) if len(w) > 3 and w not in cols and w not in _GENERIC}
        low = set(re.findall(r"[a-z0-9]+", message.lower()))
        if words & low or {w + "s" for w in words} & low or {w[:-1] for w in words if w.endswith("s")} & low:
            return "a follow-up on an earlier calculation"
    if _CALC_VERB.search(message):
        for q in asked_quantities(message):
            if not find_column(df, q):
                return f"'{q}' is not a column, so it has to be calculated"
    return None
