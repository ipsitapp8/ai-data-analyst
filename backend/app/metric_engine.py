"""Deterministic metric engine: a small, whitelisted formula language evaluated
over a dataset. No model is involved anywhere in this module.

One engine, several users: the fast path answers simple questions with it,
the verification engine recomputes claims with it, the investigator, the
copilot and the what-if simulator all build on it, and semantic-layer metric
definitions are written in its formula language.

Formula language (parsed with `ast`, never evaluated as Python):

    sum(revenue)                         one aggregation
    sum(revenue) / count_distinct(order_id)
    (sum(revenue) - sum(cost)) / sum(revenue) * 100
    avg("Unit Price")                    quote names that are not identifiers

Aggregations: sum, avg (mean), median, min, max, count, count_distinct.
`count()` with no argument counts rows. Operators: + - * / and unary minus.
Anything else -- attribute access, subscripts, other calls, names outside an
aggregation -- is rejected at parse time.

`reference_evaluate` is a second, independent implementation of the same
semantics using only the `csv` module and plain Python. Verification compares
the two; agreement is evidence that neither has a bug the other shares.
"""
from __future__ import annotations

import ast
import csv
import datetime as dt
import math
import os
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd

from app import config

AGGREGATIONS = {"sum", "avg", "mean", "median", "min", "max", "count", "count_distinct"}
NUMERIC_ONLY = {"sum", "avg", "mean", "median"}
FILTER_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in", "between", "contains"}
MAX_FORMULA_CHARS = 500
MAX_GROUPS = 500
# The values pandas.read_csv treats as missing, mirrored by the reference path.
_NA_STRINGS = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
               "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}


class FormulaError(ValueError):
    """The formula, a filter or a column reference is not valid."""


@dataclass(frozen=True)
class Formula:
    text: str
    tree: Any  # nested tuples: ("num", v) | ("agg", fn, col|None) | ("bin", op, l, r) | ("neg", x)
    aggregations: tuple[tuple[str, str | None], ...]

    @property
    def columns(self) -> list[str]:
        return sorted({c for _, c in self.aggregations if c is not None})

    @property
    def is_additive(self) -> bool:
        """True when the formula is a single sum or row count, so segment values
        add up to the total and contributions are exact."""
        return self.tree[0] == "agg" and self.tree[1] in ("sum", "count")


# ------------------------------------------------------------------ parsing --

_BIN_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def parse_formula(text: str) -> Formula:
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("Formula is empty")
    if len(text) > MAX_FORMULA_CHARS:
        raise FormulaError(f"Formula is longer than {MAX_FORMULA_CHARS} characters")
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"Formula is not valid: {e.msg}") from e
    aggs: list[tuple[str, str | None]] = []

    def walk(n: ast.AST, depth: int = 0):
        if depth > 20:
            raise FormulaError("Formula is nested too deeply")
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            if not math.isfinite(float(n.value)):
                raise FormulaError("Numbers in a formula must be finite")
            return ("num", float(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            inner = walk(n.operand, depth + 1)
            return ("neg", inner) if isinstance(n.op, ast.USub) else inner
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN_OPS:
            return ("bin", _BIN_OPS[type(n.op)], walk(n.left, depth + 1), walk(n.right, depth + 1))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = n.func.id.lower()
            if fn not in AGGREGATIONS:
                raise FormulaError(f"'{n.func.id}' is not a supported aggregation "
                                   f"(use one of: {', '.join(sorted(AGGREGATIONS))})")
            if n.keywords or len(n.args) > 1:
                raise FormulaError(f"{fn}() takes one column")
            col: str | None = None
            if n.args:
                arg = n.args[0]
                if isinstance(arg, ast.Name):
                    col = arg.id
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.strip():
                    col = arg.value
                else:
                    raise FormulaError(f"{fn}() needs a column name, e.g. {fn}(revenue) or {fn}(\"Unit Price\")")
            elif fn != "count":
                raise FormulaError(f"{fn}() needs a column")
            fn = "avg" if fn == "mean" else fn
            aggs.append((fn, col))
            return ("agg", fn, col)
        if isinstance(n, ast.Name):
            raise FormulaError(f"Column '{n.id}' must be inside an aggregation, e.g. sum({n.id})")
        raise FormulaError("Only aggregations, numbers and + - * / are allowed in a formula")

    tree = walk(node)
    if not aggs:
        raise FormulaError("A formula needs at least one aggregation, e.g. sum(revenue)")
    return Formula(text.strip(), tree, tuple(aggs))


def validate_against_profile(formula: Formula, profile: dict) -> list[str]:
    """Problems that only show once the formula meets a dataset. Empty = valid."""
    kinds = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    problems = []
    for fn, col in formula.aggregations:
        if col is None:
            continue
        if col not in kinds:
            problems.append(f"Column '{col}' does not exist in this dataset")
        elif fn in NUMERIC_ONLY and kinds[col] != "numeric":
            problems.append(f"{fn}({col}) needs a numeric column; '{col}' is {kinds[col]}")
    return problems


def normalize_filters(filters: list[dict] | None, columns: list[str] | None = None) -> list[dict]:
    out = []
    for f in filters or []:
        if not isinstance(f, dict):
            raise FormulaError("Each filter must be an object with column, op and value")
        col, op, value = f.get("column"), f.get("op", "=="), f.get("value")
        if op == "=":
            op = "=="
        if not isinstance(col, str) or not col:
            raise FormulaError("A filter needs a column")
        if columns is not None and col not in columns:
            raise FormulaError(f"Filter column '{col}' does not exist in this dataset")
        if op not in FILTER_OPS:
            raise FormulaError(f"Filter operator '{op}' is not supported")
        if op in ("in", "not_in"):
            if not isinstance(value, list) or not value or len(value) > 200:
                raise FormulaError(f"Filter '{op}' needs a list of 1 to 200 values")
        elif op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise FormulaError("Filter 'between' needs [low, high]")
        elif isinstance(value, (list, dict)) or value is None:
            raise FormulaError(f"Filter '{op}' needs a single value")
        out.append({"column": col, "op": op, "value": value})
    return out


# ------------------------------------------------------------- pandas path --

@lru_cache(maxsize=3)
def _load_cached(path: str, mtime: float, size: int) -> pd.DataFrame:  # mtime/size: cache key only
    from app.profiling import load_and_profile_csv

    df, _ = load_and_profile_csv(path)
    return df


def load_frame(path: str) -> pd.DataFrame:
    """The dataset as the profiler sees it (date columns parsed). Cached; treat
    the returned frame as read-only."""
    st = os.stat(path)
    df = _load_cached(path, st.st_mtime, st.st_size)
    if len(df) > config.ENGINE_MAX_ROWS:
        raise FormulaError(f"Dataset has more than {config.ENGINE_MAX_ROWS} rows; "
                           "this operation is limited to smaller datasets")
    return df


def _coerce_like(series: pd.Series, value: Any) -> Any:
    if pd.api.types.is_datetime64_any_dtype(series):
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            raise FormulaError(f"'{value}' is not a date")
        return ts
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise FormulaError(f"'{value}' is not a number") from e
    return str(value)


def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    filters = normalize_filters(filters, list(df.columns))
    mask = pd.Series(True, index=df.index)
    for f in filters:
        s, op, v = df[f["column"]], f["op"], f["value"]
        textual = not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))
        cmp = s.astype(str) if textual else s
        if op == "==":
            m = cmp == _coerce_like(s, v)
        elif op == "!=":
            m = cmp != _coerce_like(s, v)
        elif op in (">", ">=", "<", "<="):
            if textual:
                raise FormulaError(f"'{op}' needs a numeric or date column; '{f['column']}' is text")
            rhs = _coerce_like(s, v)
            m = {">": cmp > rhs, ">=": cmp >= rhs, "<": cmp < rhs, "<=": cmp <= rhs}[op]
        elif op == "between":
            if textual:
                raise FormulaError(f"'between' needs a numeric or date column; '{f['column']}' is text")
            m = (cmp >= _coerce_like(s, v[0])) & (cmp <= _coerce_like(s, v[1]))
        elif op == "in":
            m = cmp.isin([_coerce_like(s, x) for x in v])
        elif op == "not_in":
            m = ~cmp.isin([_coerce_like(s, x) for x in v])
        else:  # contains
            m = s.astype(str).str.contains(str(v), case=False, regex=False)
        mask &= (m.fillna(False) & s.notna()) if op != "!=" and op != "not_in" else m.fillna(True)
    return df[mask.astype(bool)]


def _agg(keys: list[pd.Series] | None, df: pd.DataFrame, fn: str, col: str | None):
    """One aggregation, over the whole frame (keys is None) or per group."""
    grouped = keys is not None
    if col is None:  # count()
        return df.groupby(keys, dropna=False, sort=True).size() if grouped else float(len(df))
    if col not in df.columns:
        raise FormulaError(f"Column '{col}' does not exist in this dataset")
    if fn in NUMERIC_ONLY or (fn in ("min", "max") and not pd.api.types.is_datetime64_any_dtype(df[col])):
        series = pd.to_numeric(df[col], errors="coerce")
    else:
        series = df[col]
    if grouped:
        series = series.groupby(keys, dropna=False, sort=True)
    if fn == "sum":
        return series.sum(min_count=1)
    if fn == "avg":
        return series.mean()
    if fn == "median":
        return series.median()
    if fn == "min":
        return series.min()
    if fn == "max":
        return series.max()
    if fn == "count":
        return series.count() if grouped else float(series.count())
    if fn == "count_distinct":
        return series.nunique() if grouped else float(series.nunique())
    raise FormulaError(f"Unsupported aggregation '{fn}'")


def _combine(tree, resolve):
    kind = tree[0]
    if kind == "num":
        return tree[1]
    if kind == "agg":
        return resolve(tree[1], tree[2])
    if kind == "neg":
        return -_combine(tree[1], resolve)
    op, left, right = tree[1], _combine(tree[2], resolve), _combine(tree[3], resolve)
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if isinstance(right, pd.Series):
        return left / right.where(right != 0)
    if right is None or (isinstance(right, float) and (right == 0 or math.isnan(right))):
        return float("nan") if not isinstance(left, pd.Series) else left * float("nan")
    return left / right


def _clean(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _key(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate(df: pd.DataFrame, formula: Formula | str, filters: list[dict] | None = None,
             group_by: list[str] | None = None) -> dict:
    """Evaluate a formula. Returns
        {"value": float|None, "rows": n}                       without group_by
        {"groups": [{"key": {...}, "value": ...}], "rows": n}  with group_by
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)
    for col in formula.columns:
        if col not in df.columns:
            raise FormulaError(f"Column '{col}' does not exist in this dataset")
    data = apply_filters(df, filters)
    coerced = _coercion_report(data, formula)
    if not group_by:
        value = _combine(formula.tree, lambda fn, col: _agg(None, data, fn, col))
        return {"value": _clean(value), "rows": int(len(data)), "coerced_non_numeric": coerced}

    for g in group_by:
        if g not in df.columns:
            raise FormulaError(f"Group-by column '{g}' does not exist in this dataset")
    if len(group_by) > 3:
        raise FormulaError("At most 3 group-by columns are supported")
    keys = [data[g] for g in group_by]
    counts = data.groupby(keys, dropna=False, sort=True).size()
    values = _combine(formula.tree, lambda fn, col: _agg(keys, data, fn, col))
    if not isinstance(values, pd.Series):
        values = counts * 0 + values
    groups = []
    for key, value in values.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        groups.append({
            "key": {g: _key(k) for g, k in zip(group_by, key_tuple)},
            "value": _clean(value),
            "rows": int(counts.get(key, 0)),
        })
    truncated = len(groups) > MAX_GROUPS
    return {"groups": groups[:MAX_GROUPS], "rows": int(len(data)), "truncated": truncated,
            "coerced_non_numeric": coerced}


def _coercion_report(data: pd.DataFrame, formula: Formula) -> dict[str, int]:
    """Values that were present but not numeric in a column used numerically."""
    out = {}
    for fn, col in formula.aggregations:
        if col is None or fn not in NUMERIC_ONLY or col not in data.columns:
            continue
        if not pd.api.types.is_numeric_dtype(data[col]):
            bad = int((pd.to_numeric(data[col], errors="coerce").isna() & data[col].notna()).sum())
            if bad:
                out[col] = bad
    return out


# ---------------------------------------------------------- reference path --

class ReferenceNotApplicable(Exception):
    """The plain-Python path cannot express this computation (it then counts
    as a check that was not run -- never as a pass)."""


def _ref_number(raw: str) -> float | None:
    raw = raw.strip()
    if raw in _NA_STRINGS:
        return None
    try:
        f = float(raw)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _ref_date(raw: Any) -> dt.datetime | None:
    text = str(raw).strip().replace("T", " ")
    for fmt, width in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M", 16), ("%Y-%m-%d", 10),
                       ("%Y/%m/%d", 10), ("%m/%d/%Y", 10), ("%d-%m-%Y", 10)):
        if len(text) < width:
            continue
        try:
            return dt.datetime.strptime(text[:width], fmt)
        except ValueError:
            continue
    return None


def _ref_match(row: dict, f: dict) -> bool:
    raw = (row.get(f["column"]) or "").strip()
    if raw in _NA_STRINGS:
        return f["op"] in ("!=", "not_in")
    op, v = f["op"], f["value"]

    def comparable(x: Any):
        n_cell, n_val = _ref_number(raw), _ref_number(str(x))
        if n_cell is not None and n_val is not None:
            return n_cell, n_val
        d_cell, d_val = _ref_date(raw), _ref_date(x)
        if d_cell is not None and d_val is not None:
            return d_cell, d_val
        return None

    def equal(x: Any) -> bool:
        pair = comparable(x)
        return pair[0] == pair[1] if pair is not None else raw == str(x)

    if op == "==":
        return equal(v)
    if op == "!=":
        return not equal(v)
    if op == "in":
        return any(equal(x) for x in v)
    if op == "not_in":
        return not any(equal(x) for x in v)
    if op == "contains":
        return str(v).lower() in raw.lower()
    bounds = v if op == "between" else [v]
    pairs = [comparable(b) for b in bounds]
    if any(p is None for p in pairs):
        raise ReferenceNotApplicable(f"cannot compare '{raw}' with {bounds!r}")
    if op == "between":
        return pairs[0][0] >= pairs[0][1] and pairs[1][0] <= pairs[1][1]
    cell, rhs = pairs[0]
    return {">": cell > rhs, ">=": cell >= rhs, "<": cell < rhs, "<=": cell <= rhs}[op]


def _ref_agg(rows: list[dict], fn: str, col: str | None) -> float | None:
    if col is None:
        return float(len(rows))
    raws = [(r.get(col) or "").strip() for r in rows]
    present = [x for x in raws if x not in _NA_STRINGS]
    if fn == "count":
        return float(len(present))
    if fn == "count_distinct":
        nums = [_ref_number(x) for x in present]
        if present and all(n is not None for n in nums):
            return float(len(set(nums)))
        return float(len(set(present)))
    nums = [n for n in (_ref_number(x) for x in present) if n is not None]
    if not nums:
        return None
    if fn == "sum":
        return math.fsum(nums)
    if fn == "avg":
        return math.fsum(nums) / len(nums)
    if fn == "median":
        return float(statistics.median(nums))
    if fn == "min":
        return min(nums)
    if fn == "max":
        return max(nums)
    raise ReferenceNotApplicable(fn)


def _ref_combine(tree, rows: list[dict]) -> float | None:
    kind = tree[0]
    if kind == "num":
        return tree[1]
    if kind == "agg":
        return _ref_agg(rows, tree[1], tree[2])
    if kind == "neg":
        inner = _ref_combine(tree[1], rows)
        return None if inner is None else -inner
    left, right = _ref_combine(tree[2], rows), _ref_combine(tree[3], rows)
    if left is None or right is None:
        return None
    op = tree[1]
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    return None if right == 0 else left / right


def reference_evaluate(csv_path: str, formula: Formula | str, filters: list[dict] | None = None,
                       group_by: list[str] | None = None) -> dict:
    """Recompute with `csv` + plain Python. Same result shape as `evaluate`.
    Group keys are the raw CSV strings."""
    if isinstance(formula, str):
        formula = parse_formula(formula)
    filters = normalize_filters(filters)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for col in [*formula.columns, *(group_by or []), *(x["column"] for x in filters)]:
            if col not in header:
                raise FormulaError(f"Column '{col}' does not exist in this dataset")
        rows = [r for r in reader if all(_ref_match(r, flt) for flt in filters)]
    if not group_by:
        return {"value": _ref_combine(formula.tree, rows), "rows": len(rows)}
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault(tuple((r.get(g) or "").strip() for g in group_by), []).append(r)
    return {"groups": [{"key": dict(zip(group_by, k)), "value": _ref_combine(formula.tree, v), "rows": len(v)}
                       for k, v in sorted(buckets.items())], "rows": len(rows)}


def values_agree(a: float | None, b: float | None, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
