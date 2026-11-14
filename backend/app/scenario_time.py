"""Time-aware what-if baselines: choose a period of the data, and compare two historical periods.

A baseline can be taken from all of a dataset or from one calendar period of it (a year, quarter or month, or a custom
date range). Periods are read from the dataset's *own* calendar: the periods on offer are the ones that have rows, not
"this month" by the clock.

Four kinds of result are kept apart, and every response says which it is:

    historical_actual      what the data recorded in a period (baselines, period comparisons)
    scenario               arithmetic on stated inputs and adjustments (app/scenarios.py)
    statistical_estimate   a quantity estimated from the data with its uncertainty (elasticity)
    forecast               not produced by this feature. A period's figures are never extrapolated forward.

A period comparison states what changed between two recorded periods and attributes the change in the headline output to
the inputs that moved (Shapley, the same method scenarios use). It describes the past; it does not say why inputs moved.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app import scenarios
from app.scenarios import ScenarioError

TIME_VERSION = "scenario-time.1"
PERIOD_TYPES = ("all", "year", "quarter", "month", "custom")
MAX_LISTED_MONTHS = 240
DATA_KINDS = {
    "historical_actual": "Recorded in your data for the period shown.",
    "scenario": "Arithmetic on the inputs and adjustments you set. Not observed, not a forecast.",
    "statistical_estimate": "Estimated from your data with its uncertainty. Not exact, not a forecast.",
    "forecast": "Not produced here: no figure is extrapolated forward in time.",
}


class TimeError(ScenarioError):
    """The date column or period is not valid; the message says why."""


def date_column_of(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise TimeError(f"The date column '{column}' does not exist in this dataset")
    s = df[column]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")
        if s.notna().mean() < 0.9:
            raise TimeError(f"The column '{column}' is not made of dates")
    return s


def _int(v: Any, what: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise TimeError(f"{what} must be a whole number from {lo} to {hi}")
    return v


def parse_period(spec: dict | None) -> dict:
    """A period as {type, start, end, label}. `start` and `end` are inclusive timestamps (None for all data)."""
    spec = spec or {"type": "all"}
    kind = spec.get("type")
    if kind not in PERIOD_TYPES:
        raise TimeError(f"A period type is one of: {', '.join(PERIOD_TYPES)}")
    if kind == "all":
        return {"type": "all", "start": None, "end": None, "label": "All data", "spec": {"type": "all"}}
    if kind == "custom":
        try:
            start, end = pd.Timestamp(spec.get("start")), pd.Timestamp(spec.get("end"))
        except (ValueError, TypeError) as e:
            raise TimeError("A custom period needs a start and an end date, for example 2025-02-01") from e
        if pd.isna(start) or pd.isna(end) or start > end:
            raise TimeError("The start of a custom period must not be after its end")
        end = end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return {"type": "custom", "start": start.normalize(), "end": end,
                "label": f"{start.date()} to {end.date()}", "spec": {"type": "custom", "start": str(start.date()), "end": str(end.date())}}
    year = _int(spec.get("year"), "The year", 1900, 2200)
    if kind == "year":
        start, nxt, label = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1), str(year)
        out_spec = {"type": "year", "year": year}
    elif kind == "quarter":
        q = _int(spec.get("quarter"), "The quarter", 1, 4)
        start = pd.Timestamp(year, 3 * q - 2, 1)
        nxt, label = start + pd.DateOffset(months=3), f"{year} Q{q}"
        out_spec = {"type": "quarter", "year": year, "quarter": q}
    else:
        m = _int(spec.get("month"), "The month", 1, 12)
        start = pd.Timestamp(year, m, 1)
        nxt, label = start + pd.DateOffset(months=1), start.strftime("%B %Y")
        out_spec = {"type": "month", "year": year, "month": m}
    return {"type": kind, "start": start, "end": nxt - pd.Timedelta(microseconds=1), "label": label, "spec": out_spec}


def period_filters(date_column: str, period: dict) -> list[dict]:
    if period["start"] is None:
        return []
    return [{"column": date_column, "op": "between", "value": [period["start"].isoformat(), period["end"].isoformat()]}]


def available_periods(df: pd.DataFrame, date_column: str) -> dict:
    """The periods that have rows, from the data's own calendar."""
    dates = date_column_of(df, date_column)
    valid = dates.dropna()
    if valid.empty:
        raise TimeError(f"The column '{date_column}' has no dates")
    by_month = valid.dt.to_period("M").value_counts().sort_index()
    months = [{"year": p.year, "month": p.month, "rows": int(n)} for p, n in by_month.items()]
    quarters = [{"year": p.year, "quarter": p.quarter, "rows": int(n)} for p, n in valid.dt.to_period("Q").value_counts().sort_index().items()]
    years = [{"year": int(y), "rows": int(n)} for y, n in valid.dt.year.value_counts().sort_index().items()]
    return {"date_column": date_column, "min": str(valid.min().date()), "max": str(valid.max().date()),
            "rows_with_date": int(len(valid)), "rows_without_date": int(len(dates) - len(valid)),
            "years": years, "quarters": quarters, "months": months[-MAX_LISTED_MONTHS:],
            "months_truncated": len(months) > MAX_LISTED_MONTHS}


def describe_period(period: dict, date_column: str | None, rows: int, df: pd.DataFrame | None = None) -> dict:
    out = {"label": period["label"], "type": period["type"], "spec": period["spec"], "date_column": date_column,
           "start": None if period["start"] is None else str(period["start"].date()),
           "end": None if period["end"] is None else str(period["end"].date()), "rows": rows}
    if df is not None and date_column and rows:
        d = date_column_of(df, date_column).dropna()
        out["data_covers"] = [str(d.min().date()), str(d.max().date())]
    return out


def require_rows(period: dict, rows: int, df: pd.DataFrame, date_column: str | None) -> None:
    """A period with no rows is an error, not a baseline of zeros."""
    if rows:
        return
    covers = ""
    if date_column:
        d = date_column_of(df, date_column).dropna()
        if len(d):
            covers = f" The data covers {d.min().date()} to {d.max().date()}."
    raise TimeError(f"No rows fall in {period['label']}.{covers}")


# ---------------------------------------------------------- period comparison --

def compare_periods(model: "str | dict", a: dict, b: dict, assumptions: dict | None = None) -> dict:
    """Outputs, changes and an attribution of the change, between two recorded sets of inputs (period A, then B)."""
    spec = scenarios._model(model)  # noqa: SLF001
    assumptions = scenarios.validate_assumptions(model, assumptions)
    for label, inputs in (("A", a), ("B", b)):
        missing = [k for k in spec["inputs"] if inputs.get(k) is None]
        if missing:
            raise TimeError(f"Period {label} has no value for: {', '.join(missing)}. Enter them so both periods are complete.")
    a, b = scenarios.validate_inputs(model, a), scenarios.validate_inputs(model, b)
    out_a, out_b = scenarios._outputs(model, a, assumptions), scenarios._outputs(model, b, assumptions)  # noqa: SLF001
    primary = spec["primary_output"]
    sh = scenarios.shapley(lambda i: scenarios._outputs(model, i, assumptions)[primary] or 0.0, a, b)  # noqa: SLF001
    notes: dict[str, list] = {}
    check = spec.get("constraint_violations")
    if check is not None:
        for label, inputs, outs in (("A", a, out_a), ("B", b, out_b)):
            found = check({**inputs, **assumptions, **{k: (math.nan if v is None else v) for k, v in outs.items()}})
            if found:
                notes[label] = found
    return {"kind": "period_comparison", "data_kind": "historical_actual", "data_kinds": DATA_KINDS,
            "engine_version": TIME_VERSION, "model": scenarios.model_key(model), "title": spec["title"],
            "assumptions": assumptions, "primary_output": primary, "formulas": spec["formulas"],
            "a": {"inputs": a, "outputs": out_a}, "b": {"inputs": b, "outputs": out_b},
            "input_changes": {k: _change(a[k], b[k]) for k in a},
            "output_changes": scenarios._delta(out_a, out_b),  # noqa: SLF001
            "contributions": {"output": primary, "from": "A", "to": "B", "by_input": sh["values"],
                              "method": ("Shapley average over the order in which inputs changed between the periods"
                                         if sh["method"] == "exact" else
                                         f"Shapley average estimated from {sh['samples']:,} sampled orders (approximate)"),
                              "computation": {k: sh[k] for k in ("method", "changed_inputs", "samples", "standard_errors")}},
            "constraint_notes": notes,
            "limits": ["This describes what the data recorded in each period. It does not say why the inputs moved, and it is "
                       "not a forecast.",
                       "The attribution splits the change in the headline output among the inputs that moved, using the "
                       "model's formulas; inputs that are linked in reality (a price change that moved volume) are attributed "
                       "separately here."]}


def _change(x: float, y: float) -> dict:
    return {"abs": y - x, "pct": (y - x) / abs(x) * 100 if x else None}
