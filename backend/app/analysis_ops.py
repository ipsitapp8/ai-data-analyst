"""Analytical operations for hypothesis-driven investigation.

A hypothesis is only as good as the test behind it. This module is the closed
set of tests the engine can run. A model may *choose* an operation and its
parameters; it can never supply code, SQL or a result. Every operation is:

  validated     parameters are checked against the dataset's real columns and
                the formula grammar of `metric_engine` before anything runs;
  deterministic computed by `metric_engine` / `investigator` on the pinned
                dataset version -- read-only, row-capped, no generated code;
  identified    by a signature (hash of the normalised operation), so an
                equivalent analysis is never executed twice;
  evaluated     against the hypothesis's stated prediction by fixed rules.

Evidence rules, in one place (thresholds are configuration, see config.py):

  supported      the observation matches the prediction and is material; where
                 the method has a significance test, the test also rejects
                 "no difference"
  weakened       the observation contradicts the prediction: wrong direction,
                 or no material movement on an adequate sample
  inconclusive   too few rows, a result that is in the predicted direction but
                 not distinguishable from noise, or a middling share

A hypothesis that could not be tested is `not_testable`, and one whose
analysis raised is `failed`. Neither is ever counted as evidence against it.

"Supported" means the data is consistent with the hypothesis. It does not mean
the hypothesis is the cause.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app import config, investigator
from app import metric_engine as me
from app.stats import result as stat_result
from app.stats import selector as stat


SUPPORTED, WEAKENED, INCONCLUSIVE, NOT_TESTABLE, FAILED = (
    "supported", "weakened", "inconclusive", "not_testable", "failed")

# operation -> the predictions that make sense for it
EXPECTATIONS: dict[str, tuple[str, ...]] = {
    "metric_change": (),
    "component_change": ("decrease", "increase"),
    "share_change": ("decrease", "increase"),
    "distribution_shift": ("decrease", "increase"),
    "segment_contribution": ("concentrated",),
    "mix_shift": ("shift",),
    "driver_decomposition": ("count_driven", "value_driven"),
    "price_volume": ("price_driven", "volume_driven"),
    "period_coverage": ("incomplete",),
    "data_quality": ("issues_present",),
    # statistical operations (app/stats): each returns a full statistical result
    "level_shift": ("decrease", "increase"),
    "pre_trend": ("pre_existing_trend",),
    "change_point": ("abrupt",),
    "paired_change": ("decrease", "increase"),
    "composition_change": ("changed",),
    "group_difference": ("differs",),
    "correlation": ("positive", "negative"),
    "adjusted_change": ("persists", "explained"),
}
# Operations whose "supported" means the data itself is suspect.
DATA_INTEGRITY_OPS = ("period_coverage", "data_quality")

CATALOGUE = """
component_change      params {"formula": "<aggregation formula>", "filters": [..optional..]}
                      Compares another formula between the two periods (e.g. count(), avg(order_value)).
                      expect: "decrease" | "increase"
share_change          params {"filters": [{"column","op","value"}, ...]}
                      Share of rows matching the filters in each period, with a two-proportion test
                      (e.g. share of rows where status == "refunded").  expect: "decrease" | "increase"
distribution_shift    params {"column": "<numeric column>"}
                      Mean of a numeric column in each period, with Welch's test and effect size.
                      expect: "decrease" | "increase"
segment_contribution  params {"dimension": "<categorical column>", "segment": "<optional value>",
                              "filters": [..optional: restrict to a sub-population, e.g. one region, to drill down..]}
                      How much of the metric's change each segment accounts for, compared with the
                      segment's share of the metric before the change.  expect: "concentrated"
mix_shift             params {"dimension": "<categorical column>"}
                      Whether the average value per row moved because the mix across segments moved
                      (as opposed to values moving within segments).  expect: "shift"
driver_decomposition  params {}
                      Splits a change in a summed metric into "fewer/more rows" and "lower/higher value
                      per row".  expect: "count_driven" | "value_driven"
price_volume          params {}
                      Price effect versus volume effect, when the dataset has a quantity column.
                      expect: "price_driven" | "volume_driven"
level_shift           params {"formula": "<optional; defaults to the metric>", "filters": [..optional..]}
                      The metric as a daily (or weekly) series: did its level differ between the periods by
                      more than its ordinary day-to-day variation? Autocorrelation-aware.
                      expect: "decrease" | "increase"
pre_trend             params {}
                      Was the metric already rising or falling during the earlier period, and how much of the
                      later change would that trend alone produce?  expect: "pre_existing_trend"
change_point          params {}
                      Is there one abrupt shift in the level of the series, and on which date?  expect: "abrupt"
paired_change         params {"unit": "<categorical column>"}
                      The metric per unit (store, product, ...) in both periods, compared as pairs: did the
                      typical unit move, or only a few?  expect: "decrease" | "increase"
composition_change    params {"dimension": "<categorical column>"}
                      Did the share of rows across this dimension's values change between the periods?
                      expect: "changed"
group_difference      params {"outcome": "<numeric column>", "dimension": "<categorical column>",
                              "period": "later" | "earlier" | "both"}
                      Does the mean of a numeric column differ between the groups of a dimension?
                      expect: "differs"
correlation           params {"x": "<numeric column>", "y": "<numeric column>", "period": "both" | "earlier" | "later"}
                      Association between two numeric columns.  expect: "positive" | "negative"
adjusted_change       params {"outcome": "<numeric column>", "controls": ["<column>", ...]}
                      Regression: does the period difference in a row-level value remain once the controls are
                      held fixed? An adjusted association, never a causal estimate.
                      expect: "persists" | "explained"
Any analysis may also carry "method": "<a test you would suggest>" and, for comparisons of a numeric
value, "estimand": "mean" | "typical". The system validates the suggestion against the data and may
override it; it never changes the result to fit.
period_coverage       params {}
                      Whether the two periods are comparable: lengths, days with data, a later period
                      that stops early.  expect: "incomplete"
data_quality          params {}
                      Missing values, duplicates and open data-quality incidents that could produce the
                      change by themselves.  expect: "issues_present"
"""


# The same operations in one line each, for a step that only has to pick one:
# name, parameters ("?" marks an optional one), what it shows, and the allowed
# predictions. Built from EXPECTATIONS, so it cannot drift from the full list.
_BRIEF = {
    "component_change": ('{"formula","filters"?}', "another formula compared between the periods"),
    "share_change": ('{"filters"}', "share of rows matching the filters in each period"),
    "distribution_shift": ('{"column"}', "mean of a numeric column in each period"),
    "segment_contribution": ('{"dimension","segment"?,"filters"?}', "how much of the change each segment accounts for; filters = drill-down"),
    "mix_shift": ('{"dimension"}', "average value moved by mix across segments or within them"),
    "driver_decomposition": ("{}", "fewer or more rows versus value per row"),
    "price_volume": ("{}", "price effect versus volume effect"),
    "level_shift": ('{"formula"?,"filters"?}', "metric as a daily or weekly series: did its level differ"),
    "pre_trend": ("{}", "was the metric already trending before the later period"),
    "change_point": ("{}", "one abrupt shift in the series, and when"),
    "paired_change": ('{"unit"}', "metric per unit in both periods, compared as pairs"),
    "composition_change": ('{"dimension"}', "did the share of rows across this dimension change"),
    "group_difference": ('{"outcome","dimension","period"?}', "mean of a numeric column across groups"),
    "correlation": ('{"x","y","period"?}', "association between two numeric columns"),
    "adjusted_change": ('{"outcome","controls":[..]}', "period difference holding controls fixed (regression; not causal)"),
    "period_coverage": ("{}", "are the two periods comparable"),
    "data_quality": ("{}", "missing values, duplicates, open incidents"),
}
CATALOGUE_BRIEF = "\n".join(
    f"{name} params {_BRIEF[name][0]} - {_BRIEF[name][1]}. expect: {' | '.join(EXPECTATIONS[name])}"
    for name in EXPECTATIONS if name in _BRIEF
) + ('\nA filter is {"column","op","value"}. Optional on any analysis: "method" (a test you would suggest); '
     'on comparisons of a numeric value: "estimand": "mean" | "typical".')


class OpError(ValueError):
    """The operation cannot be run on this dataset. `code` says why."""

    def __init__(self, message: str, code: str = "invalid_operation"):
        super().__init__(message)
        self.code = code


@dataclass
class AnalysisContext:
    """Everything fixed for one investigation: the metric, the population and
    the two periods. Operations compare period A (earlier) with period B (later)."""

    formula: me.Formula
    label: str
    base_filters: list[dict]
    time_column: str
    period_a: list[str]
    period_b: list[str]
    profile: dict
    dq_messages: list[str] = field(default_factory=list)
    total_a: float | None = None
    total_b: float | None = None
    # Work shared by the operations of one run (period rows, time series, segment breakdowns).
    # It only ever holds what the same call would compute again from the same inputs.
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def fa(self) -> list[dict]:
        return investigator._period_filters(self.time_column, self.period_a, last=False)

    @property
    def fb(self) -> list[dict]:
        return investigator._period_filters(self.time_column, self.period_b, last=True)

    @property
    def delta(self) -> float | None:
        if self.total_a is None or self.total_b is None:
            return None
        return self.total_b - self.total_a

    def kinds(self) -> dict[str, str]:
        return {c["name"]: c.get("kind") for c in self.profile.get("columns", [])}


# -------------------------------------------------------------- validation --

def _need_column(kinds: dict, name: Any, kind: str | None = None) -> str:
    if not isinstance(name, str) or name not in kinds:
        raise OpError(f"column '{name}' does not exist in this dataset", "missing_column")
    if kind is not None and kinds[name] != kind:
        raise OpError(f"column '{name}' is {kinds[name]}, but this analysis needs a {kind} column", "wrong_column_type")
    return name


def normalize(op: Any, ctx: AnalysisContext) -> dict:
    """Validate an operation proposed by a model (or built by the engine) and
    return its canonical form. Raises OpError; never runs anything."""
    if not isinstance(op, dict) or op.get("op") not in EXPECTATIONS:
        raise OpError(f"unknown analysis '{op.get('op') if isinstance(op, dict) else op}'", "unknown_operation")
    name = op["op"]
    params = op.get("params") or {}
    if not isinstance(params, dict):
        raise OpError("params must be an object", "invalid_operation")
    kinds = ctx.kinds()
    clean: dict[str, Any] = {}
    if name == "component_change":
        try:
            formula = me.parse_formula(str(params.get("formula", "")))
            problems = me.validate_against_profile(formula, ctx.profile)
            filters = me.normalize_filters(params.get("filters"), list(kinds))
        except me.FormulaError as e:
            raise OpError(str(e), "invalid_formula") from e
        if problems:
            raise OpError("; ".join(problems), "missing_column")
        clean = {"formula": formula.text, "filters": filters}
    elif name == "share_change":
        try:
            filters = me.normalize_filters(params.get("filters"), list(kinds))
        except me.FormulaError as e:
            raise OpError(str(e), "missing_column" if "does not exist" in str(e) else "invalid_filter") from e
        if not filters:
            raise OpError("share_change needs at least one filter", "invalid_operation")
        clean = {"filters": filters}
    elif name == "distribution_shift":
        clean = {"column": _need_column(kinds, params.get("column"), "numeric")}
    elif name == "level_shift":
        if params.get("formula"):
            try:
                formula = me.parse_formula(str(params["formula"]))
                problems = me.validate_against_profile(formula, ctx.profile)
            except me.FormulaError as e:
                raise OpError(str(e), "invalid_formula") from e
            if problems:
                raise OpError("; ".join(problems), "missing_column")
            if formula.text != ctx.formula.text:
                clean["formula"] = formula.text
        if params.get("filters"):
            clean["filters"] = _filters(params["filters"], kinds)
    elif name == "paired_change":
        clean = {"unit": _dimension(kinds, params.get("unit"), ctx)}
    elif name == "composition_change":
        clean = {"dimension": _dimension(kinds, params.get("dimension"), ctx)}
    elif name == "group_difference":
        clean = {"outcome": _need_column(kinds, params.get("outcome"), "numeric"),
                 "dimension": _dimension(kinds, params.get("dimension"), ctx),
                 "period": params.get("period") if params.get("period") in ("earlier", "both") else "later"}
    elif name == "correlation":
        x, y = _need_column(kinds, params.get("x"), "numeric"), _need_column(kinds, params.get("y"), "numeric")
        if x == y:
            raise OpError("correlation needs two different columns", "invalid_operation")
        clean = {"x": min(x, y), "y": max(x, y), "period": params.get("period") if params.get("period") in ("earlier", "later") else "both"}
    elif name == "adjusted_change":
        controls = params.get("controls")
        if not isinstance(controls, list) or not controls:
            raise OpError("adjusted_change needs at least one control column", "invalid_operation")
        outcome = _need_column(kinds, params.get("outcome"), "numeric")
        names = []
        for c in controls[:6]:
            c = _need_column(kinds, c)
            if c == outcome or c == ctx.time_column or kinds[c] == "datetime":
                raise OpError(f"'{c}' cannot be used as a control here", "wrong_column_type")
            if c not in names:
                names.append(c)
        clean = {"outcome": outcome, "controls": sorted(names)}
    elif name in ("segment_contribution", "mix_shift"):
        dim = _need_column(kinds, params.get("dimension"))
        if kinds[dim] == "datetime" or dim == ctx.time_column:
            raise OpError(f"'{dim}' is a date column; it cannot be used as a segment dimension", "wrong_column_type")
        clean = {"dimension": dim}
        if name == "segment_contribution" and params.get("segment") not in (None, ""):
            clean["segment"] = str(params["segment"])[:200]
        if name == "segment_contribution" and params.get("filters"):
            try:
                clean["filters"] = me.normalize_filters(params["filters"], list(kinds))
            except me.FormulaError as e:
                raise OpError(str(e), "missing_column" if "does not exist" in str(e) else "invalid_filter") from e
        if name == "mix_shift" and not (ctx.formula.tree[0] == "agg" and ctx.formula.tree[1] == "sum"):
            raise OpError("mix_shift applies to a summed metric", "not_applicable")
    elif name == "driver_decomposition":
        if not (ctx.formula.tree[0] == "agg" and ctx.formula.tree[1] == "sum"):
            raise OpError("driver_decomposition applies to a summed metric (rows x value per row)", "not_applicable")
    if name in STATISTICAL_OPS:
        # A model's own idea of the method is advice for the selector, recorded with the result.
        suggested = params.get("method") if "method" in params else params.get("suggested_method")
        if isinstance(suggested, str) and suggested.strip():
            clean["suggested_method"] = suggested.strip()[:60]
        if params.get("estimand") == "typical" and name in ("distribution_shift", "paired_change", "component_change"):
            clean["estimand"] = "typical"
    return {"op": name, "params": clean}


# Operations that carry a statistical result (facts["stat"]).
STATISTICAL_OPS = ("component_change", "share_change", "distribution_shift", "level_shift", "pre_trend", "change_point",
                   "paired_change", "composition_change", "group_difference", "correlation", "adjusted_change")


def _filters(raw: Any, kinds: dict) -> list[dict]:
    try:
        return me.normalize_filters(raw, list(kinds))
    except me.FormulaError as e:
        raise OpError(str(e), "missing_column" if "does not exist" in str(e) else "invalid_filter") from e


def _dimension(kinds: dict, name: Any, ctx: "AnalysisContext") -> str:
    dim = _need_column(kinds, name)
    if kinds[dim] == "datetime" or dim == ctx.time_column:
        raise OpError(f"'{dim}' is a date column; it cannot be used to form groups", "wrong_column_type")
    return dim


def signature(op: dict) -> str:
    """Stable identity of a normalised operation. A suggested method does not
    change what is computed, so it is not part of the identity."""
    params = {k: v for k, v in (op.get("params") or {}).items() if k != "suggested_method"}
    return hashlib.sha256(json.dumps({"op": op["op"], "params": params}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def validate_prediction(op_name: str, prediction: Any) -> dict:
    allowed = EXPECTATIONS.get(op_name, ())
    if not isinstance(prediction, dict) or prediction.get("expect") not in allowed:
        got = prediction.get("expect") if isinstance(prediction, dict) else prediction
        raise OpError(f"prediction '{got}' does not fit analysis '{op_name}' (allowed: {', '.join(allowed) or 'none'})",
                      "invalid_prediction")
    return {"expect": prediction["expect"]}


# ------------------------------------------------------------ shared work --

def _filters_key(filters: list[dict] | None) -> str:
    return json.dumps(filters or [], sort_keys=True, default=str)


def period_frame(df: pd.DataFrame, ctx: AnalysisContext, which: str, extra: list[dict] | None = None) -> pd.DataFrame:
    """The rows of the run's population in one period ("a", "b") or in the
    whole table ("all"), optionally narrowed by `extra` filters.

    Filtered once per run and then reused: the result depends only on the
    table, the context's population and periods, and `extra`. Treat the
    returned frame as read-only."""
    key = ("frame", id(df), which, _filters_key(extra))
    if key not in ctx.cache:
        ctx.cache[("table", id(df))] = df          # keeps the table alive, so its id cannot be reused within the run
        period = {"a": ctx.fa, "b": ctx.fb, "all": []}[which]
        ctx.cache[key] = me.apply_filters(df, ctx.base_filters + (extra or []) + period)
    return ctx.cache[key]


def _evaluate_in(df: pd.DataFrame, ctx: AnalysisContext, which: str, formula: me.Formula | str,
                 extra: list[dict] | None = None, group_by: list[str] | None = None) -> dict:
    """me.evaluate on one period's rows. Same result as evaluating the full
    table with the period filters; the rows are just not filtered again."""
    return me.evaluate(period_frame(df, ctx, which, extra), formula, None, group_by=group_by)


# -------------------------------------------------------------- statistics --

def _values(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(float)


def bucket_series(df: pd.DataFrame, ctx: AnalysisContext, formula: me.Formula, filters: list[dict] | None = None) -> dict:
    """The metric as a series of time buckets in each period (see
    _build_bucket_series). Built once per run for a given formula and filters:
    the level-shift, trend and change-point analyses all read the same series.
    Each caller gets its own copy, so none can alter what the others see."""
    key = ("series", id(df), formula.text, _filters_key(filters))
    if key not in ctx.cache:
        ctx.cache[("table", id(df))] = df
        ctx.cache[key] = _build_bucket_series(df, ctx, formula, filters)
    return copy.deepcopy(ctx.cache[key])


def _build_bucket_series(df: pd.DataFrame, ctx: AnalysisContext, formula: me.Formula, filters: list[dict] | None = None) -> dict:
    """The metric as a series of time buckets in each period.

    Granularity: day, unless a period is longer than 180 days, then week,
    then month. Weeks and months that a period only partly covers are left out.

    A bucket with no rows is left out for a non-additive metric (an average
    has no value there). For an additive one (a sum or a count) it counts as 0
    when such buckets are rare (at most 10%) -- an occasional day with no
    sales is a zero -- and is left out when they are common: data recorded
    every second day or on weekdays only has a calendar of its own, and
    padding it with zeros would invent variation that is not there.

    Returns {"a", "b": values, "labels_a", "labels_b", "bucket", "context"}.
    """
    scoped = period_frame(df, ctx, "all", filters)
    when = pd.to_datetime(scoped[ctx.time_column], errors="coerce")
    scoped, when = scoped[when.notna()], when[when.notna()]
    first, last = when.min(), when.max()
    bounds = {}
    for key, (start, end), closed in (("a", ctx.period_a, False), ("b", ctx.period_b, True)):
        lo, hi = max(pd.Timestamp(start), first.normalize()), pd.Timestamp(end)
        hi = min(hi if closed else hi - pd.Timedelta(microseconds=1), last)
        bounds[key] = (lo.normalize(), hi.normalize())
    additive = formula.is_additive
    warnings, assumptions, limitations, excluded = [], [], [], {}

    def build(freq: str) -> dict:
        out = {}
        for key, (lo, hi) in bounds.items():
            if hi < lo:
                out[key] = ([], [])
                continue
            if freq == "D":
                buckets = pd.date_range(lo, hi, freq="D")
                stamp = when.dt.normalize()
            elif freq == "W":
                buckets = pd.date_range(lo + pd.Timedelta(days=(7 - lo.weekday()) % 7), hi - pd.Timedelta(days=6), freq="7D")
                stamp = (when.dt.normalize() - pd.to_timedelta(when.dt.weekday, unit="D"))
            else:
                buckets = pd.date_range(lo + pd.offsets.MonthBegin(0),
                                        (hi + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.offsets.MonthBegin(1), freq="MS")
                stamp = when.dt.to_period("M").dt.to_timestamp()
            frame = scoped.assign(__bucket=stamp.dt.strftime("%Y-%m-%d"))
            frame = frame[frame["__bucket"].isin(set(buckets.strftime("%Y-%m-%d")))]
            got = {g["key"]["__bucket"]: g["value"] for g in me.evaluate(frame, formula, None, group_by=["__bucket"]).get("groups", [])} if len(frame) else {}
            names = list(buckets.strftime("%Y-%m-%d"))
            empty = sum(got.get(label) is None for label in names)
            out[key] = (names, [got.get(label) for label in names], empty, len(buckets))
        # One rule for both periods, so they are always treated alike.
        total, empties = sum(v[3] for v in out.values() if len(v) == 4), sum(v[2] for v in out.values() if len(v) == 4)
        fill = additive and total > 0 and empties <= 0.1 * total
        for key, v in out.items():
            if len(v) == 4:
                kept = [(l, 0.0 if x is None else float(x)) for l, x in zip(v[0], v[1]) if x is not None or fill]
                out[key] = ([l for l, _ in kept], [x for _, x in kept], v[2], v[3])
        out["__fill"] = fill
        return out

    built, name = None, "day"
    for freq, name in (("D", "day"), ("W", "week"), ("M", "month")):
        built = build(freq)
        if any(len(built[k]) < 4 for k in ("a", "b")) or max(built[k][3] for k in ("a", "b")) <= 180 or freq == "M":
            break
    fill = built.pop("__fill")
    for key in ("a", "b"):
        if len(built[key]) < 4:
            built[key] = ([], [], 0, 0)
    empties = {k: built[k][2] for k in ("a", "b")}
    if any(empties.values()):
        excluded["buckets_without_rows"] = {"earlier": empties["a"], "later": empties["b"],
                                            "handling": "counted as 0" if fill else "left out"}
        count = empties["a"] + empties["b"]
        if fill:
            warnings.append(f"{count} {name}(s) have no rows and were counted as 0; if data is missing for those {name}s "
                            "rather than truly zero, the comparison is distorted")
        elif additive:
            warnings.append(f"{count} {name}(s) have no rows, too many to be occasional zeros, so the series uses only "
                            f"{name}s with data; if those {name}s were truly zero, the level per {name} is overstated")
    if name == "day" and built["a"][0] and built["b"][0]:
        mix = {k: np.bincount(pd.to_datetime(built[k][0]).weekday, minlength=7) for k in ("a", "b")}
        gap = int(np.abs(mix["a"] - mix["b"]).max())
        assumptions.append({"name": "comparable calendar windows", "status": "met" if gap <= 1 else "violated",
                            "detail": f"weekday counts differ by at most {gap} between the periods"
                            + ("" if gap <= 1 else "; with a weekly pattern this alone can move the average")})
    if (last - first).days < 730:
        limitations.append("Less than two years of data: a seasonal pattern across the year cannot be told apart from "
                           "the change between these two periods.")
    return {"a": np.array(built["a"][1], float), "b": np.array(built["b"][1], float), "labels_a": built["a"][0],
            "labels_b": built["b"][0], "bucket": name,
            "context": {"warnings": warnings, "assumptions": assumptions, "limitations": limitations, "excluded": excluded,
                        "exclusion_rule": f"{name}s only partly inside a period are left out" if name != "day" else "",
                        "details": {"granularity": name}}}


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return (b - a) / abs(a) * 100


# ---------------------------------------------------------------- execution --

def execute(df: pd.DataFrame, ctx: AnalysisContext, op: dict) -> dict:
    """Run one normalised operation. Returns {"facts", "rows_a", "rows_b",
    "warnings", "table"?}. Raises OpError when the data cannot support it and
    me.FormulaError for engine-level problems."""
    name, p = op["op"], op["params"]
    base, fa, fb = ctx.base_filters, ctx.fa, ctx.fb
    warnings: list[str] = []

    def compare(formula: me.Formula | str, extra: list[dict] | None = None) -> tuple[dict, dict]:
        return _evaluate_in(df, ctx, "a", formula, extra), _evaluate_in(df, ctx, "b", formula, extra)

    if name in ("metric_change", "component_change"):
        formula = ctx.formula if name == "metric_change" else me.parse_formula(p["formula"])
        ra, rb = compare(formula, p.get("filters"))
        a, b = ra["value"], rb["value"]
        facts: dict[str, Any] = {"formula": formula.text, "a": a, "b": b,
                                 "change": None if a is None or b is None else b - a, "change_pct": _pct(a, b)}
        for res in (ra, rb):
            for col, n in (res.get("coerced_non_numeric") or {}).items():
                warnings.append(f"{n} non-numeric value(s) in '{col}' were ignored")
        if name == "component_change" and formula.tree[0] == "agg" and formula.tree[1] == "avg":
            col = formula.tree[2]
            extra = p.get("filters") or []
            facts["stat"] = stat.two_groups(_values(period_frame(df, ctx, "a", extra), col),
                                            _values(period_frame(df, ctx, "b", extra), col), what=col, units=col,
                                            estimand=p.get("estimand", "mean"), suggested=p.get("suggested_method"))
        return {"facts": facts, "rows_a": ra["rows"], "rows_b": rb["rows"], "warnings": sorted(set(warnings))}

    if name == "share_change":
        n_a, n_b = len(period_frame(df, ctx, "a")), len(period_frame(df, ctx, "b"))
        x_a, x_b = len(period_frame(df, ctx, "a", p["filters"])), len(period_frame(df, ctx, "b", p["filters"]))
        s_a, s_b = (x_a / n_a if n_a else None), (x_b / n_b if n_b else None)
        return {"facts": {"filters": p["filters"], "matching_a": x_a, "matching_b": x_b, "share_a": s_a, "share_b": s_b,
                          "share_change_points": None if s_a is None or s_b is None else (s_b - s_a) * 100,
                          "change_pct": _pct(s_a, s_b),
                          "stat": stat.proportions(x_a, n_a, x_b, n_b, what="share of rows matching the filters",
                                                   suggested=p.get("suggested_method"))},
                "rows_a": n_a, "rows_b": n_b, "warnings": []}

    if name == "distribution_shift":
        col = p["column"]
        result = stat.two_groups(_values(period_frame(df, ctx, "a"), col), _values(period_frame(df, ctx, "b"), col),
                                 what=col, units=col, estimand=p.get("estimand", "mean"), suggested=p.get("suggested_method"))
        da, db = result["descriptives"].get("earlier", {}), result["descriptives"].get("later", {})
        ma, mb = da.get("mean"), db.get("mean")
        return {"facts": {"column": col, "mean_a": ma, "mean_b": mb, "change": None if ma is None or mb is None else mb - ma,
                          "change_pct": _pct(ma, mb), "stat": result},
                "rows_a": int(da.get("n", 0)), "rows_b": int(db.get("n", 0)), "warnings": list(result["warnings"])}

    if name in ("level_shift", "pre_trend", "change_point"):
        formula = me.parse_formula(p["formula"]) if p.get("formula") else ctx.formula
        series = bucket_series(df, ctx, formula, p.get("filters"))
        what, bucket, context = (formula.text if p.get("formula") else ctx.label), series["bucket"], series["context"]
        if name == "level_shift":
            result = stat.level_shift(series["a"], series["b"], what=what, bucket=bucket, context=context)
        elif name == "pre_trend":
            result = stat.pre_trend(series["a"], series["b"], what=what, bucket=bucket, context=context)
        else:
            result = stat.change_point(np.concatenate([series["a"], series["b"]]), series["labels_a"] + series["labels_b"],
                                       what=what, bucket=bucket, context=context)
            if result["status"] == "ok" and series["labels_b"]:
                result["details"]["later_period_starts"] = series["labels_b"][0]
        mean_a = float(series["a"].mean()) if len(series["a"]) else None
        mean_b = float(series["b"].mean()) if len(series["b"]) else None
        return {"facts": {"bucket": bucket, "buckets_a": int(len(series["a"])), "buckets_b": int(len(series["b"])),
                          "mean_a": mean_a, "mean_b": mean_b, "change_pct": _pct(mean_a, mean_b), "stat": result},
                "rows_a": int(len(series["a"])), "rows_b": int(len(series["b"])), "warnings": list(result["warnings"])}

    if name == "paired_change":
        unit = p["unit"]
        ga = {str(g["key"][unit]): g["value"] for g in _evaluate_in(df, ctx, "a", ctx.formula, group_by=[unit])["groups"]}
        gb = {str(g["key"][unit]): g["value"] for g in _evaluate_in(df, ctx, "b", ctx.formula, group_by=[unit])["groups"]}
        both = sorted(k for k in ga if k in gb and ga[k] is not None and gb[k] is not None)
        result = stat.paired([ga[k] for k in both], [gb[k] for k in both], what=ctx.label, unit_name=unit,
                             estimand=p.get("estimand", "mean"), suggested=p.get("suggested_method"),
                             unmatched=len(set(ga) | set(gb)) - len(both))
        table = [{"unit": k, "a": ga[k], "b": gb[k], "change": gb[k] - ga[k]} for k in both]
        table.sort(key=lambda r: r["change"])
        down = sum(r["change"] < 0 for r in table)
        return {"facts": {"unit": unit, "units_compared": len(both), "units_down": down, "units_up": sum(r["change"] > 0 for r in table),
                          "mean_a": float(np.mean([ga[k] for k in both])) if both else None, "stat": result},
                "rows_a": len(both), "rows_b": len(both), "warnings": list(result["warnings"]), "table": table[:20]}

    if name == "composition_change":
        dim = p["dimension"]
        da, db = period_frame(df, ctx, "a"), period_frame(df, ctx, "b")
        result = stat.contingency(["earlier"] * len(da) + ["later"] * len(db), pd.concat([da[dim], db[dim]]).to_numpy(),
                                  row_name="period", col_name=dim, suggested=p.get("suggested_method"))
        sa, sb = da[dim].astype(str).value_counts(normalize=True), db[dim].astype(str).value_counts(normalize=True)
        table = sorted(({"segment": k, "share_a": float(sa.get(k, 0.0)), "share_b": float(sb.get(k, 0.0)),
                         "change_points": float((sb.get(k, 0.0) - sa.get(k, 0.0)) * 100)} for k in set(sa.index) | set(sb.index)),
                       key=lambda r: -abs(r["change_points"]))
        return {"facts": {"dimension": dim, "max_share_change_points": abs(table[0]["change_points"]) if table else None,
                          "largest_shift": table[0]["segment"] if table else None, "stat": result},
                "rows_a": len(da), "rows_b": len(db), "warnings": list(result["warnings"]), "table": table[:12]}

    if name == "group_difference":
        scope = {"earlier": base + fa, "later": base + fb}.get(p["period"])
        data = (period_frame(df, ctx, "a" if p["period"] == "earlier" else "b") if scope is not None
                else pd.concat([period_frame(df, ctx, "a"), period_frame(df, ctx, "b")]))
        groups = {str(k): _values(g, p["outcome"]) for k, g in data.groupby(p["dimension"], dropna=True)}
        if len(groups) > config.STAT_MAX_CONTROL_LEVELS:
            raise OpError(f"'{p['dimension']}' has {len(groups)} groups, too many to compare", "not_applicable")
        result = stat.k_groups(groups, what=p["outcome"], group_name=p["dimension"], units=p["outcome"], suggested=p.get("suggested_method"))
        means = {k: d["mean"] for k, d in result["descriptives"].items() if d.get("mean") is not None}
        overall = float(np.nanmean(_values(data, p["outcome"]))) if len(data) else None
        spread = (max(means.values()) - min(means.values())) if len(means) >= 2 else None
        return {"facts": {"outcome": p["outcome"], "dimension": p["dimension"], "period": p["period"], "group_means": means,
                          "largest_gap": spread, "largest_gap_pct": None if not overall or spread is None else spread / abs(overall) * 100,
                          "stat": result},
                "rows_a": len(data), "rows_b": len(data), "warnings": list(result["warnings"])}

    if name == "correlation":
        scope = {"earlier": base + fa, "later": base + fb}.get(p["period"])
        data = (period_frame(df, ctx, "a" if p["period"] == "earlier" else "b") if scope is not None
                else pd.concat([period_frame(df, ctx, "a"), period_frame(df, ctx, "b")]))
        result = stat.correlation(data[p["x"]], data[p["y"]], x_name=p["x"], y_name=p["y"], suggested=p.get("suggested_method"))
        return {"facts": {"x": p["x"], "y": p["y"], "period": p["period"], "stat": result},
                "rows_a": len(data), "rows_b": len(data), "warnings": list(result["warnings"])}

    if name == "adjusted_change":
        da, db = period_frame(df, ctx, "a"), period_frame(df, ctx, "b")
        data = pd.concat([da, db], ignore_index=True)
        later = np.r_[np.zeros(len(da)), np.ones(len(db))]
        result = stat.adjusted_difference(data[p["outcome"]], later, {c: data[c] for c in p["controls"]},
                                          what=p["outcome"], units=p["outcome"])
        mean_a = float(np.nanmean(_values(da, p["outcome"]))) if len(da) else None
        return {"facts": {"outcome": p["outcome"], "controls": p["controls"], "mean_a": mean_a, "stat": result},
                "rows_a": len(da), "rows_b": len(db), "warnings": list(result["warnings"])}

    # The two data checks describe the data itself and need no metric value.
    if name == "period_coverage":
        return _period_coverage(df, ctx)
    if name == "data_quality":
        return _data_quality(df, ctx)

    if ctx.delta is None:
        raise OpError("the metric has no value in one of the periods, so its change cannot be decomposed", "no_metric_value")


    if name == "segment_contribution":
        budget = investigator.Budget(max_engine_calls=10)
        scope = base + (p.get("filters") or [])
        total_a, delta = ctx.total_a, ctx.delta
        if p.get("filters"):  # a drill-down: the change being decomposed is the sub-population's own
            sub_a = _evaluate_in(df, ctx, "a", ctx.formula, p["filters"])["value"]
            sub_b = _evaluate_in(df, ctx, "b", ctx.formula, p["filters"])["value"]
            if sub_a is None or sub_b is None:
                raise OpError("the metric has no value for this sub-population in one of the periods", "no_metric_value")
            total_a, delta = sub_a, sub_b - sub_a
        if delta == 0:
            raise OpError("the metric did not change for this population, so there is nothing to decompose", "no_change")
        # The breakdown does not depend on which segment a hypothesis names, only on the dimension and
        # the population. The scan and a named-segment test therefore share one decomposition.
        breakdown = ("segments", id(df), p["dimension"], _filters_key(p.get("filters")), total_a, delta)
        if breakdown in ctx.cache:
            out = ctx.cache[breakdown]
        elif ctx.formula.is_additive:
            out = investigator._decompose_additive(df, ctx.formula, scope, fa, fb, p["dimension"], delta, budget)
        else:
            parts = investigator._ratio_parts(ctx.formula)
            if parts is None:
                raise OpError("this metric is neither a sum nor a ratio of sums, so segment contributions are not exact", "not_applicable")
            out = investigator._decompose_ratio(df, parts, scope, fa, fb, p["dimension"], total_a, delta, budget)
        ctx.cache[("table", id(df))] = df
        ctx.cache[breakdown] = out
        if out is None:
            raise OpError("the metric could not be decomposed along this dimension", "not_applicable")
        segments = out["segments"]
        additive = ctx.formula.is_additive and total_a
        table = [{"segment": s["segment"], "a": s["a"], "b": s["b"], "contribution": s["contribution"],
                  "share_of_change": s["share_of_change"],
                  # the segment's share of the metric before the change: the yardstick for "concentrated"
                  "base_share": (s["a"] / total_a) if additive and s["a"] is not None else None,
                  "rows_a": s["rows_a"], "rows_b": s["rows_b"], "presence": s["presence"]} for s in segments[:12]]
        return {"facts": {"dimension": p["dimension"], "segment": p.get("segment"), "filters": p.get("filters") or [],
                          "method": out["method"], "reconciles": out["reconciles"], "concentration": out["concentration"],
                          "segments_total": len(segments), "total_change": delta},
                "rows_a": sum(s["rows_a"] for s in segments), "rows_b": sum(s["rows_b"] for s in segments),
                "warnings": [] if len(segments) <= 12 else [f"only the 12 largest of {len(segments)} segments are listed"],
                "table": table}

    if name == "mix_shift":
        col = ctx.formula.tree[2]
        parts = (me.parse_formula(f'sum("{col}")'), me.parse_formula("count()"))
        avg_a = _evaluate_in(df, ctx, "a", f'sum("{col}") / count()')["value"]
        avg_b = _evaluate_in(df, ctx, "b", f'sum("{col}") / count()')["value"]
        if avg_a is None or avg_b is None:
            raise OpError("no rows in one of the periods", "no_metric_value")
        out = investigator._decompose_ratio(df, parts, base, fa, fb, p["dimension"], avg_a, avg_b - avg_a,
                                            investigator.Budget(max_engine_calls=10))
        if out is None:
            raise OpError("the average value could not be decomposed along this dimension", "not_applicable")
        mix, rate = out["mix_effect_total"], out["rate_effect_total"]
        return {"facts": {"dimension": p["dimension"], "average_value_a": avg_a, "average_value_b": avg_b,
                          "average_value_change": avg_b - avg_a, "change_pct": _pct(avg_a, avg_b),
                          "mix_effect": mix, "within_segment_effect": rate, "reconciles": out["reconciles"]},
                "rows_a": sum(s["rows_a"] for s in out["segments"]), "rows_b": sum(s["rows_b"] for s in out["segments"]),
                "warnings": [],
                "table": [{"segment": s["segment"], "mix_effect": s["mix_effect"], "within_segment_effect": s["rate_effect"]}
                          for s in out["segments"][:12]]}

    if name == "driver_decomposition":
        col = ctx.formula.tree[2]
        ra, rb = compare(me.parse_formula(f'count("{col}")'))
        n_a, n_b = ra["value"] or 0.0, rb["value"] or 0.0
        if not n_a or not n_b:
            raise OpError("no rows in one of the periods", "no_metric_value")
        v_a, v_b = ctx.total_a / n_a, ctx.total_b / n_b
        count_effect, value_effect = (n_b - n_a) * v_a, (v_b - v_a) * n_b
        return {"facts": {"rows_with_value_a": n_a, "rows_with_value_b": n_b, "value_per_row_a": v_a, "value_per_row_b": v_b,
                          "count_effect": count_effect, "value_effect": value_effect, "total_change": ctx.delta,
                          "reconciles": math.isclose(count_effect + value_effect, ctx.delta, rel_tol=1e-9,
                                                     abs_tol=1e-9 * max(1.0, abs(ctx.delta))),
                          "identity": "change = (rows_b - rows_a) x value_per_row_a + (value_per_row_b - value_per_row_a) x rows_b"},
                "rows_a": ra["rows"], "rows_b": rb["rows"], "warnings": []}

    if name == "price_volume":
        pv = investigator._price_volume(df, ctx.formula, ctx.profile, base, fa, fb, None, ctx.delta,
                                        investigator.Budget(max_engine_calls=10))
        if pv is None:
            raise OpError("this dataset has no quantity column to split price from volume", "not_applicable")
        return {"facts": {k: pv[k] for k in ("amount_column", "quantity_column", "basis", "price_effect", "volume_effect",
                                             "entry_exit_effect", "reconciles")} | {"total_change": ctx.delta},
                "rows_a": 0, "rows_b": 0, "warnings": [pv["note"]]}

    raise OpError(f"unknown analysis '{name}'", "unknown_operation")


def _period_coverage(df: pd.DataFrame, ctx: AnalysisContext) -> dict:
    scoped = period_frame(df, ctx, "all")
    ts = pd.to_datetime(scoped[ctx.time_column], errors="coerce")
    a0, a1, b0, b1 = (pd.Timestamp(x) for x in (*ctx.period_a, *ctx.period_b))
    in_a, in_b = ts[(ts >= a0) & (ts < a1)], ts[(ts >= b0) & (ts <= b1)]
    span_a = max((a1 - a0).total_seconds() / 86400, 1e-9)
    span_b = max((b1 - b0).total_seconds() / 86400, 1e-9)
    days_a, days_b = int(in_a.dt.normalize().nunique()), int(in_b.dt.normalize().nunique())
    density_a, density_b = days_a / max(span_a, 1.0), days_b / max(span_b, 1.0)
    issues = []
    if span_b < 0.8 * span_a or span_a < 0.8 * span_b:
        issues.append(f"the periods have different lengths ({span_a:.0f} and {span_b:.0f} days)")
    if density_a > 0 and density_b < 0.8 * density_a:
        issues.append(f"the later period has data on {days_b} days against {days_a} in the earlier one")
    if len(in_b) and (b1 - in_b.max()).total_seconds() / 86400 > 0.15 * span_b:
        issues.append(f"the later period's data stops on {in_b.max().date()}, before the period ends on {b1.date()}")
    if not len(in_a) or not len(in_b):
        issues.append("one of the periods has no rows at all")
    return {"facts": {"span_days_a": round(span_a, 2), "span_days_b": round(span_b, 2), "days_with_data_a": days_a,
                      "days_with_data_b": days_b,
                      "first_data_b": str(in_b.min().date()) if len(in_b) else None,
                      "last_data_b": str(in_b.max().date()) if len(in_b) else None,
                      "rows_per_day_a": round(len(in_a) / max(days_a, 1), 3), "rows_per_day_b": round(len(in_b) / max(days_b, 1), 3),
                      "incomplete": bool(issues), "issues": issues},
            "rows_a": int(len(in_a)), "rows_b": int(len(in_b)), "warnings": []}


def _data_quality(df: pd.DataFrame, ctx: AnalysisContext) -> dict:
    da, db_ = period_frame(df, ctx, "a"), period_frame(df, ctx, "b")
    issues, columns = [], {}
    for col in ctx.formula.columns:
        if col not in df.columns:
            continue
        miss_a = float(da[col].isna().mean() * 100) if len(da) else 0.0
        miss_b = float(db_[col].isna().mean() * 100) if len(db_) else 0.0
        bad_b = 0
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            bad_b = int((pd.to_numeric(db_[col], errors="coerce").isna() & db_[col].notna()).sum())
        columns[col] = {"missing_pct_a": round(miss_a, 3), "missing_pct_b": round(miss_b, 3), "non_numeric_b": bad_b}
        if miss_b - miss_a >= 10:
            issues.append(f"missing values in '{col}' rose from {miss_a:.1f}% to {miss_b:.1f}%")
        if bad_b:
            issues.append(f"{bad_b} non-numeric value(s) in '{col}' in the later period")
    dup_a = float(da.duplicated().mean() * 100) if len(da) else 0.0
    dup_b = float(db_.duplicated().mean() * 100) if len(db_) else 0.0
    if abs(dup_b - dup_a) >= 1.0 and max(dup_a, dup_b) >= 1.0:
        issues.append(f"exact duplicate rows went from {dup_a:.1f}% to {dup_b:.1f}% of the period")
    for message in ctx.dq_messages[:5]:
        issues.append(f"open data-quality incident: {message}")
    return {"facts": {"columns": columns, "duplicate_pct_a": round(dup_a, 3), "duplicate_pct_b": round(dup_b, 3),
                      "open_incidents": len(ctx.dq_messages), "issues_present": bool(issues), "issues": issues},
            "rows_a": int(len(da)), "rows_b": int(len(db_)), "warnings": []}


# --------------------------------------------------------------- evaluation --

def _directional(expect: str, change: float | None, pct: float | None, rows_a: int, rows_b: int,
                 result: dict | None, what: str) -> tuple[str, list[str], dict]:
    """Did a quantity move in the predicted direction?

    Without a statistical result (a plain total, say) the measured change is
    judged on direction and materiality alone. With one, three things are kept
    apart: the direction of the estimate, whether the data distinguishes it
    from no change, and whether its size matters:

      distinguishable, predicted direction, material      -> supported
      distinguishable, predicted direction, immaterial    -> weakened   (real but too small to explain much)
      distinguishable, opposite direction                 -> weakened
      not distinguishable, interval inside materiality    -> weakened   (any effect is too small to matter)
      not distinguishable otherwise                       -> inconclusive (not evidence of no effect)
    """
    min_rows = config.HYP_MIN_ROWS
    if result is not None and result["status"] != "ok":
        return INCONCLUSIVE, [f"no statistical conclusion for {what}: {result['reason']}"], {}
    if result is None and min(rows_a, rows_b) < min_rows:
        return INCONCLUSIVE, [f"insufficient sample: {rows_a} and {rows_b} rows (at least {min_rows} needed in each period)"], {}
    if change is None:
        return INCONCLUSIVE, [f"{what} has no value in one of the periods"], {}
    effect = {"measure": "relative change, percent", "value": pct} if pct is not None else {"measure": "change", "value": change}
    material = abs(pct) >= config.HYP_MATERIAL_PCT if pct is not None else change != 0
    moved = "decrease" if change < 0 else "increase"
    size = f"{pct:+.1f}%" if pct is not None else f"{change:+g}"
    if result is None or (result.get("test") or {}).get("p_value") is None:
        if not material:
            return WEAKENED, [f"{what} moved by {pct:+.2f}%, below the {config.HYP_MATERIAL_PCT:g}% materiality threshold"], effect
        if moved != expect:
            return WEAKENED, [f"{what} moved the other way ({size}); the hypothesis predicted a {expect}"], effect
        return SUPPORTED, [f"{what} moved as predicted ({size}). No significance test applies to this quantity."], effect

    p, iv = result["test"]["p_value"], result.get("interval")
    evidence = f"{result['method_label']}, {_p_both(result)}"
    if iv:
        effect["interval"] = [iv["low"], iv["high"]]
        effect["units"] = result.get("units")
        evidence += f"; {iv['level'] * 100:g}% interval for the difference {stat.fmt(iv['low'])} to {stat.fmt(iv['high'])}"
    if _sig(result):
        if moved != expect:
            return WEAKENED, [f"{what} moved the other way ({size}); the hypothesis predicted a {expect} ({evidence})"], effect
        if not material:
            return WEAKENED, [f"{what} moved as predicted but only by {pct:+.2f}%, below the {config.HYP_MATERIAL_PCT:g}% materiality "
                              f"threshold: statistically distinguishable, practically small ({evidence})"], effect
        return SUPPORTED, [f"{what} moved as predicted ({size}) ({evidence})"], effect
    # Not distinguishable from no change. That is only evidence against an effect if the interval rules out one that matters.
    base = abs(change / (pct / 100)) if pct else None
    band = None if base is None else base * config.HYP_MATERIAL_PCT / 100
    # The hypothesis predicts one direction, so what counts against it is the interval excluding a
    # material move in that direction.
    if iv and band is not None and ((expect == "decrease" and iv["low"] > -band) or (expect == "increase" and iv["high"] < band)):
        return WEAKENED, [f"a material {expect} in {what} is not compatible with the data: the interval excludes a move of "
                          f"{config.HYP_MATERIAL_PCT:g}% or more in that direction ({evidence})"], effect
    return INCONCLUSIVE, [f"{what} measured {size}, but the data does not distinguish that from no change ({evidence}). "
                          "This is not evidence that nothing changed"], effect


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _sig(result: dict) -> bool:
    """Meets the significance criterion -- after the multiplicity correction,
    when this test is one of a family."""
    adj = result.get("adjusted")
    return adj["p_value"] < result["alpha"] if adj else bool(result["significant"])


def _p_both(result: dict) -> str:
    text = _p(result["test"]["p_value"])
    adj = result.get("adjusted")
    if adj:
        text += f" ({_p(adj['p_value']).replace('p', 'adjusted p', 1)}, {adj['method']} over {adj['comparisons']} tests)"
    return text


def _stat_gate(result: dict, what: str) -> tuple[str, list[str], dict] | None:
    """A statistical operation whose method could not run is inconclusive, never weakened."""
    if result["status"] != "ok":
        return INCONCLUSIVE, [f"no statistical conclusion for {what}: {result['reason']}"], {}
    return None


def _concentration(row: dict, label: str) -> tuple[str, list[str], dict]:
    """Is the change concentrated in this segment? A segment that holds half
    the metric and accounts for half the change is not a finding -- the change
    is simply spread in proportion. So the share of the change is judged
    against the segment's share of the metric beforehand."""
    share, base_share = row["share_of_change"], row.get("base_share")
    status, reasons, effect = _share("concentrated", share, label)
    if base_share is not None and share is not None:
        effect["base_share"] = base_share
        reasons[0] += f"; it held {base_share * 100:.0f}% of the metric before the change"
        if status == SUPPORTED and share < 1.5 * base_share:
            return WEAKENED, [reasons[0] + ", so the change is roughly in proportion to its size, not concentrated in it"], effect
    return status, reasons, effect


def _share(status_for: str, share: float | None, label: str) -> tuple[str, list[str], dict]:
    effect = {"measure": "share of the total change", "value": share}
    if share is None:
        return INCONCLUSIVE, [f"{label}: the share of the change could not be computed"], effect
    text = f"{label} accounts for {share * 100:.0f}% of the change"
    if share >= config.HYP_STRONG_SHARE:
        return SUPPORTED, [f"{text} (at or above the {config.HYP_STRONG_SHARE * 100:.0f}% threshold)"], effect
    if share <= config.HYP_WEAK_SHARE:
        return WEAKENED, [f"{text} (at or below the {config.HYP_WEAK_SHARE * 100:.0f}% threshold)"], effect
    return INCONCLUSIVE, [f"{text}: a contributor, but not a dominant one"], effect


def _evidence(r: dict) -> str:
    text = f"{r['method_label']}, {_p_both(r)}"
    iv = r.get("interval")
    if iv:
        text += f"; {iv['level'] * 100:g}% interval {stat.fmt(iv['low'])} to {stat.fmt(iv['high'])}"
    return text


def _rule_level_shift(expect, f, r, result):
    per = f"the average {r['sample']['bucket']}ly level" if r["sample"]["bucket"] != "day" else "the average daily level"
    return _directional(expect, r["estimate"], f["change_pct"], result["rows_a"], result["rows_b"], r, per)


def _rule_paired(expect, f, r, result):
    pct = r["estimate"] / abs(f["mean_a"]) * 100 if f.get("mean_a") else None
    status, reasons, effect = _directional(expect, r["estimate"], pct, result["rows_a"], result["rows_b"], r,
                                           f"the change per {f['unit']}")
    reasons[0] += f"; {f['units_down']} of {f['units_compared']} {f['unit']} values went down, {f['units_up']} up"
    return status, reasons, effect


def _rule_pre_trend(expect, f, r, result):
    d = r["details"]
    share, observed = d["share_explained_by_trend"], d["observed_change_per_bucket"]
    effect = {"measure": "share of the change the earlier trend would produce", "value": share}
    slope = f"trend {stat.fmt(r['estimate'])} per {r['sample']['bucket']} in the earlier period ({_evidence(r)})"
    if not _sig(r):
        return INCONCLUSIVE, [f"no clear trend before the later period: {slope}. A trend is not ruled out"], effect
    if share is None or r["estimate"] * observed <= 0:
        return WEAKENED, [f"the earlier period was trending the other way: {slope}"], effect
    if share >= config.HYP_STRONG_SHARE:
        return SUPPORTED, [f"the metric was already moving this way: {slope}; continued, it would produce "
                           f"{share * 100:.0f}% of the observed change"], effect
    if share <= config.HYP_WEAK_SHARE:
        return WEAKENED, [f"{slope}, but continued it would produce only {share * 100:.0f}% of the observed change"], effect
    return INCONCLUSIVE, [f"{slope}; continued, it would produce {share * 100:.0f}% of the observed change: part of the story at most"], effect


def _rule_change_point(expect, f, r, result):
    d = r["details"]
    effect = {"measure": "difference in level after the shift", "value": r["estimate"], "units": r.get("units")}
    where = f"between {d['shift_after']} and {d['first_bucket_after_shift']}"
    if not _sig(r):
        return INCONCLUSIVE, [f"no single abrupt shift stands out ({_evidence(r)}); the most likely position is {where}. "
                              "A gradual change is not ruled out"], effect
    mean_a, mean_b = f.get("mean_a"), f.get("mean_b")
    if mean_a is not None and mean_b is not None and (mean_b - mean_a) * r["estimate"] < 0:
        return WEAKENED, [f"a shift was found {where}, but in the opposite direction to the change between the periods"], effect
    boundary = f"; the later period starts on {d['later_period_starts']}" if d.get("later_period_starts") else ""
    return SUPPORTED, [f"the level shifted {where} by {stat.fmt(r['estimate'])} ({_evidence(r)}){boundary}"], effect


def _rule_composition(expect, f, r, result):
    points = f["max_share_change_points"]
    effect = {"measure": "Cramér's V", "value": r["estimate"], "largest_share_change_points": points}
    text = (f"the largest shift is {f['largest_shift']} ({points:.1f} percentage points of rows); {_evidence(r)}")
    material = points is not None and points >= config.HYP_MATERIAL_PCT
    if _sig(r) and material:
        return SUPPORTED, [f"the mix of rows across {f['dimension']} changed: {text}"], effect
    if not material:
        return WEAKENED, [f"the mix of rows across {f['dimension']} is practically unchanged: {text}"], effect
    return INCONCLUSIVE, [f"the mix across {f['dimension']} moved, but not distinguishably from chance: {text}. "
                          "This is not evidence that it is unchanged"], effect


def _rule_group_difference(expect, f, r, result):
    gap = f["largest_gap_pct"]
    effect = {"measure": "largest gap between group means, percent of the overall mean", "value": gap}
    text = f"{_evidence(r)}; group means range over {stat.fmt(f['largest_gap'])}" + (f" ({gap:.1f}% of the overall mean)" if gap is not None else "")
    if not _sig(r):
        return INCONCLUSIVE, [f"mean {f['outcome']} does not differ distinguishably between {f['dimension']} groups ({text}). "
                              "This is not evidence that the groups are the same"], effect
    if gap is not None and gap < config.HYP_MATERIAL_PCT:
        return WEAKENED, [f"mean {f['outcome']} differs between {f['dimension']} groups, but by a practically small amount ({text})"], effect
    return SUPPORTED, [f"mean {f['outcome']} differs between {f['dimension']} groups ({text})"], effect


def _rule_correlation(expect, f, r, result):
    effect = {"measure": r["method_label"], "value": r["estimate"]}
    text = f"{r['method_label']} {r['estimate']:+.2f} ({_evidence(r)})"
    if not _sig(r):
        return INCONCLUSIVE, [f"{f['x']} and {f['y']}: {text}; no clear association, which is not evidence of none"], effect
    sign = "positive" if r["estimate"] > 0 else "negative"
    if sign != expect:
        return WEAKENED, [f"{f['x']} and {f['y']} are associated the other way: {text}"], effect
    return SUPPORTED, [f"{f['x']} and {f['y']} move together as predicted: {text}. An association, not a cause"], effect


def _rule_adjusted(expect, f, r, result):
    d = r["details"]
    att, raw = d["attenuation"], d["unadjusted"]
    effect = {"measure": "adjusted difference", "value": r["estimate"], "units": r.get("units"),
              "interval": [r["interval"]["low"], r["interval"]["high"]], "share_removed_by_controls": att}
    text = (f"unadjusted difference {stat.fmt(raw['estimate'])}; holding {', '.join(d['controls'])} fixed it is "
            f"{stat.fmt(r['estimate'])} ({_evidence(r)})")
    if raw["p_value"] is None or raw["p_value"] >= r["alpha"]:
        return INCONCLUSIVE, [f"there is no clear unadjusted difference in {f['outcome']} to account for ({text})"], effect
    if att is not None and att >= config.STAT_ATTENUATION_SHARE:
        status = SUPPORTED if expect == "explained" else WEAKENED
        return status, [f"the controls account for {att * 100:.0f}% of the period difference in {f['outcome']}: {text}. "
                        "An adjusted association, not a causal estimate"], effect
    if _sig(r) and (att is None or att <= config.HYP_WEAK_SHARE):
        status = SUPPORTED if expect == "persists" else WEAKENED
        return status, [f"the period difference in {f['outcome']} remains after adjustment: {text}. "
                        "Variables not in the dataset are not ruled out"], effect
    return INCONCLUSIVE, [f"the controls account for part of the difference in {f['outcome']}: {text}"], effect


STAT_RULES = {"level_shift": _rule_level_shift, "paired_change": _rule_paired, "pre_trend": _rule_pre_trend,
              "change_point": _rule_change_point, "composition_change": _rule_composition,
              "group_difference": _rule_group_difference, "correlation": _rule_correlation, "adjusted_change": _rule_adjusted}


def evaluate(op: dict, result: dict, prediction: dict) -> dict:
    """Judge one executed analysis against a prediction. Pure and deterministic."""
    name, expect, f = op["op"], prediction["expect"], result["facts"]
    rows_a, rows_b = result.get("rows_a", 0), result.get("rows_b", 0)
    if name == "component_change":
        status, reasons, effect = _directional(expect, f["change"], f["change_pct"], rows_a, rows_b, f.get("stat"), f["formula"])
    elif name == "share_change":
        change = None if f["share_change_points"] is None else f["share_change_points"]
        status, reasons, effect = _directional(expect, change, f["change_pct"], rows_a, rows_b, f.get("stat"),
                                               "the share of matching rows")
    elif name == "distribution_shift":
        status, reasons, effect = _directional(expect, f["change"], f["change_pct"], rows_a, rows_b, f.get("stat"),
                                               f"the mean of {f['column']}")
    elif name == "segment_contribution":
        table = result.get("table") or []
        total = f["total_change"]
        if f.get("segment"):
            row = next((r for r in table if str(r["segment"]).lower() == f["segment"].lower()), None)
            if row is None:
                status, reasons, effect = WEAKENED, [f"segment '{f['segment']}' is not among the contributors to the change"], {}
            elif min(row["rows_a"], row["rows_b"]) < config.HYP_MIN_ROWS and row["presence"] == "both":
                status, reasons, effect = INCONCLUSIVE, [f"segment '{row['segment']}' has too few rows "
                                                         f"({row['rows_a']} and {row['rows_b']})"], {}
            else:
                status, reasons, effect = _concentration(row, f"{f['dimension']} = {row['segment']}")
        else:
            same_way = [r for r in table if r["share_of_change"] is not None and r["contribution"] * total > 0]
            if not same_way:
                status, reasons, effect = WEAKENED, [f"no {f['dimension']} segment moved in the direction of the total change"], {}
            else:
                top = max(same_way, key=lambda r: r["share_of_change"])
                if min(top["rows_a"], top["rows_b"]) < config.HYP_MIN_ROWS and top["presence"] == "both":
                    status, reasons, effect = INCONCLUSIVE, [f"the leading segment '{top['segment']}' has too few rows"], {}
                else:
                    status, reasons, effect = _concentration(top, f"{f['dimension']} = {top['segment']}")
    elif name == "mix_shift":
        if f["change_pct"] is not None and abs(f["change_pct"]) < config.HYP_MATERIAL_PCT:
            status, reasons, effect = WEAKENED, [f"the average value per row moved by only {f['change_pct']:+.2f}%"], {}
        else:
            magnitude = abs(f["mix_effect"]) + abs(f["within_segment_effect"])
            share = abs(f["mix_effect"]) / magnitude if magnitude else None
            aligned = f["mix_effect"] * f["average_value_change"] > 0
            status, reasons, effect = _share(expect, share if aligned else 0.0, f"the mix across {f['dimension']}")
            effect["measure"] = "share of the change in average value per row"
    elif name in ("driver_decomposition", "price_volume"):
        key = {"count_driven": "count_effect", "value_driven": "value_effect",
               "price_driven": "price_effect", "volume_driven": "volume_effect"}[expect]
        total = f["total_change"]
        share = f[key] / total if total else None
        label = {"count_effect": "the change in the number of rows", "value_effect": "the change in value per row",
                 "price_effect": "the price effect", "volume_effect": "the volume effect"}[key]
        status, reasons, effect = _share(expect, share, label)
        if not f.get("reconciles"):
            status, reasons = INCONCLUSIVE, reasons + ["the decomposition does not reconcile to the total change"]
    elif name == "period_coverage":
        if f["incomplete"]:
            status, reasons = SUPPORTED, ["the periods are not comparable: " + "; ".join(f["issues"])]
        else:
            status, reasons = WEAKENED, ["both periods have comparable length and coverage"]
        effect = {}
    elif name == "data_quality":
        if f["issues_present"]:
            status, reasons = SUPPORTED, ["data problems that could move the metric by themselves: " + "; ".join(f["issues"])]
        else:
            status, reasons = WEAKENED, ["no missing-value, duplicate or incident signal in either period"]
        effect = {}
    elif name in STAT_RULES:
        r = f["stat"]
        status, reasons, effect = _stat_gate(r, name.replace("_", " ")) or STAT_RULES[name](expect, f, r, result)
    else:
        raise OpError(f"analysis '{name}' has no evaluation rule", "invalid_prediction")
    return {"status": status, "reasons": reasons, "effect": effect}


def stat_brief(result: dict | None) -> dict | None:
    """The part of a statistical result worth showing to a model: what was
    estimated, how, the estimate with its interval, the p-value (raw and
    adjusted), the effect sizes, and what qualifies the result: rows or
    buckets that were left out, violated assumptions, warnings and the
    limitations specific to the method. Unavailable values stay None; fields
    that carry nothing (no exclusion, no violated assumption) are left out.
    Every value is copied from the result; nothing is reworded."""
    if not result:
        return None
    if result["status"] != "ok":
        return {"status": "unavailable", "reason": result["reason"], "estimand": result["estimand"]}
    test, iv = result.get("test") or {}, result.get("interval")

    def p_text(p):
        # A very small p-value must not be rounded to 0 on its way into a prompt.
        return p if p is None or p >= 0.001 else "<0.001"

    out = {"method": result["method_label"], "estimand": result["estimand"], "estimate": result["estimate"],
           "interval": None if not iv else [iv["low"], iv["high"]], "level": None if not iv else iv["level"],
           "p_value": p_text(test.get("p_value")), "p_adjusted": p_text((result.get("adjusted") or {}).get("p_value")),
           "significant": result["significant"], "n": result["sample"].get("n"),
           "effects": {e["name"]: e["value"] for e in result.get("effects") or [] if e.get("value") is not None}}
    if result.get("units"):
        out["units"] = result["units"]
    # The few method-specific results a reader cannot do without: where a shift was found,
    # what a difference was before adjustment and how much of it the controls removed.
    details = {k: v for k, v in (result.get("details") or {}).items() if k in _BRIEF_DETAILS and v is not None}
    for key, value in details.items():
        if isinstance(value, dict) and "p_value" in value:      # a p-value inside a detail gets the same treatment
            details[key] = {**value, "p_value": p_text(value["p_value"])}
    if details:
        out["details"] = details
    excluded = _nonzero((result.get("sample") or {}).get("excluded"))
    if excluded:
        out["excluded"] = excluded
    violated = [a["name"] for a in result.get("assumptions") or [] if a["status"] == "violated"]
    if violated:
        out["assumptions_violated"] = violated
    if result.get("warnings"):
        out["warnings"] = [str(w)[:160] for w in result["warnings"][:2]]
    # What this method in particular cannot show. The notes attached to every inferential result
    # (stated once in the report and in the instructions) are not repeated per analysis.
    specific = [text for text in result.get("limitations") or [] if text not in _GENERAL_NOTES]
    if specific:
        out["limitations"] = specific
    return out


_GENERAL_NOTES = (stat_result.FRAME_NOTE, stat_result.OBSERVATIONAL_NOTE, stat_result.BIAS_NOTE)


def _nonzero(value: Any) -> Any:
    """Exclusion counts with the empty ones removed: a count of 0, an empty
    list and a group in which nothing was excluded say nothing. A label that
    describes the counts beside it (how empty buckets were handled) is kept
    only when a count remains."""
    if isinstance(value, dict):
        kept = {k: _nonzero(v) for k, v in value.items()}
        kept = {k: v for k, v in kept.items() if v not in (None, 0, [], {})}
        return kept if any(not isinstance(v, str) for v in kept.values()) else {}
    return value


_BRIEF_DETAILS = ("shift_after", "first_bucket_after_shift", "later_period_starts", "p_value_is_approximate",
                  "unadjusted", "attenuation", "controls")

# The columns of a result table a model needs in order to reason about it; the rest stays in the record.
_BRIEF_TABLE_KEYS = {
    "segment_contribution": ("segment", "contribution", "share_of_change", "base_share"),
    "mix_shift": ("segment", "mix_effect", "within_segment_effect"),
    "paired_change": ("unit", "a", "b", "change"),
    "composition_change": ("segment", "share_a", "share_b", "change_points"),
}
BRIEF_TOP_ROWS = 3


def analysis_brief(op_name: str, result: dict) -> dict:
    """A compact account of one executed analysis for a prompt: its facts, the
    statistical summary, and the leading rows of its table. Nothing here is
    recomputed -- every value is copied from the stored result."""
    facts = result.get("facts") or {}
    brief = {k: v for k, v in facts.items() if k not in ("identity", "stat") and v is not None and v != [] and v != {}}
    means = brief.get("group_means")
    if isinstance(means, dict) and len(means) > 8:
        ordered = sorted(means.items(), key=lambda kv: kv[1])
        brief["group_means"] = dict(ordered[:4] + ordered[-4:])
        brief["groups"] = len(means)
    if facts.get("stat"):
        brief["statistics"] = stat_brief(facts["stat"])
    table = result.get("table") or []
    if table:
        keys = _BRIEF_TABLE_KEYS.get(op_name)
        rows = table[:BRIEF_TOP_ROWS]
        brief["top_rows"] = [{k: r.get(k) for k in keys if k in r} for r in rows] if keys else rows
        if len(table) > BRIEF_TOP_ROWS:
            brief["rows_listed"] = f"top {BRIEF_TOP_ROWS} of {facts.get('segments_total') or len(table)}"
    return brief
