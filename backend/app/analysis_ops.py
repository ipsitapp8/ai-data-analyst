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

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app import config, investigator
from app import metric_engine as me

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
period_coverage       params {}
                      Whether the two periods are comparable: lengths, days with data, a later period
                      that stops early.  expect: "incomplete"
data_quality          params {}
                      Missing values, duplicates and open data-quality incidents that could produce the
                      change by themselves.  expect: "issues_present"
"""


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
    return {"op": name, "params": clean}


def signature(op: dict) -> str:
    """Stable identity of a normalised operation."""
    return hashlib.sha256(json.dumps(op, sort_keys=True, default=str).encode()).hexdigest()[:16]


def validate_prediction(op_name: str, prediction: Any) -> dict:
    allowed = EXPECTATIONS.get(op_name, ())
    if not isinstance(prediction, dict) or prediction.get("expect") not in allowed:
        got = prediction.get("expect") if isinstance(prediction, dict) else prediction
        raise OpError(f"prediction '{got}' does not fit analysis '{op_name}' (allowed: {', '.join(allowed) or 'none'})",
                      "invalid_prediction")
    return {"expect": prediction["expect"]}


# -------------------------------------------------------------- statistics --

def _normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2)))


def welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's unequal-variance test of two means. The p-value uses the normal
    approximation, which is why it is only reported when both samples have at
    least 30 values; below that the test is skipped, not guessed."""
    n_a, n_b = int(len(a)), int(len(b))
    if n_a < 30 or n_b < 30:
        return {"method": "none", "skipped": f"fewer than 30 values in a period ({n_a} and {n_b})", "n_a": n_a, "n_b": n_b}
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = math.sqrt(va / n_a + vb / n_b)
    if se == 0:
        return {"method": "none", "skipped": "no variation in either period", "n_a": n_a, "n_b": n_b}
    z = (float(b.mean()) - float(a.mean())) / se
    pooled = math.sqrt((va + vb) / 2)
    return {"method": "Welch's t-test (normal approximation; both samples have 30+ values)",
            "statistic": z, "p_value": _normal_two_sided_p(z), "n_a": n_a, "n_b": n_b,
            "cohens_d": (float(b.mean()) - float(a.mean())) / pooled if pooled else None}


def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int) -> dict:
    if min(n_a, n_b) < 30 or min(x_a + x_b, (n_a - x_a) + (n_b - x_b)) < 10:
        return {"method": "none", "skipped": "too few rows for a two-proportion test", "n_a": n_a, "n_b": n_b}
    pooled = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return {"method": "none", "skipped": "no variation", "n_a": n_a, "n_b": n_b}
    z = (x_b / n_b - x_a / n_a) / se
    return {"method": "two-proportion z-test", "statistic": z, "p_value": _normal_two_sided_p(z), "n_a": n_a, "n_b": n_b}


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
        extra = extra or []
        return me.evaluate(df, formula, base + extra + fa), me.evaluate(df, formula, base + extra + fb)

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
            va = pd.to_numeric(me.apply_filters(df, base + (p.get("filters") or []) + fa)[col], errors="coerce").dropna().to_numpy(float)
            vb = pd.to_numeric(me.apply_filters(df, base + (p.get("filters") or []) + fb)[col], errors="coerce").dropna().to_numpy(float)
            facts["test"] = welch_test(va, vb)
        return {"facts": facts, "rows_a": ra["rows"], "rows_b": rb["rows"], "warnings": sorted(set(warnings))}

    if name == "share_change":
        n_a, n_b = len(me.apply_filters(df, base + fa)), len(me.apply_filters(df, base + fb))
        x_a, x_b = len(me.apply_filters(df, base + p["filters"] + fa)), len(me.apply_filters(df, base + p["filters"] + fb))
        s_a, s_b = (x_a / n_a if n_a else None), (x_b / n_b if n_b else None)
        return {"facts": {"filters": p["filters"], "matching_a": x_a, "matching_b": x_b, "share_a": s_a, "share_b": s_b,
                          "share_change_points": None if s_a is None or s_b is None else (s_b - s_a) * 100,
                          "change_pct": _pct(s_a, s_b), "test": two_proportion_test(x_a, n_a, x_b, n_b)},
                "rows_a": n_a, "rows_b": n_b, "warnings": []}

    if name == "distribution_shift":
        col = p["column"]
        va = pd.to_numeric(me.apply_filters(df, base + fa)[col], errors="coerce").dropna().to_numpy(float)
        vb = pd.to_numeric(me.apply_filters(df, base + fb)[col], errors="coerce").dropna().to_numpy(float)
        ma, mb = (float(va.mean()) if len(va) else None), (float(vb.mean()) if len(vb) else None)
        return {"facts": {"column": col, "mean_a": ma, "mean_b": mb, "change": None if ma is None or mb is None else mb - ma,
                          "change_pct": _pct(ma, mb), "test": welch_test(va, vb)},
                "rows_a": int(len(va)), "rows_b": int(len(vb)), "warnings": []}

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
            sub_a, sub_b = me.evaluate(df, ctx.formula, scope + fa)["value"], me.evaluate(df, ctx.formula, scope + fb)["value"]
            if sub_a is None or sub_b is None:
                raise OpError("the metric has no value for this sub-population in one of the periods", "no_metric_value")
            total_a, delta = sub_a, sub_b - sub_a
        if delta == 0:
            raise OpError("the metric did not change for this population, so there is nothing to decompose", "no_change")
        if ctx.formula.is_additive:
            out = investigator._decompose_additive(df, ctx.formula, scope, fa, fb, p["dimension"], delta, budget)
        else:
            parts = investigator._ratio_parts(ctx.formula)
            if parts is None:
                raise OpError("this metric is neither a sum nor a ratio of sums, so segment contributions are not exact", "not_applicable")
            out = investigator._decompose_ratio(df, parts, scope, fa, fb, p["dimension"], total_a, delta, budget)
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
        avg_a = me.evaluate(df, f'sum("{col}") / count()', base + fa)["value"]
        avg_b = me.evaluate(df, f'sum("{col}") / count()', base + fb)["value"]
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
    scoped = me.apply_filters(df, ctx.base_filters)
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
    da, db_ = me.apply_filters(df, ctx.base_filters + ctx.fa), me.apply_filters(df, ctx.base_filters + ctx.fb)
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
                 test: dict | None, what: str) -> tuple[str, list[str], dict]:
    min_rows = config.HYP_MIN_ROWS
    if min(rows_a, rows_b) < min_rows:
        return INCONCLUSIVE, [f"insufficient sample: {rows_a} and {rows_b} rows (at least {min_rows} needed in each period)"], {}
    if change is None:
        return INCONCLUSIVE, [f"{what} has no value in one of the periods"], {}
    effect = {"measure": "relative change, percent", "value": pct} if pct is not None else {"measure": "change", "value": change}
    material = abs(pct) >= config.HYP_MATERIAL_PCT if pct is not None else change != 0
    if not material:
        return WEAKENED, [f"{what} moved by {pct:+.2f}%, below the {config.HYP_MATERIAL_PCT:g}% materiality threshold"], effect
    moved = "decrease" if change < 0 else "increase"
    size = f"{pct:+.1f}%" if pct is not None else f"{change:+g}"
    if moved != expect:
        return WEAKENED, [f"{what} moved the other way ({size}); the hypothesis predicted a {expect}"], effect
    if test and test.get("p_value") is not None:
        if test["p_value"] >= config.HYP_SIGNIFICANCE_ALPHA:
            return INCONCLUSIVE, [f"{what} moved as predicted ({size}), but the difference is not statistically "
                                  f"distinguishable from noise (p = {test['p_value']:.3f})"], effect
        return SUPPORTED, [f"{what} moved as predicted ({size}); {test['method']}, p = {test['p_value']:.4f}"], effect
    note = f" No significance test was applied ({test['skipped']})." if test and test.get("skipped") else ""
    return SUPPORTED, [f"{what} moved as predicted ({size}).{note}"], effect


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


def evaluate(op: dict, result: dict, prediction: dict) -> dict:
    """Judge one executed analysis against a prediction. Pure and deterministic."""
    name, expect, f = op["op"], prediction["expect"], result["facts"]
    rows_a, rows_b = result.get("rows_a", 0), result.get("rows_b", 0)
    if name == "component_change":
        status, reasons, effect = _directional(expect, f["change"], f["change_pct"], rows_a, rows_b, f.get("test"), f["formula"])
    elif name == "share_change":
        change = None if f["share_change_points"] is None else f["share_change_points"]
        status, reasons, effect = _directional(expect, change, f["change_pct"], rows_a, rows_b, f.get("test"),
                                               "the share of matching rows")
    elif name == "distribution_shift":
        status, reasons, effect = _directional(expect, f["change"], f["change_pct"], rows_a, rows_b, f.get("test"),
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
    else:
        raise OpError(f"analysis '{name}' has no evaluation rule", "invalid_prediction")
    return {"status": status, "reasons": reasons, "effect": effect}
