"""The statistical result: one validated, versioned shape for every method.

A value that could not be computed is None, never 0. `build` refuses a result
whose numbers are not internally possible (a p-value outside [0, 1], an
interval whose ends are reversed, a non-finite estimate)."""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = "1"
METHODS_VERSION = "1"

# Said of every inferential result: what an interval or test can and cannot fix.
FRAME_NOTE = ("The dataset holds every recorded row for the compared groups, so the measured difference is exact for "
              "this data. The interval and test answer a narrower question: whether a difference of this size is "
              "larger than ordinary variation between rows would produce.")
OBSERVATIONAL_NOTE = ("The data is observational. A statistical association, however strong, does not show what "
                      "caused it; other things that changed at the same time are not ruled out.")
BIAS_NOTE = "An interval or a test does not correct for missing, duplicated or unrepresentative records."


class StatResultError(ValueError):
    """A method produced a result that is not internally valid."""


def num(value: Any) -> float | None:
    """A finite float, or None. NaN and infinity are 'unavailable', not numbers."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Interval(BaseModel):
    low: float
    high: float
    level: float = Field(gt=0, lt=1)
    method: str

    @model_validator(mode="after")
    def _ordered(self):
        if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.low > self.high:
            raise ValueError("interval ends must be finite and ordered")
        return self


class Test(BaseModel):
    name: str
    null: str
    alternative: str
    statistic_name: str | None = None
    statistic: float | None = None
    df: float | list[float] | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    exact: bool = False
    resamples: int | None = None

    @field_validator("statistic")
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("test statistic must be finite or unavailable")
        return v


class Effect(BaseModel):
    name: str
    value: float | None
    units: str | None = None
    interval: Interval | None = None
    note: str | None = None

    @field_validator("value")
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("effect size must be finite or unavailable")
        return v


class Assumption(BaseModel):
    name: str
    status: Literal["met", "violated", "not_checked", "not_applicable"]
    detail: str


class Adjusted(BaseModel):
    p_value: float = Field(ge=0, le=1)
    method: str
    family: str
    comparisons: int = Field(ge=1)


class StatResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    methods_version: str = METHODS_VERSION
    status: Literal["ok", "unavailable"]
    reason: str | None = None                      # why unavailable
    method: str                                    # stable identifier, e.g. "welch_t"
    method_label: str
    design: Literal["independent_groups", "paired", "association", "time_series", "regression", "descriptive"]
    estimand: str                                  # what quantity is being estimated, in words
    units: str | None = None
    selection: list[str] = []                      # why this method
    rejected_methods: list[dict] = []              # [{"method", "reason"}]
    suggested_method: dict | None = None           # {"method", "outcome": accepted|overridden|unknown, "reason"}
    sample: dict = {}                              # n, per-group n, exclusions
    descriptives: dict = {}
    assumptions: list[Assumption] = []
    estimate: float | None = None
    interval: Interval | None = None
    test: Test | None = None
    effects: list[Effect] = []
    alpha: float = Field(gt=0, lt=1)
    significant: bool | None = None                # raw p < alpha; None when there is no p-value
    adjusted: Adjusted | None = None
    comparisons: list[dict] = []                   # pairwise results inside one analysis
    warnings: list[str] = []
    limitations: list[str] = []
    interpretation: list[str] = []
    details: dict = {}                             # method-specific extras (coefficients, diagnostics)

    @field_validator("estimate")
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("estimate must be finite or unavailable")
        return v

    @model_validator(mode="after")
    def _coherent(self):
        if self.status == "unavailable" and not self.reason:
            raise ValueError("an unavailable result must say why")
        p = self.test.p_value if self.test else None
        if self.status == "ok" and (self.significant is None) != (p is None):
            raise ValueError("significance is stated exactly when a p-value exists")
        if p is not None and self.significant != (p < self.alpha):
            raise ValueError("significance must follow from the p-value and alpha")
        return self


def fmt(value: float | None, units: str | None = None) -> str:
    if value is None:
        return "unavailable"
    a = abs(value)
    text = f"{value:,.0f}" if a >= 1000 else f"{value:,.2f}" if a >= 1 else f"{value:.3g}" if a else "0"
    return f"{text} {units}" if units else text


def p_text(p: float | None) -> str:
    if p is None:
        return "no p-value"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def interpret(r: dict) -> list[str]:
    """Plain-language reading of a result. Fixed wording: a non-significant
    result is never read as 'no effect', a significant one never as 'large'
    or 'caused by'."""
    if r["status"] != "ok":
        return [f"No statistical conclusion: {r['reason']}"]
    out = []
    units = r.get("units")
    if r.get("estimate") is not None:
        line = f"Estimated {r['estimand']}: {fmt(r['estimate'], units)}"
        iv = r.get("interval")
        if iv:
            line += f" ({iv['level'] * 100:g}% interval {fmt(iv['low'])} to {fmt(iv['high'])})"
        out.append(line + ".")
        if iv:
            out.append(f"The interval is the range of values of the {r['estimand']} that the data is reasonably compatible "
                       f"with. It is not a {iv['level'] * 100:g}% probability that the true value lies inside, and it says "
                       "nothing about errors in how the data was collected.")
    t = r.get("test")
    if t and t.get("p_value") is not None:
        p = t["p_value"]
        if r["significant"]:
            out.append(f"{t['name']}: {p_text(p)}. A difference this large would be unusual if {t['null']}. "
                       "This is not a statement about how large or how important the difference is, or about its cause.")
        else:
            out.append(f"{t['name']}: {p_text(p)}. The data does not distinguish this from {t['null']}. "
                       "That is not evidence that there is no difference: the interval shows which sizes remain plausible.")
        adj = r.get("adjusted")
        if adj:
            out.append(f"After {adj['method']} correction across {adj['comparisons']} tests ({adj['family']}): {p_text(adj['p_value'])}.")
    return out


def build(**fields: Any) -> dict:
    """Validate and return a result as a plain dict (JSON-serialisable)."""
    for key in ("estimate",):
        if key in fields:
            fields[key] = num(fields[key])
    fields.setdefault("limitations", [])
    try:
        result = StatResult(**fields).model_dump()
    except ValidationError as e:
        raise StatResultError(str(e)) from e
    if not result["interpretation"]:
        result["interpretation"] = interpret(result)
    return result


def unavailable(*, method: str, method_label: str, design: str, estimand: str, reason: str, alpha: float,
                **fields: Any) -> dict:
    return build(status="unavailable", reason=reason, method=method, method_label=method_label, design=design,
                 estimand=estimand, alpha=alpha, **fields)


def with_adjustment(result: dict, p_adjusted: float, method: str, family: str, comparisons: int) -> dict:
    """A copy of `result` carrying its multiplicity-adjusted p-value."""
    out = {**result, "adjusted": {"p_value": p_adjusted, "method": method, "family": family, "comparisons": comparisons},
           "interpretation": []}
    return build(**out)
