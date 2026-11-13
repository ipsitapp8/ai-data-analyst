"""Hypothetical scenarios on the user's own data, with every assumption named.

Three kinds, all arithmetic on observed figures unless a trained model is used:

* column_change  "what if salaries increase by 10%": the column's total and
  average before and after. If this conversation's Random Forest uses the
  column as an input, the model's average prediction is recomputed with the
  changed values and reported separately, labelled as a model-based estimate.
* conversion_change  "conversion improves from 3% to 5%": expected conversions
  and revenue from the observed leads and observed revenue per conversion.
* rate_change  "churn decreases by 5%": a share of rows with one value of a
  categorical column; relative change and percentage points must not be
  confused, so the unit has to be stated.

Nothing here predicts how the world responds to a change; the result says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analyst.business import Insufficient
from app.analyst.data import business_roles, finite, kind_of

HYPOTHETICAL = ("Hypothetical: this is arithmetic under the stated assumptions, not an observed result or a "
                "forecast of what will happen.")


def column_change(df: pd.DataFrame, column: str, change_pct: float, memory: dict | None = None,
                  group: str | None = None) -> dict:
    if column not in df.columns:
        raise Insufficient(f"There is no column named '{column}'.")
    if kind_of(df[column]) != "numeric":
        raise Insufficient(f"'{column}' is not numeric, so a percentage change does not apply to it.")
    if not -100 < change_pct <= 500:
        raise Insufficient("The change must be between -100% and +500%.")
    x = finite(df[column])
    p = change_pct / 100
    out = {"kind": "column_change", "column": column, "change_pct": change_pct, "rows": int(len(x)),
           "observed": {"total": float(x.sum()), "average": float(x.mean())},
           "hypothetical": {"total": float(x.sum() * (1 + p)), "average": float(x.mean() * (1 + p))},
           "assumptions": [f"Every value of '{column}' changes by {change_pct:+g}%.",
                           "Nothing else in the data changes as a consequence (no knock-on effects are assumed)."],
           "caveat": HYPOTHETICAL}
    out["difference"] = {"total": out["hypothetical"]["total"] - out["observed"]["total"],
                         "average": out["hypothetical"]["average"] - out["observed"]["average"]}
    if group and group in df.columns:
        g = df.groupby(group)[column].sum()
        out["by_group"] = [{"group": str(k), "observed": float(v), "hypothetical": float(v * (1 + p)),
                            "difference": float(v * p)} for k, v in g.sort_values(ascending=False).items()]
    trained = (memory or {}).get("model")
    if trained and column in trained["features"] and column != trained["target"]:
        out["model_estimate"] = _model_estimate(df, trained, column, p)
    return out


def _model_estimate(df: pd.DataFrame, trained: dict, column: str, p: float) -> dict:
    data = df[df[trained["target"]].notna()]
    X = data[trained["features"]].copy()
    for c in X.columns:
        if kind_of(X[c]) != "numeric":
            X[c] = X[c].astype("object").where(X[c].notna(), np.nan).astype(str).replace("nan", np.nan)
    X2 = X.copy()
    X2[column] = pd.to_numeric(X2[column], errors="coerce") * (1 + p)
    model = trained["_model"]
    out = {"target": trained["target"], "task": trained["task"], "rows": int(len(X)),
           "caution": ("Model-based estimate: the Random Forest learned associations in this data, not causes, and a "
                       "forest cannot extrapolate beyond the values it saw. Treat it as a rough indication.")}
    if trained["task"] == "regression":
        before, after = float(model.predict(X).mean()), float(model.predict(X2).mean())
        out.update(average_prediction_before=before, average_prediction_after=after, difference=after - before)
    else:
        classes = list(model.classes_)
        pb, pa = model.predict_proba(X).mean(axis=0), model.predict_proba(X2).mean(axis=0)
        out["class_shares"] = [{"class": str(c), "before": float(b), "after": float(a), "difference": float(a - b)}
                               for c, b, a in zip(classes, pb, pa)]
        out["caution"] += (" The forest was trained with class weighting, so its average probability for a rare class "
                           "is higher than that class's actual share: compare before with after, not with the data.")
    return out


def conversion_change(df: pd.DataFrame, to_rate: float, from_rate: float | None = None) -> dict:
    roles = business_roles(df)["roles"]
    need = [r for r in ("leads", "conversions") if r not in roles]
    if need:
        raise Insufficient("A conversion scenario needs a leads/visitors column and a conversions column; this dataset "
                           f"has no {' or '.join(need)} column.")
    if not 0 < to_rate <= 1:
        raise Insufficient("The new conversion rate must be between 0% and 100%.")
    leads = float(df[roles["leads"]].sum())
    conv = float(df[roles["conversions"]].sum())
    observed_rate = conv / leads if leads else None
    out = {"kind": "conversion_change", "columns": {k: roles[k] for k in ("leads", "conversions", "revenue") if k in roles},
           "observed": {"leads": leads, "conversions": conv, "conversion_rate": observed_rate},
           "assumed_from_rate": from_rate, "to_rate": to_rate,
           "assumptions": [f"Leads stay at the observed {leads:,.0f}.",
                           f"The conversion rate becomes {to_rate * 100:g}%."],
           "caveat": HYPOTHETICAL}
    new_conv = leads * to_rate
    out["hypothetical"] = {"conversions": new_conv}
    if "revenue" in roles:
        rev = float(df[roles["revenue"]].sum())
        per = rev / conv if conv else None
        out["observed"].update(revenue=rev, revenue_per_conversion=per)
        out["assumptions"].append(f"Each conversion brings the observed average revenue of {per:,.2f}." if per else
                                  "Revenue per conversion is unknown.")
        if per:
            out["hypothetical"]["revenue"] = new_conv * per
            out["difference"] = {"conversions": new_conv - conv, "revenue": new_conv * per - rev}
            if from_rate:
                base = leads * from_rate * per
                out["from_rate_revenue"] = base
                out["difference_vs_from_rate"] = new_conv * per - base
    else:
        out["difference"] = {"conversions": new_conv - conv}
        out["assumptions"].append("There is no revenue column, so only conversions are estimated.")
    if from_rate is not None and observed_rate is not None and abs(from_rate - observed_rate) > 0.005:
        out["note"] = (f"You said the current rate is {from_rate * 100:g}%; the data shows {observed_rate * 100:.2f}%. "
                       "Both baselines are shown.")
    return out


def rate_change(df: pd.DataFrame, column: str, value: str, change: float, unit: str) -> dict:
    """Share of rows where `column == value`, changed by `change` (relative % or percentage points)."""
    if unit not in ("relative", "points"):
        raise Insufficient("State whether the change is relative (e.g. 5% of the current rate) or in percentage points.")
    s = df[column].dropna().astype(str)
    n = len(s)
    k = int((s == value).sum())
    rate = k / n if n else 0.0
    new_rate = rate * (1 + change / 100) if unit == "relative" else rate + change / 100
    if not 0 <= new_rate <= 1:
        raise Insufficient(f"That change would take the rate outside 0–100% (from {rate * 100:.1f}%).")
    out = {"kind": "rate_change", "column": column, "value": value, "rows": n, "unit": unit, "change": change,
           "observed": {"count": k, "rate": rate}, "hypothetical": {"count": new_rate * n, "rate": new_rate},
           "difference": {"count": new_rate * n - k, "rate_points": (new_rate - rate) * 100},
           "assumptions": [f"The share of '{value}' in '{column}' moves from {rate * 100:.1f}% to {new_rate * 100:.1f}% "
                           f"({'relative change of ' + format(change, '+g') + '%' if unit == 'relative' else format(change, '+g') + ' percentage points'}).",
                           f"The number of rows stays at {n}."],
           "caveat": HYPOTHETICAL}
    money = [c for c in df.columns if kind_of(df[c]) == "numeric" and
             any(w in c.lower() for w in ("charge", "fee", "revenue", "value", "spend", "price", "amount"))]
    if money:
        col = money[0]
        per = float(pd.to_numeric(df.loc[df[column].astype(str) == value, col], errors="coerce").mean())
        if per == per:
            out["value_column"] = col
            out["value_per_row"] = per
            out["value_difference"] = (new_rate * n - k) * per
            out["assumptions"].append(f"Each affected row is worth the observed average '{col}' of rows with "
                                      f"'{value}', {per:,.2f}.")
    return out
