"""Historical campaign performance, recommendations and budget scenarios.

Every figure is computed from the dataset's own columns, chosen by
`data.business_roles` with the reason recorded. Recommendations follow fixed,
stated rules on that evidence (see `recommend`); they are observational: the
data shows what happened where money was spent, not what spending caused.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.analyst.data import business_roles, finite
from app.stats import methods as sm

LEVEL = 0.95
MIN_ROWS_PER_GROUP = 3           # campaign rows per area before its figures are compared at all
OBSERVATIONAL = ("Observational data: it shows results where money was spent, not what spending caused "
                 "(areas that received more budget may differ in other ways).")


class Insufficient(ValueError):
    """The data cannot support the analysis; the message says what is missing."""


def _ratio(a: float, b: float) -> float | None:
    return None if b in (0, None) or b != b else float(a / b)


def performance(df: pd.DataFrame, group: str | None = None, focus: str | None = None) -> dict:
    """Per group: rows, spend, revenue, profit, ROI, CAC, conversion rate with a
    Wilson interval; and, with `focus`, that group against all others."""
    found = business_roles(df)
    roles = found["roles"]
    group = group or roles.get("area") or roles.get("campaign")
    missing = [r for r in ("spend",) if r not in roles]
    if not ({"revenue", "conversions"} & roles.keys()):
        missing.append("revenue or conversions/customers acquired")
    if group is None:
        missing.append("an area, region or campaign column to compare")
    if missing:
        raise Insufficient("This dataset does not contain the advertising history needed: no "
                           + ", no ".join(missing) + " column was found."
                           + (f" (Ambiguous columns: {found['ambiguous']}.)" if found["ambiguous"] else ""))
    if group not in df.columns:
        raise Insufficient(f"There is no column named '{group}'.")
    if focus is not None and focus not in set(df[group].astype(str)):
        known = sorted(df[group].dropna().astype(str).unique())[:12]
        raise Insufficient(f"'{focus}' does not occur in '{group}' (values: {', '.join(known)}).")

    spend, rev, conv, leads = roles.get("spend"), roles.get("revenue"), roles.get("conversions"), roles.get("leads")
    profit_col = roles.get("profit")
    rows = []
    for name, g in df.groupby(df[group].astype(str), sort=True):
        r = {"group": name, "rows": int(len(g)), "spend": float(g[spend].sum())}
        if rev:
            r["revenue"] = float(g[rev].sum())
            r["profit"] = float(g[profit_col].sum()) if profit_col else r["revenue"] - r["spend"]
            r["roi"] = _ratio(r["revenue"] - r["spend"], r["spend"])
        if conv:
            r["conversions"] = float(g[conv].sum())
            r["cac"] = _ratio(r["spend"], r["conversions"])
            if rev:
                r["revenue_per_conversion"] = _ratio(r["revenue"], r["conversions"])
        if leads and conv:
            r["leads"] = float(g[leads].sum())
            r["conversion_rate"] = _ratio(r["conversions"], r["leads"])
            ci = sm.wilson(int(round(r["conversions"])), int(round(r["leads"])), LEVEL) if r["leads"] >= 1 else None
            r["conversion_rate_ci"] = list(ci) if ci else None
        if rev:
            per_row = finite((g[rev] - g[spend]) / g[spend].replace(0, np.nan))
            r["roi_per_row_mean"] = float(per_row.mean()) if len(per_row) else None
            ci = sm.mean_interval(per_row, LEVEL) if len(per_row) >= 2 else None
            r["roi_per_row_ci"] = list(ci) if ci else None
        rows.append(r)

    out = {"group_column": group, "columns_used": {k: v for k, v in roles.items() if k in
                                                    ("spend", "revenue", "profit", "conversions", "leads", "date")},
           "column_evidence": {k: v for k, v in found["evidence"].items()
                               if (k in ("spend", "revenue", "profit", "conversions", "leads", "date") and k in roles)
                               or roles.get(k) == group},
           "groups": rows, "rows_total": int(len(df))}
    key = "roi" if rev else "cac"
    ranked = [r for r in rows if r.get(key) is not None and r["rows"] >= MIN_ROWS_PER_GROUP]
    ranked.sort(key=lambda r: -r[key] if key == "roi" else r[key])
    out["ranking_metric"] = "ROI ((revenue - spend) / spend)" if key == "roi" else "CAC (spend / conversions)"
    out["ranking"] = [r["group"] for r in ranked]
    out["too_few_rows"] = [r["group"] for r in rows if r["rows"] < MIN_ROWS_PER_GROUP]
    if focus is None and ranked:
        focus_eval = ranked[0]["group"]
    else:
        focus_eval = focus
    if focus_eval is not None:
        out["focus"] = focus_eval
        out["comparison"] = _compare(df, group, focus_eval, roles)
    if roles.get("date") and focus_eval is not None:
        out["trend"] = _trend(df[df[group].astype(str) == focus_eval], roles)
    return out


def _compare(df: pd.DataFrame, group: str, focus: str, roles: dict) -> dict:
    """The focus group against all other groups combined."""
    a = df[df[group].astype(str) != focus]
    b = df[df[group].astype(str) == focus]
    spend, rev, conv, leads = roles.get("spend"), roles.get("revenue"), roles.get("conversions"), roles.get("leads")
    res: dict = {"focus_rows": int(len(b)), "other_rows": int(len(a))}
    if rev:
        ra = finite((a[rev] - a[spend]) / a[spend].replace(0, np.nan))
        rb = finite((b[rev] - b[spend]) / b[spend].replace(0, np.nan))
        if len(ra) >= 2 and len(rb) >= 2:
            w = sm.welch(ra, rb, LEVEL)
            res["roi_test"] = {"method": "Welch's t-test on per-campaign ROI (focus minus others)",
                               "focus_mean": float(rb.mean()), "other_mean": float(ra.mean()),
                               "difference": w["estimate"], "ci": [w["interval"]["low"], w["interval"]["high"]],
                               "p_value": w["test"]["p_value"], "hedges_g": sm.hedges_g(ra, rb),
                               "n_focus": int(len(rb)), "n_other": int(len(ra))}
            if len(rb) < 10:
                res["roi_test"]["caution"] = (f"only {len(rb)} campaign rows in '{focus}': the t-test's normality "
                                              "assumption cannot be checked, so treat the p-value as approximate")
    if leads and conv:
        xa, na = int(round(a[conv].sum())), int(round(a[leads].sum()))
        xb, nb = int(round(b[conv].sum())), int(round(b[leads].sum()))
        if na > 0 and nb > 0 and xa <= na and xb <= nb:
            ci = sm.newcombe(xa, na, xb, nb, LEVEL)
            table = np.array([[xb, nb - xb], [xa, na - xa]])
            exp_ok = sm.cochran_expected_ok(sm.expected_counts(table))
            test = sm.chi_square(table) if exp_ok else sm.fisher(table)
            res["conversion_test"] = {"method": test["name"] + " on conversions vs leads",
                                      "focus_rate": xb / nb, "other_rate": xa / na, "difference": xb / nb - xa / na,
                                      "ci": list(ci) if ci else None, "p_value": test["p_value"],
                                      "assumption": "expected counts adequate for chi-square" if exp_ok
                                      else "expected counts too small for chi-square, so Fisher's exact test was used"}
    return res


def _trend(g: pd.DataFrame, roles: dict) -> dict | None:
    """First half vs second half of the period, by date."""
    date, spend, rev = roles["date"], roles.get("spend"), roles.get("revenue")
    if not rev or len(g) < 4:
        return None
    g = g.sort_values(date)
    half = len(g) // 2
    early, late = g.iloc[:half], g.iloc[half:]
    roi = lambda x: _ratio(x[rev].sum() - x[spend].sum(), x[spend].sum())
    return {"early_period": [str(early[date].min().date()), str(early[date].max().date())], "early_roi": roi(early),
            "late_period": [str(late[date].min().date()), str(late[date].max().date())], "late_roi": roi(late)}


def recommend(perf: dict, question_kind: str = "where") -> dict:
    """A recommendation from the performance evidence by fixed rules:

    * strong   the focus beats the others with a 95% interval excluding zero
               (ROI or conversion rate) and has >= 10 campaign rows
    * suggestive  better point estimates, but the interval includes zero or
               the sample is small -> a limited, controlled test
    * negative the focus loses money (ROI < 0) -> do not expand
    """
    focus = perf.get("focus")
    row = next((r for r in perf["groups"] if r["group"] == focus), None)
    if row is None:
        raise Insufficient("No group has enough campaign rows to evaluate.")
    comp = perf.get("comparison") or {}
    roi_t, conv_t = comp.get("roi_test"), comp.get("conversion_test")
    better_ci = any(t and t.get("ci") and t["ci"][0] > 0 for t in (roi_t, conv_t))
    worse_ci = any(t and t.get("ci") and t["ci"][1] < 0 for t in (roi_t, conv_t))
    n = row["rows"]
    roi = row.get("roi")
    evidence_level = "strong" if better_ci and n >= 10 else "weak"
    if roi is not None and roi < 0:
        decision = "do_not_expand"
        text = (f"Do not increase advertising in {focus}: historically it lost money (ROI below zero)."
                if question_kind == "increase" else
                f"Do not prioritise {focus}: historically its campaigns lost money (ROI below zero).")
    elif worse_ci:
        decision = "do_not_expand"
        text = f"Do not prioritise {focus}: it performed significantly worse than the other areas."
    elif evidence_level == "strong":
        decision = "expand"
        text = (f"Increase the budget in {focus} gradually, monitoring ROI as spend grows."
                if question_kind == "increase" else f"Prioritise {focus} for the next campaign.")
    else:
        decision = "test"
        text = (f"Test a limited budget increase in {focus} as a controlled experiment before committing."
                if question_kind == "increase" else
                f"Test advertising in {focus} with a limited budget before committing.")
    limitations = [OBSERVATIONAL]
    if n < 10:
        limitations.append(f"Small sample: {focus} has only {n} campaign rows.")
    if perf.get("too_few_rows"):
        limitations.append("Not compared (fewer than %d rows): %s." % (MIN_ROWS_PER_GROUP, ", ".join(perf["too_few_rows"])))
    if "conversion_test" not in comp:
        limitations.append("No leads/visits column, so conversion rates could not be compared.")
    limitations.append("Past returns may not hold at a higher budget (diminishing returns are not visible in this data).")
    next_action = ("Run a controlled experiment: raise spend in part of the area (or for a period) and keep a "
                   "comparable holdout, then compare conversions and ROI between them.")
    return {"decision": decision, "evidence_level": evidence_level, "recommendation": text, "focus": focus,
            "limitations": limitations, "next_action": next_action,
            "rules": "strong = 95% interval of the difference excludes zero and >= 10 campaign rows; "
                     "negative ROI = do not expand; otherwise a limited controlled test"}


def budget_scenario(df: pd.DataFrame, change_pct: float, group: str | None = None, focus: str | None = None) -> dict:
    """What a budget change would mean under explicitly stated assumptions --
    arithmetic, not a forecast. Two bounds: returns scale with spend (constant
    ROI and CAC) and no extra return at all; plus the break-even."""
    if not -100 < change_pct <= 500:
        raise Insufficient("A budget change must be between -100% and +500%.")
    roles = business_roles(df)["roles"]
    if "spend" not in roles or "revenue" not in roles:
        raise Insufficient("A budget scenario needs spend and revenue columns.")
    sub = df
    group = group or roles.get("area") or roles.get("campaign")
    if focus is not None and group:
        sub = df[df[group].astype(str) == focus]
        if sub.empty:
            raise Insufficient(f"'{focus}' does not occur in '{group}'.")
    s, r = float(sub[roles["spend"]].sum()), float(sub[roles["revenue"]].sum())
    p = change_pct / 100
    new_spend = s * (1 + p)
    out = {"scope": focus or "all rows", "change_pct": change_pct, "baseline_spend": s, "baseline_revenue": r,
           "baseline_roi": _ratio(r - s, s), "new_spend": new_spend, "extra_spend": new_spend - s,
           "if_returns_scale": {"assumption": "revenue grows in proportion to spend (constant ROI)",
                                "revenue": r * (1 + p), "profit": r * (1 + p) - new_spend, "roi": _ratio(r * (1 + p) - new_spend, new_spend)},
           "if_no_response": {"assumption": "the extra spend brings no extra revenue",
                              "revenue": r, "profit": r - new_spend, "roi": _ratio(r - new_spend, new_spend)},
           "break_even_extra_revenue": new_spend - s,
           "caveat": "Scenario arithmetic under stated assumptions, not a prediction: the data does not show how "
                     "revenue responds to more spend."}
    if "conversions" in roles:
        c = float(sub[roles["conversions"]].sum())
        out["baseline_conversions"] = c
        out["if_returns_scale"]["conversions"] = c * (1 + p)
    return out


def num(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"
