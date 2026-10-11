"""Root-cause investigator: an evidence-driven, bounded decomposition of why a
metric differs between two periods.

Everything here is deterministic arithmetic on the dataset through
`metric_engine`. It answers "where did the change come from?" -- which
segments, and price versus volume -- and it ranks what it finds by measured
contribution and by how much evidence stands behind each finding.

It does not answer "why" in the causal sense. A segment that accounts for 80%
of a drop is where the drop *is*; whether a price change, a stock-out or a
competitor caused it is not something a decomposition can show. Every report
says so, lists what it could not test, and names alternative explanations.

It always terminates: the number of dimensions, segments, drill-downs and
engine calls is capped, and so is wall-clock time. When evidence is thin the
report says "inconclusive" rather than promoting the largest number it found.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from app import metric_engine as me

CAVEATS = [
    "This is a descriptive decomposition: it shows where the change is located, not what caused it.",
    "Segments are compared within the dataset only; anything not recorded in it (pricing decisions, "
    "campaigns, outages, competitors) cannot appear as an explanation.",
    "Dimensions are examined one at a time. Two dimensions can describe the same rows, so their "
    "contributions must not be added together.",
]
_PRICE = re.compile(r"(^|_|\b)(unit[_ ]?price|price|rate|avg[_ ]?price)(_|\b|$)", re.IGNORECASE)
_QTY = re.compile(r"(^|_|\b)(quantity|qty|units|volume|unit[_ ]?sold|units[_ ]?sold)(_|\b|$)", re.IGNORECASE)
MATERIAL_CHANGE_PCT = 1.0
STRONG_SHARE = 0.4
MIN_ROWS_HIGH, MIN_ROWS_MEDIUM = 30, 10


@dataclass
class Budget:
    max_dimensions: int = 6
    max_segments: int = 8
    max_drilldowns: int = 2
    max_engine_calls: int = 120
    max_seconds: float = 25.0
    calls: int = 0
    started: float = field(default_factory=time.monotonic)
    exhausted: str = ""

    def spend(self) -> bool:
        """Account for one engine call. False once the budget is gone."""
        if self.exhausted:
            return False
        if self.calls >= self.max_engine_calls:
            self.exhausted = "engine_calls"
            return False
        if time.monotonic() - self.started > self.max_seconds:
            self.exhausted = "time"
            return False
        self.calls += 1
        return True

    def used(self) -> dict:
        return {"engine_calls": self.calls, "seconds": round(time.monotonic() - self.started, 3),
                "exhausted": self.exhausted,
                "limits": {"dimensions": self.max_dimensions, "segments": self.max_segments,
                           "drilldowns": self.max_drilldowns, "engine_calls": self.max_engine_calls,
                           "seconds": self.max_seconds}}


# ------------------------------------------------------------------ inputs --

def candidate_dimensions(profile: dict, exclude: set[str]) -> list[str]:
    """Categorical columns with few enough values to compare segment by segment."""
    rows = profile.get("row_count") or 0
    out = []
    for c in profile.get("columns", []):
        if c["name"] in exclude or c.get("kind") != "categorical":
            continue
        unique = c.get("unique_count") or 0
        if 2 <= unique <= 50 and (rows == 0 or unique < rows * 0.5):
            out.append((unique, c["name"]))
    return [name for _, name in sorted(out)]


def infer_spec(profile: dict, question: str, resolved_metrics: list[dict] | None = None) -> dict | None:
    """A starting spec from the question and the profile, or None when the
    dataset has no time column or no identifiable metric."""
    columns = profile.get("columns", [])
    time_cols = [c["name"] for c in columns if c.get("kind") == "datetime"]
    if not time_cols:
        return None
    formula, label, filters = None, None, []
    if resolved_metrics:
        m = resolved_metrics[0]
        formula, label, filters = m["formula"], m["name"], m.get("filters") or []
    else:
        text = re.sub(r"[_\-]+", " ", question.lower())
        numeric = [c["name"] for c in columns if c.get("kind") == "numeric"]
        hits = [n for n in numeric
                if re.search(rf"(?<![a-z0-9]){re.escape(re.sub(r'[_\-]+', ' ', n.lower()))}(?![a-z0-9])", text)]
        if len(hits) == 1:
            formula, label = f'sum("{hits[0]}")', f"Total {hits[0]}"
        elif not hits and len(numeric) == 1:
            formula, label = f'sum("{numeric[0]}")', f"Total {numeric[0]}"
    if formula is None:
        return None
    return {"formula": formula, "label": label, "filters": filters, "time_column": time_cols[0]}


def _auto_periods(df: pd.DataFrame, time_column: str) -> tuple[list[str], list[str]] | None:
    ts = pd.to_datetime(df[time_column], errors="coerce").dropna()
    if ts.empty or ts.min() == ts.max():
        return None
    lo, hi = ts.min(), ts.max()
    mid = lo + (hi - lo) / 2
    mid = mid.normalize() if (hi - lo).days >= 2 else mid
    if mid <= lo:
        mid = lo + (hi - lo) / 2
    return [lo.isoformat(), mid.isoformat()], [mid.isoformat(), hi.isoformat()]


def _period_filters(time_column: str, period: list[str], last: bool) -> list[dict]:
    """[start, end): half-open, except the final period which includes its end."""
    return [{"column": time_column, "op": ">=", "value": period[0]},
            {"column": time_column, "op": "<=" if last else "<", "value": period[1]}]


def _ratio_parts(formula: me.Formula) -> tuple[me.Formula, me.Formula] | None:
    """(numerator, denominator) when the formula is a ratio of additive parts."""
    tree = formula.tree
    if tree[0] == "agg" and tree[1] == "avg" and tree[2] is not None:
        return me.parse_formula(f'sum("{tree[2]}")'), me.parse_formula(f'count("{tree[2]}")')
    if tree[0] == "bin" and tree[1] == "/" and tree[2][0] == "agg" and tree[3][0] == "agg":
        num, den = tree[2], tree[3]
        if num[1] in ("sum", "count") and den[1] in ("sum", "count"):
            def text(t):
                return f"{t[1]}()" if t[2] is None else f'{t[1]}("{t[2]}")'
            return me.parse_formula(text(num)), me.parse_formula(text(den))
    return None


def _by_segment(result: dict, dimension: str) -> dict[str, dict]:
    return {str(g["key"].get(dimension)): g for g in result.get("groups", [])}


def _quality(rows_a: int, rows_b: int, consistent: bool | None) -> str:
    smaller = min(rows_a, rows_b)
    if smaller >= MIN_ROWS_HIGH and consistent:
        return "high"
    if smaller >= MIN_ROWS_MEDIUM and consistent is not False:
        return "medium"
    return "low"


# ------------------------------------------------------------- decomposition --

def _decompose_additive(df, formula, base_filters, fa, fb, dimension, delta, budget) -> dict | None:
    if not (budget.spend() and budget.spend()):
        return None
    a = _by_segment(me.evaluate(df, formula, base_filters + fa, [dimension]), dimension)
    b = _by_segment(me.evaluate(df, formula, base_filters + fb, [dimension]), dimension)
    segments = []
    for key in sorted(set(a) | set(b)):
        va, vb = (a.get(key) or {}).get("value") or 0.0, (b.get(key) or {}).get("value") or 0.0
        contribution = vb - va
        segments.append({
            "segment": key, "a": va, "b": vb, "contribution": contribution,
            "share_of_change": contribution / delta if delta else None,
            "rows_a": (a.get(key) or {}).get("rows", 0), "rows_b": (b.get(key) or {}).get("rows", 0),
            "presence": "new" if key not in a else "lost" if key not in b else "both",
        })
    return _finish_dimension(dimension, "additive", segments, delta)


def _decompose_ratio(df, parts, base_filters, fa, fb, dimension, total_a, delta, budget) -> dict | None:
    num, den = parts
    if not all(budget.spend() for _ in range(4)):
        return None
    na = _by_segment(me.evaluate(df, num, base_filters + fa, [dimension]), dimension)
    da = _by_segment(me.evaluate(df, den, base_filters + fa, [dimension]), dimension)
    nb = _by_segment(me.evaluate(df, num, base_filters + fb, [dimension]), dimension)
    db_ = _by_segment(me.evaluate(df, den, base_filters + fb, [dimension]), dimension)
    den_a = sum((g.get("value") or 0.0) for g in da.values())
    den_b = sum((g.get("value") or 0.0) for g in db_.values())
    if not den_a or not den_b:
        return None
    segments = []
    for key in sorted(set(da) | set(db_)):
        d_a, d_b = (da.get(key) or {}).get("value") or 0.0, (db_.get(key) or {}).get("value") or 0.0
        n_a, n_b = (na.get(key) or {}).get("value") or 0.0, (nb.get(key) or {}).get("value") or 0.0
        w_a, w_b = d_a / den_a, d_b / den_b
        r_a = n_a / d_a if d_a else total_a  # a segment new in B has no earlier rate; the identity holds for any value
        r_b = n_b / d_b if d_b else r_a
        rate_effect = w_b * (r_b - r_a)
        mix_effect = (w_b - w_a) * (r_a - total_a)
        contribution = rate_effect + mix_effect
        segments.append({
            "segment": key, "a": n_a / d_a if d_a else None, "b": n_b / d_b if d_b else None,
            "contribution": contribution, "rate_effect": rate_effect, "mix_effect": mix_effect,
            "share_of_change": contribution / delta if delta else None,
            "rows_a": (da.get(key) or {}).get("rows", 0), "rows_b": (db_.get(key) or {}).get("rows", 0),
            "presence": "new" if not d_a else "lost" if not d_b else "both",
        })
    out = _finish_dimension(dimension, "rate_and_mix", segments, delta)
    out["rate_effect_total"] = sum(s["rate_effect"] for s in segments)
    out["mix_effect_total"] = sum(s["mix_effect"] for s in segments)
    return out


def _finish_dimension(dimension: str, method: str, segments: list[dict], delta: float) -> dict:
    explained = math.fsum(s["contribution"] for s in segments)
    magnitude = math.fsum(abs(s["contribution"]) for s in segments)
    ranked = sorted(segments, key=lambda s: -abs(s["contribution"]))
    top2 = math.fsum(abs(s["contribution"]) for s in ranked[:2])
    return {
        "dimension": dimension, "method": method, "segments": ranked,
        "explained": explained,
        "reconciles": math.isclose(explained, delta, rel_tol=1e-6, abs_tol=1e-6 * max(1.0, abs(delta))),
        # How much of the total movement sits in the two largest segments. Near
        # 1.0: the change is localised along this dimension. Low: it is spread out.
        "concentration": top2 / magnitude if magnitude else 0.0,
    }


def _consistency(df, formula, base_filters, time_column, period_a, period_b, dimension, segment,
                 direction: float, budget) -> bool | None:
    """Does the segment move the same way in both halves of the comparison?
    A change that only appears in one half is weaker evidence."""
    try:
        a0, a1 = pd.Timestamp(period_a[0]), pd.Timestamp(period_a[1])
        b0, b1 = pd.Timestamp(period_b[0]), pd.Timestamp(period_b[1])
    except (ValueError, TypeError):
        return None
    am, bm = a0 + (a1 - a0) / 2, b0 + (b1 - b0) / 2
    seg = [{"column": dimension, "op": "==", "value": segment}]
    halves = [((a0, am), (b0, bm), False), ((am, a1), (bm, b1), True)]
    signs = []
    for (sa, ea), (sb, eb), last in halves:
        if not (budget.spend() and budget.spend()):
            return None
        try:
            va = me.evaluate(df, formula, base_filters + seg + _period_filters(time_column, [sa.isoformat(), ea.isoformat()], False))["value"]
            vb = me.evaluate(df, formula, base_filters + seg + _period_filters(time_column, [sb.isoformat(), eb.isoformat()], last))["value"]
        except me.FormulaError:
            return None
        if va is None or vb is None:
            return None
        signs.append(vb - va)
    return all(s * direction > 0 for s in signs)


def _price_volume(df: pd.DataFrame, formula: me.Formula, profile: dict, base_filters, fa, fb,
                  dimension: str | None, delta: float, budget: Budget) -> dict | None:
    """Price-volume split, when the metric is sum(amount) and the dataset has a
    quantity column such that amount is (close to) unit price x quantity."""
    if not (formula.tree[0] == "agg" and formula.tree[1] == "sum"):
        return None
    amount = formula.tree[2]
    names = [c["name"] for c in profile.get("columns", []) if c.get("kind") == "numeric" and c["name"] != amount]
    qty_cols = [n for n in names if _QTY.search(n)]
    if not qty_cols:
        return None
    qty = qty_cols[0]
    price_cols = [n for n in names if _PRICE.search(n) and n != qty]
    basis = "average price = amount / quantity"
    if price_cols:
        sample = df[[amount, qty, price_cols[0]]].apply(pd.to_numeric, errors="coerce").dropna().head(5000)
        if len(sample) and (sample[amount].abs() > 0).any():
            err = ((sample[price_cols[0]] * sample[qty] - sample[amount]).abs() / sample[amount].abs().clip(lower=1e-9)).median()
            if err < 0.01:
                basis = f"{amount} = {price_cols[0]} x {qty} (confirmed on a sample of {len(sample)} rows)"
    group = [dimension] if dimension else None
    q_formula = me.parse_formula(f'sum("{qty}")')
    if not all(budget.spend() for _ in range(4)):
        return None
    ra, rb = me.evaluate(df, formula, base_filters + fa, group), me.evaluate(df, formula, base_filters + fb, group)
    qa, qb = me.evaluate(df, q_formula, base_filters + fa, group), me.evaluate(df, q_formula, base_filters + fb, group)

    def table(result):
        if group:
            return {str(g["key"].get(dimension)): g.get("value") or 0.0 for g in result.get("groups", [])}
        return {"all": result.get("value") or 0.0}

    ra, rb, qa, qb = table(ra), table(rb), table(qa), table(qb)
    price_effect = volume_effect = entry_exit = 0.0
    rows = []
    for key in sorted(set(ra) | set(rb)):
        r0, r1, q0, q1 = ra.get(key, 0.0), rb.get(key, 0.0), qa.get(key, 0.0), qb.get(key, 0.0)
        if q0 > 0 and q1 > 0:
            p0, p1 = r0 / q0, r1 / q1
            vol, price = (q1 - q0) * p0, (p1 - p0) * q1
            price_effect += price
            volume_effect += vol
            rows.append({"segment": key, "price_effect": price, "volume_effect": vol,
                         "avg_price_a": p0, "avg_price_b": p1, "quantity_a": q0, "quantity_b": q1})
        else:
            entry_exit += r1 - r0
            rows.append({"segment": key, "entry_or_exit": r1 - r0, "quantity_a": q0, "quantity_b": q1})
    total = price_effect + volume_effect + entry_exit
    return {
        "amount_column": amount, "quantity_column": qty, "basis": basis, "by": dimension,
        "price_effect": price_effect, "volume_effect": volume_effect, "entry_exit_effect": entry_exit,
        "reconciles": math.isclose(total, delta, rel_tol=1e-6, abs_tol=1e-6 * max(1.0, abs(delta))),
        "segments": sorted(rows, key=lambda r: -abs(r.get("price_effect", 0) + r.get("volume_effect", 0) + r.get("entry_or_exit", 0)))[:20],
        "note": "Volume effect = change in quantity at the earlier average price. Price effect = change in "
                "average price at the later quantity. 'Average price' moves when the product mix moves, so a "
                "price effect is not necessarily a list-price change.",
    }


# ------------------------------------------------------------------ report --

def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _contribution_chart(title: str, ranked: list[dict]) -> dict:
    top = ranked[:10]
    return {
        "data": [{"type": "bar", "orientation": "h",
                  "y": [f"{r['dimension']} = {r['segment']}" for r in top][::-1],
                  "x": [r["contribution"] for r in top][::-1]}],
        "layout": {"title": {"text": title}, "xaxis": {"title": {"text": "Contribution to the change"}},
                   "margin": {"l": 160}},
    }


def investigate(df: pd.DataFrame, profile: dict, spec: dict, budget: Budget | None = None) -> dict:
    """Run one investigation. Never raises for inconclusive evidence; raises
    FormulaError only for an invalid spec (bad formula, unknown column)."""
    budget = budget or Budget()
    formula = me.parse_formula(spec["formula"])
    problems = me.validate_against_profile(formula, profile)
    if problems:
        raise me.FormulaError("; ".join(problems))
    time_column = spec.get("time_column")
    if not time_column or time_column not in df.columns:
        raise me.FormulaError("A time column that exists in the dataset is required")
    base_filters = me.normalize_filters(spec.get("filters"), list(df.columns))
    label = spec.get("label") or formula.text

    if spec.get("period_a") and spec.get("period_b"):
        period_a, period_b = list(spec["period_a"]), list(spec["period_b"])
        for p in (period_a, period_b):
            if len(p) != 2 or pd.isna(pd.to_datetime(p[0], errors="coerce")) or pd.isna(pd.to_datetime(p[1], errors="coerce")):
                raise me.FormulaError("Each period must be [start, end] dates")
        periods_source = "specified"
    else:
        scoped = me.apply_filters(df, base_filters)
        auto = _auto_periods(scoped, time_column)
        if auto is None:
            return _inconclusive(label, formula, spec, budget, "The time column has fewer than two distinct moments, "
                                 "so there are no two periods to compare.")
        period_a, period_b = auto
        periods_source = "automatic: the time range split at its midpoint"
    fa = _period_filters(time_column, period_a, last=False)
    fb = _period_filters(time_column, period_b, last=True)

    budget.spend()
    budget.spend()
    res_a = me.evaluate(df, formula, base_filters + fa)
    res_b = me.evaluate(df, formula, base_filters + fb)
    total_a, total_b = res_a["value"], res_b["value"]
    base = {"metric": {"label": label, "formula": formula.text, "filters": base_filters},
            "periods": {"a": period_a, "b": period_b, "time_column": time_column, "source": periods_source},
            "totals": {"a": total_a, "b": total_b, "rows_a": res_a["rows"], "rows_b": res_b["rows"]}}
    if total_a is None or total_b is None:
        return {**_inconclusive(label, formula, spec, budget, "The metric has no value in one of the two periods."), **base}
    delta = total_b - total_a
    pct = delta / abs(total_a) * 100 if total_a else None
    base["totals"].update({"change": delta, "change_pct": pct})
    if delta == 0 or (pct is not None and abs(pct) < MATERIAL_CHANGE_PCT):
        return {**_inconclusive(label, formula, spec, budget,
                                f"The metric moved by less than {MATERIAL_CHANGE_PCT:g}% between the periods; "
                                "there is no material change to explain."), **base}

    requested = spec.get("dimensions")
    exclude = {time_column, *formula.columns}
    dims = [d for d in requested if d in df.columns and d not in exclude] if requested \
        else candidate_dimensions(profile, exclude)
    skipped = dims[budget.max_dimensions:]
    dims = dims[:budget.max_dimensions]
    parts = None if formula.is_additive else _ratio_parts(formula)
    unresolved: list[str] = []
    if not formula.is_additive and parts is None:
        unresolved.append("This metric is neither a sum nor a ratio of sums, so segment contributions cannot be "
                          "computed exactly; only segment-level values are compared.")

    dimensions = []
    for d in dims:
        try:
            if formula.is_additive:
                out = _decompose_additive(df, formula, base_filters, fa, fb, d, delta, budget)
            elif parts is not None:
                out = _decompose_ratio(df, parts, base_filters, fa, fb, d, total_a, delta, budget)
            else:
                out = None
        except me.FormulaError:
            out = None
        if out is None:
            if budget.exhausted:
                skipped = [d, *[x for x in dims[dims.index(d) + 1:]], *skipped]
                break
            continue
        out["segments"] = out["segments"][:budget.max_segments]
        dimensions.append(out)

    ranked = []
    for dim in dimensions:
        for s in dim["segments"]:
            if s["share_of_change"] is None:
                continue
            ranked.append({"dimension": dim["dimension"], "segment": s["segment"], "contribution": s["contribution"],
                           "share_of_change": s["share_of_change"], "rows_a": s["rows_a"], "rows_b": s["rows_b"],
                           "presence": s["presence"], "method": dim["method"]})
    ranked.sort(key=lambda r: -abs(r["contribution"]))
    for r in ranked[:5]:
        consistent = None
        if r["presence"] == "both" and formula.is_additive:
            consistent = _consistency(df, formula, base_filters, time_column, period_a, period_b,
                                      r["dimension"], r["segment"], r["contribution"], budget)
        r["consistent_across_halves"] = consistent
        r["evidence_quality"] = _quality(r["rows_a"], r["rows_b"], consistent)
    for r in ranked[5:]:
        r["consistent_across_halves"] = None
        r["evidence_quality"] = _quality(r["rows_a"], r["rows_b"], None)
    weight = {"high": 1.0, "medium": 0.7, "low": 0.35}
    ranked.sort(key=lambda r: -abs(r["contribution"]) * weight[r["evidence_quality"]])

    best_dim = max(dimensions, key=lambda d: d["concentration"], default=None)
    pvm = None
    try:
        pvm = _price_volume(df, formula, profile, base_filters, fa, fb,
                            best_dim["dimension"] if best_dim else None, delta, budget)
    except me.FormulaError:
        pvm = None

    drilldowns = []
    for r in [x for x in ranked if x["share_of_change"] and x["share_of_change"] >= STRONG_SHARE][:budget.max_drilldowns]:
        others = [d for d in dims if d != r["dimension"]]
        if not others or not formula.is_additive:
            continue
        seg_filter = [{"column": r["dimension"], "op": "==", "value": r["segment"]}]
        sub = _decompose_additive(df, formula, base_filters + seg_filter, fa, fb, others[0], r["contribution"], budget)
        if sub is not None:
            sub["segments"] = sub["segments"][:5]
            drilldowns.append({"within": {"dimension": r["dimension"], "segment": r["segment"]}, **sub})

    hypotheses = _hypotheses(ranked, dimensions, pvm, delta)
    alternatives = []
    strong = [r for r in ranked if r["share_of_change"] and r["share_of_change"] >= 0.5]
    if len({r["dimension"] for r in strong}) > 1:
        alternatives.append("More than one dimension has a segment accounting for over half of the change ("
                            + ", ".join(f"{r['dimension']} = {r['segment']}" for r in strong[:3])
                            + "). These may be the same rows seen from different angles; see the drill-downs.")
    alternatives.append("A change in what was recorded (late data, a changed definition, duplicated or missing "
                        "rows) would produce the same pattern. Check the data-quality page for this dataset version.")
    if skipped:
        unresolved.append("Not examined within the budget: " + ", ".join(skipped[:10]))
    if budget.exhausted:
        unresolved.append(f"The investigation stopped at its {budget.exhausted.replace('_', ' ')} limit.")
    if not dims:
        unresolved.append("The dataset has no categorical column with between 2 and 50 values to compare by.")
    low_only = bool(ranked) and all(r["evidence_quality"] == "low" for r in ranked[:3])
    if low_only:
        unresolved.append("The leading contributors rest on few rows or do not hold in both halves of the periods.")

    conclusive = bool(ranked) and not low_only
    direction = "rose" if delta > 0 else "fell"
    summary = (f"{label} {direction} from {_fmt(total_a)} to {_fmt(total_b)} "
               f"({_fmt(delta)}{f', {pct:+.1f}%' if pct is not None else ''}).")
    if ranked:
        lead = ranked[0]
        summary += (f" The largest measured contributor is {lead['dimension']} = {lead['segment']}, accounting for "
                    f"{lead['share_of_change'] * 100:.0f}% of the change ({lead['evidence_quality']} evidence quality).")
    if pvm and pvm["reconciles"]:
        summary += f" Price effect {_fmt(pvm['price_effect'])}, volume effect {_fmt(pvm['volume_effect'])}."
    if not conclusive:
        summary += " The evidence is not strong enough to single out a contributor."

    charts = []
    if ranked:
        charts.append({"title": f"Largest contributors to the change in {label}",
                       "plotly_json": _contribution_chart(f"Contributors to the change in {label}", ranked)})
    if pvm and pvm["reconciles"]:
        charts.append({"title": "Price, volume and entry/exit effects", "plotly_json": {
            "data": [{"type": "waterfall", "measure": ["relative", "relative", "relative", "total"],
                      "x": ["Price", "Volume", "Entry / exit", "Total change"],
                      "y": [pvm["price_effect"], pvm["volume_effect"], pvm["entry_exit_effect"], delta]}],
            "layout": {"title": {"text": "Price, volume and entry/exit effects"}}}})

    return {
        **base, "status": "complete" if conclusive else "inconclusive", "summary": summary,
        "decomposition_method": "additive" if formula.is_additive else "rate_and_mix" if parts else "not_decomposable",
        "dimensions": dimensions, "ranked_contributors": ranked[:15], "price_volume": pvm,
        "drilldowns": drilldowns, "hypotheses": hypotheses, "unresolved": unresolved,
        "alternative_explanations": alternatives, "caveats": CAVEATS, "charts": charts,
        "budget": budget.used(),
    }


def _hypotheses(ranked: list[dict], dimensions: list[dict], pvm: dict | None, delta: float) -> list[dict]:
    """Candidate explanations, each with the test that was applied and what it showed."""
    out = []
    for r in ranked[:4]:
        share = r["share_of_change"] or 0.0
        if share >= STRONG_SHARE and r["evidence_quality"] != "low":
            status = "supported"
        elif share >= 0.2:
            status = "weak"
        else:
            status = "not_supported"
        out.append({
            "hypothesis": f"The change is concentrated in {r['dimension']} = {r['segment']}.",
            "test": "Share of the total change located in this segment, with row counts in both periods and a "
                    "same-direction check across the two halves of each period.",
            "result": f"{share * 100:.0f}% of the change; rows {r['rows_a']} -> {r['rows_b']}; "
                      f"consistent across halves: {r['consistent_across_halves']}.",
            "status": status, "evidence_quality": r["evidence_quality"],
        })
    if dimensions and max(d["concentration"] for d in dimensions) < 0.5:
        out.append({"hypothesis": "The change is broad-based rather than located in particular segments.",
                    "test": "Concentration of the change in the two largest segments of every dimension examined.",
                    "result": "No dimension has more than half of the movement in its two largest segments.",
                    "status": "supported", "evidence_quality": "medium"})
    if pvm is not None and pvm["reconciles"]:
        price, volume = pvm["price_effect"], pvm["volume_effect"]
        dominant = "price" if abs(price) > abs(volume) else "volume"
        out.append({"hypothesis": f"The change is mainly a {dominant} effect.",
                    "test": "Price-volume decomposition on " + pvm["basis"] + ".",
                    "result": f"Price effect {_fmt(price)}, volume effect {_fmt(volume)}, entry/exit "
                              f"{_fmt(pvm['entry_exit_effect'])} of a total change of {_fmt(delta)}.",
                    "status": "supported" if abs(price - volume) > 0.2 * abs(delta) else "weak",
                    "evidence_quality": "medium"})
    return out


def _inconclusive(label: str, formula: me.Formula, spec: dict, budget: Budget, reason: str) -> dict:
    return {
        "status": "inconclusive", "summary": reason,
        "metric": {"label": label, "formula": formula.text, "filters": spec.get("filters") or []},
        "periods": {}, "totals": {}, "decomposition_method": "none", "dimensions": [], "ranked_contributors": [],
        "price_volume": None, "drilldowns": [], "hypotheses": [], "unresolved": [reason],
        "alternative_explanations": [], "caveats": CAVEATS, "charts": [], "budget": budget.used(),
    }
