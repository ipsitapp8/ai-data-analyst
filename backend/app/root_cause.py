"""Root-cause diagnosis: the evidence layer on top of the investigator's decomposition.

`investigator.investigate` says *where* a metric changed (which segment, how much).
This module asks whether that location is a believable *contributing factor*:

* numeric drivers are scanned too (their level shift, and the change by value band);
* every leading candidate is tested against every other leading candidate by
  stratification -- does the segment's decline survive when the other factor is held
  fixed? -- so a segment that is only a proxy for another is demoted, not promoted;
* the change is re-measured with extreme rows pulled in, and the metric's inputs and
  the row volume are checked for recording artefacts;
* each candidate is placed on an evidence ladder set by code, never by a model, and
  never above "plausible contributor": an observational table cannot show causation.

Everything is deterministic arithmetic through `metric_engine` (and scipy for two
descriptive tests). Nothing here carries a confidence percentage: the strength of
evidence is the list of checks passed and failed, written out.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats as sps

from app import investigator as inv
from app import metric_engine as me

VERSION = "rc-1"
MIN_CELL_ROWS = 8            # rows needed in each period of a stratum before it is compared
MIN_NUMERIC_VALUES = 20
MAX_NUMERIC_FACTORS = 6
MAX_CANDIDATES = 6
MIN_MATERIAL_SHARE = 0.2     # a candidate must account for this share of the change to be called plausible
MARGINAL_FLOOR = 0.02        # a gap under 2 points between a segment and the rest is not an effect to adjust
MECHANICAL_RHO = 0.9          # a column this tightly tied to the outcome row by row is part of the metric, not a driver
HOLDS, VANISHES = 0.5, 0.25  # share of the marginal gap that must survive holding another factor fixed

TIER_LABEL = {
    "plausible_contributor": "Plausible contributor",
    "association": "Association only",
    "overlapping": "Overlaps another factor",
}
_DOWN = re.compile(r"\b(drop|dropp\w*|declin\w*|fell|fall\w*|decreas\w*|down|lower|lost|loss|slump\w*|shrink\w*|worse\w*|reduc\w*|underperform\w*)\b", re.I)
_UP = re.compile(r"\b(rose|rise|rising|increas\w*|grew|grow\w*|up|higher|spike\w*|jump\w*|surge\w*|climb\w*)\b", re.I)
_DESIGN = re.compile(r"(variant|treatment|experiment|test[_ ]?group|\barm\b|control|ab[_ ]?test|cohort)", re.I)


# ------------------------------------------------------------------ helpers --

def _fmt(v) -> str:
    return inv._fmt(None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))


def _pct(v: float | None, signed: bool = True) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running, m = [1.0] * len(pvalues), 0.0, len(pvalues)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _member(dfx: pd.DataFrame, dimension: str, segment: str) -> pd.Series:
    col = dfx[dimension]
    if segment in ("None", "nan", "<NA>"):
        return col.isna()
    return col.astype(str) == segment


def _row_outcome_column(formula: me.Formula) -> str | None:
    """The column that is each row's own outcome when the metric is a sum or mean of it,
    or a sum of it over a row count. None for anything where a row has no outcome of its own."""
    t = formula.tree
    if t[0] == "agg" and t[1] in ("sum", "avg") and t[2]:
        return t[2]
    if t[0] == "bin" and t[1] == "/" and t[2][0] == "agg" and t[3][0] == "agg" \
            and t[2][1] == "sum" and t[2][2] and t[3][1] == "count":
        return t[2][2]
    return None


def _change_pct(va, vb, scale: float, additive: bool) -> float | None:
    base = None if va is None else va * (scale if additive else 1.0)
    if vb is None or not base:
        return None
    return vb / base - 1.0


def _split(dfx, formula, base, fa, fb, column: str, budget) -> tuple[dict, dict] | None:
    if not (budget.spend() and budget.spend()):
        return None
    ga = inv._by_segment(me.evaluate(dfx, formula, base + fa, [column]), column)
    gb = inv._by_segment(me.evaluate(dfx, formula, base + fb, [column]), column)
    return ga, gb


# ------------------------------------------------------- problem definition --

def _problem(question: str, label: str, formula, base_filters, periods, totals, scoped_rows, spec_notes) -> dict:
    delta = totals.get("change")
    direction = "increase" if (delta or 0) > 0 else "decrease"
    said = "decrease" if _DOWN.search(question or "") and not _UP.search(question or "") else \
        "increase" if _UP.search(question or "") and not _DOWN.search(question or "") else None
    ambiguities = list(spec_notes)
    src = str(periods.get("source") or "")
    if src.startswith("automatic"):
        ambiguities.append("The comparison periods were not specified, so the time range was split at its midpoint. "
                           "A different baseline could give a different answer.")
    if said and said != direction:
        ambiguities.append(f"The question suggests a {said}, but {label} actually {'rose' if direction == 'increase' else 'fell'} "
                           "between the periods; the investigation explains the measured change.")
    if not base_filters:
        scope = "all rows"
    else:
        scope = " and ".join(f"{f['column']} {f['op']} {f['value']}" for f in base_filters)
    return {
        "outcome": label, "formula": formula.text, "direction": direction, "question_direction": said,
        "baseline": periods.get("a"), "comparison": periods.get("b"), "periods_source": src,
        "scope": scope, "scope_filters": base_filters, "rows_baseline": totals.get("rows_a"), "rows_comparison": totals.get("rows_b"),
        "value_baseline": totals.get("a"), "value_comparison": totals.get("b"),
        "change": delta, "change_pct": totals.get("change_pct"),
        "like_for_like_change": totals.get("like_for_like_change"),
        "ambiguities": ambiguities,
    }


# --------------------------------------------------------- numeric factors --

def _bands(values: pd.Series, pooled: np.ndarray) -> tuple[pd.Series, list[str]] | None:
    edges = sorted(set(np.unique(np.quantile(pooled, [0.25, 0.5, 0.75]))))
    lo, hi = float(np.min(pooled)), float(np.max(pooled))
    edges = [e for e in edges if lo < e < hi]
    if not edges:
        return None
    bounds = [-np.inf, *edges, np.inf]
    labels = []
    for i in range(len(bounds) - 1):
        if i == 0:
            labels.append(f"≤ {_fmt(edges[0])}")
        elif i == len(bounds) - 2:
            labels.append(f"> {_fmt(edges[-1])}")
        else:
            labels.append(f"{_fmt(edges[i - 1])} – {_fmt(edges[i])}")
    cut = pd.cut(values, bins=bounds, labels=labels, include_lowest=True)
    return cut.astype(object).where(values.notna(), "(missing)"), labels


def _numeric_scan(df, dfx, profile, formula, base, fa, fb, exclude: set[str], budget, scale, delta, notes) -> tuple[list[dict], list[dict]]:
    """Level shift of each numeric column between the periods, and the change by value band.
    Bands are added to `dfx` as columns named "<column> (band)". Returns (factors, band decompositions)."""
    rows = profile.get("row_count") or len(df)
    cols = []
    for c in profile.get("columns", []):
        if c.get("kind") != "numeric" or c["name"] in exclude or c["name"] not in df.columns:
            continue
        unique = c.get("unique_count") or 0
        if unique < 6:
            continue
        if unique > 0.9 * max(rows, 1):
            notes.append(f"'{c['name']}' looks like an identifier (almost every value is distinct), so it was not scanned as a factor.")
            continue
        cols.append(c["name"])
    cols = cols[:MAX_NUMERIC_FACTORS]
    scoped_a, scoped_b = me.apply_filters(df, base + fa), me.apply_filters(df, base + fb)
    out, decomps, pvals = [], [], []
    for name in cols:
        xa, xb = pd.to_numeric(scoped_a[name], errors="coerce"), pd.to_numeric(scoped_b[name], errors="coerce")
        va, vb = xa.dropna().to_numpy(float), xb.dropna().to_numpy(float)
        if len(va) < MIN_NUMERIC_VALUES or len(vb) < MIN_NUMERIC_VALUES:
            notes.append(f"'{name}' has fewer than {MIN_NUMERIC_VALUES} usable values in one period, so it was not analysed.")
            continue
        sd = math.sqrt(((len(va) - 1) * va.var(ddof=1) + (len(vb) - 1) * vb.var(ddof=1)) / (len(va) + len(vb) - 2))
        d = float((vb.mean() - va.mean()) / sd) if sd else 0.0
        try:
            p = float(sps.mannwhitneyu(va, vb, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        item = {"column": name, "n_a": len(va), "n_b": len(vb),
                "missing_a": float(xa.isna().mean()), "missing_b": float(xb.isna().mean()),
                "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                "median_a": float(np.median(va)), "median_b": float(np.median(vb)),
                "quartiles_a": [float(q) for q in np.quantile(va, [0.25, 0.75])],
                "quartiles_b": [float(q) for q in np.quantile(vb, [0.25, 0.75])],
                "range_a": [float(va.min()), float(va.max())], "range_b": [float(vb.min()), float(vb.max())],
                "mean_change_pct": float(vb.mean() / va.mean() - 1) if va.mean() else None,
                "std_diff": d, "p_value": p}
        # Relationship with the row-level outcome, when each row has one: association only.
        outcome = _row_outcome_column(formula)
        if outcome and outcome in df.columns and outcome != name:
            rel = {}
            for tag, frame in (("a", scoped_a), ("b", scoped_b)):
                pair = pd.DataFrame({"x": pd.to_numeric(frame[name], errors="coerce"),
                                     "y": pd.to_numeric(frame[outcome], errors="coerce")}).dropna()
                if len(pair) >= 30 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
                    rho = sps.spearmanr(pair["x"], pair["y"])
                    rel[tag] = {"rho": float(rho.statistic), "n": len(pair)}
            item["relationship"] = rel or None
        rel = item.get("relationship") or {}
        if rel and min(abs(v["rho"]) for v in rel.values()) >= MECHANICAL_RHO:
            rho = max(abs(v["rho"]) for v in rel.values())
            notes.append(f"'{name}' moves with the metric almost one-for-one (rank correlation {rho:.2f} with {_row_outcome_column(formula)}), so it is "
                         "treated as a component of the metric, not as a factor behind it.")
            continue
        out.append(item)
        pvals.append(p)
        pooled = np.concatenate([va, vb])
        built = _bands(pd.to_numeric(df[name], errors="coerce"), pooled)
        if built is None:
            continue
        banded, _labels = built
        dim = f"{name} (band)"
        dfx[dim] = banded
        try:
            if formula.is_additive:
                dec = inv._decompose_additive(dfx, formula, base, fa, fb, dim, delta, budget, scale)
            else:
                parts = inv._ratio_parts(formula)
                total_a = me.evaluate(dfx, formula, base + fa)["value"]
                dec = inv._decompose_ratio(dfx, parts, base, fa, fb, dim, total_a, delta, budget) if parts else None
        except me.FormulaError:
            dec = None
        if dec:
            dec["column"] = name
            decomps.append(dec)
    for item, adj in zip(out, _holm(pvals)):
        item["p_adjusted"] = adj
        item["moved"] = adj < 0.05 and abs(item["std_diff"]) >= 0.2
    return out, decomps



# ---------------------------------------------------------------- uncertainty --

BOOT_REPLICATES = 300
BOOT_MAX_ROWS = 250_000
BOOT_SEED = 20250101


def _row_parts(formula: me.Formula):
    """(numerator, denominator) per-row vectors builders for a metric that is a sum, a count, a mean or a ratio of
    sums/counts; None for anything else (the intervals are then not computed, and the report says so)."""
    t = formula.tree

    def vec(agg):
        fn, col = agg[1], agg[2]
        if fn == "sum" and col:
            return lambda f: pd.to_numeric(f[col], errors="coerce").fillna(0.0).to_numpy(float)
        if fn == "count":
            return (lambda f: f[col].notna().to_numpy(float)) if col else (lambda f: np.ones(len(f)))
        return None
    if t[0] == "agg":
        if t[1] in ("sum", "count"):
            return vec(t), None
        if t[1] == "avg" and t[2]:
            return vec(("agg", "sum", t[2])), vec(("agg", "count", t[2]))
        return None
    if t[0] == "bin" and t[1] == "/" and t[2][0] == "agg" and t[3][0] == "agg":
        n, d = vec(t[2]), vec(t[3])
        return (n, d) if n and d else None
    return None


class _Boot:
    """Poisson bootstrap of a metric's change: every row gets a random weight with mean 1, so the number of rows
    varies between replicates as well as their values. Deterministic (fixed seed)."""

    def __init__(self, dfx, formula, base, fa, fb, scale, additive):
        self.ok = False
        parts = _row_parts(formula)
        if parts is None:
            return
        sa, sb = me.apply_filters(dfx, base + fa), me.apply_filters(dfx, base + fb)
        if len(sa) > BOOT_MAX_ROWS or len(sb) > BOOT_MAX_ROWS or not len(sa) or not len(sb):
            return
        self.idx = {"a": sa.index, "b": sb.index}
        self.num = {"a": parts[0](sa), "b": parts[0](sb)}
        self.den = {"a": parts[1](sa), "b": parts[1](sb)} if parts[1] else None
        self.scale, self.additive, self.ok = scale, additive, True

    def pct(self, codes: dict[str, np.ndarray], groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Replicates of the percentage change of each group (a set of cell codes). `codes` gives each row's cell per period."""
        rng = np.random.default_rng(BOOT_SEED)
        sums = {}
        for period in ("a", "b"):
            for cell in set(c for g in groups.values() for c in g):
                rows = np.flatnonzero(codes[period] == cell)
                w = rng.poisson(1.0, size=(BOOT_REPLICATES, len(rows))).astype(float)
                sums[(period, cell)] = (w @ self.num[period][rows], None if self.den is None else w @ self.den[period][rows])
        out = {}
        for name, cells in groups.items():
            val = {}
            for period in ("a", "b"):
                n = sum(sums[(period, c)][0] for c in cells)
                if self.den is not None:
                    d = sum(sums[(period, c)][1] for c in cells)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        n = np.where(d != 0, n / d, np.nan)
                val[period] = n
            base = val["a"] * (self.scale if self.additive else 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = np.where(base != 0, val["b"] / base - 1.0, np.nan)
        return out


def _interval(x: np.ndarray, family: int = 1) -> list[float] | None:
    """Interval for a bootstrapped quantity: median +/- z standard deviations, with z set so that the whole family of
    comparisons made together has a 5% chance of a false flag (Bonferroni). `family` is how many were made."""
    x = x[~np.isnan(x)]
    if len(x) < BOOT_REPLICATES // 2:
        return None
    z = float(sps.norm.ppf(1 - 0.025 / max(1, family)))
    centre, sd = float(np.median(x)), float(np.std(x, ddof=1))
    return [centre - z * sd, centre + z * sd]


# ------------------------------------------------------------ stratification --

def _stratify(dfx, formula, base, fa, fb, c1: dict, c2: dict, scale: float, additive: bool, budget, boot=None, family: int = 1) -> dict | None:
    """Is c1's gap to the rest of the data still there once c2 is held fixed, and the reverse?"""
    m1, m2 = _member(dfx, c1["dimension"], c1["segment"]), _member(dfx, c2["dimension"], c2["segment"])
    dfx["__cell"] = (m1.astype(int) * 2 + m2.astype(int)).astype(str)
    split = _split(dfx, formula, base, fa, fb, "__cell", budget)
    if split is None:
        return None
    ga, gb = split

    def cell(k: str):
        a, b = ga.get(k) or {}, gb.get(k) or {}
        pct = _change_pct(a.get("value"), b.get("value"), scale, additive)
        usable = a.get("rows", 0) >= MIN_CELL_ROWS and b.get("rows", 0) >= MIN_CELL_ROWS and pct is not None
        return {"pct": pct if usable else None, "weight": a.get("rows", 0) + b.get("rows", 0),
                "rows_a": a.get("rows", 0), "rows_b": b.get("rows", 0), "a": a.get("value"), "b": b.get("value")}
    cells = {k: cell(k) for k in "0123"}

    def effect(hi: str, lo: str) -> list[tuple[float, int]]:
        """Gap in change between 'has the factor' (cell hi) and 'does not' (cell lo), inside one stratum of the other factor."""
        if cells[hi]["pct"] is None or cells[lo]["pct"] is None:
            return []
        return [(cells[hi]["pct"] - cells[lo]["pct"], cells[hi]["weight"])]

    # c1 inside c2 (3 vs 1) and outside c2 (2 vs 0);  c2 inside c1 (3 vs 2) and outside c1 (1 vs 0)
    one = {"within": effect("3", "1"), "outside": effect("2", "0")}
    two = {"within": effect("3", "2"), "outside": effect("1", "0")}

    reps = None
    if boot is not None and boot.ok:
        code = dfx["__cell"]
        codes = {pd_: code.loc[boot.idx[pd_]].to_numpy() for pd_ in ("a", "b")}
        reps = boot.pct(codes, {k: [k] for k in "0123"})

    def summarise(parts: dict, gap: float | None, strata: dict[str, tuple[str, str]]) -> dict:
        within = parts["within"][0][0] if parts["within"] else None
        outside = parts["outside"][0][0] if parts["outside"] else None
        pairs = parts["within"] + parts["outside"]
        weight = sum(w for _, w in pairs)
        adjusted = sum(g * w for g, w in pairs) / weight if weight else None
        interval = None
        if reps is not None and adjusted is not None:
            used = [strata[n] for n, lst in (("within", parts["within"]), ("outside", parts["outside"])) if lst]
            ws = [cells[hi]["weight"] for hi, _ in used]
            interval = _interval(sum(w * (reps[hi] - reps[lo]) for w, (hi, lo) in zip(ws, used)) / sum(ws), family)
        if gap is None or abs(gap) < MARGINAL_FLOOR:
            result = "no_gap"
        elif adjusted is None:
            result = "not_separable"
        else:
            kept = adjusted / gap
            if interval is None:
                result = "holds" if kept >= HOLDS else "vanishes" if kept < VANISHES else "attenuated"
            elif interval[0] > 0 or interval[1] < 0:        # distinguishable from no effect
                result = "holds" if kept >= HOLDS and adjusted * gap > 0 else "attenuated"
            elif not (interval[0] <= gap <= interval[1]):   # indistinguishable from zero, and the full gap is ruled out
                result = "vanishes"
            else:
                result = "unclear"                          # too noisy to say either way
        unstable = (adjusted is not None and gap is not None and abs(gap) >= MARGINAL_FLOOR
                    and (adjusted * gap < 0 or abs(adjusted) > max(1.0, 3 * abs(gap))))
        if unstable:                       # strata this thin cannot be compared: a flipped or exploding gap is an artefact, not a finding
            result, interval = "unclear", None
        interaction = (within is not None and outside is not None and abs(within - outside) >= max(0.10, 0.5 * abs(gap or 0)))
        return {"result": result, "marginal_gap": gap, "adjusted_gap": adjusted, "within": within, "outside": outside,
                "interaction": bool(interaction), "interval": interval}
    return {"first": summarise(one, c1["_gap"], {"within": ("3", "1"), "outside": ("2", "0")}),
            "second": summarise(two, c2["_gap"], {"within": ("3", "2"), "outside": ("1", "0")}), "cells": cells}


# ------------------------------------------------------------ evidence ladder --

def _steady_presence(dfx, time_column: str, dimension: str, segment, period) -> bool | None:
    """A segment that exists in one period only: is it present in both halves of that period (not a one-day blip)?"""
    try:
        t0, t1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
        ts = pd.to_datetime(dfx[time_column], errors="coerce")
        mine = dfx[dimension].astype(str) == str(segment)
        mid = t0 + (t1 - t0) / 2
        return bool(((ts >= t0) & (ts < mid) & mine).sum() >= inv.MIN_ROWS_MEDIUM and ((ts >= mid) & (ts <= t1) & mine).sum() >= inv.MIN_ROWS_MEDIUM)
    except (ValueError, TypeError, KeyError):
        return None


def _tier(c: dict, ctx: dict) -> tuple[str, list[str]]:
    """Place a candidate on the ladder. Returns (tier, the checks it failed)."""
    failed = []
    if c["evidence_quality"] == "low":
        failed.append("the finding rests on few rows or does not hold in both halves of the periods")
    elif c["evidence_quality"] == "medium" and c.get("consistent_across_halves") is None:
        failed.append("its consistency across the two halves of the periods could not be checked")
    share = abs(c["share_of_change"] or 0)
    if share < MIN_MATERIAL_SHARE:
        failed.append(f"it accounts for only {share * 100:.0f}% of the change")
    if (c["contribution"] or 0) * (ctx["delta"] or 0) <= 0:
        failed.append("it moved against the overall change, so it offsets rather than explains it")
    if c.get("numeric_unsupported"):
        failed.append(c["numeric_unsupported"])
    gap = c.get("_gap")
    ci = c.get("_gap_ci")
    if c["presence"] != "both":
        rest = c.get("rest_change_pct")
        if rest is not None and abs(rest) >= 0.5 * abs(ctx["overall_pct"] or 0) and rest * (ctx["overall_pct"] or 0) > 0:
            failed.append(f"the rest of the data already moved {_pct(rest)}, so this segment is not what makes the period different")
    elif gap is not None and (abs(gap) < max(MARGINAL_FLOOR * 2, 0.5 * abs(ctx["overall_pct"] or 0)) or (ci and ci[0] <= 0 <= ci[1])):
        failed.append(f"its change ({_pct(c['change_pct'])}) is not distinct from the rest of the data ({_pct(c['rest_change_pct'])}): "
                      "it is large only because it is a large part of the total")
    adj = c["adjustment"]["status"]
    if adj == "absorbed":
        failed.append("its effect disappears when " + " / ".join(c["adjustment"]["absorbed_by"]) + " is held fixed")
    elif adj == "overlaps" and not c.get("representative"):
        failed.append("it describes the same rows as " + " / ".join(c["adjustment"]["same_rows_as"]) + " and cannot be told apart from it")
    if ctx["within_noise"]:
        failed.append("the overall change is within the metric's ordinary variation")
    if ctx["outlier_sensitive"]:
        failed.append("the overall change depends on a few extreme rows")
    if ctx["incomparable"]:
        failed.append("the two periods are not fully comparable")
    if ctx["artefact"]:
        failed.append(f"'{ctx['artefact']['column']}' is recorded far less often in one period, so the change may be a recording effect")
    if adj == "absorbed":
        return "overlapping", failed
    if adj == "overlaps" and not c.get("representative"):
        return "overlapping", failed
    return ("plausible_contributor" if not failed else "association"), failed


# -------------------------------------------------------------- the entry --

def diagnose(df: pd.DataFrame, profile: dict, *, formula: me.Formula, label: str, base_filters: list[dict],
             time_column: str, period_a: list[str], period_b: list[str], fa: list[dict], fb: list[dict],
             scale: float, delta: float, base: dict, dimensions: list[dict], ranked: list[dict],
             question: str, inputs: dict, within_noise: bool, comparability: dict, spec_notes: list[str],
             budget: inv.Budget) -> dict:
    additive = formula.is_additive
    dfx = df.copy(deep=False)
    notes: list[str] = []
    exclude = {time_column, *formula.columns, *(f["column"] for f in base_filters)}
    problem = _problem(question, label, formula, base_filters, base["periods"], base["totals"], None, spec_notes)

    # 1. numeric drivers
    numeric_factors, band_decomps = [], []
    try:
        numeric_factors, band_decomps = _numeric_scan(df, dfx, profile, formula, base_filters, fa, fb, exclude, budget, scale, delta, notes)
    except me.FormulaError as e:
        notes.append(f"Numeric factors could not be scanned: {e}")

    # 2. candidate pool: categorical segments (already ranked) + numeric bands
    pool = []
    for r in ranked:
        pool.append({**r, "kind": "categorical", "column": r["dimension"]})
    for dec in band_decomps:
        for s in dec["segments"]:
            if s["share_of_change"] is None:
                continue
            pool.append({"dimension": dec["dimension"], "segment": s["segment"], "a": s["a"], "b": s["b"],
                         "contribution": s["contribution"], "share_of_change": s["share_of_change"],
                         "rows_a": s["rows_a"], "rows_b": s["rows_b"], "presence": s["presence"],
                         "method": dec["method"], "kind": "numeric_band", "column": dec["column"]})
    weight = {"high": 1.0, "medium": 0.7, "low": 0.35}
    pool.sort(key=lambda c: -abs(c["contribution"]))
    for rank, c in enumerate(pool):
        if "evidence_quality" not in c or c.get("consistent_across_halves") is None:
            consistent = None
            if c["presence"] == "both" and rank < 12:
                # a rate has no additive contribution: the direction to hold is the segment's own change
                direction = c["contribution"] if additive else (c["b"] or 0) - (c["a"] or 0)
                consistent = inv._consistency(dfx, formula, base_filters, time_column, period_a, period_b,
                                              c["dimension"], c["segment"], direction, budget, scale if additive else 1.0)
            c["consistent_across_halves"] = consistent
            if c["presence"] == "both":
                c["evidence_quality"] = inv._quality(c["rows_a"], c["rows_b"], consistent)
            else:   # present in one period only: judged on its rows in the period it exists, steady across that period's two halves
                here = period_b if c["presence"] == "new" else period_a
                c["consistent_across_halves"] = _steady_presence(dfx, time_column, c["dimension"], c["segment"], here)
                n = c["rows_b"] if c["presence"] == "new" else c["rows_a"]
                c["evidence_quality"] = inv._quality(n, n, c["consistent_across_halves"])
    pool.sort(key=lambda c: -abs(c["contribution"]) * weight[c["evidence_quality"]])
    # A band and its own column's categorical twin are the same thing; one candidate per (dimension, segment).
    seen, candidates = set(), []
    for c in pool:
        key = (c["dimension"], c["segment"])
        if key in seen or (c["presence"] != "both" and c["kind"] != "categorical"):
            continue
        if c["kind"] == "numeric_band":     # one band per numeric column: the strongest
            if ("band", c["column"]) in seen:
                continue
            seen.add(("band", c["column"]))
        seen.add(key)
        candidates.append(c)
    candidates = candidates[:MAX_CANDIDATES]
    for c in candidates:
        c["name"] = f"{c['dimension']} = {c['segment']}"

    numeric_by_col = {n["column"]: n for n in numeric_factors}
    for c in candidates:
        c["numeric_unsupported"] = None
        if c["kind"] == "numeric_band":
            rel = (numeric_by_col.get(c["column"]) or {}).get("relationship") or {}
            ok = len(rel) == 2 and all(abs(v["rho"]) >= 0.1 for v in rel.values()) and rel["a"]["rho"] * rel["b"]["rho"] > 0
            if not ok:
                c["numeric_unsupported"] = (f"{c['column']} itself shifted between the periods, so a value band mixes different rows in each period, "
                                            "and no consistent row-level relationship between it and the outcome was found in both periods")

    # 3. each candidate against the rest of the data, then against each other candidate
    boot = _Boot(dfx, formula, base_filters, fa, fb, scale, additive)
    if not boot.ok:
        notes.append("Intervals around the segment comparisons were not computed (the metric is not a sum, count, mean or ratio of those, "
                     "or there are too many rows); the comparisons use point estimates only.")
    for c in candidates:
        m = _member(dfx, c["dimension"], c["segment"])
        dfx["__m"] = m.astype(int).astype(str)
        split = _split(dfx, formula, base_filters, fa, fb, "__m", budget)
        pin = pout = None
        if split:
            ga, gb = split
            pin = _change_pct((ga.get("1") or {}).get("value"), (gb.get("1") or {}).get("value"), scale, additive)
            pout = _change_pct((ga.get("0") or {}).get("value"), (gb.get("0") or {}).get("value"), scale, additive)
        c["change_pct"], c["rest_change_pct"] = pin, pout
        c["_gap"] = None if pin is None or pout is None else pin - pout
        c["_gap_ci"] = None
        if boot.ok and c["_gap"] is not None:
            code = dfx["__m"]
            r = boot.pct({p_: code.loc[boot.idx[p_]].to_numpy() for p_ in ("a", "b")}, {"in": ["1"], "out": ["0"]})
            c["_gap_ci"] = _interval(r["in"] - r["out"], len(candidates))
    tests = []
    family = max(1, sum(1 for i, a in enumerate(candidates) for b in candidates[i + 1:] if a["column"] != b["column"]))
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1:]:
            if c1["column"] == c2["column"] or budget.exhausted or c1["presence"] != "both" or c2["presence"] != "both":
                continue
            res = _stratify(dfx, formula, base_filters, fa, fb, c1, c2, scale, additive, budget, boot, family)
            if res is None:
                continue
            tests.append({"a": c1["name"], "b": c2["name"], **res})
    for c in candidates:
        verdicts, absorbed_by, same_rows, inseparable, interactions = [], [], [], [], []
        for t in tests:
            if c["name"] not in (t["a"], t["b"]):
                continue
            mine, theirs, other = (t["first"], t["second"], t["b"]) if c["name"] == t["a"] else (t["second"], t["first"], t["a"])
            verdicts.append({"against": other, "result": mine["result"], "marginal_gap": mine["marginal_gap"],
                             "adjusted_gap": mine["adjusted_gap"], "within": mine["within"], "outside": mine["outside"]})
            if mine["interaction"]:
                interactions.append({"with": other, "within": mine["within"], "outside": mine["outside"]})
            if mine["result"] == "vanishes":
                (same_rows if theirs["result"] == "vanishes" else absorbed_by).append(other)
            elif mine["result"] == "not_separable":
                inseparable.append(other)
        tested = [v for v in verdicts if v["result"] not in ("no_gap", "not_separable", "unclear")]
        if absorbed_by:
            status = "absorbed"
        elif same_rows:
            status = "overlaps"
        elif c["presence"] != "both":
            status = "one_period"
        elif tested and all(v["result"] == "holds" for v in tested):
            status = "holds"
        elif tested:
            status = "attenuated"
        else:
            status = "untested"
        c["adjustment"] = {"status": status, "tests": verdicts, "absorbed_by": absorbed_by, "same_rows_as": same_rows,
                           "inseparable_from": inseparable, "interactions": interactions}

    # Candidates that describe the same rows are one finding seen several ways: the most direct view
    # (a named category over a value band, then the larger contribution) represents the group.
    name_of = {c["name"]: c for c in candidates}
    group = {c["name"]: c["name"] for c in candidates}

    def find(x):
        while group[x] != x:
            group[x] = group[group[x]]
            x = group[x]
        return x
    for c in candidates:
        for other in c["adjustment"]["same_rows_as"]:
            group[find(c["name"])] = find(other)
    for root in {find(n) for n in group}:
        members = [name_of[n] for n in group if find(n) == root]
        if len(members) > 1:
            rep = max(members, key=lambda m: (m["kind"] == "categorical", abs(m["contribution"])))
            rep["representative"] = True
            rep["adjustment"]["equivalent_views"] = [m["name"] for m in members if m is not rep]

    # 4. recording artefacts
    quality = _quality_checks(df, formula, base_filters, fa, fb, notes)
    outlier = _outlier_check(dfx, formula, base_filters, fa, fb, scale, delta)
    ctx = {"delta": delta, "overall_pct": ((base["totals"].get("like_for_like_pct") if base["totals"].get("length_adjusted") else base["totals"].get("change_pct")) or 0) / 100, "within_noise": within_noise, "outlier_sensitive": bool(outlier and outlier["sensitive"]),
           "incomparable": bool(comparability.get("issues")),
           "artefact": next((q for q in quality if q["check"] == "missing_inputs" and abs(q["missing_b"] - q["missing_a"]) >= 0.05), None)}

    # 5. ladder, wording
    factors = []
    for c in candidates:
        tier, failed = _tier(c, ctx)
        c["tier"], c["tier_reasons"] = tier, failed
    order = {"plausible_contributor": 0, "association": 1, "overlapping": 2}
    candidates.sort(key=lambda c: (order[c["tier"]], -abs(c["contribution"]) * weight[c["evidence_quality"]]))
    for c in candidates:
        factors.append(_factor_out(c, numeric_by_col.get(c["column"]), delta, label, period_a, period_b))
    lead = next((c for c in candidates if c["tier"] == "plausible_contributor"), None) or (candidates[0] if candidates else None)

    sub = _sub_location(dfx, formula, base_filters, fa, fb, lead, [d["dimension"] for d in dimensions], scale, budget) if lead and additive else None
    charts = _charts(dfx, formula, base_filters, time_column, period_a, period_b, lead, factors, numeric_factors, label, budget)
    diagnosis = _diagnosis(label, delta, factors, numeric_factors, sub, within_noise, outlier, ctx, comparability, base)
    return {
        "version": VERSION, "problem": problem, "diagnosis": diagnosis, "factors": factors,
        "numeric_factors": numeric_factors, "adjustment_tests": [_test_out(t) for t in tests],
        "sub_location": sub, "outlier_check": outlier, "data_checks": quality,
        "causal_status": _causal_status(profile, formula, base_filters, dimensions),
        "method_notes": notes, "charts": charts,
        "next_steps": _next_steps(factors, numeric_factors, sub, diagnosis, quality, problem, label),
        "how_to_read": [
            "Plausible contributor: located in the data, large enough to matter, holds across both halves of the periods and "
            "when each other leading factor is held fixed. It is a lead to investigate, not a proven cause.",
            "Association only: measured, but at least one of those checks failed; the failed checks are listed with it.",
            "Held fixed means comparing the segment with the rest of the data inside each stratum of the other factor. "
            "It adjusts for the factors in this table only; anything not recorded in the dataset is not adjusted for.",
        ],
    }


def _test_out(t: dict) -> dict:
    return {"a": t["a"], "b": t["b"], "a_given_b": t["first"], "b_given_a": t["second"],
            "cells": {k: {kk: v[kk] for kk in ("a", "b", "rows_a", "rows_b", "pct")} for k, v in t["cells"].items()}}


def _factor_out(c: dict, numeric: dict | None, delta: float, label: str, period_a, period_b) -> dict:
    share = c["share_of_change"]
    drives = (c["contribution"] or 0) * (delta or 0) > 0
    limitations = list(c["tier_reasons"])
    if c["presence"] != "both":
        limitations.append(f"it exists in only one period ({'appears' if c['presence'] == 'new' else 'disappears'}), so there is no like-for-like change to compare; "
                           "check whether it is a genuinely new/ended entity or a renamed one, and whether its rows replaced others")
    if c["adjustment"]["status"] == "untested":
        limitations.append("it could not be separated from the other leading factors (too few rows in the overlapping groups)")
    for t in c["adjustment"]["tests"]:
        if t["result"] == "attenuated" and t["marginal_gap"] is not None and t["adjusted_gap"] is not None:
            limitations.append(f"its gap to the rest shrinks from {_pct(t['marginal_gap'])} to {_pct(t['adjusted_gap'])} points when {t['against']} is held fixed, "
                               "so part of it may belong to that factor")
        elif t["result"] == "unclear":
            limitations.append(f"too noisy or unstable to tell whether it holds once {t['against']} is held fixed")
    for other in c["adjustment"]["inseparable_from"]:
        limitations.append(f"too few rows to tell it apart from {other}")
    out = {
        "factor": c["name"], "dimension": c["dimension"], "segment": c["segment"], "column": c["column"], "kind": c["kind"],
        "presence": c["presence"], "tier": c["tier"], "tier_label": TIER_LABEL[c["tier"]], "failed_checks": c["tier_reasons"],
        "role": "contributes to the change" if drives else "offsets the change",
        "evidence": {
            "value_a": c["a"], "value_b": c["b"], "rows_a": c["rows_a"], "rows_b": c["rows_b"],
            "change_pct": c["change_pct"], "rest_change_pct": c["rest_change_pct"],
            "gap_points": None if c["_gap"] is None else c["_gap"] * 100,
            "contribution": c["contribution"], "share_of_change": share, "method": c["method"],
        },
        "evidence_quality": c["evidence_quality"], "consistent_across_halves": c["consistent_across_halves"],
        "adjustment": c["adjustment"], "limitations": limitations,
    }
    if numeric:
        out["numeric_shift"] = {k: numeric[k] for k in ("mean_a", "mean_b", "mean_change_pct", "std_diff", "p_adjusted", "moved")}
        out["relationship"] = numeric.get("relationship")
    out["why_it_matters"] = _why(out, label)
    return out


def _why(f: dict, label: str) -> str:
    e = f["evidence"]
    own, rest = e["change_pct"], e["rest_change_pct"]
    if own is not None and rest is not None:
        line = (f"{label} in {f['factor']} changed {_pct(own)}, against {_pct(rest)} in the rest of the data "
                f"({e['rows_a']:,} → {e['rows_b']:,} rows). It accounts for {abs(e['share_of_change'] or 0) * 100:.0f}% of the measured change.")
    elif f.get("presence", "both") != "both" and rest is not None:
        line = (f"{f['factor']} exists in only one period ({e['rows_a']:,} → {e['rows_b']:,} rows) and accounts for "
                f"{abs(e['share_of_change'] or 0) * 100:.0f}% of the measured change; the rest of the data moved {_pct(rest)}.")
    else:
        line = f"It accounts for {abs(e['share_of_change'] or 0) * 100:.0f}% of the measured change."
    if f["adjustment"]["status"] == "holds":
        line += " The gap persists when each other leading factor is held fixed."
    for i in f["adjustment"]["interactions"][:1]:
        line += (f" The effect differs by {i['with']}: a {_pct(i['within'])} gap where it applies and {_pct(i['outside'])} elsewhere.")
    return line


# ----------------------------------------------------------- data artefacts --

def _quality_checks(df, formula, base, fa, fb, notes) -> list[dict]:
    """Things that could make the change an artefact of how rows were recorded."""
    sa, sb = me.apply_filters(df, base + fa), me.apply_filters(df, base + fb)
    checks = []
    for col in formula.columns:
        if col in df.columns:
            ma = float(pd.to_numeric(sa[col], errors="coerce").isna().mean()) if len(sa) else 0.0
            mb = float(pd.to_numeric(sb[col], errors="coerce").isna().mean()) if len(sb) else 0.0
            if abs(mb - ma) >= 0.05 or max(ma, mb) >= 0.2:
                checks.append({"check": "missing_inputs", "column": col, "missing_a": ma, "missing_b": mb,
                               "text": f"'{col}' is missing in {ma * 100:.0f}% of rows in the earlier period and {mb * 100:.0f}% in the later; "
                                       "a change in how often it was recorded could move the metric without any real change."})
    dup_a, dup_b = float(sa.duplicated().mean()) if len(sa) else 0.0, float(sb.duplicated().mean()) if len(sb) else 0.0
    if max(dup_a, dup_b) >= 0.02:
        checks.append({"check": "duplicate_rows", "dup_a": dup_a, "dup_b": dup_b,
                       "text": f"{dup_a * 100:.1f}% of earlier rows and {dup_b * 100:.1f}% of later rows are exact duplicates; "
                               "check whether they are repeated records or genuine repeats."})
    if len(sa) and len(sb) and not formula.is_additive and abs(len(sb) / len(sa) - 1) > 0.5:
        checks.append({"check": "volume_shift", "rows_a": len(sa), "rows_b": len(sb),
                       "text": f"The number of rows changed from {len(sa):,} to {len(sb):,}; a rate can move when the population it is "
                               "measured on changes in size or composition."})
    return checks


def _outlier_check(dfx, formula, base, fa, fb, scale, delta) -> dict | None:
    """Does the change survive when the most extreme 1% of values at each end are pulled in?"""
    col = formula.tree[2] if formula.tree[0] == "agg" and formula.tree[1] in ("sum", "avg") else None
    if not col or col not in dfx.columns:
        return None
    work = dfx.copy(deep=False)
    values = pd.to_numeric(work[col], errors="coerce")
    scoped = me.apply_filters(dfx, base + fa).index.union(me.apply_filters(dfx, base + fb).index)
    pool = values.loc[scoped].dropna()
    if len(pool) < 50:
        return None
    lo, hi = pool.quantile(0.01), pool.quantile(0.99)
    work[col] = values.clip(lo, hi)
    va = me.evaluate(work, formula, base + fa)["value"]
    vb = me.evaluate(work, formula, base + fb)["value"]
    if va is None or vb is None:
        return None
    trimmed = vb - va * (scale if formula.is_additive else 1.0)
    retained = trimmed / delta if delta else None
    sensitive = retained is not None and (retained < 0.5)
    return {"column": col, "raw_change": delta, "trimmed_change": trimmed, "retained": retained, "sensitive": bool(sensitive),
            "text": (f"With the most extreme 1% of '{col}' values at each end pulled in, the change is {_fmt(trimmed)} "
                     f"instead of {_fmt(delta)}"
                     + (": the change depends on a few extreme rows." if sensitive else ": the change does not depend on extreme rows."))}


def _causal_status(profile, formula, base_filters, dimensions) -> dict:
    design = [c["name"] for c in profile.get("columns", [])
              if c.get("kind") == "categorical" and _DESIGN.search(c["name"]) and 2 <= (c.get("unique_count") or 0) <= 4]
    text = ("The data is a record of what happened, not an experiment: nothing here can show that a factor caused the change. "
            "Every factor below is an association or a plausible contributor.")
    if design:
        text += (f" The columns {', '.join(design)} may encode an experiment or cohort split, but the data cannot show how rows were assigned, "
                 "so they are not treated as causal evidence.")
    return {"level": "observational", "design_columns": design, "text": text}


# ------------------------------------------------------------- drill + charts --

def _sub_location(dfx, formula, base, fa, fb, lead, dims, scale, budget) -> dict | None:
    """Where, inside the lead factor, its contribution concentrates."""
    if lead is None or lead["contribution"] == 0:
        return None
    seg = [{"column": lead["dimension"], "op": "==", "value": lead["segment"]}]
    best = None
    for d in dims:
        if d == lead["column"] or d not in dfx.columns or budget.exhausted:
            continue
        sub = inv._decompose_additive(dfx, formula, base + seg, fa, fb, d, lead["contribution"], budget, scale)
        if sub and sub["segments"]:
            s = sub["segments"][0]
            if s["share_of_change"] is not None and (best is None or abs(s["share_of_change"]) > abs(best["share"])):
                best = {"within": lead["name"], "dimension": d, "segment": s["segment"], "share": s["share_of_change"],
                        "contribution": s["contribution"], "rows_a": s["rows_a"], "rows_b": s["rows_b"], "a": s["a"], "b": s["b"]}
    return best


_TIER_COLOUR = {"plausible_contributor": "#1b6ca8", "association": "#8a94a3", "overlapping": "#d4a24c"}


def _charts(dfx, formula, base, time_column, period_a, period_b, lead, factors, numeric_factors, label, budget) -> list[dict]:
    charts = []
    top = factors[:8]
    if top:
        traces = []
        for tier in ("plausible_contributor", "association", "overlapping"):
            rows = [f for f in top if f["tier"] == tier]
            if rows:
                traces.append({"type": "bar", "orientation": "h", "name": TIER_LABEL[tier], "marker": {"color": _TIER_COLOUR[tier]},
                               "y": [f["factor"] for f in rows], "x": [f["evidence"]["share_of_change"] * 100 for f in rows],
                               "hovertemplate": "%{y}<br>%{x:.0f}% of the change<extra></extra>"})
        charts.append({"title": "Candidate factors by share of the change", "kind": "factor_ranking", "plotly_json": {
            "data": traces, "layout": {"title": {"text": "Share of the change located in each candidate factor"},
                                       "xaxis": {"title": {"text": "% of the change (negative = moved against it)"}},
                                       "yaxis": {"autorange": "reversed"}, "barmode": "overlay", "margin": {"l": 190},
                                       "legend": {"orientation": "h", "y": -0.2}}}})
    if lead is not None:
        try:
            lo, hi = pd.Timestamp(period_a[0]), pd.Timestamp(period_b[1])
            ts = pd.to_datetime(dfx[time_column], errors="coerce")
            span = (hi - lo).days
            freq = "D" if span <= 45 else "W" if span <= 200 else "M"
            dfx["__bucket"] = ts.dt.to_period(freq).dt.start_time.dt.strftime("%Y-%m-%d").where(ts.notna(), None)
            window = [{"column": time_column, "op": ">=", "value": period_a[0]}, {"column": time_column, "op": "<=", "value": period_b[1]}]
            if budget.spend() and budget.spend():
                inside = me.evaluate(dfx, formula, base + window + [{"column": lead["dimension"], "op": "==", "value": lead["segment"]}], ["__bucket"])
                rest = me.evaluate(dfx, formula, base + window + [{"column": lead["dimension"], "op": "!=", "value": lead["segment"]}], ["__bucket"])
                series = lambda r: sorted((g["key"]["__bucket"], g["value"]) for g in r["groups"] if g["key"]["__bucket"])
                si, sr = series(inside), series(rest)
                if len(si) >= 3:
                    charts.append({"title": f"{label} over time: {lead['name']} against the rest", "kind": "lead_trend", "plotly_json": {
                        "data": [{"type": "scatter", "mode": "lines", "name": lead["name"], "x": [x for x, _ in si], "y": [y for _, y in si],
                                  "line": {"color": _TIER_COLOUR["plausible_contributor"], "width": 2.5}},
                                 {"type": "scatter", "mode": "lines", "name": "All other rows", "x": [x for x, _ in sr], "y": [y for _, y in sr],
                                  "line": {"color": _TIER_COLOUR["association"], "width": 2, "dash": "dot"}}],
                        "layout": {"title": {"text": f"{label}: {lead['name']} against the rest, by {'day' if freq == 'D' else 'week' if freq == 'W' else 'month'}"},
                                   "xaxis": {"title": {"text": "Start of " + ("day" if freq == "D" else "week" if freq == "W" else "month")}},
                                   "yaxis": {"title": {"text": label}},
                                   "shapes": [{"type": "line", "x0": pd.Timestamp(period_b[0]).strftime("%Y-%m-%d"), "x1": pd.Timestamp(period_b[0]).strftime("%Y-%m-%d"),
                                               "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#555", "width": 1, "dash": "dash"}}],
                                   "annotations": [{"x": pd.Timestamp(period_b[0]).strftime("%Y-%m-%d"), "y": 1, "yref": "paper", "text": "comparison period starts",
                                                    "showarrow": False, "xanchor": "left", "font": {"size": 11}}],
                                   "legend": {"orientation": "h", "y": -0.25}}}})
        except (me.FormulaError, ValueError, KeyError):
            pass
    moved = [n for n in numeric_factors if n["moved"]]
    if moved:
        n = max(moved, key=lambda x: abs(x["std_diff"]))

        def box(tag, name, colour):
            return {"type": "box", "name": name, "q1": [n[f"quartiles_{tag}"][0]], "median": [n[f"median_{tag}"]], "q3": [n[f"quartiles_{tag}"][1]],
                    "lowerfence": [n[f"range_{tag}"][0]], "upperfence": [n[f"range_{tag}"][1]], "mean": [n[f"mean_{tag}"]],
                    "marker": {"color": colour}, "boxpoints": False, "x": [n["column"]]}
        charts.append({"title": f"How {n['column']} shifted between the periods", "kind": "numeric_shift", "plotly_json": {
            "data": [box("a", "Earlier period", _TIER_COLOUR["association"]), box("b", "Later period", _TIER_COLOUR["plausible_contributor"])],
            "layout": {"title": {"text": f"Distribution of {n['column']}: earlier and later period"}, "boxmode": "group",
                       "yaxis": {"title": {"text": n["column"]}}, "legend": {"orientation": "h", "y": -0.2}}}})
    return charts


# ---------------------------------------------------- diagnosis and wording --

def _diagnosis(label, delta, factors, numeric_factors, sub, within_noise, outlier, ctx, comparability, base) -> dict:
    plausible = [f for f in factors if f["tier"] == "plausible_contributor" and f["role"].startswith("contributes")]
    assoc = [f for f in factors if f["tier"] == "association" and f["role"].startswith("contributes")]
    totals = base["totals"]
    move = f"{label} {'fell' if delta < 0 else 'rose'} from {_fmt(totals['a'])} to {_fmt(totals['b'])} ({totals['change_pct']:+.1f}%)." \
        if totals.get("change_pct") is not None else f"{label} changed from {_fmt(totals['a'])} to {_fmt(totals['b'])}."
    if comparability.get("issues"):
        code = "insufficient_evidence"
        text = move + " The two periods are not fully comparable, so the data cannot support naming a contributing factor."
    elif outlier and outlier["sensitive"]:
        code = "insufficient_evidence"
        text = move + " " + outlier["text"] + " No factor can be named until those rows are explained."
    elif ctx["artefact"]:
        a = ctx["artefact"]
        code = "possible_artefact"
        text = (move + f" '{a['column']}' is missing in {a['missing_a'] * 100:.0f}% of earlier rows and {a['missing_b'] * 100:.0f}% of later rows, so the change may "
                "reflect what was recorded rather than what happened. Confirm that before looking for a behavioural cause.")
    elif within_noise and not plausible:
        code = "insufficient_evidence"
        text = move + " The change is within the metric's ordinary variation, so the segments below may be describing noise, not a real shift."
    elif plausible:
        lead = plausible[0]
        share = abs(lead["evidence"]["share_of_change"]) * 100
        code = "driver_located"
        ev = lead["evidence"]
        if lead.get("presence", "both") != "both":
            appears = lead["presence"] == "new"
            own = (f" and it {'has no rows in the baseline period' if appears else 'has no rows in the comparison period'} "
                   f"({ev['rows_a']:,} → {ev['rows_b']:,} rows) while the rest of the data moved {_pct(ev['rest_change_pct'])}")
        else:
            own = f" and its {label} moved {_pct(ev['change_pct'])} against {_pct(ev['rest_change_pct'])} elsewhere"
        text = (f"{move} The strongest plausible contributing factor is {lead['factor']}: it accounts for {share:.0f}% of the change"
                + own
                + (f"; the change is concentrated in {sub['dimension']} = {sub['segment']} ({abs(sub['share']) * 100:.0f}% of it)." if sub else "."))
        if len(plausible) > 1:
            code = "multiple_factors"
            text += f" {len(plausible) - 1} other factor{'s' if len(plausible) > 2 else ''} also pass every check, so the change likely has more than one source."
        if outlier and outlier["sensitive"]:
            code, text = "insufficient_evidence", move + " The change depends on a few extreme rows, so no factor can be named with confidence."
    elif not any(not any("not distinct" in x for x in f["failed_checks"]) and f["role"].startswith("contributes") for f in factors):
        code = "diffuse"
        text = move + " No segment moved distinctly from the rest of the data: the change is spread across the whole dataset, not located in one factor."
    elif assoc:
        code = "unconfirmed"
        lead = assoc[0]
        text = (f"{move} {lead['factor']} accounts for {abs(lead['evidence']['share_of_change']) * 100:.0f}% of it, but it did not pass every check: "
                + "; ".join(lead["failed_checks"][:2]) + ". Treat it as a lead, not a finding.")
    elif factors:
        code = "diffuse"
        text = move + " No single factor accounts for a material share: the change is spread across many segments."
    else:
        code = "insufficient_evidence"
        text = move + " The dataset has no factor with enough rows to compare, so the data cannot say what contributed."
    moved_numeric = [n for n in numeric_factors if n["moved"]]
    if moved_numeric and code != "insufficient_evidence":
        n = max(moved_numeric, key=lambda x: abs(x["std_diff"]))
        text += f" Separately, {n['column']} shifted from {_fmt(n['mean_a'])} to {_fmt(n['mean_b'])} on average ({_pct(n['mean_change_pct'])})."
    return {"code": code, "headline": text, "n_plausible": len(plausible), "n_association": len(assoc)}


def _next_steps(factors, numeric_factors, sub, diagnosis, quality, problem, label) -> list[str]:
    steps = []
    plausible = [f for f in factors if f["tier"] == "plausible_contributor" and f["role"].startswith("contributes")]
    for f in plausible[:2]:
        e = f["evidence"]
        steps.append(f"Find what changed for {f['factor']} around {str(problem['comparison'][0])[:10]}, when the later period starts "
                     f"({label} {_fmt(e['value_a'])} → {_fmt(e['value_b'])}); look for a pricing, process, staffing, supply or tracking change recorded "
                     "for that group, since the data itself cannot say which.")
    if sub:
        steps.append(f"Within {sub['within']}, the change sits mainly in {sub['dimension']} = {sub['segment']} "
                     f"({_fmt(sub['a'])} → {_fmt(sub['b'])}, {abs(sub['share']) * 100:.0f}% of the factor's change): start the review there.")
    for f in factors:
        if f["tier"] == "overlapping" and f["adjustment"]["absorbed_by"]:
            steps.append(f"{f['factor']} looks like a proxy for {' / '.join(f['adjustment']['absorbed_by'])}: its decline disappears once that is held fixed. "
                         "Investigate the latter instead of the former.")
            break
    for n in [x for x in numeric_factors if x["moved"]][:1]:
        steps.append(f"{n['column']} moved {_pct(n['mean_change_pct'])} on average between the periods. Check whether its definition, "
                     f"data source or the way it is set changed, and whether {label} responds to it (see its relationship under the factor).")
    if diagnosis["code"] in ("unconfirmed", "diffuse", "insufficient_evidence"):
        steps.append("Extend the comparison to more periods, or add columns that record candidate causes (pricing, campaigns, staffing, outages), "
                     "so competing explanations can be separated.")
    for q in quality[:2]:
        steps.append("Rule out a recording artefact first: " + q["text"])
    return steps


# --------------------------------------------------------------- scope read --

_STOP = {"all", "other", "none", "yes", "no", "true", "false", "total", "new", "old", "high", "low", "medium"}


def infer_scope(profile: dict, question: str, exclude: set[str]) -> tuple[list[dict], list[str]]:
    """Narrow the investigation to a segment the question names ("why did sales fall in the West?").
    Only when exactly one value of exactly one column is named; anything else is reported, not guessed."""
    text = " " + re.sub(r"[_\-]+", " ", (question or "").lower()) + " "
    hits = []
    for c in profile.get("columns", []):
        if c.get("kind") != "categorical" or c["name"] in exclude:
            continue
        for tv in c.get("top_values") or []:
            value = tv.get("value") if isinstance(tv, dict) else tv
            v = str(value).strip()
            if len(v) < 3 or v.lower() in _STOP:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(re.sub('[_-]+', ' ', v.lower()))}(?![a-z0-9])", text):
                hits.append((c["name"], v))
    cols = {c for c, _ in hits}
    if len(hits) == 1:
        c, v = hits[0]
        return [{"column": c, "op": "==", "value": v}], [f"The question names {c} = {v}, so the investigation is limited to those rows."]
    if hits and len(cols) == 1:
        c = next(iter(cols))
        return [{"column": c, "op": "in", "value": [v for _, v in hits]}], [f"The question names several {c} values ({', '.join(v for _, v in hits)}), so the investigation is limited to those rows."]
    if hits:
        return [], ["The question names values in more than one column (" + ", ".join(f"{c} = {v}" for c, v in hits[:4]) +
                    "); it was not narrowed, because it is unclear which group is meant."]
    return [], []
