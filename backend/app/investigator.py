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
# Two periods whose lengths differ by more than this are not compared total against
# total: a sum over 31 days exceeds one over 28 with nothing having changed.
LENGTH_TOLERANCE = 0.02


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
                if re.search(rf"(?<![a-z0-9]){re.escape(re.sub('[_-]+', ' ', n.lower()))}(?![a-z0-9])", text)]
        if len(hits) == 1:
            formula, label = f'sum("{hits[0]}")', f"Total {hits[0]}"
        elif not hits and len(numeric) == 1:
            formula, label = f'sum("{numeric[0]}")', f"Total {numeric[0]}"
    if formula is None:
        return None
    return {"formula": formula, "label": label, "filters": filters, "time_column": time_cols[0], "question": question}


_MONTHS = {m: i + 1 for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"))}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_QUARTER = re.compile(r"\bq([1-4])\b(?:\s+(\d{4}))?", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_RELATIVE = re.compile(r"\b(last|past|previous|prior|this|current)\s+(?:(\d{1,3})\s+)?(day|week|month|quarter|year)s?\b", re.IGNORECASE)
_NAMED_MONTH = re.compile(r"\b(?:in|during|for)\s+(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?(?:\s+(\d{4}))?",
                          re.IGNORECASE)
_UNIT_MONTHS = {"month": 1, "quarter": 3, "year": 12}


def _unit_start(ts: pd.Timestamp, unit: str) -> pd.Timestamp:
    day = ts.normalize()
    if unit == "month":
        return day.replace(day=1)
    if unit == "quarter":
        return day.replace(month=3 * ((day.month - 1) // 3) + 1, day=1)
    return day.replace(month=1, day=1)


def question_periods(question: str, df: pd.DataFrame, time_column: str) -> dict | None:
    """The two periods a question names ("last month", "the last 14 days",
    "in March"), read against the dataset's own calendar: "last" means the most
    recent stretch the data covers, not the most recent on today's calendar.
    The earlier period is the stretch of the same kind immediately before.

    Returns {"period_a", "period_b", "source"}, {"error": reason} when the
    question names a period the dataset cannot compare, or None when it names
    no period this function reads."""
    ts = pd.to_datetime(df[time_column], errors="coerce").dropna()
    if ts.empty or not question:
        return None
    lo, hi = ts.min(), ts.max()
    data_end = hi.normalize() + pd.Timedelta(days=1)          # exclusive
    relative, named = _RELATIVE.search(question), _NAMED_MONTH.search(question)
    quarter = _QUARTER.search(question)
    if not relative and _YESTERDAY.search(question):
        day = pd.Timedelta(days=1)
        return _checked_periods("yesterday", "the final day of data, against the day before",
                                data_end - 2 * day, data_end - day, data_end - day, data_end, lo, hi)
    if relative:
        word, count, unit = relative.group(1).lower(), relative.group(2), relative.group(3).lower()
        n = max(1, int(count)) if count else 1
        phrase = relative.group(0)
        if unit in ("day", "week"):
            span = pd.Timedelta(days=n * (7 if unit == "week" else 1))
            end_b, start_b = data_end, data_end - span
            start_a = start_b - span
            reading = f"the final {span.days} day(s) of data"
        elif word in ("this", "current"):
            start_b, end_b = _unit_start(hi, unit), data_end
            start_a = start_b - pd.DateOffset(months=_UNIT_MONTHS[unit])
            reading = f"the {unit} to date, against the same number of days at the start of the {unit} before"
            end_a = min(start_a + (end_b - start_b), start_b)
            return _checked_periods(phrase, reading, start_a, end_a, start_b, end_b, lo, hi)
        else:
            # The most recent complete calendar unit: the one the data ends in
            # only if the data runs to its final day.
            current = _unit_start(hi, unit)
            complete = current + pd.DateOffset(months=_UNIT_MONTHS[unit]) == data_end
            end_b = data_end if complete else current
            start_b = end_b - pd.DateOffset(months=n * _UNIT_MONTHS[unit])
            start_a = start_b - pd.DateOffset(months=n * _UNIT_MONTHS[unit])
            reading = f"the most recent {n if n > 1 else ''} complete calendar {unit}{'s' if n > 1 else ''} in the data".replace("  ", " ")
        return _checked_periods(phrase, reading, start_a, start_b, start_b, end_b, lo, hi)
    if named:
        month = _MONTHS[named.group(1).lower()]
        year = int(named.group(2)) if named.group(2) else (hi.year if month <= hi.month else hi.year - 1)
        try:
            start_b = pd.Timestamp(year=year, month=month, day=1, tz=hi.tz)
        except ValueError:
            return None
        end_b = start_b + pd.DateOffset(months=1)
        start_a = start_b - pd.DateOffset(months=1)
        return _checked_periods(named.group(0), "that calendar month, against the month before",
                                start_a, start_b, start_b, end_b, lo, hi)
    if quarter:
        first_month = 3 * (int(quarter.group(1)) - 1) + 1
        year = int(quarter.group(2)) if quarter.group(2) else (hi.year if first_month <= hi.month else hi.year - 1)
        start_b = pd.Timestamp(year=year, month=first_month, day=1, tz=hi.tz)
        return _checked_periods(quarter.group(0).strip(), "that calendar quarter, against the quarter before",
                                start_b - pd.DateOffset(months=3), start_b, start_b, start_b + pd.DateOffset(months=3), lo, hi)
    return None


def _checked_periods(phrase: str, reading: str, start_a, end_a, start_b, end_b, lo, hi) -> dict:
    last_day = str(hi.date())
    if start_b > hi or end_b <= lo:
        return {"error": f"The question asks about \"{phrase}\" ({start_b.date()} to {(end_b - pd.Timedelta(days=1)).date()}), "
                         f"but the dataset only covers {lo.date()} to {last_day}."}
    if end_a <= lo:
        return {"error": f"The question asks about \"{phrase}\" ({start_b.date()} to {(end_b - pd.Timedelta(days=1)).date()}), "
                         f"but the dataset starts on {lo.date()}, so there is no earlier period to compare it with."}
    data_end = hi.normalize() + pd.Timedelta(days=1)
    if end_b > data_end:
        # The data stops part-way through the period asked about. Comparing a
        # part-period with a whole one would show a fall that is only the
        # missing days, so both are cut to the same number of days from their start.
        end_b = data_end
        end_a = min(end_a, start_a + (end_b - start_b))
        reading += f"; the data stops on {last_day}, so both periods are cut to their first {(end_b - start_b).days} day(s)"
    # Period B's end is inclusive in the engine; stop just short of the next day.
    inclusive_end = end_b - pd.Timedelta(microseconds=1)
    return {"period_a": [start_a.isoformat(), end_a.isoformat()], "period_b": [start_b.isoformat(), inclusive_end.isoformat()],
            "source": f"read from the question: \"{phrase}\" taken as {reading} (the dataset ends on {last_day})"}


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


def _period_days(period: list[str], last: bool) -> float:
    """Length of a period in days. The later period includes its end: an end
    at midnight (a date) or at the last instant of a day counts that whole day."""
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if last and (end == end.normalize() or end + pd.Timedelta(microseconds=1) == (end + pd.Timedelta(microseconds=1)).normalize()):
        end = end.normalize() + pd.Timedelta(days=1) if end == end.normalize() else end + pd.Timedelta(microseconds=1)
    return (end - start).total_seconds() / 86400


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

def _decompose_additive(df, formula, base_filters, fa, fb, dimension, delta, budget, scale: float = 1.0) -> dict | None:
    """`scale` is (days in the later period) / (days in the earlier one) when
    the two differ: each segment's earlier value is scaled to the later
    period's length, so a segment's contribution is its change at equal length
    and the contributions sum to `delta`, the like-for-like change."""
    if not (budget.spend() and budget.spend()):
        return None
    a = _by_segment(me.evaluate(df, formula, base_filters + fa, [dimension]), dimension)
    b = _by_segment(me.evaluate(df, formula, base_filters + fb, [dimension]), dimension)
    segments = []
    for key in sorted(set(a) | set(b)):
        va, vb = (a.get(key) or {}).get("value") or 0.0, (b.get(key) or {}).get("value") or 0.0
        contribution = vb - va * scale
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
                 direction: float, budget, scale: float = 1.0) -> bool | None:
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
        signs.append(vb - va * scale)
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

def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


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
        read = question_periods(spec.get("question") or "", scoped, time_column)
        if read is not None and read.get("error"):
            return _inconclusive(label, formula, spec, budget, read["error"])
        if read is not None:
            period_a, period_b, periods_source = read["period_a"], read["period_b"], read["source"]
        else:
            auto = _auto_periods(scoped, time_column)
            if auto is None:
                return _inconclusive(label, formula, spec, budget, "The time column has fewer than two distinct moments, "
                                     "so there are no two periods to compare.")
            period_a, period_b = auto
            periods_source = "automatic: the time range split at its midpoint"
            if spec.get("question"):
                periods_source += " (the question names no period this investigator can read)"
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
    if not res_a["rows"] or not res_b["rows"]:
        which = "earlier" if not res_a["rows"] else "later"
        return {**_inconclusive(label, formula, spec, budget, f"The {which} period has no rows at all, so there is "
                                "nothing to compare the other period with."), **base}
    if total_a is None or total_b is None:
        return {**_inconclusive(label, formula, spec, budget, "The metric has no value in one of the two periods."), **base}
    delta = total_b - total_a
    pct = delta / abs(total_a) * 100 if total_a else None
    base["totals"].update({"change": delta, "change_pct": pct})

    # A total over a longer period is larger with nothing having changed. For a
    # sum or a count, the earlier value is scaled to the later period's length
    # and only the remainder (the like-for-like change) is explained.
    days_a, days_b = _period_days(period_a, last=False), _period_days(period_b, last=True)
    scale = 1.0
    if formula.is_additive and days_a > 0 and days_b > 0 and abs(days_b / days_a - 1) > LENGTH_TOLERANCE:
        scale = days_b / days_a
    length_effect = total_a * (scale - 1)
    explain = delta - length_effect
    explain_pct = explain / abs(total_a * scale) * 100 if total_a else None
    base["totals"].update({"days_a": round(days_a, 3), "days_b": round(days_b, 3), "length_adjusted": scale != 1.0,
                           "length_effect": length_effect, "like_for_like_change": explain, "like_for_like_pct": explain_pct,
                           "per_day_a": total_a / days_a if formula.is_additive and days_a else None,
                           "per_day_b": total_b / days_b if formula.is_additive and days_b else None})
    if delta == 0 or (pct is not None and abs(pct) < MATERIAL_CHANGE_PCT):
        return {**_inconclusive(label, formula, spec, budget,
                                f"The metric moved by less than {MATERIAL_CHANGE_PCT:g}% between the periods; "
                                "there is no material change to explain."), **base}
    if scale != 1.0 and (explain == 0 or (explain_pct is not None and abs(explain_pct) < MATERIAL_CHANGE_PCT)):
        return {**_inconclusive(
            label, formula, spec, budget,
            f"{label} went from {_fmt(total_a)} to {_fmt(total_b)}, but the later period is {_fmt(days_b)} days against "
            f"{_fmt(days_a)}. Per day it went from {_fmt(total_a / days_a)} to {_fmt(total_b / days_b)}: the difference in "
            "the totals is the difference in length, and there is no material change to explain."), **base}

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
                out = _decompose_additive(df, formula, base_filters, fa, fb, d, explain, budget, scale)
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
            ranked.append({"dimension": dim["dimension"], "segment": s["segment"], "a": s["a"], "b": s["b"],
                           "contribution": s["contribution"],
                           "share_of_change": s["share_of_change"], "rows_a": s["rows_a"], "rows_b": s["rows_b"],
                           "presence": s["presence"], "method": dim["method"]})
    ranked.sort(key=lambda r: -abs(r["contribution"]))
    for r in ranked[:5]:
        consistent = None
        if r["presence"] == "both" and formula.is_additive:
            consistent = _consistency(df, formula, base_filters, time_column, period_a, period_b,
                                      r["dimension"], r["segment"], r["contribution"], budget, scale)
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
        sub = _decompose_additive(df, formula, base_filters + seg_filter, fa, fb, others[0], r["contribution"], budget, scale)
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

    evidence = _statistical_evidence(df, profile, formula, label, base_filters, time_column, period_a, period_b,
                                     total_a, total_b, ranked, budget)
    overall, comparability, pre_trend = evidence["overall"], evidence["comparability"], evidence["pre_trend"]
    if scale != 1.0:
        # Unequal lengths are handled by the scaling above, so they are no longer a reason to distrust the result.
        comparability["issues"] = [i for i in comparability["issues"] if "different lengths" not in i]
        comparability["comparable"] = not comparability["issues"]
        comparability["length_adjusted"] = True
    if comparability["issues"]:
        unresolved.append("The two periods are not fully comparable: " + "; ".join(comparability["issues"]) + ".")
    trending = (pre_trend["status"] == "ok" and pre_trend["significant"] and pre_trend["same_direction"]
                and (pre_trend["share"] or 0) >= STRONG_SHARE)
    if trending:
        alternatives.insert(0, f"The metric was already moving this way before the later period began: its trend in the "
                               f"earlier period, continued, would produce {pre_trend['share'] * 100:.0f}% of the change in "
                               f"its {pre_trend['bucket']} level. Whatever explains the change may predate the later period.")
    if not total_a:
        unresolved.append("The metric was zero in the earlier period, so no percentage change is defined.")
    within_noise = overall["status"] == "ok" and not overall["significant"]
    if within_noise:
        unresolved.append(f"As a series of {overall['bucket']} values, the difference between the periods is within the "
                          f"metric's ordinary variation ({_p_text(overall['p_value'])}), so the contributors below may "
                          "be describing noise.")

    conclusive = bool(ranked) and not low_only and not within_noise and not comparability["issues"]
    direction = "rose" if delta > 0 else "fell"
    summary = (f"{label} {direction} from {_fmt(total_a)} to {_fmt(total_b)} "
               f"({_fmt(delta)}{f', {pct:+.1f}%' if pct is not None else ''}).")
    of_what = "the change"
    if scale != 1.0:
        of_what = "the like-for-like change"
        summary += (f" The later period is {_fmt(days_b)} days against {_fmt(days_a)}, which alone accounts for "
                    f"{_fmt(length_effect)}; like for like the change is {_fmt(explain)}"
                    f"{f' ({explain_pct:+.1f}%)' if explain_pct is not None else ''}.")
    lead = next((r for r in ranked if (r["share_of_change"] or 0) > 0), ranked[0] if ranked else None)
    if lead is not None:
        summary += (f" The largest measured contributor is {lead['dimension']} = {lead['segment']}, accounting for "
                    f"{lead['share_of_change'] * 100:.0f}% of {of_what} ({lead['evidence_quality']} evidence quality)"
                    + ("; other segments moved the other way." if lead["share_of_change"] > 1 else "."))
    if pvm and pvm["reconciles"] and scale == 1.0:
        # On periods of unequal length the split is of the raw totals, so its volume effect
        # contains the extra days; it stays in the evidence table, labelled, and out of the summary.
        summary += f" Price effect {_fmt(pvm['price_effect'])}, volume effect {_fmt(pvm['volume_effect'])}."
    if overall["status"] == "ok":
        summary += (f" Per {overall['bucket']}, the level differs by {_fmt(overall['estimate'])} "
                    f"({overall['level'] * 100:.0f}% interval {_fmt(overall['low'])} to {_fmt(overall['high'])}, "
                    f"{_p_text(overall['p_value'])})"
                    + ("." if overall["significant"] else ", which is within its ordinary variation."))
    if comparability["issues"]:
        summary += " The two periods are not fully comparable, so this is not a like-for-like result."
    elif not conclusive:
        summary += " The evidence is not strong enough to single out a contributor."
    length = {"days_a": days_a, "days_b": days_b, "effect": length_effect} if scale != 1.0 else None
    evidence_table = _evidence_table(ranked, pvm, delta, comparability, length, pre_trend if trending else None)
    follow_ups = _follow_ups(ranked, drilldowns, dims, skipped, comparability, overall, period_b, low_only, pre_trend, lead)

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
        "change_evidence": {"overall": overall, "pre_trend": pre_trend, "family": evidence["family"]},
        "comparability": comparability,
        "evidence_table": evidence_table, "follow_ups": follow_ups,
        "budget": budget.used(),
    }


# ------------------------------------------------------ statistical evidence --

STAT_CONTRIBUTORS = 5


def _compact_stat(result: dict | None, bucket: str = "") -> dict:
    if result is None:
        return {"status": "not_run", "reason": "outside this investigation's computation budget"}
    if result.get("status") != "ok":
        return {"status": "unavailable", "reason": result.get("reason") or "the test could not be run"}
    interval, test = result.get("interval") or {}, result.get("test") or {}
    return {"status": "ok", "method": result.get("method_label"), "bucket": bucket, "estimate": result.get("estimate"),
            "low": interval.get("low"), "high": interval.get("high"), "level": interval.get("level"),
            "p_value": test.get("p_value"), "alpha": result.get("alpha"), "significant": bool(result.get("significant")),
            "buckets": (result.get("sample") or {}).get("groups"), "limitations": result.get("limitations") or []}


def _statistical_evidence(df, profile, formula, label, base_filters, time_column, period_a, period_b,
                          total_a, total_b, ranked, budget) -> dict:
    """Uncertainty for the change and for its leading contributors.

    Each is tested as a series of time buckets (days, usually) with
    autocorrelation-robust errors, never row by row: thousands of rows from the
    same few days are not thousands of independent observations. The
    contributors' p-values are Holm-adjusted as one family, because looking at
    several segments and reporting the one that moved is how noise gets
    mistaken for a finding. Attaches `statistic` to the ranked rows in place."""
    from app import analysis_ops as ops           # imports this module, so not at the top
    from app.stats import selector as stat

    ctx = ops.AnalysisContext(formula=formula, label=label, base_filters=base_filters, time_column=time_column,
                              period_a=list(period_a), period_b=list(period_b), profile=profile,
                              total_a=total_a, total_b=total_b)

    def level_shift(filters: list[dict] | None, what: str) -> dict:
        if not budget.spend():
            return _compact_stat(None)
        try:
            series = ops.bucket_series(df, ctx, formula, filters)
            return _compact_stat(stat.level_shift(series["a"], series["b"], what=what, bucket=series["bucket"],
                                                  context=series["context"]), series["bucket"])
        except (me.FormulaError, ops.OpError, ValueError) as e:
            return {"status": "unavailable", "reason": str(e)}

    try:
        coverage = ops._period_coverage(df, ctx)["facts"]
        comparability = {"comparable": not coverage["incomplete"], "issues": coverage["issues"],
                         "span_days_a": coverage["span_days_a"], "span_days_b": coverage["span_days_b"],
                         "days_with_data_a": coverage["days_with_data_a"], "days_with_data_b": coverage["days_with_data_b"]}
        # The shared check looks for a later period that is thinner or cut short;
        # an earlier period the data only partly covers distorts a comparison just as much.
        issues = comparability["issues"]
        density_a = coverage["days_with_data_a"] / max(coverage["span_days_a"], 1.0)
        density_b = coverage["days_with_data_b"] / max(coverage["span_days_b"], 1.0)
        if density_b > 0 and density_a < 0.8 * density_b:
            first = pd.to_datetime(ops.period_frame(df, ctx, "a")[time_column], errors="coerce").min()
            late = pd.notna(first) and (first - pd.Timestamp(period_a[0])).total_seconds() / 86400 > 0.15 * coverage["span_days_a"]
            issues.append(f"the earlier period's data starts on {first.date()}, after the period begins on "
                          f"{pd.Timestamp(period_a[0]).date()}" if late else
                          f"the earlier period has data on {coverage['days_with_data_a']} days against "
                          f"{coverage['days_with_data_b']} in the later one")
        comparability["comparable"] = not issues
    except (me.FormulaError, ValueError, KeyError) as e:
        comparability = {"comparable": None, "issues": [f"period coverage could not be checked ({e})"]}

    overall = level_shift(None, label)
    pre_trend = {"status": "not_run", "reason": "outside this investigation's computation budget"}
    if budget.spend():
        try:
            series = ops.bucket_series(df, ctx, formula, None)
            result = stat.pre_trend(series["a"], series["b"], what=label, bucket=series["bucket"], context=series["context"])
            if result.get("status") == "ok":
                details = result.get("details") or {}
                observed = details.get("observed_change_per_bucket") or 0.0
                pre_trend = {"status": "ok", "bucket": series["bucket"], "slope": result.get("estimate"),
                             "p_value": (result.get("test") or {}).get("p_value"), "significant": bool(result.get("significant")),
                             "share": details.get("share_explained_by_trend"),
                             "same_direction": (result.get("estimate") or 0.0) * observed > 0}
            else:
                pre_trend = {"status": "unavailable", "reason": result.get("reason") or "the trend could not be estimated"}
        except (me.FormulaError, ops.OpError, ValueError) as e:
            pre_trend = {"status": "unavailable", "reason": str(e)}
    tested = ranked[:STAT_CONTRIBUTORS]
    for r in tested:
        r["statistic"] = level_shift([{"column": r["dimension"], "op": "==", "value": r["segment"]}],
                                     f"{label} for {r['dimension']} = {r['segment']}")
    for r in ranked[STAT_CONTRIBUTORS:]:
        r["statistic"] = {"status": "not_run", "reason": f"only the {STAT_CONTRIBUTORS} leading contributors are tested"}
    ok = [r["statistic"] for r in tested if r["statistic"]["status"] == "ok" and r["statistic"]["p_value"] is not None]
    for s, adjusted in zip(ok, stat.adjust_family([s["p_value"] for s in ok], "holm") if ok else []):
        s["p_adjusted"] = adjusted
        s["significant"] = adjusted < (s.get("alpha") or 0.05)
    return {"overall": overall, "comparability": comparability, "pre_trend": pre_trend,
            "family": {"method": "Holm", "comparisons": len(ok),
                       "note": "p-values of the contributors are adjusted together, as one family of tests"}}


def _support(r: dict) -> str:
    """How far the measurements back "the change is located here". Never more
    than that: no row here is evidence about what caused the change."""
    share, stat_ = r.get("share_of_change") or 0.0, r.get("statistic") or {}
    if share < 0:
        return "offset"          # it moved against the change: not somewhere the change is located
    tested, significant = stat_.get("status") == "ok", bool(stat_.get("significant"))
    if share >= STRONG_SHARE and r["evidence_quality"] != "low" and tested and significant:
        return "strong"
    if share >= 0.2 and r["evidence_quality"] != "low" and not (tested and not significant):
        return "moderate"
    return "weak"


def _row_limitations(r: dict) -> list[str]:
    out, stat_ = [], r.get("statistic") or {}
    if min(r["rows_a"], r["rows_b"]) < MIN_ROWS_MEDIUM:
        out.append(f"few rows ({r['rows_a']} earlier, {r['rows_b']} later)")
    if r["presence"] != "both":
        out.append(f"present in only one period ({r['presence']}), so its whole value counts as change")
    if r.get("consistent_across_halves") is False:
        out.append("the direction does not hold in both halves of the periods")
    if stat_.get("status") == "ok" and not stat_.get("significant"):
        out.append("its own difference is within its ordinary variation after adjusting for the number of segments tested")
    elif stat_.get("status") in ("unavailable", "not_run"):
        out.append("no statistical test: " + str(stat_.get("reason")))
    return out


def _evidence_table(ranked: list[dict], pvm: dict | None, delta: float, comparability: dict,
                    length: dict | None = None, pre_trend: dict | None = None) -> list[dict]:
    """One row per candidate explanation. `kind` separates what was measured in
    the data ("measured": a location of the change) from what was not and
    cannot be read off it ("hypothesis")."""
    rows = []
    if length is not None and delta:
        rows.append({"candidate": f"Length of the periods ({_fmt(length['days_b'])} days against {_fmt(length['days_a'])})",
                     "kind": "measured", "claim": "a longer or shorter period, with the daily level unchanged",
                     "value_a": None, "value_b": None, "contribution": length["effect"], "share_of_change": length["effect"] / delta,
                     "rows_a": None, "rows_b": None, "statistic": {"status": "note", "text": "exact arithmetic, not a test"},
                     "evidence_quality": "high", "support": "exact",
                     "limitations": ["the shares in the rows below are of the like-for-like change that remains"]})
    if pre_trend is not None:
        rows.append({"candidate": "A trend already under way before the later period", "kind": "measured",
                     "claim": "the earlier period's trend, continued, would produce this share of the change in level",
                     "value_a": None, "value_b": None, "contribution": None, "share_of_change": pre_trend["share"],
                     "rows_a": None, "rows_b": None,
                     "statistic": {"status": "note", "text": f"trend of {_fmt(pre_trend['slope'])} per {pre_trend['bucket']} "
                                                             f"in the earlier period ({_p_text(pre_trend['p_value'])})"},
                     "evidence_quality": "medium", "support": "moderate",
                     "limitations": ["a continued trend is an extrapolation; it says the change may predate the later period, "
                                     "not what set it off"]})
    uncomparable = bool(comparability.get("issues"))
    for r in ranked[:8]:
        offset = (r["share_of_change"] or 0.0) < 0
        support, limits = _support(r), _row_limitations(r)
        if uncomparable and support == "strong":
            support = "moderate"
            limits = limits + ["the two periods are not fully comparable, so the size of this contribution is uncertain"]
        rows.append({"candidate": f"{r['dimension']} = {r['segment']}", "kind": "measured",
                     "claim": "moved against the overall change, offsetting part of it" if offset
                     else "the change is located in this segment",
                     "value_a": r.get("a"), "value_b": r.get("b"), "contribution": r["contribution"],
                     "share_of_change": r["share_of_change"], "rows_a": r["rows_a"], "rows_b": r["rows_b"],
                     "statistic": r.get("statistic"), "evidence_quality": r["evidence_quality"],
                     "support": support, "limitations": limits})
    if pvm is not None and pvm.get("reconciles") and delta:
        for name, key in (("Price effect", "price_effect"), ("Volume effect", "volume_effect")):
            rows.append({"candidate": name, "kind": "measured", "claim": "an accounting split of the change, on " + pvm["basis"],
                         "value_a": None, "value_b": None, "contribution": pvm[key], "share_of_change": pvm[key] / delta,
                         "rows_a": None, "rows_b": None, "statistic": {"status": "note", "text": "an exact identity, not a test"},
                         "evidence_quality": "medium", "support": "moderate" if abs(pvm[key] / delta) >= 0.2 else "weak",
                         "limitations": ["a price or volume effect says how the change is composed, not why prices or volumes moved"]
                         + (["computed on the raw totals: the volume effect includes the difference in the periods' lengths"]
                            if length is not None else [])})
    rows.append({"candidate": "An event not recorded in this dataset (a pricing decision, campaign, outage, competitor)",
                 "kind": "hypothesis", "claim": "not tested", "support": "not testable with this dataset",
                 "limitations": ["nothing in the dataset's columns can confirm or rule this out"]})
    issues = comparability.get("issues") or []
    rows.append({"candidate": "A change in how the data was recorded (late, missing or duplicated rows)",
                 "kind": "hypothesis", "claim": "not tested",
                 "support": "not ruled out: the periods are not fully comparable" if issues else "not tested here",
                 "limitations": issues or ["the two periods cover comparable spans, but row-level recording was not audited"]})
    return rows


def _follow_ups(ranked, drilldowns, dims, skipped, comparability, overall, period_b, low_only,
                pre_trend: dict | None = None, lead: dict | None = None) -> list[str]:
    out = []
    pre_trend = pre_trend or {"status": "not_run"}
    if comparability.get("issues"):
        out.append("Repeat the comparison on two periods the data covers completely; the result above mixes a real "
                   "change with the difference in coverage.")
    if overall["status"] != "ok":
        out.append("Use longer periods or a finer date column so the metric forms a series long enough to test "
                   f"({overall.get('reason')}).")
    lead = lead or (ranked[0] if ranked else None)
    if lead is not None:
        where = f"{lead['dimension']} = {lead['segment']}"
        other = next((d for d in dims if d != lead["dimension"]), None)
        if other and not any(d["within"]["segment"] == lead["segment"] for d in drilldowns):
            out.append(f"Break {where} down by {other} to see whether the change sits in part of it.")
        out.append(f"Find out what changed for {where} around {str(period_b[0])[:10]} (prices, promotions, stock, outages, "
                   "tracking): the dataset shows where the change is, not what brought it about.")
        if pre_trend["status"] == "ok" and pre_trend["significant"] and pre_trend["same_direction"]:
            out.append(f"Extend the earlier period further back to find when the trend began; the metric was already moving "
                       f"this way before {str(period_b[0])[:10]}.")
        elif pre_trend["status"] != "ok":
            out.append(f"Check whether {where} was already moving before {str(period_b[0])[:10]} (not tested here: "
                       f"{pre_trend.get('reason')}); a trend that predates the period is not explained by anything in it.")
    if low_only:
        out.append("Gather more rows per segment before acting: the leading contributors rest on too little data.")
    if skipped:
        out.append("Examine the dimensions left out of this run: " + ", ".join(skipped[:6]) + ".")
    if not dims:
        out.append("Add a categorical column (region, product, channel) to the dataset so the change can be located.")
    return out


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
        "change_evidence": {"overall": {"status": "not_run", "reason": reason},
                            "pre_trend": {"status": "not_run", "reason": reason}, "family": None},
        "comparability": {"comparable": None, "issues": []}, "evidence_table": [], "follow_ups": [],
    }
