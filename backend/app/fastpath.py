"""Fast path: answer a simple aggregation deterministically, with no model call
and no generated code.

"Total revenue by region" does not need a planner, an executor, a critic and a
dashboard writer. If -- and only if -- every content word of the question is
accounted for by an aggregation word, a column name or an approved metric, the
question is answered by `metric_engine`, recomputed by its independent
reference implementation, and published with evidence.

The parser is deliberately conservative. Any word it cannot account for
("last month", "excluding", "trend", "why") means the question is not simple,
and it goes to the full pipeline instead. A fast answer to a question the
parser only half understood would be worse than a slow correct one.
"""
from __future__ import annotations

import re

from app import metric_engine as me

_AGG_WORDS = [
    ("count_distinct", r"(?:how many|number of|count of|count)\s+(?:distinct|unique|different)"),
    ("count_distinct", r"(?:distinct|unique)\s+(?:count|number) of"),
    ("count", r"how many|number of|count of|count"),
    ("sum", r"total|sum of|sum|overall"),
    ("avg", r"average|mean|avg"),
    ("median", r"median"),
    ("max", r"maximum|max|highest|largest|biggest"),
    ("min", r"minimum|min|lowest|smallest"),
]
_GROUP_WORDS = r"(?:by|per|for each|for every|across|grouped by|broken down by|split by)"
_STOP = set("""a an and are as at be by can could do does each every for from give how i in is it list me of on
our please show tell than that the their there this to us was we were what which with you your all data dataset
value values amount figure number numbers rows row records record entries entry there's whats what's is""".split())
_ROW_WORDS = r"(?:rows?|records?|entries|entry|observations?|lines?)"
MAX_GROUPS_FOR_FAST = 60


def _phrase(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", name.strip().lower()))


def _find(phrase: str, text: str) -> re.Match | None:
    variants = {phrase}
    if not phrase.endswith("s"):
        variants.add(phrase + "s")
    elif len(phrase) > 3:
        variants.add(phrase[:-1])
    pattern = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return re.search(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", text)


def _consume(text: str, match: re.Match) -> str:
    return text[:match.start()] + " " * (match.end() - match.start()) + text[match.end():]


def parse(question: str, profile: dict, resolved_metrics: list[dict] | None = None) -> dict | None:
    """Returns one of
        {"kind": "spec", "formula", "filters", "group_by", "label", "explanation"}
        {"kind": "ambiguous", "reason", "options": [{"label", "question"}]}
        None  -- not a simple question; use the full pipeline
    """
    text = re.sub(r"[^a-z0-9%\s_\-]", " ", (question or "").lower())
    text = re.sub(r"\s+", " ", _phrase(text)).strip()
    if not text or len(text) > 200:
        return None
    columns = profile.get("columns", [])
    numeric = {c["name"]: _phrase(c["name"]) for c in columns if c.get("kind") == "numeric"}
    categorical = {c["name"]: _phrase(c["name"]) for c in columns if c.get("kind") == "categorical"}
    every = {c["name"]: _phrase(c["name"]) for c in columns}
    work = text

    # 1. group-by: "<by|per|...> <column>"
    group_by: list[str] = []
    for name, phrase in sorted(every.items(), key=lambda kv: -len(kv[1])):
        m = re.search(rf"(?<![a-z0-9]){_GROUP_WORDS}\s+(?:each\s+|every\s+|the\s+)?", work)
        if not m:
            break
        after = _find(phrase, work[m.end():])
        if after is not None and after.start() == 0:
            col = next(c for c in columns if c["name"] == name)
            if col.get("kind") == "numeric" and (col.get("unique_count") or 0) > MAX_GROUPS_FOR_FAST:
                return None
            if (col.get("unique_count") or 0) > MAX_GROUPS_FOR_FAST:
                return None
            group_by.append(name)
            work = work[:m.start()] + " " * (m.end() + after.end() - m.start()) + work[m.end() + after.end():]
            break

    # 2. an approved metric named in the question
    formula = label = None
    filters: list[dict] = []
    explanation = ""
    unit = ""
    for metric in resolved_metrics or []:
        m = _find(metric["matched_term"], work)
        if m is not None:
            formula, label, filters = metric["formula"], metric["name"], list(metric.get("filters") or [])
            unit = metric.get("unit") or ""
            explanation = f"Approved metric '{metric['name']}' (version {metric['current_version']})"
            work = _consume(work, m)
            for _, pattern in _AGG_WORDS:  # "total net revenue": the aggregation word is part of the phrasing
                am = re.search(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", work)
                if am is not None:
                    work = _consume(work, am)
                    break
            break

    # 3. otherwise: aggregation word + column
    if formula is None:
        agg = None
        for fn, pattern in _AGG_WORDS:
            am = re.search(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", work)
            if am is not None:
                agg, work = fn, _consume(work, am)
                break
        if agg is None:
            return None
        pool = every if agg in ("count", "count_distinct", "min", "max") else numeric
        hits = [(name, _find(phrase, work)) for name, phrase in pool.items() if name not in group_by]
        hits = [(n, m) for n, m in hits if m is not None]
        # prefer the longest column phrase; equal-length rivals are a real ambiguity
        hits.sort(key=lambda h: -(h[1].end() - h[1].start()))
        if hits:
            longest = hits[0][1].end() - hits[0][1].start()
            rivals = [h for h in hits if h[1].end() - h[1].start() == longest]
            distinct_spans = {(h[1].start(), h[1].end()) for h in rivals}
            if len(rivals) > 1 and len(distinct_spans) == 1:
                return _ambiguous(question, agg, [h[0] for h in rivals])
            column, m = hits[0]
            work = _consume(work, m)
            if agg == "count" and column in categorical and not group_by:
                # "how many regions" means distinct regions, not non-empty cells
                agg = "count_distinct"
            formula = f'{agg}("{column}")'
            label = {"sum": "Total", "avg": "Average", "median": "Median", "max": "Maximum", "min": "Minimum",
                     "count": "Count of", "count_distinct": "Distinct"}[agg] + f" {column}"
        elif agg == "count":
            rm = re.search(rf"(?<![a-z0-9]){_ROW_WORDS}(?![a-z0-9])", work)
            if rm is None and not group_by:
                return None
            if rm is not None:
                work = _consume(work, rm)
            formula, label = "count()", "Number of rows"
        else:
            # an aggregation word with a partial column match: offer the candidates
            partial = _partial_columns(work, pool)
            if len(partial) > 1:
                return _ambiguous(question, agg, partial)
            return None
        explanation = f"Parsed as {formula}"

    # 4. everything else must be filler -- otherwise this is not a simple question
    leftovers = [w for w in work.split() if w not in _STOP]
    if leftovers:
        return None
    try:
        parsed = me.parse_formula(formula)
    except me.FormulaError:
        return None
    if me.validate_against_profile(parsed, profile):
        return None
    if group_by:
        label += f" by {group_by[0]}"
        explanation += f", grouped by {group_by[0]}"
    return {"kind": "spec", "formula": formula, "filters": filters, "group_by": group_by, "label": label,
            "explanation": explanation, "unit": unit}


def _partial_columns(text: str, pool: dict[str, str]) -> list[str]:
    words = {w for w in text.split() if w not in _STOP and len(w) > 2}
    return [name for name, phrase in pool.items() if words & set(phrase.split())]


def _ambiguous(question: str, agg: str, columns: list[str]) -> dict:
    word = {"sum": "total", "avg": "average", "count_distinct": "number of distinct", "count": "count of"}.get(agg, agg)
    return {
        "kind": "ambiguous",
        "reason": "This question matches more than one column: " + ", ".join(columns[:6]) + ".",
        "options": [{"label": c, "question": f"What is the {word} {c}?"} for c in columns[:6]],
    }


def format_value(value: float | None, unit: str = "") -> str:
    if value is None:
        return "n/a"
    text = f"{value:,.0f}" if float(value).is_integer() and abs(value) < 1e15 else f"{value:,.2f}"
    if unit == "%":
        return f"{text}%"
    return f"{unit}{text}" if unit in ("$", "€", "£") else text


def compute(csv_path: str, profile: dict, spec: dict) -> dict:
    """Evaluate the spec with the engine and, independently, with the reference
    implementation. Returns everything the publisher and the verifier need."""
    df = me.load_frame(csv_path)
    engine = me.evaluate(df, spec["formula"], spec["filters"], spec["group_by"] or None)
    try:
        reference = me.reference_evaluate(csv_path, spec["formula"], spec["filters"], spec["group_by"] or None)
        reference_error = None
    except (me.ReferenceNotApplicable, me.FormulaError, OSError, UnicodeDecodeError) as e:
        reference, reference_error = None, f"{type(e).__name__}: {e}"
    used = sorted({*me.parse_formula(spec["formula"]).columns, *spec["group_by"],
                   *(f["column"] for f in spec["filters"])})
    scoped = me.apply_filters(df, spec["filters"])
    slice_cols = used or list(df.columns[:6])
    rows = scoped[slice_cols].head(50).astype(object).where(scoped[slice_cols].head(50).notna(), None)
    data_slice = {"columns": slice_cols,
                  "rows": [[_json(v) for v in r] for r in rows.itertuples(index=False, name=None)]}
    return {"engine": engine, "reference": reference, "reference_error": reference_error, "data_slice": data_slice}


def _json(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def equivalent_code(spec: dict) -> str:
    """A readable pandas equivalent, shown in the Inspect panel. It documents
    the computation; the engine, not this text, produced the number."""
    lines = ["# Computed by the deterministic metric engine (no generated code was run).",
             f"# formula : {spec['formula']}"]
    if spec["filters"]:
        lines.append("# filters : " + "; ".join(f"{f['column']} {f['op']} {f['value']!r}" for f in spec["filters"]))
    if spec["group_by"]:
        lines.append(f"# group by: {', '.join(spec['group_by'])}")
    lines += ["import pandas as pd", 'df = pd.read_csv("/workspace/data/input.csv")',
              "# ... apply the filters above, then aggregate as the formula states"]
    return "\n".join(lines)
