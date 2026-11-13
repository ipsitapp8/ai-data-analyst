"""Comparison and association: "is there a difference?", answered by the
statistics engine rather than by generated code.

    question --> plan --> validate --> stats.selector --> report

The plan is a small structured object naming the two variables and how they
are to be compared. It comes from the deterministic planner below or, when
enabled, from a model; either way it is only a proposal. `validate` rejects
anything the dataset does not support before a single number is computed, and
the method is chosen by `stats.selector` from the data, never by the planner.
A model that proposes plans can be replaced without touching anything else in
this file: it has one job, and its output is checked.

No statistic here is written by a model, and none is produced by generated
code. The result states the estimate, its interval, an effect size, the sample
sizes, the method used, the methods rejected and why, the assumptions checked
and what the result does not show. An association is reported as an
association: a question that asks about cause is answered with what was
measured and with what evidence a causal answer would need.
"""
from __future__ import annotations

import csv
import math
import re
from typing import Any

import pandas as pd

from app import config, fastpath
from app import metric_engine as me
from app.stats import selector as stat
from app.stats.result import fmt, p_text

PLAN_VERSION = "1"
DESIGNS = ("independent", "paired")
ESTIMANDS = ("mean", "typical")
MIN_GROUP_ROWS = 2

PLAN_TOOL_NAME = "submit_comparison_plan"
PLAN_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "description": "Column whose values are compared or related. Exact column name."},
        "explanatory": {"type": "string", "description": "Column that defines the groups, or the second variable. Exact column name."},
        "levels": {"type": "array", "items": {"type": "string"}, "description": "Exactly two values of the grouping column to "
                   "compare, when the question names two. Omit to compare every group."},
        "outcome_level": {"type": "string", "description": "For a yes/no outcome: the value whose share is compared."},
        "design": {"type": "string", "enum": list(DESIGNS), "description": "paired only when the same units appear in both groups."},
        "unit": {"type": "string", "description": "For a paired design: the column identifying the unit measured twice."},
        "estimand": {"type": "string", "enum": list(ESTIMANDS), "description": "mean, or typical for a comparison of typical values."},
        "filters": {"type": "array", "items": {"type": "object", "properties": {
            "column": {"type": "string"}, "op": {"type": "string"}, "value": {"type": "string"}},
            "required": ["column", "op", "value"]}},
        "method": {"type": "string", "description": "Optional: a test you would suggest. The system chooses from the data "
                   "and records whether it agreed."},
        "claims_cause": {"type": "boolean", "description": "True when the question asks whether one variable causes, "
                         "affects or improves the other."},
        "assumptions": {"type": "array", "items": {"type": "string"}, "description": "Anything assumed in reading the question."},
    },
    "required": ["outcome", "explanatory"],
}
PLAN_SYSTEM = """You turn a question about a dataset into a comparison plan. You do not compute anything.

Name the two columns exactly as they appear in the profile. Say which two groups are compared only when the
question names them. Set claims_cause when the question asks about cause, effect, impact or improvement. If the
question cannot be answered by comparing or relating two columns of this dataset, do not invent a plan."""

EVIDENCE_FOR_CAUSE = [
    "Assignment that does not depend on the outcome: a randomised experiment, or a rule that decided who got which value.",
    "The order of events: the explanatory variable recorded before the outcome, for the same units.",
    "The other differences between the groups, measured, so that they can be held fixed.",
]


class PlanError(ValueError):
    """The plan cannot be carried out on this dataset. `code` says why."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------ profile --

def _columns(profile: dict) -> dict[str, dict]:
    return {c["name"]: c for c in profile.get("columns", [])}


def _role(column: dict) -> str:
    """How a column can take part: a measurement, a grouping, or neither."""
    kind, unique = column.get("kind"), column.get("unique_count") or 0
    if kind == "numeric":
        return "binary" if unique == 2 else "numeric"
    if kind == "categorical":
        return "categorical"
    return "unsupported"


# ----------------------------------------------------------------- validate --

def validate(plan: Any, profile: dict) -> dict:
    """The plan in canonical form, or PlanError. Runs nothing: every check
    here reads the plan and the dataset profile only."""
    if not isinstance(plan, dict):
        raise PlanError("a plan must be an object", "invalid_plan")
    unknown = sorted(set(plan) - set(PLAN_TOOL_SCHEMA["properties"]) - {"source"})
    if unknown:
        raise PlanError(f"unknown plan field(s): {', '.join(unknown)}", "invalid_plan")
    columns = _columns(profile)

    def column(name: Any, what: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise PlanError(f"the plan names no {what} column", "invalid_plan")
        if name not in columns:
            raise PlanError(f"'{name}' is not a column of this dataset", "unknown_column")
        return name

    outcome, explanatory = column(plan.get("outcome"), "outcome"), column(plan.get("explanatory"), "explanatory")
    if outcome == explanatory:
        raise PlanError("a comparison needs two different columns", "same_column")
    roles = {outcome: _role(columns[outcome]), explanatory: _role(columns[explanatory])}
    for name, role in roles.items():
        if role == "unsupported":
            raise PlanError(f"'{name}' is a {columns[name].get('kind') or 'text'} column; a comparison needs a numeric or "
                            "categorical one", "incompatible_type")

    swapped = False
    if roles[outcome] == "numeric" and roles[explanatory] == "numeric":
        family = "correlation"
    elif roles[outcome] == "numeric":
        family = "groups"
    elif roles[explanatory] == "numeric":
        # "is region related to revenue?": the same comparison, read the other way round.
        outcome, explanatory, family, swapped = explanatory, outcome, "groups", True
    else:
        family = "table"

    design = plan.get("design") or "independent"
    if design not in DESIGNS:
        raise PlanError(f"'{design}' is not a supported design (independent, paired)", "unsupported_design")
    estimand = plan.get("estimand") or "mean"
    if estimand not in ESTIMANDS:
        raise PlanError(f"'{estimand}' is not a supported estimand (mean, typical)", "unsupported_estimand")
    method = plan.get("method")
    if method is not None and stat.ALIASES.get(str(method).strip().lower().replace(" ", "_")) is None:
        raise PlanError(f"'{method}' is not a method this system supports", "unsupported_method")

    levels = plan.get("levels")
    if levels is not None:
        if family == "correlation":
            raise PlanError("levels apply to a grouping column, and both columns here are numeric", "invalid_plan")
        if not isinstance(levels, list) or len(levels) != 2 or len({str(v) for v in levels}) != 2:
            raise PlanError("levels must be exactly two different values of the grouping column", "invalid_levels")
        levels = [str(v) for v in levels]
    groups = columns[explanatory].get("unique_count") or 0
    if family != "correlation" and levels is None and groups > config.STAT_MAX_CONTROL_LEVELS:
        raise PlanError(f"'{explanatory}' has {groups} groups, too many to compare at once; name two of them",
                        "too_many_groups")
    if family == "table" and (columns[outcome].get("unique_count") or 0) > config.STAT_MAX_CONTROL_LEVELS:
        raise PlanError(f"'{outcome}' has too many distinct values to tabulate", "too_many_groups")

    unit = plan.get("unit")
    if design == "paired":
        if family != "groups":
            raise PlanError("a paired comparison needs a numeric outcome measured in two groups", "unsupported_design")
        unit = column(unit, "unit")
        if unit in (outcome, explanatory):
            raise PlanError("the unit column must differ from the two compared columns", "invalid_plan")
        if levels is None and groups != 2:
            raise PlanError("a paired comparison needs exactly two groups; name them", "invalid_levels")
    elif unit is not None:
        raise PlanError("a unit column applies to a paired design only", "invalid_plan")

    outcome_level = plan.get("outcome_level")
    if outcome_level is not None and family != "table":
        raise PlanError("outcome_level applies to a yes/no outcome only", "invalid_plan")
    try:
        filters = me.normalize_filters(plan.get("filters"), list(columns))
    except me.FormulaError as e:
        raise PlanError(str(e), "invalid_filter") from e

    return {"plan_version": PLAN_VERSION, "family": family, "outcome": outcome, "explanatory": explanatory,
            "levels": levels, "outcome_level": None if outcome_level is None else str(outcome_level),
            "design": design, "unit": unit, "estimand": estimand, "filters": filters, "method": method,
            "claims_cause": bool(plan.get("claims_cause")), "roles_swapped": swapped,
            "assumptions": [str(a)[:300] for a in (plan.get("assumptions") or [])][:8],
            "source": plan.get("source") or "unspecified"}


# ------------------------------------------------------ deterministic planner --

_INTENT = re.compile(
    r"\b(?:differ\w*|difference|compar\w*|versus|vs|higher|lower|greater|larger|smaller|better|worse|more|fewer|less|"
    r"than|relat\w+|relationship|correlat\w*|associat\w*|depend\w*|linked|link|affect\w*|impact\w*|influenc\w*|effect|"
    r"caus\w*|improv\w*|increas\w*|reduc\w*|significan\w*)\b")
_CAUSAL = re.compile(
    r"\b(?:caus\w*|affect\w*|impact\w*|influenc\w*|effect of|effects of|leads? to|because of|due to|results? in|"
    r"improv\w*|drives?|driven|boosts?|hurts?|makes?)\b")
# Words this planner cannot honour. A question that has one is left to another route
# rather than answered as if the constraint were not there.
_OUT_OF_SCOPE = re.compile(
    r"\b(?:over time|trend\w*|forecast\w*|predict\w*|month\w*|year\w*|week\w*|quarter\w*|daily|since|before|after|last|"
    r"previous|excluding|except|without|only|top|bottom|outlier\w*|anomal\w*|why)\b|\d")
# A comparison is asked as a yes/no question ("is ... higher than ...?") or in so many words
# ("compare", "related to"). "Different" alone is not enough: "the maximum for different
# cylinders" and "whose address is different" ask for a listing.
_YES_NO = re.compile(r"^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b")
_EXPLICIT = re.compile(r"\b(?:differs?|differed|compar\w+|correlat\w*|associat\w*|relationship|related to|relates to|"
                       r"depend\w*|linked to)\b")
# Requests for a listing, a count or an extreme are descriptive whatever else they say.
_DESCRIPTIVE = re.compile(r"\b(?:list|show|name|which|who|whose|whom|what|maximum|minimum|max|min|total|sum|count|how many|"
                          r"number of|highest|lowest|most|least|top|bottom)\b")
_GROUP_WORD = re.compile(r"\b(?:by|across|between|among|per|for each|depend\w* on|related to|with)\b")


def asks_for_comparison(question: str) -> bool:
    """Whether the wording asks to compare or relate things at all."""
    return bool(_INTENT.search(re.sub(r"[_\-]+", " ", (question or "").lower())))


def parse(question: str, profile: dict) -> dict | None:
    """A plan read from the question's own words, with no model.

    Returns {"kind": "plan", "plan", "explanation"}, {"kind": "ambiguous",
    "reason", "options"} when a word could mean more than one column, or None
    when the question is not a comparison this planner can read. It is
    deliberately narrow: two columns named outright (or one column and two of
    another's values), a word that asks for a comparison, and nothing it would
    have to ignore."""
    text = re.sub(r"\s+", " ", re.sub(r"[^\w\s.%]|[_\-]+", " ", (question or "").lower())).strip().rstrip(".")
    if not text or not _INTENT.search(text) or not (_YES_NO.search(text) or _EXPLICIT.search(text)):
        return None
    columns = _columns(profile)
    usable = {name: c for name, c in columns.items() if _role(c) != "unsupported"}

    found: list[tuple[int, str]] = []
    remaining = text
    for name in sorted(usable, key=lambda n: -len(fastpath._phrase(n))):
        match = fastpath._find(fastpath._phrase(name), remaining)
        if match:
            found.append((match.start(), name))
            remaining = fastpath._consume(remaining, match)
    found.sort()
    named = [name for _, name in found]
    # Judged with the column names taken out: a column called list_price or year is not a
    # request for a listing or a period.
    if _OUT_OF_SCOPE.search(remaining) or _DESCRIPTIVE.search(remaining):
        return None

    # Two values of one categorical column ("online vs retail") name the groups.
    levels: dict[str, list[tuple[int, str]]] = {}
    for name, c in usable.items():
        if _role(c) != "categorical":
            continue
        for top in c.get("top_values") or []:
            value = str(top.get("value", ""))
            match = fastpath._find(fastpath._phrase(value), remaining) if len(value) >= 2 else None
            if match:
                levels.setdefault(name, []).append((match.start(), value))
    by_levels = {name: [v for _, v in sorted(hits)] for name, hits in levels.items() if len(hits) == 2}

    if len(named) < 2 and not by_levels:
        pool = {name: fastpath._phrase(name) for name in usable if name not in named}
        partial = fastpath._partial_columns(remaining, pool)
        if len(partial) >= 2 and len(named) == 1:
            return {"kind": "ambiguous",
                    "reason": f"This could compare {named[0]} with more than one column: {', '.join(partial)}.",
                    "options": [{"label": f"{named[0]} and {p}", "question": f"Is {named[0]} related to {p}?"} for p in partial[:6]]}
        return None
    if len(named) > 2 or len(by_levels) > 1 or (by_levels and len(named) > 2):
        return None

    plan: dict[str, Any] = {"source": "deterministic", "claims_cause": bool(_CAUSAL.search(text))}
    if by_levels:
        group, values = next(iter(by_levels.items()))
        others = [n for n in named if n != group]
        if len(others) != 1:
            return None
        plan.update(outcome=others[0], explanatory=group, levels=values)
    else:
        first, second = named
        roles = (_role(usable[first]), _role(usable[second]))
        if roles[0] == "numeric" and roles[1] != "numeric":
            plan.update(outcome=first, explanatory=second)
        elif roles[1] == "numeric" and roles[0] != "numeric":
            plan.update(outcome=second, explanatory=first)
        else:
            # Two measurements, or two groupings. "Does X affect Y" names the explanatory one first;
            # otherwise it is the one after "by / across / related to".
            between = text[found[0][0]:found[1][0]]
            marker = _GROUP_WORD.search(text, found[0][0])
            if _CAUSAL.search(between):
                explanatory = first
            else:
                explanatory = second if marker is None or marker.start() < found[1][0] else first
            plan.update(outcome=first if explanatory == second else second, explanatory=explanatory)
    if re.search(r"\b(?:typical\w*|median)\b", text):
        plan["estimand"] = "typical"
    try:
        clean = validate(plan, profile)
    except PlanError:
        return None
    how = {"groups": f"compare {clean['outcome']} between the groups of {clean['explanatory']}",
           "correlation": f"measure the association between {clean['explanatory']} and {clean['outcome']}",
           "table": f"compare how {clean['outcome']} is distributed across {clean['explanatory']}"}[clean["family"]]
    return {"kind": "plan", "plan": clean, "explanation": how}


def propose_with_model(question: str, profile: dict) -> dict | None:
    """A plan proposed by the configured model, validated like any other, or
    None. One call; nothing it returns is trusted beyond what `validate`
    accepts. Off unless COMPARISON_MODEL_PLANNER is set."""
    from app import runtime
    from app.agents import llm_client, prompts

    try:
        out = llm_client.call_tool(
            system=PLAN_SYSTEM,
            user_content=f"Dataset profile:\n{prompts.profile_block(profile)}\n\nQuestion: {question!r}\n",
            tool_name=PLAN_TOOL_NAME, tool_schema=PLAN_TOOL_SCHEMA,
            tool_description="Propose how to compare or relate two columns of this dataset.", max_tokens=1024)
    except runtime.RunAborted as e:
        if not isinstance(e, runtime.ProviderUnavailable):
            raise
        return None
    except Exception:  # noqa: BLE001 - a planner that fails leaves the question to the next route
        return None
    try:
        return validate({**out, "source": "model"}, profile)
    except PlanError as e:
        runtime.record_event("comparison_plan", "rejected", {"code": e.code, "reason": str(e)[:300]})
        return None


# ------------------------------------------------------------------ execute --

def _labels(series: pd.Series) -> pd.Series:
    """Group labels as text, so 1 and 1.0 and "1" name the same group."""
    def label(v):
        if pd.isna(v):
            return None
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return series.map(label)


def _prepare(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, dict]:
    """Rows the plan applies to, with what was set aside and why. Raises
    PlanError for a plan the data itself contradicts (a level that does not
    exist, a pairing that is not one-to-one)."""
    try:
        scoped = me.apply_filters(df, plan["filters"])
    except me.FormulaError as e:
        raise PlanError(str(e), "invalid_filter") from e
    checks = {"rows_in_dataset": int(len(df)), "rows_after_filters": int(len(scoped)),
              "duplicate_rows": int(scoped.duplicated().sum())}
    if scoped.empty:
        raise PlanError("no rows are left after the plan's filters", "no_rows")
    if plan["family"] != "correlation":
        scoped = scoped.assign(**{"__group": _labels(scoped[plan["explanatory"]])})
        checks["rows_without_a_group"] = int(scoped["__group"].isna().sum())
        scoped = scoped[scoped["__group"].notna()]
        present = sorted(scoped["__group"].unique())
        if plan["levels"]:
            missing = [v for v in plan["levels"] if v not in present]
            if missing:
                raise PlanError(f"'{plan['explanatory']}' has no value {', '.join(repr(m) for m in missing)}. Its values "
                                f"include: {', '.join(present[:8])}", "level_not_found")
            scoped = scoped[scoped["__group"].isin(plan["levels"])]
        checks["groups"] = plan["levels"] or present
    return scoped, checks


def run(df: pd.DataFrame, plan: dict, seed: int | None = None) -> dict:
    """Carry out a validated plan. Returns the report; raises PlanError before
    computing anything when the data contradicts the plan."""
    data, checks = _prepare(df, plan)
    outcome, explanatory, family = plan["outcome"], plan["explanatory"], plan["family"]
    groups: list[dict] = []

    if family == "correlation":
        result = stat.correlation(data[explanatory], data[outcome], x_name=explanatory, y_name=outcome,
                                  estimand="monotonic" if plan["estimand"] == "typical" else "linear", suggested=plan["method"])
    elif family == "groups":
        values = {level: pd.to_numeric(g[outcome], errors="coerce") for level, g in data.groupby("__group", sort=True)}
        order = plan["levels"] or sorted(values)
        if plan["design"] == "paired":
            result, checks["pairing"] = _paired(data, plan, order, seed)
        elif len(order) == 2:
            a, b = order
            result = stat.two_groups(values[a], values[b], what=outcome, units=outcome, labels=(a, b),
                                     estimand=plan["estimand"], suggested=plan["method"], seed=seed)
        else:
            result = stat.k_groups({k: values[k] for k in order}, what=outcome, group_name=explanatory, units=outcome,
                                   suggested=plan["method"], seed=seed)
        groups = [{"level": k, "n": d["n"], "mean": d["mean"], "median": d["median"], "sd": d["sd"]}
                  for k, d in (result.get("descriptives") or {}).items() if isinstance(d, dict) and "n" in d]
    else:
        result, groups = _table(data, plan, seed)

    return {"plan": plan, "family": family, "status": result["status"], "statistic": result, "groups": groups,
            "data_checks": checks, "causal": _causal(plan), "limitations": _limitations(result, checks, plan),
            # "p < 0.001" is how a smaller p-value is written, so the threshold is part of the result.
            "p_reported_floor": 0.001}


def _paired(data: pd.DataFrame, plan: dict, order: list[str], seed: int | None) -> tuple[dict, dict]:
    unit, outcome = plan["unit"], plan["outcome"]
    a, b = order
    rows = data[[unit, "__group", outcome]].dropna(subset=[unit])
    if rows.duplicated([unit, "__group"]).any():
        repeated = int(rows.duplicated([unit, "__group"], keep=False).groupby(rows[unit]).any().sum())
        raise PlanError(f"{repeated} value(s) of '{unit}' appear more than once in a group, so the rows cannot be paired "
                        "one to one", "invalid_pairing")
    wide = rows.pivot(index=unit, columns="__group", values=outcome)
    both = wide.dropna(subset=[a, b]) if a in wide and b in wide else wide.iloc[0:0]
    pairing = {"unit": unit, "units_in_both_groups": int(len(both)), "units_in_one_group_only": int(len(wide) - len(both))}
    result = stat.paired(pd.to_numeric(both[a], errors="coerce") if len(both) else [],
                         pd.to_numeric(both[b], errors="coerce") if len(both) else [],
                         what=f"{outcome} ({b} minus {a})", unit_name=unit, units=outcome, estimand=plan["estimand"],
                         suggested=plan["method"], seed=seed, unmatched=pairing["units_in_one_group_only"])
    return result, pairing


def _table(data: pd.DataFrame, plan: dict, seed: int | None) -> tuple[dict, list[dict]]:
    outcome, explanatory = plan["outcome"], plan["explanatory"]
    data = data.assign(**{"__outcome": _labels(data[outcome])})
    data = data[data["__outcome"].notna()]
    outcome_values, group_values = sorted(data["__outcome"].unique()), plan["levels"] or sorted(data["__group"].unique())
    if plan["outcome_level"] is not None and plan["outcome_level"] not in outcome_values:
        raise PlanError(f"'{outcome}' has no value '{plan['outcome_level']}'", "level_not_found")
    if len(outcome_values) == 2 and len(group_values) == 2:
        positive = plan["outcome_level"] or outcome_values[-1]
        a, b = group_values
        counts = {g: (int(((data["__group"] == g) & (data["__outcome"] == positive)).sum()), int((data["__group"] == g).sum()))
                  for g in (a, b)}
        result = stat.proportions(counts[a][0], counts[a][1], counts[b][0], counts[b][1],
                                  what=f"share of {outcome} = {positive}", labels=(a, b), suggested=plan["method"])
        groups = [{"level": g, "n": n, "matching": x, "share": x / n if n else None} for g, (x, n) in counts.items()]
        return result, groups
    result = stat.contingency(data["__group"], data["__outcome"], row_name=explanatory, col_name=outcome,
                              suggested=plan["method"], seed=seed)
    sizes = data["__group"].value_counts()
    return result, [{"level": g, "n": int(sizes.get(g, 0))} for g in group_values]


def _causal(plan: dict) -> dict:
    return {"asked": plan["claims_cause"], "status": "not_established",
            "statement": "This measures an association in observational data. It does not show that "
                         f"{plan['explanatory']} changes {plan['outcome']}: groups that differ in one respect usually "
                         "differ in others, and nothing here holds those fixed.",
            "evidence_needed": EVIDENCE_FOR_CAUSE if plan["claims_cause"] else []}


def _limitations(result: dict, checks: dict, plan: dict) -> list[str]:
    out = list(result.get("limitations") or [])
    if checks["duplicate_rows"]:
        out.append(f"{checks['duplicate_rows']:,} of the rows used are exact duplicates of another row; if they are repeated "
                   "records rather than repeated events, the sample is smaller than it looks and the interval too narrow.")
    if checks.get("rows_without_a_group"):
        out.append(f"{checks['rows_without_a_group']:,} row(s) have no value for {plan['explanatory']} and were left out.")
    if plan["roles_swapped"]:
        out.append(f"The question was read as comparing {plan['outcome']} across the groups of {plan['explanatory']}.")
    comparisons = result.get("comparisons") or []
    if comparisons:
        out.append(f"{len(comparisons)} pairwise comparisons were made; their p-values are Holm-adjusted together.")
    return out


# ---------------------------------------------------------------- reference --

def _numbers(csv_path: str, plan: dict, names: list[str]) -> list[dict]:
    """Rows as the reference sees them: csv + plain Python, sharing no code
    with pandas or the statistics engine beyond the filter rule."""
    out = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if all(me._ref_match(row, flt) for flt in plan["filters"]):
                out.append({n: row.get(n) for n in names})
    return out


def _label(raw: Any) -> str | None:
    text = "" if raw is None else str(raw).strip()
    if text == "" or text.lower() in ("nan", "na", "null", "none", "n/a"):
        return None
    number = me._ref_number(text)
    return str(int(number)) if number is not None and float(number).is_integer() else text


def reference(csv_path: str, plan: dict) -> dict:
    """The headline figures recomputed independently: group sizes and means,
    the difference, Pearson's r, shares. Tests and intervals are not
    recomputed here; they are checked against independent references offline
    (backend/evals/build_comparison_fixtures.py)."""
    outcome, explanatory = plan["outcome"], plan["explanatory"]
    rows = _numbers(csv_path, plan, [outcome, explanatory])
    if plan["family"] == "correlation":
        pairs = [(me._ref_number(r[explanatory]), me._ref_number(r[outcome])) for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        n = len(pairs)
        if n < 2:
            return {"n": n, "pearson_r": None}
        mx, my = math.fsum(x for x, _ in pairs) / n, math.fsum(y for _, y in pairs) / n
        sxy = math.fsum((x - mx) * (y - my) for x, y in pairs)
        sxx, syy = math.fsum((x - mx) ** 2 for x, _ in pairs), math.fsum((y - my) ** 2 for _, y in pairs)
        return {"n": n, "pearson_r": sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None}
    if plan["design"] == "paired":
        # Pair by unit in plain Python: the change for every unit that appears in both groups.
        triples = _numbers(csv_path, plan, [outcome, explanatory, plan["unit"]])
        levels = plan["levels"] or sorted({lv for lv in (_label(t[explanatory]) for t in triples) if lv is not None})
        by_unit: dict[str, dict] = {}
        for t in triples:
            unit, level, value = _label(t[plan["unit"]]), _label(t[explanatory]), me._ref_number(t[outcome])
            if unit is not None and level in levels and value is not None:
                by_unit.setdefault(unit, {})[level] = value
        changes = [v[levels[1]] - v[levels[0]] for v in by_unit.values() if len(v) == 2] if len(levels) == 2 else []
        return {"pairs": len(changes), "mean_change": math.fsum(changes) / len(changes) if changes else None}
    keep = set(plan["levels"]) if plan["levels"] else None
    groups: dict[str, list] = {}
    for r in rows:
        level = _label(r[explanatory])
        if level is None or (keep is not None and level not in keep):
            continue
        groups.setdefault(level, []).append(r[outcome])
    if plan["family"] == "groups":
        out = {}
        for level, values in groups.items():
            numbers = [v for v in (me._ref_number(x) for x in values) if v is not None]
            out[level] = {"n": len(numbers), "mean": math.fsum(numbers) / len(numbers) if numbers else None}
        return {"groups": out}
    out = {}
    for level, values in groups.items():
        labels = [v for v in (_label(x) for x in values) if v is not None]
        counts: dict[str, int] = {}
        for v in labels:
            counts[v] = counts.get(v, 0) + 1
        out[level] = {"n": len(labels), "counts": counts}
    return {"groups": out}


# ------------------------------------------------------------------ wording --

def narrative(report: dict) -> str:
    """The published summary. Every number in it is one the engine computed;
    it reports what was measured, then what that does and does not show."""
    plan, r = report["plan"], report["statistic"]
    outcome, explanatory = plan["outcome"], plan["explanatory"]
    withheld = r.get("inference_unavailable")
    if withheld:
        # Measured, but not tested. Said in those words, so that it is read neither as "no difference"
        # nor as "not significant".
        sizes = ", ".join(f"{k}: {v:,}" for k, v in (r["sample"].get("groups") or {}).items()) or f"{r['sample'].get('n', 0):,} pairs"
        if r.get("estimate") is not None:
            measured = f"The measured {r['estimand']} is {fmt(r['estimate'], r.get('units'))} ({sizes})"
        else:
            ranked = sorted((g for g in report["groups"] if g["mean"] is not None), key=lambda g: -g["mean"])
            measured = (f"The mean {outcome} ranges from {fmt(ranked[-1]['mean'])} for {ranked[-1]['level']} to "
                        f"{fmt(ranked[0]['mean'])} for {ranked[0]['level']} ({sizes})")
        text = (f"{measured}. No test or interval is "
                f"reported for it: {withheld['summary']}. This is not a finding of no difference and not a non-significant "
                f"result; whether the difference is larger than chance would produce was not assessed.")
        other = withheld.get("supported_analysis")
        if other:
            # A different question, named as one. Its result is never worded as an answer to the question asked.
            text += (f" A different question the data can answer is {other['estimand']}: {fmt(other['estimate'])} "
                     f"({other['method_label']}, {p_text(other['p_value'])}). "
                     + ("That is unlikely to be chance if " if other["significant"] else "The data does not distinguish that from ")
                     + f"{other['null']}; it is not a test of the difference in means.")
        else:
            text += f" What the data does support: {withheld['supported_instead']}."
        return text + " " + report["causal"]["statement"]
    if r["status"] != "ok":
        text = (f"No statistical comparison of {outcome} by {explanatory} could be made: {r.get('reason')}.")
        return text + (" " + report["causal"]["statement"] if plan["claims_cause"] else "")
    parts = []
    test, interval = r.get("test") or {}, r.get("interval")
    groups = report["groups"]
    if report["family"] == "correlation":
        parts.append(f"The {'rank ' if r['method'] == 'spearman' else ''}correlation between {explanatory} and {outcome} is "
                     f"{fmt(r['estimate'])} across {r['sample']['n']:,} rows")
    elif report["family"] == "groups" and plan["design"] == "paired":
        parts.append(f"Across {r['sample'].get('n', 0):,} paired {plan['unit']} values, {r['estimand']} is {fmt(r['estimate'])}")
    elif report["family"] == "groups" and len(groups) == 2:
        a, b = groups
        word = "median" if plan["estimand"] == "typical" else "mean"
        parts.append(f"The {word} {outcome} is {fmt(a[word])} for {a['level']} ({a['n']:,} rows) and {fmt(b[word])} for "
                     f"{b['level']} ({b['n']:,} rows)")
        if r.get("estimate") is not None and plan["estimand"] != "typical":
            parts[-1] += f", a difference of {fmt(r['estimate'])}"
    elif report["family"] == "groups":
        ranked = sorted((g for g in groups if g["mean"] is not None), key=lambda g: -g["mean"])
        parts.append(f"The mean {outcome} ranges from {fmt(ranked[-1]['mean'])} for {ranked[-1]['level']} to "
                     f"{fmt(ranked[0]['mean'])} for {ranked[0]['level']}, across {len(ranked)} {explanatory} groups and "
                     f"{r['sample']['n']:,} rows")
    elif groups and "share" in groups[0]:
        a, b = groups
        parts.append(f"{r['estimand'].capitalize()} is {fmt(r['estimate'])}: {fmt(a['share'])} of {a['n']:,} rows for "
                     f"{a['level']} and {fmt(b['share'])} of {b['n']:,} rows for {b['level']}")
    else:
        parts.append(f"{outcome} and {explanatory} were cross-tabulated over {r['sample'].get('n', 0):,} rows; the strength of "
                     f"association (Cramér's V) is {fmt(r['estimate'])}")
    if interval:
        # An interval that rests on an assumption is quoted with it, never bare.
        assumed = re.search(r"\((assumes [^)]*)\)", interval.get("method") or "")
        parts[-1] += (f" ({interval['level'] * 100:g}% interval {fmt(interval['low'])} to {fmt(interval['high'])}"
                      + (f", which {assumed.group(1)}" if assumed else "") + ")")
    sentence = parts[0] + "."
    if not interval and r.get("estimate") is not None:
        # Said in the summary itself: a reader must not take a missing interval for an oversight.
        sentence += f" No interval is reported: {r.get('interval_unavailable')}."
    if test.get("p_value") is not None:
        sentence += f" {r['method_label'][0].upper() + r['method_label'][1:]}: {p_text(test['p_value'])}."
        nothing = "no association" if report["family"] in ("correlation", "table") and len(groups) != 2 else "no difference"
        sentence += (f" A result this far from {nothing} would be unusual if there were none."
                     if r["significant"] else
                     f" The data does not distinguish this from {nothing}, which is not evidence that there is none.")
    sentence += " " + report["causal"]["statement"]
    if plan["claims_cause"]:
        sentence += " A causal answer would need evidence this dataset does not contain."
    return sentence
