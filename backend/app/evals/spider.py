"""The "spider-numeric" suite: Spider validation questions with a single-row
numeric answer, put to the analytical copilot with no model.

Each case is run exactly as `copilot.handle_message(..., allow_model=False)`
handles the first message of a fresh session: rule interpretation
(`copilot.interpret`), validated operations (`copilot.apply_operations`) and the
deterministic metric engine (`copilot.compute`). The database layer around it is
left out; the only thing it would add is approved metric definitions from the
semantic layer, and the evaluation workspace has none. No model is called and no
generated code is run.

Every case is scored against the value the reference SQL returns, and also
translated from Spider's own parse of that SQL into a metric-engine formula and
filters (the "oracle"). The oracle answers a separate question: could the
engine have produced the right number at all? That separates

    deterministic_limitation  the engine or the copilot cannot express the
                              question (OR across columns, arithmetic inside an
                              aggregate, several numbers in one answer, ...)
    data_representation       the reference is not a number, or the engine,
                              reading the CSV as the application does, gets a
                              different number from SQLite for the same spec
    question_understanding    the oracle spec gives the right number, so the
                              failure is in reading the question
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASE_FILE = Path(__file__).resolve().parents[2] / "evals" / "cases" / "spider_numeric.json"
SUITE = "spider-numeric"
COMPONENT = "spider_copilot"

_AGG = {1: "max", 2: "min", 3: "count", 4: "sum", 5: "avg"}
_WHERE = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
_OPS = {"=": "==", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "!=": "!="}
_NEGATED = {"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}
FLOAT_DIGITS = 6


# ------------------------------------------------------------------ scoring --

def exact(got: float | None, want: float | None) -> bool:
    if got is None or want is None:
        return False
    return round(float(got), FLOAT_DIGITS) == round(float(want), FLOAT_DIGITS)


def matched(got: list, want: list, same) -> bool:
    """Every reference number matched by a different returned number."""
    left = list(got)
    for w in want:
        hit = next((i for i, g in enumerate(left) if same(g, w)), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def within(got: float | None, want: float | None, tol: dict) -> bool:
    if got is None or want is None:
        return False
    return math.isclose(float(got), float(want), rel_tol=tol["rel"], abs_tol=tol["abs"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------- oracle --

class NotExpressible(Exception):
    """The reference SQL has no metric-engine equivalent. The message says why."""


def _literal(value):
    if isinstance(value, str):
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            return v[1:-1]
        return v
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise NotExpressible("a condition compares with a column or a subquery")


def _column(col_id: int, names: list[str], frame_columns: list[str]) -> str:
    name = names[col_id]
    lower = {c.lower(): c for c in frame_columns}
    if name.lower() not in lower:
        raise NotExpressible(f"column {name!r} is not in the CSV")
    return lower[name.lower()]


def _formula(item, names: list[str], frame_columns: list[str]) -> str:
    agg, (unit_op, col_unit, _second) = item
    if agg == 0:
        raise NotExpressible("the answer selects a column that is not aggregated")
    if unit_op != 0:
        raise NotExpressible("arithmetic inside an aggregation (for example avg(a - b))")
    inner_agg, col_id, distinct = col_unit
    if inner_agg != 0:
        raise NotExpressible("a nested aggregation")
    fn = _AGG[agg]
    if col_id == 0:
        if fn != "count" or distinct:
            raise NotExpressible(f"{fn}(*)")
        return "count()"
    column = _column(col_id, names, frame_columns)
    if distinct:
        if fn != "count":
            raise NotExpressible(f"{fn}(DISTINCT ...)")
        fn = "count_distinct"
    return f'{fn}("{column}")'


def _condition(cond, names: list[str], frame_columns: list[str]) -> dict:
    not_op, op_id, (unit_op, col_unit, _), v1, v2 = cond
    op = _WHERE[op_id]
    if unit_op != 0 or col_unit[0] != 0:
        raise NotExpressible("a condition on an expression or an aggregate")
    column = _column(col_unit[1], names, frame_columns)
    if op == "between":
        if not_op:
            raise NotExpressible("NOT BETWEEN")
        return {"column": column, "op": "between", "value": [_literal(v1), _literal(v2)]}
    if op == "like":
        pattern = _literal(v1)
        if not_op or not (isinstance(pattern, str) and pattern.startswith("%") and pattern.endswith("%")
                          and "%" not in pattern[1:-1] and "_" not in pattern):
            raise NotExpressible(f"LIKE pattern {pattern!r} is not a plain 'contains'")
        return {"column": column, "op": "contains", "value": pattern[1:-1]}
    if op not in _OPS:
        raise NotExpressible(f"the condition operator {op.upper()}")
    engine_op = _NEGATED[_OPS[op]] if not_op else _OPS[op]
    return {"column": column, "op": engine_op, "value": _literal(v1)}


def oracle_spec(sql: dict, names: list[str], frame_columns: list[str]) -> dict:
    """Spider's parse of the reference query as metric-engine formulas (one per
    answer column) and AND-ed filters. Raises NotExpressible."""
    if sql.get("groupBy") or sql.get("having") or sql.get("intersect") or sql.get("union") or sql.get("except"):
        raise NotExpressible("grouping or a set operation")
    formulas = [_formula(item, names, frame_columns) for item in sql["select"][1]]
    where = sql.get("where") or []
    conds, joins = where[0::2], set(where[1::2])
    if "or" in joins:
        # Equalities on one column joined by OR are an IN filter; anything else is not expressible.
        parsed = [_condition(c, names, frame_columns) for c in conds]
        if joins != {"or"} or len({p["column"] for p in parsed}) != 1 or any(p["op"] != "==" for p in parsed):
            raise NotExpressible("OR across different columns or operators (engine filters are AND-ed)")
        return {"formulas": formulas, "filters": [{"column": parsed[0]["column"], "op": "in",
                                                   "value": [p["value"] for p in parsed]}]}
    return {"formulas": formulas, "filters": [_condition(c, names, frame_columns) for c in conds]}


def run_oracle(case: dict, path: str, frame) -> dict:
    from app import metric_engine as me

    try:
        spec = oracle_spec(case["sql_parse"], case["column_names"], list(frame.columns))
    except NotExpressible as e:
        return {"status": "not_expressible", "reason": str(e)}
    try:
        values = [me.evaluate(frame, f, spec["filters"])["value"] for f in spec["formulas"]]
    except me.FormulaError as e:
        return {"status": "not_expressible", "reason": f"the engine rejects the translated spec: {e}", "spec": spec}
    want = case["reference"]["numbers"]
    agrees = case["reference"]["scorable"] and len(values) == len(want) and all(
        within(g, w, case["tolerance"]) for g, w in zip(values, want))
    return {"status": "matches" if agrees else "differs", "spec": spec, "values": values}


# ------------------------------------------------------------------ copilot --

def ask_copilot(question: str, path: str, profile: dict, frame, dataset_name: str | None = None) -> dict:
    """The first turn of a copilot session, rules only. Mirrors the branches of
    `copilot.handle_message` with allow_model=False. `dataset_name` is the
    uploaded file's name, which the copilot receives in the application."""
    import inspect

    from app import copilot
    from app import metric_engine as me

    state = copilot.empty_state()
    kwargs = {"dataset_name": dataset_name} if "dataset_name" in inspect.signature(copilot.interpret).parameters else {}
    interpreted = copilot.interpret(question, state, profile, copilot.category_index(frame, profile), [], **kwargs)
    if "clarification" in interpreted:
        return {"status": "clarification", "reason": interpreted["clarification"]["reason"]}
    if interpreted.get("unmapped"):
        return {"status": "not_understood"}
    try:
        new_state, _ = copilot.apply_operations(state, interpreted["operations"], profile)
    except copilot.CopilotError as e:
        return {"status": "invalid_operation", "reason": str(e), "operations": interpreted["operations"]}
    spec = {"formula": new_state.get("formula"), "filters": copilot._engine_filters(new_state),
            "group_by": new_state.get("group_by") or [],
            "formulas": [new_state.get("formula"), *(m["formula"] for m in new_state.get("extra_metrics") or [])]}
    if not new_state.get("formula"):
        return {"status": "no_metric", "spec": spec}
    try:
        result = copilot.compute(path, new_state)
    except me.FormulaError as e:
        return {"status": "compute_error", "reason": str(e), "spec": spec}
    if result["type"] == "scalars":
        return {"status": "answered_multi", "spec": spec, "values": [v["value"] for v in result["values"]],
                "value": result["value"], "reference_check": result["reference_check"]}
    if result["type"] != "scalar":
        return {"status": "answered_table", "spec": spec, "groups": result.get("total_groups")}
    return {"status": "answered", "spec": spec, "value": result["value"], "reference_check": result["reference_check"]}


# --------------------------------------------------------------------- case --

def _filter_set(filters) -> set:
    return {(f["column"], f["op"], str(f["value"])) for f in filters or []}


def _understanding_issues(copilot_out: dict, oracle: dict) -> list[str]:
    """What went wrong in reading the question, most decisive first. Each label
    compares what the rules produced with the oracle spec, or quotes the
    copilot's own clarification."""
    status, want = copilot_out["status"], oracle["spec"]
    if status == "clarification":
        reason = copilot_out.get("reason", "")
        if "no date column" in reason:
            return ["a year in the question was read as a date range, but the table has no date column"]
        if "is a value in more than one column" in reason:
            return ["a value in the question occurs in several columns, and the rules asked which one"]
        if "matches more than one column" in reason:
            return ["the metric wording matched several columns, and the rules asked which one"]
        return [f"the rules asked for clarification: {reason}"]
    if status == "invalid_operation":
        return [f"the rules produced an operation the dataset rejects: {copilot_out.get('reason')}"]
    got = copilot_out.get("spec") or {}
    issues = []
    if not got.get("formula"):
        issues.append("a count of entities ('how many singers') was not read as a row count"
                      if want["formulas"] == ["count()"] else "the aggregation or the column is worded in a way the rules do not recognise")
    elif sorted(got.get("formulas") or [got["formula"]]) != sorted(want["formulas"]):
        if len(want["formulas"]) > 1 and len(got.get("formulas") or [got["formula"]]) == 1:
            issues.append("the question asks for several numbers; the rules read one")
        else:
            issues.append(f"wrong metric: computed {' and '.join(got.get('formulas') or [got['formula']])}, "
                          f"the question asks for {' and '.join(want['formulas'])}")
    if status == "answered_table" or got.get("group_by"):
        issues.append("the rules added a breakdown the question does not ask for")
    have, need = _filter_set(got.get("filters")), _filter_set(want["filters"])
    if need - have:
        issues.append("a condition in the question was not turned into a filter")
    if have - need:
        issues.append("a filter was applied that the question does not state")
    return issues or ["the computed spec differs from the question in a way not classified"]


def run_case(case: dict, chk) -> dict:
    """Run one case; add its checks to `chk`; return metrics for the report."""
    from app import metric_engine as me
    from app.profiling import load_and_profile_csv

    path = ROOT / case["csv"]
    intact = path.exists() and _sha256(path) == case["csv_sha256"]
    chk.add("fixture_intact", intact, case["csv"] if intact else f"{case['csv']} is missing or changed; rebuild the reference set")
    if not intact:
        return {"outcome": "fixture_missing", "cause": "not_run"}
    profile = load_and_profile_csv(str(path))[1]
    frame = me.load_frame(str(path))
    # The application gives the copilot the uploaded file's name; a Spider table uploaded on its own
    # is "<table>.csv" (the development scorer passes the same).
    out = ask_copilot(case["question"], str(path), profile, frame, dataset_name=f"{case['table']}.csv")
    oracle = run_oracle(case, str(path), frame)
    ref, tol = case["reference"], case["tolerance"]
    want = ref["numbers"]
    multi = len(want) > 1
    value = out.get("value")

    is_exact = is_close = False
    if out["status"] == "answered":
        if multi:
            outcome = "partial" if ref["scorable"] and any(within(value, w, tol) for w in want) else "wrong_value"
        elif not ref["scorable"]:
            outcome = "wrong_value"
        else:
            is_exact, is_close = exact(value, want[0]), within(value, want[0], tol)
            outcome = "correct_exact" if is_exact else "correct_tolerance" if is_close else "wrong_value"
    elif out["status"] == "answered_multi":
        got = out["values"]
        if not multi or not ref["scorable"] or len(got) != len(want):
            outcome = "wrong_shape" if not multi or len(got) != len(want) else "wrong_value"
        else:
            is_close = matched(got, want, lambda g, w: within(g, w, tol))
            is_exact = is_close and matched(got, want, exact)
            outcome = "correct_exact" if is_exact else "correct_tolerance" if is_close else "wrong_value"
    elif out["status"] == "answered_table":
        outcome = "wrong_shape"
    else:
        outcome = out["status"]  # not_understood, clarification, no_metric, invalid_operation, compute_error

    correct = outcome in ("correct_exact", "correct_tolerance")
    chk.add("answered", out["status"] in ("answered", "answered_multi"), out["status"] + (f": {out['reason']}" if out.get("reason") else ""))
    chk.add("answer_within_tolerance", correct,
            f"copilot {out.get('values', value)!r}, reference {ref['values']!r}" + ("" if ref["scorable"] else f" ({ref['note']})"))

    cause = reason = None
    issues: list[str] = []
    if not correct:
        if not ref["scorable"]:
            cause, reason = "data_representation", {"null_reference": "the reference SQL returns NULL",
                                                    "non_numeric_reference": "the reference answer is not a number"}[ref["note"]]
        elif oracle["status"] == "not_expressible":
            cause, reason = "deterministic_limitation", oracle["reason"]
        elif oracle["status"] == "differs":
            cause, reason = "data_representation", "the engine, reading the CSV as the application does, gets a different number for the same spec"
        else:
            issues = _understanding_issues(out, oracle)
            cause, reason = "question_understanding", issues[0]

    got_spec = out.get("spec") or {}
    got_formulas = got_spec.get("formulas") or [got_spec.get("formula")]
    spec_matches = (oracle.get("spec") is not None and sorted(got_formulas) == sorted(oracle["spec"]["formulas"])
                    and not got_spec.get("group_by")
                    and _filter_set(got_spec.get("filters")) == _filter_set(oracle["spec"]["filters"]))
    return {
        "outcome": outcome, "correct": correct, "exact": is_exact, "within_tolerance": is_close,
        "copilot_status": out["status"], "copilot_formula": got_spec.get("formula"),
        "copilot_filters": got_spec.get("filters"), "copilot_group_by": got_spec.get("group_by"),
        "copilot_value": value, "reference_values": ref["values"], "reference_note": ref["note"],
        "answer_shape": case["answer_shape"], "reference_id": case["reference_id"], "db_id": case["db_id"],
        "oracle_status": oracle["status"], "oracle_reason": oracle.get("reason"),
        "oracle_formulas": (oracle.get("spec") or {}).get("formulas"), "oracle_filters": (oracle.get("spec") or {}).get("filters"),
        "oracle_values": oracle.get("values"), "spec_matches_oracle": bool(spec_matches),
        "cause": cause, "cause_detail": reason, "understanding_issues": issues, "leakage": case.get("leakage", []),
        "data_fidelity_risk": case.get("data_fidelity_risk", []),
    }
