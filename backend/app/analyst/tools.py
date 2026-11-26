"""The analyst's tools: a whitelist with argument schemas, validation against
the dataset, and deterministic executors.

A tool call is {"tool": name, "args": {...}}. `validate_call` checks the name,
every argument's type, that named columns exist and have the right kind, and
that values are allowed; only then does `run` execute it. Executors reuse the
metric engine (app.metric_engine), the statistics library (app.stats.methods),
scikit-learn Random Forests (ml.py) and the business analyses (business.py).
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st

from app import metric_engine as me
from app.analyst import business, ml
from app.analyst.data import finite, kind_of, schema
from app.stats import methods as sm

LEVEL = 0.95
AGGS = ("count", "count_distinct", "sum", "avg", "median", "min", "max", "std", "var")
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains")

# name -> {arg: (type, required, column kind or None)}; "column" type means a dataset column name.
TOOLS: dict[str, dict[str, tuple]] = {
    "describe_dataset": {},
    "compute_metric": {"aggregation": ("agg", True, None), "column": ("column", False, None),
                       "filters": ("filters", False, None), "group_by": ("column", False, "groupable"),
                       "order": ("order", False, None)},
    "describe_column": {"column": ("column", True, "numeric"), "filters": ("filters", False, None)},
    "compare_groups": {"outcome": ("column", True, None), "group": ("column", True, "categorical"),
                       "filters": ("filters", False, None)},
    "correlation": {"x": ("column", True, "numeric"), "y": ("column", True, "numeric")},
    "linear_regression": {"target": ("column", True, "numeric"), "features": ("columns", True, "numeric")},
    "business_performance": {"group": ("column", False, "categorical"), "focus": ("str", False, None)},
    "recommend": {"group": ("column", False, "categorical"), "focus": ("str", False, None),
                  "kind": ("enum:where,increase", False, None)},
    "budget_scenario": {"change_pct": ("number", True, None), "focus": ("str", False, None),
                        "group": ("column", False, "categorical")},
    "random_forest": {"target": ("column", True, None), "task": ("enum:classification,regression", False, None)},
    "feature_importance": {},
    "predict": {"rows": ("rows", True, None)},
}
DESCRIPTIONS = {
    "describe_dataset": "columns, types, row count",
    "compute_metric": "aggregate a column (count, count_distinct, sum, avg, median, min, max, std, var), optional filters and group_by",
    "describe_column": "mean, median, variance, standard deviation and a 95% confidence interval of one numeric column",
    "compare_groups": "compare an outcome across groups with an appropriate test (Welch t / Welch ANOVA / chi-square), effect size and assumption checks",
    "correlation": "Pearson and Spearman correlation between two numeric columns",
    "linear_regression": "ordinary least squares of a numeric target on numeric features",
    "business_performance": "spend, revenue, ROI, CAC and conversion rate per area or campaign, with the focus compared to the rest",
    "recommend": "evidence-based recommendation for an area (kind=where: next campaign; kind=increase: budget increase)",
    "budget_scenario": "what a budget change would mean under stated assumptions (not a forecast)",
    "random_forest": "train and evaluate a Random Forest classifier or regressor for a target column",
    "feature_importance": "importance of each input in the most recently trained Random Forest",
    "predict": "predict the target for new rows with the most recently trained Random Forest",
}


class InvalidCall(ValueError):
    pass


class RoleRefusal(InvalidCall):
    """The call is well-formed, but the aggregation has no meaning for what the column is (the average of an
    identifier, the total of a price). The message is written for the user and shown as the answer."""


_ROLE_FN = {"avg": "mean"}


def role_check(df: pd.DataFrame, col, fn: str) -> None:
    """Raise RoleRefusal when `fn` of `col` is not a quantity (see semantics.aggregation_problem). The one guard
    every computing path calls before it aggregates a column of the data."""
    from app.analyst import semantics
    why = semantics.aggregation_problem(df, col, _ROLE_FN.get(fn, fn))
    if why:
        raise RoleRefusal(why)


def _role_guard(name: str, clean: dict, df: pd.DataFrame) -> None:
    from app.analyst import semantics
    checks = []
    if name == "compute_metric" and clean.get("column"):
        checks.append((clean["column"], _ROLE_FN.get(clean["aggregation"], clean["aggregation"])))
    elif name == "describe_column":
        checks.append((clean["column"], "mean"))
    elif name == "compare_groups":
        checks.append((clean["outcome"], "mean"))
    elif name == "correlation":
        checks += [(clean["x"], "mean"), (clean["y"], "mean")]
    elif name == "linear_regression":
        checks += [(clean["target"], "mean")] + [(c, "mean") for c in clean.get("features", [])]
    for col, fn in checks:
        why = semantics.aggregation_problem(df, col, fn)
        if why:
            raise RoleRefusal(why)


def catalogue() -> str:
    lines = []
    for name, args in TOOLS.items():
        spec = ", ".join(f"{a}{'' if req else '?'}: {t}" for a, (t, req, _) in args.items())
        lines.append(f"- {name}({spec}): {DESCRIPTIONS[name]}")
    return "\n".join(lines)


def _fold(text: Any) -> str:
    import re
    return " " + re.sub(r"[^a-z0-9.]+", " ", str(text).lower()).strip() + " "


def _number_written(value: float, message: str) -> bool:
    import re
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", message):
        try:
            if float(m.replace(",", "")) == value:
                return True
        except ValueError:
            pass
    return False


def check_supplied_inputs(rows: list[dict], message: str | None) -> None:
    """Prediction inputs must be the user's own: every value must appear in the
    user's message ("a Senior in Engineering with 10 years" gives three). A
    planner may not invent a row to predict for. Columns are checked to exist
    by the caller; the user need not name them."""
    if not message:
        raise InvalidCall("predict: prediction inputs must be given in your message")
    folded = _fold(message)
    for row in rows:
        for col, value in row.items():
            if isinstance(value, bool) or value is None:
                raise InvalidCall(f"predict: give a value for '{col}' in your message")
            if isinstance(value, (int, float)):
                if not _number_written(float(value), message):
                    raise InvalidCall(f"predict: the value {value} for '{col}' was not given in your message")
            elif _fold(value).strip() not in folded:
                raise InvalidCall(f"predict: the value '{value}' for '{col}' was not given in your message")


def resolve_column(name: Any, columns) -> str | None:
    """The dataset's own name for `name`: the exact name if it exists; otherwise the single column that differs only
    in letter case (product_code -> Product_Code). None if there is no such column. Several case-variants with no
    exact match raise InvalidCall (ambiguous): Silt never guesses between them."""
    if not isinstance(name, str):
        return None
    if name in columns:
        return name
    hits = [c for c in columns if isinstance(c, str) and c.casefold() == name.casefold()]
    if len(hits) > 1:
        raise InvalidCall(f"the column name {name!r} is ambiguous: it matches {', '.join(repr(h) for h in hits)} "
                          "except for letter case; use the exact name")
    return hits[0] if hits else None


def validate_call(call: Any, df: pd.DataFrame, message: str | None = None) -> dict:
    """The call, cleaned, or InvalidCall with the reason. `message` is the
    user's message: prediction inputs must appear in it."""
    if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
        raise InvalidCall("a tool call must be an object with a 'tool' name")
    name = call["tool"]
    if name not in TOOLS:
        raise InvalidCall(f"unknown tool '{name}'")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        raise InvalidCall(f"{name}: args must be an object")
    spec = TOOLS[name]
    extra = set(args) - set(spec)
    if extra:
        raise InvalidCall(f"{name}: unknown argument(s) {sorted(extra)}")
    clean = {}
    for arg, (typ, required, kind) in spec.items():
        v = args.get(arg)
        if v in (None, "", []):
            if required:
                raise InvalidCall(f"{name}: '{arg}' is required")
            continue
        clean[arg] = _check(name, arg, typ, kind, v, df)
    if name == "compute_metric":
        agg = clean["aggregation"]
        if agg != "count" and "column" not in clean:
            raise InvalidCall(f"compute_metric: '{agg}' needs a column")
        if agg in ("sum", "avg", "median", "std", "var") and kind_of(df[clean["column"]]) != "numeric":
            raise InvalidCall(f"compute_metric: '{clean['column']}' is not numeric, so {agg} does not apply")
    _role_guard(name, clean, df)
    if name == "predict":
        rows = []
        for row in clean["rows"]:
            unknown = [c for c in row if resolve_column(c, df.columns) is None]
            if unknown:
                raise InvalidCall(f"predict: no column named {unknown[0]!r}")
            rows.append({resolve_column(c, df.columns): v for c, v in row.items()})
        clean["rows"] = rows
        check_supplied_inputs(clean["rows"], message)
    return {"tool": name, "args": clean}


def _check(tool: str, arg: str, typ: str, kind: str | None, v: Any, df: pd.DataFrame) -> Any:
    where = f"{tool}.{arg}"
    if typ == "column":
        try:
            resolved = resolve_column(v, df.columns)
        except InvalidCall as e:
            raise InvalidCall(f"{where}: {e}") from None
        if resolved is None:
            raise InvalidCall(f"{where}: no column named {v!r}")
        v = resolved
        if kind == "groupable":
            # a breakdown: categories, or a number with few distinct values (an hour of day, a year, a rating)
            if not (kind_of(df[v]) == "categorical" or (kind_of(df[v]) == "numeric" and df[v].nunique() <= 50)):
                raise InvalidCall(f"{where}: '{v}' has {df[v].nunique()} distinct values; a breakdown needs a category "
                                  "or a number with few distinct values")
        elif kind and kind_of(df[v]) != kind:
            raise InvalidCall(f"{where}: '{v}' is {kind_of(df[v])}, a {kind} column is needed")
        return v
    if typ == "columns":
        if not isinstance(v, list) or not v:
            raise InvalidCall(f"{where}: a list of columns is needed")
        return [_check(tool, arg, "column", kind, c, df) for c in v]
    if typ == "agg":
        if v not in AGGS:
            raise InvalidCall(f"{where}: {v!r} is not one of {AGGS}")
        return v
    if typ == "order":
        if v not in ("desc", "asc"):
            raise InvalidCall(f"{where}: order is 'desc' or 'asc'")
        return v
    if typ.startswith("enum:"):
        allowed = typ[5:].split(",")
        if v not in allowed:
            raise InvalidCall(f"{where}: {v!r} is not one of {allowed}")
        return v
    if typ == "str":
        if not isinstance(v, (str, int, float)):
            raise InvalidCall(f"{where}: a text value is needed")
        return str(v)
    if typ == "number":
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise InvalidCall(f"{where}: a number is needed") from None
        if not math.isfinite(x):
            raise InvalidCall(f"{where}: a finite number is needed")
        return x
    if typ == "filters":
        if not isinstance(v, list):
            raise InvalidCall(f"{where}: filters are a list")
        out = []
        for f in v:
            col = resolve_column(f.get("column"), df.columns) if isinstance(f, dict) else None
            if col is None or f.get("op") not in OPS:
                raise InvalidCall(f"{where}: each filter needs an existing column and an operator in {OPS}")
            period = _partial_date_period(df[col], f["op"], f.get("value"))
            if period is not None:
                # "stay_date == 2025-02" means the whole of February 2025, not its first day
                out += [{"column": col, "op": ">=", "value": period[0]}, {"column": col, "op": "<", "value": period[1]}]
                continue
            out.append({"column": col, "op": f["op"], "value": f.get("value")})
        return out
    if typ == "rows":
        if not isinstance(v, list) or not all(isinstance(r, dict) for r in v) or len(v) > 50:
            raise InvalidCall(f"{where}: up to 50 rows, each an object of column -> value")
        return v
    raise InvalidCall(f"{where}: unsupported argument type {typ}")


_PARTIAL_DATE = re.compile(r"^\s*(\d{4})(?:-(\d{1,2}))?\s*$")


def _partial_date_period(s: pd.Series, op, value) -> tuple[str, str] | None:
    """An equality filter on a date column whose value names a whole year ("2024") or month ("2025-02"): the
    [start, next start) of that period. None for anything else (a full date, another operator, a non-date column)."""
    if op != "==" or not isinstance(value, (str, int)) or not pd.api.types.is_datetime64_any_dtype(s):
        return None
    m = _PARTIAL_DATE.match(str(value))
    if not m:
        return None
    y, mo = int(m.group(1)), m.group(2)
    if mo is None:
        return f"{y}-01-01", f"{y + 1}-01-01"
    mo = int(mo)
    if not 1 <= mo <= 12:
        return None
    return f"{y}-{mo:02d}-01", (f"{y + 1}-01-01" if mo == 12 else f"{y}-{mo + 1:02d}-01")


# ------------------------------------------------------------------ executors --

def run(call: dict, df: pd.DataFrame, memory: dict) -> dict:
    """Execute a validated call. `memory` holds the session's trained model."""
    name, a = call["tool"], call["args"]
    return _EXEC[name](df, a, memory)


def _describe_dataset(df, a, memory):
    return {"schema": schema(df)}


def _filtered(df, filters):
    return me.apply_filters(df, filters) if filters else df


def _compute_metric(df, a, memory):
    agg, col, filters, group = a["aggregation"], a.get("column"), a.get("filters"), a.get("group_by")
    from app.analyst.scope import valid_records
    excluded = []
    if col and agg not in ("count", "count_distinct") and col in df.columns:
        # one record-validity policy for every path (scope.valid_records): impossible values left out and listed
        df, excluded = valid_records(df, [col])
        excluded = [x for x in excluded if not x.get("kept")]
    data = _filtered(df, filters)
    no_group = 0
    if group and group in df.columns:
        # records with no value of the breakdown column belong to no group: left out, and counted
        no_group = int(data[group].isna().sum())
        data = data[data[group].notna()]
        df = data
    if agg in ("std", "var"):
        def f(s):
            x = finite(s)
            return float(np.std(x, ddof=1) if agg == "std" else np.var(x, ddof=1)) if len(x) > 1 else None
        if group:
            groups = [{"group": str(k), "value": f(g[col]), "rows": int(len(g))} for k, g in data.groupby(group)]
            total_groups = len(groups)
        else:
            return {"aggregation": agg, "column": col, "value": f(data[col]), "rows": int(len(data)), "filters": filters or []}
    else:
        formula = "count()" if agg == "count" and not col else f'{agg}("{col}")'
        r = me.evaluate(df, formula, filters or [], [group] if group else None, order=a.get("order", "desc"))
        if agg in ("count", "count_distinct"):
            # a count is a whole number, kept as one so it is shown as 5, not 5.00
            r["value"] = None if r.get("value") is None else int(round(r["value"]))
            for g_ in r.get("groups") or []:
                g_["value"] = None if g_.get("value") is None else int(round(g_["value"]))
        if not group:
            return {"aggregation": agg, "column": col, "value": r["value"], "rows": r["rows"], "filters": filters or [],
                    "excluded": excluded}
        groups = [{"group": str(g["key"][group]), "value": g["value"], "rows": g["rows"]} for g in r["groups"]]
        total_groups = r.get("groups_total", len(groups))
    rev = a.get("order", "desc") == "desc"
    groups.sort(key=lambda g: (g["value"] is None, -(g["value"] or 0) if rev else (g["value"] or 0)))
    lead = groups[0]["value"] if groups else None
    ties = sum(1 for g in groups if g["value"] is not None and lead is not None and g["value"] == lead)
    return {"aggregation": agg, "column": col, "group_by": group, "groups": groups, "filters": filters or [],
            "excluded": excluded,
            "top": groups[0] if groups else None, "groups_total": total_groups, "tied_at_top": ties,
            "no_group_value": no_group}


def _describe_column(df, a, memory):
    from app.analyst.scope import valid_records
    df, excluded = valid_records(df, [a["column"]])
    excluded = [e for e in excluded if not e.get("kept")]
    out = _describe_column_on(df, a, memory)
    out["excluded"] = excluded
    return out


def _describe_column_on(df, a, memory):
    x = finite(_filtered(df, a.get("filters"))[a["column"]])
    if len(x) == 0:
        return {"column": a["column"], "n": 0, "note": "no numeric values"}
    d = sm.describe(x)
    ci = sm.mean_interval(x, LEVEL)
    vc = pd.Series(x).value_counts()
    modes = sorted(float(v) for v in vc[vc == vc.max()].index) if len(vc) and vc.max() > 1 else []
    col_all = _filtered(df, a.get("filters"))[a["column"]]
    return {"column": a["column"], "n": d["n"], "missing": int(col_all.isna().sum()), "mode": modes,
            "mode_count": int(vc.max()) if len(vc) else 0, "mean": d["mean"], "median": d["median"], "sd": d["sd"],
            "variance": d["sd"] ** 2 if d["sd"] is not None else None, "min": d["min"], "max": d["max"],
            "skewness": d["skewness"], "mean_ci95": list(ci) if ci else None,
            "method": "t interval for the mean" + ("; the distribution is skewed, so the median may summarise it better"
                                                    if d["skewness"] is not None and abs(d["skewness"]) > 1 else "")}


def _compare_groups(df, a, memory):
    from app.analyst.scope import valid_records
    excluded = []
    if kind_of(df[a["outcome"]]) == "numeric":
        df, excluded = valid_records(df, [a["outcome"]])          # the one record-validity policy
        excluded = [e for e in excluded if not e.get("kept")]
    out = _compare_groups_on(df, a, memory)
    if isinstance(out, dict) and excluded:
        out["excluded"] = excluded
    return out


def _compare_groups_on(df, a, memory):
    data = _filtered(df, a.get("filters"))
    y, g = a["outcome"], a["group"]
    data = data[[y, g]].dropna()
    groups = {str(k): v[y] for k, v in data.groupby(g)}
    if len(groups) < 2:
        return {"note": f"'{g}' has fewer than two groups after filtering, so there is nothing to compare"}
    if len(groups) > 30:
        return {"note": f"'{g}' has {len(groups)} groups; too many for a meaningful comparison"}
    if kind_of(data[y]) == "numeric":
        arrays = {k: finite(v) for k, v in groups.items()}
        small = [k for k, v in arrays.items() if len(v) < 2]
        if small:
            return {"note": f"groups with fewer than 2 values cannot be compared: {', '.join(small)}"}
        summary = [{"group": k, "n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                    "sd": float(v.std(ddof=1))} for k, v in arrays.items()]
        summary.sort(key=lambda s: -s["mean"])
        out = {"outcome": y, "group": g, "kind": "numeric", "summary": summary, "assumptions": []}
        names = list(arrays)
        if len(names) == 2:
            w = sm.welch(arrays[names[0]], arrays[names[1]], LEVEL)
            out["test"] = {"name": "Welch's t-test", "difference": f"{names[1]} minus {names[0]}",
                           "estimate": w["estimate"], "ci": [w["interval"]["low"], w["interval"]["high"]],
                           "statistic": w["test"]["statistic"], "p_value": w["test"]["p_value"],
                           "hedges_g": sm.hedges_g(arrays[names[0]], arrays[names[1]])}
        else:
            w = sm.welch_anova(list(arrays.values()))
            k = sm.kruskal(list(arrays.values()))
            out["test"] = {"name": w["name"], "statistic": w["statistic"], "df": w["df"], "p_value": w["p_value"],
                           "eta_squared": sm.eta_squared(list(arrays.values())),
                           "robustness": {"name": k["name"], "p_value": k["p_value"]}}
        out["assumptions"].append("Welch's tests do not assume equal variances.")
        if min(len(v) for v in arrays.values()) < 15:
            out["assumptions"].append("Some groups have fewer than 15 values: normality of the means is not assured; "
                                      "compare with the rank-based result where given.")
        return out
    table = pd.crosstab(data[g].astype(str), data[y].astype(str))
    if table.shape[1] < 2:
        return {"note": f"'{y}' has a single value, so there is nothing to compare"}
    arr = table.to_numpy()
    ok = sm.cochran_expected_ok(sm.expected_counts(arr))
    rates = [{"group": str(idx), "n": int(row.sum()), **{f"share_{c}": float(row[c] / row.sum()) for c in table.columns}}
             for idx, row in table.iterrows()]
    out = {"outcome": y, "group": g, "kind": "categorical", "summary": rates, "assumptions": []}
    if ok:
        c = sm.chi_square(arr)
        out["test"] = {"name": c["name"], "statistic": c["statistic"], "df": c["df"], "p_value": c["p_value"],
                       "cramers_v": sm.cramers_v(c["statistic"], int(arr.sum()), arr.shape)}
        out["assumptions"].append("Expected counts are large enough for the chi-square approximation.")
    elif arr.shape == (2, 2):
        f = sm.fisher(arr)
        out["test"] = {"name": f["name"], "p_value": f["p_value"]}
        out["assumptions"].append("Expected counts are too small for chi-square, so Fisher's exact test was used.")
    else:
        out["test"] = None
        out["assumptions"].append("Expected counts are too small for chi-square and the table is larger than 2x2: "
                                  "no test is reported; combine categories or collect more data.")
    return out


def _correlation(df, a, memory):
    d = df[[a["x"], a["y"]]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 5:
        return {"note": "fewer than 5 complete pairs"}
    x, y = d[a["x"]].to_numpy(float), d[a["y"]].to_numpy(float)
    p = sm.pearson(x, y, LEVEL)
    s = st.spearmanr(x, y)
    return {"x": a["x"], "y": a["y"], "n": int(len(d)), "pearson_r": p["estimate"],
            "pearson_ci": [p["interval"]["low"], p["interval"]["high"]], "pearson_p": p["test"]["p_value"],
            "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue),
            "note": "Correlation measures association, not causation."}


def _linear_regression(df, a, memory):
    cols = [a["target"], *a["features"]]
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(d), len(a["features"])
    if n < k + 10:
        return {"note": f"{n} complete rows for {k} features: too few for a reliable regression"}
    X = np.column_stack([np.ones(n), d[a["features"]].to_numpy(float)])
    y = d[a["target"]].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k - 1
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * st.t.sf(np.abs(tvals), dof)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    names = ["intercept", *a["features"]]
    return {"target": a["target"], "n": n, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / dof,
            "coefficients": [{"term": t, "estimate": float(b), "se": float(s), "p_value": float(p)}
                             for t, b, s, p in zip(names, beta, se, pvals)],
            "note": "Coefficients describe association holding the other features fixed, not causal effects."}


def _business_performance(df, a, memory):
    return business.performance(df, a.get("group"), a.get("focus"))


def _recommend(df, a, memory):
    perf = business.performance(df, a.get("group"), a.get("focus"))
    rec = business.recommend(perf, a.get("kind", "where"))
    memory["last_recommendation"] = {"performance": perf, "recommendation": rec}
    return {"performance": perf, "recommendation": rec}


def _budget_scenario(df, a, memory):
    return business.budget_scenario(df, a["change_pct"], a.get("group"), a.get("focus"))


def _random_forest(df, a, memory):
    # Split and seed come from the user's own words for this turn (orchestrator.ml_options),
    # not from the planner: the planner's tool schema is unchanged.
    opts = memory.get("_ml_options") or {}
    trained = ml.train(df, a["target"], a.get("task"), test_size=opts.get("test_size", ml.TEST_SIZE),
                       seed=opts.get("seed", ml.SEED))
    memory["model"] = trained
    return ml.public(trained)


def _feature_importance(df, a, memory):
    trained = memory.get("model")
    if trained is None:
        raise ml.Unsuitable("No model has been trained in this conversation yet; ask for a prediction first.")
    return {"target": trained["target"], "task": trained["task"], "importance": trained["importance"],
            "metrics": trained["metrics"], "method": trained["method"], "limitations": trained["limitations"]}


def _predict(df, a, memory):
    trained = memory.get("model")
    if trained is None:
        raise ml.Unsuitable("No model has been trained in this conversation yet.")
    return {"target": trained["target"], "predictions": ml.predict(trained, a["rows"]),
            "model": {"task": trained["task"], "test_rows": trained["test_rows"], "metrics": {
                k: v for k, v in trained["metrics"].items() if k in ("balanced_accuracy", "accuracy", "mae", "r2")}}}


_EXEC = {"describe_dataset": _describe_dataset, "compute_metric": _compute_metric, "describe_column": _describe_column,
         "compare_groups": _compare_groups, "correlation": _correlation, "linear_regression": _linear_regression,
         "business_performance": _business_performance, "recommend": _recommend, "budget_scenario": _budget_scenario,
         "random_forest": _random_forest, "feature_importance": _feature_importance, "predict": _predict}
assert set(_EXEC) == set(TOOLS)
