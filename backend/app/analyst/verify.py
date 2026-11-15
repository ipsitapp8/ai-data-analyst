"""Answer verification: is the number shown the number the user asked for?

Run after an analysis program has executed, whoever planned it (Silt's rules or the local model). It looks only at
  - the user's own message (read by app/analyst/intent.py -- never the model's restatement or narration),
  - the program that was actually executed, and
  - the results it produced (their values and provenance),
and reports one status:
  "verified"                  the requirements hold AND the headline figure, recomputed here directly with pandas
                              from the request alone (not from the program's steps), matches a computed result;
  "not independently verified" the requirements hold but the request has a part this module cannot recompute
                              (a scenario, projection, association, anomaly count, ...) -- reported, never passed;
  "failed"                    a requirement is not met, or the independent figure differs: the answer is withheld.

Recomputed here: sum, mean, median, count, the rate of a yes/no column, ratios of totals (per unit, per person,
per record), each optionally by one grouping, restricted by named values, exclusions, "only <flag>" and a period.
The data rules are the same as the program's (one policy): exact duplicates are kept as recorded, impossible records
(scope.eligibility) are left out.
"""
from __future__ import annotations

import math

import pandas as pd

from app.analyst import intent, semantics
from app.analyst.data import kind_of

TOL = 1e-9

_SUPPORTED_EXTRA = ("extreme", "compare", "compared", "why", "unusual", "significance")


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def _same(x, y) -> bool:
    if isinstance(x, dict) and isinstance(y, dict):
        return set(map(str, x)) == set(map(str, y)) and all(_close(x[k], y[str(k)] if str(k) in y else y[k]) for k in x)
    if isinstance(x, dict) or isinstance(y, dict):
        return False
    return _close(x, y)


def recompute(req: intent.Request, df: pd.DataFrame, prof: dict | None = None, scope: dict | None = None,
              policy: dict | None = None):
    """(value, description) of the request's headline figure computed directly from the records, or (None, why not).
    `scope` is Silt's reading of the records the question is about (named values, a year, a month -- inherited by a
    follow-up); `policy` the row rules of the path that answered: {"valid": impossible records left out (programs, the
    calculator) or counted as recorded (the tool catalogue), "duplicates": "keep" | "drop"}."""
    prof = prof or semantics.profile(df)
    roles = prof["roles"]
    policy = {"valid": True, "duplicates": "keep", **(policy or {})}
    if req.hypothetical:
        return None, "a hypothetical scenario is not recomputed here"
    if len(req.measures) != 1 and req.quantity != "count":
        return None, "the request has no single measure to recompute"
    m = req.measures[0] if req.measures else None
    if m is not None and m not in df.columns:
        return None, f"{m} is not a column"
    from app.analyst.scope import eligibility
    cols = [c for c in [m, req.group, *req.denominators] if c and c in df.columns]
    cols += [c for c, _ in req.entities + req.excluded] + list(req.restrict)
    from app.analyst.scope import active, mask
    d = df[mask(df, scope)] if active(scope) else df
    if policy["duplicates"] == "drop":
        d = d.drop_duplicates()
    if policy["valid"]:
        base = d.assign(__row=range(len(d)))              # duplicates are kept as recorded unless dropped above
        ok, _ = eligibility(base, list(dict.fromkeys(cols)), historical=False)
        d = d[ok.values]
    by_col: dict = {}
    for c, v in req.entities:
        by_col.setdefault(c, []).append(str(v).lower())
    for c, vals in by_col.items():
        d = d[d[c].astype(str).str.lower().isin(vals)]
    for c, v in req.excluded:
        d = d[d[c].astype(str).str.lower() != str(v).lower()]
    for c in req.restrict:
        pos = roles[c].get("positive")
        if pos is None:
            return None, f"the restriction on {c} has no yes value"
        d = d[d[c].astype(str) == str(pos)]
    if req.period:
        if len(prof["dates"]) != 1:
            return None, "the period needs exactly one date column"
        t = pd.to_datetime(d[prof["dates"][0]], errors="coerce")
        d = d[(t >= pd.Timestamp(req.period["start"])) & (t <= pd.Timestamp(req.period["end"]))]
    if m is not None and roles[m]["role"] == "flag" and kind_of(df[m]) != "numeric":
        pos = roles[m].get("positive")
        if pos is None:
            return None, f"{m} has no yes value"
        x = (d[m].astype(str) == str(pos)).astype(float)
    elif m is not None:
        x = pd.to_numeric(d[m], errors="coerce")
    else:
        x = pd.Series(1.0, index=d.index)
    if req.group:
        d = d[d[req.group].notna()]
        x = x.loc[d.index]
    keys = d[req.group].astype(str) if req.group else None

    def per_group(f):
        if keys is None:
            return f(x, d)
        return {str(k): f(x.loc[g.index], g) for k, g in d.groupby(keys, sort=True)}

    q = req.quantity                    # no quantity word: the figure is not assumed (a why-only request, ...)
    if req.per and req.denominators and req.denominators != [intent._RECORDS]:  # noqa: SLF001
        if len(req.denominators) != 1:
            return None, "more than one denominator"
        den = req.denominators[0]
        if kind_of(df[den]) == "numeric":
            def f(xv, g):
                b = pd.to_numeric(g[den], errors="coerce").sum()
                return float(xv.sum() / b) if b else None
            how = f"sum of {m} ÷ sum of {den}"
        else:
            def f(xv, g):
                b = g[den].nunique()
                return float(xv.sum() / b) if b else None
            how = f"sum of {m} ÷ distinct {den}"
        return per_group(f), how
    fn = {"rate": "mean", "average": "mean", "median": "median", "total": "sum", "count": "count"}.get(q or "")
    if fn is None:
        return None, "the request names no quantity to recompute"
    if fn == "count":
        return per_group(lambda xv, g: float(len(g))), "number of records"
    if q == "rate" and not (m is not None and roles[m]["role"] in ("flag", "rate")):
        return None, "a rate of a column that is not yes/no"

    def g_(xv, _g):
        v = xv.dropna()
        return float(getattr(v, fn)()) if len(v) else None
    return per_group(g_), f"{fn} of {m}"


def check(message: str, df: pd.DataFrame, steps: list[dict], results: dict, scope: dict | None = None,
          policy: dict | None = None) -> dict:
    """{"status", "checks": [{check, ok, detail}]} for an executed computation in program-language steps (a program as
    run, or the steps an adapter below reads off another path's provenance). `results` are step summaries
    ({id: {"kind", "value", "hypothetical", ...}})."""
    prof = semantics.profile(df)
    req = intent.read(message, df, prof)
    checks = []
    probs = intent.check(steps, req, df, message)
    # a value that is also a column's name ("votes for Joly", Joly a column of votes and a value of winner) read as a
    # filter, where the question means the column
    from app.analyst.calc import named_values
    from app.analyst.scope import column_values
    as_cols = column_values(message, df, named_values(df, message))
    for s_ in steps:
        for c, vals in (s_.get("entities") or {}).items():
            for v in vals:
                if any(col_ == c and str(val_).lower() == str(v).lower() for col_, val_ in as_cols):
                    probs.append(f"'{v}' is the name of a column the question measures, not a value of {c} to filter on")
    # duplicates: the question's own rule (excluded only when asked) decides what the figure must be computed over
    from app.analyst.objectives import duplicate_policy
    if duplicate_policy(message) == "drop":
        if (policy or {}).get("duplicates", "drop") == "keep" and df.duplicated().any():
            probs.append("the question asks to exclude duplicates, but the calculation counted them")
        policy = {**(policy or {}), "duplicates": "drop"}
    probs = list(dict.fromkeys(probs))
    checks.append({"check": "the executed program meets the request (filters, grouping, measure, denominator, period, "
                            "exclusions)", "ok": not probs, "detail": "; ".join(probs) or "all requirements met"})
    # a scenario's table and everything computed from it must carry the hypothetical mark; a projection is identified
    # by its own kind ("projection") and wording, so it is not required to carry the mark
    hyp_steps = {s["id"] for s in steps if s["op"] == "scenario"}
    by = {s["id"]: s for s in steps}

    def derived_from_hyp(sid, depth=0):
        s = by.get(sid)
        if s is None or depth > 20:
            return False
        if sid in hyp_steps:
            return True
        return any(derived_from_hyp(s.get(k), depth + 1) for k in ("from", "a", "b", "base") if s.get(k))
    unmarked = [sid for sid in by if derived_from_hyp(sid) and sid in results and not results[sid].get("hypothetical")]
    # ... and an observed result is never presented as hypothetical
    overmarked = [sid for sid in by if sid in results and results[sid].get("hypothetical") and not derived_from_hyp(sid)
                  and by[sid]["op"] != "project"]
    checks.append({"check": "results derived from a scenario are marked hypothetical, and observed ones are not",
                   "ok": not unmarked and not overmarked,
                   "detail": ("unmarked hypothetical: " + ", ".join(unmarked) if unmarked else "")
                             + ("; " if unmarked and overmarked else "")
                             + ("observed but marked hypothetical: " + ", ".join(overmarked) if overmarked else "")
                             or "yes"})
    unmarked = unmarked + overmarked
    if probs or unmarked:
        return {"status": "failed", "checks": checks}
    implied = None
    if req.group is None:
        bys = {x.get("by") for x in steps if x["op"] == "aggregate" and x.get("by")}
        if len(bys) == 1:
            implied = next(iter(bys))       # "day totals": a breakdown the question implies but intent did not read
            req.group = implied
    want, how = recompute(req, df, prof, scope, policy)
    if want is None:
        checks.append({"check": "independent recomputation", "ok": True, "detail": f"not independently verified ({how})"})
        return {"status": "not independently verified", "checks": checks}
    got = [(sid, r.get("value")) for sid, r in results.items() if r.get("kind") in ("scalar", "series")
           and not str(sid).startswith(("_", "@"))]
    match = next((sid for sid, v in got if _same(want, v)), None)
    shown = want if not isinstance(want, dict) else {k: (round(v, 6) if isinstance(v, float) else v) for k, v in want.items()}
    checks.append({"check": f"independent recomputation ({how}"
                            + (f" by {req.group}" if req.group else "") + ", computed with pandas from the request)",
                   "ok": match is not None,
                   "detail": f"matches step '{match}'" if match else f"expected {shown}; no computed result matches"})
    if match and implied:
        checks.append({"check": "the breakdown asked for", "ok": True, "detail": f"not independently verified (the "
                       f"figures match a recomputation by {implied}, but the question's grouping was not recognised "
                       "independently)"})
        return {"status": "not independently verified", "checks": checks}
    return {"status": "verified" if match else "failed", "checks": checks}


# ------------------------------------------------------------------ every execution path --
# Each path that produces numbers is read into the same form -- program-language steps, their result summaries and
# the row policy the path applied -- and judged by check() above. One interface, no second copy of the logic.
_AGG = {"sum": "sum", "avg": "mean", "mean": "mean", "median": "median", "count": "count", "count_distinct": "nunique"}
# tools that describe the dataset itself (its columns, missing values, patterns): never the figure a question about a
# measure asks for
_DESCRIBES_DATA = {"describe_dataset", "missing_values", "understand", "patterns", "document_overview"}
_RE_PRESENTED = {"recall_result", "chart_of_previous_result", "chart_of_previous_answer", "compare_scenarios", "compare_results",
                 "investigation_finding"}


def _select_from_filters(filters) -> dict | None:
    """Tool-catalogue filters as a select step's arguments, or None when one has no program form."""
    sel: dict = {}
    for f in filters or []:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if op == "in":
            sel.setdefault("entities", {})[col] = [str(v) for v in val]
        elif op == "not_in":
            sel.setdefault("except", {})[col] = [str(v) for v in val]
        elif op == "==" and isinstance(val, str):
            sel.setdefault("entities", {}).setdefault(col, []).append(val)
        elif op == "!=" and isinstance(val, str):
            sel.setdefault("except", {}).setdefault(col, []).append(val)
        elif op in ("==", "!=", ">", ">=", "<", "<="):
            sel.setdefault("where", []).append({"column": col, "op": op, "value": val})
        else:
            return None
    return sel


def _scope_select(scope: dict | None) -> dict:
    """The records a path computed over because of the question's scope, as select arguments (named values,
    exclusions, a year column value); a calendar month/year is described for the record, as a date filter."""
    sel: dict = {}
    if not scope:
        return sel
    if scope.get("entities"):
        sel["entities"] = {c: list(v) for c, v in scope["entities"].items()}
    if scope.get("excluded_entities"):
        sel["except"] = {c: list(v) for c, v in scope["excluded_entities"].items()}
    for c, vals in (scope.get("numeric_eq") or {}).items():
        sel.setdefault("where", []).extend({"column": c, "op": "==", "value": v} for v in vals)
    return sel


def _tool_claim(call: dict, scope: dict | None) -> dict:
    tool, a, r = call.get("tool"), call.get("args") or {}, call.get("result") or {}
    pol = {"valid": False, "duplicates": "keep"}              # the tool catalogue counts every record as recorded
    if tool == "compute_metric" and a.get("aggregation") in _AGG:
        sel = _select_from_filters(a.get("filters"))
        if sel is None:
            return {"unverifiable": "a filter with no independent form"}
        sc = _scope_select(scope)
        for k, v in sc.items():
            if k == "where":
                sel.setdefault("where", []).extend(v)
            else:
                for c, vals in v.items():
                    sel.setdefault(k, {}).setdefault(c, []).extend(x for x in vals if x not in sel.get(k, {}).get(c, []))
        steps = ([{"id": "records", "op": "select", "from": "data", **sel}] if sel else [])
        agg = {"id": "result", "op": "aggregate", "from": "records" if sel else "data", "fn": _AGG[a["aggregation"]],
               **({"column": a["column"]} if a.get("column") else {}), **({"by": a["group_by"]} if a.get("group_by") else {})}
        val = ({str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r.get("value"))
        return {"steps": steps + [agg], "results": {"result": {"kind": "series" if isinstance(val, dict) else "scalar",
                                                              "value": val, "hypothetical": False}}, "policy": pol}
    if tool == "rank_groups" and r.get("rows"):
        sc = _scope_select(scope)
        steps = ([{"id": "records", "op": "select", "from": "data", **sc}] if sc else []) + [
            {"id": "result", "op": "aggregate", "from": "records" if sc else "data",
             "fn": "mean" if r.get("aggregation") == "mean" else "sum", "column": r["measure"], "by": r["group"]}]
        return {"partial": True, "steps": steps, "results": {"result": {"kind": "series", "hypothetical": False,
                "value": {str(x["group"]): x["value"] for x in r["rows"]}}}, "policy": pol}
    if tool == "compare_groups" and r.get("summary") and r.get("kind") == "numeric":
        steps = [{"id": "means", "op": "aggregate", "from": "data", "fn": "mean", "column": r["outcome"], "by": r["group"]},
                 {"id": "medians", "op": "aggregate", "from": "data", "fn": "median", "column": r["outcome"], "by": r["group"]}]
        res = {"means": {"kind": "series", "hypothetical": False, "value": {x["group"]: x["mean"] for x in r["summary"]}},
               "medians": {"kind": "series", "hypothetical": False, "value": {x["group"]: x["median"] for x in r["summary"]}}}
        return {"steps": steps, "results": res, "policy": {"valid": False, "duplicates": "keep", "dropna": True}}
    if tool == "describe_column" and r.get("n"):
        sel = _select_from_filters(a.get("filters"))
        if sel is not None:
            sc = _scope_select(scope)
            sel = {**sc, **sel}
            base = [{"id": "records", "op": "select", "from": "data", **sel}] if sel else []
            src = "records" if sel else "data"
            steps = base + [{"id": "mean", "op": "aggregate", "from": src, "fn": "mean", "column": r["column"]},
                            {"id": "median", "op": "aggregate", "from": src, "fn": "median", "column": r["column"]}]
            return {"steps": steps, "results": {"mean": {"kind": "scalar", "value": r.get("mean"), "hypothetical": False},
                                                "median": {"kind": "scalar", "value": r.get("median"),
                                                           "hypothetical": False}},
                    "policy": {"valid": False, "duplicates": "keep"}}
    return {"unverifiable": f"{tool} results (a statistical test, model or description) are not recomputed here",
            **({"other_analysis": tool} if tool in _DESCRIBES_DATA else {})}


def _calc_claim(memory: dict) -> dict:
    """The calculator's spec, when it is a plain aggregate of columns (no derived formula, no condition of its own
    beyond validity, duplicates and the question's scope)."""
    import re
    lc = memory.get("last_calc") or {}
    spec, r = lc.get("spec") or {}, lc.get("result") or {}
    used = set()
    for expr in list((spec.get("derive") or {}).values()) + list((spec.get("aggregate") or {}).values()):
        used |= set(re.findall(r"[A-Za-z_]\w*", str(expr)))
    reqs = {"columns": used, "conditions": [e.get("where", "") for e in spec.get("exclude") or []]}
    if spec.get("derive"):
        return {"unverifiable": "a derived measure (a formula over several columns)", "requirements": reqs}
    own = [e for e in spec.get("exclude") or [] if not re.search(r"duplicate\(\)|<\s*0\b|isnull\(|hours_between",
                                                                  e.get("where", ""))]
    if own and not all(e.get("scope") for e in own):
        return {"unverifiable": "a condition of the calculation's own (" + "; ".join(e.get("where", "") for e in own)[:80] + ")", "requirements": reqs}
    aggs = spec.get("aggregate") or {}
    prim = (lc.get("meta") or {}).get("primary")
    if prim and prim not in aggs:
        return {"unverifiable": f"a defined metric ({prim}) computed from several totals", "requirements": reqs}
    if prim and not re.fullmatch(r"\s*(sum|mean|avg|median|count)\(\s*[A-Za-z_]?\w*\s*\)\s*", str(aggs[prim])):
        return {"unverifiable": f"a defined metric ({prim}: {aggs[prim]})", "requirements": reqs}
    if prim:
        aggs = {prim: aggs[prim]}
    steps, res = [], {}
    by = spec.get("group_by")
    for name, expr in aggs.items():
        m = re.fullmatch(r"\s*(sum|mean|avg|median|count)\(\s*([A-Za-z_][\w]*)?\s*\)\s*", str(expr))
        if not m:
            continue
        fn = {"avg": "mean"}.get(m.group(1), m.group(1))
        steps.append({"id": name, "op": "aggregate", "from": "data", "fn": fn,
                      **({"column": m.group(2)} if m.group(2) and fn != "count" else {}), **({"by": by} if by else {})})
        val = ({str(g.get("group")): g.get(name) for g in r.get("groups") or []} if by else (r.get("totals") or {}).get(name))
        res[name] = {"kind": "series" if by else "scalar", "value": val, "hypothetical": False}
    if not steps:
        return {"unverifiable": "an aggregate with no independent form", "requirements": reqs}
    dup = "drop" if any("duplicate()" in e.get("where", "") for e in spec.get("exclude") or []) else "keep"
    return {"steps": steps, "results": res, "policy": {"valid": True, "duplicates": dup}}


def objective_checks(o: dict, df: pd.DataFrame | None) -> list[dict]:
    """Independent recomputation of a target / scenario objective where its figure has a direct form (a closure's
    totals, the amount completed so far towards a target on a plain column); the rest is reported as not
    independently verified."""
    from app.analyst.scope import eligibility
    nv = lambda why: [{"check": "independent recomputation", "ok": True,  # noqa: E731
                       "detail": f"not independently verified ({why})"}]
    if df is None:
        return nv("no data")
    p = o.get("params") or {}
    keep = p.get("duplicates", "keep") == "keep"

    def valid(d, cols, historical):
        base = d.assign(__row=range(len(d))) if keep else d
        ok, _ = eligibility(base, cols, historical=historical)
        return d[ok.values]
    if o["type"] == "closure" and not p.get("days"):
        m, col, who = p["measure"], p["entity_col"], p["entity"]
        d = valid(df if keep else df.drop_duplicates(), [m, col], False)
        normal = pd.to_numeric(d[m], errors="coerce").sum()
        closed = pd.to_numeric(d.loc[d[col].astype(str).str.lower() != str(who).lower(), m], errors="coerce").sum()
        good = _close(normal, o["cmp"]["b"]) and _close(closed, o["cmp"]["a"])
        return [{"check": "independent recomputation (total with and without the closed value, pandas)", "ok": good,
                 "detail": "matches" if good else f"expected {normal:.6g} and {closed:.6g}"}]
    if o["type"] == "feasibility" and not p["measure"]["steps"] and len(o["rows"]) == 1 and p.get("entity"):
        r = o["rows"][0]
        if "error" in r:
            return nv("the target was not computed")
        m, date = p["measure"]["column"], o["date"]
        d = valid(df if keep else df.drop_duplicates(), p["measure"]["inputs"], True)
        t = pd.to_datetime(d[date], errors="coerce")
        d = d[(d[p["entity_col"]].astype(str) == str(p["entity"])) & (t >= pd.Timestamp(p["start"]))
              & (t <= pd.Timestamp(o["as_of"]))]
        done = pd.to_numeric(d[m], errors="coerce").sum()
        good = _close(done, r["completed"])
        return [{"check": "independent recomputation of the amount completed so far (pandas)", "ok": good,
                 "detail": "matches; the projection itself is not independently verified" if good else
                 f"expected {done:.6g}, the program gave {r['completed']:.6g}"}]
    return nv({"closure": "a projection over a horizon", "absence": "a projection", "vs_rest": "a ranked comparison",
               "feasibility": "a derived measure or several people"}.get(o["type"], o["type"]))


def _requirement_problems(message: str, df: pd.DataFrame, reqs: dict) -> list[str]:
    import re
    from app.analyst.calc import named_values
    from app.analyst.scope import column_values
    req = intent.read(message, df)
    probs = []
    cols = {c for c in reqs.get("columns", set()) if c in df.columns}
    if req.measures and cols and not set(map(str, req.measures)) & set(map(str, cols)):
        probs.append(f"the request is about {' or '.join(map(str, req.measures))}, but the calculation uses "
                     f"{', '.join(sorted(map(str, cols)))}")
    for col, val in column_values(message, df, named_values(df, message)):
        if any(re.search(rf"['\"]{re.escape(str(val))}['\"]", c, re.I) for c in reqs.get("conditions", [])):
            probs.append(f"'{val}' is the name of a column the question measures, not a value of {col} to filter on")
    return probs


def check_answer(message: str, df: pd.DataFrame | None, out: dict, scope: dict | None, memory: dict) -> dict | None:
    """The one entry point: verification of whatever answered, by any path. None when the answer computed nothing
    (a clarification, a decline, a conversational reply). Program and objective answers carry their verification
    already (it decided whether they were shown); the other paths are read through their adapters here."""
    if df is None or out.get("plan_source") in ("conversation", "conversation_rules", "input_check"):
        return None
    ok_calls = [c for c in out.get("calls", []) if c.get("status") == "ok"]
    if not ok_calls:
        return None
    done = [c for c in out.get("verification") or [] if str(c.get("check", "")).startswith("independent recomputation")]
    if done:
        st = "failed" if not all(c["ok"] for c in out["verification"]) else (
            "not independently verified" if any("not independently verified" in c["detail"] for c in done) else "verified")
        return {"status": st, "checks": []}
    if all(c.get("tool") in _RE_PRESENTED for c in ok_calls):
        return {"status": "not independently verified", "checks": [{
            "check": "independent recomputation", "ok": True,
            "detail": "not independently verified (re-presents results computed earlier in this conversation)"}]}
    if out.get("plan_source") in ("metric", "model_calc", "refine"):
        claims = [_calc_claim(memory)]
    else:
        claims = [_tool_claim(ok_calls[0], scope)] + [
            {"unverifiable": f"a supporting analysis ({c.get('tool')}) shown with the headline"} for c in ok_calls[1:]]
    statuses, checks = [], []
    for cl in claims:
        if "requirements" in cl:
            # not recomputable, but still held to the request: the measure asked for is used, and no condition
            # filters on a value that is really a column's name
            probs = _requirement_problems(message, df, cl["requirements"])
            if probs:
                statuses.append("failed")
                checks.append({"check": "the calculation meets the request (measure, filters)", "ok": False,
                               "detail": "; ".join(probs)})
                continue
        if cl.get("other_analysis"):
            # the question asks for a figure Silt can recompute (a total, an average, a rate ...) but the answer is a
            # different analysis: not "unverified" -- it does not answer the question
            req = intent.read(message, df)
            want, how = recompute(req, df, None, scope)
            if want is not None:
                statuses.append("failed")
                checks.append({"check": "the answer computes what the question asks", "ok": False,
                               "detail": f"the question asks for {how}" + (f" by {req.group}" if req.group else "")
                                         + f", but the answer is a {cl['other_analysis'].replace('_', ' ')} result"})
                continue
        if "unverifiable" in cl:
            statuses.append("not independently verified")
            checks.append({"check": "independent recomputation", "ok": True,
                           "detail": f"not independently verified ({cl['unverifiable']})"})
            continue
        v = check(message, df, cl["steps"], cl["results"], scope=None if out.get("plan_source") in (
            "metric", "model_calc", "refine") and not scope else scope, policy=cl["policy"])
        if cl.get("partial") and v["status"] == "failed" and v["checks"][0]["ok"]:
            # a top-n ranking shows only some groups: its figures cannot all match the full breakdown
            v = {"status": "not independently verified", "checks": v["checks"][:2] + [{
                "check": "independent recomputation", "ok": True,
                "detail": "not independently verified (a ranking shows only some of the groups)"}]}
        statuses.append(v["status"])
        checks += v["checks"]
    status = "failed" if "failed" in statuses else ("verified" if statuses and all(x == "verified" for x in statuses)
                                                    else "not independently verified")
    return {"status": status, "checks": checks}


def duplicate_note(message: str, df: pd.DataFrame, scope: dict | None, out: dict) -> str | None:
    """One duplicate policy, stated on every path: exact duplicates among the records an answer used are counted as
    recorded (and flagged) unless the user asked to exclude them. None when there are none, the answer already says
    so, or the answer re-presents an earlier result."""
    import re
    from app.analyst import metrics
    from app.analyst.scope import active, mask
    if metrics._DUPES.search(message) or re.search(r"duplicat", out.get("answer", ""), re.I):  # noqa: SLF001
        return None
    if all(c.get("tool") in _RE_PRESENTED for c in out.get("calls", []) if c.get("status") == "ok"):
        return None
    if (out.get("plan") or {}).get("objective") in ("closure", "absence", "feasibility", "vs_rest"):
        return None                       # objectives list their own (counted as recorded / not counted)
    used = df[mask(df, scope)] if active(scope) else df
    dup = used.duplicated(keep="first")
    if not dup.any():
        return None
    pos = {i: n + 2 for n, i in enumerate(df.index)}
    lines = [pos[i] for i in dup[dup].index if i in pos][:10]
    return (f"_{int(dup.sum())} exact duplicate record(s) (line{'s' if len(lines) > 1 else ''} "
            f"{', '.join(map(str, lines))}) were counted as recorded. Ask to *exclude duplicates* if they should not "
            "count._")


def value_note(df: pd.DataFrame, scope: dict | None, out: dict) -> str | None:
    """Missing and impossible values in the column a plain aggregate used (compute_metric), stated rather than silently
    absorbed: a missing value is not counted; a negative amount is counted as recorded, and flagged. The analysis
    programs and objectives list their own."""
    from app.analyst.scope import _AMOUNT, active, mask  # noqa: PLC0415
    ok = [c for c in out.get("calls", []) if c.get("status") == "ok"]
    if df is None or not ok or any(c.get("tool") != "compute_metric" or (c.get("args") or {}).get("filters")
                                   for c in ok):
        return None                     # a tool's own filters: the records it used are not known here
    cols = {str((c.get("args") or {}).get("column")) for c in ok} & set(map(str, df.columns))
    used = df[mask(df, scope)] if active(scope) else df
    if used.empty:
        return None
    pos = {i: n + 2 for n, i in enumerate(df.index)}
    parts = []
    for col in sorted(cols):
        v = pd.to_numeric(used[col], errors="coerce")
        miss = used.index[used[col].isna()]
        if len(miss):
            parts.append(f"{len(miss)} record(s) with no {col.replace('_', ' ')} (line{'s' if len(miss) > 1 else ''} "
                         f"{', '.join(str(pos[i]) for i in miss[:10])}) were not counted")
        neg = used.index[v.lt(0).fillna(False)]
        if len(neg) and _AMOUNT.search(col):
            parts.append(f"{len(neg)} record(s) with a negative {col.replace('_', ' ')} (line{'s' if len(neg) > 1 else ''} "
                         f"{', '.join(str(pos[i]) for i in neg[:10])}) were counted as recorded — it may be a refund or "
                         "an error")
    return ("_" + "; ".join(parts) + "._") if parts else None


def unanswered_parts(message: str, df: pd.DataFrame | None, out: dict) -> list[str]:
    """Parts of a multi-part question the answer does not cover (a program answer lists its own in its checklist):
    'and why' with no associations, 'any unusual values' with no anomaly check."""
    if df is None or "What you asked, and where it is answered" in out.get("answer", ""):
        return []
    req = intent.read(message, df)
    tools_used = {c.get("tool") for c in out.get("calls", []) if c.get("status") == "ok"}
    text = out.get("answer", "")
    missing = []
    if req.why and not tools_used & {"associations", "correlation", "drivers", "feature_importance", "linear_regression",
                                     "compare_groups", "investigation_finding"} and "associated with" not in text:
        missing.append("what is associated with it (the 'why' part)")
    if req.unusual and not tools_used & {"outliers", "patterns"} and "unusual" not in text.lower():
        missing.append("unusual values")
    return missing
