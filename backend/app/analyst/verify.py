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

import numpy as np
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
    if len(req.measures) != 1 and req.quantity != "count" and not (req.event and not req.measures):
        return None, "the request has no single measure to recompute"
    m = req.measures[0] if req.measures else None
    if m is not None and m not in df.columns:
        return None, f"{m} is not a column"
    from app.analyst.scope import eligibility
    cols = [c for c in [m, req.group, *req.denominators] if c and c in df.columns]
    cols += [c for c, _ in req.entities + req.excluded] + list(req.restrict) + list(req.restrict_not)
    from app.analyst.scope import active, mask
    if req.share_of and scope:
        # the values the question names are the PART of the share; the whole is every record of the period
        scope = {k: v for k, v in scope.items() if k in ("time", "numeric_eq")}
    d = df[mask(df, scope)] if active(scope) else df
    if policy["duplicates"] == "drop":
        d = d.drop_duplicates()
    if policy["valid"]:
        base = d.assign(__row=range(len(d)))              # duplicates are kept as recorded unless dropped above
        ok, _ = eligibility(base, list(dict.fromkeys(cols)), historical=False)
        d = d[ok.values]
    if req.unresolved:
        return None, f"the limit “{req.unresolved[0]}” could not be tied to a column"
    if req.period:
        if req.period.get("unreadable"):
            return None, req.period["unreadable"]
        if len(prof["dates"]) != 1:
            return None, "the period needs exactly one date column"
        t = pd.to_datetime(d[prof["dates"][0]], errors="coerce").dt.normalize()
        if req.period.get("start"):
            d = d[t >= pd.Timestamp(req.period["start"])]
            t = t.loc[d.index]
        if req.period.get("end"):
            d = d[t <= pd.Timestamp(req.period["end"])]
    whole = d                                           # the records a share is taken of: scope and period only
    by_col: dict = {}
    for c, v in req.entities:
        by_col.setdefault(c, []).append(str(v).lower())
    for c, vals in by_col.items():
        d = d[d[c].astype(str).str.lower().isin(vals)]
    for c, v in req.excluded:
        d = d[d[c].astype(str).str.lower() != str(v).lower()]
    for c, yes in [(c, True) for c in req.restrict] + [(c, False) for c in req.restrict_not]:
        pair = intent.flag_values(df, c, roles[c])
        if pair is None:
            return None, f"the restriction on {c} has no yes/no values"
        d = d[intent.flag_mask(d, c, pair, yes)]
    for cd in req.conditions:
        x_ = pd.to_numeric(d[cd["column"]], errors="coerce")
        d = d[{"==": x_ == cd["value"], "!=": x_ != cd["value"], ">": x_ > cd["value"], ">=": x_ >= cd["value"],
               "<": x_ < cd["value"], "<=": x_ <= cd["value"]}[cd["op"]].fillna(False)]
    if req.share_of:
        if req.share_of not in df.columns:
            return None, f"{req.share_of} is not a column"
        total = pd.to_numeric(whole[req.share_of], errors="coerce").sum()
        if not total:
            return None, f"the total of {req.share_of} is zero"
        part = pd.to_numeric(d[req.share_of], errors="coerce")
        how = f"sum of {req.share_of} in the records named ÷ sum of {req.share_of} overall"
        if req.group:
            g = d[d[req.group].notna()]
            return {str(k): float(part.loc[x.index].sum() / total) for k, x in g.groupby(g[req.group].astype(str))}, how
        return float(part.sum() / total), how
    if req.event is not None and m is None:
        ec, ev = req.event
        if ec not in d.columns:
            return None, f"{ec} is not a column"
        x = (d[ec].astype(str).str.strip().str.lower() == str(ev).lower()).astype(float)
        x[d[ec].isna()] = float("nan")
        if req.group:
            d = d[d[req.group].notna()]
            x = x.loc[d.index]
        keys_ = d[req.group].astype(str) if req.group else None
        how = f"share of records where {ec} = {ev}"
        if keys_ is None:
            return (float(x.dropna().mean()) if x.notna().any() else None), how
        return {str(k): float(x.loc[g.index].dropna().mean()) for k, g in d.groupby(keys_, sort=True)}, how
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
        if kind_of(df[den]) == "numeric" and roles[den]["role"] not in ("identifier", "person_name"):
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
        if m is not None and roles[m]["role"] == "count" and intent.ranking_aggregate(m, m) == "sum":
            # "how many reviews does each type have" with a column of review counts: their sum, not the rows
            return per_group(lambda xv, g: float(xv.dropna().sum())), f"sum of {m}"
        if req.counted and req.counted in d.columns:
            return per_group(lambda xv, g: float(g[req.counted].nunique())), f"distinct {req.counted}"
        return per_group(lambda xv, g: float(len(g))), "number of records"
    if q == "rate" and not (m is not None and roles[m]["role"] in ("flag", "rate")):
        return None, "a rate of a column that is not yes/no"

    def g_(xv, _g):
        v = xv.dropna()
        return float(getattr(v, fn)()) if len(v) else None
    return per_group(g_), f"{fn} of {m}"


def check(message: str, df: pd.DataFrame, steps: list[dict], results: dict, scope: dict | None = None,
          policy: dict | None = None, request=None) -> dict:
    """{"status", "checks": [{check, ok, detail}]} for an executed computation in program-language steps (a program as
    run, or the steps an adapter below reads off another path's provenance). `results` are step summaries
    ({id: {"kind", "value", "hypothetical", ...}})."""
    prof = semantics.profile(df)
    import copy
    # a follow-up ("What about Ben Cole?") is checked against the reading it was built from -- the previous answer's,
    # with the follow-up's change applied -- never against its few words alone
    req = copy.deepcopy(request) if request is not None else intent.read(message, df, prof)
    checks = []
    probs = intent.check(steps, req, df, message if request is None else "")
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
    if want is None and req.quantity is None and req.extreme in ("max", "min", "either") and req.group \
            and len(req.measures) == 1 and not req.per and not req.denominators:
        # "the highest revenue region": the groups are compared by ONE aggregate, fixed by the question and the measure
        # (intent.ranking_aggregate -- the same rule the planners follow); the figures must be exactly that aggregate.
        # A question that leaves it open is not verified, whichever aggregate the answer happens to match.
        agg = intent.ranking_aggregate(message, req.measures[0])
        if agg in ("sum", "mean", "median"):
            req.quantity = {"sum": "total", "mean": "average", "median": "median"}[agg]
            w, h = recompute(req, df, prof, scope, policy)
            got = [v for sid, r in results.items() if r.get("kind") == "series" and not str(sid).startswith(("_", "@"))
                   for v in [r.get("value")]]
            if w is not None and any(_same(w, v) for v in got):
                want, how = w, h + " — how the question compares the groups"
            else:
                # not the aggregate the question implies: possibly a quantity Silt did not read from the question (a
                # ratio, "for the clinicians it has") -- not verified, and not passed off as a ranking it is not
                req.quantity = None
                how = f"the figures are not the {h} by {req.group} that the question compares the groups by"
        elif agg is None:
            how = "the question does not say whether the groups are compared by their total or their average"
        else:
            how = "each group's single largest or smallest record is not recomputed here"
    if want is None and len(req.measures) > 1 and req.group is None:
        # "total revenue, total cost and average revenue": every figure the question lists, each read from its own
        # part of the question and recomputed on its own; all of them must be among the results
        parts = _recompute_parts(message, df, prof, scope, policy)
        if parts:
            got = [(sid, r.get("value")) for sid, r in results.items() if r.get("kind") == "scalar"
                   and not str(sid).startswith(("_", "@"))]
            miss = [(h, w) for w, h in parts if not any(_same(w, v) for _, v in got)]
            checks.append({"check": "independent recomputation of each figure asked for (" + "; ".join(h for _, h in parts)
                                    + ", computed with pandas from the request)", "ok": not miss,
                           "detail": "every figure matches" if not miss else
                           "; ".join(f"expected {h} = {w:,.6g}; no computed result matches" for h, w in miss)})
            return {"status": "failed" if miss else "verified", "checks": checks}
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


def _recompute_parts(message: str, df: pd.DataFrame, prof: dict, scope: dict | None, policy: dict | None):
    """[(value, description)] for a question that lists several figures ("total revenue, total cost and average
    revenue"): each comma / "and" part read by the intent reader on its own -- a quantity word carries over to the
    measures after it -- and recomputed. None unless every part with a measure gives exactly one figure."""
    import re
    out, quantity = [], None
    whole = intent.read(message, df, prof)
    for part in re.split(r",|;|\band\b|&", message, flags=re.I):
        r = intent.read(part, df, prof)
        if r.quantity:
            quantity = r.quantity
        if not r.measures:
            continue
        if len(r.measures) != 1 or quantity not in ("total", "average", "median") or r.group:
            return None
        r.quantity = quantity
        r.entities, r.excluded = whole.entities, whole.excluded
        w, h = recompute(r, df, prof, scope, policy)
        if w is None or isinstance(w, dict):
            return None
        out.append((w, h))
    return out if len(out) >= 2 else None


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
    for w in scope.get("where") or []:
        sel.setdefault("where", []).append({"column": w["column"], "op": w["op"], "value": w["value"]})
    if scope.get("period"):
        sel["period"] = {k: scope["period"][k] for k in ("column", "start", "end")}
    return sel


def _tool_claim(call: dict, scope: dict | None) -> dict:
    tool, a, r = call.get("tool"), call.get("args") or {}, call.get("result") or {}
    pol = {"valid": True, "duplicates": "keep"}               # the one validity policy (scope.valid_records)
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
                    "policy": {"valid": True, "duplicates": "keep"}}
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
    if prim and not re.fullmatch(r"\s*(sum|mean|avg|median|count)\(\s*(?:[A-Za-z_]?\w*|\*)\s*\)\s*", str(aggs[prim])):
        return {"unverifiable": f"a defined metric ({prim}: {aggs[prim]})", "requirements": reqs}
    if prim:
        aggs = {prim: aggs[prim]}
    steps, res = [], {}
    by = spec.get("group_by")
    for name, expr in aggs.items():
        m = re.fullmatch(r"\s*(sum|mean|avg|median|count)\(\s*([A-Za-z_][\w]*|\*)?\s*\)\s*", str(expr))
        if not m:
            continue
        fn = {"avg": "mean"}.get(m.group(1), m.group(1))
        steps.append({"id": name, "op": "aggregate", "from": "data", "fn": fn,
                      **({"column": m.group(2)} if m.group(2) and m.group(2) != "*" and fn != "count" else {}),
                      **({"by": by} if by else {})})
        val = ({str(g.get("group")): g.get(name) for g in r.get("groups") or []} if by else (r.get("totals") or {}).get(name))
        res[name] = {"kind": "series" if by else "scalar", "value": val, "hypothetical": False}
    if not steps:
        return {"unverifiable": "an aggregate with no independent form", "requirements": reqs}
    dup = "drop" if any("duplicate()" in e.get("where", "") for e in spec.get("exclude") or []) else "keep"
    # the calculator says which suspicious records it counted as recorded (negative amounts: refunds, returns); the
    # recomputation counts the same records, or a correct figure would be judged against a different set of records
    kept_negative = any(str(c.get("issue", "")).startswith("negative ") for c in r.get("data_checks") or [])
    return {"steps": steps, "results": res, "policy": {"valid": not kept_negative, "duplicates": dup}}


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


_OWN_SCOPE = {"bands", "understand", "describe_dataset", "missing_values", "patterns", "reconcile_profit",
              "column_change", "budget_scenario", "predict", "random_forest", "feature_importance", "scenario",
              "what_if", "document_overview", "document_search"}


def _stated_unapplied(message: str, df: pd.DataFrame, ok_calls: list, scope: dict | None, out: dict,
                      memory: dict, request=None) -> list[str]:
    """Restrictions the question states (a threshold, a yes/no filter, a share of a measure) that the answer shows no
    sign of having applied -- read from what was executed: the calls' arguments and filters, the program's select
    steps, the calculator's rules and the scope. One check for every path; a path that carries its own program
    verification (intent.check) was already held to the same list there."""
    import json
    try:
        req = request if request is not None else intent.read(message, df)
    except Exception:  # noqa: BLE001
        return []
    if req.hypothetical or any(c.get("tool") in _OWN_SCOPE for c in ok_calls):
        return []
    if any(c.get("tool") != "analysis_program" for c in ok_calls):
        mism = _measure_mismatch(req, ok_calls, out, memory, df)
        if mism:
            return [mism]
    if not (req.conditions or req.restrict or req.restrict_not or req.unresolved or req.share_of or req.period
            or req.event):
        return []
    if any(c.get("tool") == "analysis_program" for c in ok_calls):
        return []                                       # checked step by step by intent.check before it was shown
    lc = memory.get("last_calc") or {} if out.get("plan_source") in ("metric", "model_calc", "refine") else {}
    spec = lc.get("spec") or {}
    executed = json.dumps([{"tool": c.get("tool"), "args": c.get("args")} for c in ok_calls], default=str).lower()
    executed += " " + json.dumps({k: spec.get(k) for k in ("derive", "exclude", "aggregate")}, default=str).lower()
    executed += " " + json.dumps({k: v for k, v in (scope or {}).items() if k in (
        "entities", "excluded_entities", "numeric_eq", "where", "flags")}, default=str).lower()
    calc_text = json.dumps(spec.get("exclude") or [], default=str).lower()
    if req.period and not req.period.get("unreadable") and spec and str(req.period.get("start") or req.period.get("end")) \
            not in calc_text:
        return [f"the question is limited to {req.period.get('label')}, which the calculation did not apply"]
    out_ = [f"the question's limit “{t}” could not be tied to a column and was not applied" for t in req.unresolved]
    for cd in req.conditions:
        num = f"{cd['value']:g}"
        if str(cd["column"]).lower() not in executed or num not in executed:
            out_.append(f"the question limits the records to {cd['column']} {cd['op']} {num}, which was not applied")
    for c in req.restrict:
        if f'"{str(c).lower()}"' not in executed and str(c).lower() not in executed:
            out_.append(f"the question is about records where {c} applies only, which was not applied")
    for c in req.restrict_not:
        if str(c).lower() not in executed:
            out_.append(f"the question is about records where {c} does not apply, which was not applied")
    if req.event and not req.measures:
        ec, ev = req.event
        measured = json.dumps({k: spec.get(k) for k in ("derive", "aggregate")}, default=str).lower() if spec else \
            json.dumps([c.get("args") for c in ok_calls], default=str).lower()
        if str(ev).lower() not in measured or str(ec).lower() not in measured:
            out_.append(f"the question asks for the share of records where {ec} = {ev}, but the answer measured "
                        "something else")
    if req.share_of and not (spec and str(req.share_of).lower() in executed and "/" in executed):
        out_.append(f"the question asks for a share of {req.share_of} (a part of its total), and the answer is not "
                    "that share")
    return out_


def _measure_mismatch(req, ok_calls: list, out: dict, memory: dict, df: pd.DataFrame) -> str | None:
    """The answer computed a different measure from the one the question asks about (the structured request's single
    measure), or counted records where the question counts distinct values. One check over every path's executed
    calls; a path that derives its figure from several columns (a calculation, a defined metric) is checked by the
    columns it used."""
    import re
    used: set = set()
    counted_records = False
    for c in ok_calls:
        a, r = c.get("args") or {}, c.get("result") or {}
        t = c.get("tool")
        if t == "compute_metric":
            if a.get("column"):
                used.add(a["column"])
            if a.get("aggregation") == "count" and not a.get("column"):
                counted_records = True
        elif t == "describe_column" and a.get("column"):
            used.add(a["column"])
        elif t == "rank_groups" and r.get("measure"):
            used.add(r["measure"])
        elif t == "compare_groups" and a.get("outcome"):
            used.add(a["outcome"])
        elif t == "correlation":
            used |= {a.get("x"), a.get("y")} - {None}
        elif t == "calculator":
            lc = memory.get("last_calc") or {}
            spec = lc.get("spec") or {}
            for expr in list((spec.get("derive") or {}).values()) + list((spec.get("aggregate") or {}).values()):
                used |= set(re.findall(r"[A-Za-z_]\w*", str(expr)))
        else:
            return None                                 # a tool this check has no reading of
    used = {u for u in used if u in df.columns}
    groups = set()
    for c in ok_calls:
        a, r = c.get("args") or {}, c.get("result") or {}
        groups |= {a.get("group_by"), a.get("group"), r.get("group"), r.get("group_by")} - {None}
    if any(c.get("tool") == "calculator" for c in ok_calls):
        groups.add(((memory.get("last_calc") or {}).get("spec") or {}).get("group_by"))
    groups.discard(None)
    group_role = semantics.column_role(df, req.group)["role"] if req.group and req.group in df.columns else None
    if req.group and group_role not in (None, "flag") and not req.relation and groups.isdisjoint({req.group}) \
            and not any(c == req.group for c, _ in req.entities) and any(
                c.get("tool") in ("compute_metric", "calculator", "rank_groups") for c in ok_calls):
        return f"the question breaks the result down by {req.group}, but the answer does not"
    if req.quantity == "count" and req.counted and req.counted in df.columns and counted_records:
        return (f"the question counts distinct {req.counted} values, but the answer counted records "
                f"({int(df[req.counted].nunique()):,} distinct vs {len(df):,} records overall)")
    if len(req.measures) == 1 and not req.share_of and not req.per and used and req.measures[0] not in used:
        return (f"the question is about {req.measures[0]}, but the answer computed "
                f"{', '.join(sorted(map(str, used)))}")
    return None


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
    request = out.get("_request")
    stated = _stated_unapplied(message, df, ok_calls, scope, out, memory, request)
    if stated:
        return {"status": "failed", "checks": [{"check": "every restriction the question states was applied",
                                                "ok": False, "detail": "; ".join(stated)}]}
    if out.get("plan_source") in ("metric", "model_calc", "refine"):
        claims = [_calc_claim(memory)]
    elif ok_calls[0].get("tool") in ("correlation", "rank_groups", "understand", "bands", "describe_column",
                                     "compare_groups") \
            and _directly_checkable(ok_calls[0], df) and not (ok_calls[0].get("args") or {}).get("filters"):
        claims = [{"direct": ok_calls[0]}] + [
            {"unverifiable": f"a supporting analysis ({c.get('tool')}) shown with the headline"} for c in ok_calls[1:]]
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
        if "direct" in cl:
            st_, ch_ = _direct_check(cl["direct"], df, scope)
            statuses.append(st_)
            checks += ch_
            continue
        if "unverifiable" in cl:
            statuses.append("not independently verified")
            checks.append({"check": "independent recomputation", "ok": True,
                           "detail": f"not independently verified ({cl['unverifiable']})"})
            continue
        v = check(message, df, cl["steps"], cl["results"], scope=None if out.get("plan_source") in (
            "metric", "model_calc", "refine") and not scope else scope, policy=cl["policy"], request=request)
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


def _directly_checkable(call: dict, df: pd.DataFrame) -> bool:
    """The result names the columns its figures came from, and they are in the data."""
    r, t = call.get("result") or {}, call.get("tool")
    need = {"correlation": ("x", "y"), "rank_groups": ("group", "measure"), "bands": ("column",),
            "describe_column": ("column",), "understand": (), "compare_groups": ("outcome", "group")}.get(t)
    if need is None or not r:
        return False
    return all(r.get(k) in df.columns for k in need) and (t != "understand" or "rows" in r)


def _direct_check(call: dict, df: pd.DataFrame, scope: dict | None) -> tuple[str, list[dict]]:
    """Recompute a correlation (Pearson r, Spearman rho, n) or a top-n ranking (each shown group's value, and the
    order) directly with pandas / scipy from the records in scope, and compare every figure shown."""
    import math

    from scipy import stats as _st

    from app.analyst.metrics import never_negative
    from app.analyst.scope import active, mask
    d = df[mask(df, scope)] if active(scope) else df
    r = call["result"]
    out = []

    def possible(frame, col):
        """The validity policy, applied here on its own terms: a negative value of a column that can never be
        negative is not a possible record."""
        if col in frame.columns and never_negative(col):
            return frame[~(pd.to_numeric(frame[col], errors="coerce") < 0)]
        return frame

    def same(a, b, tol=1e-9):
        return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tol)
    if call["tool"] == "understand":
        # a profile of the table: its record, column, missing-cell and duplicate counts, recomputed
        for label, shown, want in (("records", r.get("rows"), len(d)), ("columns", r.get("columns"), d.shape[1]),
                                   ("missing cells", r.get("missing_cells"), int(d.isna().sum().sum())),
                                   ("duplicated rows", r.get("duplicates"), int(d.duplicated().sum()))):
            out.append({"check": f"independent recomputation: {label}", "ok": same(shown, want),
                        "detail": f"{label}: shown {shown}, recomputed {want}"})
        return ("verified" if all(c["ok"] for c in out) else "failed"), out
    if call["tool"] == "bands":
        from app.analyst import bands as _b
        x = pd.to_numeric(d[r["column"]], errors="coerce")
        for b, row in zip(r["bands"], r["rows"]):
            m = _b.mask(x, b)
            out.append({"check": f"independent recomputation: {row['band']}", "ok": int(m.sum()) == row["records"],
                        "detail": f"{row['band']}: shown {row['records']}, recomputed {int(m.sum())}"})
            if r.get("entity"):
                dn = int(d.loc[m, r["entity"]].nunique())
                out.append({"check": f"independent recomputation: distinct {r['entity']} {row['band']}",
                            "ok": dn == row.get("distinct"), "detail": f"shown {row.get('distinct')}, recomputed {dn}"})
        return ("verified" if all(c["ok"] for c in out) else "failed"), out
    if call["tool"] == "compare_groups":
        return _check_group_comparison(r, possible(d, r["outcome"]) if r.get("kind") == "numeric" else d, same)
    if call["tool"] == "describe_column":
        d = possible(d, r["column"])
        x = pd.to_numeric(d[r["column"]], errors="coerce").dropna()
        vc = x.value_counts()
        modes = sorted(float(v) for v in vc[vc == vc.max()].index) if len(vc) and vc.max() > 1 else []
        for label, shown, want in (("n", r.get("n"), len(x)), ("mean", r.get("mean"), float(x.mean())),
                                   ("median", r.get("median"), float(x.median())),
                                   ("standard deviation", r.get("sd"), float(x.std(ddof=1))),
                                   ("minimum", r.get("min"), float(x.min())), ("maximum", r.get("max"), float(x.max()))):
            out.append({"check": f"independent recomputation: {label}", "ok": same(shown, want, 1e-9),
                        "detail": f"{label}: shown {shown}, recomputed {want}"})
        if "mode" in r:
            out.append({"check": "independent recomputation: mode", "ok": list(r["mode"]) == modes,
                        "detail": f"mode: shown {r['mode']}, recomputed {modes}"})
        return ("verified" if all(c["ok"] for c in out) else "failed"), out
    if call["tool"] == "correlation":
        if r.get("note") and r.get("n") is None:
            return "not independently verified", [{"check": "independent recomputation", "ok": True,
                                                   "detail": "not independently verified (no correlation was computed)"}]
        x = pd.to_numeric(d[r["x"]], errors="coerce")
        y = pd.to_numeric(d[r["y"]], errors="coerce")
        ok_ = x.notna() & y.notna()
        n = int(ok_.sum())
        pr = _st.pearsonr(x[ok_], y[ok_])
        sr = _st.spearmanr(x[ok_], y[ok_])
        for label, shown, want in (("n", r.get("n"), n), ("Pearson r", r.get("pearson_r"), float(pr.statistic)),
                                   ("Pearson p", r.get("pearson_p"), float(pr.pvalue)),
                                   ("Spearman rho", r.get("spearman_rho"), float(sr.statistic)),
                                   ("Spearman p", r.get("spearman_p"), float(sr.pvalue))):
            good = same(shown, want, tol=1e-9 if label == "n" else 1e-6)
            out.append({"check": f"independent recomputation: {label}", "ok": good,
                        "detail": f"{label} shown {shown}, recomputed {want}"})
    else:
        fn = r.get("aggregation") or "sum"
        d = possible(d, r["measure"])
        v = pd.to_numeric(d[r["measure"]], errors="coerce")
        full = v.groupby(d[r["group"]].astype(str)).agg(fn).dropna()
        for x_ in r["rows"]:
            want = full.get(str(x_["group"]))
            out.append({"check": f"independent recomputation: {x_['group']}", "ok": same(x_["value"], want),
                        "detail": f"{fn} of {r['measure']} for {x_['group']}: shown {x_['value']}, recomputed {want}"})
        for c, ea in (r.get("extras") or {}).items():
            ev = pd.to_numeric(d[c], errors="coerce").groupby(d[r["group"]].astype(str)).agg(ea)
            for x_ in r["rows"]:
                shown, want = (x_.get("extras") or {}).get(c), ev.get(str(x_["group"]))
                good = (shown is None and pd.isna(want)) or same(shown, want)
                out.append({"check": f"independent recomputation: {c} of {x_['group']}", "ok": bool(good),
                            "detail": f"{ea} of {c} for {x_['group']}: shown {shown}, recomputed {want}"})
        ranked = full.sort_values(ascending=bool(r.get("low")))
        cut = ranked.iloc[len(r["rows"]) - 1] if len(ranked) >= len(r["rows"]) and r["rows"] else None
        # the groups shown are the top (or bottom) ones of the full ranking -- ties at the cut may be any of them
        beyond = ranked.iloc[len(r["rows"]):]
        better = (beyond > cut) if not r.get("low") else (beyond < cut)
        order_ok = cut is not None and not bool(better.any())
        out.append({"check": "independent recomputation: the ranking", "ok": order_ok,
                    "detail": f"the {len(r['rows'])} groups shown are the {'lowest' if r.get('low') else 'highest'} "
                              f"of {len(full):,} by {fn} of {r['measure']}" if order_ok else
                              "a group outside those shown has a better value"})
    return ("verified" if all(c["ok"] for c in out) else "failed"), out


def _check_group_comparison(r: dict, d: pd.DataFrame, same) -> tuple[str, list[dict]]:
    """A group comparison recomputed from first principles -- pandas for each group's figures, and the test statistics
    from their formulas (not the library calls the comparison tool used):
      two groups      Welch's t (statistic, df, two-sided p, confidence interval of the difference) and Hedges' g
      several groups  Welch's one-way F (statistic, both df, p), Kruskal-Wallis H from ranks with the tie
                      correction (p), and eta squared
      categories      each group's shares, the chi-square statistic from observed and expected counts (df, p) and
                      Cramer's V; Fisher's exact p summed over the hypergeometric distribution.
    Every figure the answer shows must match; any that cannot be recomputed makes the answer unverified."""
    import math

    from scipy import stats as _st
    out: list[dict] = []

    def add(label, shown, want, tol=1e-6):
        ok = shown is None and want is None or (shown is not None and want is not None and not (
            isinstance(want, float) and math.isnan(want)) and math.isclose(float(shown), float(want), rel_tol=tol,
                                                                                abs_tol=tol))
        out.append({"check": f"independent recomputation: {label}", "ok": bool(ok),
                    "detail": f"{label}: shown {shown}, recomputed {want}"})
    if r.get("note") or not r.get("summary"):
        return "not independently verified", [{"check": "independent recomputation", "ok": True,
                                               "detail": "not independently verified (no comparison was computed)"}]
    y, g = r["outcome"], r["group"]
    data = d[[y, g]].dropna()
    if r.get("kind") == "numeric":
        vals = pd.to_numeric(data[y], errors="coerce")
        groups = {str(k): vals.loc[v.index].dropna().to_numpy(dtype=float)
                  for k, v in data.groupby(data[g].astype(str), sort=True)}
        for row in r["summary"]:
            x = groups.get(str(row["group"]))
            if x is None:
                out.append({"check": f"independent recomputation: {row['group']}", "ok": False,
                            "detail": f"group {row['group']} is not in the records"})
                continue
            add(f"n of {row['group']}", row["n"], float(len(x)), 1e-12)
            add(f"mean of {row['group']}", row["mean"], float(x.mean()), 1e-9)
            add(f"median of {row['group']}", row["median"], float(np.median(x)), 1e-9)
            add(f"sd of {row['group']}", row["sd"], float(x.std(ddof=1)) if len(x) > 1 else None, 1e-9)
        t = r.get("test") or {}
        names = list(groups)
        if t.get("name") == "Welch's t-test" and len(names) == 2:
            a_, b_ = groups[names[0]], groups[names[1]]
            va, vb, na, nb = a_.var(ddof=1), b_.var(ddof=1), len(a_), len(b_)
            se = math.sqrt(va / na + vb / nb)
            diff = b_.mean() - a_.mean()
            tt = diff / se
            dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
            p_ = 2 * _st.t.sf(abs(tt), dof)
            half = _st.t.ppf(0.975, dof) * se
            sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
            gg = diff / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp else None
            add("difference in means", abs(t.get("estimate")), abs(diff))
            add("Welch t (size)", abs(t.get("statistic")), abs(tt))
            add("Welch p-value", t.get("p_value"), p_)
            if t.get("ci"):
                add("confidence interval width", t["ci"][1] - t["ci"][0], 2 * half)
            if t.get("hedges_g") is not None and gg is not None:
                add("Hedges' g (size)", abs(t["hedges_g"]), abs(gg))
        elif t.get("name") and len(names) >= 3:
            arr = [groups[k] for k in names]
            k = len(arr)
            n = np.array([len(x) for x in arr], float)
            m = np.array([x.mean() for x in arr])
            w = n / np.array([x.var(ddof=1) for x in arr])
            grand = (w * m).sum() / w.sum()
            lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
            f_ = ((w * (m - grand) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k * k - 1) * lam)
            df2 = (k * k - 1) / (3 * lam)
            add("Welch F", t.get("statistic"), f_)
            add("Welch p-value", t.get("p_value"), _st.f.sf(f_, k - 1, df2))
            allv = np.concatenate(arr)
            sst = ((allv - allv.mean()) ** 2).sum()
            if t.get("eta_squared") is not None:
                add("eta squared", t["eta_squared"], sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in arr) / sst)
            rb = t.get("robustness") or {}
            if rb.get("p_value") is not None:
                ranks = pd.Series(allv).rank().to_numpy()
                cuts = np.cumsum([0] + [len(x) for x in arr])
                N = len(allv)
                h = 12 / (N * (N + 1)) * sum(ranks[cuts[i]:cuts[i + 1]].sum() ** 2 / len(arr[i]) for i in range(k)) \
                    - 3 * (N + 1)
                ties = pd.Series(allv).value_counts().to_numpy(dtype=float)
                h /= 1 - (ties ** 3 - ties).sum() / (N ** 3 - N)
                add("Kruskal-Wallis p-value", rb["p_value"], _st.chi2.sf(h, k - 1))
        elif t:
            out.append({"check": "independent recomputation: the test", "ok": True,
                        "detail": f"not independently verified ({t.get('name')} is not recomputed here)"})
    else:
        table = pd.crosstab(data[g].astype(str), data[y].astype(str))
        for row in r["summary"]:
            if str(row["group"]) not in table.index:
                out.append({"check": f"independent recomputation: {row['group']}", "ok": False,
                            "detail": f"group {row['group']} is not in the records"})
                continue
            tr = table.loc[str(row["group"])]
            add(f"n of {row['group']}", row["n"], float(tr.sum()), 1e-12)
            for c in table.columns:
                if f"share_{c}" in row:
                    add(f"share of {c} in {row['group']}", row[f"share_{c}"], float(tr[c] / tr.sum()), 1e-9)
        t = r.get("test") or {}
        obs = table.to_numpy(dtype=float)
        if t.get("statistic") is not None:
            exp = obs.sum(axis=1, keepdims=True) * obs.sum(axis=0, keepdims=True) / obs.sum()
            chi2 = float(((obs - exp) ** 2 / exp).sum())
            dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
            add("chi-square", t["statistic"], chi2)
            add("chi-square p-value", t.get("p_value"), _st.chi2.sf(chi2, dof))
            if t.get("cramers_v") is not None:
                add("Cramer's V", t["cramers_v"], math.sqrt(chi2 / (obs.sum() * (min(obs.shape) - 1))))
        elif t.get("name") == "Fisher's exact test" and obs.shape == (2, 2):
            a0, r1, c1, n_ = int(obs[0, 0]), int(obs[0].sum()), int(obs[:, 0].sum()), int(obs.sum())
            lo, hi = max(0, r1 + c1 - n_), min(r1, c1)
            pm = _st.hypergeom(n_, c1, r1).pmf
            p_obs = pm(a0)
            add("Fisher p-value", t.get("p_value"),
                min(1.0, float(sum(pm(x) for x in range(lo, hi + 1) if pm(x) <= p_obs * (1 + 1e-7)))))
    if not out:
        return "not independently verified", [{"check": "independent recomputation", "ok": True,
                                               "detail": "not independently verified (nothing to recompute)"}]
    if not all(c["ok"] for c in out):
        return "failed", out
    if any("not independently verified" in c["detail"] for c in out):
        return "not independently verified", out
    return "verified", out


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
    from app.analyst.data import file_lines
    pos = file_lines(df)
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
    from app.analyst.data import file_lines
    pos = file_lines(df)
    parts = []
    for col in sorted(cols):
        v = pd.to_numeric(used[col], errors="coerce")
        miss = used.index[used[col].isna()]
        if len(miss):
            parts.append(f"{len(miss)} record(s) with no {col.replace('_', ' ')} (line{'s' if len(miss) > 1 else ''} "
                         f"{', '.join(str(pos[i]) for i in miss[:10])}) were not counted")
        from app.analyst.metrics import never_negative  # noqa: PLC0415
        neg = used.index[v.lt(0).fillna(False)]
        if len(neg) and _AMOUNT.search(col) and not never_negative(col):   # impossible ones are left out and listed
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
