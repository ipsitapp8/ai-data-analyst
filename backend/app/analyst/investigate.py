"""Autonomous investigation of a whole dataset: analyses chosen from what the data is, run as one validated program.

`investigate(df)` reads the semantic profile (app/analyst/semantics.py), chooses only the analyses that fit -- data
quality always; associations with an outcome when there is one (else with the main measure); group differences
for low-cardinality categories; a trend when there is a date or year; unusual values in the main measures;
concentration in a category; the leading entities when names are present -- builds them as steps of one program
(app/analyst/program.py), validates and runs it, and turns step results into findings.

A finding is reported only when the evidence passes a stated bar (an adjusted p-value below 0.05 and a non-trivial
effect, a share above a threshold, ...). Each finding carries its calculation, columns, scope, number of
observations, interpretation and limitations, and a follow-up question Silt can answer. Patterns are never called
causes, and a numerical pattern is not called a business problem: it is a thing to check.
"""
from __future__ import annotations

import math
import re

import pandas as pd

from app.analyst import program as P
from app.analyst import semantics
from app.analyst.explain import f as fmt

MAX_MEASURES = 4
MAX_GROUPS = 3
MIN_GROUP = 5          # a group difference is reported only between groups of at least this many records


def _q(c: str) -> str:
    return c if str(c).isidentifier() else f"`{c}`"


def _lab(c) -> str:
    return str(c).replace("_", " ")


def plan(df: pd.DataFrame) -> tuple[list[dict], dict, dict]:
    prof = semantics.profile(df)
    roles = prof["roles"]
    order = {"amount": 0, "count": 1, "measurement": 2, "rate": 3}
    # the main measure is an additive business amount (revenue, sales, cost ...) before a level (a price, an average)
    level = re.compile(r"price|avg|average|mean|rate|pct|percent|per_|ratio|score|age|lat|lon|temp", re.I)
    key = re.compile(r"revenue|sales|income|profit|turnover|amount|value|cost|spend|hours|units|quantity|qty", re.I)
    first = re.compile(r"revenue|sales|income|profit|turnover|gmv", re.I)
    numo = set(prof.get("numeric_outcomes") or [])
    measures = sorted(prof["measures"], key=lambda c: (0 if c in numo else 1,
                                                       0 if key.search(str(c)) and not level.search(str(c)) else 1,
                                                       0 if first.search(str(c)) else 1,
                                                       1 if level.search(str(c)) else 0, order[roles[c]["role"]],
                                                       -df[c].notna().sum()))[:MAX_MEASURES]
    level = re.compile(r"price|avg|average|mean|rate|pct|percent|per_|ratio|score|age|lat|lon|temp", re.I)
    key = re.compile(r"revenue|sales|income|profit|turnover|amount|value|cost|spend|hours|units|quantity|qty", re.I)
    # standard metrics this data can derive per record (shift duration from clock times, labor cost, profit ...)
    from app.analyst import metrics
    derived_steps, derived = [], []
    for label in metrics.available(df):
        m = next((x for x in metrics.CATALOGUE if x.label == label), None)
        if m is None or m.kind != "row":
            continue
        spec, _ = metrics.row_definition(label, df)
        if spec is None:
            continue
        name = re.match(r"\s*sum\((\w+)\)", next(iter(spec["aggregate"].values()))).group(1)
        for k, v in spec["derive"].items():
            if k not in [d["name"] for d in derived_steps]:
                derived_steps.append({"name": k, "formula": v})
        derived.append(name)
    derived = list(dict.fromkeys(derived))[:2]
    has_key_amount = any(key.search(str(c)) and not level.search(str(c)) for c in prof["measures"])
    if has_key_amount:                  # a recorded amount comes first; a derived one may carry caveats (e.g. profit
        derived, derived_steps = [], []  # without costs of goods), so it is only used when nothing is recorded
    if derived:
        measures = (derived + measures)[:MAX_MEASURES]
    outcome = prof["outcomes"][0] if prof["outcomes"] else None
    groups = [g for g in prof["groups"] if g != outcome and 2 <= df[g].nunique() <= 12][:MAX_GROUPS]
    time = (prof["dates"] or prof["years"] or [None])[0]
    raw_cols = [c for c in measures if c in df.columns]
    inputs = sorted({c for d in derived_steps for c in re.findall(r"[A-Za-z_]\w*", d["formula"]) if c in df.columns})
    steps: list[dict] = [{"id": "quality", "op": "profile"},
                         {"id": "valid0", "op": "select", "valid_for": raw_cols + inputs + ([outcome] if outcome else []),
                          "historical": False, "duplicates": "keep"}]
    prev = "valid0"
    for i, d in enumerate(derived_steps):
        steps.append({"id": f"derived_{i}", "op": "derive", "from": prev, "name": d["name"], "formula": d["formula"]})
        prev = f"derived_{i}"
    steps.append({"id": "valid", "op": "select", "from": prev})
    meta_derived = derived
    main = measures[0] if measures else None
    target = outcome or main
    meta = {"profile": prof, "measures": measures, "outcome": outcome, "groups": groups, "time": time, "main": main,
            "derived": meta_derived}
    if target:
        steps.append({"id": "assoc", "op": "associate", "from": "valid", "target": target,
                      "exclude": [c for c in prof["nonfactors"] if c != target]})
    if outcome:
        pos = roles[outcome].get("positive") or roles[outcome].get("event") or str(df[outcome].dropna().unique()[0])
        meta["positive"] = pos
        val = pos if not str(pos).replace(".", "").isdigit() else None
        steps.append({"id": "flag", "op": "derive", "from": "valid", "name": "is_outcome",
                      "formula": f"{_q(outcome)} == '{val}'" if val is not None else f"{_q(outcome)} == {pos}"})
        steps.append({"id": "rate_all", "op": "aggregate", "from": "flag", "fn": "mean", "column": "is_outcome"})
        for i, g in enumerate(groups):
            steps.append({"id": f"rate_by_{i}", "op": "aggregate", "from": "flag", "fn": "mean", "column": "is_outcome", "by": g})
    if main:
        agg = "mean" if main in roles and roles[main]["role"] == "rate" else "sum"
        meta["main_agg"] = agg
        for i, g in enumerate(groups):
            steps.append({"id": f"main_by_{i}", "op": "aggregate", "from": "valid", "fn": "mean", "column": main, "by": g})
            steps.append({"id": f"n_by_{i}", "op": "aggregate", "from": "valid", "fn": "count", "by": g})
        if time:
            if time in prof["dates"]:
                span = (pd.to_datetime(df[time]).max() - pd.to_datetime(df[time]).min()).days
                per = "M" if span > 90 else "W" if span > 21 else "D"
                steps.append({"id": "over_time", "op": "aggregate", "from": "valid", "fn": agg, "column": main, "per": per})
                steps.append({"id": "n_over_time", "op": "aggregate", "from": "valid", "fn": "count", "per": per})
                meta["per"] = per
            else:
                steps.append({"id": "over_time", "op": "aggregate", "from": "valid", "fn": "mean", "column": main, "by": time})
                meta["per"] = "year"
        if prof["entity"]:
            steps.append({"id": "by_entity", "op": "aggregate", "from": "valid", "fn": agg, "column": main, "by": prof["entity"]})
            steps.append({"id": "top_entities", "op": "rank", "from": "by_entity", "n": 5})
    for i, m in enumerate(measures):
        if m in derived or df[m].notna().sum() >= 8:
            steps.append({"id": f"unusual_{i}", "op": "anomalies", "from": "valid", "column": m})
    return steps, meta, prof


def _finding(title, statement, calc, columns, scope, n, interp, limits, follow, score, kind):
    return {"title": title, "statement": statement, "calculation": calc, "columns": columns, "scope": scope, "n": n,
            "interpretation": interp, "limitations": limits, "follow_up": follow, "score": score, "kind": kind}


def investigate(df: pd.DataFrame) -> dict:
    steps, meta, prof = plan(df)
    res = P.run(steps, df)
    n_all = len(df)
    valid = res["valid"]
    n_valid = int(len(valid.value))
    left = [e for e in res["valid0"].provenance["excluded"] if not e.get("kept")]
    scope = f"{n_valid} of {n_all} records" + (" (impossible values left out)" if left else "")
    F = []
    # ---- data quality: one finding, every issue with its own caution
    q = res["quality"].value
    parts, cautions, affected, cols_q = [], [], 0, []
    miss = {c: k for c, k in q["missing"].items() if k}
    if miss:
        worst = max(miss, key=miss.get)
        parts.append(f"missing values in {len(miss)} column(s) — most in **{worst}** ({miss[worst]} of {n_all})")
        cautions.append("why values are missing is not recorded; if it is not random, averages can be biased")
        affected = max(affected, miss[worst])
        cols_q += list(miss)
    for issue in q["issues"]:
        if issue["reason"].startswith("missing"):
            continue
        parts.append(f"{issue['records']} record(s) — {issue['reason']} (lines {', '.join(map(str, issue['lines'][:6]))})")
        affected += issue["records"]
        cautions.append("a duplicate can be a genuine repeat: check the source before removing it"
                        if "duplicate" in issue["reason"] else "check these against the source; as recorded they "
                        "cannot be valid" if "negative" in issue["reason"] or "earlier than" in issue["reason"]
                        else "a record dated after today is planned, not history")
    if parts:
        F.append(_finding("Data quality to check first", "; ".join(parts) + ".",
                          "empty cells per column; validity rules (exact duplicates, negative values of columns that "
                          "cannot be negative, end before start, dates after today)", cols_q, f"all {n_all} records",
                          n_all, "These records can inflate, distort or thin out the results below.",
                          "; ".join(dict.fromkeys(cautions)).capitalize() + ".", "Summarize the data quality.",
                          1.6 + min(affected / max(n_all, 1) * 10, 1.5), "quality"))
    # ---- associations with the outcome / main measure
    if "assoc" in res:
        a = res["assoc"].value
        sig = [x for x in a["factors"] if x.get("p_adjusted", 1) < 0.05 and x["strength"] >= 0.1]
        tgt = a["target"]
        for x in sig[:3]:
            definitional = x["strength"] >= 0.9
            F.append(_finding(f"{_lab(x['factor'])} is associated with {_lab(tgt)}"
                              + (" (very strongly)" if definitional else ""),
                              f"**{_lab(x['factor'])}** and **{_lab(tgt)}**: {x['method']} {fmt(x['value'])}"
                              + (f" ({x['direction']})" if x["direction"] else "") + f", adjusted p "
                              + ("< 0.001" if x["p_adjusted"] < 0.001 else "< 0.01" if x["p_adjusted"] < 0.01
                                 else fmt(x["p_adjusted"])) + ".",
                              x["method"] + ", Holm-adjusted across all columns tested",
                              [x["factor"], tgt], scope, x["n"],
                              ("So strong that one may be calculated from the other (a total from its parts) or "
                               "both measure the same activity; check that before reading it as an insight." if definitional else
                               "These move together in this data more than chance would explain."),
                              "An association, not a cause: both may depend on something else, or the link may run "
                              "the other way.", f"What factors are associated with {_lab(tgt)}?",
                              0.7 if definitional else 2 + x["strength"], "association"))
        if a["factors"] and not sig:
            F.append(_finding(f"No clear association with {_lab(tgt)}", f"None of the {a['tested']} columns tested shows "
                              f"an association with **{_lab(tgt)}** that is unlikely to be chance.",
                              "Spearman / eta squared / Cramér's V with Holm-adjusted p-values", [tgt], scope, n_valid,
                              "The recorded columns do not explain it well on their own.",
                              "Weak or non-linear effects, or effects inside subgroups, can be missed.",
                              f"Which factors are associated with {_lab(tgt)}?", 0.8, "association"))
    # ---- outcome rate by group
    if meta["outcome"]:
        base = res["rate_all"].value
        for i, g in enumerate(meta["groups"]):
            sizes = valid.value[g].astype(str).value_counts()
            rates = {k: v for k, v in res[f"rate_by_{i}"].value.items() if v is not None and sizes.get(k, 0) >= MIN_GROUP}
            if len(rates) < 2 or base is None:
                continue
            hi = max(rates, key=rates.get)
            lo = min(rates, key=rates.get)
            if rates[hi] - rates[lo] >= 0.05:
                F.append(_finding(f"{_lab(meta['outcome'])} rate differs by {_lab(g)}",
                                  f"{_lab(meta['outcome'])} = {meta['positive']} in **{rates[hi]:.0%}** of records with "
                                  f"{_lab(g)} = {hi} vs **{rates[lo]:.0%}** with {_lab(g)} = {lo} (overall {base:.0%}).",
                                  f"share of records with {_lab(meta['outcome'])} = {meta['positive']}, per {_lab(g)}",
                                  [meta["outcome"], g], scope, n_valid,
                                  "Some groups see the outcome far more often than others.",
                                  "Group sizes differ; small groups can show extreme rates by chance. Not a cause.",
                                  f"What is the {_lab(meta['outcome'])} rate by {_lab(g)}?", 1.5 + rates[hi] - rates[lo], "groups"))
    # ---- main measure by group
    main = meta["main"]
    if main:
        for i, g in enumerate(meta["groups"]):
            counts = res[f"n_by_{i}"].value
            means = {k: v for k, v in res[f"main_by_{i}"].value.items() if v is not None and counts.get(k, 0) >= MIN_GROUP}
            if len(means) < 2:
                continue
            hi, lo = max(means, key=means.get), min(means, key=means.get)
            if means[lo] and means[hi] / means[lo] >= 1.25 or (means[lo] == 0 and means[hi] > 0):
                F.append(_finding(f"{_lab(main)} differs by {_lab(g)}",
                                  f"Average {_lab(main)} is **{fmt(means[hi])}** for {hi} ({int(counts.get(hi, 0))} records) vs "
                                  f"**{fmt(means[lo])}** for {lo} ({int(counts.get(lo, 0))} records).",
                                  f"mean of {_lab(main)} per {_lab(g)}", [main, g], scope, n_valid,
                                  f"{_lab(g)} is linked to large differences in {_lab(main)}.",
                                  "Averages hide spread; a difference in averages is not proof of a real or causal "
                                  "difference — use the comparison test for that.",
                                  f"Compare {_lab(main)} between {hi} and {lo}.",
                                  1.2 + min(means[hi] / means[lo] if means[lo] else 3, 3) / 3, "groups"))
        # ---- trend
        if "over_time" in res:
            pts = [(k, v) for k, v in res["over_time"].value.items() if v is not None]
            if len(pts) >= 3:
                first, last = pts[0], pts[-1]
                peak = max(pts, key=lambda kv: kv[1])
                change = (last[1] - first[1]) / abs(first[1]) if first[1] else None
                note = ""
                if "n_over_time" in res:
                    cnt = list(res["n_over_time"].value.values())
                    med = sorted(cnt)[len(cnt) // 2]
                    if cnt and cnt[-1] < 0.5 * med:
                        note = (f" The last period has only {int(cnt[-1])} records against a usual {int(med)}, so it is "
                                "probably incomplete.")
                F.append(_finding(f"{_lab(main)} over time", f"{meta.get('main_agg', 'sum').replace('sum', 'Total').replace('mean', 'Average')} "
                                  f"{_lab(main)} went from **{fmt(first[1])}** ({first[0]}) to **{fmt(last[1])}** ({last[0]})"
                                  + (f", {change:+.0%}" if change is not None else "") + f"; highest in {peak[0]} "
                                  f"({fmt(peak[1])})." + note, f"{meta.get('main_agg')} of {_lab(main)} per "
                                  f"{ {'M': 'month', 'W': 'week', 'D': 'day'}.get(meta.get('per'), 'year') }",
                                  [main, meta["time"]], scope, n_valid,
                                  "Describes what happened over the recorded periods.",
                                  "A trend over few periods is fragile, and it does not predict the future by itself.",
                                  f"Show the trend of {_lab(main)} over time.",
                                  1 + min(abs(change or 0), 1) + (0.5 if note else 0), "time"))
        if "top_entities" in res:
            top = res["top_entities"].value
            tot = sum(v for v in res["by_entity"].value.values() if v)
            if top and tot:
                lead = next(iter(top))
                share = top[lead] / tot
                if share >= 0.15:
                    F.append(_finding(f"{lead} leads on {_lab(main)}", f"**{lead}** accounts for {share:.0%} of total "
                                      f"{_lab(main)} ({fmt(top[lead])} of {fmt(tot)}).",
                                      f"{meta.get('main_agg')} of {_lab(main)} per {_lab(meta['profile']['entity'])}",
                                      [main, meta["profile"]["entity"]], scope, n_valid,
                                      "Results depend heavily on one person or unit.",
                                      "People with more records naturally have larger totals.",
                                      f"Compare {lead} with the others.", 1 + share, "concentration"))
    # ---- unusual values
    for i, m in enumerate(meta["measures"]):
        r = res.get(f"unusual_{i}")
        if r is None:
            continue
        v = r.value
        # mild tails of a normal spread are expected (~0.7% beyond 1.5 IQR); report extreme values (beyond 3 IQR), a
        # larger share of outliers, or outliers that make up a visible part of the total
        if v["count"] and (v.get("extreme") or v["count"] / v["checked"] >= 0.02 or (v.get("share_of_total") or 0) >= 0.02):
            share = v.get("share_of_total")
            col_min = pd.to_numeric(res["valid"].value[m], errors="coerce").min()
            where = (f"above {fmt(v['high'])}" if v["low"] < col_min else
                     f"outside {fmt(v['low'])}–{fmt(v['high'])}")
            F.append(_finding(f"Unusual values in {_lab(m)}", f"{v['count']} of {v['checked']} values of **{_lab(m)}** lie "
                              f"{where} (median {fmt(v['median'])})"
                              + (f"; together {share:.0%} of its total" if share and share >= 0.05 else "") + ". Lines: "
                              + ", ".join(str(e["line"]) for e in v["examples"]) + ".",
                              "1.5 × interquartile range rule", [m], scope, v["checked"],
                              "These records are far from the typical value and can dominate totals and averages.",
                              "Unusual is not wrong: check whether they are errors or genuine extremes.",
                              f"Find unusual records in {_lab(m)}.", 1 + min(v["count"] / v["checked"] * 5, 1)
                              + (0.5 if share and share >= 0.2 else 0), "unusual"))
    # ---- concentration in a category
    for g in meta["groups"]:
        vc = df[g].value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] >= 0.7:
            F.append(_finding(f"{_lab(g)} is dominated by one value", f"**{vc.index[0]}** makes up {vc.iloc[0]:.0%} of "
                              f"{_lab(g)}.", f"share of records per {_lab(g)}", [g], f"all {n_all} records", n_all,
                              "Comparisons across this column rest mostly on one group.",
                              "The other groups have few records, so their figures are uncertain.",
                              f"How many records are there per {_lab(g)}?", 0.9 + vc.iloc[0] - 0.7, "concentration"))
    F.sort(key=lambda x: -x["score"])
    limits = [f"{n_all} records, {len(df.columns)} columns; findings describe this data only."]
    if not meta["outcome"]:
        limits.append("No outcome column (a yes/no result such as churn or failure) was found; associations use "
                      f"{_lab(main) if main else 'no measure'} instead.")
    if not meta["time"]:
        limits.append("No date or year column: no trends.")
    if n_all < 50:
        limits.append("Small dataset: most differences could be chance.")
    from app.analyst import metrics
    derivable = metrics.available(df)
    return {"findings": F, "meta": {k: v for k, v in meta.items() if k != "profile"}, "profile": prof["columns"],
            "limitations": limits, "derivable": derivable, "program": steps,
            "results": {k: r.summary() for k, r in res.items() if k not in ("valid", "flag")}, "rows": n_all}


def text(r: dict, filename: str | None = None) -> str:
    F = r["findings"]
    lines = [f"**I investigated {filename or 'this dataset'}** ({r['rows']:,} records) and ran {len(r['program'])} "
             f"checks chosen for its columns. {len(F)} finding(s), most important first:", ""]
    for i, x in enumerate(F[:8], 1):
        lines += [f"**{i}. {x['title']}.** {x['statement']}  ",
                  f"_How:_ {x['calculation']}. _Records used:_ {x['n']:,} ({x['scope']}).  ",
                  f"_What it means:_ {x['interpretation']} _Caution:_ {x['limitations']}", ""]
    if not F:
        lines.append("Nothing stood out: no quality problems, no association or difference strong enough to report.")
    roles = {}
    for c in r["profile"]:
        roles.setdefault(c["role"], []).append(c["column"])
    lines += ["", "**What the columns are:** " + "; ".join(f"{k.replace('_', ' ')}: " + ", ".join(f"`{c}`" for c in v[:6])
                                                       for k, v in roles.items())]
    if r["derivable"]:
        lines.append("**Metrics that can be derived:** " + ", ".join(r["derivable"]) + ".")
    nxt = list(dict.fromkeys(x["follow_up"] for x in F if x["follow_up"]))[:4]
    if nxt:
        lines += ["", "**Investigate next** (ask any of these, or *tell me more about finding 2*):"] + [f"- {q}" for q in nxt]
    lines += ["", "_" + " ".join(r["limitations"]) + " Findings are patterns in the data, not causes, and not business "
                  "problems until checked._"]
    return "\n".join(lines)


def detail(x: dict, i: int) -> str:
    return "\n".join([f"**Finding {i}: {x['title']}**", "", x["statement"], "",
                      f"- **Calculation:** {x['calculation']}", f"- **Columns:** " + ", ".join(f"`{c}`" for c in x["columns"]),
                      f"- **Scope:** {x['scope']}", f"- **Observations:** {x['n']}",
                      f"- **Interpretation:** {x['interpretation']}", f"- **Limitations:** {x['limitations']}", "",
                      f"To dig further: *{x['follow_up']}*"])
