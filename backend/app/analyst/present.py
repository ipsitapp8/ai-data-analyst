"""Answers and charts for the routed tools (overview, data quality, outliers,
trends, patterns, scenarios, documents), and chart specifications for the
planner's tools. Text uses only numbers from the results (explain.ungrounded
checks the model's prose; these templates format results directly).

A chart is {"type": "bar"|"line", "title", "x", "y", "data": [{x: .., y: ..}], "format"?}.
"""
from __future__ import annotations

import re

from app.analyst.explain import f, p


def _pages(nums) -> str:
    return ("Page " if len(nums) == 1 else "Pages ") + ", ".join(map(str, nums))


def _are(nums) -> str:
    return "is a scanned image" if len(nums) == 1 else "are scanned images"


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


# ------------------------------------------------------------------ overview --

def plain_overview(r: dict) -> str:
    """The dataset in a few plain sentences, from the computed profile."""
    cols = r["column_details"]
    cats = [c for c in cols if c["kind"] == "categorical" and not c.get("time_of_day") and not c["identifier"]
            and 2 <= c["unique"] <= 30]
    cats.sort(key=lambda c: (bool(re.search(r"(^|_)(id|name|code)$", c["name"].lower())), c["unique"]))
    nums = [c for c in cols if c["kind"] == "numeric" and "mean" in c and not c["identifier"]]
    dates = [c for c in cols if c["kind"] == "datetime" and "first" in c]
    times = [c for c in cols if c.get("time_of_day")]
    out = [f"**In simple terms:** {r['filename'] or 'this file'} is a table of **{r['rows']:,} records** with "
           f"{r['columns']} pieces of information each."]
    if dates:
        out.append(f"The records run from {dates[0]['first']} to {dates[0]['last']}.")
    if cats:
        c = cats[0]
        vals = ", ".join(t["value"] for t in (c.get("top_values") or [])[:5])
        out.append(f"They are grouped by **{_lab(c['name'])}** ({c['unique']} values: {vals}"
                   + ("…" if c["unique"] > 5 else "") + ").")
    if times:
        out.append("**" + " and ".join(_lab(c["name"]) for c in times[:2]) + "** are times of day"
                   + (f" ({times[0].get('earliest')} to {times[-1].get('latest')})" if times[0].get("earliest") else "")
                   + (", so hours worked can be calculated from them." if len(times) >= 2 else "."))
    for c in nums[:2]:
        out.append(f"**{_lab(c['name'])}** ranges from {f(c['min'])} to {f(c['max'])} (typically {f(c['median'])}).")
    if r["quality_issues"]:
        out.append(f"Before relying on the numbers, note {len(r['quality_issues'])} data-quality point"
                   f"{'s' if len(r['quality_issues']) != 1 else ''} listed below.")
    return " ".join(out)


def _lab(name: str) -> str:
    return str(name).replace("_", " ")


def understand(r: dict) -> str:
    k = r["kinds"]
    lines = [plain_overview(r), "", f"### What's in {r['filename'] or 'this dataset'}",
             f"**{r['rows']:,} rows × {r['columns']} columns** — {k['numeric']} numeric, {k['categorical']} categorical"
             + (f", {k['datetime']} date" if k["datetime"] else "") + ". "
             + (f"{r['missing_cells']:,} missing cells" if r["missing_cells"] else "No missing values")
             + (f", {r['duplicates']} duplicated rows." if r["duplicates"] else ", no duplicated rows."), "",
             "| column | type | missing | summary |", "|---|---|---:|---|"]
    for c in r["column_details"]:
        if c["kind"] == "numeric" and "mean" in c:
            summ = f"mean {f(c['mean'])}, median {f(c['median'])}, range {f(c['min'])} to {f(c['max'])}"
        elif c["kind"] == "datetime" and "first" in c:
            summ = f"{c['first']} to {c['last']}"
        elif c.get("top_values"):
            summ = f"{c['unique']} values, most common: " + ", ".join(f"{t['value']} ({t['rows']})" for t in c["top_values"][:3])
        else:
            summ = f"{c['unique']} distinct values"
        if c.get("time_of_day"):
            summ = f"times of day, {c.get('earliest')} to {c.get('latest')}"
        if c["identifier"]:
            summ = "identifier (unique per row)"
        lines.append(f"| {c['name']} | {'time of day' if c.get('time_of_day') else c['kind']} | {c['missing']} | {summ} |")
    if r["quality_issues"]:
        lines += ["", "**Data quality**"] + [f"- {x}" for x in r["quality_issues"]]
    else:
        lines += ["", "**Data quality:** no missing values, duplicates or unusual values were found."]
    if r["business_roles"]:
        lines += ["", "**Business columns recognised:** " + ", ".join(f"{k} = `{v}`" for k, v in r["business_roles"].items())]
    if r["possible_targets"]:
        lines += ["", "**Columns that could be predicted:** " + ", ".join(
            f"`{t['column']}` ({t['task']})" for t in r["possible_targets"])]
    lines += ["", "**Questions worth asking next**"] + [f"- {q}" for q in r["suggested_questions"]]
    return "\n".join(lines)


def suggestions_only(r: dict) -> str:
    lines = [f"Here is what I would look at next in {r['filename'] or 'this dataset'} ({r['rows']:,} rows):", ""]
    lines += [f"- {q}" for q in r["suggested_questions"]]
    if r["quality_issues"]:
        lines += ["", "Check these data-quality points first:"] + [f"- {x}" for x in r["quality_issues"][:4]]
    return "\n".join(lines)


def missing_values(r: dict) -> str:
    bad = [c for c in r["columns"] if c["missing"]]
    lines = ["### Missing values and duplicates",
             f"{r['missing_cells']:,} missing cells; {r['rows_with_missing']:,} of {r['rows']:,} rows "
             f"({pct(r['rows_with_missing_pct'])}) have at least one missing value; {r['duplicates']} duplicated rows."]
    if bad:
        lines += ["", "| column | missing | share |", "|---|---:|---:|"]
        lines += [f"| {c['column']} | {c['missing']} | {pct(c['missing_pct'])} |" for c in bad]
        lines += ["", "Analyses skip missing values for the columns they use; Random Forest training fills them with the "
                      "median or most common value of the training rows."]
    else:
        lines.append("\nNo column has missing values.")
    return "\n".join(lines)


def outliers(r: dict) -> str:
    if not r["columns"]:
        return f"No unusual values found in {', '.join(r['checked'][:8]) or 'the numeric columns'} ({r['method']})."
    lines = [f"### Unusual values ({r['method']})", "", "| column | unusual values | share | normal range | most extreme |",
             "|---|---:|---:|---|---|"]
    for o in r["columns"]:
        ex = ", ".join(f"{f(e['value'])} (row {e['row']})" for e in o["examples"][:3])
        lines.append(f"| {o['column']} | {o['count']} | {pct(o['pct'])} | {f(o['low'])} to {f(o['high'])} | {ex} |")
    lines += ["", "Unusual is not the same as wrong: check whether these are entry errors or genuine extremes before "
                  "removing them."]
    return "\n".join(lines)


def patterns(r: dict) -> str:
    lines = ["### What stands out"]
    if r["correlations"]:
        lines += ["", "Strongest relationships between numeric columns (Pearson r):", "",
                  "| columns | r |", "|---|---:|"]
        lines += [f"| {c['x']} & {c['y']} | {f(c['r'])} |" for c in r["correlations"]]
    if r["outliers"]:
        lines += ["", "Columns with unusual values: " + ", ".join(f"{o['column']} ({o['count']})" for o in r["outliers"])]
    if not r["correlations"] and not r["outliers"]:
        lines.append("Nothing stands out: no strong relationships or unusual values.")
    lines += ["", r["note"]]
    return "\n".join(lines)


def trend(r: dict) -> str:
    ch = r["change_pct"]
    return "\n".join([
        f"### {('Average' if r['aggregation'] == 'avg' else 'Total')} {r['value']} by {r['frequency']}",
        f"From **{f(r['first']['value'])}** in {r['first']['period']} to **{f(r['last']['value'])}** in "
        f"{r['last']['period']}: a change of {f(r['change'])}" + (f" ({ch:+.1f}%)" if ch is not None else "") + ".",
        f"Highest: {r['peak']['period']} ({f(r['peak']['value'])}); lowest: {r['low']['period']} ({f(r['low']['value'])}). "
        f"Average change per {r['frequency']} (straight-line fit): {f(r['slope_per_period'])}.", "", r["note"]])


def drivers(r: dict) -> str:
    g, col = r["group"], r["group_column"]
    lines = [f"### How {g} differs from the other {col} values"]
    o = r.get("outcome")
    if o:
        lines.append(f"Average {o['column']}: {f(o['group_mean'])} in {g} vs {f(o['others_mean'])} elsewhere "
                     f"({r['rows_group']} vs {r['rows_others']} rows).")
    sig = [x for x in r["numeric"] if x["p_value"] < 0.05]
    if r["numeric"]:
        lines += ["", "| column | " + g + " | others | difference | p (Welch) |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {x['column']} | {f(x['group_mean'])} | {f(x['others_mean'])} | "
                  f"{x['difference_pct']:+.1f}% | {p(x['p_value'])[4:] if x['p_value'] >= 0.001 else '< 0.001'} |"
                  for x in r["numeric"][:10]]
    if sig:
        lines += ["", f"**Clear differences (p < 0.05):** " + "; ".join(
            f"{x['column']} {x['difference_pct']:+.1f}%" for x in sig[:5]) + "."]
    else:
        lines += ["", "No other numeric column differs clearly (all p ≥ 0.05)."]
    for c in r["categorical"][:3]:
        lines.append(f"- {c['column']} = {c['value']}: {f(c['group_share'], pct=True)} of {g} rows vs "
                     f"{f(c['others_share'], pct=True)} elsewhere.")
    lines += ["", f"_{r['note']}_"]
    return "\n".join(lines)


# ------------------------------------------------------------------ scenarios --

def scenario(r: dict) -> str:
    k = r["kind"]
    if k == "column_change":
        o, h, d = r["observed"], r["hypothetical"], r["difference"]
        lines = [f"### Scenario: {r['column']} changes by {r['change_pct']:+g}%", "",
                 "| | observed (baseline) | hypothetical | difference |", "|---|---:|---:|---:|",
                 f"| total {r['column']} | {f(o['total'])} | {f(h['total'])} | {f(d['total'])} |",
                 f"| average {r['column']} | {f(o['average'])} | {f(h['average'])} | {f(d['average'])} |"]
        if r.get("by_group"):
            lines += ["", "| group | observed | hypothetical | difference |", "|---|---:|---:|---:|"]
            lines += [f"| {g['group']} | {f(g['observed'])} | {f(g['hypothetical'])} | {f(g['difference'])} |"
                      for g in r["by_group"][:10]]
        m = r.get("model_estimate")
        if m:
            if m["task"] == "regression":
                lines += ["", f"**Model-based estimate** (the Random Forest trained in this conversation): the average "
                              f"predicted {m['target']} moves from {f(m['average_prediction_before'])} to "
                              f"{f(m['average_prediction_after'])} ({f(m['difference'])})."]
            else:
                lines += ["", f"**Model-based estimate** (the Random Forest trained in this conversation), predicted share "
                              f"of each {m['target']} class:", "", "| class | before | after |", "|---|---:|---:|"]
                lines += [f"| {c['class']} | {f(c['before'], pct=True)} | {f(c['after'], pct=True)} |" for c in m["class_shares"]]
            lines.append(f"_{m['caution']}_")
    elif k == "conversion_change":
        o, h = r["observed"], r["hypothetical"]
        lines = [f"### Scenario: conversion rate becomes {r['to_rate'] * 100:g}%", "",
                 "| | observed (baseline) | hypothetical | difference |", "|---|---:|---:|---:|",
                 f"| conversion rate | {f(o['conversion_rate'], pct=True)} | {f(r['to_rate'], pct=True)} | |",
                 f"| conversions | {f(o['conversions'])} | {f(h['conversions'])} | {f(r['difference']['conversions'])} |"]
        if "revenue" in h:
            lines.append(f"| revenue | {f(o['revenue'])} | {f(h['revenue'])} | {f(r['difference']['revenue'])} |")
        if r.get("from_rate_revenue") is not None:
            lines += ["", f"Against your stated baseline of {r['assumed_from_rate'] * 100:g}% "
                          f"(revenue {f(r['from_rate_revenue'])}), the difference is {f(r['difference_vs_from_rate'])}."]
        if r.get("note"):
            lines += ["", r["note"]]
    else:
        o, h, d = r["observed"], r["hypothetical"], r["difference"]
        lines = [f"### Scenario: share of {r['column']} = {r['value']}", "",
                 "| | observed (baseline) | hypothetical | difference |", "|---|---:|---:|---:|",
                 f"| rate | {f(o['rate'], pct=True)} | {f(h['rate'], pct=True)} | {d['rate_points']:+.1f} points |",
                 f"| rows | {o['count']} | {f(h['count'], digits=1)} | {d['count']:+.1f} |"]
        if r.get("value_difference") is not None:
            lines += ["", f"At the observed average {r['value_column']} of {f(r['value_per_row'])} per affected row, that "
                          f"is a difference of {f(r['value_difference'])}."]
    lines += ["", "**Assumptions**"] + [f"- {a}" for a in r["assumptions"]] + ["", f"_{r['caveat']}_"]
    return "\n".join(lines)


# ------------------------------------------------------------------ documents --

def doc_overview(r: dict) -> str:
    lines = [f"### {r['filename']}", f"{r['page_count']} pages; text extracted from {len(r['text_pages'])}."]
    if r["scanned_pages"]:
        lines.append(f"**{_pages(r['scanned_pages'])} {_are(r['scanned_pages'])} with no text layer.** OCR is "
                     "not available here, so their content was not read and is not used in any answer.")
    if r["empty_pages"]:
        lines.append(f"{_pages(r['empty_pages'])} {'has' if len(r['empty_pages']) == 1 else 'have'} no text.")
    if r["key_terms"]:
        lines.append("Frequent terms: " + ", ".join(r["key_terms"]) + ".")
    if r["page_openings"]:
        lines += ["", "**How each page starts**"] + [f"- p. {x['page']}: {x['text']}" for x in r["page_openings"]]
    return "\n".join(lines)


def doc_passages(question: str, hits: list[dict], meta: dict) -> str:
    if not hits:
        note = (" Some pages are scanned images that could not be read." if meta.get("scanned_pages") else "")
        return (f"I could not find anything about that in **{meta['filename']}**." + note +
                " Try other words, or ask for a summary of the document.")
    lines = [f"From **{meta['filename']}**, the most relevant passages:", ""]
    for h in hits:
        text = h["text"].replace("\n", " ")
        lines.append(f"> {text[:600]}{'…' if len(text) > 600 else ''}\n> — page {h['page']}\n")
    if meta.get("scanned_pages"):
        lines.append(f"_{_pages(meta['scanned_pages'])} {_are(meta['scanned_pages'])} and {'was' if len(meta['scanned_pages']) == 1 else 'were'} not searched._")
    return "\n".join(lines)


# ------------------------------------------------------------------ charts --

def charts(tool: str, r: dict) -> list[dict]:
    try:
        return _charts(tool, r)
    except Exception:  # noqa: BLE001 - a chart is a convenience; never fail an answer for it
        return []


def _bar(title, x, y, rows, fmt=None):
    return [{"type": "bar", "title": title, "x": x, "y": y, "data": rows, **({"format": fmt} if fmt else {})}] if rows else []


def _charts(tool: str, r: dict) -> list[dict]:
    if tool == "compute_metric" and r.get("groups"):
        return [dict(c, additive=r.get("aggregation") in ("sum", "count")) for c in _bar(f"{r['aggregation']} {r.get('column') or 'rows'} by {r['group_by']}", r["group_by"], "value",
                    [{r["group_by"]: g["group"], "value": g["value"]} for g in r["groups"][:25] if g["value"] is not None])]
    if tool == "compare_groups" and r.get("kind") == "numeric":
        return [dict(c, additive=False) for c in _bar(f"mean {r['outcome']} by {r['group']}", r["group"], "mean",
                                                      [{r["group"]: s["group"], "mean": s["mean"]} for s in r["summary"]])]
    if tool in ("business_performance", "recommend"):
        perf = r.get("performance", r)
        return _bar(f"ROI by {perf['group_column']}", perf["group_column"], "ROI %",
                    [{perf["group_column"]: g["group"], "ROI %": g["roi"] * 100} for g in perf["groups"] if g.get("roi") is not None])
    if tool == "random_forest" and r.get("test_predictions"):
        return [{"type": "avp", "x": "actual", "y": "predicted", "data": r["test_predictions"],
                 "title": f"Actual vs predicted {r['target']} (held-out rows)",
                 "caption": "Rows the model never saw in training; points on the diagonal are perfect predictions."}] + \
            _bar(f"What the '{r['target']}' model relied on", "feature", "importance",
                 [{"feature": i["feature"], "importance": i["importance"]} for i in r["importance"][:10]])
    if tool in ("random_forest", "feature_importance"):
        return _bar(f"What the '{r['target']}' model relied on", "feature", "importance",
                    [{"feature": i["feature"], "importance": i["importance"]} for i in r["importance"][:10]])
    if tool == "drivers":
        return _bar(f"{r['group']} vs the rest (% difference)", "column", "difference %",
                    [{"column": x["column"], "difference %": x["difference_pct"]} for x in r["numeric"][:8]
                     if x["difference_pct"] is not None])
    if tool == "trend":
        return [{"type": "line", "title": f"{r['value']} by {r['frequency']}", "x": "period", "y": r["value"],
                 "data": [{"period": pt["period"], r["value"]: pt["value"]} for pt in r["points"]]}]
    if tool == "missing_values":
        return _bar("missing values by column", "column", "missing",
                    [{"column": c["column"], "missing": c["missing"]} for c in r["columns"] if c["missing"]])
    if tool == "outliers":
        return _bar("unusual values by column", "column", "unusual values",
                    [{"column": o["column"], "unusual values": o["count"]} for o in r["columns"]])
    if tool == "column_change":
        return _bar(f"{r['column']}: observed vs hypothetical total", "case", r["column"],
                    [{"case": "observed", r["column"]: r["observed"]["total"]},
                     {"case": "hypothetical", r["column"]: r["hypothetical"]["total"]}])
    if tool == "conversion_change" and "revenue" in r["hypothetical"]:
        return _bar("revenue: observed vs hypothetical", "case", "revenue",
                    [{"case": "observed", "revenue": r["observed"]["revenue"]},
                     {"case": "hypothetical", "revenue": r["hypothetical"]["revenue"]}])
    if tool == "budget_scenario":
        a, b = r["if_returns_scale"], r["if_no_response"]
        return _bar("revenue under each assumption", "case", "revenue",
                    [{"case": "baseline", "revenue": r["baseline_revenue"]},
                     {"case": "returns scale", "revenue": a["revenue"]}, {"case": "no response", "revenue": b["revenue"]}])
    return []


__all__ = ["understand", "suggestions_only", "missing_values", "outliers", "patterns", "trend", "scenario",
           "doc_overview", "doc_passages", "charts", "p"]


# ---------------------------------------------------------------- calculator --

def _label(name: str) -> str:
    return str(name).replace("_", " ")


def _cellv(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f(v)
    return str(v).replace("|", "\\|")


def readable(expr: str) -> str:
    """A formula in words: `sum(hourly_rate * hours_between(clock_in, clock_out))` ->
    "total of hourly rate × hours from clock in to clock out"."""
    t = str(expr)
    t = re.sub(r"overnight_hours\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)", r"hours from \1 to \2 (past midnight counts as next day)", t)
    t = re.sub(r"hours_between\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)", r"hours from \1 to \2", t)
    t = re.sub(r"\b(sum|total)\(", "total of (", t)
    t = re.sub(r"\b(mean|avg|average)\(", "average of (", t)
    t = re.sub(r"\bcount\(\*\)|\bcount\(1\)", "number of records", t)
    t = re.sub(r"\bcount\(", "number of (", t)
    t = re.sub(r"\bmax\(", "largest (", t)
    t = re.sub(r"\bmin\(", "smallest (", t)
    t = re.sub(r"\bduplicate\(\)", "is a repeat of an earlier record", t)
    t = re.sub(r"\(([^()]*)\)", lambda m: m.group(1) if re.fullmatch(r"[\w .']+", m.group(1)) else m.group(0), t)
    t = re.sub(r"'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?)'", r"\1", t, flags=re.I)
    t = t.replace(" * ", " × ").replace(" / ", " ÷ ").replace("!=", "is not").replace("==", "is").replace(">=", "≥").replace("<=", "≤")
    return t.replace("_", " ")


_RATE_NAME = re.compile(r"rate|share|percent|pct|proportion|ratio|fraction", re.I)


def _is_rate(r: dict, name: str) -> bool:
    """A result that is a share of records (e.g. mean(attrition == 'Yes')): shown as a percentage."""
    if name in (r.get("percent") or []):
        return True                     # a ratio the metric catalogue defines as a percentage (margin, ROI, ...)
    vals = [r["totals"].get(name)] + [g.get(name) for g in r.get("groups") or []]
    vals = [v for v in vals if isinstance(v, (int, float))]
    expr = (r.get("aggregates") or {}).get(name, "")
    looks = bool(_RATE_NAME.search(name)) or bool(re.search(r"\b(mean|avg|average)\s*\(.*(==|!=|<|>)", expr))
    return looks and bool(vals) and all(0 <= v <= 1 for v in vals) and "%" not in expr and "* 100" not in expr


def _aval(r: dict, name: str, v) -> str:
    return pct(v * 100) if v is not None and _is_rate(r, name) else _cellv(v)


def calc_headline(r: dict, question: str = "") -> str:
    """The direct answer, in one or two sentences, from the computed result only."""
    aggs = list(r["aggregates"])
    if not aggs:
        return ""
    key = (r.get("primary") if r.get("primary") in aggs else None) or \
        next((a for a in aggs if a.replace("_", " ").lower() in question.lower()), None) or \
        next((n for n, e in r["aggregates"].items() if e.startswith("sum(")), aggs[0])
    est = "estimated " if r.get("answer_type") == "scenario" and not key.lower().startswith(("estimat", "predict", "project")) else ""
    if r.get("focus"):
        row = r["focus"][0]
        return f"**{row['group']}**: {est}{_label(key)} **{_aval(r, key, row.get(key))}** ({row['rows']} records)."
    groups = [g for g in r.get("groups") or [] if g.get(key) is not None]
    if r.get("group_by") and len(groups) > 1:
        low = bool(re.search(r"\b(lowest|least|smallest|worst|bottom|fewest)\b", question, re.I))
        ordered = sorted(groups, key=lambda g: g[key], reverse=not low)
        best, other = ordered[0], ordered[-1]
        share = best.get("share_of_" + key)
        total = r["totals"].get(key)
        return (f"**{best['group']}** has the {'lowest' if low else 'highest'} {_label(key)}: **{_aval(r, key, best[key])}**"
                + (f" ({pct(share * 100)} of the total {_aval(r, key, total)})" if share is not None and total else "")
                + f". {other['group']} has the {'highest' if low else 'lowest'} ({_aval(r, key, other[key])}).")
    parts = [f"{est}{_label(n)} **{_aval(r, n, v)}**" for n, v in r["totals"].items()]
    who = ""
    sel = [m.group(1) for e in r["exclusions"] for m in [re.search(r"!=\s*'([^']*[A-Za-z][^']*)'", e["where"])] if m]
    sel += [x for e in r["exclusions"] for m in [re.search(r"not\s+in\s*\[([^\]]*)\]", e["where"])] if m
            for x in re.findall(r"'([^']+)'", m.group(1))]
    if sel:
        who = f"For {' and '.join(sel[:3])}: "
    return (who + "; ".join(parts) + ".")[:1].upper() + (who + "; ".join(parts) + ".")[1:]


def calc(r: dict, question: str = "") -> str:
    """The main answer for a calculator result: the direct answer, the result, the breakdown, warnings."""
    aggs = list(r["aggregates"])
    lines = [calc_headline(r, question), ""]
    if r.get("assumptions"):
        lines += ["**Assumptions (from your question):** " + "; ".join(
            f"{_label(a['name'])}" + ("" if readable(a['formula']) == _label(a['name']) else f" = {readable(a['formula'])}")
            + (f" = **{_cellv(a['value'])}**" if a.get("value") is not None else "")
            for a in r["assumptions"]), ""]
    if r["totals"] and not (r.get("group_by") and len(r.get("groups") or []) > 1 and not r.get("focus")) or r.get("focus"):
        lines += [f"| result{' (all ' + _label(r['group_by']) + ' values)' if r.get('focus') else ''} | value |", "|---|---:|"]
        lines += [f"| **{_label(n)}** | **{_aval(r, n, v)}** |" for n, v in r["totals"].items()]
        lines.append("")
    if r.get("group_by") and r.get("groups") and not r.get("focus"):
        g = r["group_by"]
        share = next((k for k in r["groups"][0] if k.startswith("share_of_")), None)
        if r.get("primary") and share != "share_of_" + r["primary"]:
            share = None                # a share of a component (e.g. of revenue) is not the result asked for
        lines += [f"| {_label(g)} | records | " + " | ".join(_label(a) for a in aggs)
                  + (f" | share of {_label(share[9:])} |" if share else " |"),
                  "|---|---:|" + "---:|" * len(aggs) + ("---:|" if share else "")]
        for row in r["groups"][:30]:
            lines.append(f"| {row['group']} | {row['rows']} | " + " | ".join(_aval(r, a, row.get(a)) for a in aggs)
                         + (f" | {pct((row.get(share) or 0) * 100)} |" if share else " |"))
        if len(r["groups"]) > 1:
            lines.append("| **all** | **" + str(r["rows_valid"]) + "** | " + " | ".join(f"**{_aval(r, a, r['totals'].get(a))}**" for a in aggs)
                         + (" | 100% |" if share else " |"))
        lines.append("")
        if r.get("why"):
            lines += ["**Why:** " + r["why"]["sentence"].replace("_", " ") + " These describe what makes up the "
                      "total; they do not prove a cause.", ""]
    lines.append(f"Based on **{r['rows_valid']}** of {r['rows_total']} records"
                 + (f" ({r['rows_excluded']} left out — see *How this was calculated*)." if r["rows_excluded"] else "."))
    for u in r.get("inputs_used") or []:
        vals = ", ".join(f"{_cellv(v['value'])} ({v['records']} record{'s' if v['records'] != 1 else ''})" for v in u["values"])
        if u["distinct"] == 1:
            lines.append(f"\n**{_label(u['column'])} in the records used:** {vals} — one consistent value.")
        elif r.get("answer_type") == "scenario":
            lines.append(f"\n**{_label(u['column'])} varies across the {u['records']} records used** ({vals}"
                         + (" …" if u["distinct"] > len(u["values"]) else "") + "); the estimate combines them as "
                         "described under *How this was calculated*"
                         + ("" if r.get("selected") else ". No person or group was selected, so it averages over "
                            "everyone; name one to use only their records") + ".")
    checks = r.get("data_checks") or []
    if checks:
        lines += ["", "⚠️ **Data checks on the records used** (they were included as recorded):"]
        for c in checks:
            ls = c["lines"]
            lines.append(f"- line{'s' if len(ls) != 1 else ''} {', '.join(map(str, ls))}: {c['issue']} — ask to "
                         f"*{c['fix']}* if {'they' if len(ls) != 1 else 'it'} should not count.")
    for k, n in (r.get("incomplete") or {}).items():
        lines.append(f"\n_{n} kept record{'s' if n != 1 else ''} had no valid value for {_label(k)} (a missing or "
                     f"invalid time or number) and {'were' if n != 1 else 'was'} left out of results using it._")
    if r.get("answer_type") == "scenario":
        lines += ["", "_Type of answer: a **scenario calculation** from historical records and the assumptions you "
                  "stated — **not a machine-learning prediction**; no model was trained._"]
    else:
        lines += ["", "_Type of answer: an exact **calculation** on the records in this dataset — not a "
                  "machine-learning prediction._"]
    return "\n".join(x for x in lines if x is not None).strip()


def calc_details(r: dict) -> str:
    """The optional "How this was calculated" section: formulas in words, records left out, per-record averages."""
    lines = ["**Steps**"]
    for fm in r["formulas"]:
        lines.append(f"- **{_label(fm['column'])}** for each record = {readable(fm['formula'])}")
    for n, a in r["aggregates"].items():
        lines.append(f"- **{_label(n)}** = {readable(a)}"
                     + (f", for each {_label(r['group_by'])} and overall" if r.get("group_by") else ""))
    if any("overnight_hours" in x for x in [fm["formula"] for fm in r["formulas"]] + list(r["aggregates"].values())):
        lines.append("- An end time earlier than the start counts as the next day (an overnight shift).")
    if any("hours_between" in x for x in [fm["formula"] for fm in r["formulas"]] + list(r["aggregates"].values())):
        lines.append("- Hours between two clock times are the plain difference (09:00 to 17:30 is 8.5 hours); no "
                     "break is deducted unless you state one.")
    for u in r.get("inputs_used") or []:
        if u["distinct"] > 1:
            vals = ", ".join(f"{_cellv(v['value'])} ({v['records']})" for v in u["values"])
            lines.append(f"- {_label(u['column'])} values used (records): {vals}"
                         + (" …" if u["distinct"] > len(u["values"]) else ""))
    selection = [e for e in r["exclusions"] if re.search(r"!=\s*'[^']*[A-Za-z][^']*'", e["where"])]
    if r["exclusions"] and len(selection) == len(r["exclusions"]):
        lines += ["", "**Records used:** " + f"{r['rows_valid']} of {r['rows_total']} — "
                  + "; ".join(f"{e['reason']} ({e['rows']} other records left out)" for e in selection) + "."]
    elif r["exclusions"]:
        lines += ["", f"**Records left out ({r['rows_excluded']})**", "", "| rule | records |", "|---|---:|"]
        lines += [f"| {e['reason']} — {readable(e['where'])} | {e['rows']} |" for e in r["exclusions"]]
        if r["excluded_rows"]:
            cols = [c for c in r["excluded_rows"][0] if c not in ("row", "excluded_because")][:8]
            lines += ["", "| line | " + " | ".join(_label(c) for c in cols) + " | why |",
                      "|---:|" + "---|" * len(cols) + "---|"]
            for e in r["excluded_rows"][:25]:
                lines.append(f"| {e['row']} | " + " | ".join(_cellv(e.get(c)) for c in cols) + f" | {e['excluded_because']} |")
            if r["rows_excluded"] > min(25, len(r["excluded_rows"])):
                lines.append(f"\n_{r['rows_excluded'] - min(25, len(r['excluded_rows']))} more are not listed._")
        if r.get("overlap_note"):
            lines.append("\n" + r["overlap_note"])
    if r.get("group_by") and r.get("groups") and (r["groups"][0].get("per_row")):
        g = r["group_by"]
        per = list(r["groups"][0]["per_row"])
        lines += ["", f"**Per-record averages by {_label(g)}** (what each total is made of)", "",
                  f"| {_label(g)} | " + " | ".join(_label(k) for k in per) + " |", "|---|" + "---:|" * len(per)]
        for row in r["groups"][:30]:
            lines.append(f"| {row['group']} | " + " | ".join(_cellv(row["per_row"].get(k)) for k in per) + " |")
    if any("duplicate()" in e["where"] for e in r["exclusions"]):
        lines.append("\nExact duplicates: the first occurrence is kept and later identical records are left out.")
    return "\n".join(lines).strip()


_SINGLE_TOTAL = re.compile(r"(?:sum|count)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)")


def calc_charts(r: dict) -> list[dict]:
    try:
        if not (r.get("group_by") and r.get("groups") and r["aggregates"]):
            return []
        a = next(iter(r["aggregates"]))
        sums = [n for n, e in r["aggregates"].items() if e.startswith("sum(")]
        a = r["primary"] if r.get("primary") in r["aggregates"] else (sums[-1] if sums else a)
        g = r["group_by"]
        # additive (parts of a whole, so a pie is meaningful) only for ONE total or count -- not a ratio of totals
        additive = bool(_SINGLE_TOTAL.fullmatch(r["aggregates"][a].strip())) and not _is_rate(r, a)
        return [dict(c, additive=additive) for c in _bar(
            f"{_label(a)} by {_label(g)}", g, _label(a),
            [{g: row["group"], _label(a): row[a]} for row in r["groups"][:25] if row.get(a) is not None])]
    except Exception:  # noqa: BLE001
        return []
