"""Writing answers from tool results, and checking that a written answer only
states numbers the tools produced."""
from __future__ import annotations

import json
import math
import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Any

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
# Numbers an answer may state without a tool producing them: small counts and
# ordinals, the 95% level, and the significance thresholds the renderer names.
FREE_NUMBERS = {float(x) for x in range(0, 11)} | {95.0, 100.0, 0.05, 0.01, 0.001}


def f(x: Any, pct: bool = False, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if pct:
        return f"{x * 100:.1f}%"
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) >= 100):
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def p(x: Any) -> str:
    """A p-value with its relation sign: "p = 0.042", "p < 0.001"."""
    if x is None:
        return "p n/a"
    return "p < 0.001" if x < 0.001 else f"p = {x:.3f}"


# ---------------------------------------------------------------- grounding --

def evidence_numbers(obj: Any, out: set | None = None) -> set[float]:
    out = set() if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            evidence_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            evidence_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.add(float(m.replace(",", "")))
            except ValueError:
                pass
    return out


def _decimals(written: str) -> int:
    """Digits after the decimal point, as written: "98,087.50" -> 2, "98,088" -> 0."""
    return len(written.split(".", 1)[1]) if "." in written else 0


def _trailing_zeros(written: str) -> int:
    digits = written.replace(",", "").replace("-", "").split(".")[0]
    return len(digits) - len(digits.rstrip("0")) if "." not in written and digits.strip("0") else 0


def _roundings(value: float, places: int) -> set[Decimal]:
    """`value` rounded to `places` decimals (negative: tens, hundreds, ...), half-up and half-even."""
    q = Decimal(1).scaleb(-places)
    d = Decimal(repr(value))
    return {d.quantize(q, rounding=ROUND_HALF_UP), d.quantize(q, rounding=ROUND_HALF_EVEN)}


def _matches(written: str, v: float) -> bool:
    """`written` (a number as it appears in text) is a rounding of tool value `v`
    -- or of v x 100, for a percentage -- at exactly the precision it is
    written with. "98,087.50" matches 98087.5; "98,087.40" does not; "98,000"
    matches 98087.5 rounded to thousands; "109.3" matches 1.0927 as a percentage."""
    n = Decimal(written.replace(",", "").lstrip("-"))
    places = _decimals(written)
    precisions = [places] + [-k for k in range(1, _trailing_zeros(written) + 1)]
    for cand in (abs(v), abs(v) * 100):
        if not math.isfinite(cand):
            continue
        for pl in precisions:
            if any(r == n.quantize(Decimal(1).scaleb(-pl)) for r in _roundings(cand, pl)):
                return True
    return False


def ungrounded(text: str, evidence: Any, question: str = "") -> list[str]:
    """Numbers in `text` that no tool result (or the user's own question) contains."""
    pool = evidence_numbers(evidence) | evidence_numbers(question)
    bad = []
    for m in _NUM.findall(text or ""):
        n = float(m.replace(",", ""))
        if abs(n) in FREE_NUMBERS or any(_matches(m, v) for v in pool):
            continue
        bad.append(m)
    return bad


# ------------------------------------------------------------------ renderers --

def render(tool: str, r: dict) -> str:
    fn = _RENDER.get(tool)
    return fn(r) if fn else "```json\n" + json.dumps(r, indent=1, default=str)[:3000] + "\n```"


def _metric(r):
    agg = {"avg": "average", "count_distinct": "number of distinct", "std": "standard deviation", "var": "variance"}.get(
        r["aggregation"], r["aggregation"])
    what = f"{agg} {r['column']}" if r.get("column") else "number of rows"
    filt = "" if not r.get("filters") else " where " + " and ".join(f"{x['column']} {x['op']} {x['value']}" for x in r["filters"])
    if "groups" in r:
        lines = [f"**{what.capitalize()} by {r['group_by']}**{filt}:", ""]
        lines += [f"| {r['group_by']} | {what} | rows |", "|---|---:|---:|"]
        lines += [f"| {g['group']} | {f(g['value'])} | {g['rows']} |" for g in r["groups"][:25]]
        if r.get("top"):
            lines += ["", f"Highest: **{r['top']['group']}** ({f(r['top']['value'])})." if r["groups"] and
                      r["groups"][0] is r["top"] else ""]
        return "\n".join(lines)
    return f"The {what}{filt} is **{f(r['value'])}** (computed over {r['rows']} rows)."


def _describe_column(r):
    if not r.get("n"):
        return f"'{r['column']}' has no numeric values."
    ci = r.get("mean_ci95")
    return (f"**{r['column']}** (n = {r['n']}): mean {f(r['mean'])}"
            + (f" (95% CI {f(ci[0])} to {f(ci[1])})" if ci else "")
            + f", median {f(r['median'])}, standard deviation {f(r['sd'])}, variance {f(r['variance'])}, "
              f"range {f(r['min'])} to {f(r['max'])}.\n\nMethod: {r['method']}.")


def _compare(r):
    if "note" in r:
        return r["note"]
    lines = [f"**{r['outcome']} by {r['group']}**", ""]
    if r["kind"] == "numeric":
        lines += ["| group | n | mean | median | sd |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {s['group']} | {s['n']} | {f(s['mean'])} | {f(s['median'])} | {f(s['sd'])} |" for s in r["summary"]]
    else:
        keys = [k for k in r["summary"][0] if k.startswith("share_")]
        lines += ["| group | n | " + " | ".join(k[6:] for k in keys) + " |", "|---|---:|" + "---:|" * len(keys)]
        lines += [f"| {s['group']} | {s['n']} | " + " | ".join(f(s[k], pct=True) for k in keys) + " |" for s in r["summary"]]
    t = r.get("test")
    if t:
        line = f"\n**{t['name']}**: {p(t.get('p_value'))}"
        if "estimate" in t:
            line += f"; difference ({t['difference']}) {f(t['estimate'])}, 95% CI {f(t['ci'][0])} to {f(t['ci'][1])}"
        for k, label in (("hedges_g", "Hedges' g"), ("eta_squared", "eta squared"), ("cramers_v", "Cramér's V")):
            if t.get(k) is not None:
                line += f"; effect size {label} = {f(t[k])}"
        if t.get("robustness"):
            line += f". Rank-based check ({t['robustness']['name']}): {p(t['robustness']['p_value'])}"
        lines.append(line + ".")
        lines.append("Statistically " + ("significant at the 5% level." if (t.get("p_value") or 1) < 0.05 else
                                          "not significant at the 5% level: the data do not show a reliable difference."))
    for a in r.get("assumptions", []):
        lines.append(f"- {a}")
    return "\n".join(lines)


def _correlation(r):
    if "note" in r and "pearson_r" not in r:
        return r["note"]
    return (f"Between **{r['x']}** and **{r['y']}** (n = {r['n']}): Pearson r = {f(r['pearson_r'])} "
            f"(95% CI {f(r['pearson_ci'][0])} to {f(r['pearson_ci'][1])}, {p(r['pearson_p'])}); "
            f"Spearman rho = {f(r['spearman_rho'])} ({p(r['spearman_p'])}). {r['note']}")


def _regression(r):
    if "coefficients" not in r:
        return r["note"]
    lines = [f"OLS regression of **{r['target']}** (n = {r['n']}): R² = {f(r['r2'])}, adjusted R² = {f(r['adj_r2'])}.", "",
             "| term | estimate | std. error | p |", "|---|---:|---:|---:|"]
    lines += [f"| {c['term']} | {f(c['estimate'])} | {f(c['se'])} | {p(c['p_value'])} |" for c in r["coefficients"]]
    return "\n".join(lines + ["", r["note"]])


def _performance_table(perf):
    g = perf["group_column"]
    has_conv = any("conversion_rate" in r for r in perf["groups"])
    head = f"| {g} | rows | spend | revenue | ROI | CAC |" + (" conversion rate (95% CI) |" if has_conv else "")
    lines = [head, "|---|---:|---:|---:|---:|---:|" + ("---:|" if has_conv else "")]
    for r in perf["groups"]:
        row = (f"| {r['group']} | {r['rows']} | {f(r['spend'])} | {f(r.get('revenue'))} | {f(r.get('roi'), pct=True)} "
               f"| {f(r.get('cac'))} |")
        if has_conv:
            ci = r.get("conversion_rate_ci")
            row += f" {f(r.get('conversion_rate'), pct=True)}" + (f" ({f(ci[0], pct=True)} to {f(ci[1], pct=True)})" if ci else "") + " |"
        lines.append(row)
    return "\n".join(lines)


def _comparison_lines(perf):
    comp = perf.get("comparison") or {}
    out = []
    t = comp.get("roi_test")
    if t:
        out.append(f"Per-campaign ROI in {perf['focus']} averaged {f(t['focus_mean'], pct=True)} vs "
                   f"{f(t['other_mean'], pct=True)} elsewhere: difference {f(t['difference'], pct=True)} "
                   f"(95% CI {f(t['ci'][0], pct=True)} to {f(t['ci'][1], pct=True)}, Welch's t-test {p(t['p_value'])}"
                   + (f", Hedges' g = {f(t['hedges_g'])}" if t.get("hedges_g") is not None else "") + ").")
        if t.get("caution"):
            out.append(f"Caution: {t['caution']}.")
    c = comp.get("conversion_test")
    if c:
        out.append(f"Conversion rate in {perf['focus']} {f(c['focus_rate'], pct=True)} vs {f(c['other_rate'], pct=True)} "
                   f"elsewhere: difference {f(c['difference'], pct=True)}"
                   + (f" (95% CI {f(c['ci'][0], pct=True)} to {f(c['ci'][1], pct=True)}" if c.get("ci") else " (")
                   + f", {c['method'].split(' on ')[0]} {p(c['p_value'])}; {c['assumption']}).")
    tr = perf.get("trend")
    if tr:
        out.append(f"Over time in {perf['focus']}: ROI {f(tr['early_roi'], pct=True)} in {tr['early_period'][0]} to "
                   f"{tr['early_period'][1]}, then {f(tr['late_roi'], pct=True)} in {tr['late_period'][0]} to {tr['late_period'][1]}.")
    return out


def _methodology(perf):
    cols = perf["columns_used"]
    used = ", ".join(f"{k} = '{v}'" for k, v in cols.items())
    return [f"Grouped {perf['rows_total']} campaign rows by '{perf['group_column']}'; columns used: {used} "
            f"(chosen by name: {'; '.join(perf['column_evidence'].values())}).",
            "ROI = (revenue - spend) / spend; CAC = spend / conversions; conversion rate = conversions / leads with a "
            "Wilson 95% interval.",
            f"Ranked groups by {perf['ranking_metric']}; compared the focus group with all others combined "
            "(Welch's t-test on per-campaign ROI; chi-square or Fisher's exact test on conversion counts).",
            "No Random Forest or other predictive model was used: this is an analysis of historical results."]


def _performance(perf):
    lines = [f"**Campaign performance by {perf['group_column']}** (ranked by {perf['ranking_metric']}: "
             f"{', '.join(perf['ranking']) or 'none with enough rows'})", "", _performance_table(perf)]
    lines += [""] + [f"- {x}" for x in _comparison_lines(perf)]
    return "\n".join(lines)


def _recommend(r):
    perf, rec = r["performance"], r["recommendation"]
    lines = ["### Recommendation", rec["recommendation"],
             f"Evidence level: **{rec['evidence_level']}** ({rec['rules']}).", "", "### Evidence", _performance_table(perf), ""]
    lines += [f"- {x}" for x in _comparison_lines(perf)]
    lines += ["", "### Methodology"] + [f"- {x}" for x in _methodology(perf)]
    lines += ["", "### Limitations"] + [f"- {x}" for x in rec["limitations"]]
    lines += ["", "### Next action", rec["next_action"]]
    return "\n".join(lines)


def why(last: dict, question: str = "") -> str:
    """The reasons behind the previous recommendation, from its stored evidence."""
    perf, rec = last["performance"], last["recommendation"]
    row = next(r for r in perf["groups"] if r["group"] == rec["focus"])
    parts = [
        "### 1. The business question", f"Where advertising money has worked best, and what to do about {rec['focus']}.",
        "### 2. The data used", f"{perf['rows_total']} historical campaign rows, grouped by '{perf['group_column']}'.",
        "### 3. The analytical method"] + [f"- {x}" for x in _methodology(perf)] + [
        "### 4. The computed results",
        f"{rec['focus']}: spend {f(row['spend'])}, revenue {f(row.get('revenue'))}, ROI {f(row.get('roi'), pct=True)}, "
        f"CAC {f(row.get('cac'))}" + (f", conversion rate {f(row.get('conversion_rate'), pct=True)}" if row.get("conversion_rate") is not None else "")
        + f" over {row['rows']} campaign rows.", "", _performance_table(perf),
        "### 5. The statistical evidence"] + [f"- {x}" for x in _comparison_lines(perf)] + [
        "### 6. The uncertainty",
        "The 95% intervals above are the plausible range of each difference; an interval that includes zero means the "
        "data cannot rule out no real difference." + (f" With only {row['rows']} campaign rows in {rec['focus']}, the "
                                                       "estimates are imprecise." if row["rows"] < 10 else ""),
        "### 7. Why this supports the recommendation",
        f"Rule applied: {rec['rules']}. Result: evidence level **{rec['evidence_level']}**, so the recommendation is: "
        f"{rec['recommendation']}",
        "### 8. Limitations"] + [f"- {x}" for x in rec["limitations"]]
    return "\n".join(parts)


def _scenario(r):
    a, b = r["if_returns_scale"], r["if_no_response"]
    return "\n".join([
        f"**Scenario: change the budget by {r['change_pct']:+g}% ({r['scope']})**", "",
        f"Baseline: spend {f(r['baseline_spend'])}, revenue {f(r['baseline_revenue'])}, ROI {f(r['baseline_roi'], pct=True)}.",
        f"New spend: {f(r['new_spend'])} (extra {f(r['extra_spend'])}).", "",
        "| assumption | revenue | profit | ROI |", "|---|---:|---:|---:|",
        f"| {a['assumption']} | {f(a['revenue'])} | {f(a['profit'])} | {f(a['roi'], pct=True)} |",
        f"| {b['assumption']} | {f(b['revenue'])} | {f(b['profit'])} | {f(b['roi'], pct=True)} |", "",
        f"To break even on the extra spend, it must bring at least {f(r['break_even_extra_revenue'])} of extra revenue.",
        f"\n_{r['caveat']}_"])


METRIC_NOTES = {
    "classification": [
        "Accuracy: share of test rows predicted correctly. Misleading when one class dominates: always guessing the "
        "common class can score high.",
        "Balanced accuracy and macro F1 weigh every class equally, so they expose a model that ignores a rare class.",
        "Precision: of the rows predicted as a class, how many really were. Recall: of the rows really in a class, how "
        "many the model found. Which matters more depends on the cost of a false alarm versus a miss.",
        "ROC AUC (two classes): how well the model ranks positive above negative rows; 0.5 is chance, 1.0 is perfect.",
    ],
    "regression": [
        "MAE: the typical size of an error, in the target's own units. RMSE and MSE punish large errors more.",
        "R²: the share of the variation in the test rows the model explains; it can be negative when the model is "
        "worse than predicting the average.",
    ],
}


def _forest(r):
    m = r["metrics"]
    sp = r.get("split") or {}
    split = (f"{int(round(sp['train_share'] * 100))}/{int(round(sp['test_size'] * 100))} train/test split, seed {sp['seed']}"
             + (", stratified by class" if sp.get("stratified") else "")) if sp else ""
    lines = [f"### Random Forest {r['task']} for '{r['target']}'",
             f"Trained on **{r['train_rows']}** rows, evaluated on **{r['test_rows']}** held-out rows it never saw"
             + (f" ({split})" if split else "") + ".", ""]
    if r["task"] == "classification":
        lines += ["| metric | model | baseline |", "|---|---:|---:|",
                  f"| accuracy | {f(m['accuracy'])} | {f(m['baseline_accuracy'])} |",
                  f"| balanced accuracy | {f(m['balanced_accuracy'])} | {f(m.get('baseline_balanced_accuracy'))} |",
                  f"| macro precision | {f(m.get('macro_precision'))} | |",
                  f"| macro recall | {f(m.get('macro_recall'))} | |",
                  f"| macro F1 | {f(m['macro_f1'])} | |"]
        if "roc_auc" in m:
            lines.append(f"| ROC AUC | {f(m['roc_auc'])} | |")
        lines += ["", f"Baseline: {m['baseline']}."]
        if m.get("per_class"):
            lines += ["", "| class | precision | recall | F1 | test rows |", "|---|---:|---:|---:|---:|"]
            lines += [f"| {c['class']} | {f(c['precision'])} | {f(c['recall'])} | {f(c['f1'])} | {c['support']} |"
                      for c in m["per_class"]]
        cm = m.get("confusion_matrix")
        if cm:
            lines += ["", "**Confusion matrix** (rows: actual, columns: predicted)", "",
                      "| actual \\ predicted | " + " | ".join(cm["labels"]) + " |", "|---|" + "---:|" * len(cm["labels"])]
            lines += [f"| {lab} | " + " | ".join(str(v) for v in row) + " |" for lab, row in zip(cm["labels"], cm["matrix"])]
        bal = r.get("class_balance")
        if bal:
            lines += ["", "Class sizes in the data: " + ", ".join(f"{k} {v}" for k, v in bal["counts"].items())
                      + (" — **imbalanced**." if bal["imbalanced"] else ".")]
    else:
        lines += ["| metric | model | baseline |", "|---|---:|---:|",
                  f"| MAE | {f(m['mae'])} | {f(m['baseline_mae'])} |",
                  f"| RMSE | {f(m['rmse'])} | |",
                  f"| MSE | {f(m.get('mse'))} | |",
                  f"| R² | {f(m['r2'])} | |", "", f"Baseline: {m['baseline']}."]
    judged_on = "balanced accuracy" if r["task"] == "classification" else "MAE"
    lines += ["", "**Verdict:** the model " + ("beats" if r["beats_baseline"] else "does **not** clearly beat")
              + f" the simple baseline on the held-out rows, judged on {judged_on}."]
    if r["task"] == "classification" and m["accuracy"] < m["baseline_accuracy"]:
        lines.append(f"Its plain accuracy ({f(m['accuracy'])}) is below the baseline's ({f(m['baseline_accuracy'])}) because "
                     "the baseline never predicts the rare class at all; the model finds some of those cases at the cost "
                     "of false alarms (see the confusion matrix).")
    lines += ["",
              "**Most important inputs** (permutation importance on held-out rows)", "",
              "| feature | importance | sd |", "|---|---:|---:|"]
    lines += [f"| {i['feature']} | {f(i['importance'], digits=4)} | {f(i['sd'], digits=4)} |" for i in r["importance"][:10]]
    if r["dropped"]:
        lines += ["", "**Left out to keep the test honest:** " + "; ".join(f"{d['column']} ({d['reason']})" for d in r["dropped"])]
    lines += ["", f"Method: {r['method']}. Preprocessing is fitted on the training rows only."]
    lines += [f"- {x}" for x in r["limitations"]]
    lines += ["", "**Reading the metrics**"] + [f"- {x}" for x in METRIC_NOTES[r["task"]]]
    lines += ["", "To predict a new case, give its values, for example: _predict for " + ", ".join(
        f"{c} = …" for c in r["features"][:3]) + "_."]
    return "\n".join(lines)


def _importance(r):
    lines = [f"**What the '{r['target']}' model relied on** (permutation importance: how much held-out performance "
             "drops when the input is shuffled)", "", "| feature | importance | sd |", "|---|---:|---:|"]
    lines += [f"| {i['feature']} | {f(i['importance'], digits=4)} | {f(i['sd'], digits=4)} |" for i in r["importance"][:10]]
    return "\n".join(lines + ["", f"Method: {r['method']}."] + [f"- {x}" for x in r["limitations"]])


def _value(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return f(v)
    return str(v)


def _predict(r):
    lines = [f"### Predicted {r['target']}"]
    for n, x in enumerate(r["predictions"], 1):
        pred = x["prediction"] if isinstance(x["prediction"], str) else f(x["prediction"])
        lines += ["", f"**{'Case ' + str(n) + ': ' if len(r['predictions']) > 1 else ''}{r['target']} = {pred}**"]
        if x.get("probabilities"):
            lines.append("Model probabilities: " + ", ".join(f"{k} {f(v, pct=True)}" for k, v in x["probabilities"].items()) + ".")
        lines += ["", "| input | value |", "|---|---|"]
        lines += [f"| {k} | {_value(v)} |" for k, v in x["input"].items()]
        if x["imputed"]:
            lines.append(f"\nNot given, so filled with typical training values: {', '.join(x['imputed'])}.")
        if x.get("ignored"):
            lines.append(f"Not used by the model: {', '.join(x['ignored'])}.")
    mdl = r.get("model")
    if mdl:
        mm = mdl["metrics"]
        ctx = (f"balanced accuracy {f(mm['balanced_accuracy'])}" if mdl["task"] == "classification"
               else f"MAE {f(mm['mae'])}, R² {f(mm['r2'])}")
        lines += ["", f"Model: Random Forest {mdl['task']} trained in this conversation; on {mdl['test_rows']} held-out "
                      f"test rows it scored {ctx}."]
    lines.append("_A prediction is the model's best estimate from patterns in past data, not a certainty; it is only "
                 "as good as the held-out evaluation above._")
    return "\n".join(lines)


def _dataset(r):
    s = r["schema"]
    return f"{s['rows']} rows, {len(s['columns'])} columns: " + ", ".join(f"{c['name']} ({c['kind']})" for c in s["columns"])


_RENDER = {"compute_metric": _metric, "describe_column": _describe_column, "compare_groups": _compare,
           "correlation": _correlation, "linear_regression": _regression, "business_performance": _performance,
           "recommend": _recommend, "budget_scenario": _scenario, "random_forest": _forest,
           "feature_importance": _importance, "predict": _predict, "describe_dataset": _dataset}
