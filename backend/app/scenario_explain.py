"""Plain-English explanations of what-if results.

An explanation is written from a result the engines already computed and verified, using a fixed template. It never
calls a model, so it exists whether or not a provider is available, and the same result always gives the same text.

Rules it keeps, each tested:

* Every number in the text appears in the result it explains (checked with the same precision-aware grounding the Silt
  Copilot uses for model-written prose). Nothing is computed here that the engine did not compute.
* It states how the result changed and how much each input accounts for, using the model's own attribution. It says "accounts
  for", never "because" or "caused": no causal claim is made.
* Assumptions, risks and what cannot be concluded are separate sections, so a reader can find them.

`polish` can rewrite the text with a language model, but a rewrite is accepted only if every number in it is still in the
result and it adds no causal wording; otherwise the template text is kept.
"""
from __future__ import annotations

import re
from typing import Any, Callable

EXPLAIN_VERSION = "scenario-explain.1"
STANDARD = ("optimistic", "pessimistic")
CANNOT = [
    "This is arithmetic on the numbers you set, not a forecast of what customers, costs or competitors will do.",
    "Nothing here shows why an input would move, and a what-if is not evidence of causation.",
]


def n(x: Any) -> str:
    """A level, with its sign: a profit of -200 is written -200. Whole above 1,000, two decimals below."""
    if x is None:
        return "n/a"
    x = float(x)
    body = f"{abs(x):,.0f}" if abs(x) >= 1000 else (f"{abs(x):.2f}".rstrip("0").rstrip(".") or "0")
    return f"-{body}" if x < 0 and body != "0" else body


def a(x: Any) -> str:
    """A size of change, without its sign: direction is written in words beside it (rises, adds, down)."""
    return "n/a" if x is None else n(abs(float(x)))


def pc(x: Any) -> str:
    return "n/a" if x is None else f"{abs(float(x)):.1f}%"


def _move(old: float | None, new: float | None) -> str:
    if old is None or new is None:
        return "is undefined"
    if new > old:
        return "rises"
    return "falls" if new < old else "is unchanged"


def _parts(sections: dict) -> str:
    return " ".join(x for v in sections.values() for x in v)


def _drivers(contrib: dict, labels: dict, noun: str = "change") -> list[str]:
    """The inputs that account for the change, largest first, from the model's own attribution."""
    items = sorted(((k, v) for k, v in contrib.items() if v), key=lambda kv: -abs(kv[1]))
    if not items:
        return []
    out = []
    k, v = items[0]
    out.append(f"The largest share of the {noun} comes from {labels.get(k, k)}, which {'adds' if v > 0 else 'takes away'} {a(v)}.")
    if len(items) > 1:
        rest = ", ".join(f"{labels.get(k, k)} ({'adds' if v > 0 else 'takes away'} {a(v)})" for k, v in items[1:4])
        out.append(f"The rest comes from {rest}.")
        if any(v < 0 for _, v in items) and any(v > 0 for _, v in items):
            out.append("Some inputs work against each other, so the net change is smaller than the largest share on its own.")
    return out


# ------------------------------------------------------------------- scenarios --

def _scenario(res: dict) -> dict:
    out_label = res["output_labels"][res["primary_output"]]
    il = res.get("input_labels", {})
    primary = res["primary_output"]
    base = res["baseline"]["outputs"][primary]
    named = [k for k in res["scenarios"] if k not in STANDARD]
    shown = named[:3] or []
    what, why, assumptions, risks = [], [], [], []
    if not shown:
        what.append(f"No adjustment has been made yet. The baseline {out_label.lower()} is {n(base)}.")
    for name in shown:
        s = res["scenarios"][name]
        new = s["outputs"][primary]
        d = s["delta_vs_baseline"].get(primary) or {}
        what.append(f"In '{name}', {out_label.lower()} {_move(base, new)} from {n(base)} to {n(new)}"
                    + (f" ({pc(d.get('pct'))} {'more' if (d.get('abs') or 0) > 0 else 'less'})" if d.get("pct") is not None and d.get("abs") else "") + ".")
        changed = ", ".join(f"{il.get(k, k)} {_adj(adj)}" for k, adj in s["adjustments"].items())
        if changed:
            what.append(f"What was changed in '{name}': {changed}.")
        why += _drivers(s["contributions"]["by_input"], il)
        comp = s["contributions"].get("computation") or {}
        if comp.get("method") == "sampled":
            why.append("With this many changed inputs the shares are estimates from sampled orders; they still add up exactly to the total change.")
        if new is not None and base is not None and new < 0 <= base:
            risks.append(f"'{name}' turns {out_label.lower()} negative ({n(new)}).")
    if {"optimistic", "pessimistic"} <= set(res["scenarios"]):
        lo, hi = (res["scenarios"][k]["outputs"][primary] for k in ("pessimistic", "optimistic"))
        what.append(f"Moving every input in its favourable or unfavourable direction gives {out_label.lower()} between {n(lo)} and {n(hi)}.")
    sens = res.get("sensitivity", {}).get("rows") or []
    if sens:
        top = sens[0]
        risks.append(f"{out_label} is most sensitive to {top['label']}: moving it by {n(res['sensitivity']['pct'])}% either way takes it from "
                     f"{n(top['low'])} to {n(top['high'])}.")
    be = (res.get("break_even") or {}).get("baseline")
    if be and be.get("status") == "reachable" and be.get("safety_margin_pct") is not None:
        m = be["safety_margin_pct"]
        risks.append(f"Break-even is at {n(be['break_even_quantity'])} units; the baseline sells {pc(m)} "
                     + ("more than that." if m >= 0 else "fewer than that, so it is below break-even."))
    elif be and be.get("status") == "never":
        risks.append("At these prices and costs the baseline has no break-even point: " + be["explanation"].split(":")[0].rstrip(".") + ".")
    for key, notes in (res.get("constraint_notes") or {}).items():
        for note in notes:
            risks.append(f"{'The baseline' if key == 'baseline' else key} breaks a rule of the model: {note['message']}.")
    for k, v in res.get("assumptions", {}).items():
        src = (res.get("assumption_sources") or {}).get(k, {})
        assumptions.append(f"{k.replace('_', ' ').capitalize()} is {n(v)}"
                           + (" (your value)" if not src or src.get("source") == "manual" else " (taken from an estimate you chose)") + ".")
    assumptions += [u.rstrip(".") + "." for u in res.get("unsupported_assumptions", [])]
    return {"what_changed": what, "why": why, "assumptions": assumptions, "risks": risks, "cannot_conclude": list(CANNOT)}


def _adj(a_: dict) -> str:
    v = a_["value"]
    return {"pct": f"{'up' if v >= 0 else 'down'} {n(abs(v))}%", "abs": f"{'up' if v >= 0 else 'down'} by {n(abs(v))}", "set": f"set to {n(v)}"}.get(a_["type"], "changed")


# ------------------------------------------------------------------ simulation --

def _simulation(res: dict) -> dict:
    t = res["target"]
    label = t["label"].lower()
    p = t["probabilities"]
    what = [f"Across {res['config']['iterations']:,} simulated runs, {label} has an expected value of {n(t['expected'])}, with a typical range "
            f"(5th to 95th percentile) of {n(t['low'])} to {n(t['high'])}."]
    what.append(f"{pc(p['positive']['probability'] * 100)} of the runs are positive and {pc(p['negative']['probability'] * 100)} are negative.")
    if p.get("exceeds_baseline"):
        what.append(f"{pc(p['exceeds_baseline']['probability'] * 100)} of the runs beat today's {n(t['baseline_value'])}.")
    if p.get("reaches_target"):
        rt = p["reaches_target"]
        what.append(f"{pc(rt['probability'] * 100)} of the runs are {'at least' if rt['direction'] == 'at_least' else 'at most'} your target of {n(rt['target'])}.")
    why = [f"The spread comes from the inputs you made uncertain: {', '.join(res['uncertain_inputs'])}."]
    risks = []
    if p["negative"]["probability"] and p["negative"]["probability"] > 0.05:
        risks.append(f"There is a {pc(p['negative']['probability'] * 100)} chance of a negative {label} under these ranges.")
    if res.get("constraint_rejected_fraction"):
        risks.append(f"{pc(res['constraint_rejected_fraction'] * 100)} of the draws broke a rule of the model and are left out.")
    assumptions = [d.rstrip(".") + "." for d in res["disclosures"]]
    return {"what_changed": what, "why": why, "assumptions": assumptions, "risks": risks,
            "cannot_conclude": CANNOT + ["The ranges are assumptions you supplied, so the probabilities are only as good as those ranges."]}


# ------------------------------------------------------------------- goal seek --

def _goal_seek(res: dict) -> dict:
    t, sv = res["target"], res["solve_for"]
    what = [res["message"]]
    why, risks = [], []
    sols = res["solutions"]
    if res["recommended_index"] is not None:
        best = sols[res["recommended_index"]]
        pct = best["change_vs_baseline"]["pct"]
        why.append(f"{sv['label']} would move from {n(sv['baseline_value'])} to {n(best['value'])}"
                   + (f", {'a rise' if pct >= 0 else 'a cut'} of {pc(pct)}" if pct is not None else "") + f", with everything else as set; that gives {t['label'].lower()} of {n(best['achieved'])}.")
        for c in best["constraints"]:
            risks.append(f"Constraint {c['label'].lower()} {c['op']} {n(c['value'])}: {'met' if c['ok'] else 'not met'}.")
    elif res["status"] in ("no_solution", "no_feasible_solution") and res.get("achievable_range"):
        r = res["achievable_range"]
        why.append(f"Over the range searched, {t['label'].lower()} only reaches {n(r['min'])} to {n(r['max'])}.")
    if res["status"] == "multiple_solutions":
        risks.append("More than one value reaches the target; check which is realistic for your business.")
    return {"what_changed": what, "why": why, "assumptions": [d.rstrip(".") + "." for d in res["disclosures"]], "risks": risks,
            "cannot_conclude": CANNOT + ["A solution only says what value reaches the target in this model; it does not say the market would accept it."]}


# ----------------------------------------------------------- period comparison --

def _period(res: dict) -> dict:
    p, ol, il = res["primary_output"], res["output_labels"], res.get("input_labels", {})
    a, b = res["a"]["outputs"][p], res["b"]["outputs"][p]
    ch = res["output_changes"].get(p) or {}
    la = (res.get("periods", {}).get("a", {}).get("period") or {}).get("label", "period A")
    lb = (res.get("periods", {}).get("b", {}).get("period") or {}).get("label", "period B")
    what = [f"{ol[p]} {_move(a, b)} from {n(a)} in {la} to {n(b)} in {lb}"
            + (f" ({pc(ch.get('pct'))} {'more' if (ch.get('abs') or 0) > 0 else 'less'})" if ch.get("pct") is not None and ch.get("abs") else "") + "."]
    moved = [f"{il.get(k, k)} {_move(res['a']['inputs'][k], res['b']['inputs'][k]).replace('is unchanged', 'did not change')}" for k, c in res["input_changes"].items() if c["abs"]]
    if moved:
        what.append("Between the periods: " + ", ".join(moved) + ".")
    why = _drivers(res["contributions"]["by_input"], il, "recorded change")
    return {"what_changed": what, "why": why, "assumptions": ["Both periods are what your data recorded; nothing was adjusted."],
            "risks": [f"Period {k}: the recorded figures break a rule of the model: {x['message']}." for k, v in (res.get("constraint_notes") or {}).items() for x in v],
            "cannot_conclude": ["This describes what happened in the two periods. It does not say why the inputs moved, and it is not a forecast."]}


# -------------------------------------------------------------------- segments --

def _segments(res: dict) -> dict:
    p, ol = res["primary_output"], res["output_labels"]
    o = res["overall"]
    base, scn = o["baseline"]["outputs"][p], o["scenario"]["outputs"][p]
    ch = o["delta"].get(p) or {}
    what = [f"Total {ol[p].lower()} {_move(base, scn)} from {n(base)} to {n(scn)}"
            + (f" ({pc(ch.get('pct'))} {'more' if (ch.get('abs') or 0) > 0 else 'less'})" if ch.get("pct") is not None and ch.get("abs") else "") + "."]
    why = []
    c = res.get("contributions")
    if c and c["total_change"]:
        ranked = sorted(c["by_segment"].items(), key=lambda kv: -abs(kv[1]["abs"]))
        top = [(k, v) for k, v in ranked if v["abs"]][:3]
        why.append("By segment: " + ", ".join(
            (f"{k} accounts for {pc(v['share_pct'])} of the change ({a(v['abs'])})" if v["abs"] * c["total_change"] > 0
             else f"{k} works against the change by {a(v['abs'])}") for k, v in top) + ".")
    thin = [s["value"] for s in res["segments"] if s["thin"]]
    return {"what_changed": what, "why": why, "assumptions": [d.rstrip(".") + "." for d in res["disclosures"]],
            "risks": ([f"These segments rest on very few rows: {', '.join(thin)}."] if thin else []), "cannot_conclude": CANNOT}


# ------------------------------------------------------------------------ grid --

def _grid(res: dict) -> dict:
    label = res["output"]["label"].lower()
    xl, yl = res["x"]["label"].lower(), res["y"]["label"].lower()
    what = [f"Across the ranges shown, {label} runs from {n(res['worst']['value'])} to {n(res['best']['value'])}."] if res["best"] else ["No cell has a value."]
    why = []
    if res["best"]:
        why.append(f"The best cell on this grid is at {xl} {n(res['best']['x'])} and {yl} {n(res['best']['y'])}; the worst is at "
                   f"{xl} {n(res['worst']['x'])} and {yl} {n(res['worst']['y'])}.")
    cur = res["current"]
    why.append(f"Where you are now ({xl} {n(cur['x'])}, {yl} {n(cur['y'])}) {label} is {n(cur['value'])}.")
    risks = []
    b = res.get("boundary")
    if b and b["points"]:
        risks.append(f"{label.capitalize()} crosses {n(b['threshold'])} inside these ranges, with {b['cells_below']:,} of {res['cells']:,} cells on the lower side.")
    elif b:
        risks.append(f"{label.capitalize()} never crosses {n(b['threshold'])} inside these ranges.")
    return {"what_changed": what, "why": why, "assumptions": [d.rstrip(".") + "." for d in res["disclosures"]], "risks": risks,
            "cannot_conclude": CANNOT + ["The best cell is the best on this grid, not necessarily the best possible."]}


# ------------------------------------------------------------------ elasticity --

def _elasticity(res: dict) -> dict:
    if res["status"] != "estimated":
        return {"what_changed": ["No estimate was made."], "why": [x.rstrip(".") + "." for x in res["reasons"]], "assumptions": [],
                "risks": [], "cannot_conclude": ["Do not use a price elasticity from this data."]}
    e, d, rel = res["elasticity"], res["diagnostics"], res["reliability"]
    what = [f"The data suggests a price elasticity of {n(e['estimate'])}, meaning a 1% price rise goes with about a {n(e['estimate'])}% "
            f"change in quantity. The {n(e['level'] * 100)}% interval is {n(e['ci_low'])} to {n(e['ci_high'])}, from {d['observations']:,} observations."]
    return {"what_changed": what, "why": [rel["recommendation"]], "assumptions": [f"Method: {e['method']}."],
            "risks": [f.rstrip(".") + "." for f in rel["flags"]], "cannot_conclude": res["limitations"]}


_BUILDERS: dict[str, Callable[[dict], dict]] = {
    "scenario": _scenario, "simulation": _simulation, "goal_seek": _goal_seek, "period_comparison": _period,
    "segment_scenario": _segments, "sensitivity_grid": _grid, "elasticity_estimate": _elasticity,
}


def explain(result: dict) -> dict:
    """The explanation for any what-if result: a one-paragraph summary and labelled sections, all from the result."""
    build = _BUILDERS.get(result.get("kind"))
    if build is None:
        return {"version": EXPLAIN_VERSION, "source": "template", "summary": "", "sections": {}}
    sections = build(result)
    summary = " ".join(sections["what_changed"][:2] + sections["why"][:1])
    return {"version": EXPLAIN_VERSION, "source": "template", "summary": summary, "sections": sections,
            "note": "Written from the verified result by a fixed template; every number in it is in that result."}


CAUSAL = re.compile(r"\b(because|caused?|causing|due to|led to|leads to|driven by|drives|results? from|as a result|owing to|thanks to)\b", re.I)


def polish(explanation: dict, result: dict, rewrite: Callable[[str], str] | None) -> dict:
    """An optional model rewrite of the summary. Kept only if every number is still in the result and no causal wording
    appears; otherwise the template text stands and says why."""
    from app.analyst import explain as grounding

    if rewrite is None:
        return {**explanation, "polish": {"used": False, "reason": "no language model is available; the template text is shown"}}
    try:
        text = (rewrite(explanation["summary"]) or "").strip()
    except Exception as e:  # noqa: BLE001 - the model is optional; the template always works
        return {**explanation, "polish": {"used": False, "reason": f"the model could not be reached ({type(e).__name__})"}}
    if not text:
        return {**explanation, "polish": {"used": False, "reason": "the model returned nothing"}}
    bad = grounding.ungrounded(text, result)
    if bad:
        return {**explanation, "polish": {"used": False, "reason": f"the rewrite contained numbers that are not in the result ({', '.join(bad[:3])})"}}
    if CAUSAL.search(text):
        return {**explanation, "polish": {"used": False, "reason": "the rewrite used causal wording"}}
    return {**explanation, "summary": text, "source": "model_rewrite", "polish": {"used": True, "reason": "every number in the rewrite is in the result"}}
