"""Scenario discovery: which questions are worth asking, and which of them can be answered.

Three sources feed one list:
    dataset-driven   questions built on the columns and relationships actually in the profile
    domain-driven    the domain pack's templates, each checked against the profile
    user-driven      an objective ("I am a college administrator preparing for next year...") ranks the templates

Every suggestion is classified:
    ready               the data and everything the question needs are present
    needs_assumptions   a value the data does not hold must be supplied (it is named, never invented)
    needs_external_data the answer needs information no dataset holds
    unsupported         the data or the existing methods cannot answer it reliably

The whole thing is rules; no model is needed. An optional `ranker` (a language model adapter) may only re-order or
re-justify *existing* templates by id: anything it returns that is not a real template is dropped, so it cannot add a question
about a column or a method that does not exist. If the ranker fails, the rules result stands.
"""
from __future__ import annotations

import re
from typing import Callable

import pandas as pd

from app.whatif import operations
from app.whatif.domains import GENERIC_TEMPLATES, find_template, get_domain, list_domains
from app.whatif.domains.base import NEEDS_ASSUMPTIONS, NEEDS_EXTERNAL_DATA, READY, UNSUPPORTED, Template
from app.whatif.profile import build_profile, map_concepts, tokens_of

RANK = {READY: 0, NEEDS_ASSUMPTIONS: 1, NEEDS_EXTERNAL_DATA: 2, UNSUPPORTED: 3}
MAX_GENERIC = 3
STOP = {"i", "am", "a", "an", "the", "to", "of", "and", "for", "my", "we", "our", "help", "me", "identify", "important", "scenarios", "before", "next", "that", "this", "is", "are", "what", "if", "will", "can", "should", "with", "in", "on", "be", "it", "do", "how"}


def question_text(template: Template, values: dict | None = None, columns: dict | None = None) -> str:
    """The template's question with known values filled in and unknown ones shown as …"""
    v = {**{k: val for k, val in (columns or {}).items()}, **(values or {})}

    class _Fill(dict):
        def __missing__(self, key):
            return "…"
    flat = _Fill({k: (val[0] if isinstance(val, list) and val else val) for k, val in v.items()})
    return template.question.format_map(flat)


def _association_support(df: pd.DataFrame | None, profile: dict, x: str, y: str) -> tuple[bool, str]:
    """Whether the data could support estimating y from x. With the table, the same test the calculation applies."""
    if df is not None and x in df.columns and y in df.columns:
        import math
        from scipy import stats
        pair = pd.DataFrame({"x": pd.to_numeric(df[x], errors="coerce"), "y": pd.to_numeric(df[y], errors="coerce")}).dropna()
        if len(pair) < operations.MIN_ASSOCIATION_ROWS:
            return False, f"only {len(pair)} complete rows of {x} and {y} (at least {operations.MIN_ASSOCIATION_ROWS} are needed)"
        if pair["x"].nunique() < 3:
            return False, f"{x} takes too few distinct values"
        fit = stats.linregress(pair["x"], pair["y"])
        if not math.isfinite(fit.pvalue) or fit.pvalue >= operations.ALPHA:
            return False, f"no reliable relationship between {x} and {y} in this data (r = {fit.rvalue:.2f})"
        return True, ""
    rel = next((r for r in profile.get("relationships", []) if {r["a"], r["b"]} == {x, y}), None)
    if rel is None:
        return False, f"no clear relationship between {x} and {y} was found"
    return (rel["n"] >= operations.MIN_ASSOCIATION_ROWS, "" if rel["n"] >= operations.MIN_ASSOCIATION_ROWS else "too few rows")


def assess(template: Template, domain_key: str, profile: dict | None, df: pd.DataFrame | None = None, columns: dict | None = None) -> dict:
    """Feasibility of one template against a dataset profile (or none): status, available variables, what is missing and why."""
    dom = get_domain(domain_key)
    base = {"feasibility": READY, "available": {}, "missing": [], "limitations": [template.limitation] if template.limitation else []}
    if template.unsupported:
        return {**base, "feasibility": UNSUPPORTED, "limitations": [template.unsupported]}
    if template.operation == "none":
        return {**base, "feasibility": NEEDS_EXTERNAL_DATA, "missing": list(template.external)}
    mapped = {}
    if profile is not None:
        mapped = {k: v["columns"] for k, v in map_concepts(profile, domain_key).items()} if domain_key != "custom" else {}
        mapped.update({k: ([v] if isinstance(v, str) else v) for k, v in (columns or {}).items()})
        ambiguous = {k: v["candidates"] for k, v in (map_concepts(profile, domain_key).items() if domain_key != "custom" else []) if v["ambiguous"]}
    else:
        ambiguous = {}
    base["available"] = {k: v for k, v in mapped.items() if v}
    is_dataset_op = template.operation in operations.OPERATIONS
    missing: list[str] = []
    if is_dataset_op:
        if profile is None:
            return {**base, "feasibility": NEEDS_EXTERNAL_DATA, "missing": ["a dataset containing " + ", ".join(
                (dom.concept(c).label if dom.concept(c) else c) for c in template.requires)]}
        absent = [c for c in template.requires if not mapped.get(c) and c not in ambiguous]
        if absent:
            labels = [dom.concept(c).label if dom.concept(c) else c for c in absent]
            return {**base, "feasibility": NEEDS_EXTERNAL_DATA, "missing": [f"a column for {', '.join(labels)}"]}
        if ambiguous and any(k in ambiguous for k in template.requires):
            return {**base, "feasibility": NEEDS_ASSUMPTIONS, "missing": [f"which column is {k}: {', '.join(v)}" for k, v in ambiguous.items() if k in template.requires]}
        if template.operation == "association":
            x = (mapped.get(template.params["x"]) or [None])[0]
            ys = mapped.get(template.params["y"]) or []
            if len(ys) > 1:
                return {**base, "feasibility": NEEDS_ASSUMPTIONS, "missing": [f"which {template.params['y']} column to estimate: {', '.join(ys)}"]}
            ok, why = _association_support(df, profile, x, ys[0]) if x and ys else (False, "columns missing")
            if not ok:
                return {**base, "feasibility": UNSUPPORTED, "limitations": [f"Cannot be estimated reliably: {why}.", *base["limitations"]]}
        if template.operation == "forecast_trend" and df is not None:
            dc = (mapped.get(template.params["date"]) or [None])[0]
            d = pd.to_datetime(df[dc], errors="coerce") if dc in df.columns else pd.Series(dtype="datetime64[ns]")
            periods = d.dropna().dt.to_period(template.params.get("freq", "M")).nunique()
            if periods < operations.MIN_TREND_PERIODS:
                return {**base, "feasibility": UNSUPPORTED, "limitations": [f"Only {periods} periods of history; at least {operations.MIN_TREND_PERIODS} are needed.", *base["limitations"]]}
    for s in template.inputs:
        if s.role in ("baseline", "assumption") and s.default is None and not s.optional:
            if s.from_data:
                concept = s.from_data.partition(":")[2]
                if s.from_data.startswith(("row_count", "entity_count")) or mapped.get(concept):
                    continue
            missing.append(f"{s.label}" + (f" ({s.help})" if s.help else ""))
    if missing and not is_dataset_op and isinstance(template.model, str) and profile is not None and df is not None:
        from app.analyst import scenario_bridge
        fit = scenario_bridge.infer_model(df)
        if fit and fit[0] == template.model:
            missing = [m for m in missing if not any(w in m.lower() for w in ("price", "units sold", "variable cost"))]
    if missing:
        return {**base, "feasibility": NEEDS_ASSUMPTIONS, "missing": missing}
    return base


def _suggestion(template: Template, domain_key: str, a: dict, source: str, reason: str | None = None, columns: dict | None = None, values: dict | None = None,
                relevance: float = 0.0) -> dict:
    cols = columns or {}
    return {"id": f"{domain_key}:{template.id}" + (":" + "+".join(sum(([c for c in v] if isinstance(v, list) else [v] for v in cols.values()), [])) if cols else ""),
            "template_id": template.id, "domain": domain_key, "question": question_text(template, values, cols), "source": source,
            "reason": reason or template.reason, "required_variables": [(get_domain(domain_key).concept(c).label if get_domain(domain_key).concept(c) else c) for c in template.requires],
            "available_variables": a["available"], "missing_inputs": a["missing"], "method": template.method, "feasibility": a["feasibility"],
            "limitations": a["limitations"], "evidence_kind": template.evidence, "columns": cols, "relevance": round(relevance, 3),
            "inputs": [{"name": s.name, "label": s.label, "unit": s.unit, "role": s.role, "default": s.default, "min": s.min, "max": s.max,
                        "optional": s.optional, "help": s.help} for s in template.inputs]}


def _relevance(objective: str | None, t: Template) -> float:
    if not objective:
        return 0.0
    words = {w for w in tokens_of(objective) if w not in STOP and len(w) > 2}
    pool = set(t.tags) | tokens_of(t.title) | tokens_of(t.question.replace("{", " ").replace("}", " "))
    return len(words & pool) / (len(words) or 1)


def generic_suggestions(profile: dict, df: pd.DataFrame | None, domain_key: str, used: set[str]) -> list[dict]:
    """Questions built from columns and relationships the data itself shows, for any dataset."""
    out = []
    shift, assoc = GENERIC_TEMPLATES["generic_shift"], GENERIC_TEMPLATES["generic_association"]
    clean = [c for c in profile["metrics"] if c not in used and not profile_col(profile, c)["sensitive"]]
    for r in profile["relationships"]:
        if len([s for s in out if s["template_id"] == "generic_association"]) >= MAX_GENERIC:
            break
        if r["a"] in clean or r["b"] in clean:
            x, y = (r["a"], r["b"]) if r["a"] in clean else (r["b"], r["a"])
            a = assess(assoc, domain_key, profile, df, {"x": [x], "y": [y]})
            out.append(_suggestion(assoc, domain_key, a, "dataset", f"{x} and {y} move together in your data (correlation {r['pearson']:+.2f}).",
                                   {"x": [x], "y": [y]}, {"change": 10}))
    if domain_key == "custom" or not out:
        for c in clean[:MAX_GENERIC]:
            a = assess(shift, domain_key, profile, df, {"metric": [c]})
            out.append(_suggestion(shift, domain_key, a, "dataset", f"'{c}' is a numeric column in your data.", {"metric": [c]}, {"change": 10}))
    return out


def profile_col(profile: dict, name: str) -> dict:
    return next(c for c in profile["columns"] if c["name"] == name)


def suggest(profile: dict | None, domain_key: str, *, df: pd.DataFrame | None = None, objective: str | None = None,
            ranker: Callable[[str, list[dict]], list] | None = None, limit: int = 30) -> dict:
    """Suggested questions for a domain (and dataset, and objective), classified by feasibility."""
    dom = get_domain(domain_key)
    items: list[dict] = []
    for t in dom.templates:
        a = assess(t, domain_key, profile, df)
        src = "dataset" if profile is not None and a["feasibility"] in (READY, NEEDS_ASSUMPTIONS) and t.requires else "domain"
        items.append(_suggestion(t, domain_key, a, src, relevance=_relevance(objective, t)))
    used = {c for s in items for cols in s["available_variables"].values() for c in cols}
    if profile is not None:
        items += generic_suggestions(profile, df, domain_key, used)
    meta = {"domain": domain_key, "objective": objective, "ranked_by": "rules", "llm": None, "unmatched": False}
    if objective:
        if not any(i["relevance"] > 0 for i in items):
            meta["unmatched"] = True
            meta["message"] = ("None of the available scenarios match that objective closely. They are listed in the usual order; "
                               "you can also define your own in the Custom domain.")
        if ranker is not None:
            try:
                ids = ranker(objective, [{"id": i["id"], "question": i["question"], "reason": i["reason"]} for i in items])
                valid = {i["id"] for i in items}
                order = [x for x in ids if x in valid]            # anything not a real suggestion is dropped
                for pos, sid in enumerate(order):
                    next(i for i in items if i["id"] == sid)["relevance"] += 1.0 - pos / (len(order) + 1)
                meta["ranked_by"], meta["llm"] = "model+rules", "used"
            except Exception:  # noqa: BLE001 - the model is optional; rules already produced a complete answer
                meta["llm"] = "unavailable"
    items.sort(key=lambda i: (-(i["relevance"] if objective else 0), RANK[i["feasibility"]], 0 if i["source"] == "dataset" else 1))
    return {"suggestions": items[:limit], "meta": meta}


def classify_question(text: str, profile: dict | None, domain_key: str | None, df: pd.DataFrame | None = None) -> dict | None:
    """Match a typed question to a template (the dataset's domain first, then the others) and classify it.
    Returns None when no template matches: such a question is unsupported and the caller must say so."""
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    order = ([domain_key] if domain_key else []) + [d.key for d in list_domains() if d.key != domain_key]
    best = None
    for rank, dk in enumerate(order):
        for t in get_domain(dk).templates:
            spans = [m.end() - m.start() for p in t.patterns if (m := re.search(p, low))]
            # the most specific pattern wins (longest match); on a tie the dataset's own domain, being first, wins
            if spans and (best is None or max(spans) > best[0]):
                best = (max(spans), dk, t)
    if best is None:
        return None
    _, dk, t = best
    own = dk == (domain_key or dk)
    a = assess(t, dk, profile if own else None, df if own else None)
    return {"template": t, "domain": dk, "assessment": a, "suggestion": _suggestion(t, dk, a, "user")}
