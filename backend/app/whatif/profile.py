"""DatasetProfile: what a table contains, in a form that does not assume the table is about business.

`build_profile(df)` reuses the column-role reading the Copilot already uses (app/analyst/semantics.py) and adds what a
scenario needs: missing values, duplicate rows, outliers, entities, metrics, dimensions, relationships, time ranges, possible
prediction targets, and a domain guess with its evidence.

Domain detection is explainable and conservative. A domain wins only when several different concepts match columns
(a lone column name is never enough) and it clearly leads the runner-up; otherwise the table is "custom". The user can
override it. Nothing here invents a column: every mapping names a column that exists, and an ambiguous concept is returned
as candidates for the user to choose between, never resolved silently.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.analyst import semantics
from app.whatif.domains import get_domain, list_domains

NUMERIC_ROLES = ("amount", "count", "measurement", "rate")
MAX_RELATION_COLUMNS = 25
MIN_DOMAIN_SCORE = 2.0
MIN_DOMAIN_LEAD = 1.0
MIN_RELATION_ROWS = 10
SAMPLE_VALUES = 8
_GENERIC_SENSITIVE = {"name", "dob", "birth", "address", "phone", "email", "ssn", "aadhaar", "passport"}


def tokens_of(name: str) -> set[str]:
    """Lower-case words of a column name, split on punctuation and camelCase, with simple singular forms added."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    words = [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]
    out = set(words)
    for w in words:
        if len(w) > 3 and w.endswith("s"):
            out.add(w[:-1])
    joined = "_".join(words)
    out.add(joined)
    return out


def _compatible(kind: str, role: str, col: dict | None = None) -> bool:
    if kind == "numeric":
        # a numeric column whose values happen to be all different (marks of a small class) is still a measure unless it is named like an id
        id_named = bool(col and tokens_of(col["name"]) & {"id", "code", "no", "number", "ref", "key"})
        numeric_dtype = bool(col and col["dtype"][:3] in ("int", "flo", "uin"))
        return role in NUMERIC_ROLES or (role == "identifier" and numeric_dtype and not id_named)
    if kind == "datetime":
        return role == "datetime"
    if kind == "category":
        return role in ("category", "flag", "identifier", "person_name")
    if kind == "id":
        return role in ("identifier", "person_name", "category")
    return False


def _sensitive(name: str, extra: frozenset = frozenset()) -> bool:
    return bool(tokens_of(name) & (_GENERIC_SENSITIVE | set(extra)))


def map_concepts(profile: dict, domain_key: str) -> dict:
    """{concept: {"columns": [...], "candidates": [...], "ambiguous": bool}} for one domain.

    `columns` is the mapping the engine uses: all matches for a multi-column concept, the single match otherwise.
    When a single-column concept matches several columns, `columns` is empty and `candidates` lists them: the user picks."""
    dom = get_domain(domain_key)
    result = {}
    for c in dom.concepts:
        want = set(c.tokens) | {t[:-1] for t in c.tokens if len(t) > 3 and t.endswith("s")}
        hits = [col["name"] for col in profile["columns"] if _compatible(c.kind, col["role"], col) and tokens_of(col["name"]) & want
                and not tokens_of(col["name"]) & c.excludes]
        result[c.key] = {"columns": hits if (c.multi or len(hits) == 1) else [], "candidates": hits, "ambiguous": len(hits) > 1 and not c.multi}
    return result


def detect_domain(profile: dict, filename: str | None = None) -> dict:
    """Score every domain on the evidence in the column names (and file name) and say why. Returns
    {"key", "source": "detected"|"fallback", "scores": {domain: score}, "evidence": {domain: [...]}, "explanation"}."""
    scores, evidence = {}, {}
    file_tokens = tokens_of(re.sub(r"\.[A-Za-z0-9]+$", "", filename or "")) if filename else set()
    for dom in list_domains():
        if dom.key == "custom":
            continue
        mapped = map_concepts(profile, dom.key)
        ev, score = [], 0.0
        for c in dom.concepts:
            m = mapped[c.key]
            if m["candidates"]:
                ev.append({"concept": c.label, "columns": m["candidates"], "weight": c.weight})
                score += c.weight
        col_tokens = set().union(*(tokens_of(col["name"]) for col in profile["columns"])) if profile["columns"] else set()
        for kw in sorted(col_tokens & set(dom.keywords)):
            ev.append({"concept": f"word '{kw}' in a column name", "columns": [], "weight": 0.5})
            score += 0.5
        for kw in sorted(file_tokens & (set(dom.keywords) | {dom.key})):
            ev.append({"concept": f"word '{kw}' in the file name", "columns": [], "weight": 0.5})
            score += 0.5
        scores[dom.key], evidence[dom.key] = round(score, 2), ev
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else ("", 0.0))
    if top[1] >= MIN_DOMAIN_SCORE and top[1] - second[1] >= MIN_DOMAIN_LEAD:
        why = "; ".join(f"{e['concept']} ({', '.join(e['columns'])})" if e["columns"] else e["concept"] for e in evidence[top[0]])
        return {"key": top[0], "source": "detected", "scores": scores, "evidence": evidence,
                "explanation": f"Looks like {get_domain(top[0]).title} (score {top[1]:g}) because of: {why}. You can choose a different domain."}
    if top[1] >= MIN_DOMAIN_SCORE:
        reason = f"{get_domain(top[0]).title} and {get_domain(second[0]).title} score too closely ({top[1]:g} vs {second[1]:g}) to choose between them."
    else:
        reason = f"No domain has enough evidence (best is {get_domain(top[0]).title} with {top[1]:g}; at least {MIN_DOMAIN_SCORE:g} is needed)."
    return {"key": "custom", "source": "fallback", "scores": scores, "evidence": evidence,
            "explanation": f"{reason} Treated as Custom, so no sector is assumed. Pick a domain if you know which one applies."}


def _column(df: pd.DataFrame, role: dict, extra_sensitive: frozenset) -> dict:
    name = role["column"]
    s = df[name]
    kind = role["role"]
    sensitive = _sensitive(name, extra_sensitive) or kind == "person_name"
    out = {"name": name, "role": kind, "confidence": role["confidence"], "dtype": str(s.dtype), "missing": int(s.isna().sum()),
           "missing_pct": round(float(s.isna().mean() * 100), 2), "unique": int(s.nunique(dropna=True)), "sensitive": bool(sensitive),
           "evidence": role.get("evidence", [])}
    if kind in NUMERIC_ROLES:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x):
            q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out.update({"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "median": float(x.median()),
                        "std": float(x.std()) if len(x) > 1 else 0.0,
                        "outliers": int(((x < lo) | (x > hi)).sum()) if iqr > 0 else 0})
    elif kind == "datetime":
        d = pd.to_datetime(s, errors="coerce").dropna()
        if len(d):
            out.update({"start": d.min().isoformat(), "end": d.max().isoformat(), "span_days": int((d.max() - d.min()).days)})
    elif kind in ("category", "flag") and not sensitive:
        out["values"] = [str(v) for v in s.dropna().astype(str).value_counts().index[:SAMPLE_VALUES]]
    return out


def _relationships(df: pd.DataFrame, metrics: list[str]) -> list[dict]:
    out = []
    cols = metrics[:MAX_RELATION_COLUMNS]
    numeric = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols})
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = numeric[[a, b]].dropna()
            if len(pair) < MIN_RELATION_ROWS or pair[a].nunique() < 3 or pair[b].nunique() < 3:
                continue
            r = float(np.corrcoef(pair[a], pair[b])[0, 1])
            if np.isfinite(r) and abs(r) >= 0.3:
                out.append({"a": a, "b": b, "pearson": round(r, 3), "n": int(len(pair)),
                            "strength": "strong" if abs(r) >= 0.7 else "moderate"})
    return sorted(out, key=lambda x: -abs(x["pearson"]))[:10]


def build_profile(df: pd.DataFrame, *, filename: str | None = None, domain: str | None = None) -> dict:
    """The DatasetProfile for a table. `domain` overrides detection (the override is recorded, and the detection kept for comparison)."""
    extra = frozenset().union(*(d.sensitive_tokens for d in list_domains()))
    roles = [semantics.column_role(df, c) for c in df.columns]
    columns = [_column(df, r, extra) for r in roles]
    sem = semantics.profile(df) if len(df.columns) else {"outcomes": [], "numeric_outcomes": []}
    by_role = lambda *rs: [c["name"] for c in columns if c["role"] in rs]  # noqa: E731
    metrics = [c for c in by_role(*NUMERIC_ROLES) if not _sensitive(c, extra)]
    entities = []
    for c in columns:
        if c["role"] in ("identifier", "person_name"):
            noun = next((t for t in re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", c["name"]).lower())
                         if t and t not in ("id", "no", "number", "code", "ref", "key")), "record")
            entities.append({"column": c["name"], "noun": noun, "unique": c["unique"]})
    prof = {
        "rows": int(len(df)), "filename": filename, "columns": columns,
        "numeric": by_role(*NUMERIC_ROLES), "categorical": by_role("category"), "datetime": by_role("datetime"), "boolean": by_role("flag"),
        "identifiers": by_role("identifier", "person_name"), "metrics": metrics, "dimensions": by_role("category", "flag"),
        "duplicates": int(df.duplicated().sum()) if len(df.columns) else 0,
        "missing_cells": int(df.isna().sum().sum()), "entities": entities,
        "time_ranges": [{"column": c["name"], "start": c["start"], "end": c["end"], "span_days": c["span_days"]} for c in columns if "start" in c],
        "relationships": _relationships(df, metrics),
        "prediction_targets": list(dict.fromkeys([*sem.get("outcomes", []), *sem.get("numeric_outcomes", [])])),
        "sensitive_columns": [c["name"] for c in columns if c["sensitive"]],
    }
    detected = detect_domain(prof, filename)
    if domain:
        get_domain(domain)          # raises KeyError for an unknown domain: an override must name a real one
        prof["domain"] = {**detected, "detected": detected["key"], "key": domain, "source": "override",
                          "explanation": f"Domain set by you to {get_domain(domain).title} (detection suggested {get_domain(detected['key']).title})."}
    else:
        prof["domain"] = {**detected, "detected": detected["key"]}
    prof["concepts"] = map_concepts(prof, prof["domain"]["key"])
    prof["privacy"] = ("Only column names, counts and summary statistics are returned; no row is echoed. Columns that look like personal "
                       "identifiers are marked sensitive and their values are never listed.")
    return prof
