"""The one entry point for the universal what-if. The API router and the Copilot adapter both call these functions; neither
has calculations of its own, and neither needs a language model.

    profile_table -> suggest -> (build) -> run -> compare

All functions take plain data (a DataFrame, dicts) so they run without a database and are easy to test; the router adds
team scoping, dataset loading and persistence around them.
"""
from __future__ import annotations

import pandas as pd

from app.whatif import discovery, engine, scenario as sc
from app.whatif.domains import get_domain, list_domains
from app.whatif.parse import extract_values
from app.whatif.profile import build_profile


def domains() -> list[dict]:
    return [{"key": d.key, "title": d.title, "description": d.description, "entities": list(d.entities), "metrics": list(d.metrics),
             "notes": list(d.notes), "templates": [{"id": t.id, "title": t.title, "method": t.method, "evidence_kind": t.evidence} for t in d.templates]}
            for d in list_domains()]


def profile_table(df: pd.DataFrame, filename: str | None = None, domain: str | None = None) -> dict:
    return build_profile(df, filename=filename, domain=domain)


def suggest(df: pd.DataFrame | None, domain: str | None, *, filename: str | None = None, objective: str | None = None, ranker=None) -> dict:
    """Suggestions for a dataset (or none) in a domain. With a dataset and no domain given, the detected domain is used."""
    profile = build_profile(df, filename=filename, domain=domain) if df is not None else None
    key = domain or (profile["domain"]["key"] if profile else "custom")
    out = discovery.suggest(profile, key, df=df, objective=objective, ranker=ranker)
    out["domain"] = {"key": key, "title": get_domain(key).title, **({"detection": {k: profile["domain"][k] for k in ("source", "explanation", "scores", "detected")}} if profile else {})}
    return out


def build(domain: str, template_id: str, **kw) -> dict:
    return sc.new_scenario(domain, template_id, **kw)


def run(scenario: dict, df: pd.DataFrame | None = None, profile: dict | None = None) -> dict:
    return engine.run(scenario, df, profile)


def validate(scenario: dict, df: pd.DataFrame | None = None) -> dict:
    return sc.validate(scenario, df)


def compare(items: list[dict]) -> dict:
    return engine.compare(items)


def from_question(text: str, df: pd.DataFrame | None, *, filename: str | None = None, domain: str | None = None, inputs: dict | None = None) -> dict:
    """A structured scenario request from a typed question (the Copilot's route in). Returns
    {"status": "unsupported" | "needs_input" | "ready", "scenario", "suggestion", "message"}. Numbers come only from the sentence and `inputs`."""
    profile = build_profile(df, filename=filename, domain=domain) if df is not None else None
    key = domain or (profile["domain"]["key"] if profile else None)
    hit = discovery.classify_question(text, profile, key, df)
    if hit is None:
        return {"status": "unsupported", "scenario": None, "suggestion": None,
                "message": "That question does not match a scenario Silt can calculate reliably. Try one of the suggested questions, or define your own model in the Custom domain."}
    t, dk = hit["template"], hit["domain"]
    values = {**extract_values(t, text), **{k: v for k, v in (inputs or {}).items()}}
    s = sc.new_scenario(dk, t.id, inputs=values, question=text, dataset={"filename": filename} if filename else None)
    return {"status": "ready", "scenario": s, "suggestion": hit["suggestion"], "message": ""}
