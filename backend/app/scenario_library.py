"""Saved what-if scenarios as work people come back to: freshness, comparison and export (Phase 4).

Nothing here calculates a scenario. It reads what was stored (inputs, results, provenance) and says whether that is still current,
how several saved scenarios differ, and produces a file from it. A scenario is never silently re-based onto newer data: when the
data has moved on the scenario says so and keeps the numbers it was saved with.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any

from sqlalchemy.orm import Session

from app import scenario_model_store as store
from app import scenarios
from app.dataset_versions import latest_version
from app.models import Dataset, DatasetVersion, Scenario, ScenarioEvidence

LIBRARY_VERSION = "scenario-library.1"
FORMATS = ("json", "csv", "md")


# ================================================================== freshness ==

def freshness(db: Session, team_id: int, row: Scenario) -> dict:
    """Is what was saved still what the data and the model say? `status` is one of
    current / typed_inputs (no dataset was used) / stale (newer data exists) / changed (the pinned data's recorded fingerprint no longer
    matches what was saved) / unavailable (the dataset or version is gone). `reasons` lists everything found, in plain words."""
    reasons: list[str] = []
    status = "typed_inputs"
    src = ((row.adjustments_json or {}).get("baseline_source") or {})
    prov = src.get("provenance") or {}
    if row.dataset_id is not None:
        status = "current"
        dataset = db.get(Dataset, row.dataset_id)
        if dataset is None or dataset.team_id != team_id:
            status = "unavailable"
            reasons.append("The dataset this scenario was built from is no longer available. The saved numbers are unchanged.")
        else:
            pinned = db.get(DatasetVersion, row.dataset_version_id) if row.dataset_version_id else None
            latest = latest_version(db, dataset)
            if row.dataset_version_id and pinned is None:
                status = "unavailable"
                reasons.append("The dataset version this scenario was built from is no longer available.")
            else:
                if pinned is not None and prov.get("content_sha256") and pinned.content_sha256 and pinned.content_sha256 != prov["content_sha256"]:
                    status = "changed"
                    reasons.append("The data recorded for that version no longer matches what this scenario was saved with.")
                if pinned is not None and latest is not None and latest.version_number > pinned.version_number and status == "current":
                    status = "stale"
                if pinned is not None and latest is not None and latest.version_number > pinned.version_number:
                    reasons.append(f"This scenario uses version {pinned.version_number} of the dataset; version {latest.version_number} exists now. "
                                   "Its baseline was not changed. Re-take the baseline from the new version to compare.")
    note = model_note(db, team_id, row.model)
    if note:
        reasons.append(note)
    return {"status": status, "reasons": reasons, "pinned_version": src.get("dataset_version") or prov.get("dataset_version"),
            "latest_version": prov.get("latest_version") if status == "typed_inputs" else _latest_number(db, row)}


def _latest_number(db: Session, row: Scenario) -> int | None:
    dataset = db.get(Dataset, row.dataset_id) if row.dataset_id else None
    latest = latest_version(db, dataset) if dataset else None
    return latest.version_number if latest else None


def model_note(db: Session, team_id: int, ref: str) -> str | None:
    """A custom model that has been revised or archived since the scenario was saved. The scenario keeps the version it was built on."""
    if not store.is_custom(ref):
        return None
    try:
        model_id, number = store.parse_ref(ref)
        model = store.owned(db, team_id, model_id)
    except scenarios.ScenarioError:
        return "The custom model this scenario uses is no longer available. The saved numbers are unchanged."
    if number is not None and model.current_version > number:
        return f"This scenario uses version {number} of '{model.name}'; version {model.current_version} exists now. The saved numbers use version {number}."
    if model.status != "active":
        return f"The custom model '{model.name}' has been archived. The saved numbers are unchanged."
    return None


# ================================================================== comparison ==

def _primary(res: dict) -> tuple[str | None, dict]:
    return res.get("primary_output"), res.get("output_labels") or {}


def compare(items: list[dict]) -> dict:
    """What differs between saved scenarios of one model, from their stored results only.

    `items` are dicts with id, name, results (stored), freshness and the dataset version used. Every row is a scenario's own change
    against its own baseline. The warnings say when the rows are not on the same footing (different starting inputs or assumptions or data
    versions), because comparing a gain on a different base is easy to misread. 'Highest' and 'lowest' are about the number only."""
    rows, base_inputs, assumption_sets, versions = [], [], [], set()
    for it in items:
        res = it["results"] or {}
        primary, labels = _primary(res)
        base = (res.get("baseline") or {}).get("outputs") or {}
        base_inputs.append((res.get("baseline") or {}).get("inputs") or {})
        assumption_sets.append(res.get("assumptions") or {})
        if it.get("dataset_version") is not None:
            versions.add((it.get("dataset_id"), it["dataset_version"]))
        rows.append({"id": it["id"], "name": it["name"], "scenario": "baseline", "primary_output": primary, "value": base.get(primary),
                     "change": None, "change_pct": None, "outputs": base, "freshness": it["freshness"]["status"]})
        for name, s in (res.get("scenarios") or {}).items():
            out = s.get("outputs") or {}
            v, b = out.get(primary), base.get(primary)
            change = None if v is None or b is None else v - b
            rows.append({"id": it["id"], "name": it["name"], "scenario": name, "primary_output": primary, "value": v, "change": change,
                         "change_pct": None if change is None or not b else change / abs(b) * 100, "outputs": out,
                         "freshness": it["freshness"]["status"]})
    warnings = []
    if any(b != base_inputs[0] for b in base_inputs[1:]):
        warnings.append("These scenarios start from different baseline inputs, so their levels are not directly comparable. Compare the changes, not the levels.")
    if any(a != assumption_sets[0] for a in assumption_sets[1:]):
        warnings.append("These scenarios use different assumptions, such as price elasticity, so the same change can have a different effect.")
    if len(versions) > 1:
        warnings.append("These scenarios were built from different versions of the data.")
    for it in items:
        if it["freshness"]["status"] in ("stale", "changed", "unavailable"):
            warnings.append(f"'{it['name']}': {it['freshness']['reasons'][0]}" if it["freshness"]["reasons"] else f"'{it['name']}' is {it['freshness']['status']}.")
    scen = [r for r in rows if r["scenario"] != "baseline" and r["value"] is not None]
    ranked = sorted(scen, key=lambda r: r["value"], reverse=True)
    return {"rows": rows, "warnings": warnings, "highest": _label(ranked[0]) if ranked else None, "lowest": _label(ranked[-1]) if len(ranked) > 1 else None,
            "note": "Highest and lowest describe the number only; whether higher is better depends on the output."}


def comparison_csv(analysis: dict) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["saved_as", "scenario", "primary_output", "value", "change", "change_pct", "data_status"])
    for a in analysis["rows"]:
        w.writerow([_safe(a["name"]), _safe(a["scenario"]), a["primary_output"], a["value"], a["change"], a["change_pct"], a["freshness"]])
    w.writerow(["# " + scenarios.DISCLAIMER])
    for x in analysis["warnings"]:
        w.writerow(["# " + x])
    return out.getvalue()


def _label(r: dict) -> dict:
    return {"id": r["id"], "name": r["name"], "scenario": r["scenario"], "value": r["value"]}


# ===================================================================== export ==

def _safe(cell: Any) -> Any:
    """A spreadsheet runs a cell that begins with = + - @ as a formula; a scenario name is user text, so it is defused."""
    if isinstance(cell, str) and cell[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + cell
    return cell


def _evidence(db: Session, team_id: int, scenario_id: int) -> list[dict]:
    rows = (db.query(ScenarioEvidence).filter(ScenarioEvidence.team_id == team_id, ScenarioEvidence.scenario_id == scenario_id)
            .order_by(ScenarioEvidence.id).all())
    return [{"id": r.id, "kind": r.kind, "verification_status": r.verification_status, "content_sha256": r.content_sha256} for r in rows]


def _content_sha(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def export_document(db: Session, team_id: int, row: Scenario, fresh: dict) -> dict:
    """Everything needed to understand a saved scenario outside the app: what was asked, what came out, where the data came from,
    what the checks said and which sealed records back it. It carries a hash of its own content."""
    res = row.results_json or {}
    adj = row.adjustments_json or {}
    doc = {"export_version": LIBRARY_VERSION, "scenario": {"id": row.id, "name": row.name, "model": row.model, "created_at": row.created_at},
           "disclaimer": scenarios.DISCLAIMER, "data_kind": "scenario",
           "freshness": fresh, "baseline_inputs": row.baseline_json or {}, "assumptions": res.get("assumptions") or {},
           "assumption_sources": adj.get("assumption_sources") or {},
           "changes": {k: v for k, v in (adj.get("scenarios") or {}).items()},
           "baseline_source": adj.get("baseline_source"),
           "labels": {"inputs": res.get("input_labels") or {}, "outputs": res.get("output_labels") or {}},
           "baseline_outputs": (res.get("baseline") or {}).get("outputs") or {},
           "scenario_outputs": {k: v.get("outputs") for k, v in (res.get("scenarios") or {}).items()},
           "verification": res.get("verification"), "explanation": (res.get("explanation") or {}).get("summary"),
           "evidence": _evidence(db, team_id, row.id)}
    if res.get("simulation"):
        sim = res["simulation"]
        doc["simulation"] = {"config": sim.get("config"), "summary": sim.get("summary") or sim.get("outputs")}
    doc["content_sha256"] = _content_sha(doc)
    return doc


def to_csv(doc: dict) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    labels = doc["labels"]["outputs"]
    w.writerow(["scenario", "output", "label", "baseline", "value", "change"])
    base = doc["baseline_outputs"]
    for name, outs in doc["scenario_outputs"].items():
        for k, v in (outs or {}).items():
            b = base.get(k)
            w.writerow([_safe(name), k, _safe(labels.get(k, k)), b, v, None if v is None or b is None else v - b])
    w.writerow([])
    w.writerow(["# " + doc["disclaimer"]])
    w.writerow(["# content_sha256", doc["content_sha256"]])
    return out.getvalue()


def to_markdown(doc: dict) -> str:
    s = doc["scenario"]
    lab = doc["labels"]
    lines = [f"# {_md(s['name'])}", "", f"*{doc['disclaimer']}*", "", f"Model: `{s['model']}` · saved {s['created_at']} · kind: {doc['data_kind']}", ""]
    f = doc["freshness"]
    lines += [f"**Data status:** {f['status']}"] + [f"- {r}" for r in f["reasons"]] + [""]
    lines += ["## Starting inputs", ""] + [f"- {_md(doc['labels']['inputs'].get(k, k))}: {v}" for k, v in doc["baseline_inputs"].items()] + [""]
    if doc["assumptions"]:
        lines += ["## Assumptions", ""] + [f"- {_md(k)}: {v}" for k, v in doc["assumptions"].items()] + [""]
    lines += ["## Results", "", "| Scenario | Output | Baseline | Value | Change |", "|---|---|---|---|---|"]
    for name, outs in doc["scenario_outputs"].items():
        for k, v in (outs or {}).items():
            b = doc["baseline_outputs"].get(k)
            lines.append(f"| {_md(name)} | {_md(lab['outputs'].get(k, k))} | {_n(b)} | {_n(v)} | {_n(None if v is None or b is None else v - b)} |")
    if doc.get("explanation"):
        lines += ["", "## In plain English", "", _md(doc["explanation"])]
    v = doc.get("verification") or {}
    lines += ["", f"**Independent check:** {v.get('status', 'not run')}", ""]
    if doc["evidence"]:
        lines += ["## Evidence records", ""] + [f"- #{e['id']} {e['kind']} · {e['verification_status']} · `{e['content_sha256'][:16]}`" for e in doc["evidence"]] + [""]
    lines.append(f"Content fingerprint: `{doc['content_sha256']}`")
    return "\n".join(lines) + "\n"


def _md(x: Any) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _n(x: Any) -> str:
    return "—" if x is None else f"{x:,.2f}"


def render(doc: dict, fmt: str) -> tuple[str, str, str]:
    """(body, media type, file extension)."""
    if fmt == "json":
        return json.dumps(doc, indent=2, default=str), "application/json", "json"
    if fmt == "csv":
        return to_csv(doc), "text/csv", "csv"
    if fmt == "md":
        return to_markdown(doc), "text/markdown", "md"
    raise scenarios.ScenarioError(f"Choose one of: {', '.join(FORMATS)}")
