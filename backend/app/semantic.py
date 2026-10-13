"""Business semantic layer: approved metric definitions, the words people use
for them, and how they relate to datasets and columns.

Stored relationally (metrics, versions, terms, relationships). The "knowledge
graph" is those four tables read as nodes and edges; nothing here needs a graph
database.

Trust model. A definition takes effect only when a team owner or admin approves
a specific version through the API. Nothing read from an uploaded file, a
dataset cell, a knowledge note or a model's output can create, change or
approve a definition. When definitions are handed to the Planner they are
labelled as the team's approved definitions, separately from dataset content,
which is always labelled as untrusted data.

Ambiguity is surfaced, not resolved by guessing: if one phrase in a question
maps to two approved metrics, `resolve` reports it and the caller asks the user.
"""
from __future__ import annotations

import datetime as dt
import re

from sqlalchemy.orm import Session

from app import metric_engine
from app.models import (
    Dataset,
    KnowledgeNote,
    SemanticMetric,
    SemanticMetricVersion,
    SemanticRelationship,
    SemanticTerm,
)

TERM_KINDS = ("alias", "dimension", "entity")
RELATIONS = ("derived_from", "measured_by", "grouped_by", "joins", "synonym_of", "part_of")
NODE_TYPES = ("metric", "term", "dataset", "column")
MAX_NAME = 80


class SemanticError(ValueError):
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    if not slug:
        raise SemanticError("A name needs at least one letter or digit")
    return slug[:MAX_NAME]


def norm_term(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", (text or "").strip().lower()))


def _dataset_profile(db: Session, team_id: int, dataset_id: int | None) -> dict | None:
    if dataset_id is None:
        return None
    ds = db.get(Dataset, dataset_id)
    if ds is None or ds.team_id != team_id:
        raise SemanticError("Dataset not found")
    return ds.profile_json or {}


def validate_definition(db: Session, team_id: int, formula: str, dataset_id: int | None,
                        filters: list | None) -> tuple[metric_engine.Formula, list[dict]]:
    try:
        parsed = metric_engine.parse_formula(formula)
        profile = _dataset_profile(db, team_id, dataset_id)
        columns = [c["name"] for c in profile.get("columns", [])] if profile is not None else None
        clean_filters = metric_engine.normalize_filters(filters, columns)
    except metric_engine.FormulaError as e:
        raise SemanticError(str(e)) from e
    if profile is not None:
        problems = metric_engine.validate_against_profile(parsed, profile)
        if problems:
            raise SemanticError("; ".join(problems))
    return parsed, clean_filters


# ------------------------------------------------------------------ metrics --

def get_metric(db: Session, team_id: int, metric_id: int) -> SemanticMetric:
    m = db.get(SemanticMetric, metric_id)
    if m is None or m.team_id != team_id:
        raise SemanticError("Metric not found")
    return m


def versions_of(db: Session, metric: SemanticMetric) -> list[SemanticMetricVersion]:
    return (db.query(SemanticMetricVersion).filter_by(metric_id=metric.id)
            .order_by(SemanticMetricVersion.version).all())


def active_version(db: Session, metric: SemanticMetric) -> SemanticMetricVersion | None:
    return (db.query(SemanticMetricVersion)
            .filter_by(metric_id=metric.id, version=metric.current_version).first())


def create_metric(db: Session, *, team_id: int, user_id: int | None, name: str, formula: str,
                  description: str = "", unit: str = "", dataset_id: int | None = None,
                  filters: list | None = None, aliases: list[str] | None = None) -> SemanticMetric:
    name = (name or "").strip()
    if not name or len(name) > MAX_NAME:
        raise SemanticError(f"Name is required and at most {MAX_NAME} characters")
    slug = slugify(name)
    if db.query(SemanticMetric.id).filter_by(team_id=team_id, slug=slug).first():
        raise SemanticError(f"A metric named '{name}' already exists; add a new version to it instead")
    _, clean_filters = validate_definition(db, team_id, formula, dataset_id, filters)
    metric = SemanticMetric(team_id=team_id, slug=slug, name=name, description=description[:1000],
                            unit=unit[:30], status="draft", owner_id=user_id, current_version=1)
    db.add(metric)
    db.flush()
    db.add(SemanticMetricVersion(metric_id=metric.id, version=1, formula=formula.strip(), dataset_id=dataset_id,
                                 filters_json=clean_filters, source="user", created_by=user_id))
    for alias in aliases or []:
        _add_term(db, team_id, user_id, "alias", alias, metric_id=metric.id)
    db.commit()
    db.refresh(metric)
    return metric


def add_version(db: Session, metric: SemanticMetric, *, user_id: int | None, formula: str,
                dataset_id: int | None = None, filters: list | None = None, note: str = "") -> SemanticMetricVersion:
    """Propose a new definition. It does not take effect until approved; the
    current approved version keeps answering questions in the meantime."""
    _, clean_filters = validate_definition(db, metric.team_id, formula, dataset_id, filters)
    latest = max((v.version for v in versions_of(db, metric)), default=0)
    version = SemanticMetricVersion(metric_id=metric.id, version=latest + 1, formula=formula.strip(),
                                    dataset_id=dataset_id, filters_json=clean_filters, source="user",
                                    note=note[:500], created_by=user_id)
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def approve(db: Session, metric: SemanticMetric, version_number: int, user_id: int) -> SemanticMetric:
    version = db.query(SemanticMetricVersion).filter_by(metric_id=metric.id, version=version_number).first()
    if version is None:
        raise SemanticError("Version not found")
    # Re-validate at approval time: the dataset may have changed since the draft.
    validate_definition(db, metric.team_id, version.formula, version.dataset_id, version.filters_json)
    version.approved_by, version.approved_at = user_id, dt.datetime.utcnow()
    metric.status, metric.current_version = "approved", version.version
    db.commit()
    db.refresh(metric)
    return metric


def deprecate(db: Session, metric: SemanticMetric) -> SemanticMetric:
    metric.status = "deprecated"
    db.commit()
    db.refresh(metric)
    return metric


# -------------------------------------------------------------------- terms --

def _add_term(db: Session, team_id: int, user_id: int | None, kind: str, term: str, *,
              metric_id: int | None = None, dataset_id: int | None = None, column_name: str | None = None,
              unit: str = "", description: str = "") -> SemanticTerm:
    if kind not in TERM_KINDS:
        raise SemanticError(f"kind must be one of: {', '.join(TERM_KINDS)}")
    normalized = norm_term(term)
    if not normalized or len(normalized) > MAX_NAME:
        raise SemanticError(f"A term is required and at most {MAX_NAME} characters")
    if kind == "alias":
        if metric_id is None:
            raise SemanticError("An alias must point at a metric")
        get_metric(db, team_id, metric_id)
    else:
        profile = _dataset_profile(db, team_id, dataset_id)
        if profile is None or not column_name:
            raise SemanticError(f"A {kind} must map to a dataset column")
        if column_name not in [c["name"] for c in profile.get("columns", [])]:
            raise SemanticError(f"Column '{column_name}' does not exist in that dataset")
    duplicate = db.query(SemanticTerm).filter_by(team_id=team_id, kind=kind, term=normalized, metric_id=metric_id,
                                                 dataset_id=dataset_id, column_name=column_name).first()
    if duplicate is not None:
        return duplicate
    row = SemanticTerm(team_id=team_id, kind=kind, term=normalized, metric_id=metric_id, dataset_id=dataset_id,
                       column_name=column_name, unit=unit[:30], description=description[:500], created_by=user_id)
    db.add(row)
    db.flush()
    return row


def add_term(db: Session, **kwargs) -> tuple[SemanticTerm, list[dict]]:
    """Add a term. Returns it with any conflicts: other metrics already known
    by the same words. A conflict is allowed to exist -- it is reported here and
    turns into a clarifying question whenever someone uses the phrase."""
    row = _add_term(db, **kwargs)
    db.commit()
    db.refresh(row)
    return row, conflicts_for(db, row.team_id, row.term, exclude_metric_id=row.metric_id)


def metric_terms(db: Session, metric: SemanticMetric) -> list[str]:
    """Every phrase that refers to this metric: its name plus its aliases."""
    aliases = [t.term for t in db.query(SemanticTerm).filter_by(team_id=metric.team_id, kind="alias",
                                                                metric_id=metric.id).all()]
    return sorted({norm_term(metric.name), *aliases})


def conflicts_for(db: Session, team_id: int, term: str, exclude_metric_id: int | None = None) -> list[dict]:
    normalized = norm_term(term)
    out = []
    for m in db.query(SemanticMetric).filter(SemanticMetric.team_id == team_id,
                                             SemanticMetric.status != "deprecated").all():
        if m.id != exclude_metric_id and normalized in metric_terms(db, m):
            out.append({"metric_id": m.id, "name": m.name, "status": m.status})
    return out


# --------------------------------------------------------------- resolution --

def _phrase_in(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _applies(db: Session, metric: SemanticMetric, version: SemanticMetricVersion, dataset_id: int,
             profile: dict) -> bool:
    if version.dataset_id is not None and version.dataset_id != dataset_id:
        return False
    try:
        return not metric_engine.validate_against_profile(metric_engine.parse_formula(version.formula), profile)
    except metric_engine.FormulaError:
        return False


def metric_out(db: Session, metric: SemanticMetric, version: SemanticMetricVersion | None = None) -> dict:
    version = version or active_version(db, metric)
    return {
        "id": metric.id, "slug": metric.slug, "name": metric.name, "description": metric.description or "",
        "unit": metric.unit or "", "status": metric.status, "owner_id": metric.owner_id,
        "current_version": metric.current_version,
        "formula": version.formula if version else "", "filters": list(version.filters_json or []) if version else [],
        "dataset_id": version.dataset_id if version else None,
        "aliases": [t for t in metric_terms(db, metric) if t != norm_term(metric.name)],
        "created_at": metric.created_at, "updated_at": metric.updated_at,
    }


def approved_metrics(db: Session, team_id: int, dataset_id: int) -> list[dict]:
    """Approved definitions that can actually be computed on this dataset."""
    ds = db.get(Dataset, dataset_id)
    if ds is None or ds.team_id != team_id:
        return []
    out = []
    for m in db.query(SemanticMetric).filter_by(team_id=team_id, status="approved").order_by(SemanticMetric.name).all():
        v = active_version(db, m)
        if v is not None and v.approved_at is not None and _applies(db, m, v, dataset_id, ds.profile_json or {}):
            out.append({**metric_out(db, m, v), "terms": metric_terms(db, m)})
    return out


def resolve(db: Session, team_id: int, dataset_id: int, text: str) -> dict:
    """Map the words of a question to approved definitions.

    Returns {"metrics": [...], "ambiguous": [{"term", "candidates"}], "dimensions": [...]}.
    The longest matching phrase wins, so "net revenue" is not also reported as
    "revenue". A phrase claimed by more than one approved metric is ambiguous."""
    normalized = norm_term(text)
    by_term: dict[str, list[dict]] = {}
    for m in approved_metrics(db, team_id, dataset_id):
        for term in m["terms"]:
            if _phrase_in(term, normalized):
                by_term.setdefault(term, []).append(m)
    kept: list[str] = []
    for term in sorted(by_term, key=len, reverse=True):
        if not any(_phrase_in(term, longer) for longer in kept):
            kept.append(term)
    metrics, ambiguous, seen = [], [], set()
    for term in kept:
        unique = {m["id"]: m for m in by_term[term]}
        if len(unique) > 1:
            ambiguous.append({"term": term, "candidates": [
                {"metric_id": m["id"], "name": m["name"], "formula": m["formula"], "description": m["description"]}
                for m in unique.values()]})
        else:
            m = next(iter(unique.values()))
            if m["id"] not in seen:
                seen.add(m["id"])
                metrics.append({**m, "matched_term": term})
    dimensions = []
    for t in db.query(SemanticTerm).filter(SemanticTerm.team_id == team_id, SemanticTerm.dataset_id == dataset_id,
                                           SemanticTerm.kind.in_(("dimension", "entity"))).all():
        if _phrase_in(t.term, normalized):
            dimensions.append({"term": t.term, "kind": t.kind, "column": t.column_name})
    return {"metrics": metrics, "ambiguous": ambiguous, "dimensions": dimensions}


def format_for_prompt(metrics: list[dict]) -> str:
    """Approved definitions for the Planner. These come from the team's
    semantic layer (approved through the API), never from dataset content."""
    if not metrics:
        return ""
    lines = []
    for m in metrics:
        line = f"- {m['name']}: {m['formula']}"
        if m.get("filters"):
            line += " with filters " + "; ".join(f"{f['column']} {f['op']} {f['value']}" for f in m["filters"])
        if m.get("unit"):
            line += f" (unit: {m['unit']})"
        lines.append(line)
    return ("\nApproved metric definitions for this team (use these exact definitions whenever the question "
            "refers to one of these metrics; do not substitute your own):\n" + "\n".join(lines) + "\n")


# -------------------------------------------------------------------- graph --

def add_relationship(db: Session, *, team_id: int, user_id: int | None, src_type: str, src_ref: str,
                     relation: str, dst_type: str, dst_ref: str) -> SemanticRelationship:
    if src_type not in NODE_TYPES or dst_type not in NODE_TYPES:
        raise SemanticError(f"Node types must be one of: {', '.join(NODE_TYPES)}")
    if relation not in RELATIONS:
        raise SemanticError(f"relation must be one of: {', '.join(RELATIONS)}")
    for node_type, ref in ((src_type, src_ref), (dst_type, dst_ref)):
        _check_node(db, team_id, node_type, str(ref))
    row = SemanticRelationship(team_id=team_id, src_type=src_type, src_ref=str(src_ref)[:200], relation=relation,
                               dst_type=dst_type, dst_ref=str(dst_ref)[:200], created_by=user_id)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _check_node(db: Session, team_id: int, node_type: str, ref: str) -> None:
    """A relationship may only connect things this team owns."""
    try:
        if node_type == "metric":
            get_metric(db, team_id, int(ref))
        elif node_type == "term":
            t = db.get(SemanticTerm, int(ref))
            if t is None or t.team_id != team_id:
                raise SemanticError("Term not found")
        elif node_type == "dataset":
            _dataset_profile(db, team_id, int(ref))
        else:  # column: "<dataset id>:<column name>"
            ds_id, _, column = ref.partition(":")
            profile = _dataset_profile(db, team_id, int(ds_id))
            if column not in [c["name"] for c in (profile or {}).get("columns", [])]:
                raise SemanticError(f"Column '{column}' does not exist in that dataset")
    except ValueError as e:
        if isinstance(e, SemanticError):
            raise
        raise SemanticError(f"'{ref}' is not a valid {node_type} reference") from e


def graph(db: Session, team_id: int) -> dict:
    """Nodes and edges for the team: explicit relationships, plus the edges the
    definitions imply (metric -> the columns its formula reads, alias -> metric,
    dimension -> column)."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(node_type: str, ref: str, label: str) -> str:
        key = f"{node_type}:{ref}"
        nodes.setdefault(key, {"id": key, "type": node_type, "label": label})
        return key

    datasets = {d.id: d for d in db.query(Dataset).filter_by(team_id=team_id).all()}
    for m in db.query(SemanticMetric).filter_by(team_id=team_id).all():
        mk = node("metric", str(m.id), m.name)
        v = active_version(db, m)
        if v is None:
            continue
        try:
            columns = metric_engine.parse_formula(v.formula).columns
        except metric_engine.FormulaError:
            columns = []
        for col in columns:
            ref = f"{v.dataset_id}:{col}" if v.dataset_id else f"*:{col}"
            edges.append({"src": mk, "relation": "derived_from", "dst": node("column", ref, col), "implied": True})
        if v.dataset_id in datasets:
            edges.append({"src": mk, "relation": "measured_by",
                          "dst": node("dataset", str(v.dataset_id), datasets[v.dataset_id].filename), "implied": True})
    for t in db.query(SemanticTerm).filter_by(team_id=team_id).all():
        tk = node("term", str(t.id), t.term)
        if t.kind == "alias" and t.metric_id:
            edges.append({"src": tk, "relation": "synonym_of", "dst": f"metric:{t.metric_id}", "implied": True})
        elif t.column_name:
            edges.append({"src": tk, "relation": "grouped_by" if t.kind == "dimension" else "measured_by",
                          "dst": node("column", f"{t.dataset_id}:{t.column_name}", t.column_name), "implied": True})
    for r in db.query(SemanticRelationship).filter_by(team_id=team_id).all():
        src = node(r.src_type, r.src_ref, nodes.get(f"{r.src_type}:{r.src_ref}", {}).get("label", r.src_ref))
        dst = node(r.dst_type, r.dst_ref, nodes.get(f"{r.dst_type}:{r.dst_ref}", {}).get("label", r.dst_ref))
        edges.append({"id": r.id, "src": src, "relation": r.relation, "dst": dst, "implied": False})
    edges = [e for e in edges if e["src"] in nodes and e["dst"] in nodes]
    return {"nodes": list(nodes.values()), "edges": edges}


# ------------------------------------------------------------------- search --

def search(db: Session, team_id: int, query: str, limit: int = 25) -> list[dict]:
    """Definitions, terms and knowledge notes matching the words of `query`,
    all scoped to the team."""
    words = [w for w in norm_term(query).split(" ") if len(w) > 1]
    if not words:
        return []

    def score(text: str) -> int:
        hay = norm_term(text)
        return sum(1 for w in words if w in hay)

    hits: list[tuple[int, dict]] = []
    for m in db.query(SemanticMetric).filter_by(team_id=team_id).all():
        v = active_version(db, m)
        s = score(f"{m.name} {m.description} {' '.join(metric_terms(db, m))} {v.formula if v else ''}")
        if s:
            hits.append((s + 1, {"type": "metric", "id": m.id, "title": m.name, "status": m.status,
                                 "detail": v.formula if v else ""}))
    for t in db.query(SemanticTerm).filter(SemanticTerm.team_id == team_id,
                                           SemanticTerm.kind != "alias").all():
        s = score(f"{t.term} {t.description} {t.column_name or ''}")
        if s:
            hits.append((s, {"type": t.kind, "id": t.id, "title": t.term, "status": "",
                             "detail": f"column {t.column_name}"}))
    for n in db.query(KnowledgeNote).filter_by(team_id=team_id).order_by(KnowledgeNote.id.desc()).limit(500).all():
        s = score(n.text)
        if s:
            hits.append((s, {"type": f"note:{n.kind}", "id": n.id, "title": n.text[:80], "status": n.source,
                             "detail": n.text[:300]}))
    hits.sort(key=lambda h: -h[0])
    return [h for _, h in hits[:limit]]
