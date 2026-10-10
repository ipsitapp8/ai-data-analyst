"""Data observability: per-version quality snapshots, baselines, drift metrics
and incidents.

Four different things can go wrong with data, and they are kept apart because
they call for different responses:

    schema        a column disappeared, appeared or changed type
    quality       rows went missing, values went missing, rows were duplicated,
                  values left their usual range, the data is stale
    distribution  the values are all valid but are distributed differently
    (business KPI anomalies are a fourth thing: they live in the scheduler's
    alerts, not here -- a KPI can move because the business moved)

Avoiding false alarms:
- no distribution comparison is made unless both samples have at least
  `min_sample` values; the comparison is listed as skipped instead;
- numeric drift needs two independent signals to agree: a PSI above threshold
  AND a Kolmogorov-Smirnov test that rejects "same distribution" at 1%;
- the baseline is explicit and configurable. Comparing against the previous
  upload is the default; for seasonal data, pin the baseline to a version from
  the comparable season.

Every threshold is visible through the API and can be changed per dataset.
"""
from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app import config
from app.models import Dataset, DatasetVersion, DQConfig, DQIncident, DQSnapshot

logger = logging.getLogger(__name__)

SAMPLE_POINTS = 400
MAX_PROFILED_COLUMNS = 60
TOP_CATEGORIES = 30
EPS = 1e-4
SEVERITY_ORDER = {"high": 0, "warn": 1, "info": 2}


def default_thresholds() -> dict:
    return {
        "min_sample": config.DQ_MIN_SAMPLE,
        "row_count_change_pct_warn": 30.0, "row_count_change_pct_high": 60.0,
        "missing_increase_pts_warn": 10.0, "missing_increase_pts_high": 25.0,
        "duplicate_pct_warn": 1.0, "duplicate_pct_high": 10.0,
        "range_sigmas": 4.0,
        "psi_warn": config.DQ_PSI_WARN, "psi_high": config.DQ_PSI_HIGH,
        "ks_alpha": 0.01,
        "freshness_max_age_days": None,  # off unless set
    }


def merged_thresholds(overrides: dict | None) -> dict:
    out = default_thresholds()
    for key, value in (overrides or {}).items():
        if key in out and (value is None or isinstance(value, (int, float))):
            out[key] = value
    return out


def validate_thresholds(overrides: dict) -> dict:
    allowed = default_thresholds()
    clean = {}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"Unknown threshold '{key}'")
        if value is None:
            if key != "freshness_max_age_days":
                raise ValueError(f"'{key}' cannot be empty")
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"'{key}' must be a non-negative number")
        clean[key] = value
    merged = merged_thresholds(clean)
    for low, high in (("row_count_change_pct_warn", "row_count_change_pct_high"),
                      ("missing_increase_pts_warn", "missing_increase_pts_high"),
                      ("duplicate_pct_warn", "duplicate_pct_high"), ("psi_warn", "psi_high")):
        if merged[low] > merged[high]:
            raise ValueError(f"'{low}' must not exceed '{high}'")
    if not 0 < merged["ks_alpha"] < 1:
        raise ValueError("'ks_alpha' must be between 0 and 1")
    return clean


# ----------------------------------------------------------------- snapshot --

def _quantile_sample(values: np.ndarray) -> list[float]:
    """Up to SAMPLE_POINTS evenly spaced order statistics: enough to compare
    two distributions without storing the column."""
    values = np.sort(values)
    if len(values) <= SAMPLE_POINTS:
        return [float(v) for v in values]
    idx = np.linspace(0, len(values) - 1, SAMPLE_POINTS).round().astype(int)
    return [float(v) for v in values[idx]]


def build_snapshot(df: pd.DataFrame) -> dict:
    rows = int(len(df))
    duplicates = int(df.duplicated().sum()) if rows else 0
    columns: dict[str, dict] = {}
    for name in list(df.columns)[:MAX_PROFILED_COLUMNS]:
        s = df[name]
        present = s.dropna()
        entry: dict = {"dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 4) if rows else 0.0,
                       "n": int(len(present))}
        if pd.api.types.is_bool_dtype(s):
            entry["kind"] = "categorical"
            counts = present.astype(str).value_counts()
            entry["categories"] = {str(k): int(v) for k, v in counts.items()}
            entry["other"] = 0
        elif pd.api.types.is_numeric_dtype(s):
            entry["kind"] = "numeric"
            values = present.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                entry.update({"min": float(values.min()), "max": float(values.max()),
                              "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                              "sample": _quantile_sample(values)})
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry["kind"] = "datetime"
            if len(present):
                entry.update({"min": present.min().isoformat(), "max": present.max().isoformat()})
        else:
            entry["kind"] = "categorical"
            counts = present.astype(str).value_counts()
            top = counts.head(TOP_CATEGORIES)
            entry["categories"] = {str(k)[:120]: int(v) for k, v in top.items()}
            entry["other"] = int(counts.iloc[TOP_CATEGORIES:].sum())
            entry["unique"] = int(len(counts))
        columns[str(name)] = entry
    return {"rows": rows, "column_count": int(df.shape[1]), "duplicate_rows": duplicates,
            "duplicate_pct": round(duplicates / rows * 100, 4) if rows else 0.0, "columns": columns,
            "columns_profiled": len(columns), "taken_at": dt.datetime.utcnow().isoformat()}


# ------------------------------------------------------------------ metrics --

def psi(base: list[float], new: list[float]) -> float:
    """Population Stability Index of two aligned probability vectors."""
    total = 0.0
    for b, n in zip(base, new):
        b, n = max(b, EPS), max(n, EPS)
        total += (n - b) * math.log(n / b)
    return total


def numeric_psi(base_sample: list[float], new_sample: list[float]) -> float:
    base, new = np.asarray(base_sample, dtype=float), np.asarray(new_sample, dtype=float)
    edges = np.unique(np.quantile(base, np.linspace(0, 1, 11)))
    if len(edges) < 3:  # (near-)constant baseline: one bin for "same value", one for "anything else"
        value = float(base[0])
        return psi([1.0, 0.0], [float(np.mean(new == value)), float(np.mean(new != value))])
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(base, edges)[0] / len(base)
    n = np.histogram(new, edges)[0] / len(new)
    return psi(list(b), list(n))


def ks_statistic(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value."""
    x, y = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([x, y])
    d = float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x)
                            - np.searchsorted(y, grid, side="right") / len(y))))
    en = math.sqrt(len(x) * len(y) / (len(x) + len(y)))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return d, 1.0
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def categorical_psi(base: dict, new: dict) -> tuple[float, list[str], list[str]]:
    b_counts = {**base.get("categories", {}), "__other__": base.get("other", 0)}
    n_counts = {**new.get("categories", {}), "__other__": new.get("other", 0)}
    keys = sorted(set(b_counts) | set(n_counts))
    b_total, n_total = sum(b_counts.values()) or 1, sum(n_counts.values()) or 1
    value = psi([b_counts.get(k, 0) / b_total for k in keys], [n_counts.get(k, 0) / n_total for k in keys])
    appeared = [k for k in new.get("categories", {}) if k not in base.get("categories", {})]
    vanished = [k for k in base.get("categories", {}) if k not in new.get("categories", {})]
    return value, appeared, vanished


# --------------------------------------------------------------- comparison --

def _finding(category: str, check: str, severity: str, message: str, column: str | None = None,
             value: float | None = None, threshold: float | None = None) -> dict:
    return {"category": category, "check": check, "column": column, "severity": severity,
            "metric_value": None if value is None else float(value),
            "threshold": None if threshold is None else float(threshold), "message": message}


def _level(value: float, warn: float, high: float) -> str | None:
    return "high" if value >= high else "warn" if value >= warn else None


def single_version_findings(snapshot: dict, thresholds: dict, now: dt.datetime | None = None) -> list[dict]:
    """Checks that need no baseline."""
    out = []
    level = _level(snapshot.get("duplicate_pct", 0.0), thresholds["duplicate_pct_warn"], thresholds["duplicate_pct_high"])
    if level:
        out.append(_finding("quality", "duplicate_rows", level,
                            f"{snapshot['duplicate_rows']:,} rows ({snapshot['duplicate_pct']:.1f}%) are exact duplicates "
                            "of another row. Totals and counts computed on this version are inflated unless they are meant to repeat.",
                            value=snapshot["duplicate_pct"], threshold=thresholds["duplicate_pct_warn"]))
    max_age = thresholds.get("freshness_max_age_days")
    if max_age is not None:
        now = now or dt.datetime.utcnow()
        latest = [(name, c["max"]) for name, c in snapshot["columns"].items() if c.get("kind") == "datetime" and c.get("max")]
        if latest:
            name, newest = max(latest, key=lambda t: t[1])
            age = (now - pd.Timestamp(newest).to_pydatetime().replace(tzinfo=None)).days
            if age > max_age:
                out.append(_finding("quality", "freshness", "high" if age > 2 * max_age else "warn",
                                    f"The newest value in '{name}' is {age} days old (limit {max_age:g} days). "
                                    "Recent results may describe a period that has already ended.",
                                    column=name, value=age, threshold=max_age))
    return out


def compare(new: dict, base: dict, thresholds: dict) -> dict:
    """Findings for `new` against `base`. Returns {"findings": [...], "skipped": [...]}."""
    findings, skipped = [], []
    new_cols, base_cols = new["columns"], base["columns"]

    # --- schema
    for name in sorted(set(base_cols) - set(new_cols)):
        findings.append(_finding("schema", "column_removed", "high",
                                 f"Column '{name}' is no longer present. Analyses and metric definitions that use it will fail.",
                                 column=name))
    for name in sorted(set(new_cols) - set(base_cols)):
        findings.append(_finding("schema", "column_added", "info", f"New column '{name}'.", column=name))
    shared = sorted(set(new_cols) & set(base_cols))
    for name in shared:
        b, n = base_cols[name], new_cols[name]
        if b.get("kind") != n.get("kind"):
            findings.append(_finding("schema", "type_changed", "high",
                                     f"Column '{name}' changed from {b.get('kind')} to {n.get('kind')}. A numeric column "
                                     "that became text usually means non-numeric values were introduced.", column=name))
        elif b.get("dtype") != n.get("dtype"):
            findings.append(_finding("schema", "dtype_changed", "info",
                                     f"Column '{name}' storage type changed from {b.get('dtype')} to {n.get('dtype')}.", column=name))

    # --- quality
    if base["rows"]:
        change = (new["rows"] - base["rows"]) / base["rows"] * 100
        level = _level(abs(change), thresholds["row_count_change_pct_warn"], thresholds["row_count_change_pct_high"])
        if level:
            findings.append(_finding("quality", "row_count_change", level,
                                     f"Row count went from {base['rows']:,} to {new['rows']:,} ({change:+.1f}%). "
                                     + ("A drop this size often means a partial load." if change < 0
                                        else "A jump this size often means duplicated or appended history."),
                                     value=change, threshold=thresholds["row_count_change_pct_warn"]))
    for name in shared:
        b, n = base_cols[name], new_cols[name]
        increase = n.get("missing_pct", 0.0) - b.get("missing_pct", 0.0)
        level = _level(increase, thresholds["missing_increase_pts_warn"], thresholds["missing_increase_pts_high"])
        if level:
            findings.append(_finding("quality", "missing_increase", level,
                                     f"Missing values in '{name}' rose from {b.get('missing_pct', 0):.1f}% to "
                                     f"{n.get('missing_pct', 0):.1f}%. Averages and totals now rest on fewer rows.",
                                     column=name, value=increase, threshold=thresholds["missing_increase_pts_warn"]))
        if b.get("kind") == n.get("kind") == "numeric" and b.get("std") and "min" in n:
            span = thresholds["range_sigmas"] * b["std"]
            if n["max"] > b["max"] + span or n["min"] < b["min"] - span:
                findings.append(_finding("quality", "range_violation", "warn",
                                         f"'{name}' now ranges {n['min']:g} to {n['max']:g}; the baseline ranged "
                                         f"{b['min']:g} to {b['max']:g}. Check for unit changes or entry errors.",
                                         column=name, value=max(n["max"] - b["max"], b["min"] - n["min"]), threshold=span))
        if b.get("kind") == n.get("kind") == "datetime" and b.get("max") and n.get("max") and n["max"] < b["max"]:
            findings.append(_finding("quality", "stale_data", "warn",
                                     f"The newest '{name}' is {n['max'][:10]}, older than the baseline's {b['max'][:10]}. "
                                     "This looks like an older extract.", column=name))

    # --- distribution
    min_sample = thresholds["min_sample"]
    for name in shared:
        b, n = base_cols[name], new_cols[name]
        if b.get("kind") != n.get("kind") or b.get("kind") not in ("numeric", "categorical"):
            continue
        if b.get("n", 0) < min_sample or n.get("n", 0) < min_sample:
            skipped.append({"column": name, "reason": f"fewer than {min_sample:g} values in one of the two versions "
                                                      f"({b.get('n', 0)} vs {n.get('n', 0)})"})
            continue
        if b["kind"] == "numeric":
            if not b.get("sample") or not n.get("sample"):
                continue
            value = numeric_psi(b["sample"], n["sample"])
            d, p = ks_statistic(b["sample"], n["sample"])
            level = _level(value, thresholds["psi_warn"], thresholds["psi_high"])
            if level and p < thresholds["ks_alpha"]:
                findings.append(_finding("distribution", "numeric_drift", level,
                                         f"'{name}' is distributed differently from the baseline (PSI {value:.2f}, "
                                         f"KS {d:.2f}, p {p:.3g}; mean {b.get('mean', 0):g} -> {n.get('mean', 0):g}). "
                                         "The values are valid, but comparisons with earlier periods mix two populations.",
                                         column=name, value=value, threshold=thresholds["psi_warn"]))
        else:
            value, appeared, vanished = categorical_psi(b, n)
            level = _level(value, thresholds["psi_warn"], thresholds["psi_high"])
            if level:
                extra = ""
                if appeared:
                    extra += " New values: " + ", ".join(appeared[:5]) + "."
                if vanished:
                    extra += " No longer present: " + ", ".join(vanished[:5]) + "."
                findings.append(_finding("distribution", "category_drift", level,
                                         f"The mix of values in '{name}' shifted (PSI {value:.2f}).{extra}",
                                         column=name, value=value, threshold=thresholds["psi_warn"]))
    findings.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], f["category"], f["column"] or ""))
    return {"findings": findings, "skipped": skipped}


# -------------------------------------------------------------- persistence --

def get_config(db: Session, dataset: Dataset) -> DQConfig | None:
    return db.query(DQConfig).filter_by(dataset_id=dataset.id).first()


def snapshot_for(db: Session, dataset: Dataset, version: DatasetVersion, create: bool = True) -> DQSnapshot | None:
    row = db.query(DQSnapshot).filter_by(dataset_version_id=version.id).first()
    if row is not None or not create:
        return row
    from app.metric_engine import load_frame

    row = DQSnapshot(team_id=dataset.team_id, dataset_id=dataset.id, dataset_version_id=version.id,
                     snapshot_json=build_snapshot(load_frame(version.filepath)))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def resolve_baseline(db: Session, dataset: Dataset, version: DatasetVersion) -> tuple[DatasetVersion | None, str]:
    cfg = get_config(db, dataset)
    if cfg is not None and cfg.baseline_version_id and cfg.baseline_version_id != version.id:
        pinned = db.get(DatasetVersion, cfg.baseline_version_id)
        if pinned is not None and pinned.dataset_id == dataset.id:
            return pinned, "pinned"
    previous = (db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == dataset.id, DatasetVersion.version_number < version.version_number)
                .order_by(DatasetVersion.version_number.desc()).first())
    return previous, "previous version"


def run_for_version(db: Session, dataset: Dataset, version: DatasetVersion) -> dict:
    """Snapshot a version, compare it with its baseline and record incidents.
    Idempotent: a version that already has incidents recorded is not re-recorded."""
    snap = snapshot_for(db, dataset, version)
    cfg = get_config(db, dataset)
    thresholds = merged_thresholds(cfg.thresholds_json if cfg else None)
    baseline, baseline_source = resolve_baseline(db, dataset, version)
    findings = single_version_findings(snap.snapshot_json, thresholds)
    skipped: list[dict] = []
    if baseline is not None:
        base_snap = snapshot_for(db, dataset, baseline)
        result = compare(snap.snapshot_json, base_snap.snapshot_json, thresholds)
        findings += result["findings"]
        skipped = result["skipped"]
    already = db.query(DQIncident.id).filter_by(dataset_version_id=version.id).first() is not None
    if not already:
        for f in findings:
            db.add(DQIncident(team_id=dataset.team_id, dataset_id=dataset.id, dataset_version_id=version.id,
                              baseline_version_id=baseline.id if baseline else None, category=f["category"],
                              check_name=f["check"], column_name=f["column"], severity=f["severity"],
                              metric_value=f["metric_value"], threshold=f["threshold"], message=f["message"]))
        db.commit()
    return {"findings": findings, "skipped": skipped,
            "baseline_version": baseline.version_number if baseline else None, "baseline_source": baseline_source}


def run_safely(db: Session, dataset: Dataset, version: DatasetVersion) -> None:
    """For the upload path: data-quality checks must never fail an upload."""
    try:
        run_for_version(db, dataset, version)
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.warning("Data-quality checks failed for dataset %s version %s", dataset.id, version.id, exc_info=True)


def has_blocking_incidents(db: Session, dataset_id: int, version_id: int | None) -> list[DQIncident]:
    """Open, high-severity schema or quality incidents on a version: the kind
    that makes a KPI movement more likely an artefact than a business change."""
    if version_id is None:
        return []
    return (db.query(DQIncident)
            .filter(DQIncident.dataset_id == dataset_id, DQIncident.dataset_version_id == version_id,
                    DQIncident.status == "open", DQIncident.severity == "high",
                    DQIncident.category.in_(("schema", "quality"))).all())


def incident_out(i: DQIncident) -> dict:
    return {"id": i.id, "dataset_id": i.dataset_id, "dataset_version_id": i.dataset_version_id,
            "baseline_version_id": i.baseline_version_id, "category": i.category, "check": i.check_name,
            "column": i.column_name, "severity": i.severity, "metric_value": i.metric_value,
            "threshold": i.threshold, "message": i.message, "status": i.status, "created_at": i.created_at,
            "resolved_at": i.resolved_at}


def overview(db: Session, dataset: Dataset) -> dict:
    """Everything the Data Quality page shows for one dataset."""
    versions = (db.query(DatasetVersion).filter_by(dataset_id=dataset.id)
                .order_by(DatasetVersion.version_number).all())
    cfg = get_config(db, dataset)
    incidents = (db.query(DQIncident).filter_by(dataset_id=dataset.id)
                 .order_by(DQIncident.created_at.desc(), DQIncident.id.desc()).limit(300).all())
    by_version: dict[int, list[DQIncident]] = {}
    for i in incidents:
        by_version.setdefault(i.dataset_version_id, []).append(i)
    history = []
    for v in versions:
        snap = snapshot_for(db, dataset, v, create=False)
        s = snap.snapshot_json if snap else {}
        cols = (s.get("columns") or {}).values()
        vi = by_version.get(v.id, [])
        history.append({
            "version_id": v.id, "version": v.version_number, "uploaded_at": v.uploaded_at, "filename": v.filename,
            "rows": s.get("rows", v.row_count), "duplicate_pct": s.get("duplicate_pct"),
            "avg_missing_pct": round(sum(c.get("missing_pct", 0) for c in cols) / len(cols), 3) if cols else None,
            "incidents": {cat: sum(1 for i in vi if i.category == cat) for cat in ("schema", "quality", "distribution")},
            "max_drift_psi": max((i.metric_value or 0 for i in vi if i.category == "distribution"), default=None),
            "has_snapshot": snap is not None,
        })
    latest = versions[-1] if versions else None
    baseline, source = resolve_baseline(db, dataset, latest) if latest else (None, "")
    return {
        "dataset_id": dataset.id, "filename": dataset.filename,
        "latest_version": latest.version_number if latest else None,
        "baseline": {"version_id": baseline.id if baseline else None,
                     "version": baseline.version_number if baseline else None, "source": source,
                     "pinned": bool(cfg and cfg.baseline_version_id)},
        "thresholds": merged_thresholds(cfg.thresholds_json if cfg else None),
        "threshold_overrides": dict(cfg.thresholds_json or {}) if cfg else {},
        "history": history,
        "incidents": [incident_out(i) for i in incidents],
        "open_incidents": sum(1 for i in incidents if i.status == "open"),
    }


def compare_versions(db: Session, dataset: Dataset, a: DatasetVersion, b: DatasetVersion) -> dict:
    """Change summary between two versions of a dataset (a = earlier, b = later)."""
    sa, sb = snapshot_for(db, dataset, a).snapshot_json, snapshot_for(db, dataset, b).snapshot_json
    cfg = get_config(db, dataset)
    result = compare(sb, sa, merged_thresholds(cfg.thresholds_json if cfg else None))
    columns = []
    for name in sorted(set(sa["columns"]) | set(sb["columns"])):
        ca, cb = sa["columns"].get(name), sb["columns"].get(name)
        row = {"column": name, "in_a": ca is not None, "in_b": cb is not None,
               "kind_a": (ca or {}).get("kind"), "kind_b": (cb or {}).get("kind"),
               "missing_pct_a": (ca or {}).get("missing_pct"), "missing_pct_b": (cb or {}).get("missing_pct")}
        if ca and cb and ca.get("kind") == cb.get("kind") == "numeric":
            row.update({"mean_a": ca.get("mean"), "mean_b": cb.get("mean"), "min_a": ca.get("min"),
                        "min_b": cb.get("min"), "max_a": ca.get("max"), "max_b": cb.get("max")})
        columns.append(row)
    return {
        "a": {"version_id": a.id, "version": a.version_number, "rows": sa["rows"], "content_sha256": a.content_sha256},
        "b": {"version_id": b.id, "version": b.version_number, "rows": sb["rows"], "content_sha256": b.content_sha256},
        "identical_content": bool(a.content_sha256 and a.content_sha256 == b.content_sha256),
        "row_change": sb["rows"] - sa["rows"], "columns": columns,
        "findings": result["findings"], "skipped": result["skipped"],
    }
