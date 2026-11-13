"""Segment-level what-if: change inputs for some segments (a region, product, campaign) and see each segment and the total.

How double counting is prevented
--------------------------------
* Segments partition the rows by *one* column, so every row belongs to exactly one segment. A missing value is its own
  segment, "(missing)", never dropped. The result states how many rows are in scope and how many the segments hold.
* The total is not the model applied to the whole data again. For outputs that add up across segments (declared
  `additive`) it is the sum of the segment outputs; ratios (a margin) are recomputed from those sums; any other output
  is reported as not additive rather than summed.
* A cost that the data does not split by segment (a fixed cost) is a *total* that is allocated once, in shares that sum
  to 1, by an input you name (volume, say) or equally. A value that applies to every segment as entered (a unit cost) is
  stated as such.
* A segment-specific adjustment replaces a shared one for the same input, so an input is never changed twice.
* Verification recomputes the segment sizes and inputs from the file, each segment's outputs, the sums, the allocation,
  and that the segments add up to the figure computed from all rows at once. A mapped input that is not additive (a
  simple average of a price column) makes that last check fail, and the result then says it is unverified.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from app import scenarios
from app.scenarios import ScenarioError

SEGMENTS_VERSION = "scenario-segments.1"
MAX_SEGMENTS = 50
MISSING = "(missing)"
THIN_ROWS = 5
EQUAL = "equal"


class SegmentError(ScenarioError):
    """The segment request is invalid; the message says why."""


def segment_labels(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise SegmentError(f"The segment column '{column}' does not exist in this dataset")
    if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
        raise SegmentError(f"The segment column '{column}' is numeric; choose a category column such as a region or product")
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        raise SegmentError(f"The segment column '{column}' holds dates; use the period controls for time")
    return df[column].astype("object").where(df[column].notna(), MISSING).astype(str).str.strip().replace("", MISSING)


def segment_counts(df: pd.DataFrame, column: str) -> dict:
    labels = segment_labels(df, column)
    counts = labels.value_counts()
    listed = counts.head(MAX_SEGMENTS)
    return {"segment_column": column, "rows": int(len(df)), "segments": [{"value": str(k), "rows": int(v)} for k, v in listed.items()],
            "total_segments": int(len(counts)), "truncated": len(counts) > MAX_SEGMENTS}


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise SegmentError(f"{what} must be a finite number")
    return float(v)


def run(model: "str | dict", scope: pd.DataFrame, baseline_fn: Callable[[pd.DataFrame], dict], *, segment_column: str,
        assumptions: dict | None = None, shared_adjustments: dict | None = None, segment_adjustments: dict | None = None,
        shared_inputs: dict | None = None, constant_inputs: dict | None = None, verify: bool = True,
        reference: dict | None = None) -> dict:
    """Segment-level scenario over the rows in `scope`. `baseline_fn(frame)` gives {"inputs", "sources", "missing"}
    for a set of rows (built-in mapping or a custom model's bindings). `reference` (csv path and filters) lets the
    verifier recompute from the file."""
    spec = scenarios._model(model)  # noqa: SLF001
    assumptions = scenarios.validate_assumptions(model, assumptions)
    shared_adjustments, segment_adjustments = dict(shared_adjustments or {}), dict(segment_adjustments or {})
    shared_inputs, constant_inputs = dict(shared_inputs or {}), dict(constant_inputs or {})
    for name in [*shared_inputs, *constant_inputs]:
        if name not in spec["inputs"]:
            raise SegmentError(f"Unknown input '{name}'. Available: {', '.join(spec['inputs'])}")
    overlap = set(shared_inputs) & set(constant_inputs)
    if overlap:
        raise SegmentError(f"'{sorted(overlap)[0]}' is both a total to allocate and a value for every segment; choose one")
    if scope.empty:
        raise SegmentError("There are no rows in scope, so there are no segments")

    labels = segment_labels(scope, segment_column)
    counts = labels.value_counts()
    if len(counts) > MAX_SEGMENTS:
        raise SegmentError(f"'{segment_column}' has {len(counts)} distinct values; at most {MAX_SEGMENTS} segments are supported. "
                           "Filter the data or choose a column with fewer categories.")
    names = sorted(counts.index, key=lambda v: (-counts[v], v))
    for key in segment_adjustments:
        if key not in counts.index:
            raise SegmentError(f"There is no segment '{key}' in '{segment_column}'. Segments: {', '.join(names[:12])}"
                               + (" …" if len(names) > 12 else ""))
    for key, adj in segment_adjustments.items():
        if not isinstance(adj, dict):
            raise SegmentError(f"The adjustments for segment '{key}' must be an object of input adjustments")

    # ---- baselines: every segment, and all rows together (for the consistency check)
    seg_base, seg_sources = {}, {}
    for v in names:
        b = baseline_fn(scope[labels == v])
        seg_base[v], seg_sources[v] = dict(b["inputs"]), b["sources"]
    overall_b = baseline_fn(scope)
    overall_inputs, overall_sources = dict(overall_b["inputs"]), overall_b["sources"]

    allocations: dict[str, dict] = {}
    for name in spec["inputs"]:
        if all(seg_base[v][name] is not None for v in names):
            continue
        rule = spec["inputs"][name]
        if name in constant_inputs:
            value = _num(constant_inputs[name], f"The value for {rule['label']}")
            for v in names:
                seg_base[v][name], seg_sources[v][name] = value, {"source": "constant", "formula": None, "note": "the same value in every segment, as you entered"}
            overall_inputs[name] = value
            overall_sources[name] = {"source": "constant", "formula": None, "note": "as you entered"}
        elif name in shared_inputs:
            cfg = shared_inputs[name]
            if not isinstance(cfg, dict) or set(cfg) - {"total", "allocate_by"} or "total" not in cfg:
                raise SegmentError(f"The shared input '{name}' needs {{total, allocate_by}}")
            total, by = _num(cfg["total"], f"The total for {rule['label']}"), cfg.get("allocate_by", EQUAL)
            if by == EQUAL:
                shares = {v: 1 / len(names) for v in names}
            elif by in spec["inputs"] and by != name:
                driver = {v: seg_base[v][by] for v in names}
                if any(x is None for x in driver.values()):
                    raise SegmentError(f"'{by}' has no value in every segment, so it cannot be used to share out {rule['label']}")
                if any(x < 0 for x in driver.values()) or sum(driver.values()) <= 0:
                    raise SegmentError(f"'{by}' must be non-negative and add up to more than zero to share out {rule['label']}")
                shares = {v: driver[v] / sum(driver.values()) for v in names}
            else:
                raise SegmentError(f"allocate_by for '{name}' must be 'equal' or another input of the model")
            for v in names:
                seg_base[v][name] = total * shares[v]
                seg_sources[v][name] = {"source": "allocated", "formula": None,
                                        "note": f"{shares[v] * 100:.2f}% of the total {total:,.6g}, shared by {by}"}
            overall_inputs[name] = total
            overall_sources[name] = {"source": "total", "formula": None, "note": "the total you entered, allocated across segments"}
            allocations[name] = {"total": total, "allocate_by": by, "shares": shares, "shares_sum": math.fsum(shares.values())}
        else:
            raise SegmentError(f"'{rule['label']}' ({name}) has no value in the data. Give it as a total to allocate "
                               "(shared_inputs) or as a value for every segment (constant_inputs).")
    missing_overall = [n for n in spec["inputs"] if overall_inputs.get(n) is None]
    if missing_overall:
        raise SegmentError(f"No value for: {', '.join(missing_overall)}")
    for v in names:
        seg_base[v] = scenarios.validate_inputs(model, seg_base[v])
    overall_inputs = scenarios.validate_inputs(model, overall_inputs)

    # ---- scenario per segment
    segments, effective = [], {}
    for v in names:
        adj = {**shared_adjustments, **(segment_adjustments.get(v) or {})}
        effective[v] = adj
        inputs = scenarios.apply_adjustments(model, seg_base[v], adj, assumptions)
        base_out = scenarios._outputs(model, seg_base[v], assumptions)    # noqa: SLF001
        out = scenarios._outputs(model, inputs, assumptions)              # noqa: SLF001
        notes: dict = {}
        scenarios._enforce(spec, f"Segment '{v}'", inputs, assumptions, out, notes, v)  # noqa: SLF001
        segments.append({"value": v, "rows": int(counts[v]), "thin": bool(counts[v] < THIN_ROWS), "sources": seg_sources[v],
                         "adjustments": adj, "baseline": {"inputs": seg_base[v], "outputs": base_out},
                         "scenario": {"inputs": inputs, "outputs": out}, "delta": scenarios._delta(base_out, out),  # noqa: SLF001
                         "constraint_notes": notes.get(v, [])})

    # ---- the total, from the segments
    additive = list(spec.get("additive_outputs") or [])
    ratios = dict(spec.get("ratio_outputs") or {})
    overall_base = _combine(spec, additive, ratios, [s["baseline"]["outputs"] for s in segments])
    overall_scn = _combine(spec, additive, ratios, [s["scenario"]["outputs"] for s in segments])
    primary = spec["primary_output"]
    contributions = None
    if primary in additive:
        total_change = (overall_scn[primary] or 0.0) - (overall_base[primary] or 0.0)
        contributions = {"output": primary, "total_change": total_change, "method": "each segment's own change in the output; "
                         "the segments add up to the total change",
                         "by_segment": {s["value"]: {"abs": (s["scenario"]["outputs"][primary] or 0.0) - (s["baseline"]["outputs"][primary] or 0.0)} for s in segments}}
        for s in segments:
            c = contributions["by_segment"][s["value"]]
            c["share_pct"] = c["abs"] / total_change * 100 if total_change else None
    whole = scenarios._outputs(model, overall_inputs, assumptions)        # noqa: SLF001
    consistency = {k: {"from_segments": overall_base[k], "from_all_rows": whole[k],
                       "agrees": overall_base[k] is not None and whole[k] is not None and math.isclose(overall_base[k], whole[k], rel_tol=1e-9, abs_tol=1e-9)}
                   for k in additive}
    disclosures = ["Segments split the rows by one column, so each row is in exactly one segment; a missing value is its own segment."]
    if shared_adjustments:
        disclosures.append("Shared adjustments apply to every segment; a segment's own adjustment for the same input replaces the shared one.")
    for name, a in allocations.items():
        disclosures.append(f"{spec['inputs'][name]['label']}: a total of {a['total']:,.6g} is allocated across segments "
                           f"{'equally' if a['allocate_by'] == EQUAL else 'in proportion to ' + a['allocate_by']}, in shares that add to 100%.")
    if constant_inputs:
        disclosures.append("Used as entered in every segment: " + ", ".join(spec["inputs"][n]["label"] for n in constant_inputs) + ".")
    if any(s["thin"] for s in segments):
        disclosures.append(f"Segments with fewer than {THIN_ROWS} rows are marked thin: their baselines rest on very little data.")
    if not additive:
        disclosures.append("This model declares no additive outputs, so only segment results are shown, with no total.")
    unadj = [s["value"] for s in segments if not s["adjustments"]]
    if unadj and len(unadj) < len(segments):
        disclosures.append("Segments with no adjustment keep their baseline and are still counted in the total.")
    result = {
        "kind": "segment_scenario", "data_kind": "scenario", "baseline_data_kind": "historical_actual",
        "engine_version": SEGMENTS_VERSION, "model": scenarios.model_key(model), "title": spec["title"],
        "segment_column": segment_column, "assumptions": assumptions, "primary_output": primary,
        "segments": segments,
        "overall": {"baseline": {"inputs": overall_inputs, "outputs": overall_base},
                    "scenario": {"outputs": overall_scn}, "delta": scenarios._delta(overall_base, overall_scn),  # noqa: SLF001
                    "additive_outputs": additive, "ratio_outputs": ratios,
                    "not_additive": [k for k in spec["outputs"] if k not in additive and k not in ratios]},
        "contributions": contributions, "consistency": consistency,
        "partition": {"rows_in_scope": int(len(scope)), "rows_in_segments": int(sum(counts[v] for v in names)),
                      "segments": len(names), "complete": int(len(scope)) == int(sum(counts[v] for v in names)),
                      "missing_segment": MISSING in names},
        "allocations": allocations, "constant_inputs": constant_inputs, "shared_adjustments": shared_adjustments,
        "segment_adjustments": segment_adjustments, "disclosures": disclosures,
    }
    if verify:
        from app import scenario_verify
        result["verification"] = scenario_verify.safe(scenario_verify.verify_segments, result, spec, reference)
    return result


def _combine(spec: dict, additive: list[str], ratios: dict, per_segment: list[dict]) -> dict:
    """Total outputs: sums for additive outputs, ratios from those sums, nothing for the rest."""
    out: dict[str, float | None] = {k: None for k in spec["outputs"]}
    for k in additive:
        vals = [o[k] for o in per_segment]
        out[k] = None if any(v is None for v in vals) else math.fsum(vals)
    for k, r in ratios.items():
        num, den = out.get(r["numerator"]), out.get(r["denominator"])
        scale = r.get("scale") if r.get("scale") is not None else 1.0
        out[k] = None if num is None or den in (None, 0) else num / den * scale
    return out
