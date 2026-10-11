"""Deterministic verification of analytical claims, and the evidence records
that result.

The LLM Critic stays (it judges whether an analysis answers the question). This
module is the part that does not depend on a model's opinion: for every KPI,
chart and narrative on a dashboard it runs explicit checks and records what was
checked, with what outcome, and why.

Check outcomes
    pass            the check ran and held
    fail            the check ran and did not hold
    not_run         the check could not be performed
    warn            the check ran; the result holds but with a caveat
    not_applicable  the check has no meaning for this claim

Status rules (documented; see docs/PLATFORM.md "Verification")
    unverified              any required check is `fail`, or any required check
                            is `not_run`. A check that did not run is never
                            counted as passed.
    verified_with_caveats   every required check passed, and at least one check
                            is `warn` or a limitation is attached.
    verified                every required check passed, no warnings, no
                            limitations.

Three kinds of validity are reported separately and never merged:
    mathematical   do the numbers follow from the data?  (what the checks test)
    statistical    is the sample adequate, is the effect distinguishable from
                   noise?  Not established by arithmetic.
    causal         does X explain Y?  Never established here; causal wording in
                   a narrative is flagged as a limitation.
"""
from __future__ import annotations

import math
import re
import uuid
from typing import Any

from app import config
from app.change_detection import parse_kpi_value

PASS, FAIL, NOT_RUN, WARN, NA = "pass", "fail", "not_run", "warn", "not_applicable"
VERIFIED, CAVEATS, UNVERIFIED = "verified", "verified_with_caveats", "unverified"

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s?[kKmMbB](?![A-Za-z]))?\s?%?")
_CAUSAL = re.compile(
    r"\b(because|caused|causes|causing|due to|led to|leads to|resulted in|results in|as a result of|"
    r"driven by|drove|thanks to|owing to|attributable to|responsible for)\b", re.IGNORECASE)
_COUNT_LABEL = re.compile(r"\b(count|number of|how many|# of|total rows|records)\b", re.IGNORECASE)
_AVERAGE_LABEL = re.compile(r"\b(avg|average|mean|median|per|rate|ratio)\b", re.IGNORECASE)
_SHARE_LABEL = re.compile(r"\b(share|proportion|percentage of|percent of|% of|portion)\b", re.IGNORECASE)
SMALL_SAMPLE_ROWS = 30


def check(name: str, required: bool, outcome: str, reason_code: str = "", detail: str = "") -> dict:
    return {"check": name, "required": required, "outcome": outcome, "reason_code": reason_code, "detail": detail}


# ------------------------------------------------------------------ numbers --

def extract_numbers(obj: Any, _depth: int = 0) -> list[float]:
    """Every finite number in a JSON-like value, however nested."""
    out: list[float] = []
    if _depth > 12 or obj is None or isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        f = float(obj)
        if math.isfinite(f):
            out.append(f)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(extract_numbers(v, _depth + 1))
    elif isinstance(obj, (list, tuple)):
        for v in obj[:5000]:
            out.extend(extract_numbers(v, _depth + 1))
    elif isinstance(obj, str):
        stripped = obj.strip().replace(",", "")
        try:
            f = float(stripped)
            if math.isfinite(f):
                out.append(f)
        except ValueError:
            pass
    return out


def extract_strings(obj: Any, _depth: int = 0) -> list[str]:
    out: list[str] = []
    if _depth > 12 or obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(extract_strings(v, _depth + 1))
    elif isinstance(obj, (list, tuple)):
        for v in obj[:5000]:
            out.extend(extract_strings(v, _depth + 1))
    return out


def _display_precision(text: str) -> tuple[float | None, float, bool]:
    """(value, one unit of the last displayed digit, is_percent) for a display
    string such as "$1,234.5", "12.4%" or "1.2M"."""
    value = parse_kpi_value(text)
    if value is None:
        return None, 0.0, False
    m = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", text)
    literal = m.group().replace(",", "")
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    literal_value = abs(float(literal)) or 1.0
    scale = abs(value) / literal_value if value else 1.0  # 1e3 / 1e6 / 1e9 when a suffix applied
    return value, (10 ** -decimals) * scale, "%" in text[m.end():m.end() + 3]


def number_matches(display: str, candidates: list[float], rel_tol: float | None = None) -> tuple[bool, float | None]:
    """Does the number shown in `display` agree with any candidate?

    Agreement means the candidate could have been displayed this way: it is
    less than one unit of the last displayed digit away, which covers both
    rounding and truncation and nothing more ("1,235" matches 1234.6 and
    1235.9, not 1236.9). `rel_tol` (VERIFY_REL_TOLERANCE, default 0) can widen
    that for a deployment that needs it. A percentage also matches a candidate
    expressed as a fraction (12.4% vs 0.124)."""
    rel_tol = config.VERIFY_REL_TOLERANCE if rel_tol is None else rel_tol
    value, unit, is_percent = _display_precision(display)
    if value is None:
        return False, None
    tol = max(unit * (1 - 1e-6), rel_tol * abs(value)) + 1e-12
    for c in candidates:
        if abs(c - value) <= tol:
            return True, c
        if is_percent and abs(c * 100 - value) <= tol:
            return True, c
    return False, None


def narrative_numbers(narrative: str, ignore_text: str = "") -> list[str]:
    """Numeric claims in a narrative worth checking. Skipped: years, and bare
    small integers (ordinals and list sizes: "top 3", "2 regions"), and numbers
    that merely repeat the question."""
    ignore = {m.group().strip() for m in _NUMBER.finditer(ignore_text or "")}
    found = []
    for m in _NUMBER.finditer(narrative or ""):
        token = m.group().strip()
        if not token or token in ignore:
            continue
        bare = token.replace(",", "")
        if re.fullmatch(r"\d+", bare):
            n = int(bare)
            if n <= 12 or 1900 <= n <= 2100:
                continue
        found.append(token)
    return found


# ------------------------------------------------------------------- status --

def decide_status(checks: list[dict], limitations: list[str]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    required = [c for c in checks if c["required"]]
    failed = [c for c in required if c["outcome"] == FAIL]
    missing = [c for c in required if c["outcome"] == NOT_RUN]
    for c in failed:
        reasons.append(f"failed:{c['check']}:{c['reason_code']}")
    for c in missing:
        reasons.append(f"not_run:{c['check']}:{c['reason_code']}")
    if failed or missing:
        return UNVERIFIED, reasons
    warned = [c for c in checks if c["outcome"] == WARN]
    for c in warned:
        reasons.append(f"warn:{c['check']}:{c['reason_code']}")
    for lim in limitations:
        reasons.append(f"limitation:{lim}")
    if warned or limitations:
        return CAVEATS, reasons
    return VERIFIED, reasons


def _math_validity(status: str) -> str:
    return {VERIFIED: "established", CAVEATS: "established_with_caveats", UNVERIFIED: "not_established"}[status]


def _statistical_validity(dataset: dict, limitations: list[str]) -> str:
    rows = dataset.get("row_count") or 0
    if 0 < rows < SMALL_SAMPLE_ROWS:
        limitations.append("small_sample")
        return f"weak: only {rows} rows"
    return "not_assessed"


# --------------------------------------------------------------- provenance --

def _provenance_check(dataset: dict, has_log: bool) -> dict:
    recorded, now = dataset.get("fingerprint"), dataset.get("fingerprint_now")
    if not has_log:
        return check("provenance", True, FAIL, "no_execution_record",
                     "No execution record links this claim to the code that produced it.")
    if not recorded:
        return check("provenance", True, NOT_RUN, "no_dataset_fingerprint",
                     "The dataset version has no content fingerprint, so its integrity could not be confirmed.")
    if now is None:
        return check("provenance", True, NOT_RUN, "dataset_file_unavailable",
                     "The dataset file could not be read to confirm its fingerprint.")
    if now != recorded:
        return check("provenance", True, FAIL, "dataset_changed",
                     "The dataset file no longer matches the fingerprint recorded for this version.")
    return check("provenance", True, PASS, "", f"Dataset fingerprint {recorded[:12]} confirmed; execution record present.")


# --------------------------------------------------------------------- KPIs --

def _recomputation_check(display: str, numeric: bool, critic: dict | None, structured: dict | None) -> tuple[dict, str | None]:
    name = "independent_recomputation"
    if structured is not None:
        if structured.get("error"):
            return check(name, True, NOT_RUN, "reference_not_applicable", str(structured["error"])[:300]), None
        engine, ref = structured.get("engine_value"), structured.get("reference_value")
        if engine is None or ref is None:
            return check(name, True, NOT_RUN, "no_value", "One of the two computations returned no value."), None
        if not math.isclose(engine, ref, rel_tol=1e-9, abs_tol=1e-9):
            return check(name, True, FAIL, "recomputation_mismatch",
                         f"Engine computed {engine!r}; the independent reference computed {ref!r}."), repr(ref)
        if numeric and not number_matches(display, [ref])[0]:
            return check(name, True, FAIL, "display_mismatch",
                         f"The displayed value does not match the recomputed {ref!r}."), repr(ref)
        return check(name, True, PASS, "", "Recomputed by an independent implementation (csv + plain Python); "
                                           "both agree."), repr(ref)
    if critic is None or not critic.get("ran"):
        return check(name, True, NOT_RUN, "no_verification_run", "No independent verification was run."), None
    if critic.get("verdict") == "rejected":
        return check(name, True, FAIL, "critic_rejected", "The independent reviewer rejected this analysis."), None
    if not critic.get("success"):
        return check(name, True, NOT_RUN, "verification_script_failed",
                     "The independent verification script did not run successfully."), None
    result = critic.get("result")
    if numeric:
        ok, hit = number_matches(display, extract_numbers(result))
        if ok:
            return check(name, True, PASS, "", f"The independent verification recomputed {hit!r}, which matches."), repr(hit)
        return check(name, True, NOT_RUN, "no_matching_recomputed_value",
                     "The independent verification did not recompute this figure, so it is not confirmed."), None
    needle = display.strip().lower()
    if needle and any(needle == s.strip().lower() for s in extract_strings(result)):
        return check(name, True, PASS, "", "The independent verification produced the same value."), display
    return check(name, True, NOT_RUN, "no_matching_recomputed_value",
                 "The independent verification did not produce this value, so it is not confirmed."), None


def _invariant_check(label: str, display: str, value: float | None, is_percent: bool) -> dict | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return check("invariants", True, FAIL, "not_finite", "The value is not a finite number.")
    if _SHARE_LABEL.search(label) and is_percent and not (-1e-9 <= value <= 100 + 1e-9):
        return check("invariants", True, FAIL, "share_out_of_range",
                     f"A share must be between 0% and 100%; got {display}.")
    if _COUNT_LABEL.search(label) and not _AVERAGE_LABEL.search(label) and not is_percent and value < 0:
        return check("invariants", True, FAIL, "negative_count", f"A count cannot be negative; got {display}.")
    return check("invariants", True, PASS, "", "Finite, and within the bounds its label implies.")


def verify_kpi(kpi: dict, steps: dict[int, dict], critic: dict | None, dataset: dict,
               structured: dict | None = None) -> dict:
    label, display = str(kpi.get("label", "")), str(kpi.get("value", ""))
    step = steps.get(kpi.get("source_step_index"))
    checks: list[dict] = []
    limitations: list[str] = []
    value, _, is_percent = _display_precision(display)
    numeric = value is not None

    result = step.get("result") if step else None
    if step is None:
        checks.append(check("result_present", True, FAIL, "missing_step",
                            "The step this value is attributed to did not run successfully."))
    elif result in (None, {}, []):
        checks.append(check("result_present", True, FAIL, "missing_result", "The source step produced no structured result."))
    elif isinstance(result, dict) and "_unparseable_result" in result:
        checks.append(check("result_present", True, FAIL, "unparseable_result",
                            "The source step's result was not valid JSON."))
    else:
        checks.append(check("result_present", True, PASS, "", "The source step produced a structured result."))

    if step is not None and result not in (None, {}, []):
        if numeric:
            ok, _ = number_matches(display, extract_numbers(result))
            if ok:
                checks.append(check("value_in_result", True, PASS, "", "The displayed value appears in the source step's result."))
            else:
                elsewhere = any(number_matches(display, extract_numbers(s.get("result")))[0]
                                for i, s in steps.items() if i != kpi.get("source_step_index"))
                if elsewhere:
                    checks.append(check("value_in_result", True, WARN, "value_from_other_step",
                                        "The value appears in a different step's result than the one cited."))
                else:
                    checks.append(check("value_in_result", True, FAIL, "value_not_in_results",
                                        "The displayed value does not appear in any computed result."))
        else:
            needle = display.strip().lower()
            if needle and any(needle in s.lower() for s in extract_strings(result)):
                checks.append(check("value_in_result", True, PASS, "", "The displayed value appears in the source step's result."))
            else:
                checks.append(check("value_in_result", True, FAIL, "value_not_in_results",
                                    "The displayed value does not appear in the source step's result."))
    else:
        checks.append(check("value_in_result", True, NOT_RUN, "no_result_to_compare", "There is no result to compare against."))

    recompute, recomputed_value = _recomputation_check(display, numeric, critic, structured)
    checks.append(recompute)
    checks.append(_provenance_check(dataset, bool(step and step.get("execution_log_id"))))
    inv = _invariant_check(label, display, value, is_percent)
    if inv is not None:
        checks.append(inv)

    statistical = _statistical_validity(dataset, limitations)
    status, reasons = decide_status(checks, limitations)
    return {
        "claim_id": uuid.uuid4().hex, "claim_type": "kpi", "element_id": kpi.get("element_id"),
        "claim_text": f"{label}: {display}", "original_value": display, "recomputed_value": recomputed_value,
        "metric": {"label": label, "formula": (structured or {}).get("formula") or (step or {}).get("formula") or "",
                   "filters": (structured or {}).get("filters") or [], "group_by": (structured or {}).get("group_by") or [],
                   "source_step_index": kpi.get("source_step_index")},
        "execution_log_id": (step or {}).get("execution_log_id"), "artifact_refs": [],
        "checks": checks, "limitations": limitations, "status": status, "reasons": reasons,
        "validity": {"mathematical": _math_validity(status), "statistical": statistical, "causal": "not_applicable"},
    }


# ---------------------------------------------------------------- narrative --

def verify_narrative(narrative: str, element_id: str | None, kpis: list[dict], steps: dict[int, dict],
                     critic: dict | None, dataset: dict, question_text: str = "",
                     deterministic: bool = False) -> dict:
    checks: list[dict] = []
    limitations: list[str] = []
    supported_pool = extract_numbers([s.get("result") for s in steps.values()])
    for k in kpis:
        v = parse_kpi_value(str(k.get("value", "")))
        if v is not None:
            supported_pool.append(v)
    tokens = narrative_numbers(narrative, question_text)
    unsupported = [t for t in tokens if not number_matches(t, supported_pool)[0]]
    if not tokens:
        checks.append(check("narrative_numbers_supported", True, NA, "no_numeric_claims",
                            "The narrative states no checkable numbers."))
    elif unsupported:
        checks.append(check("narrative_numbers_supported", True, FAIL, "unsupported_numbers",
                            "Numbers in the narrative that no computed result supports: " + ", ".join(unsupported[:8])))
    else:
        checks.append(check("narrative_numbers_supported", True, PASS, "",
                            f"All {len(tokens)} numbers in the narrative appear in computed results."))

    if deterministic:
        checks.append(check("independent_review", True, NA, "templated_narrative",
                            "The narrative is generated from the computed values by a fixed template."))
    elif critic is None or not critic.get("ran"):
        checks.append(check("independent_review", True, NOT_RUN, "no_review", "No independent review was run."))
    elif critic.get("verdict") == "verified":
        checks.append(check("independent_review", True, PASS, "", "The independent reviewer accepted the analysis."))
    else:
        checks.append(check("independent_review", True, FAIL, "critic_rejected",
                            "The independent reviewer rejected the analysis."))

    causal_terms = sorted({m.group(1).lower() for m in _CAUSAL.finditer(narrative or "")})
    if causal_terms:
        checks.append(check("causal_language", False, WARN, "causal_claim_not_established",
                            "The narrative uses causal wording (" + ", ".join(causal_terms[:5]) + "). The analysis is "
                            "descriptive; it does not establish cause."))
        causal = "not_established"
    else:
        checks.append(check("causal_language", False, PASS, "", "No causal wording found."))
        causal = "no_causal_claim"

    statistical = _statistical_validity(dataset, limitations)
    status, reasons = decide_status(checks, limitations)
    return {
        "claim_id": uuid.uuid4().hex, "claim_type": "narrative", "element_id": element_id,
        "claim_text": (narrative or "")[:2000], "original_value": None, "recomputed_value": None,
        "metric": {}, "execution_log_id": None, "artifact_refs": [],
        "checks": checks, "limitations": limitations, "status": status, "reasons": reasons,
        "validity": {"mathematical": _math_validity(status), "statistical": statistical, "causal": causal},
    }


def verify_chart(chart: dict, steps: dict[int, dict], dataset: dict, deterministic: bool = False) -> dict:
    step = steps.get(chart.get("step_index"))
    checks = [
        check("artifact_validated", True, PASS if step else FAIL, "" if step else "missing_step",
              "The chart file was produced by a successful step and passed artifact validation."
              if step else "No successful step produced this chart."),
        _provenance_check(dataset, bool(step and step.get("execution_log_id"))),
    ]
    # Not recomputing each plotted value is recorded as a check that did not
    # run. It is not a required check for a chart, so it does not lower the
    # chart's status -- but it is never shown as a pass either.
    limitations: list[str] = []
    if deterministic:
        checks.append(check("independent_recomputation", False, PASS, "", "Drawn from values that were recomputed independently."))
    else:
        checks.append(check("independent_recomputation", False, NOT_RUN, "chart_values_not_recomputed",
                            "The values plotted in the chart were not recomputed individually."))
    status, reasons = decide_status(checks, limitations)
    return {
        "claim_id": uuid.uuid4().hex, "claim_type": "chart", "element_id": chart.get("element_id"),
        "claim_text": f"Chart: {chart.get('title', '')}", "original_value": None, "recomputed_value": None,
        "metric": {"source_step_index": chart.get("step_index")},
        "execution_log_id": (step or {}).get("execution_log_id"),
        "artifact_refs": [{"type": "plotly_figure", "step_index": chart.get("step_index"), "title": chart.get("title", "")}],
        "checks": checks, "limitations": limitations, "status": status, "reasons": reasons,
        "validity": {"mathematical": _math_validity(status), "statistical": "not_assessed", "causal": "not_applicable"},
    }


# ---------------------------------------------------------------- dashboard --

def build_evidence(*, kpis: list[dict], charts: list[dict], narrative: str, narrative_element_id: str | None,
                   steps: dict[int, dict], critic: dict | None, dataset: dict, question_text: str = "",
                   structured: dict[str, dict] | None = None, deterministic: bool = False) -> list[dict]:
    """Evidence for every element of a dashboard. Pure: no I/O."""
    structured = structured or {}
    records = [verify_kpi(k, steps, critic, dataset, structured.get(k.get("element_id"))) for k in kpis]
    records += [verify_chart(c, steps, dataset, deterministic) for c in charts]
    records.append(verify_narrative(narrative, narrative_element_id, kpis, steps, critic, dataset,
                                    question_text, deterministic))
    return records


def has_required_failure(records: list[dict]) -> bool:
    return any(c["required"] and c["outcome"] == FAIL for r in records for c in r["checks"])


def failure_feedback(records: list[dict]) -> str:
    """What failed, phrased for the Planner on a recovery pass."""
    lines = []
    for r in records:
        for c in r["checks"]:
            if c["required"] and c["outcome"] == FAIL:
                lines.append(f"- {r['claim_text'][:120]} -- {c['check']}: {c['detail']}")
    return "Deterministic verification failed these checks:\n" + "\n".join(lines[:12])


def combine_verdict(base_state: str, records: list[dict]) -> str:
    """Dashboard verdict from the reviewer's verdict and the evidence.

    A failed required check anywhere makes the dashboard UNVERIFIED no matter
    what the reviewer said. A claim that is merely unconfirmed (a required
    check that could not run) or carries caveats lowers VERIFIED to
    VERIFIED_WITH_CAVEATS; the claim itself still reads `unverified`."""
    if base_state == "UNVERIFIED" or has_required_failure(records):
        return "UNVERIFIED"
    if any(r["status"] != VERIFIED for r in records):
        return "VERIFIED_WITH_CAVEATS"
    return base_state


def summarize(records: list[dict]) -> dict:
    counts = {VERIFIED: 0, CAVEATS: 0, UNVERIFIED: 0}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"claims": len(records), **counts,
            "failed_checks": sum(1 for r in records for c in r["checks"] if c["outcome"] == FAIL),
            "checks_not_run": sum(1 for r in records for c in r["checks"] if c["required"] and c["outcome"] == NOT_RUN)}


def persist(db, records: list[dict], *, team_id: int | None, question_id: int, dashboard_id: int | None,
            dataset: dict) -> None:
    from app.models import EvidenceRecord

    for r in records:
        db.add(EvidenceRecord(
            team_id=team_id, question_id=question_id, dashboard_id=dashboard_id, element_id=r.get("element_id"),
            claim_id=r["claim_id"], claim_type=r["claim_type"], claim_text=r["claim_text"],
            metric_json=r.get("metric") or {}, dataset_version_id=dataset.get("version_id"),
            dataset_fingerprint=dataset.get("fingerprint"), execution_log_id=r.get("execution_log_id"),
            artifact_refs_json=r.get("artifact_refs") or [], original_value=r.get("original_value"),
            recomputed_value=r.get("recomputed_value"), checks_json=r["checks"],
            limitations_json=r["limitations"], status=r["status"], reasons_json=r["reasons"],
            validity_json=r["validity"],
        ))
    db.commit()


def record_out(e) -> dict:
    return {
        "claim_id": e.claim_id, "claim_type": e.claim_type, "element_id": e.element_id,
        "claim_text": e.claim_text, "status": e.status, "reasons": list(e.reasons_json or []),
        "metric": e.metric_json or {}, "original_value": e.original_value, "recomputed_value": e.recomputed_value,
        "checks": list(e.checks_json or []), "limitations": list(e.limitations_json or []),
        "validity": e.validity_json or {}, "dataset_version_id": e.dataset_version_id,
        "dataset_fingerprint": e.dataset_fingerprint, "execution_log_id": e.execution_log_id,
        "artifact_refs": list(e.artifact_refs_json or []), "created_at": e.created_at,
    }
