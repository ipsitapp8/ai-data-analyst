"""Versioned baseline report for the spider-numeric suite.

Reads the JSON a run wrote (`python -m app.evals run --suite spider-numeric
--json PATH`) and writes, in backend/evals/reports/:

    spider_numeric_copilot_baseline_v<N>.json
    spider_numeric_copilot_baseline_v<N>.md

An existing report version is never overwritten: choose the next --version.
Nothing is run or recomputed here; every figure comes from the run's results.

    cd backend && python evals/spider_baseline_report.py --run-json PATH --version 1 [--run-id ID]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
CASES = HERE / "cases" / "spider_numeric.json"
UNSUPPORTED = ("not_understood", "clarification", "no_metric", "invalid_operation")
ANSWERED_WRONG = ("wrong_value", "wrong_shape", "partial")
ERRORS = ("compute_error", "fixture_missing")


def rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def accuracy(rows: list[dict]) -> dict:
    n = len(rows)
    ex = sum(1 for r in rows if r["m"]["outcome"] == "correct_exact")
    tol = sum(1 for r in rows if r["m"]["correct"])
    return {"cases": n, "exact": ex, "exact_rate": rate(ex, n), "within_tolerance": tol, "tolerance_rate": rate(tol, n)}


def build(run: dict, run_id: int | None) -> dict:
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    rows = [{"id": r["case_id"], "m": r["metrics"], "error": r["detail"]["error"], "case": cases[r["case_id"]]}
            for r in run["results"]]
    if len(rows) != 165 or any(r["case"]["component"] != "spider_copilot" for r in rows):
        raise SystemExit("this is not a full spider-numeric run")
    outcomes = collections.Counter(r["m"]["outcome"] for r in rows)
    scorable = [r for r in rows if r["case"]["reference"]["scorable"]]
    held_out = [r for r in scorable if not r["case"]["leakage"]]
    single = [r for r in scorable if r["case"]["answer_shape"] == "scalar"]

    by_query = collections.defaultdict(list)
    for r in scorable:
        by_query[r["case"]["reference_id"]].append(r["m"]["correct"])

    failed = [r for r in rows if not r["m"]["correct"]]
    causes = collections.Counter(r["m"]["cause"] for r in failed)
    cause_detail = {c: collections.Counter(r["m"]["cause_detail"] for r in failed if r["m"]["cause"] == c).most_common()
                    for c in causes}
    issue_counts = collections.Counter(i for r in failed if r["m"]["cause"] == "question_understanding"
                                       for i in r["m"]["understanding_issues"])
    issue_counts = collections.Counter({(k.split(":")[0] if k.startswith("wrong metric") else k): v for k, v in issue_counts.items()})
    ceiling = [r for r in single if r["m"]["oracle_status"] == "matches"]
    coincidence = [r for r in rows if r["m"]["correct"] and not r["m"]["spec_matches_oracle"]]
    lat = sorted(r["m"]["latency_ms"] for r in rows)

    def listing(sel) -> list[dict]:
        return [{"case_id": r["id"], "question": r["case"]["question"], "reference_sql": r["case"]["query"],
                 "reference": r["m"]["reference_values"], "outcome": r["m"]["outcome"], "copilot_formula": r["m"]["copilot_formula"],
                 "copilot_filters": r["m"]["copilot_filters"], "copilot_value": r["m"]["copilot_value"],
                 "oracle": {"status": r["m"]["oracle_status"], "formulas": r["m"]["oracle_formulas"],
                            "filters": r["m"]["oracle_filters"], "reason": r["m"]["oracle_reason"]},
                 "cause": r["m"]["cause"], "cause_detail": r["m"]["cause_detail"],
                 "understanding_issues": r["m"]["understanding_issues"]} for r in sel]

    return {
        "report": "spider-numeric copilot no-model baseline",
        "suite_version": run["suite_version"], "mode": run["mode"], "provider": run["provider"],
        "stored_run_id": run_id, "started_at": run["started_at"], "finished_at": run["finished_at"],
        "case_file_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "path_measured": ("copilot.handle_message with allow_model=False, first turn of a new session: rule interpretation, "
                          "validated operations, deterministic metric engine. No model call, no generated code."),
        "scoring": {"exact": "equal to the reference at 6 decimal places",
                    "within_tolerance": "within 0.1% of the reference or 0.005 absolute; includes exact",
                    "scalar_row": "correct only if every number the question asks for is returned"},
        "model_calls": run["summary"]["pipeline"]["llm_calls"], "code_runs": run["summary"]["pipeline"]["sandbox_runs"],
        "accuracy": {
            "all_165": accuracy(rows),
            "numeric_reference_160": accuracy(scorable),
            "numeric_reference_cleanly_held_out": accuracy(held_out),
            "single_number_questions": accuracy(single),
            "multi_number_questions": accuracy([r for r in scorable if r["case"]["answer_shape"] == "scalar_row"]),
            "distinct_reference_queries": {"queries": len(by_query), "all_paraphrases_correct": sum(all(v) for v in by_query.values()),
                                           "some_paraphrase_correct": sum(any(v) for v in by_query.values())},
        },
        "correct_with_a_different_computation": [r["id"] for r in coincidence],
        "outcomes": dict(outcomes.most_common()),
        "outcome_groups": {
            "correct": sum(1 for r in rows if r["m"]["correct"]),
            "unsupported_declined": sum(outcomes[o] for o in UNSUPPORTED),
            "answered_wrong": sum(outcomes[o] for o in ANSWERED_WRONG),
            "errors": sum(outcomes[o] for o in ERRORS) + sum(1 for r in rows if r["error"]),
        },
        "failure_causes": dict(causes.most_common()),
        "failure_cause_detail": cause_detail,
        "question_understanding_issues": dict(issue_counts.most_common()),
        "engine_ceiling": {
            "description": ("single-number questions with a numeric reference whose reference SQL translates into a "
                            "metric-engine spec that reproduces the reference value: what perfect question reading "
                            "would reach with the current engine"),
            "cases": len(ceiling), "of": len(single), "currently_correct": sum(1 for r in ceiling if r["m"]["correct"])},
        "timing_ms": {"total": sum(lat), "mean": round(statistics.mean(lat), 1), "median": statistics.median(lat),
                      "p95": lat[int(0.95 * (len(lat) - 1))], "max": lat[-1],
                      "note": "per case, including reading and profiling the CSV (cached after the first case on a table)"},
        "cases_correct": listing([r for r in rows if r["m"]["correct"]]),
        "cases_answered_wrong": listing([r for r in rows if r["m"]["outcome"] in ANSWERED_WRONG]),
        "cases_failed": listing([r for r in rows if not r["m"]["correct"]]),
    }


def markdown(b: dict) -> str:
    a = b["accuracy"]
    def row(name, x):
        return f"| {name} | {x['cases']} | {x['exact']} ({x['exact_rate']:.1%}) | {x['within_tolerance']} ({x['tolerance_rate']:.1%}) |"
    q = a["distinct_reference_queries"]
    lines = [
        f"# Spider number-answer questions: copilot baseline without a model ({b['suite_version']})", "",
        f"Run {b['stored_run_id'] if b['stored_run_id'] is not None else '(not stored)'}, {b['started_at'][:19]} UTC. "
        f"Case file SHA-256 `{b['case_file_sha256'][:16]}…`. Model calls: {b['model_calls']}. Generated code runs: {b['code_runs']}.", "",
        f"Path measured: {b['path_measured']}", "",
        "Questions and SQL quoted below are from Spider (CC BY-SA 4.0).", "",
        "## Accuracy", "",
        "| Subset | Questions | Exact | Within tolerance |", "| --- | ---: | ---: | ---: |",
        row("All selected questions", a["all_165"]),
        row("Reference is a number", a["numeric_reference_160"]),
        row("…cleanly held out (no leakage marker)", a["numeric_reference_cleanly_held_out"]),
        row("…one number asked for", a["single_number_questions"]),
        row("…several numbers asked for", a["multi_number_questions"]), "",
        f"Distinct reference queries: {q['queries']}; all paraphrases correct for {q['all_paraphrases_correct']}, "
        f"at least one for {q['some_paraphrase_correct']}.", "",
        f"Tolerance: {b['scoring']['within_tolerance']}. Several-number questions count only if every number is returned.", "",
        f"Correct number from a different computation than the question asks for: {len(b['correct_with_a_different_computation'])} "
        f"({', '.join(b['correct_with_a_different_computation']) or 'none'}).", "",
        "## What the copilot did", "",
        "| Outcome | Questions |", "| --- | ---: |",
        *[f"| {k} | {v} |" for k, v in b["outcomes"].items()], "",
        f"Correct {b['outcome_groups']['correct']}, declined as not understood or ambiguous {b['outcome_groups']['unsupported_declined']}, "
        f"answered with a wrong number {b['outcome_groups']['answered_wrong']}, errors {b['outcome_groups']['errors']}.", "",
        "## Why the others failed", "",
        "Each failure is put to an oracle: the reference SQL, translated from Spider's own parse into a metric-engine spec "
        "and computed by the engine on the CSV the application would receive.", "",
        "| Cause | Questions | Meaning |", "| --- | ---: | --- |",
    ]
    meaning = {"question_understanding": "the oracle spec gives the right number: the rules misread the question",
               "deterministic_limitation": "the engine or the copilot cannot express what the question asks",
               "data_representation": "the reference is not a number, or SQLite and the engine read the data differently"}
    lines += [f"| {k} | {v} | {meaning.get(k, '')} |" for k, v in b["failure_causes"].items()]
    for cause, details in b["failure_cause_detail"].items():
        lines += ["", f"### {cause}", "", "| Detail | Questions |", "| --- | ---: |", *[f"| {d} | {n} |" for d, n in details]]
    lines += ["", "### Question-understanding issues (a question can have several)", "", "| Issue | Questions |", "| --- | ---: |",
              *[f"| {k} | {v} |" for k, v in b["question_understanding_issues"].items()]]
    c = b["engine_ceiling"]
    lines += ["", "## Ceiling with the current engine", "",
              f"{c['cases']} of {c['of']} single-number questions are expressible and reproduce the reference through the "
              f"engine. With perfect question reading and no engine change, at most {c['cases']} could be correct; "
              f"{c['currently_correct']} are now.", "",
              "## Time", "", f"Total {b['timing_ms']['total']} ms; per case mean {b['timing_ms']['mean']} ms, median "
              f"{b['timing_ms']['median']} ms, 95th percentile {b['timing_ms']['p95']} ms, max {b['timing_ms']['max']} ms "
              f"({b['timing_ms']['note']}).", "",
              "## Answered with a wrong number", ""]
    lines += [f"- `{x['case_id']}` {x['question']} — computed {x['copilot_formula']} = {x['copilot_value']}, reference "
              f"{x['reference']} ({x['cause_detail']})" for x in b["cases_answered_wrong"]] or ["None."]
    lines += ["", "## Correct", ""]
    lines += [f"- `{x['case_id']}` {x['question']} — {x['copilot_formula']} {x['copilot_filters'] or ''} = {x['copilot_value']}"
              for x in b["cases_correct"]]
    lines += ["", "The full per-question listing is in the JSON report beside this file."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-json", required=True, type=Path)
    ap.add_argument("--version", required=True, type=int)
    ap.add_argument("--run-id", type=int, help="the stored evaluation run id, if the run was stored")
    args = ap.parse_args(argv)
    stem = REPORTS / f"spider_numeric_copilot_baseline_v{args.version}"
    targets = [stem.with_suffix(".json"), stem.with_suffix(".md")]
    existing = [t for t in targets if t.exists()]
    if existing:
        print(f"not overwriting {', '.join(t.name for t in existing)}; choose another --version", file=sys.stderr)
        return 3
    b = build(json.loads(args.run_json.read_text(encoding="utf-8")), args.run_id)
    REPORTS.mkdir(exist_ok=True)
    targets[0].write_text(json.dumps(b, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    targets[1].write_text(markdown(b), encoding="utf-8")
    print("wrote", *(t.relative_to(HERE.parent) for t in targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
