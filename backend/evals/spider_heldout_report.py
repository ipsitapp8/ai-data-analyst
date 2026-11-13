"""Held-out comparison for the spider-numeric suite: frozen rules vs the v1 baseline.

Two steps, run once each:

    cd backend
    python evals/spider_heldout_report.py freeze --version 2
        records the SHA-256 of every file that decides the copilot's reading
        (reports/spider_numeric_copilot_rules_v<N>_freeze.json). Refuses if it exists.

    python -m app.evals run --suite spider-numeric --json RUN.json        (the single held-out run)

    python evals/spider_heldout_report.py report --version 2 --run-json RUN.json [--run-id ID]
        checks the files still match the freeze, then writes
        reports/spider_numeric_copilot_heldout_v<N>.{json,md}, compared case by
        case with reports/spider_numeric_copilot_baseline_v1.json. Refuses to overwrite.

Held out = the 138 questions with a numeric reference and no leakage marker.
Nothing is run or recomputed here.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
REPORTS = HERE / "reports"
CASES = HERE / "cases" / "spider_numeric.json"
BASELINE = REPORTS / "spider_numeric_copilot_baseline_v1.json"
FROZEN_FILES = ["app/copilot.py", "app/fastpath.py", "app/metric_engine.py", "app/profiling.py", "app/evals/spider.py",
                "app/evals/runner.py", "evals/cases/spider_numeric.json"]
DECLINED = ("not_understood", "clarification", "no_metric", "invalid_operation")
WRONG = ("wrong_value", "wrong_shape", "partial")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(version: int, note: str | None = None) -> int:
    target = REPORTS / f"spider_numeric_copilot_rules_v{version}_freeze.json"
    if target.exists():
        print(f"not overwriting {target.name}", file=sys.stderr)
        return 3
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND, capture_output=True, text=True).stdout.strip()
    doc = {"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git_head": head,
           "note": "working-tree files (uncommitted changes included) whose content decides the held-out result",
           "files": {f: sha(BACKEND / f) for f in FROZEN_FILES}}
    if note:
        doc["reason"] = note
    target.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(BACKEND)}")
    return 0


def rates(outcomes: list[str]) -> dict:
    n = len(outcomes)
    c = collections.Counter(outcomes)
    exact = c["correct_exact"]
    tol = exact + c["correct_tolerance"]
    declined = sum(c[o] for o in DECLINED)
    wrong = sum(c[o] for o in WRONG)
    errors = n - tol - declined - wrong
    pct = lambda k: round(k / n, 4) if n else None
    return {"questions": n, "exact": exact, "exact_rate": pct(exact), "within_tolerance": tol, "tolerance_rate": pct(tol),
            "declined": declined, "declined_rate": pct(declined), "incorrect": wrong, "incorrect_rate": pct(wrong),
            "errors": errors, "answered": tol + wrong,
            "precision_when_answering": round(tol / (tol + wrong), 4) if tol + wrong else None,
            "outcomes": dict(c.most_common())}


def report(version: int, run_json: Path, run_id: int | None, previous: Path | None = None) -> int:
    stem = REPORTS / f"spider_numeric_copilot_heldout_v{version}"
    if any(stem.with_suffix(s).exists() for s in (".json", ".md")):
        print(f"not overwriting {stem.name}.*", file=sys.stderr)
        return 3
    frozen = json.loads((REPORTS / f"spider_numeric_copilot_rules_v{version}_freeze.json").read_text(encoding="utf-8"))
    changed = [f for f, h in frozen["files"].items() if sha(BACKEND / f) != h]
    if changed:
        print(f"files changed since the freeze: {changed}; the run is not the frozen rules", file=sys.stderr)
        return 4
    run = json.loads(run_json.read_text(encoding="utf-8"))
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    before = {x["case_id"]: x for x in base["cases_failed"] + base["cases_correct"]}
    after = {r["case_id"]: r["metrics"] for r in run["results"]}
    if set(after) != set(cases) or set(before) != set(cases):
        print("the run or the baseline does not cover the 165 cases", file=sys.stderr)
        return 5
    held = sorted(i for i, c in cases.items() if c["reference"]["scorable"] and not c["leakage"])
    numeric = sorted(i for i, c in cases.items() if c["reference"]["scorable"])

    def subset(ids):
        return {"before_v1": rates([before[i]["outcome"] for i in ids]), "after": rates([after[i]["outcome"] for i in ids])}

    group = lambda o: ("correct" if o in ("correct_exact", "correct_tolerance") else "declined" if o in DECLINED
                       else "incorrect" if o in WRONG else "error")
    transitions = collections.Counter(f"{group(before[i]['outcome'])} -> {group(after[i]['outcome'])}" for i in held)
    declined_before = collections.Counter(before[i]["outcome"] for i in held if group(before[i]["outcome"]) == "declined")
    declined_after = collections.Counter(after[i]["outcome"] for i in held if group(after[i]["outcome"]) == "declined")
    still = [i for i in held if group(after[i]["outcome"]) == "declined"]
    causes = collections.Counter(after[i]["cause"] for i in still)
    issues = collections.Counter(after[i]["cause_detail"] for i in still)
    clar = collections.Counter((after[i].get("cause_detail") or "") for i in held if after[i]["outcome"] == "clarification")

    def listing(ids):
        return [{"case_id": i, "question": cases[i]["question"], "reference_sql": cases[i]["query"],
                 "reference": after[i]["reference_values"], "before": before[i]["outcome"], "after": after[i]["outcome"],
                 "copilot_formulas": after[i].get("copilot_formula"), "copilot_filters": after[i].get("copilot_filters"),
                 "copilot_value": after[i].get("copilot_value"), "oracle_formulas": after[i].get("oracle_formulas"),
                 "oracle_filters": after[i].get("oracle_filters"), "cause": after[i].get("cause"),
                 "cause_detail": after[i].get("cause_detail")} for i in ids]

    prev = None
    if previous is not None:
        old = {r["case_id"]: r["metrics"] for r in json.loads(previous.read_text(encoding="utf-8"))["results"]}
        moved = collections.Counter(f"{group(old[i]['outcome'])} -> {group(after[i]['outcome'])}" for i in held)
        prev = {"run_json": previous.name, "held_out_138": rates([old[i]["outcome"] for i in held]),
                "held_out_transitions": dict(moved.most_common()),
                "held_out_newly_correct_vs_previous": listing([i for i in held if group(after[i]["outcome"]) == "correct"
                                                                and group(old[i]["outcome"]) != "correct"])}
    lat = sorted(after[i]["latency_ms"] for i in after)
    doc = {
        "report": "spider-numeric held-out evaluation: frozen deterministic copilot rules vs the v1 baseline",
        "version": version, "suite_version": run["suite_version"], "stored_run_id": run_id,
        "started_at": run["started_at"], "provider": run["provider"],
        "model_calls": run["summary"]["pipeline"]["llm_calls"], "code_runs": run["summary"]["pipeline"]["sandbox_runs"],
        "freeze": frozen, "case_file_sha256": sha(CASES), "baseline": BASELINE.name,
        "held_out_definition": "numeric reference and no leakage marker (question_also_in_train / database_family_in_train)",
        "previous_held_out_run": prev,
        "held_out_138": subset(held), "numeric_reference_160": subset(numeric), "all_165": subset(sorted(cases)),
        "held_out_transitions": dict(transitions.most_common()),
        "held_out_declined_by_outcome": {"before_v1": dict(declined_before.most_common()), "after": dict(declined_after.most_common())},
        "held_out_still_declined_by_cause": dict(causes.most_common()),
        "held_out_still_declined_by_detail": issues.most_common(),
        "held_out_clarifications": clar.most_common(),
        "held_out_spec_matches_oracle_among_correct": sum(1 for i in held if group(after[i]["outcome"]) == "correct"
                                                          and after[i]["spec_matches_oracle"]),
        "timing_ms": {"total": sum(lat), "median": lat[len(lat) // 2], "max": lat[-1]},
        "held_out_incorrect": listing([i for i in held if group(after[i]["outcome"]) == "incorrect"]),
        "held_out_newly_correct": listing([i for i in held if group(after[i]["outcome"]) == "correct"
                                           and group(before[i]["outcome"]) != "correct"]),
        "held_out_lost": listing([i for i in held if group(before[i]["outcome"]) == "correct"
                                  and group(after[i]["outcome"]) != "correct"]),
    }
    REPORTS.mkdir(exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(doc), encoding="utf-8")
    print("wrote", stem.with_suffix(".json").relative_to(BACKEND), stem.with_suffix(".md").relative_to(BACKEND))
    return 0


def markdown(d: dict) -> str:
    def table(name, s):
        b, a = s["before_v1"], s["after"]
        row = lambda label, k, r: f"| {label} | {b[k]} ({b[r]:.1%}) | {a[k]} ({a[r]:.1%}) |"
        return [f"### {name} ({a['questions']} questions)", "", "| | Before (v1 baseline) | After (frozen rules) |",
                "| --- | ---: | ---: |", row("Exact", "exact", "exact_rate"), row("Within tolerance", "within_tolerance", "tolerance_rate"),
                row("Declined (abstained)", "declined", "declined_rate"), row("Incorrect answer", "incorrect", "incorrect_rate"),
                f"| Errors | {b['errors']} | {a['errors']} |",
                f"| Correct when it answers | {b['precision_when_answering'] if b['precision_when_answering'] is not None else '—'} | "
                f"{a['precision_when_answering'] if a['precision_when_answering'] is not None else '—'} |", ""]
    f = d["freeze"]
    lines = [f"# Spider number questions, held out: frozen copilot rules vs baseline (v{d['version']})", "",
             f"Single run {d['stored_run_id']}, {d['started_at'][:19]} UTC, suite {d['suite_version']}. Model calls: {d['model_calls']}. "
             f"Generated code runs: {d['code_runs']}. Rules frozen {f['frozen_at']} (Git HEAD {f['git_head']} plus the working tree; "
             "file hashes in the JSON). Baseline: " + d["baseline"] + ".", "",
             *([f"Why this freeze: {f['reason']}", ""] if f.get("reason") else []),
             f"Held out: {d['held_out_definition']}.", "", "Questions and SQL quoted below are from Spider (CC BY-SA 4.0).", "",
             "## Accuracy", ""]
    lines += table("Held out", d["held_out_138"]) + table("Every numeric reference", d["numeric_reference_160"]) + table("All selected", d["all_165"])
    if d.get("previous_held_out_run"):
        p, a = d["previous_held_out_run"], d["held_out_138"]["after"]
        lines += ["## Compared with the previous held-out run", "",
                  "| Held out (138) | Previous run | This run |", "| --- | ---: | ---: |",
                  *[f"| {label} | {p['held_out_138'][k]} | {a[k]} |" for label, k in
                    (("Exact", "exact"), ("Within tolerance", "within_tolerance"), ("Declined", "declined"),
                     ("Incorrect", "incorrect"), ("Errors", "errors"))], "",
                  "| Previous -> this run | Questions |", "| --- | ---: |",
                  *[f"| {k} | {v} |" for k, v in p["held_out_transitions"].items()], ""]
    lines += ["## What changed on the held-out questions (vs the v1 baseline)", "", "| Before -> after | Questions |", "| --- | ---: |",
              *[f"| {k} | {v} |" for k, v in d["held_out_transitions"].items()], "",
              "Declined, by kind:", "", "| | Before | After |", "| --- | ---: | ---: |"]
    kinds = sorted(set(d["held_out_declined_by_outcome"]["before_v1"]) | set(d["held_out_declined_by_outcome"]["after"]))
    lines += [f"| {k} | {d['held_out_declined_by_outcome']['before_v1'].get(k, 0)} | {d['held_out_declined_by_outcome']['after'].get(k, 0)} |" for k in kinds]
    lines += ["", "Still declined, by cause:", "", "| Cause | Questions |", "| --- | ---: |",
              *[f"| {k} | {v} |" for k, v in d["held_out_still_declined_by_cause"].items()], "",
              "| Detail | Questions |", "| --- | ---: |", *[f"| {k} | {v} |" for k, v in d["held_out_still_declined_by_detail"]], "",
              "## Incorrect answers (held out)", ""]
    lines += [f"- `{x['case_id']}` {x['question']} — computed {x['copilot_formulas']} {x['copilot_filters'] or ''} = "
              f"{x['copilot_value']}; reference {x['reference']} from `{x['reference_sql']}` ({x['cause_detail']})"
              for x in d["held_out_incorrect"]] or ["None."]
    lines += ["", "## Lost (correct before, not now)", ""]
    lines += [f"- `{x['case_id']}` {x['question']} — now {x['after']} ({x['cause_detail']})" for x in d["held_out_lost"]] or ["None."]
    lines += ["", f"Correct answers whose computation matches the reference SQL's: {d['held_out_spec_matches_oracle_among_correct']} "
              f"of {d['held_out_138']['after']['within_tolerance']}.", "",
              f"Time: {d['timing_ms']['total']} ms for all 165, median {d['timing_ms']['median']} ms per question.", "",
              "The newly correct questions and every per-question detail are in the JSON report beside this file."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--version", type=int, required=True)
    f.add_argument("--note", help="why this freeze exists")
    r = sub.add_parser("report")
    r.add_argument("--version", type=int, required=True)
    r.add_argument("--run-json", type=Path, required=True)
    r.add_argument("--run-id", type=int)
    r.add_argument("--previous-run-json", type=Path, help="an earlier held-out run to compare with")
    args = ap.parse_args(argv)
    return (freeze(args.version, args.note) if args.cmd == "freeze"
            else report(args.version, args.run_json, args.run_id, args.previous_run_json))


if __name__ == "__main__":
    sys.exit(main())
