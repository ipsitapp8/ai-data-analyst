"""The 14 acceptance scenarios for the Analyst Copilot, run through the full
orchestrator on the invented demo datasets.

    cd backend
    ../.venv313/bin/python evals/analyst_acceptance.py --label NAME [--model on|off]

--model on needs the analyst model being served (training/scripts/serve_analyst_model.sh).
Writes backend/evals/reports/analyst_acceptance_<label>.{json,md}; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.analyst import demo_data as dd, orchestrator, planner  # noqa: E402

REPORTS = ROOT / "evals/reports"


def tools_of(r):
    return [c.get("tool") for c in r["calls"]]


def ok_calls(r):
    return all(c.get("status") == "ok" for c in r["calls"]) and bool(r["calls"])


# (dataset, conversation, question, check, expectation)
SCENARIOS = [
    ("employees", "c1", "Hello.", lambda r: r["plan"]["type"] == "chat" and not r["calls"], "conversational answer, no tools"),
    ("employees", "c1", "What is Random Forest?", lambda r: r["plan"]["type"] == "chat" and "tree" in r["answer"].lower(),
     "general explanation"),
    ("employees", "c1", "What is the average salary?",
     lambda r: tools_of(r) == ["compute_metric"] and ok_calls(r) and
     abs(r["calls"][0]["result"]["value"] - dd.employees()["salary"].mean()) < 1e-6, "correct computed value"),
    ("employees", "c1", "Which department pays the most?",
     lambda r: "compute_metric" in tools_of(r) and ok_calls(r) and
     next(c for c in r["calls"] if c["tool"] == "compute_metric")["result"]["groups"][0]["group"] ==
     dd.employees().groupby("department")["salary"].mean().idxmax(), "grouping and comparison"),
    ("customers", "c2", "Predict customer churn.",
     lambda r: tools_of(r) == ["random_forest"] and ok_calls(r) and r["calls"][0]["result"]["task"] == "classification",
     "Random Forest classification"),
    ("employees", "c3", "Predict salary.",
     lambda r: tools_of(r) == ["random_forest"] and ok_calls(r) and r["calls"][0]["result"]["task"] == "regression",
     "Random Forest regression"),
    ("campaigns", "c4", "Which area should receive our next advertising campaign?",
     lambda r: tools_of(r) == ["recommend"] and ok_calls(r) and "### Recommendation" in r["answer"],
     "historical campaign analysis and recommendation"),
    ("campaigns", "c4", "Should we increase the advertising budget in Area A?",
     lambda r: tools_of(r) == ["recommend"] and ok_calls(r) and r["calls"][0]["args"].get("focus") == "Area A"
     and "ROI" in r["answer"], "performance, ROI, comparison, uncertainty, recommendation"),
    ("campaigns", "c4", "Why do you recommend Area A?",
     lambda r: r["plan"]["type"] == "explain_previous" and "statistical evidence" in r["answer"],
     "statistics, method and limitations of the previous recommendation"),
    ("customers", "c2", "Which factors influenced your prediction?",
     lambda r: tools_of(r) == ["feature_importance"] and ok_calls(r) and "not what causes" in r["answer"],
     "feature importance from the trained model"),
    ("campaigns", "c4", "What if we increase the budget by 20%?",
     lambda r: tools_of(r) == ["budget_scenario"] and ok_calls(r) and r["calls"][0]["args"]["change_pct"] == 20
     and "not a prediction" in r["answer"], "scenario with disclosed assumptions"),
    ("employees", "c5", "Predict churn.", lambda r: r["plan"]["type"] == "clarify", "clarification, no guess"),
    ("inventory", "c6", "Which area should receive our next advertising campaign?",
     lambda r: r["plan"]["type"] == "decline" or any(c.get("status") == "refused" for c in r["calls"]),
     "explains the data is insufficient"),
    ("stores_small", "c7", "Predict churn.",
     lambda r: any(c.get("status") == "refused" for c in r["calls"]) or r["plan"]["type"] in ("decline", "clarify"),
     "explains why a prediction cannot responsibly be made"),
]
FRAMES = {"employees": dd.employees, "customers": dd.customers, "campaigns": dd.campaigns, "inventory": dd.no_marketing,
          "stores_small": dd.tiny}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", choices=["on", "off"], default="off")
    args = ap.parse_args()
    jpath, mpath = REPORTS / f"analyst_acceptance_{args.label}.json", REPORTS / f"analyst_acceptance_{args.label}.md"
    if jpath.exists() or mpath.exists():
        print(f"refusing to overwrite {jpath.name}")
        return 1
    use_model = args.model == "on"
    if use_model and not planner.model_available():
        print(f"the analyst model is not being served at {planner.MODEL_URL}")
        return 2
    frames = {k: f() for k, f in FRAMES.items()}
    memories: dict[str, dict] = {}
    rows = []
    for i, (ds, conv, q, check, expect) in enumerate(SCENARIOS, 1):
        r = orchestrator.ask(q, frames[ds], memories.setdefault(conv, {}), use_model=use_model)
        try:
            passed = bool(check(r))
        except Exception as e:  # noqa: BLE001
            passed = False
            r.setdefault("notes", []).append(f"check raised {type(e).__name__}: {e}")
        first_pass_ok = passed and (not use_model or r["plan_source"] == "model")
        rows.append({"n": i, "dataset": ds, "question": q, "expected": expect, "passed": passed,
                     "first_pass_passed": first_pass_ok, "first_pass": r.get("first_pass"),
                     "plan_source": r["plan_source"], "plan": r["plan"], "tools": tools_of(r),
                     "statuses": [c.get("status") for c in r["calls"]], "notes": r["notes"],
                     "grounding": r["grounding"], "elapsed_ms": r["elapsed_ms"], "answer": r["answer"][:4000]})
        print(f"{i:2d} {'PASS' if passed else 'FAIL'} [{r['plan_source']}] {q}  {tools_of(r) or r['plan']['type']}")
    summary = {"label": args.label, "model": "on" if use_model else "off", "passed": sum(r["passed"] for r in rows),
               "first_pass_passed": sum(r["first_pass_passed"] for r in rows),
               "total": len(rows), "model_planned": sum(r["plan_source"] == "model" for r in rows), "rows": rows}
    REPORTS.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    lines = [f"# Analyst acceptance {args.label} (model {summary['model']})", "",
             f"Passed {summary['passed']}/{summary['total']} (first pass, model plan only: {summary['first_pass_passed']}/{summary['total']}); "
             f"plans written by the model: {summary['model_planned']}.", "",
             "| # | question | expected | planner | tools / plan | result |", "|---:|---|---|---|---|---|"]
    lines += [f"| {r['n']} | {r['question']} | {r['expected']} | {r['plan_source']} | "
              f"{', '.join(r['tools']) or r['plan']['type']} | {'pass' if r['passed'] else '**fail**'} |" for r in rows]
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{summary['passed']}/{summary['total']} passed ({summary['first_pass_passed']} on the model's first pass); wrote {jpath.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
