"""Does the model-planned path actually execute the right tools and give
grounded answers? Runs questions end to end (model plan -> validation ->
deterministic tools -> answer) on invented datasets, including the two domains
the model never saw in training, and the business questions on campaign data.

    cd backend
    ../.venv313/bin/python evals/analyst_model_execution.py --label NAME

Needs the analyst model being served (training/scripts/serve_analyst_model.sh).
For reference, each question is also answered by the deterministic planner.
Writes backend/evals/reports/analyst_model_execution_<label>.{json,md}; refuses to overwrite.

A question passes when: the plan came from the model (no fallback), every tool
call passed validation and ran, the expected tool ran, and every number in the
final answer occurs in the tool results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "training/scripts"))
from app.analyst import demo_data as dd, explain, orchestrator, planner  # noqa: E402

REPORTS = ROOT / "evals/reports"


def unseen_frames():
    import build_analyst_sft as b
    rng = np.random.default_rng(99)
    b.R.seed(99)
    edu, edu_meta = b.domain_frame("education", rng, n=120)
    est, est_meta = b.domain_frame("real_estate", rng, n=120)
    return (edu, edu_meta), (est, est_meta)


def questions():
    (edu, em), (est, rm) = unseen_frames()
    K, E, C = dd.campaigns(), dd.employees(), dd.customers()
    n = lambda c: c.replace("_", " ")
    e_score, e_group, e_cls = em["measure"][0], em["group"][0], em["cls_target"]
    r_price, r_city, r_cls = rm["measure"][0], rm["group"][0], rm["cls_target"]
    # (group, dataset name, frame, conversation, question, acceptable tool sets)
    return [
        ("unseen_education", "education", edu, "u1", f"What is the average {n(e_score)}?", [["compute_metric"]]),
        ("unseen_education", "education", edu, "u1", f"Which {n(e_group)} has the highest average {n(e_score)}?",
         [["compute_metric", "compare_groups"], ["compute_metric"]]),
        ("unseen_education", "education", edu, "u1", f"Is there a significant difference in {n(e_score)} between {n(e_group)} groups?",
         [["compare_groups"]]),
        ("unseen_education", "education", edu, "u1", f"Predict {n(e_cls)}", [["random_forest"]]),
        ("unseen_education", "education", edu, "u1", "Which factors influenced your prediction?", [["feature_importance"]]),
        ("unseen_real_estate", "real_estate", est, "u2", f"What is the median {n(r_price)} in each {n(r_city)}?", [["compute_metric"]]),
        ("unseen_real_estate", "real_estate", est, "u2", f"Predict {n(r_price)}", [["random_forest"]]),
        ("unseen_real_estate", "real_estate", est, "u2", f"Can you predict which properties will {rm['class_word']}?", [["random_forest"]]),
        ("unseen_real_estate", "real_estate", est, "u2", "Which area should receive our next advertising campaign?", [[]]),
        ("aggregation_filter", "employees", E, "e1", "What is the average salary in Engineering?", [["compute_metric"]]),
        ("aggregation_filter", "employees", E, "e1", "How many employees have a salary above 100000?", [["compute_metric"]]),
        ("aggregation_filter", "employees", E, "e1", "Show the total salary by department", [["compute_metric"]]),
        ("statistics", "employees", E, "e1", "Is years of experience correlated with salary?", [["correlation"]]),
        ("statistics", "employees", E, "e1", "What is the standard deviation of salary?", [["describe_column"]]),
        ("prediction", "customers", C, "p1", "Predict customer churn.", [["random_forest"]]),
        ("prediction", "customers", C, "p1", "Which factors influenced your prediction?", [["feature_importance"]]),
        ("prediction", "employees", E, "p2", "Predict salary.", [["random_forest"]]),
        ("business", "campaigns", K, "b1", "Should we advertise in Area A?", [["recommend"]]),
        ("business", "campaigns", K, "b1", "Which area should receive the next campaign?", [["recommend"]]),
        ("business", "campaigns", K, "b1", "Why do you recommend Area A?", [[]]),
        ("business", "campaigns", K, "b1", "Should we increase the budget?", [["recommend"]]),
        ("business", "campaigns", K, "b1", "What happens if the budget increases by 20%?", [["budget_scenario"]]),
        ("business", "campaigns", K, "b1", "Is the difference between these areas statistically significant?",
         [["compare_groups"], ["business_performance"], ["recommend"]]),
        ("business", "campaigns", K, "b1", "Show ROI and CAC by area", [["business_performance"]]),
        ("business", "inventory", dd.no_marketing(), "b2", "Should we advertise in the Tools category?", [[]]),
    ]


def run(use_model: bool):
    memories: dict = {}
    out = []
    for group, ds, df, conv, q, expected in questions():
        r = orchestrator.ask(q, df, memories.setdefault(conv, {}), use_model=use_model)
        tools_run = [c.get("tool") for c in r["calls"]]
        results = [c.get("result") for c in r["calls"] if c.get("status") == "ok"]
        evidence = results + ([r["plan"]] if r["plan"]["type"] in ("clarify", "decline") else [])
        if r["plan"]["type"] == "explain_previous":
            evidence.append(memories[conv].get("last_recommendation") or memories[conv].get("model", {}).get("importance"))
        bad = explain.ungrounded(r["answer"], orchestrator._jsonable(evidence), q)
        all_ok = all(c.get("status") in ("ok", "refused") for c in r["calls"])
        tool_ok = tools_run in expected or (expected == [[]] and not any(c.get("status") == "ok" for c in r["calls"]))
        out.append({"group": group, "dataset": ds, "question": q, "plan_source": r["plan_source"], "plan_type": r["plan"]["type"],
                    "tools": tools_run, "statuses": [c.get("status") for c in r["calls"]], "expected": expected,
                    "tool_ok": tool_ok, "calls_ok": all_ok, "ungrounded": bad, "grounded": not bad,
                    "model_summary": r["answer"].split("\n\n---\n\n")[0] if "\n\n---\n\n" in r["answer"] else None,
                    "notes": r["notes"], "elapsed_ms": r["elapsed_ms"], "answer": r["answer"][:3000]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    jpath, mpath = REPORTS / f"analyst_model_execution_{args.label}.json", REPORTS / f"analyst_model_execution_{args.label}.md"
    if jpath.exists() or mpath.exists():
        print(f"refusing to overwrite {jpath.name}")
        return 1
    if not planner.model_available():
        print(f"the analyst model is not being served at {planner.MODEL_URL}")
        return 2
    model, rules = run(True), run(False)
    for m in model:
        m["passed"] = m["plan_source"] == "model" and m["tool_ok"] and m["calls_ok"] and m["grounded"]
    for r in rules:
        r["passed"] = r["tool_ok"] and r["calls_ok"] and r["grounded"]
    by_group = {}
    for m in model:
        g = by_group.setdefault(m["group"], {"n": 0, "passed": 0})
        g["n"] += 1
        g["passed"] += m["passed"]
    summary = {"label": args.label, "questions": len(model), "model_passed": sum(m["passed"] for m in model),
               "model_planned": sum(m["plan_source"] == "model" for m in model),
               "tool_executions": sum(len(m["tools"]) for m in model),
               "tool_executions_ok": sum(s == "ok" for m in model for s in m["statuses"]),
               "answers_grounded": sum(m["grounded"] for m in model),
               "rules_passed": sum(r["passed"] for r in rules), "by_group": by_group, "model": model, "rules": rules}
    REPORTS.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    lines = [f"# Analyst model execution {args.label}", "",
             f"Model path: {summary['model_passed']}/{summary['questions']} passed; plans written by the model "
             f"{summary['model_planned']}/{summary['questions']}; tool executions ok {summary['tool_executions_ok']}/"
             f"{summary['tool_executions']}; answers with every number grounded {summary['answers_grounded']}/{summary['questions']}. "
             f"Deterministic planner on the same questions: {summary['rules_passed']}/{summary['questions']}.", "",
             "| group | question | model plan | tools run | grounded | pass | rules pass |", "|---|---|---|---|---|---|---|"]
    for m, r in zip(model, rules):
        lines.append(f"| {m['group']} | {m['question']} | {m['plan_source']}:{m['plan_type']} | {', '.join(m['tools']) or '-'} "
                     f"| {'yes' if m['grounded'] else 'NO ' + ','.join(m['ungrounded'][:3])} | {'pass' if m['passed'] else '**fail**'} "
                     f"| {'pass' if r['passed'] else 'fail'} |")
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"model {summary['model_passed']}/{summary['questions']}, rules {summary['rules_passed']}/{summary['questions']}; wrote {jpath.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
