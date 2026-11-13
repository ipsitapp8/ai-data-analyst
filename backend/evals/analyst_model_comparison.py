"""Base model vs fine-tuned model on the 25 frozen execution questions.

The questions and the pass rule are imported unchanged from
evals/analyst_model_execution.py, whose deterministic-planner result (20/25) is
frozen in evals/reports/analyst_execution_baseline_rules_freeze.json.

    cd backend
    # with the BASE model served (serve_analyst_model.sh with ANALYST_ADAPTER unset -> see --help there):
    ../.venv313/bin/python evals/analyst_model_comparison.py run --model-label base --adapter none
    # with the FINE-TUNED adapter served:
    ../.venv313/bin/python evals/analyst_model_comparison.py run --model-label fine_tuned --adapter training/adapters/analyst_qwen3_4b_lora
    ../.venv313/bin/python evals/analyst_model_comparison.py compare

For every question it records: the model's raw reply (interpretation), the
parsed plan, backend validation of each call, the tools actually executed with
their status, and whether the final answer is correct and grounded.
Writes evals/reports/analyst_model_comparison_*.{json,md}; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))
import analyst_model_execution as frozen  # noqa: E402
from app.analyst import explain, orchestrator, planner, tools  # noqa: E402

REPORTS = ROOT / "evals/reports"
FREEZE = REPORTS / "analyst_execution_baseline_rules_freeze.json"


def _key_result(call: dict):
    r = call.get("result") or {}
    if call.get("tool") == "compute_metric":
        return {"value": r.get("value"), "top": r.get("top"), "rows": r.get("rows")}
    if call.get("tool") == "random_forest":
        return {"task": r.get("task"), "target": r.get("target"), "metrics": r.get("metrics")}
    if call.get("tool") == "recommend":
        return (r.get("recommendation") or {}).get("recommendation")
    if call.get("tool") in ("compare_groups",):
        return r.get("test")
    if call.get("tool") == "correlation":
        return {"pearson_r": r.get("pearson_r"), "pearson_p": r.get("pearson_p")}
    if call.get("tool") == "business_performance":
        return {"ranking": r.get("ranking"), "focus": r.get("focus")}
    return None if call.get("status") == "ok" else call.get("reason")


def run(label: str, adapter: str) -> int:
    path = REPORTS / f"analyst_model_comparison_{label}.json"
    if path.exists():
        print(f"refusing to overwrite {path.name}")
        return 1
    if not planner.model_available():
        print(f"no model served at {planner.MODEL_URL}")
        return 2
    memories: dict = {}
    rows = []
    for i, (group, ds, df, conv, q, expected) in enumerate(frozen.questions(), 1):
        mem = memories.setdefault(conv, {})
        raw = planner.model_chat(planner.build_prompt(q, df, mem))          # 1. interpretation
        plan = planner.parse_plan(raw)                                        # 3. structured plan
        validation = []                                                       # 4. backend validation
        if plan and plan["type"] == "tools":
            for c in plan["calls"]:
                try:
                    tools.validate_call(c, df, q)
                    validation.append({"tool": c.get("tool"), "valid": True})
                except tools.InvalidCall as e:
                    validation.append({"tool": c.get("tool") if isinstance(c, dict) else None, "valid": False, "reason": str(e)})
        r = orchestrator.ask(q, df, mem, use_model=True)                      # 5. execution and answer
        tools_run = [c.get("tool") for c in r["calls"]]
        results = [c.get("result") for c in r["calls"] if c.get("status") == "ok"]
        evidence = results + ([r["plan"]] if r["plan"]["type"] in ("clarify", "decline") else [])
        if r["plan"]["type"] == "explain_previous":
            evidence.append(mem.get("last_recommendation") or (mem.get("model") or {}).get("importance"))
        bad = explain.ungrounded(r["answer"], orchestrator._jsonable(evidence), q)
        tool_ok = tools_run in expected or (expected == [[]] and not any(c.get("status") == "ok" for c in r["calls"]))
        calls_ok = all(c.get("status") in ("ok", "refused") for c in r["calls"])
        passed = r["plan_source"] == "model" and tool_ok and calls_ok and not bad      # 6. correctness (frozen rule): first pass only
        passed_after_retry = r["plan_source"] in ("model", "model_retry") and tool_ok and calls_ok and not bad
        rows.append({"n": i, "group": group, "question": q, "raw_reply": raw[:800], "parsed_plan": plan,
                     "validation": validation, "used_plan_source": r["plan_source"], "fallback_notes": r["notes"],
                     "executed": [{"tool": c.get("tool"), "args": c.get("args"), "status": c.get("status"),
                                   "key_result": _key_result(c)} for c in r["calls"]],
                     "expected_tools": expected, "tool_ok": tool_ok, "calls_ok": calls_ok, "ungrounded": bad,
                     "passed": passed, "passed_after_retry": passed_after_retry, "first_pass": r.get("first_pass"),
                     "answer": r["answer"][:2500]})
        print(f"{i:2d} {'PASS' if passed else 'FAIL'} [{r['plan_source']}] {q} -> {tools_run or r['plan']['type']}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT.parent / "training/scripts"))
    import inference_record
    inference = inference_record.record(adapter, {
        "server": "mlx_lm.server (OpenAI-compatible), request model 'default_model'",
        "plan": {"temperature": 0.0, "max_tokens": 700}, "narration": {"temperature": 0.0, "max_tokens": 350},
        "client": "app.analyst.planner.model_chat"})
    path.write_text(json.dumps({"model_label": label, "model_url": planner.MODEL_URL, "inference": inference,
                                "passed_after_retry": sum(r["passed_after_retry"] for r in rows),
                                "passed": sum(r["passed"] for r in rows), "total": len(rows), "rows": rows},
                               indent=1, default=str) + "\n", encoding="utf-8")
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)}; wrote {path.name}")
    return 0


def compare(labels: list[str], out_name: str) -> int:
    out_j, out_m = REPORTS / f"analyst_model_comparison_{out_name}.json", REPORTS / f"analyst_model_comparison_{out_name}.md"
    if out_j.exists() or out_m.exists():
        print(f"refusing to overwrite {out_j.name}")
        return 1
    runs = {lb: json.loads((REPORTS / f"analyst_model_comparison_{lb}.json").read_text()) for lb in labels}
    rows = {lb: {r["n"]: r for r in run["rows"]} for lb, run in runs.items()}
    rules = json.loads(FREEZE.read_text())
    rule_pass = {o["n"]: o["outcome"] == "pass" for o in rules["outcomes"]}
    missed = [n for n, ok in rule_pass.items() if not ok]
    passed = [n for n, ok in rule_pass.items() if ok]
    after = lambda r: r.get("passed_after_retry", r["passed"])
    summary = {"rules_frozen": rules["result"], "models": {
        lb: {"first_pass": sum(r["passed"] for r in rows[lb].values()), "after_retry": sum(after(r) for r in rows[lb].values()),
             "total": len(rows[lb]), "fixes_rule_misses_first_pass": sum(rows[lb][n]["passed"] for n in missed),
             "regressions_on_rule_passes_first_pass": [n for n in passed if not rows[lb][n]["passed"]],
             "regressions_on_rule_passes_after_retry": [n for n in passed if not after(rows[lb][n])],
             "inference": runs[lb].get("inference")} for lb in labels}}
    out_j.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")

    def desc(r):
        plan = r["parsed_plan"]
        if plan is None:
            return "unparseable reply"
        if plan["type"] == "tools":
            return "; ".join(f"{c.get('tool')}({json.dumps(c.get('args'), default=str)})" for c in plan["calls"])
        return plan["type"] + (f": {plan.get('options')}" if plan["type"] == "clarify" else "")

    lines = [f"# Execution comparison: {', '.join(labels)}", "",
             f"Deterministic planner (frozen): {rules['result']['passed']}/{rules['result']['total']}.", "",
             "| model | first pass | after one reconsideration | fixes rule misses (first pass) | regressions on rule passes (first pass / after retry) |",
             "|---|---:|---:|---:|---|"]
    for lb, m in summary["models"].items():
        lines.append(f"| {lb} | {m['first_pass']}/{m['total']} | {m['after_retry']}/{m['total']} | {m['fixes_rule_misses_first_pass']}/{len(missed)} "
                     f"| {m['regressions_on_rule_passes_first_pass'] or 'none'} / {m['regressions_on_rule_passes_after_retry'] or 'none'} |")
    lines += ["", "## Per question (first pass; R = passed only after reconsideration)", "",
              "| # | question | rules | " + " | ".join(labels) + " |", "|---:|---|---|" + "---|" * len(labels)]
    for n in sorted(rule_pass):
        cells = []
        for lb in labels:
            r = rows[lb][n]
            cells.append("pass" if r["passed"] else ("R" if after(r) else f"**fail** ({desc(r)[:60]})"))
        lines.append(f"| {n} | {rows[labels[0]][n]['question']} | {'pass' if rule_pass[n] else 'fail'} | " + " | ".join(cells) + " |")
    out_m.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({lb: {k: v for k, v in m.items() if k != "inference"} for lb, m in summary["models"].items()}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-label", required=True, help="base, fine_tuned (v1), or a new label such as base_postfix, v1_postfix, v2")
    r.add_argument("--adapter", required=True, help="the adapter the server was started with, or 'none'")
    c = sub.add_parser("compare")
    c.add_argument("--labels", nargs="+", required=True)
    c.add_argument("--out", required=True, help="name of the summary report")
    a = ap.parse_args()
    return run(a.model_label, a.adapter) if a.cmd == "run" else compare(a.labels, a.out)


if __name__ == "__main__":
    sys.exit(main())
