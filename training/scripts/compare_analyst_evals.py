"""Base vs v1 vs v2 on the held-out analyst test sample, all scored by the same
(post-fix) checker. Reads existing reports only; never re-runs a model.

    .venv313/bin/python training/scripts/compare_analyst_evals.py

Inputs:  training/reports/analyst_finetune_eval_v1_postfix.json  (base + v1, post-fix checker)
         training/reports/analyst_finetune_eval_v2.json          (base + v2, post-fix checker)
Writes:  training/reports/analyst_finetune_comparison_postfix.{json,md}; refuses to overwrite.
The frozen v1 report (analyst_finetune_eval_v1.*) is not read or changed.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "training/reports"
_spec = importlib.util.spec_from_file_location("ev", ROOT / "training/scripts/eval_analyst_model.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
from app.analyst import planner  # noqa: E402  (path set by eval_analyst_model)

ADS = re.compile(r"advertis|\bads?\b|campaign|marketing|budget|spend|roi\b|\bcac\b", re.I)


def plan_of(reply: str):
    try:
        return planner.parse_plan(reply)
    except Exception:  # noqa: BLE001
        return None


def focused(examples, rows):
    """Behaviour-level rates the user asked for, from the per-example replies."""
    acc = {k: [] for k in ("regression_direction", "decline_when_ad_columns_absent", "recommend_when_ad_columns_present",
                           "prediction_target_and_task", "explain_prediction", "clarification")}
    for ex, r in zip(examples, rows):
        if ex["intent"].startswith("narrate"):
            continue
        ref = json.loads(ex["messages"][-1]["content"])
        q = ex["messages"][1]["content"].rsplit("Question:", 1)[-1]
        got = plan_of(r["reply"]) or {}
        calls = got.get("calls") or []
        it = ex["intent"]
        if it == "regression":
            ra = ref["calls"][0]["args"]
            ga = calls[0].get("args", {}) if calls and calls[0].get("tool") == "linear_regression" else {}
            acc["regression_direction"].append(ga.get("target") == ra["target"] and
                                               list(ga.get("features") or []) == list(ra["features"]))
        elif it == "decline" and ADS.search(q):
            acc["decline_when_ad_columns_absent"].append(got.get("type") == "decline")
        elif it in ("recommendation", "performance"):
            acc["recommend_when_ad_columns_present"].append(bool(r.get("args_ok")))
        elif it == "prediction":
            acc["prediction_target_and_task"].append(bool(r.get("args_ok")))
        elif it == "explain_prediction":
            acc["explain_prediction"].append(bool(r.get("tools_ok")))
        elif it == "clarify":
            acc["clarification"].append(got.get("type") == "clarify")
    return {k: {"rate": round(sum(v) / len(v), 4) if v else None, "n": len(v), "correct": sum(v)} for k, v in acc.items()}


def main() -> int:
    out_j, out_m = REPORTS / "analyst_finetune_comparison_postfix.json", REPORTS / "analyst_finetune_comparison_postfix.md"
    if out_j.exists() or out_m.exists():
        print("refusing to overwrite", out_j.name)
        return 1
    r1 = json.loads((REPORTS / "analyst_finetune_eval_v1_postfix.json").read_text())
    r2 = json.loads((REPORTS / "analyst_finetune_eval_v2.json").read_text())
    assert r1["test_split_sha256"] == r2["test_split_sha256"], "different test splits"
    examples = ev.sample(20)  # both runs used --per-intent 20 with the default seed
    assert len(examples) == len(r1["models"]["base"]["rows"]) == len(r2["models"]["fine_tuned"]["rows"])
    base_same = [a["reply"] for a in r1["models"]["base"]["rows"]] == [b["reply"] for b in r2["models"]["base"]["rows"]]
    models = {"base": r1["models"]["base"], "v1": r1["models"]["fine_tuned"], "v2": r2["models"]["fine_tuned"]}
    table = {}
    for name, m in models.items():
        table[name] = {"measures": {k: v["rate"] for k, v in m["measures"].items()},
                       "n": {k: v["n"] for k, v in m["measures"].items()},
                       "unseen": {d: {k: v["rate"] for k, v in ms.items()} for d, ms in m["measures_unseen"].items()},
                       "focused": focused(examples, m["rows"])}
    data = {"test_split_sha256": r1["test_split_sha256"], "examples": len(examples),
            "base_replies_identical_across_runs": base_same,
            "checker": "post-fix (precision-aware grounding; validation receives the question)",
            "adapters": {"v1": r1["adapter"], "v2": r2["adapter"]}, "inference": {"v1_run": r1["inference"], "v2_run": r2["inference"]},
            "models": table}
    out_j.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")

    def f(x):
        return "–" if x is None else f"{x:.3f}"
    L = ["# Analyst planner held-out comparison: base vs v1 vs v2 (same post-fix checker)", "",
         f"Test split sha256 `{r1['test_split_sha256'][:16]}…`, {len(examples)} examples, greedy decoding, identical prompts. "
         f"Base replies identical in both runs: {base_same}.", "", "## Standard measures", "",
         "| measure | n | base | v1 | v2 |", "|---|---:|---:|---:|---:|"]
    for k in table["base"]["measures"]:
        L.append(f"| {k} | {table['v2']['n'][k]} | " + " | ".join(f(table[m]['measures'][k]) for m in models) + " |")
    L += ["", "## Behaviours checked specifically", "", "| behaviour | n | base | v1 | v2 |", "|---|---:|---:|---:|---:|"]
    for k in table["base"]["focused"]:
        L.append(f"| {k} | {table['v2']['focused'][k]['n']} | " + " | ".join(
            f"{f(table[m]['focused'][k]['rate'])} ({table[m]['focused'][k]['correct']})" for m in models) + " |")
    for d in table["base"]["unseen"]:
        L += ["", f"## Unseen domain: {d}", "", "| measure | base | v1 | v2 |", "|---|---:|---:|---:|"]
        for k in table["base"]["unseen"][d]:
            if any(table[m]["unseen"][d][k] is not None for m in models):
                L.append(f"| {k} | " + " | ".join(f(table[m]['unseen'][d][k]) for m in models) + " |")
    out_m.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
