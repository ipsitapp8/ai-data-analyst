"""Reliability evaluation of the analyst Copilot on the demo datasets, with the model on or off.

Every expected value is computed here with pandas from the data, independently of Silt. Each question is classified:

  first_pass        correct answer, no plan was rejected or repaired on the way
  repaired          correct answer after Silt rejected or repaired part of a plan
  correct_refusal   a clarification / refusal where one was the right answer (no number shown)
  incorrect_refusal a clarification / refusal where the data could answer it
  incorrect_answer  a number was shown that is not the expected one (or an answer where a refusal was right)
  tool_failure      an exception or an explicit tool error
  ml_failure        an ML request whose result lacks a held-out evaluation against a baseline

Usage (from backend/):  python evals/analyst_reliability.py --model off|on --out PATH.json
With --model on the local v1 server (127.0.0.1:8080) must be running; model calls are counted per question.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analyst import data, orchestrator, planner  # noqa: E402
from app.analyst.explain import f as fmt  # noqa: E402

UPLOADS = Path.home() / "silt-demo" / "data" / "uploads"
FILES = {"timesheet": "32b0ebf899cb4117a431f69461e95d30_timesheet_test.csv",
         "campaigns": "7336cedab94640e2bbd0c64a34c173fd_campaigns.csv",
         "sales": "c23f6c17351a4132a4c4602c9a49131a_sales.csv",
         "shift_costs": "ec5205647bf940bbb99bf715ed5a7e31_shift_costs.csv",
         "inventory": "e04e3e4940774ed1ac094ca9debf7965_inventory.csv",
         "hr": "e8da81ee9d084cfba856960a374c1fd2_hr_attrition.csv"}
MODEL_LOG = Path.home() / "silt-demo" / "model_server.log"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _h(t: str) -> float:
    a, b = t.split(":")
    return int(a) + int(b) / 60


def truths(d: dict[str, pd.DataFrame]) -> list[dict]:
    ts, camp, sales = d["timesheet"], d["campaigns"], d["sales"]
    ts_h = ts.assign(h=[_h(b) - _h(a) for a, b in zip(ts.clock_in, ts.clock_out)])
    valid = ts_h[(ts_h.h >= 0) & (ts_h.hourly_rate >= 0)]
    cost = valid.h * valid.hourly_rate
    rahul = valid[valid.employee_name == "Rahul Sharma"]
    rahul_dedup = rahul.drop_duplicates(subset=[c for c in ts.columns])
    by_dept = cost.groupby(valid.department).sum()
    ot = (valid.h - 8).clip(lower=0).groupby(valid.employee_name).sum()
    c = camp.groupby("channel")[["conversions", "leads", "revenue", "ad_spend"]].sum()
    a = camp.groupby("area")[["conversions", "ad_spend", "revenue"]].sum()
    s = sales.groupby("region")[["conversions", "leads", "sales"]].sum()
    sc = d["shift_costs"]
    inv = d["inventory"]
    hr = d["hr"]
    att = hr.assign(y=hr.attrition.eq("Yes")).groupby("department").y.mean()
    day = valid.drop_duplicates(subset=list(ts.columns)).assign(c=lambda x: x.h * x.hourly_rate)
    day = day[pd.to_datetime(day.date) <= pd.Timestamp.today()]
    per_day = day.groupby("date").c.sum()
    complete = per_day[day.groupby("date").employee_id.nunique() >= 0.75 * day.groupby("date").employee_id.nunique().median()]
    cases = [
        # ---- timesheet: derived metrics, scope, follow-ups
        {"ds": "timesheet", "q": "What is Rahul Sharma's total shift duration?", "expect": "value",
         "values": [fmt(float((ts_h[ts_h.employee_name == "Rahul Sharma"].h).sum()))]},
        {"ds": "timesheet", "q": "Total hours worked by each employee", "expect": "value",
         "values": [fmt(float(valid.h.groupby(ts_h.employee_name).sum().max()))]},
        {"ds": "timesheet", "q": "What is the total labor cost by department?", "expect": "value",
         "values": [fmt(float(by_dept.max())), fmt(float(by_dept.sum()))], "conv": "ts1"},
        {"ds": "timesheet", "q": "Only Rahul.", "expect": "value",
         "values": [fmt(float((rahul.h * rahul.hourly_rate).sum()))], "conv": "ts1"},
        {"ds": "timesheet", "q": "Exclude duplicates.", "expect": "value",
         "values": [fmt(float((rahul_dedup.h * rahul_dedup.hourly_rate).sum()))], "conv": "ts1"},
        {"ds": "timesheet", "q": "What if the rate increases by 10%?", "expect": "value",
         "values": [fmt(float((rahul_dedup.h * rahul_dedup.hourly_rate).sum() * 1.1))], "conv": "ts1"},
        {"ds": "timesheet", "q": "How much overtime beyond 8 hours did each employee work?", "expect": "value",
         "values": [fmt(float(ot.max()))]},
        {"ds": "timesheet", "q": "How much overtime did each employee work?", "expect": "refuse"},
        {"ds": "timesheet", "q": "Total labor cost excluding Rahul Sharma", "expect": "value",
         "values": [fmt(float(cost[valid.employee_name != "Rahul Sharma"].sum()))]},
        {"ds": "timesheet", "q": "Predict the next day's total labor cost across all employees.", "expect": "value",
         "values": [fmt(float(complete.mean()))]},
        {"ds": "timesheet", "q": "Labor cost by department excluding cancelled shifts", "expect": "refuse"},
        {"ds": "timesheet", "q": "idqjnfiW", "expect": "refuse"},
        # ---- campaigns: ratio metrics
        {"ds": "campaigns", "q": "What is the conversion rate by channel?", "expect": "value",
         "values": [pct(float((c.conversions / c.leads).max()))]},
        {"ds": "campaigns", "q": "Which channel has the highest ROI?", "expect": "value",
         "values": [pct(float(((c.revenue - c.ad_spend) / c.ad_spend).max()))]},
        {"ds": "campaigns", "q": "What is the profit margin for each channel?", "expect": "value",
         "values": [pct(float(((c.revenue - c.ad_spend) / c.revenue).max()))]},
        {"ds": "campaigns", "q": "Cost per conversion by area", "expect": "value",
         "values": [fmt(float((a.ad_spend / a.conversions).max()))]},
        {"ds": "campaigns", "q": "Profit by area (revenue minus ad spend)", "expect": "value",
         "values": [fmt(float((a.revenue - a.ad_spend).max()))]},
        {"ds": "campaigns", "q": "Return on ad spend by channel", "expect": "value",
         "values": [fmt(float((c.revenue / c.ad_spend).max()))]},
        # ---- sales
        {"ds": "sales", "q": "Average order value by region", "expect": "refuse"},
        {"ds": "sales", "q": "Total sales by region", "expect": "value", "values": [fmt(float(s.sales.max()))]},
        {"ds": "sales", "q": "What is the conversion rate for each region?", "expect": "value",
         "values": [pct(float((s.conversions / s.leads).max()))]},
        {"ds": "sales", "q": "Top 3 regions by sales", "expect": "value", "values": [fmt(float(s.sales.max()))]},
        {"ds": "sales", "q": "Which product line has the most records?", "expect": "value",
         "values": [str(int(sales.product_line.value_counts().max()))]},
        # ---- other schemas
        {"ds": "shift_costs", "q": "Total labor cost by team", "expect": "value",
         "values": [fmt(float((sc.hours * sc.rate).groupby(sc.team).sum().max()))]},
        {"ds": "inventory", "q": "What is the total stock value (stock times unit price) by category?", "expect": "value",
         "values": [fmt(float((inv.stock * inv.unit_price).groupby(inv.category).sum().max()))]},
        {"ds": "hr", "q": "What is the attrition rate by department?", "expect": "value", "values": [pct(float(att.max()))]},
        {"ds": "hr", "q": "Train a model to predict attrition.", "expect": "ml"},
    ]
    return cases


def classify(case: dict, out: dict | None, err: str | None) -> tuple[str, str]:
    if err:
        return "tool_failure", err.splitlines()[-1][:200]
    ans = out["answer"]
    ptype = (out.get("plan") or {}).get("type")
    refused = ptype in ("clarify", "decline", "rephrase", "error") or out.get("plan_source") in ("input_check",
                                                                                               "model_calc_failed")
    if ptype == "error" or re.search(r"could not be run|calculation of .* failed|tool error", ans, re.I):
        return "tool_failure", ans[:200]
    if case["expect"] == "ml":
        r = next((c.get("result") for c in out.get("calls", []) if c.get("tool") == "random_forest"
                  and c.get("status") == "ok"), None)
        ok = r and r.get("metrics") and "beats_baseline" in r and (r["metrics"].get("cross_validation") or r.get("test_rows"))
        return ("first_pass" if ok else "ml_failure"), "" if ok else ans[:200]
    if case["expect"] == "refuse":
        return ("correct_refusal", "") if refused else ("incorrect_answer", "answered instead of asking: " + ans[:160])
    if refused:
        return "incorrect_refusal", ans[:200]
    missing = [v for v in case["values"] if v not in ans]
    if missing:
        return "incorrect_answer", f"expected {missing} in: " + re.sub(r"\s+", " ", ans)[:220]
    fp = out.get("first_pass") or {}
    repaired = fp.get("valid") is False or any(re.search(r"corrected|Validation rejected|rewrite|retry", n, re.I)
                                               for n in out.get("notes", []))
    return ("repaired" if repaired else "first_pass"), ""


def model_calls() -> int:
    try:
        return MODEL_LOG.read_text(errors="ignore").count("POST /v1/completions") + \
            MODEL_LOG.read_text(errors="ignore").count("POST /v1/chat/completions")
    except OSError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["on", "off"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    use = args.model == "on"
    if not use:
        planner.model_available = lambda: False
    elif not planner.model_available():
        sys.exit("the local model server is not reachable")
    frames = {k: data.load_table(str(UPLOADS / v)) for k, v in FILES.items()}
    rows, mems = [], {}
    for case in truths(frames):
        mem = mems.setdefault(case.get("conv") or id(case), {})
        before, t0 = model_calls(), time.monotonic()
        out, err = None, None
        try:
            out = orchestrator.ask(case["q"], frames[case["ds"]], mem, use_model=use)
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
        cls, why = classify(case, out, err)
        rows.append({"dataset": case["ds"], "question": case["q"], "expect": case["expect"], "outcome": cls,
                     "detail": why, "plan_source": out and out.get("plan_source"),
                     "model_calls": model_calls() - before if use else 0,
                     "seconds": round(time.monotonic() - t0, 2)})
        print(f"{cls:18s} {case['ds']:11s} {case['q'][:70]}  [{out and out.get('plan_source')}] {why[:120]}", flush=True)
    summary = {}
    for r in rows:
        summary[r["outcome"]] = summary.get(r["outcome"], 0) + 1
    report = {"model": "v1 (Qwen3-4B + Silt LoRA v1)" if use else "offline", "questions": len(rows), "summary": summary,
              "model_calls_total": sum(r["model_calls"] for r in rows), "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(summary), "model calls:", report["model_calls_total"])


if __name__ == "__main__":
    main()
