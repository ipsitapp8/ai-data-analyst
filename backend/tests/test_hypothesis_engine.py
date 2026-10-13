"""Hypothesis-driven investigation: operations, evidence rules, the reasoning
loop, its limits, and its integration with jobs, the API and evidence.

The model is scripted in every test; every analysis is really executed on
fixture data, and expected numbers are computed here independently."""
from __future__ import annotations

import copy
import csv
import io
import time

import numpy as np
import pandas as pd
import pytest
from platform_helpers import FIXTURES, SALES_CSV, ask, job_row, question_row, run_job, workspace

from app import analysis_ops as ops
from app import config, hypothesis_engine as he, jobs, runtime
from app import metric_engine as me
from app.agents import llm_client, router
from app.database import SessionLocal
from app.evals.scripted import ScriptedProvider
from app.models import InvestigationAnalysis, InvestigationHypothesis
from app.profiling import load_and_profile_csv, profile_dataframe

SPLIT = "2025-03-01"
QUESTION = "Why did revenue drop by 20% in the last two months?"
ROWS = list(csv.DictReader(io.StringIO(SALES_CSV.decode())))


def ref_sum(column: str, later: bool, **where) -> float:
    """Independent reference: plain Python over the CSV."""
    return sum(float(r[column]) for r in ROWS
               if (r["order_date"] >= SPLIT) == later and all(r[k] == v for k, v in where.items()))


REV_A, REV_B = ref_sum("revenue", False), ref_sum("revenue", True)
WEST_DELTA = ref_sum("revenue", True, region="West") - ref_sum("revenue", False, region="West")

UNDERSTANDING = {"metric_formula": "sum(revenue)", "metric_label": "Revenue", "time_column": "order_date",
                 "period_a": ["2025-01-01", SPLIT], "period_b": [SPLIT, "2025-04-29"], "population_filters": [],
                 "dimensions": ["region", "product"], "expected_direction": "decrease", "claimed_change_pct": 20,
                 "requested_outcome": "explain the drop", "assumptions": ["Periods read as Jan-Feb versus Mar-Apr."],
                 "missing_information": []}


def hyp(statement, op, params, expect, priority=3, columns=(), **extra):
    return {"statement": statement, "measurable_prediction": f"{op} shows {expect}", "analysis": {"op": op, "params": params},
            "prediction": {"expect": expect}, "required_columns": list(columns), "confounders": ["seasonality"],
            "priority": priority, **extra}


HYPOTHESES = {"hypotheses": [
    hyp("Revenue fell because the West region sold less", "segment_contribution", {"dimension": "region", "segment": "West"},
        "concentrated", 5, ["region", "revenue"]),
    hyp("Fewer units were sold overall", "component_change", {"formula": "sum(units)"}, "decrease", 4, ["units"]),
    hyp("Unit prices were cut", "distribution_shift", {"column": "unit_price"}, "decrease", 3, ["unit_price"]),
    hyp("The drop is spread across channels in proportion", "segment_contribution", {"dimension": "channel"}, "concentrated", 2, ["channel"]),
    hyp("The later period is incomplete", "period_coverage", {}, "incomplete", 4),
    hyp("Refunds increased", "share_change", {"filters": [{"column": "refund_status", "op": "==", "value": "refunded"}]},
        "increase", 4, ["refund_status"]),                                                   # column does not exist
    hyp("Fewer units were sold overall", "component_change", {"formula": "sum(units)"}, "decrease", 1),   # duplicate
    hyp("Magic", "run_python", {"code": "import os"}, "decrease", 5),                        # not an operation
    hyp("Prices moved", "distribution_shift", {"column": "unit_price"}, "concentrated", 2),  # prediction does not fit
    {"statement": "", "analysis": {}},                                                     # malformed
]}
NEXT_STEPS = [
    {"decision": "analyze", "unresolved_question": "Is the unit decline a volume effect?", "hypothesis_id": "H2",
     "analysis": {"op": "price_volume", "params": {}}, "prediction": {"expect": "volume_driven"}, "rationale": "separate price from volume"},
    {"decision": "analyze", "unresolved_question": "Which product inside West?", "hypothesis_id": "H1",
     "analysis": {"op": "segment_contribution", "params": {"dimension": "product", "filters": [{"column": "region", "op": "==", "value": "West"}]}},
     "prediction": {"expect": "concentrated"}, "rationale": "drill into the supported segment"},
    {"decision": "stop", "rationale": "Every hypothesis is resolved."},
]
GOOD_REPORT = {"executive_summary": f"Revenue fell from {REV_A:,.0f} to {REV_B:,.0f}. The data is consistent with the decline "
                                    "being located in the West region, where fewer units were sold. Prices did not fall.",
               "possible_explanations": ["Demand in the West region may have weakened."],
               "follow_up_analyses": ["Check West order counts by week."]}


def script(**overrides) -> dict:
    base = {"submit_understanding": [UNDERSTANDING], "submit_hypotheses": [HYPOTHESES],
            "submit_next_step": NEXT_STEPS, "submit_report": [GOOD_REPORT]}
    base.update(overrides)
    return copy.deepcopy(base)


@pytest.fixture(scope="module")
def sales():
    return load_and_profile_csv(str(FIXTURES / "sales.csv"))


def engine_for(df, profile, replies, question=QUESTION, **kw):
    provider = ScriptedProvider(replies)
    eng = he.HypothesisEngine(question=question, df=df, profile=profile,
                              call_model=lambda **k: provider("test", **k), **kw)
    return eng, provider


def context(df, profile, formula="sum(revenue)") -> ops.AnalysisContext:
    ctx = ops.AnalysisContext(formula=me.parse_formula(formula), label="Metric", base_filters=[], time_column="order_date",
                              period_a=["2025-01-01", SPLIT], period_b=[SPLIT, "2025-04-29"], profile=profile)
    base = ops.execute(df, ctx, {"op": "metric_change", "params": {}})
    ctx.total_a, ctx.total_b = base["facts"]["a"], base["facts"]["b"]
    return ctx


# ======================================================== operations ========

@pytest.mark.parametrize("op,code", [
    ({"op": "run_sql", "params": {"sql": "DROP TABLE users"}}, "unknown_operation"),
    ({"op": "component_change", "params": {"formula": "__import__('os').system('id')"}}, "invalid_formula"),
    ({"op": "component_change", "params": {"formula": "sum(ghost)"}}, "missing_column"),
    ({"op": "distribution_shift", "params": {"column": "region"}}, "wrong_column_type"),
    ({"op": "distribution_shift", "params": {"column": "nope"}}, "missing_column"),
    ({"op": "segment_contribution", "params": {"dimension": "order_date"}}, "wrong_column_type"),
    ({"op": "segment_contribution", "params": {"dimension": "region", "filters": [{"column": "ghost", "op": "==", "value": 1}]}}, "missing_column"),
    ({"op": "share_change", "params": {"filters": []}}, "invalid_operation"),
    ({"op": "share_change", "params": {"filters": [{"column": "region", "op": "; DROP", "value": "x"}]}}, "invalid_filter"),
    ("not even a dict", "unknown_operation"),
    ({"op": "component_change", "params": "sum(revenue)"}, "invalid_operation"),
])
def test_proposed_operations_are_validated_before_anything_runs(sales, op, code):
    df, profile = sales
    with pytest.raises(ops.OpError) as e:
        ops.normalize(op, context(df, profile))
    assert e.value.code == code


def test_operations_that_do_not_apply_to_the_metric_are_refused(sales):
    df, profile = sales
    ctx = context(df, profile, "avg(unit_price)")
    for name in ("driver_decomposition", "mix_shift"):
        with pytest.raises(ops.OpError) as e:
            ops.normalize({"op": name, "params": {"dimension": "region"}}, ctx)
        assert e.value.code == "not_applicable"


def test_signature_identifies_equivalent_analyses(sales):
    df, profile = sales
    ctx = context(df, profile)
    a = ops.normalize({"op": "segment_contribution", "params": {"dimension": "region", "segment": "West"}}, ctx)
    b = ops.normalize({"params": {"segment": "West", "dimension": "region", "junk": 1}, "op": "segment_contribution"}, ctx)
    c = ops.normalize({"op": "segment_contribution", "params": {"dimension": "region"}}, ctx)
    assert ops.signature(a) == ops.signature(b) != ops.signature(c)
    assert ops.signature(ops.normalize({"op": "component_change", "params": {"formula": " sum(units) "}}, ctx)) == \
        ops.signature(ops.normalize({"op": "component_change", "params": {"formula": "sum(units)", "filters": []}}, ctx))


@pytest.mark.parametrize("op_name,prediction", [("component_change", {"expect": "concentrated"}), ("period_coverage", {"expect": "decrease"}),
                                               ("segment_contribution", {}), ("data_quality", "issues_present"), ("metric_change", {"expect": "decrease"})])
def test_a_prediction_must_fit_its_analysis(op_name, prediction):
    with pytest.raises(ops.OpError) as e:
        ops.validate_prediction(op_name, prediction)
    assert e.value.code == "invalid_prediction"


def test_metric_change_and_segment_contribution_match_the_reference(sales):
    df, profile = sales
    ctx = context(df, profile)
    assert (ctx.total_a, ctx.total_b) == (REV_A, REV_B)
    out = ops.execute(df, ctx, ops.normalize({"op": "segment_contribution", "params": {"dimension": "region"}}, ctx))
    west = next(r for r in out["table"] if r["segment"] == "West")
    assert west["contribution"] == pytest.approx(WEST_DELTA) and out["facts"]["reconciles"]
    assert west["share_of_change"] == pytest.approx(WEST_DELTA / (REV_B - REV_A))
    assert west["base_share"] == pytest.approx(ref_sum("revenue", False, region="West") / REV_A)
    assert sum(r["contribution"] for r in out["table"]) == pytest.approx(REV_B - REV_A)


def test_drill_down_decomposes_the_sub_population_s_own_change(sales):
    df, profile = sales
    ctx = context(df, profile)
    op = ops.normalize({"op": "segment_contribution", "params": {"dimension": "product", "filters": [{"column": "region", "op": "==", "value": "West"}]}}, ctx)
    out = ops.execute(df, ctx, op)
    assert out["facts"]["total_change"] == pytest.approx(WEST_DELTA)
    assert sum(r["contribution"] for r in out["table"]) == pytest.approx(WEST_DELTA)
    for r in out["table"]:
        expected = ref_sum("revenue", True, region="West", product=r["segment"]) - ref_sum("revenue", False, region="West", product=r["segment"])
        assert r["contribution"] == pytest.approx(expected)


def test_driver_decomposition_is_an_exact_identity():
    df = pd.DataFrame({"d": pd.to_datetime(["2025-01-05"] * 4 + ["2025-02-05"] * 2), "amt": [10.0, 20.0, 30.0, 40.0, 50.0, 30.0]})
    profile = profile_dataframe(df)
    ctx = ops.AnalysisContext(formula=me.parse_formula("sum(amt)"), label="Amount", base_filters=[], time_column="d",
                              period_a=["2025-01-01", "2025-02-01"], period_b=["2025-02-01", "2025-02-28"], profile=profile,
                              total_a=100.0, total_b=80.0)
    f = ops.execute(df, ctx, ops.normalize({"op": "driver_decomposition", "params": {}}, ctx))["facts"]
    # 4 rows at 25 each -> 2 rows at 40 each: (2-4)*25 = -50 from fewer rows, (40-25)*2 = +30 from value per row
    assert (f["count_effect"], f["value_effect"], f["reconciles"]) == (-50.0, 30.0, True)
    assert f["count_effect"] + f["value_effect"] == ctx.delta == -20.0
    verdict = ops.evaluate({"op": "driver_decomposition", "params": {}}, {"facts": f, "rows_a": 4, "rows_b": 2}, {"expect": "count_driven"})
    assert verdict["status"] == ops.SUPPORTED and verdict["effect"]["value"] == pytest.approx(2.5)
    other = ops.evaluate({"op": "driver_decomposition", "params": {}}, {"facts": f, "rows_a": 4, "rows_b": 2}, {"expect": "value_driven"})
    assert other["status"] == ops.WEAKENED, "value per row moved the opposite way to the total"


def test_price_volume_reconciles_and_not_applicable_without_a_quantity_column(sales):
    df, profile = sales
    ctx = context(df, profile)
    f = ops.execute(df, ctx, {"op": "price_volume", "params": {}})["facts"]
    assert f["reconciles"] and f["price_effect"] > 0 > f["volume_effect"]
    assert f["price_effect"] + f["volume_effect"] + f["entry_exit_effect"] == pytest.approx(REV_B - REV_A)
    slim = df[["order_date", "region", "revenue"]]
    ctx2 = context(slim, profile_dataframe(slim))
    with pytest.raises(ops.OpError) as e:
        ops.execute(slim, ctx2, {"op": "price_volume", "params": {}})
    assert e.value.code == "not_applicable"


# ====================================================== evidence rules ======

def _cc(change_pct, a=100.0, rows=(200, 200)):
    b = a * (1 + change_pct / 100)
    return {"facts": {"formula": "sum(x)", "a": a, "b": b, "change": b - a, "change_pct": change_pct},
            "rows_a": rows[0], "rows_b": rows[1]}


CC = {"op": "component_change", "params": {"formula": "sum(x)"}}


def test_evidence_can_support_weaken_or_leave_inconclusive():
    assert ops.evaluate(CC, _cc(-15), {"expect": "decrease"})["status"] == ops.SUPPORTED
    opposite = ops.evaluate(CC, _cc(+15), {"expect": "decrease"})
    assert opposite["status"] == ops.WEAKENED and "other way" in opposite["reasons"][0]
    flat = ops.evaluate(CC, _cc(-0.5), {"expect": "decrease"})
    assert flat["status"] == ops.WEAKENED and "materiality" in flat["reasons"][0]
    thin = ops.evaluate(CC, _cc(-40, rows=(200, 4)), {"expect": "decrease"})
    assert thin["status"] == ops.INCONCLUSIVE and "insufficient sample" in thin["reasons"][0], "a big move on 4 rows proves nothing"
    assert ops.evaluate(CC, _cc(-15), {"expect": "decrease"})["effect"] == {"measure": "relative change, percent", "value": -15}


def _two(mean_b, sd=30.0, n=200, seed=3):
    """A real statistical result for two normal samples, attached to a component_change result."""
    from app.stats import selector

    rng = np.random.default_rng(seed)
    a, b = rng.normal(100, sd, n), rng.normal(mean_b, sd, n)
    r = selector.two_groups(a, b, what="x", units="x")
    ma, mb = float(a.mean()), float(b.mean())
    return {"facts": {"formula": "avg(x)", "a": ma, "b": mb, "change": mb - ma, "change_pct": (mb - ma) / ma * 100, "stat": r},
            "rows_a": n, "rows_b": n}


def test_statistical_method_is_selected_and_interpreted(sales):
    """The engine's inferential operations go through the selector (numerical
    correctness of the methods themselves is covered in test_statistics.py)."""
    clear = ops.evaluate(CC, _two(80), {"expect": "decrease"})
    assert clear["status"] == ops.SUPPORTED and "Welch's t-test, p < 0.001" in clear["reasons"][0]
    assert clear["effect"]["interval"][1] < 0, "the interval for the difference is carried with the verdict"
    opposite = ops.evaluate(CC, _two(80), {"expect": "increase"})
    assert opposite["status"] == ops.WEAKENED and "other way" in opposite["reasons"][0]

    noisy = ops.evaluate(CC, _two(97), {"expect": "decrease"})
    assert noisy["status"] == ops.INCONCLUSIVE and "not evidence that nothing changed" in noisy["reasons"][0], \
        "a non-significant difference is never read as 'no effect'"
    tight = ops.evaluate(CC, _two(100.0, sd=1.0, n=5000), {"expect": "decrease"})
    assert tight["status"] == ops.WEAKENED and "not compatible with the data" in tight["reasons"][0], \
        "only an interval that excludes a material move counts against the hypothesis"
    tiny_but_real = ops.evaluate(CC, _two(99.0, sd=1.0, n=5000), {"expect": "decrease"})
    assert tiny_but_real["status"] == ops.WEAKENED and "statistically distinguishable, practically small" in tiny_but_real["reasons"][0], \
        "significance alone does not support a hypothesis"

    df, profile = sales
    ctx = context(df, profile)
    avg = ops.execute(df, ctx, ops.normalize({"op": "component_change", "params": {"formula": "avg(unit_price)", "method": "t-test"}}, ctx))
    stat = avg["facts"]["stat"]
    assert stat["method"] == "welch_t" and stat["suggested_method"]["outcome"] == "accepted" and stat["test"]["df"] > 100
    total = ops.execute(df, ctx, ops.normalize({"op": "component_change", "params": {"formula": "sum(units)"}}, ctx))
    assert "stat" not in total["facts"], "a plain total has no row-level test"
    assert "No significance test applies" in ops.evaluate(CC, total, {"expect": "decrease"})["reasons"][0]
    share = ops.execute(df, ctx, ops.normalize({"op": "share_change", "params": {"filters": [{"column": "region", "op": "==", "value": "West"}]}}, ctx))
    assert share["facts"]["stat"]["method"] == "chi_square" and share["facts"]["share_a"] == pytest.approx(0.25)
    assert share["facts"]["stat"]["interval"]["method"] == "Newcombe hybrid score interval"

    few = df.groupby(df["order_date"] >= SPLIT).head(1)                 # one row per period
    unavailable = ops.execute(few, context(few, profile), ops.normalize({"op": "distribution_shift", "params": {"column": "units"}}, ctx))
    verdict = ops.evaluate({"op": "distribution_shift", "params": {}}, unavailable, {"expect": "decrease"})
    assert unavailable["facts"]["stat"]["status"] == "unavailable" and unavailable["facts"]["stat"]["test"] is None
    assert verdict["status"] == ops.INCONCLUSIVE and "at least 2 values" in verdict["reasons"][0], "too little data is not evidence against"

    a = ops.normalize({"op": "distribution_shift", "params": {"column": "units", "method": "welch"}}, ctx)
    b = ops.normalize({"op": "distribution_shift", "params": {"column": "units", "method": "mann-whitney"}}, ctx)
    assert ops.signature(a) == ops.signature(b), "a suggested method does not make a different analysis"


def test_concentration_is_judged_against_the_segment_s_size(sales):
    df, profile = sales
    ctx = context(df, profile)
    by_region = ops.execute(df, ctx, ops.normalize({"op": "segment_contribution", "params": {"dimension": "region"}}, ctx))
    v = ops.evaluate({"op": "segment_contribution", "params": {}}, by_region, {"expect": "concentrated"})
    west_base = ref_sum("revenue", False, region="West") / REV_A
    assert v["status"] == ops.SUPPORTED and "West" in v["reasons"][0] and f"{west_base:.0%} of the metric" in v["reasons"][0]
    by_channel = ops.execute(df, ctx, ops.normalize({"op": "segment_contribution", "params": {"dimension": "channel"}}, ctx))
    v = ops.evaluate({"op": "segment_contribution", "params": {}}, by_channel, {"expect": "concentrated"})
    assert v["status"] == ops.WEAKENED and "in proportion" in v["reasons"][0], "half the base, half the change: not a finding"
    named = ops.execute(df, ctx, ops.normalize({"op": "segment_contribution", "params": {"dimension": "region", "segment": "East"}}, ctx))
    assert ops.evaluate({"op": "segment_contribution", "params": {}}, named, {"expect": "concentrated"})["status"] == ops.WEAKENED
    ghost = ops.execute(df, ctx, ops.normalize({"op": "segment_contribution", "params": {"dimension": "region", "segment": "Atlantis"}}, ctx))
    assert "not among the contributors" in ops.evaluate({"op": "segment_contribution", "params": {}}, ghost, {"expect": "concentrated"})["reasons"][0]


def test_incomplete_period_and_data_quality_problems_are_detected(sales):
    df, profile = sales
    ok = ops.execute(df, context(df, profile), {"op": "period_coverage", "params": {}})
    assert ok["facts"]["incomplete"] is False and ops.evaluate({"op": "period_coverage", "params": {}}, ok, {"expect": "incomplete"})["status"] == ops.WEAKENED

    cut = df[df["order_date"] < "2025-03-20"]                      # the later period stops three weeks in
    ctx = context(cut, profile)
    short = ops.execute(cut, ctx, {"op": "period_coverage", "params": {}})
    assert short["facts"]["incomplete"] and any("stops on" in i for i in short["facts"]["issues"])
    assert ops.evaluate({"op": "period_coverage", "params": {}}, short, {"expect": "incomplete"})["status"] == ops.SUPPORTED

    ctx_uneven = ops.AnalysisContext(formula=me.parse_formula("sum(revenue)"), label="m", base_filters=[], time_column="order_date",
                                     period_a=["2025-01-01", "2025-04-01"], period_b=["2025-04-01", "2025-04-29"], profile=profile)
    assert any("different lengths" in i for i in ops.execute(df, ctx_uneven, {"op": "period_coverage", "params": {}})["facts"]["issues"])

    dirty = df.copy()
    later = dirty["order_date"] >= SPLIT
    dirty.loc[dirty.index[later][:150], "revenue"] = None          # 42% of the later period missing
    dirty = pd.concat([dirty, dirty[later].head(60)], ignore_index=True)
    ctx = context(dirty, profile)
    ctx.dq_messages = ["Column 'channel' is no longer present."]
    dq = ops.execute(dirty, ctx, {"op": "data_quality", "params": {}})
    issues = " | ".join(dq["facts"]["issues"])
    assert "missing values in 'revenue' rose" in issues and "duplicate rows" in issues and "open data-quality incident" in issues
    assert ops.evaluate({"op": "data_quality", "params": {}}, dq, {"expect": "issues_present"})["status"] == ops.SUPPORTED
    clean = ops.execute(df, context(df, profile), {"op": "data_quality", "params": {}})
    assert ops.evaluate({"op": "data_quality", "params": {}}, clean, {"expect": "issues_present"})["status"] == ops.WEAKENED


# ========================================================== the engine ======

@pytest.fixture(scope="module")
def full_run(sales):
    df, profile = sales
    eng, provider = engine_for(df, profile, script())
    return eng.run(), eng, provider


def test_understanding_is_structured_and_validated(full_run):
    report, _, _ = full_run
    u = report["understanding"]
    assert u["metric"]["formula"] == "sum(revenue)" and "validated against the schema" in u["metric"]["source"]
    assert u["time_column"] == "order_date" and u["periods_source"] == "stated in the question"
    assert u["expected_direction"] == "decrease" and u["claimed_change_pct"] == 20 and u["population"] == "all rows"
    assert set(u["available_dimensions"]) == {"region", "product", "channel"} and u["assumptions"]
    assert any("about 20%" in f and "-9.9%" in f for f in report["observed_facts"]), "the question's own claim is checked against the data"


def test_hypotheses_are_validated_deduplicated_and_rejected_with_reasons(full_run):
    report, _, _ = full_run
    accepted = {h["id"]: h for h in report["hypotheses"]}
    assert len(accepted) == 5 and all(set(h) >= {"id", "statement", "measurable_prediction", "analysis", "prediction",
                                                 "required_columns", "confounders", "priority", "status", "evidence"} for h in accepted.values())
    assert [accepted[k]["priority"] for k in ("H1", "H2", "H3", "H4", "H5")] == [5, 4, 4, 3, 2], "ordered by priority"
    reasons = {r["statement"]: r["reason"] for r in report["rejected_hypotheses"]}
    assert "refund_status" in reasons["Refunds increased"] and reasons["Refunds increased"].startswith("schema-incompatible")
    assert reasons["Fewer units were sold overall"].startswith("duplicate")
    assert "unknown_operation" in reasons["Magic"] and "invalid_prediction" in reasons["Prices moved"]
    assert reasons["(empty)"].startswith("untestable")
    assert len(report["rejected_hypotheses"]) == 5


def test_every_status_comes_from_an_executed_analysis(full_run):
    report, _, _ = full_run
    analyses = {a["id"]: a for a in report["analyses"]}
    by_statement = {h["statement"]: h for h in report["hypotheses"]}
    for h in report["hypotheses"]:
        assert h["evidence"], h["id"]
        for e in h["evidence"]:
            a = analyses[e["analysis_id"]]
            assert a["status"] == "ok" and h["id"] in a["hypothesis_ids"] and a["signature"]
    west = by_statement["Revenue fell because the West region sold less"]
    assert west["status"] == "supported" and west["effect"]["value"] == pytest.approx(WEST_DELTA / (REV_B - REV_A))
    assert analyses[west["evidence"][0]["analysis_id"]]["op"] == "segment_contribution"
    assert by_statement["Fewer units were sold overall"]["status"] == "supported"
    prices = by_statement["Unit prices were cut"]
    assert prices["status"] == "inconclusive" and "not evidence that nothing changed" in prices["reasons"][0], \
        "prices measured higher, but a cut is still inside the interval: not contradicted, not supported"
    assert prices["effect"]["interval"][0] < 0 < prices["effect"]["interval"][1]
    assert by_statement["The drop is spread across channels in proportion"]["status"] == "weakened"
    assert by_statement["The later period is incomplete"]["status"] == "weakened"
    assert report["hypothesis_counts"] == {"supported": 2, "weakened": 2, "inconclusive": 1, "not_testable": 0, "failed": 0, "proposed": 0}
    assert [f["status"] for f in report["findings"]][:2] == ["supported", "supported"], "findings ranked by evidence"


def test_the_loop_adapts_then_stops_and_stays_inside_its_budget(full_run):
    report, eng, provider = full_run
    assert [a["params"]["dimension"] for a in report["analyses"] if a["requested_by"] == "scan"] == ["region", "product", "channel"]
    follow = [a for a in report["analyses"] if a["requested_by"] == "follow_up"]
    assert [a["op"] for a in follow] == ["price_volume", "segment_contribution"]
    assert follow[1]["params"]["filters"] == [{"column": "region", "op": "==", "value": "West"}], "it drilled into the supported segment"
    h2 = next(h for h in report["hypotheses"] if h["statement"].startswith("Fewer units"))
    assert h2["follow_ups"][0]["status_after"] == "supported" and len(h2["evidence"]) == 2
    west = next(h for h in report["hypotheses"] if "West" in h["statement"])
    drill = west["evidence"][1]
    assert (drill["role"], drill["assessment"]) == ("refinement", "weakened") and west["status"] == "supported", \
        "where inside West the change sits does not decide whether it sits in West"
    assert "Gizmo" in west["refinements"][0] and all("Gizmo" not in r for r in west["reasons"])
    coverage = next(a for a in report["analyses"] if a["op"] == "period_coverage")
    assert coverage["requested_by"] == "baseline" and len(coverage["hypothesis_ids"]) == 1, "the baseline check was reused, not re-run"
    assert report["stop_reason"] == "model_stopped" and report["budget"]["rounds"] == 3
    assert any("reason for stopping (unverified wording): Every hypothesis is resolved." in n for n in eng.model_notes)
    b = report["budget"]
    assert b["model_calls"] == provider.count() == 6 <= b["limits"]["model_calls"]
    assert b["analyses"] == len(report["analyses"]) == 12 <= b["limits"]["analyses"]
    assert [t for _, t in provider.calls] == ["submit_understanding", "submit_hypotheses", "submit_next_step",
                                               "submit_next_step", "submit_next_step", "submit_report"]


def test_report_is_grounded_and_separates_fact_from_explanation(full_run):
    report, _, _ = full_run
    assert report["determination"]["code"] == "explanations_supported" and report["status"] == "complete"
    assert "do not prove what caused it" in report["determination"]["text"]
    assert report["executive_summary"]["source"] == "model" and report["summary"].startswith("Revenue fell from 145,800 to 131,310")
    assert report["totals"]["a"] == REV_A and report["totals"]["b"] == REV_B and report["totals"]["change"] == REV_B - REV_A
    assert report["possible_explanations"] == ["Demand in the West region may have weakened."] and report["follow_ups"]
    assert any("Welch" in s for s in report["statistical_interpretations"])
    assert any("not proof of cause" in s for s in report["limitations"])
    repro = report["reproducibility"]
    assert repro["analysis_ids"] == [a["id"] for a in report["analyses"]] and len(repro["analysis_signatures"]) == 12
    assert repro["evidence_thresholds"]["strong_share"] == config.HYP_STRONG_SHARE and "deterministic" in repro["note"]
    assert report["charts"] and report["charts"][0]["plotly_json"]["data"][0]["type"] == "bar"


@pytest.mark.parametrize("summary,reason", [
    ("Revenue fell from 145,800 to 131,310, a 37.5% collapse driven by 4,200 lost orders.", "numbers that no executed analysis produced"),
    ("", "returned no summary"),
])
def test_a_summary_with_invented_numbers_is_rejected(sales, summary, reason):
    df, profile = sales
    eng, _ = engine_for(df, profile, script(submit_report=[{"executive_summary": summary, "possible_explanations": [], "follow_up_analyses": []}]))
    report = eng.run()
    assert report["executive_summary"]["source"] == "template" and reason in report["executive_summary"]["rejected_reason"]
    assert "37.5" not in report["summary"] and "4,200" not in report["summary"]
    assert report["summary"].startswith("Revenue fell from 145,800 to 131,310 (-9.9%)."), "the template uses executed results only"


def test_a_concentration_the_model_overlooks_is_still_reported(sales):
    """Seen on a real run: the model tested products and channels but never the
    region, which is where the change is."""
    df, profile = sales
    narrow = {"hypotheses": [hyp("Fewer units were sold overall", "component_change", {"formula": "sum(units)"}, "decrease", 5),
                             hyp("One channel lost revenue", "segment_contribution", {"dimension": "channel"}, "concentrated", 4)]}
    wording = {"executive_summary": "Revenue fell from 145,800 to 131,310. The West region accounts for 117% of the change.",
               "possible_explanations": ["West demand may have weakened.", "A 37% price cut in the East may have backfired."],
               "follow_up_analyses": ["Look at West by week.", "Survey 5,000 customers."]}
    eng, provider = engine_for(df, profile, script(submit_hypotheses=[narrow], submit_next_step=[{"decision": "stop", "rationale": "done"}],
                                                   submit_report=[wording]))
    prompts_seen, real = {}, eng.call_model
    eng.call_model = lambda **k: (prompts_seen.__setitem__(k["tool_name"], k["user_content"]), real(**k))[1]
    report = eng.run()
    west = [f for f in report["observed_facts"] if "region = West" in f]
    assert len(west) == 1 and "117% of the change" in west[0] and west[0].endswith("No supported hypothesis covers this.")
    assert not any("channel" in f or "product" in f for f in report["observed_facts"]), "proportional segments are not findings"
    assert '"segment":"West"' in prompts_seen["submit_hypotheses"], "the scan result is in front of the model before it proposes anything"
    assert report["executive_summary"]["source"] == "model", "a share stated as a percentage is a faithful form of the computed fraction"
    assert report["possible_explanations"] == ["West demand may have weakened."], "an explanation carrying an invented figure is dropped"
    assert report["follow_ups"] == ["Look at West by week."]

    covered = engine_for(df, profile, script())[0].run()
    assert not any("No supported hypothesis covers this" in f for f in covered["observed_facts"])


def test_failed_execution_is_never_counted_against_a_hypothesis(sales, monkeypatch):
    df, profile = sales
    real = ops.execute

    def flaky(frame, ctx, op):
        if op["op"] == "distribution_shift":
            raise RuntimeError("worker ran out of memory")
        return real(frame, ctx, op)

    monkeypatch.setattr(ops, "execute", flaky)
    eng, _ = engine_for(df, profile, script(submit_next_step=[{"decision": "stop", "rationale": "done"}]))
    report = eng.run()
    prices = next(h for h in report["hypotheses"] if h["statement"] == "Unit prices were cut")
    assert prices["status"] == "failed" and prices["evidence"][0]["assessment"] == "failed"
    assert "out of memory" not in str(report), "internal error text stays out of the report"
    failed = next(a for a in report["analyses"] if a["op"] == "distribution_shift")
    assert failed["status"] == "failed" and failed["result"] == {}
    assert report["hypothesis_counts"]["weakened"] == 2, "not added to the weakened pile"
    assert any("H" in l and "neither supported nor weakened" in l for l in report["limitations"])


def test_an_untestable_hypothesis_is_reported_as_not_testable(sales):
    df, profile = sales
    slim = df[["order_date", "region", "revenue"]]
    hyps = {"hypotheses": [hyp("Volume fell rather than price", "price_volume", {}, "volume_driven", 5)]}
    eng, _ = engine_for(slim, profile_dataframe(slim), script(submit_hypotheses=[hyps], submit_next_step=[{"decision": "stop", "rationale": "x"}]))
    report = eng.run()
    h = report["hypotheses"][0]
    assert h["status"] == "not_testable" and "not_applicable" in h["evidence"][0]["reasons"][0]
    assert report["determination"]["code"] == "cause_not_determined" and "could not be determined" in report["determination"]["text"]
    assert report["status"] == "inconclusive"


def test_unreliable_data_blocks_a_conclusion_even_with_a_supported_hypothesis(sales):
    df, profile = sales
    cut = df[df["order_date"] < "2025-03-20"]
    eng, _ = engine_for(cut, profile, script(submit_next_step=[{"decision": "stop", "rationale": "x"}],
                                             submit_report=[{"executive_summary": "The cause is the West region.", "possible_explanations": [], "follow_up_analyses": []}]))
    report = eng.run()
    assert any(h["status"] == "supported" for h in report["hypotheses"])
    assert report["determination"]["code"] == "data_cannot_support_conclusion" and report["status"] == "inconclusive"
    assert "may be an artefact" in report["determination"]["text"]
    assert report["executive_summary"]["source"] == "template", "a summary asserting a cause is rejected when none was determined"
    assert "asserted a cause" in report["executive_summary"]["rejected_reason"]
    assert any("not directly comparable" in l for l in report["limitations"]) and report["data_quality"]


def test_no_material_change_stops_before_any_hypothesis_is_generated(sales):
    df, profile = sales
    u = {**UNDERSTANDING, "metric_formula": "count_distinct(product)"}
    eng, provider = engine_for(df, profile, script(submit_understanding=[u]))
    report = eng.run()
    assert report["determination"]["code"] == "no_material_change" and report["hypotheses"] == []
    assert [t for _, t in provider.calls] == ["submit_understanding", "submit_report"], "no hypotheses were requested"
    assert report["stop_reason"] == "no_material_change"


def test_budgets_stop_the_loop_and_are_reported(sales):
    df, profile = sales
    eng, provider = engine_for(df, profile, script(), budget=he.Budget(max_llm_calls=2))
    report = eng.run()
    assert provider.count() == 2 and report["stop_reason"] == "budget:model_calls"
    assert report["executive_summary"]["source"] == "template" and "no model call was left" in report["executive_summary"]["rejected_reason"]
    assert any("model calls limit" in l for l in report["limitations"]) and len(report["hypotheses"]) == 5

    eng, _ = engine_for(df, profile, script(), budget=he.Budget(max_analyses=9))     # 4 baseline + 3 scan + 2 tests
    report = eng.run()
    assert report["budget"]["analyses"] == 9 and report["stop_reason"] == "budget:analyses"
    untested = [h for h in report["hypotheses"] if h["status"] == "proposed"]
    assert len(untested) == 3 and all(h["evidence"] == [] for h in untested), "untested stays untested, not weakened"

    eng, provider = engine_for(df, profile, script(), budget=he.Budget(max_seconds=0.0))
    report = eng.run()
    assert report["stop_reason"] == "budget:time" and provider.count() == 0 and report["analyses"] == []
    assert report["determination"]["code"] == "not_measured", "nothing measured, so nothing claimed about the metric"
    assert report["totals"]["a"] is None and report["status"] == "inconclusive"

    used = {"n": 0}
    eng, provider = engine_for(df, profile, script(), budget=he.Budget(max_tokens=1000, tokens_used=lambda: used["n"]))
    real = eng.call_model
    eng.call_model = lambda **k: (used.__setitem__("n", used["n"] + 600), real(**k))[1]
    report = eng.run()
    assert report["stop_reason"] == "budget:tokens" and provider.count() == 2

    endless = [{"decision": "analyze", "hypothesis_id": "H3", "analysis": {"op": "component_change", "params": {"formula": f"sum(units) * {i}"}},
                "prediction": {"expect": "decrease"}, "rationale": "again"} for i in range(1, 30)]
    eng, _ = engine_for(df, profile, script(submit_next_step=endless), budget=he.Budget(max_rounds=3, max_llm_calls=50))
    report = eng.run()
    assert report["budget"]["rounds"] == 3 and report["stop_reason"] == "budget:rounds", "a model that never stops is stopped"


def test_equivalent_analyses_are_never_executed_twice(sales, monkeypatch):
    df, profile = sales
    executed = []
    real = ops.execute
    monkeypatch.setattr(ops, "execute", lambda frame, ctx, op: (executed.append(ops.signature(op)), real(frame, ctx, op))[1])
    two = {"hypotheses": [hyp("Units fell", "component_change", {"formula": "sum(units)"}, "decrease", 5),
                          hyp("Units rose", "component_change", {"formula": "sum(units)", "filters": []}, "increase", 4)]}
    repeat = {"decision": "analyze", "hypothesis_id": "H1", "analysis": {"op": "component_change", "params": {"formula": "sum(units)"}},
              "prediction": {"expect": "decrease"}, "rationale": "check again"}
    eng, provider = engine_for(df, profile, script(submit_hypotheses=[two], submit_next_step=[repeat]))
    report = eng.run()
    assert len(executed) == len(set(executed)) == 8, "4 baseline analyses + 3 scans + 1 shared test"
    h1, h2 = report["hypotheses"][0], next(h for h in report["hypotheses"] if h["statement"] == "Units rose")
    assert h1["evidence"][0]["analysis_id"] == h2["evidence"][0]["analysis_id"], "one execution, judged against both predictions"
    assert (h1["status"], h2["status"]) == ("supported", "weakened")
    shared = next(a for a in report["analyses"] if a["op"] == "component_change")
    assert sorted(shared["hypothesis_ids"]) == ["H1", "H2"]
    assert report["stop_reason"].startswith("no_new_analysis") and provider.count("submit_next_step") == 2
    assert sum("repeated an analysis" in n for n in report["model_notes"]) == 2


@pytest.mark.parametrize("bad_next", [
    {"decision": "analyze", "hypothesis_id": "H99", "analysis": {"op": "price_volume", "params": {}}, "prediction": {"expect": "volume_driven"}, "rationale": "x"},
    {"decision": "analyze", "hypothesis_id": "H1", "analysis": {"op": "exec", "params": {"code": "rm -rf /"}}, "prediction": {"expect": "decrease"}, "rationale": "x"},
    {"decision": "analyze", "hypothesis_id": "H1", "analysis": {"op": "price_volume", "params": {}}, "prediction": {"expect": "definitely"}, "rationale": "x"},
])
def test_invalid_next_steps_are_discarded_not_executed(sales, bad_next):
    df, profile = sales
    eng, _ = engine_for(df, profile, script(submit_next_step=[bad_next]))
    report = eng.run()
    assert report["stop_reason"].startswith("no_new_analysis") and not [a for a in report["analyses"] if a["requested_by"] == "follow_up"]
    assert len(report["model_notes"]) == 2


@pytest.mark.parametrize("garbage", [None, "revenue went down", ["a list"], {"hypotheses": "none"}, {"hypotheses": [1, "x", None]}, {}])
def test_malformed_hypothesis_output_leads_to_an_honest_non_answer(sales, garbage):
    df, profile = sales
    eng, _ = engine_for(df, profile, script(submit_hypotheses=[garbage]))
    report = eng.run()
    assert report["hypotheses"] == [] and report["stop_reason"] == "no_testable_hypotheses"
    assert report["determination"]["code"] == "cause_not_determined" and report["totals"]["a"] == REV_A
    assert any("no hypothesis that could be tested" in l for l in report["limitations"])


def test_malformed_understanding_falls_back_to_what_the_schema_supports(sales):
    df, profile = sales
    bad = {"metric_formula": "DROP TABLE revenue;", "time_column": "nope", "period_a": ["2031-01-01", "2031-02-01"],
           "period_b": ["2031-02-01", "2031-03-01"], "population_filters": [{"column": "ghost", "op": "==", "value": 1}],
           "expected_direction": "sideways", "claimed_change_pct": "lots", "assumptions": "none", "missing_information": None}
    eng, _ = engine_for(df, profile, script(submit_understanding=[bad]))
    u = eng.run()["understanding"]
    assert u["metric"]["formula"] == 'sum("revenue")' and "inferred from a column named in the question" in u["metric"]["source"]
    assert u["time_column"] == "order_date" and u["periods_source"].startswith("automatic") and u["metric"]["filters"] == []
    assert u["expected_direction"] == "unknown" and u["claimed_change_pct"] is None
    assert any("fall outside the data" in n for n in eng.model_notes) and any("population filters" in n for n in eng.model_notes)

    eng, _ = engine_for(df, profile, script(submit_understanding=[bad]), question="Why did things get worse?")
    with pytest.raises(he.CannotInvestigate, match="No metric could be identified"):
        eng.run()
    no_dates = df.drop(columns=["order_date"])
    eng, _ = engine_for(no_dates, profile_dataframe(no_dates), script())
    with pytest.raises(he.CannotInvestigate, match="no date column"):
        eng.run()


def test_an_approved_metric_definition_overrides_the_model_s_formula(sales):
    df, profile = sales
    approved = [{"name": "Net revenue", "formula": "sum(revenue) - sum(cost)", "filters": [], "current_version": 2, "matched_term": "net revenue"}]
    eng, _ = engine_for(df, profile, script(), question="Why did net revenue fall?", approved_metrics=approved)
    u = eng.run()["understanding"]
    assert u["metric"]["formula"] == "sum(revenue) - sum(cost)" and "approved metric 'Net revenue' (version 2)" in u["metric"]["source"]


def test_prompts_carry_aggregates_and_schema_but_no_identifier_values(sales):
    df, profile = sales
    private = df.copy()
    private["customer_email"] = [f"person{i}@secret.example" for i in range(len(private))]
    provider = ScriptedProvider(script())
    seen = []
    eng = he.HypothesisEngine(question=QUESTION, df=private, profile=profile_dataframe(private),
                              call_model=lambda **k: (seen.append(k), provider("t", **k))[1])
    eng.run()
    prompts_text = "\n".join(k["user_content"] + k["system"] for k in seen)
    assert "secret.example" not in prompts_text and "customer_email" in prompts_text, "the column is known; its values are not sent"
    assert "sample values withheld" in prompts_text and "UNTRUSTED DATA" in prompts_text
    assert "West" in prompts_text, "values of low-cardinality dimensions are needed to reason about segments"
    assert all(len(k["user_content"]) < 40_000 for k in seen)
    import re
    assert not re.search(r"\d\.\d{6,}", prompts_text), "facts are rounded before the model sees them"


def test_an_interrupted_run_resumes_without_repeating_work(sales, monkeypatch):
    df, profile = sales
    store = he.MemoryStore()
    calls = {"n": 0}

    def die_late():
        calls["n"] += 1
        if calls["n"] == 13:                      # after the baseline, the scan and three hypothesis tests
            raise runtime.RunCancelled()

    eng, first = engine_for(df, profile, script(), store=store, checkpoint=die_late)
    with pytest.raises(runtime.RunCancelled):
        eng.run()
    done_before = [a["id"] for a in store.analyses]
    tested_before = [h["id"] for h in store.hypotheses if h["evidence"]]
    assert store.understanding and len(store.hypotheses) == 5 and 3 < len(done_before) < 12 and tested_before

    executed = []
    real = ops.execute
    monkeypatch.setattr(ops, "execute", lambda frame, ctx, op: (executed.append(op["op"]), real(frame, ctx, op))[1])
    eng2, second = engine_for(df, profile, script(), store=store)
    report = eng2.run()
    assert report["resumed"] is True
    assert "submit_understanding" not in [t for _, t in second.calls] and "submit_hypotheses" not in [t for _, t in second.calls]
    assert "metric_change" not in executed and "period_coverage" not in executed, "finished analyses are reused, not re-run"
    assert len(report["analyses"]) == 12 and [a["id"] for a in report["analyses"]][:len(done_before)] == done_before
    assert report["hypothesis_counts"]["supported"] == 2 and report["determination"]["code"] == "explanations_supported"


def test_losing_the_model_mid_run_keeps_the_evidence(sales):
    df, profile = sales
    provider = ScriptedProvider(script())

    def model(**k):
        if k["tool_name"] in ("submit_next_step", "submit_report"):
            raise runtime.ProviderUnavailable("quota exhausted")
        return provider("t", **k)

    eng = he.HypothesisEngine(question=QUESTION, df=df, profile=profile, call_model=model)
    report = eng.run()
    assert report["stop_reason"] == "model_unavailable" and report["hypothesis_counts"]["supported"] == 2
    assert report["executive_summary"]["source"] == "template" and "quota" not in str(report)
    assert any("became unavailable" in l for l in report["limitations"])

    def dead(**k):
        raise runtime.ProviderUnavailable("no key")

    with pytest.raises(runtime.ProviderUnavailable):       # nothing at all: the caller chooses the fallback
        he.HypothesisEngine(question=QUESTION, df=df, profile=profile, call_model=dead).run()


def test_tool_schemas_are_accepted_by_both_provider_sdks(sales):
    """A real Gemini call first failed here: the scripted provider does not
    validate schemas, the SDK does."""
    from google.genai import types

    def walk(node, path="$"):
        if isinstance(node, dict):
            if "type" in node and "properties" not in path.rsplit(".", 1)[-1]:
                assert isinstance(node["type"], str), f"{path}: one type per field, got {node['type']!r}"
                assert node["type"] != "object" or node.get("properties"), f"{path}: an object needs properties"
            for key, value in node.items():
                if key == "properties":
                    for name, sub in value.items():
                        assert sub, f"{path}.{name}: untyped field"
                        walk(sub, f"{path}.{name}")
                elif key == "items":
                    walk(value, f"{path}[]")

    for name, schema in (("submit_understanding", he.UNDERSTANDING_SCHEMA), ("submit_hypotheses", he.HYPOTHESES_SCHEMA),
                         ("submit_next_step", he.NEXT_STEP_SCHEMA), ("submit_report", he.REPORT_SCHEMA)):
        types.FunctionDeclaration(name=name, description="d", parameters=schema)     # raises on an invalid schema
        walk(schema)

    df, profile = sales                 # values arrive as text and are compared as the column's own type
    ctx = context(df, profile)
    op = ops.normalize({"op": "share_change", "params": {"filters": [{"column": "units", "op": ">", "value": "20"}]}}, ctx)
    out = ops.execute(df, ctx, op)["facts"]
    earlier = [r for r in ROWS if r["order_date"] < SPLIT]
    assert out["matching_a"] == sum(float(r["units"]) > 20 for r in earlier) > 0
    for claim, question, kept in ((0, QUESTION, None), (35, QUESTION, None), (20, QUESTION, 20), (-20, QUESTION, -20),
                                  (2, QUESTION, None), (20, "Why did revenue drop in 2025?", None), (True, QUESTION, None)):
        eng, _ = engine_for(df, profile, script(submit_understanding=[{**UNDERSTANDING, "claimed_change_pct": claim}],
                                                submit_next_step=[{"decision": "stop", "rationale": "x"}]), question=question)
        assert eng.run()["understanding"]["claimed_change_pct"] == kept, (claim, question)
    no_claim = {k: v for k, v in UNDERSTANDING.items() if k != "claimed_change_pct"}
    eng, _ = engine_for(df, profile, script(submit_understanding=[no_claim], submit_next_step=[{"decision": "stop", "rationale": "x"}]))
    assert eng.run()["understanding"]["claimed_change_pct"] is None


# ======================================================== integration ======

@pytest.fixture
def ws(client):
    return workspace(client)


def test_routing_uses_the_engine_only_when_a_model_can_answer(monkeypatch):
    profile = {"row_count": 9, "columns": [{"name": "d", "kind": "datetime"}, {"name": "revenue", "kind": "numeric"}]}
    none = {"metrics": [], "ambiguous": [], "dimensions": []}
    assert router.decide("Why did revenue drop?", profile, none, model_available=True)["route"] == "hypothesis"
    assert router.decide("Why did revenue drop?", profile, none, model_available=False)["route"] == "root_cause"
    assert router.decide("Why did revenue drop?", profile, none, forced="root_cause", model_available=True)["route"] == "root_cause"
    assert router.decide("total revenue", profile, none, model_available=True)["route"] == "fast", "simple questions stay cheap"
    monkeypatch.setattr(config, "HYPOTHESIS_ENGINE_ENABLED", False)
    assert router.decide("Why did revenue drop?", profile, none, model_available=True)["route"] == "root_cause"
    assert router.decide("Why did revenue drop?", profile, none, forced="hypothesis", model_available=True)["route"] == "hypothesis"
    no_dates = {"row_count": 9, "columns": [{"name": "revenue", "kind": "numeric"}]}
    assert router.decide("Why did revenue drop?", no_dates, none, forced="hypothesis", model_available=True)["route"] != "hypothesis"

    assert llm_client.provider_available("submit_understanding") is False, "no keys in the test environment"
    with runtime.provider_override(ScriptedProvider({})):
        assert llm_client.provider_available("submit_understanding") is False
    with runtime.provider_override(ScriptedProvider(script())):
        assert llm_client.provider_available("submit_understanding") is True
    assert router.ROUTE_BUDGETS["hypothesis"]["sandbox_runs"] == 0, "the engine never runs generated code"


def test_a_question_runs_the_whole_investigation_through_the_job_system(client, ws):
    created = ask(client, ws, QUESTION)
    assert run_job(created["job_id"], script()) == "succeeded"
    q = question_row(created["question_id"])
    assert (q.status, q.route) == ("verified", "hypothesis")

    dash = client.get(f"/api/questions/{q.id}/dashboard", headers=ws["h"]).json()
    assert dash["verdict_state"] == "VERIFIED_WITH_CAVEATS" and [k["value"] for k in dash["kpis"]] == ["145,800", "131,310", "-14,490"]
    assert {k["evidence_status"] for k in dash["kpis"]} == {"verified"} and dash["narrative"].startswith("Revenue fell from 145,800")
    ev = client.get(f"/api/questions/{q.id}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(r for r in ev if r["claim_type"] == "narrative")
    assert narrative["status"] == "verified_with_caveats" and narrative["validity"]["causal"] == "not_established"
    assert {c["check"]: c["outcome"] for c in narrative["checks"]}["model_summary_grounded"] == "pass"
    assert {c["check"]: c["outcome"] for c in narrative["checks"]}["narrative_numbers_supported"] == "pass"

    inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    assert inv["engine"] == "hypothesis" and inv["status"] == "complete" and inv["question_id"] == q.id
    detail = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()
    assert [h["status"] for h in detail["hypotheses"]].count("supported") == 2 and len(detail["analyses"]) == 12
    assert all(a["status"] == "ok" and a["signature"] for a in detail["analyses"])
    assert detail["report"]["reproducibility"]["dataset_fingerprint"] and detail["report"]["reproducibility"]["models"] == ["scripted:scripted"]
    assert detail["report"]["reproducibility"]["investigation_id"] == inv["id"]

    trace = client.get(f"/api/questions/{q.id}/trace", headers=ws["h"]).json()
    assert [e["name"] for e in trace["events"] if e["kind"] == "node"] == ["router", "hypothesis"]
    assert trace["usage"]["llm_calls"] == 6 and trace["usage"]["sandbox_runs"] == 0
    assert trace["budget"]["llm_calls"] == config.HYP_MAX_LLM_CALLS
    manifest = client.get(f"/api/questions/{q.id}/manifest", headers=ws["h"]).json()
    assert manifest["question"]["route"] == "hypothesis" and manifest["models"]["scripted:scripted"]["calls"] == 6
    status = client.get(f"/api/questions/{q.id}/status", headers=ws["h"]).json()
    assert status["current_stage"] == "done" and status["steps"][0]["status"] == "done"


def test_the_investigation_endpoint_validation_idempotency_and_tenancy(client, ws):
    body = {"dataset_id": ws["ds"], "question": QUESTION}
    r = client.post("/api/investigations/hypothesis", json=body, headers=ws["h"])
    assert r.status_code == 503 and "No AI model provider" in r.json()["detail"], "refused up front without a model"
    with runtime.provider_override(ScriptedProvider(script())):
        h = {**ws["h"], "Idempotency-Key": "hyp-1"}
        first = client.post("/api/investigations/hypothesis", json=body, headers=h)
        assert first.status_code == 201 and first.json()["limits"]["analyses"] == config.HYP_MAX_ANALYSES
        again = client.post("/api/investigations/hypothesis", json=body, headers=h).json()
        assert again["investigation_id"] == first.json()["investigation_id"] and again["created"] is False
        assert client.post("/api/investigations/hypothesis", json={**body, "question": "hi"}, headers=ws["h"]).status_code == 422
        assert client.post("/api/investigations/hypothesis", json={**body, "dataset_id": 10**9}, headers=ws["h"]).status_code == 404
        flat = workspace(client, b"a,b\n1,2\n3,4\n", "flat.csv")
        r = client.post("/api/investigations/hypothesis", json={"dataset_id": flat["ds"], "question": QUESTION}, headers=flat["h"])
        assert r.status_code == 400 and "no date column" in r.json()["detail"]
        other = workspace(client)
        assert client.post("/api/investigations/hypothesis", json=body, headers=other["h"]).status_code == 404
        assert client.post("/api/investigations/hypothesis", json=body).status_code == 401
    created = first.json()
    assert question_row(created["question_id"]).route == "hypothesis"
    assert run_job(created["job_id"], script()) == "succeeded"
    assert client.get(f"/api/investigations/{created['investigation_id']}", headers=ws["h"]).json()["status"] == "complete"
    assert client.get(f"/api/investigations/{created['investigation_id']}", headers=other["h"]).status_code == 404


def test_progress_is_visible_and_a_cancelled_run_resumes_on_retry(client, ws):
    created = ask(client, ws, QUESTION)
    provider = ScriptedProvider(script())
    seen = {}

    def cancelling(provider_hint, **k):
        if k["tool_name"] == "submit_next_step" and "cancelled" not in seen:
            seen["cancelled"] = True
            inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
            seen["live"] = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()
            seen["stage"] = client.get(f"/api/questions/{created['question_id']}/status", headers=ws["h"]).json()
            client.post(f"/api/questions/{created['question_id']}/cancel", headers=ws["h"])
            time.sleep(2.2)      # a cancel request is picked up within 2 s; a real model call takes at least that long
        return provider(provider_hint, **k)

    cancelling.can_answer = provider.can_answer
    with runtime.provider_override(cancelling):
        state = jobs.run_inline(created["job_id"])
    assert state == "cancelled" and job_row(created["job_id"]).failure_class == "cancelled"
    assert question_row(created["question_id"]).status == "cancelled"
    live = seen["live"]
    assert live["status"] == "running" and len(live["hypotheses"]) == 5 and len(live["analyses"]) == 10, "state is readable mid-run"
    assert {h["status"] for h in live["hypotheses"]} == {"supported", "weakened", "inconclusive"}
    assert seen["stage"]["current_stage"] == "investigating" and "choosing the next analysis" in seen["stage"]["stage_detail"]
    assert client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"]).status_code == 404

    assert client.post(f"/api/jobs/{created['job_id']}/retry", headers=ws["h"]).status_code == 200
    resumed = ScriptedProvider(script())
    with runtime.provider_override(resumed):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    assert [t for _, t in resumed.calls] == ["submit_next_step", "submit_next_step", "submit_next_step", "submit_report"], \
        "the understanding and the hypotheses were not asked for again"
    db = SessionLocal()
    try:
        inv_id = live["id"]
        assert db.query(InvestigationAnalysis).filter_by(investigation_id=inv_id).count() == 12
        assert db.query(InvestigationHypothesis).filter_by(investigation_id=inv_id).count() == 5
    finally:
        db.close()
    report = client.get(f"/api/investigations/{inv_id}", headers=ws["h"]).json()["report"]
    assert report["resumed"] is True and report["status"] == "complete"


def test_no_model_at_the_first_step_falls_back_to_the_deterministic_investigator(client, ws):
    created = ask(client, ws, "Why did revenue change?")

    def down(provider_hint, **k):
        raise runtime.ProviderUnavailable("both providers failed")

    down.can_answer = lambda tool: True      # configured, but failing
    with runtime.provider_override(down):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    q = question_row(created["question_id"])
    assert (q.status, q.route) == ("verified", "root_cause")
    trace = client.get(f"/api/questions/{q.id}/trace", headers=ws["h"]).json()
    assert [e["name"] for e in trace["events"] if e["kind"] == "node"] == ["router", "hypothesis", "investigate"]
    assert any("fell back" in r for e in trace["events"] if e["kind"] == "route" for r in e["detail"]["reasons"])
    inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    report = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()["report"]
    assert report["ranked_contributors"][0]["segment"] == "West" and report.get("engine") != "hypothesis"


def test_losing_the_model_before_any_hypothesis_also_falls_back(client, ws, sales):
    """Seen on a real run: the model answered the first call, then returned 503."""
    df, profile = sales
    provider = ScriptedProvider(script())

    def flaky(provider_hint=None, **k):
        if k["tool_name"] != "submit_understanding":
            raise runtime.ProviderUnavailable("503 then 401")
        return provider(provider_hint, **k)

    eng = he.HypothesisEngine(question=QUESTION, df=df, profile=profile, call_model=flaky)
    report = eng.run()
    assert report["stop_reason"] == "model_unavailable" and report["hypotheses"] == [] and report["totals"]["a"] == REV_A
    assert any("before any hypothesis" in l for l in report["limitations"])
    assert not any("no usable list" in n for n in eng.model_notes), "an outage is not reported as a bad answer"

    created = ask(client, ws, QUESTION)
    flaky.can_answer = lambda tool: True
    with runtime.provider_override(flaky):
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    q = question_row(created["question_id"])
    assert (q.status, q.route) == ("verified", "root_cause")
    inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    assert client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()["report"]["ranked_contributors"][0]["segment"] == "West"


def test_periods_written_with_inclusive_end_dates_lose_no_day(sales):
    """Seen on a real run: the model wrote 1 Jan - 28 Feb and 1 Mar - 30 Apr."""
    df, profile = sales
    u = {**UNDERSTANDING, "period_a": ["2025-01-01", "2025-02-28"], "period_b": ["2025-03-01", "2025-04-30"]}
    eng, _ = engine_for(df, profile, script(submit_understanding=[u], submit_next_step=[{"decision": "stop", "rationale": "x"}]))
    report = eng.run()
    assert (report["totals"]["a"], report["totals"]["b"]) == (REV_A, REV_B), "28 February is part of the earlier period"
    assert report["understanding"]["period_a"] == ["2025-01-01T00:00:00", "2025-03-01T00:00:00"]
    assert report["understanding"]["period_b"][1] == "2025-04-30T23:59:59" and report["understanding"]["periods_source"] == "stated in the question"
    gap = {**UNDERSTANDING, "period_a": ["2025-01-01", "2025-01-31"], "period_b": ["2025-03-01", "2025-03-31"]}
    eng, _ = engine_for(df, profile, script(submit_understanding=[gap], submit_next_step=[{"decision": "stop", "rationale": "x"}]))
    assert eng.run()["understanding"]["period_a"][1] == "2025-02-01T00:00:00", "January ends with 31 January, not at the later period"


def test_a_model_summary_may_state_a_fall_as_a_positive_magnitude(client, ws):
    """Seen on a real run: "a decrease of 9,500" for a change of -9,500 passed
    the engine's check but was marked unverified by evidence verification."""
    wording = {"executive_summary": "Revenue declined from 145,800 to 131,310, a decrease of 14,490 or about 9.9%. "
                                    "2 of the 5 hypotheses tested are consistent with the data.",
               "possible_explanations": [], "follow_up_analyses": []}
    created = ask(client, ws, QUESTION)
    assert run_job(created["job_id"], script(submit_report=[wording])) == "succeeded"
    ev = client.get(f"/api/questions/{created['question_id']}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(r for r in ev if r["claim_type"] == "narrative")
    checks = {c["check"]: c for c in narrative["checks"]}
    assert checks["narrative_numbers_supported"]["outcome"] == "pass" and narrative["status"] == "verified_with_caveats"
    assert "checked against the executed analyses" in checks["independent_review"]["detail"]

    invented = {**wording, "executive_summary": "Revenue declined by 14,490, of which 9,999 came from returns."}
    created = ask(client, ws, QUESTION + " ")
    assert run_job(created["job_id"], script(submit_report=[invented])) == "succeeded"
    dash = client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"]).json()
    assert "9,999" not in dash["narrative"], "the invented figure never reaches the dashboard"


def test_the_engine_s_model_budget_is_enforced_by_the_run_context(client, ws, monkeypatch):
    monkeypatch.setitem(router.ROUTE_BUDGETS, "hypothesis", {"llm_calls": 3, "sandbox_runs": 0, "seconds": 120})
    monkeypatch.setattr(config, "HYP_MAX_LLM_CALLS", 20)      # the engine's own limit is looser than the run's
    created = ask(client, ws, QUESTION)
    assert run_job(created["job_id"], script()) == "failed"
    assert job_row(created["job_id"]).failure_class == "budget_exceeded"
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=ws["h"]).json()
    assert trace["usage"]["llm_calls"] == 3 and trace["usage"]["sandbox_runs"] == 0


def test_a_question_the_data_cannot_answer_fails_with_a_clear_reason(client, ws):
    created = ask(client, ws, "Why did customer satisfaction drop?")
    u = {**UNDERSTANDING, "metric_formula": "avg(satisfaction_score)"}
    assert run_job(created["job_id"], script(submit_understanding=[u])) == "failed"
    q = question_row(created["question_id"])
    assert q.status == "failed" and "No metric could be identified" in q.error
    inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    assert inv["status"] == "inconclusive" and "No metric could be identified" in inv["summary"]
