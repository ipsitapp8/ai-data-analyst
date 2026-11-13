"""The spider-numeric evaluation suite: the harness reproduces the copilot's
no-model path exactly, scores and attributes failures correctly, stays out of
the default suite, and its import is reproducible.

Most tests use the repository's own sales fixture. The tests marked as needing
the Spider reference set skip when it has not been built locally (it is not in
Git; see training/data/eval/spider_dev_single_table/REPORT.md)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
from platform_helpers import workspace

from app import metric_engine as me
from app.evals import runner, spider

SALES = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "sales.csv"
REPO = Path(__file__).resolve().parents[2]
SPIDER_BUILT = (REPO / "training/data/eval/spider_dev_single_table/cases.jsonl").exists() and \
    (REPO / "training/data/raw/spider_official/spider_data/dev.json").exists()
COLUMNS = ["*", "order_date", "region", "product", "channel", "units", "unit_price", "revenue", "cost"]


def _col(name: str, agg: int = 0, distinct: bool = False) -> list:
    return [agg, COLUMNS.index(name), distinct]


def _sql(select: list, where: list | None = None) -> dict:
    """A query in Spider's parse format: select items are (agg_id, val_unit)."""
    return {"select": [False, select], "from": {"table_units": [["table_unit", 0]], "conds": []},
            "where": where or [], "groupBy": [], "having": [], "orderBy": [], "limit": None,
            "intersect": None, "union": None, "except": None}


def _sel(agg: int, name: str, distinct: bool = False, unit_op: int = 0) -> list:
    return [agg, [unit_op, _col(name, 0, distinct), None if unit_op == 0 else _col("cost")]]


def _cond(name: str, op_id: int, value, not_op: bool = False, value2=None) -> list:
    return [not_op, op_id, [0, _col(name), None], value, value2]


def _case(question: str, sql: dict, values: list, note: str | None = None) -> dict:
    import hashlib
    numbers = [None if v is None else float(v) for v in values]
    return {"id": "t", "component": "spider_copilot", "question": question, "csv": str(SALES.relative_to(REPO)),
            "csv_sha256": hashlib.sha256(SALES.read_bytes()).hexdigest(), "answer_shape": "scalar" if len(values) == 1 else "scalar_row",
            "reference_id": "ref-t", "db_id": "sales", "table": "sales", "query": "", "sql_parse": sql, "column_names": COLUMNS,
            "reference": {"columns": [], "values": values, "numbers": numbers, "scorable": note is None, "note": note},
            "tolerance": {"rel": 1e-3, "abs": 5e-3}, "leakage": [], "data_fidelity_risk": []}


def _run(case: dict) -> tuple[dict, runner._Check]:
    chk = runner._Check()
    return spider.run_case(case, chk), chk


def _truth(formula: str, filters: list | None = None) -> float:
    return me.evaluate(me.load_frame(str(SALES)), formula, filters)["value"]


# ------------------------------------------------- fidelity to the copilot --

FIDELITY_MESSAGES = [
    "total revenue", "average unit_price", "How many records are there?", "max cost for region = West",
    "total revenue by region", "How many singers do we have?", "total revenue in 2025", "units", "median units only East",
]


@pytest.mark.parametrize("message", FIDELITY_MESSAGES)
def test_harness_gives_what_the_copilot_api_gives_without_a_model(client, message):
    """Each message as the first turn of a new session through the real API
    (rules only: no provider key in tests) against the harness on the same file."""
    ws = workspace(client)
    sid = client.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=ws["h"]).json()["id"]
    turn = client.post(f"/api/copilot/sessions/{sid}/messages", json={"message": message}, headers=ws["h"]).json()
    from app.profiling import load_and_profile_csv
    harness = spider.ask_copilot(message, str(SALES), load_and_profile_csv(str(SALES))[1], me.load_frame(str(SALES)))

    status = turn["action"]["status"]
    if status == "applied":
        result = turn.get("result")
        api = "no_metric" if result is None else "answered" if result["type"] == "scalar" else "answered_table"
    else:
        api = {"not_understood": "not_understood", "clarification": "clarification",
               "invalid_operation": "invalid_operation"}[status]
    assert harness["status"] == api, (message, turn["content"])
    if api == "answered":
        assert harness["value"] == turn["result"]["value"]
        assert harness["spec"]["formula"] == turn["session"]["state"]["formula"]
    assert turn["action"].get("interpreted_by", "rules") == "rules"


# ---------------------------------------------------------------- scoring --

def test_a_correctly_understood_question_scores_exact():
    value = _truth('sum("revenue")')
    m, chk = _run(_case("total revenue", _sql([_sel(4, "revenue")]), [value]))
    assert m["outcome"] == "correct_exact" and m["exact"] and m["within_tolerance"] and chk.passed
    assert m["spec_matches_oracle"] and m["cause"] is None


def test_rounding_inside_tolerance_is_not_exact():
    value = _truth('avg("unit_price")')
    m, chk = _run(_case("average unit_price", _sql([_sel(5, "unit_price")]), [round(value, 1) + 0.001]))
    assert m["outcome"] == "correct_tolerance" and not m["exact"] and chk.passed


def test_a_different_computation_that_happens_to_agree_is_flagged():
    """'How many regions' is read as distinct regions; if the reference counted
    rows filtered to one region per row, the number may agree by coincidence."""
    regions = _truth('count_distinct("region")')
    m, chk = _run(_case("How many regions are there?", _sql([_sel(3, "*")]), [regions]))
    assert m["correct"] and not m["spec_matches_oracle"], "right number, different spec"


def test_unparsed_question_with_an_expressible_answer_is_a_question_understanding_failure():
    rows = _truth("count()", [{"column": "units", "op": ">", "value": 10}])
    m, chk = _run(_case("How many orders sold more than 10 units?", _sql([_sel(3, "*")], [_cond("units", 3, 10)]), [rows]))
    assert not chk.passed and m["outcome"] in ("not_understood", "no_metric", "clarification")
    assert m["oracle_status"] == "matches" and m["cause"] == "question_understanding"
    assert m["understanding_issues"]


def test_several_numbers_in_one_answer_are_scored_together():
    case = _case("average and maximum revenue", _sql([_sel(5, "revenue"), _sel(1, "revenue")]),
                 [_truth('avg("revenue")'), _truth('max("revenue")')])
    m, chk = _run(case)
    assert m["copilot_status"] == "answered_multi" and m["correct"] and m["spec_matches_oracle"] and chk.passed
    wrong = {**case, "reference": {**case["reference"], "values": [1.0, 2.0], "numbers": [1.0, 2.0]}}
    m, chk = _run(wrong)
    assert m["outcome"] == "wrong_value" and not chk.passed, "every number must match, not just one"


def test_or_across_columns_and_arithmetic_are_not_expressible():
    where = [_cond("units", 3, 20), "or", _cond("revenue", 4, 200)]
    m, _ = _run(_case("How many orders sold over 20 units or under 200 revenue?", _sql([_sel(3, "*")], where), [1.0]))
    assert m["oracle_status"] == "not_expressible" and m["cause"] == "deterministic_limitation"
    m, _ = _run(_case("average margin", _sql([_sel(5, "revenue", unit_op=1)]), [1.0]))
    assert m["oracle_status"] == "not_expressible" and "arithmetic" in m["oracle_reason"]


def test_oracle_translates_or_on_one_column_like_and_negation():
    frame_cols = COLUMNS[1:]
    spec = spider.oracle_spec(_sql([_sel(3, "*")], [_cond("region", 2, '"East"'), "or", _cond("region", 2, '"West"')]),
                              COLUMNS, frame_cols)
    assert spec == {"formulas": ["count()"], "filters": [{"column": "region", "op": "in", "value": ["East", "West"]}]}
    spec = spider.oracle_spec(_sql([_sel(3, "region", distinct=True)], [_cond("product", 9, '"%idg%"')]), COLUMNS, frame_cols)
    assert spec["formulas"] == ['count_distinct("region")'] and spec["filters"][0]["op"] == "contains"
    spec = spider.oracle_spec(_sql([_sel(4, "revenue")], [_cond("units", 3, 10, not_op=True)]), COLUMNS, frame_cols)
    assert spec["filters"] == [{"column": "units", "op": "<=", "value": 10}]
    with pytest.raises(spider.NotExpressible):
        spider.oracle_spec(_sql([_sel(3, "*")], [_cond("product", 9, '"Wid%"')]), COLUMNS, frame_cols)


def test_a_reference_that_is_not_a_number_is_attributed_to_data_not_to_the_copilot():
    m, chk = _run(_case("total revenue", _sql([_sel(4, "revenue")]), [None], note="null_reference"))
    assert not chk.passed and m["cause"] == "data_representation"


def test_a_changed_fixture_is_not_scored():
    case = {**_case("total revenue", _sql([_sel(4, "revenue")]), [1.0]), "csv_sha256": "0" * 64}
    m, chk = _run(case)
    assert m["outcome"] == "fixture_missing" and not chk.passed


# ------------------------------------------------------- suite separation --

def test_default_suite_is_unchanged_by_the_separate_suite():
    default = runner.load_suite()
    assert not any(c["component"] == "spider_copilot" for c in default["cases"])
    assert "spider_copilot" not in runner.COMPONENTS
    assert "spider" not in default["suite_version"]


def test_unknown_or_unbuilt_suite_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        runner.load_suite(suite_name="nope")
    monkeypatch.setattr(runner, "SUITE_PATH", tmp_path / "suite.json")
    with pytest.raises(FileNotFoundError):
        runner.load_suite(suite_name="spider-numeric")


# ----------------------------------------------- needs the Spider reference --

def _importer():
    spec = importlib.util.spec_from_file_location("build_spider_cases",
                                                  Path(__file__).resolve().parents[1] / "evals" / "build_spider_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SPIDER_BUILT, reason="the Spider reference set is not built locally")
def test_import_is_verified_deterministic_and_selects_the_165():
    importer = _importer()
    first, verification = importer.build()
    second, _ = importer.build()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["selection"]["cases"] == 165 and first["selection"]["by_answer_shape"] == {"scalar": 143, "scalar_row": 22}
    assert verification["csv_hash_ok"] == 165
    assert verification.get("reference_reexecuted_identical", 0) + verification.get("reference_not_reexecuted", 0) == 90
    assert all(c["component"] == "spider_copilot" for c in first["cases"])


@pytest.mark.skipif(os.environ.get("SPIDER_HELDOUT_TESTS") != "1",
                    reason="runs all 165 Spider cases, the held-out evaluation questions included: opt-in only, "
                           "set SPIDER_HELDOUT_TESTS=1 when that run is authorised")
@pytest.mark.skipif(not (SPIDER_BUILT and (REPO / "backend/evals/cases/spider_numeric.json").exists()),
                    reason="the spider-numeric case file is not built locally")
def test_spider_suite_runs_without_a_model_or_generated_code():
    """Crash check over the whole suite. It executes the held-out questions, so it
    never runs in an ordinary test run (see the opt-in above)."""
    report = runner.run_suite("offline", suite_name="spider-numeric")
    assert report["summary"]["cases"] == 165
    assert report["summary"]["pipeline"]["llm_calls"] == 0 and report["summary"]["pipeline"]["sandbox_runs"] == 0
    assert all(r["detail"]["error"] is None for r in report["results"]), "no case crashes"
    assert all(r["metrics"]["outcome"] != "fixture_missing" for r in report["results"])
