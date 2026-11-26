"""Regressions for the last v4 evaluation failures (model output, deterministic repair).

1. Narration rounding: the model writes a tool value mis-rounded at its own precision (16.6962 -> "16.69",
   2439.5448 -> "2439.55", p 0.0496 -> "0.049"). The grounding check rightly rejects these, so the whole summary
   was dropped. explain.repair_roundings corrects ONLY a number that is a mis-rounding of exactly one result value
   at the precision it is written with; anything else is left for the grounding check to reject, as before.
2. Column case: the model wrote `product_code` for the column `Product_Code`. An exact name always wins; a
   unique case-insensitive match resolves to the dataset's own name; several matches are rejected as ambiguous.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.analyst import explain, plan_checks, tools


# ------------------------------------------------------------------ narration --
def fix(text, evidence, question=""):
    out, fixes = explain.repair_roundings(text, evidence, question)
    return out, fixes, explain.ungrounded(out, evidence, question)


@pytest.mark.parametrize("text,value,expected", [
    ("The average age is 16.69.", 16.696202531645568, "The average age is 16.70."),          # truncation
    ("It is 2439.55 metres.", 2439.5447776119404, "It is 2439.54 metres."),                   # wrong last digit
    ("The test gives p = 0.049.", 0.0496, "The test gives p = 0.050."),                       # p-value, 3 dp kept
    ("Late fees average 2.89.", 2.895357142857143, "Late fees average 2.90."),               # carry into tenths
    ("Fuel is 10.99 tonnes.", 10.998666666666667, "Fuel is 11.00 tonnes."),                  # carry into units
    ("Conversion was 12.34%.", 0.12346, "Conversion was 12.35%."),                            # percentage of a share
    ("Revenue was $5,663.60.", 5663.5805, "Revenue was $5,663.58."),                          # currency, commas kept
    ("The balance fell to -3.45.", -3.4567, "The balance fell to -3.46."),                    # negative
    ("To three decimals it is 16.695.", 16.696202531645568, "To three decimals it is 16.696."),  # requested precision kept
])
def test_misrounded_values_are_corrected_at_the_written_precision(text, value, expected):
    out, fixes, bad = fix(text, [{"value": value}])
    assert out == expected
    assert len(fixes) == 1 and bad == []


@pytest.mark.parametrize("text,value", [
    ("The average is 16.70.", 16.696202531645568),         # already right
    ("The average is 16.69.", 16.6949),                    # just below the boundary: right as written
    ("It is 2.35.", 2.345), ("It is 2.34.", 2.345),        # exact half: half-up and half-even both accepted
    ("Total 1,234,567.9 units.", 1234567.89),              # large value
    ("Counted 30,000 rows.", 30000),                       # integer
    ("The rate is 2.50.", 2.5),                            # trailing zero
    ("p < 0.001.", 0.00004),                               # small p-value
])
def test_correct_values_are_never_changed(text, value):
    out, fixes, bad = fix(text, [{"value": value}])
    assert out == text and fixes == [] and bad == []


def test_numbers_that_are_not_a_near_rounding_are_left_for_the_check_to_reject():
    out, fixes, bad = fix("The average is 18.40.", [{"value": 16.6962}])     # a different number, not a rounding slip
    assert out == "The average is 18.40." and fixes == [] and bad == ["18.40"]
    out, fixes, bad = fix("About 19 people.", [{"value": 16.6962}])           # integers are never adjusted
    assert fixes == [] and bad == ["19"]


def test_ambiguous_near_values_are_not_guessed():
    out, fixes, bad = fix("It is 2.45.", [{"a": 2.4449, "b": 2.4551}])        # 2.44 or 2.46: no unique correction
    assert fixes == [] and bad == ["2.45"]


def test_question_numbers_are_never_adjusted():
    out, fixes, bad = fix("Rows above 2,439 were kept.", [{"value": 2439.5448}], "How many above 2,439?")
    assert out == "Rows above 2,439 were kept." and fixes == [] and bad == []


def test_several_values_in_one_sentence():
    ev = [{"groups": [{"value": 16.696202531645568}, {"value": 2439.5447776119404}]}]
    out, fixes, bad = fix("A is 16.69 and B is 2439.55, over 67 rows.", ev + [{"rows": 67}])
    assert out == "A is 16.70 and B is 2439.54, over 67 rows." and len(fixes) == 2 and bad == []


def test_intermediate_values_are_not_rounded():
    ev = [{"value": 16.696202531645568}]
    explain.repair_roundings("The average age is 16.69.", ev, "")
    assert ev == [{"value": 16.696202531645568}]


def test_narration_with_a_rounding_slip_is_shown_corrected(monkeypatch):
    from app.analyst import orchestrator, planner
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "The average age is 16.69, computed over 395 rows.")
    notes: list[str] = []
    summary, grounding = orchestrator._narrate("What is the average age?", [{"value": 16.696202531645568, "rows": 395}], notes)
    assert summary == "The average age is 16.70, computed over 395 rows."
    assert grounding["ungrounded"] == [] and grounding.get("rounding_repaired")
    assert any("16.69" in n and "16.70" in n for n in notes)


def test_narration_with_an_unsupported_number_is_still_withheld(monkeypatch):
    from app.analyst import orchestrator, planner
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "The average age is 18.40, computed over 395 rows.")
    summary, grounding = orchestrator._narrate("What is the average age?", [{"value": 16.6962, "rows": 395}], [])
    assert summary is None and grounding["ungrounded"] == ["18.40"]


# ---------------------------------------------------------------- column case --
def frame(*cols):
    return pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in cols} | {"region": ["a", "b", "a", "b"]})


def metric(col, df):
    # count_distinct: valid for a code column (averaging one is refused by the role rule, correctly)
    return tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "count_distinct", "column": col}}, df)


def test_exact_match_is_used():
    assert metric("Product_Code", frame("Product_Code"))["args"]["column"] == "Product_Code"


def test_exact_match_wins_over_a_case_variant():
    df = frame("Product_Code", "product_code")
    assert metric("product_code", df)["args"]["column"] == "product_code"
    assert metric("Product_Code", df)["args"]["column"] == "Product_Code"


def test_unique_case_insensitive_match_resolves_to_the_dataset_name():
    assert metric("product_code", frame("Product_Code"))["args"]["column"] == "Product_Code"
    call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": "count", "filters": [
        {"column": "REGION", "op": "==", "value": "a"}]}}, frame("Product_Code"))
    assert call["args"]["filters"][0]["column"] == "region"
    lr = tools.validate_call({"tool": "linear_regression", "args": {"target": "units_sold", "features": ["UNITS"]}},
                             frame("Units_Sold", "units"))
    assert lr["args"] == {"target": "Units_Sold", "features": ["units"]}


def test_ambiguous_case_insensitive_match_is_rejected():
    with pytest.raises(tools.InvalidCall, match="ambiguous"):
        metric("PRODUCT_CODE", frame("Product_Code", "product_code"))
    with pytest.raises(tools.InvalidCall):
        metric("PRODUCT_CODE", frame("Product_Code", "product_CODE", "PRODUCT_code"))


def test_unknown_column_is_still_rejected():
    with pytest.raises(tools.InvalidCall, match="no column"):
        metric("product_id", frame("Product_Code"))


def test_clarify_options_are_resolved_to_dataset_names():
    df = frame("Product_Code", "units")
    q = "Which column should I predict?"
    kept, dropped = plan_checks.canonical_options(["product_code", "Units", "week"], df, q)
    assert kept == ["Product_Code", "units"] and dropped == ["week"]
    kept, dropped = plan_checks.canonical_options(["PRODUCT_CODE"], frame("Product_Code", "product_code"), q)
    assert kept == [] and dropped == ["PRODUCT_CODE"]


def test_clarify_options_that_are_values_not_columns_are_kept():
    kept, dropped = plan_checks.canonical_options(["North", "South"], frame("units"), "Which market do you mean?")
    assert kept == ["North", "South"] and dropped == []
