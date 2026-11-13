"""The interpretation record (app/interpretation.py): an additive account of how
the copilot read a message -- which words, which rule, which reference -- that
changes nothing about the reading itself.

Datasets here are invented for the tests; none comes from the Spider benchmark."""
from __future__ import annotations

import copy

import pandas as pd
import pytest
from platform_helpers import workspace

from app import copilot
from app import interpretation as interp_mod
from app.profiling import profile_dataframe

SINGERS = {"Name": ["A", "B", "C", "D"], "Country": ["France", "France", "Spain", "Italy"], "Age": [30, 40, 50, 25],
           "Song_Name": ["w", "x", "y", "z"], "Hours": [10, 20, 30, 40], "active": ["Y", "N", "Y", "Y"]}
PETS = {"PetType": ["dog", "cat", "dog", "bird"], "weight": [12.0, 4.0, 9.5, 0.3], "pet_age": [3, 5, 1, 2]}
SALES = {"order_date": pd.to_datetime(["2025-01-05", "2025-03-02", "2025-03-20", "2026-05-05"]),
         "region": ["West", "East", "West", "North"], "revenue": [1.0, 2.0, 3.0, 4.0], "cost": [0.5, 1.0, 1.0, 2.0],
         "units": [1, 2, 3, 4]}
TWO_DATES = {**SALES, "ship_date": pd.to_datetime(["2025-01-06", "2025-03-03", "2025-03-21", "2026-05-06"])}
CONCERTS = {"Year": [2014, 2015, 2015, 2016], "Stadium": ["North", "East", "North", "West"], "Attendance": [10, 20, 30, 40]}
TWO_YEARS = {**CONCERTS, "Founded_year": [1990, 1991, 1992, 1993]}
LANGUAGES = {"Country": ["France", "Spain"], "Language": ["French", "Spanish"], "speakers": [1, 2]}
BIRTHPLACES = {"Country": ["Spain", "Italy"], "Birthplace": ["Spain", "France"], "Age": [1, 2]}
SCORES = {"home_score": [1, 2], "away_score": [3, 4], "team": ["a", "b"]}
LARGE = {"district": [f"D{i}" for i in range(399)] + ["Gelderland"], "attendance": list(range(400))}
NET = {"id": 7, "name": "Net revenue", "formula": "sum(revenue) - sum(cost)", "filters": [], "current_version": 2,
       "matched_term": "net revenue", "unit": ""}


def _read(message: str, rows: dict, dataset_name: str | None, resolved: list | None = None, traced: bool = True):
    df = pd.DataFrame(rows)
    profile = profile_dataframe(df)
    trace = interp_mod.Trace(message) if traced else None
    out = copilot.interpret(message, copilot.empty_state(), profile, copilot.category_index(df, profile),
                            copy.deepcopy(resolved or []), dataset_name=dataset_name, trace=trace)
    return out, trace


def _interpret(message, rows, dataset_name=None, resolved=None):
    out, trace = _read(message, rows, dataset_name, resolved)
    return out, interp_mod.build(message, out, trace)


# Every reading rule, each clarification and the ways of declining.
CASES = [
    ("How many singers do we have?", SINGERS, "singer.csv", None, "resolved"),
    ("How many singers are older than 35?", SINGERS, "singer.csv", None, "resolved"),
    ("What is the average age of singers whose age is at least 30?", SINGERS, "singer.csv", None, "resolved"),
    ("How many singers have more than 20 hours?", SINGERS, "singer.csv", None, "resolved"),
    ("How many singers have greater age than 30?", SINGERS, "singer.csv", None, "resolved"),
    ("How many singers are 30 years old?", SINGERS, "singer.csv", None, "resolved"),
    ("What is the weight of the heaviest dog?", PETS, "pets.csv", None, "resolved"),
    ("Show me Total_Revenue where region = West", SALES, "sales.csv", None, "resolved"),
    ("total revenue where region is East", SALES, "sales.csv", None, "resolved"),
    ("What is the total revenue excluding West?", SALES, "sales.csv", None, "resolved"),
    ("total revenue for East or West", SALES, "sales.csv", None, "resolved"),
    ("How many French singers are there?", SINGERS, "singer.csv", None, "resolved"),
    ("total attendance of the Gelderland district", LARGE, "events.csv", None, "resolved"),
    ("total revenue in the region East", SALES, "sales.csv", None, "resolved"),
    ("total revenue by region", SALES, "sales.csv", None, "resolved"),
    ("total revenue across region", SALES, "sales.csv", None, "resolved"),
    ("What is the average number of units?", SALES, "sales.csv", None, "resolved"),
    ("What is the total number of hours?", SINGERS, "singer.csv", None, "resolved"),
    ("How many hours in total?", SINGERS, "singer.csv", None, "resolved"),
    ("What are the average, minimum and maximum age of all singers?", SINGERS, "singer.csv", None, "resolved"),
    ("Count the number of different countries and song names.", SINGERS, "singer.csv", None, "resolved"),
    ("total revenue since 2025-03-01", SALES, "sales.csv", None, "resolved"),
    ("total revenue in March 2025", SALES, "sales.csv", None, "resolved"),
    ("total revenue in 2025", SALES, "sales.csv", None, "resolved"),
    ("total attendance in 2015", CONCERTS, "concert.csv", None, "resolved"),
    ("net revenue", SALES, "sales.csv", [NET], "resolved"),
    ("total net revenue by region", SALES, "sales.csv", [NET], "resolved"),
    ("reset", SALES, "sales.csv", None, "resolved"),
    ("use the latest data", SALES, "sales.csv", None, "resolved"),
    ("clear filters", SALES, "sales.csv", None, "resolved"),
    ("remove the filters on region", SALES, "sales.csv", None, "resolved"),
    ("drill up", SALES, "sales.csv", None, "resolved"),
    ("How many hours?", SINGERS, "singer.csv", None, "needs_clarification"),
    ("total attendance in 2015", TWO_YEARS, "concert.csv", None, "needs_clarification"),
    ("total revenue since 2025-03-01", TWO_DATES, "sales.csv", None, "needs_clarification"),
    ("total attendance since 2015-03-01", CONCERTS, "concert.csv", None, "needs_clarification"),
    ("total speakers of French countries", LANGUAGES, "languages.csv", None, "needs_clarification"),
    ("average age for Spain", BIRTHPLACES, "people.csv", None, "needs_clarification"),
    ("total score", SCORES, "games.csv", None, "needs_clarification"),
    ("Why did revenue drop?", SALES, "sales.csv", None, "unmapped"),
    ("Count the singers who are active.", SINGERS, "singer.csv", None, "unmapped"),
    ("total attendance of the D1 district and D2 district", LARGE, "events.csv", None, "unmapped"),
]


@pytest.mark.parametrize("message, rows, name, resolved, status", CASES)
def test_the_record_leaves_the_reading_unchanged_and_accounts_for_it(message, rows, name, resolved, status):
    plain, _ = _read(message, rows, name, resolved, traced=False)
    traced, trace = _read(message, rows, name, resolved)
    assert traced == plain, "the trace only observes"
    interp = interp_mod.build(message, traced, trace)
    assert interp["status"] == status
    assert interp_mod.validate(interp, message, traced.get("operations")) == []
    assert interp["trace"] == {"complete": True, "realigned": [], "error": None}
    if status == "resolved":
        assert interp_mod.to_operations(interp) == traced["operations"], "converts back to exactly the operations"
        items = interp["measures"] + interp["filters"] + interp["group_by"] + interp["state_ops"]
        assert items and all(item["evidence"] for item in items), "every reading has its words"
        assert interp["confidence"] in interp_mod.TIERS
    elif status == "needs_clarification":
        (amb,) = interp["ambiguities"]
        assert len(amb["candidates"]) == len(traced["clarification"]["options"])
        assert [c["label"] for c in amb["candidates"]] == [o["label"] for o in traced["clarification"]["options"]]
        assert interp["confidence"] is None
    else:
        assert interp["confidence"] is None and not interp["ambiguities"]


def _evidence(interp: dict, kind: str, role: str, index: int = 0) -> list[dict]:
    return [e for e in interp[kind][index]["evidence"] if e["role"] == role]


@pytest.mark.parametrize("message, rows, name, kind, index, role, text, rewritten", [
    # an adjective that names its quantity: the column is read from "older"
    ("How many singers are older than 35?", SINGERS, "singer.csv", "filters", 0, "column", "older", False),
    ("How many singers are older than 35?", SINGERS, "singer.csv", "filters", 0, "value", "35", False),
    ("How many singers are older than 35?", SINGERS, "singer.csv", "filters", 0, "operator", "older than", False),
    # the rows were counted because the file is singers: the original word, rewritten to "rows"
    ("How many singers are older than 35?", SINGERS, "singer.csv", "measures", 0, "rows", "singers", True),
    # a word consumed earlier ("French") is not part of what a later rewrite stands for
    ("How many French singers are there?", SINGERS, "singer.csv", "measures", 0, "rows", "singers", True),
    ("How many French singers are there?", SINGERS, "singer.csv", "filters", 0, "value", "French", False),
    # spans are the message's own characters: case and separators kept
    ("Show me Total_Revenue where region = West", SALES, "sales.csv", "measures", 0, "column", "Revenue", False),
    ("Show me Total_Revenue where region = West", SALES, "sales.csv", "measures", 0, "aggregation", "Total", False),
    ("Show me Total_Revenue where region = West", SALES, "sales.csv", "filters", 0, "operator", "=", False),
    ("  total   revenue  by   region", SALES, "sales.csv", "group_by", 0, "column", "region", False),
    ("What's the total revenue -- by region?", SALES, "sales.csv", "group_by", 0, "group_keyword", "by", False),
    ("What is the weight of the heaviest dog?", PETS, "pets.csv", "measures", 0, "column",
     "the weight of the heaviest", True),
    ("total revenue in March 2025", SALES, "sales.csv", "filters", 0, "time_range", "in March 2025", False),
    ("total attendance in 2015", CONCERTS, "concert.csv", "filters", 0, "value", "2015", False),
    ("total attendance of the Gelderland district", LARGE, "events.csv", "filters", 0, "value",
     "Gelderland district", False),
    # a segment that names no column takes the next one's: the evidence is that column's words
    ("What are the average, minimum and maximum age of all singers?", SINGERS, "singer.csv", "measures", 0,
     "column", "age", True),
    ("Count the number of different countries and song names.", SINGERS, "singer.csv", "measures", 1, "column",
     "song names", False),
    ("What is the total revenue excluding West?", SALES, "sales.csv", "filters", 0, "operator", "excluding", False),
    ("total net revenue by region", SALES, "sales.csv", "measures", 0, "metric", "net revenue", False),
])
def test_evidence_spans_are_the_original_words(message, rows, name, kind, index, role, text, rewritten):
    _, interp = _interpret(message, rows, name, [NET] if "net revenue" in message else None)
    (ev,) = _evidence(interp, kind, role, index)[:1]
    assert ev["text"] == text
    assert " ".join(message[s:e] for s, e in ev["spans"]) == text
    assert bool(ev.get("via")) is rewritten


@pytest.mark.parametrize("message, rows, name, resolved, confidence", [
    ("total revenue", SALES, "sales.csv", None, "lexical"),
    ("total revenue by region", SALES, "sales.csv", None, "lexical"),
    ("net revenue", SALES, "sales.csv", [NET], "definitional"),
    ("How many singers do we have?", SINGERS, "singer.csv", None, "derived"),     # read from the file name
    ("How many French singers are there?", SINGERS, "singer.csv", None, "derived"),
    ("What is the weight of the heaviest dog?", PETS, "pets.csv", None, "derived"),
])
def test_confidence_is_the_weakest_evidence_label(message, rows, name, resolved, confidence):
    _, interp = _interpret(message, rows, name, resolved)
    assert interp["confidence"] == confidence


def test_an_approved_metric_is_referenced_by_id_and_version():
    _, interp = _interpret("net revenue", SALES, "sales.csv", [NET])
    (m,) = interp["measures"]
    assert m["metric_ref"] == {"metric_id": 7, "version": 2, "name": "Net revenue", "term": "net revenue"}
    assert (m["aggregation"], m["columns"]) == ("formula", ["cost", "revenue"])


def test_several_measures_keep_their_order_and_one_operation():
    out, interp = _interpret("What are the average, minimum and maximum age of all singers?", SINGERS, "singer.csv")
    assert [(m["part"], m["aggregation"], m["column"]) for m in interp["measures"]] == \
        [(0, "avg", "Age"), (1, "min", "Age"), (2, "max", "Age")]
    assert {m["op_index"] for m in interp["measures"]} == {0}
    assert interp_mod.to_operations(interp) == out["operations"]


def test_unread_words_are_listed_with_their_spans():
    message = "Why did revenue drop?"
    _, interp = _interpret(message, SALES, "sales.csv")
    assert [u["text"] for u in interp["unread"]] == ["Why", "did", "revenue", "drop"]
    assert all(" ".join(message[s:e] for s, e in u["spans"]) == u["text"] for u in interp["unread"])


def _valid() -> tuple[str, dict, list]:
    message = "How many singers are older than 35?"
    out, interp = _interpret(message, SINGERS, "singer.csv")
    assert interp_mod.validate(interp, message, out["operations"]) == []
    return message, interp, out["operations"]


def _shift_span(i):
    i["filters"][0]["evidence"][0]["spans"][0][0] += 1


def _span_outside(i):
    i["filters"][0]["evidence"][0]["spans"] = [[0, 999]]


def _wrong_literal(i):
    i["filters"][0]["value"] = 36.0


def _measure_with_value(i):
    i["measures"][0]["value"] = 4


def _unknown_operator(i):
    i["filters"][0]["operator"] = "~"


def _overstated_confidence(i):
    i["confidence"] = "lexical"


def _no_operations(i):
    i["measures"], i["filters"] = [], []


def _unknown_source(i):
    i["filters"][0]["evidence"][0]["source"] = "guess"


@pytest.mark.parametrize("corrupt, fragment", [
    (_shift_span, "the message at"),
    (_span_outside, "not an ordered range inside the message"),
    (_wrong_literal, "is not written in the message"),
    (_measure_with_value, "numbers come only from the engine"),
    (_unknown_operator, "is not supported"),
    (_overstated_confidence, "the weakest evidence is 'derived'"),
    (_no_operations, "a resolved reading has no operations"),
    (_unknown_source, "unknown evidence source"),
])
def test_validation_catches_a_record_that_misstates_the_reading(corrupt, fragment):
    message, interp, ops = _valid()
    corrupt(interp)
    assert any(fragment in p for p in interp_mod.validate(interp, message)), interp_mod.validate(interp, message)


def test_validation_catches_a_record_that_does_not_convert_back():
    message, interp, ops = _valid()
    other = copy.deepcopy(ops)
    other[0]["formula"] = 'count_distinct("Age")'
    assert any("converts to" in p for p in interp_mod.validate(interp, message, other))


def test_a_fault_in_the_trace_never_changes_the_reading(monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("injected")
    monkeypatch.setattr(interp_mod.Trace, "_fragments", broken)
    message = "How many singers are older than 35?"
    plain, _ = _read(message, SINGERS, "singer.csv", traced=False)
    traced, trace = _read(message, SINGERS, "singer.csv")
    assert traced == plain
    interp = interp_mod.build(message, traced, trace)
    assert interp["trace"]["complete"] is False and "injected" in interp["trace"]["error"]
    assert interp["confidence"] is None
    assert interp_mod.to_operations(interp) == plain["operations"], "the operations are still recorded exactly"


def test_without_a_trace_the_result_has_no_record():
    out, _ = _read("Why did revenue drop?", SALES, "sales.csv", traced=False)
    assert out == {"unmapped": True}


# ------------------------------------------------------------------ the API --

def _turn(client, ws, message: str) -> dict:
    sid = client.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=ws["h"]).json()["id"]
    return client.post(f"/api/copilot/sessions/{sid}/messages", json={"message": message}, headers=ws["h"]).json()


@pytest.mark.parametrize("message, status, action_status", [
    ("total revenue by region", "resolved", "applied"),
    ("How many units?", "needs_clarification", "clarification"),
    ("Why did revenue drop?", "unmapped", "not_understood"),
])
def test_a_copilot_turn_records_its_interpretation(client, message, status, action_status):
    ws = workspace(client)
    turn = _turn(client, ws, message)
    action = turn["action"]
    assert action["status"] == action_status
    interp = action["interpretation"]
    assert interp["status"] == status and interp["interpreted_by"] == "rules"
    assert interp_mod.validate(interp, message, action.get("operations")) == []
    assert interp["trace"]["complete"] is True
    if status == "resolved":
        assert interp_mod.to_operations(interp) == action["operations"]
        assert turn["evidence"]["computed_by"] == "deterministic metric engine"


def test_a_definition_clash_is_recorded_as_a_semantic_clarification(client):
    ws = workspace(client)
    ids = []
    for name, formula in (("Gross margin", "sum(revenue) - sum(cost)"), ("Unit margin", "sum(revenue) / sum(units)")):
        m = client.post("/api/semantic/metrics", json={"name": name, "formula": formula, "aliases": ["margin"]},
                        headers=ws["h"]).json()
        assert client.post(f"/api/semantic/metrics/{m['id']}/versions/1/approve", headers=ws["h"]).status_code == 200
        ids.append(m["id"])
    message = "What is our Margin?"
    turn = _turn(client, ws, message)
    interp = turn["action"]["interpretation"]
    assert turn["action"]["status"] == "clarification"
    assert (interp["status"], interp["interpreted_by"]) == ("needs_clarification", "semantic")
    (amb,) = interp["ambiguities"]
    assert [c["binding"]["metric_id"] for c in amb["candidates"]] == ids
    assert [message[s:e] for s, e in amb["spans"]] == ["Margin"]
    assert interp_mod.validate(interp, message) == []
