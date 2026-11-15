"""Suggested questions for the Home page: built from real column names, never invented values."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from home_dynamic import GENERIC_QUESTIONS, dataset_questions, has_active_run, suggestions  # noqa: E402


def _ds(cols, rows=500, id=1, at="2026-01-01"):
    return {"id": id, "row_count": rows, "uploaded_at": at, "profile": {"columns": cols}}


SALES = [
    {"name": "order_id", "kind": "numeric", "unique_count": 500},
    {"name": "order_date", "kind": "datetime", "unique_count": 300},
    {"name": "region", "kind": "categorical", "unique_count": 4},
    {"name": "total_revenue", "kind": "numeric", "unique_count": 480},
    {"name": "units_sold", "kind": "numeric", "unique_count": 40},
]


def test_questions_use_real_columns_and_skip_identifiers():
    qs = dataset_questions(_ds(SALES))
    assert "Which region has the highest total revenue?" in qs
    assert "How has total revenue changed over time?" in qs
    assert not any("order id" in q for q in qs), "an id column is not something to total"


def test_no_usable_columns_gives_no_questions_and_falls_back_to_generic():
    assert dataset_questions(_ds([{"name": "notes", "kind": "text", "unique_count": 500}])) == []
    got = suggestions([_ds([{"name": "notes", "kind": "text", "unique_count": 500}])])
    assert [q for _, q in got] == GENERIC_QUESTIONS and all(i is None for i, _ in got)
    assert [q for _, q in suggestions([])] == GENERIC_QUESTIONS


def test_newest_dataset_comes_first_and_carries_its_id():
    old = _ds(SALES, id=1, at="2026-01-01")
    new = _ds([{"name": "city", "kind": "categorical", "unique_count": 5},
               {"name": "price", "kind": "numeric", "unique_count": 50}], id=2, at="2026-06-01")
    got = suggestions([old, new])
    assert got[0] == (2, "Which city has the highest price?")
    assert len(got) == 4 and len({q for _, q in got}) == 4


def test_a_running_analysis_is_detected():
    assert has_active_run([{"status": "verified"}, {"status": "running"}])
    assert not has_active_run([{"status": "verified"}, {"status": "failed"}])
    assert not has_active_run([])
