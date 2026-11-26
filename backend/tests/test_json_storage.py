"""JSON columns must accept whatever an analysis step prints. Generated code writes NaN (a month with no previous
month) and Postgres rejects it, which ended the whole run with "an internal error". These run on SQLite and, in CI,
on Postgres."""
from __future__ import annotations

import json

import numpy as np
from platform_helpers import ask, workspace

from app.database import SessionLocal, _json_serializer
from app.models import ExecutionLog
from app.sandbox.runner import RESULT_MARKER, get_runner

# what `print("RESULT_JSON:" + json.dumps(result))` emits for a first month with nothing before it
PRINTED = json.dumps({"monthly": [{"month": "2025-01", "change": float("nan"), "ratio": float("inf")},
                                  {"month": "2025-02", "change": 2.5, "ratio": -float("inf")}], "note": "a\x00b"})
CLEAN = {"monthly": [{"month": "2025-01", "change": None, "ratio": None}, {"month": "2025-02", "change": 2.5, "ratio": None}],
         "note": "ab"}


def test_the_serializer_writes_valid_json():
    def refuse(token):
        raise AssertionError(f"{token} is not JSON")

    out = _json_serializer({"a": float("nan"), "b": [np.float64("inf"), (1.5, np.nan)], "c": np.int64(3), float("nan"): 1})
    assert json.loads(out, parse_constant=refuse) == {"a": None, "b": [None, [1.5, None]], "c": "3", "null": 1}


def test_a_step_result_with_missing_numbers_is_saved(client):
    ws = workspace(client)
    question_id = ask(client, ws, "How did the score change each month?")["question_id"]
    result = get_runner()._parse_stdout(f"{RESULT_MARKER}{PRINTED}")
    assert np.isnan(result["monthly"][0]["change"]), "the sandbox hands the NaN on as printed"

    db = SessionLocal()
    try:
        row = ExecutionLog(question_id=question_id, step_index=0, step_description="monthly change", attempt_number=1,
                           code="", stdout="", stderr="", success=True, result_json=result)
        db.add(row)
        db.commit()
        db.expire_all()
        assert db.get(ExecutionLog, row.id).result_json == CLEAN
    finally:
        db.close()
