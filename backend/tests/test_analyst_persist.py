"""The result register and open objective survive a restart (app/analyst/persist.py) -- for their own team, user,
dataset, document and conversation only."""
from __future__ import annotations

import json
import os
import stat
import time

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator, persist, planner, scope

TODAY = pd.Timestamp("2025-06-10")
KEY = "t1:u1:7:None:conv-a"


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(scope, "today", lambda: TODAY)
    monkeypatch.setenv("ANALYST_MEMORY_DIR", str(tmp_path / "mem"))
    orchestrator._MEMORY.clear()                     # noqa: SLF001
    yield
    orchestrator._MEMORY.clear()                     # noqa: SLF001


def shops(seed=5, n=160):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"outlet": rng.choice(["Quay", "Mall", "Depot"], n),
                         "sold_on": TODAY - pd.to_timedelta(rng.integers(0, 40, n), "D"),
                         "takings": rng.gamma(5, 80, n).round(2)})


def turn(key, q, df):
    mem = orchestrator.memory_for(key, persist=True)
    out = orchestrator.ask(q, df, mem, use_model=False)
    orchestrator.save_memory(key, mem)
    return out


def restart():
    orchestrator._MEMORY.clear()                     # noqa: SLF001 - the process memory is gone


def test_references_and_reruns_work_after_a_restart():
    df = shops()
    turn(KEY, "Total takings for Quay", df)
    turn(KEY, "Total takings for Mall", df)
    restart()
    out = turn(KEY, "Compare that with the previous result.", df)
    r = out["calls"][0]["result"]
    assert out["calls"][0]["tool"] == "compare_results" and (r["a"], r["b"]) == (1, 2)
    assert r["figures"]["value"]["a"] == pytest.approx(df[df.outlet == "Quay"].takings.sum())
    turn(KEY, "Can Quay reach 30000 takings this month?", df)
    restart()
    out = turn(KEY, "What if the target is 35000?", df)
    v = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"]["Quay"]["verdict"]["value"]
    assert v["target"] == 35000, "the open objective was restored and re-run"


@pytest.mark.parametrize("other", ["t1:u2:7:None:conv-a", "t2:u1:7:None:conv-a", "t1:u1:8:None:conv-a",
                                   "t1:u1:7:None:conv-b"])
def test_nothing_is_restored_into_another_team_user_dataset_or_conversation(other):
    df = shops()
    turn(KEY, "Total takings for Quay", df)
    turn(KEY, "Total takings for Mall", df)
    restart()
    out = turn(other, "Compare that with the previous result.", df)
    assert "nothing earlier in this conversation" in out["answer"] and not out["calls"]


def test_a_file_under_another_keys_name_is_not_loaded_and_files_are_private():
    df = shops()
    turn(KEY, "Total takings for Quay", df)
    p = persist.path_for(KEY)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600 and stat.S_IMODE(os.stat(p.parent).st_mode) == 0o700
    other = "t1:u1:7:None:conv-z"
    q = persist.path_for(other)
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(p.read_text())                      # a copied file: its stored key is not this conversation's
    restart()
    assert persist.load(other) == {} and orchestrator.memory_for(other, persist=True) == {}


def test_expired_memory_is_not_restored_and_tests_do_not_write_files():
    df = shops()
    turn(KEY, "Total takings for Quay", df)
    p = persist.path_for(KEY)
    data = json.loads(p.read_text())
    data["saved_at"] = time.time() - (persist.RETENTION_DAYS + 1) * 86400
    p.write_text(json.dumps(data))
    restart()
    assert persist.load(KEY) == {} and not p.exists()
    mem = orchestrator.memory_for("t9:u9:1:None:c", persist=False)   # in-process use (tests, evaluation) writes nothing
    orchestrator.ask("Total takings for Quay", df, mem, use_model=False)
    assert not persist.path_for("t9:u9:1:None:c").exists()


# ------------------------------------------------------------- corruption and recovery --
@pytest.mark.parametrize("content", ["", "{", '{"version": 1, "key": "t1:u1:7:None:conv-a", "regis', "[1, 2]", "null",
                                     "\xff\xfe garbage", '{"version": 1, "key": "t1:u1:7:None:conv-a", "saved_at": "soon",'
                                     ' "register": []}'])
def test_unreadable_memory_is_never_restored_and_never_crashes(content):
    df = shops()
    p = persist.path_for(KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode("latin-1"))
    restart()
    assert persist.load(KEY) == {}
    out = turn(KEY, "Compare that with the previous result.", df)
    assert "nothing earlier in this conversation" in out["answer"] and not out["calls"]
    out = turn(KEY, "Total takings for Quay", df)                 # the conversation carries on and is saved again
    assert persist.load(KEY)["register"][0]["n"] == 1


def test_malformed_entries_are_dropped_not_half_used():
    p = persist.path_for(KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    good = {"n": 1, "question": "q", "kind": "rules", "figures": {"value": 1.0}, "label": "sum of x", "hypothetical": False}
    for reg in ([good, {"question": "no number"}], [good, {**good, "n": 1}], "not a list", [good, "text"]):
        p.write_text(json.dumps({"version": 1, "key": KEY, "saved_at": time.time(), "register": reg,
                                 "objective": "x"}))
        assert persist.load(KEY) == {}, reg
    p.write_text(json.dumps({"version": 1, "key": KEY, "saved_at": time.time(), "register": [good]}))
    assert persist.load(KEY) == {"register": [good]}


def test_a_failed_write_keeps_the_previous_memory(monkeypatch):
    df = shops()
    turn(KEY, "Total takings for Quay", df)
    before = persist.path_for(KEY).read_text()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(persist.os, "replace", boom)
    turn(KEY, "Total takings for Mall", df)                       # the turn is answered; only the save fails
    assert persist.path_for(KEY).read_text() == before
    assert not list(persist.path_for(KEY).parent.glob("*.tmp")), "no temporary file is left behind"


def test_a_withheld_answer_is_not_kept_and_a_refinement_does_not_reuse_it(monkeypatch):
    from app.analyst import nl_planner, tools
    df = shops()
    real = tools._EXEC["compute_metric"]

    def wrong(d, a, m):
        r = real(d, a, m)
        for g in r.get("groups", []):
            g["value"] = g["value"] * 1.1
        return r
    monkeypatch.setitem(tools._EXEC, "compute_metric", wrong)
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)
    out = turn(KEY, "Total takings by outlet", df)
    assert out["verify_status"] == "withheld"
    restart()
    assert "register" not in persist.load(KEY), "a withheld answer is not a result later turns can refer to"
    out = turn(KEY, "Go back to the first result.", df)
    assert not out["calls"]


def test_a_clarification_keeps_its_context_without_inventing_a_result():
    df = shops().assign(staff_hours=lambda d: (d.takings / 40).round(1))
    mem = orchestrator.memory_for(KEY, persist=True)
    first = orchestrator.ask("Which outlet performs worst and why?", df, mem, use_model=False)
    assert first["plan"]["type"] == "clarify" and not first["calls"]
    assert not mem.get("register"), "a question asked back is not a result"
    out = orchestrator.ask("takings", df, mem, use_model=False)
    want = df.groupby("outlet").takings
    assert out["calls"] and any(f"{v:,.2f}" in out["answer"] for v in list(want.sum()) + list(want.mean()))
    assert len(mem["register"]) == 1
