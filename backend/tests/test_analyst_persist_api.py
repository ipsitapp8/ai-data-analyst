"""Persistent conversational memory through the real API route (/api/analyst/ask): what survives a restart, and
that it never crosses users, teams, datasets, documents or conversations. A restart is simulated by clearing the
process memory; the memory directory is a temporary one per test."""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator, persist
from tests.platform_helpers import add_member, workspace


def sales_csv(seed=3, n=120) -> bytes:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"department": rng.choice(["Retail", "Online", "Wholesale"], n),
                       "sold_on": pd.Timestamp("2026-11-14") - pd.to_timedelta(rng.integers(0, 50, n), "D"),
                       "revenue": rng.gamma(4, 60, n).round(2)})
    return df.to_csv(index=False).encode()


@pytest.fixture(autouse=True)
def memory_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYST_MEMORY_DIR", str(tmp_path / "mem"))
    orchestrator._MEMORY.clear()                     # noqa: SLF001
    yield tmp_path / "mem"
    orchestrator._MEMORY.clear()                     # noqa: SLF001


def ask(client, headers, ds, q, conv="conv-a", **extra):
    body = {"dataset_id": ds, "message": q, "use_model": False, **extra}
    if conv is not None:
        body["conversation_id"] = conv
    r = client.post("/api/analyst/ask", json=body, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def restart():
    orchestrator._MEMORY.clear()                     # noqa: SLF001


def total(csv: bytes) -> float:
    import io
    return float(pd.read_csv(io.BytesIO(csv)).revenue.sum())


def test_a_conversation_is_restored_after_a_restart_through_the_api(client):
    csv = sales_csv()
    ws = workspace(client, csv=csv, filename="sales.csv")
    ask(client, ws["h"], ws["ds"], "What is the total revenue?")
    ask(client, ws["h"], ws["ds"], "What is the average revenue by department?")
    restart()
    out = ask(client, ws["h"], ws["ds"], "Go back to the first result.")
    assert out["calls"][0]["tool"] == "recall_result"
    assert out["calls"][0]["result"]["figures"]["value"] == pytest.approx(total(csv))
    out = ask(client, ws["h"], ws["ds"], "Compare that with the previous result.")
    assert (out["calls"][0]["result"]["a"], out["calls"][0]["result"]["b"]) == (1, 2), \
        "'the previous result' is the previous computed result, not the recall just shown"


@pytest.mark.parametrize("who", ["another user of the same team", "another team", "another dataset",
                                 "another conversation", "no conversation id"])
def test_memory_never_crosses_sessions(client, who):
    csv = sales_csv()
    ws = workspace(client, csv=csv, filename="sales.csv")
    ask(client, ws["h"], ws["ds"], "What is the total revenue?")
    ask(client, ws["h"], ws["ds"], "What is the average revenue by department?")
    restart()
    headers, ds, conv = ws["h"], ws["ds"], "conv-a"
    if who == "another user of the same team":
        headers = add_member(client, ws)
    elif who == "another team":
        other = workspace(client, csv=csv, filename="sales.csv")
        headers, ds = other["h"], other["ds"]
    elif who == "another dataset":
        ds = client.post("/api/datasets/upload", files={"file": ("b.csv", sales_csv(9), "text/csv")},
                         headers=ws["h"]).json()["id"]
    elif who == "another conversation":
        conv = "conv-b"
    else:
        conv = None
    out = ask(client, headers, ds, "Compare that with the previous result.", conv=conv)
    assert not out["calls"] and "nothing earlier in this conversation" in out["answer"], who
    out = ask(client, headers, ds, "Go back to the first result.", conv=conv)
    assert not out["calls"], who


def test_a_request_without_a_conversation_id_is_not_written_to_disk(client, memory_dir):
    ws = workspace(client, csv=sales_csv(), filename="sales.csv")
    ask(client, ws["h"], ws["ds"], "What is the total revenue?", conv=None)
    assert not memory_dir.exists() or not any(memory_dir.rglob("*.json"))
    ask(client, ws["h"], ws["ds"], "What is the total revenue?")
    assert len(list(memory_dir.rglob("*.json"))) == 1


def test_ids_containing_separators_cannot_stand_for_other_parts():
    a = orchestrator.memory_key(1, 2, None, "x:y", "z")
    b = orchestrator.memory_key(1, 2, None, "x", "y:z")
    assert a != b and persist.path_for(a) != persist.path_for(b)
    assert orchestrator.memory_key(1, 2, 5, None, "abc123") == "1:2:5:None:abc123", "ordinary ids are unchanged"
    p = persist.path_for(orchestrator.memory_key("../../etc", "..", 1, None, "c"))
    assert persist.root() in p.parents, "no id can lead outside the memory directory"


def test_concurrent_turns_in_one_conversation_are_all_kept(client):
    ws = workspace(client, csv=sales_csv(), filename="sales.csv")
    qs = ["What is the total revenue?", "What is the average revenue by department?", "What is the median revenue?",
          "What is the total revenue by department?"]
    errors = []

    def run(q):
        try:
            ask(client, ws["h"], ws["ds"], q)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=run, args=(q,)) for q in qs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    key = orchestrator.memory_key(ws["team"], next(iter(orchestrator._MEMORY)).split(":")[1], ws["ds"], None,  # noqa: SLF001
                                  "conv-a")
    restart()
    reg = persist.load(key)["register"]
    assert [e["n"] for e in reg] == [1, 2, 3, 4] and sorted(e["question"] for e in reg) == sorted(qs)
