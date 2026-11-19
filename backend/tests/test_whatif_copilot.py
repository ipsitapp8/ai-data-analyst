"""Universal what-if through the Silt Copilot: same engine, structured results, asks for missing values, steps aside otherwise."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator
from app.whatif import service

SMALL = pd.DataFrame({"Student ID": range(1, 11), "Attendance": [60, 70, 75, 80, 90, 95, 75, 85, 70, 80],
                      "Marks": [30, 40, 45, 50, 55, 60, 40, 50, 45, 60]})
SALES = pd.DataFrame({"region": ["East", "West"] * 4, "units": [10, 20, 30, 40, 50, 60, 70, 80],
                      "revenue": [200, 400, 600, 800, 1000, 1200, 1400, 1600], "cost": [120, 240, 360, 480, 600, 720, 840, 960]})


def ask(message, df=SMALL, memory=None):
    memory = {} if memory is None else memory
    return orchestrator.ask(message, df, memory, use_model=False), memory


def test_copilot_answers_with_the_shared_engine_numbers():
    out, mem = ask("What if passing marks increase from 40 to 50?")
    direct = service.run(service.from_question("What if passing marks increase from 40 to 50?", SMALL)["scenario"], SMALL)
    assert out["plan"]["type"] == "whatif" and out["plan"]["template"] == "edu_pass_marks"
    m = direct["results"]["metrics"][0]
    assert f"{m['baseline']:,.2f} → {m['scenario']:,.2f}" in out["answer"] and "Exact calculation" in out["answer"]
    assert any("shared what-if engine" in n for n in out["notes"]) and mem["whatif_history"]


def test_copilot_asks_for_missing_values_then_runs_and_never_assumes():
    out, mem = ask("What if 100 additional students enroll?")
    assert out["plan"]["type"] == "clarify" and "faculty" in out["answer"].lower() and "will not assume" in out["answer"]
    out2, _ = ask("faculty is 2, seats are 15", memory=mem)
    assert out2["plan"]["type"] == "whatif"
    assert "Students per faculty member: 5.00 → 55.00" in out2["answer"]       # 10 students / 2 -> 110 / 2


def test_copilot_declines_unsupported_and_missing_columns():
    out, _ = ask("What if passing marks increase from 40 to 50?", df=SALES)
    assert "can't answer" in out["answer"] and out["plan"]["type"] == "clarify"
    out, _ = ask("What if we change the treatment protocol?")
    assert "I could not run" in out["answer"] and "clinical" in out["answer"]


def test_copilot_brainstorms_with_feasibility():
    out, _ = ask("Suggest some what-if scenarios I could explore")
    assert out["plan"]["type"] == "whatif_suggestions" and "ready to run" in out["answer"] and "needs a value from you" in out["answer"]


def test_copilot_compares_and_explains_followups():
    mem = {}
    ask("What if passing marks increase from 40 to 50?", memory=mem)
    ask("What if passing marks increase from 40 to 55?", memory=mem)
    cmp, _ = ask("Compare those scenarios", memory=mem)
    assert cmp["plan"]["type"] == "whatif_compare" and "5.00 vs 3.00" in cmp["answer"]
    why, _ = ask("What are the assumptions?", memory=mem)
    assert why["plan"]["type"] == "whatif_explain" and "What this cannot tell you" in why["answer"]


def test_business_sentences_are_left_to_the_existing_path():
    out, _ = ask("What happens if I increase the price by 10%?", df=SALES, memory={"scenario_inputs": {"fixed_cost": 500.0}})
    assert out["plan"]["type"] == "scenario"            # scenario_bridge, unchanged


def test_llm_failure_does_not_matter_to_execution(monkeypatch):
    from app.agents import llm_client
    monkeypatch.setattr(llm_client, "provider_available", lambda *a, **k: False)
    out, _ = ask("What if passing marks increase from 40 to 50?")
    assert out["plan"]["type"] == "whatif"
