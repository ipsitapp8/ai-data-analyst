"""P1 phase 7: the new deliverable charts (comparison bars, growth lines per group, correlation scatter) are drawn from
the executor's own validated specs, and several charts of one compound answer all render in the Copilot page."""
from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from style import copilot_ui as ui
from test_pages import healthy, page  # noqa: F401
from test_pages import api, ws  # noqa: F401

SALES = pd.read_csv(io.BytesIO(__import__("platform_helpers").SALES_CSV))


@pytest.fixture()
def frame(tmp_path):
    from app.analyst.data import load_table
    p = tmp_path / "sales.csv"
    SALES.to_csv(p, index=False)
    return load_table(str(p))


def _run(kind, frame, **kw):
    from app.analyst import compound
    from app.analyst.deliverables import Deliverable
    d = Deliverable(id=1, kind=kind, clause="q", order=0, **kw)
    ctx = compound.execute([d, Deliverable(id=2, kind="chart", clause="q", order=0, chart_for=1, depends_on=[1])], frame)
    assert d.status == "done", d.detail
    return ctx["results"][2]["chart_spec"], ctx["results"][1]


def test_growth_lines_are_one_line_per_group(frame):
    spec, res = _run("growth", frame, measures=["revenue"], group_by="region", groups=["East", "West"])
    d = ui.chart(spec, "light").to_dict()
    assert d["encoding"]["color"]["field"] == "group" and {r["group"] for r in spec["data"]} == {"East", "West"}
    assert d["mark"]["type"] == "line"


def test_comparison_bars_match_the_computed_values(frame):
    spec, res = _run("compare", frame, measures=["revenue"], group_by="region", groups=["East", "West"], stat="avg")
    want = SALES.groupby("region").revenue.mean()
    assert {r["region"]: r["revenue"] for r in spec["data"]} == pytest.approx({g: want[g] for g in ("East", "West")})
    assert ui.chart(spec, "dark").to_dict()["mark"]["type"] == "bar"


def test_correlation_draws_a_scatter(frame):
    spec, res = _run("correlation", frame, measures=["units", "revenue"])
    assert ui.chart(spec, "light").to_dict()["mark"]["type"] == "circle" and len(spec["data"]) == len(SALES)


PLAN = {"tasks": [
    {"do": "compare", "text": "Compare revenue between East and West", "col": "revenue", "by": "region", "groups": ["East", "West"]},
    {"do": "chart", "text": "plot the comparison", "of": 0},
    {"do": "growth", "text": "revenue growth by region", "col": "revenue", "by": "region"},
    {"do": "chart", "text": "as lines", "of": 2},
    {"do": "forecast", "text": "forecast Widget revenue for 2 months", "col": "revenue", "where": {"product": "Widget"}, "horizon": 2},
    {"do": "chart", "text": "graph it", "of": 4}]}


def test_page_renders_every_chart_of_a_compound_answer(api, ws, monkeypatch):
    from app.analyst import decomposer, planner
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(decomposer, "model_chat", lambda msgs, max_tokens=900: json.dumps(PLAN))
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no planner here")))
    at = healthy(page("12_Copilot.py", ws))
    at.chat_input[0].set_value("Compare revenue between East and West and plot the comparison, revenue growth by region "
                               "as lines, and forecast Widget revenue for 2 months - graph it.").run()
    healthy(at)
    charts = at.get("vega_lite_chart")
    assert len(charts) == 3, f"three requested charts, got {len(charts)}"
    text = "\n".join(str(m.value) for m in at.markdown)
    assert "Comparison of revenue between East and West" in text and "Growth of revenue by region" in text
    assert "All 6 requested items are answered" in text
