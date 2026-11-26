"""The Copilot renders a compound answer's forecast chart: actuals, forecast, range and the target line."""
import pandas as pd

from style import copilot_ui as ui

SPEC = {"type": "forecast", "title": "hours_worked: actual and forecast — employee = Rahul", "x": "period", "y": "value",
        "y_title": "hours_worked (hours per month)", "series": "kind", "target": 300.0,
        "target_label": "target 300 hours per month",
        "data": [{"period": f"2025-{m:02d}", "value": 280.0 + m, "kind": "actual", "lower": None, "upper": None} for m in range(1, 13)]
        + [{"period": "2026-01", "value": 296.4, "kind": "forecast", "lower": 289.9, "upper": 304.5},
           {"period": "2026-02", "value": 299.4, "kind": "forecast", "lower": 292.6, "upper": 307.8}]}


def marks(d):
    return [lay["mark"] if isinstance(lay["mark"], str) else lay["mark"]["type"] for lay in d["layer"]]


def test_forecast_chart_has_every_requested_element():
    d = ui.chart(SPEC, "light").to_dict()
    assert marks(d) == ["area", "line", "line", "rule", "text"]
    dashed = [lay for lay in d["layer"] if isinstance(lay["mark"], dict) and lay["mark"].get("strokeDash")]
    assert len(dashed) == 2                                  # the forecast line and the target rule are dashed
    assert d["layer"][1]["encoding"]["y"]["title"] == "hours_worked (hours per month)"


def test_forecast_chart_without_target_or_range_still_draws():
    spec = {**SPEC, "data": [dict(x, lower=None, upper=None) for x in SPEC["data"]]}
    spec.pop("target")
    d = ui.chart(spec, "dark").to_dict()
    assert marks(d) == ["line", "line"]


def test_actual_and_forecast_are_kept_apart():
    df = pd.DataFrame(SPEC["data"])
    assert set(df["kind"]) == {"actual", "forecast"} and (df[df.kind == "forecast"].period > df[df.kind == "actual"].period.max()).all()
