"""Chart Studio: every chart type must build from every common source figure,
and the code it shows must be the code that built the figure."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import pytest

import chart_studio as cs

_DF = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr"] * 2,
    "Revenue": [10, 12.5, 9, 14, 7, 8, 11, 6],
    "Region": ["North"] * 4 + ["South"] * 4,
})

SOURCES = {
    "grouped_bar": px.bar(_DF, x="Month", y="Revenue", color="Region"),
    "horizontal_bar": px.bar(_DF, y="Month", x="Revenue", orientation="h"),
    "line": px.line(_DF, x="Month", y="Revenue", color="Region"),
    "pie": px.pie(_DF, names="Region", values="Revenue"),
    "histogram": px.histogram(_DF, x="Revenue"),
    "heatmap": px.imshow([[1, 2], [3, 4]], x=["a", "b"], y=["r1", "r2"]),
}


def _data(name: str) -> cs.ChartData:
    return cs.extract_chart_data(json.loads(SOURCES[name].to_json()))


def test_there_are_forty_chart_types():
    assert len(cs.CHART_TYPES) == 40
    assert sum(len(cs.types_in_group(g)) for g in cs.GROUPS) == 40


def test_extract_decodes_plotly6_binary_arrays():
    data = _data("grouped_bar")
    assert list(data.df.columns) == ["series", "x", "y"]
    assert data.df["y"].tolist() == [10, 12.5, 9, 14, 7, 8, 11, 6]
    assert set(data.df["series"]) == {"North", "South"}
    assert (data.x_label, data.y_label) == ("Month", "Revenue")


def test_horizontal_bar_is_read_as_category_then_value():
    data = _data("horizontal_bar")
    assert data.df["x"].tolist()[:2] == ["Jan", "Feb"]
    assert data.df["y"].tolist()[:2] == [10, 12.5]
    assert cs.detect_studio_type(data) == "hbar"


@pytest.mark.parametrize("source", list(SOURCES))
@pytest.mark.parametrize("chart_type", list(cs.CHART_TYPES))
def test_every_type_builds_from_every_source(source, chart_type):
    data = _data(source)
    style = {"effect": "neon" if chart_type.endswith("line") else "glass", "labels": True}
    build = cs.build_chart(data, chart_type, style, "Revenue")
    assert build.fig.data, "figure has no traces"
    build.fig.to_json()
    cs.calculation_table(data, chart_type, style)


@pytest.mark.parametrize("effect", list(cs.EFFECTS))
def test_shown_code_reproduces_the_figure(effect):
    data = _data("grouped_bar")
    build = cs.build_chart(data, "column", {"effect": effect, "palette": "Light blue"}, "Revenue")
    namespace: dict = {}
    exec(build.code.replace("fig.show()", ""), namespace)  # noqa: S102
    assert namespace["fig"].to_json() == build.fig.to_json()


def test_strings_from_the_figure_cannot_become_code():
    fig = px.bar(x=['"); import os; os.system("echo pwned"); ("'], y=[1])
    data = cs.extract_chart_data(json.loads(fig.to_json()))
    build = cs.build_chart(data, "column", {}, 'title"); import os; ("')
    assert "os.system" in build.fig.data[0].x[0]  # survived as plain data


def test_normalize_style_rejects_unsafe_values():
    style = cs.normalize_style({"color": "red;}</style><script>", "effect": "explode",
                                "opacity": 99, "palette": "nope", "labels": "yes"})
    assert style["color"] == cs.DEFAULT_STYLE["color"]
    assert style["effect"] == "solid" and style["palette"] == "SILT"
    assert style["opacity"] == 1.0 and style["labels"] is False


def test_pie_calculation_shares_add_to_100():
    calc, _ = cs.calculation_table(_data("grouped_bar"), "pie")
    assert calc["Share %"].sum() == pytest.approx(100, abs=0.01)


def test_all_negative_values_give_a_clear_error_for_part_to_whole():
    data = cs.extract_chart_data(json.loads(px.bar(x=["a", "b"], y=[-1, -2]).to_json()))
    with pytest.raises(ValueError, match="pie"):
        cs.build_chart(data, "pie", {})
