"""Chart Studio: re-chart a dashboard figure as any of 40 chart types, restyle it,
and explain exactly how the new chart was built.

Pure functions only (no Streamlit), so everything here is unit-testable:

  extract_chart_data(plotly_json) -> ChartData   the tidy (series, x, y) table the
                                                  original figure actually plotted
  build_chart(data, chart_type, style, title)     -> ChartBuild(fig, code)
  formula_for(chart_type), calculation_table()    -> the math the new chart applies

The figure is produced by exec'ing the very script returned as `code`, so the
Python shown in the Inspect panel is, by construction, the code that drew the
chart. Every value interpolated into that script goes through repr() of a plain
str/int/float/bool/None (see _literal), so nothing from a stored figure or a
teammate's saved style can turn into code.
"""
from __future__ import annotations

import base64
import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------ styles --

PALETTES: dict[str, list[str]] = {
    "SILT": ["#4f8fe0", "#3fb87a", "#e0a83e", "#d1596b", "#9575cd", "#41b8c4"],
    "Light blue": ["#7cc4fa", "#4dabf7", "#a5d8ff", "#339af0", "#d0ebff", "#1c7ed6"],
    "Ocean": ["#0b84a5", "#4fb3bf", "#93d5db", "#1f5f8b", "#6ac1e5", "#2e86ab"],
    "Sunset": ["#ff7c43", "#ffa600", "#f95d6a", "#d45087", "#a05195", "#665191"],
    "Forest": ["#40916c", "#74c69d", "#2d6a4f", "#95d5b2", "#52b788", "#b7e4c7"],
    "Berry": ["#c2185b", "#e91e63", "#8e24aa", "#f06292", "#ab47bc", "#f8bbd0"],
    "Pastel": ["#a0c4ff", "#ffadad", "#caffbf", "#ffd6a5", "#bdb2ff", "#9bf6ff"],
    "Neon": ["#00f5ff", "#ff00e5", "#39ff14", "#fff01f", "#ff6b00", "#8a2be2"],
    "Earth": ["#a47148", "#d4a373", "#6b705c", "#cb997e", "#606c38", "#ddbea9"],
    "Monochrome": ["#e6e6e6", "#b5b5b5", "#8a8a8a", "#cfcfcf", "#6e6e6e", "#a0a0a0"],
}
CUSTOM_PALETTE = "Custom color"
PALETTE_NAMES = list(PALETTES) + [CUSTOM_PALETTE]

EFFECTS = {
    "solid": "Solid",
    "glass": "Glass",
    "gradient": "Gradient",
    "neon": "Neon glow",
    "outline": "Outline",
}
ANIMATIONS = {
    "none": "None",
    "fade": "Fade in",
    "grow": "Grow",
    "shimmer": "Glass shimmer",
    "pulse": "Pulse glow",
    "float": "Float",
}
SORTS = {"none": "As computed", "asc": "Ascending", "desc": "Descending"}

DEFAULT_STYLE: dict = {
    "palette": "SILT",
    "color": "#7cc4fa",
    "effect": "solid",
    "animation": "none",
    "opacity": 0.9,
    "corner_radius": 4,
    "line_width": 2.5,
    "marker_size": 8,
    "labels": False,
    "legend": True,
    "sort": "none",
}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _clamp(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


def normalize_style(raw) -> dict:
    """Coerce a stored/user style into the whitelist.

    Styles are saved per team, so a teammate (or a hand-crafted API call) could
    store anything; every field is checked here before it reaches CSS or code.
    """
    raw = raw if isinstance(raw, dict) else {}
    s = dict(DEFAULT_STYLE)
    if raw.get("palette") in PALETTE_NAMES:
        s["palette"] = raw["palette"]
    if isinstance(raw.get("color"), str) and _HEX.match(raw["color"]):
        s["color"] = raw["color"].lower()
    if raw.get("effect") in EFFECTS:
        s["effect"] = raw["effect"]
    if raw.get("animation") in ANIMATIONS:
        s["animation"] = raw["animation"]
    if raw.get("sort") in SORTS:
        s["sort"] = raw["sort"]
    s["opacity"] = round(_clamp(raw.get("opacity"), 0.2, 1.0, s["opacity"]), 2)
    s["corner_radius"] = int(_clamp(raw.get("corner_radius"), 0, 20, s["corner_radius"]))
    s["line_width"] = round(_clamp(raw.get("line_width"), 0.5, 8, s["line_width"]), 1)
    s["marker_size"] = int(_clamp(raw.get("marker_size"), 2, 24, s["marker_size"]))
    for flag in ("labels", "legend"):
        if isinstance(raw.get(flag), bool):
            s[flag] = raw[flag]
    return s


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(min(max(c, 0), 255))) for c in (r, g, b)))


def shade(hex_color: str, amount: float) -> str:
    """amount > 0 mixes toward white, < 0 toward black."""
    r, g, b = _rgb(hex_color)
    target = 255 if amount > 0 else 0
    a = abs(amount)
    return _hex(r + (target - r) * a, g + (target - g) * a, b + (target - b) * a)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _rgb(hex_color)
    return f"rgba({r},{g},{b},{round(alpha, 3)})"


_CUSTOM_STEPS = [0, 0.35, -0.3, 0.6, -0.5, 0.18, -0.15, 0.8]


def palette_colors(style: dict, n: int) -> list[str]:
    n = max(n, 1)
    if style["palette"] == CUSTOM_PALETTE:
        return [shade(style["color"], _CUSTOM_STEPS[k % len(_CUSTOM_STEPS)]) for k in range(n)]
    base = PALETTES[style["palette"]]
    return [base[k % len(base)] for k in range(n)]


def effect_colors(style: dict, colors: list[str]) -> dict:
    """Fill / area-fill / edge colors and edge width for the chosen effect."""
    op = style["opacity"]
    effect = style["effect"]
    if effect == "glass":
        return dict(fill=[rgba(c, 0.4 * op) for c in colors], area=[rgba(c, 0.22 * op) for c in colors],
                    edge=[rgba(shade(c, 0.45), 0.95) for c in colors], width=1.5)
    if effect == "neon":
        return dict(fill=[rgba(c, 0.5 * op) for c in colors], area=[rgba(c, 0.25 * op) for c in colors],
                    edge=[shade(c, 0.35) for c in colors], width=2)
    if effect == "outline":
        return dict(fill=[rgba(c, 0.08) for c in colors], area=[rgba(c, 0.06) for c in colors],
                    edge=list(colors), width=2)
    if effect == "gradient":
        return dict(fill=[rgba(c, op) for c in colors], area=[rgba(c, 0.35 * op) for c in colors],
                    edge=list(colors), width=0)
    return dict(fill=[rgba(c, op) for c in colors], area=[rgba(c, 0.45 * op) for c in colors],
                edge=list(colors), width=0)


# --------------------------------------------------------------- extraction --

@dataclass
class ChartData:
    df: pd.DataFrame                  # columns: series, x, y
    source_type: str                  # plotly trace type of the original figure
    x_label: str = ""
    y_label: str = ""
    raw: bool = False                 # values are raw observations (histogram/box source)
    meta: dict = field(default_factory=dict)


def _decode(value):
    """Plotly 6 serializes numpy arrays as {"dtype": "f8", "bdata": <base64>}."""
    if isinstance(value, dict) and "bdata" in value:
        try:
            arr = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value.get("dtype", "f8")))
            shape = value.get("shape")
            if shape:
                arr = arr.reshape([int(s) for s in str(shape).split(",")])
            return arr.tolist()
        except (ValueError, TypeError):
            return None
    if isinstance(value, (list, tuple)):
        return [_decode(v) if isinstance(v, (dict, list)) else v for v in value]
    return None


def _axis_title(layout: dict, axis: str) -> str:
    title = (layout.get(axis) or {}).get("title")
    if isinstance(title, dict):
        title = title.get("text")
    return str(title) if title else ""


_PART_TYPES = ("pie", "funnelarea", "treemap", "sunburst", "icicle")


def _trace_rows(t: dict, typ: str, name: str) -> tuple[list[tuple], bool, bool]:
    """-> (rows of (series, x, y), is_raw, is_horizontal)."""
    get = lambda k: _decode(t.get(k))  # noqa: E731

    if typ in _PART_TYPES:
        labels, values, parents = get("labels"), get("values"), get("parents")
        if not labels:
            return [], False, False
        if values is None:  # plotly counts label occurrences when values are omitted
            return [(name, k, v) for k, v in Counter(labels).items()], False, False
        if parents and any(parents):
            inner = {p for p in parents if p}
            return [(p or name, lab, v) for lab, v, p in zip(labels, values, parents) if lab not in inner], False, False
        return [(name, lab, v) for lab, v in zip(labels, values)], False, False

    if typ == "heatmap":
        z = get("z")
        if not z:
            return [], False, False
        if not isinstance(z[0], list):
            z = [z]
        xs = get("x") or list(range(len(z[0])))
        ys = get("y") or list(range(len(z)))
        return [(str(ys[r]), xs[c], z[r][c]) for r in range(min(len(z), len(ys)))
                for c in range(min(len(z[r]), len(xs)))], False, False

    if typ in ("histogram", "box", "violin"):
        x, y = get("x"), get("y")
        if typ == "histogram" and x is not None and y is not None:
            return [(name, a, b) for a, b in zip(x, y)], False, False
        if x is not None and y is not None:
            horizontal = t.get("orientation") == "h"
            cat, val = (y, x) if horizontal else (x, y)
            return [(name, a, b) for a, b in zip(cat, val)], True, False
        vals = y if y is not None else x
        return [(name, k, v) for k, v in enumerate(vals or [], start=1)], True, False

    if typ in ("scatterpolar", "barpolar"):
        x, y = get("theta"), get("r")
    elif typ in ("candlestick", "ohlc"):
        x, y = get("x"), get("close")
    else:
        x, y = get("x"), get("y")

    horizontal = t.get("orientation") == "h" or (typ == "funnel" and t.get("orientation") != "v")
    if horizontal:
        x, y = y, x
    if y is None:
        return [], False, horizontal
    if x is None:
        x = list(range(1, len(y) + 1))
    return [(name, a, b) for a, b in zip(x, y)], False, horizontal


def _literal(v):
    """Reduce any value to a plain, repr()-safe python literal."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return str(v)


def extract_chart_data(plotly_json: dict) -> ChartData:
    traces = [t for t in (plotly_json or {}).get("data") or [] if isinstance(t, dict)]
    layout = (plotly_json or {}).get("layout") or {}
    rows: list[tuple] = []
    raw = horizontal = False
    source = None
    for i, t in enumerate(traces):
        typ = t.get("type") or "scatter"
        source = source or typ
        name = str(t.get("name") or (f"Series {i + 1}" if len(traces) > 1 else "Value"))
        trace_rows, is_raw, is_h = _trace_rows(t, typ, name)
        rows.extend(trace_rows)
        raw = raw or is_raw
        horizontal = horizontal or is_h

    df = pd.DataFrame(rows, columns=["series", "x", "y"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[np.isfinite(df["y"])].reset_index(drop=True)
    df["series"] = [str(_literal(v)) for v in df["series"]]
    df["x"] = [("(blank)" if (lit := _literal(v)) is None else lit) for v in df["x"]]
    df["y"] = [float(v) for v in df["y"]]

    x_label, y_label = _axis_title(layout, "xaxis"), _axis_title(layout, "yaxis")
    if horizontal:
        x_label, y_label = y_label, x_label
    return ChartData(df=df, source_type=source or "unknown", x_label=x_label, y_label=y_label,
                     raw=raw, meta={"mode": (traces[0].get("mode") if traces else None),
                                    "hole": (traces[0].get("hole") if traces else None),
                                    "fill": (traces[0].get("fill") if traces else None),
                                    "barmode": layout.get("barmode"),
                                    "horizontal": horizontal})


def detect_studio_type(data: ChartData) -> str | None:
    """Closest Chart Studio type for the original figure (used as the starting
    point when only the style is changed, and to mark 'original' in the bar)."""
    t, m = data.source_type, data.meta
    if t == "bar":
        # px sets barmode="relative" even for one series, so only call it
        # stacked when there is actually more than one series to stack
        stacked = m.get("barmode") in ("stack", "relative") and data.df["series"].nunique() > 1
        if m.get("horizontal"):
            return "stacked_hbar" if stacked else "hbar"
        return "stacked_bar" if stacked else "column"
    if t in ("scatter", "scattergl"):
        mode = m.get("mode") or "lines"
        if m.get("fill"):
            return "area"
        if "lines" in mode:
            return "line_markers" if "markers" in mode else "line"
        return "scatter"
    if t == "pie":
        return "donut" if m.get("hole") else "pie"
    return {
        "histogram": "histogram", "box": "box", "violin": "violin", "heatmap": "heatmap",
        "funnel": "funnel", "funnelarea": "funnel_area", "treemap": "treemap",
        "sunburst": "sunburst", "icicle": "icicle", "waterfall": "waterfall",
        "barpolar": "rose", "scatterpolar": "radar", "table": "table",
    }.get(t)


# --------------------------------------------------------------- chart types --
# Each snippet runs with these names in scope (all defined in the script
# prelude): df, title, x_label, y_label, colors, fill_colors, area_fills,
# edge_colors, edge_width, line_width, marker_size, corner_radius, show_labels,
# show_legend, pick, rgba, go, np, pd, make_subplots. It must leave a `fig`.

_BAR_MARKER = "marker=dict(color=pick(fill_colors, i), line=dict(color=pick(edge_colors, i), width=edge_width))"
_LOOP = 'for i, (name, g) in enumerate(df.groupby("series", sort=False)):'


def _bars(orientation: str, barmode: str) -> str:
    if orientation == "h":
        xy, label = 'x=g["y"], y=g["x"], orientation="h"', "%{x:.3s}"
        tail = '\nfig.update_yaxes(autorange="reversed")'
    else:
        xy, label, tail = 'x=g["x"], y=g["y"]', "%{y:.3s}", ""
    return f"""
fig = go.Figure()
{_LOOP}
    fig.add_bar({xy}, name=str(name),
                {_BAR_MARKER},
                texttemplate="{label}" if show_labels else None)
fig.update_layout(barmode="{barmode}"){tail}
"""


def _percent_bars(orientation: str) -> str:
    if orientation == "h":
        xy, tail = 'x=g["share"], y=g["x"], orientation="h"', '\nfig.update_layout(xaxis_ticksuffix="%")\nfig.update_yaxes(autorange="reversed")'
        tmpl = "%{x:.0f}%"
    else:
        xy, tail, tmpl = 'x=g["x"], y=g["share"]', '\nfig.update_layout(yaxis_ticksuffix="%")', "%{y:.0f}%"
    return f"""
# share of each segment within its category, in percent
df = df.assign(share=df["y"] / df.groupby("x", sort=False)["y"].transform("sum") * 100)
fig = go.Figure()
{_LOOP}
    fig.add_bar({xy}, name=str(name), customdata=g["y"],
                {_BAR_MARKER},
                hovertemplate="%{{customdata:,.2f}} (%{{{'x' if orientation == 'h' else 'y'}:.1f}}%)<extra>" + str(name) + "</extra>",
                texttemplate="{tmpl}" if show_labels else None)
fig.update_layout(barmode="stack"){tail}
"""


def _lines(mode="lines", shape="linear", dash=None, fill=None, stack=False, y_expr='g["y"]',
           pre: str = "", tail: str = "") -> str:
    extras = []
    if fill:
        extras.append(f'fill="{fill}", fillcolor=pick(area_fills, i)')
    if stack:
        extras.append('stackgroup="one"' + (', groupnorm="percent"' if stack == "percent" else "")
                      + ", fillcolor=pick(area_fills, i)")
    if "markers" in mode:
        extras.append("marker=dict(size=marker_size, color=pick(colors, i), "
                      "line=dict(color=pick(edge_colors, i), width=edge_width))")
    extra = "".join(f",\n                    {e}" for e in extras)
    dash_kw = f', dash="{dash}"' if dash else ""
    return f"""{pre}
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["x"], y={y_expr}, name=str(name),
                    mode="{mode}+text" if show_labels else "{mode}",
                    texttemplate="%{{y:.3s}}", textposition="top center",
                    line=dict(color=pick(colors, i), width=line_width, shape="{shape}"{dash_kw}){extra})
{tail}"""


_TOTALS = """
# total per category across all series; shares need positive amounts
totals = df.groupby("x", sort=False)["y"].sum()
totals = totals[totals > 0]
labels = [str(v) for v in totals.index]
slice_colors = [pick(fill_colors, k) for k in range(len(labels))]
slice_edges = [pick(edge_colors, k) for k in range(len(labels))]
"""

_HIERARCHY = """
# series -> category hierarchy; a parent's size is the sum of its children
agg = df.groupby(["series", "x"], sort=False)["y"].sum().reset_index()
agg = agg[agg["y"] > 0]
if agg["series"].nunique() > 1:
    tops = agg.groupby("series", sort=False)["y"].sum()
    ids = [str(s) for s in tops.index] + [f"{s} / {x}" for s, x in zip(agg["series"], agg["x"])]
    labels = [str(s) for s in tops.index] + [str(x) for x in agg["x"]]
    parents = [""] * len(tops) + [str(s) for s in agg["series"]]
    values = list(tops.values) + list(agg["y"])
else:
    ids = labels = [str(x) for x in agg["x"]]
    parents = [""] * len(agg)
    values = list(agg["y"])
node_colors = [pick(fill_colors, k) for k in range(len(ids))]
node_edges = [pick(edge_colors, k) for k in range(len(ids))]
"""


def _hier(trace: str, textinfo: str, extra_marker: str = "") -> str:
    return _HIERARCHY + f"""
fig = go.Figure(go.{trace}(ids=ids, labels=labels, parents=parents, values=values,
                branchvalues="total",
                marker=dict(colors=node_colors, line=dict(color=node_edges, width=edge_width){extra_marker}),
                textinfo="{textinfo}" if show_labels else "label"))
"""


_POLAR_LAYOUT = """
fig.update_polars(bgcolor="rgba(0,0,0,0)",
                  radialaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
                  angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.15)"))
"""

_DIST = {
    "histogram": f"""
fig = go.Figure()
{_LOOP}
    fig.add_histogram(x=g["y"], name=str(name),
                      {_BAR_MARKER},
                      texttemplate="%{{y}}" if show_labels else None)
fig.update_layout(barmode="overlay" if df["series"].nunique() > 1 else "group")
""",
    "box": f"""
fig = go.Figure()
{_LOOP}
    fig.add_box(y=g["y"], name=str(name), boxmean=True, boxpoints="outliers",
                fillcolor=pick(area_fills, i), marker=dict(color=pick(colors, i), size=marker_size * 0.6),
                line=dict(color=pick(colors, i), width=max(line_width / 2, 1)))
""",
    "violin": f"""
fig = go.Figure()
{_LOOP}
    fig.add_violin(y=g["y"], name=str(name), box_visible=True, meanline_visible=True,
                   points="outliers", fillcolor=pick(area_fills, i),
                   line=dict(color=pick(colors, i), width=max(line_width / 2, 1)))
""",
    "strip": f"""
fig = go.Figure()
{_LOOP}
    # a box with invisible body: plotly's way of drawing jittered points
    fig.add_box(y=g["y"], name=str(name), boxpoints="all", jitter=0.6, pointpos=0,
                fillcolor="rgba(0,0,0,0)", line=dict(color="rgba(0,0,0,0)"), hoveron="points",
                customdata=g["x"], hovertemplate="%{{customdata}}: %{{y:,.2f}}<extra></extra>",
                marker=dict(color=pick(fill_colors, i), size=marker_size,
                            line=dict(color=pick(edge_colors, i), width=edge_width)))
""",
}

# (id, label, group, material icon, axes(x expr, y expr) | None, snippet, formula)
_TYPES: list[tuple] = [
    # ---- bar ----
    ("column", "Column", "Bar", "bar_chart", ("x_label", "y_label"), _bars("v", "group"),
     "Bar height = the value the analysis computed for that category (no re-aggregation). "
     "Several series are drawn side by side."),
    ("hbar", "Horizontal bar", "Bar", "align_horizontal_left", ("y_label", "x_label"), _bars("h", "group"),
     "Bar length = the value the analysis computed for that category (no re-aggregation)."),
    ("stacked_bar", "Stacked column", "Bar", "stacked_bar_chart", ("x_label", "y_label"), _bars("v", "stack"),
     "Each segment = one series' value; full column height = Σ of all series' values for that category."),
    ("stacked_hbar", "Stacked bar", "Bar", "view_week", ("y_label", "x_label"), _bars("h", "stack"),
     "Each segment = one series' value; full bar length = Σ of all series' values for that category."),
    ("percent_bar", "100% stacked column", "Bar", "percent", ("x_label", '"Share of category (%)"'),
     _percent_bars("v"),
     "Segment % = series value ÷ Σ(all series values in that category) × 100, so every column adds to 100%."),
    ("percent_hbar", "100% stacked bar", "Bar", "density_medium", ('"Share of category (%)"', "x_label"),
     _percent_bars("h"),
     "Segment % = series value ÷ Σ(all series values in that category) × 100, so every bar adds to 100%."),
    ("lollipop", "Lollipop", "Bar", "more_vert", ("x_label", "y_label"), f"""
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["x"], y=g["y"], name=str(name),
                    mode="markers+text" if show_labels else "markers",
                    texttemplate="%{{y:.3s}}", textposition="top center",
                    marker=dict(color=pick(fill_colors, i), size=marker_size * 1.6,
                                line=dict(color=pick(edge_colors, i), width=edge_width)),
                    # the stick: an error bar from the dot back to zero
                    error_y=dict(type="data", symmetric=False, width=0, thickness=line_width,
                                 array=(-g["y"]).clip(lower=0), arrayminus=g["y"].clip(lower=0),
                                 color=pick(colors, i)))
""", "Dot height = the computed value; the stick runs from 0 to that value."),
    ("dot", "Dot plot", "Bar", "more_horiz", ("y_label", "x_label"), f"""
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["y"], y=g["x"], name=str(name),
                    mode="markers+text" if show_labels else "markers",
                    texttemplate="%{{x:.3s}}", textposition="middle right",
                    marker=dict(color=pick(fill_colors, i), size=marker_size * 1.5,
                                line=dict(color=pick(edge_colors, i), width=edge_width)))
fig.update_yaxes(autorange="reversed", showgrid=True)
""", "Dot position along the axis = the computed value; one row per category, one dot per series."),
    ("waterfall", "Waterfall", "Bar", "waterfall_chart", ("x_label", "y_label"), """
totals = df.groupby("x", sort=False)["y"].sum()
fig = go.Figure(go.Waterfall(
    x=[str(v) for v in totals.index] + ["Total"],
    y=list(totals.values) + [0],
    measure=["relative"] * len(totals) + ["total"],
    increasing=dict(marker=dict(color=pick(fill_colors, 0), line=dict(color=pick(edge_colors, 0), width=edge_width))),
    decreasing=dict(marker=dict(color=pick(fill_colors, 1), line=dict(color=pick(edge_colors, 1), width=edge_width))),
    totals=dict(marker=dict(color=pick(fill_colors, 2), line=dict(color=pick(edge_colors, 2), width=edge_width))),
    connector=dict(line=dict(color="rgba(255,255,255,0.25)", width=1)),
    texttemplate="%{y:.3s}" if show_labels else None,
    name="Running total"))
""", "Each step = that category's value (Σ across series), stacked on the running total "
     "Rₖ = v₁ + v₂ + … + vₖ; the final 'Total' bar = Σ of all values."),
    ("pareto", "Pareto", "Bar", "signal_cellular_alt", ("x_label", "y_label"), """
totals = df.groupby("x", sort=False)["y"].sum().sort_values(ascending=False)
cumulative_pct = totals.cumsum() / totals.sum() * 100
cats = [str(v) for v in totals.index]
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_bar(x=cats, y=totals.values, name="Value", secondary_y=False,
            marker=dict(color=pick(fill_colors, 0), line=dict(color=pick(edge_colors, 0), width=edge_width)),
            texttemplate="%{y:.3s}" if show_labels else None)
fig.add_scatter(x=cats, y=cumulative_pct.values, name="Cumulative %", secondary_y=True,
                mode="lines+markers", line=dict(color=pick(colors, 1), width=line_width),
                marker=dict(size=marker_size))
fig.update_yaxes(title_text="Cumulative %", ticksuffix="%", range=[0, 105], showgrid=False, secondary_y=True)
""", "Bars = category totals sorted largest first; line = cumulative % = "
     "(running Σ of sorted totals) ÷ (grand total) × 100."),
    # ---- line & area ----
    ("line", "Line", "Line & area", "show_chart", ("x_label", "y_label"), _lines(),
     "Each point = the computed value at that x; points joined in order."),
    ("spline", "Smooth line", "Line & area", "ssid_chart", ("x_label", "y_label"), _lines(shape="spline"),
     "Same points as the line chart, joined with a smoothing spline (curve shape only — values unchanged)."),
    ("step", "Step line", "Line & area", "stairs", ("x_label", "y_label"), _lines(shape="hv"),
     "Each value is held flat until the next x, then steps to the next value."),
    ("line_markers", "Line + markers", "Line & area", "timeline", ("x_label", "y_label"),
     _lines(mode="lines+markers"), "Each marker = the computed value at that x; markers joined in order."),
    ("dashed", "Dashed line", "Line & area", "horizontal_rule", ("x_label", "y_label"), _lines(dash="dash"),
     "Each point = the computed value at that x, joined with a dashed line."),
    ("area", "Area", "Line & area", "area_chart", ("x_label", "y_label"), _lines(fill="tozeroy"),
     "Line of the computed values with the area down to zero filled."),
    ("stacked_area", "Stacked area", "Line & area", "stacked_line_chart", ("x_label", "y_label"),
     _lines(stack=True), "Each band = one series' value; top edge = Σ of all series at that x."),
    ("percent_area", "100% stacked area", "Line & area", "full_stacked_bar_chart",
     ("x_label", '"Share (%)"'), _lines(stack="percent"),
     "Band % = series value ÷ Σ(all series at that x) × 100, so the stack always reaches 100%."),
    ("cumulative", "Cumulative (running total)", "Line & area", "trending_up", ("x_label", '"Running total"'),
     _lines(y_expr='g["y"].cumsum()', fill="tozeroy"),
     "Running total Cₖ = v₁ + v₂ + … + vₖ (summed in x order, per series)."),
    ("moving_avg", "Moving average", "Line & area", "query_stats", ("x_label", "y_label"), f"""
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["x"], y=g["y"], name=f"{{name}} (actual)", mode="lines",
                    line=dict(color=rgba(pick(colors, i), 0.35), width=max(line_width / 2, 1)))
    # 3-point moving average; the first points average whatever is available
    fig.add_scatter(x=g["x"], y=g["y"].rolling(3, min_periods=1).mean(), name=f"{{name}} (3-pt avg)",
                    mode="lines+text" if show_labels else "lines",
                    texttemplate="%{{y:.3s}}", textposition="top center",
                    line=dict(color=pick(colors, i), width=line_width, shape="spline"))
""", "MAₖ = (vₖ₋₂ + vₖ₋₁ + vₖ) ÷ 3 — the mean of the current and two previous values "
     "(fewer at the start). The faint line is the actual value."),
    # ---- part-to-whole ----
    ("pie", "Pie", "Part-to-whole", "pie_chart", None, _TOTALS + """
fig = go.Figure(go.Pie(labels=labels, values=totals.values, sort=False,
                       marker=dict(colors=slice_colors, line=dict(color=slice_edges, width=edge_width)),
                       textinfo="label+percent" if show_labels else "percent"))
""", "Slice % = category total ÷ grand total × 100, where category total = Σ values across series. "
     "Zero/negative totals are left out (a share must be positive)."),
    ("donut", "Donut", "Part-to-whole", "donut_large", None, _TOTALS + """
fig = go.Figure(go.Pie(labels=labels, values=totals.values, sort=False, hole=0.58,
                       marker=dict(colors=slice_colors, line=dict(color=slice_edges, width=edge_width)),
                       textinfo="label+percent" if show_labels else "percent"))
fig.add_annotation(text=f"<b>{totals.sum():,.4g}</b><br>total", showarrow=False,
                   font=dict(size=15, color="#e6e6e6"))
""", "Slice % = category total ÷ grand total × 100; the centre shows the grand total Σ. "
     "Zero/negative totals are left out."),
    ("sunburst", "Sunburst", "Part-to-whole", "data_usage", None, _hier("Sunburst", "label+percent root"),
     "Inner ring = each series (Σ of its categories); outer ring = each category's value inside its series. "
     "% = node ÷ grand total × 100."),
    ("treemap", "Treemap", "Part-to-whole", "view_quilt", None,
     _hier("Treemap", "label+value+percent root", ", cornerradius=corner_radius"),
     "Tile area ∝ value; a group's tile = Σ of the tiles inside it. % = tile ÷ grand total × 100."),
    ("icicle", "Icicle", "Part-to-whole", "view_day", None,
     _hier("Icicle", "label+value+percent root"),
     "Bar length ∝ value; a parent bar = Σ of its children. % = node ÷ grand total × 100."),
    ("funnel", "Funnel", "Part-to-whole", "filter_alt", None, f"""
fig = go.Figure()
{_LOOP}
    fig.add_funnel(y=[str(v) for v in g["x"]], x=g["y"], name=str(name),
                   marker=dict(color=pick(fill_colors, i), line=dict(color=pick(edge_colors, i), width=edge_width)),
                   textinfo="value+percent initial" if show_labels else "value")
""", "Stage width = the computed value; '% initial' = stage value ÷ first stage value × 100."),
    ("funnel_area", "Funnel area", "Part-to-whole", "change_history", None, _TOTALS + """
fig = go.Figure(go.Funnelarea(labels=labels, values=totals.values,
                              marker=dict(colors=slice_colors, line=dict(color=slice_edges, width=edge_width)),
                              textinfo="label+percent" if show_labels else "percent"))
""", "Section area ∝ category total; % = category total ÷ grand total × 100."),
    ("rose", "Nightingale rose", "Part-to-whole", "filter_vintage", None, f"""
fig = go.Figure()
{_LOOP}
    fig.add_barpolar(r=g["y"], theta=[str(v) for v in g["x"]], name=str(name),
                     marker=dict(color=pick(fill_colors, i), line=dict(color=pick(edge_colors, i), width=edge_width)))
""" + _POLAR_LAYOUT, "Wedge radius = the computed value for that category (one wedge per category and series)."),
    ("waffle", "Waffle", "Part-to-whole", "grid_view", None, _TOTALS + """
# 10 x 10 grid: each cell = 1% of the grand total
cells = (totals / totals.sum() * 100).round().astype(int)
cells.iloc[int(np.argmax(cells.values))] += 100 - cells.sum()  # rounding remainder to the largest
cat_index = np.repeat(np.arange(len(cells)), cells.clip(lower=0).values)[:100]
cat_index = np.pad(cat_index, (0, 100 - len(cat_index)), constant_values=len(cells) - 1)
grid = cat_index.reshape(10, 10)
k = len(cells)
scale = []
for j in range(k):
    scale += [[j / k, slice_colors[j]], [(j + 1) / k, slice_colors[j]]]
fig = go.Figure(go.Heatmap(z=grid, colorscale=scale, zmin=0, zmax=max(k - 1, 1), showscale=False,
                           xgap=3, ygap=3, text=np.array(labels)[grid],
                           hovertemplate="%{text}<extra></extra>"))
for j, lab in enumerate(labels):  # legend entries
    fig.add_scatter(x=[None], y=[None], mode="markers", name=f"{lab} ({cells.iloc[j]}%)",
                    marker=dict(color=slice_colors[j], size=11, symbol="square"))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, scaleanchor="x")
""", "Cells = round(category total ÷ grand total × 100); 100 cells = 100%. "
     "Any rounding remainder goes to the largest category."),
    # ---- relationship ----
    ("scatter", "Scatter", "Relationship", "scatter_plot", ("x_label", "y_label"), f"""
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["x"], y=g["y"], name=str(name),
                    mode="markers+text" if show_labels else "markers",
                    texttemplate="%{{y:.3s}}", textposition="top center",
                    marker=dict(color=pick(fill_colors, i), size=marker_size * 1.3,
                                line=dict(color=pick(edge_colors, i), width=edge_width)))
""", "Each point = (x, computed value)."),
    ("bubble", "Bubble", "Relationship", "bubble_chart", ("x_label", "y_label"), f"""
peak = df["y"].abs().max() or 1
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["x"], y=g["y"], name=str(name),
                    mode="markers+text" if show_labels else "markers",
                    texttemplate="%{{y:.3s}}", textposition="middle center",
                    marker=dict(size=g["y"].abs(), sizemode="area", sizemin=4,
                                sizeref=2 * peak / (marker_size * 5) ** 2,
                                color=pick(fill_colors, i), line=dict(color=pick(edge_colors, i), width=edge_width)))
""", "Bubble area ∝ |value| (largest bubble = largest absolute value); vertical position = the value."),
    ("scatter_trend", "Scatter + trend line", "Relationship", "insights", ("x_label", "y_label"), f"""
fig = go.Figure()
{_LOOP}
    fig.add_scatter(x=g["x"], y=g["y"], name=str(name), mode="markers",
                    marker=dict(color=pick(fill_colors, i), size=marker_size * 1.3,
                                line=dict(color=pick(edge_colors, i), width=edge_width)))
    numeric_x = pd.to_numeric(g["x"], errors="coerce")
    # categorical/date x: fit against position 0, 1, 2, ...
    pos = numeric_x.to_numpy(dtype=float) if numeric_x.notna().all() else np.arange(len(g), dtype=float)
    if len(g) >= 2 and np.ptp(pos) > 0:
        slope, intercept = np.polyfit(pos, g["y"].to_numpy(dtype=float), 1)  # least squares
        order = np.argsort(pos)
        fig.add_scatter(x=g["x"].to_numpy()[order], y=(slope * pos + intercept)[order],
                        name=f"{{name}} trend", mode="lines",
                        line=dict(color=pick(colors, i), width=line_width, dash="dash"))
""", "Trend = least-squares line ŷ = a·x + b with a = Σ(x−x̄)(y−ȳ) ÷ Σ(x−x̄)² and b = ȳ − a·x̄ "
     "(x = position 0, 1, 2… when x is not numeric)."),
    ("radar", "Radar", "Relationship", "radar", None, f"""
fig = go.Figure()
{_LOOP}
    theta = [str(v) for v in g["x"]]
    r = list(g["y"])
    fig.add_scatterpolar(r=r + r[:1], theta=theta + theta[:1], name=str(name),  # repeat first point to close
                         mode="lines+markers", fill="toself", fillcolor=pick(area_fills, i),
                         line=dict(color=pick(colors, i), width=line_width),
                         marker=dict(size=marker_size * 0.7))
""" + _POLAR_LAYOUT, "Distance from the centre = the computed value for each category (one spoke per category)."),
    ("diverging", "Deviation from average", "Relationship", "swap_vert", ("x_label", '"Deviation from average"'), """
totals = df.groupby("x", sort=False)["y"].sum()
deviation = totals - totals.mean()
fig = go.Figure(go.Bar(
    x=[str(v) for v in totals.index], y=deviation.values, name="Deviation from average",
    customdata=totals.values,
    hovertemplate="%{x}<br>deviation %{y:,.2f}<br>value %{customdata:,.2f}<extra></extra>",
    marker=dict(color=[pick(fill_colors, 0) if d >= 0 else pick(fill_colors, 1) for d in deviation],
                line=dict(color=pick(edge_colors, 0), width=edge_width)),
    texttemplate="%{y:+.3s}" if show_labels else None))
fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)")
""", "Deviation = category total − average of all category totals (x̄ = Σ totals ÷ number of categories). "
     "Above zero = above average."),
    # ---- distribution ----
    ("histogram", "Histogram", "Distribution", "equalizer", ("y_label", '"Count"'), _DIST["histogram"],
     "Values are grouped into equal-width bins; bar height = how many values fall in each bin."),
    ("box", "Box plot", "Distribution", "candlestick_chart", (None, "y_label"), _DIST["box"],
     "Box = Q1 to Q3 (25th–75th percentile), line = median, dashed = mean, "
     "whiskers reach the furthest value within 1.5 × IQR (IQR = Q3 − Q1)."),
    ("violin", "Violin", "Distribution", "graphic_eq", (None, "y_label"), _DIST["violin"],
     "Width = kernel-density estimate of how common each value is; inner box = Q1/median/Q3."),
    ("strip", "Strip (jitter)", "Distribution", "blur_on", (None, "y_label"), _DIST["strip"],
     "Every computed value is one dot; dots are spread sideways (jitter) only so they don't overlap."),
    # ---- matrix ----
    ("heatmap", "Heatmap", "Matrix", "grid_on", ("x_label", None), """
grid = df.pivot_table(index="series", columns="x", values="y", aggfunc="sum", sort=False)
fig = go.Figure(go.Heatmap(z=grid.values, x=[str(c) for c in grid.columns], y=[str(r) for r in grid.index],
                           colorscale=[[0, rgba(pick(colors, 0), 0.08)], [1, pick(colors, 0)]],
                           xgap=2, ygap=2, colorbar=dict(outlinewidth=0),
                           texttemplate="%{z:.3s}" if show_labels else None))
""", "Cell colour intensity = Σ value for that (series, category) pair, scaled from the smallest to the largest cell."),
    ("table", "Table", "Matrix", "table_chart", None, """
fig = go.Figure(go.Table(
    header=dict(values=["Series", x_label or "Category", y_label or "Value"], align="left",
                fill_color=rgba(pick(colors, 0), 0.25), font=dict(color="#e6e6e6"),
                line_color="rgba(255,255,255,0.12)"),
    cells=dict(values=[df["series"], df["x"], df["y"].round(4)], align="left",
               fill_color="rgba(255,255,255,0.03)", font=dict(color="#e6e6e6"),
               line_color="rgba(255,255,255,0.08)")))
""", "The exact values plotted, one row per point — no further calculation."),
]

CHART_TYPES: dict[str, dict] = {
    t[0]: {"id": t[0], "label": t[1], "group": t[2], "icon": t[3], "axes": t[4],
           "snippet": t[5], "formula": t[6]}
    for t in _TYPES
}
GROUPS: list[str] = list(dict.fromkeys(t[2] for t in _TYPES))
assert len(CHART_TYPES) == 40, len(CHART_TYPES)


def types_in_group(group: str) -> list[str]:
    return [k for k, v in CHART_TYPES.items() if v["group"] == group]


def formula_for(chart_type: str) -> str:
    return CHART_TYPES[chart_type]["formula"]


_PART_TO_WHOLE = {"pie", "donut", "funnel_area", "waffle", "sunburst", "treemap", "icicle"}

# ----------------------------------------------------------- effects (code) --

_GRADIENT = """
# Gradient effect: shade each bar by its value; fade area fills to transparent
for k, tr in enumerate(fig.data):
    base = pick(colors, k)
    if tr.type == "barpolar":
        tr.marker.color, tr.marker.colorscale = list(tr.r), [[0, rgba(base, 0.25)], [1, base]]
    elif tr.type == "bar" and not isinstance(tr.marker.color, (list, tuple)):
        values = tr.x if tr.orientation == "h" else tr.y
        tr.marker.color, tr.marker.colorscale = list(values), [[0, rgba(base, 0.25)], [1, base]]
    elif tr.type == "scatter" and tr.fill:
        tr.fillgradient = dict(type="vertical", colorscale=[[0, rgba(base, 0.0)], [1, rgba(base, 0.75)]])
"""

_NEON = """
# Neon effect: a wide, faint copy under every line makes it glow
glows = []
for tr in fig.data:
    if tr.type == "scatter" and tr.mode and "lines" in tr.mode and not tr.stackgroup \\
            and isinstance(tr.line.color, str) and tr.line.color.startswith("#"):
        glows.append(go.Scatter(x=tr.x, y=tr.y, mode="lines", showlegend=False, hoverinfo="skip",
                                xaxis=tr.xaxis, yaxis=tr.yaxis,
                                line=dict(color=rgba(tr.line.color, 0.22), width=line_width * 4,
                                          shape=tr.line.shape)))
if glows:  # add the glows, then move them underneath the real lines
    fig.add_traces(glows)
    fig.data = fig.data[-len(glows):] + fig.data[:-len(glows)]
"""

_HELPERS = '''
def pick(seq, k):
    """k-th color, cycling if there are more items than colors."""
    return seq[k % len(seq)]


def rgba(hex_color, alpha):
    """'#4f8fe0', 0.5 -> 'rgba(79,143,224,0.5)'"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[j:j + 2], 16) for j in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
'''


# ------------------------------------------------------------------- build --

@dataclass
class ChartBuild:
    fig: go.Figure
    code: str


def _wrap_list(items: list, indent: str, width: int = 96) -> str:
    """repr() items into a list literal, breaking lines between items only
    (never inside a string literal, which textwrap would happily do)."""
    reps = [repr(_literal(v)) for v in items]
    lines, cur = [], ""
    for r in reps:
        piece = r + ", "
        if cur and len(indent) + len(cur) + len(piece) > width:
            lines.append(cur.rstrip())
            cur = ""
        cur += piece
    if cur:
        lines.append(cur.rstrip().rstrip(","))
    if not lines:
        return "[]"
    if len(lines) == 1:
        return "[" + lines[0] + "]"
    return "[\n" + "\n".join(indent + "    " + ln for ln in lines) + "\n" + indent + "]"


def _comment(text: str) -> str:
    return " ".join(str(text).split())[:80]


def build_chart(data: ChartData, chart_type: str, style: dict, title: str = "") -> ChartBuild:
    """Render `data` as `chart_type`. Raises ValueError if there is nothing to plot."""
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unknown chart type: {chart_type}")
    if data.df.empty:
        raise ValueError("This chart has no plottable values to re-chart.")
    style = normalize_style(style)
    spec = CHART_TYPES[chart_type]
    df = data.df
    if chart_type in _PART_TO_WHOLE and not (df["y"] > 0).any():
        raise ValueError(f"A {spec['label'].lower()} chart shows shares of a total, "
                         "so it needs at least one positive value.")

    n_colors = max(df["series"].nunique(), df["x"].nunique(), 1)
    colors = palette_colors(style, n_colors)
    fx = effect_colors(style, colors)

    lines = [
        f"# {spec['label']} — rebuilt in Chart Studio",
        "import numpy as np",
        "import pandas as pd",
        "import plotly.graph_objects as go",
        "from plotly.subplots import make_subplots",
        "",
        "# The values the analysis computed — one row per plotted point",
        f"# x = {_comment(data.x_label) or 'category'}, y = {_comment(data.y_label) or 'value'}",
        "df = pd.DataFrame({",
        f"    \"series\": {_wrap_list(list(df['series']), '    ')},",
        f"    \"x\": {_wrap_list(list(df['x']), '    ')},",
        f"    \"y\": {_wrap_list(list(df['y']), '    ')},",
        "})",
        f"title = {title!r}",
        f"x_label = {data.x_label!r}",
        f"y_label = {data.y_label!r}",
    ]
    if style["sort"] != "none":
        asc = style["sort"] == "asc"
        lines += ["", f"df = df.sort_values(\"y\", ascending={asc}).reset_index(drop=True)  # sort: {SORTS[style['sort']]}"]
    lines += [
        "",
        f"# Style: palette {style['palette']!s}"
        + (f" ({style['color']})" if style["palette"] == CUSTOM_PALETTE else "")
        + f", effect {EFFECTS[style['effect']]}, opacity {style['opacity']}",
        f"colors = {_wrap_list(colors, '')}",
        f"fill_colors = {_wrap_list(fx['fill'], '')}",
        f"area_fills = {_wrap_list(fx['area'], '')}",
        f"edge_colors = {_wrap_list(fx['edge'], '')}",
        f"edge_width = {fx['width']!r}",
        f"line_width = {style['line_width']!r}",
        f"marker_size = {style['marker_size']!r}",
        f"corner_radius = {style['corner_radius']!r}",
        f"show_labels = {style['labels']!r}",
        f"show_legend = {style['legend']!r}",
        _HELPERS.rstrip(),
        "",
        textwrap.dedent(spec["snippet"]).strip(),
    ]
    if style["effect"] == "gradient":
        lines += ["", _GRADIENT.strip()]
    elif style["effect"] == "neon":
        lines += ["", _NEON.strip()]
    lines += ["", "fig.update_layout(title=title, showlegend=show_legend, barcornerradius=corner_radius)"]
    axes = spec["axes"]
    if axes:
        kw = [f"{a}axis_title={expr}" for a, expr in zip(("x", "y"), axes) if expr]
        if kw:
            lines.append("fig.update_layout(" + ", ".join(kw) + ")")
    if style["animation"] != "none":
        lines.append(f"# animation '{ANIMATIONS[style['animation']]}' is a dashboard (CSS) effect, not part of the figure")

    code = "\n".join(lines) + "\n"
    namespace: dict = {"__name__": "chart_studio_build"}
    try:
        exec(compile(code, f"<chart_studio:{chart_type}>", "exec"), namespace)  # noqa: S102 - see module docstring
    except Exception as e:  # noqa: BLE001 - e.g. a pie of all-negative values
        raise ValueError(f"This data can't be drawn as a {spec['label'].lower()} chart ({e}).") from e
    return ChartBuild(fig=namespace["fig"], code=code + "fig.show()\n")


# ------------------------------------------------------- calculation table --

def _sorted_df(data: ChartData, style: dict) -> pd.DataFrame:
    df = data.df
    sort = normalize_style(style)["sort"]
    if sort != "none":
        df = df.sort_values("y", ascending=(sort == "asc")).reset_index(drop=True)
    return df


def calculation_table(data: ChartData, chart_type: str, style: dict | None = None) -> tuple[pd.DataFrame, str]:
    """The numbers the new chart actually derives, with a one-line caption."""
    df = _sorted_df(data, style or {})
    xl, yl = data.x_label or "Category", data.y_label or "Value"
    r = lambda s: s.round(4)  # noqa: E731

    if chart_type in ("pie", "donut", "funnel_area", "waffle"):
        totals = df.groupby("x", sort=False)["y"].sum()
        totals = totals[totals > 0]
        out = pd.DataFrame({xl: totals.index.astype(str), "Total": r(totals.values),
                            "Share %": r(totals.values / totals.sum() * 100)})
        if chart_type == "waffle":
            cells = (totals / totals.sum() * 100).round().astype(int)
            cells.iloc[int(np.argmax(cells.values))] += 100 - cells.sum()
            out["Cells (of 100)"] = cells.values
        return out, f"Grand total = {totals.sum():,.4g}"
    if chart_type in ("sunburst", "treemap", "icicle"):
        agg = df.groupby(["series", "x"], sort=False)["y"].sum().reset_index()
        agg = agg[agg["y"] > 0]
        parent = agg.groupby("series")["y"].transform("sum")
        return pd.DataFrame({"Series": agg["series"], xl: agg["x"], "Value": r(agg["y"]),
                             "% of parent": r(agg["y"] / parent * 100),
                             "% of total": r(agg["y"] / agg["y"].sum() * 100)}), "Parent = Σ of its children"
    if chart_type in ("percent_bar", "percent_hbar", "percent_area"):
        cat_total = df.groupby("x", sort=False)["y"].transform("sum")
        return pd.DataFrame({"Series": df["series"], xl: df["x"], yl: r(df["y"]),
                             "Category total": r(cat_total), "Share %": r(df["y"] / cat_total * 100)}), \
            "Share % = value ÷ category total × 100"
    if chart_type in ("stacked_bar", "stacked_hbar", "stacked_area"):
        tot = df.groupby("x", sort=False)["y"].sum()
        return pd.DataFrame({xl: tot.index, "Stack height (Σ series)": r(tot.values)}), "Σ of every series per category"
    if chart_type == "cumulative":
        return pd.DataFrame({"Series": df["series"], xl: df["x"], yl: r(df["y"]),
                             "Running total": r(df.groupby("series", sort=False)["y"].cumsum())}), "Cₖ = Σ v₁…vₖ"
    if chart_type == "moving_avg":
        ma = df.groupby("series", sort=False)["y"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        return pd.DataFrame({"Series": df["series"], xl: df["x"], yl: r(df["y"]), "3-pt average": r(ma)}), \
            "MAₖ = mean(vₖ₋₂, vₖ₋₁, vₖ)"
    if chart_type == "pareto":
        tot = df.groupby("x", sort=False)["y"].sum().sort_values(ascending=False)
        return pd.DataFrame({xl: tot.index, "Total": r(tot.values), "Running Σ": r(tot.cumsum().values),
                             "Cumulative %": r((tot.cumsum() / tot.sum() * 100).values)}), \
            f"Grand total = {tot.sum():,.4g}"
    if chart_type == "waterfall":
        tot = df.groupby("x", sort=False)["y"].sum()
        return pd.DataFrame({xl: tot.index, "Change": r(tot.values), "Running total": r(tot.cumsum().values)}), \
            f"Final total = {tot.sum():,.4g}"
    if chart_type == "diverging":
        tot = df.groupby("x", sort=False)["y"].sum()
        return pd.DataFrame({xl: tot.index, "Total": r(tot.values), "Average": round(tot.mean(), 4),
                             "Deviation": r((tot - tot.mean()).values)}), f"Average = {tot.mean():,.4g}"
    if chart_type == "scatter_trend":
        rows = []
        for name, g in df.groupby("series", sort=False):
            nx = pd.to_numeric(g["x"], errors="coerce")
            pos = nx.to_numpy(dtype=float) if nx.notna().all() else np.arange(len(g), dtype=float)
            if len(g) >= 2 and np.ptp(pos) > 0:
                yv = g["y"].to_numpy(dtype=float)
                a, b = np.polyfit(pos, yv, 1)
                ss_res = float(((yv - (a * pos + b)) ** 2).sum())
                ss_tot = float(((yv - yv.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
                rows.append({"Series": name, "Slope a": round(a, 6), "Intercept b": round(b, 6),
                             "R²": round(r2, 4), "Points": len(g)})
        return pd.DataFrame(rows), "ŷ = a·x + b (least squares)"
    if chart_type in ("histogram", "box", "violin", "strip"):
        st = df.groupby("series", sort=False)["y"].describe()
        st = st.rename(columns={"25%": "Q1", "50%": "Median", "75%": "Q3"})
        st.insert(0, "Series", st.index)
        return st.reset_index(drop=True).round(4), "Summary statistics of the plotted values"
    if chart_type == "bubble":
        peak = df["y"].abs().max() or 1
        return pd.DataFrame({"Series": df["series"], xl: df["x"], yl: r(df["y"]),
                             "Relative bubble area": r(df["y"].abs() / peak)}), "Area ∝ |value| ÷ max |value|"
    if chart_type == "heatmap":
        grid = df.pivot_table(index="series", columns="x", values="y", aggfunc="sum", sort=False)
        grid.columns = [str(c) for c in grid.columns]
        grid.insert(0, "Series", grid.index)
        return grid.reset_index(drop=True).round(4), "Cell = Σ value per (series, category)"
    return pd.DataFrame({"Series": df["series"], xl: df["x"], yl: r(df["y"])}), "Values plotted as computed"
