"""Two-way sensitivity heatmap: one output over a grid of two inputs.

Rendering only. The grid, the best and worst cells, the current position and the threshold boundary come from the backend
(`/api/scenarios/sensitivity-grid`) together with its independent check; this module draws them.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, sensitivity_grid
from style.theme import badge, esc, html, plot, stat_card
from what_if_decision import num, section, verification_panel

RESOLUTIONS = [11, 21, 31, 41, 61]


def grid_section(mk: str, model: dict, baseline: dict, assumptions: dict, adjustments: dict) -> None:
    section("Two-way sensitivity (heatmap)", "One result across a grid of two inputs. Cells are the model's own arithmetic; the axes replace any "
                                             "adjustment to those two inputs and everything else stays as set above.")
    inputs, outputs = list(model["inputs"]), list(model["outputs"])
    if len(inputs) < 2:
        st.caption("A model needs at least two inputs for a two-way grid.")
        return
    a, b, c = st.columns(3)
    x_input = a.selectbox("Across (x)", inputs, format_func=lambda k: model["inputs"][k]["label"], key=f"gr_x_{mk}")
    y_input = b.selectbox("Up (y)", inputs, index=1, format_func=lambda k: model["inputs"][k]["label"], key=f"gr_y_{mk}")
    output = c.selectbox("Result shown", outputs, index=outputs.index(model["primary_output"]), format_func=model["outputs"].get, key=f"gr_out_{mk}")
    r1, r2, r3, r4 = st.columns(4)
    xl = r1.number_input("x from, % of its value", -99.0, 1000.0, -20.0, 5.0, key=f"gr_xl_{mk}")
    xh = r2.number_input("x to, %", -99.0, 1000.0, 20.0, 5.0, key=f"gr_xh_{mk}")
    yl = r3.number_input("y from, %", -99.0, 1000.0, -20.0, 5.0, key=f"gr_yl_{mk}")
    yh = r4.number_input("y to, %", -99.0, 1000.0, 20.0, 5.0, key=f"gr_yh_{mk}")
    s1, s2, s3 = st.columns(3)
    resolution = s1.select_slider("Cells per side", RESOLUTIONS, 21, key=f"gr_res_{mk}")
    mark = s2.checkbox("Mark where the result crosses a value", True, key=f"gr_mark_{mk}")
    threshold = s3.number_input("Value (0 = break-even for profit)", value=0.0, key=f"gr_thr_{mk}", disabled=not mark)
    body = {"model": mk, "baseline": baseline, "assumptions": assumptions, "adjustments": adjustments, "x_input": x_input, "y_input": y_input,
            "output": output, "x_range": {"mode": "pct", "low": xl, "high": xh}, "y_range": {"mode": "pct", "low": yl, "high": yh},
            "resolution": int(resolution), "threshold": float(threshold) if mark else None}
    key = f"gr_result_{mk}"
    if st.button("Draw the heatmap", type="primary", key="gr_run"):
        try:
            with st.spinner("Computing the grid and checking it…"):
                st.session_state[key] = sensitivity_grid(body)
        except ApiError as e:
            st.session_state.pop(key, None)
            st.error(f"The grid could not be drawn: {e}")
    g = st.session_state.get(key)
    if g:
        _show(g)


def _show(g: dict) -> None:
    label = g["output"]["label"]
    xs, ys = g["x"]["values"], g["y"]["values"]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=g["z"], colorbar={"title": label}, hoverongaps=False,
                             hovertemplate=f"{g['x']['label']}: %{{x:,.4g}}<br>{g['y']['label']}: %{{y:,.4g}}<br>{label}: %{{z:,.2f}}<extra></extra>"))
    t = g["threshold"]
    if t is not None and g["boundary"]["points"]:
        fig.add_trace(go.Contour(x=xs, y=ys, z=g["z"], showscale=False, hoverinfo="skip", name=f"{label} = {t:g}",
                                 contours={"start": t, "end": t, "size": 1, "coloring": "lines", "showlabels": False},
                                 line={"width": 3, "dash": "dash"}))
    cur = g["current"]
    fig.add_trace(go.Scatter(x=[cur["x"]], y=[cur["y"]], mode="markers", name="Where you are now",
                             marker={"symbol": "star", "size": 15, "line": {"width": 1}}))
    if g["best"]:
        fig.add_trace(go.Scatter(x=[g["best"]["x"]], y=[g["best"]["y"]], mode="markers", name="Best cell",
                                 marker={"symbol": "diamond", "size": 12, "line": {"width": 1}}))
        fig.add_trace(go.Scatter(x=[g["worst"]["x"]], y=[g["worst"]["y"]], mode="markers", name="Worst cell",
                                 marker={"symbol": "x", "size": 12, "line": {"width": 1}}))
    fig.update_layout(title=f"{label} across {g['x']['label'].lower()} and {g['y']['label'].lower()}", xaxis_title=g["x"]["label"],
                      yaxis_title=g["y"]["label"])
    plot(fig, height=440, showlegend=True, restyle_traces=False)
    cards = [("Where you are now", num(cur["value"]), ""), ("Best on this grid", num(g["best"]["value"]) if g["best"] else "—", ""),
             ("Worst on this grid", num(g["worst"]["value"]) if g["worst"] else "—", "")]
    if t is not None:
        b = g["boundary"]
        cards.append((f"Cells above {t:g}", f"{b['cells_above']:,} of {g['cells']:,}", ""))
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(stat_card(*card), unsafe_allow_html=True)
    if g["best"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">Best cell: {esc(g["x"]["label"])} {num(g["best"]["x"], 4)}, {esc(g["y"]["label"])} '
             f'{num(g["best"]["y"], 4)}. Worst cell: {esc(g["x"]["label"])} {num(g["worst"]["x"], 4)}, {esc(g["y"]["label"])} {num(g["worst"]["y"], 4)}.</div>')
    if t is not None:
        n = len(g["boundary"]["points"])
        html(f'<div class="ds-row-meta" style="padding:3px 0;">{badge("Boundary", "neutral")} '
             + (f"The result crosses {t:g} at {n} point(s) between cells (the dashed line)." if n else f"The result never crosses {t:g} inside these ranges.") + "</div>")
    for d in g["disclosures"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(d)}</div>')
    verification_panel(g.get("verification"), "Independent check of the grid")
