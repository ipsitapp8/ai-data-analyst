"""What-if tools that work from the user's real data: elasticity estimated from history.

Rendering only: every number shown comes from the backend (`/api/scenarios/*`).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, estimate_elasticity
from style.theme import badge, esc, html, plot, stat_card
from what_if_decision import num, section, verification_panel

RELIABILITY_BADGE = {"usable": ("verified", "Usable"), "use_with_caution": ("warn", "Use with caution"),
                     "not_recommended": ("error", "Not recommended"), "not_estimated": ("error", "Not estimated")}


def _columns(ds: dict, kind: str) -> list[str]:
    return [c["name"] for c in ds["profile"].get("columns", []) if c.get("kind") == kind]


def _use_estimate(slider_key: str, value: float, applied_key: str, estimate: dict) -> None:
    """Button callback: runs before the slider is built on the next run, so it may set the slider's value."""
    st.session_state[slider_key] = float(value)
    st.session_state[applied_key] = {"value": float(value), "estimate": estimate}


def elasticity_panel(mk: str, datasets: list[dict], slider_key: str, current_value: float) -> dict:
    """Estimate price elasticity from a dataset. Returns `assumption_sources` for the request: the estimate is
    recorded as the source only while the slider still holds the value the user took from it."""
    applied_key = f"wi_el_applied_{mk}"
    result_key = f"wi_el_result_{mk}"
    with st.expander("Estimate price elasticity from your data (optional)"):
        st.caption("A suggestion from your own history, with its uncertainty. Your slider value above stays in charge: "
                   "nothing changes unless you press the button that uses the estimate.")
        if not datasets:
            st.caption("Upload a dataset with price and quantity columns to use this.")
        else:
            ds = st.selectbox("Dataset", datasets, format_func=lambda d: d["filename"], key="wi_el_ds")
            numeric = _columns(ds, "numeric")
            c1, c2 = st.columns(2)
            price = c1.selectbox("Price column", numeric, key="wi_el_price", index=None, placeholder="Choose the price column")
            qty = c2.selectbox("Quantity column", numeric, key="wi_el_qty", index=None, placeholder="Choose the quantity column")
            d1, d2, d3 = st.columns(3)
            date = d1.selectbox("Date column", [None] + _columns(ds, "datetime"), format_func=lambda v: "None (each row)" if v is None else v, key="wi_el_date")
            segment = d2.selectbox("Segment column (product, region…)", [None] + _columns(ds, "categorical"),
                                   format_func=lambda v: "None" if v is None else v, key="wi_el_segment")
            gran = d3.selectbox("Period", ["auto", "day", "week", "month", "quarter"], key="wi_el_gran", disabled=date is None)
            o1, o2 = st.columns(2)
            level = o1.select_slider("Confidence level", [0.8, 0.9, 0.95, 0.99], 0.95, key="wi_el_level")
            trim = o2.checkbox("Leave outliers out of the estimate", False, key="wi_el_trim")
            body = {"dataset_id": ds["id"], "price_column": price, "quantity_column": qty, "date_column": date,
                    "segment_column": segment, "granularity": gran, "level": level, "exclude_outliers": trim}
            if st.button("Estimate elasticity", key="wi_el_run", disabled=not (price and qty)):
                try:
                    with st.spinner("Estimating and checking…"):
                        st.session_state[result_key] = estimate_elasticity(body)
                except ApiError as e:
                    st.session_state.pop(result_key, None)
                    st.error(f"The estimate could not run: {e}")
            if not (price and qty):
                st.caption("Choose the price and quantity columns to enable the estimate.")
        r = st.session_state.get(result_key)
        if r:
            _show(r, slider_key, applied_key, current_value)
    applied = st.session_state.get(applied_key)
    if applied and abs(applied["value"] - current_value) < 1e-9:
        e = applied["estimate"]
        return {"price_elasticity": {"source": "estimate", "estimate": e["elasticity"]["estimate"], "ci_low": e["elasticity"]["ci_low"],
                                     "ci_high": e["elasticity"]["ci_high"], "level": e["elasticity"]["level"],
                                     "engine_version": e["engine_version"], "provenance": e["provenance"],
                                     "chosen_value": applied["value"]}}
    return {}


def _show(r: dict, slider_key: str, applied_key: str, current_value: float) -> None:
    p = r["provenance"]
    html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge("Statistical estimate", "neutral")} from '
         f'{esc(p["dataset_filename"])} · version {p["dataset_version"]}{" (a newer version exists)" if p["latest_version"] and p["latest_version"] != p["dataset_version"] else ""}'
         f' · {r["data"]["observations"]:,} observations</div>')
    rel = r["reliability"]
    kind, label = RELIABILITY_BADGE.get(rel["level"], ("neutral", rel["level"]))
    if r["status"] == "not_estimated":
        html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge(label, kind)} {esc(rel["recommendation"])}</div>')
        for reason in r["reasons"]:
            html(f'<div class="ds-row-meta" style="padding:2px 0;">• {esc(reason)}</div>')
        return
    e, d = r["elasticity"], r["diagnostics"]
    cards = [("Estimated elasticity", num(e["estimate"], 3), ""),
             (f"{e['level'] * 100:g}% interval", f"{num(e['ci_low'], 2)} to {num(e['ci_high'], 2)}", ""),
             ("Observations", f"{d['observations']:,}", ""), ("R²", num(d["r_squared"], 2), "")]
    for col, card in zip(st.columns(4), cards):
        col.markdown(stat_card(*card), unsafe_allow_html=True)
    html(f'<div class="ds-row-meta" style="padding:4px 0;">{badge(label, kind)} {esc(rel["recommendation"])}</div>')
    html(f'<div class="ds-row-meta" style="padding:2px 0;">Your value in the slider: <b>{current_value:g}</b>. '
         f'Observed price range: {num(d["price_range"][0])} to {num(d["price_range"][1])}; do not apply the estimate far outside it.</div>')
    if rel["can_apply"]:
        st.button(f"Use the estimate ({e['estimate']:.2f}) as my elasticity", key="wi_el_use", on_click=_use_estimate,
                  args=(slider_key, round(e["estimate"], 2), applied_key, r))
    else:
        st.caption("This estimate cannot be applied: see the recommendation above. Your own value stays as it is.")
    if r["scatter"]:
        fig = go.Figure()
        fig.add_scatter(x=[s["price"] for s in r["scatter"] if not s["outlier"]], y=[s["quantity"] for s in r["scatter"] if not s["outlier"]],
                        mode="markers", name="Observations")
        if any(s["outlier"] for s in r["scatter"]):
            fig.add_scatter(x=[s["price"] for s in r["scatter"] if s["outlier"]], y=[s["quantity"] for s in r["scatter"] if s["outlier"]],
                            mode="markers", name="Outliers", marker={"symbol": "x", "size": 10})
        fig.add_scatter(x=r["fit_curve"]["price"], y=r["fit_curve"]["quantity"], mode="lines", name="Fitted demand curve")
        fig.update_layout(title="Quantity against price, with the fitted curve", xaxis_title="Price", yaxis_title="Quantity")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        plot(fig, height=300, showlegend=True)
    with st.expander("Diagnostics and what was left out"):
        data = r["data"]
        out = d["outliers"]
        rows = [f"Rows read: {data['rows_in']:,}. Left out: {data['missing_value']:,} with a missing value, {data['price_not_positive']:,} "
                f"with a price of zero or less, {data['quantity_zero_or_negative']:,} with zero or negative quantity, "
                f"{data['non_finite']:,} not finite.",
                f"Method: {e['method']}. Standard error {num(e['standard_error'], 3)}; p-value {num(e['p_value'], 4)}.",
                f"Controls: trend — {d['controls']['trend']}; seasonality — {d['controls']['seasonality']}; segments — {d['controls']['segments']}.",
                f"Spread of ln(price): {num(d['price_variation_sd_ln'], 3)}. Residual autocorrelation: {num(d['residual_autocorrelation_lag1'], 2)}.",
                f"Outliers: {out['count']} flagged ({out['threshold']})"
                + (f"; without them the estimate is {num(out['without_outliers']['estimate'], 3)}" if out["without_outliers"] else "")
                + ("; they are excluded from this estimate" if out["excluded_from_primary"] else "") + "."]
        for line in rows + [f"Limitation: {x}" for x in r["limitations"]]:
            html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(line)}</div>')
    verification_panel(r.get("verification"), "Independent check of the estimate")
