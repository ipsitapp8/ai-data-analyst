"""What-if by segment: change inputs for some regions, products or campaigns and see each segment and the total.

Rendering only. Segments, baselines, allocation, totals, contributions and the independent check all come from the backend
(`/api/scenarios/segments`). The total is the sum of the segments, so nothing is counted twice, and the page says how
many rows are in scope and how many the segments hold.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, run_segments, segment_values
from style.theme import badge, esc, html, plot, stat_card
from what_if_decision import num, pct, section, verification_panel
from what_if_time import MAPPING_FIELDS, legend, period_picker, version_picker

MAX_ADJUSTED = 8
MODES = {"data": "From the data", "constant": "Same value in every segment", "total": "A total to share out"}


def _category_columns(ds: dict) -> list[str]:
    return [c["name"] for c in ds["profile"].get("columns", []) if c.get("kind") == "categorical"]


def segments_tab(models: dict, datasets: list[dict]) -> None:
    section("Segments", "Change inputs for some segments and see each segment and the total. Segments split the rows by one column, "
                        "so every row is in exactly one segment and nothing is counted twice; the total is the sum of the segments.")
    legend()
    if not datasets:
        st.caption("Upload a dataset to work by segment.")
        return
    model_key = st.selectbox("Model", list(models), format_func=lambda k: models[k]["title"], key="sg_model")
    model = models[model_key]
    ds = st.selectbox("Dataset", datasets, format_func=lambda d: d["filename"], key="sg_ds")
    cats = _category_columns(ds)
    if not cats:
        st.caption("This dataset has no category column (region, product, campaign…) to split by.")
        return
    version = version_picker("sg", ds)
    c1, c2 = st.columns(2)
    segment_column = c1.selectbox("Split by", cats, key="sg_column")
    with c2:
        date_column, period = period_picker("sg", ds, version)

    mapping = {}
    if model_key in MAPPING_FIELDS:
        numeric = [None] + [c["name"] for c in ds["profile"].get("columns", []) if c.get("kind") == "numeric"]
        fields = MAPPING_FIELDS[model_key]
        for col, f in zip(st.columns(len(fields)), fields):
            mapping[f] = col.selectbox(f.replace("_", " ").title(), numeric, key=f"sg_map_{model_key}_{f}", format_func=lambda v: v or "—")
    else:
        st.caption("This model's inputs are bound to approved metrics or formulas in its definition.")

    try:
        listing = segment_values(ds["id"], segment_column, version)
    except ApiError as e:
        st.error(str(e))
        return
    sizes = {s["value"]: s["rows"] for s in listing["segments"]}
    st.caption(f"{listing['total_segments']} segment(s) in '{segment_column}'"
               + (f" (showing the largest {len(sizes)}; filter the data to work with the rest)" if listing["truncated"] else "") + ".")

    chosen = st.multiselect(f"Segments to adjust (up to {MAX_ADJUSTED})", list(sizes), format_func=lambda v: f"{v} ({sizes[v]:,} rows)",
                            key="sg_pick", max_selections=MAX_ADJUSTED)
    seg_adj: dict[str, dict] = {}
    for v in chosen:
        st.markdown(f"**{v}** — change, in %")
        cols = st.columns(len(model["inputs"]))
        adj = {}
        for col, (name, rule) in zip(cols, model["inputs"].items()):
            x = col.number_input(rule["label"], -100.0, 1000.0, 0.0, 1.0, key=f"sg_adj_{model_key}_{v}_{name}")
            if x:
                adj[name] = {"type": "pct", "value": x}
        if adj:
            seg_adj[v] = adj
    shared_adj: dict[str, dict] = {}
    with st.expander("Change an input in every segment (a segment's own change above replaces this one)"):
        cols = st.columns(len(model["inputs"]))
        for col, (name, rule) in zip(cols, model["inputs"].items()):
            x = col.number_input(f"{rule['label']} %", -100.0, 1000.0, 0.0, 1.0, key=f"sg_shared_{model_key}_{name}")
            if x:
                shared_adj[name] = {"type": "pct", "value": x}

    constants, shared_inputs = {}, {}
    with st.expander("Values your data does not split by segment (a fixed cost, a unit cost…)"):
        st.caption("Leave an input 'From the data' if it is mapped or bound. Otherwise give it a value for every segment, or a total "
                   "that is shared out once, in shares that add to 100%.")
        for name, rule in model["inputs"].items():
            a, b, c = st.columns([3, 3, 3])
            mode = a.selectbox(rule["label"], list(MODES), format_func=MODES.get, key=f"sg_mode_{model_key}_{name}")
            if mode == "data":
                continue
            value = b.number_input("Value" if mode == "constant" else "Total", value=0.0, key=f"sg_val_{model_key}_{name}", format="%.4f")
            if mode == "constant":
                constants[name] = float(value)
            else:
                others = ["equal"] + [k for k in model["inputs"] if k != name]
                by = c.selectbox("Share it out", others, format_func=lambda k: "Equally" if k == "equal" else f"In proportion to {model['inputs'][k]['label']}",
                                 key=f"sg_by_{model_key}_{name}")
                shared_inputs[name] = {"total": float(value), "allocate_by": by}

    assumptions = {k: float(r["default"]) for k, r in model["assumptions"].items()}
    if "price_elasticity" in assumptions:
        assumptions["price_elasticity"] = st.slider("Price elasticity (how volume responds to a price change)", float(model["assumptions"]["price_elasticity"]["min"]),
                                                    float(model["assumptions"]["price_elasticity"]["max"]), 0.0, 0.01, key="sg_elasticity")
    body = {"dataset_id": ds["id"], "dataset_version": version, "model": model_key, "mapping": {k: v for k, v in mapping.items() if v},
            "segment_column": segment_column, "assumptions": assumptions, "shared_adjustments": shared_adj, "segment_adjustments": seg_adj,
            "shared_inputs": shared_inputs, "constant_inputs": constants}
    if date_column:
        body.update(date_column=date_column, period=period)
    if st.button("Run by segment", type="primary", key="sg_run"):
        try:
            with st.spinner("Splitting the data, running each segment and checking the totals…"):
                st.session_state["sg_result"] = run_segments(body)
        except ApiError as e:
            st.session_state.pop("sg_result", None)
            st.error(f"The segment run could not complete: {e}")
    r = st.session_state.get("sg_result")
    if r:
        _show(r, model)


def _show(r: dict, model: dict) -> None:
    part, prov = r["partition"], r["provenance"]
    html(f'<div class="ds-row-meta" style="padding:6px 0;">{badge("Historical actuals", "neutral")} baselines + {badge("Scenario", "neutral")} adjustments · '
         f'{esc(r["period"]["label"])} · dataset version {prov["dataset_version"]} · {part["rows_in_scope"]:,} rows in scope, '
         f'{part["rows_in_segments"]:,} in {part["segments"]} segments '
         f'{badge("Every row counted once", "verified") if part["complete"] else badge("Rows missing", "error")}</div>')
    primary = r["primary_output"]
    label = model["outputs"][primary]
    o = r["overall"]
    cards = [(f"Total {label.lower()}, baseline", num(o["baseline"]["outputs"][primary]), ""),
             (f"Total {label.lower()}, scenario", num(o["scenario"]["outputs"][primary]), ""),
             ("Change", num(o["delta"].get(primary, {}).get("abs") if o["delta"].get(primary) else None), ""),
             ("Change %", pct(o["delta"].get(primary, {}).get("pct") if o["delta"].get(primary) else None), "")]
    for col, card in zip(st.columns(4), cards):
        col.markdown(stat_card(*card), unsafe_allow_html=True)
    cons = r["contributions"]
    rows = []
    for s in r["segments"]:
        d = s["delta"].get(primary) or {"abs": None, "pct": None}
        c = (cons or {}).get("by_segment", {}).get(s["value"], {})
        rows.append({"Segment": s["value"] + (" (thin)" if s["thin"] else ""), "Rows": s["rows"], f"{label}, baseline": s["baseline"]["outputs"].get(primary),
                     f"{label}, scenario": s["scenario"]["outputs"].get(primary), "Change": d["abs"], "Change %": d["pct"],
                     "Share of total change %": c.get("share_pct"),
                     "Adjusted": ", ".join(f"{model['inputs'][k]['label']} {v['value']:+g}%" for k, v in s["adjustments"].items()) or "—"})
    frame = pd.DataFrame(rows)
    st.dataframe(frame.style.format({f"{label}, baseline": "{:,.2f}", f"{label}, scenario": "{:,.2f}", "Change": "{:+,.2f}", "Change %": "{:+.1f}%",
                                      "Share of total change %": "{:.1f}%"}, na_rep="—"), use_container_width=True, hide_index=True)
    if cons and any(c["abs"] for c in cons["by_segment"].values()):
        fig = go.Figure(go.Waterfall(x=list(cons["by_segment"]) + ["Total change"], y=[c["abs"] for c in cons["by_segment"].values()] + [cons["total_change"]],
                                     measure=["relative"] * len(cons["by_segment"]) + ["total"]))
        fig.update_layout(title=f"Each segment's share of the change in {label.lower()}")
        plot(fig, height=300, restyle_traces=False)
        st.caption("The segments' changes add up to the total change.")
    elif cons is None:
        st.caption("The headline output is not additive across segments, so the change is not split by segment.")
    shown = [k for k in model["outputs"] if k in o["additive_outputs"] or k in o["ratio_outputs"]]
    if shown:
        totals = pd.DataFrame({model["outputs"][k]: {"Baseline": o["baseline"]["outputs"][k], "Scenario": o["scenario"]["outputs"][k]} for k in shown}).T
        st.dataframe(totals.style.format("{:,.2f}", na_rep="—"), use_container_width=True)
    if o["not_additive"]:
        st.caption("Not added up across segments (they are not additive): " + ", ".join(model["outputs"][k] for k in o["not_additive"]) + ".")
    for d in r["disclosures"]:
        html(f'<div class="ds-row-meta" style="padding:3px 0;">• {esc(d)}</div>')
    verification_panel(r.get("verification"), "Independent check of the segments")
    with st.expander("Where each segment's inputs came from"):
        for s in r["segments"]:
            html(f'<div class="ds-row-meta" style="padding:2px 0;"><b>{esc(s["value"])}</b>: ' + "; ".join(
                f"{esc(model['inputs'][k]['label'])} — {esc(v['source'].replace('_', ' '))}" for k, v in s["sources"].items()) + "</div>")
