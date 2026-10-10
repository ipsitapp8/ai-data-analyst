"""Data quality — per-version checks, drift against a baseline, incidents and
the thresholds behind them."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import (ApiError, compare_dataset_versions, get_quality, list_datasets, run_quality,
                        set_quality_config, update_incident)
from style.theme import badge, esc, html, page_header, page_setup, plot, render_sidebar, require_active_team

page_setup("Data quality")
render_sidebar("data_quality")
require_active_team("Data quality")
page_header("Data quality and drift", "What changed in the data itself, kept separate from changes in the business.")

try:
    datasets = list_datasets()
except ApiError as e:
    st.error(f"Backend unreachable: {e}")
    st.stop()
if not datasets:
    st.info("No dataset yet — upload a CSV first.")
    st.stop()

active = st.session_state.get("active_dataset_id")
ds = st.selectbox("Dataset", datasets, index=next((i for i, d in enumerate(datasets) if d["id"] == active), 0),
                  format_func=lambda d: f"{d['filename']} · {d.get('version_count', 1)} version(s)", key="dq_ds")
try:
    q = get_quality(ds["id"])
except ApiError as e:
    st.error(str(e))
    st.stop()

history = q["history"]
if not any(h["has_snapshot"] for h in history):
    st.info("This dataset was uploaded before quality checks existed.")
    if st.button("Check it now", type="primary", key="dq_run"):
        try:
            run_quality(ds["id"])
            st.rerun()
        except ApiError as e:
            st.error(str(e))
    st.stop()

base = q["baseline"]
open_n = q["open_incidents"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest version", q["latest_version"])
k2.metric("Compared with", f"version {base['version']}" if base["version"] else "nothing yet",
          "pinned baseline" if base["pinned"] else (base["source"] if base["version"] else None), delta_color="off")
k3.metric("Open incidents", open_n)
k4.metric("Rows (latest)", f"{history[-1]['rows']:,}" if history[-1]["rows"] is not None else "—")

CATEGORY = {"schema": ("Schema", "error"), "quality": ("Data quality", "warn"), "distribution": ("Distribution drift", "running")}
SEVERITY = {"high": "error", "warn": "warn", "info": "neutral"}
html('<div class="ds-section-title" style="margin-top:16px;">Incidents</div>'
     '<div class="ds-row-meta">Schema changes break analyses. Quality failures make numbers wrong. Distribution '
     "drift means the values are valid but the population changed. KPI anomalies are reported separately, on the "
     "Scheduled page.</div>")
show = st.radio("Show", ["Open", "All"], horizontal=True, key="dq_show", label_visibility="collapsed")
incidents = [i for i in q["incidents"] if show == "All" or i["status"] != "resolved"]
if not incidents:
    html('<div class="ds-row-meta" style="padding:14px 0;">No incidents'
         + (" on record." if show == "All" else " open. The latest data matches its baseline within the thresholds below.")
         + "</div>")
versions = {h["version_id"]: h["version"] for h in history}
for i in incidents:
    label, kind = CATEGORY.get(i["category"], (i["category"], "neutral"))
    c1, c2 = st.columns([6, 1.6])
    with c1:
        html(f'<div style="padding:8px 0;border-top:1px solid var(--border);"><div style="display:flex;gap:8px;align-items:center;">'
             f'{badge(label, kind)}{badge(i["severity"].title(), SEVERITY.get(i["severity"], "neutral"))}'
             f'<span class="ds-row-title">{esc(i["check"].replace("_", " "))}'
             f'{" · " + esc(i["column"]) if i["column"] else ""}</span>'
             f'<span class="ds-row-meta">version {versions.get(i["dataset_version_id"], "?")} · {esc(i["status"])}</span></div>'
             f'<div class="ds-row-meta" style="margin-top:3px;">{esc(i["message"])}</div></div>')
    with c2:
        a, b = st.columns(2)
        if i["status"] == "open" and a.button("Ack", key=f"dq_ack_{i['id']}"):
            try:
                update_incident(i["id"], "acknowledge")
            except ApiError as e:
                st.error(str(e))
            st.rerun()
        if i["status"] != "resolved" and b.button("Resolve", key=f"dq_res_{i['id']}"):
            try:
                update_incident(i["id"], "resolve")
            except ApiError as e:
                st.error(str(e))
            st.rerun()

if len(history) > 1:
    html('<div class="ds-section-title" style="margin-top:18px;">History by version</div>')
    frame = pd.DataFrame([{"Version": h["version"], "Rows": h["rows"], "Average missing %": h["avg_missing_pct"],
                           "Duplicate rows %": h["duplicate_pct"], "Schema": h["incidents"]["schema"],
                           "Quality": h["incidents"]["quality"], "Drift": h["incidents"]["distribution"],
                           "Largest drift (PSI)": h["max_drift_psi"]} for h in history])
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=frame["Version"], y=frame["Rows"], mode="lines+markers"))
        fig.update_layout(title="Rows per version", xaxis_title="Version")
        plot(fig, height=250)
    with right:
        fig = go.Figure()
        for col in ("Schema", "Quality", "Drift"):
            fig.add_bar(x=frame["Version"], y=frame[col], name=col)
        fig.update_layout(barmode="stack", title="Incidents per version", xaxis_title="Version")
        plot(fig, height=250, showlegend=True, restyle_traces=False)
    st.dataframe(frame, use_container_width=True, hide_index=True)

    with st.expander("Compare two versions"):
        opts = {h["version"]: h["version_id"] for h in history}
        c1, c2 = st.columns(2)
        va = c1.selectbox("Earlier", list(opts), index=max(0, len(opts) - 2), key="dq_va")
        vb = c2.selectbox("Later", list(opts), index=len(opts) - 1, key="dq_vb")
        if st.button("Compare", key="dq_cmp"):
            try:
                cmp = compare_dataset_versions(ds["id"], opts[va], opts[vb])
                if cmp["identical_content"]:
                    st.success("Both versions are byte-for-byte the same file.")
                st.markdown(f"Rows: **{cmp['a']['rows']:,} → {cmp['b']['rows']:,}** ({cmp['row_change']:+,}).")
                for f in cmp["findings"]:
                    st.markdown(f"- **{f['category']} · {f['check'].replace('_', ' ')}** — {f['message']}")
                if cmp["skipped"]:
                    st.caption("Not compared (sample too small): " + ", ".join(s["column"] for s in cmp["skipped"]))
                st.dataframe(pd.DataFrame(cmp["columns"]), use_container_width=True, hide_index=True)
            except ApiError as e:
                st.error(str(e))

with st.expander("Baseline and thresholds"):
    st.caption("Changes here apply to future uploads. Only a team owner or admin can save them. For seasonal data, "
               "pin the baseline to a version from the comparable season rather than the previous upload.")
    t = q["thresholds"]
    choices = [None] + [h["version_id"] for h in history]
    current = base["version_id"] if base["pinned"] else None
    baseline_id = st.selectbox("Baseline version", choices, index=choices.index(current) if current in choices else 0,
                               format_func=lambda v: "The previous version (default)" if v is None else f"Version {versions[v]}",
                               key="dq_baseline")
    c1, c2, c3 = st.columns(3)
    new = {
        "min_sample": c1.number_input("Minimum values needed to compare distributions", 1.0, 1e6, float(t["min_sample"]), key="dq_t_min"),
        "psi_warn": c2.number_input("Drift (PSI): warning from", 0.0, 5.0, float(t["psi_warn"]), 0.01, key="dq_t_pw"),
        "psi_high": c3.number_input("Drift (PSI): high from", 0.0, 5.0, float(t["psi_high"]), 0.01, key="dq_t_ph"),
        "row_count_change_pct_warn": c1.number_input("Row count change %: warning", 0.0, 1000.0, float(t["row_count_change_pct_warn"]), key="dq_t_rw"),
        "row_count_change_pct_high": c2.number_input("Row count change %: high", 0.0, 1000.0, float(t["row_count_change_pct_high"]), key="dq_t_rh"),
        "duplicate_pct_warn": c3.number_input("Duplicate rows %: warning", 0.0, 100.0, float(t["duplicate_pct_warn"]), key="dq_t_dw"),
        "missing_increase_pts_warn": c1.number_input("Missing values: rise in points, warning", 0.0, 100.0, float(t["missing_increase_pts_warn"]), key="dq_t_mw"),
        "missing_increase_pts_high": c2.number_input("Missing values: rise in points, high", 0.0, 100.0, float(t["missing_increase_pts_high"]), key="dq_t_mh"),
    }
    fresh_on = c3.checkbox("Check freshness", value=t.get("freshness_max_age_days") is not None, key="dq_t_fon")
    if fresh_on:
        new["freshness_max_age_days"] = c3.number_input("Newest date may be at most this many days old", 0.0, 3650.0,
                                                        float(t.get("freshness_max_age_days") or 7.0), key="dq_t_fd")
    if st.button("Save", type="primary", key="dq_save"):
        try:
            set_quality_config(ds["id"], baseline_id, new)
            st.success("Saved.")
            st.rerun()
        except ApiError as e:
            st.error(str(e))
