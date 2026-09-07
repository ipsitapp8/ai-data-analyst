"""Datasets — upload CSVs, browse sources, inspect the auto-generated profile."""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, get_dataset, list_datasets, upload_dataset
from style.theme import html, page_header, page_setup, render_sidebar

page_setup("Datasets")
render_sidebar("datasets")

head_l, head_r = st.columns([2.4, 1])
with head_l:
    page_header("Datasets", "Manage and explore your data sources.")
with head_r:
    html("<div style='height:10px'></div>")
    b1, b2 = st.columns(2)
    with b1:
        upload_open = st.button("Upload Data", key="ds_upload_btn")
    with b2:
        st.button("＋  Connect Source", type="primary", key="ds_connect_btn",
                  help="Database / cloud sources are not part of this MVP yet.")

if upload_open:
    st.session_state["show_uploader"] = not st.session_state.get("show_uploader", False)

if st.session_state.get("show_uploader"):
    with st.container(key="card_upload"):
        html('<div class="ds-section-title">Upload a CSV</div>')
        html("<div style='height:10px'></div>")
        f = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        if f is not None and st.button("Profile this dataset", type="primary", key="ds_do_upload"):
            with st.spinner("Uploading & profiling…"):
                try:
                    res = upload_dataset(f.name, f.getvalue())
                    st.session_state["active_dataset_id"] = res["id"]
                    st.session_state["show_uploader"] = False
                    st.success(f"Profiled **{res['filename']}** — "
                               f"{res['row_count']:,} rows, {res['col_count']} columns.")
                    st.rerun()
                except ApiError as e:
                    st.error(f"Upload failed: {e}")
    html("<div style='height:14px'></div>")

try:
    datasets = list_datasets()
except ApiError as e:
    datasets = []
    st.error(f"Backend unreachable: {e}")

FILE_ICON = ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.7"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" '
             'stroke-linejoin="round"/><path d="M14 3v5h5" stroke-linejoin="round"/></svg>')

with st.container(key="flat_table"):
    if not datasets:
        html(
            '<div style="padding:34px 22px;text-align:center;">'
            '<div class="ds-row-title" style="margin-bottom:5px;">No datasets yet</div>'
            '<div class="ds-row-meta">Click <b>Upload Data</b> above to add your first CSV.</div>'
            "</div>"
        )
    else:
        html(
            """
            <div style="display:flex;padding:13px 22px;border-bottom:1px solid var(--border);
                        color:var(--text-secondary);font-size:0.79rem;">
              <div style="flex:2.6;">Name</div>
              <div style="flex:0.8;">Rows</div>
              <div style="flex:0.8;">Columns</div>
              <div style="flex:0.9;">Source</div>
              <div style="flex:1.1;">Last Updated</div>
              <div style="flex:1.3;">Quality</div>
            </div>
            """
        )
        for d in datasets:
            profile = d.get("profile", {})
            cols = profile.get("columns", [])
            total_cells = max(d["row_count"] * max(d["col_count"], 1), 1)
            missing = sum(c.get("missing_count", 0) for c in cols)
            quality = round(max(0.0, 100.0 - (missing / total_cells * 100)), 1)
            html(
                f"""
                <div class="ds-row" style="padding:14px 22px;">
                  <div style="flex:2.6;display:flex;align-items:center;gap:11px;">
                    <div class="ds-row-icon" style="width:29px;height:29px;">{FILE_ICON}</div>
                    <div class="ds-row-title" style="font-weight:500;">{d['filename']}</div>
                  </div>
                  <div style="flex:0.8;" class="ds-row-meta">{d['row_count']:,}</div>
                  <div style="flex:0.8;" class="ds-row-meta">{d['col_count']}</div>
                  <div style="flex:0.9;" class="ds-row-meta">CSV</div>
                  <div style="flex:1.1;" class="ds-row-meta">#{d['id']}</div>
                  <div style="flex:1.3;">
                    <div class="ds-quality">
                      <span class="ds-row-meta" style="min-width:38px;">{quality}%</span>
                      <span class="ds-quality-track">
                        <span class="ds-quality-fill" style="width:{quality}%;"></span>
                      </span>
                    </div>
                  </div>
                </div>
                """
            )

if datasets:
    html("<div style='height:20px'></div>")
    labels = [f"{d['filename']}  ·  #{d['id']}" for d in datasets]
    active = st.session_state.get("active_dataset_id")
    default = next((i for i, d in enumerate(datasets) if d["id"] == active), 0)

    sel_l, sel_r = st.columns([3, 1])
    with sel_l:
        idx = st.selectbox("Inspect dataset", range(len(datasets)),
                           index=default, format_func=lambda i: labels[i])
    with sel_r:
        html("<div style='height:28px'></div>")
        if st.button("Analyze this  →", type="primary", key="ds_go_analyze"):
            st.session_state["active_dataset_id"] = datasets[idx]["id"]
            st.switch_page("pages/3_Analyses.py")

    chosen = datasets[idx]
    st.session_state["active_dataset_id"] = chosen["id"]

    try:
        full = get_dataset(chosen["id"])
    except ApiError as e:
        full = None
        st.error(f"Could not load profile: {e}")

    if full:
        profile = full["profile"]
        html("<div style='height:6px'></div>")
        with st.container(key="flat_profile"):
            html(
                f'<div class="ds-card-head"><div class="ds-section-title">'
                f'Profile — {full["filename"]}</div></div>'
            )
            html(
                """
                <div style="display:flex;padding:11px 22px;border-bottom:1px solid var(--border);
                            color:var(--text-secondary);font-size:0.78rem;">
                  <div style="flex:1.6;">Column</div><div style="flex:0.9;">Type</div>
                  <div style="flex:1.1;">Missing</div><div style="flex:0.8;">Unique</div>
                  <div style="flex:2.4;">Detail</div>
                </div>
                """
            )
            for c in profile["columns"]:
                detail = ""
                if c["kind"] == "numeric" and c.get("stats"):
                    s = c["stats"]
                    if s.get("mean") is not None:
                        detail = (f"min {s['min']:.2f} · max {s['max']:.2f} · "
                                  f"mean {s['mean']:.2f}")
                elif c["kind"] == "categorical" and c.get("top_values"):
                    detail = ", ".join(f"{t['value']} ({t['count']})"
                                       for t in c["top_values"][:3])
                elif c["kind"] == "datetime" and c.get("stats"):
                    detail = f"{c['stats'].get('min')} → {c['stats'].get('max')}"
                html(
                    f"""
                    <div class="ds-row" style="padding:11px 22px;">
                      <div style="flex:1.6;" class="ds-row-title" >{c['name']}</div>
                      <div style="flex:0.9;" class="ds-row-meta">{c['kind']}</div>
                      <div style="flex:1.1;" class="ds-row-meta">
                        {c['missing_count']} ({c['missing_pct']}%)</div>
                      <div style="flex:0.8;" class="ds-row-meta">{c['unique_count']}</div>
                      <div style="flex:2.4;" class="ds-row-meta">{detail}</div>
                    </div>
                    """
                )
