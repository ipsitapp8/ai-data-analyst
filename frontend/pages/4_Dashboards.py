"""Dashboards — verified KPI cards, agent-generated charts, and the narrative."""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, get_dashboard, get_status, list_questions
from style.theme import (
    badge,
    esc,
    figure_from_json,
    html,
    open_inspect,
    page_header,
    page_setup,
    plot,
    render_inspect_dialog_if_open,
    render_sidebar,
)

page_setup("Dashboards")
render_sidebar("dashboards")

qid = st.session_state.get("active_question_id")

# ---- picker when nothing is selected ----
try:
    questions = list_questions()
except ApiError as e:
    questions = []
    st.error(f"Backend unreachable: {e}")

ready = [q for q in questions if q.get("has_dashboard")]

if not qid:
    page_header("Dashboards", "Generated from verified analyses.")
    if not ready:
        st.info("No dashboards yet — run an analysis first.")
        if st.button("Go to Analyses  →", type="primary", key="db_go"):
            st.switch_page("pages/3_Analyses.py")
        st.stop()
    with st.container(key="flat_pick"):
        html('<div class="ds-card-head"><div class="ds-section-title">'
                    'All Dashboards</div></div>')
        for q in ready:
            c1, c2 = st.columns([5, 1])
            with c1:
                kind = "verified" if q["status"] == "verified" else "warn"
                label = "Verified" if q["status"] == "verified" else "Needs review"
                html(
                    f'<div class="ds-row" style="border-top:none;">'
                    f'<div><div class="ds-row-title">{esc(q["text"][:70])}</div>'
                    f'<div class="ds-row-meta">Updated {q["age"]}</div></div>'
                    f'<div class="ds-row-spacer"></div>{badge(label, kind)}</div>'
                )
            with c2:
                if st.button("Open", key=f"db_open_{q['id']}"):
                    st.session_state["active_question_id"] = q["id"]
                    st.rerun()
    st.stop()

# ---- selected dashboard ----
try:
    status = get_status(qid)
except ApiError as e:
    st.error(f"Could not fetch status: {e}")
    st.stop()

if status["status"] not in ("verified", "unverified"):
    page_header("Dashboards")
    st.warning(f"This analysis isn't finished yet (status: **{status['status']}**).")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Watch it run  →", type="primary", key="db_watch"):
            st.switch_page("pages/3_Analyses.py")
    with c2:
        if st.button("←  All dashboards", key="db_back_pending"):
            st.session_state.pop("active_question_id", None)
            st.rerun()
    st.stop()

try:
    dash = get_dashboard(qid)
except ApiError as e:
    st.error(f"Could not load dashboard: {e}")
    st.stop()

if st.button("←  All dashboards", key="db_back"):
    st.session_state.pop("active_question_id", None)
    st.rerun()
html("<div style='height:4px'></div>")

verified = dash["verified"]
head_l, head_r = st.columns([3, 1])
with head_l:
    html(
        f'<div class="ds-page-title">{esc(status.get("question_text", "Analysis"))}</div>'
    )
    html(
        f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:24px;">'
        f'<span class="ds-row-meta">Analysis #{qid}</span>'
        f'{badge("Verified" if verified else "Needs review", "verified" if verified else "warn")}'
        f"</div>"
    )
with head_r:
    html("<div style='height:12px'></div>")
    if st.button("Audit Trail  →", key="db_audit"):
        st.switch_page("pages/6_Audit_Trail.py")

dashboard_id = dash["id"]

kpis = dash.get("kpis", [])
if kpis:
    cols = st.columns(min(4, len(kpis)), gap="medium")
    for col, kpi in zip(cols, kpis[:4]):
        with col:
            element_id = kpi.get("element_id")
            label = f"{kpi.get('label', '')}  \n**{kpi.get('value', '')}**"
            with st.container(key=f"kpi_{element_id or kpi.get('label', '')}"):
                if st.button(label, key=f"kpibtn_{element_id or kpi.get('label', '')}",
                             use_container_width=True, disabled=not element_id):
                    open_inspect(dashboard_id, element_id)
                    st.rerun()
    html("<div style='height:22px'></div>")

charts = dash.get("charts", [])
if charts:
    for i in range(0, len(charts), 2):
        pair = charts[i:i + 2]
        cols = st.columns(len(pair), gap="medium")
        for col, chart in zip(cols, pair):
            with col:
                element_id = chart.get("element_id")
                container_key = f"chartcard_{element_id}" if element_id else f"card_ch{i}_{chart['step_index']}"
                with st.container(key=container_key):
                    html(f'<div class="ds-section-title">{esc(chart["title"])}</div>')
                    html("<div style='height:8px'></div>")
                    try:
                        fig = figure_from_json(chart["plotly_json"])
                        event = plot(fig, height=300, showlegend=True,
                                     on_select_key=f"chart_{element_id}" if element_id else None)
                        if element_id and event and event.selection and event.selection.get("points"):
                            open_inspect(dashboard_id, element_id)
                            st.rerun()
                    except Exception as e:  # noqa: BLE001 - render one bad chart, not the page
                        html(f'<div class="ds-row-meta">Could not render: {e}</div>')
        html("<div style='height:8px'></div>")

render_inspect_dialog_if_open()

if dash.get("narrative"):
    with st.container(key="card_narr"):
        html('<div class="ds-section-title">Narrative Summary</div>')
        html(
            f'<div style="font-size:0.97rem;line-height:1.75;color:var(--text-primary);'
            f'margin-top:10px;">{esc(dash["narrative"])}</div>'
        )

if dash.get("verification_summary"):
    html("<div style='height:14px'></div>")
    with st.container(key="card_verif"):
        html(
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'{badge("Verified" if verified else "Needs review", "verified" if verified else "warn")}'
            f'<div class="ds-section-title">Verification</div></div>'
            f'<div class="ds-row-meta" style="margin-top:10px;line-height:1.7;">'
            f'{esc(dash["verification_summary"])}</div>'
        )
