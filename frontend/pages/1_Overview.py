"""Overview — workspace landing: counters, recent analyses, insights chart, dashboards."""
from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, list_datasets, list_questions
from style.theme import GREEN, ROSE, badge, html, page_setup, plot, render_sidebar, stat_card

page_setup("Overview")
render_sidebar("overview")


def _greeting() -> str:
    h = dt.datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 17 else "Good evening")


try:
    datasets = list_datasets()
except ApiError:
    datasets = []
try:
    questions = list_questions()
except ApiError:
    questions = []

verified = [q for q in questions if q["status"] == "verified"]

head_l, head_r = st.columns([3, 1])
with head_l:
    html(
        f'<div class="ds-page-title">{_greeting()}, Ipsita 🌿</div>'
        '<div class="ds-page-sub">Here\'s what\'s happening in your workspace.</div>'
    )
with head_r:
    html("<div style='height:8px'></div>")
    if st.button("＋  New Analysis", type="primary", key="ov_new"):
        st.switch_page("pages/3_Analyses.py")

cols = st.columns(4, gap="medium")
stats = [
    ("Datasets", str(len(datasets))),
    ("Analyses", str(len(questions))),
    ("Dashboards", str(len(verified))),
    ("Insights Generated", str(sum(q.get("kpi_count", 0) for q in questions))),
]
for col, (label, value) in zip(cols, stats):
    with col:
        html(stat_card(label, value))

html("<div style='height:22px'></div>")

left, right = st.columns([1.05, 1], gap="medium")

# ---- recent analyses ----
with left:
    with st.container(key="flat_recent"):
        html(
            '<div class="ds-card-head"><div class="ds-section-title">Recent Analyses</div></div>'
        )
        if not questions:
            html(
                '<div class="ds-row"><div class="ds-row-meta">No analyses yet — '
                'start one from the Analyses page.</div></div>'
            )
        icon = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="1.8"><path d="M4 18l5-6 4 3.5L20 7" stroke-linecap="round" '
                'stroke-linejoin="round"/></svg>')
        for q in questions[:4]:
            kind = {"verified": "verified", "unverified": "warn", "rejected": "warn",
                    "failed": "error"}.get(q["status"], "running")
            label = {"verified": "Verified", "unverified": "Needs review", "rejected": "Not analyzable",
                     "failed": "Failed"}.get(q["status"], "Running")
            state = "Completed" if q["status"] in ("verified", "unverified") else q["status"].capitalize()
            html(
                f"""
                <div class="ds-row">
                  <div class="ds-row-icon">{icon}</div>
                  <div>
                    <div class="ds-row-title">{q["text"][:58]}</div>
                    <div class="ds-row-meta">{state} • {q["age"]}</div>
                  </div>
                  <div class="ds-row-spacer"></div>
                  {badge(label, kind)}
                </div>
                """
            )

# ---- insights chart ----
with right:
    with st.container(key="card_insights"):
        html('<div class="ds-section-title">Insights at a Glance</div>')
        html("<div style='height:10px'></div>")

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        counts = [0] * 7
        for q in questions:
            idx = q.get("weekday")
            if isinstance(idx, int) and 0 <= idx < 7:
                counts[idx] += 1

        fig = go.Figure(
            go.Scatter(
                x=days, y=counts, mode="lines+markers",
                line=dict(color=ROSE, width=2.4, shape="spline"),
                marker=dict(color=ROSE, size=7),
                fill="tozeroy", fillcolor="rgba(239,130,150,0.10)",
                hovertemplate="%{x}: %{y} analyses<extra></extra>",
            )
        )
        fig.update_yaxes(rangemode="tozero")
        plot(fig, height=252)

html("<div style='height:22px'></div>")

# ---- recent dashboards ----
with st.container(key="flat_dash"):
    html(
        '<div class="ds-card-head"><div class="ds-section-title">Recent Dashboards</div></div>'
    )
    html('<div style="padding:0 22px 20px 22px;">')
    if verified:
        cols = st.columns(min(3, len(verified)), gap="medium")
        bars = ('<svg width="30" height="26" viewBox="0 0 34 26" fill="none">'
                + "".join(
                    f'<rect x="{2+i*5}" y="{26-h}" width="3.4" height="{h}" rx="1.2" fill="{GREEN}" '
                    f'opacity="{0.45 + i*0.09}"/>'
                    for i, h in enumerate([9, 15, 11, 19, 13, 22])
                )
                + "</svg>")
        for col, q in zip(cols, verified[:3]):
            with col:
                html(
                    f"""
                    <div class="ds-card" style="display:flex;align-items:center;gap:14px;">
                      {bars}
                      <div>
                        <div class="ds-row-title">{q["text"][:30]}</div>
                        <div class="ds-row-meta">Updated {q["age"]}</div>
                      </div>
                    </div>
                    """
                )
                if st.button("Open  →", key=f"ov_dash_{q['id']}"):
                    st.session_state["active_question_id"] = q["id"]
                    st.switch_page("pages/4_Dashboards.py")
    else:
        html('<div class="ds-row-meta">No verified dashboards yet.</div>')
    html("</div>")
