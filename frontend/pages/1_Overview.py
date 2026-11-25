"""Overview — workspace landing: counters, recent analyses, insights chart, dashboards."""
from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from api_client import ApiError, list_datasets, list_questions
from auth import current_user
from style.theme import ACCENT, badge, esc, html, page_setup, plot, render_sidebar, stat_card, require_active_team

page_setup("Overview")
render_sidebar("overview")
require_active_team("Overview")


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

user = current_user()
first_name = (user["display_name"].split() or [user["display_name"]])[0] if user else ""

head_l, head_r = st.columns([3, 1])
with head_l:
    html(
        f'<div class="ds-page-title">{_greeting()}{", " + esc(first_name) if first_name else ""}</div>'
        '<div class="ds-page-sub">Here\'s what\'s happening in your workspace.</div>'
    )
with head_r:
    html("<div style='height:8px'></div>")
    if st.button("＋  New Analysis", type="primary", key="ov_new"):
        st.switch_page("pages/3_Analyses.py")

from style.theme import page_intro
page_intro("Your workspace at a glance: what you have uploaded, asked and verified, and where to go next.")
_steps = [
    ("Upload your data", "Add a CSV or Excel file so Silt has something to read.", bool(datasets), "pages/2_Datasets.py", "Upload data"),
    ("Ask your first question", "Type it in everyday language. Silt runs the analysis and checks it.", bool(questions), "pages/3_Analyses.py", "Ask a question"),
    ("Read your dashboard", "Every verified answer comes with charts and a report.", bool(verified), "pages/4_Dashboards.py", "Open dashboards"),
]
_next = next((s for s in _steps if not s[2]), None)
if _next:
    _rows = "".join(
        f'<div style="display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-top:1px solid rgba(242,239,231,.08);">'
        f'<div style="width:22px;height:22px;border-radius:50%;flex:0 0 22px;display:grid;place-items:center;font:600 12px/1 Inter,system-ui,sans-serif;'
        f'{"background:#f3b24f;color:#1d1406;" if done else "border:1px solid rgba(242,239,231,.35);color:#bdb9cc;"}">{"✓" if done else i}</div>'
        f'<div><div style="font:600 14.5px/1.3 Inter,system-ui,sans-serif;color:#f2efe7;">{esc(t)}</div>'
        f'<div style="font:400 13px/1.5 Inter,system-ui,sans-serif;color:#8c88a3;">{esc(d)}</div></div></div>'
        for i, (t, d, done, _p, _b) in enumerate(_steps, 1))
    with st.container(key="card_getstarted"):
        html('<div class="ds-section-title">Get started in three steps</div>' + _rows)
        if st.button(f"{_next[4]}  →", type="primary", key="ov_next_step"):
            st.switch_page(_next[3])
    html("<div style='height:18px'></div>")

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
                '<div class="ds-empty">No analyses yet — '
                'start one from the Analyses page.</div>'
            )
        icon = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="1.8"><path d="M4 18l5-6 4 3.5L20 7" stroke-linecap="round" '
                'stroke-linejoin="round"/></svg>')
        for q in questions[:4]:
            kind = {"verified": "verified", "unverified": "warn", "rejected": "warn", "cancelled": "neutral", "needs_clarification": "warn",
                    "failed": "error"}.get(q["status"], "running")
            label = {"verified": "Verified", "unverified": "Needs review", "rejected": "Not analyzable", "cancelled": "Cancelled",
                     "needs_clarification": "Needs clarification",
                     "failed": "Failed"}.get(q["status"], "Running")
            state = "Completed" if q["status"] in ("verified", "unverified") else q["status"].capitalize()
            html(
                f"""
                <div class="ds-row">
                  <div class="ds-row-icon">{icon}</div>
                  <div>
                    <div class="ds-row-title">{esc(q["text"][:58])}</div>
                    <div class="ds-row-meta">{esc(state)} • {esc(q["age"])}</div>
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
                line=dict(color=ACCENT, width=2.4, shape="spline"),
                marker=dict(color=ACCENT, size=7),
                fill="tozeroy", fillcolor="rgba(79,70,229,0.10)",
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
                    f'<rect x="{2+i*5}" y="{26-h}" width="3.4" height="{h}" rx="1.2" fill="{ACCENT}" '
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
                        <div class="ds-row-title">{esc(q["text"][:30])}</div>
                        <div class="ds-row-meta">Updated {esc(q["age"])}</div>
                      </div>
                    </div>
                    """
                )
                if st.button("Open  →", key=f"ov_dash_{q['id']}"):
                    st.session_state["active_question_id"] = q["id"]
                    st.switch_page("pages/4_Dashboards.py")
    else:
        html('<div class="ds-empty">No verified dashboards yet.</div>')
    html("</div>")
