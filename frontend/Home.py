"""Landing page — hero, capability strip, how it works, and a live workspace snapshot.

The dataset/analysis counts and recent activity come straight from the
backend, same as the Overview page. The hero stage is drawn in CSS (no image asset).
"""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, list_datasets, list_questions
from style.theme import esc, html, page_setup, render_sidebar, verdict_badge

page_setup("Home")
render_sidebar("home")

_has_team = bool(st.session_state.get("active_team_id"))

try:
    _datasets = list_datasets() if _has_team else []
except ApiError:
    _datasets = []
try:
    _questions = list_questions() if _has_team else []
except ApiError:
    _questions = []

_verified = [q for q in _questions if q.get("verdict_state") in ("VERIFIED", "VERIFIED_WITH_CAVEATS")
             or (not q.get("verdict_state") and q["status"] == "verified")]


def _icon(body: str, size: int = 26) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


I_UPLOAD = _icon('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M12 17v-5M9.5 14.5L12 12l2.5 2.5"/>')
I_PLAN = _icon('<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>')
I_CODE = _icon('<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 5l-4 14"/>')
I_VERIFY = _icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>')
I_CHART = _icon('<path d="M5 20V11M12 20V4M19 20v-6"/>')
I_DB = _icon('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>', 16)
I_SPARK = _icon('<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>', 22)

I_SHIELD = _icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6z"/>', 17)
I_BOX = _icon('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5L12 12l8-4.5M12 12v9"/>', 17)
I_LOG = _icon('<path d="M6 4h12v16H6z"/><path d="M9 9h6M9 13h6M9 17h3"/>', 17)
I_DASH = _icon('<rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/>', 17)
I_FILE = _icon('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>', 17)

EXAMPLE_QUESTIONS = [
    "Which region grew fastest last quarter?",
    "Why did revenue dip in March?",
    "Which customers are most at risk of churning?",
    "Is the lift from the new campaign significant?",
]


def _steps() -> str:
    data = [
        (I_UPLOAD, "1. Upload", "Add your CSV or<br>connect your database."),
        (I_PLAN, "2. Plan", "The agent analyses your<br>question and creates a plan."),
        (I_CODE, "3. Execute", "Writes and runs code<br>in a secure sandbox."),
        (I_VERIFY, "4. Verify", "Independent model<br>checks the results."),
        (I_CHART, "5. Visualize", "Get a clean dashboard<br>and actionable insights."),
    ]
    parts = []
    for i, (ico, name, desc) in enumerate(data):
        parts.append(f'<div class="h-step"><div class="h-step-ico">{ico}</div>'
                     f'<div class="h-step-name">{name}</div><div class="h-step-desc">{desc}</div></div>')
        if i < len(data) - 1:
            parts.append('<div class="h-arrow">→</div>')
    return "".join(parts)


# ------------------------------------------------------------------- hero --
if not _has_team:
    st.info("You're not on a team yet — create or join one to upload data and run analyses.")

with st.container(key="home_hero"):
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        html('<div class="h-badge"><span class="tag">AGENTIC</span>Your autonomous data analyst</div>')
        html('<div class="h-hero">Ask your data anything.<br><span>Trust every answer.</span></div>')
        html('<div class="h-lede">An AI agent plans the analysis, writes and runs the code in a sandbox, '
             'and a second, <em>independent model checks the result</em> before it reaches you.</div>')
        with st.container(key="home_ctas"):
            b1, b2, _ = st.columns([1.15, 1, 0.5])
            with b1:
                if _has_team:
                    if st.button("New analysis", type="primary", key="btn_ask", use_container_width=True):
                        st.switch_page("pages/3_Analyses.py")
                elif st.button("Go to Workspaces", type="primary", key="btn_ws", use_container_width=True):
                    st.switch_page("pages/8_Workspaces.py")
            with b2:
                if st.button("Upload data", key="btn_upload", use_container_width=True):
                    st.switch_page("pages/2_Datasets.py")
        html('<div class="h-proof"><span>Runs in a secure sandbox</span><span>Every number verified</span>'
             '<span>Fully auditable</span></div>')
    with right:
        suggestions = "".join(f"<div>{esc(q)}</div>" for q in EXAMPLE_QUESTIONS[1:])
        html(
            f"""
            <div class="h-stage">
              <div class="h-orb"></div>
              <div class="h-float h-ask">
                <div class="h-ask-head"><div class="h-ask-ico">{I_SPARK}</div>
                  <div class="h-ask-q"><small>Ask Silt</small>{esc(EXAMPLE_QUESTIONS[0])}</div></div>
                <div class="h-ask-list">{suggestions}</div>
              </div>
              <div class="h-float h-metric">
                <div class="lbl">Analyses verified</div>
                <div class="num">{len(_verified)}</div>
                <div class="sub"><b>of {len(_questions)}</b> run in this workspace</div>
              </div>
              <div class="h-stage-title">Every number,<br>checked twice.</div>
            </div>
            """
        )

# ------------------------------------------------------ capability strip --
html(
    '<div class="h-strip-eyebrow">Built for answers you can defend</div>'
    '<div class="h-strip">'
    f'<div>{I_FILE}CSV upload</div><div>{I_BOX}Sandboxed code</div><div>{I_SHIELD}Independent critic</div>'
    f'<div>{I_LOG}Audit trail</div><div>{I_DASH}Dashboards</div></div>'
)

# ------------------------------------------------- how it works + snapshot --
how_l, snap_r = st.columns([1.75, 1.0], gap="large")
with how_l:
    html('<div class="h-how-eyebrow">How it works</div>')
    html('<div class="h-how-title">From raw data to real insights.</div>')
    html(f'<div class="h-steps">{_steps()}</div>')

with snap_r:
    if _questions:
        rows = ""
        for item in _questions[:3]:
            tag = (verdict_badge(item["verdict_state"]) if item.get("verdict_state") else
                   f'<span class="ds-badge ds-badge-neutral">{esc(item["status"].replace("_", " ").title())}</span>')
            rows += f'<div class="h-snap-row"><div class="t">{esc(item["text"][:46])}</div>{tag}</div>'
        foot = rows
    else:
        foot = (f'<div class="h-snap-foot"><div class="h-snap-ico">{I_SPARK}</div>'
                '<div>No analyses yet — upload a CSV and ask your first question.</div></div>')
    html(
        f"""
        <div class="h-snap">
          <div class="h-snap-head"><span>Workspace Snapshot</span>
            <span class="h-snap-live"><span class="silt-dot silt-dot-on"></span>Live</span></div>
          <div class="h-snap-tiles">
            <div class="h-tile"><div class="h-tile-num">{len(_datasets)}</div><div class="h-tile-lbl">{I_DB}Datasets</div></div>
            <div class="h-tile"><div class="h-tile-num">{len(_questions)}</div><div class="h-tile-lbl">{_icon('<path d="M5 20V11M12 20V4M19 20v-6"/>', 16)}Analyses</div></div>
            <div class="h-tile"><div class="h-tile-num">{len(_verified)}</div><div class="h-tile-lbl">{_icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6z"/>', 16)}Verified</div></div>
          </div>
          {foot}
        </div>
        """
    )

html('<div class="h-footer"><div>Silt <b>—</b> Autonomous Data Analyst</div>'
     '<div>Verified &nbsp;/&nbsp; Auditable &nbsp;/&nbsp; Actionable</div></div>')
