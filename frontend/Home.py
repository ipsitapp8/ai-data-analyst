"""Landing page — hero, how it works, and a live workspace snapshot.

The dataset/analysis counts and recent activity come straight from the
backend, same as the Overview page. Drop a picture at frontend/assets/hero.jpg
to replace the drawn hero artwork.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from api_client import ApiError, list_datasets, list_questions
from style.theme import asset_data_uri, esc, html, page_setup, render_sidebar, verdict_badge

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

# Drawn stand-in for a hero photo: a tilted dashboard tablet, a CSV tile, and warm light arcs.
HERO_SVG = """
<svg viewBox="0 0 640 380" xmlns="http://www.w3.org/2000/svg">
  <rect x="40" y="40" width="470" height="290" rx="14" fill="#ffffff" stroke="#e4e4e7"/>
  <text x="68" y="76" fill="#18181b" font-size="14" font-weight="600" font-family="Inter, sans-serif">Sales overview</text>
  <rect x="68" y="96" width="130" height="64" rx="8" fill="#fafafa" stroke="#e4e4e7"/>
  <rect x="212" y="96" width="130" height="64" rx="8" fill="#fafafa" stroke="#e4e4e7"/>
  <rect x="356" y="96" width="130" height="64" rx="8" fill="#eef2ff" stroke="#c7d2fe"/>
  <polyline points="68,290 130,262 190,272 250,232 310,242 370,200 430,190 486,170" fill="none" stroke="#4f46e5" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="68,300 130,290 190,294 250,274 310,282 370,262 430,256 486,248" fill="none" stroke="#0ea5a4" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  <g transform="translate(470 250)">
    <rect width="130" height="96" rx="12" fill="#ffffff" stroke="#e4e4e7"/>
    <circle cx="28" cy="30" r="12" fill="#dcfce7"/><path d="M22 30l4 4 8-8" fill="none" stroke="#16a34a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="48" y="35" fill="#18181b" font-size="12" font-weight="600" font-family="Inter, sans-serif">Verified</text>
    <rect x="18" y="58" width="94" height="6" rx="3" fill="#f0f0f2"/><rect x="18" y="72" width="64" height="6" rx="3" fill="#f0f0f2"/>
  </g>
</svg>
"""


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

hero_img = Path(__file__).parent / "assets" / "hero.jpg"
with st.container(key="home_hero"):
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        html('<div class="h-eyebrow">Your data. <b>Our analysis.</b></div>')
        html('<div class="h-hero">Ask a question about a CSV in plain English.</div>')
        html('<div class="h-lede">An AI agent plans the analysis, writes and runs the code in a sandbox, '
             'and a second, <em>independent model checks the result</em> before it reaches you.</div>')
        with st.container(key="home_ctas"):
            b1, b2, _ = st.columns([1.25, 1, 0.6])
            with b1:
                if _has_team:
                    if st.button("New analysis  →", type="primary", key="btn_ask", use_container_width=True):
                        st.switch_page("pages/3_Analyses.py")
                elif st.button("Go to Workspaces  →", type="primary", key="btn_ws", use_container_width=True):
                    st.switch_page("pages/8_Workspaces.py")
            with b2:
                if st.button("Upload data", key="btn_upload", use_container_width=True):
                    st.switch_page("pages/2_Datasets.py")
    with right:
        if hero_img.exists():
            html(f'<div class="h-art photo" style="background-image:url({asset_data_uri("hero.jpg")})"></div>')
        else:
            html(f'<div class="h-art">{HERO_SVG}</div>')

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
