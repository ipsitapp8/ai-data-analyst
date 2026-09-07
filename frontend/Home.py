"""Landing page — a real snapshot of the workspace, not a marketing mockup.

Everything below the fold is live: dataset/analysis counts and the recent-
activity list come straight from the backend, same as the Overview page.
No sidebar here; the two CTAs and the nav links are the only way in.
"""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, health, list_datasets, list_questions
from style.theme import html, page_setup

page_setup("Home", sidebar=False)

st.markdown(
    """
<style>
:root {
  --hbg: #161616;
  --hcard: #161616;
  --hborder: rgba(255,255,255,0.12);
  --htext: #e6e6e6;
  --hmuted: #969696;
  --hfaint: #666666;
  --haccent: #4f8fe0;
  --hgood: #8a8a8a;
}

.stApp { background: var(--hbg) !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
.block-container { padding: 28px 44px 40px 44px !important; max-width: 1200px !important; }

/* ---------------- top bar ---------------- */
.h-brand { display: flex; align-items: center; gap: 11px; }
.h-brand-name { font-size: 1.15rem; font-weight: 600; color: var(--htext); line-height: 1; }
.h-brand-tag { font-size: 0.74rem; color: var(--hfaint); margin-top: 3px; }

.h-status {
  display: flex; align-items: center; gap: 8px; justify-content: flex-end;
  font-size: 0.78rem; color: var(--hmuted); height: 40px;
}
.h-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.h-dot-on  { background: var(--hgood); }
.h-dot-off { background: #b05a52; }

.st-key-nav_upload button, .st-key-nav_ask button {
  font-size: 0.85rem !important; padding: 0.5em 1.1em !important; white-space: nowrap !important;
}

/* ---------------- hero ---------------- */
.h-head {
  font-size: 2rem; font-weight: 600; line-height: 1.25;
  color: var(--htext); margin: 40px 0 14px 0;
}
.h-sub { color: var(--hmuted); font-size: 0.94rem; line-height: 1.65; max-width: 480px; margin-bottom: 26px; }

.h-feats { list-style: none; margin: 0 0 8px 0; padding: 0; max-width: 480px; }
.h-feats li {
  font-size: 0.88rem; color: var(--htext); padding: 8px 0;
  border-top: 1px solid var(--hborder);
}
.h-feats li:first-child { border-top: none; }

/* ---------------- right: live snapshot ---------------- */
.h-panel { border: 1px solid var(--hborder); border-radius: 4px; margin-top: 40px; }
.h-panel-head {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 14px 18px; border-bottom: 1px solid var(--hborder);
}
.h-panel-title { font-size: 0.78rem; color: var(--hmuted); }
.h-panel-live { font-size: 0.75rem; color: var(--hgood); }

.h-stats { display: flex; }
.h-stat { flex: 1; padding: 18px; }
.h-stat + .h-stat { border-left: 1px solid var(--hborder); }
.h-stat-v { font-size: 1.6rem; color: var(--htext); font-weight: 600; }
.h-stat-l { font-size: 0.74rem; color: var(--hfaint); margin-top: 4px; }

.h-activity { border-top: 1px solid var(--hborder); }
.h-arow { display: flex; align-items: center; gap: 12px; padding: 12px 18px; border-top: 1px solid var(--hborder); }
.h-arow:first-child { border-top: none; }
.h-arow-text { font-size: 0.85rem; color: var(--htext); line-height: 1.4; flex: 1;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.h-arow-tag {
  font-size: 0.72rem; padding: 2px 7px; border-radius: 3px; white-space: nowrap;
}
.h-tag-verified { background: rgba(79,143,224,0.12); color: var(--haccent); border: 1px solid rgba(79,143,224,0.3); }
.h-tag-running  { background: rgba(138,138,138,0.12); color: var(--hgood); border: 1px solid rgba(138,138,138,0.3); }
.h-tag-other    { background: rgba(255,255,255,0.05); color: var(--hmuted); border: 1px solid var(--hborder); }

.h-empty { padding: 26px 18px; text-align: center; color: var(--hmuted); font-size: 0.85rem; line-height: 1.6; }
</style>
""",
    unsafe_allow_html=True,
)

try:
    health()
    _backend_ok = True
except ApiError:
    _backend_ok = False

_has_team = bool(st.session_state.get("active_team_id"))

try:
    _datasets = list_datasets() if _has_team else []
except ApiError:
    _datasets = []

try:
    _questions = list_questions() if _has_team else []
except ApiError:
    _questions = []

_verified = [q for q in _questions if q["status"] == "verified"]

if not _has_team:
    st.info("You're not on a team yet — create or join one to upload data and run analyses.")
    if st.button("Go to Workspaces  →", type="primary", key="home_go_workspaces"):
        st.switch_page("pages/8_Workspaces.py")

# ---------------------------------------------------------------- top bar --
top_l, top_r = st.columns([2, 1.3])
with top_l:
    html('<div class="h-brand"><div><div class="h-brand-name">Silt</div>'
         '<div class="h-brand-tag">Autonomous data analyst</div></div></div>')
with top_r:
    dot = "h-dot-on" if _backend_ok else "h-dot-off"
    status = "backend online" if _backend_ok else "backend unreachable"
    st_l, up_c, ask_c = st.columns([1.3, 1, 1])
    with st_l:
        html(f'<div class="h-status"><span class="h-dot {dot}"></span>{status}</div>')
    with up_c:
        with st.container(key="nav_upload"):
            if st.button("Upload data", key="btn_upload"):
                st.switch_page("pages/2_Datasets.py")
    with ask_c:
        with st.container(key="nav_ask"):
            if st.button("New analysis", key="btn_ask", type="primary"):
                st.switch_page("pages/3_Analyses.py")

# ------------------------------------------------------------- hero body --
left, right = st.columns([1.1, 1.0], gap="large")

with left:
    html(
        """
        <div class="h-head">Ask a question about a CSV in plain English.</div>
        <div class="h-sub">
          An agent plans the analysis, writes and runs the code in a sandbox, and a
          second, independent model checks the result before it reaches a chart.
        </div>
        <ul class="h-feats">
          <li>Plans the analysis before touching data</li>
          <li>Runs code in an isolated sandbox</li>
          <li>A second model verifies every number</li>
        </ul>
        """
    )

with right:
    stats_html = f"""
    <div class="h-stats">
      <div class="h-stat"><div class="h-stat-v">{len(_datasets)}</div><div class="h-stat-l">Datasets</div></div>
      <div class="h-stat"><div class="h-stat-v">{len(_questions)}</div><div class="h-stat-l">Analyses</div></div>
      <div class="h-stat"><div class="h-stat-v">{len(_verified)}</div><div class="h-stat-l">Verified</div></div>
    </div>
    """

    if _questions:
        rows = ""
        for q in _questions[:5]:
            if q["status"] == "verified":
                tag_cls, tag_txt = "h-tag-verified", "Verified"
            elif q["status"] in ("planning", "executing", "critic", "dashboard", "queued"):
                tag_cls, tag_txt = "h-tag-running", "Running"
            else:
                tag_cls, tag_txt = "h-tag-other", q["status"].replace("_", " ")
            rows += (
                f'<div class="h-arow"><div class="h-arow-text">{q["text"][:52]}</div>'
                f'<span class="h-arow-tag {tag_cls}">{tag_txt}</span></div>'
            )
        activity_html = f'<div class="h-activity">{rows}</div>'
    else:
        activity_html = (
            '<div class="h-empty">No analyses yet — upload a CSV and ask your first question.</div>'
        )

    html(
        f"""
        <div class="h-panel">
          <div class="h-panel-head">
            <span class="h-panel-title">Workspace snapshot</span>
            <span class="h-panel-live">live</span>
          </div>
          {stats_html}
          {activity_html}
        </div>
        """
    )
