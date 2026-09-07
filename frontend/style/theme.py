"""SILT design system: global CSS + shared UI components.

Flat, plain, and quiet on purpose: system fonts, one muted accent color, no
gradients or glow. Every page imports from here so the app reads as one
product.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from api_client import ApiError, health, inspect_element, my_workspaces
from auth import current_user, logout, require_login, require_password

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Chart data gets real, distinct color even though the UI chrome around it
# stays flat and neutral -- that split is normal (Excel, Grafana, Tableau all
# do it): plain chrome, legible/vivid data encoding.
CHART_COLORWAY = ["#4f8fe0", "#3fb87a", "#e0a83e", "#d1596b", "#9575cd", "#41b8c4"]
ACCENT = "#4f8fe0"      # clear blue -- the one interactive/positive UI color
ACCENT_2 = "#8a8a8a"    # plain gray -- in-progress/secondary state

BASE_CSS = """
<style>
:root {
  --bg-base: #161616;
  --bg-page: #161616;
  --bg-card: #161616;
  --bg-card-hover: #1c1c1c;
  --bg-inset: #101010;
  --border: rgba(255,255,255,0.12);
  --border-strong: rgba(255,255,255,0.22);

  --text-primary: #e6e6e6;
  --text-secondary: #969696;
  --text-muted: #666666;

  --accent: #4f8fe0;
  --accent-dim: #3f72b3;
  --accent-bg: rgba(79,143,224,0.12);
  --accent-border: rgba(79,143,224,0.32);

  --accent2: #8a8a8a;
  --accent2-dim: #6e6e6e;
  --accent2-bg: rgba(138,138,138,0.12);
  --accent2-border: rgba(138,138,138,0.3);

  --warn: #b6944a;
  --warn-bg: rgba(182,148,74,0.12);
  --warn-border: rgba(182,148,74,0.3);

  --error: #b05a52;
  --error-bg: rgba(176,90,82,0.12);
  --error-border: rgba(176,90,82,0.32);

  --radius-lg: 4px;
  --radius-md: 4px;
  --radius-sm: 3px;

  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-serif: var(--font);
  --font-mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg-page);
  color: var(--text-primary);
  font-family: var(--font);
}

[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
/* the sidebar's re-expand button lives inside stToolbar and would otherwise
   inherit that visibility:hidden -- restore it explicitly, or a collapsed
   sidebar becomes permanently stuck with no way back */
[data-testid="stExpandSidebarButton"] { visibility: visible !important; }
/* native form controls (radio/checkbox/slider) otherwise render in
   Streamlit's stock red regardless of the rest of the theme */
:root { accent-color: var(--accent); }
[data-testid="stSliderThumbValue"], [data-baseweb="radio"] div:first-child {
  border-color: var(--text-secondary) !important;
}
[role="radio"][aria-checked="true"] div:first-child,
[data-baseweb="radio"] input:checked + div {
  border-color: var(--accent) !important;
  background: var(--accent) !important;
}
[data-testid="stSidebarNav"] { display: none; }

.block-container {
  padding-top: 2.6rem;
  padding-bottom: 4rem;
  max-width: 1360px;
}

h1,h2,h3,h4,h5 { font-family: var(--font-serif); color: var(--text-primary); font-weight: 600; letter-spacing: -0.005em; }
p, span, div, label, li { font-family: var(--font); }

::selection { background: var(--accent-bg); color: var(--text-primary); }

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
  background: var(--bg-base);
  border-right: 1px solid var(--border);
  width: 250px !important;
}
[data-testid="stSidebar"] > div { padding-top: 0; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.1rem; }

.silt-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 26px 20px 22px 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 6px;
}
.silt-brand-name {
  font-family: var(--font-serif); font-size: 1.12rem; font-weight: 600;
  color: var(--text-primary); line-height: 1; letter-spacing: 0.01em;
}
.silt-brand-sub { font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.06em;
                  text-transform: uppercase; margin-top: 3px; }

/* sidebar nav rows -- plain list, no icon tiles */
.st-key-nav { padding: 12px 10px 0 10px; }
.st-key-nav .stButton > button {
  width: 100%;
  display: flex !important;
  justify-content: flex-start !important;
  align-items: baseline;
  gap: 12px;
  background: transparent !important;
  border: none !important;
  border-left: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--text-secondary) !important;
  font-family: var(--font) !important;
  font-weight: 400 !important;
  font-size: 0.87rem !important;
  padding: 9px 10px 9px 12px !important;
  margin: 0 !important;
  box-shadow: none !important;
  transition: border-color 0.1s ease, color 0.1s ease, background 0.1s ease;
}
.st-key-nav .stButton > button:hover {
  background: var(--bg-card-hover) !important;
  color: var(--text-primary) !important;
  border-left-color: var(--border-strong) !important;
}
.st-key-nav .stButton > button:disabled {
  background: var(--bg-card-hover) !important;
  border-left: 2px solid var(--accent) !important;
  color: var(--text-primary) !important;
  font-weight: 600 !important;
  opacity: 1 !important;
  cursor: default !important;
}

.silt-status {
  display: flex; align-items: center; gap: 8px;
  padding: 12px 20px; margin-top: 14px;
  border-top: 1px solid var(--border);
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);
}
.silt-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.silt-dot-on  { background: var(--accent2); }
.silt-dot-off { background: var(--error); }

/* ============ BUTTONS ============ */
button[kind="primary"], .stFormSubmitButton > button {
  background: var(--accent) !important;
  color: #101010 !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important;
  font-size: 0.87rem !important;
  padding: 0.55em 1.2em !important;
  box-shadow: none !important;
}
button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  background: var(--accent-dim) !important;
  color: #101010 !important;
}
button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-md) !important;
  font-weight: 500 !important;
  font-size: 0.87rem !important;
  padding: 0.53em 1.15em !important;
  box-shadow: none !important;
}
button[kind="secondary"]:hover {
  border-color: var(--accent) !important;
  background: var(--bg-card-hover) !important;
}

/* ============ CARDS ============ */
.ds-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
}
[class*="st-key-card"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 18px 20px !important;
}
[class*="st-key-flat"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 0 !important;
  overflow: hidden;
}

/* inspectable KPI cards -- a real st.button styled to look like .ds-card,
   so the whole card is clickable (see pages/4_Dashboards.py) */
[class*="st-key-kpi_"] .stButton > button {
  width: 100% !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 18px 20px !important;
  text-align: left !important;
  white-space: pre-line !important;
  box-shadow: none !important;
  position: relative;
}
[class*="st-key-kpi_"] .stButton > button p {
  font-family: var(--font-mono) !important; color: var(--text-secondary) !important;
  font-size: 0.74rem !important; letter-spacing: 0.02em; margin: 0 0 10px 0 !important;
}
[class*="st-key-kpi_"] .stButton > button strong {
  font-family: var(--font-serif) !important; color: var(--text-primary) !important;
  font-size: 1.85rem !important; font-weight: 600 !important;
}
[class*="st-key-kpi_"] .stButton > button:hover {
  border-color: var(--accent) !important;
  background: var(--bg-card-hover) !important;
}
[class*="st-key-kpi_"] .stButton > button::after {
  content: "🔍"; position: absolute; top: 14px; right: 16px;
  font-size: 0.8rem; opacity: 0; transition: opacity 0.12s ease;
}
[class*="st-key-kpi_"] .stButton > button:hover::after { opacity: 0.6; }

/* inspectable charts -- the hover hint lives on the wrapping container (keyed
   st.container(key=f"chartcard_{element_id}")) since the chart itself is a
   plotly iframe/canvas we can't style into directly */
[class*="st-key-chartcard_"] { position: relative; }
[class*="st-key-chartcard_"]::after {
  content: "🔍 click a point to inspect"; position: absolute; top: 16px; right: 18px;
  font-size: 0.7rem; color: var(--text-muted); opacity: 0; transition: opacity 0.12s ease;
  pointer-events: none;
}
[class*="st-key-chartcard_"]:hover::after { opacity: 0.8; }

.ds-page-title { font-family: var(--font-serif); font-size: 1.5rem; font-weight: 600;
                 margin: 0 0 4px 0; }
.ds-page-sub   { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 22px; }
.ds-section-title { font-family: var(--font-serif); font-size: 0.98rem; font-weight: 600; margin: 0; }

/* stat cards */
.ds-stat-label { font-family: var(--font-mono); color: var(--text-secondary);
                 font-size: 0.74rem; letter-spacing: 0.02em; margin-bottom: 10px; }
.ds-stat-value { font-family: var(--font-serif); font-size: 1.85rem; font-weight: 600;
                 line-height: 1; }
.ds-stat-delta { font-family: var(--font-mono); font-size: 0.78rem; margin-top: 11px;
                 display: flex; align-items: center; gap: 4px; }
.ds-up   { color: var(--accent2); }
.ds-down { color: var(--error); }

/* list rows */
.ds-row {
  display: flex; align-items: center; gap: 14px;
  padding: 15px 22px; border-top: 1px solid var(--border);
}
.ds-row:hover { background: var(--bg-card-hover); }
.ds-row-icon {
  width: 32px; height: 32px; border-radius: var(--radius-sm); flex-shrink: 0;
  background: var(--bg-inset); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  color: var(--text-secondary);
}
.ds-row-title { font-size: 0.93rem; font-weight: 500; color: var(--text-primary); line-height: 1.35; }
.ds-row-meta  { font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-secondary); line-height: 1.4; }
.ds-row-spacer { flex: 1; }

.ds-card-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 22px;
  border-bottom: 1px solid var(--border);
}

/* badges -- flat tags, not pills */
.ds-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 8px; border-radius: var(--radius-sm);
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 500;
  white-space: nowrap;
}
.ds-badge-verified { background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent-border); }
.ds-badge-warn     { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-border); }
.ds-badge-error    { background: var(--error-bg); color: var(--error); border: 1px solid var(--error-border); }
.ds-badge-neutral  { background: var(--bg-inset); color: var(--text-secondary); border: 1px solid var(--border); }
.ds-badge-running  { background: var(--accent2-bg); color: var(--accent2); border: 1px solid var(--accent2-border); }
.ds-badge-running::before {
  content:''; width:6px; height:6px; border-radius:50%; background: var(--accent2);
  animation: siltpulse 1.4s infinite ease-in-out;
}
@keyframes siltpulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ============ INPUTS ============ */
[data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"],
[data-testid="stNumberInputRootElement"],
.stTextInput input, .stTextArea textarea,
.react-aria-ComboBox > div,
[data-testid="stFileUploaderDropzone"] {
  background: var(--bg-inset) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  box-shadow: none !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stFileUploaderDropzone"] { border-style: dashed !important; }

/* tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 4px; background: transparent; border-bottom: 1px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent; color: var(--text-secondary);
  font-family: var(--font-mono); font-size: 0.85rem; font-weight: 500; padding: 10px 14px;
}
[data-testid="stTabs"] [aria-selected="true"] { color: var(--text-primary) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px !important; }

/* alerts */
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stAlertContainer"] * { color: var(--text-primary) !important; fill: var(--text-primary) !important; }

/* expander / popover */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary { color: var(--text-primary) !important; }
[data-testid="stPopoverBody"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
}

/* tables */
[data-testid="stTable"] table { background: transparent !important; border-collapse: collapse; width: 100%; }
[data-testid="stTable"] th {
  background: transparent !important; color: var(--text-secondary) !important;
  font-family: var(--font) !important; font-size: 0.78rem; font-weight: 500;
  border-bottom: 1px solid var(--border) !important; padding: 12px 16px !important; text-align: left;
}
[data-testid="stTable"] td {
  background: transparent !important; color: var(--text-primary) !important;
  border-bottom: 1px solid var(--border) !important; padding: 13px 16px !important; font-size: 0.87rem;
}
[data-testid="stTable"] tr:last-child td { border-bottom: none !important; }

/* code */
[data-testid="stCode"], pre {
  background: var(--bg-inset) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
code, pre, [data-testid="stCode"] * { font-family: var(--font-mono) !important; font-size: 0.83rem !important; }

hr { border-color: var(--border) !important; }
[data-testid="stMetric"] {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 18px 20px;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-family: var(--font-mono) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: var(--font-serif) !important; }

/* quality bar */
.ds-quality { display: flex; align-items: center; gap: 10px; }
.ds-quality-track {
  display: inline-block; width: 92px; height: 3px; border-radius: 0;
  background: var(--bg-inset); overflow: hidden;
}
.ds-quality-fill { display: block; height: 100%; background: var(--accent); }

/* progress bar */
.ds-progress-track { width: 100%; height: 3px; border-radius: 0; background: var(--bg-inset); overflow: hidden; }
.ds-progress-fill  { height: 100%; transition: width .3s ease; background: var(--accent); }

/* step list */
.ds-step { display: flex; gap: 13px; padding: 13px 16px; border-radius: var(--radius-sm); align-items: flex-start; }
.ds-step-active { background: var(--accent2-bg); border: 1px solid var(--accent2-border); }
.ds-step-num {
  width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
  background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-secondary);
}
.ds-step-done   { background: var(--accent-bg); border-color: var(--accent-border); color: var(--accent); }
.ds-step-run    { background: var(--accent2-bg); border-color: var(--accent2-border); color: var(--accent2); }
.ds-step-title  { font-size: 0.89rem; font-weight: 500; color: var(--text-primary); line-height: 1.35; }
.ds-step-status { font-family: var(--font-mono); font-size: 0.74rem; color: var(--text-secondary); line-height: 1.4; }

/* verification checklist */
.ds-check { display: flex; align-items: center; justify-content: space-between; padding: 11px 0; border-bottom: 1px solid var(--border); }
.ds-check:last-child { border-bottom: none; }
.ds-check-label { font-size: 0.87rem; color: var(--text-primary); }
</style>
"""


def html(markup: str) -> None:
    """Render raw HTML.

    st.markdown parses its input as markdown first, so any line indented by 4+
    spaces becomes a <pre> code block and the markup shows up as literal text.
    Stripping per-line leading whitespace lets us keep readable indentation in
    the source without it leaking into the page.
    """
    # join with a space, not "": HTML collapses inter-tag whitespace, but
    # concatenating bare would fuse words split across source lines.
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_base_css() -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def page_setup(title: str, sidebar: bool = True) -> None:
    st.set_page_config(
        page_title=f"{title} · SILT",
        page_icon="▤",
        layout="wide",
        initial_sidebar_state="expanded" if sidebar else "collapsed",
    )
    # Two gates, outer to inner. Neither changes local dev: require_password()
    # is a no-op unless APP_PASSWORD is set, and require_login() always applies
    # (there are no anonymous accounts) but is fast once a session exists.
    require_password()
    require_login()
    _ensure_active_team()
    inject_base_css()
    if not sidebar:
        st.markdown(
            "<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
            "[data-testid='stExpandSidebarButton']{display:none !important;}"
            ".block-container{max-width:100% !important;padding:0 !important;}</style>",
            unsafe_allow_html=True,
        )


def _ensure_active_team() -> None:
    """Default to the user's first team so existing pages (which call
    list_datasets()/list_questions()/etc with no team argument) have a valid
    X-Team-Id as soon as they're logged in. The sidebar switcher can then
    change it. Leaves both unset if the user belongs to no team yet -- the
    Workspaces page is where they create or join one.
    """
    if st.session_state.get("active_team_id"):
        return
    try:
        workspaces = my_workspaces()["communities"]
    except ApiError:
        return
    for community in workspaces:
        if community["teams"]:
            st.session_state["active_community_id"] = community["id"]
            st.session_state["active_team_id"] = community["teams"][0]["id"]
            return


@st.cache_data
def asset_data_uri(filename: str, mime: str = "image/jpeg") -> str:
    """Base64 data URI for a file in frontend/assets (CSS can't reach local paths)."""
    data = (ASSETS_DIR / filename).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


NAV_PAGES = [
    ("home", "Home", "Home.py"),
    ("overview", "Overview", "pages/1_Overview.py"),
    ("datasets", "Datasets", "pages/2_Datasets.py"),
    ("analyses", "Analyses", "pages/3_Analyses.py"),
    ("dashboards", "Dashboards", "pages/4_Dashboards.py"),
    ("reports", "Reports", "pages/5_Reports.py"),
    ("audit", "Audit trail", "pages/6_Audit_Trail.py"),
    ("workspaces", "Workspaces", "pages/8_Workspaces.py"),
    ("team_overview", "Team overview", "pages/9_Team_Overview.py"),
    ("settings", "Settings", "pages/7_Settings.py"),
]


@st.cache_data(ttl=5)
def _backend_alive() -> bool:
    try:
        health()
        return True
    except ApiError:
        return False
    except Exception:  # noqa: BLE001 - sidebar status must never crash the page
        return False


def _render_workspace_switcher() -> None:
    try:
        communities = my_workspaces()["communities"]
    except ApiError:
        html('<div class="silt-status">workspaces unavailable</div>')
        return

    if not communities:
        html(
            '<div class="silt-status" style="border-top:none;">'
            "No team yet — open Workspaces to create or join one."
            "</div>"
        )
        return

    with st.container(key="ws_switcher"):
        c_ids = [c["id"] for c in communities]
        c_labels = {c["id"]: c["name"] for c in communities}
        active_c = st.session_state.get("active_community_id", c_ids[0])
        if active_c not in c_ids:
            active_c = c_ids[0]
        c_idx = st.selectbox(
            "Community", range(len(c_ids)), index=c_ids.index(active_c),
            format_func=lambda i: c_labels[c_ids[i]], key="ws_community_select",
            label_visibility="collapsed",
        )
        chosen_community = communities[c_idx]
        st.session_state["active_community_id"] = chosen_community["id"]

        teams = chosen_community["teams"]
        if not teams:
            html('<div class="silt-status" style="border-top:none;">No teams in this community yet.</div>')
            return

        t_ids = [t["id"] for t in teams]
        t_labels = {t["id"]: t["name"] for t in teams}
        active_t = st.session_state.get("active_team_id")
        if active_t not in t_ids:
            active_t = t_ids[0]
        t_idx = st.selectbox(
            "Team", range(len(t_ids)), index=t_ids.index(active_t),
            format_func=lambda i: t_labels[t_ids[i]], key="ws_team_select",
            label_visibility="collapsed",
        )
        st.session_state["active_team_id"] = teams[t_idx]["id"]


def render_sidebar(current: str) -> None:
    """Brand mark, workspace switcher, numbered nav rail, and a quiet live
    system-status line."""
    with st.sidebar:
        html(
            """
            <div class="silt-brand">
              <div class="silt-brand-name">Silt</div>
              <div class="silt-brand-sub">Data Analyst</div>
            </div>
            """
        )
        _render_workspace_switcher()
        with st.container(key="nav"):
            for i, (key, label, target) in enumerate(NAV_PAGES, start=1):
                if st.button(f"{i:02d}  {label}", key=f"nav_{key}", disabled=(key == current)):
                    st.switch_page(target)

        alive = _backend_alive()
        dot_cls = "silt-dot-on" if alive else "silt-dot-off"
        status = "backend online" if alive else "backend unreachable"
        html(
            f"""
            <div class="silt-status">
              <span class="silt-dot {dot_cls}"></span>{status}
            </div>
            """
        )

        user = current_user()
        if user:
            u_l, u_r = st.columns([3, 1])
            with u_l:
                html(f'<div class="silt-status" style="border-top:none;">{user["display_name"]}</div>')
            with u_r:
                if st.button("⏻", key="nav_logout", help="Log out"):
                    logout()
                    st.rerun()


def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="ds-page-sub">{subtitle}</div>' if subtitle else ""
    html(f'<div class="ds-page-title">{title}</div>{sub}')


def badge(text: str, kind: str = "neutral") -> str:
    check = (
        '<svg width="11" height="11" viewBox="0 0 24 24" fill="none">'
        '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/>'
        '<path d="M8.5 12.2l2.4 2.4 4.6-4.9" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        if kind == "verified" else ""
    )
    return f'<span class="ds-badge ds-badge-{kind}">{check}{text}</span>'


def stat_card(label: str, value: str, delta: str = "", direction: str = "up") -> str:
    arrow = "↑" if direction == "up" else "↓"
    cls = "ds-up" if direction == "up" else "ds-down"
    d = f'<div class="ds-stat-delta {cls}">{arrow} {delta}</div>' if delta else ""
    return f"""
    <div class="ds-card">
      <div class="ds-stat-label">{label}</div>
      <div class="ds-stat-value">{value}</div>
      {d}
    </div>
    """


def figure_from_json(payload: dict) -> go.Figure:
    """Rebuild a sandbox-produced figure.

    The sandbox image pins plotly 5.x while the frontend runs plotly 6.x, and
    the default template 5.x embeds includes trace types 6.x dropped (e.g.
    `heatmapgl`), so a plain from_json raises "Invalid property". We restyle
    every chart ourselves anyway, so the embedded template is dropped before
    parsing rather than being version-matched.
    """
    data = copy.deepcopy(payload)
    layout = data.get("layout")
    if isinstance(layout, dict):
        layout.pop("template", None)
    return pio.from_json(json.dumps(data))


def style_chart(fig: go.Figure, height: int = 300, showlegend: bool = False,
                 ensure_markers: bool = False) -> go.Figure:
    """Force any figure -- including sandbox-generated ones -- into the SILT look.

    ensure_markers: sandbox line charts default to mode="lines" (px.line's
    default), which has no clickable points for Plotly's on_select -- a click
    anywhere on the bare line doesn't register as a point selection. Pass True
    (click-to-inspect charts do) to add markers to any bare-line scatter trace
    so there's something to actually click.
    """
    if ensure_markers:
        for trace in fig.data:
            if getattr(trace, "type", None) == "scatter" and trace.mode and "markers" not in trace.mode:
                trace.mode = trace.mode + "+markers"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#969696", family="system-ui, sans-serif", size=12),
        colorway=CHART_COLORWAY,
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=showlegend,
        legend=dict(font=dict(color="#969696"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1c1c1c", bordercolor="rgba(255,255,255,0.22)",
                        font=dict(color="#e6e6e6", family="system-ui, sans-serif")),
        # empty string, not None -- None leaves the title node in place and
        # Plotly renders a literal "undefined" tspan above the plot
        title=dict(text=""),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)",
                     color="#666666", showline=False, ticks="")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)",
                     color="#666666", showline=False, ticks="")
    _recolor_traces(fig)
    return fig


def _recolor_traces(fig: go.Figure) -> None:
    """Repaint solid trace colors from CHART_COLORWAY.

    `colorway` only supplies colors Plotly would otherwise auto-assign. Plotly
    Express bakes explicit per-trace colors whenever the sandbox code groups by
    a column, which would otherwise leave stock blue/red charts sitting inside
    the SILT palette. Array-valued colors (continuous scales) are left
    alone -- overwriting those would destroy the encoding.
    """
    for i, trace in enumerate(fig.data):
        color = CHART_COLORWAY[i % len(CHART_COLORWAY)]
        marker = getattr(trace, "marker", None)
        if marker is not None and not isinstance(getattr(marker, "color", None), (list, tuple)):
            try:
                trace.marker.color = color
            except (ValueError, AttributeError):
                pass
        line = getattr(trace, "line", None)
        if line is not None and not isinstance(getattr(line, "color", None), (list, tuple)):
            try:
                trace.line.color = color
            except (ValueError, AttributeError):
                pass


def plot(fig: go.Figure, height: int = 300, showlegend: bool = False,
         on_select_key: str | None = None):
    """Render a styled chart. Pass on_select_key to make it clickable -- the
    click/select event is then returned (and also lands in
    st.session_state[on_select_key]) instead of nothing."""
    kwargs = {}
    if on_select_key:
        # selection_mode="points" only (not the default points+box+lasso):
        # a plain click on a marker reliably registers as a point selection
        # this way, instead of needing an actual box/lasso drag.
        kwargs = {"on_select": "rerun", "key": on_select_key, "selection_mode": "points"}
    styled = style_chart(fig, height, showlegend, ensure_markers=bool(on_select_key))
    return st.plotly_chart(styled, use_container_width=True,
                           config={"displayModeBar": False}, **kwargs)


# ---------------------------------------------------------- click-to-inspect --
# The "Code" / "Formula" / "Data Used" panel for one dashboard element
# (backend/app/routers/inspect.py). Shared here so any page can open it the
# same way; today only pages/4_Dashboards.py does.

def open_inspect(dashboard_id: int, element_id: str) -> None:
    """Call this from a click handler (e.g. inside `if st.button(...):`)."""
    st.session_state["_inspect_open"] = True
    st.session_state["_inspect_dashboard_id"] = dashboard_id
    st.session_state["_inspect_element_id"] = element_id


def render_inspect_dialog_if_open() -> None:
    """Call once, near the end of a page's script. No-op unless open_inspect()
    was called earlier in this run (or a prior rerun still marks it open)."""
    if st.session_state.get("_inspect_open"):
        _inspect_dialog()


@st.dialog("Inspect", width="large")
def _inspect_dialog() -> None:
    dashboard_id = st.session_state.get("_inspect_dashboard_id")
    element_id = st.session_state.get("_inspect_element_id")

    cache = st.session_state.setdefault("inspect_cache", {})
    cache_key = f"{dashboard_id}:{element_id}"
    if cache_key not in cache:
        try:
            cache[cache_key] = inspect_element(dashboard_id, element_id)
        except ApiError as e:
            st.error(f"Could not load this element: {e}")
            if st.button("Close", key="inspect_close_err"):
                st.session_state["_inspect_open"] = False
                st.rerun()
            return

    data = cache[cache_key]

    html('<div class="ds-section-title">Code</div>')
    st.code(data.get("code") or "No code recorded for this element.", language="python")

    html('<div class="ds-section-title" style="margin-top:18px;">Formula</div>')
    html(f'<div style="font-size:0.9rem;color:var(--text-primary);margin-top:6px;line-height:1.6;">'
         f'{data.get("formula_explanation", "")}</div>')

    html('<div class="ds-section-title" style="margin-top:18px;">Data Used</div>')
    data_slice = data.get("data_slice") or {}
    columns, rows = data_slice.get("columns") or [], data_slice.get("rows") or []
    if columns and rows:
        st.dataframe(pd.DataFrame(rows, columns=columns), use_container_width=True, height=240)
    else:
        html('<div class="ds-row-meta" style="margin-top:6px;">No data slice was recorded for this element.</div>')

    html("<div style='height:10px'></div>")
    if st.button("Close", key="inspect_close"):
        st.session_state["_inspect_open"] = False
        st.rerun()
