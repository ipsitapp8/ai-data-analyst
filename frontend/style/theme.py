"""SILT design system: global CSS + shared UI components.

Light, airy, and minimal on purpose: Inter, white surfaces, hairline borders,
one indigo accent, no gradients or glow. Every page imports from here so the app reads as one
product.
"""
from __future__ import annotations

import base64
import copy
import html as _html_stdlib
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from api_client import ApiError, health, inspect_element, list_alerts, my_workspaces
from auth import current_user, logout, require_login, require_password

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Chart data gets real, distinct color even though the UI chrome around it
# stays flat and neutral -- that split is normal (Excel, Grafana, Tableau all
# do it): plain chrome, legible/vivid data encoding.
CHART_COLORWAY = ["#4f46e5", "#0ea5a4", "#f59e0b", "#e11d48", "#8b5cf6", "#64748b"]
ACCENT = "#4f46e5"      # indigo -- the one interactive/brand UI color
ACCENT_2 = "#94a3b8"    # slate -- in-progress/secondary state

BASE_CSS = """
<style>
:root {
  --bg-base: #ffffff;
  --bg-page: #fafafa;
  --bg-card: #ffffff;
  --bg-card-hover: #f4f4f5;
  --bg-inset: #f4f4f5;
  --border: #e4e4e7;
  --border-strong: #d4d4d8;

  --text-primary: #18181b;
  --text-secondary: #52525b;
  --text-muted: #a1a1aa;

  --accent: #4f46e5;
  --accent-dim: #4338ca;
  --accent-bg: #eef2ff;
  --accent-border: #c7d2fe;

  --ok: #16a34a;
  --ok-bg: #f0fdf4;
  --ok-border: #bbf7d0;
  --warn: #b45309;
  --warn-bg: #fffbeb;
  --warn-border: #fde68a;
  --error: #dc2626;
  --error-bg: #fef2f2;
  --error-border: #fecaca;

  --accent2: #64748b;
  --accent2-dim: #475569;
  --accent2-bg: #f1f5f9;
  --accent2-border: #e2e8f0;

  --radius-lg: 12px;
  --radius-md: 8px;
  --radius-sm: 6px;

  --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
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
.silt-dot-on  { background: var(--ok); }
.silt-dot-off { background: var(--error); }

/* ============ BUTTONS ============ */
button[kind="primary"], .stFormSubmitButton > button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important;
  font-size: 0.87rem !important;
  padding: 0.55em 1.2em !important;
  box-shadow: none !important;
}
button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  background: var(--accent-dim) !important;
  color: #ffffff !important;
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
  content: "🔍 click a point to inspect"; position: absolute; bottom: 8px; right: 18px;
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
.ds-up   { color: var(--ok); }
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
.ds-badge-verified { background: var(--ok-bg); color: var(--ok); border: 1px solid var(--ok-border); }
.ds-badge-warn     { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-border); }
.ds-badge-error    { background: var(--error-bg); color: var(--error); border: 1px solid var(--error-border); }
.ds-badge-neutral  { background: var(--bg-inset); color: var(--text-secondary); border: 1px solid var(--border); }
.ds-badge-running  { background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent-border); }
.ds-badge-running::before {
  content:''; width:6px; height:6px; border-radius:50%; background: var(--accent);
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
.ds-step-active { background: var(--accent-bg); border: 1px solid var(--accent-border); }
.ds-step-num {
  width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
  background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-secondary);
}
.ds-step-done   { background: var(--ok-bg); border-color: var(--ok-border); color: var(--ok); }
.ds-step-run    { background: var(--accent-bg); border-color: var(--accent-border); color: var(--accent); }
.ds-step-title  { font-size: 0.89rem; font-weight: 500; color: var(--text-primary); line-height: 1.35; }
.ds-step-status { font-family: var(--font-mono); font-size: 0.74rem; color: var(--text-secondary); line-height: 1.4; }

/* verification checklist */
.ds-check { display: flex; align-items: center; justify-content: space-between; padding: 11px 0; border-bottom: 1px solid var(--border); }
.ds-check:last-child { border-bottom: none; }
.ds-check-label { font-size: 0.87rem; color: var(--text-primary); }

/* ============ VERDICT BANNER ============ */
.ds-verdict { width: 100%; border-radius: var(--radius-sm); margin: 0 0 20px 0; border: 1px solid; }
.ds-verdict > summary, .ds-verdict > .ds-verdict-head {
  list-style: none; display: flex; align-items: center; gap: 12px;
  padding: 15px 20px; font-size: 1.02rem; font-weight: 600;
}
.ds-verdict > summary { cursor: pointer; }
.ds-verdict > summary::-webkit-details-marker { display: none; }
.ds-verdict-hint { margin-left: auto; font-size: 0.78rem; font-weight: 400; opacity: 0.85; }
.ds-verdict-body { padding: 4px 20px 18px 20px; font-size: 0.9rem; line-height: 1.65; color: var(--text-primary); }
.ds-verdict-body ul { margin: 6px 0 12px 18px; padding: 0; }
.ds-verdict-ok   { background: var(--ok-bg); color: var(--ok); border-color: var(--ok-border); }
.ds-verdict-warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-border); }
.ds-verdict-bad  { background: var(--error-bg); color: var(--error); border-color: var(--error-border); }
.ds-flag { color: var(--warn); font-weight: 700; }
.ds-flag-bad { color: var(--error); font-weight: 700; }
.ds-trend-up { color: var(--ok); } .ds-trend-down { color: var(--error); } .ds-trend-flat { color: var(--text-secondary); }
/* ============ CHART STUDIO (pages/4_Dashboards.py) ============ */
.st-key-card_studio { border-color: var(--border-strong) !important; }
.ds-studio-label { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted);
                   letter-spacing: 0.06em; text-transform: uppercase; margin: 2px 0 6px 0; }
.st-key-card_studio [data-testid="stButtonGroup"] button { font-size: 0.8rem !important; }
.st-key-card_studio [data-testid="stButtonGroup"] button[aria-checked="true"],
.st-key-card_studio [data-testid="stButtonGroup"] button[kind*="Active"] {
  border-color: var(--accent) !important; color: var(--text-primary) !important;
  background: var(--accent-bg) !important;
}
[class*="st-key-cs_edit_"] .stButton > button {
  padding: 2px 10px !important; font-size: 0.74rem !important; float: right;
}
@keyframes cs-fade  { from { opacity: 0 } to { opacity: 1 } }
@keyframes cs-grow  { from { opacity: 0; transform: scaleY(0.2) } to { opacity: 1; transform: scaleY(1) } }
@keyframes cs-float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-4px) } }
@keyframes cs-sheen { 0% { left: -60% } 55%,100% { left: 130% } }
@media (prefers-reduced-motion: reduce) {
  [class*="st-key-chartcard_"], [class*="st-key-chartcard_"] *,
  [class*="st-key-card_ch"], [class*="st-key-card_ch"] * { animation: none !important; }
}
</style>
"""


def esc(value) -> str:
    """Escape a value for safe interpolation into an html()/f-string call.

    Every page builds markup as an f-string and renders it via html(), which
    goes to st.markdown(unsafe_allow_html=True) with no sanitization of its
    own. Any user-controlled text embedded unescaped -- a question's text, a
    dataset filename, a community/team/display name, a chart title -- is a
    stored XSS: it runs as script in the browser of every other team member
    who later views that page. Wrap any such value in esc() before it goes
    into an f-string destined for html().
    """
    return _html_stdlib.escape(str(value), quote=True)


def html(markup: str) -> None:
    """Render raw HTML.

    st.markdown parses its input as markdown first, so any line indented by 4+
    spaces becomes a <pre> code block and the markup shows up as literal text.
    Stripping per-line leading whitespace lets us keep readable indentation in
    the source without it leaking into the page.

    This does NOT escape its input -- markup is meant to contain real tags.
    Escape untrusted values individually with esc() before interpolating them.
    """
    # join with a space, not "": HTML collapses inter-tag whitespace, but
    # concatenating bare would fuse words split across source lines.
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600'
    '&display=swap" rel="stylesheet">'
)

# Line icons for the sidebar nav, drawn as CSS masks so they inherit the row's colour.
_NAV_ICONS = {
    "home": '<path d="M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10"/>',
    "overview": '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/>',
    "datasets": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "analyses": '<path d="M3 3v18h18M7 15l4-4 3 3 5-6"/>',
    "scheduled": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "dashboards": '<rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/>',
    "reports": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M8.5 13h7M8.5 17h5"/>',
    "audit": '<path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5M12 8v4l3 2"/>',
    "workspaces": '<path d="M3 21h18M5 21V8l7-4 7 4v13M9 21v-6h6v6"/>',
    "team_overview": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6M16 5a3 3 0 0 1 0 6M18 14c1.8.6 3 2.4 3 4.5"/>',
    "logout": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
}


def _nav_icon_css() -> str:
    rules = []
    for key, body in _NAV_ICONS.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
               "stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>" + body.replace('"', "'") + "</svg>")
        uri = "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe="/:=' ,.-()")
        rules.append(f'.st-key-nav_{key} button {{ --icon: url("{uri}"); }}')
    return " ".join(rules)


def inject_base_css() -> None:
    skin = (Path(__file__).with_name("silt.css")).read_text(encoding="utf-8")
    # Flatten to one line: markdown ends an HTML block at the first blank line,
    # which would spill the rest of the stylesheet onto the page as text.
    skin = " ".join(line.strip() for line in skin.splitlines() if line.strip())
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    st.markdown(_FONTS_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{skin} {_nav_icon_css()}</style>", unsafe_allow_html=True)


def page_setup(title: str, sidebar: bool = True, public: bool = False) -> None:
    st.set_page_config(
        page_title=f"{title} · SILT",
        page_icon="▤",
        layout="wide",
        initial_sidebar_state="expanded" if sidebar else "collapsed",
    )
    if public:
        # Anonymous read-only pages (share links): no gates, no team, no nav.
        # The page itself may only call endpoints that need no login.
        inject_base_css()
        st.markdown(
            "<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
            "[data-testid='stExpandSidebarButton']{display:none !important;}"
            ".block-container{padding-top:2.2rem !important;max-width:980px !important;}</style>",
            unsafe_allow_html=True,
        )
        return
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


_WORKSPACES_CACHE_TTL_SECONDS = 8


def _cached_my_workspaces() -> dict:
    """Session-scoped, short-TTL cache for /api/me/workspaces.

    render_sidebar() calls this on every single page render, and the Analyses
    page's live-run fragment reruns every 2 seconds while an analysis is in
    progress -- without this, every one of those reruns re-fetched the full
    workspace list. Deliberately NOT @st.cache_data: that caches globally by
    function args, and my_workspaces() takes none, so every user would share
    one cached result -- a cross-tenant leak. session_state is per-browser-
    session already, so a manual TTL here stays correctly scoped per user.
    """
    now = time.monotonic()
    cached = st.session_state.get("_ws_cache")
    if cached and now - cached[0] < _WORKSPACES_CACHE_TTL_SECONDS:
        return cached[1]
    data = my_workspaces()
    st.session_state["_ws_cache"] = (now, data)
    return data


def invalidate_workspaces_cache() -> None:
    """Call after creating/joining a community or team so the switcher and
    _ensure_active_team() see it on the very next render, not after the TTL."""
    st.session_state.pop("_ws_cache", None)


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
        workspaces = _cached_my_workspaces()["communities"]
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
    ("scheduled", "Scheduled", "pages/10_Scheduled.py"),
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
        communities = _cached_my_workspaces()["communities"]
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


_ALERTS_CACHE_TTL_SECONDS = 8


def _unread_alert_count() -> int:
    """Unread in-app alerts for the active team, short-TTL cached per session
    (same reasoning as _cached_my_workspaces: the sidebar renders on every rerun)."""
    if not st.session_state.get("active_team_id"):
        return 0
    now = time.monotonic()
    cached = st.session_state.get("_alerts_cache")
    if cached and now - cached[0] < _ALERTS_CACHE_TTL_SECONDS and cached[2] == st.session_state["active_team_id"]:
        return cached[1]
    try:
        n = int(list_alerts(unread_only=True, limit=1).get("unread", 0))
    except Exception:  # noqa: BLE001 - a badge must never break a page
        n = 0
    st.session_state["_alerts_cache"] = (now, n, st.session_state["active_team_id"])
    return n


def invalidate_alerts_cache() -> None:
    st.session_state.pop("_alerts_cache", None)


def _active_workspace_name(user: dict | None) -> str:
    try:
        communities = _cached_my_workspaces()["communities"]
    except ApiError:
        communities = []
    active_team = st.session_state.get("active_team_id")
    for c in communities:
        if any(t["id"] == active_team for t in c["teams"]):
            return c["name"]
    return (user or {}).get("display_name", "Workspace")


def render_topbar() -> None:
    """Fixed top strip, right-aligned: live backend status and the active workspace."""
    user = current_user()
    alive = _backend_alive()
    name = _active_workspace_name(user)
    initials = "".join(w[0] for w in (user or {}).get("display_name", name).split()[:2]).upper() or "•"
    html(
        f"""
        <div class="silt-topbar">
          <div class="silt-tb-status"><span class="silt-dot {'silt-dot-on' if alive else 'silt-dot-off'}"></span>
            {'Backend online' if alive else 'Backend unreachable'}</div>
          <div class="silt-tb-user"><div class="silt-tb-avatar">{esc(initials)}</div>
            <div class="silt-tb-ws">{esc(name)}</div></div>
        </div>
        """
    )


_LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" stroke-width="2" '
    'stroke-linecap="round"><path d="M5 20L17 4M9 21l10-13M4 15l8-11"/></svg>'
)


def render_sidebar(current: str) -> None:
    """Logo, icon nav rail, workspace switcher, log-out; plus the top status strip."""
    render_topbar()
    with st.sidebar:
        html(f'<div class="silt-brand2">{_LOGO}<div class="n">Silt</div></div>')
        with st.container(key="nav"):
            unread = _unread_alert_count()
            for key, label, target in NAV_PAGES:
                shown = f"{label}  ·  {unread}" if key == "scheduled" and unread else label
                if st.button(shown, key=f"nav_{key}", disabled=(key == current)):
                    st.switch_page(target)
            if current_user() and st.button("Log out", key="nav_logout"):
                logout()
                st.rerun()

        html('<div class="silt-ws-label">Workspace</div>')
        _render_workspace_switcher()


def require_active_team(title: str = "") -> None:
    """Stop the page with a pointer to Workspaces when the user has no active
    team. Team-scoped API calls need an X-Team-Id, so without one they fail
    with a 400 that would otherwise be shown as "Backend unreachable"."""
    if st.session_state.get("active_team_id"):
        return
    if title:
        page_header(title)
    st.info("You're not in a team yet. Create a workspace or accept an invite to use this page.")
    if st.button("Go to Workspaces  →", type="primary", key="need_team_go"):
        st.switch_page("pages/8_Workspaces.py")
    st.stop()


def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="ds-page-sub">{esc(subtitle)}</div>' if subtitle else ""
    html(f'<div class="ds-page-title">{esc(title)}</div>{sub}')


def badge(text: str, kind: str = "neutral") -> str:
    check = (
        '<svg width="11" height="11" viewBox="0 0 24 24" fill="none">'
        '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/>'
        '<path d="M8.5 12.2l2.4 2.4 4.6-4.9" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        if kind == "verified" else ""
    )
    return f'<span class="ds-badge ds-badge-{kind}">{check}{esc(text)}</span>'


VERDICT_META = {
    "VERIFIED": ("verified", "Verified"),
    "VERIFIED_WITH_CAVEATS": ("warn", "Verified with caveats"),
    "UNVERIFIED": ("error", "Unverified"),
}


def verdict_badge(state: str | None) -> str:
    kind, label = VERDICT_META.get(state or "", ("neutral", "No verdict"))
    return badge(label, kind)


def _rejections_html(rejections: list[dict]) -> str:
    parts = []
    for r in rejections:
        issues = "".join(f"<li>{esc(i)}</li>" for i in r.get("issues") or [])
        parts.append(
            f'<div><b>Critic&#39;s reasoning:</b> {esc(r.get("summary") or "No summary given.")}'
            f'{"<ul>" + issues + "</ul>" if issues else ""}</div>'
        )
    return "".join(parts)


def render_verdict_banner(dash: dict) -> None:
    """Full-width trust banner; call before any KPI. Amber/red expand inline to
    the Critic's actual rejection reasoning."""
    state = dash.get("verdict_state") or ("VERIFIED" if dash.get("verified") else "UNVERIFIED")
    rejections = dash.get("rejections") or []
    if state == "VERIFIED":
        html('<div class="ds-verdict ds-verdict-ok"><div class="ds-verdict-head">✓ Fully Verified</div></div>')
        return
    if state == "VERIFIED_WITH_CAVEATS":
        cls, icon = "ds-verdict-warn", "⚠"
        n = dash.get("flagged_count", 0)
        title = f"Verified with caveats — {n} item(s) flagged"
    else:
        cls, icon = "ds-verdict-bad", "⛔"
        title = "Unverified — Critic could not confirm this analysis"
    body = _rejections_html(rejections) or esc(dash.get("verification_summary") or "No reasoning was recorded.")
    html(
        f'<details class="ds-verdict {cls}"><summary><span>{icon}</span><span>{esc(title)}</span>'
        f'<span class="ds-verdict-hint">Show Critic reasoning ▾</span></summary>'
        f'<div class="ds-verdict-body">{body}</div></details>'
    )


def trend_html(trend: str | None) -> str:
    arrow = {"up": "▲ Up", "down": "▼ Down", "flat": "▬ Flat"}.get(trend or "")
    return f'<span class="ds-trend-{trend}">{arrow}</span>' if arrow else '<span class="ds-row-meta">—</span>'


def trigger_badge(trigger: str | None) -> str:
    return badge("Scheduled", "running") if trigger == "scheduled" else badge("Manual", "neutral")


def stat_card(label: str, value: str, delta: str = "", direction: str = "up") -> str:
    arrow = "↑" if direction == "up" else "↓"
    cls = "ds-up" if direction == "up" else "ds-down"
    d = f'<div class="ds-stat-delta {cls}">{arrow} {esc(delta)}</div>' if delta else ""
    return f"""
    <div class="ds-card">
      <div class="ds-stat-label">{esc(label)}</div>
      <div class="ds-stat-value">{esc(value)}</div>
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
                 ensure_markers: bool = False, recolor: bool = True) -> go.Figure:
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
        font=dict(color="#52525b", family="Inter, system-ui, sans-serif", size=12),
        colorway=CHART_COLORWAY,
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=showlegend,
        legend=dict(font=dict(color="#52525b"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d4d4d8",
                        font=dict(color="#18181b", family="Inter, system-ui, sans-serif")),
        # empty string, not None -- None leaves the title node in place and
        # Plotly renders a literal "undefined" tspan above the plot
        title=dict(text=""),
    )
    fig.update_xaxes(gridcolor="#f0f0f2", zerolinecolor="#e4e4e7",
                     color="#71717a", showline=False, ticks="")
    fig.update_yaxes(gridcolor="#f0f0f2", zerolinecolor="#e4e4e7",
                     color="#71717a", showline=False, ticks="")
    if recolor:
        _recolor_traces(fig)
    return fig


def chart_card_css(container_key: str, style: dict, selected: bool = False) -> str:
    """Scoped CSS for one chart card: Chart Studio's card-level effects
    (glass/neon) and animations, plus the 'selected in the studio bar'
    outline. `style` must already be chart_studio.normalize_style()d -- its
    color is then a validated #rrggbb, and container_key is ours (a hex
    element id), so nothing user-controlled reaches the stylesheet raw."""
    from chart_studio import palette_colors, rgba

    sel = f".st-key-{container_key}"
    accent = palette_colors(style, 1)[0]
    rules = []
    if selected:
        rules.append(f"{sel} {{ border-color: var(--accent) !important; }}")
    effect, anim = style.get("effect"), style.get("animation")
    if effect == "glass" or anim == "shimmer":
        rules.append(
            f"{sel} {{ background: linear-gradient(135deg, {rgba(accent, 0.10)}, rgba(255,255,255,0.015) 60%) !important;"
            f" border-color: {rgba(accent, 0.35)} !important; backdrop-filter: blur(14px);"
            f" box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 10px 30px rgba(0,0,0,0.35);"
            f" overflow: hidden; }}"
        )
    if effect == "neon":
        rules.append(f"{sel} {{ border-color: {rgba(accent, 0.6)} !important;"
                     f" box-shadow: 0 0 18px {rgba(accent, 0.25)}, inset 0 0 12px {rgba(accent, 0.08)}; }}")
    if anim == "fade":
        rules.append(f"{sel} .stPlotlyChart {{ animation: cs-fade 0.9s ease both; }}")
    elif anim == "grow":
        rules.append(f"{sel} .stPlotlyChart {{ transform-origin: bottom; animation: cs-grow 0.8s cubic-bezier(.2,.8,.2,1) both; }}")
    elif anim == "float":
        rules.append(f"{sel} .stPlotlyChart {{ animation: cs-float 4s ease-in-out infinite; }}")
    elif anim == "pulse":
        name = f"cs-pulse-{container_key}"
        rules.append(f"@keyframes {name} {{ 0%,100% {{ box-shadow: 0 0 0 {rgba(accent, 0)}; }}"
                     f" 50% {{ box-shadow: 0 0 22px {rgba(accent, 0.45)}; }} }}")
        rules.append(f"{sel} {{ animation: {name} 2.6s ease-in-out infinite; }}")
    elif anim == "shimmer":
        rules.append(f"{sel} {{ position: relative; overflow: hidden; }}")
        rules.append(
            f"{sel}::before {{ content: ''; position: absolute; top: 0; bottom: 0; left: -60%; width: 45%;"
            f" background: linear-gradient(100deg, transparent, {rgba(shade_hex(accent), 0.16)}, transparent);"
            f" transform: skewX(-18deg); animation: cs-sheen 3.6s ease-in-out infinite;"
            f" pointer-events: none; z-index: 2; }}"
        )
    if not rules:
        return ""
    return "<style>" + "\n".join(rules) + "</style>"


def shade_hex(hex_color: str) -> str:
    from chart_studio import shade
    return shade(hex_color, 0.6)


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
         on_select_key: str | None = None, restyle_traces: bool = True):
    """Render a styled chart. Pass on_select_key to make it clickable -- the
    click/select event is then returned (and also lands in
    st.session_state[on_select_key]) instead of nothing.

    restyle_traces=False keeps the figure's own trace colors/modes (Chart
    Studio figures were already styled exactly as the user picked)."""
    kwargs = {}
    if on_select_key:
        # selection_mode="points" only (not the default points+box+lasso):
        # a plain click on a marker reliably registers as a point selection
        # this way, instead of needing an actual box/lasso drag.
        kwargs = {"on_select": "rerun", "key": on_select_key, "selection_mode": "points"}
    styled = style_chart(fig, height, showlegend, ensure_markers=bool(on_select_key) and restyle_traces,
                         recolor=restyle_traces)
    return st.plotly_chart(styled, use_container_width=True,
                           config={"displayModeBar": False}, **kwargs)


# ---------------------------------------------------------- click-to-inspect --
# The "Code" / "Formula" / "Data Used" panel for one dashboard element
# (backend/app/routers/inspect.py). Shared here so any page can open it the
# same way; today only pages/4_Dashboards.py does.

def open_inspect(dashboard_id: int, element_id: str, studio: dict | None = None) -> None:
    """Call this from a click handler (e.g. inside `if st.button(...):`).

    studio: for a chart re-drawn in Chart Studio -- {"label", "code",
    "formula", "calc" (DataFrame), "calc_caption", "plotted" (DataFrame)} --
    so the panel explains the chart as currently shown, not as generated."""
    st.session_state["_inspect_open"] = True
    st.session_state["_inspect_dashboard_id"] = dashboard_id
    st.session_state["_inspect_element_id"] = element_id
    st.session_state["_inspect_studio"] = studio


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
    studio = st.session_state.get("_inspect_studio")
    sub_title = 'style="font-size:0.86rem;color:var(--text-secondary);margin-top:12px;"'
    body = 'style="font-size:0.9rem;color:var(--text-primary);margin-top:6px;line-height:1.6;"'

    if studio:
        html(f'<div class="ds-row-meta" style="margin-bottom:10px;">Shown as <b>{esc(studio["label"])}</b> '
             f'(Chart Studio). The numbers are unchanged — only how they are drawn.</div>')

    html('<div class="ds-section-title">Code</div>')
    analysis_code = data.get("code") or "# No analysis code was recorded for this element."
    if studio:
        code = (
            "# ===== Step 1 · Analysis — ran in the sandbox on your uploaded CSV =====\n"
            f"{analysis_code.rstrip()}\n\n\n"
            f"# ===== Step 2 · {studio['label']} chart — Chart Studio =====\n"
            "# Rebuilds the chart from the values step 1 produced (run on its own, it needs only pandas/plotly).\n"
            f"{studio['code']}"
        )
    else:
        code = analysis_code
    st.code(code, language="python")
    st.download_button("Download .py", code, file_name=f"chart_{element_id[:8]}.py",
                       mime="text/x-python", key="inspect_dl")

    html('<div class="ds-section-title" style="margin-top:18px;">Formula</div>')
    if studio:
        html(f'<div {sub_title}>How the numbers were calculated from your CSV</div>')
    html(f'<div {body}>{esc(data.get("formula_explanation", ""))}</div>')
    if studio:
        html(f'<div {sub_title}>How the {esc(studio["label"].lower())} chart uses them</div>')
        html(f'<div {body}>{esc(studio["formula"])}</div>')
        calc = studio.get("calc")
        if calc is not None and len(calc):
            html(f'<div class="ds-row-meta" style="margin:8px 0 4px 0;">Calculation · {esc(studio.get("calc_caption", ""))}</div>')
            st.dataframe(calc, use_container_width=True, hide_index=True, height=min(240, 38 + 35 * len(calc)))

    html('<div class="ds-section-title" style="margin-top:18px;">Data Used</div>')
    data_slice = data.get("data_slice") or {}
    columns, rows = data_slice.get("columns") or [], data_slice.get("rows") or []
    if columns and rows:
        frame = pd.DataFrame(rows, columns=columns)
        lines = data.get("csv_lines") or []
        traced = sum(1 for n in lines if n)
        source = esc(data.get("csv_filename") or "the uploaded CSV")
        if traced:
            padded = (list(lines) + [None] * len(frame))[:len(frame)]
            frame.insert(0, "CSV line", pd.array([n or None for n in padded], dtype="Int64"))
            note = (f"{traced} of {len(frame)} rows traced to their exact line in <b>{source}</b> "
                    f"(line 1 is the header).")
        else:
            note = (f"Rows the analysis computed from <b>{source}</b>. They contain derived/aggregated "
                    f"values, so they don't map to single CSV lines.")
        html(f'<div class="ds-row-meta" style="margin:6px 0;">{note}</div>')
        st.dataframe(frame, use_container_width=True, height=240, hide_index=True)
    else:
        html('<div class="ds-row-meta" style="margin-top:6px;">No data slice was recorded for this element.</div>')
    if studio and studio.get("plotted") is not None:
        html(f'<div {sub_title}>Values plotted in this chart</div>')
        st.dataframe(studio["plotted"], use_container_width=True, hide_index=True, height=200)

    if data.get("flagged"):
        html('<div class="ds-section-title ds-flag-bad" style="margin-top:18px;">⚠ Critic reasoning</div>')
        issues = "".join(f"<li>{esc(i)}</li>" for i in data.get("critic_issues") or [])
        html(f'<div style="font-size:0.9rem;color:var(--text-primary);margin-top:6px;line-height:1.6;">'
             f'{esc(data.get("critic_reasoning") or "")}'
             f'{"<ul>" + issues + "</ul>" if issues else ""}</div>')

    html("<div style='height:10px'></div>")
    if st.button("Close", key="inspect_close"):
        st.session_state["_inspect_open"] = False
        st.rerun()
