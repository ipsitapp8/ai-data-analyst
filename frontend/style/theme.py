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

from api_client import (
    ApiError,
    accept_invitation,
    accept_invite,
    create_community,
    create_team,
    health,
    inspect_element,
    list_alerts,
    my_invites,
    my_workspaces,
)
from auth import current_user, logout, require_login, require_password

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Chart data gets real, distinct color even though the UI chrome around it
# stays flat and neutral -- that split is normal (Excel, Grafana, Tableau all
# do it): plain chrome, legible/vivid data encoding.
CHART_COLORWAY = ["#a78bfa", "#f472b6", "#7dd3fc", "#fbbf24", "#6d5efc", "#4ade80"]
ACCENT = "#a78bfa"      # violet -- the one interactive/brand UI color on the indigo skin
ACCENT_2 = "#7dd3fc"    # sky -- in-progress/secondary state

BASE_CSS = """
<style>
:root {
  --bg-base: #ffffff;
  --bg-page: #fbf8fb;
  --bg-card: #ffffff;
  --bg-card-hover: #f4f4f5;
  --bg-inset: #f4f4f5;
  --border: #e4e4e7;
  --border-strong: #d4d4d8;

  --text-primary: #18181b;
  --text-secondary: #52525b;
  --text-muted: #a1a1aa;

  --accent: #250e2c;
  --accent-dim: #3b1a47;
  --accent-bg: #f6eef6;
  --accent-border: #e6cfe2;

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


# ------------------------------------------------- dashboard look (opt-in) --
# A second set of design tokens for the dashboard view: a sage page, one large
# white rounded surface, a warm orange accent and pastel data colours. It is
# applied only by a page that calls inject_dashboard_theme() (today
# pages/4_Dashboards.py), and only to the main area, dialogs and popovers --
# the sidebar and every other page keep the base tokens above.
DASHBOARD_PALETTE: dict[str, str] = {
    "violet": "#A78BFA", "pink": "#F472B6", "sky": "#7DD3FC",
    "amber": "#FBBF24", "indigo": "#6D5EFC", "mint": "#4ADE80",
}
# Violet leads; the pink, sky and amber follow for further series. chart_studio's "Dashboard"
# palette is this same list (a test keeps the two in step).
DASHBOARD_COLORWAY = list(DASHBOARD_PALETTE.values())
DASHBOARD_TOKENS: dict[str, str] = {
    "--bg-page": "#0B0820",
    "--bg-card": "#1B1347",
    "--bg-card-hover": "#261A5E",
    "--bg-inset": "#120D33",
    "--border": "#34296F",
    "--border-strong": "#4C3D9A",
    "--text-primary": "#ECE8FF",
    "--text-secondary": "#B6B0E0",
    "--text-muted": "#8A84BD",
    "--accent": DASHBOARD_PALETTE["violet"],
    "--accent-dim": "#8B5CF6",
    "--accent-ink": "#C4B5FD",        # light violet that reads as text on the indigo surface
    "--accent-bg": "rgba(139,92,246,0.18)",
    "--accent-border": "rgba(167,139,250,0.38)",
    "--radius-lg": "16px",
    "--radius-md": "12px",
    "--radius-sm": "8px",
    "--dash-surface": "#140E3A",
    "--dash-surface-radius": "28px",
    "--dash-gap": "20px",
    "--dash-shadow": "0 1px 2px rgba(3,2,16,0.4), 0 18px 40px rgba(3,2,16,0.5)",
    "--dash-card-shadow": "0 1px 2px rgba(3,2,16,0.4), 0 4px 12px rgba(3,2,16,0.3)",
    **{f"--dash-c{i}": colour for i, colour in enumerate(DASHBOARD_PALETTE.values(), start=1)},
}
# Plotly draws on a canvas that page CSS cannot reach, so the chart's own colours are set here.
DASHBOARD_CHART = {"text": "#ece8ff", "muted": "#a9a3d6", "grid": "rgba(167,139,250,0.12)", "line": "rgba(167,139,250,0.28)"}

# .stMain is the main scroll area (its data-testid varies with the page's content)
_DASHBOARD_SCOPE = '.stMain, [data-testid="stDialog"], [data-baseweb="popover"]'

_DASHBOARD_CSS = """
.stMain { background: var(--bg-page) !important; accent-color: var(--accent); }
.stMain [data-testid="stBottom"], .stMain [data-testid="stBottom"] > div,
.stMain [data-testid="stBottomBlockContainer"] { background: var(--bg-page) !important; }

/* the one large surface every dashboard element sits on */
.stMain [data-testid="stMainBlockContainer"] {
  background: var(--dash-surface); border: 1px solid var(--border);
  border-radius: var(--dash-surface-radius); box-shadow: var(--dash-shadow);
  width: calc(100% - 48px) !important; max-width: 1240px;
  margin: 82px auto 30px auto !important; padding: 34px 38px 40px 38px !important;
}
.stMain [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: var(--dash-gap); }

/* typography */
.stMain { color: var(--text-primary); }
.stMain .ds-page-title { font-size: 1.75rem; font-weight: 700; line-height: 1.22; letter-spacing: -0.022em; color: var(--text-primary); }
.stMain .ds-page-sub { color: var(--text-muted); }
.stMain .ds-section-title { font-size: 1rem; font-weight: 650; color: var(--text-primary); }
.stMain .ds-row-title { color: var(--text-primary); }
.stMain .ds-row-meta { color: var(--text-muted); }
.dash-head-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 4px 0 12px 0; }
.dash-eyebrow { display: flex; align-items: baseline; gap: 8px; font-size: 1.02rem; font-weight: 650;
                letter-spacing: -0.01em; color: var(--text-primary); margin: 12px 0 -4px 2px; }
.dash-eyebrow b { font-size: 0.74rem; font-weight: 600; color: var(--accent-ink); background: var(--accent-bg);
                  border-radius: 999px; padding: 1px 8px; }
.dash-narrative { font-size: 0.97rem; line-height: 1.75; color: var(--text-primary); margin-top: 10px; }

/* cards: white, thin border, a restrained shadow; they sit on the white surface */
.stMain [class*="st-key-card"], .stMain [class*="st-key-flat"],
.stMain [class*="st-key-chartcard_"] {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important; box-shadow: var(--dash-card-shadow) !important;
}
.stMain [class*="st-key-card"], .stMain [class*="st-key-chartcard_"] { padding: 20px 22px !important; }
.stMain .st-key-card_studio { background: var(--bg-inset) !important; border-color: var(--border) !important; box-shadow: none !important; }
.stMain .st-key-card_narr { border-left: 4px solid var(--dash-c3) !important; }
.stMain .st-key-card_verif { background: var(--bg-inset) !important; box-shadow: none !important; }
.stMain [data-testid="stExpander"] { border-radius: var(--radius-lg) !important; }
.stMain [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius-md); overflow: hidden; }
.stMain .ds-verdict { border-radius: var(--radius-md); margin-bottom: 16px; }
/* stylesheets and the fixed top strip are page furniture, not rows of the surface */
.stMain [data-testid="stElementContainer"]:has(style, link[rel], .silt-topbar):not(:has(.dash-eyebrow)) { position: absolute !important; }

/* buttons */
.stMain button[kind="secondary"] { border-radius: 999px !important; border-color: var(--border-strong) !important; }
.stMain button[kind="secondary"]:hover { border-color: var(--accent) !important; color: var(--accent-ink) !important; }
/* dark text on the orange: white on #FF824D is too faint to read */
.stMain button[kind="primary"], .stMain button[kind="primary"]:hover,
[data-baseweb="popover"] button[kind="primary"], [data-baseweb="popover"] button[kind="primary"]:hover {
  border-radius: 999px !important; color: var(--text-primary) !important;
}
.stMain button[kind="primary"] p, [data-baseweb="popover"] button[kind="primary"] p { color: var(--text-primary) !important; }
/* the title row's actions sit side by side and wrap when there is no room */
/* on a narrow surface the actions drop under the title instead of squeezing beside it */
.stMain [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-dash_actions) { flex-wrap: wrap; row-gap: 8px; }
.stMain [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-dash_actions) > [data-testid="stColumn"] {
  min-width: min(340px, 100%) !important; flex: 1 1 340px !important;
}
/* likewise the studio's heading keeps its text readable and lets its two buttons wrap */
.st-key-card_studio [data-testid="stColumn"]:has(> div .ds-section-title) {
  min-width: min(320px, 100%) !important; flex: 1 1 320px !important; padding-bottom: 16px;
}
.st-key-dash_actions { flex-wrap: wrap !important; justify-content: flex-end; gap: 10px !important; }
.st-key-dash_actions > * { flex: 0 0 auto !important; width: auto !important; }
.st-key-dash_actions button { padding-left: 1.1em !important; padding-right: 1.1em !important; }
.st-key-dash_actions button, .st-key-dash_actions [data-testid="stPopover"] { width: max-content !important; max-width: none !important; }
.st-key-dash_actions button div, .st-key-dash_actions button p {
  white-space: nowrap; overflow: visible !important; text-overflow: clip !important; max-width: none !important;
}

/* A selected segment is a radio too, so the base radio rule paints a solid block behind its
   label; inside the dashboard the segment's own tint is the selection mark. */
.stMain [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] div { background: transparent !important; }
.stMain [data-testid="stButtonGroup"] button[aria-checked="true"] p { color: var(--accent-ink) !important; font-weight: 600; }

/* KPI grid: as many equal cards per row as fit. Each card is white, with a short bar in the
   next palette colour so a row of cards reads as one set. */
.st-key-dash_kpis [data-testid="stHorizontalBlock"] {
  display: grid !important; grid-template-columns: repeat(auto-fit, minmax(min(220px, 100%), 1fr)); gap: var(--dash-gap);
}
.st-key-dash_kpis [data-testid="stColumn"] { width: auto !important; min-width: 0 !important; flex: none !important; --kpi-mark: var(--dash-c1); }
.st-key-dash_kpis [data-testid="stColumn"]:nth-child(6n+2) { --kpi-mark: var(--dash-c3); }
.st-key-dash_kpis [data-testid="stColumn"]:nth-child(6n+3) { --kpi-mark: var(--dash-c2); }
.st-key-dash_kpis [data-testid="stColumn"]:nth-child(6n+4) { --kpi-mark: var(--dash-c5); }
.st-key-dash_kpis [data-testid="stColumn"]:nth-child(6n+5) { --kpi-mark: var(--dash-c6); }
.st-key-dash_kpis [data-testid="stColumn"]:nth-child(6n+6) { --kpi-mark: var(--dash-c4); }
.st-key-dash_kpis [class*="st-key-kpi_"] .stButton > button {
  border-radius: var(--radius-lg) !important; border: 1px solid var(--border) !important;
  background: var(--bg-card) !important; box-shadow: var(--dash-card-shadow) !important;
  padding: 36px 22px 20px 22px !important; min-height: 132px; opacity: 1 !important;
  align-items: flex-start !important; justify-content: flex-start !important;
}
.st-key-dash_kpis [class*="st-key-kpi_"] .stButton > button::before {
  content: ""; position: absolute; top: 20px; left: 22px; width: 28px; height: 4px;
  border-radius: 999px; background: var(--kpi-mark);
}
.st-key-dash_kpis [class*="st-key-kpi_"] .stButton > button > div { width: 100%; justify-content: flex-start; text-align: left; }
.st-key-dash_kpis [class*="st-key-kpi_"] .stButton > button:hover:not(:disabled) { border-color: var(--accent) !important; background: var(--bg-card) !important; }
.st-key-dash_kpis [class*="st-key-kpi_"] .stButton > button p {
  font-family: var(--font) !important; font-size: 0.8rem !important; font-weight: 500 !important;
  color: var(--text-muted) !important; letter-spacing: 0 !important; line-height: 1.5 !important;
  text-align: left; margin: 0 !important;
}
.st-key-dash_kpis [class*="st-key-kpi_"] .stButton > button strong {
  display: inline-block; margin: 8px 0 4px 0; font-size: 1.95rem !important; font-weight: 700 !important;
  color: var(--text-primary) !important; line-height: 1.15; letter-spacing: -0.025em; overflow-wrap: anywhere;
}

/* chart grid: two across, one when the surface is narrow; an odd last chart spans the row */
.st-key-dash_charts [data-testid="stHorizontalBlock"]:has([class*="st-key-chartcard_"], [class*="st-key-card_ch"]) {
  display: grid !important; gap: var(--dash-gap);
  grid-template-columns: repeat(auto-fit, minmax(min(100%, max(380px, calc(50% - var(--dash-gap)))), 1fr));
}
.st-key-dash_charts [data-testid="stColumn"]:has([class*="st-key-chartcard_"], [class*="st-key-card_ch"]) {
  width: auto !important; min-width: 0 !important; flex: none !important;
}
.st-key-dash_charts [data-testid="stColumn"]:has([class*="st-key-chartcard_"], [class*="st-key-card_ch"]):last-child:nth-child(odd) {
  grid-column: 1 / -1;
}
/* the studio's control rows wrap instead of squeezing their controls */
.st-key-card_studio [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 10px; }
.st-key-card_studio [data-testid="stColumn"] { min-width: min(110px, 100%) !important; }
.st-key-card_studio [data-testid="stColumn"]:has(.st-key-cs_inspect, .st-key-cs_reset) { min-width: min(160px, 100%) !important; }
/* the dashboard list: each row's Open button sits on the row's centre line */
.st-key-flat_pick [data-testid="stHorizontalBlock"] { align-items: center; padding-right: 18px; }
.st-key-flat_pick [data-testid="stHorizontalBlock"] + [data-testid="stHorizontalBlock"],
.st-key-flat_pick [data-testid="stLayoutWrapper"] + [data-testid="stLayoutWrapper"] { border-top: 1px solid var(--border); }
.st-key-card_studio button p { white-space: nowrap; }
.stMain [class*="st-key-cs_edit_"] .stButton > button, .stMain [class*="st-key-cs_edit_"] .stButton > button p { white-space: nowrap !important; }
.st-key-dash_summary [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: var(--dash-gap); }
.st-key-dash_summary [data-testid="stColumn"] { min-width: min(300px, 100%) !important; }

@media (max-width: 900px) {
  .stMain [data-testid="stMainBlockContainer"] {
    width: calc(100% - 20px) !important; padding: 20px 16px 26px 16px !important; border-radius: 20px;
  }
}
"""


def dashboard_theme_css(tokens: dict[str, str] | None = None) -> str:
    """The dashboard stylesheet, with its tokens declared on the main area
    (plus dialogs and popovers, which render outside it) rather than :root."""
    declared = " ".join(f"{name}: {value};" for name, value in (tokens or DASHBOARD_TOKENS).items())
    css = f"{_DASHBOARD_SCOPE} {{ {declared} }} {_DASHBOARD_CSS}"
    return "<style>" + " ".join(line.strip() for line in css.splitlines() if line.strip()) + "</style>"


def inject_dashboard_theme() -> None:
    """Call after page_setup() on a page that should use the dashboard look."""
    st.markdown(dashboard_theme_css(), unsafe_allow_html=True)
    glass = (Path(__file__).with_name("dashboard_glass.css")).read_text(encoding="utf-8")
    indigo = (Path(__file__).with_name("indigo.css")).read_text(encoding="utf-8")   # re-assert after the dashboard sheet
    st.markdown("<style>" + " ".join(line.strip() for line in (glass + chr(10) + indigo).splitlines() if line.strip()) + "</style>", unsafe_allow_html=True)


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
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Poppins:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&family=JetBrains+Mono:wght@400;500'
    '&display=swap" rel="stylesheet">'
)

# Line icons for the sidebar nav, drawn as CSS masks so they inherit the row's colour.
_NAV_ICONS = {   # Lucide (lucide.dev), one consistent stroke family
    "home": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "overview": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "datasets": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "analyses": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="m19 9-5 5-4-4-3 3"/>',
    "analyst": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "investigations": '<path d="m21 21-4.34-4.34"/><circle cx="11" cy="11" r="8"/>',
    "what_if": '<line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
    "copilot": '<path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z"/><path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1"/>',
    "scheduled": '<path d="M12 6v6l4 2"/><circle cx="12" cy="12" r="10"/>',
    "dashboards": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "reports": '<path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "audit": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "data_quality": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "semantic": '<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/>',
    "evaluation": '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.453 15h11.094"/><path d="M8.5 2h7"/>',
    "discussion": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
    "team_overview": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "workspaces": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/><path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>',
    "settings": '<path d="M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915"/><circle cx="12" cy="12" r="3"/>',
    "logout": '<path d="m16 17 5-5-5-5"/><path d="M21 12H9"/><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>',
}


def _nav_icon_css() -> str:
    rules = []
    for key, body in _NAV_ICONS.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
               "stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>" + body.replace('"', "'") + "</svg>")
        uri = "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe="/:=' ,.-()")
        rules.append(f'.st-key-nav_{key} button {{ --icon: url("{uri}"); }}')
    return " ".join(rules)


# Silt's pages are light, but Streamlit takes its text colour from the browser's
# preference: in a dark-mode browser widget values, chat messages and expander
# text were near-white on white. These rules name only those Streamlit
# elements, so nothing else on a page changes.
_READABLE_TEXT_CSS = (
    "<style>"
    "[data-testid='stChatMessage']{background:rgba(24,17,60,.7);border:1px solid rgba(167,139,250,.22);border-radius:10px;}"
    "[data-testid='stChatMessage'] *,[data-testid='stExpander'] *{color:#ece8ff;}"
    "[data-testid='stChatMessage'] th,[data-testid='stChatMessage'] td{border:1px solid rgba(167,139,250,.22);padding:2px 8px;}"
    # light text on the indigo surface: code chips and blocks get a deep, translucent background
    "[data-testid='stChatMessage'] code,[data-testid='stChatMessage'] pre,"
    "[data-testid='stExpander'] code,[data-testid='stExpander'] pre{background:rgba(8,5,26,.8) !important;}"
    "[data-testid='stExpander'] summary,[data-testid='stExpander'] [data-testid='stJson']"
    "{background:rgba(24,17,60,.55) !important;}"
    "[data-testid='stTabs'] [role='tab'][aria-selected='false'] p{color:#a9a3d6 !important;}"
    "[data-testid='stSelectbox'] input,[data-testid='stSelectbox'] p,[data-testid='stMultiSelect'] input,"
    "[data-testid='stTextInput'] input,[data-testid='stTextArea'] textarea,[data-testid='stNumberInput'] input"
    "{color:#ece8ff !important;-webkit-text-fill-color:#ece8ff !important;}"
    "[data-testid='stWidgetLabel'] p{color:#b6b0e0 !important;-webkit-text-fill-color:#b6b0e0 !important;}"
    "</style>"
)


def _sidebar_css() -> str:
    """The sidebar's final stylesheet (style/sidebar.css). Loaded after every other skin so it has the last word."""
    raw = Path(__file__).with_name("sidebar.css").read_text(encoding="utf-8")
    return "<style>" + " ".join(line.strip() for line in raw.splitlines() if line.strip()) + "</style>"


def inject_base_css() -> None:
    skin = (Path(__file__).with_name("silt.css")).read_text(encoding="utf-8") + "\n" + (
        Path(__file__).with_name("glass.css")).read_text(encoding="utf-8") + "\n" + (
        Path(__file__).with_name("mindset.css")).read_text(encoding="utf-8") + "\n" + (
        Path(__file__).with_name("indigo.css")).read_text(encoding="utf-8") + chr(10) + (
        Path(__file__).with_name("workbench.css")).read_text(encoding="utf-8")
    # Flatten to one line: markdown ends an HTML block at the first blank line,
    # which would spill the rest of the stylesheet onto the page as text.
    skin = " ".join(line.strip() for line in skin.splitlines() if line.strip())
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    st.markdown(_FONTS_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{skin} {_nav_icon_css()}</style>", unsafe_allow_html=True)
    st.markdown(_READABLE_TEXT_CSS, unsafe_allow_html=True)
    st.markdown(_ONBOARD_HANDOFF_CSS, unsafe_allow_html=True)
    st.markdown(_sidebar_css(), unsafe_allow_html=True)


# After "Create workspace" the next run (the real page) can take several seconds on a slow backend or database, and
# Streamlit keeps the previous run's forms on screen, dimmed, until it finishes. Those stale onboarding cards are hidden and a
# short message is shown instead, so it reads as "loading" rather than as a broken, repeated form.
_ONBOARD_HANDOFF_CSS = """<style>
[class*="st-key-card_onboard"]:has([data-stale="true"]), [class*="st-key-onb_logout"]:has([data-stale="true"]) { display: none !important; }
[data-testid="stMainBlockContainer"]:has([class*="st-key-card_onboard"] [data-stale="true"])::before {
  content: "Setting up your workspace…"; display: block; margin: 22vh auto 0; text-align: center; font-size: 1.05rem; color: #bdb9cc; letter-spacing: .01em; }
</style>"""


def page_setup(title: str, sidebar: bool = True, public: bool = False) -> None:
    st.set_page_config(
        page_title=f"{title} · SILT",
        page_icon="▤",
        layout="wide",
        initial_sidebar_state="auto" if sidebar else "collapsed",
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
    _consume_invite_param()
    _ensure_active_team()
    inject_base_css()
    _require_team_onboarding()
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


def _consume_invite_param() -> None:
    """Accept an invitation arriving as ?invite=<token> (the link in an invite
    email or the one an admin shares). Runs after login so the server can bind
    the token to the signed-in account's email."""
    token = st.query_params.get("invite")
    if not token:
        return
    try:
        result = accept_invitation(token)
        invalidate_workspaces_cache()
        st.session_state["active_team_id"] = result["team_id"]
        st.session_state["active_community_id"] = result["community_id"]
        st.session_state["flash"] = f"You joined {result['team_name']}."
    except ApiError as e:
        st.session_state["flash_error"] = (
            "That invitation link couldn't be used. It may have expired, been used already, or been "
            f"sent to a different email address. ({e})"
        )
    st.query_params.clear()


def _reload_page() -> None:
    """Load the page afresh after a workspace step. Redrawing in place left the onboarding card on screen above
    the new page (the browser kept that block when the next run drew over it); a full load starts from a clean page,
    and the sign-in cookie brings the user straight back to Home."""
    import streamlit.components.v1 as components
    st.markdown("<style>[class*='st-key-card_onboard']{display:none !important}</style>", unsafe_allow_html=True)
    components.html("<script>window.parent.location.reload();</script>", height=0)
    st.stop()


def _require_team_onboarding() -> None:
    """Full-screen first-run step for a logged-in user with no team: accept a
    pending invite or create a workspace, with no app nav shown. Every
    team-scoped page needs an X-Team-Id, so nothing else is usable until this
    is done -- better to ask up front than to show "not in a team" on each page."""
    if st.session_state.get("active_team_id"):
        return
    try:
        _cached_my_workspaces()
    except ApiError:
        return  # backend down: let the page show its own error rather than onboarding
    st.markdown(
        "<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
        "[data-testid='stExpandSidebarButton']{display:none !important;}"
        ".block-container{max-width:640px !important;padding-top:4rem !important;}</style>",
        unsafe_allow_html=True,
    )
    user = current_user() or {}
    html(f'<div class="silt-brand2">{_LOGO}<div class="n">Silt</div></div>')
    page_header(
        f"Welcome, {user.get('display_name', '')}".rstrip(", "),
        "Everything in Silt lives in a workspace. Join one you were invited to, or create your own.",
    )

    if st.session_state.get("flash_error"):
        st.error(st.session_state.pop("flash_error"))
    try:
        invites = my_invites()
    except ApiError:
        invites = []
    if invites:
        with st.container(key="card_onboard_invites"):
            html('<div class="ds-section-title">You\'ve been invited</div>')
            for inv in invites:
                c1, c2 = st.columns([3, 1])
                with c1:
                    html(
                        f'<div class="ds-row-title">{esc(inv["team_name"])}</div>'
                        f'<div class="ds-row-meta">in {esc(inv["community_name"])} · role: {esc(inv["role"])}</div>'
                    )
                with c2:
                    if st.button("Accept", key=f"onb_accept_{inv['team_id']}", type="primary"):
                        try:
                            accept_invite(inv["team_id"])
                            invalidate_workspaces_cache()
                            st.session_state["_go_home"] = True
                            _reload_page()
                        except ApiError as e:
                            st.error(f"Could not accept: {e}")
        html("<div style='height:14px'></div>")

    with st.container(key="card_onboard_create"):
        html('<div class="ds-section-title">Create a workspace</div>')
        with st.form("onboard_create_form"):
            ws_name = st.text_input("Workspace name", placeholder="e.g. Acme Inc.")
            team_name = st.text_input("First team", value="General")
            submitted = st.form_submit_button("Create workspace", type="primary")
        if submitted:
            if not ws_name.strip() or not team_name.strip():
                st.error("Enter a workspace name and a team name.")
            else:
                try:
                    community = create_community(ws_name.strip())
                    team = create_team(community["id"], team_name.strip())
                    st.session_state["active_community_id"] = community["id"]
                    st.session_state["active_team_id"] = team["id"]
                    invalidate_workspaces_cache()
                    st.session_state["_go_home"] = True
                    _reload_page()
                except ApiError as e:
                    st.error(f"Could not create workspace: {e}")

    with st.container(key="card_onboard_code"):
        html('<div class="ds-section-title">Have an invitation link or code?</div>')
        with st.form("onboard_code_form"):
            code = st.text_input("Invitation code", placeholder="Paste the code from your invitation link")
            go = st.form_submit_button("Join with invitation")
        if go and code.strip():
            try:
                result = accept_invitation(code.strip().split("invite=")[-1])
                invalidate_workspaces_cache()
                st.session_state["active_team_id"] = result["team_id"]
                st.session_state["active_community_id"] = result["community_id"]
                st.session_state["_go_home"] = True
                _reload_page()
            except ApiError as e:
                st.error(f"Could not use that invitation: {e}")

    if st.button("Log out", key="onb_logout"):
        logout()
        st.rerun()
    st.stop()


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
    ("analyst", "Silt Copilot", "pages/12_Copilot.py"),
    ("investigations", "Root cause", "pages/13_Investigations.py"),
    ("what_if", "What-if", "pages/14_What_If.py"),
    ("copilot", "Drill-down sessions", "pages/19_Drilldown_Sessions.py"),
    ("scheduled", "Scheduled", "pages/10_Scheduled.py"),
    ("dashboards", "Dashboards", "pages/4_Dashboards.py"),
    ("reports", "Reports", "pages/5_Reports.py"),
    ("audit", "Audit trail", "pages/6_Audit_Trail.py"),
    ("data_quality", "Data quality", "pages/15_Data_Quality.py"),
    ("semantic", "Semantic layer", "pages/16_Semantic_Layer.py"),
    ("evaluation", "Evaluation", "pages/17_Evaluation.py"),
    ("discussion", "Discussion", "pages/18_Discussion.py"),
    ("workspaces", "Workspaces", "pages/8_Workspaces.py"),
    ("team_overview", "Team overview", "pages/9_Team_Overview.py"),
]


# Everyday pages sit in the rail; the rest are one click away under "More" (nothing is removed, and the section opens by
# itself when you are on one of its pages). Every key is a key of NAV_PAGES.
NAV_MAIN = ["home", "overview", "datasets", "analyses", "analyst", "investigations", "what_if", "dashboards", "reports"]
NAV_MORE = ["copilot", "scheduled", "audit", "data_quality", "semantic", "evaluation",
            "discussion", "team_overview", "workspaces"]


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
    name = _active_workspace_name(user)
    initials = "".join(w[0] for w in (user or {}).get("display_name", name).split()[:2]).upper() or "•"
    html(
        f"""
        <div class="silt-topbar">
          <div class="silt-tb-user"><div class="silt-tb-avatar">{esc(initials)}</div>
            <div class="silt-tb-ws">{esc(name)}</div></div>
        </div>
        """
    )
    if user:
        # Always visible, unlike the sidebar one at the bottom of the long nav list.
        with st.container(key="topbar_logout"):
            if st.button("Log out", key="topbar_logout_btn"):
                logout()
                st.rerun()


_LOGO_MARK = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linecap="round"><path d="M5 20L17 4M9 21l10-13M4 15l8-11"/></svg>'
)
_LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#837ab6" stroke-width="2" '
    'stroke-linecap="round"><path d="M5 20L17 4M9 21l10-13M4 15l8-11"/></svg>'
)


def _account_card() -> None:
    """The workspace / account selector at the foot of the sidebar: a card showing where you are; it opens the workspace
    switcher and Log out. Only the label is new, the switcher and the log-out button are the existing ones."""
    user = current_user() or {}
    active_team = st.session_state.get("active_team_id")
    workspace, team = _active_workspace_name(user), ""
    try:
        for c in _cached_my_workspaces()["communities"]:
            for t in c["teams"]:
                if t["id"] == active_team:
                    workspace, team = c["name"], t["name"]
    except ApiError:
        pass
    who = (user.get("display_name") or user.get("email") or "Account").strip()
    initial = (workspace[:1] or who[:1] or "S").upper()
    sub = f"{team} \u00b7 {who}" if team else who
    css_str = lambda v: '"' + str(v).replace("\\", "").replace('"', "").replace("\n", " ")[:42] + '"'
    st.markdown(f"<style>.st-key-sb_account button{{--ini:{css_str(initial)};--sub:{css_str(sub)};}}</style>", unsafe_allow_html=True)
    with st.container(key="sb_account"):
        with st.popover(workspace or "Workspace", use_container_width=True):
            _render_workspace_switcher()
            if current_user() and st.button("Log out", key="nav_logout"):
                logout()
                st.rerun()


def render_sidebar(current: str) -> None:
    """Brand, grouped navigation, and the workspace / account card; plus the top status strip."""
    render_topbar()
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if st.session_state.get("flash_error"):
        st.error(st.session_state.pop("flash_error"))
    with st.sidebar:
        html(f'<div class="sb-brand"><div class="sb-logo">{_LOGO_MARK}</div>'
             '<div class="sb-wm"><b>Silt</b><span>Data intelligence</span></div></div>')
        with st.container(key="nav"):
            unread = _unread_alert_count()
            pages = {key: (label, target) for key, label, target in NAV_PAGES}

            def _item(key: str) -> None:
                label, target = pages[key]
                shown = f"{label}  \u00b7  {unread}" if key == "scheduled" and unread else label
                if st.button(shown, key=f"nav_{key}", disabled=(key == current)):
                    st.switch_page(target)

            for key in NAV_MAIN:
                _item(key)
            with st.expander("More", expanded=current in NAV_MORE):
                for key in NAV_MORE:
                    _item(key)
        _account_card()


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


EVIDENCE_META = {
    "verified": ("verified", "Verified"),
    "verified_with_caveats": ("warn", "Verified with caveats"),
    "unverified": ("error", "Not verified"),
}
CHECK_META = {"pass": ("verified", "Passed"), "fail": ("error", "Failed"), "not_run": ("warn", "Not run"),
              "warn": ("warn", "Caveat"), "not_applicable": ("neutral", "Not applicable")}


def evidence_badge(status: str | None) -> str:
    """Badge for one claim's deterministic evidence status."""
    kind, label = EVIDENCE_META.get(status or "", ("neutral", "No evidence record"))
    return badge(label, kind)


def render_evidence(record: dict | None) -> None:
    """One evidence record: status, each check with its outcome and reason, the
    three kinds of validity, limitations and provenance."""
    if not record:
        html('<div class="ds-row-meta">No evidence record exists for this element (it predates verification).</div>')
        return
    html(f'<div style="margin:6px 0 10px 0;">{evidence_badge(record.get("status"))}</div>')
    rows = []
    for c in record.get("checks") or []:
        kind, label = CHECK_META.get(c.get("outcome"), ("neutral", c.get("outcome", "")))
        need = "required" if c.get("required") else "informational"
        rows.append(
            f'<div style="padding:7px 0;border-top:1px solid var(--border);">'
            f'<div style="display:flex;gap:10px;align-items:center;">{badge(label, kind)}'
            f'<span class="ds-row-title">{esc(str(c.get("check", "")).replace("_", " "))}</span>'
            f'<span class="ds-row-meta">{need}</span></div>'
            f'<div class="ds-row-meta" style="margin-top:3px;">{esc(c.get("detail") or "")}</div></div>')
    html("".join(rows))
    validity = record.get("validity") or {}
    html('<div class="ds-row-meta" style="margin-top:10px;line-height:1.7;">'
         f'<b>Mathematical validity:</b> {esc(str(validity.get("mathematical", "—")).replace("_", " "))}<br>'
         f'<b>Statistical validity:</b> {esc(str(validity.get("statistical", "—")).replace("_", " "))}<br>'
         f'<b>Causal validity:</b> {esc(str(validity.get("causal", "—")).replace("_", " "))}</div>')
    limits = record.get("limitations") or []
    if limits:
        html('<div class="ds-row-meta" style="margin-top:6px;"><b>Limitations:</b> '
             + esc(", ".join(str(x).replace("_", " ") for x in limits)) + "</div>")
    metric = record.get("metric") or {}
    bits = []
    if metric.get("formula"):
        bits.append(f'formula {esc(metric["formula"])}')
    if record.get("recomputed_value"):
        bits.append(f'recomputed value {esc(record["recomputed_value"])}')
    if record.get("dataset_fingerprint"):
        bits.append(f'dataset {esc(record["dataset_fingerprint"][:12])}…')
    if bits:
        html('<div class="ds-row-meta" style="margin-top:6px;"><b>Provenance:</b> ' + " · ".join(bits) + "</div>")


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
    summary = dash.get("evidence_summary") or {}
    not_full = summary.get("verified_with_caveats", 0) + summary.get("unverified", 0)
    if state == "VERIFIED_WITH_CAVEATS":
        cls, icon = "ds-verdict-warn", "⚠"
        n = dash.get("flagged_count", 0)
        if n:
            title = f"Verified with caveats — {n} item(s) flagged"
        elif not_full:
            title = f"Verified with caveats — {not_full} of {summary.get('claims', not_full)} claims not fully verified"
        else:
            title = "Verified with caveats"
    else:
        cls, icon = "ds-verdict-bad", "⛔"
        title = "Unverified — verification could not confirm this analysis"
    body = _rejections_html(rejections) or esc(dash.get("verification_summary") or "No reasoning was recorded.")
    html(
        f'<details class="ds-verdict {cls}"><summary><span>{icon}</span><span>{esc(title)}</span>'
        f'<span class="ds-verdict-hint">Show reasoning ▾</span></summary>'
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
                 ensure_markers: bool = False, recolor: bool = True,
                 colorway: list[str] | None = None) -> go.Figure:
    """Force any figure -- including sandbox-generated ones -- into the SILT look.

    ensure_markers: sandbox line charts default to mode="lines" (px.line's
    default), which has no clickable points for Plotly's on_select -- a click
    anywhere on the bare line doesn't register as a point selection. Pass True
    (click-to-inspect charts do) to add markers to any bare-line scatter trace
    so there's something to actually click.
    """
    colorway = colorway or CHART_COLORWAY
    if ensure_markers:
        for trace in fig.data:
            if getattr(trace, "type", None) == "scatter" and trace.mode and "markers" not in trace.mode:
                trace.mode = trace.mode + "+markers"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b6b0e0", family="Inter, system-ui, sans-serif", size=12),
        colorway=colorway,
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=showlegend,
        legend=dict(font=dict(color="#b6b0e0"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1b1347", bordercolor="#4c3d9a",
                        font=dict(color="#ece8ff", family="Inter, system-ui, sans-serif")),
        # empty string, not None -- None leaves the title node in place and
        # Plotly renders a literal "undefined" tspan above the plot
        title=dict(text=""),
    )
    fig.update_xaxes(gridcolor="rgba(167,139,250,0.12)", zerolinecolor="rgba(167,139,250,0.28)",
                     color="#a9a3d6", showline=False, ticks="")
    fig.update_yaxes(gridcolor="rgba(167,139,250,0.12)", zerolinecolor="rgba(167,139,250,0.28)",
                     color="#a9a3d6", showline=False, ticks="")
    if recolor:
        _recolor_traces(fig, colorway)
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


def _recolor_traces(fig: go.Figure, colorway: list[str] | None = None) -> None:
    """Repaint solid trace colors from the colorway (CHART_COLORWAY by default).

    `colorway` only supplies colors Plotly would otherwise auto-assign. Plotly
    Express bakes explicit per-trace colors whenever the sandbox code groups by
    a column, which would otherwise leave stock blue/red charts sitting inside
    the SILT palette. Array-valued colors (continuous scales) are left
    alone -- overwriting those would destroy the encoding.
    """
    colorway = colorway or CHART_COLORWAY
    for i, trace in enumerate(fig.data):
        color = colorway[i % len(colorway)]
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


def dashboard_chart(fig: go.Figure, own_traces: bool = False) -> go.Figure:
    """Finish a styled figure in the dashboard look: muted labels, light
    gridlines, the legend in a row above the plot, rounded bars, even line and
    marker weights. Only presentation changes -- no trace's data, type, hover
    text or name is touched.

    own_traces: the figure was drawn by Chart Studio with the weights and
    corner radius the user chose, so only the frame around the traces is set."""
    c = DASHBOARD_CHART
    family = "Inter, system-ui, sans-serif"
    named = [t for t in fig.data if getattr(t, "name", None)]
    if not own_traces and len(fig.data) == 1 and not named:
        fig.update_layout(showlegend=False)          # one unnamed series: a legend reading "trace 0" says nothing
    legend_shown = bool(fig.layout.showlegend)
    fig.update_layout(
        font=dict(color=c["muted"], family=family, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=c["muted"], size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1b1347", bordercolor=c["line"], font=dict(color=c["text"], family=family)),
        margin=dict(l=8, r=8, t=40 if legend_shown else 12, b=8),
    )
    axis = dict(gridcolor=c["grid"], zerolinecolor=c["line"], linecolor=c["line"], color=c["muted"],
                tickfont=dict(color=c["muted"], size=11), title_font=dict(color=c["muted"], size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if not own_traces:
        fig.update_layout(barcornerradius=6, bargap=0.3)
        for trace in fig.data:
            if getattr(trace, "type", None) == "scatter":
                if trace.line.width is None:
                    trace.line.width = 2.5
                if trace.marker.size is None:
                    trace.marker.size = 7
    return fig


def plot(fig: go.Figure, height: int = 300, showlegend: bool = False,
         on_select_key: str | None = None, restyle_traces: bool = True,
         colorway: list[str] | None = None, finish=None):
    """Render a styled chart. Pass on_select_key to make it clickable -- the
    click/select event is then returned (and also lands in
    st.session_state[on_select_key]) instead of nothing.

    restyle_traces=False keeps the figure's own trace colors/modes (Chart
    Studio figures were already styled exactly as the user picked).
    colorway replaces the default data colours for this chart only; finish,
    if given, is called with the styled figure for a page's own last touches."""
    kwargs = {}
    if on_select_key:
        # selection_mode="points" only (not the default points+box+lasso):
        # a plain click on a marker reliably registers as a point selection
        # this way, instead of needing an actual box/lasso drag.
        kwargs = {"on_select": "rerun", "key": on_select_key, "selection_mode": "points"}
    styled = style_chart(fig, height, showlegend, ensure_markers=bool(on_select_key) and restyle_traces,
                         recolor=restyle_traces, colorway=colorway)
    if finish is not None:
        styled = finish(styled) or styled
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

    html('<div class="ds-section-title" style="margin-top:18px;">Evidence</div>')
    render_evidence(data.get("evidence"))

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


_INTRO_CSS = """<style>
.pi { margin: 4px 0 26px; max-width: 760px; }
.pi .pi-lede { font: 400 15.5px/1.65 "Inter", system-ui, sans-serif; color: #b4b0c6; margin: 0; }
.pi details { margin-top: 14px; }
.pi summary { list-style: none; display: inline-flex; align-items: center; gap: 8px; cursor: pointer; font: 500 13.5px/1 "Inter", system-ui, sans-serif; color: #d9d5e8; padding: 7px 0; }
.pi summary::-webkit-details-marker { display: none; }
.pi summary::before { content: ""; width: 7px; height: 7px; border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: rotate(-45deg); transition: transform .15s ease; margin-right: 2px; }
.pi details[open] summary::before { transform: rotate(45deg); }
.pi summary:hover { color: #f3b24f; }
.pi .pi-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0 28px; margin: 10px 0 0; padding: 16px 0 0; border-top: 1px solid rgba(242,239,231,.10); list-style: none; counter-reset: pi; }
.pi .pi-steps li { counter-increment: pi; font: 400 14px/1.55 "Inter", system-ui, sans-serif; color: #b4b0c6; padding: 0 0 14px; }
.pi .pi-steps li::before { content: counter(pi, decimal-leading-zero); display: block; font: 500 12px/1 "JetBrains Mono", ui-monospace, Consolas, monospace; color: #f3b24f; margin-bottom: 8px; letter-spacing: .04em; }
.pi .pi-note { margin: 2px 0 0; font: 400 13px/1.55 "Inter", system-ui, sans-serif; color: #8c88a3; }
</style>"""


def page_intro(purpose: str, steps: list[str] | None = None, tip: str | None = None) -> None:
    """A short plain-language description under a page title, with the steps folded behind "How it works".
    Written for someone seeing the page for the first time. Static text only; it never calls the backend."""
    body = f'<p class="pi-lede">{esc(purpose)}</p>'
    if steps:
        items = "".join(f"<li>{esc(s)}</li>" for s in steps)
        note = f'<p class="pi-note">{esc(tip)}</p>' if tip else ""
        body += f'<details><summary>How it works</summary><ol class="pi-steps">{items}</ol>{note}</details>'
    elif tip:
        body += f'<p class="pi-note" style="margin-top:8px;">{esc(tip)}</p>'
    st.markdown(_INTRO_CSS, unsafe_allow_html=True)
    html(f'<div class="pi">{body}</div>')
