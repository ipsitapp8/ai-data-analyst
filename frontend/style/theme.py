"""DataSage design system: global CSS + shared UI components.

Near-black surfaces, sage-green primary accent, rose secondary accent.
Every page imports from here so the app reads as one product.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from auth import require_password

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Chart colorway: green first (positive/primary), rose second (negative/secondary),
# then muted supporting tones. Matches the dashboard mockups.
CHART_COLORWAY = ["#a9d4a4", "#ef8296", "#c9b6e8", "#e8d5a8", "#8fb8d9", "#8b8b88"]
GREEN = "#a9d4a4"
ROSE = "#ef8296"

BASE_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --bg-base: #0b0b0b;
  --bg-page: #0f0f0f;
  --bg-card: #151515;
  --bg-card-hover: #1b1b1b;
  --bg-inset: #101010;
  --border: rgba(255,255,255,0.07);
  --border-strong: rgba(255,255,255,0.12);

  --text-primary: #f2f2f0;
  --text-secondary: #8b8b88;
  --text-muted: #63635f;

  --green: #a9d4a4;
  --green-dim: #7fae7a;
  --green-bg: rgba(169,212,164,0.10);
  --green-border: rgba(169,212,164,0.28);

  --rose: #ef8296;
  --rose-dim: #d06a7d;
  --rose-bg: rgba(239,130,150,0.10);

  --radius-lg: 14px;
  --radius-md: 10px;
  --radius-sm: 7px;

  --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'Consolas', monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg-page);
  color: var(--text-primary);
  font-family: var(--font);
}

[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }

.block-container {
  padding-top: 2.4rem;
  padding-bottom: 4rem;
  max-width: 1400px;
}

h1,h2,h3,h4,h5 { font-family: var(--font); color: var(--text-primary); letter-spacing: -0.01em; }
p, span, div, label, li { font-family: var(--font); }

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
  background: var(--bg-base);
  border-right: 1px solid var(--border);
  width: 260px !important;
}
/* The landing page collapses the sidebar, and Streamlit persists that state
   across page switches -- so workspace pages must force it back open.
   (Home re-hides it with display:none, which is injected after this rule.) */
[data-testid="stSidebar"][aria-expanded="false"] {
  transform: none !important;
  width: 260px !important;
  min-width: 260px !important;
  visibility: visible !important;
}
[data-testid="stSidebar"] > div { padding-top: 0; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.15rem; }

.ds-brand {
  display: flex; align-items: center; gap: 11px;
  padding: 22px 6px 26px 6px;
}
.ds-brand-mark {
  width: 34px; height: 34px; border-radius: 9px;
  background: var(--bg-card); border: 1px solid var(--border-strong);
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.ds-brand-name { font-size: 0.97rem; font-weight: 700; color: var(--text-primary); line-height: 1.15; }
.ds-brand-sub  { font-size: 0.72rem; color: var(--text-secondary); line-height: 1.3; }

/* sidebar nav rows */
.st-key-nav .stButton > button {
  width: 100%;
  display: flex !important;
  justify-content: flex-start !important;
  align-items: center;
  gap: 12px;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  padding: 10px 13px !important;
  margin: 1px 0 !important;
  box-shadow: none !important;
  transition: background 0.12s ease, color 0.12s ease;
}
.st-key-nav .stButton > button:hover {
  background: var(--bg-card) !important;
  color: var(--text-primary) !important;
}
.st-key-nav .stButton > button:disabled {
  background: var(--green-bg) !important;
  border: 1px solid var(--green-border) !important;
  color: var(--green) !important;
  font-weight: 600 !important;
  opacity: 1 !important;
  cursor: default !important;
}
.st-key-nav .stButton > button [data-testid="stIconMaterial"] { font-size: 19px !important; }

.ds-workspace {
  display: flex; align-items: center; gap: 11px;
  padding: 12px; margin-top: 14px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-md);
}
.ds-workspace-badge {
  width: 30px; height: 30px; border-radius: 7px; flex-shrink: 0;
  background: var(--green-bg); border: 1px solid var(--green-border);
  color: var(--green); font-size: 0.7rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.ds-workspace-label { font-size: 0.68rem; color: var(--text-secondary); line-height: 1.2; }
.ds-workspace-name  { font-size: 0.86rem; color: var(--text-primary); font-weight: 600; line-height: 1.3; }

/* ============ BUTTONS ============ */
/* Not scoped as `.stButton > button` -- a button with help= gets wrapped in an
   extra tooltip element, which breaks the direct-child selector and leaves it
   painted in Streamlit's stock red. */
button[kind="primary"], .stFormSubmitButton > button {
  background: var(--green) !important;
  color: #0d1a0c !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 0.58em 1.25em !important;
  box-shadow: none !important;
  transition: filter 0.12s ease;
}
button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  filter: brightness(1.08);
  color: #0d1a0c !important;
}
button[kind="secondary"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-md) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 0.55em 1.1em !important;
  box-shadow: none !important;
}
button[kind="secondary"]:hover {
  background: var(--bg-card-hover) !important;
  border-color: var(--border-strong) !important;
  color: var(--text-primary) !important;
}

/* ============ CARDS ============ */
.ds-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 22px;
}
[class*="st-key-card"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 20px 22px !important;
}
[class*="st-key-flat"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 0 !important;
  overflow: hidden;
}

.ds-page-title { font-size: 1.7rem; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.02em; }
.ds-page-sub   { color: var(--text-secondary); font-size: 0.93rem; margin-bottom: 26px; }
.ds-section-title { font-size: 1.02rem; font-weight: 600; margin: 0; }

/* stat cards */
.ds-stat-label { color: var(--text-secondary); font-size: 0.83rem; margin-bottom: 9px; }
.ds-stat-value { font-size: 2.05rem; font-weight: 700; letter-spacing: -0.025em; line-height: 1; }
.ds-stat-delta { font-size: 0.79rem; margin-top: 10px; display: flex; align-items: center; gap: 4px; }
.ds-up   { color: var(--green); }
.ds-down { color: var(--rose); }

/* list rows */
.ds-row {
  display: flex; align-items: center; gap: 14px;
  padding: 15px 22px; border-top: 1px solid var(--border);
}
.ds-row:hover { background: var(--bg-card-hover); }
.ds-row-icon {
  width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
  background: var(--bg-inset); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  color: var(--text-secondary);
}
.ds-row-title { font-size: 0.93rem; font-weight: 600; color: var(--text-primary); line-height: 1.35; }
.ds-row-meta  { font-size: 0.79rem; color: var(--text-secondary); line-height: 1.35; }
.ds-row-spacer { flex: 1; }

.ds-card-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 22px;
}

/* badges */
.ds-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px;
  font-size: 0.77rem; font-weight: 600; white-space: nowrap;
}
.ds-badge-verified { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
.ds-badge-warn     { background: rgba(232,213,168,0.10); color: #e8d5a8; border: 1px solid rgba(232,213,168,0.28); }
.ds-badge-error    { background: var(--rose-bg); color: var(--rose); border: 1px solid rgba(239,130,150,0.28); }
.ds-badge-neutral  { background: var(--bg-inset); color: var(--text-secondary); border: 1px solid var(--border); }
.ds-badge-running  { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
.ds-badge-running::before {
  content:''; width:7px; height:7px; border-radius:50%; background: var(--green);
  animation: dspulse 1.4s infinite ease-in-out;
}
@keyframes dspulse { 0%,100%{opacity:1} 50%{opacity:.35} }

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
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stFileUploaderDropzone"] { border-style: dashed !important; }

/* tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 4px; background: transparent; border-bottom: 1px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent; color: var(--text-secondary);
  font-size: 0.9rem; font-weight: 500; padding: 10px 14px;
}
[data-testid="stTabs"] [aria-selected="true"] { color: var(--text-primary) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--green) !important; }

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

/* tables (st.table -- real HTML, unlike canvas-based st.dataframe) */
[data-testid="stTable"] table { background: transparent !important; border-collapse: collapse; width: 100%; }
[data-testid="stTable"] th {
  background: transparent !important; color: var(--text-secondary) !important;
  font-size: 0.79rem; font-weight: 500; text-transform: none;
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
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* quality bar (datasets table) */
.ds-quality { display: flex; align-items: center; gap: 10px; }
/* inline-block/block are required: these are <span>s, and width/height have no
   effect on inline boxes, so the fill would collapse to nothing */
.ds-quality-track {
  display: inline-block; width: 92px; height: 5px; border-radius: 99px;
  background: var(--bg-inset); overflow: hidden;
}
.ds-quality-fill { display: block; height: 100%; border-radius: 99px; background: var(--green); }

/* progress bar */
.ds-progress-track { width: 100%; height: 6px; border-radius: 99px; background: var(--bg-inset); overflow: hidden; }
.ds-progress-fill  { height: 100%; border-radius: 99px; background: var(--green); transition: width .3s ease; }

/* step list */
.ds-step { display: flex; gap: 13px; padding: 13px 16px; border-radius: var(--radius-md); align-items: flex-start; }
.ds-step-active { background: var(--rose-bg); border: 1px solid rgba(239,130,150,0.25); }
.ds-step-num {
  width: 24px; height: 24px; border-radius: 7px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 600;
  background: var(--bg-inset); border: 1px solid var(--border); color: var(--text-secondary);
}
.ds-step-done   { background: var(--green-bg); border-color: var(--green-border); color: var(--green); }
.ds-step-run    { background: var(--rose-bg); border-color: rgba(239,130,150,0.3); color: var(--rose); }
.ds-step-title  { font-size: 0.89rem; font-weight: 600; color: var(--text-primary); line-height: 1.35; }
.ds-step-status { font-size: 0.77rem; color: var(--text-secondary); line-height: 1.35; }

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
        page_title=f"{title} · DataSage",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded" if sidebar else "collapsed",
    )
    # Gate before anything else renders. No-op unless APP_PASSWORD is set, so
    # this changes nothing for local development.
    require_password()
    inject_base_css()
    if not sidebar:
        st.markdown(
            "<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
            "[data-testid='stExpandSidebarButton']{display:none !important;}"
            ".block-container{max-width:100% !important;padding:0 !important;}</style>",
            unsafe_allow_html=True,
        )


@st.cache_data
def asset_data_uri(filename: str, mime: str = "image/jpeg") -> str:
    """Base64 data URI for a file in frontend/assets (CSS can't reach local paths)."""
    data = (ASSETS_DIR / filename).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


HEX_LOGO = """
<svg width="17" height="17" viewBox="0 0 24 24" fill="none">
  <path d="M12 2.5 20.5 7.2v9.6L12 21.5 3.5 16.8V7.2z" stroke="#a9d4a4" stroke-width="1.7"
        stroke-linejoin="round"/>
</svg>
"""

NAV_PAGES = [
    ("overview", "Overview", "home", "pages/1_Overview.py"),
    ("datasets", "Datasets", "database", "pages/2_Datasets.py"),
    ("analyses", "Analyses", "monitoring", "pages/3_Analyses.py"),
    ("dashboards", "Dashboards", "dashboard", "pages/4_Dashboards.py"),
    ("reports", "Reports", "description", "pages/5_Reports.py"),
    ("audit", "Audit Trail", "receipt_long", "pages/6_Audit_Trail.py"),
    ("settings", "Settings", "settings", "pages/7_Settings.py"),
]


def render_sidebar(current: str) -> None:
    """Brand mark, nav rows, and the workspace switcher pinned at the bottom."""
    with st.sidebar:
        html(
            f"""
            <div class="ds-brand">
              <div class="ds-brand-mark">{HEX_LOGO}</div>
              <div>
                <div class="ds-brand-name">DataSage</div>
                <div class="ds-brand-sub">AI Data Analyst</div>
              </div>
            </div>
            """
        )
        with st.container(key="nav"):
            for key, label, icon, target in NAV_PAGES:
                if st.button(label, key=f"nav_{key}", icon=f":material/{icon}:",
                             disabled=(key == current)):
                    st.switch_page(target)

        html(
            """
            <div class="ds-workspace">
              <div class="ds-workspace-badge">GT</div>
              <div>
                <div class="ds-workspace-label">Workspace</div>
                <div class="ds-workspace-name">Growth Team</div>
              </div>
            </div>
            """
        )


def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="ds-page-sub">{subtitle}</div>' if subtitle else ""
    html(f'<div class="ds-page-title">{title}</div>{sub}')


def badge(text: str, kind: str = "neutral") -> str:
    check = (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none">'
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


def style_chart(fig: go.Figure, height: int = 300, showlegend: bool = False) -> go.Figure:
    """Force any figure -- including sandbox-generated ones -- into the DataSage look."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b8b88", family="Inter, sans-serif", size=12),
        colorway=CHART_COLORWAY,
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=showlegend,
        legend=dict(font=dict(color="#8b8b88"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1b1b1b", bordercolor="rgba(255,255,255,0.12)",
                        font=dict(color="#f2f2f0", family="Inter")),
        # empty string, not None -- None leaves the title node in place and
        # Plotly renders a literal "undefined" tspan above the plot
        title=dict(text=""),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                     color="#63635f", showline=False, ticks="")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                     color="#63635f", showline=False, ticks="")
    _recolor_traces(fig)
    return fig


def _recolor_traces(fig: go.Figure) -> None:
    """Repaint solid trace colors from CHART_COLORWAY.

    `colorway` only supplies colors Plotly would otherwise auto-assign. Plotly
    Express bakes explicit per-trace colors whenever the sandbox code groups by
    a column, which would otherwise leave stock blue/red charts sitting inside
    the DataSage palette. Array-valued colors (continuous scales) are left
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


def plot(fig: go.Figure, height: int = 300, showlegend: bool = False) -> None:
    st.plotly_chart(style_chart(fig, height, showlegend), use_container_width=True,
                    config={"displayModeBar": False})
