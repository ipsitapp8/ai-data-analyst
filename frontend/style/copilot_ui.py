"""Silt Copilot's look: a light/dark glass-style purple theme, the typing indicator, and charts.

Kept apart from the page so the pieces can be tested without a running app. Charts are drawn only from data the
backend computed; nothing here invents or transforms values beyond sorting and labelling them.
"""
from __future__ import annotations

from html import escape

import altair as alt
import pandas as pd

THEMES = {
    "light": {
        "bg": "linear-gradient(180deg,#f6f3ff 0%,#fbfaff 45%,#ffffff 100%)",
        "surface": "rgba(255,255,255,0.78)", "surface_solid": "#ffffff", "page": "#ffffff", "border": "rgba(109,79,224,0.16)",
        "text": "#1d1a2e", "muted": "#5d5875", "accent": "#6d4fe0", "accent_soft": "rgba(109,79,224,0.10)",
        "user_bg": "linear-gradient(135deg,#6d4fe0 0%,#8a6cf0 100%)", "user_text": "#ffffff",
        "input_bg": "#ffffff", "input_text": "#1d1a2e", "placeholder": "#8a84a6", "caret": "#6d4fe0",
        "code_bg": "#f1edff", "table_head": "#f4f1ff", "table_border": "#e6e1fa", "grid": "#ebe8f5",
        "palette": ["#6d4fe0", "#0f9488", "#d97706", "#db2777", "#2563eb"],
        "shadow": "0 6px 24px rgba(76,52,170,0.08)", "warn_bg": "#fff7e8", "warn_text": "#7a4a00",
    },
    "dark": {
        "bg": "linear-gradient(180deg,#14111f 0%,#181428 50%,#1b1730 100%)",
        "surface": "rgba(38,32,62,0.72)", "surface_solid": "#221d36", "page": "#1b1730", "border": "rgba(167,139,250,0.22)",
        "text": "#ece9ff", "muted": "#b3abd6", "accent": "#a78bfa", "accent_soft": "rgba(167,139,250,0.14)",
        "user_bg": "linear-gradient(135deg,#5b3fd1 0%,#7a5ae6 100%)", "user_text": "#ffffff",
        "input_bg": "#221d36", "input_text": "#f4f2ff", "placeholder": "#9a93c0", "caret": "#c4b5fd",
        "code_bg": "#2c2546", "table_head": "#2a2343", "table_border": "#3a3260", "grid": "#332c52",
        "palette": ["#9a7cf5", "#0fa392", "#b07a06", "#ec4899", "#4f8ff7"],
        "shadow": "0 6px 24px rgba(0,0,0,0.35)", "warn_bg": "#3a2e12", "warn_text": "#ffd98a",
    },
}


def css(theme: str = "light") -> str:
    t = THEMES.get(theme, THEMES["light"])
    msg = "[data-testid='stChatMessage']"
    rules = f"""
:root {{ --cp-text:{t['text']}; --cp-muted:{t['muted']}; --cp-accent:{t['accent']}; }}
[data-testid='stAppViewContainer'], [data-testid='stMain'], .stMain {{ background:{t['bg']} !important; }}
[data-testid='stHeader'] {{ background:transparent !important; }}
.block-container {{ max-width:1180px !important; }}
.cp-shell {{ max-width:880px; margin:0 auto; }}
.cp-title {{ font-size:1.6rem; font-weight:700; margin:0; color:{t['text']}; letter-spacing:-0.01em; }}
.cp-title span {{ background:linear-gradient(90deg,{t['accent']},#c084fc); -webkit-background-clip:text;
  background-clip:text; color:transparent; }}
.cp-sub {{ color:{t['muted']}; margin:2px 0 10px 0; font-size:.92rem; }}
.cp-pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
  background:{t['accent_soft']}; color:{t['accent']}; border:1px solid {t['border']}; }}
.cp-pill.off {{ background:{t['warn_bg']}; color:{t['warn_text']}; }}
.cp-src {{ color:{t['muted']}; font-size:.78rem; margin-top:8px; }}
.cp-ctx {{ background:{t['surface']}; backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
  border:1px solid {t['border']}; border-radius:14px; padding:8px 14px; margin:4px 0 12px 0; color:{t['text']};
  font-size:.88rem; }}
.cp-empty {{ background:{t['surface']}; backdrop-filter:blur(8px); border:1px solid {t['border']}; border-radius:16px;
  padding:18px 20px; margin:8px 0 14px 0; color:{t['text']}; box-shadow:{t['shadow']}; }}
.cp-file {{ font-size:.82rem; color:{t['muted']}; }}
.cp-file b {{ color:{t['text']}; }}

/* chat bubbles */
{msg} {{ background:{t['surface']} !important; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border:1px solid {t['border']} !important; border-radius:18px !important; padding:14px 16px !important;
  box-shadow:{t['shadow']}; max-width:880px; margin-left:auto; margin-right:auto; }}
{msg} *, [data-testid='stExpander'] * {{ color:{t['text']} !important; }}
{msg}:has([data-testid='stChatMessageAvatarUser']) {{ background:{t['user_bg']} !important; border:none !important; }}
{msg}:has([data-testid='stChatMessageAvatarUser']) * {{ color:{t['user_text']} !important; }}
{msg} p {{ line-height:1.55; }}
{msg} code, {msg} pre, [data-testid='stExpander'] code {{ background:{t['code_bg']} !important; border-radius:6px; }}
{msg} .cp-src, {msg} .cp-src * {{ color:{t['muted']} !important; }}

/* tables: fit the bubble, wrap text, scroll sideways only when they must */
{msg} table, [data-testid='stExpander'] table {{ display:block; max-width:100%; overflow-x:auto; border-collapse:collapse;
  font-size:.86rem; margin:6px 0 10px 0; }}
{msg} th, {msg} td, [data-testid='stExpander'] th, [data-testid='stExpander'] td {{
  border:1px solid {t['table_border']} !important; padding:4px 10px !important; white-space:normal;
  overflow-wrap:break-word; word-break:normal; vertical-align:top; }}
{msg} th, [data-testid='stExpander'] th {{ background:{t['table_head']} !important; font-weight:600; }}
{msg} td[align='right'], {msg} th[align='right'], {msg} td[style*='right'] {{ font-variant-numeric:tabular-nums;
  white-space:nowrap; }}
@media (max-width: 720px) {{
  {msg} table {{ font-size:.78rem; }}
  {msg} th, {msg} td {{ padding:3px 6px !important; }}
  .cp-title {{ font-size:1.3rem; }}
}}

/* expander */
[data-testid='stExpander'] details {{ background:{t['surface_solid']} !important; border:1px solid {t['border']} !important;
  border-radius:12px !important; }}
[data-testid='stExpander'] summary {{ background:transparent !important; }}
.stMain [data-testid='stExpander'] details {{ border-color:{t['border']} !important; }}

/* the message box: readable text, placeholder and caret in both themes */
[data-testid='stBottom'], [data-testid='stBottom'] > div, [data-testid='stBottomBlockContainer'] {{
  background:{t['page']} !important; }}
[data-testid='stChatInput'] {{ background:{t['input_bg']} !important; border:1px solid {t['border']} !important;
  border-radius:16px !important; box-shadow:{t['shadow']}; max-width:880px; margin:0 auto; }}
[data-testid='stChatInput'] > div {{ background:{t['input_bg']} !important; border:none !important;
  box-shadow:none !important; }}
[data-testid='stChatInput'] textarea, [data-testid='stChatInputTextArea'] {{ color:{t['input_text']} !important;
  -webkit-text-fill-color:{t['input_text']} !important; caret-color:{t['caret']} !important;
  background:{t['input_bg']} !important; }}
[data-testid='stChatInput'] textarea::placeholder {{ color:{t['placeholder']} !important;
  -webkit-text-fill-color:{t['placeholder']} !important; opacity:1; }}
[data-testid='stChatInput']:focus-within {{ border-color:{t['accent']} !important;
  box-shadow:0 0 0 3px {t['accent_soft']} !important; }}
[data-testid='stChatInputSubmitButton'] {{ background:{t['accent']} !important; border-radius:10px !important;
  opacity:1 !important; }}
[data-testid='stChatInputSubmitButton'] svg {{ fill:#ffffff !important; color:#ffffff !important; }}
[data-testid='stChatInputSubmitButton']:hover {{ filter:brightness(1.08); }}
[data-testid='stChatInputSubmitButton']:disabled {{ background:{t['muted']} !important; opacity:.55 !important; }}
[data-testid='stChatInput']:has(textarea:disabled) {{ opacity:.7; }}

/* buttons and selectors on this page (main area only: the app's sidebar keeps its own look) */
.stMain .stButton button, .stMain [data-testid='stPopover'] button {{
  border-radius:12px !important; border:1px solid {t['border']} !important;
  background:{t['surface_solid']} !important; color:{t['text']} !important; }}
.stMain .stButton button *, .stMain [data-testid='stPopover'] button * {{ color:{t['text']} !important; }}
.stMain .stButton button:hover {{ border-color:{t['accent']} !important; }}
.stMain .stButton button[kind='primary'] {{ background:{t['accent']} !important; border:none !important; }}
.stMain .stButton button[kind='primary'] * {{ color:#ffffff !important; }}
.stMain [data-testid='stWidgetLabel'], .stMain [data-testid='stWidgetLabel'] *,
.stMain label, .stMain [data-testid='stCaptionContainer'],
.stMain [data-testid='stCaptionContainer'] p {{ color:{t['muted']} !important; -webkit-text-fill-color:{t['muted']} !important; }}
.stMain [data-testid='stMarkdownContainer'] p, .stMain [data-testid='stMarkdownContainer'] li {{ color:{t['text']}; }}
[data-testid='stPopoverBody'] {{ background:{t['surface_solid']} !important; color:{t['text']} !important; }}

/* typing indicator */
.cp-typing {{ display:inline-flex; gap:6px; align-items:center; padding:6px 2px; }}
.cp-typing span.dot {{ width:8px; height:8px; border-radius:50%; background:{t['accent']}; opacity:.35;
  animation:cpBlink 1.2s infinite ease-in-out; }}
.cp-typing span.dot:nth-child(2) {{ animation-delay:.2s; }}
.cp-typing span.dot:nth-child(3) {{ animation-delay:.4s; }}
.cp-typing .lbl {{ margin-left:8px; font-size:.84rem; color:{t['muted']} !important; }}
@keyframes cpBlink {{ 0%,80%,100% {{ opacity:.25; transform:translateY(0); }} 40% {{ opacity:1; transform:translateY(-3px); }} }}
@media (prefers-reduced-motion: reduce) {{ .cp-typing span.dot {{ animation:none; opacity:.8; }} }}
"""
    return "<style>" + " ".join(line.strip() for line in rules.splitlines() if line.strip()) + "</style>"


def typing_html(label: str = "Thinking") -> str:
    return (f"<div class='cp-typing' role='status' aria-live='polite' aria-label='{escape(label)}'>"
            "<span class='dot'></span><span class='dot'></span><span class='dot'></span>"
            f"<span class='lbl'>{escape(label)}…</span></div>")


def _base(c: alt.Chart, spec: dict, theme: str, height: int) -> alt.Chart:
    t = THEMES.get(theme, THEMES["light"])
    return (c.properties(title=spec.get("title", ""), height=height, background=t["surface_solid"])
            .configure_axis(labelColor=t["text"], titleColor=t["muted"], gridColor=t["grid"], domainColor=t["grid"],
                            tickColor=t["grid"], labelLimit=220)
            .configure_title(color=t["text"], fontSize=14, anchor="start")
            .configure_legend(labelColor=t["text"], titleColor=t["muted"])
            .configure_view(stroke=None))


def chart(spec: dict, theme: str = "light") -> alt.Chart | None:
    """An Altair chart for a backend chart spec: bar (optionally by series), line, pie, scatter, histogram, avp
    (actual vs predicted). None when there is nothing to draw."""
    df = pd.DataFrame(spec.get("data") or [])
    if df.empty:
        return None
    t = THEMES.get(theme, THEMES["light"])
    pal = t["palette"]
    kind, x, y = spec.get("type", "bar"), spec.get("x"), spec.get("y")
    tip = [alt.Tooltip(c, format=",.2f") if pd.api.types.is_float_dtype(df[c]) else alt.Tooltip(c)
           for c in df.columns if c != "start"]
    if kind == "line":
        c = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=70, filled=True, color=pal[0]), color=pal[0],
                                    strokeWidth=2).encode(
            x=alt.X(f"{x}:N", title=None, sort=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{y}:Q", title=y), tooltip=tip)
        return _base(c, spec, theme, 300)
    if kind == "pie":
        c = alt.Chart(df).mark_arc(innerRadius=60, stroke=t["surface_solid"], strokeWidth=2).encode(
            theta=alt.Theta(f"{y}:Q"), color=alt.Color(f"{x}:N", title=x, sort=None,
                                                       scale=alt.Scale(range=pal + ["#9ca3af"] * 8)),
            tooltip=tip)
        return _base(c, spec, theme, 300)
    if kind in ("scatter", "avp"):
        pts = alt.Chart(df).mark_circle(size=64, opacity=0.7, color=pal[0]).encode(
            x=alt.X(f"{x}:Q", title=x, scale=alt.Scale(zero=False)), y=alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
            tooltip=tip)
        if kind == "avp":
            lo = float(min(df[x].min(), df[y].min()))
            hi = float(max(df[x].max(), df[y].max()))
            diag = alt.Chart(pd.DataFrame({x: [lo, hi], y: [lo, hi]})).mark_line(
                strokeDash=[5, 4], color=t["muted"], strokeWidth=1.5).encode(x=f"{x}:Q", y=f"{y}:Q")
            return _base(diag + pts, spec, theme, 340)
        return _base(pts, spec, theme, 320)
    if kind == "histogram":
        c = alt.Chart(df).mark_bar(color=pal[0], cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(f"{x}:N", title=spec.get("x_title", "range"), sort=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{y}:Q", title=y), tooltip=tip)
        return _base(c, spec, theme, 280)
    series = spec.get("series")
    height = max(170, min(80 + 40 * df[x].nunique() * (2 if series else 1), 460))
    enc = {"y": alt.Y(f"{x}:N", title=None, sort=None if series else ("-x" if len(df) > 2 else None),
                      axis=alt.Axis(labelOverlap=False)),
           "x": alt.X(f"{y}:Q", title=y), "tooltip": tip}
    if series:
        enc["yOffset"] = alt.YOffset(f"{series}:N", sort=None)
        enc["color"] = alt.Color(f"{series}:N", title=None, sort=None, scale=alt.Scale(range=pal),
                                 legend=alt.Legend(orient="top"))
        c = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(**enc)
    else:
        c = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=pal[0]).encode(**enc)
    return _base(c, spec, theme, height)


def file_kind(name: str) -> str:
    n = name.lower()
    return ("PDF" if n.endswith(".pdf") else "Excel" if n.endswith((".xlsx", ".xls")) else "CSV" if n.endswith(".csv")
            else "file")


def human_size(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{max(1, round(n / 1024))} KB"


def unique_labels(labels: dict) -> dict:
    """Option labels that are unique: where two options share a label (the same file uploaded twice), each gets its id
    appended ("sales_report.pdf · 4 pages · #b64ede"), so a choice made in the browser maps back to the right option."""
    seen: dict = {}
    for label in labels.values():
        seen[label] = seen.get(label, 0) + 1
    return {k: (f"{v} · #{str(k)[:6]}" if seen[v] > 1 else v) for k, v in labels.items()}
