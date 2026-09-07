"""Reports — every completed analysis, packaged as a downloadable summary."""
from __future__ import annotations

import json

import streamlit as st

from api_client import ApiError, get_dashboard, list_questions
from style.theme import badge, esc, html, page_header, page_setup, render_sidebar

page_setup("Reports")
render_sidebar("reports")

page_header("Reports", "Every completed analysis, packaged and verified.")

try:
    questions = list_questions()
except ApiError as e:
    questions = []
    st.error(f"Backend unreachable: {e}")

done = [q for q in questions if q.get("has_dashboard")]

DOC_ICON = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.7"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" '
            'stroke-linejoin="round"/><path d="M14 3v5h5M8.5 13h7M8.5 16.5h4.5" '
            'stroke-linecap="round"/></svg>')


def build_report(question: dict) -> str:
    """Plain-text report bundling the narrative, KPIs, and verification verdict."""
    try:
        dash = get_dashboard(question["id"])
    except ApiError as e:
        return f"Could not load dashboard: {e}"

    lines = [
        f"REPORT — {question['text']}",
        "=" * 72,
        f"Analysis ID : {question['id']}",
        f"Status      : {question['status']}",
        f"Generated   : {question['age']}",
        "",
        "KPIs",
        "-" * 72,
    ]
    for k in dash.get("kpis", []):
        lines.append(f"  {k.get('label','')}: {k.get('value','')}")
    lines += ["", "NARRATIVE", "-" * 72, dash.get("narrative", "(none)"), ""]
    lines += ["VERIFICATION", "-" * 72, dash.get("verification_summary", "(none)"), ""]
    charts = dash.get("charts", [])
    if charts:
        lines += ["CHARTS", "-" * 72]
        lines += [f"  - {c['title']} (from step {c['step_index']})" for c in charts]
    return "\n".join(lines)


with st.container(key="flat_reports"):
    html('<div class="ds-card-head"><div class="ds-section-title">All Reports</div></div>')

    if not done:
        html(
            '<div style="padding:34px 22px;text-align:center;">'
            '<div class="ds-row-title" style="margin-bottom:5px;">No reports yet</div>'
            '<div class="ds-row-meta">Reports appear here once an analysis produces '
            "a dashboard.</div></div>"
        )

    for q in done:
        kind = "verified" if q["status"] == "verified" else "warn"
        label = "Verified" if q["status"] == "verified" else "Needs review"
        row, act = st.columns([5, 1.4])
        with row:
            html(
                f'<div class="ds-row">'
                f'<div class="ds-row-icon">{DOC_ICON}</div>'
                f'<div><div class="ds-row-title">{esc(q["text"][:66])}</div>'
                f'<div class="ds-row-meta">Generated {q["age"]} • '
                f'{q.get("kpi_count", 0)} KPIs</div></div>'
                f'<div class="ds-row-spacer"></div>{badge(label, kind)}</div>'
            )
        with act:
            html("<div style='height:14px'></div>")
            st.download_button(
                "Download",
                data=build_report(q),
                file_name=f"report_{q['id']}.txt",
                mime="text/plain",
                key=f"rp_dl_{q['id']}",
            )
