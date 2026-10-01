"""Reports — every completed analysis, packaged as a downloadable summary."""
from __future__ import annotations

import json

import streamlit as st

from api_client import ApiError, get_dashboard, list_questions
from style.theme import esc, html, page_header, page_setup, render_sidebar, verdict_badge, require_active_team

page_setup("Reports")
render_sidebar("reports")
require_active_team("Reports")

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


BANNER_TITLES = {
    "VERIFIED": "FULLY VERIFIED",
    "VERIFIED_WITH_CAVEATS": "VERIFIED WITH CAVEATS - {n} ITEM(S) FLAGGED",
    "UNVERIFIED": "UNVERIFIED - CRITIC COULD NOT CONFIRM THIS ANALYSIS",
}


def verdict_banner_text(dash: dict) -> str:
    """The dashboard's trust banner, in text form, so a shared/downloaded
    report never loses it. Non-verified states include the Critic's reasoning."""
    state = dash.get("verdict_state") or ("VERIFIED" if dash.get("verified") else "UNVERIFIED")
    title = BANNER_TITLES.get(state, state).format(n=dash.get("flagged_count", 0))
    bar = "!" * 72 if state != "VERIFIED" else "=" * 72
    out = [bar, f"  {title}", bar]
    for r in dash.get("rejections", []):
        out.append(f"  Critic's reasoning: {r.get('summary') or 'No summary given.'}")
        out += [f"    - {i}" for i in r.get("issues", [])]
    return "\n".join(out)


def build_report(question: dict) -> str:
    """Plain-text report bundling the narrative, KPIs, and verification verdict."""
    try:
        dash = get_dashboard(question["id"])
    except ApiError as e:
        return f"Could not load dashboard: {e}"

    lines = [verdict_banner_text(dash), "", f"REPORT — {question['text']}", "=" * 72,
        f"Analysis ID : {question['id']}",
        f"Trigger     : {question.get('trigger', 'manual')}",
        f"Status      : {question['status']}",
        f"Generated   : {question['age']}",
        "",
        "KPIs",
        "-" * 72,
    ]
    for k in dash.get("kpis", []):
        flag = "  [FLAGGED BY CRITIC]" if k.get("flagged") else ""
        lines.append(f"  {k.get('label','')}: {k.get('value','')}{flag}")
    narr_flag = "  [FLAGGED BY CRITIC]" if dash.get("narrative_flagged") else ""
    lines += ["", f"NARRATIVE{narr_flag}", "-" * 72, dash.get("narrative", "(none)"), ""]
    lines += ["VERIFICATION", "-" * 72, dash.get("verification_summary", "(none)"), ""]
    charts = dash.get("charts", [])
    if charts:
        lines += ["CHARTS", "-" * 72]
        lines += [
            f"  - {c['title']} (from step {c['step_index']})"
            + ("  [FLAGGED BY CRITIC]" if c.get("flagged") else "")
            for c in charts
        ]
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
        row, act = st.columns([5, 1.4])
        with row:
            html(
                f'<div class="ds-row">'
                f'<div class="ds-row-icon">{DOC_ICON}</div>'
                f'<div><div class="ds-row-title">{esc(q["text"][:66])}</div>'
                f'<div class="ds-row-meta">Generated {q["age"]} • '
                f'{q.get("kpi_count", 0)} KPIs</div></div>'
                f'<div class="ds-row-spacer"></div>{verdict_badge(q.get("verdict_state"))}</div>'
            )
        with act:
            html("<div style='height:14px'></div>")
            cache_key = f"report_text_{q['id']}"
            if cache_key in st.session_state:
                st.download_button(
                    "Download",
                    data=st.session_state[cache_key],
                    file_name=f"report_{q['id']}.txt",
                    mime="text/plain",
                    key=f"rp_dl_{q['id']}",
                )
            else:
                # Deferred until asked for: building a report fetches the full
                # dashboard, and this list can hold many rows -- doing that
                # eagerly for every row on every page load meant N reports
                # cost N dashboard fetches just to render the page at all.
                if st.button("Prepare", key=f"rp_prep_{q['id']}"):
                    st.session_state[cache_key] = build_report(q)
                    st.rerun()
