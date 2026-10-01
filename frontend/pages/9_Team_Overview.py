"""Team Overview — a chronological timeline of this team's past analyses."""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, list_questions
from style.theme import (
    require_active_team,
    badge,
    esc,
    html,
    page_header,
    page_setup,
    render_sidebar,
    trigger_badge,
    verdict_badge,
)

page_setup("Team Overview")
render_sidebar("team_overview")

require_active_team("Team Overview")

page_header("Team Overview", "Every analysis this team has run, newest first.")

try:
    questions = list_questions()
except ApiError as e:
    questions = []
    st.error(f"Backend unreachable: {e}")

with st.container(key="flat_timeline"):
    if not questions:
        html(
            '<div style="padding:34px 22px;text-align:center;">'
            '<div class="ds-row-title" style="margin-bottom:5px;">No analyses yet</div>'
            '<div class="ds-row-meta">Upload a dataset and ask your first question.</div>'
            "</div>"
        )
    for q in questions:
        if q.get("verdict_state"):
            kind, label = None, None  # rendered via verdict_badge below
        elif q["status"] in ("planning", "executing", "critic", "dashboard", "queued", "running"):
            kind, label = "running", "Running"
        elif q["status"] == "rejected":
            kind, label = "neutral", "Rejected"
        else:
            kind, label = "warn", q["status"].replace("_", " ").title()

        c1, c2 = st.columns([5, 1])
        with c1:
            html(
                f'<div class="ds-row" style="border-top:1px solid var(--border);">'
                f'<div><div class="ds-row-title">{esc(q["text"][:80])}</div>'
                f'<div class="ds-row-meta">{esc(q["age"])} · {q["kpi_count"]} KPI(s)</div></div>'
                f'<div class="ds-row-spacer"></div>{trigger_badge(q.get("trigger"))}&nbsp;'
                f'{verdict_badge(q["verdict_state"]) if q.get("verdict_state") else badge(label, kind)}</div>'
            )
        with c2:
            if q.get("has_dashboard") and st.button("Open", key=f"tov_open_{q['id']}"):
                st.session_state["active_question_id"] = q["id"]
                st.switch_page("pages/4_Dashboards.py")
