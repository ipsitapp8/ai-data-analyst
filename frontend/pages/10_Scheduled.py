"""Scheduled — questions DataSage re-runs on an interval, alerting the team
only when a KPI moves past its threshold or the Critic can't verify the result."""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, delete_scheduled, list_scheduled, update_scheduled
from style.theme import (
    require_active_team,
    esc,
    html,
    page_header,
    page_setup,
    render_sidebar,
    trend_html,
    verdict_badge,
)

page_setup("Scheduled")
render_sidebar("scheduled")

require_active_team("Scheduled Analyses")

page_header("Scheduled Analyses", "Tracked questions, re-run automatically on the latest data.")

try:
    rows = list_scheduled()
except ApiError as e:
    rows = []
    st.error(f"Backend unreachable: {e}")


def _when(value: str | None) -> str:
    return value.replace("T", " ")[:16] + " UTC" if value else "—"


with st.container(key="flat_scheduled"):
    if not rows:
        html(
            '<div style="padding:34px 22px;text-align:center;">'
            '<div class="ds-row-title" style="margin-bottom:5px;">Nothing tracked yet</div>'
            '<div class="ds-row-meta">Open a dashboard and choose <b>Track this question</b>.</div>'
            "</div>"
        )
    for sa in rows:
        info, actions = st.columns([5, 2.2])
        with info:
            status = "Active" if sa["is_active"] else "Paused"
            summary = sa.get("last_change_summary") or ""
            html(
                f'<div class="ds-row" style="border-top:1px solid var(--border);">'
                f'<div><div class="ds-row-title">{esc(sa["question_text"][:90])}</div>'
                f'<div class="ds-row-meta">{esc(sa["interval"].title())} · alert over '
                f'{sa["change_threshold_pct"]:g}% · last run {esc(_when(sa.get("last_run_at")))} · '
                f'next {esc(_when(sa.get("next_run_at"))) if sa["is_active"] else "paused"}</div>'
                f'<div class="ds-row-meta">Since last run: {trend_html(sa.get("last_trend"))}'
                f'{" · " + esc(summary) if summary else ""}</div></div>'
                f'<div class="ds-row-spacer"></div>'
                f'{verdict_badge(sa.get("last_verdict_state")) if sa.get("last_verdict_state") else ""}'
                f'&nbsp;<span class="ds-row-meta">{status}</span></div>'
            )
        with actions:
            html("<div style='height:10px'></div>")
            b1, b2, b3 = st.columns(3)
            with b1:
                if sa.get("last_question_id") and st.button("Open", key=f"sc_open_{sa['id']}"):
                    st.session_state["active_question_id"] = sa["last_question_id"]
                    st.switch_page("pages/4_Dashboards.py")
            with b2:
                label = "Pause" if sa["is_active"] else "Resume"
                if st.button(label, key=f"sc_toggle_{sa['id']}"):
                    try:
                        update_scheduled(sa["id"], is_active=not sa["is_active"])
                    except ApiError as e:
                        st.error(str(e))
                    st.rerun()
            with b3:
                if st.button("Delete", key=f"sc_del_{sa['id']}"):
                    try:
                        delete_scheduled(sa["id"])
                    except ApiError as e:
                        st.error(str(e))
                    st.rerun()
