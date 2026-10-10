"""Scheduled — questions DataSage re-runs on an interval, alerting the team
only when a KPI moves past its threshold or the Critic can't verify the result."""
from __future__ import annotations

import streamlit as st

from api_client import (ApiError, delete_scheduled, list_alerts, list_scheduled, mark_alert_read,
                        mark_all_alerts_read, update_scheduled)
from style.theme import (
    badge,
    invalidate_alerts_cache,
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

# ---- alert feed: what changed, went unusual, or failed verification ----
try:
    feed = list_alerts(limit=20)
except ApiError:
    feed = {"unread": 0, "alerts": []}

ALERT_KIND = {"change": ("Changed", "warn"), "anomaly": ("Unusual", "running"), "unverified": ("Unverified", "error")}
with st.container(key="card_alerts"):
    a_head, a_btn = st.columns([4, 1])
    with a_head:
        unread_note = f" · {feed['unread']} unread" if feed["unread"] else ""
        html(f'<div class="ds-section-title">Alerts{unread_note}</div>')
    with a_btn:
        if feed["unread"] and st.button("Mark all read", key="alerts_read_all"):
            try:
                mark_all_alerts_read()
            except ApiError as e:
                st.error(str(e))
            invalidate_alerts_cache()
            st.rerun()
    if not feed["alerts"]:
        html('<div class="ds-row-meta" style="padding:8px 0;">No alerts yet. They appear here when a tracked '
             'metric moves past its threshold, behaves unusually against its own history, or fails verification.</div>')
    for al in feed["alerts"]:
        label, kind = ALERT_KIND.get(al["kind"], ("Alert", "neutral"))
        c1, c2 = st.columns([6, 1.4])
        with c1:
            html(f'<div style="padding:8px 0;{"" if al["read"] else "font-weight:600;"}">'
                 f'<div style="display:flex;gap:10px;align-items:center;">{badge(label, kind)}'
                 f'<span class="ds-row-title">{esc(al["title"])}</span></div>'
                 f'<div class="ds-row-meta" style="margin-top:3px;">{esc(al["created_at"].replace("T", " ")[:16])} UTC'
                 f'{" · " + esc(al["detail"].splitlines()[0][:140]) if al["detail"] else ""}</div></div>')
        with c2:
            b1, b2 = st.columns(2)
            with b1:
                if al.get("question_id") and st.button("Open", key=f"al_open_{al['id']}"):
                    if not al["read"]:
                        try:
                            mark_alert_read(al["id"])
                        except ApiError:
                            pass
                        invalidate_alerts_cache()
                    st.session_state["active_question_id"] = al["question_id"]
                    st.switch_page("pages/4_Dashboards.py")
            with b2:
                if not al["read"] and st.button("✓", key=f"al_read_{al['id']}", help="Mark as read"):
                    try:
                        mark_alert_read(al["id"])
                    except ApiError as e:
                        st.error(str(e))
                    invalidate_alerts_cache()
                    st.rerun()
html("<div style='height:16px'></div>")

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
