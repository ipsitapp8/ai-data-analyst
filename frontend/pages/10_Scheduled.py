"""Scheduled — questions DataSage re-runs on an interval, alerting the team
only when a KPI moves past its threshold or the Critic can't verify the result."""
from __future__ import annotations

import streamlit as st

from api_client import (ApiError, delete_scheduled, get_alert, list_alerts, list_scheduled, mark_alert_read,
                        mark_all_alerts_read, update_alert, update_scheduled)
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

ALERT_KIND = {"change": ("Changed", "warn"), "anomaly": ("Unusual", "running"), "unverified": ("Unverified", "error"),
              "data_quality": ("Data quality", "error")}
ALERT_STATUS = {"open": "Open", "acknowledged": "Acknowledged", "resolved": "Resolved"}
DELIVERY = {"sent": "email sent", "failed": "email failed, retrying", "gave_up": "email failed", "skipped": "no email",
            "none": ""}
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
        c1, c2 = st.columns([5, 2.4])
        with c1:
            seen = f' · seen {al["occurrences"]} times' if al.get("occurrences", 1) > 1 else ""
            sent = DELIVERY.get(al.get("delivery_state", "none"), "")
            html(f'<div style="padding:8px 0;{"" if al["read"] else "font-weight:600;"}">'
                 f'<div style="display:flex;gap:10px;align-items:center;">{badge(label, kind)}'
                 f'<span class="ds-row-title">{esc(al["title"])}</span></div>'
                 f'<div class="ds-row-meta" style="margin-top:3px;">{esc(al["created_at"].replace("T", " ")[:16])} UTC'
                 f' · {ALERT_STATUS.get(al.get("status", "open"), "Open")}{seen}{" · " + sent if sent else ""}'
                 f'{" · " + esc(al["detail"].splitlines()[0][:120]) if al["detail"] else ""}</div></div>')
            with st.popover("Why this alert", use_container_width=False):
                try:
                    ex = get_alert(al["id"]).get("explanation") or {}
                except ApiError as e:
                    ex = {}
                    st.caption(str(e))
                if not ex:
                    st.caption("No explanation was recorded for this alert.")
                else:
                    st.markdown(f"Compared with **{ex.get('compared_with', 'the previous run')}**, threshold "
                                f"**{ex.get('threshold_pct', 0):g}%**"
                                + (f", minimum change **{ex['min_effect_abs']:g}**" if ex.get("min_effect_abs") else "")
                                + f". Verdict of the run: **{str(ex.get('verdict', '')).replace('_', ' ').lower()}**.")
                    if ex.get("changes"):
                        st.dataframe([{"KPI": c["label"], "Before": c["old"], "Now": c["new"],
                                       "Change %": None if c["pct"] is None else round(c["pct"], 1),
                                       "Counts as a change": c["crossed"]} for c in ex["changes"]],
                                     use_container_width=True, hide_index=True)
                    if ex.get("history"):
                        st.dataframe([{"KPI": k, "Earlier runs": v["runs"], "Mean": round(v["mean"], 2),
                                       "Std dev": round(v["stdev"], 2), "Min": v["min"], "Max": v["max"]}
                                      for k, v in ex["history"].items()], use_container_width=True, hide_index=True)
                    for d in ex.get("data_quality") or []:
                        st.warning(d.get("message", ""))
        with c2:
            b1, b2, b3, b4 = st.columns(4)
            with b3:
                if al.get("status", "open") == "open" and st.button("Ack", key=f"al_ack_{al['id']}",
                                                                    help="Acknowledge: someone is looking at it"):
                    try:
                        update_alert(al["id"], "acknowledge")
                    except ApiError as e:
                        st.error(str(e))
                    invalidate_alerts_cache()
                    st.rerun()
            with b4:
                if al.get("status", "open") != "resolved" and st.button("Resolve", key=f"al_res_{al['id']}",
                                                                        help="Close this incident"):
                    try:
                        update_alert(al["id"], "resolve")
                    except ApiError as e:
                        st.error(str(e))
                    invalidate_alerts_cache()
                    st.rerun()
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
                f'{sa["change_threshold_pct"]:g}% vs {esc(str(sa.get("comparison", "previous")).replace("_", " "))}'
                f'{" · min change " + format(sa["min_effect_abs"], "g") if sa.get("min_effect_abs") else ""}'
                f'{"" if sa.get("suppress_on_dq", True) else " · data-quality gate off"}'
                f'{"" if sa.get("notify_email", True) else " · email off"}'
                f' · last run {esc(_when(sa.get("last_run_at")))} · '
                f'next {esc(_when(sa.get("next_run_at"))) if sa["is_active"] else "paused"}</div>'
                f'<div class="ds-row-meta">Since last run: {trend_html(sa.get("last_trend"))}'
                f'{" · " + esc(summary) if summary else ""}</div></div>'
                f'<div class="ds-row-spacer"></div>'
                f'{verdict_badge(sa.get("last_verdict_state")) if sa.get("last_verdict_state") else ""}'
                f'&nbsp;<span class="ds-row-meta">{status}</span></div>'
            )
        with actions:
            html("<div style='height:10px'></div>")
            with st.popover("Monitoring settings", use_container_width=True):
                modes = ["previous", "same_weekday", "rolling_mean"]
                mode = st.selectbox("Compare each run with", modes, index=modes.index(sa.get("comparison", "previous")),
                                    format_func=lambda m: {"previous": "The previous run",
                                                           "same_weekday": "The latest run on the same weekday",
                                                           "rolling_mean": "The mean of recent runs"}[m],
                                    key=f"sc_mode_{sa['id']}")
                thr = st.number_input("Alert when a KPI changes by more than (%)", min_value=0.0,
                                      value=float(sa["change_threshold_pct"]), key=f"sc_thr_{sa['id']}")
                min_abs = st.number_input("…and by at least this much in the KPI's own units (0 = no minimum)",
                                          min_value=0.0, value=float(sa.get("min_effect_abs") or 0.0), key=f"sc_min_{sa['id']}")
                window = st.number_input("Runs in the mean (for the mean comparison)", min_value=2, max_value=30,
                                         value=int(sa.get("window_runs") or 4), key=f"sc_win_{sa['id']}")
                gate = st.checkbox("Hold KPI alerts while the data has open quality problems",
                                   value=bool(sa.get("suppress_on_dq", True)), key=f"sc_gate_{sa['id']}")
                mail = st.checkbox("Send email (in-app alerts are always recorded)",
                                   value=bool(sa.get("notify_email", True)), key=f"sc_mail_{sa['id']}")
                if st.button("Save", key=f"sc_save_{sa['id']}", type="primary"):
                    fields = {"comparison": mode, "change_threshold_pct": thr, "min_effect_abs": min_abs, "window_runs": int(window)}
                    if gate != bool(sa.get("suppress_on_dq", True)):
                        fields["suppress_on_dq"] = gate   # owner/admin only; only sent when changed
                    if mail != bool(sa.get("notify_email", True)):
                        fields["notify_email"] = mail
                    try:
                        update_scheduled(sa["id"], **fields)
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))
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
