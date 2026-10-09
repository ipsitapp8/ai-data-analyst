"""Dashboards — verified KPI cards, agent-generated charts, and the narrative."""
from __future__ import annotations

import streamlit as st

from api_client import (ApiError, ask_question, chat_dashboard, create_scheduled, create_share, get_dashboard,
                        get_status, list_questions, list_shares, revoke_share)
from style.theme import (
    require_active_team,
    esc,
    figure_from_json,
    html,
    open_inspect,
    page_header,
    page_setup,
    plot,
    render_inspect_dialog_if_open,
    render_sidebar,
    render_verdict_banner,
    verdict_badge,
)

page_setup("Dashboards")
render_sidebar("dashboards")
require_active_team("Dashboards")

# Deep link from alert emails: /Dashboards?question=<id>
_linked = st.query_params.get("question")
if _linked and str(_linked).isdigit():
    st.session_state["active_question_id"] = int(_linked)
    st.query_params.clear()

qid = st.session_state.get("active_question_id")

# ---- picker when nothing is selected ----
try:
    questions = list_questions()
except ApiError as e:
    questions = []
    st.error(f"Backend unreachable: {e}")

ready = [q for q in questions if q.get("has_dashboard")]

if not qid:
    page_header("Dashboards", "Generated from verified analyses.")
    if not ready:
        st.info("No dashboards yet — run an analysis first.")
        if st.button("Go to Analyses  →", type="primary", key="db_go"):
            st.switch_page("pages/3_Analyses.py")
        st.stop()
    with st.container(key="flat_pick"):
        html('<div class="ds-card-head"><div class="ds-section-title">'
                    'All Dashboards</div></div>')
        for q in ready:
            c1, c2 = st.columns([5, 1])
            with c1:
                html(
                    f'<div class="ds-row" style="border-top:none;">'
                    f'<div><div class="ds-row-title">{esc(q["text"][:70])}</div>'
                    f'<div class="ds-row-meta">Updated {q["age"]}</div></div>'
                    f'<div class="ds-row-spacer"></div>{verdict_badge(q.get("verdict_state"))}</div>'
                )
            with c2:
                if st.button("Open", key=f"db_open_{q['id']}"):
                    st.session_state["active_question_id"] = q["id"]
                    st.rerun()
    st.stop()

# ---- selected dashboard ----
try:
    status = get_status(qid)
except ApiError as e:
    st.error(f"Could not fetch status: {e}")
    st.stop()

if status["status"] not in ("verified", "unverified"):
    page_header("Dashboards")
    st.warning(f"This analysis isn't finished yet (status: **{status['status']}**).")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Watch it run  →", type="primary", key="db_watch"):
            st.switch_page("pages/3_Analyses.py")
    with c2:
        if st.button("←  All dashboards", key="db_back_pending"):
            st.session_state.pop("active_question_id", None)
            st.rerun()
    st.stop()

try:
    dash = get_dashboard(qid)
except ApiError as e:
    st.error(f"Could not load dashboard: {e}")
    st.stop()

if st.button("←  All dashboards", key="db_back"):
    st.session_state.pop("active_question_id", None)
    st.rerun()
html("<div style='height:4px'></div>")

verdict_state = dash.get("verdict_state")
head_l, head_r = st.columns([3, 1])
with head_l:
    html(
        f'<div class="ds-page-title">{esc(status.get("question_text", "Analysis"))}</div>'
    )
    html(
        f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:24px;">'
        f'<span class="ds-row-meta">Analysis #{qid}</span>'
        f'{verdict_badge(verdict_state)}'
        + (f'<span class="ds-row-meta">Data v{dash["dataset_version"]}</span>' if dash.get("dataset_version") else "")
        + "</div>"
    )
with head_r:
    html("<div style='height:12px'></div>")
    if st.button("Audit Trail  →", key="db_audit"):
        st.switch_page("pages/6_Audit_Trail.py")
    with st.popover("Share", use_container_width=True):
        if verdict_state == "UNVERIFIED":
            st.caption("The Critic could not verify this analysis, so it can't be shared publicly.")
        else:
            st.caption("Anyone with the link can view a read-only copy. No login needed.")
            days = st.selectbox("Link expires after", [1, 7, 30, 90], index=1,
                                format_func=lambda d: f"{d} day{'s' if d != 1 else ''}", key="share_days")
            if st.button("Create link", type="primary", key="share_create"):
                try:
                    st.session_state["_new_share_url"] = create_share(qid, days)["url"]
                except ApiError as e:
                    st.error(f"Could not create link: {e}")
            if st.session_state.get("_new_share_url"):
                st.code(st.session_state["_new_share_url"], language=None)
            try:
                active_links = list_shares(qid)
            except ApiError:
                active_links = []
            for link in active_links:
                lc1, lc2 = st.columns([3, 1])
                with lc1:
                    st.caption(f"Expires {link['expires_at'][:10]}")
                with lc2:
                    if st.button("Revoke", key=f"share_rev_{link['id']}"):
                        try:
                            revoke_share(link["id"])
                            st.session_state.pop("_new_share_url", None)
                            st.rerun()
                        except ApiError as e:
                            st.error(str(e))
    with st.popover("Track this question", use_container_width=True):
        interval = st.radio("Re-run", ["daily", "weekly"], horizontal=True, key="track_interval")
        threshold = st.number_input("Alert when a KPI changes by more than (%)", min_value=0.0,
                                    value=10.0, step=1.0, key="track_threshold")
        if st.button("Start tracking", type="primary", key="track_go"):
            src = next((q for q in questions if q["id"] == qid), None)
            if not src:
                st.error("Could not find this question's dataset.")
            else:
                try:
                    create_scheduled(src["dataset_id"], status.get("question_text") or src["text"],
                                     interval, threshold)
                    st.success("Tracking started — see the Scheduled page.")
                except ApiError as e:
                    st.error(f"Could not schedule: {e}")

# The trust signal: first thing on the page, above every KPI.
render_verdict_banner(dash)

dashboard_id = dash["id"]

kpis = dash.get("kpis", [])
if kpis:
    cols = st.columns(min(4, len(kpis)), gap="medium")
    for col, kpi in zip(cols, kpis[:4]):
        with col:
            element_id = kpi.get("element_id")
            flag = "⚠️ " if kpi.get("flagged") else ""
            label = f"{flag}{kpi.get('label', '')}  \n**{kpi.get('value', '')}**"
            with st.container(key=f"kpi_{element_id or kpi.get('label', '')}"):
                if st.button(label, key=f"kpibtn_{element_id or kpi.get('label', '')}",
                             use_container_width=True, disabled=not element_id):
                    open_inspect(dashboard_id, element_id)
                    st.rerun()
    html("<div style='height:22px'></div>")

charts = dash.get("charts", [])
if charts:
    for i in range(0, len(charts), 2):
        pair = charts[i:i + 2]
        cols = st.columns(len(pair), gap="medium")
        for col, chart in zip(cols, pair):
            with col:
                element_id = chart.get("element_id")
                container_key = f"chartcard_{element_id}" if element_id else f"card_ch{i}_{chart['step_index']}"
                with st.container(key=container_key):
                    if chart.get("flagged") and element_id:
                        t_col, f_col = st.columns([5, 1])
                        with t_col:
                            html(f'<div class="ds-section-title">{esc(chart["title"])}</div>')
                        with f_col:
                            if st.button("⚠️", key=f"flag_{element_id}", help="Flagged by the Critic — see why"):
                                open_inspect(dashboard_id, element_id)
                                st.rerun()
                    else:
                        html(f'<div class="ds-section-title">{esc(chart["title"])}</div>')
                    html("<div style='height:8px'></div>")
                    try:
                        fig = figure_from_json(chart["plotly_json"])
                        event = plot(fig, height=300, showlegend=True,
                                     on_select_key=f"chart_{element_id}" if element_id else None)
                        if element_id and event and event.selection and event.selection.get("points"):
                            open_inspect(dashboard_id, element_id)
                            st.rerun()
                    except Exception as e:  # noqa: BLE001 - render one bad chart, not the page
                        html(f'<div class="ds-row-meta">Could not render: {e}</div>')
        html("<div style='height:8px'></div>")

render_inspect_dialog_if_open()

if dash.get("narrative"):
    with st.container(key="card_narr"):
        n_col, nf_col = st.columns([6, 1])
        with n_col:
            html('<div class="ds-section-title">Narrative Summary</div>')
        with nf_col:
            if dash.get("narrative_flagged"):
                if dash.get("narrative_element_id"):
                    if st.button("⚠️", key="flag_narrative", help="Flagged by the Critic — see why"):
                        open_inspect(dashboard_id, dash["narrative_element_id"])
                        st.rerun()
                else:
                    html('<span class="ds-flag" title="Flagged by the Critic">⚠️</span>')
        html(
            f'<div style="font-size:0.97rem;line-height:1.75;color:var(--text-primary);'
            f'margin-top:10px;">{esc(dash["narrative"])}</div>'
        )

if dash.get("verification_summary"):
    html("<div style='height:14px'></div>")
    with st.container(key="card_verif"):
        html(
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'{verdict_badge(verdict_state)}'
            f'<div class="ds-section-title">Verification</div></div>'
            f'<div class="ds-row-meta" style="margin-top:10px;line-height:1.7;">'
            f'{esc(dash["verification_summary"])}</div>'
        )


# ------------------------------------------------------- ask your dashboard --
html("<div style='height:22px'></div>")
chat_key = f"chat_{qid}"
history = st.session_state.setdefault(chat_key, [])
with st.container(key="card_chat"):
    html('<div class="ds-section-title">Ask this dashboard</div>')
    html('<div class="ds-page-sub" style="margin:4px 0 12px 0;">Follow-up questions are answered only from '
         'this analysis, and every answer is fact-checked by a second model. Anything it cannot answer, '
         'you can run as a new, fully verified analysis.</div>')
    for i, turn in enumerate(history):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("verified") is True:
                    html(f'<span class="ds-badge ds-badge-verified">Checked against this analysis</span>')
                elif turn.get("verified") is False:
                    claims = "".join(f"<li>{esc(c)}</li>" for c in turn.get("unsupported_claims") or [])
                    html(f'<span class="ds-badge ds-badge-warn">Not fully supported</span>'
                         f'<div class="ds-row-meta" style="margin-top:6px;">The checker could not confirm:'
                         f'<ul style="margin:4px 0 0 18px;">{claims}</ul></div>')
                elif turn.get("needs_new_analysis") is not True:
                    html('<span class="ds-badge ds-badge-neutral">Not independently checked</span>')
                if turn.get("sources"):
                    st.caption("Based on: " + " · ".join(turn["sources"]))
                if turn.get("needs_new_analysis") and turn.get("suggested_question"):
                    st.caption("This needs a new calculation.")
                    if st.button(f"Run as new analysis: {turn['suggested_question']}", key=f"chat_run_{qid}_{i}"):
                        src = next((q for q in questions if q["id"] == qid), None)
                        if src:
                            try:
                                res = ask_question(src["dataset_id"], turn["suggested_question"])
                                st.session_state["active_question_id"] = res["question_id"]
                                st.session_state["question_running"] = True
                                st.switch_page("pages/3_Analyses.py")
                            except ApiError as e:
                                st.error(f"Could not start analysis: {e}")

prompt = st.chat_input("Ask a follow-up about this analysis…", max_chars=600, key=f"chat_input_{qid}")
if prompt:
    sent = [{"role": t["role"], "content": t["content"]} for t in history][-8:]
    history.append({"role": "user", "content": prompt})
    with st.spinner("Reading the analysis and checking the answer…"):
        try:
            reply = chat_dashboard(qid, prompt, sent)
            history.append({"role": "assistant", "content": reply["answer"], **{
                k: reply.get(k) for k in ("verified", "unsupported_claims", "sources",
                                           "needs_new_analysis", "suggested_question")}})
        except ApiError as e:
            history.append({"role": "assistant", "content": f"Sorry, I couldn't answer that: {e}",
                            "verified": None, "needs_new_analysis": True})
    st.rerun()
