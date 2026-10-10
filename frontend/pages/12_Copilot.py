"""Copilot — a multi-turn analytical session over one dataset. Every answer is
recomputed from the data; the conversation only records the decisions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import (ApiError, create_copilot_session, delete_copilot_session, get_copilot_session,
                        list_copilot_sessions, list_datasets, send_copilot_message)
from style.theme import (badge, esc, figure_from_json, html, page_header, page_setup, plot, render_sidebar,
                         require_active_team)

page_setup("Copilot")
render_sidebar("copilot")
require_active_team("Copilot")
page_header("Analytical copilot", "Ask, filter and drill down step by step. Each answer is computed from the data.")

try:
    datasets, sessions = list_datasets(), list_copilot_sessions()
except ApiError as e:
    st.error(f"Backend unreachable: {e}")
    st.stop()

if not datasets:
    st.info("No dataset yet — upload a CSV first.")
    if st.button("Go to Datasets  →", type="primary", key="cp_go_ds"):
        st.switch_page("pages/2_Datasets.py")
    st.stop()

left, right = st.columns([1.2, 3.2], gap="large")

with left:
    html('<div class="ds-section-title">Sessions</div>')
    with st.popover("New session", use_container_width=True):
        ds = st.selectbox("Dataset", datasets, format_func=lambda d: f"{d['filename']} · v{d.get('version', 1)}", key="cp_new_ds")
        title = st.text_input("Name (optional)", key="cp_new_title", max_chars=120)
        if st.button("Start", type="primary", key="cp_new_go"):
            try:
                st.session_state["copilot_session_id"] = create_copilot_session(ds["id"], title)["id"]
                st.rerun()
            except ApiError as e:
                st.error(str(e))
    if not sessions:
        html('<div class="ds-row-meta" style="padding:10px 0;">No sessions yet. Start one to explore a dataset '
             "conversationally. Sessions are private to you.</div>")
    for s in sessions:
        active = st.session_state.get("copilot_session_id") == s["id"]
        if st.button(("● " if active else "") + s["title"][:38], key=f"cp_open_{s['id']}", use_container_width=True):
            st.session_state["copilot_session_id"] = s["id"]
            st.rerun()

sid = st.session_state.get("copilot_session_id")
if sid and not any(s["id"] == sid for s in sessions):
    sid = None
    st.session_state.pop("copilot_session_id", None)

with right:
    if not sid:
        html('<div style="padding:40px 10px;text-align:center;"><div class="ds-row-title">Pick or start a session</div>'
             '<div class="ds-row-meta" style="margin-top:6px;">Try: “total revenue by region”, then “only West”, '
             "“in 2025”, “drill into product”, “overall”, “reset”.</div></div>")
        st.stop()
    try:
        session = get_copilot_session(sid)
    except ApiError as e:
        st.error(str(e))
        st.stop()

    state = session["state"]
    head_l, head_r = st.columns([5, 1])
    with head_l:
        chips = [badge(state.get("metric_label") or "No metric selected", "verified" if state.get("formula") else "neutral")]
        chips += [badge(f'{f["column"]} {f["op"]} {f["value"]}', "running") for f in state.get("filters") or []]
        chips += [badge(f"by {g}", "warn") for g in state.get("group_by") or []]
        html(f'<div class="ds-row-title">{esc(session["title"])}</div>'
             f'<div class="ds-row-meta" style="margin:4px 0 8px 0;">{esc(session.get("dataset_filename") or "dataset removed")} '
             f'· version {session.get("dataset_version")}</div>'
             f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{"".join(chips)}</div>')
    with head_r:
        if st.button("Delete", key="cp_delete"):
            try:
                delete_copilot_session(sid)
            except ApiError as e:
                st.error(str(e))
            st.session_state.pop("copilot_session_id", None)
            st.rerun()
    if session.get("stale"):
        st.warning(f"A newer version of this dataset exists (version {session['latest_version']}). This session is "
                   f"still on version {session['dataset_version']}. Send “use latest data” to switch.")
    if not session.get("dataset_available", True):
        st.error("The dataset this session was built on is no longer available.")
    html("<div style='height:8px'></div>")

    turns = session.get("turns") or []
    if not turns:
        html('<div class="ds-row-meta" style="padding:14px 0;">No messages yet. Start with a metric, for example '
             "“total revenue by region”.</div>")
    for t in turns:
        with st.chat_message("user" if t["role"] == "user" else "assistant"):
            st.markdown(t["content"])
            action, result = t.get("action") or {}, t.get("result")
            if action.get("status") == "clarification":
                for i, opt in enumerate(action.get("options") or []):
                    if st.button(opt["label"], key=f"cp_opt_{t['seq']}_{i}"):
                        try:
                            send_copilot_message(sid, opt["message"])
                        except ApiError as e:
                            st.error(str(e))
                        st.rerun()
            if action.get("suggested_question"):
                if st.button("Run this as a full analysis  →", key=f"cp_full_{t['seq']}"):
                    st.session_state["prefill_question"] = action["suggested_question"]
                    st.session_state["active_dataset_id"] = session["dataset_id"]
                    st.session_state.pop("active_question_id", None)
                    st.switch_page("pages/3_Analyses.py")
            if result:
                if result["type"] == "table":
                    plot(figure_from_json(result["chart"]), height=280)
                    st.dataframe(pd.DataFrame(result["rows_table"]), use_container_width=True, hide_index=True,
                                 height=min(300, 38 + 35 * len(result["rows_table"])))
                ev = t.get("evidence") or {}
                if ev:
                    check = {"agrees": "an independent recomputation agrees", "disagrees": "an independent recomputation DISAGREES",
                             "not_comparable": "independent recomputation not comparable for this shape",
                             "not_run": "independent recomputation could not run"}.get(ev.get("reference_check"), "")
                    filters = "; ".join(f'{f["column"]} {f["op"]} {f["value"]}' for f in ev.get("filters") or []) or "none"
                    st.caption(f"Evidence — formula {ev.get('formula')} · filters: {filters} · {ev.get('rows_used', 0):,} rows · "
                               f"dataset version {ev.get('dataset_version')} ({(ev.get('dataset_fingerprint') or '')[:10]}…) · {check}")

    message = st.chat_input("Ask about this dataset, add a filter, or drill down", max_chars=500)
    if message:
        try:
            send_copilot_message(sid, message)
        except ApiError as e:
            st.error(f"Could not send: {e}")
        else:
            st.rerun()
