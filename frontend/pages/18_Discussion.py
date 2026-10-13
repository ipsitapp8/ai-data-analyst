"""Discussion — a team-scoped feed of comments and notes, optionally pinned to a
specific analysis or dataset, with a review (open / resolved) state."""
from __future__ import annotations

import streamlit as st

from api_client import (
    ApiError,
    delete_comment,
    list_comments,
    list_datasets,
    list_questions,
    post_comment,
    update_comment,
)
from style.theme import badge, esc, html, page_header, page_setup, render_sidebar, require_active_team

page_setup("Discussion")
render_sidebar("discussion")
require_active_team("Discussion")

page_header("Discussion", "Comments and notes for this team. Pin one to an analysis or dataset to keep the context.")

try:
    questions = list_questions()
except ApiError:
    questions = []
try:
    datasets = list_datasets()
except ApiError:
    datasets = []

targets: dict[str, tuple[str | None, int | None]] = {"General (no attachment)": (None, None)}
for q in questions[:50]:
    targets[f"Analysis · {q['text'][:60]}"] = ("question", q["id"])
for d in datasets[:50]:
    targets[f"Dataset · {d['filename']}"] = ("dataset", d["id"])
target_labels = {(t, i): label for label, (t, i) in targets.items() if t}

with st.container(key="card_compose"):
    html('<div class="ds-section-title">New post</div>')
    with st.form("compose_form", clear_on_submit=True):
        body = st.text_area("Message", label_visibility="collapsed", height=90,
                            placeholder="Share a finding, ask a question, or leave a note for your team…")
        c1, c2, c3 = st.columns([3, 1.4, 1])
        with c1:
            attach = st.selectbox("Attach to", list(targets), label_visibility="collapsed")
        with c2:
            kind = st.selectbox("Type", ["comment", "note"], label_visibility="collapsed")
        with c3:
            sent = st.form_submit_button("Post", type="primary", use_container_width=True)
    if sent:
        if not body.strip():
            st.error("Write something first.")
        else:
            t_type, t_id = targets[attach]
            try:
                post_comment(body.strip(), t_type, t_id, kind)
                st.rerun()
            except ApiError as e:
                st.error(f"Could not post: {e}")

html("<div style='height:12px'></div>")
flt = st.radio("Show", ["All", "Open", "Resolved"], horizontal=True, label_visibility="collapsed")
status = {"All": None, "Open": "open", "Resolved": "resolved"}[flt]

try:
    comments = list_comments(status=status)
except ApiError as e:
    comments = []
    st.error(f"Could not load the discussion: {e}")

if not comments:
    html(
        '<div style="padding:34px 22px;text-align:center;">'
        '<div class="ds-row-title" style="margin-bottom:5px;">Nothing here yet</div>'
        '<div class="ds-row-meta">Start the conversation with the form above.</div></div>'
    )

editing = st.session_state.get("editing_comment")
for c in comments:
    pinned = target_labels.get((c["target_type"], c["target_id"]))
    meta = f'{esc(c["author_name"])} · {esc((c["created_at"] or "")[:16].replace("T", " "))}'
    if c["edited_at"]:
        meta += f' · edited ({c["edit_count"]})'
    if pinned:
        meta += f' · {esc(pinned)}'
    state = badge("Resolved", "verified") if c["status"] == "resolved" else badge("Open", "neutral")
    kind_badge = badge("Note", "warn") if c["kind"] == "note" else ""
    with st.container(key=f"comment_{c['id']}"):
        html(
            f'<div class="ds-row" style="padding:14px 0;border-top:1px solid var(--border);">'
            f'<div><div class="ds-row-meta">{meta}</div>'
            f'<div class="ds-row-title" style="white-space:pre-wrap;font-weight:400;margin-top:4px;">{esc(c["body"])}</div></div>'
            f'<div class="ds-row-spacer"></div>{kind_badge}&nbsp;{state}</div>'
        )
        if editing == c["id"]:
            with st.form(f"edit_form_{c['id']}"):
                new_body = st.text_area("Edit", value=c["body"], label_visibility="collapsed")
                s1, s2 = st.columns(2)
                save = s1.form_submit_button("Save", type="primary")
                cancel = s2.form_submit_button("Cancel")
            if save and new_body.strip():
                try:
                    update_comment(c["id"], body=new_body.strip())
                    st.session_state.pop("editing_comment", None)
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not save: {e}")
            if cancel:
                st.session_state.pop("editing_comment", None)
                st.rerun()
        else:
            b1, b2, b3, _sp = st.columns([1, 1.2, 1, 5])
            if c["can_edit"] and b1.button("Edit", key=f"edit_{c['id']}"):
                st.session_state["editing_comment"] = c["id"]
                st.rerun()
            if c["can_moderate"]:
                label = "Reopen" if c["status"] == "resolved" else "Resolve"
                if b2.button(label, key=f"res_{c['id']}"):
                    try:
                        update_comment(c["id"], status="open" if c["status"] == "resolved" else "resolved")
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))
                if b3.button("Delete", key=f"del_{c['id']}"):
                    try:
                        delete_comment(c["id"])
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))
