"""Workspaces — communities, the teams inside them, and team membership."""
from __future__ import annotations

import streamlit as st

from api_client import (
    ApiError,
    accept_invite,
    create_community,
    create_team,
    invite_member,
    list_communities,
    list_members,
    my_invites,
)
from config import APP_URL
from style.theme import badge, esc, html, invalidate_workspaces_cache, page_header, page_setup, render_sidebar

page_setup("Workspaces")
render_sidebar("workspaces")

page_header("Workspaces", "Communities hold teams; each team is its own isolated workspace.")

try:
    invites = my_invites()
except ApiError:
    invites = []

if invites:
    with st.container(key="card_invites"):
        html('<div class="ds-section-title">Pending invites</div>')
        html("<div style='height:8px'></div>")
        for inv in invites:
            c1, c2 = st.columns([4, 1])
            with c1:
                html(
                    f'<div class="ds-row-title">{esc(inv["team_name"])}</div>'
                    f'<div class="ds-row-meta">in {esc(inv["community_name"])} · role: {esc(inv["role"])}</div>'
                )
            with c2:
                if st.button("Accept", key=f"accept_{inv['team_id']}"):
                    try:
                        accept_invite(inv["team_id"])
                        invalidate_workspaces_cache()
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Could not accept: {e}")
    html("<div style='height:20px'></div>")

try:
    communities = list_communities()
except ApiError as e:
    communities = []
    st.error(f"Backend unreachable: {e}")

col_l, col_r = st.columns([1.3, 1], gap="large")

with col_l:
    with st.container(key="card_communities"):
        html('<div class="ds-section-title">Your communities</div>')
        html("<div style='height:8px'></div>")
        if not communities:
            html('<div class="ds-row-meta">No communities yet — create one to get started.</div>')
        for c in communities:
            team_count_badge = badge(f"{len(c['teams'])} team(s)", "neutral")
            is_active_community = st.session_state.get("active_community_id") == c["id"]
            hc1, hc2 = st.columns([4, 1])
            with hc1:
                html(f'<div class="ds-row" style="padding:12px 0;border-top:1px solid var(--border);">'
                     f'<div class="ds-row-title">{esc(c["name"])}</div>'
                     f'<div class="ds-row-spacer"></div>'
                     f'{team_count_badge}</div>')
            with hc2:
                if st.button("Selected" if is_active_community else "Select", key=f"pick_community_{c['id']}",
                             disabled=is_active_community):
                    st.session_state["active_community_id"] = c["id"]
                    st.rerun()
            for t in c["teams"]:
                sel = st.session_state.get("active_team_id") == t["id"]
                cc1, cc2 = st.columns([4, 1])
                with cc1:
                    html(f'<div class="ds-row-meta" style="padding-left:14px;">'
                         f'{esc(t["name"])} — {esc(t["role"])}</div>')
                with cc2:
                    if st.button("Switch to" if not sel else "Active", key=f"switch_{t['id']}",
                                 disabled=sel):
                        st.session_state["active_community_id"] = c["id"]
                        st.session_state["active_team_id"] = t["id"]
                        st.rerun()

    html("<div style='height:14px'></div>")
    with st.container(key="card_new_community"):
        html('<div class="ds-section-title">Create a community</div>')
        html("<div style='height:8px'></div>")
        with st.form("new_community_form", clear_on_submit=True):
            name = st.text_input("Community name", label_visibility="collapsed",
                                  placeholder="e.g. Acme Inc.")
            submitted = st.form_submit_button("Create community", type="primary")
        if submitted and name.strip():
            try:
                create_community(name.strip())
                invalidate_workspaces_cache()
                st.rerun()
            except ApiError as e:
                st.error(f"Could not create community: {e}")

with col_r:
    active_community_id = st.session_state.get("active_community_id")
    active_team_id = st.session_state.get("active_team_id")
    active_community = next((c for c in communities if c["id"] == active_community_id), None)

    with st.container(key="card_new_team"):
        html('<div class="ds-section-title">Create a team</div>')
        html("<div style='height:8px'></div>")
        if not active_community:
            html('<div class="ds-row-meta">Pick a community above first.</div>')
        else:
            html(f'<div class="ds-row-meta">In <b>{esc(active_community["name"])}</b></div>')
            with st.form("new_team_form", clear_on_submit=True):
                name = st.text_input("Team name", label_visibility="collapsed",
                                      placeholder="e.g. Growth")
                submitted = st.form_submit_button("Create team", type="primary")
            if submitted and name.strip():
                try:
                    new_team = create_team(active_community["id"], name.strip())
                    st.session_state["active_team_id"] = new_team["id"]
                    invalidate_workspaces_cache()
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not create team: {e}")

    html("<div style='height:14px'></div>")
    with st.container(key="card_members"):
        html('<div class="ds-section-title">Team members</div>')
        html("<div style='height:8px'></div>")
        if not active_team_id:
            html('<div class="ds-row-meta">No active team selected.</div>')
        else:
            try:
                members = list_members(active_team_id)
            except ApiError as e:
                members = []
                st.error(f"Could not load members: {e}")
            for m in members:
                kind = "verified" if m["status"] == "active" else "warn"
                html(
                    f'<div class="ds-row" style="padding:10px 0;border-top:1px solid var(--border);">'
                    f'<div><div class="ds-row-title">{esc(m["display_name"] or m["email"])}</div>'
                    f'<div class="ds-row-meta">{esc(m["email"])} · {esc(m["role"])}</div></div>'
                    f'<div class="ds-row-spacer"></div>{badge(m["status"], kind)}</div>'
                )

            html("<div style='height:12px'></div>")

            # Persisted in session_state rather than a bare st.success() right
            # before st.rerun(): a message shown immediately before a rerun
            # gets wiped before it's ever visible. This is also exactly the
            # spot that needs to survive the rerun -- "did the email actually
            # send" was previously impossible to tell from the UI at all.
            last = st.session_state.get("last_invite_result")
            if last and last["team_id"] == active_team_id:
                if last["email_sent"]:
                    st.success(f"Invited {last['email']} — confirmation email sent.")
                else:
                    st.warning(
                        f"Invited {last['email']} — the row was created, but the email "
                        f"couldn't be sent (SMTP not configured or delivery failed). "
                        f"Share this link with them yourself: {APP_URL}"
                    )

            with st.form("invite_form", clear_on_submit=True):
                email = st.text_input("Invite by email", placeholder="teammate@company.com")
                submitted = st.form_submit_button("Send invite", type="primary")
            if submitted and email.strip():
                try:
                    result = invite_member(active_team_id, email.strip())
                    st.session_state["last_invite_result"] = {
                        "team_id": active_team_id,
                        "email": email.strip(),
                        "email_sent": result["email_sent"],
                    }
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not invite: {e}")
