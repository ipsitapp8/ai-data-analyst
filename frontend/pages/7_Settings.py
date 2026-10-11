"""Settings — account/team context, plus a read-only view of live backend config."""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, health, my_workspaces
from auth import current_user
from style.theme import badge, esc, html, page_header, page_setup, render_sidebar

page_setup("Settings")
render_sidebar("settings")

page_header("Settings", "Your account, your team, and agent configuration.")

try:
    hz = health()
except ApiError as e:
    hz = {}
    st.error(f"Backend unreachable: {e}")


def row(label: str, value: str, last: bool = False) -> str:
    border = "" if last else "border-bottom:1px solid var(--border);"
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:15px 22px;{border}">'
        f'<span style="font-size:0.9rem;color:var(--text-primary);">{label}</span>'
        f'<span style="font-size:0.9rem;color:var(--text-primary);font-weight:600;">'
        f"{value}</span></div>"
    )


# ---------------------------------------------------------------- account --
user = current_user()
active_team_id = st.session_state.get("active_team_id")

active_community_name, active_team_name, active_team_role = None, None, None
try:
    for community in my_workspaces().get("communities", []):
        for team in community["teams"]:
            if team["id"] == active_team_id:
                active_community_name = community["name"]
                active_team_name = team["name"]
                active_team_role = team["role"]
except ApiError:
    pass

with st.container(key="flat_account"):
    html('<div class="ds-card-head"><div class="ds-section-title">Account</div></div>')
    html(
        row("Name", esc(user["display_name"]) if user else "—")
        + row("Email", esc(user["email"]) if user else "—")
    )
    if active_team_name:
        html(
            row(
                "Active workspace",
                f"{esc(active_community_name)} / {esc(active_team_name)} "
                f"{badge(active_team_role, 'neutral')}",
                last=True,
            )
        )
    else:
        html(row("Active workspace", "None selected", last=True))

c1, _c2 = st.columns([1, 4])
with c1:
    if st.button("Manage workspaces  →", key="settings_go_workspaces"):
        st.switch_page("pages/8_Workspaces.py")

html("<div style='height:18px'></div>")

sandbox = hz.get("sandbox_backend", "—")
img_ready = hz.get("sandbox_image_ready", False)
sandbox_val = f"Docker · {'image ready' if img_ready else 'image not built'}" \
    if sandbox == "docker" else "Subprocess (not isolated)"

with st.container(key="flat_ws"):
    html('<div class="ds-card-head"><div class="ds-section-title">Workspace</div></div>')
    html(
        row("Default compute", sandbox_val)
        + row("Agent autonomy", "Plan, execute &amp; verify")
        + row("Verification threshold", "Critic must pass")
        + row("Data retention", "Local SQLite (no expiry)", last=True)
    )

html("<div style='height:18px'></div>")

gem = hz.get("gemini_key_configured", False)
lla = hz.get("llama_key_configured", False)

with st.container(key="flat_models"):
    html('<div class="ds-card-head"><div class="ds-section-title">Models</div></div>')
    # "Key set", not "Connected": health only reports whether a key string is
    # present -- it does not call the provider, so a present-but-invalid key
    # would still show here.
    html(
        row("Planner / Executor / Dashboard",
            f"Gemini {badge('Key set', 'verified') if gem else badge('No key', 'error')}")
        + row("Critic (independent verifier)",
              f"Llama {badge('Key set', 'verified') if lla else badge('No key', 'error')}",
              last=True)
    )
    html(
        '<div class="ds-row-meta" style="padding:0 22px 18px 22px;line-height:1.65;">'
        "The Critic runs on a different model family than the Executor on purpose — a "
        "different model catches different mistakes than the same model re-checking its own "
        "work. If the Llama key is missing or invalid, the Critic falls back to Gemini and "
        "the audit trail records that the check was <b>not</b> cross-model.</div>"
    )

html("<div style='height:18px'></div>")

with st.container(key="flat_sys"):
    html('<div class="ds-card-head"><div class="ds-section-title">System</div></div>')
    api_ok = bool(hz)
    db_ok = hz.get("database_ok", False)
    html(
        row("Backend API", "Reachable" if api_ok else "Unreachable")
        + row("Database", "Connected" if db_ok else "Unreachable")
        + row("Sandbox backend", sandbox)
        + row("Docker image", "Built" if img_ready else "Not built", last=True)
    )

if sandbox == "docker" and not img_ready:
    html("<div style='height:14px'></div>")
    st.warning(
        "The sandbox image isn't built yet, so analyses will fail at the execution step. "
        "Start Docker Desktop, then run `./backend/app/sandbox/build.ps1`."
    )
