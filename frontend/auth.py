"""Shared-password gate for the deployed app.

Deliberately minimal: one password, held in an environment variable (a Space
secret in the HF deployment), checked in constant time. This is a "don't let
strangers burn my Gemini quota" control, not an auth system -- there are no
accounts, no sessions beyond Streamlit's own, and no per-user isolation.

If APP_PASSWORD is unset or empty the gate is disabled entirely, so local
development is unchanged.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_SESSION_KEY = "_auth_ok"

def _gate_css() -> str:
    """Login-screen stylesheet (style/gate.css), flattened to one line: markdown
    ends an HTML block at the first blank line and would print the rest as text."""
    raw = (Path(__file__).parent / "style" / "gate.css").read_text(encoding="utf-8")
    return "<style>" + " ".join(line.strip() for line in raw.splitlines() if line.strip()) + "</style>"


_GATE_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600'
    '&family=Newsreader:opsz,wght@6..72,300..600&display=swap" rel="stylesheet">'
)
_GATE_LOGO = (
    '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#f1d9b5" stroke-width="1.7" '
    'stroke-linecap="round"><path d="M5 20L17 4M9 21l10-13M4 15l8-11"/></svg>'
)


# Runs in Streamlit's same-origin component iframe and reaches up to the page.
# One delegated listener (installed once) flips a class on <body>; the stylesheet
# does the sliding. The class lives on <body>, which React doesn't manage, so it
# survives reruns such as a failed-login error message.
_AUTH_SWITCH_JS = """
<script>
(function () {
  var doc = window.parent.document;
  if (window.parent.__siltAuthSwitch) return;
  window.parent.__siltAuthSwitch = true;
  doc.addEventListener("click", function (e) {
    var link = e.target.closest && e.target.closest("a.auth-switch");
    if (!link) return;
    e.preventDefault();
    doc.body.classList.toggle("auth-mode-signup", link.getAttribute("data-to") === "signup");
  });
})();
</script>
"""


def _render_gate_header(eyebrow: str, title: str, sub: str) -> None:
    st.markdown(_gate_css(), unsafe_allow_html=True)
    st.markdown(_GATE_FONTS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="gate-brand">{_GATE_LOGO}<div class="gate-mark">Silt</div></div>'
        f'<div class="gate-eyebrow">{eyebrow}</div>'
        f'<div class="gate-title">{title}</div>'
        f'<div class="gate-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def _password_is_set() -> str:
    return os.getenv("APP_PASSWORD", "").strip()


def require_password() -> None:
    """Halt rendering until the correct shared password is entered.

    Call after st.set_page_config. Does nothing when APP_PASSWORD is unset.
    """
    expected = _password_is_set()
    if not expected:
        return
    if st.session_state.get(_SESSION_KEY):
        return

    _render_gate_header("Private deployment", "Enter the access password.",
                        "This workspace is private. Enter the shared password to continue.")

    with st.form("auth_gate", clear_on_submit=False):
        entered = st.text_input("Password", type="password", label_visibility="collapsed",
                                placeholder="Access password")
        submitted = st.form_submit_button("Enter", type="primary", use_container_width=True)

    if submitted:
        # compare_digest over utf-8 bytes: constant time, and avoids the
        # UnicodeEncodeError compare_digest raises on non-ascii str input.
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            st.session_state[_SESSION_KEY] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


_USER_SESSION_KEY = "auth_user"


def current_user() -> dict | None:
    return st.session_state.get(_USER_SESSION_KEY)


def logout() -> None:
    for key in (_USER_SESSION_KEY, "auth_token", "active_community_id", "active_team_id"):
        st.session_state.pop(key, None)


def require_login() -> None:
    """Per-user login/signup, layered on top of require_password()'s shared
    deployment gate. Halts rendering with a Login/Sign up form until the
    visitor has a real account and a valid session token.
    """
    from api_client import ApiError, login as api_login, signup as api_signup

    if st.session_state.get(_USER_SESSION_KEY) and st.session_state.get("auth_token"):
        return

    st.markdown(_gate_css(), unsafe_allow_html=True)
    st.markdown(_GATE_FONTS, unsafe_allow_html=True)
    st.markdown("<style>.block-container { max-width: 1020px !important; padding-top: 8vh !important; }</style>",
                unsafe_allow_html=True)

    with st.container(key="auth_stage"):
        # The gold panel always covers the *inactive* side; switching modes only
        # flips its class, and the CSS transition slides it across.
        st.markdown(
            '<div class="auth-overlay">'
            f'<div class="auth-ov-brand">{_GATE_LOGO}<div class="gate-mark">Silt</div></div>'
            '<div class="gate-eyebrow">Your data. Our analysis.</div>'
            '<div class="gate-title">Ask a question in plain English.</div>'
            '<div class="gate-sub">An agent plans, runs and explains the analysis, and an independent '
            'model checks it before it reaches you.</div>'
            '<ul class="auth-ov-list"><li>Runs in a secure sandbox</li><li>Every number is verified</li>'
            '<li>Fully auditable</li></ul></div>',
            unsafe_allow_html=True,
        )
        col_signup, col_login = st.columns(2)

        with col_signup:
            st.markdown('<div class="auth-panel-title">Create your account.</div>'
                        '<div class="auth-panel-sub">Start asking questions of your data.</div>',
                        unsafe_allow_html=True)
            with st.form("signup_form", clear_on_submit=False):
                name = st.text_input("Display name", key="signup_name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password",
                                          help="At least 8 characters.")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                try:
                    result = api_signup(email.strip().lower(), password, name)
                    st.session_state[_USER_SESSION_KEY] = result["user"]
                    st.session_state["auth_token"] = result["access_token"]
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not sign up: {e}")
            st.markdown('<a class="auth-switch" data-to="login">Already have an account? Log in →</a>',
                        unsafe_allow_html=True)

        with col_login:
            st.markdown('<div class="auth-panel-title">Welcome back.</div>'
                        '<div class="auth-panel-sub">Sign in to your team&#39;s workspace.</div>',
                        unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                try:
                    result = api_login(email.strip().lower(), password)
                    st.session_state[_USER_SESSION_KEY] = result["user"]
                    st.session_state["auth_token"] = result["access_token"]
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not log in: {e}")
            st.markdown('<a class="auth-switch" data-to="signup">New to Silt? Create an account →</a>',
                        unsafe_allow_html=True)

    components.html(_AUTH_SWITCH_JS, height=0)

    st.stop()
