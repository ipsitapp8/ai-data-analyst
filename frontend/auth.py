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
import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_SESSION_KEY = "_auth_ok"

def _landing_css() -> str:
    """Landing/sign-in stylesheet (style/landing.css), flattened like _gate_css."""
    raw = (Path(__file__).parent / "style" / "landing.css").read_text(encoding="utf-8")
    return "<style>" + " ".join(line.strip() for line in raw.splitlines() if line.strip()) + "</style>"


def _gate_css() -> str:
    """Login-screen stylesheet (style/gate.css), flattened to one line: markdown
    ends an HTML block at the first blank line and would print the rest as text."""
    raw = (Path(__file__).parent / "style" / "gate.css").read_text(encoding="utf-8")
    return "<style>" + " ".join(line.strip() for line in raw.splitlines() if line.strip()) + "</style>"


_GATE_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600'
    '&display=swap" rel="stylesheet">'
)
_GATE_LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:#837ab6" '
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


# "Stay signed in": Streamlit forgets session_state on every browser refresh, so
# the API token would be lost and the visitor sent back to the login form. We keep
# the same JWT the API issued (7-day expiry, validated by the API on every use) in
# a browser cookie and restore the session from it. The cookie is script-readable
# because Streamlit offers no server-side cookie writing; it carries no more than
# the session token already held in memory, and logging out clears it.
_COOKIE = "silt_token"
_COOKIE_MAX_AGE = 7 * 24 * 3600


def _cookie_js(token: str | None) -> str:
    if token:
        value = "encodeURIComponent(" + json.dumps(token).replace("</", "<\\/") + ")"
        attrs = "'; path=/; max-age=" + str(_COOKIE_MAX_AGE) + "; SameSite=Lax'"
    else:
        value = "''"
        attrs = "'; path=/; max-age=0; SameSite=Lax'"
    return (
        "<script>(function(){var secure = window.parent.location.protocol === 'https:' ? '; Secure' : '';"
        f"window.parent.document.cookie = '{_COOKIE}=' + {value} + {attrs} + secure;}})();</script>"
    )


def _write_cookie(token: str | None) -> None:
    components.html(_cookie_js(token), height=0)


def _restore_session_from_cookie() -> bool:
    """Re-establish the login from the browser cookie. Returns True on success."""
    if st.session_state.get("_logged_out"):
        return False
    try:
        token = st.context.cookies.get(_COOKIE)
    except Exception:  # noqa: BLE001 - no request context (e.g. headless test)
        return False
    if not token:
        return False
    import api_client

    try:
        # Raw call, not api_client.me(): a stale cookie must not trigger the
        # "session expired" flow (which reruns the script) -- it just means "log in".
        resp = api_client.requests.get(f"{api_client.BACKEND_BASE_URL}/api/auth/me",
                                       headers={"Authorization": f"Bearer {token}"}, timeout=api_client.TIMEOUT)
    except Exception:  # noqa: BLE001 - backend down: show the normal login form
        return False
    if not resp.ok:
        st.session_state["_clear_cookie"] = True
        return False
    st.session_state[_USER_SESSION_KEY] = resp.json()
    st.session_state["auth_token"] = token
    return True


def _google_enabled() -> bool:
    """Google sign-in needs both Streamlit's [auth] OIDC settings and the secret
    the API checks. Missing either one simply hides the button."""
    from config import INTERNAL_AUTH_SECRET

    if not INTERNAL_AUTH_SECRET:
        return False
    try:
        return bool(st.secrets.get("auth"))
    except Exception:  # noqa: BLE001 - no secrets file at all
        return False


def _sign_in_with_google_if_ready() -> bool:
    """After Google redirects back, trade the verified identity for an app session."""
    if st.session_state.get("_logged_out"):
        return False
    try:
        user = st.user
        if not user.is_logged_in:
            return False
        email = str(user.get("email") or "")
        name = str(user.get("name") or "")
        verified = bool(user.get("email_verified", False))
    except Exception:  # noqa: BLE001
        return False

    from api_client import ApiError, google_exchange
    from config import INTERNAL_AUTH_SECRET

    try:
        result = google_exchange(email, name, verified, INTERNAL_AUTH_SECRET)
    except ApiError as e:
        st.session_state["auth_notice"] = f"Google sign-in could not be completed: {e}"
        return False
    st.session_state[_USER_SESSION_KEY] = result["user"]
    st.session_state["auth_token"] = result["access_token"]
    st.session_state["_set_cookie"] = result["access_token"]
    return True


def logout() -> None:
    try:
        if st.user.is_logged_in:
            st.logout()  # also end the Google identity, or the next load would sign straight back in
    except Exception:  # noqa: BLE001 - no OIDC configured
        pass
    for key in (_USER_SESSION_KEY, "auth_token", "active_community_id", "active_team_id"):
        st.session_state.pop(key, None)
    # Don't restore from the cookie that is still on this very connection.
    st.session_state["_logged_out"] = True
    st.session_state["_clear_cookie"] = True


def require_login() -> None:
    """Per-user login/signup, layered on top of require_password()'s shared
    deployment gate. Halts rendering with a Login/Sign up form until the
    visitor has a real account and a valid session token.
    """
    from api_client import ApiError, login as api_login, signup as api_signup

    if st.session_state.get(_USER_SESSION_KEY) and st.session_state.get("auth_token"):
        if "view" in st.query_params:  # leftover from the sign-in page address
            del st.query_params["view"]
        pending = st.session_state.pop("_set_cookie", None)
        if pending:
            _write_cookie(pending)
        return

    if _restore_session_from_cookie():
        return
    if _google_enabled() and _sign_in_with_google_if_ready():
        st.rerun()
    if st.session_state.pop("_clear_cookie", None):
        _write_cookie(None)

    # Two pages on one URL: the marketing landing page (default) and the sign-in page (?view=login or
    # ?view=signup). "Get started" and "Log in" are plain links, so each page is its own address.
    view = str(st.query_params.get("view", "")).lower()
    # Why the visitor is back at this form (set when the API rejects a session token). Shown once.
    notice = st.session_state.pop("auth_notice", None)
    if notice and view not in ("login", "signup"):
        view = "login"
        st.query_params["view"] = "login"  # stay on the sign-in page across reruns

    st.markdown(_landing_css(), unsafe_allow_html=True)
    st.markdown(_GATE_FONTS, unsafe_allow_html=True)

    if view not in ("login", "signup"):
        st.markdown(
            f'<div class="lp-nav"><a class="lp-brand" href="?">{_GATE_LOGO}Silt</a>'
            '<div class="lp-nav-r"><a class="lp-ghost" href="?view=login" target="_self">Log in</a>'
            '<a class="lp-pill" href="?view=signup" target="_self">Get started</a></div></div>'
            '<div class="lp-hero"><div>'
            '<div class="lp-badge"><b>AGENTIC</b>Your autonomous data analyst</div>'
            '<h1 class="lp-title">Ask your data anything.<br><span>Trust every answer.</span></h1>'
            '<p class="lp-sub">Silt is an AI data analyst for teams. Ask a question in plain English; an agent plans, '
            'runs and explains the analysis, and an independent model checks it before it reaches you.</p>'
            '<a class="lp-cta" href="?view=signup" target="_self">Get started<i>&rarr;</i></a>'
            '<div class="lp-points"><span>Runs in a secure sandbox</span><span>Every number verified</span>'
            '<span>Fully auditable</span></div></div>'
            '<div class="lp-stage"><div class="lp-orb"></div>'
            '<div class="lp-float lp-q"><small>Ask Silt</small>Which region grew fastest last quarter?</div>'
            '<div class="lp-float lp-v"><div class="row"><span class="ok">&#10003;</span>Checked by an independent model</div>'
            '<div class="bar"></div><div class="bar s"></div></div></div></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # ---- sign-in page ----
    st.markdown(
        f'<div class="lp-nav"><a class="lp-brand" href="?" target="_self">{_GATE_LOGO}Silt</a>'
        '<div class="lp-nav-r"><a class="lp-ghost" href="?" target="_self">&larr; Back to home</a></div></div>'
        '<div class="lp-login-pad"></div>',
        unsafe_allow_html=True,
    )
    if notice:
        st.warning(notice)

    with st.container(key="auth_card"):
        if _google_enabled():
            if st.button("Continue with Google", key="google_login", use_container_width=True):
                st.login()
            st.markdown('<div class="lp-or"><span>or use email</span></div>', unsafe_allow_html=True)
        labels = ["Create account", "Log in"] if view == "signup" else ["Log in", "Create account"]
        _tabs = dict(zip(labels, st.tabs(labels), strict=True))
        tab_login, tab_signup = _tabs["Log in"], _tabs["Create account"]

        with tab_login:
            st.markdown('<div class="lp-card-title">Welcome back.</div>'
                        '<div class="lp-card-sub">Sign in to your team&#39;s workspace.</div>',
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
                    st.session_state["_set_cookie"] = result["access_token"]
                    st.session_state.pop("_logged_out", None)
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not log in: {e}")

        with tab_signup:
            st.markdown('<div class="lp-card-title">Create your account.</div>'
                        '<div class="lp-card-sub">Start asking questions of your data.</div>',
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
                    st.session_state["_set_cookie"] = result["access_token"]
                    st.session_state.pop("_logged_out", None)
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not sign up: {e}")

    st.stop()
