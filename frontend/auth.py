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

import streamlit as st

_SESSION_KEY = "_auth_ok"

GATE_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
.stApp { background: #0a0908 !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
.block-container { max-width: 460px !important; padding-top: 14vh !important; }
.gate-mark {
  font-size: 30px; letter-spacing: .18em; color: #f2f0ea;
  font-weight: 600; margin-bottom: 6px;
}
.gate-sub { color: #97968f; font-size: 14px; margin-bottom: 26px; line-height: 1.55; }
.gate-card {
  border: 1px solid rgba(255,255,255,.085); border-radius: 14px;
  background: rgba(17,15,15,.78); padding: 26px 24px 8px 24px;
}
</style>
"""


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

    st.markdown(GATE_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='gate-mark'>⬡ DATASAGE</div>"
        "<div class='gate-sub'>This is a private demo deployment. "
        "Enter the access password to continue.</div>",
        unsafe_allow_html=True,
    )

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
