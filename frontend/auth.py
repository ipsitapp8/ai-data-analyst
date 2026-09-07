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
.stApp { background: #161616 !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
.block-container { max-width: 420px !important; padding-top: 16vh !important; }
.gate-mark {
  font-size: 22px; color: #e6e6e6; font-weight: 600; margin-bottom: 8px;
}
.gate-sub {
  color: #969696; font-size: 13px; margin-bottom: 24px; line-height: 1.6;
}
[data-testid="stTextInput"] input {
  background: #101010 !important; border: 1px solid rgba(255,255,255,0.22) !important;
  color: #e6e6e6 !important; border-radius: 4px !important;
}
.stFormSubmitButton > button {
  background: #4f8fe0 !important;
  color: #101010 !important; border: none !important;
  border-radius: 4px !important; font-weight: 600 !important;
  box-shadow: none !important;
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
        "<div class='gate-mark'>Silt</div>"
        "<div class='gate-sub'>Private deployment — enter the access password to continue.</div>",
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
