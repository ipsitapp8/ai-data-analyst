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

from landing_scene import LANDING_SCENE_HTML
import streamlit.components.v1 as components

_SESSION_KEY = "_auth_ok"

def _lp_words(text: str, start: int = 0) -> str:
    """A title split into words, so each can rise in turn when it scrolls into view."""
    import html as _h
    return " ".join(f'<span class="w" style="--i:{start + i}">{_h.escape(w)}</span>' for i, w in enumerate(text.split()))


# Every tool in the product, grouped. Names and one-liners follow what each page of the app says it does.
_LP_GROUPS = [
    ("Ask", "Questions in, answers out.", [
        ("Analyses", "Ask in plain English", "An agent plans the analysis, writes the code and runs it in a sandbox."),
        ("Silt Copilot", "Chat with your data", "Free-form questions, calculations and follow-ups in one conversation."),
        ("Drill-down", "Break a metric down", "Filter and split one metric step by step, with rules only."),
        ("Datasets", "Bring your data", "Upload CSV or Excel files, keep versions and explore your sources."),
    ]),
    ("Verify", "Nothing reaches you unchecked.", [
        ("Critic", "A second model checks it", "An independent model re-verifies every number before the answer reaches you."),
        ("Audit trail", "Every step is logged", "Full traceability for each analysis: the plan, the code and the evidence."),
        ("Semantic layer", "One definition per metric", "Approved metric definitions that analyses use instead of guessing."),
        ("Evaluation", "Measured against references", "Regression results for the pipeline and each of its parts."),
    ]),
    ("Investigate", "Find what moved, and where.", [
        ("Root cause", "Where did the number move?", "Locate where a metric changed between two periods. Computed from your data."),
        ("Data quality", "Spot drift early", "What changed in the data itself, kept separate from changes in the business."),
        ("Alerts", "Hear about changes first", "What changed, went unusual or failed verification, in one feed."),
    ]),
    ("Decide", "Look ahead, then share it.", [
        ("What-if", "Test a decision first", "Change an input and see the range of outcomes, with the evidence behind each."),
        ("Dashboards", "Verified dashboards", "Generated from verified analyses, ready to read at a glance."),
        ("Reports", "Packaged answers", "Every completed analysis, packaged and verified."),
    ]),
    ("Operate", "Run it as a team.", [
        ("Scheduled", "Re-run automatically", "Tracked questions re-run on the latest data without anyone asking again."),
        ("Workspaces", "Isolated by team", "Communities hold teams, and each team is its own isolated workspace."),
        ("Shared links", "Share a result", "Send a read-only link to an analysis, no account needed to view it."),
        ("Discussion", "Notes with context", "Comments pinned to an analysis or a dataset, so the context stays with it."),
    ]),
]


# A small looping animation per tool, shown at the top of its card. Pure markup: the motion is CSS, and it only plays
# once the card has scrolled into view.
_LP_VIS = {
    "Analyses": '<div class="vis v-type"><span class="q s1">Which region grew fastest?</span><span class="a s2">West, up 18.4%<u></u></span></div>',
    "Silt Copilot": '<div class="vis v-chat"><span class="u s1">Why did March dip?</span><span class="dots"><i></i><i></i><i></i></span><span class="r s3">Mostly West: 8.4 of 12.4 points</span></div>',
    "Drill-down": '<div class="vis v-drill"><span class="s1">Revenue</span><b class="s2">&rsaquo;</b><span class="s2">West</span><b class="s3">&rsaquo;</b><span class="s3">Q3</span><b class="s4">&rsaquo;</b><span class="s4">Online</span></div>',
    "Datasets": '<div class="vis v-data"><div class="file s1"><em>sales.csv</em><i><u class="gx1"></u></i></div><div class="vers"><span class="s3">v1</span><span class="s4">v2</span><span class="s5">v3</span></div></div>',
    "Critic": '<div class="vis v-crit"><p class="s1"><i>&#10003;</i>Totals match source</p><p class="s2"><i>&#10003;</i>No double counting</p><p class="s3"><i>&#10003;</i>12 of 12 reproduced</p></div>',
    "Audit trail": '<div class="vis v-log"><p class="s1"><u></u>Plan saved</p><p class="s2"><u></u>Code executed</p><p class="s3"><u></u>Evidence kept</p></div>',
    "Semantic layer": '<div class="vis v-sem"><code class="s1">Revenue = SUM(amount)</code><span class="stamp s3">&#10003; approved</span></div>',
    "Evaluation": '<div class="vis v-eval"><div class="row"><span>Planner</span><i><u class="gx1"></u></i></div><div class="row"><span>Executor</span><i><u class="gx2"></u></i></div><div class="row"><span>Critic</span><i><u class="gx3"></u></i></div></div>',
    "Root cause": '<div class="vis v-rc"><div class="hd s1"><span>Revenue, Q2 to Q3</span><b>&minus;12.4%</b></div><div class="row"><span>West</span><i><u class="gx1" style="--w:70%"></u></i></div><div class="row"><span>Online</span><i><u class="gx2" style="--w:34%"></u></i></div></div>',
    "Data quality": '<div class="vis v-dq"><svg viewBox="0 0 200 60" preserveAspectRatio="none"><path pathLength="100" d="M0 42 L28 38 L56 40 L84 36 L112 39 L140 22 L170 12 L200 8"/><circle class="s4" cx="140" cy="22" r="4"/></svg><span class="tag s4">drift detected</span></div>',
    "Alerts": '<div class="vis v-al"><div class="toast s1"><i class="pulse"></i><div><b>Revenue looks unusual</b><span>West &middot; just now</span></div></div><div class="toast s3 dim"><i></i><div><b>A check failed verification</b><span>Scheduled run &middot; 06:00</span></div></div></div>',
    "What-if": '<div class="vis v-wi"><div class="row"><span>Price</span><i class="sl"><u class="knob"></u></i></div><div class="row"><span>Revenue</span><b class="s2">118k &ndash; 131k</b></div></div>',
    "Dashboards": '<div class="vis v-db"><i style="--h:46%;--n:0"></i><i style="--h:72%;--n:1"></i><i style="--h:58%;--n:2"></i><i style="--h:88%;--n:3"></i><i style="--h:66%;--n:4"></i></div>',
    "Reports": '<div class="vis v-rp"><p class="l s1"></p><p class="l s2"></p><p class="l short s3"></p><span class="stamp s4">&#10003; verified</span></div>',
    "Scheduled": '<div class="vis v-sc"><svg viewBox="0 0 44 44"><circle class="track" cx="22" cy="22" r="18"/><circle class="arc" cx="22" cy="22" r="18" pathLength="100"/></svg><div><b class="s1">Every Monday</b><span class="s2">next run 06:00</span></div></div>',
    "Workspaces": '<div class="vis v-ws"><span class="t s1">Sales</span><span class="t s2">Finance</span><span class="t s3">Ops</span><em class="s4">Each team is isolated</em></div>',
    "Shared links": '<div class="vis v-sh"><span class="link s1">shared link</span><span class="dash"></span><span class="badge s3">view only</span></div>',
    "Discussion": '<div class="vis v-ds"><p class="s1"><i>A</i>Check the West drop?</p><p class="s3 me"><i>B</i>Pinned to Q3 analysis</p></div>',
}


def _lp_feature_grid() -> str:
    """The full tool list as one markup string: a labelled block per group, a small card per tool."""
    import html as _h
    total = sum(len(g[2]) for g in _LP_GROUPS)
    out = ['<div class="lp-sec lm-all"><div class="lp-eyebrow">Everything inside</div>',
           f'<h2 class="lp-h2">{_lp_words("Every tool, in one workspace.")}</h2>',
           f'<p class="lm-all-sub">{total} tools across asking, verifying, investigating, deciding and running your analytics.</p>',
           '<div class="lm-groups">']
    for gi, (name, line, items) in enumerate(_LP_GROUPS, 1):
        cards = "".join(
            f'<div class="lm-fcard">{_LP_VIS.get(tag, "")}<div class="n">{_h.escape(tag)}</div><b>{_h.escape(title)}</b><p>{_h.escape(text)}</p></div>'
            for tag, title, text in items)
        out.append(f'<div class="lm-fgroup"><div class="lm-fhead"><div class="k">{gi:02d} / {_h.escape(name.upper())}</div>'
                   f'<div class="t">{_h.escape(line)}</div></div><div class="lm-fcards">{cards}</div></div>')
    out.append('</div></div>')
    return "".join(out)


def _lp_ticker() -> str:
    """The tool names as a loop for the closing ticker (the list is doubled so the loop has no seam)."""
    import html as _h
    names = [tag for _, _, items in _LP_GROUPS for tag, _, _ in items]
    chips = "".join(f'<span><i></i>{_h.escape(n)}</span>' for n in names)
    return chips + chips


# Scroll choreography for the landing page. Runs in Streamlit's same-origin component iframe and reaches up to the page:
# the first things rise in on load, the lower sections appear one by one as they come into view, feature columns drift at
# different speeds, a light follows the cursor over each card and the steps line draws with the scroll. Elements are
# hidden only after they are marked, so if this does not run everything is simply visible.
_LP_REVEAL_JS = """
<script>
(function () {
  var w, d;
  try { w = window.parent; d = w.document; } catch (e) { return; }
  if (!('IntersectionObserver' in w)) return;
  if (w.__lmStop) { try { w.__lmStop(); } catch (e) {} }
  w.__lmIntroAt = Date.now();
  var SEL = '.lp-strip span,.lp-eyebrow,.lp-h2,.lm-all-sub,.lp-feat,.lm-fgroup,.lm-fcard,.lp-steps>div,.lm-marq,.lm-final-card,.lm-big';
  var STAGGER = '.lp-strip span,.lp-feat,.lm-fcard,.lp-steps>div';
  var INTRO = ['.lp-kicker', '.lp-title', '.lp-sub', '.lp-ctas'];
  var io = new w.IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('lm-in'); io.unobserve(e.target); } });
  }, { threshold: 0.15, rootMargin: '0px 0px -5% 0px' });
  function intro() {
    if (Date.now() - w.__lmIntroAt > 2500) return;
    INTRO.forEach(function (sel, i) {
      d.querySelectorAll(sel).forEach(function (el) {
        if (el.__lmi) return; el.__lmi = 1;
        el.style.animationDelay = (i * 140) + 'ms'; el.classList.add('lm-intro');
      });
    });
  }
  function scan() {
    intro();
    var H = w.innerHeight;
    d.querySelectorAll(SEL).forEach(function (el) {
      if (el.__lm) return; el.__lm = 1;
      var par = el.parentElement, idx = par ? Array.prototype.indexOf.call(par.children, el) : 0;
      if (el.matches(STAGGER)) el.style.transitionDelay = (idx * 90) + 'ms';
      if (el.classList.contains('lp-feat')) el.setAttribute('data-par', [0, 0.03, 0.06][idx % 3]);
      if (el.classList.contains('lp-feat') || el.classList.contains('lm-fcard')) {
        el.addEventListener('mousemove', function (e) { var b = el.getBoundingClientRect(); el.style.setProperty('--mx', (e.clientX - b.left) + 'px'); el.style.setProperty('--my', (e.clientY - b.top) + 'px'); });
      }
      el.classList.add('lm-r');
      if (el.getBoundingClientRect().top < H * 0.92) el.classList.add('lm-now', 'lm-in');
      else io.observe(el);
    });
  }
  function line() {
    d.querySelectorAll('.lp-feat[data-par]').forEach(function (el) {
      var r = el.getBoundingClientRect(), off = r.top + r.height / 2 - w.innerHeight / 2;
      el.style.setProperty('--py', (-off * parseFloat(el.getAttribute('data-par'))).toFixed(1) + 'px');
    });
    var s = d.querySelector('.lp-steps'); if (!s) return;
    var r = s.getBoundingClientRect(), p = (w.innerHeight * 0.85 - r.top) / (w.innerHeight * 0.5);
    s.style.setProperty('--lm-p', Math.max(0, Math.min(1, p)).toFixed(3));
  }
  var raf = 0;
  function tick() { if (raf) return; raf = w.requestAnimationFrame(function () { raf = 0; scan(); line(); }); }
  var mo = new w.MutationObserver(tick);
  mo.observe(d.body, { childList: true, subtree: true });
  d.addEventListener('scroll', tick, true);
  w.addEventListener('resize', tick);
  w.__lmStop = function () { io.disconnect(); mo.disconnect(); d.removeEventListener('scroll', tick, true); w.removeEventListener('resize', tick); };
  tick();
})();
</script>"""


def _landing_css() -> str:
    """Landing/sign-in stylesheet (style/landing.css), flattened like _gate_css."""
    raw = (Path(__file__).parent / "style" / "landing.css").read_text(encoding="utf-8")
    raw += chr(10) + (Path(__file__).parent / "style" / "landing_indigo.css").read_text(encoding="utf-8")
    raw += chr(10) + (Path(__file__).parent / "style" / "landing_hero.css").read_text(encoding="utf-8")
    raw += chr(10) + (Path(__file__).parent / "style" / "landing_motion.css").read_text(encoding="utf-8")
    return "<style>" + " ".join(line.strip() for line in raw.splitlines() if line.strip()) + "</style>"


def _gate_css() -> str:
    """Login-screen stylesheet (style/gate.css), flattened to one line: markdown
    ends an HTML block at the first blank line and would print the rest as text."""
    raw = (Path(__file__).parent / "style" / "gate.css").read_text(encoding="utf-8")
    return "<style>" + " ".join(line.strip() for line in raw.splitlines() if line.strip()) + "</style>"


_GATE_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&family=JetBrains+Mono:wght@400;500'
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
    st.session_state["_go_home"] = True
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


def go_home_if_asked() -> None:
    """After an explicit sign-in or workspace step, land on Home whichever page the browser tab was on.

    Signing in re-runs the page the tab already had open, so someone who logged out from Analyses came back
    to Analyses. The flag is set only by those explicit steps (never by a page refresh or a deep link), is
    used once, and is ignored when the tab is already on Home."""
    if not st.session_state.pop("_go_home", None):
        return
    try:
        from urllib.parse import urlparse
        path = urlparse(str(getattr(st.context, "url", "") or "")).path.strip("/")
    except Exception:  # noqa: BLE001 - if the address cannot be read, stay where we are
        return
    if path:
        st.switch_page("Home.py")


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
        go_home_if_asked()
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
            '<div class="lm-scope"></div>'
            f'<div class="lp-nav"><a class="lp-brand" href="?" target="_self">{_GATE_LOGO}Silt</a>'
            '<div class="lp-navpill"><a href="?" target="_self">Home</a><a class="off" href="?view=login" target="_self">Log in</a></div>'
            '<div class="lp-nav-r"><a class="lp-pill" href="?view=signup" target="_self">Get started &#8599;</a></div></div>'
            '<div class="lp-aurora" aria-hidden="true"><i class="a1"></i><i class="a2"></i><i class="a3"></i></div><div class="lp-spot" aria-hidden="true"></div>'
            '<div class="lp-hero2"><div class="lp-hcopy">'
            '<div class="lp-kicker"><span class="dot"></span>Autonomous AI data analyst</div>'
            '<h1 class="lp-title"><span class="l1">Data Matters.</span>Ask Your Data<br><span class="grad">In Plain English</span><br>Trust Every Answer</h1>'
            '<p class="lp-sub">An autonomous AI data analyst for teams. An agent plans, runs and explains the analysis, '
            'and an independent model checks it before it reaches you.</p>'
            '<div class="lp-ctas"><a class="lp-btn light" href="?view=login" target="_self">Log in</a>'
            '<a class="lp-btn dark" href="?view=signup" target="_self">Get Started <span class="arr">&rarr;</span></a></div></div>'
            '<div class="lp-core-stage" id="lp-core">'
            '<div class="lp-panel lp-p1"><div class="lp-ptag"><i></i>PLANNER &middot; 01</div><div class="lp-ptitle">Plans the analysis</div>'
            '<div class="lp-pline">&ldquo;Which region grew fastest?&rdquo;</div><div class="lp-pline lp-dim">3 steps &middot; group by region</div>'
            '<div class="lp-pmeter"><b></b></div></div>'
            '<div class="lp-panel lp-p2"><div class="lp-ptag"><i></i>ANALYST &middot; 02</div><div class="lp-ptitle">Runs it in a sandbox</div>'
            '<div class="lp-pline">Q3 &middot; 2,481 rows</div><div class="lp-pline lp-dim">code, result and evidence logged</div>'
            '<div class="lp-pmeter"><b></b></div></div>'
            '<div class="lp-panel lp-p3"><div class="lp-ptag"><i></i>CRITIC &middot; 03</div><div class="lp-ptitle">Checks it independently</div>'
            '<div class="lp-pline">12 of 12 numbers reproduced</div><div class="lp-pline lp-dim">a second model, a second method</div>'
            '<div class="lp-pmeter"><b></b></div></div>'
            '<div class="lp-core-cap">Illustrative demonstration &middot; sample question, not a live analysis</div>'
            '</div></div>'
            '<div class="lm-cue"><span>Scroll</span><i></i></div>'
            '<div class="lp-strip"><span>CSV &amp; Excel</span><span>PostgreSQL</span><span>Sandboxed Python</span>'
            '<span>Audit trail</span><span>Scheduled reports</span></div>'
            '<div class="lp-sec"><div class="lp-eyebrow">Why Silt</div>'
            f'<h2 class="lp-h2">{_lp_words("An analyst that shows its work.")}</h2>'
            '<div class="lp-feats">'
            '<div class="lp-feat"><div class="lp-fvis"><div class="lp-chat"><span class="u">Which region grew fastest?</span>'
            '<span class="a">West, up 18.4% quarter over quarter.<i class="caret"></i></span></div></div>'
            '<b>Ask in plain English</b><p>No SQL, no dashboards to build. Type the question and the agent plans the analysis.</p></div>'
            '<div class="lp-feat"><div class="lp-fvis"><div class="lp-checks"><span><i>&#10003;</i>Totals match source</span>'
            '<span><i>&#10003;</i>No double counting</span><span><i>&#10003;</i>Dates align</span></div></div>'
            '<b>A second model checks it</b><p>An independent critic re-verifies every number before the answer reaches you.</p></div>'
            '<div class="lp-feat"><div class="lp-fvis"><div class="lp-log"><span><u></u>Loaded sales.csv</span>'
            '<span><u></u>Grouped by region</span><span><u></u>Compared Q2 vs Q3</span></div></div>'
            '<b>Every step is logged</b><p>See the plan, the code and the evidence behind each answer in the audit trail.</p></div>'
            '</div></div>'
            f'{_lp_feature_grid()}'
            '<div class="lp-sec"><div class="lp-eyebrow">How it works</div>'
            f'<h2 class="lp-h2">{_lp_words("From question to trusted answer.")}</h2>'
            '<div class="lp-steps"><div><em>1</em><b>Upload your data</b><p>CSV, Excel or a database.</p></div>'
            '<div><em>2</em><b>Ask a question</b><p>The planner agent writes the analysis.</p></div>'
            '<div><em>3</em><b>Run in a sandbox</b><p>Code executes in isolation.</p></div>'
            '<div><em>4</em><b>Verified answer</b><p>The critic signs off, then you see it.</p></div></div></div>'
            f'<div class="lm-finale"><div class="lm-marq"><div class="track">{_lp_ticker()}</div></div>'
            '<div class="lm-final-card"><div class="lp-eyebrow">Your turn</div>'
            f'<h2 class="lm-big">{_lp_words("Ask your")} <em>{_lp_words("first question.", 2)}</em></h2>'
            '<p>Create a workspace, upload a CSV and get a verified answer in minutes.</p>'
            '<div class="lm-cta"><a class="lp-btn dark" href="?view=signup" target="_self">Get Started <span class="arr">&rarr;</span></a>'
            '<a class="lp-btn light" href="?view=login" target="_self">Log in</a></div>'
            '<div class="lm-proof"><span>Runs in a secure sandbox</span><span>Every number verified</span><span>Fully auditable</span></div></div>'
            f'<div class="lm-marq rev"><div class="track">{_lp_ticker()}</div></div>'
            '<div class="lm-foot"><span>Silt &mdash; Autonomous Data Analyst</span><span>Verified / Auditable / Actionable</span></div></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="lp_scene"):
            st.iframe(LANDING_SCENE_HTML, height=200)
        components.html(_LP_REVEAL_JS, height=0)
        st.stop()

    # ---- sign-in page ----
    st.markdown(
        f'<div class="lp-nav"><a class="lp-brand" href="?" target="_self">{_GATE_LOGO}Silt</a>'
        '<div class="lp-nav-r"><a class="lp-ghost" href="?" target="_self">&larr; Back to home</a></div></div>'
        '<div class="lp-login-pad"></div>'
        '<div class="lp-blobs"><i class="b1"></i><i class="b2"></i><i class="b3"></i><i class="b4"></i><span class="g"></span></div>'
        '<div class="lp-chip lp-chip-l"><span class="ok">&#10003;</span>Checked by an independent model</div>'
        '<div class="lp-chip lp-chip-r"><b>12 / 12</b>numbers verified</div>'
        '<div class="lp-chip lp-chip-b"><span class="pulse"></span>Planner agent is thinking&hellip;</div>',
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
                    st.session_state["_go_home"] = True
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
                    st.session_state["_go_home"] = True
                    st.session_state.pop("_logged_out", None)
                    st.rerun()
                except ApiError as e:
                    st.error(f"Could not sign up: {e}")

    st.stop()
