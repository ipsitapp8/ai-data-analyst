"""Landing page — hero, capability strip, how it works, and a live workspace snapshot.

The dataset/analysis counts and recent activity come straight from the
backend, same as the Overview page. The hero stage is drawn in CSS (no image asset).
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from api_client import ApiError, list_datasets, list_questions
from hero_scene import SCENE_HTML
from home_dynamic import GENERIC_QUESTIONS, has_active_run, suggestions
from style.theme import esc, html, page_setup, render_sidebar, verdict_badge

page_setup("Home")
render_sidebar("home")

# Page-scoped skin (style/home.css): every rule hangs off .hm-scope, so no other page is affected.
_FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
          '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500'
          '&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">')
_css = (Path(__file__).resolve().parent / "style" / "home.css").read_text(encoding="utf-8")
# one line, because markdown ends an HTML block at the first blank line
st.markdown(_FONTS + "<style>" + " ".join(line.strip() for line in _css.splitlines() if line.strip()) + "</style>"
            '<div class="hm-scope"></div>', unsafe_allow_html=True)

# Reveal-as-you-scroll. Runs in Streamlit's same-origin component iframe and reaches up to the page: marks the lower
# sections, shows each one as it comes into view, and grows the timeline line with the scroll. Elements are hidden only
# after they are marked, so if this does not run (or the browser lacks IntersectionObserver) the page is just fully visible.
_REVEAL_JS = """
<script>
(function () {
  var w, d;
  try { w = window.parent; d = w.document; } catch (e) { return; }
  if (!('IntersectionObserver' in w)) return;
  if (w.__hmStop) { try { w.__hmStop(); } catch (e) {} }
  w.__hmIntroAt = Date.now();
  var SEL = '.h-strip-eyebrow,.h-strip>div,.h-feat-eyebrow,.h-feat-title,.h-feat,.h-how-eyebrow,.h-how-title,.h-step,.h-snap,.h-tile,.h-snap-row,.h-snap-foot,.h-footer';
  var INTRO = ['.h-badge', '.h-hero', '.h-lede', '[class*="st-key-home_ctas"]', '.h-try', '[class*="st-key-home_chips"]', '.h-proof'];
  var io = new w.IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('hm-in'); io.unobserve(e.target); } });
  }, { threshold: 0.15, rootMargin: '0px 0px -5% 0px' });
  function intro() {
    if (Date.now() - w.__hmIntroAt > 2500) return;            // only the first moments after the page opens
    INTRO.forEach(function (sel, i) {
      d.querySelectorAll(sel).forEach(function (el) {
        if (el.__hmi) return; el.__hmi = 1;
        el.style.animationDelay = (i * 130) + 'ms'; el.classList.add('hm-intro');
      });
    });
  }
  function scan() {
    intro();
    var H = w.innerHeight;
    d.querySelectorAll(SEL).forEach(function (el) {
      if (el.__hm) return;
      el.__hm = 1;
      var par = el.parentElement, idx = par ? Array.prototype.indexOf.call(par.children, el) : 0;
      if (/(^|\\s)(h-tile|h-snap-row|h-feat)(\\s|$)/.test(el.className) || (par && /(^|\\s)h-strip(\\s|$)/.test(par.className))) el.style.transitionDelay = (idx * 90) + 'ms';
      if (el.classList.contains('h-feat') && !el.__hmm) {                // a soft light that follows the cursor over each card
        el.__hmm = 1;
        el.addEventListener('mousemove', function (e) { var b = el.getBoundingClientRect(); el.style.setProperty('--mx', (e.clientX - b.left) + 'px'); el.style.setProperty('--my', (e.clientY - b.top) + 'px'); });
      }
      el.classList.add('hm-r');
      if (el.getBoundingClientRect().top < H * 0.92) { el.classList.add('hm-now', 'hm-in'); }   // already on screen (or passed): no animation
      else io.observe(el);
    });
  }
  function line() {
    d.querySelectorAll('.h-feat[data-par]').forEach(function (el) {   // columns drift at different speeds: depth
      var r = el.getBoundingClientRect(), off = r.top + r.height / 2 - w.innerHeight / 2;
      el.style.setProperty('--py', (-off * parseFloat(el.getAttribute('data-par'))).toFixed(1) + 'px');
    });
    var s = d.querySelector('.h-steps'); if (!s) return;
    var r = s.getBoundingClientRect(), p = (w.innerHeight * 0.62 - r.top) / Math.max(1, r.height);
    s.style.setProperty('--hm-p', Math.max(0, Math.min(1, p)).toFixed(3));
  }
  var raf = 0;
  function tick() { if (raf) return; raf = w.requestAnimationFrame(function () { raf = 0; scan(); line(); }); }
  var mo = new w.MutationObserver(tick);
  mo.observe(d.body, { childList: true, subtree: true });
  d.addEventListener('scroll', tick, true);
  w.addEventListener('resize', tick);
  w.__hmStop = function () { io.disconnect(); mo.disconnect(); d.removeEventListener('scroll', tick, true); w.removeEventListener('resize', tick); };
  tick();
})();
</script>"""
components.html(_REVEAL_JS, height=0)

_has_team = bool(st.session_state.get("active_team_id"))
_REFRESH = "10s"          # how often the live panels re-read the backend
_MAX_AGE = 8.0            # seconds; two panels refreshing together share one fetch


def _load() -> tuple[list, list]:
    """(datasets, questions) for the active team, re-fetched at most every _MAX_AGE seconds."""
    if not _has_team:
        return [], []
    key = ("home_data", st.session_state.get("active_team_id"))
    cached = st.session_state.get("home_cache")
    if cached and cached["key"] == key and time.time() - cached["at"] < _MAX_AGE:
        return cached["datasets"], cached["questions"]
    try:
        datasets = list_datasets()
    except ApiError:
        datasets = cached["datasets"] if cached and cached["key"] == key else []
    try:
        questions = list_questions()
    except ApiError:
        questions = cached["questions"] if cached and cached["key"] == key else []
    st.session_state["home_cache"] = {"key": key, "at": time.time(), "datasets": datasets, "questions": questions}
    return datasets, questions


def _verified_of(questions: list) -> list:
    return [q for q in questions if q.get("verdict_state") in ("VERIFIED", "VERIFIED_WITH_CAVEATS")
            or (not q.get("verdict_state") and q["status"] == "verified")]


_datasets, _questions = _load()
_verified = _verified_of(_questions)


def _icon(body: str, size: int = 26) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


I_UPLOAD = _icon('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M12 17v-5M9.5 14.5L12 12l2.5 2.5"/>')
I_PLAN = _icon('<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>')
I_CODE = _icon('<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 5l-4 14"/>')
I_VERIFY = _icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>')
I_CHART = _icon('<path d="M5 20V11M12 20V4M19 20v-6"/>')
I_DB = _icon('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>', 16)
I_SPARK = _icon('<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>', 22)

I_SHIELD = _icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6z"/>', 17)
I_BOX = _icon('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5L12 12l8-4.5M12 12v9"/>', 17)
I_LOG = _icon('<path d="M6 4h12v16H6z"/><path d="M9 9h6M9 13h6M9 17h3"/>', 17)
I_DASH = _icon('<rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/>', 17)
I_FILE = _icon('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>', 17)

EXAMPLE_QUESTIONS = GENERIC_QUESTIONS


def _steps() -> str:
    data = [
        (I_UPLOAD, "Upload", "Add your CSV or<br>connect your database."),
        (I_PLAN, "Plan", "The agent analyses your<br>question and creates a plan."),
        (I_CODE, "Execute", "Writes and runs code<br>in a secure sandbox."),
        (I_VERIFY, "Verify", "Independent model<br>checks the results."),
        (I_CHART, "Visualize", "Get a clean dashboard<br>and actionable insights."),
    ]
    parts = []
    for i, (ico, name, desc) in enumerate(data):
        parts.append(f'<div class="h-step"><div class="h-step-ico">{ico}</div>'
                     f'<div class="h-step-name">{name}</div><div class="h-step-desc">{desc.replace("<br>", " ")}</div></div>')
        if i < len(data) - 1:
            parts.append('<div class="h-arrow">→</div>')
    return "".join(parts)



@st.fragment(run_every=_REFRESH if _has_team else None)
def _hero_stage() -> None:
    datasets, questions = _load()
    verified = _verified_of(questions)
    pool = [q for _, q in suggestions(datasets)] or list(EXAMPLE_QUESTIONS)
    sub = (f"<b>of {len(questions)}</b> run in this workspace" if questions
           else "<b>ready</b> when you are")
    running = ('<div class="h-running"><span class="silt-dot silt-dot-on"></span>Analysis running</div>'
               if has_active_run(questions) else "")
    nxt = "".join(f"<div>{esc(q)}</div>" for q in pool[1:4])
    html(
        f"""
        <div class="h-stage">
          <div class="h-float h-ask">
            <div class="h-ask-head"><div class="h-ask-ico">{I_SPARK}</div>
              <div class="h-ask-q"><small>Ask Silt</small>{esc(pool[0])}</div></div>
            <div class="h-ask-list">{nxt}</div>
          </div>
          <div class="h-float h-metric">
            <div class="lbl">Analyses verified</div>
            <div class="num h-count" style="--n:{len(verified)}" aria-label="{len(verified)}"></div>
            <div class="sub">{sub}</div>
            {running}
          </div>
          <div class="h-stage-title">Every number,<br>checked twice.</div>
        </div>
        """
    )


@st.fragment(run_every=_REFRESH if _has_team else None)
def _chips() -> None:
    """Clickable suggested questions, drawn from the newest datasets."""
    picks = suggestions(_load()[0])
    html('<div class="h-try">Try asking</div>')
    with st.container(key="home_chips"):
        for i, (ds_id, q) in enumerate(picks):
            if st.button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state["prefill_question"] = q
                if ds_id is not None:
                    st.session_state["active_dataset_id"] = ds_id
                st.switch_page("pages/3_Analyses.py")


@st.fragment(run_every=_REFRESH if _has_team else None)
def _snapshot() -> None:
    datasets, questions = _load()
    verified = _verified_of(questions)
    if questions:
        rows = ""
        for item in questions[:3]:
            tag = (verdict_badge(item["verdict_state"]) if item.get("verdict_state") else
                   f'<span class="ds-badge ds-badge-neutral">{esc(item["status"].replace("_", " ").title())}</span>')
            rows += f'<div class="h-snap-row"><div class="t">{esc(item["text"][:46])}</div>{tag}</div>'
        foot = rows
    else:
        foot = (f'<div class="h-snap-foot"><div class="h-snap-ico">{I_SPARK}</div>'
                '<div>No analyses yet — upload a CSV and ask your first question.</div></div>')
    live = "Live" if _has_team else "Idle"
    html(
        f"""
        <div class="h-snap">
          <div class="h-snap-head"><span>Workspace Snapshot</span>
            <span class="h-snap-live"><span class="silt-dot silt-dot-on"></span>{live}</span></div>
          <div class="h-snap-tiles">
            <div class="h-tile"><div class="h-tile-num h-count" style="--n:{len(datasets)}" aria-label="{len(datasets)}"></div><div class="h-tile-lbl">{I_DB}Datasets</div></div>
            <div class="h-tile"><div class="h-tile-num h-count" style="--n:{len(questions)}" aria-label="{len(questions)}"></div><div class="h-tile-lbl">{_icon('<path d="M5 20V11M12 20V4M19 20v-6"/>', 16)}Analyses</div></div>
            <div class="h-tile"><div class="h-tile-num h-count" style="--n:{len(verified)}" aria-label="{len(verified)}"></div><div class="h-tile-lbl">{_icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6z"/>', 16)}Verified</div></div>
          </div>
          {foot}
        </div>
        """
    )


# ------------------------------------------------------------------- hero --
if not _has_team:
    st.info("You're not on a team yet — create or join one to upload data and run analyses.")

with st.container(key="home_hero"):
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        html('<div class="h-badge"><span class="tag">AGENTIC</span>Your autonomous data analyst</div>')
        html('<div class="h-hero">Ask your data anything.<br><span>Trust every answer.</span></div>')
        html('<div class="h-lede">An AI agent plans the analysis, writes and runs the code in a sandbox, '
             'and a second, <em>independent model checks the result</em> before it reaches you.</div>')
        with st.container(key="home_ctas"):
            b1, b2, _ = st.columns([1.15, 1, 0.5])
            with b1:
                if _has_team:
                    if st.button("New analysis", type="primary", key="btn_ask", use_container_width=True):
                        st.switch_page("pages/3_Analyses.py")
                elif st.button("Go to Workspaces", type="primary", key="btn_ws", use_container_width=True):
                    st.switch_page("pages/8_Workspaces.py")
            with b2:
                if st.button("Upload data", key="btn_upload", use_container_width=True):
                    st.switch_page("pages/2_Datasets.py")
        if _has_team:
            _chips()
        html('<div class="h-proof"><span>Runs in a secure sandbox</span><span>Every number verified</span>'
             '<span>Fully auditable</span></div>')
    with right:
        with st.container(key="home_stage"):
            st.iframe(SCENE_HTML, height=470)   # canvas scene; cards below float over it
            _hero_stage()

html('<div class="hm-cue"><span>Scroll</span><i></i></div>')

# ------------------------------------------------------ capability strip --
html(
    '<div class="h-strip-eyebrow">Built for answers you can defend</div>'
    '<div class="h-strip">'
    f'<div>{I_FILE}CSV upload</div><div>{I_BOX}Sandboxed code</div><div>{I_SHIELD}Independent critic</div>'
    f'<div>{I_LOG}Audit trail</div><div>{I_DASH}Dashboards</div></div>'
)

def _words(text: str) -> str:
    """A title split into words, so each can rise in turn when the title scrolls into view."""
    return " ".join(f'<span class="w" style="--i:{i}">{esc(w)}</span>' for i, w in enumerate(text.split()))


# ------------------------------------------------------------ what is inside --
_FEATS = [
    ("Analyses", "Ask in plain English", "Type a question. An agent plans the analysis, writes the code and runs it in a sandbox."),
    ("Verification", "A second model checks it", "An independent critic re-verifies every number before the answer reaches you."),
    ("Root cause", "Find where a number moved", "Compare two periods and see which segments drove the change. Every figure is computed from your data."),
    ("What-if", "Test a decision first", "Change an input and see the range of outcomes, with goal-seek and the evidence behind each result."),
    ("Copilot", "Chat with your data", "Ask follow-ups in plain language and get answers with the numbers behind them."),
    ("Audit trail", "Every answer shows its work", "The plan, code, result and checks are kept for each dashboard, so anyone can inspect them."),
]
_cards = "".join(
    f'<div class="h-feat" data-par="{(0, 0.03, 0.06)[i % 3]}"><div class="k">{i + 1:02d} · {esc(tag)}</div>'
    f'<div class="t">{esc(title)}</div><p>{esc(text)}</p></div>'
    for i, (tag, title, text) in enumerate(_FEATS))
html('<div class="h-feat-eyebrow">What is inside</div>'
     f'<div class="h-feat-title">{_words("One workspace for asking, checking and deciding.")}</div>'
     f'<div class="h-feats">{_cards}</div>')

# ------------------------------------------------- how it works + snapshot --
how_l, snap_r = st.columns([1.75, 1.0], gap="large")
with how_l:
    html('<div class="h-how-eyebrow">How it works</div>')
    html(f'<div class="h-how-title">{_words("From raw data to real insights.")}</div>')
    html(f'<div class="h-steps">{_steps()}</div>')

with snap_r:
    _snapshot()

html('<div class="h-footer"><div>Silt <b>—</b> Autonomous Data Analyst</div>'
     '<div>Verified &nbsp;/&nbsp; Auditable &nbsp;/&nbsp; Actionable</div></div>')
