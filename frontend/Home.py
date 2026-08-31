"""Landing page — full-bleed VERITY hero, no sidebar.

Everything here is presentational except the two CTAs and Book Demo, which
enter the real flow (dataset first, then question).
"""
from __future__ import annotations

import streamlit as st

from style.theme import asset_data_uri, html, page_setup

page_setup("Home", sidebar=False)

BG = asset_data_uri("hero-mesh.jpg")

st.markdown(
    f"""
<style>
:root {{
  --pink: #ef5f77;
  --olive: #a9b878;
  --olive-btn: #9dae6b;
  --cream: #f2f0ea;
  --muted: #97968f;
  --card: rgba(17,15,15,0.78);
  --line: rgba(255,255,255,0.085);
}}

/* Base colour goes on .stApp and the view container stays transparent -- an
   opaque view container paints over .stApp::before and hides the artwork. */
.stApp {{ background: #0a0908 !important; }}
[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
.block-container {{
  padding: 22px 40px 20px 40px !important;
  max-width: 1580px !important;
}}

/* ---- background artwork ---- */
.stApp::before {{
  content: '';
  position: fixed; inset: 0 0 0 30%;
  background-image: url("{BG}");
  background-size: cover;
  /* bias right so the bright pink/amber peaks land beside the cards rather
     than the darker green left edge of the plate */
  background-position: 70% 46%;
  /* hold the fade off until well past the headline column -- the reference
     keeps the left ~40% essentially pure black */
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, transparent 8%, rgba(0,0,0,0.22) 24%, #000 54%, #000 100%);
  mask-image: linear-gradient(90deg, transparent 0%, transparent 8%, rgba(0,0,0,0.22) 24%, #000 54%, #000 100%);
  animation: meshDrift 36s ease-in-out infinite alternate;
  transform-origin: 65% 50%;
  z-index: 0; pointer-events: none;
}}
@keyframes meshDrift {{
  from {{ transform: scale(1.00); }}
  to   {{ transform: scale(1.06) translate3d(-1%, -0.8%, 0); }}
}}
[data-testid="stMainBlockContainer"] {{ position: relative; z-index: 1; }}

/* ---------------- top nav ---------------- */
.v-logo {{ display:flex; align-items:center; gap:12px; }}
.v-logo-mark {{
  width:46px; height:46px; border-radius:50%; flex-shrink:0;
  background: radial-gradient(circle at 35% 30%, #6f7d43, #3d4724);
  border:1px solid rgba(169,184,120,0.45);
  display:flex; align-items:center; justify-content:center;
}}
.v-name {{
  font-size:1.62rem; font-weight:600; letter-spacing:0.14em;
  color:var(--cream); line-height:1; margin-bottom:4px;
}}
.v-name i {{ font-style:normal; color:var(--olive); }}
.v-tag {{ font-size:0.72rem; color:var(--muted); letter-spacing:0.05em; }}

.v-links {{ display:flex; gap:34px; align-items:center; height:46px; }}
.v-links span {{ color:#d6d4ce; font-size:0.94rem; cursor:default; white-space:nowrap; }}
.v-signin {{
  display:flex; align-items:center; justify-content:flex-end;
  height:46px; color:#d6d4ce; font-size:0.94rem; white-space:nowrap;
}}
.st-key-demo button {{
  background: transparent !important;
  border: 1px solid rgba(169,184,120,0.55) !important;
  color: #c3d18e !important; border-radius: 999px !important;
  padding: 0.62em 1.35em !important; font-size: 0.92rem !important;
  white-space: nowrap !important;
}}
.st-key-demo button:hover {{ background: rgba(169,184,120,0.10) !important; }}

/* ---------------- left column ---------------- */
.v-eyebrow {{
  display:flex; align-items:center; gap:9px; white-space:nowrap;
  color:var(--pink); font-size:0.78rem; font-weight:600;
  letter-spacing:0.16em; margin:34px 0 16px 0;
}}
.v-spark {{ animation: twinkle 3.4s ease-in-out infinite; }}
@keyframes twinkle {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.45;transform:scale(.82)}} }}

.v-head {{
  font-size: clamp(2.3rem, 0.9rem + 3.1vw, 4.2rem);
  font-weight:400; line-height:1.05; letter-spacing:-0.035em;
  color:var(--cream); margin:0 0 24px 0;
}}
.v-head .grad {{
  background: linear-gradient(95deg, var(--olive) 0%, #c8a97e 46%, var(--pink) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}}
.v-sub {{
  color:#bfbdb7; font-size:0.95rem; line-height:1.75;
  max-width:470px; margin-bottom:22px;
}}

.st-key-cta button {{
  border-radius:999px !important; font-size:0.95rem !important;
  padding:0.78em 1.35em !important; font-weight:500 !important;
  white-space:nowrap !important;
}}
.st-key-cta button[kind="primary"] {{
  background: var(--olive-btn) !important; color:#171a0e !important; border:none !important;
}}
.st-key-cta button[kind="secondary"] {{
  background: rgba(255,255,255,0.03) !important;
  border:1px solid rgba(255,255,255,0.22) !important; color:var(--cream) !important;
}}

.v-rule {{ height:1px; background:var(--line); margin:26px 0 16px 0; max-width:520px; }}
.v-feats {{ display:flex; max-width:600px; }}
.v-feat {{ display:flex; align-items:flex-start; gap:11px; padding-right:22px; }}
.v-feat + .v-feat {{ border-left:1px solid var(--line); padding-left:22px; }}
.v-feat-t {{ font-size:0.88rem; font-weight:600; color:var(--cream); line-height:1.3; white-space:nowrap; }}
.v-feat-s {{ font-size:0.78rem; color:var(--muted); line-height:1.3; white-space:nowrap; }}

/* ---------------- right cards ----------------
   Width is capped so the mesh stays visible to the RIGHT of the cards, as in
   the reference -- letting them fill the column buries the artwork. */
.v-cards {{ position:relative; max-width:364px; margin:16px 0 0 6px; }}
.v-card {{
  background:var(--card); border:1px solid var(--line);
  border-radius:16px; backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  box-shadow:0 40px 90px -40px rgba(0,0,0,0.9);
}}
.v-card-1 {{ padding:14px 16px 15px 16px; }}

.v-search {{
  display:flex; align-items:center; gap:10px;
  background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10);
  border-radius:10px; padding:8px 11px; margin-bottom:14px;
}}
.v-search-ic {{
  width:28px; height:28px; border-radius:8px; flex-shrink:0;
  background:rgba(169,184,120,0.16); border:1px solid rgba(169,184,120,0.30);
  display:flex; align-items:center; justify-content:center;
  animation: glowPulse 3.2s ease-in-out infinite;
}}
@keyframes glowPulse {{
  0%,100% {{ box-shadow:0 0 0 0 rgba(169,184,120,0.30); }}
  50%     {{ box-shadow:0 0 13px 3px rgba(169,184,120,0.17); }}
}}
.v-search-t {{ color:var(--cream); font-size:0.86rem; white-space:nowrap; }}

.v-label {{
  color:var(--pink); font-size:0.64rem; font-weight:600;
  letter-spacing:0.15em; margin-bottom:7px;
}}
.v-steps {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.v-step {{
  display:flex; align-items:center; gap:12px; padding:6px 12px;
  opacity:0; animation: rowIn .5s ease forwards;
}}
.v-step + .v-step {{ border-top:1px solid var(--line); }}
.v-step:nth-child(1) {{ animation-delay:.15s }}
.v-step:nth-child(2) {{ animation-delay:.42s }}
.v-step:nth-child(3) {{ animation-delay:.69s }}
.v-step:nth-child(4) {{ animation-delay:.96s }}
@keyframes rowIn {{ from{{opacity:0;transform:translateY(7px)}} to{{opacity:1;transform:none}} }}
.v-step-n {{ color:var(--muted); font-size:0.75rem; font-variant-numeric:tabular-nums; }}
.v-step-t {{ color:var(--cream); font-size:0.82rem; flex:1; white-space:nowrap; }}

.v-insight {{ font-size:1.04rem; color:var(--cream); font-weight:500;
              letter-spacing:-0.015em; white-space:nowrap; }}
.v-insight b {{ color:var(--pink); font-weight:600; }}
.v-driver {{ color:#adaba5; font-size:0.78rem; margin-top:5px; white-space:nowrap; }}

.v-card-2 {{ padding:12px 15px 10px 15px; width:114%; margin:-10px 0 0 12px; }}
.v-chart-l {{ color:#adaba5; font-size:0.64rem; letter-spacing:0.13em; margin-bottom:4px; }}
.v-draw {{ stroke-dasharray:1000; stroke-dashoffset:1000; animation: draw 2.1s ease-out .5s forwards; }}
@keyframes draw {{ to {{ stroke-dashoffset:0; }} }}
.v-endpt {{ animation: endPulse 2.2s ease-in-out 2.4s infinite; }}
@keyframes endPulse {{ 0%,100%{{r:4.7;opacity:1}} 50%{{r:6.6;opacity:.65}} }}
.v-verified {{
  display:flex; align-items:center; gap:9px; white-space:nowrap;
  border-top:1px solid var(--line); padding-top:9px; margin-top:4px;
}}
.v-verified b {{ color:var(--olive); font-size:0.8rem; letter-spacing:0.09em; font-weight:600; }}
.v-verified span {{ color:#adaba5; font-size:0.8rem; }}

/* right-edge legend */
.v-legend {{ position:fixed; right:30px; bottom:76px; z-index:2; }}
.v-legend div {{
  display:flex; align-items:center; gap:8px; padding:3px 0;
  color:var(--pink); font-size:0.68rem; letter-spacing:0.13em;
}}
.v-legend i {{ width:5px; height:5px; border-radius:50%; background:var(--pink);
               display:inline-block; animation: blink 3s ease-in-out infinite; }}
.v-legend div:nth-child(2) i {{ animation-delay:.4s }}
.v-legend div:nth-child(3) i {{ animation-delay:.8s }}
.v-legend div:nth-child(4) i {{ animation-delay:1.2s }}
.v-legend div:nth-child(5) i {{ animation-delay:1.6s }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.28}} }}

/* Narrow viewports can't hold the reference composition -- shed the pieces
   that would otherwise overlap instead of letting them collide. */
@media (max-width: 1450px) {{ .v-legend {{ display:none; }} }}
@media (max-width: 1250px) {{ .stApp::before {{ opacity:0.45; }} }}
@media (max-width: 1180px) {{
  .v-links {{ display:none; }}
  .st-key-cta [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; gap:12px; }}
  .st-key-cta [data-testid="stColumn"] {{ flex:1 1 200px !important; min-width:200px; }}
  .v-cards {{ max-width:100%; }}
  .v-card-2 {{ width:100%; margin-left:0; }}
}}
@media (max-width: 900px) {{
  .v-feats {{ flex-wrap:wrap; gap:14px; }}
  .v-feat + .v-feat {{ border-left:none; padding-left:0; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

CHECK = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#cbc9c3" '
         'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
         '<path d="M4.5 12.5l5 5 10-11"/></svg>')

# ---------------------------------------------------------------- top nav --
nav_l, nav_c, nav_r = st.columns([1.65, 2.15, 1.25], gap="small")
with nav_l:
    html(
        """
        <div class="v-logo">
          <div class="v-logo-mark">
            <svg width="23" height="23" viewBox="0 0 24 24" fill="none">
              <path d="M12 3.4c3.4 1.5 6 1.6 6 1.6v6.2c0 4.3-2.7 7.6-6 9.4-3.3-1.8-6-5.1-6-9.4V5c0 0 2.6-.1 6-1.6z"
                    stroke="#c3d18e" stroke-width="1.5" stroke-linejoin="round"/>
              <path d="M12 8.2v7.4M12 8.2c-1.6 0-2.6 1-2.6 2.4S10.4 13 12 13" stroke="#c3d18e"
                    stroke-width="1.5" stroke-linecap="round"/>
            </svg>
          </div>
          <div>
            <div class="v-name">VER<i>i</i>TY</div>
            <div class="v-tag">Autonomous Data Intelligence</div>
          </div>
        </div>
        """
    )
with nav_c:
    html(
        '<div class="v-links"><span>Product</span><span>Solutions</span>'
        "<span>Pricing</span><span>Documentation</span></div>"
    )
with nav_r:
    s_l, s_r = st.columns([1, 1.2])
    with s_l:
        html('<div class="v-signin">Sign In</div>')
    with s_r:
        with st.container(key="demo"):
            if st.button("Book Demo", key="btn_demo"):
                st.switch_page("pages/1_Overview.py")

# ------------------------------------------------------------ hero body --
left, right = st.columns([1.06, 1.0], gap="medium")

with left:
    html(
        """
        <div class="v-eyebrow">
          <svg class="v-spark" width="14" height="14" viewBox="0 0 24 24" fill="#ef5f77">
            <path d="M12 2l1.9 6.6L20 12l-6.1 3.4L12 22l-1.9-6.6L4 12l6.1-3.4z"/>
          </svg>
          AUTONOMOUS AI DATA ANALYST
        </div>
        <h1 class="v-head">From Data to<br/><span class="grad">Decisions.</span></h1>
        <div class="v-sub">
          Ask any business question. Our AI agent plans the analysis, executes the code,
          verifies the result, and turns your data into an answer you can trust.
        </div>
        """
    )

    with st.container(key="cta"):
        # narrow button columns so the two CTAs sit close together as in the
        # reference -- wide columns leave a large dead gap between them
        c1, c2, _ = st.columns([0.82, 0.76, 1.72])
        with c1:
            if st.button("⬆   Upload your data", key="btn_upload", type="primary"):
                st.switch_page("pages/2_Datasets.py")
        with c2:
            if st.button("🔍   Ask a question", key="btn_ask"):
                st.switch_page("pages/3_Analyses.py")

    html(
        """
        <div class="v-rule"></div>
        <div class="v-feats">
          <div class="v-feat">
            <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="#ef5f77" stroke-width="1.5">
              <path d="M9.5 4.2A2.7 2.7 0 0 0 6.8 7c-1.3.3-2.3 1.5-2.3 2.9 0 .6.2 1.2.5 1.7-.6.5-1 1.3-1 2.2 0 1.4 1 2.6 2.4 2.8.2 1.4 1.4 2.4 2.8 2.4.5 0 1-.1 1.4-.4V4.6c-.3-.2-.7-.4-1.1-.4z"
                    stroke-linejoin="round"/>
              <path d="M14.5 4.2A2.7 2.7 0 0 1 17.2 7c1.3.3 2.3 1.5 2.3 2.9 0 .6-.2 1.2-.5 1.7.6.5 1 1.3 1 2.2 0 1.4-1 2.6-2.4 2.8-.2 1.4-1.4 2.4-2.8 2.4-.5 0-1-.1-1.4-.4V4.6c.3-.2.7-.4 1.1-.4z"
                    stroke-linejoin="round"/>
            </svg>
            <div><div class="v-feat-t">Autonomous</div><div class="v-feat-s">Plans &amp; analyzes</div></div>
          </div>
          <div class="v-feat">
            <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="#ef5f77" stroke-width="1.5">
              <path d="M12 3.2l7.2 2.6v5.6c0 4.3-3 7.7-7.2 9.4-4.2-1.7-7.2-5.1-7.2-9.4V5.8z" stroke-linejoin="round"/>
              <path d="M8.7 12.1l2.4 2.4 4.4-4.7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div><div class="v-feat-t">Verified</div><div class="v-feat-s">Checks &amp; validates</div></div>
          </div>
          <div class="v-feat">
            <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="#ef5f77" stroke-width="1.5">
              <rect x="5" y="10.4" width="14" height="9.4" rx="2.2"/>
              <path d="M8.3 10.4V7.7a3.7 3.7 0 0 1 7.4 0v2.7" stroke-linecap="round"/>
              <circle cx="12" cy="15.1" r="1.25" fill="#ef5f77" stroke="none"/>
            </svg>
            <div><div class="v-feat-t">Auditable</div><div class="v-feat-s">Insights you can trust</div></div>
          </div>
        </div>
        """
    )

with right:
    html(
        f"""
        <div class="v-cards">
          <div class="v-card v-card-1">
            <div class="v-search">
              <div class="v-search-ic">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#c3d18e" stroke-width="2">
                  <circle cx="11" cy="11" r="6.2"/><path d="M15.6 15.6L20 20" stroke-linecap="round"/>
                </svg>
              </div>
              <div class="v-search-t">Why did revenue drop in Q3?</div>
            </div>

            <div class="v-label">ANALYSIS PROCESS</div>
            <div class="v-steps">
              <div class="v-step"><span class="v-step-n">01</span>
                <span class="v-step-t">Understand data</span>{CHECK}</div>
              <div class="v-step"><span class="v-step-n">02</span>
                <span class="v-step-t">Plan analysis</span>{CHECK}</div>
              <div class="v-step"><span class="v-step-n">03</span>
                <span class="v-step-t">Execute code</span>{CHECK}</div>
              <div class="v-step"><span class="v-step-n">04</span>
                <span class="v-step-t">Verify results</span>{CHECK}</div>
            </div>

            <div class="v-label" style="margin-top:20px;">VERIFIED INSIGHT</div>
            <div class="v-insight">Revenue dropped <b>11.8%</b> in Q3.</div>
            <div class="v-driver">Primary driver: Electronics — North region</div>
          </div>

          <div class="v-card v-card-2">
            <div class="v-chart-l">REVENUE OVER TIME</div>
            <svg viewBox="0 0 430 152" style="width:100%;height:auto;overflow:visible;">
              <defs>
                <linearGradient id="vfill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#ef5f77" stop-opacity="0.34"/>
                  <stop offset="100%" stop-color="#ef5f77" stop-opacity="0.02"/>
                </linearGradient>
              </defs>
              <g fill="#8d8b86" font-size="9.5" font-family="Inter, sans-serif" text-anchor="end">
                <text x="44" y="16">₹12Cr</text><text x="44" y="44">₹10Cr</text>
                <text x="44" y="72">₹8Cr</text><text x="44" y="100">₹6Cr</text>
                <text x="44" y="128">₹4Cr</text>
              </g>
              <path d="M66,105 L112,89 L158,75 L204,79 L250,56 L296,28 L342,44 L388,67 L410,100
                       L410,134 L66,134 Z" fill="url(#vfill)"/>
              <polyline class="v-draw" points="66,105 112,89 158,75 204,79 250,56 296,28 342,44 388,67 410,100"
                        fill="none" stroke="#ef5f77" stroke-width="2.1"
                        stroke-linecap="round" stroke-linejoin="round"/>
              <g fill="#ef5f77">
                <circle cx="66" cy="105" r="3.1"/><circle cx="112" cy="89" r="3.1"/>
                <circle cx="158" cy="75" r="3.1"/><circle cx="204" cy="79" r="3.1"/>
                <circle cx="250" cy="56" r="3.1"/><circle cx="296" cy="28" r="3.1"/>
                <circle cx="342" cy="44" r="3.1"/><circle cx="388" cy="67" r="3.1"/>
              </g>
              <circle class="v-endpt" cx="410" cy="100" r="4.7" fill="#ef5f77"
                      stroke="rgba(239,95,119,0.35)" stroke-width="4.5"/>
              <g fill="#8d8b86" font-size="9.5" font-family="Inter, sans-serif" text-anchor="middle">
                <text x="80" y="150">Apr</text><text x="146" y="150">May</text>
                <text x="212" y="150">Jun</text><text x="278" y="150">Jul</text>
                <text x="344" y="150">Aug</text><text x="410" y="150">Sep</text>
              </g>
            </svg>
            <div class="v-verified">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a9b878" stroke-width="1.6">
                <path d="M12 3.4l6.6 2.4v5.2c0 3.9-2.7 7-6.6 8.6-3.9-1.6-6.6-4.7-6.6-8.6V5.8z" stroke-linejoin="round"/>
                <path d="M9.1 12l2.2 2.2 4-4.3" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              <b>VERIFIED</b><span>All results validated</span>
            </div>
          </div>
        </div>
        """
    )

html(
    """
    <div class="v-legend">
      <div><i></i>DATA</div><div><i></i>ANALYTICS</div><div><i></i>INSIGHT</div>
      <div><i></i>PREDICTION</div><div><i></i>MODEL</div>
    </div>
    """
)
