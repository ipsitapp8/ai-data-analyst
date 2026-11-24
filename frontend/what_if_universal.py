"""What-if, any domain: choose a domain and a starting point, discover questions, build a scenario, run it, read the result.

Everything here calls the backend's /api/whatif routes. Calculation, validation and explanation happen there; this page
only collects inputs and shows results. It never needs the Copilot's language model to run a scenario."""
from __future__ import annotations

import re
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import (ApiError, analyst_ask, whatif_compare, whatif_delete, whatif_domains, whatif_get, whatif_profile, whatif_run, whatif_save,
                        whatif_saved, whatif_suggest)
from style.theme import esc, html, plot
from what_if_result import render as render_result

S = st.session_state
STATE = {"ready": ("Ready to simulate", "wx-ready"), "needs_assumptions": ("Needs assumptions", "wx-assume"),
         "needs_external_data": ("Needs external data", "wx-ext"), "unsupported": ("Unsupported", "wx-no")}
DOMAIN_UI = {"auto": ("travel_explore", "Detect from my data", "Silt reads your columns and says which domain fits, and why."),
             "business": ("storefront", "Business", "Pricing, cost, volume, marketing and headcount."),
             "education": ("school", "Education", "Marks, attendance rules, pass/fail, class capacity, enrolment."),
             "healthcare": ("local_hospital", "Healthcare", "Beds, staff, admissions and waiting times. Operational only, never clinical."),
             "travel": ("flight_takeoff", "Travel", "Trip budget, duration, accommodation and itinerary from values you type."),
             "government": ("account_balance", "Government & public services", "Population, service capacity and project budgets."),
             "custom": ("tune", "Custom", "Your own variables and formulas, or any column of any table.")}
START_UI = {"Explore suggestions": ("lightbulb", "See questions worth asking based on your data and domain.", ""),
            "Use my dataset": ("database", "Start from your data and real columns.", ""),
            "Build a scenario manually": ("edit_note", "Define your own values, assumptions and constraints.", ""),
            "Brainstorm with Silt AI": ("auto_awesome", "Describe your goal and get a ranked list of scenarios.", "")}
START = ["Explore suggestions", "Use my dataset", "Build a scenario manually", "Brainstorm with Silt AI"]
METHOD_LABEL = {"deterministic": "Exact calculation", "regression": "Estimate from my data", "forecast": "Trend forecast", "monte_carlo": "Monte Carlo simulation",
                "sensitivity": "Sensitivity analysis", "break_even": "Break-even", "goal_seek": "Goal seek"}


def _cached(key: tuple, fetch):
    """Remember a backend answer for this browser session, so typing into the builder does not refetch the domain list, the profile
    and the suggestions on every rerun. Anything that changes the answer (domain, dataset, objective) is part of the key."""
    cache = S.setdefault("wxu_cache", {})
    if key not in cache:
        cache[key] = fetch()
    return cache[key]


def _fmt(x) -> str:
    if x is None:
        return "—"
    return f"{x:,.0f}" if abs(x) >= 1000 or float(x).is_integer() else f"{x:,.2f}"


STEPS = ["1 Â· Setup", "2 Â· Questions", "3 Â· Scenario", "Saved"]
_KEEP = ("wxu_pick_domain", "wxu_pick_start", "wxu_ds", "wxu_obj")


def _goto(step: str) -> None:
    S["wxu_goto"] = step
    st.rerun()


def _context(datasets: list[dict]) -> tuple[str, str, dict | None, str | None]:
    """The choices made in Setup, read from session state so every step sees them even while Setup is not on screen."""
    domain = S.get("wxu_pick_domain", "auto")
    start = S.get("wxu_pick_start", START[0])
    can_use_data = start in ("Explore suggestions", "Use my dataset", "Brainstorm with Silt AI") or domain == "auto"
    ds = S.get("wxu_ds") if can_use_data and datasets else None
    objective = S.get("wxu_obj") if start == "Brainstorm with Silt AI" else None
    return domain, start, ds, objective


def _setup_step(datasets: list[dict], ui: dict) -> None:
    def _reset() -> None:
        S["wxu_sel"], S["wxu_result"] = None, None

    left, right = st.columns([1, 1], gap="medium")
    with left:
        with st.container(border=True, key="wxu_panel_domain"):
            html('<div class="wx-ph"><div><div class="wx-pk">Step 1a Â· Context</div><div class="wx-pt">Domain</div>'
                 '<div class="wx-ps">Choose the context for your analysis.</div></div></div>')
            domain = st.radio("Domain", list(ui), format_func=lambda k: ui[k][1], key="wxu_pick_domain", on_change=_reset, label_visibility="collapsed")
            html(f'<div class="wx-detail">{esc(ui[domain][2])}</div>')
    with right:
        with st.container(border=True, key="wxu_panel_start"):
            html('<div class="wx-ph"><div><div class="wx-pk">Step 1b Â· How to begin</div><div class="wx-pt">Starting point</div>'
                 '<div class="wx-ps">How would you like to begin your scenario?</div></div></div>')
            start = st.radio("Starting point", START, format_func=lambda k: k, captions=[START_UI[k][1] for k in START], key="wxu_pick_start",
                             on_change=_reset, label_visibility="collapsed")
    S["wxu_domain"], S["wxu_start"] = domain, start      # the names the rest of the page (and its tests) read
    needs_data = start == "Use my dataset" or (domain == "auto" and start != "Build a scenario manually")
    can_use_data = start in ("Explore suggestions", "Use my dataset", "Brainstorm with Silt AI") or domain == "auto"
    ds = None
    with st.container(border=True, key="wxu_panel_data"):
        html('<div class="wx-ph"><div><div class="wx-pk">Step 1c Â· Data</div><div class="wx-pt">Dataset</div>'
             '<div class="wx-ps">The table Silt reads to find questions it can answer.</div></div></div>')
        if can_use_data and datasets:
            none_label = "Choose a datasetâ€¦" if needs_data else "None: use the domain's templates (optional)"
            ds = st.selectbox("Dataset", [None] + datasets, format_func=lambda d: none_label if d is None else d["filename"], key="wxu_ds",
                              label_visibility="collapsed")
        elif can_use_data and needs_data:
            st.info("No dataset yet. Upload one on the Datasets page, or choose â€œBuild a scenario manuallyâ€.")
        else:
            st.caption("This starting point does not use a dataset.")
        if ds and start != "Build a scenario manually":
            _profile_panel(ds["id"], None if domain == "auto" else domain)
    if start == "Brainstorm with Silt AI":
        with st.container(border=True, key="wxu_panel_goal"):
            html('<div class="wx-ph"><div><div class="wx-pk">Step 1d Â· Goal</div><div class="wx-pt">What do you want to decide?</div>'
                 '<div class="wx-ps">Describe your role and goal, or ask a question about your data.</div></div></div>')
            st.text_area("Describe your role and what you want to decide", key="wxu_obj", height=80, label_visibility="collapsed",
                         placeholder="I am a college administrator. Help me identify important scenarios to evaluate before the next academic year.")
    manual = start == "Build a scenario manually" and domain == "custom"
    if st.button("Continue to scenario  â†’" if manual else "Continue to questions  â†’", type="primary", key="wxu_next_setup"):
        _goto(STEPS[2] if manual else STEPS[1])


def _questions_step(datasets: list[dict]) -> None:
    domain, start, ds, objective = _context(datasets)
    if start == "Build a scenario manually" and domain == "custom":
        st.info("Custom scenarios are defined by you, so there are no suggested questions.")
        if st.button("Go to the scenario builder  â†’", key="wxu_to_builder"):
            _goto(STEPS[2])
        return
    ds_id = ds["id"] if ds else None
    _suggestions(ds_id, None if domain == "auto" else domain, objective, start)


def _scenario_step(datasets: list[dict]) -> None:
    domain, start, ds, _ = _context(datasets)
    ds_id = ds["id"] if ds else None
    sel = S.get("wxu_sel")
    if sel and sel["id"] != "custom":
        if st.button("â†  Back to questions", key="wxu_back"):
            S["wxu_sel"], S["wxu_result"] = None, None
            _goto(STEPS[1])
        if str(sel["id"]).startswith("saved"):
            if S.get("wxu_result"):
                _result(S["wxu_result"], (S["wxu_scn"].get("dataset") or {}).get("dataset_id"), sel)
        else:
            _builder(sel, ds_id)
    elif start == "Build a scenario manually" and domain == "custom":
        _custom_builder()
    else:
        st.info("No question chosen yet. Pick one in step 2, or press â€œUseâ€ on a suggested question.")
        if st.button("â†  Go to questions", key="wxu_to_questions"):
            _goto(STEPS[1])


def explore_tab(datasets: list[dict]) -> None:
    try:
        domains = _cached(("domains",), whatif_domains)
    except ApiError as e:
        st.error(f"Backend unreachable: {e}")
        return
    ui = {**DOMAIN_UI}
    for d in domains:                                   # the backend's own domain list decides which domains exist
        ui.setdefault(d["key"], ("category", d["title"], d["description"]))
    for k in _KEEP:                                     # a widget that is not on screen would otherwise lose its value
        if k in S:
            S[k] = S[k]
    if S.get("wxu_goto"):
        S["wxu_step"] = S.pop("wxu_goto")
    elif S.get("wxu_step") is None:
        S["wxu_step"] = STEPS[0]
    step = st.segmented_control("Step", STEPS, key="wxu_step", label_visibility="collapsed", selection_mode="single") or STEPS[0]
    domain, start, ds, _ = _context(datasets)
    if step != STEPS[0]:
        st.caption(" Â· ".join(x for x in (ui.get(domain, ("", ""))[1], start, ds["filename"] if ds else None) if x) + "  â€”  change this in step 1")
    if step == STEPS[0]:
        _setup_step(datasets, ui)
    elif step == STEPS[1]:
        _questions_step(datasets)
    elif step == STEPS[2]:
        _scenario_step(datasets)
    else:
        _saved_panel()


def _profile_panel(ds_id: int, domain: str | None) -> None:
    try:
        p = _cached(("profile", ds_id, domain), lambda: whatif_profile(ds_id, domain))
    except ApiError as e:
        st.warning(str(e))
        return
    dom = p["domain"]
    with st.expander(f"What Silt understood about this data — {dom['key'].title()} ({dom['source']})"):
        st.write(dom["explanation"])
        st.caption(f"{p['rows']:,} rows · {len(p['columns'])} columns · {p['duplicates']} duplicate rows · {p['missing_cells']} blank cells")
        st.dataframe(pd.DataFrame([{"column": c["name"], "role": c["role"], "missing": c["missing"], "unique": c["unique"],
                                    **({"min": c["min"], "max": c["max"]} if "min" in c else {})} for c in p["columns"]]), hide_index=True, use_container_width=True)
        if p["relationships"]:
            st.markdown("**Relationships found**")
            for r in p["relationships"][:5]:
                st.caption(f"{r['a']} ↔ {r['b']}: correlation {r['pearson']:+.2f} over {r['n']} rows ({r['strength']})")
        if p["sensitive_columns"]:
            st.caption("Marked sensitive (values are never shown): " + ", ".join(p["sensitive_columns"]))


_DATA_Q = re.compile(r"^\s*(which|who|whom|whose|what|how many|how much|how|when|where|show|list|compare|find|rank|top)\b", re.I)


def _is_data_question(text: str | None) -> bool:
    """A question about the data itself ("which diagnosis has the highest average bill?"), as opposed to a what-if or a statement of a goal."""
    t = (text or "").strip()
    if not t or re.search(r"\bwhat\s+if\b|\bsuppose\b|\bscenario\b", t, re.I):
        return False
    return t.endswith("?") or bool(_DATA_Q.match(t))


def _answer_data_question(ds_id: int, question: str) -> None:
    """Answer a data question with the Copilot's analyst, because scenario suggestions alone leave it unanswered."""
    html('<div class="wx-section">Answer from your data</div>')
    try:
        r = _cached(("answer", ds_id, question), lambda: analyst_ask(ds_id, question, "whatif-brainstorm"))
    except ApiError as e:
        st.warning(f"This question could not be answered: {e}")
        return
    if r.get("error"):
        st.warning(r["error"])
        return
    st.markdown(r.get("answer") or "No answer was produced for this question.")
    if r.get("details"):
        with st.expander("How this was calculated"):
            st.markdown(r["details"])
    st.caption("Answered by the Copilot's analyst from your data. The scenario ideas below are what-if questions you could run next.")


def _suggestions(ds_id, domain, objective, start) -> None:
    if domain is None and ds_id is None:
        st.caption("Choose a domain or a dataset to see suggestions.")
        return
    if start == "Brainstorm with Silt AI" and not objective:
        st.caption("Describe your objective in step 1, then press Suggest.")
    go_btn = st.button("Suggest questions" if start != "Brainstorm with Silt AI" else "Suggest for my objective", key="wxu_go") if start == "Brainstorm with Silt AI" else True
    if not go_btn:
        return
    try:
        out = _cached(("suggest", ds_id, domain, objective, start == "Brainstorm with Silt AI"),
                      lambda: whatif_suggest(ds_id, domain, objective, use_model=start == "Brainstorm with Silt AI"))
    except ApiError as e:
        st.error(str(e))
        return
    meta = out["meta"]
    if start == "Brainstorm with Silt AI" and ds_id and _is_data_question(objective):
        _answer_data_question(ds_id, objective)
        meta = {**meta, "message": None}      # the "no scenario matches" notice is misleading once the question has been answered
    if meta.get("message"):
        st.info(meta["message"])
    if meta.get("llm") == "unavailable":
        st.caption("Silt AI is unavailable right now; these were ranked by rules and are just as valid.")
    html('<div class="wx-section">Questions to explore</div>')
    shown = len(out["suggestions"])
    st.caption(f"{shown} questions for {out['domain']['title']}. Each one says whether it can be answered now, needs something from you, or cannot be answered reliably.")
    for i, s in enumerate(out["suggestions"]):
        label, cls = STATE[s["feasibility"]]
        with st.container(key=f"wxu_row_{i}"):
            a, b = st.columns([6, 1], vertical_alignment="center")
            a.markdown(f'<div class="wx-q">{esc(s["question"])}</div>'
                       f'<div><span class="wx-state {cls}">{label}</span><span class="wx-meta">{esc(METHOD_LABEL.get(s["method"], s["method"]))}</span></div>'
                       f'<div class="wx-meta">{esc(s["reason"])}</div>'
                       + (f'<div class="wx-meta"><b>Needs</b> {esc("; ".join(s["missing_inputs"]))}</div>' if s["missing_inputs"] else "")
                       + (f'<div class="wx-meta"><b>Limit</b> {esc(s["limitations"][0])}</div>' if s["limitations"] else ""), unsafe_allow_html=True)
            if s["feasibility"] in ("ready", "needs_assumptions"):
                if b.button("Use", key=f"wxu_use_{s['id']}"):
                    S["wxu_sel"], S["wxu_result"] = s, None
                    _goto(STEPS[2])


def _inputs_form(sel: dict) -> tuple[dict, dict]:
    inputs, uncertainty = {}, {}
    groups = {"value": "Question", "baseline": "Starting values", "assumption": "Assumptions", "change": "What changes"}
    for role, title in groups.items():
        specs = [i for i in sel["inputs"] if i["role"] == role]
        if not specs:
            continue
        st.markdown(f"**{title}**")
        cols = st.columns(min(3, len(specs)))
        for col, i in zip(cols * 3, specs):
            key = f"wxu_in_{sel['id']}_{i['name']}"
            label = f"{i['label']}" + (f" ({i['unit']})" if i["unit"] else "") + (" — optional" if i["optional"] else "")
            v = col.number_input(label, min_value=i["min"], max_value=i["max"], value=i["default"], key=key, help=i["help"] or None, step=None, format="%g")
            if v is not None:
                inputs[i["name"]] = float(v)
    return inputs, uncertainty


def _builder(sel: dict, ds_id) -> None:
    html('<div class="wx-section">Scenario builder</div>')
    st.markdown(f"**{sel['question']}**")
    inputs, _ = _inputs_form(sel)
    methods = {"formula": ["deterministic", "monte_carlo", "goal_seek"]}
    method = sel["method"]
    if sel["template_id"].startswith(("hc_", "biz_", "gov_", "tr_", "edu_enrolment_capacity", "edu_faculty")) and method in ("deterministic", "monte_carlo", "sensitivity", "break_even"):
        opts = list(dict.fromkeys([method, "deterministic", "monte_carlo"] + (["sensitivity", "break_even"] if sel["template_id"].startswith("biz_") else [])))
        method = st.selectbox("Method", opts, index=opts.index(method), format_func=lambda m: METHOD_LABEL[m], key=f"wxu_method_{sel['id']}")
    uncertainty = {}
    if method == "monte_carlo":
        base_names = [i["name"] for i in sel["inputs"] if i["role"] == "baseline"]
        u1, u2, u3 = st.columns(3)
        who = u1.selectbox("Uncertain input", base_names, key=f"wxu_unc_{sel['id']}")
        sd = u2.number_input("Spread (standard deviation, % of its value)", 1.0, 100.0, 15.0, 1.0, key=f"wxu_sd_{sel['id']}")
        seed = u3.number_input("Seed (repeat a run exactly)", 0, 2**31 - 1, 42, key=f"wxu_seed_{sel['id']}")
        uncertainty = {who: {"type": "normal", "sd_pct": float(sd)}}
        st.caption("The spread is your assumption, not a measurement.")
    else:
        seed = None
    scenario = {"id": uuid.uuid4().hex[:12], "name": sel["question"][:100], "domain": sel["domain"], "template_id": sel["template_id"], "question": sel["question"],
                "method": method, "columns": {k: v for k, v in (S.get(f"wxu_cols_{sel['id']}") or {}).items()}, "inputs": inputs,
                "uncertainty": uncertainty, "seed": int(seed) if seed is not None else None, "assumptions": {}, "constraints": [], "target": None, "custom_model": None}
    if sel.get("columns"):
        scenario["columns"].update(sel["columns"])
    if method == "goal_seek":
        t1, t2, t3 = st.columns(3)
        scenario["target"] = {"output": t1.text_input("Output to reach (e.g. profit)", "profit", key=f"wxu_to_{sel['id']}"),
                              "value": t2.number_input("Target value", value=0.0, key=f"wxu_tv_{sel['id']}"),
                              "solve_for": t3.text_input("Input to solve for (e.g. price)", "price", key=f"wxu_ts_{sel['id']}")}
    if st.button("Run scenario", type="primary", key=f"wxu_run_{sel['id']}"):
        try:
            S["wxu_result"] = whatif_run(scenario, ds_id)
            S["wxu_scn"] = scenario
        except ApiError as e:
            st.error(str(e))
    res = S.get("wxu_result")
    if res:
        _result(res, ds_id, sel)


def _result(res: dict, ds_id, sel) -> None:
    v = res["validation"]
    if v["status"] != "valid":
        st.warning({"needs_input": "Silt needs a few more things before it can run this:", "invalid": "These inputs cannot be used:",
                    "unsupported": "This question cannot be answered by calculation:"}.get(v["status"], v["status"]))
        for e in v["errors"]:
            st.error(e)
        for m in v["missing_inputs"]:
            if m["kind"] == "column":
                pick = st.selectbox(m["label"], m["candidates"], key=f"wxu_pick_{m['name']}")
                S.setdefault(f"wxu_cols_{sel['id']}", {})[m["name"].split(":", 1)[1]] = [pick]
            else:
                st.markdown(f"- **{m['label']}** — {m.get('why', '')}")
        return
    render_result(res, S.get("wxu_scn") or {}, ds_id, sel)


def _custom_builder() -> None:
    html('<div class="ds-section-title" style="margin-top:8px;">Define your own model</div>')
    st.caption("Name your variables with starting values, then give each output a formula using those names. Allowed: + − × ÷, powers, min, max, abs, sqrt, log, exp.")
    vars_ = st.data_editor(pd.DataFrame({"name": ["price", "volume"], "label": ["Price", "Volume"], "baseline": [10.0, 100.0], "change_pct": [0.0, 0.0]}),
                           num_rows="dynamic", key="wxu_vars", use_container_width=True)
    outs = st.data_editor(pd.DataFrame({"name": ["revenue"], "label": ["Revenue"], "formula": ["price * volume"]}), num_rows="dynamic", key="wxu_outs", use_container_width=True)
    cons = st.text_area("Constraints (one per line: expression | message)", "price >= 1 | Price must be at least 1", key="wxu_cons", height=70)
    method = st.selectbox("Method", ["deterministic", "sensitivity"], format_func=lambda m: METHOD_LABEL[m], key="wxu_cm")
    vars_ = vars_.dropna(subset=["name"])
    outs = outs.dropna(subset=["name", "formula"])
    if st.button("Run my model", type="primary", key="wxu_cr") and len(vars_) and len(outs):
        defn = {"inputs": {r["name"]: {"label": r["label"] or r["name"], "unit": "", "min": 0} for _, r in vars_.iterrows()},
                "outputs": {r["name"]: {"label": r["label"] or r["name"], "unit": "", "formula": r["formula"]} for _, r in outs.iterrows()},
                "primary_output": outs.iloc[0]["name"],
                "constraints": [{"expr": a.strip(), "message": (b or a).strip(), "severity": "error"} for a, _, b in (ln.partition("|") for ln in cons.splitlines() if ln.strip())]}
        scenario = {"id": uuid.uuid4().hex[:12], "name": "Custom model", "domain": "custom", "template_id": "custom_model", "question": "My own model", "method": method,
                    "inputs": {}, "columns": {}, "uncertainty": {}, "assumptions": {}, "constraints": [], "target": None, "custom_model": defn,
                    "baseline": {r["name"]: float(r["baseline"]) for _, r in vars_.iterrows()},
                    "changes": {r["name"]: {"type": "pct", "value": float(r["change_pct"])} for _, r in vars_.iterrows() if r["change_pct"]}}
        try:
            S["wxu_result"], S["wxu_scn"] = whatif_run(scenario, None), scenario
            S["wxu_sel"] = {"id": "custom", "question": "My own model", "inputs": [], "domain": "custom", "template_id": "custom_model", "method": method}
        except ApiError as e:
            st.error(str(e))
    if S.get("wxu_result") and (S.get("wxu_sel") or {}).get("id") == "custom":
        _result(S["wxu_result"], None, S["wxu_sel"])
        S["wxu_sel"] = None


def _saved_panel() -> None:
    try:
        rows = whatif_saved()
    except ApiError:
        return
    html(f'<div class="wx-saved"><span class="wx-pico wx-i-bookmark"></span><div><div class="wx-pt">Saved scenarios ({len(rows)})</div>'
         f'<div class="wx-ps">{"Nothing saved yet. Run a scenario and press Save." if not rows else "Open one to review it, or tick two or more to compare."}</div></div></div>')
    if not rows:
        return
    picked = []
    for r in rows:
        a, b, c = st.columns([5, 1, 1])
        if a.checkbox(f"{r['name']} — {r.get('domain') or ''} · {r.get('evidence_kind') or ''}", key=f"wxu_cmp_{r['id']}"):
            picked.append(r["id"])
        if b.button("Open", key=f"wxu_open_{r['id']}"):
            full = whatif_get(r["id"])["scenario"]
            S["wxu_result"], S["wxu_scn"] = full, full
            S["wxu_sel"] = {"id": f"saved{r['id']}", "question": full.get("question") or r["name"], "inputs": [], "domain": full["domain"],
                            "template_id": full["template_id"], "method": full["method"]}
            _goto(STEPS[2])
        if c.button("Delete", key=f"wxu_del_{r['id']}"):
            whatif_delete(r["id"])
            st.rerun()
    if len(picked) >= 2 and st.button("Compare selected", key="wxu_compare"):
        try:
            cmp = whatif_compare(picked)
        except ApiError as e:
            st.error(str(e))
            return
        names = {s["id"]: s["name"] for s in cmp["scenarios"]}
        table = pd.DataFrame([{"Measure": m["label"], **{f"{names[i]} (simulated)": v for i, v in m["scenario"].items()}} for m in cmp["metrics"]])
        st.dataframe(table, hide_index=True, use_container_width=True)
        for n in cmp["notes"]:
            st.caption(n)
