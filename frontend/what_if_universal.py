"""What-if, any domain: choose a domain and a starting point, then follow that starting point's own workflow to a result.

    Explore suggestions        discover the domain's questions, grouped by what Silt can calculate
    Use my dataset             inspect a table's columns and pick a question built from them
    Build a scenario manually  define objective, variables, values, formulas and constraints (what_if_manual)
    Brainstorm with Silt AI    describe a goal; a language model proposes ranked ideas, or the page says it could not

Everything here calls the backend's /api/whatif routes. Calculation, validation and explanation happen there; this page
only collects inputs and shows results. It never needs the Copilot's language model to run a scenario."""
from __future__ import annotations

import re
import uuid

import pandas as pd
import streamlit as st

import what_if_manual as manual
from api_client import (ApiError, analyst_ask, whatif_brainstorm, whatif_compare, whatif_delete, whatif_domains, whatif_get, whatif_profile,
                        whatif_run, whatif_saved, whatif_suggest)
from style.theme import esc, html
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
            "Brainstorm with Silt AI": ("lightbulb", "Describe your goal and get a ranked list of scenarios.", "")}
START = ["Explore suggestions", "Use my dataset", "Build a scenario manually", "Brainstorm with Silt AI"]
EXPLORE, DATASET, MANUAL, BRAINSTORM = START
# what steps 2 and 3 are called, and the way on from step 1, for each starting point
FLOW = {EXPLORE: ("2 · Suggestions", "3 · Scenario", "Continue to suggestions  →", "suggestions"),
        DATASET: ("2 · Your data", "3 · Scenario", "Continue to your data  →", "your data"),
        MANUAL: ("2 · Build", "3 · Result", "Continue to builder  →", "builder"),
        BRAINSTORM: ("2 · AI ideas", "3 · Scenario", "Continue to brainstorm  →", "ideas")}
# state that belongs to one choice of (starting point, domain, dataset); it is dropped when that choice changes
_WORK = ("wxu_sel", "wxu_result", "wxu_scn", "wxu_ideas", "wxu_ai_error", "wxu_fallback", "wxu_save_name")
_WORK_PREFIXES = ("wxu_in_", "wxu_cols_", "wxu_col_", "wxu_method_", "wxu_unc_", "wxu_sd_", "wxu_seed_", "wxu_gs_", "wxu_f_", "wxb_")
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


STEPS = ["1 · Setup", "2 · Questions", "3 · Scenario", "Saved"]
_KEEP = ("wxu_pick_domain", "wxu_pick_start", "wxu_ds", "wxu_obj")


def _goto(step: str) -> None:
    S["wxu_goto"] = step
    st.rerun()


def _reset_work() -> None:
    """Forget every question, selection, typed scenario value, AI idea and result. The manual draft is the person's own and stays."""
    for k in list(S):
        if k in _WORK or str(k).startswith(_WORK_PREFIXES):
            del S[k]


def _context(datasets: list[dict]) -> tuple[str, str, dict | None, str | None]:
    """The choices made in Setup, read from session state so every step sees them even while Setup is not on screen."""
    domain = S.get("wxu_pick_domain", "auto")
    start = S.get("wxu_pick_start", START[0])
    ds = S.get("wxu_ds") if start != MANUAL and datasets else None
    objective = (S.get("wxu_obj") or "").strip() or None if start == BRAINSTORM else None
    return domain, start, ds, objective


def _sync(datasets: list[dict]) -> None:
    """Nothing produced under one starting point, domain or dataset may show up under another."""
    domain, start, ds, _ = _context(datasets)
    signature = (start, domain, ds["id"] if ds else None)
    if S.get("wxu_ctx", signature) != signature:
        _reset_work()
    S["wxu_ctx"] = signature


def _setup_step(datasets: list[dict], ui: dict) -> None:
    left, right = st.columns([1, 1], gap="medium")
    with left:
        with st.container(border=True, key="wxu_panel_domain"):
            html('<div class="wx-ph"><div><div class="wx-pk">Step 1a · Context</div><div class="wx-pt">Domain</div>'
                 '<div class="wx-ps">Choose the context for your analysis.</div></div></div>')
            domain = st.radio("Domain", list(ui), format_func=lambda k: ui[k][1], key="wxu_pick_domain", on_change=_reset_work, label_visibility="collapsed")
            html(f'<div class="wx-detail">{esc(ui[domain][2])}</div>')
    with right:
        with st.container(border=True, key="wxu_panel_start"):
            html('<div class="wx-ph"><div><div class="wx-pk">Step 1b · How to begin</div><div class="wx-pt">Starting point</div>'
                 '<div class="wx-ps">How would you like to begin your scenario?</div></div></div>')
            start = st.radio("Starting point", START, format_func=lambda k: k, captions=[START_UI[k][1] for k in START], key="wxu_pick_start",
                             on_change=_reset_work, label_visibility="collapsed")
    S["wxu_domain"], S["wxu_start"] = domain, start      # the names the rest of the page (and its tests) read
    needs_data = start == DATASET or (domain == "auto" and start == EXPLORE)
    ds = None
    with st.container(border=True, key="wxu_panel_data"):
        html('<div class="wx-ph"><div><div class="wx-pk">Step 1c · Data</div><div class="wx-pt">Dataset</div>'
             '<div class="wx-ps">The table Silt reads to find questions it can answer.</div></div></div>')
        if start == MANUAL:
            st.caption("This starting point does not use a dataset: you define every variable yourself in the next step.")
        elif datasets:
            none_label = ("Choose a dataset…" if needs_data else "None: brainstorm without data (optional)" if start == BRAINSTORM
                          else "None: use the domain's templates (optional)")
            ds = st.selectbox("Dataset", [None] + datasets, format_func=lambda d: none_label if d is None else d["filename"], key="wxu_ds",
                              label_visibility="collapsed")
        elif needs_data:
            st.info("No dataset yet. Upload one on the Datasets page, or choose “Build a scenario manually”.")
        else:
            st.caption("No dataset uploaded. You can continue without one.")
        if start != MANUAL and S.get("wxu_hidden"):
            st.caption(f"{S['wxu_hidden']} older dataset(s) are not listed because their file is no longer on the server. Upload them again to use them.")
        if ds and start != MANUAL:
            _profile_panel(ds["id"], None if domain == "auto" else domain)
    if start == BRAINSTORM:
        with st.container(border=True, key="wxu_panel_goal"):
            html('<div class="wx-ph"><div><div class="wx-pk">Step 1d · Goal</div><div class="wx-pt">What do you want to decide?</div>'
                 '<div class="wx-ps">Describe your role and goal, or ask a question about your data.</div></div></div>')
            st.text_area("Describe your role and what you want to decide", key="wxu_obj", height=80, max_chars=1000, on_change=_reset_work,
                         label_visibility="collapsed",
                         placeholder="I am a college administrator. Help me identify important scenarios to evaluate before the next academic year.")
    if st.button(FLOW[start][2], type="primary", key="wxu_next_setup"):
        _goto(STEPS[1])


def _questions_step(datasets: list[dict]) -> None:
    domain, start, ds, objective = _context(datasets)
    ds_id = ds["id"] if ds else None
    key = None if domain == "auto" else domain
    if start == MANUAL:
        if manual.builder(MANUAL):
            _goto(STEPS[2])
    elif start == DATASET:
        _dataset_view(ds_id, key)
    elif start == BRAINSTORM:
        _brainstorm_view(ds_id, key, objective)
    else:
        _explore_view(ds_id, key)


def _scenario_step(datasets: list[dict]) -> None:
    domain, start, ds, _ = _context(datasets)
    ds_id = ds["id"] if ds else None
    sel = S.get("wxu_sel")
    saved = bool(sel) and str(sel["id"]).startswith("saved")
    if sel and not saved and sel.get("workflow") != start:
        _reset_work()
        sel = None
    if sel:
        back = "←  Edit assumptions" if start == MANUAL and not saved else f"←  Back to {FLOW[start][3]}"
        if st.button(back, key="wxu_back"):
            if start != MANUAL or saved:
                S["wxu_sel"], S["wxu_result"] = None, None
            _goto(STEPS[1])
    if saved or (sel and start == MANUAL):
        if S.get("wxu_result"):
            _result(S["wxu_result"], (S["wxu_scn"].get("dataset") or {}).get("dataset_id") if saved else None, sel)
    elif sel and sel.get("custom_model"):
        manual.idea_builder(sel)
        if S.get("wxu_result"):
            _result(S["wxu_result"], None, sel)
    elif sel:
        _builder(sel, ds_id)
    else:
        st.info("Nothing has been run yet. Define your scenario in step 2 and press Run." if start == MANUAL
                else "No question chosen yet. Pick one in step 2 by pressing “Use”.")
        if st.button(f"←  Go to {FLOW[start][3]}", key="wxu_to_questions"):
            _goto(STEPS[1])


def explore_tab(datasets: list[dict]) -> None:
    # a dataset whose file is gone for good (no copy on disk or in the database) cannot be used, so it is not offered
    hidden = [d for d in datasets if d.get("file_available") is False]
    datasets = [d for d in datasets if d.get("file_available") is not False]
    S["wxu_hidden"] = len(hidden)
    try:
        domains = _cached(("domains",), whatif_domains)
    except ApiError as e:
        st.error(f"Backend unreachable: {e}")
        return
    ui = {**DOMAIN_UI}
    for d in domains:                                   # the backend's own domain list decides which domains exist
        ui.setdefault(d["key"], ("category", d["title"], d["description"]))
    for k in list(S):                                   # a widget that is not on screen would otherwise lose its value
        if k in _KEEP or str(k).startswith(manual.KEEP_PREFIX):
            S[k] = S[k]
    _sync(datasets)
    if S.get("wxu_goto"):
        S["wxu_step"] = S.pop("wxu_goto")
    elif S.get("wxu_step") is None:
        S["wxu_step"] = STEPS[0]
    domain, start, ds, _ = _context(datasets)
    names = {STEPS[1]: FLOW[start][0], STEPS[2]: FLOW[start][1]}
    step = st.segmented_control("Step", STEPS, format_func=lambda s: names.get(s, s), key="wxu_step", label_visibility="collapsed",
                                selection_mode="single") or STEPS[0]
    if step != STEPS[0]:
        st.caption(" · ".join(x for x in (ui.get(domain, ("", ""))[1], start, ds["filename"] if ds else None) if x) + "  —  change this in step 1")
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


def _card(row: str, s: dict, workflow: str, title: str, lines: list[tuple[str, str]]) -> None:
    """One scenario in a list: its name, whether it can run, what it explores and needs, and the button that picks it."""
    label, cls = STATE[s["feasibility"]]
    with st.container(key=f"wxu_row_{row}"):
        a, b = st.columns([6, 1], vertical_alignment="center")
        body = "".join(f'<div class="wx-meta">{f"<b>{esc(k)}</b> " if k else ""}{esc(v)}</div>' for k, v in lines if v)
        a.markdown(f'<div class="wx-q">{esc(title)}</div>'
                   f'<div><span class="wx-state {cls}">{label}</span><span class="wx-meta">{esc(METHOD_LABEL.get(s["method"], s["method"]))}</span></div>'
                   + body, unsafe_allow_html=True)
        if s["feasibility"] in ("ready", "needs_assumptions") and b.button("Use", key=f"wxu_use_{s['id']}"):
            _reset_work_for_choice()
            S["wxu_sel"], S["wxu_result"] = {**s, "workflow": workflow}, None
            _goto(STEPS[2])


def _reset_work_for_choice() -> None:
    """A newly chosen question starts clean: no value or column choice carried over from an earlier one."""
    for k in list(S):
        if str(k).startswith(_WORK_PREFIXES) and not str(k).startswith("wxu_f_"):
            del S[k]


def _fetch(ds_id, domain, mode: str) -> dict | None:
    try:
        return _cached((mode, ds_id, domain), lambda: whatif_suggest(ds_id, domain, mode))
    except ApiError as e:
        st.error(str(e))
        return None


def _explore_lines(s: dict) -> list[tuple[str, str]]:
    return [("Explores", s["reason"]), ("Data needed", " · ".join(s["needs"])), ("Missing now", "; ".join(s["missing_inputs"])),
            ("Limit", (s["limitations"] or [""])[0])]


def _explore_view(ds_id, domain) -> None:
    """Discovery: the domain's questions, arranged by the kind of calculation behind them."""
    if domain is None and ds_id is None:
        st.info("Choose a domain, or a dataset to detect it from, in step 1.")
        return
    out = _fetch(ds_id, domain, "explore")
    if out is None:
        return
    dom = out["domain"]
    html(f'<div class="wx-section">Discover what you could explore in {esc(dom["title"])}</div>')
    st.caption(dom["description"] + (f" Typical measures: {', '.join(dom['metrics'])}." if dom["metrics"] else ""))
    if domain is None and dom.get("detection"):
        st.caption(dom["detection"]["explanation"])
    if not out["meta"]["with_data"]:
        st.caption("No dataset chosen: questions that read a table show what they would need.")
    if not out["suggestions"]:
        st.info(f"{dom['title']} has no ready-made questions. Choose another domain or a dataset, or build a scenario manually.")
        return
    caps = {c["key"]: c for c in out["capabilities"]}
    pick = st.selectbox("Kind of question", [None, *caps], key="wxu_f_capability",
                        format_func=lambda k: f"All kinds ({len(out['suggestions'])})" if k is None else f"{caps[k]['title']} ({caps[k]['count']})")
    n = 0
    for key, cap in caps.items():
        if pick not in (None, key):
            continue
        html(f'<div class="wx-section">{esc(cap["title"])}</div>')
        st.caption(cap["what"])
        for s in (x for x in out["suggestions"] if x["capability"] == key):
            _card(str(n), s, EXPLORE, f"{s['title']}: {s['question']}", _explore_lines(s))
            n += 1


def _dataset_view(ds_id, domain) -> None:
    """The table first (what each column is and whether a scenario can use it), then only the questions its columns support."""
    if ds_id is None:
        st.info("Choose a dataset in step 1. These scenarios are built from its columns.")
        if st.button("←  Choose a dataset", key="wxu_to_setup"):
            _goto(STEPS[0])
        return
    out = _fetch(ds_id, domain, "dataset")
    if out is None:
        return
    meta = out["meta"]
    html(f'<div class="wx-section">What is in {esc(meta["filename"] or "your data")}</div>')
    st.caption(f"{meta['rows']:,} records · {len(out['columns'])} columns · read as {out['domain']['title']}")
    st.dataframe(pd.DataFrame(out["columns"]).rename(columns={"column": "Column", "type": "Type", "missing": "Blank", "observed": "Values seen",
                                                              "use": "How a scenario can use it"}), hide_index=True, use_container_width=True)
    for r in meta["relationships"]:
        st.caption(f"Moves together: {r['a']} and {r['b']} (correlation {r['pearson']:+.2f} over {r['n']} rows)")
    for note in meta["notes"]:
        st.caption(note)
    used = [c["column"] for c in out["columns"] if any(c["column"] in s["used_columns"] for s in out["suggestions"])]
    html('<div class="wx-section">Scenarios built from these columns</div>')
    if not out["suggestions"]:
        st.info("No calculation can use these columns: none is numeric or a date. Build a scenario manually instead.")
        return
    column = st.selectbox("Show scenarios using", [None, *used], key="wxu_f_column", format_func=lambda c: "Any column" if c is None else c)
    n = 0
    for g in out["groups"]:
        rows = [s for s in out["suggestions"] if s["group"] == g["key"] and (column is None or column in s["used_columns"])]
        if not rows:
            continue
        html(f'<div class="wx-section">{esc(g["title"])}</div>')
        for s in rows:
            needs = "; ".join(s["missing_inputs"]) or ", ".join(i["label"].lower() for i in s["inputs"] if i["default"] is None and not i["optional"])
            _card(str(n), s, DATASET, s["question"], [("Columns", ", ".join(s["used_columns"])), ("Why", s["reason"]), ("You provide", needs),
                                                       ("Limit", (s["limitations"] or [""])[0])])
            n += 1


def _brainstorm_view(ds_id, domain, objective) -> None:
    """Ideas written by the language model for this objective. If it cannot answer, that is said; the rules-based list is a
    separate choice, labelled as not AI."""
    if not objective or len(objective) < 5:
        st.info("Describe your objective in step 1, then brainstorm here.")
        if st.button("←  Describe my objective", key="wxu_to_setup"):
            _goto(STEPS[0])
        return
    signature = (objective, ds_id, domain)
    html('<div class="wx-section">Brainstorm with Silt AI</div>')
    st.caption(f"Objective: {objective}")
    made = S.get("wxu_ideas") if (S.get("wxu_ideas") or {}).get("signature") == signature else None
    if st.button("Brainstorm scenarios", key="wxu_go", type="primary"):      # one label: a button that renames itself loses its click
        _reset_work()
        made = None
        try:
            with st.spinner("Silt AI is thinking about your objective…"):
                made = S["wxu_ideas"] = {"signature": signature, "out": whatif_brainstorm(objective, ds_id, domain)}
        except ApiError as e:
            S["wxu_ai_error"] = {"signature": signature, "message": re.sub(r"^\d{3}:\s*", "", str(e))}
    failed = S.get("wxu_ai_error") if (S.get("wxu_ai_error") or {}).get("signature") == signature else None
    if ds_id and _is_data_question(objective) and (made or failed):
        _answer_data_question(ds_id, objective)
    if failed:
        st.error(failed["message"])
        _fallback(ds_id, domain, signature)
    if not made:
        return
    out, meta = made["out"], made["out"]["meta"]
    html('<div class="wx-section">Ideas from Silt AI</div>')
    st.caption(f"{len(out['suggestions'])} ideas written by {meta['generated_by']} for your objective, ranked by {meta['ranked_by']}. "
               "Silt checked each against your data and calculations."
               + (f" {meta['dropped']} idea(s) were left out because they referred to something that does not exist." if meta["dropped"] else ""))
    for n, s in enumerate(out["suggestions"]):
        ai = s["ai"]
        runs = s["question"] if s["feasibility"] in ("ready", "needs_assumptions") and not s.get("custom_model") else ""
        _card(str(n), s, BRAINSTORM, f"{s['rank']}. {ai['title']}: {ai['question']}",
              [("Why (Silt AI)", ai["why"]), ("Usefulness", f"{ai['usefulness']} of 5, rated by Silt AI"), ("Data needed", " · ".join(s["needs"])),
               ("Missing now", "; ".join(s["missing_inputs"])), ("Silt will calculate", runs), ("Limit", (s["limitations"] or [""])[0])])


def _fallback(ds_id, domain, signature) -> None:
    if ds_id is None and domain is None:
        st.caption("Choose a dataset or a domain in step 1 to see rules-based scenarios instead, or try again shortly.")
        return
    if S.get("wxu_fallback") != signature:
        if st.button("Show rules-based scenarios instead (not AI)", key="wxu_fallback_btn"):
            S["wxu_fallback"] = signature
            st.rerun()
        return
    out = _fetch(ds_id, domain, "dataset" if ds_id else "explore")
    if out is None:
        return
    html('<div class="wx-section">Rules-based scenarios (not written by Silt AI)</div>')
    st.caption("These come from your " + ("dataset's columns" if ds_id else "domain") + ". They have not been chosen or ranked for your objective.")
    for n, s in enumerate(out["suggestions"]):
        _card(str(n), s, BRAINSTORM, s["question"], _explore_lines(s))


def _inputs_form(sel: dict) -> tuple[dict, dict]:
    inputs, uncertainty = {}, {}
    groups = {"value": "Your values", "baseline": "Starting values", "assumption": "Assumptions", "change": "What changes"}
    for role, title in groups.items():
        specs = [i for i in sel["inputs"] if i["role"] == role]
        if not specs:
            continue
        st.markdown(f"**{title}**")
        cols = st.columns(min(3, len(specs)))
        for col, i in zip(cols * 3, specs):
            key = f"wxu_in_{sel['id']}_{i['name']}"
            label = f"{i['label']}" + (f" ({i['unit']})" if i["unit"] else "") + (" — optional" if i["optional"] else "")
            lo, hi, default = (None if i[k] is None else float(i[k]) for k in ("min", "max", "default"))
            v = col.number_input(label, min_value=lo, max_value=hi, value=default, key=key, help=i["help"] or None, step=None, format="%g")
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
        scenario["target"] = {"output": t1.text_input("Output to reach (e.g. profit)", "profit", key=f"wxu_gs_o_{sel['id']}"),
                              "value": t2.number_input("Target value", value=0.0, key=f"wxu_gs_v_{sel['id']}"),
                              "solve_for": t3.text_input("Input to solve for (e.g. price)", "price", key=f"wxu_gs_s_{sel['id']}")}
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
                pick = st.selectbox(m["label"], m["candidates"], key=f"wxu_col_{sel['id']}_{m['name']}")
                S.setdefault(f"wxu_cols_{sel['id']}", {})[m["name"].split(":", 1)[1]] = [pick]
            else:
                st.markdown(f"- **{m['label']}** — {m.get('why', '')}")
        return
    render_result(res, S.get("wxu_scn") or {}, ds_id, sel)


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
            _reset_work()
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
