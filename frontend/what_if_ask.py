"""Ask a what-if in plain words.

Rendering only. The sentence is read, validated, calculated, checked and explained by the backend (`/api/scenarios/ask`, the same
service the Silt Copilot uses). The page shows how the sentence was understood *before* it shows the answer, asks when
something is unclear, and keeps the conversation explicit: it holds the context the backend returned and sends it back, and
drops it when the model or the numbers change.
"""
from __future__ import annotations

import streamlit as st

from api_client import ApiError, ask_scenario
from style.theme import badge, esc, html
from what_if_decision import num, section, verification_panel

EXAMPLES = ["What happens if I increase the price by 10%?", "What price would give me a 25% profit margin?", "How many units do I need to break even?",
            "Compare a 10% price increase with a 15% volume increase."]


def _apply(mk: str, adjustments: dict) -> None:
    """Callback: put a scenario's percentage changes into the builder's fields (runs before they are drawn)."""
    for name, a in adjustments.items():
        if a["type"] == "pct":
            st.session_state[f"wi_adj_{mk}_{name}"] = float(a["value"])


def ask_box(mk: str, model: dict, baseline: dict, assumptions: dict) -> None:
    section("Ask in plain words", "Type a what-if or a goal. You will see how it was understood first, then the answer, calculated by the same "
                                  "engines as the form below and independently checked.")
    ctx_key, last_key = f"ask_ctx_{mk}", f"ask_last_{mk}"
    text = st.text_input("Your question", key=f"ask_text_{mk}", max_chars=500, placeholder=EXAMPLES[0], label_visibility="collapsed")
    c1, c2, c3 = st.columns([1, 2, 1])
    go = c1.button("Ask", type="primary", key="ask_go", disabled=not text.strip())
    use_model = c2.checkbox("Let a language model read sentences the rules cannot", False, key=f"ask_model_{mk}",
                            help="It only translates your wording. It never supplies a number, and its reading is checked against this model's inputs "
                                 "and the numbers you wrote.")
    if c3.button("Start over", key="ask_reset"):
        st.session_state.pop(ctx_key, None)
        st.session_state.pop(last_key, None)
        st.rerun()
    if go:
        body = {"message": text.strip(), "model": mk, "baseline": baseline, "assumptions": assumptions, "use_model": bool(use_model),
                "context": st.session_state.get(ctx_key)}
        try:
            with st.spinner("Reading, calculating and checking…"):
                out = ask_scenario(body)
            st.session_state[last_key] = out
            if out.get("context"):
                st.session_state[ctx_key] = out["context"]
        except ApiError as e:
            st.session_state.pop(last_key, None)
            st.error(f"That could not be answered: {e}")
    out = st.session_state.get(last_key)
    if not out:
        st.caption("For example: " + " · ".join(EXAMPLES[:3]))
        return
    _show(mk, model, out)


def _show(mk: str, model: dict, out: dict) -> None:
    status = out["status"]
    if out.get("interpretation"):
        html('<div class="ds-row-meta" style="padding:4px 0;"><b>How I understood it</b></div>')
        for line in out["interpretation"]:
            html(f'<div class="ds-row-meta" style="padding:1px 0;">• {esc(line)}</div>')
    for n in out.get("notes", []):
        html(f'<div class="ds-row-meta" style="padding:1px 0;">{esc(n)}</div>')
    if out.get("translated_by") == "model":
        html(f'<div class="ds-row-meta" style="padding:2px 0;">{badge("Read by a language model", "warn")} checked against this model\'s inputs and the numbers you wrote</div>')
    if status == "clarify":
        c = out["clarification"]
        st.warning(c["question"])
        if c["options"]:
            st.caption("Choose one and ask again, naming it: " + ", ".join(o["label"] for o in c["options"]))
        return
    if status == "unsupported":
        st.info(out["answer"])
        st.caption("Try: " + " · ".join(out.get("examples", EXAMPLES)[:3]))
        return
    st.markdown(out["answer"].replace("\n", "  \n"))
    result = out.get("result")
    if not result:
        return
    ver = result.get("verification")
    if ver:
        verification_panel(ver, "Independent check of this answer")
    if result.get("kind") == "scenario" and out.get("intent", {}).get("action") == "evaluate":
        scns = out["intent"]["scenarios"]
        name = list(scns)[-1]
        s = result["scenarios"][name]
        rows = []
        for k, label in result["output_labels"].items():
            rows.append(f"{label}: {num(result['baseline']['outputs'][k])} → {num(s['outputs'][k])}")
        with st.expander("All the numbers for this scenario"):
            for r in rows:
                html(f'<div class="ds-row-meta" style="padding:2px 0;">{esc(r)}</div>')
        pct = {k: a for k, a in scns[name].items() if a["type"] == "pct"}
        other = {k: a for k, a in scns[name].items() if a["type"] != "pct"}
        if pct:
            st.button("Put these changes in the form below", key="ask_apply", on_click=_apply, args=(mk, pct))
        if other:
            st.caption("The form takes percentage changes only, so " + ", ".join(model["inputs"][k]["label"] for k in other) + " cannot be copied into it; the answer above uses them as written.")
    if result.get("kind") == "goal_seek" and result.get("solutions"):
        st.caption("This is the value that reaches the goal in this model, found by the same solver as the Goal seek section below.")
