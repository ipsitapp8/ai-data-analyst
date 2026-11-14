"""Silt Copilot: talk to your data and documents.

Upload a CSV, Excel workbook or PDF, then ask in ordinary language. The local fine-tuned model reads the question,
plans the analysis and explains the result; deterministic tools compute every number (calculator, statistics,
scikit-learn models, scenario arithmetic, document search). Technical detail is kept in an optional "How this was
calculated" section. Conversations are kept for this browser session; questions asked while one is being answered
wait in a queue and are answered in order.
"""
from __future__ import annotations

import io
import re
import time
import uuid
from html import escape as html_escape

import pandas as pd
import streamlit as st

from api_client import (ApiError, analyst_ask, analyst_documents, analyst_preview, analyst_status,
                        analyst_upload_document, list_datasets, upload_dataset)
from style import copilot_ui as ui
from style.theme import page_setup, render_sidebar, require_active_team

page_setup("Silt Copilot")
render_sidebar("analyst")
require_active_team("Silt Copilot")

S = st.session_state
S.setdefault("cp_convs", {})
S.setdefault("cp_upload_n", 0)
S.setdefault("cp_theme", "light")
st.markdown(ui.css(S["cp_theme"]), unsafe_allow_html=True)

SOURCE = {"model": "Planned by the fine-tuned local model",
          "model_retry": "Planned by the fine-tuned local model, after one reconsideration",
          "rules": "Planned by Silt's deterministic planner",
          "router": "Answered by Silt's analysis tools",
          "tool": "Computed by Silt's analysis tools",
          "document": "Answered from the document",
          "conversation": "Explained by the fine-tuned local model from the computed results",
          "conversation_rules": "Explained by Silt from the computed results",
          "model_calc": "Planned and explained by the fine-tuned local model · numbers computed exactly by Silt's calculator",
          "model_calc_failed": "The local model's calculation plan could not be verified · nothing was computed",
          "input_check": "Checked by Silt before analysis · nothing was computed",
          "metric": "Metric definition resolved and computed by Silt's calculator · no language model needed",
          "refine": "The previous calculation re-run with your change by Silt's calculator",
          "conversation_state": "Read against your open request · nothing was calculated"}
EXAMPLES_DATA = ["Explain this dataset in simple terms.", "Which group has the highest total?",
                 "What trends do you notice?", "Find unusual records.", "What should we do next?",
                 "Show me a chart of the main measure by group."]
EXAMPLES_DOC = ["Summarize the document.", "What does the report say about risks?"]
WAIT_SECONDS = 300           # a question being answered by another run is waited for, then retried once


def new_conv(dataset_id=None, doc_id=None) -> str:
    cid = uuid.uuid4().hex[:12]
    S["cp_convs"][cid] = {"title": "New conversation", "dataset_id": dataset_id, "doc_id": doc_id, "log": [],
                          "created": time.time()}
    S["cp_active"] = cid
    return cid


def enqueue(conv: dict, q: str) -> None:
    """Add a question to the conversation's queue; an identical question still waiting is not added twice."""
    q = q.strip()
    if not q:
        return
    if any(t["q"] == q and t.get("r") is None for t in conv["log"]):
        S["cp_dup"] = True
        return
    conv["log"].append({"q": q, "r": None})
    if conv["title"] == "New conversation":
        conv["title"] = q


try:
    status = analyst_status()
    datasets = list_datasets()
    docs = analyst_documents()
except ApiError as e:
    st.error(f"Silt's backend could not be reached, so the Copilot cannot run right now. ({e})")
    st.stop()

ds_by_id = {d["id"]: d for d in datasets}
doc_by_id = {d["id"]: d for d in docs}
if S.get("cp_active") not in S["cp_convs"]:
    new_conv(datasets[0]["id"] if datasets else None)
conv = S["cp_convs"][S["cp_active"]]

# --------------------------------------------------------------------------- header --
h1, h2 = st.columns([3, 1.5])
with h1:
    st.markdown("<div class='cp-title'><span>Silt</span> Copilot</div><div class='cp-sub'>Ask about your data in plain "
                "language. The local model understands the question and explains the answer; every number is "
                "computed from your data.</div>", unsafe_allow_html=True)
with h2:
    served = status.get("served") or {}
    if served.get("verified") and served.get("adapter_is_v1"):
        ident = (f"{html_escape(str(served.get('base_model', '')).split('/')[-1])} + Silt LoRA v1 "
                 f"({str(served.get('adapter_sha256', ''))[:8]}) · on this computer")
    elif served.get("verified"):
        ident = (f"{html_escape(str(served.get('base_model', '')))} · adapter "
                 f"{html_escape(str(served.get('adapter') or 'none'))} — <b>not the v1 adapter</b>")
    else:
        ident = html_escape(status.get("model_name", ""))
    pill = ("<span class='cp-pill'>● Local model running</span>" if status.get("model_available") else
            "<span class='cp-pill off'>● Local model offline — Silt's tools only</span>")
    a, b = st.columns([3, 1])
    a.markdown(f"<div style='text-align:right;margin-top:6px'>{pill}<div class='cp-src'>{ident}</div></div>",
               unsafe_allow_html=True)
    if b.button("☾" if S["cp_theme"] == "light" else "☀", key="cp_theme_btn",
                help="Switch between light and dark theme"):
        S["cp_theme"] = "dark" if S["cp_theme"] == "light" else "light"
        st.rerun()

left, main = st.columns([1, 3.4], gap="medium")

# --------------------------------------------------------------- conversations panel --
with left:
    if st.button("＋ New conversation", width="stretch", type="primary"):
        new_conv(conv["dataset_id"], conv["doc_id"])
        st.rerun()
    st.caption("Conversations")
    for cid, c in sorted(S["cp_convs"].items(), key=lambda kv: -kv[1]["created"]):
        ctx = ds_by_id.get(c["dataset_id"], {}).get("filename") or doc_by_id.get(c["doc_id"], {}).get("filename") or "—"
        label = f"{'▸ ' if cid == S['cp_active'] else ''}{c['title'][:38]}"
        if st.button(label, key=f"cv_{cid}", width="stretch", help=f"{len(c['log'])} messages · {ctx}"):
            S["cp_active"] = cid
            st.rerun()

    with st.popover("⤒ Upload a file", width="stretch"):
        st.caption("CSV, Excel (.xlsx/.xls) or text-based PDF. Data files become datasets; PDFs become searchable "
                   "documents with page references.")
        up = st.file_uploader("File", type=["csv", "xlsx", "xls", "pdf"], key=f"cp_up_{S['cp_upload_n']}",
                              label_visibility="collapsed")
        sheet = None
        if up is not None:
            st.markdown(f"<div class='cp-file'><b>{html_escape(up.name)}</b> · {ui.file_kind(up.name)} · "
                        f"{ui.human_size(len(up.getvalue()))} · ready to add</div>", unsafe_allow_html=True)
        if up is not None and up.name.lower().endswith((".xlsx", ".xls")):
            try:
                names = pd.ExcelFile(io.BytesIO(up.getvalue())).sheet_names
            except Exception:  # noqa: BLE001
                names = []
                st.error("This Excel file could not be opened. It may be damaged or password-protected.")
            if len(names) > 1:
                sheet = st.selectbox("Sheet", names, help="Which worksheet holds the table?")
        if up is not None and st.button("Add to Silt", type="primary", width="stretch"):
            try:
                with st.spinner(f"Uploading and reading {up.name}…"):
                    if up.name.lower().endswith(".pdf"):
                        d = analyst_upload_document(up.name, up.getvalue())
                        target = {"dataset_id": None, "doc_id": d["id"]}
                        scanned = d.get("scanned_pages") or []
                        S["cp_flash"] = (f"Added **{d['filename']}** (PDF): {d['page_count']} pages, text found on "
                                         f"{len(d['text_pages'])}. It is now selected."
                                         + (f" Pages {', '.join(map(str, scanned))} are scanned images with no text "
                                            "layer: OCR would be needed to read them, so they are not searched."
                                            if scanned else ""))
                    else:
                        d = upload_dataset(up.name, up.getvalue(), sheet)
                        target = {"dataset_id": d["id"], "doc_id": None}
                        S["cp_flash"] = (f"Added **{d['filename']}** ({ui.file_kind(up.name)}"
                                         + (f", sheet {sheet}" if sheet else "") + f"): {d['row_count']:,} rows × "
                                         f"{d['col_count']} columns. It is now selected.")
                if conv["log"]:
                    new_conv(**target)
                else:
                    conv.update(target)
                S["cp_upload_n"] += 1
                st.rerun()
            except ApiError as e:
                st.error(f"**{up.name}** could not be added: " + re.sub(r"^\d{3}: ", "", str(e)))

# --------------------------------------------------------------------- context bar --
with main:
    if S.get("cp_flash"):
        st.success(S.pop("cp_flash"))
    c1, c2 = st.columns(2)
    ds_ids = [None] + [d["id"] for d in datasets]
    doc_ids = [None] + [d["id"] for d in docs]
    # The browser reports a selectbox choice by its label, so two options with the same label (the same file uploaded
    # twice) map back to the first one. That made the selection disagree with the conversation and the page re-ran
    # before reading the chat box, dropping the question. Labels are made unique with the id where they collide.
    ds_labels = ui.unique_labels({i: f"{d['filename']} · {d.get('row_count', '?'):,} rows" for i, d in ds_by_id.items()})
    doc_labels = ui.unique_labels({i: f"{d['filename']} · {d['page_count']} pages" for i, d in doc_by_id.items()})
    with c1:
        sel_ds = st.selectbox("Dataset", ds_ids, index=ds_ids.index(conv["dataset_id"]) if conv["dataset_id"] in ds_ids else 0,
                              format_func=lambda i: "— none —" if i is None else ds_labels[i])
    with c2:
        sel_doc = st.selectbox("Document", doc_ids, index=doc_ids.index(conv["doc_id"]) if conv["doc_id"] in doc_ids else 0,
                               format_func=lambda i: "— none —" if i is None else doc_labels[i])
    if (sel_ds, sel_doc) != (conv["dataset_id"], conv["doc_id"]):
        if conv["log"]:          # a different source is a new conversation; the old one stays in the list
            new_conv(sel_ds, sel_doc)
        else:
            conv.update(dataset_id=sel_ds, doc_id=sel_doc)
        st.rerun()

    parts = []
    if conv["dataset_id"] in ds_by_id:
        d = ds_by_id[conv["dataset_id"]]
        parts.append(f"<b>{html_escape(d['filename'])}</b> ({ui.file_kind(d['filename'])}, "
                     f"{d.get('row_count', '?'):,} rows)")
    if conv["doc_id"] in doc_by_id:
        d = doc_by_id[conv["doc_id"]]
        parts.append(f"<b>{html_escape(d['filename'])}</b> (PDF, {d['page_count']} pages)")
    st.markdown(f"<div class='cp-ctx'>Talking about {' and '.join(parts) if parts else 'nothing yet — choose or upload a dataset or document'}. "
                "Follow-up questions keep this context.</div>", unsafe_allow_html=True)
    if conv["dataset_id"] in ds_by_id:
        with st.expander("Preview the data"):
            try:
                pv = analyst_preview(conv["dataset_id"])
                st.caption(f"First {len(pv['preview'])} of {pv['rows']:,} rows")
                st.dataframe(pd.DataFrame(pv["preview"]), width="stretch", hide_index=True)
                st.dataframe(pd.DataFrame([{"column": c["name"], "type": c["kind"], "missing": c["missing"],
                                            "distinct": c["unique"]} for c in pv["schema"]]),
                             width="stretch", hide_index=True)
            except ApiError as e:
                st.error(f"The preview could not be loaded. ({e})")

    # ------------------------------------------------------------------ messages --
    def show_charts(r: dict) -> None:
        for spec in r.get("charts") or []:
            c = ui.chart(spec, S["cp_theme"])
            if c is not None:
                st.altair_chart(c, width="stretch")
                if spec.get("caption"):
                    st.caption(spec["caption"])

    def show_answer(r: dict, i: int) -> None:
        if "error" in r:
            st.error(r["error"])
            if st.button("Try again", key=f"retry_{S['cp_active']}_{i}"):
                conv["log"][i]["r"] = None
                conv["log"][i].pop("running_since", None)
                st.rerun()
            return
        answer = r.get("answer") or ""
        if r.get("chart_requested") and r.get("charts"):
            # a chart was asked for: the short direct answer, then the chart, then the rest
            head, _, rest = answer.partition("\n\n")
            st.markdown(head)
            show_charts(r)
            if rest.strip():
                st.markdown(rest)
        else:
            st.markdown(answer)
            show_charts(r)
        st.markdown(f"<div class='cp-src'>{html_escape(SOURCE.get(r.get('plan_source'), str(r.get('plan_source'))))} · "
                    f"{r.get('elapsed_ms', 0) / 1000:.1f} s</div>", unsafe_allow_html=True)
        details, notes = r.get("details") or "", r.get("notes") or []
        if details or notes:
            with st.expander("How this was calculated"):
                if details:
                    st.markdown(details)
                if notes:
                    st.markdown("**How the question was handled**\n\n" + "\n".join(f"- {n}" for n in notes))
                if st.toggle("Show the technical trace", key=f"trace_{S['cp_active']}_{i}"):
                    st.json({"plan": r.get("plan"), "tool_calls": [{k: v for k, v in c.items() if k != "result"}
                                                                   for c in r.get("calls", [])],
                             "first_pass": r.get("first_pass"), "grounding": r.get("grounding")}, expanded=False)

    if not conv["log"]:
        doc_only = conv["doc_id"] and not conv["dataset_id"]
        st.markdown("<div class='cp-empty'><b>Start with a question.</b> Ask it the way you would ask a colleague — "
                    "no column names or formulas needed. For example:</div>", unsafe_allow_html=True)
        ex = EXAMPLES_DOC if doc_only else EXAMPLES_DATA + (EXAMPLES_DOC[:1] if conv["doc_id"] else [])
        cols = st.columns(3)
        for i, q in enumerate(ex):
            if cols[i % 3].button(q, key=f"ex_{i}", width="stretch", disabled=not (conv["dataset_id"] or conv["doc_id"])):
                enqueue(conv, q)
                st.rerun()

    for i, turn in enumerate(conv["log"]):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        if turn.get("r") is not None:
            with st.chat_message("assistant"):
                show_answer(turn["r"], i)

    pending = [t for t in conv["log"] if t.get("r") is None]
    if S.pop("cp_dup", False):
        st.caption("That question is already waiting to be answered.")
    if len(pending) > 1:
        st.caption(f"{len(pending) - 1} more question{'s' if len(pending) > 2 else ''} queued; they will be answered in order.")
    typing_slot = st.empty()

busy = bool(pending)
q = st.chat_input(("Silt is answering… you can queue another question" if busy else
                   "Ask about this dataset or document…") if (conv["dataset_id"] or conv["doc_id"]) else
                  "Choose or upload a dataset or document first", disabled=not (conv["dataset_id"] or conv["doc_id"]))
if q:
    enqueue(conv, q)
    st.rerun()

if pending:
    turn = pending[0]
    low = turn["q"].lower()
    label = ("Training and testing a model" if re.search(r"\b(train|model|forecast|predict)\w*", low) else
             "Reading the document" if conv["doc_id"] and not conv["dataset_id"] else "Thinking")
    with typing_slot.container():
        with st.chat_message("assistant"):
            st.markdown(ui.typing_html(label), unsafe_allow_html=True)
    started = turn.get("running_since")
    if started and time.time() - started < WAIT_SECONDS:
        # another run of this page is already waiting for this answer (the user typed while it worked): don't ask
        # twice, just wait for it to land
        time.sleep(0.8)
        st.rerun()
    turn["running_since"] = time.time()
    try:
        turn["r"] = analyst_ask(conv["dataset_id"], turn["q"], S["cp_active"], document_id=conv["doc_id"])
    except ApiError as e:
        turn["r"] = {"error": "Sorry, that question could not be answered: " + re.sub(r"^\d{3}: ", "", str(e))
                     + " Your conversation is kept; you can try again."}
    except Exception as e:  # noqa: BLE001 - never lose the conversation over one failed request
        turn["r"] = {"error": f"Sorry, something went wrong ({type(e).__name__}). Your conversation is kept; "
                              "you can try again."}
    st.rerun()
