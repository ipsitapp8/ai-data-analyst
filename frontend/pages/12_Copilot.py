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
                        analyst_upload_document, copilot_delete, copilot_get, copilot_list,
                        copilot_rename, copilot_save, copilot_save_message, list_datasets, upload_dataset)
from style import copilot_ui as ui
from style.theme import page_setup, render_sidebar, require_active_team

page_setup("Silt Copilot")
render_sidebar("analyst")
require_active_team("Silt Copilot")

S = st.session_state


def _user_letter() -> str:
    from auth import current_user
    return (((current_user() or {}).get("display_name") or "U")[:1] or "U").upper()


USER_LETTER = _user_letter()
S.setdefault("cp_convs", {})
S.setdefault("cp_upload_n", 0)
S["cp_theme"] = "dark"          # the site is one deep-indigo skin; the page no longer has a light mode
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
          "conversation_state": "Read against your open request · nothing was calculated",
          "program": "Analysis program: validated steps run by Silt's deterministic tools",
          "compound": "Every part of your request computed by Silt's tools and checked off",
          "model_program": "Analysis program planned by the local model; validated and computed by Silt",
          "rule_program": "Analysis program built by Silt from the request's parts; validated and computed",
          "withheld": "Withheld by Silt's answer verification · no figure shown"}
# the answer-verification status of a computed answer (backend app/analyst/verify.py), shown with its source
VERIFY = {"verified": "✓ Verified independently", "not independently verified": "◐ Not independently verified",
          "partially verified": "◐ Partly verified (see the checklist)",
          "withheld": "✕ Withheld — did not pass verification"}
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
    conv["log"].append({"q": q, "r": None, "t": time.time()})
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

# --------------------------------------------------------------------------- saved chats --
def _safe(fn, *a, **k):
    """Saving a chat must never break answering one: failures are swallowed (the chat stays in this session)."""
    try:
        return fn(*a, **k)
    except Exception:  # noqa: BLE001
        return None


def save_context() -> None:
    _safe(copilot_save, S["cp_active"], conv["dataset_id"], conv["doc_id"], None)


def save_turn(i: int) -> None:
    t = conv["log"][i]
    r = t.get("r")
    text = (r.get("answer") or r.get("error") or "") if isinstance(r, dict) else ""
    _safe(copilot_save, S["cp_active"], conv["dataset_id"], conv["doc_id"], None)
    _safe(copilot_save_message, S["cp_active"], i, t["q"], r, text)
    S["cp_hist"] = None


def history(force: bool = False) -> list[dict]:
    cached = S.get("cp_hist")
    if force or not cached or time.time() - cached[0] > 30:
        rows = _safe(copilot_list) or []
        S["cp_hist"] = (time.time(), rows)
        return rows
    return cached[1]


def open_saved(cid: str) -> None:
    """Restore a saved chat (its messages, dataset and document) into the current workspace."""
    try:
        saved = copilot_get(cid)
    except ApiError:
        st.toast("That chat could not be opened.")
        return
    S["cp_convs"][cid] = {"title": saved["title"], "dataset_id": saved["dataset_id"], "doc_id": saved["document_id"],
                          "created": time.time(),
                          "log": [{"q": m["question"], "r": m["response"], "t": _epoch(m.get("created_at"))} for m in saved["messages"]]}
    S["cp_active"] = cid
    st.rerun()


def _epoch(iso):
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(iso).replace(tzinfo=_dt.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def fmt_time(ts) -> str:
    """Clock time of a message in the visitor's own time zone (the browser reports it)."""
    import datetime as _dt
    if not ts:
        return ""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(st.context.timezone) if st.context.timezone else None
    except Exception:  # noqa: BLE001
        tz = None
    return _dt.datetime.fromtimestamp(ts, tz).strftime("%I:%M %p").lstrip("0")



def when_group(iso: str) -> str:
    import datetime as _dt
    days = (_dt.datetime.utcnow().date() - _dt.datetime.fromisoformat(iso).date()).days
    return "Today" if days <= 0 else "Yesterday" if days == 1 else "Previous 7 days" if days <= 7 else "Older"


def start_new_chat() -> None:
    new_conv(conv["dataset_id"], conv["doc_id"])
    st.rerun()


# --------------------------------------------------------------------------- header --
_skin = (__import__("pathlib").Path(__file__).resolve().parent.parent / "style" / "copilot_chat.css").read_text(encoding="utf-8")
st.markdown(f"<style>:root {{ --cp-init: '{USER_LETTER}'; }}{_skin}</style><div class='cp-scope'></div>", unsafe_allow_html=True)

with st.container(key="cp_head"):
    t_col, s_col, h_col, n_col = st.columns([6, 0.7, 0.7, 1.55], vertical_alignment="center")
    t_col.markdown("<div class='cp-title'><span>Silt</span> Copilot <i class='cp-spark'>✦</i></div>"
                   "<div class='cp-tag'>Your AI data analyst. Ask questions, get insights and run analysis.</div>",
                   unsafe_allow_html=True)

    # ---- search: titles, questions, answers and dataset names of this user's saved chats
    with s_col.popover("Search", icon=":material/search:", help="Search your chats"):
        term = st.text_input("Search chats", key="cp_search", placeholder="Search your chats…",
                             label_visibility="collapsed")
        if term.strip():
            with st.spinner("Searching…"):
                try:
                    found = copilot_list(term.strip())
                except ApiError:
                    found = None
            if found is None:
                st.caption("Search is unavailable right now. Try again in a moment.")
            elif not found:
                st.caption(f"No chats match “{term.strip()}”.")
            for c in found or []:
                if st.button(c["title"][:60], key=f"sr_{c['id']}", width="stretch",
                             help=f"{c.get('dataset_name') or 'No dataset'} · {c['message_count']} messages"):
                    open_saved(c["id"])
                if c.get("snippet") and c.get("matched_in") != "title":
                    st.markdown(f"<div class='cp-snip'>{html_escape(c['snippet'])}</div>", unsafe_allow_html=True)
        else:
            st.caption("Find a chat by its title, a question, an answer or the dataset it used.")

    # ---- history: recent chats grouped by age; open, rename or delete
    with h_col.popover("History", icon=":material/history:", help="Recent chats"):
        rows = history()
        if not rows:
            st.caption("Your chats are saved here once you start one.")
        order = ["Today", "Yesterday", "Previous 7 days", "Older"]
        for grp in order:
            items = [c for c in rows if when_group(c["updated_at"]) == grp]
            if not items:
                continue
            st.markdown(f"<div class='cp-group'>{grp}</div>", unsafe_allow_html=True)
            for c in items:
                if S.get("cp_renaming") == c["id"]:
                    new_title = st.text_input("Name", value=c["title"], key=f"rn_{c['id']}", max_chars=120,
                                              label_visibility="collapsed")
                    ok, cancel = st.columns(2)
                    if ok.button("Save name", key=f"rns_{c['id']}", type="primary", width="stretch") and new_title.strip():
                        _safe(copilot_rename, c["id"], new_title.strip())
                        if c["id"] in S["cp_convs"]:
                            S["cp_convs"][c["id"]]["title"] = new_title.strip()
                        S["cp_renaming"], S["cp_hist"] = None, None
                        st.rerun()
                    if cancel.button("Cancel", key=f"rnc_{c['id']}", width="stretch"):
                        S["cp_renaming"] = None
                        st.rerun()
                    continue
                o, e, d = st.columns([8, 1.3, 1.3], vertical_alignment="center")
                label = f"{'● ' if c['id'] == S['cp_active'] else ''}{c['title'][:40]}"
                if o.button(label, key=f"hx_{c['id']}", width="stretch",
                            help=f"{c.get('dataset_name') or 'No dataset'} · {c['message_count']} messages"):
                    open_saved(c["id"])
                if e.button("✎", key=f"ed_{c['id']}", help="Rename", width="stretch"):
                    S["cp_renaming"] = c["id"]
                    st.rerun()
                if d.button("✕", key=f"dl_{c['id']}", help="Delete this chat", width="stretch"):
                    _safe(copilot_delete, c["id"])
                    S["cp_convs"].pop(c["id"], None)
                    S["cp_hist"] = None
                    st.rerun()

    if n_col.button("New chat", icon=":material/add:", type="primary", width="stretch", key="cp_new"):
        start_new_chat()

def add_file(f) -> None:
    """Upload a file attached in the message box. Data files become datasets, PDFs become documents."""
    name, data = f.name, f.getvalue()
    try:
        with st.spinner(f"Uploading and reading {name}…"):
            if name.lower().endswith(".pdf"):
                d = analyst_upload_document(name, data)
                target = {"dataset_id": None, "doc_id": d["id"]}
                scanned = d.get("scanned_pages") or []
                S["cp_flash"] = (f"Added **{d['filename']}** (PDF): {d['page_count']} pages, text found on "
                                 f"{len(d['text_pages'])}. It is now selected."
                                 + (f" Pages {', '.join(map(str, scanned))} are scanned images with no text "
                                    "layer: OCR would be needed to read them, so they are not searched."
                                    if scanned else ""))
            else:
                d = upload_dataset(name, data, None)
                target = {"dataset_id": d["id"], "doc_id": None}
                how = (((d.get("profile") or {}).get("ingestion") or {}).get("notes") or [])
                if d.get("already_existed"):
                    S["cp_flash"] = f"**{d['filename']}** is already in your datasets, so it was not added again. It is selected."
                else:
                    S["cp_flash"] = (f"Added **{d['filename']}** ({ui.file_kind(name)}): {d['row_count']:,} rows × "
                                     f"{d['col_count']} columns. It is now selected."
                                     + ("\n\n**How the file was read:** " + " ".join(how[:6]) if how else ""))
        if conv["log"]:
            new_conv(**target)
        else:
            conv.update(target)
        S["cp_upload_n"] += 1
        st.rerun()
    except ApiError as e:
        st.error(f"**{name}** could not be added: " + re.sub(r"^\d{3}: ", "", str(e)))


# --------------------------------------------------------------------- context bar --
main = st.container()
with main:
    if S.get("cp_flash"):
        st.success(S.pop("cp_flash"))
    ds_ids = [None] + [d["id"] for d in datasets]
    doc_ids = [None] + [d["id"] for d in docs]
    # The browser reports a selectbox choice by its label, so two options with the same label (the same file uploaded
    # twice) map back to the first one. Labels are made unique with the id where they collide.
    ds_labels = ui.unique_labels({i: f"{d['filename']} · {d.get('row_count', '?'):,} rows" for i, d in ds_by_id.items()})
    doc_labels = ui.unique_labels({i: f"{d['filename']} · {d['page_count']} pages" for i, d in doc_by_id.items()})
    cur_ds, cur_doc = ds_by_id.get(conv["dataset_id"]), doc_by_id.get(conv["doc_id"])
    changing = bool(S.get("cp_change")) or (cur_ds is None and cur_doc is None)
    with st.container(key="cp_ds"):
        info, act = st.columns([4, 1.5], vertical_alignment="center")
        if cur_ds or cur_doc:
            lines = []
            if cur_ds:
                lines.append(f"<div class='cp-ds-n'>{html_escape(cur_ds['filename'])}</div>"
                             f"<div class='cp-ds-m'>{cur_ds.get('row_count', 0):,} rows • {cur_ds.get('col_count', '?')} columns</div>")
            if cur_doc:
                lines.append(f"<div class='cp-ds-n'>{html_escape(cur_doc['filename'])}</div>"
                             f"<div class='cp-ds-m'>PDF • {cur_doc['page_count']} pages</div>")
            info.markdown("<div class='cp-ds'><span class='cp-ds-i'>" + ui.TABLE_ICON + "</span><div>" + "".join(lines) + "</div></div>",
                          unsafe_allow_html=True)
            if act.button("Change dataset", icon=":material/swap_horiz:", key="cp_chg_btn", width="stretch"):
                S["cp_change"] = not S.get("cp_change")
                st.rerun()
        else:
            info.markdown("<div class='cp-ds'><span class='cp-ds-i'>" + ui.TABLE_ICON + "</span><div>"
                          "<div class='cp-ds-n'>No dataset yet</div>"
                          "<div class='cp-ds-m'>Pick one below, or attach a CSV, Excel or PDF with the paperclip.</div></div></div>",
                          unsafe_allow_html=True)
    sel_ds, sel_doc = conv["dataset_id"], conv["doc_id"]
    if changing:
        c1, c2 = st.columns([3, 2] if docs else [1, 0.001])
        with c1:
            sel_ds = st.selectbox("Dataset", ds_ids, index=ds_ids.index(conv["dataset_id"]) if conv["dataset_id"] in ds_ids else 0,
                                  format_func=lambda i: "Choose a dataset…" if i is None else ds_labels[i],
                                  label_visibility="collapsed")
        if docs:
            with c2:
                sel_doc = st.selectbox("Document", doc_ids, index=doc_ids.index(conv["doc_id"]) if conv["doc_id"] in doc_ids else 0,
                                       format_func=lambda i: "No document" if i is None else doc_labels[i],
                                       label_visibility="collapsed")
    if (sel_ds, sel_doc) != (conv["dataset_id"], conv["doc_id"]):
        if conv["log"]:          # a different source is a new conversation; the old one stays in the list
            new_conv(sel_ds, sel_doc)
        else:
            conv.update(dataset_id=sel_ds, doc_id=sel_doc)
        S["cp_change"] = False
        st.rerun()

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
        status = VERIFY.get(r.get("verify_status") or "")
        st.markdown(f"<div class='cp-src'>{html_escape(SOURCE.get(r.get('plan_source'), str(r.get('plan_source') or 'Computed by Silt')))} · "
                    + (f"<b>{html_escape(status)}</b> · " if status else "")
                    + f"{r.get('elapsed_ms', 0) / 1000:.1f} s</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='cp-hello'>What would you like to know?</div>"
                    "<div class='cp-hello-sub'>Ask the way you would ask a colleague. Try one of these, or type your own below.</div>",
                    unsafe_allow_html=True)
        ex = EXAMPLES_DOC if doc_only else EXAMPLES_DATA + (EXAMPLES_DOC[:1] if conv["doc_id"] else [])
        cols = st.columns(3)
        for i, q in enumerate(ex):
            if cols[i % 3].button(q, key=f"ex_{i}", width="stretch", disabled=not (conv["dataset_id"] or conv["doc_id"])):
                enqueue(conv, q)
                save_turn(len(conv["log"]) - 1)
                st.rerun()

    for i, turn in enumerate(conv["log"]):
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(turn["q"])
        st.markdown(f"<div class='cp-time u'>{fmt_time(turn.get('t'))}</div>", unsafe_allow_html=True)
        if turn.get("r") is not None:
            with st.chat_message("assistant", avatar=ui.LOGO_AVATAR):
                show_answer(turn["r"], i)
            st.markdown(f"<div class='cp-time a'>{fmt_time(turn.get('ta') or turn.get('t'))}</div>", unsafe_allow_html=True)

    pending = [t for t in conv["log"] if t.get("r") is None]
    if S.pop("cp_dup", False):
        st.caption("That question is already waiting to be answered.")
    if len(pending) > 1:
        st.caption(f"{len(pending) - 1} more question{'s' if len(pending) > 2 else ''} queued; they will be answered in order.")
    typing_slot = st.empty()

busy = bool(pending)
has_source = bool(conv["dataset_id"] or conv["doc_id"])
composer = st.chat_input("Silt is answering… you can queue another question" if busy else
                         "Ask about this dataset or document…" if has_source else
                         "Attach a CSV, Excel or PDF, or choose a dataset above",
                         accept_file="multiple", file_type=["csv", "xlsx", "xls", "pdf"], key=f"cp_in_{S['cp_upload_n']}")
q = None
if composer:
    for f in (composer.files or []):
        add_file(f)
    q = (composer.text or "").strip() or None
    if q and not has_source:
        st.warning("Choose a dataset above, or attach a file, before asking.")
        q = None
if not q and has_source and S.get("cp_handoff"):
    q = S.pop("cp_handoff")          # a question handed over from the What-If page, with its scenario context in the text
if q:
    enqueue(conv, q)
    save_turn(len(conv["log"]) - 1)
    st.rerun()

if pending:
    turn = pending[0]
    low = turn["q"].lower()
    label = ("Training and testing a model" if re.search(r"\b(train|model|forecast|predict)\w*", low) else
             "Reading the document" if conv["doc_id"] and not conv["dataset_id"] else "Thinking")
    with typing_slot.container():
        with st.chat_message("assistant", avatar=ui.LOGO_AVATAR):
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
    turn["ta"] = time.time()
    save_turn(conv["log"].index(turn))
    st.rerun()
