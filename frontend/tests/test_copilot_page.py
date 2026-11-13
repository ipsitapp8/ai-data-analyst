"""Silt Copilot page (pages/12_Copilot.py) run headlessly against the real API, with the local model switched off:
no LLM is called. Covers answers, tables and charts in the chat, the message queue, the typing indicator, light/dark
input readability, uploads (CSV, Excel, PDF), errors that keep the conversation, and conversation navigation."""
from __future__ import annotations

import io
import re

import pandas as pd
import pytest

import api_client
from style import copilot_ui as ui
from test_pages import healthy, page, text_of  # noqa: F401 - shared harness (in-process API, fixtures)
from test_pages import api, ws  # noqa: F401


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    from app.analyst import model_info, planner
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    monkeypatch.setattr(model_info, "served", lambda url: {"verified": False})


def _ask(at, q):
    at.chat_input[0].set_value(q).run()
    return healthy(at)


def _assistant_text(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


# ------------------------------------------------------------------- answers --

def test_page_renders_empty_state_and_answers_a_question(api, ws):
    at = healthy(page("12_Copilot.py", ws))
    assert "Start with a question" in text_of(at)
    assert "Local model offline" in text_of(at)
    _ask(at, "Explain this dataset in simple terms.")
    text = _assistant_text(at)
    assert "In simple terms:" in text and "720 records" in text      # fixture: 720 rows
    assert len(at.chat_message) == 2
    assert any("analysis tools" in str(m.value) for m in at.markdown)       # the source line


def test_a_ranking_answer_shows_a_table_and_a_chart_built_from_the_data(api, ws):
    df = pd.read_csv(io.BytesIO(__import__("platform_helpers").SALES_CSV))
    truth = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    at = _ask(healthy(page("12_Copilot.py", ws)), "Show the top two products.")
    text = _assistant_text(at)
    top = truth.index[0]
    assert f"**{top}**" in text and f"{truth.iloc[0]:,.0f}" in text
    assert "| rank |" in text                                           # a markdown table in the bubble
    charts = at.get("vega_lite_chart")
    assert charts, "the ranking is drawn"


def test_follow_up_keeps_context_and_conversations_persist(api, ws):
    at = healthy(page("12_Copilot.py", ws))
    _ask(at, "Show the top two products.")
    _ask(at, "Explain this in simple terms.")
    assert len(at.chat_message) == 4
    assert "In short" in _assistant_text(at) or "top" in _assistant_text(at).lower()
    first = at.session_state["cp_active"]
    at.button[[b.label for b in at.button].index("＋ New conversation")].click().run()
    healthy(at)
    assert at.session_state["cp_active"] != first and len(at.chat_message) == 0
    at.button(key=f"cv_{first}").click().run()
    healthy(at)
    assert at.session_state["cp_active"] == first and len(at.chat_message) == 4


def test_queue_answers_every_question_in_order(api, ws):
    at = healthy(page("12_Copilot.py", ws))
    conv = at.session_state["cp_convs"][at.session_state["cp_active"]]
    conv["log"] += [{"q": "Find unusual records.", "r": None}, {"q": "Show the top two products.", "r": None}]
    at.run()
    healthy(at)
    log = at.session_state["cp_convs"][at.session_state["cp_active"]]["log"]
    assert [t["q"] for t in log] == ["Find unusual records.", "Show the top two products."]
    assert all(t["r"] and "answer" in t["r"] for t in log)
    assert len(at.chat_message) == 4


def test_a_duplicate_of_a_waiting_question_is_queued_once(api, ws):
    at = healthy(page("12_Copilot.py", ws))
    conv = at.session_state["cp_convs"][at.session_state["cp_active"]]
    conv["log"].append({"q": "Find unusual records.", "r": None})
    at.chat_input[0].set_value("Find unusual records.").run()
    healthy(at)
    log = at.session_state["cp_convs"][at.session_state["cp_active"]]["log"]
    assert len(log) == 1 and log[0]["r"]


def test_a_question_left_by_a_stopped_run_is_answered_once(api, ws, monkeypatch):
    import time
    calls = []
    real = api_client.analyst_ask
    monkeypatch.setattr(api_client, "analyst_ask", lambda *a, **k: calls.append(a) or real(*a, **k))
    at = healthy(page("12_Copilot.py", ws))
    conv = at.session_state["cp_convs"][at.session_state["cp_active"]]
    conv["log"].append({"q": "Find unusual records.", "r": None, "running_since": time.time() - 10_000})
    at.run()
    healthy(at)
    assert len(calls) == 1 and at.session_state["cp_convs"][at.session_state["cp_active"]]["log"][0]["r"]


def test_an_error_is_shown_without_losing_the_conversation(api, ws, monkeypatch):
    at = healthy(page("12_Copilot.py", ws))
    _ask(at, "Find unusual records.")
    monkeypatch.setattr(api_client, "analyst_ask", lambda *a, **k: (_ for _ in ()).throw(api_client.ApiError("503: busy")))
    at.chat_input[0].set_value("Show the top two products.").run()
    assert not at.exception
    assert at.error and "could not be answered" in at.error[0].value and "conversation is kept" in at.error[0].value
    assert len(at.chat_message) == 4                 # the earlier answer is still there
    assert any(b.label == "Try again" for b in at.button)


def test_details_are_in_an_optional_section_not_raw_json(api, ws):
    at = _ask(healthy(page("12_Copilot.py", ws)), "Find unusual records.")
    assert any(e.label == "How this was calculated" for e in at.expander)
    assert not at.json, "the technical trace is hidden unless asked for"


def test_theme_toggle_switches_to_dark(api, ws):
    at = healthy(page("12_Copilot.py", ws))
    assert at.session_state["cp_theme"] == "light"
    at.button(key="cp_theme_btn").click().run()
    healthy(at)
    assert at.session_state["cp_theme"] == "dark"
    assert any(ui.THEMES["dark"]["input_text"] in str(m.value) for m in at.markdown)


# ------------------------------------------------------------------- uploads --

def test_csv_excel_and_pdf_uploads_reach_the_copilot(api, ws):
    df = pd.DataFrame({"team": ["A", "B", "A"], "cost": [1.0, 2.0, 3.0]})
    up = api.post("/api/datasets/upload", files={"file": ("costs.csv", df.to_csv(index=False).encode(), "text/csv")},
                  headers=ws["h"])
    assert up.status_code == 200 and up.json()["row_count"] == 3
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    up = api.post("/api/datasets/upload", files={"file": ("costs.xlsx", buf.getvalue(), "application/octet-stream")},
                  headers=ws["h"])
    assert up.status_code == 200 and up.json()["col_count"] == 2
    pdf = _pdf_with_text("Revenue rose in the West region. Risks: supplier delays.")
    up = api.post("/api/analyst/documents", files={"file": ("report.pdf", pdf, "application/pdf")}, headers=ws["h"])
    assert up.status_code == 200, up.text
    d = up.json()
    assert d["page_count"] == 1 and d["text_pages"] == [1] and not d["scanned_pages"]
    bad = api.post("/api/analyst/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}, headers=ws["h"])
    assert bad.status_code == 400
    at = healthy(page("12_Copilot.py", ws))
    labels = " ".join(str(o) for s in at.selectbox for o in s.options)
    assert "costs.csv" in labels and "costs.xlsx" in labels and "report.pdf" in labels


def _pdf_with_text(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    w = PdfWriter()
    p = w.add_blank_page(612, 792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    p[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    s = DecodedStreamObject()
    s.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    p[NameObject("/Contents")] = w._add_object(s)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


# --------------------------------------------------------------- UI pieces --

def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_chat_input_text_placeholder_and_caret_are_readable(theme):
    t = ui.THEMES[theme]
    assert _contrast(t["input_text"], t["input_bg"]) >= 7
    assert _contrast(t["placeholder"], t["input_bg"]) >= 3
    assert _contrast(t["caret"], t["input_bg"]) >= 3
    css = ui.css(theme)
    for rule in (f"color:{t['input_text']} !important", f"-webkit-text-fill-color:{t['input_text']}",
                 f"caret-color:{t['caret']}", "::placeholder", "stChatInputSubmitButton", ":focus-within",
                 ":disabled", "overflow-x:auto", "overflow-wrap:break-word", "max-width:100%"):
        assert rule in css, rule
    assert "<style>" in css and "\n" not in css            # one line: markdown must not split the stylesheet


def test_typing_indicator_has_three_animated_dots_and_no_fake_progress():
    h = ui.typing_html("Thinking")
    assert h.count("class='dot'") == 3 and "role='status'" in h and "%" not in h
    assert "@keyframes cpBlink" in ui.css("light") and "prefers-reduced-motion" in ui.css("dark")


@pytest.mark.parametrize("spec", [
    {"type": "bar", "x": "dept", "y": "cost", "data": [{"dept": "A", "cost": 3.0}, {"dept": "B", "cost": 1.0}]},
    {"type": "bar", "x": "dept", "y": "value", "series": "case",
     "data": [{"dept": "A", "case": "now", "value": 1.0}, {"dept": "A", "case": "with the change", "value": 1.1}]},
    {"type": "line", "x": "period", "y": "revenue", "data": [{"period": "2025-01", "revenue": 5.0}, {"period": "2025-02", "revenue": 7.0}]},
    {"type": "pie", "x": "dept", "y": "cost", "data": [{"dept": "A", "cost": 3.0}, {"dept": "B", "cost": 1.0}]},
    {"type": "scatter", "x": "u", "y": "r", "data": [{"u": 1, "r": 2.0}, {"u": 2, "r": 4.0}, {"u": 3, "r": 5.0}]},
    {"type": "histogram", "x": "range", "y": "count", "data": [{"range": "0 – 1", "start": 0.0, "count": 2}]},
    {"type": "avp", "x": "actual", "y": "predicted", "data": [{"actual": 1.0, "predicted": 1.2}, {"actual": 2.0, "predicted": 1.9}]},
])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_chart_type_renders_the_backend_values_unchanged(spec, theme):
    c = ui.chart(spec, theme)
    assert c is not None
    d = c.to_dict()
    values = [row for ds in d.get("datasets", {}).values() for row in ds]
    for row in spec["data"]:
        assert row in values or {k: v for k, v in row.items()} in values
    assert d["config"]["background"] if "background" in d.get("config", {}) else True


def test_empty_chart_is_not_drawn():
    assert ui.chart({"type": "bar", "x": "a", "y": "b", "data": []}) is None


def test_file_labels():
    assert ui.file_kind("a.XLSX") == "Excel" and ui.file_kind("b.pdf") == "PDF" and ui.file_kind("c.csv") == "CSV"
    assert ui.human_size(2048) == "2 KB" and re.match(r"1\.\d MB", ui.human_size(1_200_000))


# ------------------------------------- regression: the same file uploaded twice --

def test_duplicate_file_names_get_distinct_selector_labels():
    labels = ui.unique_labels({"b64ede694ad94ead": "sales_report.pdf · 4 pages", "a30a90cbd648483f": "sales_report.pdf · 4 pages",
                               "c1": "other.pdf · 1 pages"})
    assert len(set(labels.values())) == 3
    assert labels["b64ede694ad94ead"].endswith("#b64ede") and labels["c1"] == "other.pdf · 1 pages"
    assert ui.unique_labels({7: "t.csv · 11 rows", 4: "t.csv · 11 rows"}) == {7: "t.csv · 11 rows · #7", 4: "t.csv · 11 rows · #4"}


def test_a_question_after_uploading_a_file_with_an_existing_name_is_answered(api, ws):
    """The browser reports a selectbox choice by its label; with two documents labelled alike the page used to switch
    to the older copy and re-run before reading the chat box, dropping the question."""
    import time
    pdf = _pdf_with_text("Next review date: 15 January.")
    ids = [api.post("/api/analyst/documents", files={"file": ("report.pdf", pdf, "application/pdf")}, headers=ws["h"]).json()["id"]
           for _ in range(2)]
    convs = {"c1": {"title": "New conversation", "dataset_id": None, "doc_id": ids[1], "log": [], "created": time.time()}}
    at = healthy(page("12_Copilot.py", ws, cp_convs=convs, cp_active="c1"))
    doc_box = next(s for s in at.selectbox if s.label == "Document")
    shown = list(doc_box.options)                      # AppTest exposes the labels the browser sees
    assert len(set(shown)) == len(shown) == 3, shown     # none + two copies, each with its own label
    assert doc_box.value == ids[1]
    at.chat_input[0].set_value("When is the next review date?").run()
    healthy(at)
    log = at.session_state["cp_convs"]["c1"]["log"]
    assert at.session_state["cp_active"] == "c1" and at.session_state["cp_convs"]["c1"]["doc_id"] == ids[1]
    assert [t["q"] for t in log] == ["When is the next review date?"] and "15 January" in log[0]["r"]["answer"]
