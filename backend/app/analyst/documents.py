"""PDF documents for the analyst: text extraction per page, honest reporting of
pages that have no extractable text (scanned images need OCR, which is not
available here), and search that returns passages with their page numbers.

A document is not a dataset: it is never fed to statistics or machine learning.
Answers about a document quote it, with page references.

Storage: DATA_DIR/analyst_documents/<team_id>/<doc_id>.pdf and <doc_id>.json
(the extracted pages). No database table is needed.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

from app import config

ROOT = Path(config.DATA_DIR) / "analyst_documents"
MAX_BYTES = 25 * 1024 * 1024
MIN_PAGE_CHARS = 25           # fewer extractable characters than this on a page with images: treated as scanned
PASSAGE_CHARS = 700
_WORD = re.compile(r"[a-z0-9][a-z0-9'%.-]*")
_STOP = set("""a an and are as at be by can did do does for from has have how i in is it its of on or our that the
their there these this to was we were what when where which who why will with you your about into than then them
they not no yes per tell me show give document pdf report page pages say says said""".split())


class DocumentError(ValueError):
    """The file cannot be used as a document; the message says why."""


def _dir(team_id: int) -> Path:
    d = ROOT / str(int(team_id))
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract(path: Path) -> dict:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:  # noqa: BLE001
                raise DocumentError("The PDF is password-protected.") from e
        pages = []
        for i, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one bad page must not lose the others
                text = ""
            text = re.sub(r"[ \t]+", " ", text).strip()
            try:
                images = len(page.images)
            except Exception:  # noqa: BLE001
                images = 0
            pages.append({"page": i, "text": text, "chars": len(text), "images": images})
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as e:
        raise DocumentError(f"The file is not a readable PDF ({type(e).__name__}).") from e
    if not pages:
        raise DocumentError("The PDF has no pages.")
    scanned = [p["page"] for p in pages if p["chars"] < MIN_PAGE_CHARS and p["images"] > 0]
    empty = [p["page"] for p in pages if p["chars"] < MIN_PAGE_CHARS and p["images"] == 0]
    return {"pages": pages, "page_count": len(pages), "scanned_pages": scanned, "empty_pages": empty,
            "text_pages": [p["page"] for p in pages if p["chars"] >= MIN_PAGE_CHARS],
            "chars": sum(p["chars"] for p in pages)}


def save(team_id: int, filename: str, data: bytes) -> dict:
    if not filename.lower().endswith(".pdf"):
        raise DocumentError("Only PDF documents are supported.")
    if len(data) > MAX_BYTES:
        raise DocumentError(f"The PDF is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF"):
        raise DocumentError("The file does not start like a PDF; it may be damaged or another format renamed to .pdf.")
    doc_id = uuid.uuid4().hex[:16]
    d = _dir(team_id)
    pdf = d / f"{doc_id}.pdf"
    pdf.write_bytes(data)
    try:
        info = extract(pdf)
    except DocumentError:
        pdf.unlink(missing_ok=True)
        raise
    meta = {"id": doc_id, "filename": Path(filename).name, **info}
    (d / f"{doc_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return summary(meta)


def summary(meta: dict) -> dict:
    return {k: meta[k] for k in ("id", "filename", "page_count", "scanned_pages", "empty_pages", "text_pages", "chars")}


def load(team_id: int, doc_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{16}", doc_id or ""):
        raise DocumentError("Unknown document.")
    p = _dir(team_id) / f"{doc_id}.json"
    if not p.exists():
        raise DocumentError("Unknown document.")
    return json.loads(p.read_text(encoding="utf-8"))


def list_docs(team_id: int) -> list[dict]:
    out = []
    for p in sorted(_dir(team_id).glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            out.append(summary(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001
            continue
    return out


def _tokens(text: str) -> list[str]:
    return [w.strip(".'-") for w in _WORD.findall(text.lower()) if w.strip(".'-") not in _STOP and len(w.strip(".'-")) > 1]


def passages(meta: dict) -> list[dict]:
    out = []
    for p in meta["pages"]:
        paras = [x.strip() for x in re.split(r"\n\s*\n|\n(?=[A-Z0-9•\-])", p["text"]) if x.strip()]
        buf = ""
        for para in paras:
            if buf and len(buf) + len(para) > PASSAGE_CHARS:
                out.append({"page": p["page"], "text": buf.strip()})
                buf = ""
            buf += ("\n" if buf else "") + para
        if buf.strip():
            out.append({"page": p["page"], "text": buf.strip()})
    return out


def search(meta: dict, query: str, k: int = 4) -> list[dict]:
    """BM25 over passages; returns the best passages with page numbers and the matched words."""
    ps = passages(meta)
    q = list(dict.fromkeys(_tokens(query)))
    if not ps or not q:
        return []
    docs = [_tokens(p["text"]) for p in ps]
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    scored = []
    for p, d in zip(ps, docs):
        tf = Counter(d)
        score = 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if score > 0:
            scored.append({**p, "score": round(score, 3), "matched": [w for w in q if tf[w]]})
    scored.sort(key=lambda x: -x["score"])
    return scored[:k]


def overview(meta: dict) -> dict:
    words = Counter(w for p in meta["pages"] for w in _tokens(p["text"]) if not re.fullmatch(r"[\d.,%-]+", w))
    firsts = []
    for p in meta["pages"]:
        line = next((ln.strip() for ln in p["text"].splitlines() if len(ln.strip()) > 3), "")
        if line:
            firsts.append({"page": p["page"], "text": line[:160]})
    return {"filename": meta["filename"], "page_count": meta["page_count"], "text_pages": meta["text_pages"],
            "scanned_pages": meta["scanned_pages"], "empty_pages": meta["empty_pages"],
            "key_terms": [w for w, _ in words.most_common(12)], "page_openings": firsts[:12]}
