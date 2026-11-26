"""Data helpers for the Home page: suggested questions drawn from the user's own datasets.

Pure functions (no Streamlit, no network) so they can be unit-tested. Questions are built from
column *names and kinds* in the stored dataset profile; no values are read and no model is called.
"""
from __future__ import annotations

from typing import Any

GENERIC_QUESTIONS = [
    "Which region grew fastest last quarter?",
    "Why did revenue dip in March?",
    "Which customers are most at risk of churning?",
    "Is the lift from the new campaign significant?",
]

# Columns that are identifiers, not something to total or average.
_ID_HINTS = ("id", "uuid", "index", "key", "zip", "postcode", "phone")


def _label(name: str) -> str:
    return str(name).replace("_", " ").strip().lower()


def _is_identifier(col: dict[str, Any], row_count: int) -> bool:
    name = str(col.get("name", "")).lower()
    tokens = name.replace("-", "_").split("_")
    if name == "id" or any(t in _ID_HINTS for t in tokens):
        return True
    unique = col.get("unique_count") or 0
    return bool(row_count and unique >= row_count and col.get("kind") != "numeric")


def _columns(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    cols = (dataset.get("profile") or {}).get("columns") or []
    return [c for c in cols if isinstance(c, dict) and c.get("name")]


def dataset_questions(dataset: dict[str, Any], limit: int = 4) -> list[str]:
    """Up to `limit` questions that refer to real columns of one dataset."""
    rows = int(dataset.get("row_count") or 0)
    cols = _columns(dataset)
    numeric = [c for c in cols if c.get("kind") == "numeric" and not _is_identifier(c, rows)]
    groups = [c for c in cols if c.get("kind") == "categorical" and not _is_identifier(c, rows)
              and 2 <= (c.get("unique_count") or 0) <= 30]
    dates = [c for c in cols if c.get("kind") == "datetime"]

    out: list[str] = []
    if numeric and groups:
        out.append(f"Which {_label(groups[0]['name'])} has the highest {_label(numeric[0]['name'])}?")
    if numeric and dates:
        out.append(f"How has {_label(numeric[0]['name'])} changed over time?")
        out.append(f"Why did {_label(numeric[0]['name'])} change last month?")
    if len(numeric) >= 2:
        out.append(f"Is {_label(numeric[0]['name'])} related to {_label(numeric[1]['name'])}?")
    if len(groups) >= 2 and numeric:
        out.append(f"Compare {_label(numeric[0]['name'])} across {_label(groups[1]['name'])}.")
    if not out and numeric:
        out.append(f"What is the average {_label(numeric[0]['name'])}, and are there outliers?")
    # Keep order, drop repeats.
    seen: set[str] = set()
    return [q for q in out if not (q in seen or seen.add(q))][:limit]


def suggestions(datasets: list[dict[str, Any]], per_dataset: int = 2, limit: int = 4
                ) -> list[tuple[int | None, str]]:
    """(dataset_id, question) pairs from the newest datasets first; generic fallback with id None."""
    picked: list[tuple[int | None, str]] = []
    for ds in sorted(datasets, key=lambda d: str(d.get("uploaded_at") or ""), reverse=True):
        for q in dataset_questions(ds, per_dataset):
            picked.append((ds.get("id"), q))
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for q in GENERIC_QUESTIONS:
            if len(picked) >= limit:
                break
            if all(q != p[1] for p in picked):
                picked.append((None, q))
    return picked[:limit]


def has_active_run(questions: list[dict[str, Any]]) -> bool:
    done = {"verified", "unverified", "failed", "rejected", "cancelled", "needs_clarification"}
    return any(str(q.get("status")) not in done for q in questions)
