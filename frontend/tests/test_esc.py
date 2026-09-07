"""esc() is the one thing standing between user-controlled text and a stored
XSS across every page (see style/theme.py's docstring on it). Codifies the
fix from the hardening pass so a future edit can't silently regress it.
"""
from __future__ import annotations

from style.theme import esc


def test_escapes_script_tags():
    assert "<script>" not in esc("<script>alert(1)</script>")


def test_escapes_quotes_for_attribute_context():
    # quote=True so a value can't break out of a double-quoted HTML attribute
    # (e.g. `<div title="{esc(value)}">`).
    out = esc('" onmouseover="alert(1)')
    assert '"' not in out


def test_plain_text_survives_unescaped_semantically():
    assert "hello world" in esc("hello world")


def test_handles_non_string_input():
    assert esc(42) == "42"
    assert esc(None) == "None"
