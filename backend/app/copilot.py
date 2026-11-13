"""Stateful analytical copilot: a multi-turn session over one dataset version.

The session holds an explicit analytical state -- metric, filters, grouping --
and every user message is turned into controlled operations on that state:

    set_metric · add_filter · remove_filter · clear_filters · set_time_range
    drill_down · drill_up · rebase (move to the latest data) · reset

After each turn the answer is recomputed from the dataset by the deterministic
metric engine. The conversation is a record of decisions, never the source of
a number: nothing is answered from memory, and when a message refers to
something the state does not contain, the copilot says so instead of inventing
it.

A message is interpreted by rules first. Only when no rule applies, and a model
is configured, is a model asked to translate the message into operations --
and its output is validated against the same whitelist and the dataset's real
columns before anything runs. A model never computes the answer.

Ambiguity produces a question back to the user (which column? which metric?),
with the options listed, and leaves the state unchanged.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy.orm import Session

from app import fastpath, interpretation, provenance, semantic
from app import metric_engine as me
from app.dataset_versions import latest_version
from app.models import CopilotSession, CopilotTurn, Dataset, DatasetVersion

logger = logging.getLogger(__name__)

OPERATIONS = ("set_metric", "add_filter", "remove_filter", "clear_filters", "set_time_range",
              "drill_down", "drill_up", "rebase", "reset")
MAX_TURNS = 200
MAX_MESSAGE_CHARS = 500
MAX_CATEGORY_VALUES = 300
MAX_LARGE_COLUMN_VALUES = 50_000   # larger columns are read only next to their name (see ValueIndex)
RESULT_LIMIT = 25
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
_NEEDS_ANALYSIS = re.compile(r"\bwhy\b|root cause|forecast|predict|trend|correlat|outlier|anomal|explain", re.IGNORECASE)

ACTIONS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OPERATIONS)},
                    "formula": {"type": "string", "description": "For set_metric, e.g. sum(revenue)"},
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "in", "contains"]},
                    "value": {"type": "string"},
                    "start": {"type": "string"}, "end": {"type": "string"},
                },
                "required": ["op"],
            },
        },
        "cannot_map": {"type": "boolean", "description": "True if the message is not one of these operations."},
    },
    "required": ["operations", "cannot_map"],
}
ACTIONS_SYSTEM = (
    "You translate one user message into operations on an analytical state (metric, filters, group-by). "
    "Use ONLY the listed operations and ONLY the column names given. Never answer the question yourself and "
    "never invent a column, a value or a number. If the message is not one of these operations, set cannot_map."
)


class CopilotError(ValueError):
    pass


def empty_state() -> dict:
    return {"formula": None, "metric_label": None, "filters": [], "group_by": [], "history_depth": 0}


# ------------------------------------------------------------ interpretation --

def _phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(text).strip().lower()))


class ValueIndex(dict):
    """lower-cased value -> [(column, original value)], plus `large`: columns with
    too many values to be matched anywhere in a question, kept as
    {column: {lower-cased value: original}} to be read only next to the column's name."""
    large: dict[str, dict[str, str]]


def category_index(df: pd.DataFrame, profile: dict) -> dict[str, list[tuple[str, str]]]:
    """lower-cased value -> [(column, original value)] for categorical columns."""
    index = ValueIndex()
    index.large = {}
    for c in profile.get("columns", []):
        if c.get("kind") != "categorical" or c["name"] not in df.columns:
            continue
        values = df[c["name"]].dropna().astype(str).unique()
        if len(values) > MAX_CATEGORY_VALUES:
            if len(values) <= MAX_LARGE_COLUMN_VALUES:
                index.large[c["name"]] = {_phrase(v): v for v in values}
            continue
        for v in values:
            index.setdefault(_phrase(v), []).append((c["name"], v))
    return index


def _time_columns(profile: dict) -> list[str]:
    return [c["name"] for c in profile.get("columns", []) if c.get("kind") == "datetime"]


def _parse_time_range(text: str) -> tuple[str | None, str | None, tuple[int, int]] | None:
    """(start, end, span of the matched phrase) from the raw, lower-cased
    message -- before punctuation is normalised, so ISO dates are still intact."""
    iso = r"(\d{4}-\d{2}-\d{2})"
    m = re.search(rf"\b(?:from|between)\s+{iso}\s+(?:to|and|until|through)\s+{iso}", text)
    if m:
        return m.group(1), m.group(2), m.span()
    m = re.search(rf"\b(?:since|after|from)\s+{iso}", text)
    if m:
        return m.group(1), None, m.span()
    m = re.search(rf"\b(?:before|until|up to)\s+{iso}", text)
    if m:
        return None, m.group(1), m.span()
    m = re.search(r"\b(?:in|during|for)\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", text)
    if m:
        month, year = _MONTHS[m.group(1)], int(m.group(2))
        end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return f"{year:04d}-{month:02d}-01", end.strftime("%Y-%m-%d"), m.span()
    m = re.search(r"\b(?:in|during|for)\s+(?:the year\s+)?((?:19|20)\d{2})\b", text)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(1)}-12-31", m.span()
    return None


# ------------------------------------------------- question reading helpers --
#
# General rules for reading one question. Each one either accounts for words of
# the question exactly or leaves them alone; a word nothing accounts for still
# makes the copilot decline, so these widen what is understood without guessing.

# Words a dataset's file name carries that do not name what its rows are.
_GENERIC_NAME_WORDS = {"data", "dataset", "table", "info", "information", "details", "list", "log", "csv", "file",
                       "export", "sheet", "records", "record", "rows", "row", "all", "final", "new", "old", "copy"}
# Words that may be left over in a question once its metric and filters are read.
_FILLER = {"there", "exist", "exists", "existing", "total", "database", "find", "return", "compute", "calculate",
           "display", "get", "provide", "have", "has", "do", "does", "we", "us", "our", "these", "those",
           "listed", "recorded", "registered", "altogether", "currently", "overall", "whole", "entire",
           # connectives between a metric and conditions that were read as filters; if a
           # condition was not read, its own words are still left over and the copilot declines
           "whose", "who", "which", "that", "where", "either", "or"}
# Words that are never read as a category value on their own, even if a column holds them.
_NOT_A_VALUE = fastpath._STOP | {"how", "many", "much", "total", "sum", "average", "mean", "avg", "median", "max",
                                 "maximum", "min", "minimum", "highest", "lowest", "largest", "smallest", "biggest",
                                 "count", "number", "different", "distinct", "unique", "than", "more", "less", "least",
                                 "most", "over", "under", "above", "below", "between", "only", "not", "without"}
_AGG_TOKEN = (r"average|mean|avg|median|minimum|min|lowest|smallest|maximum|max|highest|largest|biggest|"
              r"total|sum of|sum")
_NUM = r"-?\d+(?:\.\d+)?"
# Comparison phrases -> operator. Longest phrases first; every one must be followed by a number.
_COMPARATORS = [
    # "not higher than 4" is "at most 4"; the negated phrases come first so "higher than" never reads half of one
    (r"no less than|not less than|at least|greater than or equal to|more than or equal to|"
     r"not (?:fewer|lower|smaller|younger|shorter|lighter) than|not (?:under|below)", ">="),
    (r"no more than|not more than|at most|less than or equal to|"
     r"not (?:greater|higher|larger|bigger|heavier|older|taller|longer) than|not (?:over|above|exceeding)", "<="),
    (r"more than|greater than|higher than|larger than|bigger than|heavier than|older than|taller than|longer than|"
     r"later than|over|above|exceeding|after", ">"),
    (r"less than|fewer than|lower than|smaller than|lighter than|younger than|shorter than|earlier than|under|"
     r"below|before", "<"),
]
_CMP_PATTERN = "|".join(p for p, _ in _COMPARATORS)
# A comparative adjective names its quantity: "heavier than 10" is about weight.
_IMPLIED = {"heavier": "weight", "lighter": "weight", "older": "age", "younger": "age", "taller": "height",
            "shorter": "height", "longer": "length"}
_YEARISH = re.compile(r"\byears?\b|\byr\b")


def _op_for(phrase: str) -> str:
    for pattern, op in _COMPARATORS:
        if re.fullmatch(pattern, phrase):
            return op
    raise ValueError(phrase)


def _number_forms(phrase: str) -> set[str]:
    """Singular and plural spellings of a phrase's last word."""
    words = phrase.split()
    if not words:
        return set()
    head, last = words[:-1], words[-1]
    forms = {last, last + "s", last + "es"}
    if last.endswith("ies") and len(last) > 4:
        forms.add(last[:-3] + "y")
    if last.endswith("y") and len(last) > 2:
        forms.add(last[:-1] + "ies")
    if last.endswith("es") and len(last) > 3:
        forms.add(last[:-2])
    if last.endswith("s") and len(last) > 2:
        forms.add(last[:-1])
    return {" ".join(head + [f]) for f in forms}


def entity_phrases(dataset_name: str | None) -> list[str]:
    """What a row of the dataset is, read from its file name: "singer.csv" ->
    singer, singers. Longest first. Empty when the name says nothing usable."""
    if not dataset_name:
        return []
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", dataset_name.strip()).split("__")[-1]
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem).lower()
    words = [w for w in re.split(r"[^a-z]+", stem) if w and w not in _GENERIC_NAME_WORDS]
    if not words:
        return []
    words = words[-3:]
    # every word in its singular or plural spelling: "customers_cards" -> "customer cards", ...
    combos = [[]]
    for i, w in enumerate(words):
        spellings = _number_forms(w) if i < len(words) - 1 else {w}
        combos = [c + [s] for c in combos for s in sorted(spellings | {w})]
    forms = {f for c in combos for f in _number_forms(" ".join(c))} | _number_forms(words[-1])
    return sorted(forms, key=lambda f: (-len(f), f))


def _plural(word: str) -> str:
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + ("es" if re.search(r"(?:s|x|z|ch|sh)$", word) else "s")


# Words that never take part in a reordered column phrase: "num of factories" is
# not "factories num", and an identifier stays as written.
_NO_REORDER = {"num", "number", "no", "nbr", "count", "total", "id", "code", "key", "ref", "is", "has", "in", "per"}


def _inner_plurals(phrase: str) -> set[str]:
    """"final table made" -> "final tables made": a plural on a word before the
    last one. (The last word's plural is read wherever a column is matched.)"""
    words = phrase.split()
    if len(words) < 2:
        return set()
    out = set()
    for i, w in enumerate(words[:-1]):
        if len(w) > 2 and w.isalpha() and w not in fastpath._STOP and w not in _NO_REORDER:
            out.add(" ".join(words[:i] + [_plural(w)] + words[i + 1:]))
    return out


def _reordered(phrase: str) -> set[str]:
    """The same two-word column in the other order: PetType -> "type of pet",
    GovernmentForm -> "form of government", Level_of_membership -> "membership
    level". Only two content words, so nothing is guessed about longer names."""
    words = phrase.split()
    if len(words) == 3 and words[1] == "of":
        a, b = words[2], words[0]               # "level of membership" -> "membership level"
        return set() if {a, b} & (_NO_REORDER | fastpath._STOP) or not (a.isalpha() and b.isalpha()) \
            else {f"{a} {b}"}
    if len(words) == 2:
        a, b = words                            # "pet type" -> "type of pet", "types of pet"
        if {a, b} & (_NO_REORDER | fastpath._STOP) or not (a.isalpha() and b.isalpha()) or len(a) < 3 or len(b) < 3:
            return set()
        return {f"{b} of {a}", f"{_plural(b)} of {a}"}
    return set()


def column_aliases(profile: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Other phrasings of each column. `aliases` apply everywhere (CamelCase
    split, -y/-ies plural); `measure` only to measured quantities ("factories"
    for Num_of_Factories), never to counts, where they would change the meaning."""
    aliases: dict[str, list[str]] = {}
    measure: dict[str, list[str]] = {}
    for c in profile.get("columns", []):
        name = c["name"]
        base = _phrase(name)
        camel = _phrase(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name))
        found = {camel} | {f for p in (base, camel) for f in _number_forms(p) if f.endswith("ies")}
        for p in {base, camel}:
            found |= _inner_plurals(p) | _reordered(p)
        aliases[name] = sorted(found - {base})
        for p in {base, camel}:
            m = re.match(r"(?:num|number|no|nbr|count|total)\s+(?:of\s+)?(.+)$", p)
            if m and m.group(1) not in ("", base):
                measure.setdefault(name, []).append(m.group(1))
    return aliases, measure


def _column_source(name: str, matched: str) -> str:
    """Evidence source for words that named a column: its own name in any
    spelling (case, separators, CamelCase, singular/plural) is "column_name";
    a phrasing the rules derived from it (reordered words, "factories" for
    Num_of_Factories) is "column_phrasing"."""
    base = _phrase(name)
    camel = _phrase(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name))
    own = {f for p in (base, camel) for f in _number_forms(p) | {p}}
    return "column_name" if _phrase(matched) in own else "column_phrasing"


def _column_matches(text: str, profile: dict, aliases: dict, kinds: tuple = ("numeric",)) -> list[tuple[str, re.Match]]:
    out = []
    for c in profile.get("columns", []):
        if c.get("kind") not in kinds:
            continue
        for p in [_phrase(c["name"]), *aliases.get(c["name"], [])]:
            for f in _number_forms(p) | {p}:
                for m in re.finditer(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", text):
                    out.append((c["name"], m))
    return out


def _read_comparisons(text: str, profile: dict, aliases: dict, measure: dict,
                      trace=interpretation.NULL_TRACE) -> tuple[list[dict], str]:
    """"weight heavier than 10", "price of at least 180", "more than 20 shops",
    "heavier than 10" (the adjective names the column) -> numeric filters.
    Only numeric columns; the phrase read is removed from the text."""
    ops: list[dict] = []
    every = {**{k: list(v) for k, v in aliases.items()}}
    for k, v in measure.items():
        every.setdefault(k, []).extend(v)
    numeric = {c["name"] for c in profile.get("columns", []) if c.get("kind") == "numeric"}
    link = r"(?:\s+(?:is|are|was|were|of|with|has|have|had|being|that is|which is))?"
    while True:
        found = None
        parts: list[tuple] = []   # (start, end, source, role, rule): the words read, for the interpretation record
        cols = sorted(_column_matches(text, profile, every), key=lambda h: -(h[1].end() - h[1].start()))
        # 1. <column> [is|of|...] <comparator> <number>
        for name, cm in cols:
            m = re.match(rf"{link}\s+({_CMP_PATTERN})\s+({_NUM})(?!\.?\d)", text[cm.end():])
            if m:
                found = (name, _op_for(m.group(1)), m.group(2), cm.start(), cm.end() + m.end())
                rule = "copilot.comparison.column_first"
                parts = [(cm.start(), cm.end(), _column_source(name, cm.group(0)), "column", rule),
                         (cm.end() + m.start(1), cm.end() + m.end(1), "comparison_phrase", "operator", rule),
                         (cm.end() + m.start(2), cm.end() + m.end(2), "literal", "value", rule)]
                break
        # 2. <comparator> <number> <column>: "more than 20 shops"
        if not found:
            for m in re.finditer(rf"(?<![a-z0-9])({_CMP_PATTERN})\s+({_NUM})\s+", text):
                hit = next(((n, cm) for n, cm in cols if cm.start() == m.end()), None)
                if hit:
                    found = (hit[0], _op_for(m.group(1)), m.group(2), m.start(), hit[1].end())
                    rule = "copilot.comparison.number_first"
                    parts = [(m.start(1), m.end(1), "comparison_phrase", "operator", rule),
                             (m.start(2), m.end(2), "literal", "value", rule),
                             (hit[1].start(), hit[1].end(), _column_source(hit[0], hit[1].group(0)), "column", rule)]
                    break
        # 3. <more|less|...> <column> than <number>: "greater weight than 10"
        if not found:
            for name, cm in cols:
                before = re.search(r"(more|greater|higher|larger|bigger|less|fewer|lower|smaller)\s+$", text[:cm.start()])
                after = re.match(rf"\s+than\s+({_NUM})(?!\.?\d)", text[cm.end():])
                if before and after:
                    op = ">" if before.group(1) in ("more", "greater", "higher", "larger", "bigger") else "<"
                    found = (name, op, after.group(1), before.start(), cm.end() + after.end())
                    rule = "copilot.comparison.comparative_column"
                    parts = [(before.start(1), before.end(1), "comparison_phrase", "operator", rule),
                             (cm.start(), cm.end(), _column_source(name, cm.group(0)), "column", rule),
                             (cm.end(), cm.end() + after.start(1), "comparison_phrase", "operator", rule),
                             (cm.end() + after.start(1), cm.end() + after.end(1), "literal", "value", rule)]
                    break
        # 4. <adjective> than <number>, the adjective naming exactly one numeric column
        if not found:
            for m in re.finditer(rf"(?<![a-z0-9])((?:not\s+)?)({'|'.join(_IMPLIED)}) than\s+({_NUM})(?!\.?\d)", text):
                stem = _IMPLIED[m.group(2)]
                owners = [n for n in numeric if stem in _phrase(n).split()]
                if len(owners) == 1:
                    # "not older than 40" is "at most 40", never "older than 40" with a stray "not"
                    found = (owners[0], _op_for(f"{m.group(1)}{m.group(2)} than"), m.group(3), m.start(), m.end())
                    rule = "copilot.comparison.implied_adjective"
                    parts = [(m.start(), m.start(3), "comparison_phrase", "operator", rule),
                             (m.start(2), m.end(2), "column_inference", "column", rule),
                             (m.start(3), m.end(3), "literal", "value", rule)]
                    break
        if not found:
            return ops, text
        name, op, number, start, end = found
        if name not in numeric:
            return ops, text
        new_op = {"op": "add_filter", "column": name, "operator": op, "value": float(number)}
        ops.append(new_op)
        for s, e, source, role, rule in parts:
            trace.note(new_op, text, s, e, source=source, rule=rule, role=role,
                       ref={"column": name} if role == "column" else None)
        before_text = text
        text = text[:start] + " " + text[end:]
        trace.edit(before_text, start, end, " ", "copilot.read_comparisons")


def _read_bare_years(raw: str, profile: dict, trace=interpretation.NULL_TRACE) -> tuple[list[dict] | dict | None, str]:
    """"in 1980", "in year 2014 or 2015" when the dataset has no date column but
    one numeric year column. Returns (operations, raw text without the phrase),
    a clarification when several year columns fit, or (None, raw) untouched."""
    if any(c.get("kind") == "datetime" for c in profile.get("columns", [])):
        return None, raw
    year = r"(?:1[5-9]|20)\d{2}"
    m = re.search(rf"\b(?:in|during|for|from|of|on)\s+(?:the\s+)?(?:years?\s+)?(?:of\s+)?({year}(?:\s*(?:,|or|and)\s*{year})*)\b", raw)
    if not m:
        return None, raw
    years = [float(y) for y in re.findall(year, m.group(1))]
    cols = [c["name"] for c in profile.get("columns", []) if c.get("kind") == "numeric" and _YEARISH.search(_phrase(c["name"]))]
    if not cols:
        return None, raw
    if len(cols) > 1:
        reason = "This dataset has more than one year column. Which one does the year refer to?"
        trace.ambiguity(slot="filter", reason=reason, text=raw, start=m.start(), end=m.end(),
                        candidates=[{"column": c, "operator": "==", "value": years[0]} for c in cols])
        return {"clarification": {"reason": reason,
                                  "options": [{"label": c, "message": f"{raw[:m.start()]}where {c} = {int(years[0])}{raw[m.end():]}"}
                                              for c in cols]}}, raw
    op = {"op": "add_filter", "column": cols[0], "operator": "==" if len(years) == 1 else "in",
          "value": years[0] if len(years) == 1 else years}
    trace.note(op, raw, m.start(1), m.end(1), source="literal", rule="copilot.bare_year", role="value")
    trace.note(op, raw, m.start(), m.end(), source="column_inference", rule="copilot.bare_year.only_year_column",
               role="column", ref={"column": cols[0]})
    trace.edit(raw, m.start(), m.end(), " ", "copilot.bare_year")
    return [op], raw[:m.start()] + " " + raw[m.end():]


# What may follow the counted entity: a verb, a preposition, a connective, or nothing.
# Another noun means the entity word was a modifier ("how many movie directors").
_AFTER_ENTITY = (r"(?=\s*(?:$|[.?!,;:]|(?:the|are|is|were|was|do|does|did|have|has|had|in|with|who|that|which|whose|whom|from|of|"
                 r"there|by|for|on|at|exist|exists|listed|recorded|registered|not|and|or|than|without|excluding|"
                 r"located|based|total|altogether|we|you|they|records?|rows?|entries)\b))")


def _read_row_count(text: str, entities: list[str], profile: dict | None = None,
                    trace=interpretation.NULL_TRACE) -> str:
    """"how many singers", "the number of singers", "count the singers" on a
    dataset of singers count its rows: the entity becomes "rows", which the
    fast path reads as count(). Only the dataset's own entity is read this way;
    "how many customers" on a sales table stays unanswered."""
    if not entities:
        return text
    ent = "|".join(re.escape(e) for e in entities)
    count = r"how many|(?:the\s+)?(?:total\s+)?number of|count(?:\s+of)?(?:\s+the)?(?:\s+total)?(?:\s+number of)?"
    inside = [cm.span() for _, cm in _column_matches(text, profile or {}, {}, ("numeric", "categorical", "datetime"))]
    for m in re.finditer(rf"\b({count})\s+(?:the\s+|all\s+)?(?:{ent})(?![a-z0-9]){_AFTER_ENTITY}", text):
        if any(s <= m.start() < e for s, e in inside):
            continue   # "count" inside a column's name ("access count of documents")
        if trace:
            # the same pattern with the entity captured: which of its spellings was read
            em = re.compile(rf"\b({count})\s+(?:the\s+|all\s+)?({ent})(?![a-z0-9]){_AFTER_ENTITY}").match(text, m.start())
            if em is not None and em.end() == m.end():
                trace.set_entity(text, em.start(2), em.end(2), meaning="rows", rule="copilot.row_count_entity",
                                 ref={"entity": em.group(2)})
            trace.edit(text, m.end(1), m.end(), " rows", "copilot.row_count_entity")
        return text[:m.start()] + f"{m.group(1)} rows" + text[m.end():]
    return text


def _drop_entity_filler(text: str, entities: list[str], profile: dict, aliases: dict | None = None,
                        trace=interpretation.NULL_TRACE) -> str:
    """"... of all the payments", "... of the Marketing and Finance departments":
    naming the dataset's own entity says nothing about the computation, so the
    word is read as filler -- unless it is also a column's name, where it may
    be the metric ("count of distinct artists"), or part of one as written
    ("types of pets" for PetType on a table of pets)."""
    columns = {f for c in profile.get("columns", []) for f in _number_forms(_phrase(c["name"])) | {_phrase(c["name"])}}
    ent = [e for e in entities if e not in columns]
    if not ent:
        return text
    inside = [cm.span() for _, cm in _column_matches(text, profile, aliases or {}, ("numeric", "categorical", "datetime"))
              if " " in cm.group(0)]

    def drop(m: re.Match) -> str:
        return m.group(0) if any(s < m.end() and m.start() < e for s, e in inside) else " " * len(m.group(0))

    alt = "|".join(re.escape(e) for e in ent)
    patterns = (rf"\b(?:of|for|across|among|in|from|on)\s+(?:all\s+)?(?:the\s+|these\s+|those\s+)?(?:{alt})(?![a-z0-9])",
                rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])")
    for pattern in patterns:
        before = text
        text = re.sub(pattern, drop, text)
        if trace:
            for m in re.finditer(pattern, before):
                if not drop(m).strip():
                    trace.set_entity(before, m.start(), m.end(), meaning="context", rule="copilot.entity_filler",
                                     ref={"entity": m.group(0).split()[-1]})
            trace.resub(before, text, pattern, drop, "copilot.entity_filler")
    return text


def _consume_filter_column_words(text: str, ops: list[dict], profile: dict, aliases: dict,
                                 trace=interpretation.NULL_TRACE) -> str:
    """"products in the category Seeds": once Seeds became a filter on
    product_category_code, the words naming that column ("category") are read
    too -- unless another column's name also uses them."""
    names = [c["name"] for c in profile.get("columns", [])]
    words_of = {n: set(_phrase(n).split()) | {w for a in aliases.get(n, []) for w in a.split()} for n in names}
    for col in {o["column"] for o in ops if o["op"] == "add_filter"}:
        others = set().union(*(words_of[n] for n in names if n != col)) if len(names) > 1 else set()
        for w in sorted(words_of.get(col, set()) - others - fastpath._STOP, key=len, reverse=True):
            if len(w) > 1:
                pattern = rf"(?<![a-z0-9])(?:{'|'.join(re.escape(f) for f in _number_forms(w) | {w})})(?![a-z0-9])"
                before = text
                text = re.sub(pattern, " ", text)
                if trace and text != before:
                    source = "column_name" if w in _phrase(col).split() else "column_phrasing"
                    for m in re.finditer(pattern, before):
                        for o in ops:
                            if o["op"] == "add_filter" and o["column"] == col:
                                trace.note(o, before, m.start(), m.end(), source=source, role="column",
                                           rule="copilot.filter_column_words", ref={"column": col})
                    trace.resub(before, text, pattern, " ", "copilot.filter_column_words")
    return text


_ID_WORDS = {"id", "code", "key", "no", "num", "number", "nbr", "ref", "zip", "phone"}


def _read_count_of_measure(text: str, profile: dict, aliases: dict, measure: dict,
                           trace=interpretation.NULL_TRACE) -> str | dict:
    """"total number of hours" is a sum of the hours column. "how many hours
    ... in total" is too. "how many hours" alone could be a total or a count of
    rows, so the copilot asks. Identifier columns (…_id, …_code) are left alone:
    counting them is meaningful."""
    forms: dict[str, str] = {}
    for c in profile.get("columns", []):
        base = _phrase(c["name"])
        if c.get("kind") != "numeric" or (base.split() or [""])[-1] in _ID_WORDS:
            continue
        for p in [base, *aliases.get(c["name"], []), *measure.get(c["name"], [])]:
            for f in _number_forms(p) | {p}:
                forms[f] = c["name"]
    if not forms:
        return text
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    before = text
    total_pattern = rf"\btotal number of\s+(?:the\s+)?({alt})(?![a-z0-9])"
    text = re.sub(total_pattern, r"total \1", text)
    trace.resub(before, text, total_pattern, r"total \1", "copilot.total_number_of")
    m = re.search(rf"\b(how many|(?:the\s+)?number of)\s+(?:the\s+)?({alt})(?![a-z0-9])", text)
    if not m or re.search(r"\b(total|sum|average|mean|avg|median|maximum|minimum|max|min)\b", text[:m.start()]):
        return text
    rest = text[:m.start()] + " " + text[m.end():]
    in_total = r"\b(in total|total|altogether|combined|summed)\b"
    if re.search(in_total, rest):
        new = text[:m.start()] + f"total {m.group(2)}" + re.sub(in_total, " ", text[m.end():])
        if trace:
            # the words that said "in total" decide the reading; they are part of the measure's evidence
            for t in re.finditer(in_total, text[m.end():]):
                trace.note("pending_measure", text, m.end() + t.start(), m.end() + t.end(), source="aggregation_word",
                           rule="copilot.count_of_measure_in_total", role="aggregation")
            mid = text[:m.start()] + f"total {m.group(2)}" + text[m.end():]
            trace.edit(text, m.start(), m.end(), f"total {m.group(2)}", "copilot.count_of_measure_in_total")
            trace.resub(mid, new, in_total, " ", "copilot.count_of_measure_in_total",
                        start=m.start() + len(f"total {m.group(2)}"))
        return new
    col = forms[m.group(2)]
    reason = f"'{m.group(0).strip()}' could mean the total of {col} or the number of rows. Which do you mean?"
    trace.ambiguity(slot="measure", reason=reason, text=text, start=m.start(), end=m.end(),
                    candidates=[{"aggregation": "sum", "column": col}, {"aggregation": "count", "column": None}])
    return {"clarification": {
        "reason": reason,
        "options": [{"label": f"Total {col}", "message": f"total {col}"},
                    {"label": "Number of rows", "message": "number of rows"}]}}


def _read_several_metrics(text: str, profile: dict, kw: dict, trace=interpretation.NULL_TRACE) -> list[dict] | None:
    """"average, minimum and maximum age", "total and minimum enrollment",
    "maximum price and minimum weight": several aggregations, each read by the
    fast path. A segment that names no column takes the column of the next one.
    None unless every segment is read and at least two distinct metrics result."""
    aliases = kw.get("aliases") or {}
    inside = [cm.span() for _, cm in _column_matches(text, profile, aliases, ("numeric", "categorical", "datetime"))]
    starts = [m for m in re.finditer(rf"(?<![a-z0-9])(?:{_AGG_TOKEN})(?![a-z0-9])", text)
              if not any(s <= m.start() < e for s, e in inside)]   # "max" in a column called max_speed
    if len(starts) < 2:
        return None
    prefix = text[:starts[0].start()].split()
    if any(w not in fastpath._STOP and w not in _FILLER for w in prefix):
        return None
    segments, bounds = [], []
    for i, m in enumerate(starts):
        seg = text[m.start(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        segments.append(re.sub(r"(?:,|\band\b|\bas well as\b|\bthe\b|\s)+$", " ", seg).strip())
        # where the segment the fast path reads stands in `text` (it starts at the aggregation word)
        exact = text[m.start(): m.start() + len(segments[-1])] == segments[-1]
        bounds.append((m.start(), m.start() + (len(segments[-1]) if exact else len(seg))))
    parsed: list[dict | None] = [None] * len(segments)
    children: list = [None] * len(segments)   # each segment's reading, kept only if the whole list is read
    shared = None
    shared_spans: list = []
    for i in range(len(segments) - 1, -1, -1):
        child = trace.child([("slice", text, *bounds[i])])
        p = fastpath.parse(segments[i], profile, **kw, trace=child)
        if (p is None or p["kind"] != "spec") and shared is not None:
            child = trace.child([("slice", text, *bounds[i]), ("synth", " ", [], "copilot.several_metrics"),
                                 ("synth", shared, shared_spans, "copilot.several_metrics.shared_column")])
            p = fastpath.parse(f"{segments[i]} {shared}", profile, **kw, trace=child)
        if p is None or p["kind"] != "spec" or p["group_by"] or p["filters"]:
            return None
        parsed[i] = p
        children[i] = child
        cols = me.parse_formula(p["formula"]).columns
        if len(cols) == 1 and child is not None:
            shared_spans = [s for e in child.evidence_of("measure") if e.get("role") == "column" for s in e["spans"]]
        shared = _phrase(cols[0]) if len(cols) == 1 else shared
    metrics = [{"formula": p["formula"], "label": p["label"]} for p in parsed]
    if len({m["formula"] for m in metrics}) != len(metrics):
        return None
    trace.stash("several", children)
    return metrics


def _numeric_owner(profile: dict, stems: tuple[str, ...]) -> str | None:
    """The one numeric column whose name contains one of these words, or None."""
    owners = [c["name"] for c in profile.get("columns", [])
              if c.get("kind") == "numeric" and set(_phrase(c["name"]).split()) & set(stems)]
    return owners[0] if len(owners) == 1 else None


# A superlative adjective names an extreme of one quantity: "the oldest" is the maximum age.
_SUPERLATIVES = {"oldest": ("maximum", ("age",)), "youngest": ("minimum", ("age",)),
                 "heaviest": ("maximum", ("weight",)), "lightest": ("minimum", ("weight",)),
                 "tallest": ("maximum", ("height",)), "shortest": ("minimum", ("height", "length")),
                 "longest": ("maximum", ("length",))}


def _read_superlatives(text: str, profile: dict, aliases: dict, trace=interpretation.NULL_TRACE) -> str:
    """"the age of the oldest dog", "the oldest age", "the ages of the oldest and
    youngest students" -> "maximum age" / "maximum and minimum age". Only when
    the quantity the adjective implies is itself named and is exactly one
    numeric column: "the name of the oldest dog" asks for a row, not a maximum,
    and is left alone."""
    sup = "|".join(_SUPERLATIVES)
    for c in profile.get("columns", []):
        if c.get("kind") != "numeric":
            continue
        name = c["name"]
        forms = sorted({f for p in [_phrase(name), *aliases.get(name, [])] for f in _number_forms(p) | {p}},
                       key=len, reverse=True)
        col = "|".join(re.escape(f) for f in forms)
        pair = rf"({sup})(?:\s+and\s+(?:the\s+)?({sup}))?"
        for pattern, groups in ((rf"(?<![a-z0-9])(?:the\s+)?(?:{col})\s+of\s+(?:all\s+)?(?:the\s+)?{pair}(?![a-z0-9])", (1, 2)),
                                (rf"(?<![a-z0-9])(?:the\s+)?{pair}\s+(?:{col})(?![a-z0-9])", (1, 2))):
            m = re.search(pattern, text)
            if not m:
                continue
            words = [m.group(g) for g in groups if m.group(g)]
            if any(_numeric_owner(profile, _SUPERLATIVES[w][1]) != name for w in words):
                continue
            aggs = " and ".join(_SUPERLATIVES[w][0] for w in words)
            trace.edit(text, m.start(), m.end(), f" {aggs} {_phrase(name)} ", "copilot.superlative")
            return text[:m.start()] + f" {aggs} {_phrase(name)} " + text[m.end():]
    return text


def _read_years_old(text: str, profile: dict, trace=interpretation.NULL_TRACE) -> str:
    """"18 years old" names the age column when there is exactly one."""
    owner = _numeric_owner(profile, ("age",))
    if owner is None:
        return text
    pattern, repl = rf"(?<![a-z0-9])({_NUM})\s+years?\s+old(?![a-z0-9])", rf"\1 {_phrase(owner)}"
    out = re.sub(pattern, repl, text)
    trace.resub(text, out, pattern, repl, "copilot.years_old")
    return out


def _read_number_equalities(text: str, profile: dict, aliases: dict, measure: dict,
                            trace=interpretation.NULL_TRACE) -> tuple[list[dict], str]:
    """"cars with 8 cylinders", "students who are 18 years old": a number
    written right before a numeric column, after a linking verb, is an
    equality filter. Comparisons ("more than 8 cylinders") are read before this."""
    every = {k: list(v) for k, v in aliases.items()}
    for k, v in measure.items():
        every.setdefault(k, []).extend(v)
    ops: list[dict] = []
    while True:
        cols = sorted(_column_matches(text, profile, every), key=lambda h: -(h[1].end() - h[1].start()))
        found = None
        for m in re.finditer(rf"(?<![a-z0-9])(?:with|having|have|has|had|are|is|were|was|exactly)\s+({_NUM})\s+", text):
            hit = next(((n, cm) for n, cm in cols if cm.start() == m.end()), None)
            if hit:
                found = (hit[0], m.group(1), m.start(), hit[1].end())
                break
        if not found:
            return ops, text
        name, number, start, end = found
        new_op = {"op": "add_filter", "column": name, "operator": "==", "value": float(number)}
        ops.append(new_op)
        rule = "copilot.number_equality"
        trace.note(new_op, text, m.start(), m.start(1), source="comparison_phrase", rule=rule, role="operator")
        trace.note(new_op, text, m.start(1), m.end(1), source="literal", rule=rule, role="value")
        trace.note(new_op, text, hit[1].start(), hit[1].end(), source=_column_source(name, hit[1].group(0)),
                   rule=rule, role="column", ref={"column": name})
        before = text
        text = text[:start] + " " + text[end:]
        trace.edit(before, start, end, " ", rule)


def _read_several_distinct_counts(text: str, profile: dict, kw: dict,
                                  trace=interpretation.NULL_TRACE) -> list[dict] | None:
    """"the number of different series names and contents": one distinct count
    per column. None unless every part is exactly one column."""
    m = re.match(r"^(.*?)(?<![a-z0-9])(?:how many|(?:the\s+)?number of|count of)\s+(?:different|distinct|unique)\s+(.+)$",
                 text.strip())
    if not m or any(w not in fastpath._STOP and w not in _FILLER for w in m.group(1).split()):
        return None
    pieces = [p.strip() for p in re.split(r"(?<![a-z0-9])and(?![a-z0-9])", m.group(2))]
    if len(pieces) < 2 or not all(pieces):
        return None
    children: list = []
    if trace:
        # where each piece and the shared "number of different" stand in `text`
        lead = len(text) - len(text.lstrip())
        count_spans = interpretation.spans_of(trace.evidence(text, lead + m.end(1), lead + m.start(2), source="keyword",
                                                             rule="copilot.several_distinct_counts", role="aggregation"))
        seps = list(re.finditer(r"(?<![a-z0-9])and(?![a-z0-9])", m.group(2)))
        cuts = [0, *(s.end() for s in seps)], [*(s.start() for s in seps), len(m.group(2))]
        piece_bounds = []
        for a, b in zip(*cuts):
            # the cleaned piece is the raw one without surrounding spaces and leading the/different/...
            raw_piece = m.group(2)[a:b]
            stripped = raw_piece.strip()
            cleaned = re.sub(r"^(?:(?:the|different|distinct|unique)\s+)+", "", stripped)
            s = lead + m.start(2) + a + (len(raw_piece) - len(raw_piece.lstrip())) + (len(stripped) - len(cleaned))
            piece_bounds.append((s, s + len(cleaned)) if text[s:s + len(cleaned)] == cleaned
                                else (lead + m.start(2) + a, lead + m.start(2) + b))
    metrics = []
    for k, piece in enumerate(pieces):
        piece = re.sub(r"^(?:(?:the|different|distinct|unique)\s+)+", "", piece)
        child = trace.child([("synth", "number of different ", count_spans, "copilot.several_distinct_counts"),
                             ("slice", text, *piece_bounds[k])]) if trace else None
        p = fastpath.parse(f"number of different {piece}", profile, **kw, trace=child)
        if p is None or p["kind"] != "spec" or p["group_by"] or p["filters"] or not p["formula"].startswith("count_distinct("):
            return None
        metrics.append({"formula": p["formula"], "label": p["label"]})
        children.append(child)
    if len({x["formula"] for x in metrics}) != len(metrics):
        return None
    trace.stash("several", children)
    return metrics


# Adjectives of place -> the place's names as a dataset may write them. Closed
# list; words that name more than one place ("American", "English", "Korean")
# or are common verbs ("polish") are left out.
_DEMONYMS = {
    "french": ("France",), "german": ("Germany",), "italian": ("Italy",), "spanish": ("Spain",),
    "japanese": ("Japan",), "chinese": ("China",), "canadian": ("Canada",), "mexican": ("Mexico",),
    "brazilian": ("Brazil",), "russian": ("Russia",), "indian": ("India",), "dutch": ("Netherlands", "Holland"),
    "swedish": ("Sweden",), "norwegian": ("Norway",), "danish": ("Denmark",), "finnish": ("Finland",),
    "greek": ("Greece",), "turkish": ("Turkey",), "irish": ("Ireland",), "scottish": ("Scotland",),
    "welsh": ("Wales",), "portuguese": ("Portugal",), "argentine": ("Argentina",), "argentinian": ("Argentina",),
    "australian": ("Australia",), "egyptian": ("Egypt",), "swiss": ("Switzerland",), "austrian": ("Austria",),
    "belgian": ("Belgium",), "chilean": ("Chile",), "colombian": ("Colombia",), "nigerian": ("Nigeria",),
    "kenyan": ("Kenya",), "hungarian": ("Hungary",), "ukrainian": ("Ukraine",),
    "british": ("United Kingdom", "UK", "Great Britain"),
    "african": ("Africa",), "european": ("Europe",), "asian": ("Asia",),
}


def _large_value_reading(text: str, column: str, values: dict[str, str], aliases: dict) -> tuple[str, tuple[int, int]] | None | bool:
    """A value of a column too large for the value index, read only where it
    stands right next to the column's name: "the Gelderland district",
    "district Gelderland", "district named Gelderland". Returns (value, span),
    None when nothing is there, or False when two different values could be meant."""
    forms = sorted({f for p in [_phrase(column), *aliases.get(column, [])] for f in _number_forms(p) | {p}},
                   key=len, reverse=True)
    word = r"[a-z0-9][a-z0-9.'&]*"
    found: dict[str, tuple[int, int]] = {}
    for cm in re.finditer(rf"(?<![a-z0-9])(?:{'|'.join(re.escape(f) for f in forms)})(?![a-z0-9])", text):
        for n in (3, 2, 1):
            b = re.search(rf"(?<![a-z0-9])({word}(?:\s+{word}){{{n - 1}}})\s+$", text[:cm.start()])
            if b and b.group(1) in values and not all(w in _NOT_A_VALUE for w in b.group(1).split()):
                found.setdefault(values[b.group(1)], (b.start(1), cm.end()))
                break
        for n in (3, 2, 1):
            a = re.match(rf"\s+(?:named\s+|called\s+)?({word}(?:\s+{word}){{{n - 1}}})(?![a-z0-9])", text[cm.end():])
            if a and a.group(1) in values and not all(w in _NOT_A_VALUE for w in a.group(1).split()):
                found.setdefault(values[a.group(1)], (cm.start(), cm.end() + a.end(1)))
                break
    if not found:
        return None
    if len(found) > 1:
        return False
    return next(iter(found.items()))


def interpret(message: str, state: dict, profile: dict, cat_index: dict, resolved_metrics: list[dict],
              dataset_name: str | None = None, trace: "interpretation.Trace | None" = None) -> dict:
    """Rules only. Returns {"operations": [...]} or {"clarification": {...}} or
    {"unmapped": True}. `dataset_name` is the uploaded file's name: it tells the
    rules what one row is ("singers.csv"), so "how many singers" counts rows.

    `trace` (an `interpretation.Trace` of `message`) is told which words each
    rule reads and how the working text is rewritten, for the interpretation
    record. It only observes: the result is the same with or without it."""
    t = trace if trace is not None else interpretation.NULL_TRACE
    if trace is not None:
        trace.column_source, trace.dataset_name = _column_source, dataset_name
    raw = message.lower()
    year_ops, raw = _read_bare_years(raw, profile, t)
    if isinstance(year_ops, dict):
        return year_ops
    time_range = _parse_time_range(raw)
    time_ev = None
    if time_range is not None:
        start, end = time_range[2]
        time_ev = t.evidence(raw, start, end, source="literal", rule="copilot.time_range", role="time_range")
        t.edit(raw, start, end, " ", "copilot.time_range")
        raw = raw[:start] + " " + raw[end:]
    text = _phrase(re.sub(r"[^\w\s%.<>=!\-:/]", " ", raw))
    t.sync(text, "normalize")
    ops: list[dict] = list(year_ops or [])
    columns = {c["name"]: _phrase(c["name"]) for c in profile.get("columns", [])}
    kinds = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    aliases, measure = column_aliases(profile)
    entities = entity_phrases(dataset_name)

    state_m = re.search(r"\b(reset|start over|start again|clear everything)\b", text)
    if state_m:
        op = {"op": "reset"}
        t.note(op, text, *state_m.span(), source="keyword", rule="copilot.reset", role="state")
        return {"operations": [op]}
    state_m = re.search(r"\b(use|switch to|move to|update to|rebase)\b.*\b(latest|newest|new|current)\b.*\b(data|version|dataset)\b", text)
    if state_m or text in ("rebase", "use latest data"):
        op = {"op": "rebase"}
        t.note(op, text, *(state_m.span() if state_m else (0, len(text))), source="keyword", rule="copilot.rebase",
               role="state")
        return {"operations": [op]}
    state_m = re.search(r"\b(clear|remove|drop)\s+(all\s+)?(the\s+)?filters?\b", text)
    if state_m:
        named = [name for name, phrase in columns.items() if re.search(rf"\b{re.escape(phrase)}\b", text)]
        if named:
            removed = [{"op": "remove_filter", "column": n} for n in named]
            for op in removed:
                t.note(op, text, *state_m.span(), source="keyword", rule="copilot.remove_filter", role="state")
                if t:
                    cm = re.search(rf"\b{re.escape(columns[op['column']])}\b", text)
                    t.note(op, text, *cm.span(), source="column_name", rule="copilot.remove_filter", role="column",
                           ref={"column": op["column"]})
            return {"operations": removed}
        op = {"op": "clear_filters"}
        t.note(op, text, *state_m.span(), source="keyword", rule="copilot.clear_filters", role="state")
        return {"operations": [op]}
    state_m = re.search(r"\b(overall|in total|drill up|remove (the )?(breakdown|grouping)|no breakdown|ungroup)\b", text)
    if state_m:
        op = {"op": "drill_up"}
        t.note(op, text, *state_m.span(), source="keyword", rule="copilot.drill_up", role="state")
        ops.append(op)

    if time_range is not None:
        time_cols = _time_columns(profile)
        if not time_cols:
            reason = "This dataset has no date column, so a time range cannot be applied."
            t.ambiguity(slot="filter", reason=reason, candidates=[], spans=interpretation.spans_of(time_ev))
            return {"clarification": {"reason": reason,
                                      "options": []}}
        named_time = None
        if len(time_cols) > 1:
            named = [c for c in time_cols if re.search(rf"\b{re.escape(_phrase(c))}\b", text)]
            if len(named) != 1:
                reason = "This dataset has more than one date column. Which one should the time range use?"
                t.ambiguity(slot="filter", reason=reason, spans=interpretation.spans_of(time_ev),
                            candidates=[{"column": c, "start": time_range[0], "end": time_range[1]} for c in time_cols])
                return {"clarification": {
                    "reason": reason,
                    "options": [{"label": c, "message": f"{message} (by {c})"} for c in time_cols]}}
            time_cols = named
            named_time = re.search(rf"\b{re.escape(_phrase(named[0]))}\b", text)
        op = {"op": "set_time_range", "column": time_cols[0], "start": time_range[0], "end": time_range[1]}
        t.add(op, time_ev)
        if named_time is not None:
            t.note(op, text, *named_time.span(), source="column_name", rule="copilot.time_range", role="column",
                   ref={"column": time_cols[0]})
        else:
            t.add(op, interpretation.restate(time_ev, source="column_inference", role="column",
                                             rule="copilot.time_range.only_date_column", ref={"column": time_cols[0]}))
        ops.append(op)

    # "the age of the oldest dog" -> "maximum age"; "count the number of different X" says "number of" once
    text = _read_superlatives(text, profile, aliases, t)
    before = text
    pattern = r"\bcount\s+(?:the\s+)?(?:total\s+)?number of\s+(?=(?:different|distinct|unique)\b)"
    text = re.sub(pattern, "number of ", text)
    t.resub(before, text, pattern, "number of ", "copilot.count_number_of_different")
    text = _read_years_old(text, profile, t)

    # comparisons in words: "weight heavier than 10", "at least 180", "more than 20 shops"
    cmp_ops, text = _read_comparisons(text, profile, aliases, measure, t)
    # a number written before a column: "with 8 cylinders"
    eq_ops, text = _read_number_equalities(text, profile, aliases, measure, t)
    cmp_ops += eq_ops
    ops += cmp_ops

    # explicit comparison: "where price > 10", "quantity >= 5"
    for name, phrase in sorted(columns.items(), key=lambda kv: -len(kv[1])):
        m = re.search(rf"\b{re.escape(phrase)}\s*(>=|<=|!=|==|=|>|<|is not|is|equals)\s*([\w.\-]+)((?: [\w.\-]+){{0,2}})", text)
        if m:
            op = {"=": "==", "is": "==", "equals": "==", "is not": "!="}.get(m.group(1), m.group(1))
            # The value is one word, unless a longer run of words is a real
            # value of this column ("region = new south wales").
            words = [m.group(2), *m.group(3).split()]
            value, used, known = words[0], 1, False
            for n in range(len(words), 0, -1):
                hit = [v for c, v in cat_index.get(" ".join(words[:n]), []) if c == name]
                if hit:
                    value, used, known = hit[0], n, True
                    break
            # "weight is heavier": a word after "is" becomes a filter only if it
            # is a real value of the column (or a number, for a numeric one).
            if m.group(1) in ("is", "is not", "equals") and not known and not (
                    kinds.get(name) == "numeric" and re.fullmatch(_NUM, value)):
                continue
            end = m.start(2) + len(" ".join(words[:used]))
            new_op = {"op": "add_filter", "column": name, "operator": op, "value": value}
            ops.append(new_op)
            rule = "copilot.explicit_comparison"
            t.note(new_op, text, m.start(), m.start() + len(phrase), source="column_name", rule=rule, role="column",
                   ref={"column": name})
            t.note(new_op, text, *m.span(1), rule=rule, role="operator",
                   source="comparison_phrase" if m.group(1)[0].isalpha() else "explicit_operator")
            t.note(new_op, text, m.start(2), end, rule=rule, role="value",
                   source="category_value" if known else "literal",
                   ref={"column": name, "value": value} if known else None)
            t.edit(text, m.start(), end, " ", rule)
            text = text[:m.start()] + " " + text[end:]

    # a category value named on its own: "only west", "for enterprise", "excluding returns"
    negation = re.search(r"\b(exclud\w*|without|except|not)\b", text)
    negate = bool(negation)
    negation_ev = t.evidence(text, *negation.span(), source="keyword", rule="copilot.negated_value",
                             role="operator") if negation else None
    by_value: dict[str, dict] = {}   # filters read from category values in this message, by column
    marked = {_phrase(q) for q in re.findall(r"[\"“‘']([^\"“”‘’']{1,60})[\"”’']", message)} | \
             {_phrase(t) for t in re.findall(r"\b[A-Z]{2,}\b", message)}

    def add_value(column: str, original: str, *evidence) -> None:
        if column in by_value:
            # "Tokyo or Beijing": a second value of the same column widens the filter to a list.
            op = by_value[column]
            if op["operator"] in ("==", "!="):
                op["operator"], op["value"] = ("in" if op["operator"] == "==" else "not_in"), [op["value"]]
            if original not in op["value"]:
                op["value"].append(original)
        else:
            by_value[column] = {"op": "add_filter", "column": column, "operator": "!=" if negate else "==", "value": original}
            ops.append(by_value[column])
            if negate:
                t.add(by_value[column], negation_ev)
        t.add(by_value[column], *evidence)

    # a place adjective: "French singers" -> Country = France, only if France is a value of one column
    # and "French" is not itself a value anywhere ("French" may be a language)
    for word, places in _DEMONYMS.items():
        m = re.search(rf"(?<![a-z0-9]){word}(?![a-z0-9])", text)
        if not m:
            continue
        following = re.search(rf"(?i)\b{word}\s+(\S)", message)
        if following and following.group(1).isupper():
            continue       # part of a proper name: "Australian Grand Prix"
        if any(word in vp.split() and len(vp.split()) > 1 and re.search(rf"(?<![a-z0-9]){re.escape(vp)}(?![a-z0-9])", text)
               for vp in cat_index):
            continue       # inside a longer value the value loop reads ("French Guiana")
        owners = sorted({(c, v) for place in places for c, v in cat_index.get(_phrase(place), [])})
        if not owners:
            continue
        literal = cat_index.get(word, [])
        if literal or len({c for c, _ in owners}) > 1:
            options = [*owners, *literal]
            reason = f"'{word}' could refer to more than one value. Which do you mean?"
            t.ambiguity(slot="value", reason=reason, text=text, start=m.start(), end=m.end(),
                        candidates=[{"column": c, "operator": "==", "value": v} for c, v in options])
            return {"clarification": {
                "reason": reason,
                "options": [{"label": f"{c} = {v}", "message": f"{c} = {v}"} for c, v in options]}}
        add_value(*owners[0], t.evidence(text, m.start(), m.end(), source="place_adjective", rule="copilot.place_adjective",
                                         role="value", ref={"column": owners[0][0], "value": owners[0][1]}))
        t.edit(text, m.start(), m.end(), " ", "copilot.place_adjective")
        text = text[:m.start()] + " " + text[m.end():]

    entity_words = set(entities)
    column_words = {f for p in columns.values() for w in p.split() for f in _number_forms(w) | {w}}
    for value_phrase in sorted(cat_index, key=len, reverse=True):
        if len(value_phrase) < 2:
            continue
        # "republics" for the value Republic; never a plural that names the rows or a column ("students")
        plural = "(?:s|es)?" if len(value_phrase) > 2 and value_phrase[-1].isalpha() and value_phrase[-1] != "s" else ""
        pattern = rf"(?<![a-z0-9]){re.escape(value_phrase)}{plural}(?![a-z0-9])"
        vm = re.search(pattern, text)
        if vm is None:
            continue
        if vm.group(0) != value_phrase:
            if vm.group(0) in entity_words or vm.group(0).split()[-1] in column_words or vm.group(0) in cat_index:
                pattern = rf"(?<![a-z0-9]){re.escape(value_phrase)}(?![a-z0-9])"
                if not re.search(pattern, text):
                    continue
        if all(w in _NOT_A_VALUE for w in value_phrase.split()) and value_phrase not in marked:
            continue  # a question word that happens to be a value ("total", "number") is not a filter,
            #           unless the user quoted it or wrote it in capitals ("division 'AS'")
        owners = cat_index[value_phrase]
        already = {o.get("column") for o in ops if o["op"] == "add_filter" and o.get("column") not in by_value}
        owners = [o for o in owners if o[0] not in already]
        if not owners:
            continue
        if len({c for c, _ in owners}) > 1:
            reason = f"'{owners[0][1]}' is a value in more than one column. Which do you mean?"
            t.ambiguity(slot="value", reason=reason, text=text, start=vm.start(), end=vm.end(),
                        candidates=[{"column": c, "operator": "==", "value": v} for c, v in owners])
            return {"clarification": {
                "reason": reason,
                "options": [{"label": f"{c} = {v}", "message": f"{c} = {v}"} for c, v in owners]}}
        read = re.search(pattern, text) if t else None   # the words the substitution below removes
        add_value(*owners[0], t.evidence(text, *read.span(), source="category_value", rule="copilot.category_value",
                                         role="value", ref={"column": owners[0][0], "value": owners[0][1]})
                  if read else None)
        before = text
        text = re.sub(pattern, " ", text, count=1)
        t.resub(before, text, pattern, " ", "copilot.category_value", count=1)

    # a value of a column too large to index, read only next to the column's name: "the Gelderland district"
    for column, values in getattr(cat_index, "large", {}).items():
        if any(o.get("column") == column for o in ops if o["op"] == "add_filter"):
            continue
        reading = _large_value_reading(text, column, values, aliases)
        if reading is False:
            t.set_reason(f"two different values of {column} could be meant")
            return {"unmapped": True}
        if reading:
            original, (start, end) = reading
            add_value(column, original, t.evidence(text, start, end, source="category_value", role="value",
                                                   rule="copilot.large_column_value", ref={"column": column, "value": original}))
            t.edit(text, start, end, " ", "copilot.large_column_value")
            text = text[:start] + " " + text[end:]
    text = _consume_filter_column_words(text, list(by_value.values()) + cmp_ops, profile, aliases, t)

    # grouping: "by region", "break down by product", "drill into channel"
    m = re.search(r"\b(?:by|per|for each|break(?:ing)? (?:it |this |that )?down by|split by|drill (?:down )?(?:into|by|on))\s+(?:the\s+)?", text)
    if m:
        rest = text[m.end():]
        for name, phrase in sorted(columns.items(), key=lambda kv: -len(kv[1])):
            gm = re.match(rf"{re.escape(phrase)}s?(?![a-z0-9])", rest)
            if gm:
                op = {"op": "drill_down", "column": name}
                ops.append(op)
                t.note(op, text, *m.span(), source="keyword", rule="copilot.group_by", role="group_keyword")
                t.note(op, text, m.end(), m.end() + gm.end(), source="column_name", rule="copilot.group_by",
                       role="column", ref={"column": name})
                t.edit(text, m.start(), m.end() + len(phrase), " ", "copilot.group_by")
                text = text[:m.start()] + " " + rest[len(phrase):]
                break

    # a metric: an approved definition, or "<aggregation> <column>", or several of them
    kw = {"aliases": aliases, "measure_aliases": measure, "extra_filler": _FILLER, "agg_outside_columns": True}
    before = text
    pattern = r"\b(average|mean|avg|median|minimum|min|maximum|max|highest|lowest|largest|smallest)\s+(?:the\s+)?number of\b"
    text = re.sub(pattern, r"\1", text)   # "average number of floors" is an average, not a count
    t.resub(before, text, pattern, r"\1", "copilot.average_number_of")
    if any(o["operator"] in ("!=", "not_in") for o in by_value.values()):
        before = text
        text = re.sub(r"\b(exclud\w*|without|except|not)\b", " ", text)   # the negation became the filter
        t.resub(before, text, r"\b(exclud\w*|without|except|not)\b", " ", "copilot.negated_value")
    counted = _read_row_count(text, entities, profile, t)
    counts_entity = counted != text
    text = _drop_entity_filler(counted, entities, profile, aliases, t)
    text = _read_count_of_measure(text, profile, aliases, measure, t)
    if isinstance(text, dict):
        return text
    several = (_read_several_metrics(text, profile, kw, t) or _read_several_distinct_counts(text, profile, kw, t)) \
        if text.strip() else None
    if several:
        op = {"op": "set_metric", "formula": several[0]["formula"], "label": several[0]["label"],
              "filters": [], "metrics": several}
        ops.insert(0, op)
        for i, child in enumerate(t.take("several") or []):
            t.adopt(child, {"measure": ("measure", id(op), i)})
        t.attach_pending(op, "pending_measure")
        return {"operations": ops}
    child = t.child([("slice", text, 0, len(text))]) if text.strip() else None
    parsed = fastpath.parse(text, profile, resolved_metrics, **kw, trace=child) if text.strip() else None
    if counts_entity and parsed is not None and (parsed["kind"] != "spec" or parsed["formula"] != "count()"):
        # "count the customers who are active": the question counts rows; a column it
        # also names is a condition that was not read, so nothing is answered.
        t.adopt(child, {})
        t.set_reason("the question counts rows and also names a column that was not read as a condition")
        return {"unmapped": True}
    if parsed is not None and parsed["kind"] == "ambiguous":
        t.adopt(child, {})
        return {"clarification": {"reason": parsed["reason"] + " Which one do you mean?",
                                  "options": [{"label": o["label"], "message": o["question"]} for o in parsed["options"]]}}
    if parsed is not None:
        op = {"op": "set_metric", "formula": parsed["formula"], "label": parsed["label"], "filters": parsed["filters"]}
        ops.insert(0, op)
        mapping = {"measure": ("measure", id(op), 0)}
        for g in parsed["group_by"]:
            if not any(o["op"] == "drill_down" and o["column"] == g for o in ops):
                group_op = {"op": "drill_down", "column": g}
                ops.append(group_op)
                mapping[("group", g)] = group_op
        t.adopt(child, mapping)
        t.attach_pending(op, "pending_measure")
    else:
        t.adopt(child, {})
    if not ops:
        return {"unmapped": True}
    return {"operations": ops}


def interpret_with_model(message: str, state: dict, profile: dict) -> dict:
    """Ask a model to map the message onto operations. Its answer is data: it is
    validated by `apply_operations` like any other input."""
    from app.agents import llm_client, prompts

    cols = [f"{c['name']} ({c.get('kind')})" for c in profile.get("columns", [])][:120]
    user = (f"Columns: {', '.join(cols)}\nCurrent state: metric={state.get('formula')}, "
            f"filters={state.get('filters')}, group_by={state.get('group_by')}\n"
            f"Operations: {', '.join(OPERATIONS)}\nUser message: {message!r}")
    try:
        out = llm_client.call_tool(system=ACTIONS_SYSTEM + prompts.UNTRUSTED_DATA_NOTICE, user_content=user,
                                   tool_name="submit_operations", tool_schema=ACTIONS_TOOL_SCHEMA,
                                   tool_description="Submit the operations for this message.", max_tokens=600)
    except Exception:  # noqa: BLE001 - no model, no quota: fall back to "not understood"
        logger.info("Copilot model interpretation unavailable", exc_info=True)
        return {"unmapped": True}
    if out.get("cannot_map") or not out.get("operations"):
        return {"unmapped": True}
    ops = []
    for raw in list(out["operations"])[:5]:
        if not isinstance(raw, dict) or raw.get("op") not in OPERATIONS:
            return {"unmapped": True}
        op = {"op": raw["op"]}
        for key in ("formula", "column", "operator", "value", "start", "end"):
            if raw.get(key) not in (None, ""):
                op[key] = raw[key]
        ops.append(op)
    return {"operations": ops, "interpreted_by": "model"}


# -------------------------------------------------------------------- state --

def apply_operations(state: dict, operations: list[dict], profile: dict) -> tuple[dict, list[str]]:
    """Apply validated operations. Returns (new state, human-readable decisions).
    Raises CopilotError on anything that is not a valid operation on this dataset."""
    columns = {c["name"]: c.get("kind") for c in profile.get("columns", [])}
    new = {"formula": state.get("formula"), "metric_label": state.get("metric_label"),
           "filters": [dict(f) for f in state.get("filters") or []], "group_by": list(state.get("group_by") or []),
           "history_depth": int(state.get("history_depth") or 0) + 1}
    if state.get("extra_metrics"):
        new["extra_metrics"] = [dict(m) for m in state["extra_metrics"]]
    decisions: list[str] = []

    def need_column(name) -> str:
        if not isinstance(name, str) or name not in columns:
            raise CopilotError(f"'{name}' is not a column of this dataset")
        return name

    for op in operations:
        kind = op.get("op")
        if kind == "reset":
            new = {**empty_state(), "history_depth": new["history_depth"]}
            decisions.append("Cleared the metric, filters and grouping.")
        elif kind == "set_metric":
            try:
                parsed = me.parse_formula(str(op.get("formula", "")))
            except me.FormulaError as e:
                raise CopilotError(str(e)) from e
            problems = me.validate_against_profile(parsed, profile)
            if problems:
                raise CopilotError("; ".join(problems))
            new["formula"], new["metric_label"] = parsed.text, str(op.get("label") or parsed.text)[:120]
            for f in op.get("filters") or []:
                new["filters"] = [x for x in new["filters"] if x["column"] != f["column"]] + [dict(f)]
            new.pop("extra_metrics", None)
            extra = []
            for m in (op.get("metrics") or [])[1:6]:   # several numbers asked for at once
                try:
                    more = me.parse_formula(str(m.get("formula", "")))
                except me.FormulaError as e:
                    raise CopilotError(str(e)) from e
                problems = me.validate_against_profile(more, profile)
                if problems:
                    raise CopilotError("; ".join(problems))
                extra.append({"formula": more.text, "label": str(m.get("label") or more.text)[:120]})
            if extra:
                new["extra_metrics"] = extra
                decisions.append("Metrics set to " + ", ".join(
                    f"{m['label']} ({m['formula']})" for m in [{"label": new["metric_label"], "formula": parsed.text}, *extra]) + ".")
            else:
                decisions.append(f"Metric set to {new['metric_label']} ({parsed.text}).")
        elif kind == "add_filter":
            column = need_column(op.get("column"))
            operator = op.get("operator", "==")
            value = op.get("value")
            try:
                clean = me.normalize_filters([{"column": column, "op": operator, "value": value}], list(columns))[0]
            except me.FormulaError as e:
                raise CopilotError(str(e)) from e
            new["filters"] = [f for f in new["filters"] if not (f["column"] == column and not f.get("time_range"))]
            new["filters"].append(clean)
            decisions.append(f"Filter added: {column} {clean['op']} {clean['value']}.")
        elif kind == "remove_filter":
            column = need_column(op.get("column"))
            before = len(new["filters"])
            new["filters"] = [f for f in new["filters"] if f["column"] != column]
            decisions.append(f"Filter on {column} removed." if len(new["filters"]) < before
                             else f"There was no filter on {column}.")
        elif kind == "clear_filters":
            new["filters"] = []
            decisions.append("All filters removed.")
        elif kind == "set_time_range":
            column = need_column(op.get("column"))
            if columns[column] != "datetime":
                raise CopilotError(f"'{column}' is not a date column")
            start, end = op.get("start"), op.get("end")
            for bound in (start, end):
                if bound is not None and pd.isna(pd.to_datetime(bound, errors="coerce")):
                    raise CopilotError(f"'{bound}' is not a date")
            if start is None and end is None:
                raise CopilotError("A time range needs a start or an end")
            new["filters"] = [f for f in new["filters"] if not f.get("time_range")]
            if start is not None:
                new["filters"].append({"column": column, "op": ">=", "value": start, "time_range": True})
            if end is not None:
                new["filters"].append({"column": column, "op": "<=", "value": f"{end} 23:59:59", "time_range": True})
            decisions.append(f"Time range on {column}: {start or 'the beginning'} to {end or 'the end'}.")
        elif kind == "drill_down":
            column = need_column(op.get("column"))
            if column not in new["group_by"]:
                if len(new["group_by"]) >= 2:
                    new["group_by"] = new["group_by"][1:]
                new["group_by"].append(column)
            decisions.append(f"Broken down by {', '.join(new['group_by'])}.")
        elif kind == "drill_up":
            if new["group_by"]:
                new["group_by"] = new["group_by"][:-1]
                decisions.append("Removed one level of breakdown." if new["group_by"] else "Showing the overall figure.")
            else:
                decisions.append("There is no breakdown to remove.")
        elif kind == "rebase":
            continue  # handled by the caller: it changes the session's dataset version
        else:
            raise CopilotError(f"Unsupported operation '{kind}'")
    if new.get("extra_metrics") and new["group_by"]:
        # A breakdown shows one metric; the others are set aside, and the user is told.
        decisions.append(f"A breakdown shows one metric, so it shows {new['metric_label']}; "
                         + ", ".join(m["label"] for m in new["extra_metrics"]) + " set aside.")
        new.pop("extra_metrics")
    return new, decisions


def prune_state(state: dict, profile: dict) -> tuple[dict, list[str]]:
    """After a dataset change: drop whatever no longer exists, and say so."""
    columns = {c["name"] for c in profile.get("columns", [])}
    notes = []
    new = dict(state)
    kept_filters = [f for f in state.get("filters") or [] if f["column"] in columns]
    if len(kept_filters) != len(state.get("filters") or []):
        notes.append("Removed filters on columns that the new data no longer has.")
    new["filters"] = kept_filters
    kept_groups = [g for g in state.get("group_by") or [] if g in columns]
    if len(kept_groups) != len(state.get("group_by") or []):
        notes.append("Removed a breakdown by a column that the new data no longer has.")
    new["group_by"] = kept_groups
    if state.get("formula"):
        try:
            if me.validate_against_profile(me.parse_formula(state["formula"]), profile):
                raise me.FormulaError("columns missing")
        except me.FormulaError:
            new["formula"], new["metric_label"] = None, None
            notes.append("The metric used a column the new data no longer has, so it was cleared.")
    if state.get("extra_metrics"):
        kept = [m for m in state["extra_metrics"] if new.get("formula")
                and not me.validate_against_profile(me.parse_formula(m["formula"]), profile)]
        if len(kept) != len(state["extra_metrics"]):
            notes.append("Metrics that used columns the new data no longer has were cleared.")
        if kept:
            new["extra_metrics"] = kept
        else:
            new.pop("extra_metrics", None)
    return new, notes


# ------------------------------------------------------------------ compute --

def _engine_filters(state: dict) -> list[dict]:
    return [{"column": f["column"], "op": f["op"], "value": f["value"]} for f in state.get("filters") or []]


def compute(csv_path: str, state: dict) -> dict:
    """The current state, evaluated on the dataset."""
    df = me.load_frame(csv_path)
    filters, group_by = _engine_filters(state), list(state.get("group_by") or [])
    engine = me.evaluate(df, state["formula"], filters, group_by or None)
    try:
        ref = me.reference_evaluate(csv_path, state["formula"], filters, group_by or None)
        agrees = None
        if not group_by:
            agrees = me.values_agree(engine["value"], ref["value"], rel_tol=1e-9, abs_tol=1e-9)
        elif len(group_by) == 1:
            ref_map = {str(g["key"][group_by[0]]): g["value"] for g in ref["groups"]}
            eng_map = {str(g["key"][group_by[0]]): g["value"] for g in engine["groups"]}
            if set(ref_map) == set(eng_map):
                agrees = all(me.values_agree(eng_map[k], ref_map[k], rel_tol=1e-9, abs_tol=1e-9) for k in eng_map)
        reference_check = {True: "agrees", False: "disagrees", None: "not_comparable"}[agrees]
    except (me.ReferenceNotApplicable, me.FormulaError, OSError, UnicodeDecodeError):
        reference_check = "not_run"
    label = state.get("metric_label") or state["formula"]
    result: dict = {"rows": engine["rows"], "label": label}
    if not group_by and state.get("extra_metrics"):
        # Several numbers asked for at once: each from the engine, each recomputed independently.
        values = [{"label": label, "formula": state["formula"], "value": engine["value"]}]
        checks = [reference_check]
        for m in state["extra_metrics"]:
            value = me.evaluate(df, m["formula"], filters)["value"]
            try:
                ref = me.reference_evaluate(csv_path, m["formula"], filters)
                checks.append("agrees" if me.values_agree(value, ref["value"], rel_tol=1e-9, abs_tol=1e-9) else "disagrees")
            except (me.ReferenceNotApplicable, me.FormulaError, OSError, UnicodeDecodeError):
                checks.append("not_run")
            values.append({"label": m["label"], "formula": m["formula"], "value": value})
        for v in values:
            v["display"] = fastpath.format_value(v["value"])
        reference_check = ("disagrees" if "disagrees" in checks else "not_run" if "not_run" in checks
                           else "agrees" if all(c == "agrees" for c in checks) else "not_comparable")
        result.update({"type": "scalars", "values": values, "value": engine["value"],
                       "display": "; ".join(f"{v['label']}: {v['display']}" for v in values)})
    elif not group_by:
        result.update({"type": "scalar", "value": engine["value"], "display": fastpath.format_value(engine["value"])})
    else:
        groups = [g for g in engine["groups"] if g["value"] is not None]
        groups.sort(key=lambda g: -g["value"])
        shown = groups[:RESULT_LIMIT]
        result.update({
            "type": "table", "group_by": group_by, "total_groups": len(groups),
            "rows_table": [{**{k: v for k, v in g["key"].items()}, "value": g["value"], "rows": g["rows"]} for g in shown],
            "chart": {"data": [{"type": "bar",
                                "x": [" / ".join(str(g["key"][c]) for c in group_by) for g in shown],
                                "y": [g["value"] for g in shown]}],
                      "layout": {"title": {"text": f"{label} by {', '.join(group_by)}"}}},
        })
    result["reference_check"] = reference_check
    result["coerced_non_numeric"] = engine.get("coerced_non_numeric") or {}
    return result


def describe(state: dict, result: dict) -> str:
    scope = ""
    filters = state.get("filters") or []
    if filters:
        scope = " where " + " and ".join(f"{f['column']} {f['op']} {f['value']}" for f in filters)
    if result["type"] == "scalar":
        return f"{result['label']}{scope}: {result['display']} (from {result['rows']:,} rows)."
    if result["type"] == "scalars":
        return f"{result['display']}{scope} (from {result['rows']:,} rows)."
    top = result["rows_table"][:3]
    lead = "; ".join(f"{' / '.join(str(r[c]) for c in result['group_by'])}: {fastpath.format_value(r['value'])}" for r in top)
    more = f" Showing the top {len(result['rows_table'])} of {result['total_groups']}." \
        if result["total_groups"] > len(result["rows_table"]) else ""
    return f"{result['label']} by {', '.join(result['group_by'])}{scope}. Highest: {lead}.{more}"


# ------------------------------------------------------------------ session --

def get_session(db: Session, team_id: int, user_id: int, session_id: int) -> CopilotSession:
    s = db.get(CopilotSession, session_id)
    # Same answer for "not found", "another team's" and "another user's".
    if s is None or s.team_id != team_id or s.user_id != user_id:
        raise CopilotError("Session not found")
    return s


def create_session(db: Session, *, team_id: int, user_id: int, dataset: Dataset, title: str = "") -> CopilotSession:
    version = latest_version(db, dataset)
    s = CopilotSession(team_id=team_id, user_id=user_id, dataset_id=dataset.id,
                       dataset_version_id=version.id if version else None,
                       title=(title or f"Session on {dataset.filename}")[:120], state_json=empty_state())
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _add_turn(db: Session, session: CopilotSession, role: str, content: str, **extra) -> CopilotTurn:
    seq = (db.query(CopilotTurn).filter_by(session_id=session.id).count()) + 1
    turn = CopilotTurn(session_id=session.id, seq=seq, role=role, content=content,
                       action_json=extra.get("action"), result_json=extra.get("result"),
                       evidence_json=extra.get("evidence"))
    db.add(turn)
    return turn


def turn_out(t: CopilotTurn) -> dict:
    return {"seq": t.seq, "role": t.role, "content": t.content, "action": t.action_json,
            "result": t.result_json, "evidence": t.evidence_json, "created_at": t.created_at}


def session_out(db: Session, s: CopilotSession, with_turns: bool = False) -> dict:
    dataset = db.get(Dataset, s.dataset_id)
    version = db.get(DatasetVersion, s.dataset_version_id) if s.dataset_version_id else None
    latest = latest_version(db, dataset) if dataset else None
    out = {"id": s.id, "title": s.title, "dataset_id": s.dataset_id,
           "dataset_filename": dataset.filename if dataset else None,
           "dataset_version": version.version_number if version else None,
           "latest_version": latest.version_number if latest else None,
           "stale": bool(latest and version and latest.id != version.id),
           "dataset_available": dataset is not None and version is not None,
           "state": s.state_json or empty_state(), "created_at": s.created_at, "updated_at": s.updated_at}
    if with_turns:
        turns = db.query(CopilotTurn).filter_by(session_id=s.id).order_by(CopilotTurn.seq).all()
        out["turns"] = [turn_out(t) for t in turns]
    return out


def _record(make, on_error: object = "record"):
    """Build an additive record of a turn. A failure is logged and stored as
    such; it never changes the turn itself."""
    try:
        return make()
    except Exception:  # noqa: BLE001 - see docstring
        logger.warning("Copilot interpretation record could not be built", exc_info=True)
        if on_error == "record":
            return {"schema": interpretation.SCHEMA, "error": "the interpretation could not be recorded"}
        return on_error


def handle_message(db: Session, session: CopilotSession, message: str, allow_model: bool = True) -> dict:
    """One turn: interpret, apply, recompute, record. Returns the assistant turn."""
    message = (message or "").strip()
    if not message:
        raise CopilotError("Message is empty")
    if len(message) > MAX_MESSAGE_CHARS:
        raise CopilotError(f"Message is longer than {MAX_MESSAGE_CHARS} characters")
    if db.query(CopilotTurn).filter_by(session_id=session.id).count() >= MAX_TURNS:
        raise CopilotError("This session has reached its length limit; start a new one")
    dataset = db.get(Dataset, session.dataset_id)
    version = db.get(DatasetVersion, session.dataset_version_id) if session.dataset_version_id else None
    _add_turn(db, session, "user", message)

    def reply(content: str, **extra) -> dict:
        turn = _add_turn(db, session, "assistant", content, **extra)
        db.commit()
        db.refresh(turn)
        return {**turn_out(turn), "session": session_out(db, session)}

    if dataset is None or version is None:
        return reply("The dataset this session was built on is no longer available, so nothing can be computed. "
                     "Start a new session on a current dataset.", action={"status": "dataset_unavailable"})
    latest = latest_version(db, dataset)
    state = dict(session.state_json or empty_state())
    profile = version.profile_json or {}
    try:
        df = me.load_frame(version.filepath)
    except (OSError, me.FormulaError) as e:
        return reply(f"The data for this session could not be read ({type(e).__name__}).",
                     action={"status": "dataset_unreadable"})
    resolved = semantic.resolve(db, session.team_id, dataset.id, message)
    if resolved["ambiguous"]:
        amb = resolved["ambiguous"][0]
        return reply(f"'{amb['term']}' has more than one approved definition. Which one do you mean?",
                     action={"status": "clarification",
                             "options": [{"label": f"{c['name']} = {c['formula']}", "message": c["formula"]}
                                         for c in amb["candidates"]],
                             "interpretation": _record(lambda: interpretation.semantic_ambiguity(
                                 message, amb, semantic.norm_term(message)))})
    trace = _record(lambda: interpretation.Trace(message), on_error=None)
    cat_index = category_index(df, profile)
    try:
        interpreted = interpret(message, state, profile, cat_index, resolved["metrics"],
                                dataset_name=dataset.filename, trace=trace)
    except Exception:
        if trace is None:
            raise
        # The trace only observes, and `interpret` is pure: read the message again
        # exactly as without a record, so a fault in the record never changes a turn.
        logger.warning("Copilot interpretation trace failed; reading the message without it", exc_info=True)
        trace = None
        interpreted = interpret(message, state, profile, cat_index, resolved["metrics"], dataset_name=dataset.filename)
    if interpreted.get("unmapped") and allow_model:
        interpreted = interpret_with_model(message, state, profile)
    # What was understood and from which words: an additive record, built before
    # anything is applied. It changes no operation, value or reply.
    interp = _record(lambda: interpretation.build(message, interpreted, trace,
                                                  interpreted_by=interpreted.get("interpreted_by", "rules")))
    if "clarification" in interpreted:
        c = interpreted["clarification"]
        return reply(c["reason"], action={"status": "clarification", "options": c["options"], "interpretation": interp})
    if interpreted.get("unmapped"):
        suggestion = message if _NEEDS_ANALYSIS.search(message) else None
        text = ("I could not turn that into a metric, a filter or a breakdown on this dataset, so I have not "
                "changed anything.")
        if suggestion:
            text += " A question like this needs a full analysis; you can run it from the Analyses page."
        else:
            text += " Try, for example: \"total <numeric column> by <column>\", \"only <value>\", \"in 2024\", \"drill up\", \"reset\"."
        return reply(text, action={"status": "not_understood", "suggested_question": suggestion, "interpretation": interp})

    operations = interpreted["operations"]
    notes: list[str] = []
    if any(o["op"] == "rebase" for o in operations):
        if latest is None or latest.id == version.id:
            notes.append("This session is already on the latest data.")
        else:
            version, profile = latest, latest.profile_json or {}
            session.dataset_version_id = latest.id
            state, pruned = prune_state(state, profile)
            notes.append(f"Moved to version {latest.version_number} of the data.")
            notes += pruned
    try:
        new_state, decisions = apply_operations(state, operations, profile)
    except CopilotError as e:
        db.rollback()
        _add_turn(db, session, "user", message)
        return reply(f"I did not change anything: {e}.", action={"status": "invalid_operation", "operations": operations,
                                                                  "interpretation": interp})
    session.state_json = new_state
    stale = latest is not None and latest.id != version.id
    stale_note = (f" Note: a newer version of this dataset exists (version {latest.version_number}); this session "
                  f"is still on version {version.version_number}. Say \"use latest data\" to switch.") if stale else ""
    action = {"status": "applied", "operations": operations, "decisions": notes + decisions,
              "interpreted_by": interpreted.get("interpreted_by", "rules"), "state": new_state,
              "interpretation": interp}
    if not new_state.get("formula"):
        return reply(" ".join(notes + decisions) + " No metric is selected yet, so there is nothing to compute. "
                     "Name one, for example \"total <numeric column>\"." + stale_note, action=action)
    try:
        result = compute(version.filepath, new_state)
    except me.FormulaError as e:
        return reply(" ".join(notes + decisions) + f" That could not be computed: {e}." + stale_note, action=action)
    evidence = {
        "formula": new_state["formula"], "filters": _engine_filters(new_state), "group_by": new_state["group_by"],
        **({"formulas": [new_state["formula"], *(m["formula"] for m in new_state["extra_metrics"])]}
           if new_state.get("extra_metrics") else {}),
        "rows_used": result["rows"], "dataset_version": version.version_number,
        "dataset_fingerprint": provenance.fingerprint(db, version),
        "computed_by": "deterministic metric engine", "reference_check": result["reference_check"],
        "note": "Recomputed from the dataset on this turn; not taken from the conversation.",
    }
    return reply(" ".join(notes + decisions) + " " + describe(new_state, result) + stale_note,
                 action=action, result=result, evidence=evidence)
