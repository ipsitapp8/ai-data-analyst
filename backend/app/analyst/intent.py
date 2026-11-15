"""What a request asks for, read from its language and the schema -- and whether a program delivers it.

A program can be valid (columns exist, types fit) and still answer a different question: an average where a rate
was asked, a total where the request divides by the number of people, the first half of "which ... and why". This
module reads the request's requirements without any model:

  quantity     rate / share, average, median, total, count            ("what proportion", "on average", "how many")
  measures     the columns the quantity is about                       (named, by unit word, derivable, or by concept)
  group        the column results are broken down by                   ("by depot", "each team", "which outlet")
  denominator  what a per-unit figure divides by                       ("per driver", "allowing for how far ...")
  entities     values the request names                                ("North", "Dr Lena Fischer")
  compare / extreme / why / unusual / hypothetical                     the other parts of a composite request

`check(steps, req, df)` returns the requirements a program does not meet (each a reason the model is shown), and
`complete(steps, req, df)` adds the parts that have exactly one deterministic form -- the associations behind a
"why", the unusual values of the measure -- saying so. Nothing here knows a dataset or a question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from app.analyst import semantics
from app.analyst.data import kind_of

_RATE = re.compile(r"\b(rate|rates|share|proportion|percentage|percent of|fraction|how often|how likely|likelihood|"
                   r"chance|what part of)\b", re.I)
_MEDIAN = re.compile(r"\b(median|typical)\b", re.I)
_AVG = re.compile(r"\b(average|averages|mean|avg|on average)\b", re.I)
_TOTAL = re.compile(r"\b(total|totals|in all|altogether|how much|sum of)\b", re.I)
_COUNT = re.compile(r"\b(how many|number of|count of)\s+(\w+)", re.I)
_GROUP_WORDS = r"(?:by|per|each|every|across|between|among|which|what|for each|in each|of the|the)"
_PER = [re.compile(p, re.I) for p in (
    r"\bper\s+(?:each\s+)?([a-z]+(?: on [a-z]+)?)\b",
    r"\bfor each ([a-z]+) (?:it|they) (?:has|have|employs?|uses?|staffs?)\b",
    r"\bout of each ([a-z]+)\b",
    r"\b(?:after|once|when)\s+(?:you\s+)?(?:account(?:ing)?|adjust(?:ing)?|allow(?:ing)?|control(?:ling)?|"
    r"correct(?:ing)?)\s+for\s+([^,.?;]{1,60})",
    r"\ballowing for\s+([^,.?;]{1,60})",
    r"\btaking\s+([^,.?;]{1,40})\s+into account\b",
    r"\b([a-z ]{1,25}) (?:is|are) taken into account\b",
    r"\brelative to (?:its|their|the|each)\s+([^,.?;]{1,40})",
    r"\b(?:relative to|compared with|out of|for) (how many [a-z]+(?: (?:it|they) (?:runs?|has|have|handles?|makes?|does))?)")]
_RECORDS = "(records)"                                   # a denominator that is the number of records
_RESTRICT = re.compile(r"\b(only|among|excluding|except|that were|which were|limited to|just the)\b", re.I)
_PEOPLE = {"person", "people", "driver", "drivers", "staff", "employee", "employees", "worker", "workers", "agent",
           "agents", "member", "members", "head", "heads", "clinician", "clinicians", "picker", "pickers", "rep", "reps",
           "operator", "operators", "courier", "couriers", "individual", "individuals", "team member"}
_COMPARE = re.compile(r"\b(compare|compared|comparison|versus|vs\.?|than|difference between|differ|differs|different|"
                      r"stack up|against|relative to)\b", re.I)
_EXTREME = re.compile(r"\b(which|what|who|where)\b[^?]{0,80}\b(most|highest|largest|biggest|best|top|longest|least|lowest|"
                      r"smallest|fewest|worst|slowest|shortest|bottom|fastest|weakest|strongest|poorest)\b", re.I)
_SINGLE = re.compile(r"\b(single|maximum|minimum|max|min|individual|one (?:order|sale|record|transaction|row)|"
                     r"per (?:order|sale|record|transaction|row)|largest one|biggest one)\b", re.I)
_LOW = re.compile(r"\b(least|lowest|smallest|fewest|bottom|shortest|weakest|poorest)\b", re.I)
_WHY = re.compile(r"\bwhy\b|\b(associated|linked|related|link|goes with|go with|what drives|drivers of|explains?|"
                  r"factors?|because|what lies behind|what is behind|influences?)\b", re.I)
_TALLY_NAMES = {"count", "counts", "n", "number", "qty", "quantity", "tally", "frequency", "freq", "total"}
_QUANTITY_NAMES = {"total", "count", "sum", "average", "mean", "number", "amount", "value", "values", "median"}
_GENERIC = {"total", "num", "no", "number", "count", "of", "per", "pct", "rate", "id", "avg", "mean", "the", "on", "at",
            "in", "amount", "value", "level", "levels", "x", "y"}
_UNUSUAL = re.compile(r"\b(unusual|odd|outliers?|anomal\w*|implausible|out of the ordinary|strange|suspicious|"
                      r"extreme values?|data errors?|look wrong|irregular)\b", re.I)
_HYPO = re.compile(r"\b(?:what if|suppose|imagine|assuming|if)\b[^?]{0,80}\b(?:were|was|had been|closed|shut|stopped|"
                   r"removed|dropped|rose|fell|doubled|halved|went (?:up|down)|increased|decreased|keeps?|continu\w*)\b",
                   re.I)


@dataclass
class Request:
    quantity: str | None = None
    measures: list = field(default_factory=list)
    group: str | None = None
    per: bool = False
    denominators: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    compare: bool = False
    extreme: str | None = None
    why: bool = False
    unusual: bool = False
    hypothetical: bool = False
    why_target: str | None = None
    restrict: list = field(default_factory=list)
    unknown_group: str | None = None
    excluded: list = field(default_factory=list)
    period: dict | None = None
    compared: bool = False              # the grouping is values set against each other ("dinner versus lunch")
    significance: bool = False          # "is that difference meaningful": a statistical question, not a ratio

    def parts(self) -> list[str]:
        out = []
        q = {"rate": "the rate (a share of records)", "average": "the average", "median": "the median",
             "total": "the total", "count": "the count"}.get(self.quantity or "", "the measure")
        what = f"{q} of {' / '.join(map(str, self.measures))}" if self.measures else q
        if self.per:
            what += " per " + (" / ".join(map(str, self.denominators)) if self.denominators else "unit")
        if self.group:
            what += f" by {self.group}"
        out.append(what)
        if self.extreme:
            out.append(f"which {self.group or 'one'} is {'lowest' if self.extreme == 'min' else 'highest'}"
                       if self.extreme != "either" else f"which {self.group or 'one'} stands out")
        if self.why:
            out.append("what is associated with it (evidence, not causes)")
        if self.unusual:
            out.append("unusual values")
        if self.hypothetical:
            out.append("the hypothetical scenario")
        if self.significance:
            out.append("whether the difference is statistically meaningful")
        return out


def _cols_in(text: str, df: pd.DataFrame) -> list:
    from app.analyst.orchestrator import followup_cols
    return list(followup_cols(df, text))


def read(message: str, df: pd.DataFrame, prof: dict | None = None) -> Request:
    from app.analyst.calc import named_values
    from app.analyst.metrics import UNIT_WORDS
    prof = prof or semantics.profile(df)
    roles = prof["roles"]
    low = message.lower()
    r = Request()
    named = _cols_in(message, df)
    _denominator(message, df, roles, r)
    if not r.denominators:
        _conversion(message, df, roles, r)
    # a year column named after a grouping word ("by season", "each year") breaks the result down too
    cats = [c for c in named if roles[c]["role"] in ("category", "flag", "person_name", "year") and c not in r.denominators]
    # the grouping: a category named after a grouping word ("by depot", "each team", "which outlet", "between depots")
    for c in cats:
        w = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()
        from app.analyst.scope import _negated
        if re.search(rf"\b{_GROUP_WORDS}\s+(?:\w+\s+){{0,2}}?{re.escape(w)}s?\b", low) and c not in [
                x for x, v in named_values(df, message) if not _negated(message, v)]:   # "excluding East" still groups
            yes_no = roles[c]["role"] == "flag" and roles[c].get("positive") and not roles[c].get("positive_is_minority")
            if yes_no and not re.search(rf"\b(?:by|between|across|versus|vs)\s+{re.escape(w)}", low):
                continue                                # "among escalated tickets" filters; it does not group
            r.group = c
            break
    if r.group is None:
        # "smokers ... than non-smokers", "Ash versus Birch": a category compared by its own values or plural name
        for c in cats:
            w = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()
            if re.search(rf"\b{re.escape(w)}s\b", low) and _COMPARE.search(message):
                r.group = c
                break
    if r.group is not None and r.group in r.denominators:
        r.denominators.remove(r.group)
        r.per = bool(r.denominators) or r.per
    _rest(message, df, roles, r, named, low)
    if r.group is None and r.compare:
        _compared_groups(message, df, roles, r, low)
    return r


_VS = r"(?:versus|vs\.?|compared (?:with|to)|against|or)"
_OTHER_STATE = re.compile(rf"\b([a-z]{{4,}})(?:\s+[a-z]+)?\s+{_VS}\s+(?:the\s+)?(?:ordinary|normal|regular|usual|other|"
                          r"standard|non[- ]?\1|no[- ]?\1|without)\b|\b(?:with|on)\s+(?:a\s+|an\s+)?([a-z]{4,})\s+"
                          rf"(?:[a-z]+\s+)?{_VS}\s+without\b", re.I)


def _compared_groups(message, df, roles, r, low):
    """'the typical bill at dinner versus lunch', 'promotion days versus ordinary days': values of one column (or the
    two states of a yes/no column) set against each other are a breakdown by that column -- restricted to the values
    named -- not a filter to all of them at once."""
    by_col: dict = {}
    for c, v in r.entities:
        by_col.setdefault(c, []).append(v)
    pick = [c for c, vs in by_col.items() if len(vs) >= 2 and roles[c]["role"] in ("category", "flag", "person_name")]
    if len(pick) == 1:
        r.group, r.compared = pick[0], True
        return
    m = _OTHER_STATE.search(low)
    if m:
        w = (m.group(1) or m.group(2)).lower()
        flags = [c for c in df.columns if roles[c]["role"] == "flag" and c not in r.measures and any(
            t == w or _same_stem(w, t) or (len(t) >= 4 and w.startswith(t)) for t in semantics._tokens(c))]  # noqa: SLF001
        if len(flags) == 1:
            r.group, r.compared = flags[0], True
            r.restrict = [c for c in r.restrict if c != flags[0]]


# a figure of one measure for each unit of another, said without "per": "turns visitors into takings", "converts hours
# into parcels", "the slice of the bill left as a tip", "tips as a share of the bill", "takings for every visitor".
# The phrase names the denominator; it counts only when it names a numeric column.
_CONVERSION = [re.compile(p, re.I) for p in (
    r"\b(?:turns?|turning|turned|converts?|converting|converted)\s+(?:its\s+|their\s+|the\s+|each\s+)?"
    r"([a-z_]+(?:\s+[a-z_]+)?)\s+into\b",
    r"\b(?:slice|share|portion|fraction|proportion|percentage|percent|part)\s+of\s+(?:the\s+|their\s+|its\s+|each\s+)?"
    r"([a-z_]+(?:\s+[a-z_]+)?)\s+(?:as|left as|given as|paid as|going to|spent on|that goes to)\b",
    r"\bas\s+an?\s+(?:share|percentage|fraction|proportion|percent|part|slice)\s+of\s+(?:the\s+|their\s+|its\s+)?"
    r"([a-z_]+(?:\s+[a-z_]+)?)",
    r"\bfor every\s+([a-z_]+)\b")]
_SIGNIFICANCE = re.compile(r"\b(meaningful|significant|significance|statistically|by chance|real difference|reliable "
                           r"difference|genuine difference)\b", re.I)


def _conversion(message, df, roles, r):
    for pat in _CONVERSION:
        m = pat.search(message)
        if not m:
            continue
        words = set(re.findall(r"[a-z]+", m.group(1).lower()))
        stems = {f for w in words for f in semantics._forms(w)}  # noqa: SLF001
        d = [c for c in df.columns if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year", "flag")
             and ((semantics._tokens(c) - _GENERIC) & stems or str(c).lower() in words)]  # noqa: SLF001
        if len(d) == 1:
            r.per, r.denominators = True, d
            return


def _denominator(message, df, roles, r):
    """The denominator of a per-unit figure ('per driver', 'allowing for how far each drives', 'once you account
    for staffing levels'): columns named in that phrase, by unit word, by stem, or the people column."""
    from app.analyst.metrics import UNIT_WORDS
    for pat in _PER:
        m = pat.search(message)
        if not m:
            continue
        phrase = m.group(1).strip().lower()
        if phrase.startswith("how many"):
            # "relative to how many trips it runs": divided by the number of records
            r.per, r.denominators = True, [_RECORDS]
            break
        pc = _cols_in(phrase, df)
        if pc and all(roles[c]["role"] in ("category", "flag") for c in pc):
            continue                                    # "per depot" groups; it does not divide
        r.per = True
        d = list(pc)
        words = set(re.findall(r"[a-z]+", phrase))
        stems = {f for w in words for f in semantics._forms(w)} | {w + "s" for w in words}  # noqa: SLF001
        d += [c for c in df.columns if kind_of(df[c]) == "numeric" and (semantics._tokens(c) - _GENERIC) & stems]  # noqa: SLF001
        for forms in UNIT_WORDS.values():
            if words & forms:
                d += [c for c in df.columns if kind_of(df[c]) == "numeric" and semantics._tokens(c) & forms]  # noqa: SLF001
        if words & _PEOPLE or "how many people" in phrase:
            d += [c for c in df.columns if roles[c]["role"] == "person_name"
                  or (roles[c]["role"] in ("identifier", "category") and semantics._tokens(c) & _PEOPLE)]  # noqa: SLF001
        if words & {"far", "distance", "long", "time", "size", "sizes"}:
            d += [c for c in df.columns if kind_of(df[c]) == "numeric" and semantics._tokens(c) & {  # noqa: SLF001
                "km", "mi", "miles", "distance", "hours", "minutes", "duration", "size", "sizes"}]
        r.denominators = list(dict.fromkeys(r.denominators + d))
        if r.denominators:
            break                                       # else a later phrase may name what to divide by


def _rest(message, df, roles, r, named, low):
    from app.analyst.calc import named_values
    r.why = bool(_WHY.search(message))
    # the quantity
    cm = _COUNT.search(message)
    if cm and _RECORDS in r.denominators:
        cm = None                                       # "how many trips" there is the denominator, not the question
    plain = message                                       # "hourly rate" in hourly_rate names a column, not a share
    for c in df.columns:
        parts = [t for t in re.split(r"[^a-z0-9]+", semantics._words(str(c)).lower()) if t]  # noqa: SLF001
        if len(parts) > 1:
            plain = re.sub(r"\b" + r"[\s_]+".join(map(re.escape, parts)) + r"s?\b", " ", plain, flags=re.I)
    if _RATE.search(plain):
        r.quantity = "rate"
    elif _MEDIAN.search(message):
        r.quantity = "median"
    elif _AVG.search(message):
        r.quantity = "average"
    elif cm and not _unit(cm.group(2), df) and not _UNUSUAL.search(message):
        r.quantity = "count"                            # ("how many unusual values": the anomaly count, not a count)
    elif _TOTAL.search(message) or (cm and _unit(cm.group(2), df)):
        r.quantity = "total"
    # the measures: named numeric (or flag, for a rate) columns other than the grouping and denominators
    ms = [c for c in named if c != r.group and c not in r.denominators and (
        kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year")
        or (r.quantity == "rate" and roles[c]["role"] == "flag"))]
    words = re.findall(r"[a-z]+", low)
    for w in words:
        ms += [c for c in _unit(w, df) if c not in r.denominators]
        # a distinctive word of exactly one numeric column ("fuel" for fuel_litres, "bill" for total_bill)
        hits = [c for c in df.columns if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year")
                and w in (semantics._tokens(c) - _GENERIC) and c not in r.denominators and c != r.group]  # noqa: SLF001
        if len(hits) == 1 and len(w) > 2:
            ms += hits
        flags = [c for c in semantics.concept_candidates(w, df) if roles.get(c, {}).get("role") == "flag"]
        # "escalation" for escalated, "cancellations" for cancelled: the same stem
        flags += [c for c in df.columns if roles[c]["role"] == "flag" and len(w) > 4
                  and any(_same_stem(w, t) for t in semantics._tokens(c))]  # noqa: SLF001
        flags = [c for c in dict.fromkeys(flags) if c != r.group]     # the grouping is never also the measure
        if r.quantity == "rate":
            ms += flags
        if r.why and flags and r.why_target is None:
            r.why_target = flags[0]                     # "is lateness linked to anything": the outcome is late
    # a yes/no column named with "only" / "among" restricts the records ("among escalated tickets only")
    if _RESTRICT.search(message):
        r.restrict = [c for c in named if roles[c]["role"] == "flag" and roles[c].get("positive")
                      and not roles[c].get("positive_is_minority") and c != r.group and c not in ms]
    # a grouping noun that is no column ("each waiter"): nothing may be grouped by something else instead
    gm = re.search(r"\b(?:each|every|per|by|which|for each)\s+([a-z]{3,})\b", low)
    if gm and r.group is None and not _cols_in(gm.group(1), df) and gm.group(1) not in _PEOPLE | {
            "one", "day", "days", "month", "week", "year", "time", "other", "record", "row", "of", "the", "is", "has",
            "are", "was", "does", "do"} and not semantics.concept_candidates(gm.group(1), df) and \
            not any(gm.group(1).rstrip("s") in str(v).lower() for _, v in named_values(df, message)):
        r.unknown_group = gm.group(1)
    if not ms:
        # "time to resolve" -> resolution_hours: a distinctive word sharing a long stem with exactly one numeric
        # column's name
        for w in words:
            if len(w) <= 4 or w in _GENERIC:
                continue
            hits = [c for c in df.columns if kind_of(df[c]) == "numeric" and c != r.group and c not in r.denominators
                    and roles[c]["role"] not in ("identifier", "year", "flag")
                    and any(_same_stem(w, t) for t in semantics._tokens(c) - _GENERIC)]  # noqa: SLF001
            if len(hits) == 1:
                ms = hits
                break
    if not ms:
        # "which department has the highest attrition": a yes/no column named as the measure means its rate
        fl = [c for c in named if roles[c]["role"] == "flag" and c != r.group]
        fl = [c for c in fl if c not in r.restrict]
        if len(fl) == 1:
            ms = fl
            if not (r.why and not r.group and not _EXTREME.search(message)):
                r.quantity = r.quantity or "rate"     # "what is linked to refunds?" asks what goes with it, not its rate
    if _RECORDS in r.denominators and any(roles.get(c, {}).get("role") == "flag" for c in ms):
        r.quantity = "rate"                             # yes/no records out of all records: a rate
    ms = list(dict.fromkeys(ms))
    if len(ms) > 1:
        # a column named after a quantity word ("total", "count") is the measure only when nothing else is named:
        # "the total number of votes for Joly" asks for a total of Joly
        ms = [c for c in ms if str(c).lower() not in _QUANTITY_NAMES] or ms
    cm2 = _COUNT.search(message)
    if r.quantity == "count" and not ms and cm2:
        # "how many medals did each nation win" on a table of tallies: one count column, named as a count or after
        # the noun counted, holds the counts -- its sum is the answer, not the number of rows
        noun = semantics._forms(cm2.group(2).lower())  # noqa: SLF001
        tallies = [c for c in df.columns if roles[c]["role"] == "count" and c not in r.denominators and c != r.group
                   and (str(c).lower() in _TALLY_NAMES or semantics._tokens(c) & (noun | _TALLY_NAMES))]  # noqa: SLF001
        if len(tallies) == 1:
            ms, r.quantity = tallies, "total"
    if r.quantity == "count" and _TOTAL.search(message) and any(
            kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("flag", "identifier", "year") for c in ms):
        r.quantity = "total"            # "the total number of votes for Joly": a sum of the Joly column, not a count
    r.measures = ms
    if r.why and r.why_target is None and len(r.measures) == 1:
        r.why_target = r.measures[0]
    from app.analyst.scope import _negated
    from app.analyst.scope import column_values
    hits = named_values(df, message)
    as_cols = column_values(message, df, hits)
    named_vals = [(c, v) for c, v in hits if (c, v) not in as_cols]
    r.entities = [(c, v) for c, v in named_vals if not _negated(message, v)]
    r.excluded = [(c, v) for c, v in named_vals if _negated(message, v)]
    r.period = relative_period(message, df)
    r.compare = bool(_COMPARE.search(message))
    em = _EXTREME.search(message)
    if em:
        r.extreme = "min" if _LOW.search(em.group(2)) else ("either" if em.group(2).lower() in ("worst", "best") else "max")
    r.unusual = bool(_UNUSUAL.search(message))
    r.hypothetical = bool(_HYPO.search(message))
    r.significance = bool(_SIGNIFICANCE.search(message)) and (r.compare or bool(r.group))


_NUMW = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "twelve": 12, "fourteen": 14, "thirty": 30, "sixty": 60, "ninety": 90}
_LASTN = re.compile(r"\b(?:last|past|previous|recent)\s+(\d+|" + "|".join(_NUMW) + r")\s+(day|week|month|year)s?\b", re.I)
_THIS = re.compile(r"\b(this|current|last|previous)\s+(week|month|quarter|year)\b|\b(year|month|quarter)[- ]to[- ]date\b|"
                   r"\b(ytd|mtd|qtd)\b", re.I)


def relative_period(message: str, df: pd.DataFrame) -> dict | None:
    """'the last 30 days', 'this quarter', 'last month', 'year to date' as {start, end, label}, counted back from
    today -- or from the last record when the data ends well before today (said in the label). None if the message
    names no relative period or the data has no date column."""
    from app.analyst.nl_planner import anchor_date
    if not any(kind_of(df[c]) == "datetime" for c in df.columns):
        return None
    a, why = anchor_date(df)
    m = _LASTN.search(message)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1).lower()]
        unit = m.group(2).lower()
        start = a - (pd.Timedelta(days=n - 1) if unit == "day" else pd.Timedelta(weeks=n) + pd.Timedelta(days=-1)
                     if unit == "week" else pd.DateOffset(months=n) - pd.Timedelta(days=1) if unit == "month"
                     else pd.DateOffset(years=n) - pd.Timedelta(days=1))
        return {"start": str(pd.Timestamp(start).date()), "end": str(a.date()), "label": f"the last {n} {unit}s ({why})"}
    m = _THIS.search(message)
    if not m:
        return None
    if m.group(4) or m.group(3):
        u = {"ytd": "year", "mtd": "month", "qtd": "quarter"}.get((m.group(4) or "").lower(), (m.group(3) or "").lower())
        which, unit = "this", u
    else:
        which, unit = m.group(1).lower(), m.group(2).lower()
    freq = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}[unit]
    per = a.to_period(freq) - (1 if which in ("last", "previous") else 0)
    end = min(per.end_time.normalize(), a) if which in ("this", "current") else per.end_time.normalize()
    return {"start": str(per.start_time.date()), "end": str(end.date()), "label": f"{which} {unit} ({per}; {why})"}


def _same_stem(a: str, b: str) -> bool:
    """escalation / escalated, cancellations / cancelled: a long shared beginning."""
    import os
    n = len(os.path.commonprefix([a, b]))
    return n >= 5 and n >= 0.7 * min(len(a), len(b))


def _unit(word: str, df: pd.DataFrame) -> list:
    from app.analyst.metrics import _unit_columns
    return _unit_columns(word, df)


# ------------------------------------------------------------------- checking --
def _measure_cols(s: dict, by: dict) -> set:
    """The data columns a step's value is computed from (through derive steps)."""
    out = set()
    c = s.get("column")
    src = by.get(s.get("from"))
    seen = 0
    while src is not None and seen < 20:
        if src["op"] == "derive" and src.get("name") == c:
            out |= set(re.findall(r"[A-Za-z_]\w*", str(src["formula"])))
        src = by.get(src.get("from"))
        seen += 1
    if c:
        out.add(c)
    return out


# words that limit a request to a period. A bare unit is not one: "promotion days", "per week", "each month" describe
# records or a breakdown; a period is qualified ("this month", "the last 30 days", "by month end") or a named date.
_TIME_WORDS = re.compile(r"\b(today|yesterday|since|until|till|before|after|during|recent|recently|latest|ytd|mtd|qtd|"
                        r"to date|so far|\d{4}|january|february|march|april|may|june|july|august|september|october|"
                        r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|pace)\b|"
                        r"\b(?:this|last|past|next|previous|current|coming|recent|same)\s+(?:\d+\s+|few\s+|"
                        r"(?:two|three|four|five|six|seven|eight|nine|ten|twelve|thirty|sixty|ninety)\s+)?"
                        r"(?:day|week|fortnight|month|quarter|year|period)s?\b|"
                        r"\b(?:end|start|beginning|middle)\s+of\s+(?:the\s+)?(?:day|week|month|quarter|year)\b|"
                        r"\b(?:month|year|quarter|week)[- ]end\b|\bbetween\b[^?]{0,30}\band\b[^?]{0,10}\d", re.I)


def check(steps: list[dict], req: Request, df: pd.DataFrame, message: str = "") -> list[str]:
    """Requirements of the request the program does not meet -- a valid program that computes something else."""
    by = {s["id"]: s for s in steps}
    aggs = [s for s in steps if s["op"] == "aggregate"]
    ratios = [s for s in steps if s["op"] == "ratio"]
    numer = {s["id"] for s in aggs if not (s["fn"] in ("count", "nunique") and any(r["b"] == s["id"] for r in ratios))}
    used = set()
    for s in steps:
        if s["op"] in ("aggregate", "project", "anomalies") and (s["op"] != "aggregate" or s["id"] in numer):
            used |= _measure_cols(s, by)
        if s["op"] == "associate":
            used.add(s["target"])
    problems = []
    if req.measures and not used & set(req.measures) and not any(s["op"] in ("scenario", "compare") for s in steps):
        problems.append(f"the request is about {' or '.join(map(str, req.measures))}, but the program computes "
                        f"{', '.join(sorted(map(str, used))) or 'nothing from it'}")
    fns = {s["fn"] for s in aggs if s["id"] in numer}
    flag_like = {s["id"] for s in aggs if s["fn"] == "mean" and any(
        str(c).startswith("is_") or (c in df.columns and semantics.column_role(df, c)["role"] in ("flag", "rate"))
        for c in _measure_cols(s, by))}
    if req.quantity == "rate" and not (flag_like or ratios):
        problems.append("the request asks for a rate (a share of records); compute it as the mean of a 0/1 flag "
                        "(derive '<column> == '<value>'', then aggregate mean) or as a count divided by a count")
    if req.quantity == "average" and not req.per and "mean" not in fns and not ratios:
        problems.append(f"the request asks for an average, but the program computes {', '.join(sorted(fns)) or 'none'}")
    if req.quantity == "median" and "median" not in fns:
        problems.append("the request asks for the median (the typical value); use fn median")
    if req.quantity == "total" and not req.per and not ({"sum"} & fns or any(s["op"] in ("project", "compare")
                                                                              for s in steps)):
        problems.append(f"the request asks for a total, but the program computes {', '.join(sorted(fns)) or 'none'}")
    if req.extreme in ("min", "max", "either") and req.group and req.quantity is None and fns and fns <= {"max", "min"} \
            and not _SINGLE.search(message):
        problems.append(f"the request ranks each {req.group} by its overall level of the measure; 'highest' and "
                        f"'lowest' compare the groups, not each group's single largest or smallest record: aggregate "
                        f"with fn sum or mean by \"{req.group}\"")
    if req.quantity == "count" and not ({"count", "nunique", "sum"} & fns):
        problems.append("the request asks how many; use count (records) or nunique (distinct values)")
    for c in req.restrict:
        if not any(c in (s.get("entities") or {}) or any(w.get("column") == c for w in s.get("where") or [])
                   for s in steps if s["op"] == "select"):
            problems.append(f"the request is limited to records where {c} applies; select them (where {c} == ...)")
    if req.per and _RECORDS in req.denominators:
        if not (flag_like or ratios):
            problems.append("the request divides by the number of records: use the mean of a 0/1 flag, or a count "
                            "divided by a count")
    elif req.per:
        if not ratios:
            problems.append("the request asks for a figure per " + (" / ".join(map(str, req.denominators)) or "unit")
                            + "; divide a total by it with a ratio step")
        elif req.denominators:
            dens = set()
            for r in ratios:
                b = by.get(r["b"])
                if b is not None:
                    dens |= _measure_cols(b, by) if b["op"] == "aggregate" else set()
            if not dens & set(req.denominators):
                problems.append(f"the figure must be divided by {' or '.join(map(str, req.denominators))} "
                                f"(the program divides by {', '.join(sorted(map(str, dens))) or 'something else'})")
    if req.group and not any(s.get("by") == req.group for s in aggs) and not (
            req.entities and any(c == req.group for c, _ in req.entities)):
        problems.append(f"the request breaks the result down by {req.group}; aggregate with by \"{req.group}\"")
    for c, v in req.entities:
        if c == req.group:
            continue
        hit = any(v.lower() in [str(x).lower() for x in vals] for s in steps
                  for key in ("entities", "except") for vals in (s.get(key) or {}).values()) or \
            any(str(w.get("value")).lower() == v.lower() for s in steps for w in s.get("where") or []) or \
            any(str(v).lower() in str(ch).lower() for s in steps if s["op"] == "scenario" for ch in s.get("changes") or [])
        if not hit:
            problems.append(f"the request is about {v} ({c}); the program does not select it")
    plain = message
    for c in df.columns:                                  # "monthly" in monthly_pay names a column, not a period
        parts = [t for t in re.split(r"[^a-z0-9]+", semantics._words(str(c)).lower()) if t]  # noqa: SLF001
        if parts:
            plain = re.sub(r"\b" + r"[\s_]+".join(map(re.escape, parts)) + r"s?\b", " ", plain, flags=re.I)
    if message and not req.period and not _TIME_WORDS.search(plain) and any(
            s["op"] == "select" and s.get("period") for s in steps):
        problems.append("the request names no period, so the program must not select one (use all the records)")
    if req.period and not any(s["op"] == "select" and (s.get("period") or {}).get("start") == req.period["start"]
                              for s in steps):
        problems.append(f"the request is limited to {req.period['label']}: select period start {req.period['start']} "
                        f"end {req.period['end']}")
    for c, v in req.excluded:
        if not any(v.lower() in [str(x).lower() for x in (s.get("except") or {}).get(c, [])] for s in steps) and \
                not any(str(w.get("value")).lower() == v.lower() and w.get("op") == "!=" for s in steps
                        for w in s.get("where") or []):
            problems.append(f"the request leaves out {v} ({c}); the program must exclude it")
    if req.hypothetical and not any(s["op"] in ("scenario", "project") for s in steps):
        problems.append("the request is hypothetical; express it with a scenario step (or a projection), not observed "
                        "totals alone")
    return problems


def complete(steps: list[dict], req: Request, df: pd.DataFrame, prof: dict, fixed: list) -> list[dict]:
    """Parts of a composite request with one deterministic form, added when the program leaves them out."""
    by = {s["id"]: s for s in steps}
    main = next((s for s in steps if s["op"] == "aggregate" and s.get("fn") not in ("count", "nunique")), None)
    src = main.get("from", "data") if main else "data"
    while src in by and by[src]["op"] in ("derive",):
        src = by[src].get("from", "data")              # the records the main result was computed from
    cols = _measure_cols(main, by) if main else set()
    outcome = req.why_target or next((c for c in req.measures if c in df.columns), None) or next(
        (c for c in cols if c in df.columns and kind_of(df[c]) != "datetime"), None)
    if req.why and not any(s["op"] == "associate" for s in steps) and outcome is not None:
        steps = steps + [{"id": "why_links", "op": "associate", "from": src, "target": outcome,
                          "exclude": [c for c in prof["nonfactors"] if c != outcome]}]
        fixed.append(f"the request asks why: Silt added the associations of {outcome} with the other columns")
    num = next((c for c in [*req.measures, *cols] if c in df.columns and kind_of(df[c]) == "numeric"
                and prof["roles"][c]["role"] not in ("flag", "identifier", "year")), None)
    if req.unusual and not any(s["op"] == "anomalies" for s in steps) and num is not None:
        steps = steps + [{"id": "unusual_values", "op": "anomalies", "from": src, "column": num}]
        fixed.append(f"the request asks about unusual values: Silt added the unusual values of {num}")
    return steps


# ------------------------------------------------------- deterministic program --
def rule_program(req: Request, df: pd.DataFrame, prof: dict) -> list[dict] | None:
    """A program for a request whose parts are all grounded (one measure, a quantity; a grouping, or a selection of
    named entities / a restriction / a period), built without a model -- used before the model when it can answer,
    and as the fallback when the model is off or its plans fail. None when anything would be a guess."""
    scoped = bool(req.entities or req.excluded or req.restrict or req.period)
    if len(req.measures) != 1 or req.hypothetical or req.unknown_group or (req.group is None and not scoped):
        return None
    m = req.measures[0]
    role = prof["roles"][m]["role"]
    steps: list[dict] = []
    base = "data"
    # the records the request is about: named entities (not the grouping column), "only <flag>", a period
    sel: dict = {}
    ents = {}
    for c, v in req.entities:
        if c != req.group:
            ents.setdefault(c, []).append(v)
    named_in_group = [v for c, v in req.entities if c == req.group]
    if named_in_group and set(map(str, named_in_group)) != set(df[req.group].dropna().astype(str)):
        ents[req.group] = named_in_group               # "Ash versus Birch": only the values named are compared
    if ents:
        sel["entities"] = ents
    exc = {}
    for c, v in req.excluded:
        exc.setdefault(c, []).append(v)
    if exc:
        sel["except"] = exc
    if req.restrict:
        where = []
        for c in req.restrict:
            pos = prof["roles"][c].get("positive")
            if not pos:
                return None
            where.append({"column": c, "op": "==", "value": pos})
        sel["where"] = where
    if req.period:
        if len(prof["dates"]) != 1:
            return None
        sel["period"] = {"start": req.period["start"], "end": req.period["end"]}
    if sel:
        steps.append({"id": "records", "op": "select", "from": "data", **sel})
        base = "records"
    src, col = base, m
    if role == "flag" and kind_of(df[m]) != "numeric":
        pos = prof["roles"][m].get("positive")
        if not pos:
            return None
        steps.append({"id": "flag", "op": "derive", "from": base, "name": f"is_{re.sub(r'[^0-9a-zA-Z_]', '_', str(m))}",
                      "formula": f"{m} == '{pos}'"})
        src, col = "flag", steps[-1]["name"]
    q = req.quantity or ("rate" if role == "flag" else None)
    by = {"by": req.group} if req.group else {}
    if req.per and req.denominators == [_RECORDS] and col != m:
        q = "rate"                                      # flag records out of all records: the mean of the flag
    elif req.per:
        if len(req.denominators) != 1:
            return None
        d = req.denominators[0]
        dfn = "sum" if kind_of(df[d]) == "numeric" else "nunique"
        steps += [{"id": "numerator", "op": "aggregate", "from": src, "fn": "sum", "column": col, **by},
                  {"id": "denominator", "op": "aggregate", "from": base, "fn": dfn, "column": d, **by},
                  {"id": "per_unit", "op": "ratio", "a": "numerator", "b": "denominator"}]
        answer = "per_unit"
    if not (req.per and req.denominators != [_RECORDS]):
        fn = {"rate": "mean", "average": "mean", "median": "median", "total": "sum", "count": "count"}.get(q or "")
        if fn is None or (fn == "mean" and q == "rate" and src == base and role not in ("rate", "flag")):
            return None
        steps.append({"id": "result", "op": "aggregate", "from": src, "fn": fn,
                      **({} if fn == "count" else {"column": col}), **by})
        answer = "result"
    if req.why:
        t = req.why_target or m
        steps.append({"id": "why_links", "op": "associate", "from": base, "target": t,
                      "exclude": [c for c in prof["nonfactors"] if c != t]})
    if req.unusual and kind_of(df[m]) == "numeric" and role not in ("flag",):
        steps.append({"id": "unusual_values", "op": "anomalies", "from": base, "column": m})
    return steps if not check(steps, req, df) else None
