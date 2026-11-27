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
_TOTAL = re.compile(r"\b(total|totals|in all|altogether|how much|sum of|sum up|sums up|summed up|add(?:s|ed|ing)? up|"
                    r"combined|in total|aggregate|grand total|tally up)\b", re.I)
_DISTINCT = r"(?:distinct|unique|different|separate|individual)"
_COUNT = re.compile(rf"\b(how many|number of|count of|count the|counts? (?:by|per|for each|in each))\s+(?:{_DISTINCT}\s+)?(\w+)",
                    re.I)
_DISTINCT_COUNT = re.compile(rf"\b(?:how many|number of|count of|count the|count)\s+{_DISTINCT}\b|\b{_DISTINCT}\s+count\b",
                             re.I)
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
_EXTREME = re.compile(r"\b(which|what|who|where|identify|find|name|determine)\b[^?]{0,80}\b(most|highest|largest|biggest|best|top|longest|least|lowest|"
                      r"smallest|fewest|worst|slowest|shortest|bottom|fastest|weakest|strongest|poorest)\b", re.I)
_SINGLE = re.compile(r"\b(single|maximum|minimum|max|min|individual|one (?:order|sale|record|transaction|row)|"
                     r"per (?:order|sale|record|transaction|row)|largest one|biggest one)\b", re.I)
# Ranking groups ("the highest revenue region"): what each group is compared by. A flow adds up over records (revenue,
# cost, units sold) -- groups are compared by their total; a level describes each record (a price, a rate, a salary,
# a duration) -- by their average. Words in the question decide first; the measure's name second; neither: ask.
_FLOW_NAME = re.compile(r"revenue|sales|turnover|takings|income|gmv|spend|expense|cost|profit|budget|amount|qty|"
                        r"quantity|units|volume|orders|visits|visitors|bookings|tickets|bill|"
                        # counts of events: a company's reviews, a video's views -- added up per group
                        r"reviews?\b|votes?\b|comments?\b|likes?\b|followers?|downloads?|installs?|views?\b|clicks?\b|"
                        r"calls?\b|applications?|signups?|subscribers?|purchases?|transactions?|complaints?", re.I)
_LEVEL_NAME = re.compile(r"rate|price|percent|pct|ratio|avg|average|score|rating|margin|\bage\b|tenure|temperature|"
                         r"salary|wage|\bpay\b|fee|fare|duration|minutes|length|size|weight|discount|satisfaction", re.I)
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
    counted: str | None = None          # "how many companies": the column whose distinct values are counted
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
    # --- restrictions the question states; each must be applied by whatever answers, or the answer is not shown ---
    conditions: list = field(default_factory=list)      # [{"column", "op", "value", "text"}]  "csat below 3"
    unresolved: list = field(default_factory=list)      # a stated threshold no column could be found for ("over 500")
    restrict_not: list = field(default_factory=list)    # yes/no columns that must be "no"   ("not escalated")
    unmatched: list = field(default_factory=list)       # names the question gives that occur nowhere in the data
    share_of: str | None = None         # "what share of revenue ...": a share of this column's total, not of records
    share_word: str | None = None       # ... when the word names a measure no column could be found for
    ambiguous_names: list = field(default_factory=list)  # [(name, [values containing it])] in a long text column
    distinct: bool = False              # "how many distinct / unique / different X": distinct values, not records
    event: tuple | None = None          # "cancellation rate": (column, value) whose share of records is asked for
    calendar: dict | None = None        # a named month / year ("in August 2026"), applied through the scope
    sort: str | None = None             # "asc" / "desc": the order a breakdown is asked to be shown in
    record_noun: bool = False           # "how many shifts": the thing counted is the record itself (one id each)
    relation: tuple | None = None       # "does X correlate with / go with / cause Y": two measures, as (x, y)
    causal: bool = False                # ... asked as cause and effect ("cause", "drive", "lead to", "because")
    relation_kind: str = "numbers"      # "numbers" (two measures: correlation) | "groups" (a measure across a category)

    def parts(self) -> list[str]:
        out = []
        q = {"rate": "the rate (a share of records)", "average": "the average", "median": "the median",
             "total": "the total", "count": "the count"}.get(self.quantity or "", "the measure")
        what = f"{q} of {' / '.join(map(str, self.measures))}" if self.measures else q
        if self.share_of:
            what = f"the share of {self.share_of} (a part of its total)"
        elif self.event:
            what = f"the rate (a share of records) of {self.event[0]} = {self.event[1]}"
        elif self.quantity == "count" and self.counted:
            what = f"the number of distinct {self.counted} values"
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


def request_to_dict(r: Request) -> dict:
    """A reading as plain data (kept in conversation memory, persisted between turns)."""
    import dataclasses
    return {k: v for k, v in dataclasses.asdict(r).items() if not k.startswith("_")}


def request_from_dict(d: dict) -> Request:
    """Back from conversation memory: pairs become tuples again; unknown keys (an older memory) are ignored."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(Request)}
    r = Request(**{k: v for k, v in (d or {}).items() if k in names})
    r.entities = [tuple(x) for x in r.entities]
    r.excluded = [tuple(x) for x in r.excluded]
    for k in ("event", "relation"):
        if getattr(r, k) is not None:
            setattr(r, k, tuple(getattr(r, k)))
    r.conditions = [dict(x) for x in r.conditions]
    return r


def _cols_in(text: str, df: pd.DataFrame) -> list:
    from app.analyst.orchestrator import followup_cols
    return list(followup_cols(df, text))


_READ_CACHE: dict = {}


def read(message: str, df: pd.DataFrame, prof: dict | None = None) -> Request:
    """The structured reading of one question -- the single interpretation every path, planner and verifier uses.
    Cached per (message, table), and returned as a copy, so no caller's adjustments leak into another's."""
    import copy
    key = (message, id(df), df.shape, tuple(map(str, df.columns)))
    hit = _READ_CACHE.get(key)
    if hit is not None and hit[0] is df:
        return copy.deepcopy(hit[1])
    r = _read(message, df, prof)
    if r.group is not None and r.group in r.denominators:
        # the grouping column is never also the divisor: "mean adr per property" is the mean for each property (the
        # same rule _read applies when the grouping is found first; here it may be found after the denominator)
        r.denominators = [d for d in r.denominators if d != r.group]
        r.per = bool(r.denominators)
    if len(_READ_CACHE) > 256:
        _READ_CACHE.clear()
    _READ_CACHE[key] = (df, copy.deepcopy(r))
    return r


def _read(message: str, df: pd.DataFrame, prof: dict | None = None) -> Request:
    from app.analyst.calc import named_values
    from app.analyst.metrics import UNIT_WORDS
    prof = prof or semantics.profile(df)
    distinct = bool(_DISTINCT_COUNT.search(message))
    if distinct:
        # "how many different channels": 'different' qualifies the count -- it is not a comparison of channels
        message = re.sub(rf"\b((?:how many|number of|count of|count the|count)\s+){_DISTINCT}\s+", r"\1", message, flags=re.I)
    roles = prof["roles"]
    low = message.lower()
    r = Request()
    named = _cols_in(message, df)
    _denominator(message, df, roles, r)
    if not r.denominators:
        _conversion(message, df, roles, r)
    # a year column named after a grouping word ("by season", "each year") breaks the result down too
    cats = [c for c in named if roles[c]["role"] in ("category", "flag", "person_name", "year") and c not in r.denominators]
    # the grouping: a category named after a grouping word ("by depot", "each team", "which outlet", "between depots");
    # a quantity word does not count towards the words in between ("the highest average revenue region")
    low_g = re.sub(r"\b(?:average|mean|median|total|overall)\s+", "", low)
    # "top 10 companies by rating", "rank the cities by revenue": the column after "by" is what is ranked on
    ranked_on = re.search(r"\b(?:top|bottom|best|worst|highest|lowest|rank\w*|sort\w*|order\w*)\b[^.?!]{0,40}?\bby\s+"
                          r"(?:the\s+)?([a-z][a-z_ ]{0,30})", low_g)
    ranked_cols = set(_cols_in(ranked_on.group(1), df)) if ranked_on else set()
    for c in cats:
        if c in ranked_cols:
            continue
        w = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()
        from app.analyst.scope import _negated
        split = re.sub(r"[^a-z0-9]+", " ", semantics._words(str(c))).split()  # noqa: SLF001
        if len(split) >= 2 and split[-1] in ("code", "cd", "id"):
            wp = rf"(?:{re.escape(w)}|{re.escape(' '.join(split[:-1]))})"    # "the depots" for depotCode
        else:
            wp = re.escape(w)
        if re.search(rf"\b{_GROUP_WORDS}\s+(?:\w+\s+){{0,2}}?{wp}s?\b", low_g) and c not in [
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
    if r.group is None:
        # "which company", "top 10 companies", "each company": the column that names companies (app/analyst/
        # schema_profile.entity_column), never an attribute of the company such as company_type
        from app.analyst.schema_profile import entity_column
        for gm in re.finditer(rf"\b(?:{_GROUP_WORDS}|top|bottom|best|worst)\s+(?:\w+\s+){{0,2}}?([a-z]+)\b", low_g):
            ent = entity_column(df, gm.group(1))
            if ent is not None and ent not in r.denominators:
                r.group = ent
                break
    if r.group is not None and r.group in r.denominators:
        r.denominators.remove(r.group)
        r.per = bool(r.denominators) or r.per
    r.distinct = distinct
    if re.search(r"\b(?:ascending|lowest (?:to|first)|smallest (?:to|first)|least (?:to|first)|low to high|bottom up|"
                 r"from (?:the )?(?:lowest|smallest|least))\b", low):
        r.sort = "asc"
    elif re.search(r"\b(?:descending|highest (?:to|first)|largest (?:to|first)|most (?:to|first)|high to low|"
                   r"from (?:the )?(?:highest|largest|most))\b", low):
        r.sort = "desc"
    _rest(message, df, roles, r, named, low)
    if r.group is None and r.compare:
        _compared_groups(message, df, roles, r, low)
    if r.counted and r.group == r.counted:
        r.group = None                  # "how many channels": the channels are counted, not a breakdown
    try:
        from app.analyst.scope import _time
        r.calendar = None if r.period else _time(df, message)
    except Exception:  # noqa: BLE001
        r.calendar = None
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
        if pat is _PER[0]:
            # "per product line": the words after "per" may name a category in two or three words
            tail = re.findall(r"[a-z_]+", message[m.start(1):m.start(1) + 60].lower())[:3]
            for k in (3, 2):
                pc_ = [c for c in _cols_in(" ".join(tail[:k]), df) if roles[c]["role"] in ("category", "flag")
                       and re.split(r"[^a-z0-9]+", semantics._words(str(c)).lower())[0] == tail[0]]  # noqa: SLF001
                if len(tail) >= k and pc_:
                    phrase = " ".join(tail[:k])
                    break
        unit_adj = {"hour": "hourly", "day": "daily", "week": "weekly", "month": "monthly", "year": "yearly",
                    "annum": "annual", "night": "nightly"}
        u = phrase.split()[0] if phrase else ""
        adj = unit_adj.get(u.rstrip("s"))
        if adj and pat is _PER[0]:
            # "pay rate per hour" names the hourly_rate column ("per hour" = hourly); it divides nothing
            hits = [c for c in df.columns if kind_of(df[c]) == "numeric" and adj in semantics._tokens(c)]  # noqa: SLF001
            if len(hits) == 1:
                r._per_unit_measure = hits[0]
                continue
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
    from app.analyst.schema_profile import entity_column
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
    ms = ([r._per_unit_measure] if getattr(r, "_per_unit_measure", None) else []) + [
        c for c in named if c != r.group and c not in r.denominators and (
        kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year")
        or (r.quantity == "rate" and roles[c]["role"] == "flag"))]
    if getattr(r, "_per_unit_measure", None) and r.quantity == "rate" and not _RATE.search(
            re.sub(r"\b(?:pay|hourly|daily|weekly|monthly)\s+rate\b", "", low)):
        r.quantity = "average" if _AVG.search(message) else None
    # the words of a date window ("the last 30 days", "this month") say when, never what is measured
    words = re.findall(r"[a-z]+", _THIS.sub(" ", _LASTN.sub(" ", low)))
    for w in words:
        ms += [c for c in _unit(w, df) if c not in r.denominators]
        # a distinctive word of exactly one numeric column ("fuel" for fuel_litres, "bill" for total_bill)
        hits = [c for c in df.columns if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year")
                and w in (semantics._tokens(c) - _GENERIC) and c not in r.denominators and c != r.group]  # noqa: SLF001
        if hits and entity_column(df, w) is not None:
            hits = []                     # "company" names the companies (an entity), not Company_Age
        if len(hits) == 1 and len(w) > 2:
            ms += hits
        flags = [c for c in semantics.concept_candidates(w, df) if roles.get(c, {}).get("role") == "flag"]
        # "escalation" for escalated, "cancellations" for cancelled: the same stem
        flags += [c for c in df.columns if roles[c]["role"] == "flag" and len(w) >= 4
                  and any(_same_stem(w, t) or (len(t) >= 4 and lemma(w) == lemma(t))
                          for t in semantics._tokens(c))]  # noqa: SLF001 -- "pass" / passed
        flags = [c for c in dict.fromkeys(flags) if c != r.group]     # the grouping is never also the measure
        if r.quantity == "rate":
            ms += flags
        if r.why and flags and r.why_target is None:
            r.why_target = flags[0]                     # "is lateness linked to anything": the outcome is late
    # a yes/no column named with "only" / "among" restricts the records ("among escalated tickets only")
    if _RESTRICT.search(message):
        r.restrict = [c for c in named if roles[c]["role"] == "flag" and roles[c].get("positive")
                      and not roles[c].get("positive_is_minority") and c != r.group and c not in ms]
    ms = _stated_filters(message, df, roles, r, named, ms, low)
    # a grouping noun that is no column ("each waiter"): nothing may be grouped by something else instead
    gm = re.search(r"\b(?:each|every|per|by|which|for each)\s+([a-z]{3,})\b", low)
    if gm and r.group is None and entity_column(df, gm.group(1)) is not None:
        r.group = entity_column(df, gm.group(1))         # "which company": the column naming the companies
        ms = [c for c in ms if c != r.group]
    if gm and r.group is None and not _cols_in(gm.group(1), df) and gm.group(1) not in _PEOPLE | {
            "one", "day", "days", "month", "week", "year", "time", "other", "record", "row", "of", "the", "is", "has",
            "are", "was", "does", "do"} and not semantics.concept_candidates(gm.group(1), df) and \
            not any(gm.group(1).rstrip("s") in str(v).lower() for _, v in named_values(df, message)):
        r.unknown_group = gm.group(1)
    if not ms:
        # "time to resolve" -> resolution_hours: a distinctive word sharing a long stem with exactly one numeric
        # column's name
        for w in words:
            if len(w) <= 4 or w in _GENERIC or entity_column(df, w) is not None:
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
        fl = [c for c in fl if c not in r.restrict and c not in r.restrict_not]
        if len(fl) == 1:
            ms = fl
            if not (r.why and not r.group and not _EXTREME.search(message)):
                r.quantity = r.quantity or "rate"     # "what is linked to refunds?" asks what goes with it, not its rate
    if _RECORDS in r.denominators and any(roles.get(c, {}).get("role") == "flag" for c in ms):
        r.quantity = "rate"                             # yes/no records out of all records: a rate
    ms = list(dict.fromkeys(ms))
    if len(ms) > 1:
        # a column named after a quantity word ("total", "count") is the measure only when nothing else is named:
        # "the total number of votes for Joly" asks for a total of Joly -- unless the word is used as the column's
        # name, not as a quantity ("discount and amount", "higher amount": no "of" follows it)
        ms = [c for c in ms if str(c).lower() not in _QUANTITY_NAMES
              or re.search(rf"\b{re.escape(str(c).lower())}\b(?!\s+of\b)", low)] or ms
    cm2 = _COUNT.search(message)
    if r.quantity == "count" and not ms and cm2 and entity_column(df, cm2.group(2)) not in (None, r.group):
        r.counted = entity_column(df, cm2.group(2))       # "how many companies": distinct companies, not rows
    if r.quantity == "count" and not ms and cm2 and not cm2.group(1).lower().startswith("count") \
            and entity_column(df, cm2.group(2)) is None:     # "how many companies": distinct companies, not a tally
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
    if r.quantity == "count" and not r.counted and cm2 and not r.per and not re.match(r"counts?\s+(?:by|per|for|in)\b",
                                                                                   cm2.group(1), re.I):
        # ("counts by status": status is the breakdown, not what is counted)
        ct = count_target(cm2.group(2), df, roles, r.distinct)
        if ct is not None:
            r.counted = ct
            ms = [c for c in ms if c != ct]
    if r.quantity == "count" and r.conditions:
        ms = [c for c in ms if c not in {x["column"] for x in r.conditions}]   # "how many ... longer than 10 hours"
    if r.quantity == "count" and not r.counted and cm2 and not ms and _is_record_noun(cm2.group(2), df, roles):
        r.record_noun = True
    if r.quantity == "count" and r.counted and ms and all(roles[c]["role"] == "identifier" for c in ms):
        ms = []                                         # "how many distinct customer ids": the ids are counted
    r.measures = ms
    nums = [c for c in ms if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year", "flag")]
    cats_named = [c for c in named if roles[c]["role"] in ("category", "flag") and df[c].nunique() <= 30
                  and c not in {x for x, _ in named_values(df, message)}]
    if len(nums) == 1 and len(cats_named) == 1 and _RELATION.search(message) and not r.quantity and not r.extreme:
        # "does the region affect amount": a measure compared across the groups of one category
        r.relation = (cats_named[0], nums[0])
        r.relation_kind = "groups"
        r.causal = bool(_CAUSAL.search(message))
        r.group = None
    cats_all = [c for c in named if roles[c]["role"] in ("category", "flag") and df[c].nunique() <= 30]
    if not nums and len(cats_all) == 2 and _RELATION.search(message) and not r.quantity and not r.extreme \
            and r.relation is None:
        # "is status related to region": how one category's shares differ across the other's groups (chi-square)
        order = sorted(cats_all, key=lambda c: _first_mention(low, c))
        r.relation, r.relation_kind = (order[1], order[0]), "groups"
        r.causal = bool(_CAUSAL.search(message))
        r.group = None
    if len(nums) == 2 and _RELATION.search(message) and not r.group and not r.quantity:
        # "does discount correlate with higher amount", "is X related to Y", "does X cause Y": an association between
        # two measures -- asked as cause and effect or not, it is measured as an association
        order = sorted(nums, key=lambda c: _first_mention(low, c))
        r.relation = (order[0], order[1])
        r.causal = bool(_CAUSAL.search(message))
    if r.why and r.why_target is None and len(r.measures) == 1:
        r.why_target = r.measures[0]
    from app.analyst.scope import _negated
    from app.analyst.scope import column_values
    hits = named_values(df, message)
    as_cols = column_values(message, df, hits)
    named_vals = [(c, v) for c, v in hits if (c, v) not in as_cols]
    found, r.ambiguous_names = getattr(r, "_long_names", ([], []))
    named_vals += [(c, v) for c, v in found if (c, v) not in named_vals and c != r.group]
    r.entities = [(c, v) for c, v in named_vals if not _negated(message, v)]
    r.excluded = [(c, v) for c, v in named_vals if _negated(message, v)]
    if r.quantity == "rate" and not r.measures and not r.share_of:
        _event(message, df, roles, r, low)
    r.period = relative_period(message, df) or explicit_period(message, df)
    r.compare = bool(_COMPARE.search(message))
    em = _EXTREME.search(re.sub(r"\bat (?:least|most)\b|\bno (?:more|less|fewer) than\b", " ", message, flags=re.I))
    if em:
        r.extreme = "min" if _LOW.search(em.group(2)) else ("either" if em.group(2).lower() in ("worst", "best") else "max")
    r.unusual = bool(_UNUSUAL.search(message))
    r.hypothetical = bool(_HYPO.search(message))
    r.significance = bool(_SIGNIFICANCE.search(message)) and (r.compare or bool(r.group))


# ------------------------------------------------- stated restrictions: thresholds, yes/no filters, names, shares --
_CMP = ((r"at least|no less than|no fewer than|a minimum of|minimum of|>=|≥", ">="),
        (r"at most|no more than|a maximum of|maximum of|<=|≤", "<="),
        (r"more than|greater than|higher than|bigger than|larger than|longer than|above|over|exceeding|exceeds?|>", ">"),
        (r"less than|fewer than|lower than|smaller than|shorter than|below|under|<", "<"),
        (r"equal to|equals|exactly|==|=", "=="))
_CMP_ALT = "|".join(p for p, _ in _CMP)
_NUMBER = r"[$£€₹]?\s*(-?\d[\d,]*(?:\.\d+)?)(?:\s*(k\b|m\b|%|percent\b))?"
_COND = re.compile(rf"(?P<before>(?:[a-z_]+[\s-]+){{0,4}})(?P<neg>not\s+)?(?<![a-z])(?P<cmp>{_CMP_ALT})\s*{_NUMBER}"
                   rf"(?P<after>(?:\s+(?!(?:and|or|but)\b)[a-z_]+){{0,3}})", re.I)   # never across "and more than 1000 reviews"
_COND_TAIL = re.compile(rf"(?P<before>(?:[a-z_]+[\s-]+){{1,4}}?)(?:of\s+|is\s+|are\s+|was\s+|were\s+|=\s*)?{_NUMBER}\s+"
                        r"(?:or|and)\s+(?P<dir>more|above|over|higher|greater|up|less|fewer|below|under|lower)\b", re.I)
_NOT_A_THRESHOLD = re.compile(r"^\s*(?:days?|weeks?|months?|years?|quarters?)\s+(?:ago|back|old)\b|^\s*(?:january|february|"
                              r"march|april|may|june|july|august|september|october|november|december)\b", re.I)
_FLIP = {">": "<=", ">=": "<", "<": ">=", "<=": ">", "==": "!="}


def _stated_conditions(message: str, df: pd.DataFrame, roles: dict, r: "Request") -> None:
    """'tickets with csat below 3', 'orders over 500 in amount', 'taking more than 10 hours', 'rated 4 or above':
    a threshold on a numeric column, read with the column it is said about. A threshold whose column cannot be found
    is recorded as unresolved -- the answer may not ignore it."""
    from app.analyst import bands
    try:
        if bands.parse(message, df) is not None:
            return                                      # several value bands of one column: counted per band, not filtered
    except Exception:  # noqa: BLE001
        pass
    low = message.lower()
    if _HYPO.search(message) or re.search(r"\b(what if|targets?|goals?|quotas?|forecast\w*|predict\w*|project\w*)\b", low):
        return                  # a number in a scenario, target or prediction is an input to it, not a record filter

    def numeric_cols(text):
        return [c for c in _cols_in(text, df) if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("year",)]

    def column_for(before, after):
        aw = re.findall(r"[a-z_]+", after.lower())
        if aw and aw[0] in ("in", "of", "for", "on") and len(aw) > 1:
            hit = numeric_cols(" ".join(aw[1:]))
            if hit:
                return hit[0]
        if aw and aw[0] not in ("in", "of", "for", "on", "and", "or", "the", "a"):
            hit = [c for c in numeric_cols(aw[0]) if roles[c]["role"] != "identifier"]
            if hit:
                return hit[0]                           # "more than 5 units": the noun counted is the column
        if aw:                                          # "more than 10 hours": the unit names the column
            u = [c for c in _unit(aw[0], df) if kind_of(df[c]) == "numeric"]
            if len(u) == 1:
                return u[0]
        words = re.findall(r"[a-z_]+", before.lower())
        for n in range(1, len(words) + 1):              # the nearest column named before the comparison
            hit = [c for c in numeric_cols(" ".join(words[-n:])) if roles[c]["role"] != "identifier"]
            if hit:
                return hit[-1]
        words = words + [x for x in aw if x not in ("than", "the", "a", "an")]
        for w in reversed(words):                       # ... or a distinctive word of exactly one numeric column
            if len(w) <= 3 or w in _GENERIC:
                continue
            hit = [c for c in df.columns if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("year", "flag")
                   and roles[c]["role"] != "identifier" and (w in semantics._tokens(c) - _GENERIC  # noqa: SLF001
                        or any(_same_stem(w, t) for t in semantics._tokens(c) - _GENERIC))]  # noqa: SLF001
            if len(hit) == 1:
                return hit[0]
        return None

    def value_of(num, suffix, col):
        v = float(num.replace(",", ""))
        suffix = (suffix or "").lower()
        if suffix == "k":
            v *= 1_000
        elif suffix == "m":
            v *= 1_000_000
        elif suffix in ("%", "percent") and col is not None and roles[col].get("scale") == "0-1":
            v /= 100
        return v

    spans = []
    for m in _COND.finditer(low):
        if _NOT_A_THRESHOLD.search(m.group("after") or "") or re.search(r"\b(?:top|bottom|first|last|past|next)\s*$",
                                                                       m.group("before") or ""):
            continue
        op = next(o for p_, o in _CMP if re.fullmatch(p_, m.group("cmp"), re.I))
        if m.group("neg"):
            op = _FLIP[op]
        col = column_for(m.group("before") or "", m.group("after") or "")
        text = low[m.start("cmp") if not m.group("neg") else m.start("neg"):m.end(5) if m.group(5) else m.end(4)].strip()
        spans.append((m.start(), m.end()))
        if col is None:
            r.unresolved.append(text)
        else:
            r.conditions.append({"column": col, "op": op, "value": value_of(m.group(4), m.group(5), col), "text": text})
    for m in _COND_TAIL.finditer(low):
        if any(a <= m.start(2) < b for a, b in spans):
            continue
        col = column_for(m.group("before"), "")
        if col is None:
            continue                                    # "5 or more" with nothing measured before it: not a threshold
        op = ">=" if m.group("dir") in ("more", "above", "over", "higher", "greater", "up") else "<="
        r.conditions.append({"column": col, "op": op, "value": value_of(m.group(2), m.group(3), col),
                             "text": low[m.start(2):m.end()].strip()})


def _flag_words_of(c) -> list[str]:
    w = re.sub(r"[^a-z0-9]+", " ", semantics._words(str(c)).lower()).strip()  # noqa: SLF001
    w = re.sub(r"^(?:is|was|has|had|did) ", "", w)
    return [re.escape(w).replace(r"\ ", r"[\s_-]+")]


def flag_values(df: pd.DataFrame, c, role: dict):
    """(yes, no) values of a yes/no column -- numbers for 0/1 codes, the written values otherwise; None when the
    column's two values are not a yes/no pair."""
    if role.get("role") != "flag" or role.get("positive") is None or role.get("positive_is_minority"):
        return None
    if kind_of(df[c]) == "numeric":
        vals = sorted(pd.to_numeric(df[c], errors="coerce").dropna().unique())
        pos = float(role["positive"])
        neg = next((float(v) for v in vals if float(v) != pos), None)
        return (pos, neg) if neg is not None else None
    from app.analyst.metrics import yes_no
    return yes_no(df, c)


def flag_mask(d: pd.DataFrame, c, pair, yes: bool = True) -> pd.Series:
    want = pair[0] if yes else pair[1]
    if isinstance(want, float):
        return pd.to_numeric(d[c], errors="coerce") == want
    return d[c].astype(str).str.strip().str.lower() == str(want).lower()


def _stated_flags(message: str, df: pd.DataFrame, roles: dict, r: "Request", named: list, ms: list, low: str) -> list:
    """'average amount of gift wrapped orders', 'how many tickets were not escalated': a yes/no column said about the
    records, next to another measure or in a count, selects records -- it is not the number asked for."""
    from app.analyst.scope import _negated
    others = [c for c in ms if roles[c]["role"] != "flag"]
    for c in named:
        role = roles[c]
        if role["role"] != "flag" or c == r.group or c in r.denominators or flag_values(df, c, role) is None:
            continue
        w = _flag_words_of(c)[0]
        both = re.search(rf"\b(?:non|not|un)[\s-]*{w}\b.*\b{w}\b|\b{w}\b[^.?!]*\b(?:non|not|un)[\s-]*{w}\b|"
                         rf"\b{w}\b\s+(?:and|or|vs\.?|versus|against|compared (?:to|with))\s+(?:the\s+)?(?:non|not|other|rest|"
                         rf"ordinary|normal|regular|un)", low)
        if both or re.search(rf"\b(?:by|per|across|between|whether|if)\s+(?:the\s+)?(?:\w+\s+)?{w}\b", low):
            continue                                    # both states set against each other: a breakdown, not a filter
        said_about_records = bool(others) or r.quantity == "count" or c in r.restrict
        if not said_about_records or (r.quantity == "rate" and not others and c not in r.restrict):
            continue                                    # the flag itself is what is measured ("escalation rate")
        neg = re.search(rf"\b(?:not|non|never|no|un)[\s-]*(?:been\s+|yet\s+|being\s+)?{w}\b|n't\s+(?:been\s+)?{w}\b",
                        low) or _negated(message, re.sub(r"\\.|\[.*?\]\+", " ", w).strip() or str(c))
        if neg:
            r.restrict = [x for x in r.restrict if x != c]
            if c not in r.restrict_not:
                r.restrict_not.append(c)
        elif c not in r.restrict:
            r.restrict.append(c)
        ms = [x for x in ms if x != c]
    return ms


_SHARE_OF = re.compile(r"\b(?:share|percentage|percent|proportion|fraction|portion|part|how much)\s+of\s+"
                       r"(?:the\s+|our\s+|all\s+|total\s+|overall\s+|their\s+|its\s+)*([a-z_]+(?:\s+[a-z_]+)?)|"
                       r"%\s+of\s+(?:the\s+|our\s+|all\s+|total\s+|overall\s+)*([a-z_]+(?:\s+[a-z_]+)?)", re.I)


def _stated_share(message: str, df: pd.DataFrame, roles: dict, r: "Request", low: str) -> None:
    """'what percentage of revenue comes from East': a share of a measure's total (East's revenue ÷ all revenue), not
    the share of records that are East. The measure must be one additive numeric column; a measure word no column is
    named after ('revenue' on a table with amount and unit_price) is asked about, never answered as a share of records."""
    m = _SHARE_OF.search(low)
    if not m or r.per or r.denominators:
        return                  # "gratuity as a share of the bill": one measure over another, read by _conversion
    phrase = (m.group(1) or m.group(2) or "").strip()
    words = re.findall(r"[a-z_]+", phrase)
    if not words:
        return

    def measure_cols(text):
        return [c for c in _cols_in(text, df) if kind_of(df[c]) == "numeric" and roles[c]["role"] not in (
            "identifier", "year", "flag") and semantics.additive(df, c, roles[c])]
    cols = measure_cols(phrase) or measure_cols(words[0])
    # "percentage of bills paid by smokers" counts bills (records); "percentage of the bill", "share of revenue",
    # "share of units" name the measure itself: the word must be the column's own word as written, not its plural
    cols = [c for c in cols if set(words) & (semantics._tokens(c) | {str(c).lower()})]  # noqa: SLF001
    if not cols:
        cols = [c for c in df.columns if kind_of(df[c]) == "numeric" and roles[c]["role"] not in ("identifier", "year",
                "flag") and semantics.additive(df, c, roles[c]) and words[0] in semantics._tokens(c) - _GENERIC]  # noqa: SLF001
    if len(cols) == 1:
        r.share_of = cols[0]
        return
    money = semantics._AMOUNT_WORDS | semantics.CONCEPTS[2] | semantics.CONCEPTS[3] | semantics.CONCEPTS[6]  # noqa: SLF001
    w = next((x for x in words if x in money), None)    # as written: "bills" counts bills, "revenue" is a measure
    if w and not cols:
        r.share_word = w


_NAME_PREP = (r"(?:for|of|by|from|about|to|with|versus|vs\.?|than|against|in|at|only|just|except|excluding|including|"
              r"and|or|is|are|was|were|did|does|has|have|had)")
# "Compare TCS and Accenture": the first name follows the comparison verb (capitalised at the start of a sentence)
_NAMED = re.compile(rf"(?:\b{_NAME_PREP}|(?i:\b(?:compare|compared|comparing|between)))\s+(?:the\s+)?((?:[A-Z][\w'’.-]*[A-Za-z])(?:\s+[A-Z][\w'’-]*[A-Za-z]){{0,3}})"
                    r"(?:'s|’s)?")
_KNOWN_NAMES = set("""january february march april may june july august september october november december jan feb mar
apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur
thurs fri sat sun q1 q2 q3 q4 silt copilot random forest welch pearson spearman kruskal wallis holm mann whitney cramer
cramér tukey anova bonferroni student fisher chi excel python pandas csv pdf iqr roi aov kpi ytd mtd qtd ai ml i ok
english today yesterday tomorrow christmas easter diwali ramadan eid""".split())


def _vocabulary(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Every word the data itself contains -- (all of them, those of column names and short category columns). A
    word of the question may match the second set loosely (Engineers / Engineering); the names of a long column
    (4,000 companies) only exactly, or almost any word would resemble one of them."""
    cached = df.attrs.get("_silt_vocabulary")
    if cached is not None and cached[0] == (len(df), tuple(map(str, df.columns))):
        return cached[1]
    vocab: set[str] = set()
    core: set[str] = set()
    for c in df.columns:
        names = {t for t in re.split(r"[^a-z0-9]+", semantics._words(str(c)).lower()) if t}  # noqa: SLF001
        names |= {t for t in re.split(r"[^a-z0-9]+", str(c).lower()) if t}
        vocab |= names
        core |= names
        if kind_of(df[c]) == "categorical":
            vals = df[c].dropna().astype(str).unique()
            words = {t for v in vals[:20000] for t in re.split(r"[^a-z0-9]+", v.lower()) if t}
            vocab |= words
            if len(vals) <= 50:
                core |= words
    try:
        df.attrs["_silt_vocabulary"] = ((len(df), tuple(map(str, df.columns))), (vocab, core))
    except Exception:  # noqa: BLE001
        pass
    return vocab, core


def _title_case(message: str) -> bool:
    """Nearly every longer word is capitalised ("Show Total Amount By Region", "TOTAL SALES"): the capitals are a
    writing style, not names."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]+", message) if len(w) > 3]
    return len(words) >= 3 and sum(w[0].isupper() for w in words) >= 0.8 * len(words)


def unmatched_names(message: str, df: pd.DataFrame) -> list[str]:
    """Names the question gives ('for Zed Zorro', 'in Atlantis') that occur nowhere in the data -- not as a column,
    not as (part of) any value. Such a question is about records that do not exist; it is never answered over all the
    records instead. Only capitalised names written inside a sentence are read this way, so ordinary words are not."""
    if _title_case(message):
        return []                                       # Title Case Or ALL CAPS: capitals say nothing here
    out, vocab, core = [], None, set()
    for m in _NAMED.finditer(message):
        toks = [t for t in re.split(r"[^A-Za-z0-9]+", re.sub(r"['’]s\b", "", m.group(1))) if len(t) > 1]
        toks = [t for t in toks if not (t.isupper() and len(t) <= 5)]           # acronyms: KPI, ROI, USA
        if not toks:
            continue
        if vocab is None:
            vocab, core = _vocabulary(df)
        lows = [t.lower() for t in toks]

        def known(t):
            return (t in vocab or t.rstrip("s") in vocab or t in _KNOWN_NAMES or t + "s" in vocab
                    or (len(t) >= 5 and any(_same_stem(t, v) for v in core if len(v) >= 5)))    # "Engineers"
        if all(known(t) for t in lows):
            continue                    # every word of it is in the data (or is a calendar / method word)
        name = " ".join(toks)
        if name not in out:
            out.append(name)
    return out


def names_in_long_columns(message: str, df: pd.DataFrame, already: list) -> tuple[list, list]:
    """Capitalised names the question gives that are values of a text column too long to scan word by word (4,000
    company names): ([(column, value)], [(name, [candidate values])]). A name equal to one value -- or contained, as
    whole words, in exactly one -- selects that value; a name several values contain is returned for Silt to ask
    about. Without this, "average rating for Infosys" would be answered over every company."""
    found, unsure = [], []
    big = [c for c in df.columns if kind_of(df[c]) == "categorical" and df[c].nunique() > 50]
    if not big:
        return found, unsure
    # a name that is a value of a short column ("Africa" the continent) is that value -- read by the scope as before
    # -- and never also a part of some long column's value ("South Africa" the country)
    taken = {str(v).lower() for _, v in already}
    for c in df.columns:
        if kind_of(df[c]) == "categorical" and c not in big:
            taken |= {str(v).strip().lower() for v in df[c].dropna().unique()}
    if _title_case(message):
        return found, unsure
    for m in _NAMED.finditer(message):
        name = re.sub(r"['’]s$", "", m.group(1).strip())
        low = name.lower()
        if low in taken or all(t.lower() in _KNOWN_NAMES for t in re.findall(r"[A-Za-z0-9]+", name)):
            continue
        if any(re.sub(r"[^a-z0-9]+", " ", semantics._words(str(c)).lower()).strip() == low for c in df.columns):  # noqa: SLF001
            continue                                    # a column's own name, capitalised
        for c in big:
            vals = df[c].dropna().astype(str)
            exact = vals[vals.str.strip().str.lower() == low]
            if len(exact):
                found.append((c, str(exact.iloc[0]).strip()))
                break
            part = vals[vals.str.contains(rf"(?<![\w-]){re.escape(name)}(?![\w-])", case=False, regex=True)].unique()
            if len(part) == 1:
                found.append((c, str(part[0])))
                break
            if len(part) > 1:
                unsure.append((name, [str(x) for x in part[:6]]))
                break
    return found, unsure


def _stated_filters(message, df, roles, r, named, ms, low):
    """Everything the question says about WHICH records it means, beyond named values and periods; returns the
    measures with the columns that turned out to be filters taken out."""
    _stated_conditions(message, df, roles, r)
    cond_cols = {c["column"] for c in r.conditions}
    if cond_cols:
        rest = [c for c in ms if c not in cond_cols]
        if rest or r.quantity == "count":
            ms = rest                                   # "total hours for tickets with csat below 3": csat filters
    if not (_HYPO.search(message) or re.search(r"\b(what if|forecast\w*|predict\w*)\b", low)):
        # (the values given to a prediction or a scenario are its inputs, not a selection of records)
        ms = _stated_flags(message, df, roles, r, named, ms, low)
    _stated_share(message, df, roles, r, low)
    if r.share_of:
        ms = [r.share_of]
        r.quantity = "rate"
    r.unmatched = unmatched_names(message, df)
    r._long_names = names_in_long_columns(message, df, []) if not r.unmatched else ([], [])
    return ms


_SUFFIXES = ("ations", "ation", "itions", "ition", "ions", "ion", "ments", "ment", "ings", "ing", "als", "al", "ed",
             "es", "s", "e")


def lemma(w: str) -> str:
    """cancellation / cancelled / cancel -> cancel; returned / returns -> return; escalation / escalated -> escalat."""
    w = w.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 4 and w[-1] == w[-2] else w


_RELATION = re.compile(r"\b(correlat\w*|relat(?:ed|es|ionship)|associat\w*|linked|link between|go(?:es)? (?:up |down )?"
                       r"(?:together|with)|move together|track|cause[sd]?|causing|drives?|driving|leads? to|"
                       r"results? in|affects?|impacts?|influences?)\b", re.I)
_CAUSAL = re.compile(r"\b(cause[sd]?|causing|because|drives?|driving|leads? to|results? in|makes?|affects?|impacts?|"
                     r"influences?)\b", re.I)


def _first_mention(low: str, col) -> int:
    words = [w for w in re.split(r"[^a-z0-9]+", semantics._words(str(col)).lower()) if w]  # noqa: SLF001
    pos = [low.find(w[:5]) for w in words if low.find(w[:5]) >= 0]
    return min(pos) if pos else 10 ** 6


def _is_record_noun(noun: str, df: pd.DataFrame, roles: dict) -> bool:
    """'how many shifts' on a table with one shift_id per record, or 'how many records / rows': the records."""
    from app.analyst.schema_profile import _tokens as st
    w = noun.lower()
    if w in ("records", "rows", "entries", "lines", "observations"):
        return True
    forms = {w, w.rstrip("s"), re.sub(r"ies$", "y", w), w[:-2] if w.endswith("es") else w}
    for c in df.columns:
        t = st(c)
        if t and t[0] in forms and roles[c]["role"] == "identifier":
            v = df[c].dropna()
            if len(v) and v.nunique() >= 0.95 * len(v):
                return True
    return False


def count_target(noun: str, df: pd.DataFrame, roles: dict, distinct: bool = False):
    """What "how many <noun>" counts: the distinct values of the column naming those things (customers ->
    customer_id when customers repeat across records; agents -> agent), or None for the records themselves (tickets,
    shifts: one record each -- counted as records, duplicates disclosed by the duplicate policy)."""
    from app.analyst.schema_profile import _tokens as st, entity_column
    w = noun.lower()
    if w in ("records", "rows", "entries", "lines", "observations"):
        return None
    forms = {w, w.rstrip("s"), re.sub(r"ies$", "y", w), w[:-2] if w.endswith("es") else w}
    cands = []
    ent = entity_column(df, w)
    if ent is not None:
        cands.append(ent)
    for c in df.columns:
        t = st(c)
        if t and t[0] in forms and set(t) <= forms | {"id", "no", "number", "code", "key", "ref", "name"} \
                and kind_of(df[c]) != "datetime" and roles[c]["role"] not in ("year",):
            cands.append(c)
    cands = list(dict.fromkeys(cands))
    if not cands:
        return None
    c = cands[0]
    v = df[c].dropna()
    if not distinct and len(v) and v.nunique() >= 0.98 * len(v) and roles[c]["role"] == "identifier":
        return None                     # one value per record: the records are what is counted
    return c


def _event(message, df, roles, r, low):
    """'cancellation rate', 'how often are orders returned', 'what share of orders were returned': the share of
    records with one value of a category column -- found by the word's stem against the column's values. A value
    the question names outright is taken as the event only when nothing else is measured and it is the last value
    named (the earlier ones say which records)."""
    cands = []
    words = [w for w in re.findall(r"[a-z]+", low) if len(w) > 4]
    for c in df.columns:
        if roles[c]["role"] not in ("category", "flag") or df[c].nunique() > 30 or c == r.group:
            continue
        for v in df[c].dropna().astype(str).unique():
            lv = lemma(v) if re.fullmatch(r"[A-Za-z]+", v) else None
            if lv and len(lv) >= 4 and any(lemma(w) == lv for w in words):
                cands.append((c, v))
    named = [(c, v) for c, v in r.entities if roles[c]["role"] in ("category", "flag") and c != r.group]
    if not cands and named and len(named) >= 1:
        last = max(named, key=lambda cv: low.rfind(str(cv[1]).lower()))
        if len(named) == 1 or len({c for c, _ in named}) > 1:
            cands = [last]
    cands = list(dict.fromkeys(cands))
    if len(cands) > 1:
        # "share of orders cancelled in the North": a value after "in / for / from / among ..." says which records;
        # the other one ("cancelled") is the event
        def filtered(v):
            return re.search(rf"\b(?:in|for|from|at|among|within|of|across)\s+(?:the\s+)?{re.escape(str(v).lower())}\b", low)
        left = [cv for cv in cands if not filtered(cv[1])]
        if len(left) == 1:
            cands = left
    if len(cands) == 1:
        r.event = cands[0]
        r.entities = [e for e in r.entities if e != cands[0]]


_CLAUSE = re.compile(r",?\s+and\s+(?=(?:what|how|which|who|where|when|is|are|does|do|show|give|list|tell)\b)", re.I)
_CLAUSE_QTY = re.compile(r",?\s+and\s+(?=(?:also\s+)?(?:the\s+)?(?:average|mean|median|total|sum|number|count|share|"
                         r"rate|percentage|minimum|maximum)\s+(?:of\s+)?[a-z])", re.I)


def clauses(message: str, quantity_split: bool = False) -> list[str]:
    """A message asking two things ("How many orders per region and what is the typical order size?"), as its
    clauses -- split only where 'and' starts a new question, so "total amount for East and West" stays whole. With
    quantity_split, also where 'and' starts a new quantity ("... for East and the average units for West"); that
    split is used only when the parts are about different records, never for a list of measures on one table."""
    rx = _CLAUSE_QTY if quantity_split else _CLAUSE
    parts = [p.strip(" ,?.") for p in rx.split(message) if p and p.strip(" ,?.")]
    if len(parts) >= 2 and len(parts[-1].split()) < 2:
        return [message]
    return parts if len(parts) >= 2 else [message]


def separate_scopes(message: str, df: pd.DataFrame) -> list[str] | None:
    """The clauses of a message whose parts are about DIFFERENT records ("total amount for East and the average
    units for West"): each must be computed on its own records, never on all of them together. None otherwise."""
    def scopes_of(reads):
        return [frozenset(r.entities) | frozenset(("not",) + e for e in r.excluded) for r in reads]
    parts = clauses(message)
    if len(parts) >= 2:
        reads = [read(p, df) for p in parts]
        if all(r.quantity or r.measures or r.counted or r.event for r in reads):
            asks = [(r.quantity, tuple(r.measures), r.counted, r.event, r.record_noun) for r in reads]
            if len(set(scopes_of(reads))) >= 2 or len(set(asks)) >= 2:
                return parts            # two questions ("how many ... and what is ..."): each answered
    parts = clauses(message, quantity_split=True)
    if len(parts) >= 2:
        reads = [read(p, df) for p in parts]
        named = [bool(r.entities or r.excluded) for r in reads]
        if all(r.quantity or r.measures for r in reads) and all(named) and len(set(scopes_of(reads))) >= 2:
            return parts                # "total amount for East and the average units for West"
    return None


def stated_filters_text(r: "Request") -> list[str]:
    """The stated restrictions, as the user would recognise them."""
    out = [f"{c['column']} {c['op']} {c['value']:g}" for c in r.conditions]
    out += [f"only records where {c} applies" for c in r.restrict]
    out += [f"only records where {c} does not apply" for c in r.restrict_not]
    out += [f"“{t}” (no column found for it)" for t in r.unresolved]
    return out


_NUMW = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "twelve": 12, "fourteen": 14, "thirty": 30, "sixty": 60, "ninety": 90}
_LASTN = re.compile(r"\b(?:last|past|previous|recent)\s+(\d+|" + "|".join(_NUMW) + r")\s+(day|week|month|year)s?\b", re.I)
_THIS = re.compile(r"\b(this|current|last|previous)\s+(week|month|quarter|year)\b|\b(year|month|quarter)[- ]to[- ]date\b|"
                   r"\b(ytd|mtd|qtd)\b", re.I)


def relative_period(message: str, df: pd.DataFrame) -> dict | None:
    """'the last 30 days', 'this quarter', 'last month', 'year to date' as {start, end, label}, counted back from
    today -- or from the last record when the data ends well before today (said in the label). None if the message
    names no relative period or the data has no date column.

    The date-window convention, the same on every path (program select, shared scope mask, calculator rule, verifier):
      * "today" is the anchor date: today's date, or the last recorded date when the data ends more than 31 days
        before today (nl_planner.anchor_date); every answer that uses a window says which.
      * "last / past N days" = N calendar days ENDING ON and INCLUDING the anchor date: anchor - (N-1) .. anchor.
        "the last 7 days" on 2026-11-17 is 2026-11-11 .. 2026-11-17 (7 days).
      * "last / past N weeks" = 7N calendar days including the anchor; "last N months / years" = from the same day
        N months / years earlier plus one day, to the anchor.
      * "this week / month / quarter / year" = its first day .. the anchor; "last / previous month" (no number) = the
        whole previous calendar month (likewise week, quarter, year).
      * Both ends are included, and records are compared by calendar day: a timestamp later on the end day is inside.
    """
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


_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = (rf"(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MON}(?:,?\s+\d{{4}})?|{_MON}\s+\d{{1,2}}(?:st|nd|rd|th)?"
         rf"(?:,?\s+\d{{4}})?|\d{{1,2}}/\d{{1,2}}/\d{{4}})")
_RANGE = re.compile(rf"\b(?:between|from)\s+(?:the\s+)?({_DATE})\s+(?:and|to|until|till|through|-)\s+(?:the\s+)?({_DATE})", re.I)
_OPEN = re.compile(rf"\b(since|after|from|starting|before|until|till|up to|through)\s+(?:the\s+)?({_DATE})", re.I)


def _parse_date(text: str, year_hint: int | None):
    """A date written in the question, or (None, why) when it cannot be read without guessing."""
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip(), flags=re.I)
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 12 and b <= 12 and a != b:
            return None, f"“{text}” could be day/month or month/day"
        d, mo = (a, b) if a > 12 else (b, a)
        return pd.Timestamp(year=int(m.group(3)), month=mo, day=d), None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:                               # ISO: year-month-day, never read day-first
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3))), None
        except ValueError:
            return None, f"“{text}” is not a valid date"
    if not re.search(r"\d{4}", t):
        if year_hint is None:
            return None, f"“{text}” has no year"
        t = f"{t} {year_hint}"
    try:
        # only written-out months reach here ("1 August 2026", "Aug 1, 2026"): unambiguous either way
        return pd.Timestamp(pd.to_datetime(t)).normalize(), None
    except (ValueError, TypeError):
        return None, f"“{text}” is not a date Silt can read"


def explicit_period(message: str, df: pd.DataFrame) -> dict | None:
    """'between 1 August 2026 and 15 August 2026', 'from Aug 1 to Aug 15, 2026', 'since 1 October 2026', 'before
    2026-06-01': a date range, both ends included ('before' / 'after' exclude the date itself). A date that cannot be
    read without guessing (01/02/2026) is returned as {"unreadable": why} -- applied by nobody, asked about."""
    if not any(kind_of(df[c]) == "datetime" for c in df.columns):
        return None
    years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", message)]
    hint = years[-1] if years else None
    m = _RANGE.search(message)
    if m:
        a, wa = _parse_date(m.group(1), hint)
        b, wb = _parse_date(m.group(2), hint)
        if a is None or b is None:
            return {"unreadable": wa or wb}
        if b < a:
            return {"unreadable": f"the range ends ({b.date()}) before it starts ({a.date()})"}
        return {"start": str(a.date()), "end": str(b.date()), "label": f"{a.date()} to {b.date()} (both days included)"}
    m = _OPEN.search(message)
    if m:
        d, why = _parse_date(m.group(2), hint)
        if d is None:
            return {"unreadable": why}
        w = m.group(1).lower()
        if w in ("since", "from", "starting"):
            return {"start": str(d.date()), "end": None, "label": f"from {d.date()} (included)"}
        if w == "after":
            d2 = d + pd.Timedelta(days=1)
            return {"start": str(d2.date()), "end": None, "label": f"after {d.date()} (from {d2.date()})"}
        if w == "before":
            d2 = d - pd.Timedelta(days=1)
            return {"start": None, "end": str(d2.date()), "label": f"before {d.date()} (to {d2.date()})"}
        return {"start": None, "end": str(d.date()), "label": f"up to {d.date()} (included)"}
    return None


def _same_stem(a: str, b: str) -> bool:
    """escalation / escalated, cancellations / cancelled: a long shared beginning."""
    import os
    n = len(os.path.commonprefix([a, b]))
    # (a long shared beginning: "readmi" for readmission / readmitted)
    return (n >= 5 and n >= 0.7 * min(len(a), len(b))) or (n >= 6 and n >= 0.6 * min(len(a), len(b)))


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


def ranking_aggregate(message: str, measure) -> str | None:
    """How a question that ranks groups by `measure` compares them: "max"/"min" (each group's single record --
    "largest individual transaction"), "median", "mean" ("highest average revenue"), "sum" ("highest revenue region",
    revenue being a flow), or None when neither the question nor the measure says (Silt asks)."""
    if _SINGLE.search(message):
        return "min" if _LOW.search(message) else "max"
    if _MEDIAN.search(message):
        return "median"
    if _AVG.search(message):
        return "mean"
    if _TOTAL.search(message) or re.search(r"\boverall\b", message, re.I):
        return "sum"
    flow, level = bool(_FLOW_NAME.search(str(measure))), bool(_LEVEL_NAME.search(str(measure)))
    if flow != level:
        return "sum" if flow else "mean"
    return None


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
    if req.extreme in ("min", "max", "either") and req.group and req.quantity is None and len(req.measures) == 1 \
            and not ratios and fns and fns <= {"sum", "mean", "median"}:
        # "the highest revenue region" compares the regions' total revenue; "the highest unit price region" their average
        want = ranking_aggregate(message, req.measures[0])
        if want in ("sum", "mean", "median") and want not in fns:
            problems.append(f"the request ranks each {req.group} by its {dict(sum='total', mean='average', median='median')[want]} "
                            f"{req.measures[0]}; aggregate with fn {want}, not {', '.join(sorted(fns))}")
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
    selects = [s for s in steps if s["op"] == "select"]
    for c in req.restrict_not:
        if not any(c in (s.get("entities") or {}) or c in (s.get("except") or {})
                   or any(w.get("column") == c for w in s.get("where") or []) for s in selects):
            problems.append(f"the request is limited to records where {c} does not apply; select them")
    for cd in req.conditions:
        def same(w, cd=cd):
            try:
                return w.get("column") == cd["column"] and w.get("op") == cd["op"] and \
                    abs(float(w.get("value")) - cd["value"]) <= 1e-9 * max(1.0, abs(cd["value"]))
            except (TypeError, ValueError):
                return False
        if not any(same(w) for s in selects for w in s.get("where") or []):
            problems.append(f"the request is limited to records where {cd['column']} {cd['op']} {cd['value']:g}; "
                            "select them")
    for t in req.unresolved:
        problems.append(f"the request states a limit (“{t}”) that could not be tied to a column, so it was not applied")
    if req.event and not req.measures:
        ec, ev = req.event
        hit = any(s["op"] == "derive" and ec in str(s.get("formula")) and str(ev).lower() in str(s.get("formula")).lower()
                  for s in steps) or (ratios and any(str(ev).lower() in [str(x).lower() for x in vals] for s in steps
                                                     for vals in (s.get("entities") or {}).values()))
        if not hit:
            problems.append(f"the request asks for the share of records where {ec} = {ev}; derive "
                            f"'{ec} == '{ev}'' and aggregate its mean")
    if req.share_of and not any(by.get(r_["a"], {}).get("column") == req.share_of
                                and by.get(r_["b"], {}).get("column") == req.share_of for r_ in ratios):
        problems.append(f"the request asks for a share of {req.share_of} (its total in the records named ÷ its total "
                        "overall), not a share of records")
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
    grouped = {s.get("by") for s in aggs if s.get("by")}
    named_all = {c for c in grouped if c in df.columns and {str(x).lower() for x in df[c].dropna().astype(str).unique()}
                 <= {str(v).lower() for c2, v in req.entities if c2 == c}}
    for c, v in req.entities:
        if c == req.group or c in named_all:
            continue                    # every value of a column the result is broken down by: shown, not selected
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
    if req.period and req.period.get("unreadable"):
        problems.append(f"the request names a period Silt cannot read: {req.period['unreadable']}")
    elif req.period and not any(s["op"] == "select" and (s.get("period") or {}).get("start") == req.period["start"]
                                and (s.get("period") or {}).get("end") == req.period["end"] for s in steps):
        problems.append(f"the request is limited to {req.period['label']}: select period start {req.period['start']} "
                        f"end {req.period['end']}")
    if message:
        # a numeric condition the question does not state ("Revenue >= 0" when nobody asked to drop refunds) is the
        # model's own exclusion rule: records are kept unless the request filters them (one policy, see
        # metrics.never_negative); a threshold the question states, or a sign it names, is the user's own
        from app.analyst.calc import stated_numbers
        stated = stated_numbers(message)
        signed = re.search(r"\b(negative|positive|non-?negative|below zero|above zero|zero|refunds?|losses|loss-?making|"
                           r"profitable|credits?)\b", message, re.I)
        for st in steps:
            if st["op"] != "select":
                continue
            for w in st.get("where") or []:
                v = w.get("value")
                if isinstance(v, (int, float)) and not isinstance(v, bool) and w.get("op") in (">", ">=", "<", "<=") \
                        and float(v) not in stated and not (float(v) == 0 and signed):
                    problems.append(f"the request sets no condition {w.get('column')} {w.get('op')} {v}; the program "
                                    "must not filter records the question does not ask to filter")
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
def rule_program(req: Request, df: pd.DataFrame, prof: dict, plain: bool = False) -> list[dict] | None:
    """A program for a request whose parts are all grounded (one measure, a quantity; a grouping, or a selection of
    named entities / a restriction / a period), built without a model -- used before the model when it can answer,
    and as the fallback when the model is off or its plans fail. None when anything would be a guess."""
    scoped = bool(req.entities or req.excluded or req.restrict or req.period or req.conditions or req.restrict_not)
    if req.unresolved or req.unmatched or req.share_word or req.ambiguous_names or (req.period or {}).get("unreadable"):
        return None                                     # a stated limit nothing could apply: never a program without it
    counting = req.quantity == "count" and not req.measures and not req.per and not req.relation
    event = req.event if (req.quantity == "rate" and not req.measures and req.event) else None
    if (len(req.measures) != 1 and not counting and not event) or req.hypothetical or req.unknown_group or \
            req.relation or (req.group is None and not scoped and not (counting and (req.counted or req.record_noun))
                             and not event and not (plain and req.quantity and len(req.measures) == 1)):
        return None
    m = req.measures[0] if req.measures else None
    role = prof["roles"][m]["role"] if m is not None else None
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
    where = []
    for c, yes in [(c, True) for c in req.restrict] + [(c, False) for c in req.restrict_not]:
        pair = flag_values(df, c, prof["roles"][c])
        if pair is None:
            return None
        where.append({"column": c, "op": "==", "value": pair[0] if yes else pair[1]})
    where += [{"column": cd["column"], "op": cd["op"], "value": cd["value"]} for cd in req.conditions]
    if where:
        sel["where"] = where
    period = None
    if req.period:
        if len(prof["dates"]) != 1:
            return None
        period = {"start": req.period["start"], "end": req.period["end"]}
    if req.share_of:
        return _share_program(req, df, sel, period)
    if period:
        sel["period"] = period
    if sel:
        steps.append({"id": "records", "op": "select", "from": "data", **sel})
        base = "records"
    src, col = base, m
    if event:
        ec, ev = event
        val = f"'{ev}'" if kind_of(df[ec]) != "numeric" else ev
        steps.append({"id": "flag", "op": "derive", "from": base, "name": f"is_{re.sub(r'[^0-9a-zA-Z_]', '_', str(ev)).lower()}",
                      "formula": f"{ec} == {val}"})
        steps.append({"id": "result", "op": "aggregate", "from": "flag", "fn": "mean", "column": steps[-1]["name"],
                      **({"by": req.group} if req.group else {})})
        return steps if not check(steps, req, df) else None
    if counting:
        fn = "nunique" if req.counted else "count"
        steps.append({"id": "result", "op": "aggregate", "from": base, "fn": fn,
                      **({"column": req.counted} if req.counted else {}), **({"by": req.group} if req.group else {})})
        return steps if not check(steps, req, df) else None
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
        # per customer / per person: the distinct ones (an identifier is counted, never added up); per km / per hour: a sum
        dfn = "sum" if kind_of(df[d]) == "numeric" and prof["roles"][d]["role"] not in ("identifier", "person_name") \
            else "nunique"
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


def _share_program(req: Request, df: pd.DataFrame, sel: dict, period: dict | None) -> list[dict] | None:
    """A share of a measure's total: the measure summed over the records the question names (or per group), divided
    by its total over all the records of the same period."""
    m = req.share_of
    steps, whole = [], "data"
    if period:
        steps.append({"id": "period", "op": "select", "from": "data", "period": period})
        whole = "period"
    part = whole
    if sel:
        steps.append({"id": "records", "op": "select", "from": whole, **sel})
        part = "records"
    elif not req.group:
        return None                                     # nothing names the part: the share would be 100%
    by = {"by": req.group} if req.group else {}
    steps += [{"id": "numerator", "op": "aggregate", "from": part, "fn": "sum", "column": m, **by},
              {"id": "denominator", "op": "aggregate", "from": whole, "fn": "sum", "column": m},
              {"id": "per_unit", "op": "ratio", "a": "numerator", "b": "denominator", "broadcast": True}]
    return steps if not check(steps, req, df) else None
