"""Conversation state: one analytical task carried across turns.

The orchestrator answers one message at a time; this module is what makes a sequence of messages one conversation.
It keeps a structured task in the conversation's memory (memory["task"]):

  original     the request being worked on, as the user would have asked it in full
  status       "awaiting" (a clarification is open) | "done" (answered) | "new"
  pending      the open clarification: the exact question Silt asked, the options it offered, what kind of answer
               is expected ("choice" | "number" | "period" | "yes_no" | "free"), the action a "yes" would take,
               and how many times it has been asked
  offers       follow-up actions the last answer offered ("exclude duplicates", "treat them as overnight shifts")
  turn         a counter, so stale state can be told apart

Every new message is first read against that state (`resolve`), deterministically -- no model call, so a model outage
cannot lose a pending task:

  confirmation ("yes", "okay do it", "go ahead") -> continue the pending task with its only/proposed option, or
                take the single offered action; if several are possible, ask which -- once, concisely
  refusal / "the other one"                      -> the alternative option of a two-way choice
  answer to the open question                    -> merged into the original request, which is then run
  correction ("actually, use Aarav instead")     -> applied to the previous task
  topic change ("forget that, ...")              -> the old task and its filters are dropped
  anything else                                  -> a new request (the task is replaced when it is answered)

The result is an *effective message*: the original request with the answer folded in, which goes through the same
analysis path as any question, so the executed plan is the one the user would have got by asking in full.
Nothing here knows a dataset: options come from the clarification, values from the data.
"""
from __future__ import annotations

import re

import pandas as pd

_CONFIRM_WORDS = {"yes", "yeah", "yep", "yup", "y", "ok", "okay", "okey", "k", "sure", "go", "ahead", "proceed",
                  "continue", "please", "do", "it", "that", "this", "show", "me", "calculate", "compute", "run", "on",
                  "sounds", "good", "correct", "right", "fine", "alright", "great", "lets", "let's", "us", "then",
                  "now", "thanks", "thank", "you", "absolutely", "definitely", "of", "course", "affirmative", "carry",
                  "and", "just", "that's", "thats", "one", "the", "same", "agreed", "perfect", "cool"}
_CONFIRM_KEYS = {"yes", "yeah", "yep", "yup", "y", "ok", "okay", "okey", "sure", "proceed", "continue", "go", "do",
                 "show", "calculate", "compute", "run", "correct", "right", "fine", "alright", "absolutely",
                 "definitely", "affirmative", "agreed", "perfect", "cool", "carry"}
_NEGATE = re.compile(r"^\W*(?:no|nope|nah|not (?:that|this|it)|wrong|neither)\b", re.I)
_OTHER = re.compile(r"\b(?:the )?other (?:one|column|option|value)\b|\bthe second\b|\bthe first\b", re.I)
_TOPIC = re.compile(r"^\W*(?:forget (?:that|it|this|about (?:that|it))|never ?mind|scratch that|start over|let'?s start "
                    r"over|new (?:question|task|topic)|change of topic|something else|different question)\b[\s,.;:!-]*",
                    re.I)
_CORRECT = re.compile(r"^\W*(?:actually|sorry|oops|correction|rather|wait)\b[\s,.;:!-]*|^\W*no[\s,.;:!-]+(?=i meant|use|"
                      r"for|make it|switch|change|try)|^\W*(?:i meant|i mean)\b[\s,.;:!-]*", re.I)
_QUESTION = re.compile(r"\b(what|which|how|why|who|when|where|show|list|give|compare|predict|forecast|estimate|plot|"
                       r"chart|calculate|compute|total|average|sum|count|explain|describe|summari[sz]e|train|rank|top)\b",
                       re.I)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def is_confirmation(message: str) -> bool:
    w = _words(message)
    return bool(w) and len(w) <= 7 and set(w) <= _CONFIRM_WORDS and bool(set(w) & _CONFIRM_KEYS)


def classify(message: str) -> str:
    if _TOPIC.search(message):
        return "topic_change"
    if is_confirmation(message):
        return "confirm"
    if _NEGATE.search(message) and (_OTHER.search(message) or len(_words(message)) <= 2):
        return "negate"
    if _CORRECT.search(message) or re.search(r"\binstead\b", message, re.I):
        return "correction"
    return "other"


# ------------------------------------------------------------- recording --
_OPTION_TICKS = re.compile(r"`([^`]+)`")


def _options(out: dict) -> list[str]:
    plan = out.get("plan") or {}
    opts = [str(o) for o in plan.get("options") or []]
    if not opts:
        head = re.split(r"\n\n", out.get("answer", ""), maxsplit=1)[0]
        if re.search(r"\b(which|options?|one of|choose|pick)\b", head, re.I):
            opts = _OPTION_TICKS.findall(head)
    return list(dict.fromkeys(opts))[:8]


def _expects(question: str, options: list[str]) -> tuple[str, str | None]:
    q = question.lower()
    if options or re.search(r"\bwhich\b", q):
        return ("choice", None) if options else ("free", None)     # "which one should I use?" is not a yes/no
    m = re.search(r"\b(?:should i|shall i|do you want me to|would you like me to|want me to|do you want to)\s+([^?]+)\?", q)
    if m:
        return "yes_no", m.group(1).strip()
    if re.search(r"\bthreshold\b|\bbeyond how many\b|\bhow many (hours|minutes|days)\b|\bgive a percentage\b|\bby how much\b",
                 q):
        return "number", None
    if re.search(r"\bwhich (period|date|dates|month|year|week|quarter|day)\b|\bdate range\b|\bwhat period\b", q):
        return "period", None
    return "free", None


_OFFER = re.compile(r"\bask (?:to|for) \*([^*]{3,60})\*", re.I)


def record(memory: dict, effective: str, out: dict) -> None:
    """After a turn: what is open, what was answered, what was offered."""
    task = memory.get("task") or {}
    turn = task.get("turn", 0) + 1
    plan = out.get("plan") or {}
    if out.get("plan_source") == "input_check" and plan.get("type") == "rephrase":
        return                                          # unreadable input changes nothing
    answered = any(c.get("status") == "ok" for c in out.get("calls", []))
    if plan.get("type") == "clarify" and not answered:
        question = re.sub(r"\s+", " ", re.split(r"\n\n", out.get("answer", ""), maxsplit=1)[0]).strip()
        opts = _options(out)
        expects, action = _expects(question, opts)
        same = task.get("status") == "awaiting" and task.get("original") == effective
        # the request as completed so far: after an answer that leads to a further question, it already includes
        # that answer ("Show team totals (profit)"), so the next answer refines it rather than the first version
        original = effective
        memory["task"] = {"original": original, "status": "awaiting", "turn": turn, "offers": [],
                          "pending": {"question": question[:400], "options": opts, "expects": expects, "action": action,
                                      "asked": (task.get("pending") or {}).get("asked", 0) + 1 if same else 1}}
    elif answered:
        offers = list(dict.fromkeys(o.strip() for o in _OFFER.findall(out.get("answer", ""))))
        memory["task"] = {"original": effective, "status": "done", "turn": turn, "pending": None, "offers": offers[:4]}
    else:
        task["turn"] = turn                             # an explanation or a chat reply: the task stays as it was
        memory["task"] = task
    memory.pop("_resolved_from_pending", None)


# ------------------------------------------------------------ resolution --
def _norm(s: str) -> str:
    return re.sub(r"[_\s]+", " ", s.lower()).strip(" .!?`*")


def _match_option(message: str, options: list[str]) -> str | None:
    m = _norm(message)
    exact = [o for o in options if _norm(o) == m]
    if exact:
        return exact[0]
    inside = [o for o in options if re.search(rf"\b{re.escape(_norm(o))}\b", m)]
    if len(inside) == 1:
        return inside[0]
    if len(m) >= 3:
        start = [o for o in options if _norm(o).startswith(m) or m in _norm(o)]
        if len(start) == 1:
            return start[0]
    return None


def _fold(original: str, phrase: str) -> str:
    return original.rstrip(" ?.!") + " " + phrase.strip() + "?"


def _answer_phrase(pending: dict, message: str, df: pd.DataFrame | None, original: str) -> str | None:
    """The user's reply as a phrase to fold into the original request, or None if it does not answer the question."""
    from app.analyst.calc import named_values
    exp, opts = pending["expects"], pending.get("options") or []
    if len(_words(message)) > 4 and re.match(r"^\W*(?:what|which|how|why|who|when|where|show|list|give|compare|predict|"
                                              r"forecast|estimate|plot|chart|calculate|compute|explain|describe|"
                                              r"summari[sz]e|train|rank|is|are|does|do|can|could|tell)\b", message, re.I):
        return None                     # a complete question of its own: a new request, not an answer
    if exp == "choice":
        o = _match_option(message, opts)
        if o is None:
            return None
        if df is not None and o in map(str, df.columns):
            return f"using {o}"
        return f"({o})" if not re.search(r"\b(relative|points?|%)\b", o) else o
    if exp == "number":
        n = re.search(r"-?\d+(?:\.\d+)?\s*%?", message)
        if not n:
            return None
        unit = re.search(r"\b(hours?|minutes?|days?)\b", message + " " + pending["question"], re.I)
        num = n.group(0).strip()
        if "%" in num or re.search(r"percentage|by how much", pending["question"], re.I):
            return f"by {num.rstrip('%')}%"
        return f"beyond {num} {unit.group(1) if unit else ''}".strip()
    if exp == "period":
        if re.search(r"\b(last|this|next|previous|past)\s+\w+|\b(19|20)\d\d\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|"
                     r"nov|dec)\w*\b|\b(today|yesterday|ytd|q[1-4])\b", message, re.I):
            return "for " + message.strip(" ?.!")
        return None
    if exp == "free":
        if df is not None and named_values(df, message):
            return "for " + " and ".join(v for _, v in named_values(df, message))
        if len(_words(message)) <= 4 and not _QUESTION.search(message):
            return message.strip(" ?.!")
    return None


def resolve(message: str, memory: dict, df: pd.DataFrame | None) -> dict | None:
    """How this message relates to the conversation's task. None: an ordinary new request. Otherwise
    {"message": effective request, "note": what was understood} or {"reply": a concise question}."""
    task = memory.get("task") or {}
    kind = classify(message)
    pending = task.get("pending") if task.get("status") == "awaiting" else None
    if kind == "topic_change":
        rest = _TOPIC.sub("", message).strip()
        for k in ("task", "scope", "last_focus"):
            memory.pop(k, None)
        memory["last_analysis"] = "other"
        if not rest:
            return {"reply": "Okay — I've dropped the previous request and its filters. What would you like to look at?",
                    "kind": kind}
        return {"message": rest, "kind": kind, "note": "Started a new task: the previous request and its filters "
                                                      "were dropped."}
    if pending:
        original, opts = task["original"], pending.get("options") or []
        if kind == "confirm":
            if pending["expects"] == "yes_no" and pending.get("action"):
                return _go(memory, pending["action"], f"You confirmed: {pending['action']}.", kind)
            if pending["expects"] == "choice" and len(opts) == 1:
                return _go(memory, _fold(original, _answer_phrase(pending, opts[0], df, original) or opts[0]),
                           f"You confirmed the only option offered ({opts[0]}) for your request “{original}”.", kind)
            return {"reply": _reask(original, pending, f"“{message.strip()}”"), "kind": kind, "keep": True}
        if kind == "negate":
            if pending["expects"] == "yes_no":
                memory["task"] = {**task, "status": "done", "pending": None}
                return {"reply": "Okay, I won't. " + f"Your earlier request was “{original}” — tell me what to change, "
                                 "or ask something new.", "kind": kind}
            if len(opts) == 2 and _OTHER.search(message):
                pick = opts[1] if re.search(r"other|second", message, re.I) else opts[0]
                return _go(memory, _fold(original, _answer_phrase(pending, pick, df, original) or pick),
                           f"Using {pick} for your request “{original}”.", kind)
            return {"reply": _reask(original, pending, f"“{message.strip()}”"), "kind": kind, "keep": True}
        phrase = _answer_phrase(pending, message, df, original)
        if phrase is not None:
            return _go(memory, _fold(original, phrase), f"Continuing your request “{original}” with your answer "
                                                        f"“{message.strip()}”.", "answer")
        # not an answer to the open question: a new request (the pending one is replaced when this is answered)
        return None
    if kind == "confirm":
        offers = task.get("offers") or []
        if len(offers) == 1 and memory.get("last_calc"):
            return _go(memory, offers[0], f"You confirmed the action offered with the last answer: {offers[0]}.", kind)
        if len(offers) > 1:
            return {"reply": "Which of these should I do? " + "; ".join(f"*{o}*" for o in offers) + ".", "kind": kind}
        if task.get("status") == "done":
            return {"reply": f"The last request (“{task['original']}”) is already answered above. What should I do "
                             "next — for example narrow it to someone, break it down by a group, or chart it?",
                    "kind": kind}
        return {"reply": "What would you like me to do? Ask a question about the dataset, for example a total, an "
                         "average or a comparison by group.", "kind": kind}
    if kind == "correction":
        rest = message
        for _ in range(3):
            rest = _CORRECT.sub("", rest).strip()
        rest = re.sub(r"\binstead\b", "", rest, flags=re.I).strip(" ,.!?") or message
        return {"message": rest, "kind": kind, "note": f"Correction applied to the previous request: “{rest}”.",
                "correction": True}
    return None


def _go(memory: dict, effective: str, note: str, kind: str) -> dict:
    memory["_resolved_from_pending"] = True
    return {"message": effective, "note": note, "kind": kind}


def _reask(original: str, pending: dict, said: str) -> str:
    opts = pending.get("options") or []
    if opts:
        return (f"To continue “{original}” I need one choice: " + ", ".join(f"**{o}**" for o in opts)
                + f". ({said} doesn't say which.)")
    return f"To continue “{original}” I still need this: {pending['question']}"


_AGG_WORD = {"mean": "average", "sum": "total", "median": "median", "max": "maximum", "min": "minimum",
             "count": "number of", "nunique": "number of"}


def retarget(rest: str, memory: dict, df: pd.DataFrame | None) -> str | None:
    """"No, I meant hourly rate": the previous request with its measure replaced -- same aggregation, same people or
    groups, same breakdown -- written out in full. None when the correction is not about the measure."""
    from app.analyst import metrics
    from app.analyst.data import find_column, kind_of
    last = memory.get("last_calc")
    if df is None or not last or len(_words(rest)) > 5:
        return None
    nums = [c for c in find_column(df, rest) if kind_of(df[c]) == "numeric"]
    if not nums and not metrics.recognise(rest):
        return None
    res = last["result"]
    primary = res.get("primary") or next(iter(res.get("aggregates") or {}), None)
    func = re.match(r"\s*(\w+)\(", (res.get("aggregates") or {}).get(primary, "") or "")
    word = _AGG_WORD.get(func.group(1).lower(), "total") if func else "total"
    sc = memory.get("scope") or {}
    names = [v for vals in (sc.get("entities") or {}).values() for v in vals]
    q = f"What is the {word} {rest.strip(' .?!')}"
    if names:
        q += " for " + " and ".join(names)
    if res.get("group_by"):
        q += " by " + str(res["group_by"]).replace("_", " ")
    return q + "?"
