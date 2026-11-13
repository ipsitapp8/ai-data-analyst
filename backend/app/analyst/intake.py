"""Is the message a question at all? Checked before any planning, calculation or prediction.

Deliberately conservative: only text that is clearly not language is stopped -- empty input, punctuation only, or
words that cannot be words (no vowels, keyboard runs such as "asdf", impossible consonant runs, one letter repeated,
or two or more letter pairs that real words practically never contain, as in random typing like "idqjnfiW").
Short questions, misspellings ("whcih departmnet"), abbreviations and unfamiliar business terms all pass, because the
test is on the shape of each word, not on a dictionary. A word that matches a column or a value in the data is never
treated as noise, whatever it looks like (product codes, IDs).

Three outcomes:
  None                -> a question; analysis proceeds as before.
  {"kind": "unintelligible"} -> nothing readable: ask the user to rephrase; nothing is computed.
  {"kind": "unknown_term", "terms": [...]} -> a real question built around a word that is not a word and is not in
                         the data ("total of cdfvgbhnjmk"): say so and offer the real columns; nothing is computed.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
# Letter pairs that occur in fewer than 10 of the ~236,000 words of a standard English word list (macOS
# /usr/share/dict/words, measured once and fixed here so nothing depends on that file at run time). One such pair
# happens in real words, names and acronyms (fjord, Qatar, json, sql, dvd); two or more in one word practically never
# do, but they are typical of random typing (idqjnfiW: dq, qj, jn).
_RARE_PAIRS = frozenset("""
bq bx bz cb cf cg cj cv cx dq dx fj fk fq fv fx fz gj gq gv gx gz hj hq hx hz jb jc jd jf jg jh jj jk jl jm jn jp jq jr
js jt jv jw jx jy jz kq kv kx kz lj lq lx mj mq mx mz pq pv px pz qa qb qc qd qe qf qg qh qi qj qk ql qm qn qo qp qq qr
qs qt qv qw qx qy qz rx sx tq tx vb vc vd vf vg vh vj vk vl vm vn vp vq vs vt vw vx vz wj wq wv wx xg xj xk xn xr xv xx
xz yj yq yy zc zf zg zh zj zk zm zn zp zq zr zs zv zw zx""".split())


def rare_pairs(word: str) -> int:
    w = word.lower()
    return sum(w[i:i + 2] in _RARE_PAIRS for i in range(len(w) - 1))
_VOWELS = set("aeiouy")


def _keyboard_run(word: str) -> bool:
    """Keys typed along one keyboard row, either direction: the whole word ("asdf", "lkjh", "qwer"), or a run of five
    or more inside a longer one ("sdfghjkl"). A four-key run inside a word is ordinary English ("property": erty)."""
    for row in _ROWS:
        for seq in (row, row[::-1]):
            if len(word) >= 4 and word in seq:
                return True
            for i in range(len(seq) - 4):
                if seq[i:i + 5] in word:
                    return True
    return False


def _longest_consonant_run(word: str) -> int:
    return max((len(m) for m in re.findall(r"[^aeiouy]+", word)), default=0)


def noise_word(word: str) -> bool:
    """A token that cannot be a word in any natural language written in Latin letters."""
    w = word.lower()
    if len(w) < 4 or not w.isalpha() or (word.isupper() and len(word) <= 6):    # short acronyms: HTTPS, SKU, CRM
        return False
    if not w.isascii():
        return False          # a word with letters outside English (försäljning, résumé) is judged by no English rule
    if re.search(r"(.)\1{3,}", w):                              # "aaaa", "zzzzzz"
        return True
    if len(w) >= 5 and not (_VOWELS & set(w)):                  # "cdfvgbhnjmk", "fjdksl"
        return True
    if _longest_consonant_run(w) >= 6:                          # English tops out at 5 ("lengths")
        return True
    if rare_pairs(w) >= 2:                                      # "idqjnfiW", "ajsdhakjsdh": vowels, but no word shape
        return True
    return _keyboard_run(w)                                     # "asdf", "qwer", "zxcv", "hjkl"


def _schema_words(df: pd.DataFrame | None) -> set[str]:
    if df is None:
        return set()
    out: set[str] = set()
    for c in df.columns:
        out |= {w for w in re.split(r"[^a-z0-9]+", str(c).lower()) if w}
        s = df[c]
        if s.dtype == object and s.nunique(dropna=True) <= 500:
            for v in s.dropna().astype(str).unique():
                out |= {w for w in re.split(r"[^a-z0-9]+", v.lower()) if w}
    return out


def check(message: str, df: pd.DataFrame | None = None) -> dict | None:
    text = (message or "").strip()
    if not re.search(r"[^\W_]", text):
        return {"kind": "unintelligible", "why": "empty" if not text else "no letters or numbers"}
    words = [w for w in re.findall(r"[^\W\d_]+", text) if len(w) >= 2]
    if not words:
        if re.search(r"\d", text):
            return None       # a number on its own ("10%", "2025") can answer a clarifying question
        return {"kind": "unintelligible", "why": "no words"}
    known = _schema_words(df)
    noise = [w for w in words if w.lower() not in known and noise_word(w)]
    if not noise:
        return None
    real = [w for w in words if w not in noise]
    # Mostly noise, or nothing longer than a couple of letters besides it: there is no question to answer.
    if len(noise) >= len(real) or not any(len(w) >= 3 for w in real):
        return {"kind": "unintelligible", "why": "no recognisable words", "terms": noise}
    return {"kind": "unknown_term", "terms": list(dict.fromkeys(noise))}


def reply(result: dict, df: pd.DataFrame | None, filename: str | None = None) -> str:
    if result["kind"] == "unintelligible":
        return ("I couldn't understand that. Please ask a valid question about your dataset or document."
                + (" (The message was empty.)" if result.get("why") == "empty" else "")
                + "\n\nYou could rephrase it, for example: *What does this dataset contain?*, *Which group has the "
                "highest total?* or *Show the trend over time.* Nothing was calculated.")
    terms = result["terms"]
    cols = [str(c) for c in (df.columns if df is not None else [])]
    near = []
    for t in terms:
        near += difflib.get_close_matches(t.lower(), [c.lower() for c in cols], n=2, cutoff=0.6)
    listed = ", ".join(f"`{c}`" for c in cols[:15]) + (" …" if len(cols) > 15 else "")
    return (f"I couldn't find **{', '.join(terms)}** in {('**' + filename + '**') if filename else 'this data'} — it isn't "
            "a column or a value there, so I haven't calculated anything with it."
            + (f" Did you mean {', '.join(f'`{n}`' for n in dict.fromkeys(near))}?" if near else "")
            + (f"\n\nThe columns are: {listed}. Which one did you mean?" if cols else ""))
