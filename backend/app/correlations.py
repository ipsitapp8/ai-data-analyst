"""Strongest relationships between numeric columns (deterministic, no LLM).

Pearson and Spearman are both computed; a pair is only reported when the two
agree in sign and the sample is big enough, so one outlier can't fake a link.
"""
from __future__ import annotations

import pandas as pd

MIN_ROWS = 10
MIN_ABS_R = 0.5
MAX_PAIRS = 10


def _strength(r: float) -> str:
    a = abs(r)
    return "very strong" if a >= 0.9 else "strong" if a >= 0.7 else "moderate"


def top_correlations(df: pd.DataFrame, min_abs: float = MIN_ABS_R, limit: int = MAX_PAIRS) -> list[dict]:
    num = df.select_dtypes("number")
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if len(num) < MIN_ROWS or num.shape[1] < 2:
        return []
    pearson = num.corr(method="pearson")
    spearman = num.corr(method="spearman")
    cols, out = list(num.columns), []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p, s = pearson.at[a, b], spearman.at[a, b]
            if pd.isna(p) or pd.isna(s) or p * s <= 0 or abs(p) < min_abs:
                continue
            n = int(num[[a, b]].dropna().shape[0])
            if n < MIN_ROWS:
                continue
            out.append({"a": str(a), "b": str(b), "pearson": round(float(p), 3),
                        "spearman": round(float(s), 3), "n": n,
                        "direction": "positive" if p > 0 else "negative", "strength": _strength(p)})
    return sorted(out, key=lambda x: -abs(x["pearson"]))[:limit]
