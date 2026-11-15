"""Which columns are associated with an outcome -- measured directly, without training a model.

For each other column (identifiers, free text and the outcome itself excluded):
  number vs number      Spearman rank correlation (monotonic association, robust to outliers)
  category vs number    eta squared: the share of the outcome's variance between the category's groups
  category vs category  Cramer's V from the contingency table
Each comes with the records used and a p-value; p-values are Holm-adjusted across all the factors tested, since
testing many factors at once finds some "significant" ones by chance. These are associations in this data, not
causes, and every answer says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.analyst.data import kind_of

MIN_N = 10
MAX_LEVELS = 30


def _holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def _id_like(s: pd.Series) -> bool:
    v = s.dropna()
    return len(v) > 20 and v.nunique() > 0.9 * len(v) and kind_of(s) == "categorical"


def compute(df: pd.DataFrame, target: str, exclude: list | None = None) -> dict:
    y = df[target]
    num_target = kind_of(y) == "numeric" and y.nunique() > 2
    rows = []
    for c in df.columns:
        if c == target or c in (exclude or []) or kind_of(df[c]) == "datetime" or _id_like(df[c]):
            continue
        x = df[c]
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(d) < MIN_N:
            continue
        x_num = kind_of(x) == "numeric" and x.nunique() > 2
        if not x_num and not 2 <= d["x"].nunique() <= MAX_LEVELS:
            continue
        try:
            if num_target and x_num:
                r, p = stats.spearmanr(d["x"], d["y"])
                rows.append({"factor": c, "method": "Spearman correlation", "strength": abs(float(r)), "value": float(r),
                             "direction": "higher with higher " + c if r > 0 else "lower with higher " + c, "p": float(p),
                             "n": int(len(d))})
            elif num_target:
                groups = [g["y"].to_numpy(float) for _, g in d.groupby("x")]
                grand = d["y"].mean()
                ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
                ss_t = float(((d["y"] - grand) ** 2).sum())
                eta2 = ss_b / ss_t if ss_t else 0.0
                p = float(stats.kruskal(*groups).pvalue) if len(groups) > 1 else 1.0
                means = d.groupby("x")["y"].mean().sort_values()
                rows.append({"factor": c, "method": "eta squared (variance between groups)", "strength": float(eta2),
                             "value": float(eta2), "direction": f"highest for {means.index[-1]}, lowest for "
                                                                  f"{means.index[0]}", "p": p, "n": int(len(d))})
            else:
                yy = d["y"].astype(str)
                if x_num:
                    xb = pd.cut(d["x"], bins=min(5, d["x"].nunique()), duplicates="drop").astype(str)
                else:
                    xb = d["x"].astype(str)
                tab = pd.crosstab(xb, yy)
                if tab.shape[0] < 2 or tab.shape[1] < 2:
                    continue
                chi2, p, _, _ = stats.chi2_contingency(tab)
                v = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))
                rows.append({"factor": c, "method": "Cramér's V" + (" (number grouped into ranges)" if x_num else ""),
                             "strength": v, "value": v, "direction": "", "p": float(p), "n": int(len(d))})
        except Exception:  # noqa: BLE001 - one factor that cannot be measured does not stop the others
            continue
    if rows:
        for r, a in zip(rows, _holm([r["p"] for r in rows])):
            r["p_adjusted"] = a
    rows.sort(key=lambda r: -r["strength"])
    return {"target": target, "factors": rows, "tested": len(rows),
            "target_kind": "numeric" if num_target else "categorical"}


def text(r: dict) -> str:
    from app.analyst.explain import f
    rows = r["factors"]
    if not rows:
        return f"No column could be compared with **{r['target']}** (each needs at least {MIN_N} complete records)."
    top = rows[0]
    sig = [x for x in rows if x["p_adjusted"] < 0.05]
    lines = [f"**Most strongly associated with {r['target']}: {top['factor']}** ({top['method']} {f(top['value'])}"
             + (f"; {top['direction']}" if top["direction"] else "") + f", {top['n']} records)."
             + f" {len(sig)} of {r['tested']} columns tested show an association unlikely to be chance (p < 0.05 after "
               "adjusting for testing many columns).", "",
             "| column | measure | value | direction | records | adjusted p |", "|---|---|---:|---|---:|---:|"]
    for x in rows[:10]:
        lines.append(f"| {x['factor']} | {x['method']} | {f(x['value'])} | {x['direction'] or '—'} | {x['n']} | "
                     f"{'< 0.001' if x['p_adjusted'] < 0.001 else f(x['p_adjusted'])} |")
    lines += ["", "_These are **associations** in this data, measured directly — not causes, and not a machine-learning "
                  "model. Columns can be associated because they share a cause, or by coincidence in a small sample. "
                  "For which inputs a predictive model relies on, ask to train one._"]
    return "\n".join(lines)
