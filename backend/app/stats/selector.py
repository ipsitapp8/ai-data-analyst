"""Method selection. Given the design (what is being compared) and the data,
choose the method, run it, and return a full result that records why that
method was chosen, which alternatives were rejected, which assumptions were
checked, and what the result does not show.

The rules are documented in docs/STATISTICAL_REASONING.md. In short:

  two independent groups, means   Welch's t-test when Cochran's rule (n >= 25 * skewness^2) holds in both
                                  groups; otherwise a permutation test of the same difference in means
                                  with a bootstrap interval. Mann-Whitney only when the question is about
                                  typical values rather than the mean.
  paired units                    paired t-test under the same rule on the differences; otherwise the
                                  sign-flip permutation test. Wilcoxon only for "typical".
  two proportions                 chi-square when every expected count is at least 5, otherwise Fisher's
                                  exact test. Newcombe/Wilson score intervals in both cases.
  r x c table                     chi-square under Cochran's condition; Fisher for a sparse 2x2; a
                                  permutation chi-square for a sparse larger table.
  several groups, means           Welch's one-way test; a permutation one-way test when the rule fails.
                                  Pairwise comparisons are Holm-adjusted within the analysis.
  two numeric variables           Pearson; Spearman when extreme outliers could dominate it.
  a series over time              least squares with Newey-West errors (never an independent-rows test).
  adjusted comparison             least squares with HC3 errors.

A method a model suggests is recorded as accepted or overridden; it never
decides."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as st

from app import config
from app.stats import methods as m
from app.stats import series as ts
from app.stats.result import BIAS_NOTE, FRAME_NOTE, OBSERVATIONAL_NOTE, build, fmt, unavailable

LIMITS = [FRAME_NOTE, OBSERVATIONAL_NOTE, BIAS_NOTE]
ALIASES = {
    "welch": "welch_t", "welch_t": "welch_t", "t_test": "welch_t", "t-test": "welch_t", "ttest": "welch_t",
    "student_t": "welch_t", "permutation": "permutation_means", "permutation_means": "permutation_means",
    "mann_whitney": "mann_whitney", "mann-whitney": "mann_whitney", "wilcoxon_rank_sum": "mann_whitney",
    "paired_t": "paired_t", "paired_t_test": "paired_t", "sign_flip": "sign_flip_permutation",
    "wilcoxon": "wilcoxon_signed_rank", "wilcoxon_signed_rank": "wilcoxon_signed_rank",
    "chi_square": "chi_square", "chi-square": "chi_square", "chi2": "chi_square", "fisher": "fisher_exact",
    "fisher_exact": "fisher_exact", "z_test": "chi_square", "two_proportion_z": "chi_square",
    "anova": "welch_anova", "welch_anova": "welch_anova", "kruskal": "kruskal_wallis", "kruskal_wallis": "kruskal_wallis",
    "pearson": "pearson", "spearman": "spearman", "regression": "ols_hc3", "ols": "ols_hc3",
}


def _alpha(alpha: float | None) -> float:
    return config.HYP_SIGNIFICANCE_ALPHA if alpha is None else alpha


def _level(level: float | None) -> float:
    return config.STAT_CONFIDENCE_LEVEL if level is None else level


def _suggestion(suggested: Any, chosen: str, why_not: str) -> dict | None:
    """What became of a method proposed by a model."""
    if not suggested:
        return None
    key = ALIASES.get(str(suggested).strip().lower().replace(" ", "_"))
    if key is None:
        return {"method": str(suggested)[:60], "outcome": "unknown", "reason": "not a supported method; the selector chose independently"}
    if key == chosen:
        return {"method": key, "outcome": "accepted", "reason": "matches the method the data supports"}
    return {"method": key, "outcome": "overridden", "reason": why_not}


def _cochran(label: str, d: dict) -> tuple[bool, dict]:
    """Is a t-based interval for this group's mean justified?

    Cochran's rule (n >= 25 * skewness^2), applied to the upper end of what the
    skewness could plausibly be: a skewness measured on a handful of values is
    itself uncertain (standard error about sqrt(6 / n)), so a small sample
    cannot vouch for its own symmetry."""
    if d["n"] < 3 or d["skewness"] is None:
        return False, {"name": f"normality of the mean ({label})", "status": "not_checked",
                       "detail": f"{d['n']} value(s): too few to measure skewness"}
    bound = abs(d["skewness"]) + 2 * math.sqrt(6 / d["n"])
    need = m.cochran_n(bound)
    ok = d["n"] >= need
    return ok, {"name": f"normality of the mean ({label})", "status": "met" if ok else "violated",
                "detail": f"skewness {d['skewness']:.2f} (plausibly up to {bound:.2f} at this sample size); "
                          f"Cochran's rule needs n >= {math.ceil(need)}, n = {d['n']}"}


def _outlier_warnings(groups: dict[str, dict]) -> list[str]:
    return [f"{d['extreme_outliers']} extreme value(s) in '{label}' (beyond 3 interquartile ranges); they were kept, "
            "and they pull the mean" for label, d in groups.items() if d["extreme_outliers"]]


INDEPENDENCE = {"name": "independent observations", "status": "not_checked",
                "detail": "rows are treated as independent; repeated rows from the same customer, order or day would "
                          "make the interval too narrow"}


def _power_warning(pieces: dict, alpha: float) -> list[str]:
    floor = pieces.get("smallest_attainable_p")
    if floor is not None and floor > alpha:
        # Resampling so few values cannot describe their uncertainty either.
        pieces["interval"] = None
        return ["No interval is reported: there are too few observations to resample.", f"With this few observations the smallest p-value any data could produce is {floor:.3f}, above "
                f"{alpha:g}: the test cannot detect an effect of any size."][::-1]
    return []


# ----------------------------------------------------- two independent groups --

def two_groups(a: Any, b: Any, *, what: str, units: str | None = None, labels: tuple[str, str] = ("earlier", "later"),
               estimand: str = "mean", suggested: Any = None, alpha: float | None = None, level: float | None = None,
               seed: int | None = None) -> dict:
    """Compare a numeric outcome between two independent groups."""
    alpha, level, seed = _alpha(alpha), _level(level), config.STAT_SEED if seed is None else seed
    a, miss_a = m.clean(a)
    b, miss_b = m.clean(b)
    da, db = m.describe(a), m.describe(b)
    typical = estimand == "typical"
    common = dict(design="independent_groups", units=units, alpha=alpha,
                  estimand=f"difference in {'median' if typical else 'mean'} {what} ({labels[1]} minus {labels[0]})",
                  sample={"n": da["n"] + db["n"], "groups": {labels[0]: da["n"], labels[1]: db["n"]},
                          "excluded": {"missing_or_non_numeric": {labels[0]: miss_a, labels[1]: miss_b}},
                          "exclusion_rule": "values that are missing or not numeric are left out; nothing is imputed"},
                  descriptives={labels[0]: da, labels[1]: db}, limitations=list(LIMITS))
    if min(da["n"], db["n"]) < 2:
        return unavailable(method="none", method_label="no test", reason=f"each group needs at least 2 values to estimate "
                           f"its variation ({da['n']} and {db['n']} available)", **common)
    if not da["sd"] and not db["sd"]:
        return unavailable(method="none", method_label="no test", reason="there is no variation within either group, so the "
                           "difference is exact and there is nothing to test", **common)

    ok_a, as_a = _cochran(labels[0], da)
    ok_b, as_b = _cochran(labels[1], db)
    assumptions, rejected, extra = [as_a, as_b, INDEPENDENCE], [], {}
    if typical:
        method, label = "mann_whitney", "Mann-Whitney U test"
        reasons = ["the question is about typical values rather than the mean, so groups are compared by rank"]
        pieces = m.rank_sum(a, b, level, seed)
        extra["effects"] = [{"name": "probability a later value exceeds an earlier one", "value": pieces["probability_of_superiority"]}]
        why_not = "the question asks about typical values, which a test of means does not address"
    elif ok_a and ok_b:
        method, label = "welch_t", "Welch's t-test"
        reasons = ["the estimand is a difference in means between two independent groups",
                   "Welch's test is used because it does not assume the groups have equal variances",
                   "both groups satisfy Cochran's rule, so the t-based interval is appropriate"]
        rejected.append({"method": "mann_whitney", "reason": "it tests a different quantity (rank ordering), not the mean"})
        pieces = m.welch(a, b, level)
        why_not = "the data supports Welch's t-test for a difference in means"
    else:
        method, label = "permutation_means", "permutation test of the difference in means"
        reasons = ["the estimand is a difference in means between two independent groups",
                   "the distribution is too skewed for the sample size (or the sample too small to check), so the "
                   "p-value comes from permuting group labels and the interval from the bootstrap"]
        rejected.append({"method": "welch_t", "reason": "Cochran's rule is not met, so a t-based interval may be inaccurate"})
        rejected.append({"method": "mann_whitney", "reason": "it tests a different quantity (rank ordering), not the mean"})
        pieces = m.permutation_means(a, b, level, seed)
        why_not = "Cochran's rule is not met, so a t-based method may be inaccurate here"

    p = pieces["test"]["p_value"]
    power = _power_warning(pieces, alpha)
    effects = extra.get("effects", [])
    if not typical:
        effects = [{"name": "mean difference", "value": pieces["estimate"], "units": units, "interval": pieces["interval"]},
                   {"name": "relative difference, percent", "value": pieces["estimate"] / abs(da["mean"]) * 100 if da["mean"] else None,
                    "note": None if da["mean"] else "not defined: the earlier mean is zero"},
                   {"name": "Hedges' g (standardised mean difference)", "value": m.hedges_g(a, b)}]
    null = (f"values in the two groups are equally likely to be higher" if typical else f"the two groups have the same mean {what}")
    return build(status="ok", method=method, method_label=label, selection=reasons, rejected_methods=rejected,
                 suggested_method=_suggestion(suggested, method, why_not), assumptions=assumptions,
                 estimate=pieces["estimate"], interval=pieces["interval"],
                 test={**pieces["test"], "null": null, "alternative": "they differ (two-sided)"},
                 effects=effects, significant=p < alpha,
                 warnings=_outlier_warnings({labels[0]: da, labels[1]: db}) + power, **common)


# ------------------------------------------------------------------- paired --

def paired(first: Any, second: Any, *, what: str, unit_name: str, units: str | None = None, estimand: str = "mean",
           suggested: Any = None, alpha: float | None = None, level: float | None = None, seed: int | None = None,
           unmatched: int = 0) -> dict:
    """Compare the same units measured twice (second minus first)."""
    alpha, level, seed = _alpha(alpha), _level(level), config.STAT_SEED if seed is None else seed
    x, y = np.asarray(first, float), np.asarray(second, float)
    keep = np.isfinite(x) & np.isfinite(y)
    d = (y - x)[keep]
    desc = m.describe(d)
    typical = estimand == "typical"
    common = dict(design="paired", units=units, alpha=alpha,
                  estimand=f"{'median' if typical else 'mean'} change in {what} per {unit_name} (later minus earlier)",
                  sample={"n": desc["n"], "pairs": desc["n"],
                          "excluded": {"units_not_in_both_periods": unmatched, "missing": int((~keep).sum())},
                          "exclusion_rule": f"only {unit_name} values present in both periods are compared"},
                  descriptives={"differences": desc}, limitations=list(LIMITS))
    if desc["n"] < 2:
        return unavailable(method="none", method_label="no test", reason=f"at least 2 {unit_name} values must appear in "
                           f"both periods ({desc['n']} do)", **common)
    if not desc["sd"]:
        return unavailable(method="none", method_label="no test", reason="every unit changed by exactly the same amount, "
                           "so the change is exact and there is nothing to test", **common)
    ok, assumption = _cochran("differences", desc)
    pairing = {"name": "valid pairing", "status": "met", "detail": f"each {unit_name} is compared with itself, which removes "
               f"stable differences between {unit_name} values"}
    rejected = [{"method": "welch_t", "reason": "the two periods contain the same units, so they are not independent groups"}]
    if typical:
        if np.all(d == 0):
            return unavailable(method="none", method_label="no test", reason="no unit changed", **common)
        method, label, pieces = "wilcoxon_signed_rank", "Wilcoxon signed-rank test", m.signed_rank(d, level, seed)
        reasons = ["the same units are observed in both periods, so they are compared as pairs",
                   "the question is about the typical change, so signed ranks are used"]
        why_not = "the question asks about the typical change"
    elif ok:
        method, label, pieces = "paired_t", "paired t-test", m.paired_t(d, level)
        reasons = ["the same units are observed in both periods, so they are compared as pairs",
                   "the differences satisfy Cochran's rule, so the t-based interval is appropriate"]
        why_not = "the paired differences support a paired t-test"
    else:
        method, label, pieces = "sign_flip_permutation", "paired sign-flip permutation test", m.sign_flip(d, level, seed)
        reasons = ["the same units are observed in both periods, so they are compared as pairs",
                   "there are too few pairs, or the differences are too skewed, for a t-based interval; the p-value "
                   "comes from flipping the sign of each unit's change"]
        rejected.append({"method": "paired_t", "reason": "Cochran's rule is not met for the differences"})
        why_not = "Cochran's rule is not met for the paired differences"
    p = pieces["test"]["p_value"]
    power = _power_warning(pieces, alpha)
    effects = [{"name": "mean change per unit", "value": float(d.mean()), "units": units},
               {"name": "standardised mean change (d_z)", "value": float(d.mean() / desc["sd"])}]
    null = "units are as likely to go up as down" if typical else f"the mean change per {unit_name} is zero"
    return build(status="ok", method=method, method_label=label, selection=reasons, rejected_methods=rejected,
                 suggested_method=_suggestion(suggested, method, why_not), assumptions=[pairing, assumption],
                 estimate=pieces["estimate"], interval=pieces["interval"],
                 test={**pieces["test"], "null": null, "alternative": "it is not (two-sided)"}, effects=effects,
                 significant=p < alpha, warnings=_outlier_warnings({"differences": desc}) + power, **common)


# -------------------------------------------------------------- proportions --

def proportions(x_a: int, n_a: int, x_b: int, n_b: int, *, what: str, labels: tuple[str, str] = ("earlier", "later"),
                suggested: Any = None, alpha: float | None = None, level: float | None = None) -> dict:
    """Compare a share between two independent groups (later minus earlier)."""
    alpha, level = _alpha(alpha), _level(level)
    common = dict(design="independent_groups", units="share (0 to 1)", alpha=alpha,
                  estimand=f"difference in the {what} ({labels[1]} minus {labels[0]})",
                  sample={"n": n_a + n_b, "groups": {labels[0]: n_a, labels[1]: n_b}, "matching": {labels[0]: x_a, labels[1]: x_b}},
                  limitations=list(LIMITS))
    if min(n_a, n_b) <= 0:
        return unavailable(method="none", method_label="no test", reason="one of the groups has no rows", **common)
    pa, pb = x_a / n_a, x_b / n_b
    wa, wb = m.wilson(x_a, n_a, level), m.wilson(x_b, n_b, level)
    common["descriptives"] = {labels[0]: {"n": n_a, "matching": x_a, "share": pa, "wilson_interval": list(wa)},
                              labels[1]: {"n": n_b, "matching": x_b, "share": pb, "wilson_interval": list(wb)}}
    diff = m.newcombe(x_a, n_a, x_b, n_b, level)
    interval = {"low": diff[0], "high": diff[1], "level": level, "method": "Newcombe hybrid score interval"}
    effects = [{"name": "risk difference, percentage points", "value": (pb - pa) * 100,
                "interval": {"low": diff[0] * 100, "high": diff[1] * 100, "level": level, "method": interval["method"]}}]
    effects += m.ratio_effects(x_a, n_a, x_b, n_b, level)
    table = np.array([[x_a, n_a - x_a], [x_b, n_b - x_b]], float)
    reasons = ["the outcome is binary (a row matches or does not), compared between two independent groups",
               "the interval is a score interval, which stays valid for shares near 0 or 1 where the textbook normal "
               "interval does not"]
    if table.sum(axis=0).min() == 0:
        return build(status="ok", method="none", method_label="no test (the outcome does not vary)",
                     selection=reasons + ["every row is on the same side of the outcome, so there is no association to test"],
                     estimate=pb - pa, interval=interval, effects=effects, significant=None, assumptions=[INDEPENDENCE], **common)
    expected = m.expected_counts(table)
    counts = {"name": "expected counts", "detail": f"smallest expected count {expected.min():.1f}", "status": "met"}
    if m.cochran_expected_ok(expected):
        method, label, test = "chi_square", "chi-square test of independence", m.chi_square(table)
        reasons.append("every expected count is at least 5, so the chi-square approximation is appropriate")
        rejected = [{"method": "fisher_exact", "reason": "not needed: the chi-square approximation's condition is met"}]
        why_not = "expected counts are large enough for the chi-square test"
    else:
        method, label, test = "fisher_exact", "Fisher's exact test", m.fisher(table)
        counts["status"] = "violated"
        reasons.append("an expected count is below 5, so the chi-square approximation is unreliable and the exact test is used")
        rejected = [{"method": "chi_square", "reason": "an expected count is below 5"}]
        why_not = "an expected count is below 5, so an exact test is needed"
    return build(status="ok", method=method, method_label=label, selection=reasons, rejected_methods=rejected,
                 suggested_method=_suggestion(suggested, method, why_not), assumptions=[counts, INDEPENDENCE],
                 estimate=pb - pa, interval=interval,
                 test={**test, "null": f"the {what} is the same in both groups", "alternative": "it differs (two-sided)"},
                 effects=effects, significant=test["p_value"] < alpha, **common)


def contingency(rows: Any, cols: Any, *, row_name: str, col_name: str, suggested: Any = None, alpha: float | None = None,
                seed: int | None = None) -> dict:
    """Association between two categorical variables."""
    import pandas as pd

    alpha, seed = _alpha(alpha), config.STAT_SEED if seed is None else seed
    frame = pd.DataFrame({"r": pd.Series(rows).astype("object"), "c": pd.Series(cols).astype("object")})
    missing = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna()
    table = pd.crosstab(frame["r"].astype(str), frame["c"].astype(str))
    common = dict(design="association", alpha=alpha, estimand=f"association between {row_name} and {col_name}",
                  sample={"n": int(len(frame)), "excluded": {"missing": missing},
                          "exclusion_rule": "rows missing either value are left out"},
                  limitations=list(LIMITS))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return unavailable(method="none", method_label="no test", reason=f"{row_name} or {col_name} has a single value in "
                           "this data, so no association can be measured", **common)
    values = table.to_numpy(float)
    expected = m.expected_counts(values)
    counts = {"name": "expected counts", "status": "met",
              "detail": f"smallest expected count {expected.min():.1f}; {(expected < 5).mean() * 100:.0f}% of cells below 5"}
    chi2 = float(((values - expected) ** 2 / expected).sum())
    reasons = [f"both variables are categorical ({table.shape[0]} x {table.shape[1]} table of counts)"]
    if m.cochran_expected_ok(expected):
        method, label, test = "chi_square", "chi-square test of independence", m.chi_square(values)
        reasons.append("expected counts satisfy Cochran's condition for the chi-square approximation")
        rejected, why_not = [], "expected counts are large enough for the chi-square test"
    elif values.shape == (2, 2):
        method, label, test = "fisher_exact", "Fisher's exact test", m.fisher(values)
        counts["status"] = "violated"
        reasons.append("the table is 2 x 2 with small expected counts, so the exact test is used")
        rejected, why_not = [{"method": "chi_square", "reason": "an expected count is below 5"}], "expected counts are too small"
    else:
        method, label = "permutation_chi_square", "permutation chi-square test"
        test = m.permutation_chi_square(frame["r"].astype(str).to_numpy(), frame["c"].astype(str).to_numpy(), seed)
        counts["status"] = "violated"
        reasons.append("the table is sparse, so the p-value comes from shuffling one variable rather than the chi-square approximation")
        rejected, why_not = [{"method": "chi_square", "reason": "too many small expected counts"}], "the table is too sparse"
    v = m.cramers_v(chi2, int(values.sum()), values.shape)
    shown = table.iloc[:12, :12]
    return build(status="ok", method=method, method_label=label, selection=reasons, rejected_methods=rejected,
                 suggested_method=_suggestion(suggested, method, why_not), assumptions=[counts, INDEPENDENCE],
                 estimate=v, test={**test, "null": f"{row_name} and {col_name} are unrelated", "alternative": "they are associated"},
                 effects=[{"name": "Cramér's V (0 = none, 1 = complete association)", "value": v}],
                 significant=test["p_value"] < alpha,
                 details={"table": {"rows": [str(i) for i in shown.index], "columns": [str(c) for c in shown.columns],
                                    "counts": shown.to_numpy(int).tolist(), "truncated": table.shape != shown.shape}}, **common)


# ---------------------------------------------------------------- k groups --

def k_groups(groups: dict[str, Any], *, what: str, group_name: str, units: str | None = None, suggested: Any = None,
             alpha: float | None = None, level: float | None = None, seed: int | None = None) -> dict:
    """Compare a numeric outcome across the levels of a categorical variable."""
    alpha, level, seed = _alpha(alpha), _level(level), config.STAT_SEED if seed is None else seed
    cleaned = {str(k): m.clean(v)[0] for k, v in groups.items()}
    usable = {k: v for k, v in cleaned.items() if len(v) >= 2}
    dropped = sorted(set(cleaned) - set(usable))
    desc = {k: m.describe(v) for k, v in usable.items()}
    common = dict(design="independent_groups", units=units, alpha=alpha, estimand=f"differences in mean {what} between {group_name} groups",
                  sample={"n": int(sum(len(v) for v in usable.values())), "groups": {k: len(v) for k, v in usable.items()},
                          "excluded": {"groups_with_fewer_than_2_values": dropped},
                          "exclusion_rule": "a group needs at least 2 values to estimate its variation"},
                  descriptives=desc, limitations=list(LIMITS))
    if len(usable) < 2:
        return unavailable(method="none", method_label="no test", reason=f"fewer than 2 {group_name} groups have enough values to compare", **common)
    arrays = list(usable.values())
    if all(not d["sd"] for d in desc.values()):
        return unavailable(method="none", method_label="no test", reason="there is no variation within any group", **common)
    checks = [_cochran(k, d) for k, d in desc.items()]
    assumptions = [c[1] for c in checks] + [INDEPENDENCE]
    if all(c[0] for c in checks) and all(d["sd"] for d in desc.values()):
        method, label, test = "welch_anova", "Welch's one-way test", m.welch_anova(arrays)
        reasons = [f"the estimand is a difference in means across {len(usable)} independent groups",
                   "Welch's one-way test is used because it does not assume equal variances",
                   "every group satisfies Cochran's rule"]
        rejected = [{"method": "kruskal_wallis", "reason": "it tests rank ordering, not the mean"}]
        why_not = "the data supports Welch's one-way test of means"
    else:
        res = st.permutation_test(arrays, lambda *g, axis=-1: st.f_oneway(*g, axis=axis).statistic, vectorized=True,
                                  n_resamples=m._resamples(sum(len(v) for v in arrays)), alternative="greater",
                                  rng=np.random.default_rng(seed))
        method, label = "permutation_anova", "permutation one-way test of equal means"
        test = {"name": label, "statistic_name": "F", "statistic": float(res.statistic), "p_value": float(res.pvalue),
                "resamples": int(len(res.null_distribution))}
        reasons = [f"the estimand is a difference in means across {len(usable)} independent groups",
                   "at least one group is too skewed or too small for an F-based p-value, so group labels are permuted instead"]
        rejected = [{"method": "welch_anova", "reason": "Cochran's rule is not met in every group"},
                    {"method": "kruskal_wallis", "reason": "it tests rank ordering, not the mean"}]
        why_not = "Cochran's rule is not met in every group"

    comparisons, warnings = [], _outlier_warnings(desc)
    names = list(usable)
    if 2 < len(names) <= config.STAT_MAX_PAIRWISE_GROUPS:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        raw = [two_groups(usable[x], usable[y], what=what, units=units, labels=(x, y), alpha=alpha, level=level, seed=seed) for x, y in pairs]
        ps = [r["test"]["p_value"] if r["status"] == "ok" and r["test"] else None for r in raw]
        adjusted = iter(m.holm([p for p in ps if p is not None]))
        for (x, y), r, p in zip(pairs, raw, ps):
            comparisons.append({"groups": [x, y], "estimate": r.get("estimate"), "interval": r.get("interval"), "method": r["method"],
                                "p_value": p, "p_adjusted": None if p is None else next(adjusted)})
    elif len(names) > config.STAT_MAX_PAIRWISE_GROUPS:
        warnings.append(f"{len(names)} groups: pairwise comparisons were not run (more than {config.STAT_MAX_PAIRWISE_GROUPS} "
                        "groups would mean too many tests to interpret)")
    return build(status="ok", method=method, method_label=label, selection=reasons, rejected_methods=rejected,
                 suggested_method=_suggestion(suggested, method, why_not), assumptions=assumptions,
                 test={**test, "null": f"every {group_name} group has the same mean {what}", "alternative": "at least one differs"},
                 effects=[{"name": "eta squared (share of variance between groups)", "value": m.eta_squared(arrays)}],
                 significant=test["p_value"] < alpha, comparisons=comparisons, warnings=warnings,
                 details={"pairwise_family": f"all pairwise comparisons of {group_name} in this analysis",
                          "pairwise_correction": "Holm" if comparisons else None}, **common)


# -------------------------------------------------------------- correlation --

def correlation(x: Any, y: Any, *, x_name: str, y_name: str, estimand: str = "linear", suggested: Any = None,
                alpha: float | None = None, level: float | None = None) -> dict:
    import pandas as pd

    alpha, level = _alpha(alpha), _level(level)
    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"), "y": pd.to_numeric(pd.Series(y), errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    missing = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna()
    xs, ys = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    dx, dy = m.describe(xs), m.describe(ys)
    monotonic = estimand == "monotonic"
    common = dict(design="association", alpha=alpha, sample={"n": len(xs), "excluded": {"missing_or_non_numeric": missing},
                                                             "exclusion_rule": "rows missing either value are left out"},
                  descriptives={x_name: dx, y_name: dy}, limitations=list(LIMITS))
    name = f"association between {x_name} and {y_name}"
    if len(xs) < 4:
        return unavailable(method="none", method_label="no test", estimand=name, reason=f"{len(xs)} complete rows: at least 4 are needed", **common)
    if not dx["sd"] or not dy["sd"]:
        return unavailable(method="none", method_label="no test", estimand=name, reason="one of the variables does not vary", **common)
    outliers = dx["extreme_outliers"] + dy["extreme_outliers"]
    if monotonic or outliers:
        method, label, pieces = "spearman", "Spearman rank correlation", m.spearman(xs, ys, level)
        reasons = (["the question is about a monotonic relationship"] if monotonic else
                   [f"{outliers} extreme value(s) could dominate a Pearson correlation, so ranks are used"])
        rejected, why_not = [{"method": "pearson", "reason": "sensitive to the extreme values present"}], "extreme values are present"
        estimand_text = f"monotonic {name} (Spearman's rho)"
    else:
        method, label, pieces = "pearson", "Pearson correlation", m.pearson(xs, ys, level)
        reasons = ["both variables are numeric with no extreme values, so the linear correlation is reported"]
        rejected, why_not = [], "no extreme values are present, so the linear correlation is appropriate"
        estimand_text = f"linear {name} (Pearson's r)"
    return build(status="ok", method=method, method_label=label, estimand=estimand_text, selection=reasons,
                 rejected_methods=rejected, suggested_method=_suggestion(suggested, method, why_not),
                 assumptions=[INDEPENDENCE], estimate=pieces["estimate"], interval=pieces["interval"],
                 test={**pieces["test"], "null": f"{x_name} and {y_name} are unrelated", "alternative": "they are related (two-sided)"},
                 effects=[{"name": "share of variance in common (r squared)", "value": pieces["estimate"] ** 2}],
                 significant=pieces["test"]["p_value"] < alpha, **common)


# -------------------------------------------------------------- time series --

SERIES_LIMITS = [
    "Consecutive time buckets are not independent; the standard error allows for autocorrelation (Newey-West), "
    "which is approximate when the series is short.",
    OBSERVATIONAL_NOTE, BIAS_NOTE]


def _series_ok(n: int) -> bool:
    return n >= config.STAT_MIN_SERIES_POINTS


def level_shift(a: Any, b: Any, *, what: str, bucket: str, units: str | None = None, alpha: float | None = None,
                level: float | None = None, context: dict | None = None) -> dict:
    """Did the level of a series differ between two consecutive periods, beyond
    its ordinary bucket-to-bucket variation?"""
    alpha, level, context = _alpha(alpha), _level(level), context or {}
    a, b = np.asarray(a, float), np.asarray(b, float)
    common = dict(design="time_series", units=units, alpha=alpha, method="ols_newey_west",
                  method_label="difference in level with Newey-West standard errors",
                  estimand=f"difference in {what} per {bucket} (later minus earlier)",
                  sample={"n": len(a) + len(b), "groups": {"earlier": len(a), "later": len(b)}, "bucket": bucket,
                          "excluded": context.get("excluded", {}), "exclusion_rule": context.get("exclusion_rule", "")},
                  descriptives={"earlier": m.describe(a), "later": m.describe(b)},
                  limitations=SERIES_LIMITS + context.get("limitations", []), warnings=context.get("warnings", []),
                  assumptions=context.get("assumptions", []))
    floor = config.STAT_MIN_SERIES_POINTS
    if not (_series_ok(len(a)) and _series_ok(len(b))):
        return unavailable(reason=f"{len(a)} and {len(b)} {bucket} buckets: at least {floor} are needed in each period to "
                           "estimate the series' own variation and its autocorrelation", **common)
    if np.concatenate([a, b]).std() == 0:
        return unavailable(reason="the series is constant", **common)
    fit = ts.level_shift(a, b, level)
    if fit["p_value"] is None:
        return unavailable(reason="the series does not vary within the periods, so the difference is exact", **common)
    r1 = fit["autocorrelation"]
    common["assumptions"] = common["assumptions"] + [{
        "name": "autocorrelation", "status": "met",
        "detail": f"lag-1 autocorrelation of the residuals {r1:+.2f}; handled with Newey-West errors using {fit['lags']} lag(s)"
        if r1 is not None else "could not be estimated"}]
    mean_a = float(a.mean())
    return build(status="ok", selection=[
        f"the metric is observed as a series of {bucket} values, which are not independent of each other",
        "the period difference is therefore estimated by least squares with autocorrelation-robust standard errors",
        "a row-level test is not used: it would treat thousands of rows from the same days as independent evidence"],
        rejected_methods=[{"method": "welch_t", "reason": "assumes independent observations"}],
        estimate=fit["estimate"], interval={"low": fit["low"], "high": fit["high"], "level": level, "method": "t interval with Newey-West standard error"},
        test={"name": "t-test of the period difference (Newey-West)", "statistic_name": "t", "statistic": fit["statistic"],
              "df": float(fit["df"]), "p_value": fit["p_value"], "null": f"the average {what} per {bucket} is the same in both periods",
              "alternative": "it differs (two-sided)"},
        effects=[{"name": f"difference per {bucket}", "value": fit["estimate"], "units": units},
                 {"name": "relative difference, percent", "value": fit["estimate"] / abs(mean_a) * 100 if mean_a else None}],
        significant=fit["p_value"] < alpha, details={"lags": fit["lags"], "autocorrelation": r1, **context.get("details", {})}, **common)


def pre_trend(a: Any, b: Any, *, what: str, bucket: str, units: str | None = None, alpha: float | None = None,
              level: float | None = None, context: dict | None = None) -> dict:
    """Was the metric already trending before the later period began, and how
    much of the later change would that trend alone have produced?"""
    alpha, level, context = _alpha(alpha), _level(level), context or {}
    a, b = np.asarray(a, float), np.asarray(b, float)
    common = dict(design="time_series", units=units, alpha=alpha, method="trend_newey_west",
                  method_label="linear trend with Newey-West standard errors",
                  estimand=f"trend in {what} per {bucket} during the earlier period",
                  sample={"n": len(a), "bucket": bucket, "excluded": context.get("excluded", {})},
                  limitations=SERIES_LIMITS + ["A straight-line trend is a simplification; extending it into the later period "
                                               "is an extrapolation, not a forecast."] + context.get("limitations", []),
                  warnings=context.get("warnings", []), assumptions=context.get("assumptions", []))
    if not _series_ok(len(a)) or len(b) < 1:
        return unavailable(reason=f"{len(a)} {bucket} buckets in the earlier period: at least {config.STAT_MIN_SERIES_POINTS} "
                           "are needed to estimate a trend", **common)
    if a.std() == 0:
        return unavailable(reason="the earlier series is constant, so it has no trend", **common)
    fit = ts.linear_trend(a, level)
    if fit["p_value"] is None:
        return unavailable(reason="the trend's uncertainty could not be estimated", **common)
    # The trend line carried through the later period, averaged over its buckets.
    later_t = len(a) + (len(b) - 1) / 2
    projected = fit["intercept"] + fit["estimate"] * later_t
    observed_change, projected_change = float(b.mean() - a.mean()), float(projected - a.mean())
    share = projected_change / observed_change if observed_change else None
    return build(status="ok", selection=[f"a pre-existing trend is a slope over {bucket} buckets within the earlier period",
                                         "standard errors allow for autocorrelation (Newey-West)"],
                 estimate=fit["estimate"], interval={"low": fit["low"], "high": fit["high"], "level": level,
                                                     "method": "t interval with Newey-West standard error"},
                 test={"name": "t-test of the trend slope (Newey-West)", "statistic_name": "t", "statistic": fit["statistic"],
                       "df": float(fit["df"]), "p_value": fit["p_value"], "null": f"{what} had no trend in the earlier period",
                       "alternative": "it was rising or falling"},
                 effects=[{"name": f"trend per {bucket}", "value": fit["estimate"], "units": units},
                          {"name": "share of the observed change the earlier trend would produce", "value": share}],
                 significant=fit["p_value"] < alpha,
                 details={"observed_change_per_bucket": observed_change, "change_projected_from_trend": projected_change,
                          "share_explained_by_trend": share, "lags": fit["lags"], "autocorrelation": fit["autocorrelation"]}, **common)


def change_point(values: Any, labels: list[str], *, what: str, bucket: str, units: str | None = None,
                 alpha: float | None = None, context: dict | None = None) -> dict:
    """Is there a single shift in the level of the series, and where?"""
    alpha, context = _alpha(alpha), context or {}
    x = np.asarray(values, float)
    common = dict(design="time_series", units=units, alpha=alpha, method="pettitt", method_label="Pettitt's change-point test",
                  estimand=f"location of a shift in the level of {what} per {bucket}",
                  sample={"n": len(x), "bucket": bucket, "excluded": context.get("excluded", {})},
                  limitations=["Pettitt's test looks for one shift in level; a gradual drift or several shifts are not "
                               "described well by it.", OBSERVATIONAL_NOTE, BIAS_NOTE] + context.get("limitations", []),
                  warnings=context.get("warnings", []))
    floor = 2 * config.STAT_MIN_SERIES_POINTS
    if len(x) < floor:
        return unavailable(reason=f"{len(x)} {bucket} buckets: at least {floor} are needed to look for a shift", **common)
    if x.std() == 0:
        return unavailable(reason="the series is constant", **common)
    raw = ts.pettitt(x)
    # Autocorrelation makes a test that assumes independence too eager. It is measured within the two
    # segments (a real shift would otherwise look like autocorrelation) and removed when clearly present.
    idx = raw["index"]
    seg = np.concatenate([x[:idx + 1] - x[:idx + 1].mean(), x[idx + 1:] - x[idx + 1:].mean()])
    r1 = ts.autocorr1(seg)
    threshold = 2 / math.sqrt(len(x))
    assumption = {"name": "independent observations", "status": "met",
                  "detail": f"lag-1 autocorrelation within segments {r1:+.2f} (threshold {threshold:.2f})" if r1 is not None else "not estimable"}
    result = raw
    if r1 is not None and abs(r1) > threshold:
        assumption["status"] = "violated"
        assumption["detail"] += "; the test was run on the series with that autocorrelation removed"
        white = ts.pettitt(ts.prewhiten(x, r1))
        result = {**white, "index": idx}
    before, after = x[:idx + 1], x[idx + 1:]
    shift = float(after.mean() - before.mean())
    return build(status="ok", selection=["a rank-based test for a single shift in level, which makes no assumption about the "
                                         "shape of the distribution"],
                 assumptions=[assumption], estimate=shift,
                 test={"name": "Pettitt's test", "statistic_name": "K", "statistic": result["statistic"], "p_value": result["p_value"],
                       "null": "the level of the series did not shift", "alternative": "it shifted once"},
                 effects=[{"name": "difference in level after the estimated shift", "value": shift, "units": units}],
                 significant=result["p_value"] < alpha,
                 details={"shift_after": labels[idx], "first_bucket_after_shift": labels[idx + 1], "position": idx + 1,
                          "mean_before": float(before.mean()), "mean_after": float(after.mean()), "autocorrelation": r1,
                          "p_value_is_approximate": True}, **common)


# -------------------------------------------------------------- regression --

def adjusted_difference(y: Any, later: Any, controls: dict[str, Any], *, what: str, units: str | None = None,
                        alpha: float | None = None, level: float | None = None) -> dict:
    """The period difference in a row-level outcome, before and after holding
    the control variables fixed (least squares, HC3 standard errors)."""
    import pandas as pd

    alpha, level = _alpha(alpha), _level(level)
    frame = pd.DataFrame({"__y": pd.to_numeric(pd.Series(y), errors="coerce"), "__later": np.asarray(later, float)})
    kinds: dict[str, str] = {}
    for name, values in controls.items():
        s = pd.Series(values).reset_index(drop=True)
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        frame[name] = pd.to_numeric(s, errors="coerce") if numeric else s.astype("object")
        kinds[name] = "numeric" if numeric else "categorical"
    total = len(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    used, dropped = [], []
    for name in controls:
        levels = frame[name].nunique()
        if levels < 2:
            dropped.append({"control": name, "reason": "it has a single value in the rows analysed"})
        elif kinds[name] == "categorical" and levels > config.STAT_MAX_CONTROL_LEVELS:
            dropped.append({"control": name, "reason": f"{levels} distinct values (more than {config.STAT_MAX_CONTROL_LEVELS})"})
        else:
            used.append(name)
    common = dict(design="regression", units=units, alpha=alpha, method="ols_hc3",
                  method_label="least-squares regression with HC3 robust standard errors",
                  estimand=f"difference in mean {what} between the periods, holding {', '.join(used) or 'nothing'} fixed",
                  sample={"n": int(len(frame)), "excluded": {"rows_with_missing_values": total - len(frame), "controls_dropped": dropped},
                          "exclusion_rule": "rows missing the outcome or any control are left out (complete cases)"},
                  limitations=[
                      "This is an adjusted association, not a causal estimate. It holds fixed only the listed variables; "
                      "anything not in the dataset that changed between the periods is not accounted for (omitted-variable bias).",
                      "The model is linear and additive: it assumes the period difference is the same within every level of the controls.",
                      FRAME_NOTE, BIAS_NOTE])
    if not used:
        return unavailable(reason="no usable control variable", **common)
    cols, names = [np.ones(len(frame)), frame["__later"].to_numpy(float)], ["(intercept)", "later period"]
    reference = {}
    for name in used:
        if kinds[name] == "numeric":
            cols.append(frame[name].to_numpy(float))
            names.append(name)
        else:
            counts = frame[name].astype(str).value_counts()
            reference[name] = str(counts.index[0])
            for lvl in counts.index[1:]:
                cols.append((frame[name].astype(str) == lvl).to_numpy(float))
                names.append(f"{name} = {lvl}")
    X, yv = np.column_stack(cols), frame["__y"].to_numpy(float)
    need = config.STAT_ROWS_PER_COEFFICIENT * X.shape[1]
    if len(yv) < need:
        return unavailable(reason=f"{len(yv)} complete rows for {X.shape[1]} coefficients: at least {need} are needed "
                           f"({config.STAT_ROWS_PER_COEFFICIENT} per coefficient)", **common)
    if frame["__later"].nunique() < 2:
        return unavailable(reason="all complete rows fall in one period", **common)
    try:
        fit = ts.ols(X, yv, cov="HC3")
        base = ts.ols(X[:, :2], yv, cov="HC3")
    except ts.RegressionError as e:
        return unavailable(reason=f"the model cannot be estimated: {e}", **common)
    adj, raw = ts.coefficient(fit, 1, level), ts.coefficient(base, 1, level)
    if adj["p_value"] is None:
        return unavailable(reason="the outcome does not vary once the controls are accounted for", **common)
    attenuation = 1 - adj["estimate"] / raw["estimate"] if raw["estimate"] else None
    coefficients = [{"term": n, **{k: ts.coefficient(fit, i, level)[k] for k in ("estimate", "se", "low", "high", "p_value")}}
                    for i, n in enumerate(names)][:40]
    warnings = []
    if fit["condition_number"] > 1e6:
        warnings.append("the controls are nearly collinear, which makes the adjusted estimate unstable")
    resid_desc = m.describe(fit["resid"])
    return build(status="ok", selection=[
        "the question is whether the period difference remains once other variables are held fixed, which needs a regression",
        "HC3 standard errors are used because they do not assume the same variance in every group",
        f"categorical controls enter as indicators (reference levels: {reference})" if reference else "all controls are numeric"],
        assumptions=[{"name": "linear, additive model", "status": "not_checked", "detail": "interactions between the period and the controls are not modelled"},
                     {"name": "no omitted confounders", "status": "not_checked", "detail": "cannot be checked from the data"},
                     {"name": "residual outliers", "status": "violated" if resid_desc["extreme_outliers"] else "met",
                      "detail": f"{resid_desc['extreme_outliers']} extreme residual(s)"}, INDEPENDENCE],
        estimate=adj["estimate"], interval={"low": adj["low"], "high": adj["high"], "level": level, "method": "t interval with HC3 standard error"},
        test={"name": "t-test of the adjusted period coefficient (HC3)", "statistic_name": "t", "statistic": adj["statistic"],
              "df": float(adj["df"]), "p_value": adj["p_value"],
              "null": f"mean {what} is the same in both periods once {', '.join(used)} is held fixed", "alternative": "it differs (two-sided)"},
        effects=[{"name": "adjusted difference", "value": adj["estimate"], "units": units},
                 {"name": "unadjusted difference", "value": raw["estimate"], "units": units,
                  "interval": {"low": raw["low"], "high": raw["high"], "level": level, "method": "t interval with HC3 standard error"}},
                 {"name": "share of the unadjusted difference removed by the controls", "value": attenuation}],
        significant=adj["p_value"] < alpha, warnings=warnings,
        details={"unadjusted": {k: raw[k] for k in ("estimate", "low", "high", "p_value")}, "attenuation": attenuation,
                 "controls": used, "reference_levels": reference, "coefficients": coefficients, "r_squared": fit["r_squared"],
                 "n": fit["n"], "parameters": fit["k"]}, **common)


# -------------------------------------------------------------- multiplicity --

def adjust_family(p_values: list[float], method: str = "holm") -> list[float]:
    if method == "holm":
        return m.holm(p_values)
    if method in ("bh", "fdr", "benjamini_hochberg"):
        return m.benjamini_hochberg(p_values)
    raise ValueError(f"unknown multiplicity method '{method}'")


__all__ = ["two_groups", "paired", "proportions", "contingency", "k_groups", "correlation", "level_shift", "pre_trend",
           "change_point", "adjusted_difference", "adjust_family", "fmt"]
