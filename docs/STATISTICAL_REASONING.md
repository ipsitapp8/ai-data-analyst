# Statistical reasoning

Decides whether a statistical method applies, chooses one by explicit rules, computes it with SciPy/NumPy,
and attaches the result to the hypothesis it is evidence for. A model may suggest a method; it never
chooses one and never supplies a number.

Code: `backend/app/stats/` (`result.py`, `methods.py`, `series.py`, `selector.py`), used by
`analysis_ops.py` and `hypothesis_engine.py`. Builds on [HYPOTHESIS_ENGINE.md](HYPOTHESIS_ENGINE.md).

## What replaced what

Before this, the only inference was two hand-written tests using a normal approximation behind a fixed
"n ≥ 30" rule, with no interval or degrees of freedom, and "p ≥ 0.05" or an immaterial point estimate
decided a hypothesis by itself. Those functions are gone; the three operations that used them now go
through the selector.

## Supported methods

| Design | Method | Selected when | Interval | Effect sizes |
| --- | --- | --- | --- | --- |
| Two independent groups, mean | Welch's t-test | Cochran's rule holds in both groups | t, Welch–Satterthwaite df | mean difference, relative difference, Hedges' g |
| | Permutation test of the mean difference | rule fails, or too few values to check | bootstrap (BCa) | same |
| | Mann–Whitney U | only when `estimand: typical` is asked | bootstrap, median difference | probability of superiority |
| Paired units | Paired t-test | rule holds for the differences | paired t | mean change, d_z |
| | Sign-flip permutation (exact for few pairs) | rule fails | bootstrap | same |
| | Wilcoxon signed-rank | only for `estimand: typical` | bootstrap, median | |
| Two proportions | Chi-square | every expected count ≥ 5 | Newcombe hybrid score | risk difference, risk ratio, odds ratio |
| | Fisher's exact | an expected count < 5 | Newcombe | same |
| r × c table | Chi-square / Fisher (2×2) / permutation chi-square | Cochran's condition; sparse 2×2; sparse larger | — | Cramér's V |
| Several groups, mean | Welch's one-way test | rule holds in every group | per pair | eta squared; Holm-adjusted pairwise |
| | Permutation one-way test | otherwise | per pair | same |
| Two numeric columns | Pearson | no extreme values | Fisher z | r² |
| | Spearman | extreme values present | Fisher z (approximate) | |
| Series over time | Period difference, least squares + Newey–West | always for a series | t with HAC error | difference per bucket, relative |
| | Linear trend + Newey–West | pre-existing trend | same | share of the change the trend would produce |
| | Pettitt's change-point test | abrupt shift | — | shift size, date |
| Adjusted comparison | Least squares, HC3 errors | regression | t with HC3 error | adjusted and unadjusted difference, attenuation |

Multiplicity: Holm (default) or Benjamini–Hochberg (`STAT_MULTIPLICITY_METHOD`).

Not implemented, deliberately: forecasting and forecast evaluation, seasonal decomposition, logistic or
mixed models, causal designs (difference-in-differences, instruments, matching). Nothing in the project's
data justifies them yet, and none can be validated on the fixtures.

## Selection rules

- **Cochran's rule.** A t-based interval for a mean is used when `n ≥ 25 × skewness²` in each group, with
  the skewness taken at the upper end of its own uncertainty (`|g| + 2·√(6/n)`), because a skewness
  measured on a handful of values cannot vouch for the sample. This replaces any fixed sample-size cut-off:
  400 mildly skewed values pass, 25 heavily skewed ones do not, and 5 values never do.
- **The fallback keeps the estimand.** When the rule fails, the difference in means is still what is
  tested (permutation) and estimated (bootstrap). A rank test answers a different question and is used
  only when that question is asked.
- **Too few observations.** When the smallest p-value any data could produce exceeds the criterion
  (4 pairs: 2/2⁴ = 0.125), the result says the test cannot detect an effect of any size and reports no
  interval.
- **Proportions.** Chi-square under Cochran's expected-count condition, otherwise exact. Intervals are
  score intervals, never the textbook normal interval. An undefined ratio (empty cell) is unavailable, not 0.
- **Time.** A metric observed over days is analysed as a series with autocorrelation-robust errors; a
  row-level test is rejected for it, with the reason recorded. Day buckets, or weeks beyond 180 days, then
  months; partial weeks and months are left out. Empty days count as zero for a sum when rare (≤ 10%) and
  are left out when common (data recorded every second day has its own calendar). At least
  `STAT_MIN_SERIES_POINTS` buckets per period, else no conclusion.
- **Regression.** At least `STAT_ROWS_PER_COEFFICIENT` complete rows per coefficient; a categorical control
  with more than `STAT_MAX_CONTROL_LEVELS` levels is dropped; collinear controls are refused.
- **A suggested method** is recorded as accepted, overridden (with the reason) or unknown. It changes no number.
- Missing or non-numeric values are left out and counted. Nothing is imputed. Outliers are flagged and kept.

## How results decide a hypothesis

Direction, distinguishability and size are judged separately (`analysis_ops._directional`):

| Distinguishable from no change | Direction | Size | Status |
| --- | --- | --- | --- |
| yes | as predicted | material (≥ `HYP_MATERIAL_PCT`) | supported |
| yes | as predicted | immaterial | weakened — real but too small to explain much |
| yes | opposite | | weakened |
| no | | interval excludes a material move in the predicted direction | weakened |
| no | | otherwise | inconclusive — not evidence of no effect |
| method could not run | | | inconclusive |

"Distinguishable" uses the Holm-adjusted p-value when the test is one of a family. Statistical
significance alone never supports a hypothesis, and its absence alone never counts against one.

## Multiplicity

The family is every distinct statistical test used as test evidence for a hypothesis in one investigation.
With two or more, p-values are adjusted and each hypothesis is judged again; raw and adjusted values are
both stored. Outside the family, by definition: the baseline test of the headline change (one
pre-specified test), drill-down refinements, and the exploratory segment scan, which carries no p-values.
Pairwise comparisons inside one `group_difference` analysis form their own Holm family.

## Result model

Every method returns one validated shape (`stats/result.py`, schema version 1): method and version,
estimand and units, design, why the method was chosen and which were rejected, sample sizes and exclusions,
descriptives, assumption checks, estimate, interval, test (null, alternative, statistic, df, p), effect
sizes, significance against the fixed criterion, adjusted p, warnings, limitations and a plain-language
interpretation. Unavailable values are `null`. A result with a p-value outside [0, 1], a reversed
interval or a non-finite estimate is refused. Stored inside `investigation_analyses.result_json`
(no migration); only aggregates are stored, never rows.

## Limits of what this shows

- The dataset is usually every recorded row, so a measured difference is exact for that data. Intervals
  and tests answer whether it exceeds ordinary variation between rows or days, not a sampling question.
- The data is observational. Adjusted associations hold fixed only the listed columns.
- Intervals and tests do not correct for missing, duplicated or unrepresentative records.
- Rows are assumed independent in row-level tests; repeated rows per customer would make intervals too narrow.
  There is no clustering correction.
- Newey–West errors are approximate for short series; Pettitt's p-value is approximate and assumes one shift.
- Less than two years of data: seasonality across the year cannot be separated from a period change.

## API and UI

- `GET /api/investigations/{id}/statistics` — all results with analysis and hypothesis ids, multiplicity,
  settings. `GET /api/investigations/{id}` — report gains `statistics`, `statistical_evidence`,
  `practical_significance`, `multiplicity`, `how_to_read_statistics`.
- Investigations page: Observed result, Statistical evidence, Practical significance, an expander with the
  technical table and the reasoning behind each method, How to read these statistics.

## Configuration

`HYP_SIGNIFICANCE_ALPHA` (0.05), `STAT_CONFIDENCE_LEVEL` (0.95), `STAT_SEED`, `STAT_MIN_SERIES_POINTS` (6),
`STAT_MAX_PAIRWISE_GROUPS` (6), `STAT_MAX_CONTROL_LEVELS` (20), `STAT_ROWS_PER_COEFFICIENT` (10),
`STAT_MULTIPLICITY_METHOD` (holm), `STAT_ATTENUATION_SHARE` (0.5). Dependency added: SciPy.

## Verification status

Offline: `backend/tests/test_statistics.py` and 11 `statistics` evaluation cases. Real model: not
validated for this feature — on 2026-10-12 the Gemini key was rate-limited (429) and the Llama key rejected;
the one attempt made took the fallback path and produced no statistical claims.
