# Comparison workflow

`backend/app/comparison.py`, node `compare_node` in `backend/app/agents/deterministic.py`, route `comparison`.

Answers "is there a difference?" and "are these related?" questions about two columns of a dataset:

- Is average revenue different between regions?
- Do online orders have a higher unit price than retail?
- Is revenue correlated with units?
- Does the share of returned orders differ by product?

Before this, such questions took the `statistical` route: a model wrote pandas code under a text hint, in a sandbox
image that has no statistics library, and the published evidence recorded statistical validity as `not_assessed`.
Now they are answered by the statistics engine (`backend/app/stats/`) from a validated plan. No code is generated
and nothing runs in the sandbox.

```
question ──> plan ──> validate ──> stats.selector ──> report ──> evidence ──> dashboard
             │           │               │
             │           │               └ chooses the method from the data; a suggested method is recorded
             │           └ rejects unknown columns, wrong types, unsupported designs and methods
             └ deterministic planner, or (optionally) a model: a proposal only
```

## Who decides what

| Step | Decided by | Trusted? |
| --- | --- | --- |
| Which two columns, which groups, paired or not, whether the question asks about cause | The planner | No. It is a proposal. |
| Whether the plan can be carried out on this dataset | `comparison.validate`, then `comparison.run` against the rows | Yes |
| Which statistical method | `stats.selector`, from the data | Yes |
| Every number | `stats.selector`; headline figures recomputed by `comparison.reference` (csv + plain Python) | Yes |
| The wording of the summary | A fixed template over the computed result | Yes |

The planner is the only place a model can take part, and it has one job. That is the slot a fine-tuned model is
meant to fill: it can be swapped in or out without touching validation, method selection, computation or evidence.

## The plan

```json
{
  "outcome": "revenue",              "explanatory": "channel",
  "levels": ["Online", "Retail"],    "outcome_level": null,
  "design": "independent",           "unit": null,
  "estimand": "mean",                "filters": [],
  "method": null,                    "claims_cause": false,
  "assumptions": []
}
```

Only `outcome` and `explanatory` are required. The kind of comparison is not in the plan: it follows from the column
types.

| Outcome | Explanatory | Analysis |
| --- | --- | --- |
| numeric | categorical, two groups | difference between two groups |
| numeric | categorical, more groups | differences across groups, with Holm-adjusted pairwise comparisons |
| numeric | numeric | correlation |
| categorical | categorical | two proportions (2 × 2) or a contingency table |
| categorical | numeric | read the other way round, and the report says so |

`method` is a suggestion. The engine records it as accepted or overridden; a name it does not know is rejected.

## What is rejected, and when

Before anything runs (`validate`): a plan that is not an object or has unknown fields, an unknown column, the same
column twice, a date or free-text column, an unsupported design, estimand or method, levels that are not exactly two
distinct values, more than `STAT_MAX_CONTROL_LEVELS` groups with none named, a paired design without a unit column,
an invalid filter.

Against the rows, still before any statistic (`run`): a named group that does not exist, no rows left after the
filters, a pairing that is not one-to-one.

A run that is refused fails with the reason. A rejected model plan is recorded in the run trace
(`comparison_plan: rejected`, with the code) and the question keeps the route it had.

## Planning

**Deterministic (default).** `comparison.parse` reads a plan from the question's own words. It is narrow on purpose:
the question must be a yes/no comparison ("is … higher than …?") or say so outright ("compare", "related to",
"differ"), name two columns (or one column and two values of another), and contain nothing the planner would have
to ignore: a period, a threshold, "excluding", "top". Requests to list, count or find an extreme are left alone
whatever else they say. A word that could mean two columns produces a clarifying question. Anything it declines
goes to the route it had before.

Checked against real questions: none of the 490 verified single-table Spider validation questions is taken for a
comparison.

**Model (off by default, `COMPARISON_MODEL_PLANNER=true`).** When the deterministic planner declines and the
wording still asks for a comparison, one model call proposes a plan through the `submit_comparison_plan` tool. It is
validated exactly as above. The deterministic planner always goes first, so enabling this never changes how a
question it can already read is answered. No live model was called in building or testing this; the tests use the
scripted provider.

## The result

`report["statistic"]` is the engine's standard result (`stats/result.py`): estimate, interval, test and p-value,
effect sizes, sample sizes and exclusions, the method chosen and why, the methods rejected and why, the assumptions
checked, warnings and limitations. Around it the report adds per-group figures, data checks (rows used, duplicate
rows, rows without a group), and the causal status.

The published summary reports what was measured, then what it does and does not show. It never contains a number
the engine did not compute, and that is checked by the same evidence rule as every other route.

## Association, not cause

Every result carries `causal.status = "not_established"`, and the narrative evidence record carries
`association_not_causal`. When the question asks about cause, effect or impact, the plan is marked `claims_cause`
and the answer gives the association, says it does not show that one variable changes the other, and lists what a
causal answer would need: assignment that does not depend on the outcome, the order of events, and the other
differences between the groups measured.

## Verification

Three separate things, kept apart:

1. **At run time.** Group sizes, group means, the difference, Pearson's r and shares are recomputed by
   `comparison.reference` with the `csv` module and plain Python, and compared with the engine's values.
2. **Offline, against an independent reference.** `backend/evals/build_comparison_fixtures.py` generates 15 seeded
   fixtures and their expected statistics using the standard library only: its own t, F and chi-square distribution
   functions, and full enumeration of permutations where a test is exact. The application computes the same
   quantities through `scipy.stats`. Cases are in `backend/evals/cases/comparison.json` and run in the evaluation lab
   (`python -m app.evals run --component comparison`).
3. **Offline, by simulation.** `python -m app.evals.calibration` draws datasets whose truth is known and counts how
   often a true null is rejected, how often the interval contains the truth, and how often a planted effect is found.

### Results

| Check | Result |
| --- | --- |
| Expected method chosen on every scenario | 15 of 15 |
| Estimate, interval and p-value within 1e-6 of the independent reference | 15 of 15 |
| Invalid plans rejected | 29 of 29 (22 at validation, 3 against the rows, 4 proposed by a model) |
| Numbers in a summary that were not computed | 0 across 20 reports |
| Causal claims in a summary | 0 across 20 reports |
| Tests: false-positive rate, by the criterion for each kind of test | 11 of 11 methods |
| Intervals: no evidence of covering less often than stated | 9 of 9 methods |
| Verified Spider questions whose route changes | 0 of 490 |

### Calibration criteria

One band for every method ("false positives between 3% and 7%") was the first criterion. It cannot be right for
all of them, and it was replaced, not relaxed:

- **Approximate tests** (t, F, chi-square, correlation, large-sample rank tests) claim a false-positive rate of α.
  Criterion: α lies inside the 95% Wilson interval of the rate observed over 4,000 null datasets per design. At that
  size the interval is about ±0.7 points, tighter than the old band.
- **Exact and permutation tests** guarantee a rate of *at most* α and fall below it when the data allow only a few
  distinct p-values. Criterion: no evidence of a rate above α. Where the exact size can be worked out from the design,
  the observed rate must also agree with it; that is what shows the conservatism is the test's and not a defect.
- **Intervals.** Criterion: no evidence of covering less often than the stated 95%. Covering more often is reported
  as conservatism. An interval the engine declines to report is counted, not left out.

The original bands are still evaluated and recorded for every method (last column below), and the first run's report
is kept unchanged in `backend/evals/reports/comparison_calibration_v1.json`.

### Calibration by method

Seed 20261013. Primary run: 400 simulations per design, with and without a planted effect. Confirmation run: 4,000
null simulations per design (seed offset 50,000,000). α = 0.05. Brackets are 95% Wilson intervals.

| Method | Claim | False positives, primary | False positives, confirmation | Coverage | Power | Criteria | Old bands (rate / coverage) |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `chi_square` | approx. | 5.5% (4.2–7.2), n = 854 | 5.2% (4.8–5.7), n = 8,578 | 94.7% (93.2–95.8), n = 1,150 | 70.0% | met | in / in |
| `fisher_exact` | exact | 0.9% (0.3–2.5), n = 346 | 1.2% (0.9–1.6), n = 3,421. Exact size 1.52% | 96.2% (94.0–97.6), n = 450 | 30.8% | met | out / in |
| `brunner_munzel` | approx. | 4.0% (2.5–6.4), n = 400 | 4.8% (4.2–5.5), n = 4,000 | 94.9% (93.1–96.2), n = 800 | 60.5% | met | in / in |
| `paired_t` | approx. | 7.2% (5.1–10.2), n = 400 | 5.1% (4.5–5.9), n = 4,000 | 94.1% (92.3–95.6), n = 800 | 75.5% | met | out / in |
| `pearson` | approx. | 3.9% (2.4–6.2), n = 411 | 4.3% (3.8–5.0), n = 4,053 | 96.3% (94.7–97.4), n = 808 | 84.9% | met | in / in |
| `permutation_anova` | exact | 5.7% (3.9–8.5), n = 401 | 5.5% (4.9–6.3), n = 4,000 | – | 59.8% | met | in / – |
| `permutation_means` | exact | 5.3% (3.5–7.8), n = 438 | 5.2% (4.6–5.9), n = 4,490 | 95.1% (93.5–96.4), n = 881 | 85.3% | met | in / in |
| `sign_flip_permutation` | exact | 5.1% (3.4–7.6), n = 448 | 4.5% (3.9–5.2), n = 4,481. Exact size 4.69% | 95.4% (93.8–96.6), n = 895 | 62.6% | met | in / in |
| `spearman` | approx. | 5.1% (3.4–7.8), n = 389 | 5.2% (4.6–6.0), n = 3,947 | 95.4% (92.9–97.1), n = 392 | – | met | in / in |
| `welch_anova` | approx. | 5.8% (3.9–8.5), n = 399 | 4.3% (3.8–5.0), n = 4,000 | – | 84.0% | met | in / – |
| `welch_t` | approx. | 6.0% (4.1–8.8), n = 400 | 4.3% (3.8–5.0), n = 4,000 | 95.2% (93.5–96.5), n = 800 | 77.5% | met | in / in |

Results are grouped by the method the engine chose, so a design can contribute to more than one row. Power is for
the one planted effect in each design and is not comparable between rows. Spearman's power is not reported: three
planted-effect datasets fell to it.

Reading this table honestly:

- Four designs each rejected exactly 174 of 4,000 null datasets. That looked like a seeding fault and was checked:
  the datasets rejected are different ones (2 to 4 in common out of about 68 in the first 1,500, which is what
  chance gives). It is a coincidence.
- Three t- and F-type tests sit about 1.9 standard errors below 5% in the confirmation run. A further 40,000 null
  datasets per design were run to settle it (`comparison_calibration_large_null.json`, seed offset 200,000,000):
  Welch's t 4.95% (4.74–5.16), paired t 5.06% (4.84–5.27), Pearson 5.10% (4.89–5.32), Welch's one-way test 4.78%
  (4.58–4.99). The first three are calibrated. **Welch's one-way test is slightly conservative**, by about 0.2
  points, in a design with unequal variances; that is on the safe side of α and is left as measured.
- The paired t-test's 7.2% in the first run (29 rejections in 400) was sampling variation.
- In 1 of the 48,000 confirmation datasets the workflow refused the plan: a 12 + 12 table in which no row had the
  outcome value the plan named. It is counted in the report (`plans_refused_by_the_workflow`), not dropped silently.
- "No evidence of" is a statement about these designs and these simulation counts. `permutation_anova` at 5.5%
  (4.9–6.3) is consistent with α and is the row closest to the edge of its criterion.

The table is the current report (`comparison_calibration.json`). Since the previous one (kept as
`comparison_calibration_v2.json`): `brunner_munzel` stands where `mann_whitney` did; a design "two groups, small"
(normal, 12 + 12) was added; and in three designs ("two groups, skewed, small", "two groups, small", "paired, few
pairs") 2,224 samples were measured and reported without a test, so they appear under no method. The last two of
those designs are simulated twice over, so that `permutation_means` and `sign_flip_permutation` are still judged on
at least 340 and 3,400 tested samples. Their rates are therefore rates among samples that were tested.

### What the first run found, and what was done

The first calibration run (kept in `comparison_calibration_v1.json`) failed four of its gates and one reference
check. Each was investigated; none was made to pass by changing a number.

**Spearman interval, reference mismatch — fixed.** The engine used `1.03 / sqrt(n − 3)` for the standard error on the
Fisher z scale. The published variance for a rank correlation is `1.06 / (n − 3)` (Fieller, Hartley and Pearson,
1957), whose square root is 1.0296. The engine now uses it, and the bounds agree with the reference to 1e-6.

**Bootstrap intervals on small samples, 86.0% and 87.2% coverage — replaced.** An independent study
(`backend/evals/study_small_sample_intervals.py`, 2,000 seeded simulations per design, nothing imported from the
application) reproduced the shortfall and compared five interval methods across thirteen designs. Coverage, in
percent, nominal 95:

| Design | Assumption of the inversion interval | Bootstrap BCa (was used) | Percentile | Studentised bootstrap | Welch / paired t | Test inversion (now used) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| two groups 12 + 12, skewed, shift (calibration design) | holds | 86.0 | 90.4 | 91.6 | 96.4 | 94.8 |
| two groups 8 + 20, skewed, shift | holds | 85.4 | 88.8 | 89.8 | 94.2 | 95.0 |
| two groups 6 + 6, normal, shift | holds | 89.2 | 89.1 | 95.1 | 95.5 | 95.1 |
| two groups 10 + 10, very skewed, shift | holds | 81.7 | 89.8 | 92.5 | 98.5 | 95.8 |
| two groups 5 + 5, very skewed, shift | holds | 83.7 | 87.7 | 97.6 | 98.6 | 95.8 |
| two groups 12 + 12, skewed, rescaled | **violated** | 84.4 | 88.3 | 89.6 | 93.8 | 92.0 |
| two groups 8 + 8, exponential, rescaled | **violated** | 86.4 | 87.5 | 91.8 | 94.0 | 92.3 |
| two groups 16 + 8, normal, unequal spread | **violated** | 87.9 | 88.1 | 94.0 | 94.3 | 82.0 |
| 8 pairs, normal differences (calibration design) | holds | 89.1 | 89.3 | 94.6 | 94.8 | 95.0 |
| 12 pairs, heavy-tailed symmetric differences | holds | 87.6 | 90.3 | 93.0 | 95.8 | 95.2 |
| 5 pairs, normal differences | holds | 84.2 | 84.7 | 95.1 | 95.8 | unbounded |
| 10 pairs, skewed differences | **violated** | 84.9 | 82.6 | 91.1 | 86.8 | 87.5 |
| 8 pairs, very skewed differences | **violated** | 75.4 | 71.1 | 88.1 | 76.3 | 77.8 |

What that showed:

- The bootstrap interval fell short in every design, 75% to 89%. A dozen values cannot stand in for the tails of
  their distribution however often they are resampled.
- No method holds 95% in every design. That is the known position for a mean without assumptions, not a gap in the
  candidates.
- Welch's interval looked best for two groups, then failed further tests (`welch_interval_stress.json`): 89.5%,
  84.7% and 83.7% when a small skewed group sits beside a larger, differently shaped one.
- Declining to give an interval when the sample looks skewed made coverage worse among the intervals that were given
  (as low as 62%), because it selects unrepresentative samples. It is not done.
- For skewed paired differences nothing reached 93% even with 80 pairs (`paired_interval_by_sample_size.json`).

What the engine now does, on the permutation path only:

- It reports the interval **the permutation test itself implies**: every difference that test does not reject. Under
  the assumption the p-value already rests on (the groups differ only by a shift; paired differences are symmetric)
  it is exact, and it held the true value in 94.8% to 95.8% of samples in every design where the assumption held.
  The interval excludes "no difference" exactly when the exact test rejects (300 of 300 seeded cases, for groups
  and for pairs).
- The assumption is part of the interval's name, appears as an unchecked assumption in the result, and is quoted
  with the interval in the published summary. The result's limitations give the measured coverage when the
  assumption fails: 92%, 82%, and 78% to 88%. Unequal group sizes add a warning.
- When no bounded interval exists (5 pairs or fewer, 3 + 3 groups) or the sample is over 5,000 observations, **no
  interval is reported**, the result says why, and nothing is put in its place. The estimate and the test stand.
- The bootstrap interval is listed among the rejected methods, with the reason.

This is a choice with a cost: when the assumption fails, the interval is too narrow, and a small sample cannot
reveal that. The alternative considered was to report no interval at all on this path.

**Fisher's exact test, 0.9% false positives — the criterion was wrong, not the test.** On a 12 + 12 table with a
true share of 0.30 the test's exact size, by enumeration of all 169 pairs of counts, is 1.52%. It cannot reject 5%
of the time. Observed over 3,421 datasets: 1.2% (0.9–1.6), which agrees.

**Paired t-test, 7.2% — sampling variation.** See above: 5.06% over 39,998 datasets.

A defect unrelated to calibration surfaced on the way: the count of possible arrangements was converted to a float,
which overflows above about 1,030 observations, so a large skewed sample raised an error on the permutation path.
It is fixed and tested.

### Audit of the permutation, inversion and rank-test paths

A second pass looked at the tests themselves, not only their intervals: `backend/evals/audit_permutation_paths.py`.
It calls the engine on simulated data in which the quantity each method estimates is truly zero. Seed 20261201;
simulation i of design d uses seed 20261201 + 100,000 d + i; 3,000 simulations per design (2,000 for the rank
tests). Brackets are 95% Wilson intervals. "Before" is the engine as it stood (`permutation_audit_before.json`),
"after" is the engine now (`permutation_audit.json`). **These figures describe these designs and nothing else.**

**Two independent groups, permutation path.** False positives at α = 0.05, and how often the interval held the
true difference:

| Design | False positives, before | False positives, after | Coverage, before | Coverage, after |
| --- | --- | --- | --- | --- |
| normal 12 + 12, equal spread | 5.4% (4.7–6.3) | 5.4% (4.6–6.3) | 94.5% (93.7–95.3) | 94.6% (93.7–95.4) |
| normal 6 + 18, equal spread | 5.2% (4.4–6.0) | 5.4% (4.6–6.3) | 94.8% (93.9–95.5) | 94.7% (93.8–95.4) |
| lognormal 12 + 12, same shape | 5.6% (4.8–6.4) | 5.6% (4.8–6.5) | 94.5% (93.6–95.2) | 94.5% (93.6–95.2) |
| normal 12 + 12, spread 5 vs 20 | 6.1% (5.3–7.0) | 6.1% (5.3–7.0) | 93.9% (93.0–94.7) | 93.9% (93.0–94.7) |
| normal 8 + 24, larger group has spread 20 (smaller 5) | 0.2% (0.1–0.5) | 3.8% (3.2–4.6) | 99.8% (99.6–99.9) | 96.2% (95.5–96.9) |
| normal 24 + 8, smaller group has spread 20 (larger 5) | **21.3%** (19.8–22.8) | 6.7% (5.8–7.6) | 78.8% (77.3–80.3) | 93.3% (92.3–94.1) |
| normal 30 + 10, smaller group has spread 20 (larger 5) | **21.7%** (20.2–23.2) | 6.4% (5.6–7.4) | 78.3% (76.8–79.7) | 93.6% (92.7–94.4) |
| normal 15 + 5, smaller group has spread 20 (larger 5) | **20.9%** (19.5–22.4) | 8.2% (7.3–9.3) | 79.0% (77.5–80.5) | 91.7% (90.7–92.7) |
| normal 16 + 8, smaller group has spread 20 (larger 10) | **11.0%** (10.0–12.2) | 6.3% (5.5–7.3) | 88.9% (87.7–90.0) | 93.6% (92.7–94.4) |
| normal 5 + 5, spread 5 vs 20 | 7.6% (6.7–8.6) | 7.6% (6.7–8.6) | 92.4% (91.4–93.3) | 92.4% (91.4–93.3) |
| lognormal 8 + 20, equal means, larger group more skewed | 6.3% (5.5–7.3) | 7.1% (6.2–8.0) | 93.5% (92.5–94.3) | 92.9% (91.9–93.8) |
| lognormal 20 + 8, equal means, smaller group more skewed | **23.6%** (22.1–25.2) | **19.0%** (17.7–20.5) | 76.7% (75.2–78.2) | 81.0% (79.6–82.4) |
| exponential 8 vs normal 20, equal means | **23.5%** (22.0–25.0) | **12.5%** (11.4–13.7) | 76.8% (75.3–78.3) | 87.6% (86.4–88.7) |

What was wrong, and what changed:

- The test permuted the raw difference in means. That is exact when the groups are exchangeable, and badly wrong
  when a smaller group is the more variable one: a fifth of true nulls rejected.
- It now permutes **Welch's t**. Exact in the same cases, and in these designs within 4% to 8% when normal groups
  differ two- to fourfold in spread.
- The two-sided rule changed from "twice the smaller tail" to **"at least as extreme in absolute value"**. Both are
  exact for exchangeable groups. With unequal spread a single sample's permutation distribution is lopsided, and the
  first rule takes the favourable side each time: 13.0% against 7.5% in the 24 + 8 design. (This was found because
  an independent implementation of the "same" test disagreed with the engine.)
- The interval is the inversion of that same test, so it moved with it. It still excludes "no difference" exactly
  when the exact test rejects (300 of 300 seeded cases, for equal and for unequal group sizes).

What remains, unfixed: **a small, strongly skewed group beside a differently shaped one.** 19.0% and 12.5% false
positives, 81.0% and 87.6% coverage. An independent comparison (same file, "alternatives") shows Welch's t-test and
the permuted Welch t fail there too (19.1% and 18.9%; 11.2% and 12.4%). Nothing tested fixes it. Every result on this
path states the assumption as unchecked and carries these figures in its limitations; unequal group sizes add a
warning. A two-group result with 5 against 15 values and a fourfold spread ratio (8.2%) is the worst of the
normal designs.

**Several groups, permutation path.** The same defect and the same repair: Welch's F is permuted instead of the
classic F.

| Design | False positives, before | False positives, after |
| --- | --- | --- |
| 3 groups of 8, normal, equal spread | 4.8% (4.1–5.6) | 4.7% (4.0–5.5) |
| 3 groups 20 + 20 + 6, smallest has spread 20 (others 5) | **31.4%** (29.8–33.1) | 7.4% (6.5–8.4) |
| 3 groups 6 + 6 + 20, largest has spread 20 (others 5) | 0.3% (0.2–0.6) | 4.2% (3.5–4.9) |

Skewed groups were not simulated for more than two groups.

**Paired observations, sign-flip path.** A different assumption from the independent-groups case: not "same shape"
but "each unit's change is symmetric about its centre". Unchanged by this audit, because nothing does better:

| Design | Assumption | False positives | Coverage |
| --- | --- | --- | --- |
| 8 pairs, normal | holds | 4.3% (3.7–5.1) | 95.7% (94.9–96.3) |
| 12 pairs, heavy-tailed symmetric | holds | 4.8% (4.1–5.7) | 95.2% (94.3–95.9) |
| 8 pairs, skewed | violated | **13.8%** (12.6–15.0) | 86.2% (85.0–87.4) |
| 12 pairs, skewed | violated | **13.1%** (11.9–14.3) | 86.9% (85.7–88.1) |
| 20 pairs, skewed | violated | **11.6%** (10.5–12.8) | 88.4% (87.2–89.5) |
| 40 pairs, skewed | violated | **9.9%** (8.8–11.0) | 90.0% (88.8–91.0) |
| 15 pairs, mostly small losses and a few large gains | violated | **9.6%** (8.6–10.7) | 90.3% (89.2–91.3) |

With skewed changes whose true mean is zero, the test rejects twice to three times as often as it should. The
result says so. In the 40-pair design, 7 of 3,000 samples happened to look symmetric enough for the paired t-test
instead, and 5 of those 7 rejected: choosing the method from the data selects unrepresentative samples. Seven runs
prove nothing, but the mechanism is real and applies to every data-dependent choice of method in the engine.

**Mann-Whitney (typical values, two groups).** Reviewed on its own; it does not inherit anything from the
permutation interval. Its interval is still a bootstrap of the difference in medians.

| Design | Is the test's null true? | Rejections | Interval coverage |
| --- | --- | --- | --- |
| lognormal 8 + 8, identical | yes | 4.5% (3.7–5.5) | 95.6% (94.6–96.4) |
| lognormal 15 + 15, identical | yes | 4.3% (3.5–5.3) | 95.8% (94.8–96.5) |
| lognormal 30 + 30, identical | yes | 4.3% (3.5–5.3) | 95.2% (94.1–96.0) |
| lognormal 10 + 30, identical | yes | 4.7% (3.9–5.7) | 94.3% (93.2–95.2) |
| ratings 1 to 5, 20 + 20, identical | yes | 5.5% (4.6–6.6) | 99.1% (98.6–99.4) |
| normal 12 + 12, same median, spread 1 vs 4 | yes | 7.0% (5.9–8.1) | 94.2% (93.0–95.1) |
| normal 24 + 8, same median, smaller group spread 4 | yes | **14.0%** (12.5–15.6) | 93.0% (91.8–94.0) |
| lognormal 20 + 8, same median, smaller group more skewed | no | 11.7% (10.3–13.1) | 93.7% (92.5–94.6) |
| lognormal 8 + 8, shifted by 10 | no | 20.7% | 95.5% (94.4–96.3) |
| lognormal 15 + 15, shifted by 10 | no | 34.4% | 95.7% (94.7–96.5) |
| lognormal 30 + 30, shifted by 10 | no | 62.6% | 95.5% (94.5–96.3) |
| normal 6 + 6, shifted by 8 | no | 19.6% | 95.1% (94.1–96.0) |

- The test is sound when the groups have the same shape (4.3% to 5.5%) and not when a smaller group is more
  variable (14.0%). **Not fixed.** The Brunner-Munzel test, built for that case, gave 5.7% there and 5.3% to 5.9%
  elsewhere (`rank_test_alternative.json`); replacing Mann-Whitney with it is the recommended next change.
- The bootstrap interval for the difference in medians held 93.0% to 99.1%. It is short of 95% in the two
  unequal-spread designs with unequal sizes (93.0%, 93.7%) and adequate in the rest. **Not replaced**: no
  distribution-free interval for a difference in medians was available to put in its place.
- Both limits are now stated in every Mann-Whitney result.
- **Superseded.** Mann-Whitney has since been replaced by Brunner-Munzel; see *Method decisions (report version 3)* below.

**Wilcoxon signed-rank (typical change, paired).** Also reviewed on its own.

| Design | Is the test's null true? | Rejections | Coverage, bootstrap (before) | Coverage, order statistics (after) |
| --- | --- | --- | --- | --- |
| 8 pairs, normal, median 0 | yes | 4.2% (3.5–5.2) | 92.1% (90.8–93.2) | 98.9% (98.3–99.2) |
| 15 pairs, normal, median 0 | yes | 5.6% (4.7–6.7) | 92.2% (91.0–93.3) | 96.2% (95.3–97.0) |
| 30 pairs, normal, median 0 | yes | 5.0% (4.1–6.0) | 93.6% (92.4–94.6) | 95.4% (94.4–96.2) |
| 8 pairs, normal, median 3 | no | 38.2% | 92.8% (91.5–93.8) | 99.2% (98.7–99.5) |
| 15 pairs, normal, median 3 | no | 73.6% | 92.5% (91.3–93.6) | 96.7% (95.8–97.4) |
| 30 pairs, normal, median 3 | no | 97.8% | 93.5% (92.4–94.5) | 95.2% (94.2–96.1) |
| 12 pairs, skewed, median 0 | no | 6.6% (5.5–7.7) | 93.5% (92.3–94.5) | 96.2% (95.3–97.0) |
| 30 pairs, skewed, median 0 | no | 14.0% (12.5–15.6) | 95.0% (93.9–95.8) | 96.3% (95.4–97.0) |
| 20 pairs, whole numbers −2 to 3 | no | 18.2% | 98.4% (97.8–98.9) | 99.8% (99.4–99.9) |

- The bootstrap interval for the median fell short for plain normal changes at every size tested. It is replaced by
  the **order-statistic interval**: the k-th smallest and k-th largest changes bracket the median with a probability
  that is a binomial sum, whatever the distribution. Coverage is at least the stated level, with no assumption about
  shape; it is conservative for few pairs (98.9% at 8). With 5 pairs or fewer no such interval reaches 95%, and none
  is reported.
- The test's null was stated as "units are as likely to go up as down". That is not what the signed-rank test
  tests: with skewed changes whose median was zero it rejected 14.0% of the time at 30 pairs. The null is now
  stated as "the changes are symmetric about zero", and the result carries that limitation.

**How a missing interval is represented.** Every result now has an interval or an `interval_unavailable` reason,
enforced when the result is built: never neither, never both.

| Where | What it shows |
| --- | --- |
| Result (`StatResult`) | `interval: null`, `interval_unavailable: "<reason>"`; the same sentence in `warnings` |
| Effects inside the result | the mean-difference effect carries no interval of its own |
| Published summary | "No interval is reported: <reason>." and no interval quoted |
| Evidence record of the summary | limitation `interval_not_available` |
| Dashboard figures | none stands in for an interval |
| Investigations page, compact views, model briefs | the reason, in place of "—" or an empty range |

Two views would have printed an empty interval ("0% interval — to —") if given a result without one; both now say
there is none and why. A paired comparison also published the wrong dashboard figures (it was handled as a table);
it now has its own figures and its own independent recomputation.

### Method decisions (report version 3)

`backend/evals/study_method_decisions.py` → `backend/evals/reports/method_decisions_v3.json` (seed 20270101, 2,000
simulations per design or cell, every rate with a 95% Wilson interval). Earlier reports are unchanged;
`comparison_calibration_v2.json` is the calibration report as it stood before these decisions.

**1. Brunner-Munzel replaces Mann-Whitney for a question about typical values.** The result already stated the
null "values in the two groups are equally likely to be higher", reported the difference in medians as the
estimate and the probability of superiority as the effect. Mann-Whitney tests that null only when the groups have
the same shape; Brunner-Munzel tests exactly it. So the question, the estimate, the interval and the effect size
are unchanged, and only the p-value's validity changes. Neither is a test of medians, and the result says so.

| Design (null true) | Mann-Whitney | Brunner-Munzel, t | Brunner-Munzel, permutation |
| --- | --- | --- | --- |
| identical groups, 5 to 30 each | 3.0% to 6.2% | 4.1% to 8.5% | 3.6% to 5.9% |
| same centre, spread 1 vs 4, 24 + 8 | 13.8% | 5.6% | 6.0% |
| same centre, spread 1 vs 4, 30 + 10 | 14.4% | 5.6% | 6.0% |
| same centre, spread 1 vs 4, 15 + 5 | 9.3% | 7.5% | 7.3% |
| same centre, spread 1 vs 3, 60 + 20 | 13.5% | 5.1% | 5.5% |
| lognormal, same median, shape 0.4 vs 1.2, 20 + 8 | 11.7% | 6.0% | 6.4% |
| ratings, flat vs peaked, 12 + 36 | 10.7% | 5.0% | 5.0% |

Power with a real shift was comparable (normal 15 + 15: 52.0%, 55.5%, 53.9%). The t form is liberal with few
values (8.2% at 5 + 5), so the engine uses the permutation form when the smaller group has fewer than 30 values
and the t form from 30. The engine's implementation matches SciPy's statistic and t p-value, and an exact
standard-library enumeration in two reference fixtures (one with ties and unequal sizes).

**2. A mean comparison on a sample that is itself too skewed for its size is measured, not tested.** No test of
a difference in means is valid in small samples without an assumption about shape, and the sample cannot check
it. Whether the measured skewness is low is *not* used as evidence that the assumption holds: among samples from a
strongly skewed group that happened to look symmetric, the permutation test rejected a true null in up to 44% of
cases. What is used is the opposite, as evidence against: when a sample fails Cochran's rule on its own measured
skewness (n < 25 × skewness²), the result has `status: unavailable`, no test, no interval, no significance, and
`inference_unavailable = {code: too_skewed_for_sample_size, summary, reason, supported_instead}`. The estimate,
the group descriptives and the sample sizes are kept. The question the data does support (which group tends to be
higher; the median change) is named and not run in its place.

**3. The complete workflow, selector and test together** (null true; "of all" counts a sample reported without a
test as not rejected, which is the workflow's error rate; "of tested" is among samples given a p-value):

| Design | Rule off (as before): of all | Engine: of all | Engine: of tested | Cells above 7.5%, before → now |
| --- | --- | --- | --- | --- |
| Two groups, 84 cells (7 families × 12 size pairs, 5 + 5 to 80 + 80) | 2.8% to 24.1% | 0.0% to 6.0% | 0.0% to 43.5% | 30 → 0 |
| Pairs, 42 cells (6 families × 5 to 80 pairs) | 0.0% to 17.4% | 0.0% to 9.9% | 0.0% to 65.8% | 10 → 1 |

The remaining cell is "mostly small losses, a few large gains" at 8 pairs (9.9%): one sample in six contains no
large gain and looks symmetric. Coverage where an interval was given is in the report per cell; in the families
that violate the assumption it is poor among the few samples still tested (as low as 56.5% and 34.2%).

The price, measured with a shift of 0.8 standard deviations:

| Design | Size | Power before | Power now (of all samples) | Reported without a test |
| --- | --- | --- | --- | --- |
| normal groups | 5 + 5 / 12 + 12 / 20 + 20 / 30 + 30 | 19.7% / 46.4% / 70.0% / 85.5% | 2.6% / 24.9% / 59.9% / 83.4% | 87% / 45% / 16% / 3% (null) |
| lognormal groups, same shape | 12 + 12 / 30 + 30 / 80 + 80 | 68.5% / 89.5% / 99.2% | 1.7% / 0.8% / 1.1% | nearly all |
| normal changes | 8 / 12 / 20 / 30 pairs | 48.7% / 71.0% / 92.5% / 99.1% | 25.4% / 52.2% / 85.7% / 97.9% | 44% / 26% / 8% / 2% (null) |

Skewed groups of the same shape are the case the old test handled exactly; they are indistinguishable, from the
sample, from the case where it failed at 24%, and are now reported without a test of means.

For a question about typical values the same grid was run through the selector, on the families whose null is
true for those tests (identical groups, equal centres with unequal spread, tied ratings; symmetric changes):
Brunner-Munzel rejected in 3.3% to 7.8% of samples across 48 cells (7.8% at 5 + 15 with unequal spread, the only
cell above 7.5%), with power close to the mean tests' (normal 12 + 12: 44.8%; lognormal 12 + 12: 86.2%); the
median-difference interval held 93.2% to 99.9%; Wilcoxon rejected in 0.0% to 5.5% and its interval held 95.5% to
100%. Six families with equal means but unequal typical values are not a null for these tests and are left out.

Zero variance: both groups constant gives no test and the exact difference; one constant group is tested by
permutation; a single value in a group gives no test.

### Methods version

These changes alter published numbers, so `METHODS_VERSION` in `backend/app/stats/result.py` is now `2`. It is
part of the evidence context of a hypothesis investigation. The consequences, from the code:

- Nothing stored is rewritten. A finished investigation keeps its results and its `methods_version: "1"` label and
  still reads as complete.
- When an investigation made under version 1 is run again, or an interrupted one is resumed, its earlier analyses
  are renamed and marked superseded (kept on record, never reused) and everything is computed again under version 2
  (`agents/hypothesis.DbStore.begin`).
- Affected numbers: Spearman intervals (by about 0.0001); p-values and intervals of the permutation tests of means
  (two groups and several groups), which now permute Welch's statistics; the interval beside the sign-flip test; and
  the interval beside the Wilcoxon test. Estimates are unchanged. All of it is one version step, 1 to 2: none of it
  has been released separately.
- Also in version 2: a question about typical values is tested with Brunner-Munzel instead of Mann-Whitney
  (different p-value; same estimate, interval and effect size), and a mean comparison on a sample too skewed for its
  size has no p-value or interval at all where it had a permutation one. In a hypothesis investigation such an
  analysis is inconclusive ("measured, but it was not tested") and is not counted in the family of tests that
  p-values are adjusted across.

## Reproducing

```bash
python backend/evals/build_comparison_fixtures.py                # fixtures and reference values (standard library)
cd backend
python -m app.evals run --component comparison --no-store        # 15 reference cases
python -m app.evals.calibration --json evals/reports/comparison_calibration.json    # about 5 minutes
python evals/study_small_sample_intervals.py                     # the interval study, about 3 minutes
pytest tests/test_comparison.py                                  # about 75 seconds
```

The tests rerun the whole primary calibration and two designs of the confirmation run from their seeds and compare
them with the stored report, so the figures above cannot drift from the code. The other ten confirmation designs
and the 40,000-simulation run are regenerated by the commands above, not by the test suite.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `COMPARISON_ROUTE_ENABLED` | `true` | Off: these questions take the `statistical` route as before |
| `COMPARISON_MODEL_PLANNER` | `false` | On: a model may propose the plan when the deterministic planner declines |

Significance level, confidence level, seed and group limits are the statistics engine's (`STAT_*`,
`HYP_SIGNIFICANCE_ALPHA`).

## Limits

- Two columns only. No adjustment for other variables, no interactions, no time: a question with a period goes
  elsewhere.
- Rows are treated as independent. Repeated rows from one customer, order or day make an interval too narrow; this
  is stated as an unchecked assumption, and exact duplicate rows are counted and reported.
- The deterministic planner finds a group's values only among each column's five most common ones.
- The run-time reference recomputes headline figures, not test statistics or intervals; those are covered offline.
- Calibration covers the designs listed. It says nothing about data unlike them.
- An interval beside a permutation test assumes a shift (or symmetric differences). When that fails it is too
  narrow, and the sample cannot show it. The summary quotes the assumption with the interval for that reason.
- A mean comparison on a sample that is itself too skewed for its size is measured and not tested. This costs
  power: 87% of normal 5 + 5 samples, 45% of 12 + 12 and 16% of 20 + 20 are reported without a test, and skewed
  groups of the same shape, where the permutation test was exact, are now almost never tested.
- A small sample that does not look skewed is still tested, under an assumption it cannot check. When the data was
  in fact strongly skewed, the samples that got a p-value were wrong far more often than 5% (up to 44% for two
  groups, up to 66% for pairs). Counted over all samples the workflow stayed at or below 6.0% (two groups) and
  9.9% (pairs; one design at 8 pairs is above the 7.5% criterion).
- The same rule has not been applied to, or measured for, comparisons of more than two groups.
- Brunner-Munzel is close to its level, not exact: 3.3% to 7.8% through the selector in the designs tested (one
  cell above 7.5%: 5 + 15 with the smaller group three times as variable). The bootstrap interval for the median difference held 93% to 99%.
- Method choice depends on the data (Cochran's rule). Selecting a method from a sample can select unrepresentative
  samples; this is now measured for two groups and pairs (`method_decisions_v3.json`), not for other designs.
- Permutation tests for more than two groups were simulated with normal data only.
