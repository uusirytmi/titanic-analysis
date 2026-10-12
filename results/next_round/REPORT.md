# 3 hypotheses: exploratory comparison (2026-10-12)

Common parent `1d0eef8cbc074bcb6332d8c8f957ccf0fa2ed2a3`. Each hypothesis has its
own branch, code, tests and actual results. Master is untouched. The prior OOF
informed these hypotheses: repeated CV here is exploratory, not confirmation on
independent passengers. Historical ~0.844 is a different evaluation and cannot
serve as a like-for-like target.

## Same random-passenger evaluation

All branches use exactly the same stratified 5-fold splits for seeds 20261012,
20261029 and 20261113, and the same HGB reference. Reference OOF probabilities
were verified identical across all three branches. Accuracy below is the mean
of the 15 folds. Delta is paired against HGB within those exact folds; SDs are
sample standard deviations, not confidence intervals. Seeds reuse passengers.

| Branch / candidate | Accuracy | Delta | Fold SD | Seed-mean SD | Log loss | Worse folds / 15 |
|---|---:|---:|---:|---:|---:|---:|
| Common HGB reference | 0.836895 | 0 | 0.022033 | 0.004238 | 0.420217 | 0 |
| Seeds: HGB regularized | 0.835028 | -0.001866 | 0.027063 | 0.002219 | 0.403836 | 9 |
| Seeds: RF family | 0.834647 | -0.002247 | 0.024614 | 0.001279 | 0.413203 | 8 |
| Seeds: LR regularized | 0.827529 | -0.009365 | 0.024687 | 0.002787 | 0.418881 | 10 |
| Groups: HGB companions | 0.827153 | -0.009742 | 0.018720 | 0.004045 | 0.425087 | 13 |
| Groups: RF companions | 0.831647 | -0.005248 | 0.018512 | 0.006232 | 0.409946 | 8 |
| Ensemble: equal average | 0.835390 | -0.001504 | 0.024900 | 0.004274 | 0.402603 | 7 |
| Ensemble: inner-weighted | 0.835022 | -0.001873 | 0.022689 | 0.001133 | 0.405222 | 6 |
| Ensemble: LR interactions | 0.824910 | -0.011985 | 0.027187 | 0.004045 | 0.418857 | 11 |

No added candidate improved mean Accuracy over the common reference. The equal
average improves mean log loss by 0.017614, and regularized HGB by 0.016381.
These are promising for probability quality, not a demonstrated Accuracy win.
Reference seed means range 0.833890–0.841743. Equal-average deltas by seed are
+0.003365, -0.004488, -0.003390; its first-seed improvement does not repeat.
Inner-weighted seed deltas are +0.001117, -0.007840, +0.001105. Its smaller seed SD
is descriptive evidence from only three correlated repetitions.

The inner procedure selected equal weights in 6/15 outer folds, RF-heavy in 4,
LR-heavy in 4, HGB-heavy in 1. It minimizes **inner log loss**, while Accuracy is
the primary outer metric, so those objectives need not move together. It uses
outer-training OOF labels only, never outer validation labels.

## Separate evaluation: unseen family / travel-party proxies

These scores use StratifiedGroupKFold and answer a different question. Do not
rank them against the random-passenger table. The 671 connected components
(534 singletons, maximum size 8) link normalized ticket OR surname plus stated
family size. This is a conservative covariate-based proxy, not verified kinship.
No connected component appears on both sides of an outer split.

| Candidate | Group Accuracy | Delta vs group HGB | Fold SD | Seed-mean SD | Log loss | Worse folds / 15 |
|---|---:|---:|---:|---:|---:|---:|
| HGB reference | 0.819711 | 0 | 0.024704 | 0.007205 | 0.456118 | 0 |
| HGB companions | 0.814091 | -0.005620 | 0.024817 | 0.011226 | 0.464594 | 8 |
| RF family | **0.833161** | **+0.013450** | 0.020188 | 0.006801 | 0.424792 | 1 |
| RF companions | 0.821958 | +0.002247 | 0.023133 | 0.007458 | 0.428103 | 6 |

RF family exceeded group HGB in all three seed means (+0.022435, +0.016861,
+0.001054). This is the clearest robustness hypothesis to retain, with a small
third-seed gain and no independent confirmation. It does not isolate a causal
feature effect because both model family and features differ from HGB.

The companion additions worsened their **own corresponding model**: HGB by
-0.009742 in stratified CV and -0.005620 in group CV; RF by -0.003000 and -0.011203,
respectively. Group-split validation has no matching training companions, so
count features cannot transfer there. Cabin deck, ticket prefix, family bands
and counts were bundled: this run cannot identify which caused the losses.
No target/family-survival encodings were used anywhere.

## Execution, failures and artifacts

Final bounded run: seeds 60 fits / 24.4 seconds; groups 120 fits / 36.6 seconds;
ensemble 180 fits / 41.7 seconds. All 360 fits completed; runtime failures zero.
Branches ran concurrently with one numerical thread each, so timings are
observed wall time, not a controlled model-speed benchmark. Per-candidate
ensemble times include shared component work and are not additive. Branch total
wall time is in its manifest.

A development test exposed floating-point tie-breaking in equal-quality
ensemble weights. A 1e-12 tie rule was added before ensemble execution. The
already completed seeds run was repeated with corrected common code; all its
OOF probabilities were identical. Its 60 extra fits are retained in the seeds
branch under `seeds_pre_fix` with the old plan and source hashes. Total work was
420 fits; only the final 360 are in the comparison. The failed synthetic test
was a software defect, not a failed model evaluation.

Each branch saves its own `manifest.json`, `splits.json`, anonymous `groups.json`,
`fold_scores.csv`, `seed_scores.csv`, `summary.csv`, row-position-only `oof.csv`,
source snapshot, log and `failures.json`. Ensemble additionally saves full inner
OOF and weight-choice losses. `comparison.csv` and `comparison_provenance.json`
are the shared comparison with source hashes and branch identity. Names, tickets,
raw passenger rows and credentials are not added to git.

Eight tests per branch passed, covering original fold-local preprocessing,
training-only companion counts, self exclusion, batch invariance, label
independence, transitive groups, and numerical weight ties. Saved-result audits
recomputed every outer metric, verified group isolation, inner split containment,
OOF completeness, chosen weights and all hashes. Independent review checked the
protocol and results. Its completion-state/exception-stage findings were fixed
before final runs. See `verification.log` and `REVIEW.md`.

No Kaggle submission/scoring, external holdout, demographic fairness audit or
additional seed search was performed. No master merge, charge or credential
change. Next hypotheses worth testing on fresh evaluation are RF family for
unseen groups and regularization/averaging for probability quality; an Accuracy
improvement on random passengers remains unproven.
