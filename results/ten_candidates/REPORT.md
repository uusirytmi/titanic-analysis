# Ten prespecified candidates: target not reached

All ten new candidates completed, in the planned order. Stop after trial 10:
**no unrounded primary accuracy exceeded 0.844**. No candidate was omitted.
One trial = one fixed feature/model/configuration evaluated on all five folds
for each of seeds 20261012, 20261029, 20261113. Primary = the equal-weight mean
of the three seed means of fold Accuracy (probability >= .5). All candidates
used fold-fitted interactions features. Group CV is not part of this decision.

The plan and full parameters were saved in `plan.json` before any fit. The split
file is byte-identical to the prior three-seed experiment. Candidate order and
settings were not changed after seeing results. No inner parameter search or
outer-label ensemble weighting was performed; candidate 10 used fixed weights
2:2:1 for candidates 1, 4, 7, refitted separately in each outer training fold.
SVC probability estimation uses only outer-training data.

| Trial | Candidate | Primary Accuracy | Delta vs prior HGB | Fold SD | Seed SD | Worse folds / 15 |
|---|---|---:|---:|---:|---:|---:|
| 1 | gb_depth3 | 0.833904547 | -0.002989978 | 0.027571 | 0.003005 | 7 |
| 2 | gb_depth2 | 0.829410165 | -0.007484360 | 0.024105 | 0.001948 | 8 |
| 3 | gb_leaf10 | 0.835398489 | -0.001496035 | 0.029000 | 0.004216 | 8 |
| 4 | rf_depth8 | 0.832783044 | -0.004111481 | 0.025579 | 0.006838 | 7 |
| 5 | rf_leaf2 | 0.827175528 | -0.009718997 | 0.025178 | 0.006835 | 8 |
| 6 | extra_trees | 0.813694474 | -0.023200050 | 0.026405 | 0.008961 | 14 |
| 7 | svc_c3 | 0.831278639 | -0.005615885 | 0.025271 | 0.004635 | 9 |
| 8 | svc_c10 | 0.831274454 | -0.005620070 | 0.022498 | 0.002309 | 8 |
| 9 | hgb_small_leaves | 0.833525830 | -0.003368694 | 0.030454 | 0.007325 | 8 |
| 10 | fixed_vote | 0.836890340 | -0.000004185 | 0.026846 | 0.003413 | 7 |

Best: trial 10, **0.8368903395894797** (83.6890%),
short of the target by 0.7110 percentage points.
Seed means range 0.833933840–0.840625196; fold SD 0.026845725,
seed-mean SD 0.003412908. These SDs are descriptive, not confidence intervals:
the repetitions reuse passengers and overlapping training sets.

The prior HGB reference scored 0.8368945243027223 on these exact splits, so even
the best new candidate is very slightly lower in Accuracy. Its mean log loss
is 0.398405873 (prior HGB 0.420217239), suggesting better probability
quality, but the requested Accuracy target remains unmet. ExtraTrees was the
weakest candidate; its result is retained along with every worsened trial.

Ten candidates, 150 outer evaluations, 180 component pipeline fits including
the fixed vote. Measured total candidate time 150.8 seconds.
Technical runtime failures: **0**. Model reruns: **0**. Target hits: **0**.
No additional candidate was launched after number 10. The deliberate synthetic
failure in the failure-recording unit test is not a model failure or extra trial.

An independent review identified missing summary-exception recording and extra
hash checks. These were hardened in the current runner without changing the
executing process, models, preprocessing, split membership, predictions, metric,
threshold or candidate order. The original executed source is `runner.py`, whose
hash matches the frozen plan; current source additionally has a guarded runner,
preprocessing/baseline hash checks and a configurable fresh output path. No
numerical rerun was required. Baseline and preprocessing hashes were verified
against the committed parent after execution and recorded in `verification.json`.

Thirteen tests pass, including strict unrounded threshold, all-15-fold requirement,
fixed candidate count/weights, fold-local preprocessing and failure persistence.
The OOF audit recomputes all 150 fold metrics from the local labels, all seed
means/SDs, paired differences and stopping decisions and checks hashes, row
coverage and split isolation. Independent review separately checks saved
predictions, fold-local fitting and the stop rule; see REVIEW.md.

This is further exploratory reuse of inspected data, not independent evidence
of generalization. Historical ~0.844 used a different protocol. No external
Titanic answer table, target aggregates, family-survival lookup, Kaggle submission,
external holdout, group-CV score substitution, master merge, billing or credential
changes were used. Original passenger CSVs remain untracked.

Reproduce in a fresh output directory after installing experiments/requirements.txt
and placing the same raw input in data/train.csv:

```bash
python -m unittest discover -s tests -v
python -m experiments.ten_candidates prepare --out results/ten_candidates_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.ten_candidates run --out results/ten_candidates_reproduction
python -m experiments.audit_ten  # audits the committed original results
```
