# Frozen three-branch plan, 2026-10-12 (before fits)

Common parent: 1d0eef8cbc074bcb6332d8c8f957ccf0fa2ed2a3. These are exploratory
follow-ups informed by previous OOF results, not independent confirmation.

All branches: 891 labelled rows (same SHA256 as run1); model seed 20261012;
outer split seeds 20261012, 20261029, 20261113, five folds each. Primary metric:
accuracy at p >= .5; secondary log loss. Common reference: run1 HGB interactions
(100 iterations, 15 leaves, minimum leaf 20, l2=1, learning rate .08,
early_stopping=False). Models fit all preprocessing inside the training fold.
No Kaggle/test labels, no target encodings, no family survival lookups.
No choices based on outer labels, no adaptive candidate additions.

1. seeds: StratifiedKFold; HGB reference, HGB regularized (7 leaves, minimum
   leaf 30, l2=5), RF family (run1 settings), LR interactions C=.1. 60 fits.
2. groups: those same stratified splits AND StratifiedGroupKFold, separately.
   Reference HGB, HGB companions, RF family, RF companions. 120 fits.
   Groups are connected components of shared normalized ticket or surname +
   stated family size (surname only when family size >1). Built from X solely
   for splitting, never labels. Conservative proxy, not verified true families.
   Persist only anonymous group integers. Components remain intact in group CV.
   Companion features add cabin deck, ticket prefix, family-size band and
   counts of *other* known training passengers with matching ticket/family key.
   fit records training counts and passenger IDs; transform subtracts self only
   for training IDs. Unseen keys count zero. Validation/test batches never
   contribute counts, and no survival aggregates are used. This explicitly
   models access to labelled-cohort covariates, with a distribution shift under
   new-group evaluation. Counts cannot transfer when a group is wholly unseen.
3. ensemble: same stratified splits. Components HGB reference, RF family,
   LR interactions C=1; equal probability average and one inner-selected average.
   For each outer training fold, generate component OOF with stratified 3-fold
   seed=(outer seed+100+fold). Choose minimum mean inner-fold log loss from just
   four weights in component order HGB/RF/LR: equal, (.6,.2,.2), (.2,.6,.2),
   (.2,.2,.6). Ties use listed order. Refit components on outer training rows.
   180 fits; no outer-labelled weight selection, no stacking or calibration.

Total 360 fits; single thread; persist each fold's probabilities, paired baseline
accuracy deltas, fold and seed variability, timing, failures, selection evidence,
code/data/version hashes, split membership and model parameters. Predictions
contain row positions only, not names or raw passenger records. Save failed and
worsened trials. Compare branches ONLY within matching split protocol/seed/fold;
never rank group scores against random passenger scores. Repeated seeds reuse
passengers and training sets: SD is descriptive, not an independent CI or p-value.
No final model selection claim or promise to exceed historical ~.844 (incompatible
old CV). Use fresh external evaluation for confirmation. No merge or submission.
