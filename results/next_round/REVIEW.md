# Independent review

Pre-run review found no critical leakage. Completion-state and exception-stage
robustness issues were corrected before final runs. A separate synthetic test
found numerical ties, which were fixed and verified as described in DEVELOPMENT.md.

Post-run review independently recomputed (using a separate standard-library
implementation) all 15 ensemble inner log losses, tie-tolerant selections and
outer weighted probabilities. They match. All three split SHA256 values and
common HGB OOF predictions match exactly. Every group train/validation partition
has disjoint connected components. No result-invalidating issue remains.

Interpretation checked: stratified Accuracy did not improve; group RF family
improved by .013450 against group HGB; ensemble log loss improved while Accuracy
fell. These are exploratory, correlated repeated-CV results. README now explicitly
separates historical untested items from the new repeated/group evaluations.
See verification.log for eight passing tests and result audit on this branch.
