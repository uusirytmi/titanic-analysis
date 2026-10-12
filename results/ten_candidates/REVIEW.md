# Independent review

Pre-run inspection found no critical leakage or stop-rule defect. Two robustness
suggestions were implemented in the current runner: persist exceptions from
summary/output steps, and verify preprocessing/baseline hashes. The actual
executed snapshot is preserved. Differences from the current source are limited
to guarded failure recording, hash checks and --out; numerical model/feature/
metric/stop code is unchanged. No refit was needed for these nonnumerical edits.

Post-run independent audit used a separate implementation to recompute all
10 x 2673 OOF records, 150 fold accuracies and log losses, all candidate seed
means and paired baseline deltas. All match. Strict unrounded >.844 is false
for every candidate. Ten completed candidates and final stop status agree.
The fixed-vote probabilities equal the 2:2:1 average of candidates 1/4/7 to
within 1e-12. Snapshot, preprocessing and split hashes match.

Best: fixed_vote = 0.8368903395894797. Difference from the same-split HGB
reference is -0.000004184713, not an improvement despite rounded equality.
All preprocessing remains inside outer training folds and no validation labels
are used in the fixed ensemble weights or model fitting. No result-invalidating
issue remains. Review was read-only.

A final low-priority suggestion was also implemented: accidental reruns against
completed output now preserve its final status. A thirteenth test verifies this.
