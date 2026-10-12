# Development corrections

1. Independent pre-run review found that completion was recorded before summary
   generation and exception stages could be misleading. Corrected before all
   final runs; summary failures now persist and complete stays false.
2. Synthetic equal-prediction weight test initially failed: expected first
   weight, received index 2 due to floating-point arithmetic. Use minimum loss
   with deterministic first-listed tie within 1e-12. Eight tests now pass.
3. The seeds run had already completed when that synthetic test result was read.
   It did not use the faulty selection function. Preserve it as seeds_pre_fix
   on the seeds branch and rerun under the final common plan/code anyway.
   OOF frames from the first and corrected seeds runs are exactly identical.
   This is an extra 60 fits, not an additional search or independent replicate.
