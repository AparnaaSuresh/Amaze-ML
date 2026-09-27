# Tracking Report

## Current state

- **Champion:** Not selected.
- **Dataset:** Local and ignored; initial streaming profile completed.
- **Pipeline:** Stage 00 EDA and three-fold split completed; later full-data runs pending.
- **Known risk:** Candidate recall and index size are unmeasured; do not claim a score.
- **Next action:** Run `eda`, `split`, then candidate-budget sweeps on the fixed folds.

## Latest measured facts

| Measure | Value |
|---|---:|
| Train S1 entities | 2,206,821 |
| Train S2 / S3 records | 5,034,616 / 5,285,603 |
| Train singleton S1 labels | 123,247 |
| Test S1 France records | 259,452 |
| Test S2 / S3 records | 4,887,273 / 5,082,316 |
| Train S2 / S3 missing addresses | 168,967 / 175,916 |
| Validation fold sizes | 735,611 / 735,606 / 735,604 |

Update this file only after a completed, reproducible run. Link its configuration
ID and the corresponding row in `EXPERIMENT_REGISTRY.md`.
