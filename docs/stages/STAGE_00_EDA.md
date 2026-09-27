# Stage 00 — EDA and Evaluation

**Status:** Completed on the full supplied dataset.

## Objective

Validate TSV schemas, profile supplied data, implement organizer-compatible
macro-F0.5, and create deterministic S1-level folds.

## Commands

`pipeline.py eda` and `pipeline.py split`

## Required handoff

Commit/configuration ID, data file sizes, report paths, scorer test output,
fold distribution, failures, and the Stage 01 decision.

## Measured results

- All six source files passed the four-column schema check.
- Train row counts: S1 2,206,821; S2 5,034,616; S3 5,285,603.
- Test row counts: S1 1,732,544; S2 4,887,273; S3 5,082,316.
- Names are complete; addresses are missing in 168,967 train-S2 and 175,916 train-S3 rows.
- Ground truth contains 123,247 singleton S1 entities and match multiplicities up to 11.
- Fixed fold sizes are 735,611, 735,606, and 735,604.

Artifacts: `artifacts/reports/eda.json` and `artifacts/splits/folds.json` (ignored locally).
