# Business Entity Resolution Pipeline

This is an offline, CPU-first entity-resolution pipeline. It uses only the
challenge TSVs. Generated indexes, model files, reports and candidate metadata
belong under the Git-ignored `artifacts/` directory.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r student_resource\code\business_entity_resolution\requirements.txt
```

Run commands from the repository root. The entry point is:

```powershell
python student_resource\code\business_entity_resolution\src\pipeline.py --help
```

Defaults live in `config/default.json`. Every command persists its resolved
arguments to `artifacts/configs/<command>_latest.json`; CLI flags override the
versioned defaults.

## Reproducible progression

```powershell
# Stage 00: inspect and create the fixed S1 folds
python student_resource\code\business_entity_resolution\src\pipeline.py eda
python student_resource\code\business_entity_resolution\src\pipeline.py split

# Stage 01/02: build a training-target index and final pre-model candidates
python student_resource\code\business_entity_resolution\src\pipeline.py candidates --split train --rebuild --index artifacts\cache\train_targets.sqlite --out artifacts\candidates\train.tsv --meta artifacts\candidates\train.jsonl --budget 40

# Train and evaluate a provisional matcher (sweep threshold/budget in experiments)
python student_resource\code\business_entity_resolution\src\pipeline.py train --index artifacts\cache\train_targets.sqlite --meta artifacts\candidates\train.jsonl
python student_resource\code\business_entity_resolution\src\pipeline.py predict --split train --index artifacts\cache\train_targets.sqlite --meta artifacts\candidates\train.jsonl --model artifacts\models\matcher.joblib --out artifacts\reports\train_matching.tsv --threshold 0.80

# Build test candidates, infer, then validate before upload
python student_resource\code\business_entity_resolution\src\pipeline.py candidates --split test --rebuild --index artifacts\cache\test_targets.sqlite --out student_resource\output\candidate_pairs.tsv --meta artifacts\candidates\test.jsonl --budget 40
python student_resource\code\business_entity_resolution\src\pipeline.py predict --split test --index artifacts\cache\test_targets.sqlite --meta artifacts\candidates\test.jsonl --model artifacts\models\matcher.joblib --out student_resource\output\matching_results.tsv --threshold 0.80
python student_resource\code\business_entity_resolution\src\pipeline.py validate --matching student_resource\output\matching_results.tsv --candidate student_resource\output\candidate_pairs.tsv
```

`candidate_pairs.tsv` is the final, deduplicated input set to the matcher. It
must not be replaced by an earlier, broader blocking set. Select the candidate
policy on the documented F0.5/candidate-size Pareto frontier, not by an assumed
combined score.

`full` runs the provisional baseline path with one configuration. It is not a
replacement for the required fixed-fold budget, route, feature, and threshold
ablations:

```powershell
python student_resource\code\business_entity_resolution\src\pipeline.py full --rebuild --budget 40 --threshold 0.80
```
