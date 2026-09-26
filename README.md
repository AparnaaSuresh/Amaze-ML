# Amazon ML Challenge 2026: Business Entity Resolution

Team repository for matching noisy business records from Source 2 and Source 3 to the deduplicated Source 1 reference records.

## Repository layout

```text
student_resource/
|-- dataset/                              # local only; ignored by Git
|-- output/                               # generated locally; ignored by Git
|-- utils/validate_submission.py          # official format validator
|-- code/business_entity_resolution/
|   |-- src/                              # team pipeline code
|   |-- README.md                         # reproducibility instructions
|   `-- requirements.txt
|-- Documentation_template.md
`-- README.md                             # official challenge statement
```

## First-time setup

1. Clone this repository.
2. Download the official student resource archive from the challenge portal.
3. Place the six source TSV files and the training ground truth under `student_resource/dataset/train/` and `student_resource/dataset/test/` as described in `student_resource/README.md`.
4. Create and activate a virtual environment.
5. Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r student_resource\code\business_entity_resolution\requirements.txt
```

Never commit the dataset, generated model files, submission outputs, API keys, or passwords. The challenge prohibits external business-identity lookups, so the pipeline must use only the supplied data.

## Team workflow

Before starting work, pull the latest `main` branch and create a focused branch:

```powershell
git pull
git switch -c feature/short-description
```

Commit small, understandable changes, push the branch, and open a pull request for teammate review. Do not commit directly to `main` once branch protection is enabled.

## Submission validation

Run the official validator before every upload:

```powershell
cd student_resource
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

The leaderboard scores `matching_results.tsv` using macro-averaged F0.5. The final package must also contain `candidate_pairs.tsv`, runnable source code, pinned dependencies, and the completed methodology document.

