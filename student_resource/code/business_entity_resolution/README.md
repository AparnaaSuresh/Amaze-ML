# Business Entity Resolution Pipeline

This directory will contain the reproducible end-to-end solution:

1. inspect and normalize records;
2. split training entities for validation;
3. generate high-recall candidate pairs;
4. create name, address, number, and country comparison features;
5. train and calibrate a matching model;
6. tune decision rules for macro F0.5;
7. generate and validate both required TSV outputs.

All commands and final dependency versions will be documented here as the pipeline is implemented.

