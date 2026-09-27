Member 3 pairwise classifier
===========================

This module trains a LightGBM binary probability model using the existing 12
features. It does not select a decision threshold or resolve final entities.

For Member 1 integration use `saved_split=split_json` or CLI `--saved-split`.
See [SAVED_SPLIT.md](SAVED_SPLIT.md) for the authoritative partition contract.
The random-split description below applies only when no saved split is supplied.

Input JSONL has one labeled candidate pair per line:

```json
{"pair_id":"p1","source1_id":"a1","source2_id":"b1","source1":{"name":"Example Shop","address":"12 Road","postal_code":"00123","country":"India"},"source2":{"name":"Example Shop","address":"12 Road","postal_code":"00123","country":"India"},"label":1}
```

IDs must be nonempty strings. Labels are integer 0 (nonmatch) or 1 (match).
Duplicate pair IDs, duplicate entity pairs, and inconsistent feature fields for
the same source-qualified entity ID are rejected. Feature fields are strings or
None; missing fields are allowed. Convert NaN upstream and retain postal zeros.
For Python use `CandidatePair` from `pair_data` and `train_classifier` from
`training`. `result.model.score_pairs(...)` accepts unlabeled pairs and returns
IDs and probabilities in the same input order.

Splitting and evaluation
------------------------

A seeded GroupShuffleSplit holds out whole Source 1 entities; the validation
fraction applies to groups, not rows. IDs, labels and records travel together.
Both classes must occur in training; an unsuitable split fails without searching
for a favorable seed. One-class validation returns null ROC AUC and average
precision, while log loss and Brier score remain available.

TF-IDF is fitted once on unique source-qualified records occurring in training
pairs. Validation only transforms features. Source 2 entities can occur in both
splits under this Source 1 grouping policy. If entity aliases or cross-source
connected components also need isolation, the team must supply canonical Source 1
group IDs or agree on a stricter split before production evaluation.

Fixed boosting rounds, seed, deterministic CPU mode and a default single thread
make runs reproducible for the same ordered data and dependency versions. No
early stopping, hyperparameter search, threshold optimization or full-data refit
uses validation data. All probabilities describe the supplied candidate population;
real candidate sampling and imbalance affect their interpretation.

Running in PowerShell from the project root
-----------------------------------------

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m business_entity_resolution.matching.train --synthetic --output models/member3_synthetic_smoke
.\.venv\Scripts\python.exe -m business_entity_resolution.matching.train --input labeled_pairs.jsonl --config training_config.json --output models/member3_production_run
```

The output directory must be new. Optional config JSON overrides `TrainingConfig`
fields, for example `{"num_boost_round":100,"min_data_in_leaf":20,"random_state":42}`.
The synthetic mode tests mechanics only and must not be interpreted as evidence
of matching quality. No real dataset is bundled.

Artifacts and handoff
--------------------

- `model.joblib`: `PairwiseModel`, containing fitted TF-IDF vectorizers, classifier
  and feature order. Load via `training.load_model` using the same package and
  dependency versions. Only load trusted joblib files.
- `lightgbm.txt`: native booster export; it still requires the saved feature pipeline.
- `report.json`: config, effective model parameters, versions, ordered input hash,
  feature names, train/validation IDs, iteration count and threshold-free metrics.
- `validation_predictions.json`: pair IDs, both entity IDs, labels and probabilities
  for Member 4. No hard decisions or optimized threshold are saved.

Integration still needs real labeled candidates, stable canonical entity IDs,
agreed upstream normalization and representative negative sampling. Retain an
independent test set for evaluation after Member 4 selects thresholds; the internal
validation scores are not an untouched final test set. Persist the training input
separately if exact reproduction is required; artifacts contain its hash, not its
full text. Confirm performance on realistic volumes before deploying per-pair
feature extraction.
