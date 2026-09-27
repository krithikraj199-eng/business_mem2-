Feature pipeline
================

`FeaturePipeline` combines the 3 name, 5 address and 4 structured features.
Existing feature functions retain their original behavior.

```python
from business_entity_resolution.features.feature_pipeline import FeaturePipeline, FEATURE_NAMES

pipeline = FeaturePipeline().fit(training_records)
X_train = pipeline.transform(training_pairs)
X_valid = pipeline.transform(validation_pairs)
X_test = pipeline.transform(test_pairs)
```

Integration contract:

- Each record is a mapping with `name`, `address`, `postal_code`, `country`.
  Values must be strings or `None`; omitted fields are missing. Convert tabular
  NaN values to `None` upstream and preserve postal codes as strings, including
  leading zeros. IDs and labels are ignored.
- Each pair is `(left_record, right_record)`. Resolve candidate IDs to records
  upstream. Rows preserve input pair order; align labels with that order.
- Split data before fitting. Supply only training records, preferably once per
  record ID. Both name and address training corpora need usable text. Fit uses
  the same casefold/strip normalization as the existing TF-IDF functions.
- Output is a float64 NumPy array of shape `(number_of_pairs, 12)`; column order
  is `FEATURE_NAMES`. `extract_pair(left, right)` returns a named row.
- Reuse the fitted pipeline for validation, test and inference. Persist it with
  the eventual model and feature order; do not refit on evaluation records.
- Country matching normalizes case and whitespace, but does not resolve country
  aliases (for example, `IN` versus `India`). Agree on upstream normalization.
- Extraction currently processes pairs individually. Benchmark larger candidate
  sets before production use. This class has distinct record-fit and pair-transform
  inputs; it is not a drop-in sklearn Pipeline transformer.

Run from the project root in PowerShell using the existing environment:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m pytest tests\test_feature_pipeline.py -q
.\.venv\Scripts\python.exe -m business_entity_resolution.features.feature_pipeline
```

The runnable example and automated tests use synthetic records only. No classifier
is trained. This project currently has no packaging metadata, so use `src` on
`PYTHONPATH` until the team configures installation.
