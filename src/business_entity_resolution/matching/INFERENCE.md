Saved-model inference
=====================

Postal contract update: mappings may optionally include a `postal_codes` path
alongside the four scalar fields. Its value must be a list/tuple, not serialized
text. Collection inputs require a `postal-collection-v2` model; unversioned models
retain `postal-scalar-v1` behavior and reject collections. See
../features/POSTAL_CONTRACT.md. Both model and pipeline versions are validated.

`InferencePipeline` loads the existing trusted `model.joblib`, including its fitted
name/address TF-IDF vectorizers. It reuses `PairwiseModel.score_pairs`; it never
trains, fits vectorizers, optimizes thresholds, filters candidates or sorts rows.
The saved feature schema and native booster column order must match all 12
training features. Use the training dependency versions and package on PYTHONPATH.

Configurable input adapter
--------------------------

The following is an adapter example, not a proposed final team schema. Member 1
and Member 2 can choose arbitrary field names and nesting. Supply a mapping JSON
with literal key paths. A flat column uses a one-element path. All four feature
fields must be mapped for each side; use null to explicitly mark an unavailable
field. A mapped path that is absent raises an error, preventing silent schema drift.
Values at feature paths must be strings or null. Preserve leading-zero IDs/postal
codes as strings and convert NaN upstream. ID-only candidate tables must first be
joined to the corresponding records by the upstream integration code.

```json
{
  "pair_id": ["candidate"],
  "source1_id": ["left_id"],
  "target_id": ["right_id"],
  "target_source": ["source"],
  "source1_fields": {
    "name": ["left", "name"], "address": ["left", "address"],
    "postal_code": ["left", "postal_code"], "country": ["left", "country"]
  },
  "target_fields": {
    "name": ["right", "name"], "address": ["right", "address"],
    "postal_code": ["right", "postal_code"], "country": ["right", "country"]
  }
}
```

Target source values are `S2` or `S3`. For a file with one target source, omit
`target_source` and set `target_source_value` to `S2` or `S3`. Source-qualified
target IDs distinguish S2 and S3 even when their raw IDs are identical. IDs are
otherwise preserved verbatim. Candidate IDs must be unique across the submitted
set. Duplicate edges `(source1_id, target_source, target_id)` and inconsistent
records for the same entity are rejected, following the existing scorer contract.
Invalid input fails as a whole, with no silently skipped or deduplicated rows.

Python API
----------

```python
import json
from business_entity_resolution.matching.inference import InferencePipeline, InputMapping

with open("mapping.json", encoding="utf-8") as stream:
    mapping = InputMapping(**json.load(stream))
inference = InferencePipeline("models/approved_run/model.joblib")
scores = inference.score(candidate_rows, mapping, batch_size=1024)
```

`candidate_rows` is an iterable of mappings. Empty input returns `[]`. For each
valid row, the API returns exactly one score in input order. Feature calculation
and prediction use bounded batches, but global validation and returned scores
remain in memory; this is not a streaming API. Reuse the loaded instance for
multiple calls. No production training is performed by this module.

File entry point
----------------

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m business_entity_resolution.matching.predict --model models/approved_run/model.joblib --input candidates.jsonl --mapping mapping.json --output scores.jsonl --batch-size 1024
```

Input/output use UTF-8 JSONL; blank input lines are ignored. Output must be a new
file with an existing parent directory. Validation and scoring complete before
the output is created. Empty input produces an empty file. Existing files are
never overwritten. The CLI prints the number of scored candidates.

Member 4 handoff
---------------

Each output object has exactly these fields (an example, not a computed score):

```json
{"pair_id":"candidate-17","source1_id":"001","target_source":"S3","target_id":"009","probability":0.72}
```

`probability` is the saved binary model's positive-class match probability in
[0, 1]. There is no label, threshold, hard decision, rank or final entity assignment.
Member 4 receives these ordered ID-linked rows and owns threshold selection and
final matching. Retain the model run/report alongside the output for provenance.

Before production: finalize upstream field mappings and candidate ID uniqueness,
confirm normalization matches training, and supply the approved real-data model.
The existing synthetic smoke artifact is suitable only for integration checks;
in particular, quality on S1-to-S3 candidates requires real-data evaluation.
