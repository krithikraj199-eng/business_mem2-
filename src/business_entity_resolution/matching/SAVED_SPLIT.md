Training with Member 1's authoritative split
===========================================

The existing API accepts optional `saved_split`, a decoded validation_split.json
mapping. It reuses the adapter's SavedSplit validator and the same feature fitting,
LightGBM implementation, evaluation, and persistence as random-split training.

```python
import json
from business_entity_resolution.matching.training import train_classifier

with open('validation_split.json', encoding='utf-8') as stream:
    saved_split = json.load(stream)
# joined comes from Member1Adapter.join(..., ground_truth=complete_lookup,
#                                      require_labels=True)
pairs = [item.ml_pair(collection_aware=True) for item in joined]
result = train_classifier(pairs, saved_split=saved_split)
```

Both positives and negatives are assigned solely by full Source 1 ID. Required
keys are `train_source1_entity_ids` and `validation_source1_entity_ids`. Duplicate
IDs, overlapping partitions, unknown candidate S1 IDs, incorrect pair split labels,
and invalid binary labels raise before fitting. Both candidate partitions must be
nonempty and training must contain both classes. Single-class validation is allowed
with undefined ranking metrics reported as null. Saved IDs without candidates are
allowed; they are not synthesized into candidate rows.

CandidatePair now has an optional `split` field. The Member 1 adapter preserves it
when exporting. Previous positional arguments and JSONL rows without split remain
supported. If pairs carry assignments, saved_split is required rather than silently
ignoring them. Without saved_split and without assignments, the original seeded
GroupShuffleSplit behavior remains unchanged.

CLI (input is existing ML-contract JSONL with joined records, not raw ID-only TSV):

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m business_entity_resolution.matching.train --input labeled_pairs.jsonl --saved-split validation_split.json --output models/new_real_run
```

Optional row `split` is `train` or `validation`; absent/null means no asserted
assignment. `label` must be a JSON integer 0/1. The CLI does not convert string
labels, resolve missing records, or infer negatives. File input is treated as real;
synthetic fixtures are used only by tests when exercising this entry point.

Saved-split mode ignores validation_fraction for partitioning; random_state still
controls LightGBM. Reports identify `member1_saved`, include the canonical saved-ID
partition hash and counts, and retain actual candidate IDs/group IDs in each split.
The report's config.validation_fraction describes configured defaults, not the
effective saved split. Keep the original split JSON for exact provenance.

TF-IDF fits only unique records from training pairs. Validation-only records never
enter its vocabulary. S2/S3 records shared with training may occur in validation,
consistent with the team's Source 1 grouping policy. Postal collections retain
`postal-collection-v2` in both model and report. No threshold selection is added.

Real training remains gated by available normalized records, Member 2 candidate
pairs with stable IDs, verified positive AND negative labels, complete ground-truth
coverage, and the actual saved split. This API checks structural consistency; it
cannot establish the truth of externally supplied labels. No real inputs or model
are substituted with synthetic data by this implementation.
