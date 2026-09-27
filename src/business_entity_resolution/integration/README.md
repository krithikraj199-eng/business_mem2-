Member 1 compatibility adapter
==============================

Collection export is now available via `collection_aware=True` on `ml_record`,
`ml_pair`, `inference_row`, and `inference_mapping` (as applicable). It preserves
all codes for the v2 feature contract. The scalar bridge below remains the default
for existing callers. See ../features/POSTAL_CONTRACT.md for version requirements;
the earlier collection-feature prerequisite below is now implemented, but a real
v2 model is still required. Saved-split training is documented in
../matching/SAVED_SPLIT.md.

This isolated adapter consumes records and artifacts from Member 1's audited
commit `9c45750d48889efa37ce03e93523efccd3d9c3cc`. It does not read large files,
train models, refit features, generate a split, or change existing ML modules.

```python
from business_entity_resolution.integration.member1 import (
    Member1Adapter, inference_mapping, partition_candidates,
)

adapter = Member1Adapter(source1_records, source2_records, source3_records, split_json)
joined = adapter.join(candidate_rows, ground_truth=complete_match_lookup, require_labels=True)
train, validation = partition_candidates(joined)
```

Records require `entity_id`, `business_name_normalized`,
`business_address_normalized`, `country_normalized`, and `postal_codes`.
Use strings or None; blank/whitespace-only feature values become missing.
IDs remain full strings including S1-/S2-/S3- prefixes; wrong prefixes and duplicate
records are rejected even if duplicate rows have identical fields. Source names
are retained as `source1`, `source2`, `source3` on adapted records.

Candidate rows require `pair_id`, `source1_entity_id`, `candidate_entity_id`, and
`candidate_source` (`source2` or `source3`). Optional `label` must be integer 0/1;
optional `split` must agree exactly with the saved split. Member 1's positive pair
artifact lacks pair IDs, so Member 2 must provide stable unique IDs before joining.
Duplicate edges and pair IDs are rejected. No rows are silently dropped or sorted.

The split object uses `train_source1_entity_ids` and
`validation_source1_entity_ids`. Overlap, duplicates and unknown candidate S1 IDs
raise errors. Only candidates are required to be covered: source tables may also
contain records not referenced by this candidate batch. Partitioning preserves
relative input order and never uses a random seed.

Ground truth maps each covered S1 ID to its COMPLETE collection of matching S2/S3
IDs. A present empty collection confirms no matches. An absent S1 key is missing
coverage and raises when ground truth is supplied, even if a row already has a
label. Supplied labels are checked against ground truth. With no ground-truth
argument, existing labels are caller-authoritative; missing labels remain None
unless `require_labels=True`. Completeness cannot be inferred from a partial lookup;
the caller must establish that unlisted candidates are confirmed nonmatches.

Postal collection boundary
--------------------------

`NormalizedRecord.postal_codes` preserves each space-separated value in a tuple,
including leading zeros and ZIP+4 hyphens. `original_postal_codes` preserves the
input serialization. This parser is specific to Member 1's numeric postal-code
serialization, not a parser for international codes with internal spaces.

`joined_item.ml_pair()` produces the existing CandidatePair contract for records
with at most one postal code. Multiple codes raise by default: the unchanged scalar
feature pipeline cannot represent their semantics. No code is selected or joined.
An explicit `mask_multiple_postal_codes=True` exports that scalar field as None;
the full collection remains in the joined record. This deliberately disables
postal evidence for that record and must not be mistaken for collection-aware
matching. Production use should wait for versioned collection-aware features.

Inference compatibility (no scoring occurs in the adapter):

```python
rows = [item.inference_row() for item in joined]  # fails safely on multiple codes
scores = existing_inference.score(rows, inference_mapping())
```

For an explicitly approved temporary missing-postal projection, pass
`mask_multiple_postal_codes=True` to `inference_row`. Rows retain both postal lists,
original candidate source, label and split as metadata; the existing inference
mapper reads only the scalar ML fields. Returned scores retain original candidate
and entity IDs and use `S2`/`S3` for target_source, as before.

Remaining integration work
--------------------------

- Pass `saved_split=split_json` to `train_classifier` with collection-aware ML pairs.
  Adapter exports retain their split assignments; training checks them again.
- Use a model trained with the v2 collection-aware feature contract for full-fidelity scoring.
- Obtain real negative candidates, complete ground-truth coverage, and stable pair
  IDs; confirm upstream normalization and dataset versions.
- Plan bounded-memory joins for real datasets: this adapter builds in-memory indexes.

The tiny JSON fixture under tests/fixtures documents a complete synthetic contract.
New adapter tests perform no fitting or model training. Existing full-suite tests
retain their original temporary synthetic training behavior.
