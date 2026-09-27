Postal feature contracts
========================

The 12 feature names/order remain unchanged. New FeaturePipeline instances use
`postal-collection-v2`; historical unversioned pickles are interpreted as
`postal-scalar-v1`. The model and fitted pipeline must agree on their contract.
Unknown/mismatched versions fail at load and scoring. Existing artifact files
are never rewritten. Old models reject explicit collections, including empty
ones, rather than silently adopting new semantics.

Scalar `postal_code` remains a string or None. It is NEVER split: `641 001`
still normalizes to `641001`. Existing scalar functions are unchanged.
The optional `postal_codes` field must be a list/tuple of individual strings or
None. Each item uses the existing scalar normalizer. Empty normalized items are
discarded; agreement is any intersection and availability requires two nonempty
sets. No format validation or ZIP+4 truncation is added. If both fields are
provided, a nonempty scalar must represent exactly the same singleton collection;
otherwise the input is rejected as conflicting. Empty collections are missing,
not an instruction to fall back to a different scalar value.

Member 1's adapter owns parsing of its space-separated serialization. Enable:

```python
pairs = [item.ml_pair(collection_aware=True) for item in joined]
rows = [item.inference_row(collection_aware=True) for item in joined]
mapping = inference_mapping(collection_aware=True)
scores = inference.score(rows, mapping)  # requires a v2 artifact
```

The adapter defaults retain the prior scalar-export behavior for compatibility.
Collection export and scalar masking cannot be combined. Generic inference JSON
mappings can add a `postal_codes` path to either record's four existing fields;
the path must resolve to an explicit collection, not the serialized Member 1 text.

New training saves the contract in model.joblib, its fitted feature pipeline, and
report.json. The training input hash includes collections. Native lightgbm.txt
alone does not encode the preprocessing contract: retain model.joblib and report.
Do not relabel a historical model as v2; a real v2 model must be trained under the
new contract when authorized. The existing scalar artifact remains usable on
scalar inputs only.

Member 1 saved-split training is available through `train_classifier(...,
saved_split=split_json)` and CLI `--saved-split`. See ../matching/SAVED_SPLIT.md.
Calls without assignments or a saved split retain random grouped splitting.
No production model is retrained by this update.
