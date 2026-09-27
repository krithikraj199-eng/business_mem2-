# Member 3 Model Artifacts Notice

> [!CAUTION]
> **CRITICAL: NON-PRODUCTION SYNTHETIC SMOKE MODEL**
>
> The model artifact located in `models/member3_synthetic_smoke/` is strictly a **synthetic smoke-test fixture**.
> - It was trained on **20 toy synthetic pairs** (32 train pairs, 8 validation pairs) merely to verify serialization, deserialization, feature schema alignment, and bounded batch inference plumbing.
> - **DO NOT** report its evaluation metrics or validation scores as model accuracy.
> - **DO NOT** deliver or use this artifact as a final production model.
> - It outputs mock baseline probabilities solely for pipeline verification and dry runs.

## Production Model Requirements

Production training will occur once the following upstream artifacts are finalized and provided:
1. **Member 1**: Complete ground-truth labels (`train_ground_truth.tsv`) and official entity-level split (`validation_split.json` or `validation_membership.tsv`).
2. **Member 1**: Full normalized records (`normalized_source1.tsv`, `normalized_source2.tsv`, `normalized_source3.tsv`) indexed via SQLite `DiskEntityLookup`.
3. **Member 2**: Final candidate pair blocking artifact (`candidate_pairs.tsv` or `candidate_train.tsv`).
4. **Member 3 Real Training**: Execution of `business-entity-resolution-prepare-training` followed by LightGBM training on real prepared data with character n-gram TF-IDF fit exclusively on the training split.
