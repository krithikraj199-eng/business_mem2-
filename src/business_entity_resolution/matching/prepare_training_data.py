"""CLI and API for preparing labeled, negative-sampled training datasets with bounded memory."""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from ..external.member1.disk_lookup import DiskEntityLookup
from ..integration.member1 import DiskMember1Adapter, JoinedCandidate, load_ground_truth_tsv, load_saved_split
from ..integration.member2 import Member2TSVAdapter
from .sampling import sample_training_candidates


def estimate_model_fitting_memory(
    train_pairs_count: int,
    val_pairs_count: int,
    unique_train_records_est: int | None = None,
) -> dict[str, Any]:
    """Estimate realistic RAM required by train_classifier on a given prepared dataset.

    Distinguishes bounded streaming components from materialized components:
    - validate_pairs materializes list[CandidatePair], seen_ids, seen_edges, records (~1.2 KB/pair).
    - input_sha256 materializes payload list of dicts + contiguous json.dumps() string (~650 bytes/pair).
    - FeaturePipeline.fit builds TF-IDF vocabulary on unique training records.
    - LightGBM dataset holds 12 float32 features + bin histograms.
    - Validation scoring transforms and scores all validation pairs.
    """
    total_pairs = train_pairs_count + val_pairs_count
    if unique_train_records_est is None:
        unique_train_records_est = min(train_pairs_count * 2, max(train_pairs_count // 3, 10))

    validate_pairs_mb = (total_pairs * 1200) / (1024 * 1024)
    tfidf_fit_mb = min(unique_train_records_est * 2000, 1500 * 1024 * 1024) / (1024 * 1024) + 50.0
    feature_matrix_train_mb = (train_pairs_count * 12 * 4) / (1024 * 1024)
    feature_matrix_val_mb = (val_pairs_count * 12 * 4) / (1024 * 1024)
    lightgbm_dataset_mb = feature_matrix_train_mb * 1.5 + 20.0
    provenance_hash_mb = (total_pairs * 650) / (1024 * 1024)
    validation_scoring_mb = feature_matrix_val_mb + (val_pairs_count * 250) / (1024 * 1024)
    base_overhead_mb = 180.0

    peak_ram_mb = base_overhead_mb + validate_pairs_mb + provenance_hash_mb + max(
        tfidf_fit_mb, feature_matrix_train_mb + lightgbm_dataset_mb
    )
    peak_ram_gb = peak_ram_mb / 1024.0

    # Operational safety limit on a 16 GB machine (leaves 4 GB for OS, SQLite, and Python overhead)
    fits_in_16gb = peak_ram_gb <= 12.0
    status = "SAFE" if fits_in_16gb else "UNSAFE_EXCEEDS_16GB"

    bottlenecks = []
    if total_pairs > 1_500_000:
        bottlenecks.append("validate_pairs: materializing >1.5M CandidatePair objects in RAM exceeds 2 GB")
    if total_pairs > 2_000_000:
        bottlenecks.append("train_classifier input_sha256: json.dumps(payload) creates a multi-gigabyte contiguous JSON string in RAM")
    if val_pairs_count > 1_000_000:
        bottlenecks.append("unsampled validation set: scoring >1M validation pairs un-batched in model.score_pairs() risks high memory pressure")

    recommendation = (
        "Existing trainer can safely run within 16 GB RAM."
        if fits_in_16gb
        else (
            "Smallest necessary change: batch validation evaluation and stream provenance hashing "
            "in train_classifier rather than holding all pairs in RAM simultaneously."
        )
    )

    return dict(
        total_candidate_pairs=total_pairs,
        train_pairs=train_pairs_count,
        validation_pairs=val_pairs_count,
        breakdown_mb=dict(
            validate_pairs=round(validate_pairs_mb, 2),
            tfidf_fit=round(tfidf_fit_mb, 2),
            lightgbm_dataset=round(lightgbm_dataset_mb, 2),
            provenance_hash=round(provenance_hash_mb, 2),
            validation_scoring=round(validation_scoring_mb, 2),
            base_runtime=round(base_overhead_mb, 2),
        ),
        peak_estimated_ram_mb=round(peak_ram_mb, 2),
        peak_estimated_ram_gb=round(peak_ram_gb, 2),
        fits_in_16gb=fits_in_16gb,
        status=status,
        identified_bottlenecks=bottlenecks,
        recommendation=recommendation,
    )


def prepare_training_data(
    candidates_path: str | Path,
    lookup: str | Path | DiskEntityLookup | DiskMember1Adapter,
    ground_truth_path: str | Path,
    split_path_or_artifact: str | Path | Mapping,
    output_path: str | Path | None = None,
    *,
    negative_ratio: int | None = 10,
    max_zero_match_negatives: int = 10,
    seed: int = 42,
    batch_size: int = 1024,
    collection_aware: bool = False,
    mask_multiple_postal_codes: bool | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stream candidates, join records, derive verified labels, sample negatives, and write JSONL.

    When dry_run=True:
    - Performs complete validation, join, and negative sampling without writing files.
    - Reports Source 1 entity counts, raw vs prepared candidate counts, preserved positives,
      sampled training negatives, unsampled validation pairs, estimated file size, and
      model-fitting memory requirements.

    Invariants:
    1. Preserves 100% of candidate positive matches present in the candidate set.
    2. Keeps validation candidates 100% un-sampled.
    3. Deterministically samples training negatives per Source 1 entity.
    4. Writes labeled records incrementally to output_path (O(1) memory during write).
    5. Validates complete ground-truth coverage for all candidate Source 1 entities.
    6. Preserves Member 1's authoritative Source 1 entity-level split.
    """
    out_file = None
    if not dry_run:
        if output_path is None:
            raise ValueError("output_path is required when dry_run=False")
        out_file = Path(output_path)
        if out_file.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {out_file}")
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth mapping
    truth = load_ground_truth_tsv(ground_truth_path)

    # 2. Load authoritative entity split
    split = load_saved_split(split_path_or_artifact)

    # 3. Initialize disk-backed lookup adapter
    disk_adapter = (
        lookup
        if isinstance(lookup, DiskMember1Adapter)
        else DiskMember1Adapter(lookup, split)
    )

    # 4. Stream joined candidate batches with ground-truth label derivation & tracking
    tsv_adapter = Member2TSVAdapter(candidates_path)

    raw_stats = {
        'train_pos': 0,
        'train_neg': 0,
        'val_pos': 0,
        'val_neg': 0,
        'train_s1': set(),
        'val_s1': set(),
    }

    def _stream_joined():
        for batch in tsv_adapter.stream_joined_batches(
            disk_adapter,
            batch_size=batch_size,
            ground_truth=truth,
            require_labels=True,
        ):
            for j in batch:
                s1_id = j.source1.entity_id
                if j.split == 'train':
                    raw_stats['train_s1'].add(s1_id)
                    if j.label == 1:
                        raw_stats['train_pos'] += 1
                    else:
                        raw_stats['train_neg'] += 1
                elif j.split == 'validation':
                    raw_stats['val_s1'].add(s1_id)
                    if j.label == 1:
                        raw_stats['val_pos'] += 1
                    else:
                        raw_stats['val_neg'] += 1
                yield j

    # 5. Apply stratified negative sampling per Source 1 entity
    sampled_stream = sample_training_candidates(
        _stream_joined(),
        negative_ratio=negative_ratio,
        max_zero_match_negatives=max_zero_match_negatives,
        seed=seed,
    )

    # 6. Configure postal export policy
    postal_kwargs: dict[str, Any] = {}
    if collection_aware:
        postal_kwargs['collection_aware'] = True
    else:
        postal_kwargs['mask_multiple_postal_codes'] = (
            True if mask_multiple_postal_codes is None else mask_multiple_postal_codes
        )

    # 7. Incremental write or read-only statistics aggregation
    hasher = hashlib.sha256()
    total_written = 0
    train_pos = 0
    train_neg = 0
    val_pos = 0
    val_neg = 0
    seen_s1_train = set()
    seen_s1_val = set()
    sample_line_lengths: list[int] = []

    fh = None
    try:
        if out_file is not None:
            fh = out_file.open('wb')

        for item in sampled_stream:
            if isinstance(item, JoinedCandidate):
                pair = item.ml_pair(**postal_kwargs)
            else:
                pair = item
            row = dict(
                pair_id=pair.pair_id,
                source1_id=pair.source1_id,
                source2_id=pair.source2_id,
                source1=pair.source1,
                source2=pair.source2,
                label=pair.label,
                split=pair.split,
            )
            line = (json.dumps(row, allow_nan=False) + '\n').encode('utf-8')

            if len(sample_line_lengths) < 100:
                sample_line_lengths.append(len(line))

            if fh is not None:
                hasher.update(line)
                fh.write(line)

            total_written += 1

            if pair.split == 'train':
                seen_s1_train.add(pair.source1_id)
                if pair.label == 1:
                    train_pos += 1
                elif pair.label == 0:
                    train_neg += 1
            elif pair.split == 'validation':
                seen_s1_val.add(pair.source1_id)
                if pair.label == 1:
                    val_pos += 1
                elif pair.label == 0:
                    val_neg += 1
    finally:
        if fh is not None:
            fh.close()

    avg_line_bytes = (
        sum(sample_line_lengths) / len(sample_line_lengths)
        if sample_line_lengths
        else 350.0
    )
    estimated_output_bytes = int(total_written * avg_line_bytes)

    memory_est = estimate_model_fitting_memory(
        train_pairs_count=train_pos + train_neg,
        val_pairs_count=val_pos + val_neg,
    )

    report = dict(
        mode="dry_run" if dry_run else "production_prepare",
        output_path=str(out_file) if out_file is not None else None,
        output_sha256=hasher.hexdigest() if out_file is not None else None,
        total_pairs_written=total_written,
        train=dict(
            source1_entities=len(seen_s1_train),
            positives=train_pos,
            negatives=train_neg,
            total_pairs=train_pos + train_neg,
        ),
        validation=dict(
            source1_entities=len(seen_s1_val),
            positives=val_pos,
            negatives=val_neg,
            total_pairs=val_pos + val_neg,
        ),
        source1_entities=dict(
            total=len(seen_s1_train | seen_s1_val),
            train=len(seen_s1_train),
            validation=len(seen_s1_val),
        ),
        ground_truth=dict(
            covered_entities=len(truth),
            total_positive_matches=sum(len(m) for m in truth.values()),
        ),
        raw_candidates=dict(
            total=raw_stats['train_pos'] + raw_stats['train_neg'] + raw_stats['val_pos'] + raw_stats['val_neg'],
            train=dict(
                positives=raw_stats['train_pos'],
                negatives=raw_stats['train_neg'],
                total=raw_stats['train_pos'] + raw_stats['train_neg'],
            ),
            validation=dict(
                positives=raw_stats['val_pos'],
                negatives=raw_stats['val_neg'],
                total=raw_stats['val_pos'] + raw_stats['val_neg'],
            ),
        ),
        prepared_candidates=dict(
            total=total_written,
            train=dict(
                positives=train_pos,
                negatives=train_neg,
                dropped_negatives=raw_stats['train_neg'] - train_neg,
                positive_preservation_pct=(
                    100.0 if raw_stats['train_pos'] == train_pos
                    else (round(train_pos / raw_stats['train_pos'] * 100, 2) if raw_stats['train_pos'] else 100.0)
                ),
                total=train_pos + train_neg,
            ),
            validation=dict(
                positives=val_pos,
                negatives=val_neg,
                total=val_pos + val_neg,
                unsampled=True,
            ),
        ),
        estimated_output_file=dict(
            total_lines=total_written,
            avg_line_bytes=round(avg_line_bytes, 1),
            estimated_bytes=estimated_output_bytes,
            estimated_mb=round(estimated_output_bytes / (1024 * 1024), 2),
        ),
        memory_requirements=memory_est,
        production_readiness=dict(
            trainer_fits_in_16gb=memory_est["fits_in_16gb"],
            verdict=memory_est["status"],
            identified_bottlenecks=memory_est["identified_bottlenecks"],
            recommendation=memory_est["recommendation"],
        ),
        config=dict(
            negative_ratio=negative_ratio,
            max_zero_match_negatives=max_zero_match_negatives,
            seed=seed,
            batch_size=batch_size,
            collection_aware=collection_aware,
        ),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, help='Member 2 candidate TSV path')
    parser.add_argument('--lookup', required=True, help='Member 1 SQLite index or normalized data dir')
    parser.add_argument('--ground-truth', required=True, help='Member 1 ground truth TSV path')
    parser.add_argument('--saved-split', required=True, help='Member 1 validation_split.json or TSV')
    parser.add_argument('--output', help='Destination JSONL path (required unless --dry-run is set)')
    parser.add_argument('--dry-run', action='store_true', help='Perform read-only audit: report counts, size, and memory requirements without writing files')
    parser.add_argument('--negative-ratio', type=int, default=10, help='Max negatives per positive (default: 10)')
    parser.add_argument('--max-zero-match-negatives', type=int, default=10, help='Max negatives for zero-match entities')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic sampling')
    parser.add_argument('--batch-size', type=int, default=1024, help='Lookup batch size')
    parser.add_argument('--collection-aware', action='store_true', help='Export full postal code collections')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting output file')
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is specified")

    report = prepare_training_data(
        candidates_path=args.candidates,
        lookup=args.lookup,
        ground_truth_path=args.ground_truth,
        split_path_or_artifact=args.saved_split,
        output_path=args.output,
        negative_ratio=args.negative_ratio,
        max_zero_match_negatives=args.max_zero_match_negatives,
        seed=args.seed,
        batch_size=args.batch_size,
        collection_aware=args.collection_aware,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

__all__ = [
    'estimate_model_fitting_memory',
    'prepare_training_data',
]
