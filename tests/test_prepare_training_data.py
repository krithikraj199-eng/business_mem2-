"""Tests for bounded-memory training data preparation workflow."""

import json
from pathlib import Path
import pytest

from business_entity_resolution.external.member1 import (
    build_lookup_index,
    SOURCE_COLUMNS,
)
from business_entity_resolution.matching.train import read_pairs
from business_entity_resolution.matching.prepare_training_data import (
    estimate_model_fitting_memory,
    prepare_training_data,
)
from business_entity_resolution.matching.training import train_classifier, TrainingConfig


def _write_tsv(path: Path, rows, newline="\n"):
    with path.open("wb") as fh:
        fh.write(("\t".join(SOURCE_COLUMNS) + newline).encode("utf-8"))
        for row in rows:
            fh.write(("\t".join(row) + newline).encode("utf-8"))


def _row(entity_id, name="name", address="addr", country="us", postal_codes=""):
    return [
        entity_id,
        name,
        address,
        country,
        name,
        f'["{name}"]',
        address,
        f'["{address}"]',
        "[]",
        postal_codes,
        country,
    ]


@pytest.fixture
def synthetic_training_setup(tmp_path):
    """Synthetic environment with both train and validation entities."""
    data_dir = tmp_path / "normalized"
    data_dir.mkdir()

    # S1 records: S1-001 & S1-002 (train), S1-003 & S1-004 (validation)
    s1_rows = [
        _row("S1-001", "Alpha Industries", "10 Market St", "US", "10001"),
        _row("S1-002", "Beta Global", "20 Park Ave", "US", "10002"),
        _row("S1-003", "Gamma Enterprises", "30 Elm St", "US", "10003"),
        _row("S1-004", "Delta Logistics", "40 Oak Rd", "US", "10004"),
    ]

    # Target S2 and S3 records
    s2_rows = [
        _row("S2-001", "Alpha Industries Inc", "10 Market St", "US", "10001"),
        _row("S2-002", "Alpha Commercial", "99 Random Rd", "US", "99999"),
        _row("S2-003", "Gamma Enterprise LLC", "30 Elm St", "US", "10003"),
        _row("S2-004", "Unrelated Corp", "11 First St", "US", "12345"),
        _row("S2-005", "Beta Corp", "55 Fifth Ave", "US", "10002"),
        _row("S2-006", "Other Co", "77 Broad St", "US", "54321"),
        _row("S2-007", "Random Firm", "88 High St", "US", "67890"),
    ]

    s3_rows = [
        _row("S3-001", "Beta Global Ltd", "20 Park Ave", "US", "10002"),
        _row("S3-002", "Beta Worldwide", "20 Park Avenue", "US", "10002"),
        _row("S3-003", "Zeta Parts", "12 Second St", "US", "11111"),
        _row("S3-005", "Eta Holdings", "14 Third St", "US", "22222"),
        _row("S3-006", "Theta Group", "16 Fourth St", "US", "33333"),
        _row("S3-008", "Iota Services", "18 Fifth St", "US", "44444"),
    ]

    _write_tsv(data_dir / "normalized_source1.tsv", s1_rows)
    _write_tsv(data_dir / "normalized_source2.tsv", s2_rows)
    _write_tsv(data_dir / "normalized_source3.tsv", s3_rows)

    index_path = tmp_path / "lookup" / "entity_lookup.sqlite3"
    build_lookup_index(data_dir, index_path)

    # Saved split with both train and validation
    split_file = tmp_path / "validation_split.json"
    split_file.write_text(
        json.dumps({
            "train_source1_entity_ids": ["S1-001", "S1-002"],
            "validation_source1_entity_ids": ["S1-003", "S1-004"],
        }),
        encoding="utf-8",
    )

    # Ground truth: complete coverage for all 4 S1 entities
    # S1-001 has S2-001 (pos)
    # S1-002 has S3-001, S3-002 (pos)
    # S1-003 has S2-003 (pos)
    # S1-004 has no matches (empty set)
    gt_file = tmp_path / "train_ground_truth.tsv"
    gt_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\tS3-001,S3-002\n"
        "S1-003\tS2-003\n"
        "S1-004\t\n",
        encoding="utf-8",
    )

    # Member 2 candidate output
    cand_file = tmp_path / "candidate_train.tsv"
    cand_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S2-002,S3-003,S2-004,S3-005,S2-006\n"
        "S1-002\tS3-001,S3-002,S2-005,S3-006,S2-007,S3-008\n"
        "S1-003\tS2-003,S2-001,S3-001\n"
        "S1-004\tS2-002,S3-002,S3-003\n",
        encoding="utf-8",
    )

    return {
        "index_path": index_path,
        "split_file": split_file,
        "gt_file": gt_file,
        "cand_file": cand_file,
    }


def test_prepare_training_data_workflow(synthetic_training_setup, tmp_path):
    """Test full preparation preserves positives, samples train negatives, and keeps validation unsampled."""
    cand_file = synthetic_training_setup["cand_file"]
    index_path = synthetic_training_setup["index_path"]
    gt_file = synthetic_training_setup["gt_file"]
    split_file = synthetic_training_setup["split_file"]

    out_file = tmp_path / "outputs" / "prepared_pairs.jsonl"

    report = prepare_training_data(
        candidates_path=cand_file,
        lookup=index_path,
        ground_truth_path=gt_file,
        split_path_or_artifact=split_file,
        output_path=out_file,
        negative_ratio=2,
        seed=42,
    )

    assert out_file.exists()
    assert report["total_pairs_written"] == 15

    # Check train partition:
    # S1-001: 1 pos + 2 sampled negs = 3
    # S1-002: 2 pos + 4 sampled negs = 6
    # Total train = 9
    assert report["train"]["source1_entities"] == 2
    assert report["train"]["positives"] == 3
    assert report["train"]["negatives"] == 6
    assert report["train"]["total_pairs"] == 9

    # Check validation partition:
    # S1-003: 1 pos + 2 negs = 3 (unsampled)
    # S1-004: 0 pos + 3 negs = 3 (unsampled)
    # Total validation = 6
    assert report["validation"]["source1_entities"] == 2
    assert report["validation"]["positives"] == 1
    assert report["validation"]["negatives"] == 5
    assert report["validation"]["total_pairs"] == 6

    # Verify rows can be read by read_pairs and used to train LightGBM
    pairs = read_pairs(out_file)
    assert len(pairs) == 15

    # Positives must all be intact
    pos_pairs = [p for p in pairs if p.label == 1]
    assert len(pos_pairs) == 4
    assert {p.pair_id for p in pos_pairs} == {
        "S1-001::S2-001",
        "S1-002::S3-001",
        "S1-002::S3-002",
        "S1-003::S2-003",
    }

    # Verify model training succeeds with prepared dataset
    saved_split = json.loads(split_file.read_text(encoding="utf-8"))
    config = TrainingConfig(num_boost_round=10, min_data_in_leaf=1)
    train_res = train_classifier(pairs, config, data_kind="synthetic", saved_split=saved_split)
    assert train_res.model is not None
    assert train_res.report["validation"]["count"] == 6


def test_prepare_training_data_overwrite_protection(synthetic_training_setup, tmp_path):
    """FileExistsError raised when destination exists unless overwrite=True."""
    cand_file = synthetic_training_setup["cand_file"]
    index_path = synthetic_training_setup["index_path"]
    gt_file = synthetic_training_setup["gt_file"]
    split_file = synthetic_training_setup["split_file"]

    out_file = tmp_path / "prepared_pairs.jsonl"
    out_file.write_text("existing content", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output file already exists"):
        prepare_training_data(
            cand_file, index_path, gt_file, split_file, out_file, overwrite=False
        )

    # With overwrite=True, succeeds
    report = prepare_training_data(
        cand_file, index_path, gt_file, split_file, out_file, overwrite=True
    )
    assert report["total_pairs_written"] > 0


def test_prepare_training_data_missing_coverage_raises(synthetic_training_setup, tmp_path):
    """Missing ground-truth coverage for candidate S1 entity is strictly rejected."""
    cand_file = synthetic_training_setup["cand_file"]
    index_path = synthetic_training_setup["index_path"]
    split_file = synthetic_training_setup["split_file"]

    # Ground truth missing S1-004
    bad_gt = tmp_path / "incomplete_gt.tsv"
    bad_gt.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\tS3-001,S3-002\n"
        "S1-003\tS2-003\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="Missing ground-truth coverage for S1-004"):
        prepare_training_data(
            cand_file, index_path, bad_gt, split_file, out_file
        )


def test_prepare_training_data_dry_run_synthetic(synthetic_training_setup):
    """Dry run verifies S1 counts, raw/prepared candidate counts, size, and memory without writing files."""
    cand_file = synthetic_training_setup["cand_file"]
    index_path = synthetic_training_setup["index_path"]
    gt_file = synthetic_training_setup["gt_file"]
    split_file = synthetic_training_setup["split_file"]

    report = prepare_training_data(
        candidates_path=cand_file,
        lookup=index_path,
        ground_truth_path=gt_file,
        split_path_or_artifact=split_file,
        output_path=None,
        negative_ratio=2,
        seed=42,
        dry_run=True,
    )

    assert report["mode"] == "dry_run"
    assert report["output_path"] is None
    assert report["output_sha256"] is None

    # Source 1 entities
    assert report["source1_entities"]["total"] == 4
    assert report["source1_entities"]["train"] == 2
    assert report["source1_entities"]["validation"] == 2

    # Ground truth
    assert report["ground_truth"]["covered_entities"] == 4
    assert report["ground_truth"]["total_positive_matches"] == 4

    # Raw candidate counts before sampling
    assert report["raw_candidates"]["total"] == 18
    assert report["raw_candidates"]["train"]["positives"] == 3
    assert report["raw_candidates"]["train"]["negatives"] == 9
    assert report["raw_candidates"]["validation"]["positives"] == 1
    assert report["raw_candidates"]["validation"]["negatives"] == 5

    # Prepared candidate counts after sampling
    assert report["prepared_candidates"]["total"] == 15
    assert report["prepared_candidates"]["train"]["positives"] == 3
    assert report["prepared_candidates"]["train"]["positive_preservation_pct"] == 100.0
    assert report["prepared_candidates"]["train"]["negatives"] == 6
    assert report["prepared_candidates"]["train"]["dropped_negatives"] == 3
    assert report["prepared_candidates"]["validation"]["positives"] == 1
    assert report["prepared_candidates"]["validation"]["negatives"] == 5
    assert report["prepared_candidates"]["validation"]["unsampled"] is True

    # Estimated output size
    assert report["estimated_output_file"]["total_lines"] == 15
    assert report["estimated_output_file"]["estimated_bytes"] > 0
    assert report["estimated_output_file"]["avg_line_bytes"] > 100.0

    # Model fitting memory requirements
    mem = report["memory_requirements"]
    assert mem["total_candidate_pairs"] == 15
    assert mem["train_pairs"] == 9
    assert mem["validation_pairs"] == 6
    assert mem["fits_in_16gb"] is True
    assert mem["status"] == "SAFE"

    # Production readiness
    assert report["production_readiness"]["trainer_fits_in_16gb"] is True
    assert report["production_readiness"]["verdict"] == "SAFE"


def test_estimate_model_fitting_memory_scaling():
    """Verify memory model flags out-of-memory risks for large unsampled validation sets."""
    # Small dataset: fits safely in 16 GB
    safe_res = estimate_model_fitting_memory(train_pairs_count=10_000, val_pairs_count=10_000)
    assert safe_res["fits_in_16gb"] is True
    assert safe_res["status"] == "SAFE"
    assert safe_res["identified_bottlenecks"] == []

    # Large dataset with huge unsampled validation set: exceeds 16 GB limit
    unsafe_res = estimate_model_fitting_memory(train_pairs_count=100_000, val_pairs_count=15_000_000)
    assert unsafe_res["fits_in_16gb"] is False
    assert unsafe_res["status"] == "UNSAFE_EXCEEDS_16GB"
    assert unsafe_res["peak_estimated_ram_gb"] > 16.0
    assert any("unsampled validation set" in b for b in unsafe_res["identified_bottlenecks"])
    assert any("validate_pairs" in b for b in unsafe_res["identified_bottlenecks"])
    assert any("input_sha256" in b for b in unsafe_res["identified_bottlenecks"])
    assert "Smallest necessary change" in unsafe_res["recommendation"]
