"""Integration test verifying Member 3 pipeline on the real-data pilot.

Ensures candidate streaming, DiskMember1Adapter joins, 12-feature pipeline,
and InferencePipeline scoring conform strictly to all contracts:
- 15 Source 1 entities covered
- Exactly 15,648 candidate pairs scored
- Pair IDs and candidate order preserved
- Probabilities valid and in [0.0, 1.0]
- Zero missing joins
- 51 ground-truth pairs cross-checked
- Saved split membership verified (all 15 belong to train; no held-out validation)
- Synthetic smoke model used only for workflow verification (non-production)

This test module is safely skipped on clean checkouts where git-ignored pilot
data files are absent, keeping real-data integration separated from synthetic tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import pytest

from business_entity_resolution.external.member1.disk_lookup import DiskEntityLookup
from business_entity_resolution.integration.member1 import DiskMember1Adapter, SavedSplit
from business_entity_resolution.integration.member2 import Member2TSVAdapter
from business_entity_resolution.matching.inference import InferencePipeline
from business_entity_resolution.matching.prepare_training_data import prepare_training_data
from business_entity_resolution.matching.train import read_pairs

PILOT_DIR = Path(__file__).parent.parent / "pilot" / "member3_real_data_pilot"
SMOKE_MODEL_PATH = Path(__file__).parent.parent / "models" / "member3_synthetic_smoke" / "model.joblib"

HAS_REAL_PILOT = (
    PILOT_DIR.is_dir()
    and (PILOT_DIR / "candidate_pilot.tsv").is_file()
    and (PILOT_DIR / "ground_truth_pilot.tsv").is_file()
    and (PILOT_DIR / "validation_membership.tsv").is_file()
    and (PILOT_DIR / "lookup" / "entity_lookup.sqlite3").is_file()
)

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not HAS_REAL_PILOT,
        reason="Real-data pilot files not present in clean checkout (git-ignored)",
    ),
]


def _parse_ground_truth_matches(raw: str) -> list[str]:
    match_token_re = re.compile(r"S[23]-[A-Za-z0-9_]+")
    if not raw.strip() or raw.strip() in ("[]", "null", "None"):
        return []
    return match_token_re.findall(raw)


def test_pilot_artifacts_and_ground_truth_cross_check():
    """Verify pilot ground truth has 51 pairs and all 15 S1 entities belong to train split."""
    candidate_tsv = PILOT_DIR / "candidate_pilot.tsv"
    ground_truth_tsv = PILOT_DIR / "ground_truth_pilot.tsv"
    membership_tsv = PILOT_DIR / "validation_membership.tsv"

    assert candidate_tsv.exists(), "candidate_pilot.tsv must exist"
    assert ground_truth_tsv.exists(), "ground_truth_pilot.tsv must exist"
    assert membership_tsv.exists(), "validation_membership.tsv must exist"

    # Cross-check split membership
    train_ids, val_ids = [], []
    with membership_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "split"]
        for row in reader:
            (train_ids if row[1] == "train" else val_ids).append(row[0])

    assert len(train_ids) == 15, "All 15 pilot entities must be in train split"
    assert len(val_ids) == 0, "Zero entities in validation split"

    # Cross-check ground truth
    total_gt_pairs = 0
    gt_by_s1: dict[str, set[str]] = {}
    with ground_truth_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        s1_idx = header.index("source1_entity_id")
        m_idx = header.index("matched_entity_ids")
        for row in reader:
            matches = set(_parse_ground_truth_matches(row[m_idx]))
            gt_by_s1[row[s1_idx]] = matches
            total_gt_pairs += len(matches)

    assert total_gt_pairs == 51, f"Expected 51 ground truth pairs, got {total_gt_pairs}"
    assert len(gt_by_s1) == 15, f"Expected 15 S1 entities in ground truth, got {len(gt_by_s1)}"


def test_pilot_candidate_and_disk_lookup_coverage():
    """Verify SQLite lookup index contains all pilot records and candidate stream matches counts."""
    candidate_tsv = PILOT_DIR / "candidate_pilot.tsv"
    norm_dir = PILOT_DIR / "normalized"
    index_path = PILOT_DIR / "lookup" / "entity_lookup.sqlite3"

    assert index_path.exists(), "entity_lookup.sqlite3 must exist"

    # Verify DiskEntityLookup with hash verification
    with DiskEntityLookup(index_path, data_dir=norm_dir, verify_hash=True) as lookup:
        # Check Source 1
        s1_rec = lookup.get("S1-100190437")
        assert s1_rec is not None
        assert s1_rec["entity_id"] == "S1-100190437"
        assert s1_rec["business_name_normalized"] == "guffey metro digital associates"

        # Check Source 2
        s2_rec = lookup.get("S2-725882042")
        assert s2_rec is not None
        assert s2_rec["entity_id"] == "S2-725882042"

        # Check Source 3
        s3_rec = lookup.get("S3-998746725")
        assert s3_rec is not None
        assert s3_rec["entity_id"] == "S3-998746725"

    adapter = Member2TSVAdapter(candidate_tsv, duplicate_store="disk")
    cov = adapter.inspect_coverage()
    assert cov is not None
    assert cov.total_source1_entities == 15
    assert cov.total_candidate_pairs == 15648
    assert len(cov.empty_source1_ids) == 0


def test_pilot_end_to_end_scoring_workflow(tmp_path):
    """Run full bounded-batch inference and verify all contracts on scored output."""
    candidate_tsv = PILOT_DIR / "candidate_pilot.tsv"
    membership_tsv = PILOT_DIR / "validation_membership.tsv"
    norm_dir = PILOT_DIR / "normalized"
    index_path = PILOT_DIR / "lookup" / "entity_lookup.sqlite3"

    train_ids = []
    with membership_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        train_ids = [r[0] for r in reader if r[1] == "train"]

    split = SavedSplit({
        "train_source1_entity_ids": train_ids,
        "validation_source1_entity_ids": [],
    })

    inference = InferencePipeline(SMOKE_MODEL_PATH)
    adapter = Member2TSVAdapter(candidate_tsv, duplicate_store="disk")

    out_file = tmp_path / "test_scored.jsonl"
    with DiskMember1Adapter(index_path, split, data_dir=norm_dir, verify_hash=True) as disk_adapter:
        scored_count = adapter.score_file(out_file, inference, disk_adapter, batch_size=2048)

    assert scored_count == 15648

    # Verify generated output line by line
    with out_file.open("r", encoding="utf-8") as fh:
        scored_rows = [json.loads(line) for line in fh]

    assert len(scored_rows) == 15648

    candidates = list(adapter.iter_candidate_pairs())
    assert len(candidates) == 15648

    covered_s1 = set()
    for cand, scored in zip(candidates, scored_rows, strict=True):
        assert scored["pair_id"] == cand["pair_id"]
        assert scored["source1_id"] == cand["source1_entity_id"]
        assert scored["target_id"] == cand["candidate_entity_id"]
        assert 0.0 <= scored["probability"] <= 1.0
        covered_s1.add(scored["source1_id"])

    assert len(covered_s1) == 15


def test_pilot_prepare_training_data_workflow(tmp_path):
    """Integration test: prepare_training_data on real pilot preserves 51 positives and samples negatives."""
    out_file = tmp_path / "pilot_prepared.jsonl"
    report = prepare_training_data(
        candidates_path=PILOT_DIR / "candidate_pilot.tsv",
        lookup=PILOT_DIR / "lookup" / "entity_lookup.sqlite3",
        ground_truth_path=PILOT_DIR / "ground_truth_pilot.tsv",
        split_path_or_artifact=PILOT_DIR / "validation_membership.tsv",
        output_path=out_file,
        negative_ratio=10,
        seed=42,
    )

    assert report["total_pairs_written"] == 561
    assert report["train"]["positives"] == 51
    assert report["train"]["negatives"] == 510
    assert report["train"]["source1_entities"] == 15
    assert report["validation"]["total_pairs"] == 0

    pairs = read_pairs(out_file)
    assert len(pairs) == 561
    assert sum(1 for p in pairs if p.label == 1) == 51
    assert sum(1 for p in pairs if p.label == 0) == 510


def test_pilot_prepare_training_data_dry_run():
    """Integration test: dry-run on real pilot produces accurate counts and memory estimate without writing files."""
    report = prepare_training_data(
        candidates_path=PILOT_DIR / "candidate_pilot.tsv",
        lookup=PILOT_DIR / "lookup" / "entity_lookup.sqlite3",
        ground_truth_path=PILOT_DIR / "ground_truth_pilot.tsv",
        split_path_or_artifact=PILOT_DIR / "validation_membership.tsv",
        output_path=None,
        negative_ratio=10,
        seed=42,
        dry_run=True,
    )

    assert report["mode"] == "dry_run"
    assert report["output_path"] is None
    assert report["source1_entities"]["total"] == 15
    assert report["source1_entities"]["train"] == 15
    assert report["source1_entities"]["validation"] == 0

    assert report["ground_truth"]["covered_entities"] == 15
    assert report["ground_truth"]["total_positive_matches"] == 51

    # Raw candidate counts before sampling
    assert report["raw_candidates"]["train"]["positives"] == 51
    assert report["raw_candidates"]["train"]["negatives"] == 15597
    assert report["raw_candidates"]["train"]["total"] == 15648

    # Prepared candidate counts
    assert report["prepared_candidates"]["train"]["positives"] == 51
    assert report["prepared_candidates"]["train"]["positive_preservation_pct"] == 100.0
    assert report["prepared_candidates"]["train"]["negatives"] == 510
    assert report["prepared_candidates"]["train"]["dropped_negatives"] == 15087
    assert report["prepared_candidates"]["total"] == 561

    # Estimated file size
    assert report["estimated_output_file"]["total_lines"] == 561
    assert report["estimated_output_file"]["estimated_bytes"] > 0

    # Memory requirement
    assert report["production_readiness"]["trainer_fits_in_16gb"] is True
    assert report["production_readiness"]["verdict"] == "SAFE"
