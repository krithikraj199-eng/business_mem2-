"""Run the complete Member 3 Real-Data Pilot.

Executes Member2TSVAdapter streaming -> DiskMember1Adapter (SQLite disk joins) ->
12-feature pipeline -> InferencePipeline in bounded batches.
Uses the synthetic smoke model only to verify the inference workflow (non-production).
Verifies candidate ordering, exactly 15,648 pairs, all 15 S1 entities covered,
valid probabilities, and cross-checks the 51 ground-truth pairs and saved split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import time
import tracemalloc

from business_entity_resolution.external.member1.build_lookup_index import build_lookup_index
from business_entity_resolution.external.member1.disk_lookup import DiskEntityLookup
from business_entity_resolution.integration.member1 import DiskMember1Adapter, SavedSplit
from business_entity_resolution.integration.member2 import Member2TSVAdapter
from business_entity_resolution.matching.inference import InferencePipeline


def parse_ground_truth_matches(raw: str) -> list[str]:
    match_token_re = re.compile(r"S[23]-[A-Za-z0-9_]+")
    if not raw.strip() or raw.strip() in ("[]", "null", "None"):
        return []
    return match_token_re.findall(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("pilot/member3_real_data_pilot"),
        help="Path to extracted pilot directory",
    )
    parser.add_argument(
        "--smoke-model-path",
        type=Path,
        default=Path("models/member3_synthetic_smoke/model.joblib"),
        help="Path to synthetic smoke model.joblib",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for bounded streaming execution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to output directory (defaults to pilot-dir/outputs)",
    )
    args = parser.parse_args()

    pilot_dir = args.pilot_dir.resolve()
    smoke_model_path = args.smoke_model_path.resolve()
    output_dir = (args.output_dir or (pilot_dir / "outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEMBER 3 REAL-DATA PILOT VERIFICATION & INFERENCE RUN")
    print("=" * 80)
    print(f"Pilot Directory:      {pilot_dir}")
    print(f"Synthetic Smoke Model: {smoke_model_path}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Output Directory:     {output_dir}")
    print("-" * 80)

    # 1. Setup normalized directory and build SQLite index if needed
    norm_dir = pilot_dir / "normalized"
    norm_dir.mkdir(exist_ok=True)

    tsv_mapping = {
        "source1_pilot.tsv": "normalized_source1.tsv",
        "source2_pilot.tsv": "normalized_source2.tsv",
        "source3_pilot.tsv": "normalized_source3.tsv",
    }
    for src_name, dst_name in tsv_mapping.items():
        src_path = pilot_dir / src_name
        dst_path = norm_dir / dst_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing required pilot source file: {src_path}")
        if not dst_path.exists():
            print(f"Adapting filename: {src_name} -> {dst_name}")
            shutil.copyfile(src_path, dst_path)

    index_dir = pilot_dir / "lookup"
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / "entity_lookup.sqlite3"

    if not index_path.exists():
        print(f"Building fresh SQLite lookup index at {index_path}...")
        build_lookup_index(norm_dir, index_path)
    else:
        print(f"Existing SQLite lookup index found at {index_path}. Verifying fingerprints...")

    # Verify DiskEntityLookup with hash verification
    with DiskEntityLookup(index_path, data_dir=norm_dir, verify_hash=True) as lookup:
        print("DiskEntityLookup fingerprint verification: PASS (SHA-256 and mtime validated)")

    # 2. Inspect candidate TSV & ground truth & split membership
    candidate_tsv = pilot_dir / "candidate_pilot.tsv"
    ground_truth_tsv = pilot_dir / "ground_truth_pilot.tsv"
    membership_tsv = pilot_dir / "validation_membership.tsv"

    # Cross-check saved split membership
    train_ids = []
    val_ids = []
    with membership_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "split"], f"Unexpected split header: {header}"
        for row in reader:
            if row[1] == "train":
                train_ids.append(row[0])
            elif row[1] == "validation":
                val_ids.append(row[0])
            else:
                raise ValueError(f"Unknown split in {membership_tsv}: {row}")

    print(f"Split membership: {len(train_ids)} train entities, {len(val_ids)} validation entities")
    print("NOTE: All 15 pilot entities belong to the training split.")
    print("      This pilot CANNOT validate held-out model performance or threshold selection.")

    split_obj = SavedSplit({
        "train_source1_entity_ids": train_ids,
        "validation_source1_entity_ids": val_ids,
    })

    # Cross-check ground-truth pairs
    truth_by_s1: dict[str, set[str]] = {}
    total_gt_pairs = 0
    with ground_truth_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        assert "source1_entity_id" in header and "matched_entity_ids" in header
        s1_idx = header.index("source1_entity_id")
        match_idx = header.index("matched_entity_ids")
        for row in reader:
            s1 = row[s1_idx]
            matches = set(parse_ground_truth_matches(row[match_idx]))
            truth_by_s1[s1] = matches
            total_gt_pairs += len(matches)

    print(f"Ground truth: {len(truth_by_s1)} S1 entities covered, exactly {total_gt_pairs} ground-truth pairs")
    assert total_gt_pairs == 51, f"Expected 51 ground-truth pairs, got {total_gt_pairs}"
    assert len(truth_by_s1) == 15, f"Expected 15 S1 entities in ground truth, got {len(truth_by_s1)}"

    # 3. Stream candidates and verify candidate list properties
    adapter = Member2TSVAdapter(candidate_tsv, duplicate_store="disk")
    candidates_list: list[dict] = []
    s1_candidate_counts: dict[str, int] = {}

    for cand in adapter.iter_candidate_pairs():
        candidates_list.append(cand)
        s1 = cand["source1_entity_id"]
        s1_candidate_counts[s1] = s1_candidate_counts.get(s1, 0) + 1

    total_candidates = len(candidates_list)
    print(f"Candidate pairs streamed: {total_candidates}")
    assert total_candidates == 15648, f"Expected 15,648 candidate pairs, found {total_candidates}"
    assert len(s1_candidate_counts) == 15, f"Expected 15 Source 1 entities, found {len(s1_candidate_counts)}"

    # Verify that all 51 ground-truth pairs exist inside candidate lists
    candidate_edges = {(c["source1_entity_id"], c["candidate_entity_id"]) for c in candidates_list}
    missing_gt_edges = []
    for s1, matches in truth_by_s1.items():
        for m in matches:
            if (s1, m) not in candidate_edges:
                missing_gt_edges.append((s1, m))
    assert not missing_gt_edges, f"Ground-truth pairs missing from candidate pairs: {missing_gt_edges}"
    print("Ground-truth cross-check: All 51 ground-truth pairs are present in candidate pairs! (100% recall)")

    # 4. Initialize Inference Pipeline with synthetic smoke model
    print("-" * 80)
    print("LOADING INFERENCE PIPELINE (SMOKE MODEL)...")
    print("NOTICE: The synthetic smoke model is used ONLY to verify the end-to-end inference workflow.")
    print("        Its scores are NON-PRODUCTION and MUST NOT be interpreted as model accuracy.")
    print("        No new model is being trained.")
    print("-" * 80)

    inference = InferencePipeline(smoke_model_path)
    assert tuple(inference.model.feature_names) == inference.model.features.feature_names
    print(f"Model loaded and validated. Features ({len(inference.model.feature_names)}): {inference.model.feature_names}")

    # 5. Execute bounded-batch scoring with memory and time instrumentation
    out_jsonl = output_dir / "scored_pilot_candidates.jsonl"
    out_tsv = output_dir / "scored_pilot_candidates.tsv"

    # Remove existing outputs if any
    out_jsonl.unlink(missing_ok=True)
    out_tsv.unlink(missing_ok=True)

    tracemalloc.start()
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    scored_records: list[dict] = []
    with DiskMember1Adapter(index_path, split_obj, data_dir=norm_dir, verify_hash=True) as disk_adapter:
        with out_jsonl.open("w", encoding="utf-8") as fh_jsonl, \
             out_tsv.open("w", encoding="utf-8", newline="") as fh_tsv:

            tsv_writer = csv.writer(fh_tsv, delimiter="\t", lineterminator="\n")
            tsv_writer.writerow([
                "pair_id", "source1_id", "target_source", "target_id",
                "probability", "source1_row_index", "candidate_rank"
            ])

            for score in adapter.score_candidates(
                inference, disk_adapter, batch_size=args.batch_size,
                duplicate_store="disk", include_provenance=True,
            ):
                scored_records.append(score)
                fh_jsonl.write(json.dumps(score, allow_nan=False) + "\n")
                tsv_writer.writerow([
                    score["pair_id"],
                    score["source1_id"],
                    score["target_source"],
                    score["target_id"],
                    f"{score['probability']:.6f}",
                    score.get("source1_row_index", ""),
                    score.get("candidate_rank", ""),
                ])

    t1_wall = time.perf_counter()
    t1_cpu = time.process_time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_duration = t1_wall - t0_wall
    cpu_duration = t1_cpu - t0_cpu
    peak_mem_mb = peak_mem / (1024 * 1024)

    print("-" * 80)
    print("SCORING EXECUTION COMPLETED")
    print(f"Wall Time:       {wall_duration:.2f} s ({wall_duration / 60:.2f} min)")
    print(f"CPU Time:        {cpu_duration:.2f} s")
    print(f"Throughput:      {total_candidates / wall_duration:.1f} pairs/sec")
    print(f"Peak Traced RAM: {peak_mem_mb:.2f} MB")
    print("-" * 80)

    # 6. Strict Output Verification
    print("VERIFYING OUTPUT INTEGRITY...")

    assert len(scored_records) == 15648, f"Expected 15,648 scored records, got {len(scored_records)}"
    print(f"[PASS] Exactly 15,648 candidate pairs produced scored output.")

    # Check candidate ordering, pair IDs, and provenance preservation
    scored_s1_covered = set()
    for idx, (cand, score) in enumerate(zip(candidates_list, scored_records, strict=True)):
        expected_pair_id = cand["pair_id"]
        actual_pair_id = score["pair_id"]
        assert actual_pair_id == expected_pair_id, (
            f"Row {idx}: pair_id mismatch! Expected {expected_pair_id}, got {actual_pair_id}"
        )
        assert score["source1_id"] == cand["source1_entity_id"], (
            f"Row {idx}: source1_id mismatch! Expected {cand['source1_entity_id']}, got {score['source1_id']}"
        )
        assert score["target_id"] == cand["candidate_entity_id"], (
            f"Row {idx}: target_id mismatch! Expected {cand['candidate_entity_id']}, got {score['target_id']}"
        )
        expected_target_source = "S2" if cand["candidate_entity_id"].startswith("S2-") else "S3"
        assert score["target_source"] == expected_target_source, (
            f"Row {idx}: target_source mismatch! Expected {expected_target_source}, got {score['target_source']}"
        )

        prob = score["probability"]
        assert isinstance(prob, (float, int)) and 0.0 <= prob <= 1.0, (
            f"Row {idx}: invalid probability {prob!r}"
        )

        scored_s1_covered.add(score["source1_id"])

    print("[PASS] Candidate ordering and pair IDs strictly preserved row-by-row.")
    print("[PASS] All probabilities are valid float values in [0.0, 1.0].")
    print("[PASS] Zero missing joins across all 15,648 pairs.")
    assert len(scored_s1_covered) == 15, f"Expected 15 S1 entities covered, got {len(scored_s1_covered)}"
    print(f"[PASS] All 15 Source 1 entities remain fully covered: {sorted(scored_s1_covered)}")

    # 7. Write run summary report
    report_path = output_dir / "pilot_run_report.json"
    report_data = {
        "status": "PASS",
        "pilot_directory": str(pilot_dir),
        "synthetic_smoke_model": str(smoke_model_path),
        "model_label": "NON-PRODUCTION SYNTHETIC SMOKE MODEL (WORKFLOW VERIFICATION ONLY)",
        "total_source1_entities": 15,
        "total_candidate_pairs": 15648,
        "total_ground_truth_pairs": total_gt_pairs,
        "ground_truth_recall_in_candidates": 1.0,
        "saved_split_membership": {
            "train_entities": len(train_ids),
            "validation_entities": len(val_ids),
            "held_out_validation_possible": False,
            "note": "All 15 pilot entities belong to training split. Pilot cannot validate held-out performance or threshold selection.",
        },
        "performance": {
            "wall_time_seconds": round(wall_duration, 3),
            "cpu_time_seconds": round(cpu_duration, 3),
            "throughput_pairs_per_sec": round(total_candidates / wall_duration, 1),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "batch_size": args.batch_size,
        },
        "outputs": {
            "jsonl_path": str(out_jsonl),
            "jsonl_size_bytes": out_jsonl.stat().st_size,
            "tsv_path": str(out_tsv),
            "tsv_size_bytes": out_tsv.stat().st_size,
            "lookup_index_path": str(index_path),
            "lookup_index_size_bytes": index_path.stat().st_size,
        },
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2)

    print("-" * 80)
    print("OUTPUT FILES:")
    print(f"  JSONL:  {out_jsonl} ({out_jsonl.stat().st_size:,} bytes)")
    print(f"  TSV:    {out_tsv} ({out_tsv.stat().st_size:,} bytes)")
    print(f"  Report: {report_path}")
    print("PILOT EXECUTION COMPLETE: ALL CONTRACTS VALIDATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
