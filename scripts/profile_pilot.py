"""Detailed profiling script for Member 3 Real-Data Pilot pipeline.

Measures exact time spent in:
1. Candidate TSV parsing & duplicate tracking
2. SQLite record lookup (DiskEntityLookup.get_many)
3. Record adaptation (adapt_record & JoinedCandidate)
4. 12-feature extraction (with breakdown for TF-IDF, Levenshtein, Jaccard, Structured, Postal)
5. Model inference (LightGBM booster.predict)
6. Output writing (JSONL & TSV)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from business_entity_resolution.external.member1.disk_lookup import DiskEntityLookup
from business_entity_resolution.integration.member1 import DiskMember1Adapter, SavedSplit, inference_mapping, adapt_record, JoinedCandidate
from business_entity_resolution.integration.member2 import Member2TSVAdapter, validate_candidate_batch
from business_entity_resolution.matching.inference import InferencePipeline


def profile_pipeline(batch_size: int = 1024):
    pilot_dir = Path("pilot/member3_real_data_pilot")
    norm_dir = pilot_dir / "normalized"
    index_path = pilot_dir / "lookup" / "entity_lookup.sqlite3"
    cand_path = pilot_dir / "candidate_pilot.tsv"
    model_path = Path("models/member3_synthetic_smoke/model.joblib")
    membership_tsv = pilot_dir / "validation_membership.tsv"
    out_dir = pilot_dir / "outputs" / "profile_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "profile_out.jsonl"
    out_tsv = out_dir / "profile_out.tsv"
    out_jsonl.unlink(missing_ok=True)
    out_tsv.unlink(missing_ok=True)

    train_ids = []
    with membership_tsv.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        train_ids = [r[0] for r in reader if r[1] == "train"]
    split = SavedSplit({"train_source1_entity_ids": train_ids, "validation_source1_entity_ids": []})

    inference = InferencePipeline(model_path)
    mapping = inference_mapping(collection_aware=False)

    print("=" * 80)
    print("PROFILING EXISTING MEMBER 3 PIPELINE")
    print(f"Candidate file: {cand_path}")
    print(f"Batch size:     {batch_size}")
    print("=" * 80)

    # Accumulators
    t_candidate_parse_dup = 0.0
    t_sqlite_lookup = 0.0
    t_adaptation = 0.0
    t_pair_validation = 0.0
    t_feature_extraction = 0.0
    t_model_inference = 0.0
    t_output_writing = 0.0

    total_pairs = 0
    total_batches = 0

    tracemalloc.start()
    t_start_total = time.perf_counter()

    adapter = Member2TSVAdapter(cand_path, duplicate_store="disk")

    with DiskEntityLookup(index_path, data_dir=norm_dir, verify_hash=True) as lookup:
        with out_jsonl.open("w", encoding="utf-8") as fh_jsonl, \
             out_tsv.open("w", encoding="utf-8", newline="") as fh_tsv:
            
            tsv_writer = csv.writer(fh_tsv, delimiter="\t", lineterminator="\n")
            tsv_writer.writerow(["pair_id", "source1_id", "target_source", "target_id", "probability"])

            batch_iter = adapter.iter_candidate_batches(batch_size=batch_size, duplicate_store="disk")
            
            while True:
                # 1. Candidate TSV parsing & duplicate tracking
                t0 = time.perf_counter()
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                t_candidate_parse_dup += time.perf_counter() - t0

                total_batches += 1
                total_pairs += len(batch)

                # 2. SQLite Record Lookup
                t0 = time.perf_counter()
                needed_ids = []
                for row in batch:
                    needed_ids.append(row["source1_entity_id"])
                    needed_ids.append(row["candidate_entity_id"])
                unique_ids = list(dict.fromkeys(needed_ids))
                records_raw = lookup.get_many(unique_ids)
                record_map = dict(zip(unique_ids, records_raw, strict=True))
                t_sqlite_lookup += time.perf_counter() - t0

                # 3. Record Adaptation
                t0 = time.perf_counter()
                joined = []
                for row in batch:
                    pair_id = row["pair_id"]
                    source = row["candidate_source"]
                    left = row["source1_entity_id"]
                    right = row["candidate_entity_id"]
                    assignment = split.assignment(left)
                    raw_left = record_map[left]
                    raw_right = record_map[right]
                    left_record = adapt_record(raw_left, "source1")
                    right_record = adapt_record(raw_right, source)
                    joined.append(JoinedCandidate(pair_id, left_record, right_record, source, None, assignment))
                t_adaptation += time.perf_counter() - t0

                # 4. Feature Extraction & Pair Mapping
                t0 = time.perf_counter()
                rows = [item.inference_row(mask_multiple_postal_codes=True) for item in joined]
                pairs = []
                for row in rows:
                    pair, src, target_id = mapping.map_row(row)
                    pairs.append(pair)
                matrix = inference.model.features.transform((p.source1, p.source2) for p in pairs)
                t_feature_extraction += time.perf_counter() - t0

                # 5. Model Inference (LightGBM booster predict)
                t0 = time.perf_counter()
                probabilities = np.asarray(inference.model.booster.predict(matrix))
                t_model_inference += time.perf_counter() - t0

                # 6. Output Writing
                t0 = time.perf_counter()
                for cand, prob in zip(batch, probabilities, strict=True):
                    rec = {
                        "pair_id": cand["pair_id"],
                        "source1_id": cand["source1_entity_id"],
                        "target_source": "S2" if cand["candidate_entity_id"].startswith("S2-") else "S3",
                        "target_id": cand["candidate_entity_id"],
                        "probability": float(prob),
                    }
                    fh_jsonl.write(json.dumps(rec) + "\n")
                    tsv_writer.writerow([rec["pair_id"], rec["source1_id"], rec["target_source"], rec["target_id"], f"{rec['probability']:.6f}"])
                t_output_writing += time.perf_counter() - t0

    t_total = time.perf_counter() - t_start_total
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("-" * 80)
    print(f"Total Batches: {total_batches}, Total Pairs: {total_pairs}")
    print(f"Total Wall Time: {t_total:.2f} s ({t_total/60:.2f} min)")
    print(f"Peak Traced Memory: {peak_mem / (1024*1024):.2f} MB")
    print("-" * 80)
    print("DETAILED TIME BREAKDOWN:")
    breakdown = [
        ("1. Candidate TSV & Duplicate Tracking", t_candidate_parse_dup),
        ("2. SQLite Record Lookup (get_many)", t_sqlite_lookup),
        ("3. Record Adaptation (adapt_record & JoinedCandidate)", t_adaptation),
        ("4. 12-Feature Extraction (transform)", t_feature_extraction),
        ("5. Model Inference (booster.predict)", t_model_inference),
        ("6. Output Writing (JSONL & TSV)", t_output_writing),
    ]
    accounted = sum(t for _, t in breakdown)
    overhead = t_total - accounted

    for name, t in breakdown:
        pct = (t / t_total) * 100
        print(f"  {name:<55}: {t:8.3f} s ({pct:5.1f}%)")
    print(f"  {'Other overhead (loop / batch coordination)':<55}: {overhead:8.3f} s ({(overhead/t_total)*100:5.1f}%)")
    print("-" * 80)

    # Clean up test output dir
    out_jsonl.unlink(missing_ok=True)
    out_tsv.unlink(missing_ok=True)
    out_dir.rmdir()


if __name__ == "__main__":
    profile_pipeline()
