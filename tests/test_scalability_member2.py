"""Scalability and disk-backed duplicate detection tests for Member 2 candidate streaming.

Verifies bounded-memory streaming, disk-backed exact duplicate detection parity with
in-memory tracking, resource cleanup, parameter validation, and high-volume synthetic workloads.
"""

import io
import json
import os
from pathlib import Path
import tempfile

import pytest

from business_entity_resolution.integration.member2 import (
    Member2TSVAdapter,
    Source1Coverage,
    parse_member2_tsv,
    read_member2_tsv,
)


def test_duplicate_store_disk_and_memory_parity():
    """Disk-backed and in-memory duplicate stores produce byte-identical candidate outputs and coverage."""
    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001,S2-002\n"
        "S1-002\t\n"
        "S1-003\tS3-002,S2-003\n"
    )

    adapter_disk = Member2TSVAdapter(tsv, duplicate_store='disk')
    pairs_disk = list(adapter_disk.iter_candidate_pairs())
    cov_disk = adapter_disk.coverage
    assert cov_disk is not None

    adapter_mem = Member2TSVAdapter(tsv, duplicate_store='memory')
    pairs_mem = list(adapter_mem.iter_candidate_pairs())
    cov_mem = adapter_mem.coverage
    assert cov_mem is not None

    assert pairs_disk == pairs_mem
    assert cov_disk.all_source1_ids == cov_mem.all_source1_ids
    assert cov_disk.empty_source1_ids == cov_mem.empty_source1_ids
    assert cov_disk.non_empty_source1_ids == cov_mem.non_empty_source1_ids
    assert cov_disk.candidate_counts == cov_mem.candidate_counts
    assert cov_disk.total_candidate_pairs == cov_mem.total_candidate_pairs == 5


def test_duplicate_store_validation():
    """Invalid duplicate_store argument raises ValueError."""
    with pytest.raises(ValueError, match="duplicate_store must be 'disk' or 'memory'"):
        Member2TSVAdapter("S1-1\tS2-1", duplicate_store='invalid')

    adapter = Member2TSVAdapter("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1")
    with pytest.raises(ValueError, match="duplicate_store must be 'disk' or 'memory'"):
        list(adapter.iter_candidate_pairs(duplicate_store='redis'))


def test_disk_backed_rejects_duplicates_intra_and_cross_line():
    """Disk-backed tracker strictly rejects duplicate pairs on same line and across lines."""
    # Same line duplicate
    tsv_same = "source1_entity_id\tcandidate_entity_ids\nS1-001\tS2-001,S2-001\n"
    with pytest.raises(ValueError, match=r"TSV line 2: duplicate candidate pair \('S1-001', 'S2-001'\)"):
        list(Member2TSVAdapter(tsv_same, duplicate_store='disk').iter_candidate_pairs())

    # Cross line duplicate
    tsv_cross = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-001\tS2-001\n"
    )
    with pytest.raises(ValueError, match=r"TSV line 3: duplicate candidate pair \('S1-001', 'S2-001'\)"):
        list(Member2TSVAdapter(tsv_cross, duplicate_store='disk').iter_candidate_pairs())


def test_custom_db_dir_used_and_cleaned_up(tmp_path):
    """Custom db_dir is used for temporary SQLite file and cleaned up upon completion."""
    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002\n"
    )
    custom_dir = tmp_path / 'sqlite_scratch'
    custom_dir.mkdir()

    adapter = Member2TSVAdapter(tsv, duplicate_store='disk', db_dir=custom_dir)
    pairs = list(adapter.iter_candidate_pairs())
    assert len(pairs) == 3

    # Ensure no leftover files in custom_dir
    assert list(custom_dir.iterdir()) == []


def test_disk_backed_cleanup_on_error(tmp_path):
    """Disk-backed SQLite file is cleaned up even when TSV processing raises ValueError."""
    tsv_dup = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-001\tS2-001\n"
    )
    custom_dir = tmp_path / 'sqlite_err'
    custom_dir.mkdir()

    adapter = Member2TSVAdapter(tsv_dup, duplicate_store='disk', db_dir=custom_dir)
    with pytest.raises(ValueError, match="duplicate candidate pair"):
        list(adapter.iter_candidate_pairs())

    # Ensure no leftover temporary directories or files
    assert list(custom_dir.iterdir()) == []


def test_disk_backed_cleanup_on_generator_break(tmp_path):
    """Early exit from generator cleanly closes SQLite and deletes temporary directory."""
    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002,S3-002\n"
    )
    custom_dir = tmp_path / 'sqlite_break'
    custom_dir.mkdir()

    adapter = Member2TSVAdapter(tsv, duplicate_store='disk', db_dir=custom_dir)
    gen = adapter.iter_candidate_pairs()
    first = next(gen)
    assert first['source1_entity_id'] == 'S1-001'
    del gen  # Triggers GeneratorExit and finally block

    assert list(custom_dir.iterdir()) == []


def test_synthetic_high_volume_candidate_streaming(tmp_path):
    """Scalability test: 25,000 synthetic candidate pairs stream through disk-backed adapter with bounded memory."""
    tsv_file = tmp_path / 'high_volume_candidates.tsv'
    num_s1 = 500
    cands_per_s1 = 50  # 500 * 50 = 25,000 pairs

    with tsv_file.open('w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_idx in range(num_s1):
            s1_id = f"S1-{s1_idx:05d}"
            cand_ids = [
                f"S2-{s1_idx * cands_per_s1 + c_idx:07d}" if c_idx % 2 == 0
                else f"S3-{s1_idx * cands_per_s1 + c_idx:07d}"
                for c_idx in range(cands_per_s1)
            ]
            f.write(f"{s1_id}\t{','.join(cand_ids)}\n")

    adapter = Member2TSVAdapter(tsv_file, duplicate_store='disk')
    batch_count = 0
    total_yielded = 0

    for batch in adapter.iter_candidate_batches(batch_size=1000):
        batch_count += 1
        total_yielded += len(batch)
        assert len(batch) <= 1000

    assert total_yielded == 25000
    assert batch_count == 25

    cov = adapter.coverage
    assert cov is not None
    assert cov.total_source1_entities == 500
    assert cov.total_candidate_pairs == 25000
    assert len(cov.empty_source1_ids) == 0


def test_synthetic_high_volume_duplicate_detection(tmp_path):
    """High-volume duplicate detection detects duplicate injected at tail of large stream."""
    tsv_file = tmp_path / 'duplicate_candidates.tsv'
    num_pairs = 10000

    with tsv_file.open('w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        # Line 2: S1-00000 has S2-00000
        f.write("S1-00000\tS2-00000\n")
        # Middle lines: distinct pairs
        for i in range(1, num_pairs):
            f.write(f"S1-{i:05d}\tS2-{i:07d}\n")
        # Duplicate injected at the end
        f.write("S1-00000\tS2-00000\n")

    adapter = Member2TSVAdapter(tsv_file, duplicate_store='disk')
    with pytest.raises(ValueError, match="duplicate candidate pair"):
        for _ in adapter.iter_candidate_pairs():
            pass


def test_read_and_parse_tsv_disk_and_memory_options(tmp_path):
    """read_member2_tsv and parse_member2_tsv respect duplicate_store and db_dir kwargs."""
    tsv_file = tmp_path / 'helpers.tsv'
    tsv_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\tS3-001\n"
    )

    pairs_disk, cov_disk = parse_member2_tsv(tsv_file, duplicate_store='disk', db_dir=tmp_path)
    pairs_mem, cov_mem = parse_member2_tsv(tsv_file, duplicate_store='memory')

    assert pairs_disk == pairs_mem
    assert cov_disk.total_source1_entities == cov_mem.total_source1_entities == 2

    streamed_disk = list(read_member2_tsv(tsv_file, duplicate_store='disk'))
    streamed_mem = list(read_member2_tsv(tsv_file, duplicate_store='memory'))
    assert streamed_disk == streamed_mem
