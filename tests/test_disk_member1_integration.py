"""Integration tests for DiskMember1Adapter and Member 1's DiskEntityLookup.

Tests disk-backed record retrieval, S2/S3 joins, missing ID rejections,
bounded batching, source-file integrity checks, ordering preservation,
postal-collection-v2 handling, and end-to-end inference pipeline scoring.
"""

from pathlib import Path
import pytest

from business_entity_resolution.external.member1 import (
    DiskEntityLookup,
    LookupIntegrityError,
    build_lookup_index,
    SOURCE_COLUMNS,
)
from business_entity_resolution.integration.member1 import (
    DiskMember1Adapter,
    Member1Adapter,
    Member1DiskAdapter,
    SavedSplit,
    inference_mapping,
    partition_candidates,
)
from business_entity_resolution.integration.member2 import Member2TSVAdapter
from business_entity_resolution.matching.inference import InferencePipeline


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
def synthetic_fixtures(tmp_path):
    """Create minimal synthetic S1, S2, S3 normalized TSVs and build Member 1 SQLite index."""
    data_dir = tmp_path / "normalized"
    data_dir.mkdir()

    s1_rows = [
        _row("S1-001", "alpha shop", "12 road", "india", "600001 600002"),
        _row("S1-002", "beta corp", "34 avenue", "usa", "90210"),
        _row("S1-003", "gamma inc", "56 lane", "canada", ""),
        _row("S1-004", "delta llc", "78 blvd", "uk", "SW1A 1AA"),
    ]
    s2_rows = [
        _row("S2-001", "alpha shop inc", "12 road st", "india", "600001"),
        _row("S2-002", "beta corporation", "34 ave", "usa", "90210"),
    ]
    s3_rows = [
        _row("S3-001", "alpha store", "12 rd", "india", "600001 600002"),
        _row("S3-002", "gamma incorporated", "56 lane rd", "canada", ""),
    ]

    _write_tsv(data_dir / "normalized_source1.tsv", s1_rows)
    _write_tsv(data_dir / "normalized_source2.tsv", s2_rows)
    _write_tsv(data_dir / "normalized_source3.tsv", s3_rows)

    index_path = tmp_path / "lookup" / "entity_lookup.sqlite3"
    build_lookup_index(data_dir, index_path)

    split = {
        "train_source1_entity_ids": ["S1-001", "S1-003"],
        "validation_source1_entity_ids": ["S1-002", "S1-004"],
    }

    return {
        "data_dir": data_dir,
        "index_path": index_path,
        "split": split,
    }


def test_disk_member1_adapter_s2_and_s3_joins(synthetic_fixtures):
    """Test disk-backed join with both S2 and S3 candidates preserves order and attributes."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    candidates = [
        {"pair_id": "p1", "source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "source2"},
        {"pair_id": "p2", "source1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "source3"},
        {"pair_id": "p3", "source1_entity_id": "S1-002", "candidate_entity_id": "S2-002", "candidate_source": "source2"},
    ]

    with DiskMember1Adapter(index_path, split) as adapter:
        joined = adapter.join(candidates)
        assert len(joined) == 3

        # Preserve exact candidate order
        assert [j.pair_id for j in joined] == ["p1", "p2", "p3"]

        # Check S2 join
        assert joined[0].source1.entity_id == "S1-001"
        assert joined[0].target.entity_id == "S2-001"
        assert joined[0].candidate_source == "source2"
        assert joined[0].split == "train"

        # Check S3 join
        assert joined[1].source1.entity_id == "S1-001"
        assert joined[1].target.entity_id == "S3-001"
        assert joined[1].candidate_source == "source3"
        assert joined[1].split == "train"

        # Check validation split assignment
        assert joined[2].source1.entity_id == "S1-002"
        assert joined[2].split == "validation"


def test_disk_member1_adapter_alias(synthetic_fixtures):
    """Member1DiskAdapter is an alias for DiskMember1Adapter."""
    assert Member1DiskAdapter is DiskMember1Adapter


def test_disk_member1_adapter_with_existing_lookup_instance(synthetic_fixtures):
    """DiskMember1Adapter works when passed an existing DiskEntityLookup instance."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    with DiskEntityLookup(index_path) as lookup:
        adapter = DiskMember1Adapter(lookup, split)
        joined = adapter.join([
            {"pair_id": "p1", "source1_entity_id": "S1-002", "candidate_entity_id": "S2-002", "candidate_source": "source2"},
        ])
        assert len(joined) == 1
        assert joined[0].source1.entity_id == "S1-002"
        # Since adapter did not create lookup, closing adapter does not close caller's lookup
        adapter.close()
        assert lookup.get("S1-002") is not None


def test_disk_member1_adapter_missing_ids_rejected(synthetic_fixtures):
    """Missing Source 1 or candidate IDs are strictly rejected with ValueError."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    adapter = DiskMember1Adapter(index_path, split)

    # Missing Source 1 ID (must be in split first to reach join lookup)
    split_with_missing = {
        "train_source1_entity_ids": ["S1-001", "S1-999"],
        "validation_source1_entity_ids": ["S1-002"],
    }
    adapter_missing_s1 = DiskMember1Adapter(index_path, split_with_missing)
    with pytest.raises(ValueError, match="Missing record join: S1-999"):
        adapter_missing_s1.join([
            {"pair_id": "p1", "source1_entity_id": "S1-999", "candidate_entity_id": "S2-001", "candidate_source": "source2"},
        ])

    # Missing S2 candidate ID
    with pytest.raises(ValueError, match="Missing record join: S2-999"):
        adapter.join([
            {"pair_id": "p1", "source1_entity_id": "S1-001", "candidate_entity_id": "S2-999", "candidate_source": "source2"},
        ])

    # Missing S3 candidate ID
    with pytest.raises(ValueError, match="Missing record join: S3-999"):
        adapter.join([
            {"pair_id": "p1", "source1_entity_id": "S1-001", "candidate_entity_id": "S3-999", "candidate_source": "source3"},
        ])


def test_empty_string_missing_values_and_postal_collection_v2(synthetic_fixtures):
    """Preserves empty-string missing values as None/() and enforces postal-collection-v2 contract."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    adapter = DiskMember1Adapter(index_path, split)

    # S1-003 has empty postal code ("")
    joined_empty = adapter.join([
        {"pair_id": "p-empty", "source1_entity_id": "S1-003", "candidate_entity_id": "S3-002", "candidate_source": "source3"},
    ])[0]
    assert joined_empty.source1.postal_codes == ()
    assert joined_empty.source1.original_postal_codes == ""
    assert joined_empty.source1.ml_record()["postal_code"] is None

    # S1-001 has multiple postal codes ("600001 600002")
    joined_multi = adapter.join([
        {"pair_id": "p-multi", "source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "source2"},
    ])[0]
    assert joined_multi.source1.postal_codes == ("600001", "600002")

    # Scalar inference requires explicit masking
    with pytest.raises(ValueError, match="Multiple postal codes require collection-aware features or explicit masking"):
        joined_multi.ml_pair()

    ml_masked = joined_multi.ml_pair(mask_multiple_postal_codes=True)
    assert ml_masked.source1["postal_code"] is None

    # Collection-aware inference exports full list
    ml_coll = joined_multi.ml_pair(collection_aware=True)
    assert ml_coll.source1["postal_codes"] == ["600001", "600002"]


def test_source_file_integrity_verification(synthetic_fixtures):
    """Source file modification after indexing is caught by DiskEntityLookup."""
    data_dir = synthetic_fixtures["data_dir"]
    index_path = synthetic_fixtures["index_path"]

    # Append byte to source1 file
    with (data_dir / "normalized_source1.tsv").open("ab") as fh:
        fh.write(b" ")

    with pytest.raises(LookupIntegrityError, match="Indexed source file changed"):
        DiskEntityLookup(index_path)


def test_streaming_tsv_adapter_integration(synthetic_fixtures, tmp_path):
    """Member2TSVAdapter streaming into DiskMember1Adapter in bounded batches."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    tsv_content = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002\n"
        "S1-003\tS3-002\n"
    )

    disk_adapter = DiskMember1Adapter(index_path, split)
    tsv_adapter = Member2TSVAdapter(tsv_content)

    # Batch size 2 exercises batch boundary splitting (4 total pairs -> batches of 2 and 2)
    batch_sizes = []
    total_joined = 0
    for joined_batch in tsv_adapter.stream_joined_batches(disk_adapter, batch_size=2):
        batch_sizes.append(len(joined_batch))
        total_joined += len(joined_batch)

    assert total_joined == 4
    assert batch_sizes == [2, 2]


def test_end_to_end_inference_compatibility(synthetic_fixtures, tmp_path):
    """Full pipeline: TSV -> Member2TSVAdapter -> DiskMember1Adapter -> InferencePipeline scoring."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    model_path = Path(__file__).parent.parent / "models" / "member3_synthetic_smoke" / "model.joblib"
    inference = InferencePipeline(model_path)
    disk_adapter = DiskMember1Adapter(index_path, split)

    tsv_content = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002\n"
    )
    tsv_adapter = Member2TSVAdapter(tsv_content)

    scores = list(tsv_adapter.score_candidates(inference, disk_adapter, batch_size=2))
    assert len(scores) == 3

    assert [s["pair_id"] for s in scores] == ["S1-001::S2-001", "S1-001::S3-001", "S1-002::S2-002"]
    assert [s["target_source"] for s in scores] == ["S2", "S3", "S2"]
    assert all(0.0 <= s["probability"] <= 1.0 for s in scores)

    # Test score_file writes valid JSONL
    out_jsonl = tmp_path / "scored_output.jsonl"
    count = tsv_adapter.score_file(out_jsonl, inference, disk_adapter, batch_size=2)
    assert count == 3
    assert out_jsonl.exists()


def test_ground_truth_and_labels(synthetic_fixtures):
    """DiskMember1Adapter correctly derives and checks ground truth and explicit labels."""
    index_path = synthetic_fixtures["index_path"]
    split = synthetic_fixtures["split"]

    disk_adapter = DiskMember1Adapter(index_path, split)
    candidates = [
        {"pair_id": "p1", "source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "source2"},
        {"pair_id": "p2", "source1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "source3"},
    ]

    truth = {"S1-001": ["S2-001"]}  # S2-001 is match, S3-001 is non-match
    joined = disk_adapter.join(candidates, ground_truth=truth)
    assert [j.label for j in joined] == [1, 0]

    # Partition candidates
    train, val = partition_candidates(joined)
    assert len(train) == 2
    assert len(val) == 0
