import json
from pathlib import Path

import pytest

from business_entity_resolution.integration.member1 import (
    Member1Adapter, SavedSplit, adapt_record, inference_mapping, partition_candidates,
    load_ground_truth_tsv, parse_ground_truth_matches, load_saved_split, load_split_dict,
)
from business_entity_resolution.matching.pair_data import validate_pairs


@pytest.fixture
def data():
    return json.loads((Path(__file__).parent / 'fixtures' / 'member1_small.json').read_text())


def adapter(data):
    return Member1Adapter(data['source1'], data['source2'], data['source3'], data['split'])


def test_mapping_postal_collection_and_missing(data):
    record = adapt_record(data['source1'][0], 'source1')
    assert record.entity_id == 'S1-001'
    assert record.original_source == 'source1'
    assert (record.name, record.address, record.country) == ('alpha shop', '12 road', 'india')
    assert record.postal_codes == ('600001', '600002')
    assert record.original_postal_codes == '600001 600002'
    with pytest.raises(ValueError, match='Multiple postal'):
        record.ml_record()
    assert record.ml_record(mask_multiple_postal_codes=True)['postal_code'] is None
    missing = adapt_record(data['source1'][1], 'source1')
    assert missing.ml_record() == dict(name=None, address=None, country=None, postal_code=None)
    assert missing.postal_codes == ()
    assert adapt_record(data['source3'][0], 'source3').ml_record()['postal_code'] == '00123'


def test_joins_labels_ids_order_and_saved_split(data):
    joined = adapter(data).join(iter(data['candidates']), ground_truth=data['ground_truth'], require_labels=True)
    assert [p.pair_id for p in joined] == ['p-validation', 'p-positive', 'p-negative']
    assert [p.label for p in joined] == [0, 1, 0]
    assert [p.target.entity_id for p in joined] == ['S3-001', 'S2-001', 'S3-001']
    train, validation = partition_candidates(joined)
    assert [p.pair_id for p in train] == ['p-positive', 'p-negative']
    assert [p.pair_id for p in validation] == ['p-validation']
    validate_pairs([p.ml_pair(mask_multiple_postal_codes=True) for p in joined], require_labels=True)


def test_inference_contract_without_training(data):
    joined = adapter(data).join(data['candidates'])
    rows = [p.inference_row(mask_multiple_postal_codes=True) for p in joined]
    mapped = [inference_mapping().map_row(row) for row in rows]
    validate_pairs(item[0] for item in mapped)
    assert [item[0].pair_id for item in mapped] == [p.pair_id for p in joined]
    assert [(item[1], item[2]) for item in mapped] == [('S3', 'S3-001'), ('S2', 'S2-001'), ('S3', 'S3-001')]
    assert rows[1]['source1_postal_codes'] == ['600001', '600002']
    assert rows[1]['source1']['postal_code'] is None
    with pytest.raises(ValueError, match='Multiple postal'):
        joined[1].inference_row()


def test_unknown_truth_never_negative(data):
    instance = adapter(data)
    assert all(p.label is None for p in instance.join(data['candidates']))
    with pytest.raises(ValueError, match='coverage'):
        instance.join(data['candidates'], ground_truth={'S1-001': ['S2-001']})
    with pytest.raises(ValueError, match='coverage'):
        instance.join(data['candidates'], ground_truth={})
    with pytest.raises(ValueError, match='required'):
        instance.join(data['candidates'], require_labels=True)
    data['candidates'][0]['label'] = 1
    with pytest.raises(ValueError, match='conflicts'):
        instance.join(data['candidates'], ground_truth=data['ground_truth'])
    assert instance.join(data['candidates'])[0].label == 1


@pytest.mark.parametrize('mutation', ['overlap', 'duplicate', 'unknown', 'conflict'])
def test_split_rejections(data, mutation):
    if mutation == 'overlap':
        data['split']['train_source1_entity_ids'].append('S1-002')
    elif mutation == 'duplicate':
        data['split']['train_source1_entity_ids'].append('S1-001')
    elif mutation == 'unknown':
        data['split']['validation_source1_entity_ids'] = []
    else:
        data['candidates'][0]['split'] = 'train'
    with pytest.raises(ValueError):
        adapter(data).join(data['candidates'])


@pytest.mark.parametrize('mutation', ['missing_left', 'missing_right', 'duplicate_record',
                                     'wrong_prefix', 'wrong_source', 'duplicate_pair', 'duplicate_edge'])
def test_join_rejections(data, mutation):
    if mutation == 'missing_left':
        data['source1'] = data['source1'][:1]
    elif mutation == 'missing_right':
        data['source3'] = []
    elif mutation == 'duplicate_record':
        data['source2'].append(dict(data['source2'][0]))
    elif mutation == 'wrong_prefix':
        data['source2'][0]['entity_id'] = 'S3-001'
    elif mutation == 'wrong_source':
        data['candidates'][0]['candidate_source'] = 'source2'
    elif mutation == 'duplicate_pair':
        data['candidates'][1]['pair_id'] = 'p-validation'
    else:
        data['candidates'].append(dict(data['candidates'][0], pair_id='another'))
    with pytest.raises(ValueError):
        adapter(data).join(data['candidates'])


@pytest.mark.parametrize('label', ['0', True, 2, 0.5])
def test_invalid_labels(data, label):
    data['candidates'][0]['label'] = label
    with pytest.raises(ValueError, match='Labels'):
        adapter(data).join(data['candidates'])


def test_empty_input_and_no_mutation(data):
    before = json.dumps(data)
    instance = adapter(data)
    assert instance.join([]) == []
    assert partition_candidates([]) == ([], [])
    instance.join(data['candidates'], ground_truth=data['ground_truth'])
    assert json.dumps(data) == before


def test_bad_ground_truth_and_missing_field(data):
    with pytest.raises(TypeError):
        adapter(data).join(data['candidates'], ground_truth={'S1-001': None})
    del data['source1'][0]['postal_codes']
    with pytest.raises(KeyError):
        adapter(data)


def test_parse_ground_truth_matches():
    assert parse_ground_truth_matches("S2-001,S3-002") == ["S2-001", "S3-002"]
    assert parse_ground_truth_matches("S2-001") == ["S2-001"]
    assert parse_ground_truth_matches("") == []
    assert parse_ground_truth_matches("   ") == []
    assert parse_ground_truth_matches("[]") == []
    assert parse_ground_truth_matches("null") == []
    assert parse_ground_truth_matches("None") == []
    assert parse_ground_truth_matches("['S2-123', 'S3-456']") == ["S2-123", "S3-456"]
    with pytest.raises(TypeError):
        parse_ground_truth_matches(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_ground_truth_matches("S2-001???invalid")


def test_load_ground_truth_tsv_valid(tmp_path):
    tsv_file = tmp_path / "ground_truth.tsv"
    tsv_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\t\n"
        "S1-003\tS3-002\n",
        encoding="utf-8",
    )
    result = load_ground_truth_tsv(tsv_file)
    assert result == {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": set(),
        "S1-003": {"S3-002"},
    }


def test_load_ground_truth_tsv_errors(tmp_path):
    # Missing file
    with pytest.raises(FileNotFoundError):
        load_ground_truth_tsv(tmp_path / "nonexistent.tsv")

    # Empty file
    empty_file = tmp_path / "empty.tsv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty ground-truth"):
        load_ground_truth_tsv(empty_file)

    # Bad header
    bad_hdr = tmp_path / "bad_hdr.tsv"
    bad_hdr.write_text("col1\tcol2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected ground-truth header"):
        load_ground_truth_tsv(bad_hdr)

    # Duplicate S1
    dup_file = tmp_path / "dup.tsv"
    dup_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-001\tS3-002\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate Source 1 row"):
        load_ground_truth_tsv(dup_file)

    # Malformed ID
    malformed_file = tmp_path / "malformed.tsv"
    malformed_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "BAD-ID\tS2-001\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected a full S1-"):
        load_ground_truth_tsv(malformed_file)


def test_load_saved_split_from_json_and_tsv(tmp_path):
    # Test JSON file loading
    json_file = tmp_path / "validation_split.json"
    json_file.write_text(
        json.dumps({
            "train_source1_entity_ids": ["S1-001", "S1-002"],
            "validation_source1_entity_ids": ["S1-003"],
        }),
        encoding="utf-8",
    )
    split_from_json = load_saved_split(json_file)
    assert split_from_json.assignment("S1-001") == "train"
    assert split_from_json.assignment("S1-003") == "validation"

    # Test TSV file loading (validation_membership.tsv format)
    tsv_file = tmp_path / "validation_membership.tsv"
    tsv_file.write_text(
        "source1_entity_id\tsplit\n"
        "S1-001\ttrain\n"
        "S1-002\ttrain\n"
        "S1-003\tvalidation\n",
        encoding="utf-8",
    )
    split_from_tsv = load_saved_split(tsv_file)
    assert split_from_tsv.assignment("S1-001") == "train"
    assert split_from_tsv.assignment("S1-002") == "train"
    assert split_from_tsv.assignment("S1-003") == "validation"

    # SavedSplit constructor directly accepts path
    split_direct = SavedSplit(tsv_file)
    assert split_direct.assignment("S1-001") == "train"

    # SavedSplit directly accepts another SavedSplit instance
    split_copy = SavedSplit(split_direct)
    assert split_copy.assignment("S1-001") == "train"


def test_load_saved_split_errors(tmp_path):
    # Missing file
    with pytest.raises(FileNotFoundError):
        load_saved_split(tmp_path / "nonexistent.tsv")

    # Bad header
    bad_hdr = tmp_path / "bad_split.tsv"
    bad_hdr.write_text("s1_id\tpartition\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected header in split TSV"):
        load_saved_split(bad_hdr)

    # Invalid split value
    bad_val = tmp_path / "bad_val.tsv"
    bad_val.write_text("source1_entity_id\tsplit\nS1-001\ttest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid split value"):
        load_saved_split(bad_val)
