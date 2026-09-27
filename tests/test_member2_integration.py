"""Synthetic tests for Member 2 candidate validation; no model training."""

import json
from pathlib import Path

import pytest

from business_entity_resolution.integration.member2 import (
    validate_candidate_batch, candidate_batch_summary,
    enrich_candidate_batch, enrich_candidates, generate_pair_id,
    Member2TSVAdapter, Source1Coverage, parse_member2_tsv, read_member2_tsv,
)


@pytest.fixture
def valid_candidates():
    return [
        {'pair_id': 'p-001', 'source1_entity_id': 'S1-001',
         'candidate_entity_id': 'S2-001', 'candidate_source': 'source2'},
        {'pair_id': 'p-002', 'source1_entity_id': 'S1-001',
         'candidate_entity_id': 'S3-001', 'candidate_source': 'source3'},
        {'pair_id': 'p-003', 'source1_entity_id': 'S1-002',
         'candidate_entity_id': 'S2-002', 'candidate_source': 'source2',
         'label': 1, 'split': 'train'},
    ]


def test_valid_batch_passes(valid_candidates):
    result = validate_candidate_batch(valid_candidates)
    assert len(result) == 3
    assert all('pair_id' in row for row in result)
    assert result[2]['label'] == 1
    assert result[2]['split'] == 'train'
    assert result[0].get('label') is None


def test_summary_counts(valid_candidates):
    summary = candidate_batch_summary(valid_candidates)
    assert summary['total_candidates'] == 3
    assert summary['unique_source1_entities'] == 2
    assert summary['labeled_count'] == 1
    assert summary['unlabeled_count'] == 2
    assert summary['candidate_sources'] == ['source2', 'source3']


def test_empty_batch():
    assert validate_candidate_batch([]) == []
    summary = candidate_batch_summary([])
    assert summary['total_candidates'] == 0


def test_generator_input(valid_candidates):
    result = validate_candidate_batch(iter(valid_candidates))
    assert len(result) == 3


@pytest.mark.parametrize('field', [
    'pair_id', 'source1_entity_id', 'candidate_entity_id', 'candidate_source',
])
def test_missing_required_field(valid_candidates, field):
    del valid_candidates[0][field]
    with pytest.raises(ValueError, match='missing required'):
        validate_candidate_batch(valid_candidates)


def test_duplicate_pair_id(valid_candidates):
    valid_candidates[1]['pair_id'] = 'p-001'
    with pytest.raises(ValueError, match='duplicate pair_id'):
        validate_candidate_batch(valid_candidates)


def test_duplicate_edge(valid_candidates):
    valid_candidates[1] = dict(valid_candidates[0], pair_id='p-different')
    with pytest.raises(ValueError, match='duplicate edge'):
        validate_candidate_batch(valid_candidates)


def test_wrong_source1_prefix(valid_candidates):
    valid_candidates[0]['source1_entity_id'] = 'S2-001'
    with pytest.raises(ValueError, match='S1-prefixed'):
        validate_candidate_batch(valid_candidates)


def test_wrong_candidate_prefix(valid_candidates):
    valid_candidates[0]['candidate_entity_id'] = 'S3-001'  # source2 expects S2-
    with pytest.raises(ValueError, match='S2-prefixed'):
        validate_candidate_batch(valid_candidates)


def test_invalid_candidate_source(valid_candidates):
    valid_candidates[0]['candidate_source'] = 'source4'
    with pytest.raises(ValueError, match='source2 or source3'):
        validate_candidate_batch(valid_candidates)


@pytest.mark.parametrize('label', ['0', True, 2, 0.5])
def test_invalid_label(valid_candidates, label):
    valid_candidates[0]['label'] = label
    with pytest.raises(ValueError, match='label must be'):
        validate_candidate_batch(valid_candidates)


def test_invalid_split(valid_candidates):
    valid_candidates[0]['split'] = 'test'
    with pytest.raises(ValueError, match='split must be'):
        validate_candidate_batch(valid_candidates)


def test_non_mapping_row():
    with pytest.raises(TypeError, match='mapping'):
        validate_candidate_batch(['not a dict'])


def test_empty_pair_id(valid_candidates):
    valid_candidates[0]['pair_id'] = '  '
    with pytest.raises(ValueError, match='nonempty string'):
        validate_candidate_batch(valid_candidates)


def test_extra_fields_preserved(valid_candidates):
    """Extra metadata from Member 2's blocker is passed through, not stripped."""
    valid_candidates[0]['blocking_key'] = 'name_norm:alpha'
    valid_candidates[0]['rank'] = 3
    result = validate_candidate_batch(valid_candidates)
    assert result[0]['blocking_key'] == 'name_norm:alpha'
    assert result[0]['rank'] == 3


def test_raw_two_column_input_requires_enrichment():
    """Member 2's raw two-column output fails clearly before enrichment."""
    raw = [
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
    ]
    with pytest.raises(ValueError, match='missing required'):
        validate_candidate_batch(raw)


def test_validated_compatible_with_member1_join():
    """Validated rows (including extra fields) work with Member1Adapter.join()."""
    from business_entity_resolution.integration.member1 import Member1Adapter

    data = json.loads(
        (Path(__file__).parent / 'fixtures' / 'member1_small.json').read_text()
    )
    # Add extra metadata to simulate Member 2's enriched output.
    for row in data['candidates']:
        row['blocking_key'] = 'test_key'
    validated = validate_candidate_batch(data['candidates'])
    assert all(row['blocking_key'] == 'test_key' for row in validated)
    # Extra fields do not interfere with Member1Adapter.join().
    adapter = Member1Adapter(
        data['source1'], data['source2'], data['source3'], data['split'],
    )
    joined = adapter.join(validated)
    assert len(joined) == 3
    assert [p.pair_id for p in joined] == ['p-validation', 'p-positive', 'p-negative']


def test_enrich_raw_s2_and_s3_candidates():
    """Raw two-column input with S2 and S3 prefixes derives source and adds pair_id."""
    raw = [
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S3-001'},
        {'source1_entity_id': 'S1-002', 'candidate_entity_id': 'S2-002', 'label': 1, 'split': 'train'},
    ]
    enriched = enrich_candidate_batch(raw)
    assert len(enriched) == 3
    assert enriched[0]['candidate_source'] == 'source2'
    assert enriched[1]['candidate_source'] == 'source3'
    assert enriched[2]['candidate_source'] == 'source2'
    assert enriched[0]['pair_id'] == 'S1-001::S2-001'
    assert enriched[1]['pair_id'] == 'S1-001::S3-001'
    assert enriched[2]['pair_id'] == 'S1-002::S2-002'
    assert enriched[2]['label'] == 1
    assert enriched[2]['split'] == 'train'

    summary = candidate_batch_summary(enriched)
    assert summary['total_candidates'] == 3
    assert summary['unique_source1_entities'] == 2
    assert summary['candidate_sources'] == ['source2', 'source3']


def test_enrich_deterministic_pair_ids():
    """Pair IDs are deterministic and collision-safe across multiple invocations."""
    raw = [
        {'source1_entity_id': 'S1-A', 'candidate_entity_id': 'S2-B'},
        {'source1_entity_id': 'S1-A', 'candidate_entity_id': 'S3-C'},
    ]
    run1 = enrich_candidate_batch(raw)
    run2 = enrich_candidate_batch(raw)
    assert [r['pair_id'] for r in run1] == [r['pair_id'] for r in run2]
    assert run1[0]['pair_id'] == generate_pair_id('S1-A', 'S2-B')
    assert run1[1]['pair_id'] == generate_pair_id('S1-A', 'S3-C')

    # Collision-safety check: different entity pairs produce distinct pair IDs
    assert generate_pair_id('S1-001', 'S2-002') != generate_pair_id('S1-002', 'S2-001')
    assert generate_pair_id('S1-A', 'S2-B') != generate_pair_id('S1-A', 'S3-B')


def test_enrich_backward_compatibility_preserves_existing_pair_id():
    """Already-assigned pair_id and candidate_source are preserved if valid."""
    rows = [
        {'pair_id': 'custom-pair-1', 'source1_entity_id': 'S1-001',
         'candidate_entity_id': 'S2-001', 'candidate_source': 'source2'},
        {'source1_entity_id': 'S1-002', 'candidate_entity_id': 'S3-002'},
    ]
    enriched = enrich_candidate_batch(rows)
    assert enriched[0]['pair_id'] == 'custom-pair-1'
    assert enriched[0]['candidate_source'] == 'source2'
    assert enriched[1]['pair_id'] == 'S1-002::S3-002'
    assert enriched[1]['candidate_source'] == 'source3'


@pytest.mark.parametrize('s1_id, cand_id, match_err', [
    ('S2-001', 'S2-001', 'S1-prefixed'),
    ('X-001', 'S2-001', 'S1-prefixed'),
    ('S1-', 'S2-001', 'S1-prefixed'),
    ('S1-001', 'S1-002', 'S2- or S3-prefixed'),
    ('S1-001', 'S4-001', 'S2- or S3-prefixed'),
    ('S1-001', '001', 'S2- or S3-prefixed'),
    ('S1-001', 'S2-', 'S2- or S3-prefixed'),
    ('S1-001', 'S3-', 'S2- or S3-prefixed'),
])
def test_enrich_rejects_invalid_prefixes(s1_id, cand_id, match_err):
    raw = [{'source1_entity_id': s1_id, 'candidate_entity_id': cand_id}]
    with pytest.raises(ValueError, match=match_err):
        enrich_candidate_batch(raw)


def test_enrich_rejects_conflicting_candidate_source():
    raw = [
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001',
         'candidate_source': 'source3'},
    ]
    with pytest.raises(ValueError, match='conflicts with derived source'):
        enrich_candidate_batch(raw)


def test_enrich_rejects_duplicate_pairs():
    """Duplicate candidate pairs are rejected even if pair_id differs."""
    # Exact duplicate raw rows
    raw_exact = [
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
    ]
    with pytest.raises(ValueError, match='duplicate'):
        enrich_candidate_batch(raw_exact)

    # Same candidate pair with different custom pair IDs
    raw_diff_id = [
        {'pair_id': 'p-1', 'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
        {'pair_id': 'p-2', 'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'},
    ]
    with pytest.raises(ValueError, match='duplicate'):
        enrich_candidate_batch(raw_diff_id)


def test_enrich_preserves_metadata_and_order():
    """Ordering and extra blocker metadata (blocking_key, rank, score) are preserved."""
    raw = [
        {'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001',
         'blocking_key': 'bk:alpha', 'rank': 1, 'score': 0.95, 'method': 'fts5'},
        {'source1_entity_id': 'S1-002', 'candidate_entity_id': 'S3-001',
         'blocking_key': 'bk:beta', 'rank': 2, 'score': 0.85, 'method': 'deterministic'},
        {'source1_entity_id': 'S1-003', 'candidate_entity_id': 'S2-002',
         'blocking_key': 'bk:gamma', 'rank': 3, 'score': 0.70, 'method': 'fts5'},
    ]
    raw_copy = [dict(r) for r in raw]
    enriched = enrich_candidate_batch(raw)

    # Ordering is preserved
    assert [r['source1_entity_id'] for r in enriched] == ['S1-001', 'S1-002', 'S1-003']
    assert [r['candidate_entity_id'] for r in enriched] == ['S2-001', 'S3-001', 'S2-002']

    # Extra diagnostic metadata fields are preserved
    for original, item in zip(raw, enriched):
        assert item['blocking_key'] == original['blocking_key']
        assert item['rank'] == original['rank']
        assert item['score'] == original['score']
        assert item['method'] == original['method']

    # Input dictionaries are not mutated in place
    assert raw == raw_copy
    assert all('pair_id' not in r for r in raw)


def test_enrich_generator_and_empty():
    assert enrich_candidate_batch([]) == []
    raw = [{'source1_entity_id': 'S1-001', 'candidate_entity_id': 'S2-001'}]
    result = enrich_candidate_batch(iter(raw))
    assert len(result) == 1
    assert result[0]['pair_id'] == 'S1-001::S2-001'


def test_enrich_missing_fields_and_non_mapping():
    with pytest.raises(TypeError, match='mapping'):
        enrich_candidate_batch(['not-a-dict'])

    with pytest.raises(ValueError, match='missing required raw candidate fields'):
        enrich_candidate_batch([{'source1_entity_id': 'S1-001'}])

    with pytest.raises(ValueError, match='missing required raw candidate fields'):
        enrich_candidate_batch([{'candidate_entity_id': 'S2-001'}])


def test_enrich_compatible_with_member1_adapter():
    """End-to-end: raw 2-column candidates enriched and joined via Member1Adapter."""
    from business_entity_resolution.integration.member1 import (
        Member1Adapter, partition_candidates,
    )

    data = json.loads(
        (Path(__file__).parent / 'fixtures' / 'member1_small.json').read_text()
    )
    # Strip candidates down to raw blocker output + split/metadata
    raw = []
    for i, c in enumerate(data['candidates']):
        raw.append({
            'source1_entity_id': c['source1_entity_id'],
            'candidate_entity_id': c['candidate_entity_id'],
            'split': c.get('split'),
            'blocking_key': f'key_{i}',
        })

    enriched = enrich_candidate_batch(raw)
    assert len(enriched) == 3
    assert all(r['candidate_source'] in ('source2', 'source3') for r in enriched)
    assert all('::' in r['pair_id'] for r in enriched)

    adapter = Member1Adapter(
        data['source1'], data['source2'], data['source3'], data['split'],
    )
    joined = adapter.join(enriched)
    assert len(joined) == 3
    assert [p.pair_id for p in joined] == [r['pair_id'] for r in enriched]

    train, validation = partition_candidates(joined)
    assert len(train) == 2
    assert len(validation) == 1


def test_enrich_candidates_alias():
    """enrich_candidates is an alias of enrich_candidate_batch."""
    assert enrich_candidates is enrich_candidate_batch


def test_tsv_adapter_s2_and_s3_candidates():
    """TSV input with S2 and S3 candidates expands and derives correct sources and IDs."""
    tsv_content = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002\n"
    )
    adapter = Member2TSVAdapter(tsv_content)
    pairs = list(adapter.iter_candidate_pairs())
    assert len(pairs) == 3

    # Pair 0
    assert pairs[0]['pair_id'] == 'S1-001::S2-001'
    assert pairs[0]['source1_entity_id'] == 'S1-001'
    assert pairs[0]['candidate_entity_id'] == 'S2-001'
    assert pairs[0]['candidate_source'] == 'source2'
    assert pairs[0]['source1_row_index'] == 0
    assert pairs[0]['candidate_rank'] == 0

    # Pair 1
    assert pairs[1]['pair_id'] == 'S1-001::S3-001'
    assert pairs[1]['source1_entity_id'] == 'S1-001'
    assert pairs[1]['candidate_entity_id'] == 'S3-001'
    assert pairs[1]['candidate_source'] == 'source3'
    assert pairs[1]['source1_row_index'] == 0
    assert pairs[1]['candidate_rank'] == 1

    # Pair 2
    assert pairs[2]['pair_id'] == 'S1-002::S2-002'
    assert pairs[2]['source1_entity_id'] == 'S1-002'
    assert pairs[2]['candidate_entity_id'] == 'S2-002'
    assert pairs[2]['candidate_source'] == 'source2'
    assert pairs[2]['source1_row_index'] == 1
    assert pairs[2]['candidate_rank'] == 0

    # Coverage verification
    cov = adapter.coverage
    assert cov is not None
    assert cov.total_source1_entities == 2
    assert cov.all_source1_ids == ('S1-001', 'S1-002')
    assert cov.empty_source1_ids == ()
    assert cov.non_empty_source1_ids == ('S1-001', 'S1-002')
    assert cov.total_candidate_pairs == 3


def test_tsv_adapter_empty_candidate_lists():
    """Source 1 rows with empty candidate lists create no fake pairs and preserve coverage."""
    tsv_content = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\t\n"
        "S1-003\t   \n"
        "S1-004\tS3-001\n"
    )
    adapter = Member2TSVAdapter(tsv_content)
    pairs = list(adapter.iter_candidate_pairs())

    # Only 2 real pairs generated; no fake pairs for S1-002 or S1-003
    assert len(pairs) == 2
    assert [p['source1_entity_id'] for p in pairs] == ['S1-001', 'S1-004']
    assert [p['candidate_entity_id'] for p in pairs] == ['S2-001', 'S3-001']

    # Coverage records all 4 Source 1 entities
    cov = adapter.coverage
    assert cov is not None
    assert cov.total_source1_entities == 4
    assert cov.all_source1_ids == ('S1-001', 'S1-002', 'S1-003', 'S1-004')
    assert cov.empty_source1_ids == ('S1-002', 'S1-003')
    assert cov.non_empty_source1_ids == ('S1-001', 'S1-004')
    assert cov.candidate_counts == {'S1-001': 1, 'S1-002': 0, 'S1-003': 0, 'S1-004': 1}

    # Member 4 empty match records
    empty_records = cov.empty_match_entries()
    assert len(empty_records) == 2
    assert [r['source1_id'] for r in empty_records] == ['S1-002', 'S1-003']
    assert all(r['matches'] == [] and r['num_candidates'] == 0 for r in empty_records)


@pytest.mark.parametrize('line, match_err', [
    ('S1-001', 'expected at least 2 tab-separated columns'),
    ('S1-001 S2-001', 'expected at least 2 tab-separated columns'),
    ('S2-001\tS2-001', 'S1-prefixed string'),
    ('001\tS2-001', 'S1-prefixed string'),
    ('S1-\tS2-001', 'S1-prefixed string'),
    (' S1-001\tS2-001', 'S1-prefixed string'),
    ('S1-001\tS1-002', 'must be S2- or S3-prefixed'),
    ('S1-001\tS4-001', 'must be S2- or S3-prefixed'),
    ('S1-001\t001', 'must be S2- or S3-prefixed'),
    ('S1-001\tS2-', 'empty identifier body'),
    ('S1-001\tS3-', 'empty identifier body'),
    ('S1-001\tS2-001,,S3-001', 'empty token'),
    ('S1-001\tS2-001,', 'empty token'),
    ('S1-001\t,S2-001', 'empty token'),
    ('S1-001\t S2-001', 'leading/trailing whitespace'),
])
def test_tsv_adapter_rejects_malformed_inputs(line, match_err):
    tsv = f"source1_entity_id\tcandidate_entity_ids\n{line}\n"
    adapter = Member2TSVAdapter(tsv)
    with pytest.raises(ValueError, match=match_err):
        list(adapter.iter_candidate_pairs())


def test_tsv_adapter_rejects_duplicates():
    """Duplicate candidate pairs on same line or across lines raise ValueError."""
    # Duplicate on same line
    tsv_same_line = "source1_entity_id\tcandidate_entity_ids\nS1-001\tS2-001,S2-001\n"
    with pytest.raises(ValueError, match='duplicate candidate pair'):
        list(Member2TSVAdapter(tsv_same_line).iter_candidate_pairs())

    # Duplicate across lines
    tsv_cross_line = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-001\tS2-001\n"
    )
    with pytest.raises(ValueError, match='duplicate candidate pair'):
        list(Member2TSVAdapter(tsv_cross_line).iter_candidate_pairs())


def test_tsv_adapter_ordering_preservation():
    """Candidates are emitted in strict input order, never sorted or rearranged."""
    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-003\tS3-009,S2-001\n"
        "S1-001\tS2-005\n"
        "S1-002\tS3-002,S2-004\n"
    )
    pairs = list(Member2TSVAdapter(tsv).iter_candidate_pairs())
    expected = [
        ('S1-003', 'S3-009'),
        ('S1-003', 'S2-001'),
        ('S1-001', 'S2-005'),
        ('S1-002', 'S3-002'),
        ('S1-002', 'S2-004'),
    ]
    assert [(p['source1_entity_id'], p['candidate_entity_id']) for p in pairs] == expected


def test_tsv_adapter_batch_boundaries():
    """Batching produces correct slices across boundary conditions."""
    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S2-002\n"
        "S1-002\t\n"
        "S1-003\tS3-001,S3-002,S2-003\n"
    )
    adapter = Member2TSVAdapter(tsv)

    # batch_size = 2 on 5 total candidates -> [2, 2, 1]
    batches_2 = list(adapter.iter_candidate_batches(batch_size=2))
    assert [len(b) for b in batches_2] == [2, 2, 1]
    assert sum(len(b) for b in batches_2) == 5

    # batch_size = 5 -> [5]
    batches_5 = list(adapter.iter_candidate_batches(batch_size=5))
    assert [len(b) for b in batches_5] == [5]

    # batch_size = 10 -> [5]
    batches_10 = list(adapter.iter_candidate_batches(batch_size=10))
    assert [len(b) for b in batches_10] == [5]

    # Empty TSV -> 0 batches
    empty_adapter = Member2TSVAdapter("source1_entity_id\tcandidate_entity_ids\n")
    assert list(empty_adapter.iter_candidate_batches(batch_size=2)) == []

    # Invalid batch_size raises
    with pytest.raises(ValueError, match='batch_size must be a positive integer'):
        list(adapter.iter_candidate_batches(batch_size=0))
    with pytest.raises(ValueError, match='batch_size must be a positive integer'):
        list(adapter.iter_candidate_batches(batch_size=-1))


def test_tsv_adapter_file_path_and_helpers(tmp_path):
    """Reading from Path and top-level helper functions work correctly."""
    tsv_file = tmp_path / 'candidates.tsv'
    tsv_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001\n"
        "S1-002\t\n"
    )
    candidates, cov = parse_member2_tsv(tsv_file)
    assert cov is not None
    assert len(candidates) == 1
    assert candidates[0]['pair_id'] == 'S1-001::S2-001'
    assert cov.total_source1_entities == 2
    assert cov.empty_source1_ids == ('S1-002',)

    streamed = list(read_member2_tsv(tsv_file))
    assert len(streamed) == 1


def test_tsv_adapter_end_to_end_inference():
    """End-to-end: streaming TSV through Member1 join and InferencePipeline scoring."""
    from business_entity_resolution.integration.member1 import Member1Adapter
    from business_entity_resolution.matching.inference import InferencePipeline

    data = json.loads(
        (Path(__file__).parent / 'fixtures' / 'member1_small.json').read_text()
    )
    member1 = Member1Adapter(
        data['source1'], data['source2'], data['source3'], data['split'],
    )
    model_path = Path(__file__).parent.parent / 'models' / 'member3_synthetic_smoke' / 'model.joblib'
    inference = InferencePipeline(model_path)

    # TSV with S2 candidate, S3 candidate, and an empty Source 1 entity
    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\t\n"
    )
    adapter = Member2TSVAdapter(tsv)
    scores = list(adapter.score_candidates(inference, member1, batch_size=1))

    assert len(scores) == 2
    # Probability in [0, 1]
    assert all(0.0 <= s['probability'] <= 1.0 for s in scores)
    assert scores[0]['source1_id'] == 'S1-001'
    assert scores[0]['target_source'] == 'S2'
    assert scores[0]['target_id'] == 'S2-001'
    assert scores[0]['candidate_rank'] == 0
    assert scores[1]['source1_id'] == 'S1-001'
    assert scores[1]['target_source'] == 'S3'
    assert scores[1]['target_id'] == 'S3-001'
    assert scores[1]['candidate_rank'] == 1

    # Member 4 empty match coverage for S1-002
    cov = adapter.coverage
    assert cov is not None
    assert cov.empty_source1_ids == ('S1-002',)
    empty_matches = cov.empty_match_entries()
    assert len(empty_matches) == 1
    assert empty_matches[0]['source1_id'] == 'S1-002'


def test_tsv_adapter_score_file(tmp_path):
    """score_file writes scores to a JSONL file in bounded batches."""
    from business_entity_resolution.integration.member1 import Member1Adapter
    from business_entity_resolution.matching.inference import InferencePipeline

    data = json.loads(
        (Path(__file__).parent / 'fixtures' / 'member1_small.json').read_text()
    )
    member1 = Member1Adapter(
        data['source1'], data['source2'], data['source3'], data['split'],
    )
    model_path = Path(__file__).parent.parent / 'models' / 'member3_synthetic_smoke' / 'model.joblib'
    inference = InferencePipeline(model_path)

    tsv = (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
    )
    output_file = tmp_path / 'scores.jsonl'
    adapter = Member2TSVAdapter(tsv)
    count = adapter.score_file(output_file, inference, member1, batch_size=1)
    assert count == 2
    assert output_file.exists()

    lines = [json.loads(l) for l in output_file.read_text().splitlines()]
    assert len(lines) == 2
    assert [l['target_source'] for l in lines] == ['S2', 'S3']

    # Will not overwrite existing file
    with pytest.raises(FileExistsError):
        adapter.score_file(output_file, inference, member1)
