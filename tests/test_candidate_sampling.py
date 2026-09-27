"""Tests for bounded-memory candidate labeling and stratified negative sampling."""

import pytest

from business_entity_resolution.matching.pair_data import CandidatePair
from business_entity_resolution.matching.sampling import (
    sample_training_candidates,
)


def _make_pair(s1_id, cand_id, label, split='train'):
    return CandidatePair(
        pair_id=f"{s1_id}::{cand_id}",
        source1_id=s1_id,
        source2_id=cand_id,
        source1={"name": f"Name {s1_id}", "address": "123 Main St", "postal_code": "10001", "country": "US"},
        source2={"name": f"Name {cand_id}", "address": "123 Main St", "postal_code": "10001", "country": "US"},
        label=label,
        split=split,
    )


def test_sampling_preserves_100_percent_positives():
    """All positive matches in training candidates are 100% preserved."""
    # S1-001 has 3 positives and 30 negatives
    pairs = [
        _make_pair("S1-001", f"S2-{i:03d}", label=1 if i in (5, 15, 25) else 0)
        for i in range(33)
    ]
    sampled = list(sample_training_candidates(pairs, negative_ratio=2, seed=42))

    # All 3 positives must be present
    sampled_pos = [p for p in sampled if p.label == 1]
    assert len(sampled_pos) == 3
    assert {p.source2_id for p in sampled_pos} == {"S2-005", "S2-015", "S2-025"}

    # Negatives sampled to 3 * 2 = 6
    sampled_neg = [p for p in sampled if p.label == 0]
    assert len(sampled_neg) == 6
    assert len(sampled) == 9


def test_sampling_leaves_validation_unaffected():
    """Validation candidates are never downsampled; full candidate space is preserved."""
    pairs = [
        _make_pair("S1-VAL", f"S2-{i:03d}", label=1 if i == 0 else 0, split="validation")
        for i in range(50)
    ]
    sampled = list(sample_training_candidates(pairs, negative_ratio=2, seed=42))
    assert len(sampled) == 50
    assert sampled == pairs


def test_zero_match_entity_sampling():
    """Entities with zero positive matches retain up to max_zero_match_negatives."""
    pairs = [
        _make_pair("S1-ZERO", f"S2-{i:03d}", label=0, split="train")
        for i in range(40)
    ]
    sampled = list(sample_training_candidates(pairs, negative_ratio=10, max_zero_match_negatives=7, seed=42))
    assert len(sampled) == 7
    assert all(p.label == 0 for p in sampled)


def test_deterministic_reproducibility():
    """Identical seeds produce identical candidate selections; different seeds produce different negatives."""
    pairs = [
        _make_pair("S1-001", f"S2-{i:03d}", label=1 if i == 0 else 0)
        for i in range(30)
    ]
    run1 = list(sample_training_candidates(pairs, negative_ratio=5, seed=123))
    run2 = list(sample_training_candidates(pairs, negative_ratio=5, seed=123))
    run3 = list(sample_training_candidates(pairs, negative_ratio=5, seed=999))

    assert [p.pair_id for p in run1] == [p.pair_id for p in run2]
    # Positives are preserved in both
    assert [p.pair_id for p in run1 if p.label == 1] == [p.pair_id for p in run3 if p.label == 1]
    # Sampled negatives differ across different seeds
    assert [p.pair_id for p in run1 if p.label == 0] != [p.pair_id for p in run3 if p.label == 0]


def test_relative_order_preserved():
    """Sampling preserves original input ordering among retained candidates."""
    pairs = [
        _make_pair("S1-001", f"S2-{i:03d}", label=1 if i in (1, 10) else 0)
        for i in range(20)
    ]
    sampled = list(sample_training_candidates(pairs, negative_ratio=3, seed=42))
    sampled_indices = [int(p.source2_id.split("-")[1]) for p in sampled]
    assert sampled_indices == sorted(sampled_indices)


def test_unlabeled_training_candidate_raises():
    """Training candidate with missing label raises ValueError."""
    pairs = [
        _make_pair("S1-001", "S2-001", label=None, split="train")
    ]
    with pytest.raises(ValueError, match="Unlabeled candidate"):
        list(sample_training_candidates(pairs))


def test_parameter_validation():
    """Invalid arguments raise ValueError."""
    pairs = [_make_pair("S1-001", "S2-001", label=1)]
    with pytest.raises(ValueError, match="negative_ratio"):
        list(sample_training_candidates(pairs, negative_ratio=0))
    with pytest.raises(ValueError, match="max_zero_match_negatives"):
        list(sample_training_candidates(pairs, max_zero_match_negatives=-1))


def test_multi_entity_streaming():
    """Streaming across multiple entities processes each entity independently."""
    pairs = []
    # Entity 1: train, 1 pos, 20 negs -> 1 pos + 2 negs
    pairs.extend([_make_pair("S1-001", f"S2-A{i}", label=1 if i == 0 else 0, split="train") for i in range(21)])
    # Entity 2: validation, 1 pos, 10 negs -> all 11
    pairs.extend([_make_pair("S1-002", f"S2-B{i}", label=1 if i == 0 else 0, split="validation") for i in range(11)])
    # Entity 3: train, 0 pos, 15 negs -> 3 negs
    pairs.extend([_make_pair("S1-003", f"S2-C{i}", label=0, split="train") for i in range(15)])

    sampled = list(sample_training_candidates(pairs, negative_ratio=2, max_zero_match_negatives=3, seed=42))

    e1 = [p for p in sampled if p.source1_id == "S1-001"]
    e2 = [p for p in sampled if p.source1_id == "S1-002"]
    e3 = [p for p in sampled if p.source1_id == "S1-003"]

    assert len(e1) == 1 + 2  # 1 pos + 2 negs
    assert len(e2) == 11     # validation unaffected
    assert len(e3) == 3      # 3 zero-match negs
