"""Candidate-pair contract and Source 1 grouped splitting."""

from collections.abc import Mapping
from dataclasses import dataclass

from sklearn.model_selection import GroupShuffleSplit
from ..features.postal_collections import record_postal_codes


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    source1_id: str
    source2_id: str
    source1: Mapping
    source2: Mapping
    label: int | None = None
    split: str | None = None


def validate_pairs(pairs, *, require_labels=False):
    """Validate and materialize input without changing row order."""
    pairs = list(pairs)
    seen_ids, seen_edges, records = set(), set(), {}
    for pair in pairs:
        if not isinstance(pair, CandidatePair):
            raise TypeError("Expected CandidatePair instances.")
        for value in (pair.pair_id, pair.source1_id, pair.source2_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Pair and entity IDs must be nonempty strings.")
        edge = (pair.source1_id, pair.source2_id)
        if pair.pair_id in seen_ids or edge in seen_edges:
            raise ValueError("Duplicate pair ID or candidate entity pair.")
        seen_ids.add(pair.pair_id)
        seen_edges.add(edge)
        if pair.split is not None and pair.split not in ('train', 'validation'):
            raise ValueError('Pair split must be train or validation.')
        if require_labels or pair.label is not None:
            if type(pair.label) is not int or pair.label not in (0, 1):
                raise ValueError("Labels must be integers 0 or 1.")
        for source, entity_id, record in (
            (1, pair.source1_id, pair.source1), (2, pair.source2_id, pair.source2)
        ):
            if not isinstance(record, Mapping):
                raise TypeError("Records must be mappings.")
            fields = tuple(record.get(key) for key in ('name', 'address', 'postal_code', 'country'))
            if any(value is not None and not isinstance(value, str) for value in fields):
                raise TypeError("Record fields must contain strings or None.")
            fields = fields + (record_postal_codes(record),)
            key = (source, entity_id)
            if key in records and records[key] != fields:
                raise ValueError(f"Inconsistent record for source {source} entity {entity_id}.")
            records[key] = fields
    return pairs


def split_pairs(pairs, validation_fraction=0.2, random_state=42):
    """One reproducible group split; never retry splits based on labels or scores."""
    pairs = validate_pairs(pairs, require_labels=True)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    groups = [pair.source1_id for pair in pairs]
    if len(set(groups)) < 2:
        raise ValueError("At least two Source 1 entities are required.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)
    train_indices, valid_indices = next(splitter.split(pairs, groups=groups))
    train = [pairs[int(i)] for i in train_indices]
    valid = [pairs[int(i)] for i in valid_indices]
    if {pair.label for pair in train} != {0, 1}:
        raise ValueError("Training split must contain both classes; supply more labeled groups.")
    return train, valid


def unique_training_records(pairs):
    """Deduplicate by source-qualified entity ID, not by text or pair frequency."""
    records = {}
    for pair in pairs:
        records.setdefault((1, pair.source1_id), pair.source1)
        records.setdefault((2, pair.source2_id), pair.source2)
    return list(records.values())
