"""Bounded-memory candidate labeling and stratified negative sampling for training."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import random
from typing import Any, TypeVar

from .pair_data import CandidatePair
from ..integration.member1 import JoinedCandidate

T = TypeVar('T', JoinedCandidate, CandidatePair)


@dataclass(frozen=True)
class SamplingSummary:
    total_entities: int
    train_entities: int
    validation_entities: int
    train_positives: int
    train_negatives_retained: int
    train_negatives_dropped: int
    validation_pairs: int

    @property
    def total_retained_pairs(self) -> int:
        return self.train_positives + self.train_negatives_retained + self.validation_pairs


def _get_s1_id(item: JoinedCandidate | CandidatePair) -> str:
    if isinstance(item, CandidatePair):
        return item.source1_id
    if isinstance(item, JoinedCandidate):
        return item.source1.entity_id
    raise TypeError(f"Unsupported candidate type: {type(item).__name__}")


def _process_entity_group(
    group: list[T],
    negative_ratio: int | None,
    max_zero_match_negatives: int,
    seed: int,
    s1_id: str,
) -> Iterator[T]:
    if not group:
        return

    split = group[0].split
    if split == 'validation':
        yield from group
        return

    positives = []
    negatives = []
    for item in group:
        if item.label == 1:
            positives.append(item)
        elif item.label == 0:
            negatives.append(item)
        else:
            raise ValueError(f"Unlabeled candidate for training entity {s1_id}: label={item.label!r}")

    if negative_ratio is None:
        yield from group
        return

    target_count = len(positives) * negative_ratio if positives else max_zero_match_negatives
    if len(negatives) <= target_count:
        retained_negatives = negatives
    else:
        rng = random.Random(f"{seed}-{s1_id}")
        selected_indices = set(rng.sample(range(len(negatives)), target_count))
        retained_negatives = [negatives[i] for i in sorted(selected_indices)]

    retained_ids = set(id(x) for x in positives) | set(id(x) for x in retained_negatives)
    for item in group:
        if id(item) in retained_ids:
            yield item


def sample_training_candidates(
    candidates: Iterable[T],
    *,
    negative_ratio: int | None = 10,
    max_zero_match_negatives: int = 10,
    seed: int = 42,
) -> Iterator[T]:
    """Stream candidate pairs, preserving 100% of positive matches and sampling negatives.

    Invariants:
    1. Validation candidates (split == 'validation') pass through 100% un-sampled.
    2. Training positive matches (split == 'train' and label == 1) pass through 100% un-sampled.
       Never drops a positive match present in the candidate set.
    3. Training negative matches (split == 'train' and label == 0) are deterministically
       downsampled per Source 1 entity:
       - If an entity has positive matches, retain min(len(negatives), len(positives) * negative_ratio).
       - If an entity has zero positive matches, retain min(len(negatives), max_zero_match_negatives).
    4. Deterministic sampling uses a seeded PRNG keyed by (seed, source1_id).
    5. Preserves relative input order among retained pairs.
    6. Memory is strictly bounded to the candidate count of a single Source 1 entity.
    """
    if negative_ratio is not None and (type(negative_ratio) is not int or negative_ratio < 1):
        raise ValueError("negative_ratio must be a positive integer or None.")
    if type(max_zero_match_negatives) is not int or max_zero_match_negatives < 0:
        raise ValueError("max_zero_match_negatives must be a nonnegative integer.")

    current_s1 = None
    group = []

    for item in candidates:
        s1 = _get_s1_id(item)
        if current_s1 is not None and s1 != current_s1:
            yield from _process_entity_group(group, negative_ratio, max_zero_match_negatives, seed, current_s1)
            group = []
        current_s1 = s1
        group.append(item)

    if group and current_s1 is not None:
        yield from _process_entity_group(group, negative_ratio, max_zero_match_negatives, seed, current_s1)


__all__ = [
    'SamplingSummary',
    'sample_training_candidates',
]
