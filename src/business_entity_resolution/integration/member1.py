"""Member 1 artifact adapter. No I/O, fitting, or split generation."""

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import json
import re

import importlib

from ..external.member1.disk_lookup import DiskEntityLookup, LookupIntegrityError

try:
    _member1_data_mod = importlib.import_module("src.data.disk_lookup")
    DiskEntityLookup = getattr(_member1_data_mod, "DiskEntityLookup", DiskEntityLookup)
    LookupIntegrityError = getattr(_member1_data_mod, "LookupIntegrityError", LookupIntegrityError)
except ImportError:
    pass

from ..matching.pair_data import CandidatePair, validate_pairs
from ..matching.inference import InputMapping

SOURCES = {'source1': 'S1', 'source2': 'S2', 'source3': 'S3'}


def _id(value, source):
    if (not isinstance(value, str) or value != value.strip() or
            not value.startswith(source + '-') or not value[len(source) + 1:].strip()):
        raise ValueError(f'Expected a full {source}- entity ID; got {value!r}.')
    return value


def _text(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError('Normalized fields must be strings or None (convert NaN upstream).')
    return value if value.strip() else None


@dataclass(frozen=True)
class NormalizedRecord:
    entity_id: str
    original_source: str
    name: str | None
    address: str | None
    country: str | None
    postal_codes: tuple[str, ...]
    original_postal_codes: str | None

    def ml_record(self, *, mask_multiple_postal_codes=False, collection_aware=False):
        """Scalar bridge: reject multiple codes unless explicitly masked as missing.

        The full collection remains on this record; never choose or concatenate.
        Masking is a temporary, explicit opt-in, not collection-aware matching.
        """
        if collection_aware:
            if mask_multiple_postal_codes:
                raise ValueError('Collection export and scalar masking are mutually exclusive.')
            return dict(name=self.name, address=self.address, country=self.country,
                        postal_code=None, postal_codes=list(self.postal_codes))
        if len(self.postal_codes) > 1 and not mask_multiple_postal_codes:
            raise ValueError('Multiple postal codes require collection-aware features or explicit masking.')
        return dict(name=self.name, address=self.address, country=self.country,
                    postal_code=self.postal_codes[0] if len(self.postal_codes) == 1 else None)


def adapt_record(row, source):
    if source not in SOURCES:
        raise ValueError('Source must be source1, source2 or source3.')
    raw_postal = _text(row['postal_codes'])
    return NormalizedRecord(
        _id(row['entity_id'], SOURCES[source]), source,
        _text(row['business_name_normalized']), _text(row['business_address_normalized']),
        _text(row['country_normalized']), tuple(raw_postal.split()) if raw_postal else (),
        row['postal_codes'],
    )


def load_split_dict(artifact: str | Path | Mapping) -> dict[str, list[str]]:
    """Load split definition mapping from a dict, JSON file path, or TSV file path."""
    if isinstance(artifact, Mapping):
        return dict(artifact)
    path = Path(artifact)
    if not path.is_file():
        raise FileNotFoundError(f"Saved split file not found: {path}")

    # JSON format (e.g. validation_split.json)
    if path.suffix.lower() == '.json':
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
        return dict(data)

    # TSV format (e.g. validation_membership.tsv)
    train_ids = []
    val_ids = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty split file: {path}")
        header_lower = [h.strip().lower() for h in header]
        if 'source1_entity_id' not in header_lower or 'split' not in header_lower:
            raise ValueError(
                f"Unexpected header in split TSV {path}. "
                f"Expected 'source1_entity_id' and 'split', got {header}"
            )
        s1_idx = header_lower.index('source1_entity_id')
        split_idx = header_lower.index('split')

        for line_no, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Split line {line_no}: expected {len(header)} columns, got {len(row)}")
            s1_id = row[s1_idx].strip()
            split_val = row[split_idx].strip().lower()
            if split_val == 'train':
                train_ids.append(s1_id)
            elif split_val == 'validation':
                val_ids.append(s1_id)
            else:
                raise ValueError(
                    f"Split line {line_no}: invalid split value {row[split_idx]!r}. "
                    "Must be 'train' or 'validation'."
                )
    return {
        'train_source1_entity_ids': train_ids,
        'validation_source1_entity_ids': val_ids,
    }


class SavedSplit:
    """Use Member 1's saved ID lists exactly; unknown IDs never default to train."""

    def __init__(self, artifact):
        if isinstance(artifact, SavedSplit):
            self.train_ids = artifact.train_ids
            self.validation_ids = artifact.validation_ids
            return
        if isinstance(artifact, (str, Path)):
            artifact = load_split_dict(artifact)
        def read(key):
            values = artifact[key]
            if not isinstance(values, (list, tuple)):
                raise TypeError(f'{key} must be a list of IDs.')
            ids = [_id(value, 'S1') for value in values]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate ID in saved split.')
            return frozenset(ids)
        self.train_ids = read('train_source1_entity_ids')
        self.validation_ids = read('validation_source1_entity_ids')
        if self.train_ids & self.validation_ids:
            raise ValueError('Overlapping Source 1 split IDs.')

    def assignment(self, source1_id):
        _id(source1_id, 'S1')
        if source1_id in self.train_ids:
            return 'train'
        if source1_id in self.validation_ids:
            return 'validation'
        raise ValueError(f'Unknown Source 1 ID in saved split: {source1_id}')


def load_saved_split(artifact: str | Path | Mapping) -> SavedSplit:
    """Load SavedSplit from mapping, JSON file path, or TSV membership file path."""
    return SavedSplit(artifact)


@dataclass(frozen=True)
class JoinedCandidate:
    pair_id: str
    source1: NormalizedRecord
    target: NormalizedRecord
    candidate_source: str
    label: int | None
    split: str

    def ml_pair(self, **postal_policy):
        return CandidatePair(self.pair_id, self.source1.entity_id, self.target.entity_id,
                             self.source1.ml_record(**postal_policy),
                             self.target.ml_record(**postal_policy), self.label, self.split)

    def inference_row(self, **postal_policy):
        pair = self.ml_pair(**postal_policy)
        return dict(pair_id=self.pair_id, source1_id=pair.source1_id,
                    target_id=pair.source2_id, target_source=SOURCES[self.candidate_source],
                    source1=pair.source1, target=pair.source2,
                    source1_postal_codes=list(self.source1.postal_codes),
                    target_postal_codes=list(self.target.postal_codes),
                    candidate_source=self.candidate_source, label=self.label, split=self.split)


def inference_mapping(*, collection_aware=False):
    """Mapping for JoinedCandidate.inference_row and the unchanged inference API."""
    fields = ('name', 'address', 'country', 'postal_code')
    if collection_aware:
        fields += ('postal_codes',)
    return InputMapping(['pair_id'], ['source1_id'], ['target_id'],
                        {key: ['source1', key] for key in fields},
                        {key: ['target', key] for key in fields}, target_source=['target_source'])


_MATCH_TOKEN_RE = re.compile(r"S[23]-[A-Za-z0-9_]+")


def parse_ground_truth_matches(raw: str) -> list[str]:
    """Parse common ground-truth match ID encodings; empty/null returns empty list."""
    if not isinstance(raw, str):
        raise TypeError(f"Expected string for matched_entity_ids; got {type(raw).__name__}")
    text = raw.strip()
    if not text or text in ("[]", "null", "None"):
        return []
    tokens = _MATCH_TOKEN_RE.findall(text)
    remainder = _MATCH_TOKEN_RE.sub("", text)
    if re.search(r"[^\s,;\[\]\(\)\{\}'\"|]", remainder):
        raise ValueError(f"Unrecognized characters in matched_entity_ids: {raw!r}")
    return tokens


def load_ground_truth_tsv(path: str | Path) -> dict[str, set[str]]:
    """Load and validate Member 1 ground-truth TSV artifact.

    Maps covered Source 1 entity IDs to complete sets of true-match target IDs (S2/S3).
    A present empty set confirms zero matches for that entity.
    Rejects duplicate Source 1 rows, malformed entity IDs, and unrecognized schemas.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty ground-truth file: {path}")
        if "source1_entity_id" not in header or "matched_entity_ids" not in header:
            raise ValueError(
                f"Unexpected ground-truth header in {path}. "
                f"Expected 'source1_entity_id' and 'matched_entity_ids', found {header}"
            )
        s1_idx = header.index("source1_entity_id")
        m_idx = header.index("matched_entity_ids")

        for line_no, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(
                    f"Ground-truth line {line_no}: expected {len(header)} columns, got {len(row)}"
                )
            s1_id = row[s1_idx].strip()
            _id(s1_id, "S1")
            if s1_id in result:
                raise ValueError(f"Duplicate Source 1 row in ground truth: {s1_id}")

            raw_matches = row[m_idx]
            match_ids = parse_ground_truth_matches(raw_matches)
            for m in match_ids:
                source = "S2" if m.startswith("S2-") else "S3"
                _id(m, source)
            result[s1_id] = set(match_ids)

    return result


def _ground_truth(coverage):
    if not isinstance(coverage, Mapping):
        raise TypeError('Ground truth must map Source 1 IDs to complete collections of match IDs.')
    result = {}
    for source1_id, targets in coverage.items():
        _id(source1_id, 'S1')
        if not isinstance(targets, (list, tuple, set, frozenset)):
            raise TypeError('Ground-truth values must be collections, not strings or None.')
        checked = set()
        for target in targets:
            source = 'S2' if isinstance(target, str) and target.startswith('S2-') else 'S3'
            checked.add(_id(target, source))
        result[source1_id] = checked
    return result


class Member1Adapter:
    """Index normalized tables and join candidates without changing input order.

    Ground truth must be complete for each covered S1 entity. A present empty set
    confirms no matches; an absent key is unknown. Without ground truth, supplied
    labels are preserved as caller-authoritative and absent labels stay None.
    """

    def __init__(self, source1, source2, source3, split):
        self.split = SavedSplit(split)
        self.records = {}
        for source, rows in zip(SOURCES, (source1, source2, source3), strict=True):
            index = {}
            for row in rows:
                record = adapt_record(row, source)
                if record.entity_id in index:
                    raise ValueError(f'Ambiguous record join: duplicate {record.entity_id}')
                index[record.entity_id] = record
            self.records[source] = index

    def join(self, candidates, *, ground_truth=None, require_labels=False):
        truth = _ground_truth(ground_truth) if ground_truth is not None else None
        joined = []
        seen_pairs, seen_edges = set(), set()
        for row in candidates:
            pair_id = row['pair_id']
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError('A nonempty pair_id from the candidate producer is required.')
            source = row['candidate_source']
            if source not in ('source2', 'source3'):
                raise ValueError('candidate_source must be source2 or source3.')
            left = _id(row['source1_entity_id'], 'S1')
            right = _id(row['candidate_entity_id'], SOURCES[source])
            if pair_id in seen_pairs or (left, right) in seen_edges:
                raise ValueError('Duplicate pair ID or candidate edge.')
            seen_pairs.add(pair_id)
            seen_edges.add((left, right))
            assignment = self.split.assignment(left)
            if 'split' in row and row['split'] != assignment:
                raise ValueError('Candidate split assignment conflicts with saved split.')
            try:
                left_record = self.records['source1'][left]
                right_record = self.records[source][right]
            except KeyError as exc:
                raise ValueError(f'Missing record join: {exc.args[0]}') from exc
            label = row.get('label')
            if label is not None:
                if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
                    raise ValueError('Labels must be integer 0 or 1.')
                label = int(label)
            if truth is not None:
                if left not in truth:
                    raise ValueError(f'Missing ground-truth coverage for {left}.')
                derived = int(right in truth[left])
                if label is not None and label != derived:
                    raise ValueError('Supplied label conflicts with ground truth.')
                label = derived
            if require_labels and label is None:
                raise ValueError('Labeled candidates or complete ground-truth coverage required.')
            joined.append(JoinedCandidate(pair_id, left_record, right_record, source, label, assignment))
        # Reuse existing pair validation, explicitly masking only in this validation
        # projection. The returned records still retain every original postal code.
        validate_pairs(item.ml_pair(mask_multiple_postal_codes=True) for item in joined)
        return joined


def partition_candidates(joined):
    """Preserve relative row order in each already-assigned partition."""
    train, validation = [], []
    for candidate in joined:
        if candidate.split not in ('train', 'validation'):
            raise ValueError('Invalid split assignment.')
        (train if candidate.split == 'train' else validation).append(candidate)
    return train, validation


class DiskMember1Adapter:
    """Disk-backed Member 1 record join adapter using Member 1's DiskEntityLookup.

    Retrieves normalized records on-demand using DiskEntityLookup.get_many() in
    bounded batches without preloading full entity catalogs into RAM.

    Preserves candidate order, pair IDs, labels, split assignments, complete
    entity IDs, normalized fields, empty-string missing values, and postal
    collection v2 behavior. Rejects missing or ambiguous record joins.
    """

    def __init__(self, lookup, split, *, data_dir=None, verify_hash=False):
        if isinstance(lookup, (str, Path)):
            self.lookup = DiskEntityLookup(lookup, data_dir=data_dir, verify_hash=verify_hash)
            self._owns_lookup = True
        elif hasattr(lookup, 'get_many'):
            self.lookup = lookup
            self._owns_lookup = False
        else:
            raise TypeError('lookup must be a DiskEntityLookup instance or a path to SQLite index.')

        self.split = split if isinstance(split, SavedSplit) else SavedSplit(split)

    def close(self):
        if self._owns_lookup and self.lookup is not None:
            self.lookup.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def join(self, candidates, *, ground_truth=None, require_labels=False):
        """Join candidate batch against disk lookup without preloading entity catalogs.

        Retrieves required Source 1, Source 2, and Source 3 records in a single
        bounded batch call to DiskEntityLookup.get_many().
        Preserves candidate order and all candidate metadata.
        Rejects missing records, ambiguous joins, and invalid labels/splits.
        """
        candidate_list = list(candidates)
        if not candidate_list:
            return []

        truth = _ground_truth(ground_truth) if ground_truth is not None else None
        seen_pairs, seen_edges = set(), set()
        needed_ids = []

        for row in candidate_list:
            pair_id = row['pair_id']
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError('A nonempty pair_id from the candidate producer is required.')
            source = row['candidate_source']
            if source not in ('source2', 'source3'):
                raise ValueError('candidate_source must be source2 or source3.')
            left = _id(row['source1_entity_id'], 'S1')
            right = _id(row['candidate_entity_id'], SOURCES[source])
            if pair_id in seen_pairs or (left, right) in seen_edges:
                raise ValueError('Duplicate pair ID or candidate edge.')
            seen_pairs.add(pair_id)
            seen_edges.add((left, right))
            assignment = self.split.assignment(left)
            if 'split' in row and row['split'] != assignment:
                raise ValueError('Candidate split assignment conflicts with saved split.')
            needed_ids.append(left)
            needed_ids.append(right)

        # Retrieve records in a bounded batch using DiskEntityLookup.get_many()
        unique_ids = list(dict.fromkeys(needed_ids))
        records_raw = self.lookup.get_many(unique_ids)
        record_map = dict(zip(unique_ids, records_raw, strict=True))

        joined = []
        for row in candidate_list:
            pair_id = row['pair_id']
            source = row['candidate_source']
            left = row['source1_entity_id']
            right = row['candidate_entity_id']
            assignment = self.split.assignment(left)

            raw_left = record_map.get(left)
            if raw_left is None:
                raise ValueError(f'Missing record join: {left}')
            raw_right = record_map.get(right)
            if raw_right is None:
                raise ValueError(f'Missing record join: {right}')

            left_record = adapt_record(raw_left, 'source1')
            right_record = adapt_record(raw_right, source)

            label = row.get('label')
            if label is not None:
                if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
                    raise ValueError('Labels must be integer 0 or 1.')
                label = int(label)
            if truth is not None:
                if left not in truth:
                    raise ValueError(f'Missing ground-truth coverage for {left}.')
                derived = int(right in truth[left])
                if label is not None and label != derived:
                    raise ValueError('Supplied label conflicts with ground truth.')
                label = derived
            if require_labels and label is None:
                raise ValueError('Labeled candidates or complete ground-truth coverage required.')

            joined.append(JoinedCandidate(pair_id, left_record, right_record, source, label, assignment))

        validate_pairs(item.ml_pair(mask_multiple_postal_codes=True) for item in joined)
        return joined


Member1DiskAdapter = DiskMember1Adapter

__all__ = [
    'SOURCES',
    'NormalizedRecord',
    'SavedSplit',
    'JoinedCandidate',
    'Member1Adapter',
    'DiskMember1Adapter',
    'Member1DiskAdapter',
    'adapt_record',
    'inference_mapping',
    'partition_candidates',
    'load_ground_truth_tsv',
    'parse_ground_truth_matches',
    'load_saved_split',
    'load_split_dict',
]
