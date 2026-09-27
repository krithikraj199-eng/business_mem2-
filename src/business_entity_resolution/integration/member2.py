"""Member 2 candidate adapter. Validates blocking output before record joins.

Member 2 owns candidate generation (deterministic blocking with FTS5 fallback).
This adapter handles both Member 2's raw candidate output formats:
1. Two-column candidate schema (``source1_entity_id``, ``candidate_entity_id``)
   via ``enrich_candidate_batch()``.
2. Two-column candidate TSV output (``source1_entity_id``, ``candidate_entity_ids``)
   with comma-separated candidate lists via ``Member2TSVAdapter``.

Enrichment boundary
-------------------

Before candidate pairs can be scored or joined to Member 1's record tables,
rows must satisfy Member 3's 4-column contract:

- ``pair_id``: a stable, collision-safe, deterministic string identifying each
  candidate pair (generated as ``f'{source1_id}::{candidate_id}'`` if absent).
- ``candidate_source``: ``'source2'`` or ``'source3'``, derived from the entity
  ID prefix (``'S2-'`` -> ``'source2'``, ``'S3-'`` -> ``'source3'``).

Optional fields (``label``, ``split``) may be added during enrichment or by
downstream callers. All other fields on the row (e.g. ``blocking_key``,
``match_score``, ``rank``) are **preserved as-is** and passed through to
``Member1Adapter.join()``, which safely ignores unknown fields.

TSV Streaming & Bounded-Memory Batching
---------------------------------------

Member 2's candidate TSV output contains ``source1_entity_id`` and a comma-separated
list of ``candidate_entity_ids``. ``Member2TSVAdapter`` streams this file:
- Expands candidate lists into individual candidate pairs.
- Preserves candidate ordering and diagnostic provenance (row index, candidate rank).
- Validates entity ID prefixes and rejects unexpected duplicates and malformed IDs.
- Tracks Source 1 coverage (including rows with empty candidate lists) without creating
  fake pairs, enabling Member 4 to produce complete empty match lists downstream.
- Provides bounded-memory batching (``iter_candidate_batches``, ``stream_joined_batches``,
  ``score_candidates``) suitable for millions of candidate pairs without out-of-memory
  pressure.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import io
import json
from numbers import Integral
from pathlib import Path
import sqlite3
import tempfile


# Required fields on each enriched candidate row.
REQUIRED_CANDIDATE_FIELDS = frozenset({
    'pair_id', 'source1_entity_id', 'candidate_entity_id', 'candidate_source',
})

# Fields with structured validation beyond presence checks.
VALIDATED_OPTIONAL_FIELDS = frozenset({'label', 'split'})

VALID_CANDIDATE_SOURCES = frozenset({'source2', 'source3'})

SOURCE_PREFIXES = {'source2': 'S2-', 'source3': 'S3-'}


def validate_candidate_batch(candidates):
    """Validate and materialize an enriched candidate batch.

    Returns a list of validated candidate dicts ready for Member1Adapter.join().
    Raises on schema violations, including missing fields, wrong types,
    invalid sources, and duplicate pair IDs or edges.

    All input fields are preserved in the output, including any extra metadata
    from Member 2's blocker (e.g. blocking_key, rank). Member1Adapter.join()
    safely ignores fields it does not consume.

    This validation is intentionally strict: invalid rows fail the batch
    rather than being silently dropped. Member1Adapter.join() performs
    additional validation (record joins, split consistency, ground truth).
    """
    if not isinstance(candidates, list):
        candidates = list(candidates)
    seen_pair_ids = set()
    seen_edges = set()
    validated = []

    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise TypeError(
                f'Candidate row {index}: expected a mapping, got {type(row).__name__}.'
            )

        # Check required fields.
        missing = REQUIRED_CANDIDATE_FIELDS - set(row)
        if missing:
            raise ValueError(
                f'Candidate row {index}: missing required fields {sorted(missing)}.'
            )

        pair_id = row['pair_id']
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(
                f'Candidate row {index}: pair_id must be a nonempty string.'
            )

        source1_id = row['source1_entity_id']
        if not isinstance(source1_id, str) or not source1_id.startswith('S1-') or not source1_id[3:].strip():
            raise ValueError(
                f'Candidate row {index}: source1_entity_id must be an S1-prefixed string.'
            )

        candidate_source = row['candidate_source']
        if candidate_source not in VALID_CANDIDATE_SOURCES:
            raise ValueError(
                f'Candidate row {index}: candidate_source must be source2 or source3.'
            )

        candidate_id = row['candidate_entity_id']
        expected_prefix = SOURCE_PREFIXES[candidate_source]
        if (not isinstance(candidate_id, str) or not candidate_id.startswith(expected_prefix)
                or not candidate_id[3:].strip()):
            raise ValueError(
                f'Candidate row {index}: candidate_entity_id must be '
                f'{expected_prefix}prefixed for {candidate_source}.'
            )

        # Duplicate detection.
        if pair_id in seen_pair_ids:
            raise ValueError(f'Candidate row {index}: duplicate pair_id {pair_id!r}.')
        seen_pair_ids.add(pair_id)

        edge = (source1_id, candidate_id)
        if edge in seen_edges:
            raise ValueError(f'Candidate row {index}: duplicate edge {edge}.')
        seen_edges.add(edge)

        # Validate optional fields if present.
        label = row.get('label')
        if label is not None:
            if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
                raise ValueError(
                    f'Candidate row {index}: label must be integer 0 or 1.'
                )

        split = row.get('split')
        if split is not None and split not in ('train', 'validation'):
            raise ValueError(
                f'Candidate row {index}: split must be train or validation.'
            )

        # Pass through all fields; normalize only validated optional values.
        validated_row = dict(row)
        if label is not None:
            validated_row['label'] = int(label)
        validated.append(validated_row)

    return validated


def candidate_batch_summary(candidates):
    """Return a diagnostic summary of a validated candidate batch.

    Useful for logging and integration debugging without exposing record data.
    """
    candidates = validate_candidate_batch(candidates)
    s1_ids = {row['source1_entity_id'] for row in candidates}
    sources = {row['candidate_source'] for row in candidates}
    labeled = sum(1 for row in candidates if row.get('label') is not None)
    split_counts = {}
    for row in candidates:
        s = row.get('split', 'unassigned')
        split_counts[s] = split_counts.get(s, 0) + 1
    return dict(
        total_candidates=len(candidates),
        unique_source1_entities=len(s1_ids),
        candidate_sources=sorted(sources),
        labeled_count=labeled,
        unlabeled_count=len(candidates) - labeled,
        split_counts=split_counts,
    )


def generate_pair_id(source1_entity_id, candidate_entity_id):
    """Generate a deterministic, collision-safe pair_id from two complete entity IDs."""
    return f'{source1_entity_id}::{candidate_entity_id}'


def enrich_candidate_batch(candidates):
    """Enrich Member 2 raw candidate rows into the Member 3 candidate contract.

    Converts Member 2's documented two-column candidate output
    (``source1_entity_id``, ``candidate_entity_id``) into the 4-column contract
    required by Member 3 (adding ``pair_id`` and ``candidate_source``), while
    preserving row ordering, optional fields (``label``, ``split``), and any
    diagnostic metadata (e.g. ``blocking_key``, ``rank``).

    - ``pair_id``: generated deterministically and collision-safely from the
      two complete entity IDs (``f'{source1_id}::{candidate_id}'``) if not
      already provided.
    - ``candidate_source``: derived from the complete candidate entity ID prefix
      (``'S2-'`` -> ``'source2'``, ``'S3-'`` -> ``'source3'``).
    - Rejects invalid source prefixes, non-mapping rows, and duplicate
      candidate pairs.
    - Reuses ``validate_candidate_batch()`` for schema validation and integrity
      checks.
    """
    if not isinstance(candidates, list):
        candidates = list(candidates)

    enriched = []
    seen_edges = set()

    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise TypeError(
                f'Candidate row {index}: expected a mapping, got {type(row).__name__}.'
            )

        missing = {'source1_entity_id', 'candidate_entity_id'} - set(row)
        if missing:
            raise ValueError(
                f'Candidate row {index}: missing required raw candidate fields {sorted(missing)}.'
            )

        source1_id = row['source1_entity_id']
        if not isinstance(source1_id, str) or not source1_id.startswith('S1-') or not source1_id[3:].strip():
            raise ValueError(
                f'Candidate row {index}: source1_entity_id must be an S1-prefixed string.'
            )

        candidate_id = row['candidate_entity_id']
        if not isinstance(candidate_id, str):
            raise ValueError(
                f'Candidate row {index}: candidate_entity_id must be an S2- or S3-prefixed string.'
            )

        if candidate_id.startswith('S2-'):
            derived_source = 'source2'
        elif candidate_id.startswith('S3-'):
            derived_source = 'source3'
        else:
            raise ValueError(
                f'Candidate row {index}: candidate_entity_id must be an S2- or S3-prefixed string; got {candidate_id!r}.'
            )

        if not candidate_id[3:].strip():
            raise ValueError(
                f'Candidate row {index}: candidate_entity_id must be an S2- or S3-prefixed string.'
            )

        # Reject duplicate candidate pairs (edges).
        edge = (source1_id, candidate_id)
        if edge in seen_edges:
            raise ValueError(
                f'Candidate row {index}: duplicate candidate pair {edge}.'
            )
        seen_edges.add(edge)

        # Candidate source derivation / verification.
        candidate_source = row.get('candidate_source')
        if candidate_source is None:
            candidate_source = derived_source
        elif candidate_source != derived_source:
            raise ValueError(
                f'Candidate row {index}: candidate_source {candidate_source!r} conflicts with '
                f'derived source {derived_source!r} from candidate_entity_id {candidate_id!r}.'
            )

        # Pair ID generation / preservation.
        pair_id = row.get('pair_id')
        if pair_id is None:
            pair_id = generate_pair_id(source1_id, candidate_id)
        elif not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(
                f'Candidate row {index}: pair_id must be a nonempty string.'
            )

        enriched_row = dict(row)
        enriched_row['pair_id'] = pair_id
        enriched_row['candidate_source'] = candidate_source
        enriched.append(enriched_row)

    return validate_candidate_batch(enriched)


# Backward-compatible alias
enrich_candidates = enrich_candidate_batch


@dataclass(frozen=True)
class Source1Coverage:
    """Coverage tracking for Source 1 entities from Member 2's blocker TSV.

    Preserves complete Source 1 entity coverage (including entities with zero
    candidates) so Member 4 can produce valid empty match lists downstream.
    """
    all_source1_ids: tuple[str, ...]
    empty_source1_ids: tuple[str, ...]
    candidate_counts: Mapping[str, int]
    total_candidate_pairs: int

    @property
    def total_source1_entities(self) -> int:
        return len(self.all_source1_ids)

    @property
    def non_empty_source1_ids(self) -> tuple[str, ...]:
        return tuple(s1 for s1 in self.all_source1_ids if self.candidate_counts.get(s1, 0) > 0)

    def empty_match_entries(self) -> list[dict]:
        """Return empty match records for Member 4 for Source 1 entities with 0 candidates."""
        return [
            dict(source1_id=s1_id, matches=[], num_candidates=0)
            for s1_id in self.empty_source1_ids
        ]


def _iter_tsv_lines(tsv_source):
    """Normalize tsv_source to an iterable of line strings, closing file if opened."""
    if isinstance(tsv_source, str):
        if '\n' in tsv_source or '\r' in tsv_source:
            yield from io.StringIO(tsv_source)
            return
        with open(tsv_source, 'r', encoding='utf-8') as stream:
            for line in stream:
                yield line
    elif isinstance(tsv_source, Path):
        with tsv_source.open('r', encoding='utf-8') as stream:
            for line in stream:
                yield line
    else:
        for line in tsv_source:
            yield line


class _DiskEdgeTracker:
    """Bounded-memory disk-backed exact duplicate edge tracker using SQLite.

    Ensures O(1) RAM usage regardless of whether candidate pairs number in the
    millions. Uses an ephemeral SQLite database with PRAGMA synchronous = OFF,
    PRAGMA journal_mode = OFF, and a bounded cache_size.
    """

    def __init__(self, db_dir: str | Path | None = None, cache_size_kb: int = 8192):
        self._temp_dir = tempfile.TemporaryDirectory(dir=db_dir)
        self._db_path = Path(self._temp_dir.name) / 'seen_edges.db'
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute('PRAGMA synchronous = OFF')
        self._conn.execute('PRAGMA journal_mode = OFF')
        self._conn.execute(f'PRAGMA cache_size = -{cache_size_kb}')
        self._conn.execute(
            'CREATE TABLE seen_edges ('
            '  source1_id TEXT NOT NULL,'
            '  candidate_id TEXT NOT NULL,'
            '  PRIMARY KEY (source1_id, candidate_id)'
            ') WITHOUT ROWID;'
        )
        self._insert_count = 0
        self._closed = False

    def check_and_add(self, source1_id: str, candidate_id: str, line_no: int) -> None:
        try:
            self._conn.execute(
                'INSERT INTO seen_edges VALUES (?, ?)',
                (source1_id, candidate_id),
            )
        except sqlite3.IntegrityError:
            edge = (source1_id, candidate_id)
            raise ValueError(f'TSV line {line_no}: duplicate candidate pair {edge}.')
        self._insert_count += 1
        if self._insert_count % 10000 == 0:
            self._conn.commit()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._conn.close()
            except Exception:
                pass
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class _MemoryEdgeTracker:
    """In-memory exact duplicate edge tracker using a standard Python set."""

    def __init__(self):
        self._seen_edges: set[tuple[str, str]] = set()

    def check_and_add(self, source1_id: str, candidate_id: str, line_no: int) -> None:
        edge = (source1_id, candidate_id)
        if edge in self._seen_edges:
            raise ValueError(f'TSV line {line_no}: duplicate candidate pair {edge}.')
        self._seen_edges.add(edge)

    def close(self) -> None:
        self._seen_edges.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Member2TSVAdapter:
    """Streaming adapter for Member 2's two-column candidate-generation TSV output.

    Expands comma-separated candidate lists into individual candidate pairs,
    derives candidate_source (source2/source3), and generates deterministic,
    collision-safe pair IDs (f'{source1_id}::{candidate_id}').

    Handles Source 1 rows with empty candidate lists without creating fake pairs,
    while tracking complete Source 1 coverage for Member 4 downstream.

    Supports bounded-memory batching for streaming millions of candidates
    through Member 1 record joins and the Member 3 inference pipeline.

    By default, global duplicate detection is disk-backed (SQLite) with bounded
    page cache memory, ensuring safe execution on datasets with millions of pairs.
    """

    def __init__(self, tsv_source=None, *, duplicate_store: str = 'disk',
                 db_dir: str | Path | None = None):
        if duplicate_store not in ('disk', 'memory'):
            raise ValueError(f"duplicate_store must be 'disk' or 'memory'; got {duplicate_store!r}.")
        self.tsv_source = tsv_source
        self.duplicate_store = duplicate_store
        self.db_dir = db_dir
        self._coverage: Source1Coverage | None = None

    @property
    def coverage(self) -> Source1Coverage | None:
        """Return coverage statistics from the most recent streaming run."""
        return self._coverage

    def iter_candidate_pairs(self, tsv_source=None, *, duplicate_store=None, db_dir=None):
        """Stream individual candidate pair dicts from the TSV source.

        Expands comma-separated candidate IDs into individual candidate pair dicts
        in input order. Derives candidate_source from S2/S3 ID prefix and generates
        deterministic pair_id.

        Tracks Source 1 coverage (including empty candidate lists) in self.coverage.
        Rejects malformed IDs and duplicate pairs with ValueError.
        Uses disk-backed exact duplicate detection by default to guarantee bounded RAM.
        """
        source = tsv_source if tsv_source is not None else self.tsv_source
        if source is None:
            raise ValueError('No TSV source provided.')

        dup_store = duplicate_store if duplicate_store is not None else self.duplicate_store
        if dup_store not in ('disk', 'memory'):
            raise ValueError(f"duplicate_store must be 'disk' or 'memory'; got {dup_store!r}.")
        tracker_dir = db_dir if db_dir is not None else self.db_dir

        all_s1_ids = []
        seen_s1_ids = set()
        candidate_counts = {}
        total_pairs = 0
        extra_col_names = []

        header_parsed = False
        data_row_index = 0

        tracker = (
            _DiskEdgeTracker(db_dir=tracker_dir)
            if dup_store == 'disk'
            else _MemoryEdgeTracker()
        )
        try:
            for line_no, raw_line in enumerate(_iter_tsv_lines(source), 1):
                line = raw_line.rstrip('\r\n')
                if not line.strip():
                    continue

                parts = line.split('\t')

                if not header_parsed:
                    p0 = parts[0].strip().lower()
                    p1 = parts[1].strip().lower() if len(parts) > 1 else ''
                    if p0 == 'source1_entity_id' and p1 == 'candidate_entity_ids':
                        header_parsed = True
                        if len(parts) > 2:
                            extra_col_names = [col.strip() for col in parts[2:]]
                        continue
                    header_parsed = True

                if len(parts) < 2:
                    raise ValueError(
                        f'TSV line {line_no}: expected at least 2 tab-separated columns, got {len(parts)}.'
                    )

                source1_id = parts[0]
                if (not isinstance(source1_id, str) or source1_id != source1_id.strip()
                        or not source1_id.startswith('S1-') or not source1_id[3:].strip()):
                    raise ValueError(
                        f'TSV line {line_no}: source1_entity_id must be an S1-prefixed string; got {source1_id!r}.'
                    )

                if source1_id not in seen_s1_ids:
                    all_s1_ids.append(source1_id)
                    seen_s1_ids.add(source1_id)

                raw_cands = parts[1].strip()
                if not raw_cands:
                    if source1_id not in candidate_counts:
                        candidate_counts[source1_id] = 0
                    data_row_index += 1
                    continue

                cand_tokens = parts[1].split(',')
                for cand_index, token in enumerate(cand_tokens):
                    cand_id = token.strip()
                    if not cand_id:
                        raise ValueError(
                            f'TSV line {line_no}: malformed candidate list for {source1_id}; empty token in {parts[1]!r}.'
                        )
                    if token != cand_id:
                        raise ValueError(
                            f'TSV line {line_no}: candidate ID {token!r} has leading/trailing whitespace.'
                        )
                    if cand_id.startswith('S2-'):
                        derived_source = 'source2'
                    elif cand_id.startswith('S3-'):
                        derived_source = 'source3'
                    else:
                        raise ValueError(
                            f'TSV line {line_no}: candidate ID must be S2- or S3-prefixed; got {cand_id!r}.'
                        )
                    if not cand_id[3:].strip():
                        raise ValueError(
                            f'TSV line {line_no}: candidate ID {cand_id!r} has empty identifier body.'
                        )

                    tracker.check_and_add(source1_id, cand_id, line_no)

                    candidate_counts[source1_id] = candidate_counts.get(source1_id, 0) + 1
                    total_pairs += 1

                    pair_id = generate_pair_id(source1_id, cand_id)
                    pair_dict = {
                        'pair_id': pair_id,
                        'source1_entity_id': source1_id,
                        'candidate_entity_id': cand_id,
                        'candidate_source': derived_source,
                        'source1_row_index': data_row_index,
                        'candidate_rank': cand_index,
                    }
                    if extra_col_names and len(parts) > 2:
                        for col_name, val in zip(extra_col_names, parts[2:]):
                            pair_dict[col_name] = val

                    yield pair_dict

                data_row_index += 1
        finally:
            tracker.close()

        empty_s1_ids = [s1 for s1 in all_s1_ids if candidate_counts.get(s1, 0) == 0]
        self._coverage = Source1Coverage(
            all_source1_ids=tuple(all_s1_ids),
            empty_source1_ids=tuple(empty_s1_ids),
            candidate_counts=dict(candidate_counts),
            total_candidate_pairs=total_pairs,
        )

    def inspect_coverage(self, tsv_source=None, *, duplicate_store=None, db_dir=None):
        """Consume TSV stream to calculate Source 1 coverage without storing candidate pairs in memory."""
        for _ in self.iter_candidate_pairs(tsv_source, duplicate_store=duplicate_store, db_dir=db_dir):
            pass
        return self.coverage

    def iter_candidate_batches(self, tsv_source=None, *, batch_size=1024,
                               duplicate_store=None, db_dir=None):
        """Yield bounded batches of validated candidate pair dicts.

        Each batch contains at most batch_size candidate pairs. All pairs in each
        batch are validated via validate_candidate_batch() before yielding.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')

        batch = []
        for pair in self.iter_candidate_pairs(tsv_source, duplicate_store=duplicate_store, db_dir=db_dir):
            batch.append(pair)
            if len(batch) >= batch_size:
                yield validate_candidate_batch(batch)
                batch = []
        if batch:
            yield validate_candidate_batch(batch)

    def stream_joined_batches(self, member1_adapter, tsv_source=None, *,
                              batch_size=1024, ground_truth=None, require_labels=False,
                              duplicate_store=None, db_dir=None):
        """Stream batches of JoinedCandidate instances joined via Member1Adapter.

        Reuses Member1Adapter.join() on each bounded candidate batch.
        """
        for batch in self.iter_candidate_batches(tsv_source, batch_size=batch_size,
                                                 duplicate_store=duplicate_store, db_dir=db_dir):
            yield member1_adapter.join(batch, ground_truth=ground_truth, require_labels=require_labels)

    def score_candidates(self, inference_pipeline, member1_adapter, tsv_source=None, *,
                         batch_size=1024, collection_aware=False,
                         mask_multiple_postal_codes=None, include_provenance=True,
                         duplicate_store=None, db_dir=None):
        """Stream scored candidate dicts using InferencePipeline and bounded-memory batching.

        Yields dictionaries with:
            {'pair_id': ..., 'source1_id': ..., 'target_source': ..., 'target_id': ..., 'probability': ...}
        plus any preserved candidate provenance ('source1_row_index', 'candidate_rank').

        Reuses Member1Adapter.join() and InferencePipeline.score() without duplicating
        feature engineering or LightGBM model evaluation.
        """
        from .member1 import inference_mapping
        mapping = inference_mapping(collection_aware=collection_aware)
        postal_kwargs = {}
        if collection_aware:
            postal_kwargs['collection_aware'] = True
        else:
            postal_kwargs['mask_multiple_postal_codes'] = (
                True if mask_multiple_postal_codes is None else mask_multiple_postal_codes
            )

        for batch in self.iter_candidate_batches(tsv_source, batch_size=batch_size,
                                                 duplicate_store=duplicate_store, db_dir=db_dir):
            joined = member1_adapter.join(batch)
            rows = [item.inference_row(**postal_kwargs) for item in joined]
            scores = inference_pipeline.score(rows, mapping, batch_size=batch_size)
            for score, cand in zip(scores, batch, strict=True):
                result = dict(score)
                if include_provenance:
                    for prov_key in ('source1_row_index', 'candidate_rank'):
                        if prov_key in cand:
                            result[prov_key] = cand[prov_key]
                yield result

    stream_scores = score_candidates

    def score_file(self, output_path, inference_pipeline, member1_adapter, tsv_source=None, *,
                   batch_size=1024, collection_aware=False,
                   mask_multiple_postal_codes=None, include_provenance=True,
                   duplicate_store=None, db_dir=None):
        """Score candidate TSV and stream JSONL output to file with bounded memory."""
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(f"Output file already exists: {output_path}")
        count = 0
        with output_path.open('x', encoding='utf-8') as stream:
            for score in self.stream_scores(
                inference_pipeline, member1_adapter, tsv_source,
                batch_size=batch_size, collection_aware=collection_aware,
                mask_multiple_postal_codes=mask_multiple_postal_codes,
                include_provenance=include_provenance,
                duplicate_store=duplicate_store, db_dir=db_dir,
            ):
                stream.write(json.dumps(score, allow_nan=False) + '\n')
                count += 1
        return count


def read_member2_tsv(tsv_source, *, duplicate_store='disk', db_dir=None):
    """Convenience generator to stream candidate pair dicts from Member 2 TSV output."""
    adapter = Member2TSVAdapter(tsv_source, duplicate_store=duplicate_store, db_dir=db_dir)
    yield from adapter.iter_candidate_pairs()


def parse_member2_tsv(tsv_source, *, duplicate_store='disk', db_dir=None) -> tuple[list[dict], Source1Coverage]:
    """Convenience function: parse Member 2 TSV and return (candidate_pairs, coverage)."""
    adapter = Member2TSVAdapter(tsv_source, duplicate_store=duplicate_store, db_dir=db_dir)
    candidates = list(adapter.iter_candidate_pairs())
    assert adapter.coverage is not None
    return candidates, adapter.coverage


__all__ = [
    'REQUIRED_CANDIDATE_FIELDS',
    'SOURCE_PREFIXES',
    'VALID_CANDIDATE_SOURCES',
    'VALIDATED_OPTIONAL_FIELDS',
    'Member2TSVAdapter',
    'Source1Coverage',
    'candidate_batch_summary',
    'enrich_candidate_batch',
    'enrich_candidates',
    'generate_pair_id',
    'parse_member2_tsv',
    'read_member2_tsv',
    'validate_candidate_batch',
]
