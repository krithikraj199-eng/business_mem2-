"""Configurable candidate mapping and saved-model inference without fitting."""

from collections.abc import Mapping
from dataclasses import dataclass
import json

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..features.feature_pipeline import FEATURE_NAMES
from .pair_data import CandidatePair, validate_pairs
from .training import load_model

RECORD_FIELDS = ('name', 'address', 'postal_code', 'country')


def _read(row, path):
    value = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f'Missing mapped input path: {path!r}')
        value = value[key]
    return value


@dataclass(frozen=True)
class InputMapping:
    """Paths are lists of literal dictionary keys, not dotted expressions.

    Each record field must map to a path or None (explicitly unavailable).
    Configure either a target_source path or a constant target_source_value.
    """

    pair_id: list[str]
    source1_id: list[str]
    target_id: list[str]
    source1_fields: dict
    target_fields: dict
    target_source: list[str] | None = None
    target_source_value: str | None = None

    def __post_init__(self):
        if (self.target_source is None) == (self.target_source_value is None):
            raise ValueError('Configure exactly one target source path or constant.')
        if self.target_source_value is not None and self.target_source_value not in ('S2', 'S3'):
            raise ValueError('target_source_value must be S2 or S3.')
        paths = [self.pair_id, self.source1_id, self.target_id]
        if self.target_source is not None:
            paths.append(self.target_source)
        for fields in (self.source1_fields, self.target_fields):
            if not isinstance(fields, Mapping) or set(fields) not in (set(RECORD_FIELDS), set(RECORD_FIELDS) | {'postal_codes'}):
                raise ValueError(f'Field mappings must explicitly contain {RECORD_FIELDS}.')
            paths.extend(path for path in fields.values() if path is not None)
        if any(not isinstance(path, list) or not path or
               any(not isinstance(key, str) or not key for key in path) for path in paths):
            raise ValueError('Each input path must be a nonempty list of string keys.')

    def map_row(self, row):
        source = _read(row, self.target_source) if self.target_source is not None else self.target_source_value
        if source not in ('S2', 'S3'):
            raise ValueError('Target source must be S2 or S3.')
        target_id = _read(row, self.target_id)
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError('Target IDs must be nonempty strings.')
        def record(fields):
            return {key: None if path is None else _read(row, path) for key, path in fields.items()}
        # Qualify target IDs internally so identical S2/S3 IDs cannot collide.
        pair = CandidatePair(_read(row, self.pair_id), _read(row, self.source1_id),
                             json.dumps([source, target_id]),
                             record(self.source1_fields), record(self.target_fields))
        return pair, source, target_id


class InferencePipeline:
    """Load a trusted model.joblib and score mapped records in bounded batches.

    All candidates are validated before scoring. Return exactly one output per
    valid input row in input order; invalid input raises rather than dropping rows.
    """

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self._check_model()

    def _check_model(self):
        self.model.validate_contract()
        if (tuple(self.model.feature_names) != FEATURE_NAMES or
                tuple(self.model.features.feature_names) != FEATURE_NAMES or
                tuple(self.model.booster.feature_name()) != FEATURE_NAMES):
            raise ValueError('Saved model feature names/order do not match the training schema.')
        for attr in ('name_vectorizer_', 'address_vectorizer_'):
            vectorizer = getattr(self.model.features, attr, None)
            if vectorizer is None:
                raise ValueError(f'Saved model is missing {attr}.')
            check_is_fitted(vectorizer, ['vocabulary_', 'idf_'])

    def score(self, rows, mapping: InputMapping, *, batch_size=1024):
        """Return pair_id, source1_id, target_source, target_id, probability.

        IDs are preserved verbatim. No labels, decisions, filtering or reranking
        are introduced. Empty input returns [] without calling the predictor.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        self._check_model()
        mapped = []
        for index, row in enumerate(rows):
            try:
                mapped.append(mapping.map_row(row))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Candidate row {index}: {exc}') from exc
        validate_pairs(item[0] for item in mapped)
        output = []
        for start in range(0, len(mapped), batch_size):
            batch = mapped[start:start + batch_size]
            scores = self.model.score_pairs(item[0] for item in batch)
            for (pair, source, target_id), score in zip(batch, scores, strict=True):
                if (score['pair_id'], score['source1_id'], score['source2_id']) != (
                        pair.pair_id, pair.source1_id, pair.source2_id):
                    raise ValueError('Model scoring changed candidate alignment.')
                probability = score['probability']
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Model returned an invalid probability.')
                output.append(dict(pair_id=pair.pair_id, source1_id=pair.source1_id,
                                   target_source=source, target_id=target_id,
                                   probability=float(probability)))
        return output
