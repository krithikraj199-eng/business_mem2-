"""Pair features with TF-IDF vocabulary learned only from training records.

Records are mappings with name, address, postal_code and country fields containing
strings or None; omitted fields are missing. Version 2 also accepts explicit
postal_codes lists/tuples. Extra fields (IDs, labels) are ignored.
Call fit(training_records) after splitting the data, then transform(record_pairs)
for every split. Output is a float64 array with columns in FEATURE_NAMES order.
"""

from collections.abc import Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from .postal_collections import CURRENT_CONTRACT, LEGACY_CONTRACT, record_postal_codes

from .address_features import (
    address_token_jaccard, address_levenshtein_similarity,
    address_char_tfidf_similarity, address_numeric_overlap,
    address_numeric_available,
)
from .name_features import (
    name_token_jaccard, name_levenshtein_similarity, name_char_tfidf_similarity,
)
from .structured_features import (
    country_match, country_available,
)

FEATURE_NAMES = (
    "name_token_jaccard", "name_levenshtein_similarity",
    "name_char_tfidf_similarity", "address_token_jaccard",
    "address_levenshtein_similarity", "address_char_tfidf_similarity",
    "address_numeric_overlap", "address_numeric_available",
    "postal_code_match", "postal_code_available", "country_match",
    "country_available",
)
Record = Mapping[str, object]


def _fields(record: Record) -> tuple[str, ...]:
    if not isinstance(record, Mapping):
        raise TypeError("Each record must be a mapping.")
    values = []
    for key in ("name", "address", "postal_code", "country"):
        value = record.get(key)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{key} must be a string or None; got {type(value).__name__}.")
        values.append(value or "")
    return tuple(values)


def _batch_char_tfidf_similarity(
    vectorizer: TfidfVectorizer,
    list_a: list[str],
    list_b: list[str],
) -> np.ndarray:
    n = len(list_a)
    sims = np.zeros(n, dtype=np.float64)
    valid_indices = [i for i in range(n) if list_a[i] and list_b[i]]
    if not valid_indices:
        return sims

    sub_a = [list_a[i] for i in valid_indices]
    sub_b = [list_b[i] for i in valid_indices]

    unique_a, inv_a = np.unique(sub_a, return_inverse=True)
    va = vectorizer.transform(unique_a)[inv_a] if len(unique_a) < len(sub_a) else vectorizer.transform(sub_a)

    unique_b, inv_b = np.unique(sub_b, return_inverse=True)
    vb = vectorizer.transform(unique_b)[inv_b] if len(unique_b) < len(sub_b) else vectorizer.transform(sub_b)

    dot_products = np.asarray(va.multiply(vb).sum(axis=1)).ravel()
    sims[valid_indices] = dot_products
    return sims


class FeaturePipeline:
    """Fit on training records and extract the existing 12 pair features.

    Empty training data, or a training corpus with no usable names or addresses,
    raises ValueError. No synthetic vocabulary or evaluation data is substituted.
    A failed refit leaves the previously fitted vectorizers intact.
    """

    feature_names = FEATURE_NAMES

    def __init__(self, feature_contract_version=CURRENT_CONTRACT):
        if feature_contract_version not in (LEGACY_CONTRACT, CURRENT_CONTRACT):
            raise ValueError('Unsupported feature contract.')
        self.feature_contract_version = feature_contract_version

    @property
    def contract_version(self):
        # Historical pickles have no version attribute: keep scalar semantics.
        return self.__dict__.get('feature_contract_version', LEGACY_CONTRACT)

    def _check_fitted(self) -> None:
        if not all(hasattr(self, attr) for attr in ("name_vectorizer_", "address_vectorizer_")):
            raise NotFittedError("Call fit(training_records) before extracting features.")

    def fit(self, training_records: Iterable[Record]) -> "FeaturePipeline":
        records = []
        for record in training_records:
            fields = _fields(record)
            record_postal_codes(record, self.contract_version)
            records.append(fields)
        if not records:
            raise ValueError("At least one training record is required.")
        vectorizers = []
        for index, field in enumerate(("name", "address")):
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            # Match the normalization used by the existing similarity functions.
            corpus = [record[index].casefold().strip() for record in records]
            try:
                vectorizer.fit(corpus)
            except ValueError as exc:
                raise ValueError(f"Training {field} corpus has no usable TF-IDF vocabulary.") from exc
            vectorizers.append(vectorizer)
        self.name_vectorizer_, self.address_vectorizer_ = vectorizers
        return self

    def extract_pair(self, record_a: Record, record_b: Record) -> dict[str, float]:
        """Return a named feature row, preserving FEATURE_NAMES order."""
        self._check_fitted()
        na, aa, pa, ca = _fields(record_a)
        nb, ab, pb, cb = _fields(record_b)
        postal_a = record_postal_codes(record_a, self.contract_version)
        postal_b = record_postal_codes(record_b, self.contract_version)
        values = (
            name_token_jaccard(na, nb),
            name_levenshtein_similarity(na, nb),
            name_char_tfidf_similarity(na, nb, self.name_vectorizer_),
            address_token_jaccard(aa, ab),
            address_levenshtein_similarity(aa, ab),
            address_char_tfidf_similarity(aa, ab, self.address_vectorizer_),
            address_numeric_overlap(aa, ab),
            address_numeric_available(aa, ab),
            int(bool(postal_a & postal_b)), int(bool(postal_a) and bool(postal_b)),
            country_match(ca, cb), country_available(ca, cb),
        )
        return dict(zip(FEATURE_NAMES, map(float, values), strict=True))

    def transform(self, pairs: Iterable[tuple[Record, Record]]) -> np.ndarray:
        """Transform ordered record pairs without fitting; empty input is (0, 12)."""
        self._check_fitted()
        pair_list = list(pairs)
        n = len(pair_list)
        if n == 0:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        if n == 1:
            row = list(self.extract_pair(pair_list[0][0], pair_list[0][1]).values())
            return np.asarray([row], dtype=np.float64).reshape(-1, len(FEATURE_NAMES))

        matrix = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
        clean_names_a: list[str] = []
        clean_names_b: list[str] = []
        clean_addrs_a: list[str] = []
        clean_addrs_b: list[str] = []

        for i, (ra, rb) in enumerate(pair_list):
            na, aa, pa, ca = _fields(ra)
            nb, ab, pb, cb = _fields(rb)
            postal_a = record_postal_codes(ra, self.contract_version)
            postal_b = record_postal_codes(rb, self.contract_version)

            matrix[i, 0] = name_token_jaccard(na, nb)
            matrix[i, 1] = name_levenshtein_similarity(na, nb)
            # index 2: name_char_tfidf_similarity (vectorized below)
            matrix[i, 3] = address_token_jaccard(aa, ab)
            matrix[i, 4] = address_levenshtein_similarity(aa, ab)
            # index 5: address_char_tfidf_similarity (vectorized below)
            matrix[i, 6] = address_numeric_overlap(aa, ab)
            matrix[i, 7] = address_numeric_available(aa, ab)
            matrix[i, 8] = float(bool(postal_a & postal_b))
            matrix[i, 9] = float(bool(postal_a) and bool(postal_b))
            matrix[i, 10] = float(country_match(ca, cb))
            matrix[i, 11] = float(country_available(ca, cb))

            clean_names_a.append((na or "").casefold().strip())
            clean_names_b.append((nb or "").casefold().strip())
            clean_addrs_a.append((aa or "").casefold().strip())
            clean_addrs_b.append((ab or "").casefold().strip())

        matrix[:, 2] = _batch_char_tfidf_similarity(self.name_vectorizer_, clean_names_a, clean_names_b)
        matrix[:, 5] = _batch_char_tfidf_similarity(self.address_vectorizer_, clean_addrs_a, clean_addrs_b)

        return matrix


if __name__ == "__main__":
    training = [
        {"name": "Sri Krishna Textiles", "address": "12 Gandhi Road, Coimbatore",
         "postal_code": "641001", "country": "India"},
        {"name": "Kumar Medical Store", "address": "45 Nehru Street, Chennai",
         "postal_code": "600001", "country": "India"},
    ]
    query = {"name": "Sri Krishna Textile", "address": "12 Gandhi Rd, Coimbatore",
             "postal_code": "641 001", "country": "INDIA"}
    pipeline = FeaturePipeline().fit(training)
    print(pipeline.extract_pair(training[0], query))
    print("Feature matrix shape:", pipeline.transform([(training[0], query)]).shape)
