"""Explicit postal collections; scalar strings are never split here."""

from .structured_features import normalize_postal_code

LEGACY_CONTRACT = 'postal-scalar-v1'
CURRENT_CONTRACT = 'postal-collection-v2'


def normalize_postal_collection(values):
    if not isinstance(values, (list, tuple)):
        raise TypeError('postal_codes must be a list or tuple of strings/None, not a serialized string.')
    if any(value is not None and not isinstance(value, str) for value in values):
        raise TypeError('Postal collection items must be strings or None.')
    return frozenset(code for value in values if (code := normalize_postal_code(value)))


def record_postal_codes(record, contract=CURRENT_CONTRACT):
    if contract not in (LEGACY_CONTRACT, CURRENT_CONTRACT):
        raise ValueError('Unsupported feature contract.')
    if 'postal_codes' in record:
        if contract == LEGACY_CONTRACT:
            raise ValueError('Legacy feature contract cannot consume postal collections.')
        codes = normalize_postal_collection(record['postal_codes'])
        scalar = normalize_postal_code(record.get('postal_code'))
        if scalar and codes != frozenset([scalar]):
            raise ValueError('Conflicting scalar and collection postal fields.')
        return codes
    return normalize_postal_collection([record.get('postal_code')])


def postal_collection_match(left, right):
    return int(bool(normalize_postal_collection(left) & normalize_postal_collection(right)))


def postal_collection_available(left, right):
    return int(bool(normalize_postal_collection(left)) and bool(normalize_postal_collection(right)))
