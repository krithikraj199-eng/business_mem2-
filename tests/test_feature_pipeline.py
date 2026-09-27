"""Synthetic-only regression tests for Member 3's feature pipeline."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from business_entity_resolution.features.feature_pipeline import FEATURE_NAMES, FeaturePipeline
from business_entity_resolution.features import name_features as nf
from business_entity_resolution.features import address_features as af
from business_entity_resolution.features import structured_features as sf


@pytest.fixture
def records():
    return [
        dict(name="Straße Textiles", address="12 Gandhi Road", postal_code="AB-12", country="India"),
        dict(name="Kumar Shop", address="45 Market Street", postal_code="600001", country="France"),
    ]


@pytest.fixture
def pipeline(records):
    return FeaturePipeline().fit(records)


def test_all_columns_delegate_to_existing_features(pipeline, records):
    a = records[0]
    b = dict(name="Strasse Textile", address="12 Gandhi Rd", postal_code="ab 12", country=" INDIA ")
    expected = [
        nf.name_token_jaccard(a['name'], b['name']),
        nf.name_levenshtein_similarity(a['name'], b['name']),
        nf.name_char_tfidf_similarity(a['name'], b['name'], pipeline.name_vectorizer_),
        af.address_token_jaccard(a['address'], b['address']),
        af.address_levenshtein_similarity(a['address'], b['address']),
        af.address_char_tfidf_similarity(a['address'], b['address'], pipeline.address_vectorizer_),
        af.address_numeric_overlap(a['address'], b['address']),
        af.address_numeric_available(a['address'], b['address']),
        sf.postal_code_match(a['postal_code'], b['postal_code']),
        sf.postal_code_available(a['postal_code'], b['postal_code']),
        sf.country_match(a['country'], b['country']),
        sf.country_available(a['country'], b['country']),
    ]
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 12
    assert tuple(pipeline.extract_pair(a, b)) == FEATURE_NAMES
    np.testing.assert_allclose(pipeline.transform([(a, b), (b, a)]), [expected, expected])


def test_identical_and_casefolded_records(pipeline, records):
    a = records[0]
    b = {key: value.casefold() for key, value in a.items()}
    np.testing.assert_allclose(pipeline.transform([(a, b)]), np.ones((1, 12)))


def test_transform_never_fits_or_leaks(pipeline, monkeypatch):
    unseen = dict(name="zzzzzz", address="qqqqqq")
    snapshots = []
    def forbidden(*args, **kwargs):
        pytest.fail("transform attempted to fit")
    for vectorizer in (pipeline.name_vectorizer_, pipeline.address_vectorizer_):
        snapshots.append((dict(vectorizer.vocabulary_), vectorizer.idf_.copy()))
        monkeypatch.setattr(vectorizer, "fit", forbidden)
        monkeypatch.setattr(vectorizer, "fit_transform", forbidden)
    row = pipeline.extract_pair(unseen, unseen)
    assert row['name_char_tfidf_similarity'] == row['address_char_tfidf_similarity'] == 0
    for vectorizer, (vocabulary, idf) in zip(
        (pipeline.name_vectorizer_, pipeline.address_vectorizer_), snapshots
    ):
        assert vectorizer.vocabulary_ == vocabulary
        np.testing.assert_array_equal(vectorizer.idf_, idf)


@pytest.mark.parametrize('missing', [{}, dict(name=None, address=None),
                                      dict(name='  ', address=' ', postal_code=' - ', country=' ')])
def test_missing_values_are_zero(pipeline, records, missing):
    np.testing.assert_array_equal(pipeline.transform([(missing, missing), (records[0], missing)]),
                                  np.zeros((2, 12)))


def test_empty_and_generator_input(pipeline, records):
    assert pipeline.transform(iter([])).shape == (0, 12)
    matrix = pipeline.transform((a, a) for a in records)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, np.ones((2, 12)))


def test_requires_fit():
    with pytest.raises(NotFittedError):
        FeaturePipeline().transform([])
    with pytest.raises(NotFittedError):
        FeaturePipeline().extract_pair({}, {})


@pytest.mark.parametrize('training', [[], [{}], [dict(name='Shop')], [dict(address='Road')]])
def test_unusable_training_rejected(training):
    with pytest.raises(ValueError, match='training record|corpus'):
        FeaturePipeline().fit(training)


@pytest.mark.parametrize('bad', [dict(name=123), dict(postal_code=123), dict(country=np.nan), 'record'])
def test_invalid_records_rejected(pipeline, bad):
    with pytest.raises(TypeError):
        pipeline.extract_pair(bad, {})
    with pytest.raises(TypeError):
        FeaturePipeline().fit([bad])


def test_failed_refit_preserves_state(pipeline, records):
    before = pipeline.transform([(records[0], records[1])])
    with pytest.raises(ValueError):
        pipeline.fit([dict(name='New name')])
    np.testing.assert_array_equal(pipeline.transform([(records[0], records[1])]), before)


def test_structured_feature_regressions():
    assert af.address_numeric_overlap('12A Road 7', '12a Road 8') == pytest.approx(1 / 3)
    assert af.address_numeric_available('12 Road', 'Road') == 0
    assert sf.postal_code_match(None, None) == 0
    assert sf.country_match('  United   States ', 'united states') == 1
    assert sf.country_available('India', 'France') == 1
    assert sf.country_match('India', 'France') == 0
