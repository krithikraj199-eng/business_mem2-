from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from business_entity_resolution.features.feature_pipeline import FeaturePipeline, FEATURE_NAMES
from business_entity_resolution.features.postal_collections import (
    CURRENT_CONTRACT, LEGACY_CONTRACT, postal_collection_match, postal_collection_available,
)
from business_entity_resolution.features.structured_features import postal_code_match
from business_entity_resolution.integration.member1 import Member1Adapter, inference_mapping
from business_entity_resolution.matching.inference import InferencePipeline
from business_entity_resolution.matching.train import synthetic_pairs
from business_entity_resolution.matching.training import train_classifier, TrainingConfig, save_training_result, load_model


@pytest.mark.parametrize('left,right,match,available', [
    ([], [], 0, 0), ([''], ['00123'], 0, 0), ([None, ' - '], ['00123'], 0, 0),
    (['00123'], ['00123'], 1, 1), (['00123', '600001'], ['600001', '600002'], 1, 1),
    (['00123'], ['600001', '600002'], 0, 1), (['12345-6789'], ['123456789'], 1, 1),
    (['641 001'], ['641001'], 1, 1),
])
def test_collection_features(left, right, match, available):
    assert postal_collection_match(left, right) == match
    assert postal_collection_available(left, right) == available


def test_scalar_and_explicit_collection_contracts():
    assert postal_code_match('641 001', '641001') == 1
    record = dict(name='Shop', address='12 Road', postal_code='641 001')
    for version in (LEGACY_CONTRACT, CURRENT_CONTRACT):
        pipeline = FeaturePipeline(version).fit([record])
        assert pipeline.extract_pair(record, dict(record, postal_code='641001'))['postal_code_match'] == 1
    with pytest.raises(TypeError):
        postal_collection_match('641001 641002', [])
    with pytest.raises(ValueError, match='Conflicting'):
        FeaturePipeline().fit([dict(record, postal_codes=['99999'])])
    with pytest.raises(ValueError, match='Legacy'):
        FeaturePipeline(LEGACY_CONTRACT).fit([dict(record, postal_codes=['641001'])])


def test_adapter_training_inference_consistency_and_versions(tmp_path, monkeypatch):
    data = json.loads((Path(__file__).parent / 'fixtures/member1_small.json').read_text())
    adapter = Member1Adapter(data['source1'], data['source2'], data['source3'], data['split'])
    joined = adapter.join(data['candidates'], ground_truth=data['ground_truth'])
    pairs = [p.ml_pair(collection_aware=True) for p in joined]
    rows = [p.inference_row(collection_aware=True) for p in joined]
    mapping = inference_mapping(collection_aware=True)
    # Fit only a temporary synthetic workflow model, never a production artifact.
    training = [replace(p, source1=dict(p.source1, postal_codes=[p.source1['postal_code']]),
                        source2=dict(p.source2, postal_codes=[p.source2['postal_code']]))
                for p in synthetic_pairs()]
    result = train_classifier(training, TrainingConfig(num_boost_round=2, min_data_in_leaf=2), data_kind='synthetic')
    output = save_training_result(result, tmp_path / 'run')
    assert json.loads((output / 'report.json').read_text())['feature_contract_version'] == CURRENT_CONTRACT
    inference = InferencePipeline(output / 'model.joblib')
    def forbidden(*args, **kwargs):
        pytest.fail('Inference refitted features')
    monkeypatch.setattr(FeaturePipeline, 'fit', forbidden)
    direct = inference.model.features.transform((p.source1, p.source2) for p in pairs)
    mapped_pairs = [mapping.map_row(row)[0] for row in rows]
    mapped = inference.model.features.transform((p.source1, p.source2) for p in mapped_pairs)
    np.testing.assert_array_equal(direct, mapped)
    assert direct.shape == (3, 12)
    assert direct[1, FEATURE_NAMES.index('postal_code_match')] == 1
    expected = inference.model.score_pairs(pairs)
    scored = inference.score(rows, mapping)
    assert [p['probability'] for p in scored] == [p['probability'] for p in expected]
    for version in ('unknown-v99', LEGACY_CONTRACT):
        bad = load_model(output / 'model.joblib')
        bad.feature_contract_version = version
        path = tmp_path / (version + '.joblib')
        joblib.dump(bad, path)
        with pytest.raises(ValueError, match='contract'):
            InferencePipeline(path)
    legacy = load_model(output / 'model.joblib')
    del legacy.feature_contract_version
    del legacy.features.feature_contract_version
    path = tmp_path / 'legacy.joblib'
    joblib.dump(legacy, path)
    old = InferencePipeline(path)
    assert old.model.validate_contract() == LEGACY_CONTRACT
    assert old.model.score_pairs(synthetic_pairs())
    with pytest.raises(ValueError, match='Legacy'):
        old.score(rows, mapping)


def test_existing_artifact_is_legacy_and_unchanged():
    path = Path(__file__).parents[1] / 'models/member3_synthetic_smoke/model.joblib'
    before = path.read_bytes()
    model = load_model(path)
    assert model.validate_contract() == LEGACY_CONTRACT
    assert model.score_pairs(synthetic_pairs())
    assert path.read_bytes() == before
