"""Inference tests use an isolated, temporary synthetic model only."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest

from business_entity_resolution.features.feature_pipeline import FeaturePipeline
from business_entity_resolution.matching.inference import InferencePipeline, InputMapping
from business_entity_resolution.matching.predict import predict_file
from business_entity_resolution.matching.train import synthetic_pairs
from business_entity_resolution.matching.training import TrainingConfig, save_training_result, train_classifier


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    result = train_classifier(synthetic_pairs(), TrainingConfig(num_boost_round=8, min_data_in_leaf=2),
                              data_kind='synthetic')
    return save_training_result(result, tmp_path_factory.mktemp('inference') / 'run') / 'model.joblib'


@pytest.fixture
def mapping():
    return InputMapping(pair_id=['candidate'], source1_id=['left_id'], target_id=['right_id'],
                        target_source=['source'],
                        source1_fields={k: ['left', k] for k in ('name', 'address', 'postal_code', 'country')},
                        target_fields={k: ['right', k] for k in ('name', 'address', 'postal_code', 'country')})


@pytest.fixture
def rows():
    pairs = synthetic_pairs()
    # Same raw right ID in S2/S3, with distinct records; mixed order is intentional.
    return [dict(candidate='p3', left_id='a', right_id='001', source='S3',
                 left=pairs[0].source1, right=pairs[1].source2),
            dict(candidate='p2', left_id='a', right_id='001', source='S2',
                 left=pairs[0].source1, right=pairs[0].source2)]


def test_reload_alignment_and_exactly_once_without_fit(artifact, mapping, rows, monkeypatch):
    inference = InferencePipeline(artifact)
    pairs = [mapping.map_row(row)[0] for row in rows]
    expected = inference.model.score_pairs(pairs)
    snapshots = [(dict(v.vocabulary_), v.idf_.copy()) for v in
                 (inference.model.features.name_vectorizer_, inference.model.features.address_vectorizer_)]
    def forbidden(*args, **kwargs):
        pytest.fail('Inference attempted to fit')
    monkeypatch.setattr(FeaturePipeline, 'fit', forbidden)
    for vectorizer in (inference.model.features.name_vectorizer_, inference.model.features.address_vectorizer_):
        monkeypatch.setattr(vectorizer, 'fit', forbidden)
        monkeypatch.setattr(vectorizer, 'fit_transform', forbidden)
    scored_ids = []
    original = inference.model.score_pairs
    def spy(pairs):
        pairs = list(pairs)
        scored_ids.extend(p.pair_id for p in pairs)
        return original(pairs)
    monkeypatch.setattr(inference.model, 'score_pairs', spy)
    scores = inference.score(iter(rows), mapping, batch_size=1)
    assert scored_ids == ['p3', 'p2']
    assert [(s['pair_id'], s['source1_id'], s['target_source'], s['target_id']) for s in scores] == [
        ('p3', 'a', 'S3', '001'), ('p2', 'a', 'S2', '001')]
    np.testing.assert_allclose([s['probability'] for s in scores], [s['probability'] for s in expected])
    assert scores[0]['probability'] < scores[1]['probability']
    for vectorizer, (vocabulary, idf) in zip(
        (inference.model.features.name_vectorizer_, inference.model.features.address_vectorizer_), snapshots
    ):
        assert vectorizer.vocabulary_ == vocabulary
        np.testing.assert_array_equal(vectorizer.idf_, idf)


def test_empty_does_not_predict(artifact, mapping, monkeypatch):
    inference = InferencePipeline(artifact)
    def forbidden(*args, **kwargs):
        pytest.fail('Empty input called predictor')
    monkeypatch.setattr(inference.model, 'score_pairs', forbidden)
    assert inference.score([], mapping) == []


def test_flat_mapping_and_constant_source(artifact):
    fields = dict(name=['company'], address=None, postal_code=None, country=None)
    mapping = InputMapping(['pid'], ['lid'], ['rid'], fields, fields, target_source_value='S3')
    scores = InferencePipeline(artifact).score([dict(pid='x', lid='01', rid='02', company=None)], mapping)
    assert scores[0]['target_source'] == 'S3'
    assert np.isfinite(scores[0]['probability'])


@pytest.mark.parametrize('change', ['duplicate_id', 'duplicate_edge', 'missing', 'bad_source', 'bad_id', 'inconsistent'])
def test_invalid_candidates_fail_before_predict(artifact, mapping, rows, change, monkeypatch):
    inference = InferencePipeline(artifact)
    if change == 'duplicate_id':
        rows[1]['candidate'] = rows[0]['candidate']
    elif change == 'duplicate_edge':
        rows[1]['source'] = 'S3'
    elif change == 'missing':
        del rows[1]['right']
    elif change == 'bad_source':
        rows[1]['source'] = 'S4'
    elif change == 'bad_id':
        rows[1]['right_id'] = 123
    else:
        rows[1]['left'] = dict(rows[1]['left'], name='Different')
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid input reached predictor')
    monkeypatch.setattr(inference.model, 'score_pairs', forbidden)
    with pytest.raises(ValueError):
        inference.score(rows, mapping, batch_size=1)


def test_incompatible_booster_schema_rejected(artifact, tmp_path):
    model = joblib.load(artifact)
    text = model.booster.model_to_string().replace('name_token_jaccard', 'wrong_feature_name')
    import lightgbm as lgb
    model.booster = lgb.Booster(model_str=text)
    path = tmp_path / 'bad.joblib'
    joblib.dump(model, path)
    with pytest.raises(ValueError, match='feature'):
        InferencePipeline(path)


def test_file_entrypoint_and_empty_output(artifact, mapping, rows, tmp_path):
    source, config, output = (tmp_path / name for name in ('input.jsonl', 'mapping.json', 'scores.jsonl'))
    source.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    config.write_text(json.dumps(asdict(mapping)), encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    run = subprocess.run([sys.executable, '-m', 'business_entity_resolution.matching.predict',
                          '--model', str(artifact), '--input', str(source), '--mapping', str(config),
                          '--output', str(output), '--batch-size', '1'],
                         env=env, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['scored_candidates'] == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == InferencePipeline(artifact).score(rows, mapping)
    with pytest.raises(FileExistsError):
        predict_file(artifact, source, config, output)
    source.write_text('', encoding='utf-8')
    empty = tmp_path / 'empty.jsonl'
    assert predict_file(artifact, source, config, empty) == 0
    assert empty.read_text() == ''
    source.write_text('{invalid', encoding='utf-8')
    invalid_output = tmp_path / 'invalid.jsonl'
    with pytest.raises(ValueError, match='line 1'):
        predict_file(artifact, source, config, invalid_output)
    assert not invalid_output.exists()
