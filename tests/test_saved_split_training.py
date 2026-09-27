"""Small synthetic tests only; never produce production model artifacts."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from business_entity_resolution.features.feature_pipeline import FeaturePipeline
from business_entity_resolution.integration.member1 import Member1Adapter
from business_entity_resolution.matching import training
from business_entity_resolution.matching.pair_data import unique_training_records


@pytest.fixture
def inputs():
    data = json.loads((Path(__file__).parent / 'fixtures/member1_small.json').read_text())
    data['source1'][1]['business_name_normalized'] = 'zzzzzz'
    data['source1'][1]['business_address_normalized'] = 'qqqqqq'
    adapter = Member1Adapter(data['source1'], data['source2'], data['source3'], data['split'])
    joined = adapter.join(data['candidates'], ground_truth=data['ground_truth'], require_labels=True)
    return [p.ml_pair(collection_aware=True) for p in joined], data['split']


def test_exact_split_fit_only_training_and_persistence(inputs, tmp_path, monkeypatch):
    pairs, saved = inputs
    captured = []
    fit = FeaturePipeline.fit
    def spy(self, records):
        records = list(records)
        captured.append(records)
        return fit(self, records)
    def forbidden(*args, **kwargs):
        pytest.fail('Saved split invoked random splitting')
    monkeypatch.setattr(FeaturePipeline, 'fit', spy)
    monkeypatch.setattr(training, 'split_pairs', forbidden)
    result = training.train_classifier(iter(pairs), training.TrainingConfig(num_boost_round=2),
                                       saved_split=saved, data_kind='synthetic')
    assert captured == [unique_training_records(pairs[1:])]
    report = result.report
    assert report['split']['train_pair_ids'] == ['p-positive', 'p-negative']
    assert report['split']['validation_pair_ids'] == ['p-validation']
    assert report['split']['method'] == 'member1_saved'
    assert report['feature_contract_version'] == 'postal-collection-v2'
    assert 'zzz' not in result.model.features.name_vectorizer_.vocabulary_
    assert 'qqq' not in result.model.features.address_vectorizer_.vocabulary_
    row = result.validation_predictions[0]
    assert (row['pair_id'], row['source1_id'], row['source2_id'], row['label']) == (
        'p-validation', 'S1-002', 'S3-001', 0)
    directory = training.save_training_result(result, tmp_path / 'run')
    loaded = training.load_model(directory / 'model.joblib')
    assert loaded.validate_contract() == 'postal-collection-v2'
    assert loaded.score_pairs(pairs) == result.model.score_pairs(pairs)
    assert json.loads((directory / 'report.json').read_text()) == report


@pytest.mark.parametrize('case', ['overlap', 'unknown', 'conflict', 'duplicate_id',
                                 'empty_train', 'empty_validation', 'one_class', 'invalid_split'])
def test_bad_partitions_fail_before_fit(inputs, monkeypatch, case):
    pairs, saved = inputs
    if case == 'overlap':
        saved['train_source1_entity_ids'].append('S1-002')
    elif case == 'unknown':
        saved['validation_source1_entity_ids'] = ['S1-999']
    elif case == 'conflict':
        pairs[0] = replace(pairs[0], split='train')
    elif case == 'duplicate_id':
        saved['train_source1_entity_ids'].append('S1-001')
    elif case == 'empty_train':
        pairs = pairs[:1]
    elif case == 'empty_validation':
        pairs = pairs[1:]
    elif case == 'one_class':
        pairs = [replace(p, label=1) for p in pairs]
    else:
        pairs[0] = replace(pairs[0], split='test')
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid inputs reached feature fitting')
    monkeypatch.setattr(FeaturePipeline, 'fit', forbidden)
    with pytest.raises(ValueError):
        training.train_classifier(pairs, saved_split=saved, data_kind='synthetic')


@pytest.mark.parametrize('label', [None, '0', True, 2, 0.0])
def test_bad_labels(inputs, label):
    pairs, saved = inputs
    pairs[0] = replace(pairs[0], label=label)
    with pytest.raises(ValueError, match='Labels'):
        training.train_classifier(pairs, saved_split=saved, data_kind='synthetic')


def test_split_labels_not_silently_ignored(inputs):
    pairs, _ = inputs
    with pytest.raises(ValueError, match='require saved_split'):
        training.train_classifier(pairs, data_kind='synthetic')


def test_pairs_without_split_annotations_and_unused_saved_ids(inputs):
    pairs, saved = inputs
    pairs = [replace(p, split=None) for p in pairs]
    saved['train_source1_entity_ids'].append('S1-unused')
    result = training.train_classifier(pairs, training.TrainingConfig(num_boost_round=1),
                                       saved_split=saved, data_kind='synthetic')
    assert result.report['split']['train_pair_ids'] == ['p-positive', 'p-negative']


def test_saved_split_cli(inputs, tmp_path):
    pairs, saved = inputs
    source = tmp_path / 'pairs.jsonl'
    source.write_text('\n'.join(json.dumps(asdict(p)) for p in pairs), encoding='utf-8')
    split = tmp_path / 'validation_split.json'
    split.write_text(json.dumps(saved), encoding='utf-8')
    config = tmp_path / 'config.json'
    config.write_text('{"num_boost_round": 1}', encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    output = tmp_path / 'run'
    run = subprocess.run([sys.executable, '-m', 'business_entity_resolution.matching.train',
                          '--input', str(source), '--saved-split', str(split),
                          '--config', str(config), '--output', str(output)],
                         env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    report = json.loads((output / 'report.json').read_text())
    assert report['split']['validation_pair_ids'] == ['p-validation']
    assert report['feature_contract_version'] == 'postal-collection-v2'
