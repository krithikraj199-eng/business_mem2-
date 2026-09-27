"""Synthetic workflow tests; metric values are not production quality claims."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from business_entity_resolution.features.feature_pipeline import FeaturePipeline
from business_entity_resolution.matching.pair_data import split_pairs, unique_training_records, validate_pairs
from business_entity_resolution.matching.train import read_pairs, synthetic_pairs
from business_entity_resolution.matching.training import (
    TrainingConfig, evaluate_probabilities, load_model, save_training_result, train_classifier,
)


@pytest.fixture
def pairs():
    return synthetic_pairs()


@pytest.fixture
def config():
    return TrainingConfig(num_boost_round=12, min_data_in_leaf=2)


def test_group_split_preserves_ids_labels_and_order(pairs):
    pairs = pairs[::2] + pairs[1::2]  # Labels deliberately clustered in input.
    train, valid = split_pairs(pairs)
    assert not {p.source1_id for p in train} & {p.source1_id for p in valid}
    assert {p.pair_id for p in train + valid} == {p.pair_id for p in pairs}
    for partition in (train, valid):
        ids = {p.pair_id for p in partition}
        assert partition == [p for p in pairs if p.pair_id in ids]
    assert split_pairs(pairs) == (train, valid)


def test_training_only_fit_and_prediction_alignment(pairs, config, monkeypatch):
    train, valid = split_pairs(pairs, config.validation_fraction, config.random_state)
    valid_ids = {p.source1_id for p in valid}
    pairs = [replace(p, source1=dict(p.source1, name='zzzzzz', address='qqqqqq'))
             if p.source1_id in valid_ids else p for p in pairs]
    captured = []
    original_fit = FeaturePipeline.fit
    def spy_fit(self, records):
        records = list(records)
        captured.append(records)
        return original_fit(self, records)
    monkeypatch.setattr(FeaturePipeline, 'fit', spy_fit)
    result = train_classifier(pairs, config, data_kind='synthetic')
    assert captured == [unique_training_records(train)]
    assert len(captured[0]) == 3 * len({p.source1_id for p in train})
    assert 'zzz' not in result.model.features.name_vectorizer_.vocabulary_
    assert 'qqq' not in result.model.features.address_vectorizer_.vocabulary_
    expected = {p.pair_id: p for p in pairs}
    assert [p['pair_id'] for p in result.validation_predictions] == [p.pair_id for p in valid]
    for row in result.validation_predictions:
        pair = expected[row['pair_id']]
        assert (row['source1_id'], row['source2_id'], row['label']) == (pair.source1_id, pair.source2_id, pair.label)
        assert 0 <= row['probability'] <= 1
    assert result.model.booster.num_feature() == 12
    assert result.report['data_kind'] == 'synthetic'


def test_reproducibility_and_artifact_roundtrip(pairs, config, tmp_path):
    result = train_classifier(pairs, config, data_kind='synthetic')
    repeated = train_classifier(pairs, config, data_kind='synthetic')
    assert result.report == repeated.report
    assert result.validation_predictions == repeated.validation_predictions
    directory = save_training_result(result, tmp_path / 'run')
    assert {p.name for p in directory.iterdir()} == {
        'model.joblib', 'lightgbm.txt', 'report.json', 'validation_predictions.json'}
    assert json.loads((directory / 'report.json').read_text()) == result.report
    loaded = load_model(directory / 'model.joblib')
    assert loaded.score_pairs(pairs) == result.model.score_pairs(pairs)
    assert loaded.score_pairs([]) == []
    unlabeled = [replace(p, label=None) for p in pairs[:2]]
    assert all(p['label'] is None for p in loaded.score_pairs(unlabeled))
    with pytest.raises(FileExistsError):
        save_training_result(result, directory)


def test_single_class_validation_metrics():
    report = evaluate_probabilities([0, 0], [0.1, 0.2])
    assert report['roc_auc'] is report['average_precision'] is None
    assert report['brier_score'] == pytest.approx(0.025)
    assert report['log_loss'] is not None and np.isfinite(report['log_loss'])


@pytest.mark.parametrize('labels,scores', [([], []), ([0], [0.2, 0.3]), ([2], [0.5]),
                                          ([1], [float('nan')]), ([0], [1.1])])
def test_invalid_metrics_rejected(labels, scores):
    with pytest.raises(ValueError):
        evaluate_probabilities(labels, scores)


@pytest.mark.parametrize('label', [None, '1', 2, True, float('nan')])
def test_invalid_labels_rejected(pairs, label):
    with pytest.raises(ValueError, match='Labels'):
        train_classifier([replace(pairs[0], label=label)] + pairs[1:])


def test_bad_pairs_and_insufficient_groups(pairs):
    with pytest.raises(ValueError, match='Duplicate'):
        validate_pairs(pairs + [pairs[0]])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_pairs(pairs + [replace(pairs[0], pair_id='another')])
    with pytest.raises(ValueError, match='Inconsistent'):
        validate_pairs([pairs[0], replace(pairs[1], source1={'name': 'Changed'})])
    with pytest.raises(ValueError, match='two Source 1'):
        split_pairs(pairs[:2])
    with pytest.raises(ValueError, match='both classes'):
        train_classifier([replace(p, label=0) for p in pairs])


@pytest.mark.parametrize('kwargs', [dict(num_boost_round=0), dict(num_leaves=1),
                                    dict(num_threads=0), dict(validation_fraction=1),
                                    dict(learning_rate=float('nan')), dict(random_state=-1)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_jsonl_production_entrypoint(pairs, tmp_path):
    from dataclasses import asdict
    source = tmp_path / 'pairs.jsonl'
    source.write_text('\n'.join(json.dumps(asdict(p)) for p in pairs), encoding='utf-8')
    assert read_pairs(source) == pairs
    configuration = tmp_path / 'config.json'
    configuration.write_text(json.dumps(dict(num_boost_round=3, min_data_in_leaf=2)))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    completed = subprocess.run(
        [sys.executable, '-m', 'business_entity_resolution.matching.train',
         '--input', str(source), '--config', str(configuration), '--output', str(tmp_path / 'cli-run')],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['validation']['count'] == 8
    # Input is synthetic even though this test exercises the real-input CLI path.
    assert load_model(tmp_path / 'cli-run' / 'model.joblib').score_pairs(pairs)
