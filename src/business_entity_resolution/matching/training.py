"""LightGBM training and reusable probability scoring; no threshold selection."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import joblib
import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ..features.feature_pipeline import FEATURE_NAMES, FeaturePipeline
from ..features.postal_collections import CURRENT_CONTRACT, LEGACY_CONTRACT
from .pair_data import split_pairs, unique_training_records, validate_pairs


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    validation_fraction: float = 0.2
    num_boost_round: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 20
    num_threads: int = 1

    def __post_init__(self):
        for key in ('num_boost_round', 'num_leaves', 'min_data_in_leaf', 'num_threads'):
            value = getattr(self, key)
            if type(value) is not int or value < (2 if key == 'num_leaves' else 1):
                raise ValueError(f"Invalid {key}.")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**31:
            raise ValueError("random_state must be a nonnegative 32-bit integer.")
        if not 0 < self.validation_fraction < 1 or not 0 < self.learning_rate <= 1:
            raise ValueError("Invalid validation_fraction or learning_rate.")

    def model_params(self):
        return dict(objective='binary', metric='binary_logloss',
                    learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                    min_data_in_leaf=self.min_data_in_leaf, num_threads=self.num_threads,
                    seed=self.random_state, deterministic=True, force_col_wise=True,
                    verbosity=-1)


def evaluate_probabilities(labels, probabilities):
    """Threshold-free metrics; ranking metrics are undefined for one-class data."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Expected aligned, nonempty one-dimensional labels and probabilities.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid labels or probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    both_classes = len(np.unique(labels)) == 2
    return dict(
        count=int(len(labels)), positives=int(labels.sum()),
        log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
        brier_score=float(brier_score_loss(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)) if both_classes else None,
        average_precision=float(average_precision_score(labels, probabilities)) if both_classes else None,
    )


@dataclass
class PairwiseModel:
    features: FeaturePipeline
    booster: lgb.Booster
    feature_names: tuple = FEATURE_NAMES
    feature_contract_version: str | None = None

    def __post_init__(self):
        if self.feature_contract_version is None:
            self.feature_contract_version = self.features.contract_version

    def validate_contract(self):
        version = self.__dict__.get('feature_contract_version', LEGACY_CONTRACT)
        if version not in (LEGACY_CONTRACT, CURRENT_CONTRACT) or version != self.features.contract_version:
            raise ValueError('Unsupported or inconsistent model feature contract.')
        return version

    def score_pairs(self, pairs):
        self.validate_contract()
        pairs = validate_pairs(pairs)
        if not pairs:
            return []
        if tuple(self.features.feature_names) != tuple(self.feature_names):
            raise ValueError("Feature schema differs from the saved model.")
        matrix = self.features.transform((p.source1, p.source2) for p in pairs)
        probabilities = np.asarray(self.booster.predict(matrix, num_threads=1))
        return [dict(pair_id=p.pair_id, source1_id=p.source1_id, source2_id=p.source2_id,
                     label=p.label, probability=float(score))
                for p, score in zip(pairs, probabilities, strict=True)]


@dataclass
class TrainingResult:
    model: PairwiseModel
    report: dict
    validation_predictions: list


def train_classifier(pairs, config=None, *, data_kind='real', saved_split=None):
    """Fit training groups only. Validation does not select rounds or thresholds."""
    if data_kind not in ('real', 'synthetic'):
        raise ValueError("data_kind must be real or synthetic.")
    config = config or TrainingConfig()
    pairs = validate_pairs(pairs, require_labels=True)
    split_metadata = dict(method='group_shuffle')
    if saved_split is None:
        if any(p.split is not None for p in pairs):
            raise ValueError('Pairs with split assignments require saved_split; assignments cannot be ignored.')
        train, valid = split_pairs(pairs, config.validation_fraction, config.random_state)
    else:
        # Local import avoids the adapter/inference/training module import cycle.
        from ..integration.member1 import SavedSplit
        partition = SavedSplit(saved_split)
        train, valid = [], []
        for pair in pairs:
            assignment = partition.assignment(pair.source1_id)
            if pair.split is not None and pair.split != assignment:
                raise ValueError(f'Pair {pair.pair_id} split conflicts with saved split.')
            (train if assignment == 'train' else valid).append(pair)
        if not train or not valid:
            raise ValueError('Saved split must yield nonempty training and validation candidate partitions.')
        if {p.label for p in train} != {0, 1}:
            raise ValueError('Training split must contain both classes; supply verified negative and positive pairs.')
        canonical_split = dict(train_source1_entity_ids=sorted(partition.train_ids),
                               validation_source1_entity_ids=sorted(partition.validation_ids))
        split_metadata = dict(
            method='member1_saved',
            saved_split_sha256=hashlib.sha256(json.dumps(canonical_split, sort_keys=True).encode()).hexdigest(),
            saved_train_entity_count=len(partition.train_ids),
            saved_validation_entity_count=len(partition.validation_ids),
        )
    features = FeaturePipeline().fit(unique_training_records(train))
    matrix = features.transform((p.source1, p.source2) for p in train)
    dataset = lgb.Dataset(matrix, label=[p.label for p in train], feature_name=list(FEATURE_NAMES))
    booster = lgb.train(config.model_params(), dataset, num_boost_round=config.num_boost_round)
    model = PairwiseModel(features, booster)
    predictions = model.score_pairs(valid)
    # Fingerprint normalized input plus pair order for reproducible provenance.
    payload = [dict(pair_id=p.pair_id, source1_id=p.source1_id, source2_id=p.source2_id,
                    label=p.label, source1={k: p.source1.get(k) for k in ('name', 'address', 'postal_code', 'country', 'postal_codes')},
                    source2={k: p.source2.get(k) for k in ('name', 'address', 'postal_code', 'country', 'postal_codes')})
               for p in pairs]
    report = dict(
        schema_version=1, data_kind=data_kind, config=asdict(config),
        feature_contract_version=model.validate_contract(),
        model_params=config.model_params(), feature_names=list(FEATURE_NAMES),
        input_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        versions=dict(python=platform.python_version(), lightgbm=lgb.__version__,
                      sklearn=sklearn.__version__, numpy=np.__version__, joblib=joblib.__version__),
        split=dict(**split_metadata, train_pair_ids=[p.pair_id for p in train],
                   validation_pair_ids=[p.pair_id for p in valid],
                   train_source1_ids=sorted({p.source1_id for p in train}),
                   validation_source1_ids=sorted({p.source1_id for p in valid})),
        validation=evaluate_probabilities([p.label for p in valid], [p['probability'] for p in predictions]),
        boosting_iterations=booster.current_iteration(),
    )
    return TrainingResult(model, report, predictions)


def save_training_result(result, output_directory):
    """Write a new run directory, refusing to overwrite any existing work."""
    output = Path(output_directory)
    version = result.model.validate_contract()
    if result.report.get('feature_contract_version') != version:
        raise ValueError('Report and model feature contracts differ.')
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(result.model, output / 'model.joblib')
    result.model.booster.save_model(str(output / 'lightgbm.txt'))
    (output / 'report.json').write_text(json.dumps(result.report, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'validation_predictions.json').write_text(
        json.dumps(result.validation_predictions, indent=2, allow_nan=False), encoding='utf-8')
    return output


def load_model(path):
    """Load only trusted joblib artifacts, using the training dependency versions."""
    model = joblib.load(path)
    if not isinstance(model, PairwiseModel) or tuple(model.feature_names) != FEATURE_NAMES:
        raise ValueError("Unsupported model artifact or feature schema.")
    model.validate_contract()
    return model
