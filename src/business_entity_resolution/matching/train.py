"""CLI: explicit labeled JSONL input or a synthetic workflow smoke test."""

import argparse
import json
from pathlib import Path

from ..integration.member1 import load_saved_split
from .pair_data import CandidatePair
from .training import TrainingConfig, save_training_result, train_classifier


def synthetic_pairs():
    pairs = []
    for index in range(20):
        left = dict(name=f'Example Textiles {index}', address=f'{index + 1} Market Road',
                    postal_code=f'{index:05d}', country='India')
        different = dict(name=f'Other Pharmacy {index}', address=f'{index + 101} Lake Avenue',
                         postal_code=f'{index + 100:05d}', country='France')
        for label, right in ((1, dict(left)), (0, different)):
            pairs.append(CandidatePair(f'pair-{index}-{label}', f's1-{index}',
                                       f's2-{index}-{label}', left, right, label))
    return pairs


def read_pairs(path):
    pairs = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                pairs.append(CandidatePair(**json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid candidate pair on line {line_number}: {exc}') from exc
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='Labeled candidate pairs in JSONL format')
    source.add_argument('--synthetic', action='store_true', help='Workflow smoke test only')
    parser.add_argument('--output', required=True, help='New run directory')
    parser.add_argument('--config', help='JSON object of TrainingConfig overrides')
    parser.add_argument('--saved-split', help='Member 1 validation_split.json; authoritative S1 partitions')
    args = parser.parse_args()
    config = TrainingConfig(**json.loads(Path(args.config).read_text(encoding='utf-8'))) if args.config else TrainingConfig()
    pairs = synthetic_pairs() if args.synthetic else read_pairs(args.input)
    saved_split = load_saved_split(args.saved_split) if args.saved_split else None
    result = train_classifier(pairs, config, data_kind='synthetic' if args.synthetic else 'real',
                              saved_split=saved_split)
    output = save_training_result(result, args.output)
    print(json.dumps(dict(artifacts=str(output), data_kind=result.report['data_kind'],
                          validation=result.report['validation']), indent=2))


if __name__ == '__main__':
    main()
