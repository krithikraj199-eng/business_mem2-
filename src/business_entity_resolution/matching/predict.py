"""File entry point: configurable JSONL candidates to ID-linked JSONL scores."""

import argparse
import json
from pathlib import Path

from .inference import InferencePipeline, InputMapping


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'Invalid JSON on line {line_number}.') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected a JSON object on line {line_number}.')
            yield row


def predict_file(model_path, input_path, mapping_path, output_path, *, batch_size=1024):
    """Validate and score before creating a new output file; never overwrite."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    mapping = InputMapping(**json.loads(Path(mapping_path).read_text(encoding='utf-8')))
    scores = InferencePipeline(model_path).score(read_jsonl(input_path), mapping, batch_size=batch_size)
    with output_path.open('x', encoding='utf-8') as stream:
        for row in scores:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    return len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    count = predict_file(args.model, args.input, args.mapping, args.output, batch_size=args.batch_size)
    print(json.dumps(dict(scored_candidates=count, output=args.output)))


if __name__ == '__main__':
    main()
