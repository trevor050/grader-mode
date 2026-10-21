"""Verify completed Phase 6 chunks against the frozen requests before scoring."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_stage(directory, stage, expected):
    directory = Path(directory)
    done = json.loads((directory / f'{stage}.done.json').read_text())
    if done.get('status') != 'completed' or done.get('requests') != len(expected):
        raise ValueError('only a complete admitted stage may be collected')
    lookup = {r['id']: r for r in expected}
    result, seen, tokens = [], set(), 0
    for item in done['chunks']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != stage:
            raise ValueError('unsafe or mixed-stage chunk path')
        path = directory / rel
        if sha(path) != item['sha256']:
            raise ValueError('retained chunk checksum differs from remote completion manifest')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != item['count'] or sum(r['output_tokens'] for r in rows) != item['output_tokens']:
            raise ValueError('chunk counts/tokens disagree with completion manifest')
        for row in rows:
            rid = row['id']
            if rid in seen or rid not in lookup or row.get('excluded_from_main') is not False:
                raise ValueError('duplicate/unexpected/pilot row in study stage')
            if row.get('stage') != stage or row.get('requests_sha256') != done['requests_sha256']:
                raise ValueError('mixed stage/request provenance')
            for key, value in lookup[rid].items():
                if json.dumps(row.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
                    raise ValueError(f'raw row differs from registered request: {key}')
            if len(row['output_token_ids']) != row['output_tokens']:
                raise ValueError('raw token count differs from saved IDs')
            seen.add(rid)
            result.append(row)
            tokens += row['output_tokens']
    if seen != set(lookup) or tokens != done['output_tokens']:
        raise ValueError('stage identities/tokens incomplete')
    return result, {'stage': stage, 'requests': len(result), 'output_tokens': tokens,
                    'done_sha256': sha(directory / f'{stage}.done.json'), 'chunks': done['chunks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--source-dir', action='append', required=True, help='3.8|3.5=/path/to/worker-output')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--drop-stage', action='append', choices=('3.8-prefill', '3.5-main'), default=[])
    args = parser.parse_args()
    spec = json.loads(args.requests.read_text())
    from src.vast_generate_6 import validate_spec
    validate_spec(spec, '3.8')
    sources = dict(item.split('=', 1) for item in args.source_dir)
    expected_models = {'3.8'} if '3.5-main' in args.drop_stage else {'3.8', '3.5'}
    if set(sources) != expected_models:
        raise ValueError('source models disagree with declared whole-stage drops')
    rows, receipts = [], []
    for model in ('3.8', '3.5'):
        if model not in sources:
            continue
        for stage in ('main', 'prefill') if model == '3.8' else ('main',):
            if f'{model}-{stage}' in args.drop_stage:
                continue
            expected = [r for r in spec['models'][model]['requests'] if r['arm'] == stage]
            added, receipt = collect_stage(sources[model], stage, expected)
            rows += added
            receipts.append({'model': model, 'source_dir': sources[model], **receipt})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'generations.jsonl'
    if output.exists():
        raise FileExistsError('collection output already exists; do not overwrite scientific data')
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (args.output_dir / 'manifest.json').write_text(json.dumps(spec['manifest'], indent=2) + '\n')
    receipt = {'requests_sha256': sha(args.requests), 'generations_sha256': sha(output),
               'rows': len(rows), 'stages': receipts, 'dropped_stages': args.drop_stage,
               'pilot_rows_included': False}
    (args.output_dir / 'collection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'{len(rows)} completed study replies collected with verified chunk checksums')


if __name__ == '__main__':
    main()
