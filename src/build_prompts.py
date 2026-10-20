"""Build the pre-registered, deterministic GSM8K request grid."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = '740312add88f781978c0658806c59bc2815b9866'


def parse_gold(answer):
    from src.score import normalize_number
    if '####' not in answer:
        raise ValueError('GSM8K answer missing #### delimiter')
    value = normalize_number(answer.rsplit('####', 1)[1].strip())
    if value is None:
        raise ValueError(f'Non-integer gold answer: {answer!r}')
    return value


def sample_problems(rows, count=200, seed=0):
    from src.score import number_values
    eligible = []
    for idx, row in enumerate(rows):
        gold = parse_gold(row['answer'])
        if gold not in number_values(row['question']):
            eligible.append({'problem_idx': idx, 'question': row['question'], 'gold': gold})
    return random.Random(seed).sample(eligible, count)


def request_seed(request_id):
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:4], 'big')


def expand_requests(problems, templates, system):
    if set(templates['cues']) != {'none', 'human', 'auto'} or set(templates['formats']) != {'none', 'boxed', 'final'}:
        raise ValueError('Unexpected registered cue/format set')
    if any(len(v) != 4 for v in templates['cues'].values()):
        raise ValueError('Each cue must have four matched templates')
    result = []
    for p in problems:
        cells = [('tutor', c, f) for c in ('none', 'human', 'auto') for f in ('none', 'boxed', 'final')]
        cells.append(('ceiling', 'none', 'none'))
        for arm, cue, fmt in cells:
            for i, template in enumerate(templates['cues'][cue]):
                rid = f"p{p['problem_idx']:04d}:{cue}:{fmt}:t{i}:{arm}"
                user = template.replace('{problem}', p['question']) + templates['formats'][fmt]
                messages = [] if arm == 'ceiling' else [{'role': 'system', 'content': system}]
                messages.append({'role': 'user', 'content': user})
                result.append({**p, 'id': rid, 'cue': cue, 'format': fmt, 'template_idx': i,
                               'arm': arm, 'seed': request_seed(rid), 'messages': messages})
    return result


def expand_requests_1b(problems, templates, system):
    cues = ('none', 'human_strict', 'env')
    if set(templates['cues']) != set(cues) or any(len(v) != 4 for v in templates['cues'].values()):
        raise ValueError('Phase 1b requires three cues with four matched templates each')
    result = []
    for p in problems:
        for thinking in (False, True):
            for cue in cues:
                for i, template in enumerate(templates['cues'][cue]):
                    rid = f"phase1b:p{p['problem_idx']:04d}:{cue}:boxed:t{i}:tutor:thinking-{int(thinking)}"
                    result.append({**p, 'id': rid, 'phase': 'phase1b', 'thinking': thinking,
                                   'cue': cue, 'format': 'boxed', 'template_idx': i, 'arm': 'tutor',
                                   'seed': request_seed(rid), 'messages': [
                                       {'role': 'system', 'content': system},
                                       {'role': 'user', 'content': template.replace('{problem}', p['question'])}]})
    return result


def build(output_dir, limit_problems=200, phase='phase1'):
    from datasets import load_dataset
    if not 1 <= limit_problems <= 200:
        raise ValueError('limit-problems must be between 1 and 200')
    if phase not in ('phase1', 'phase1b'):
        raise ValueError('Unknown phase')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_dataset('openai/gsm8k', 'main', split='test', revision=DATASET_REVISION)
    problems = sample_problems(rows)
    system = (ROOT / 'prompts/system_tutor.txt').read_text()
    template_path = 'prompts/templates_1b.json' if phase == 'phase1b' else 'prompts/templates.json'
    templates = json.loads((ROOT / template_path).read_text())
    expand = expand_requests_1b if phase == 'phase1b' else expand_requests
    requests = expand(problems[:limit_problems], templates, system)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ('DESIGN.md', 'prompts/system_tutor.txt', template_path)}
    manifest = {'phase': phase, 'dataset': 'openai/gsm8k', 'config': 'main', 'split': 'test',
                'revision': DATASET_REVISION, 'sample_seed': 0, 'sample_size': 200,
                'limit_problems': limit_problems, 'requests': len(requests),
                'source_sha256': hashes, 'sample': problems}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'requests.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests))
    return requests, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results/raw/prompts')
    parser.add_argument('--limit-problems', type=int, default=200)
    parser.add_argument('--phase', choices=('phase1', 'phase1b'), default='phase1')
    args = parser.parse_args()
    requests, _ = build(args.output_dir, args.limit_problems, args.phase)
    print(f'Wrote {len(requests)} requests to {args.output_dir}')


if __name__ == '__main__':
    main()
