"""Build the pre-registered, deterministic GSM8K request grid."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = '740312add88f781978c0658806c59bc2815b9866'
PHASE1C_PROSE = ('plain', 'env')
PHASE1C_FORMATS = ('none', 'boxed', 'dapo', 'answer_tag', 'result_tag', 'boxed_step')


def parse_gold(answer):
    from src.score import normalize_number
    if '####' not in answer:
        raise ValueError('GSM8K answer missing #### delimiter')
    value = normalize_number(answer.rsplit('####', 1)[1].strip())
    if value is None:
        raise ValueError(f'Non-integer gold answer: {answer!r}')
    return value


def _eligible_problems(rows):
    from src.score import number_values
    eligible = []
    for idx, row in enumerate(rows):
        gold = parse_gold(row['answer'])
        if gold not in number_values(row['question']):
            eligible.append({'problem_idx': idx, 'question': row['question'], 'gold': gold})
    return eligible


def sample_problems(rows, count=200, seed=0):
    return random.Random(seed).sample(_eligible_problems(rows), count)


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


def _validate_1c_variant(ladder, checkpoint):
    if ladder not in ('instruct', 'think') or checkpoint not in ('SFT', 'DPO', 'RLVR'):
        raise ValueError('Phase 1c requires ladder instruct|think and checkpoint SFT|DPO|RLVR')


def expand_requests_1c(problems, templates, system, ladder='instruct', checkpoint='RLVR', calibration=False):
    """Expand one checkpoint's 52-row grid (four plain/none rows in calibration)."""
    _validate_1c_variant(ladder, checkpoint)
    if (set(templates['prose']) != set(PHASE1C_PROSE)
            or any(len(v) != 4 for v in templates['prose'].values())
            or set(templates['formats']) != set(PHASE1C_FORMATS)):
        raise ValueError('Phase 1c requires two prose cues, four templates, and six registered formats')
    phase = 'phase1c-calibration' if calibration else 'phase1c'
    cells = [('tutor', 'plain', 'none')] if calibration else [
        ('tutor', cue, fmt) for cue in PHASE1C_PROSE for fmt in PHASE1C_FORMATS
    ] + [('ceiling', 'plain', 'none')]
    result = []
    for p in problems:
        for arm, cue, fmt in cells:
            for i, template in enumerate(templates['prose'][cue]):
                rid = f"{phase}:{ladder}:{checkpoint}:p{p['problem_idx']:04d}:{cue}:{fmt}:t{i}:{arm}"
                user = template.replace('{problem}', p['question']) + templates['formats'][fmt]
                messages = [] if arm == 'ceiling' else [{'role': 'system', 'content': system}]
                messages.append({'role': 'user', 'content': user})
                result.append({**p, 'id': rid, 'phase': phase, 'ladder': ladder, 'checkpoint': checkpoint,
                               'thinking': ladder == 'think', 'cue': cue, 'prose': cue, 'format': fmt, 'template_idx': i,
                               'arm': arm, 'seed': request_seed(rid), 'messages': messages})
    return result


def build_1c(output_dir, ladder, checkpoint, limit_problems=None,
             system_path='prompts/system_tutor.txt', calibration=False):
    """Persist Phase 1c requests and provenance; limits select a fixed sample prefix.

    ``sample`` always preserves the full Phase 1 seed-0 order. Calibration uses
    seed 1 on eligible rows outside all 200, retained as ``calibration_sample``.
    ``selected_sample`` records the actual requested prefix in either mode.
    """
    from datasets import load_dataset
    _validate_1c_variant(ladder, checkpoint)
    size = 20 if calibration else (100 if ladder == 'think' else 200)
    limit = size if limit_problems is None else limit_problems
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= size:
        raise ValueError(f'limit-problems must be between 1 and {size}')
    rows = load_dataset('openai/gsm8k', 'main', split='test', revision=DATASET_REVISION)
    problems = sample_problems(rows)
    selected = problems[:size]
    calibration_sample = None
    if calibration:
        main_ids = {p['problem_idx'] for p in problems}
        heldout = [p for p in _eligible_problems(rows) if p['problem_idx'] not in main_ids]
        calibration_sample = random.Random(1).sample(heldout, 20)
        selected = calibration_sample
    system_path = str(system_path)
    system = (ROOT / system_path).read_text()
    template_path = 'prompts/templates_1c.json'
    templates = json.loads((ROOT / template_path).read_text())
    requests = expand_requests_1c(selected[:limit], templates, system, ladder, checkpoint, calibration)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ('DESIGN.md', system_path, template_path)}
    manifest = {'phase': 'phase1c-calibration' if calibration else 'phase1c',
                'ladder': ladder, 'checkpoint': checkpoint, 'thinking': ladder == 'think',
                'calibration': calibration, 'system_path': system_path,
                'dataset': 'openai/gsm8k', 'config': 'main', 'split': 'test',
                'revision': DATASET_REVISION, 'sample_seed': 0, 'sample_size': 200,
                'limit_problems': limit, 'requests': len(requests), 'source_sha256': hashes,
                'sample': problems, 'selected_sample': selected[:limit]}
    if calibration:
        manifest.update(calibration_seed=1, calibration_sample=calibration_sample)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'requests.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests))
    return requests, manifest


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
