"""Registered Phase 4 request construction and Modal-free shared vLLM entrypoint."""
import argparse
import hashlib
import json
from pathlib import Path

from src.build_prompts import ROOT, DATASET_REVISION, sample_problems, request_seed
from src.models import MODELS


def expand_requests(problems, templates, phase1_templates, system, prefill=True):
    rows = []
    for p in problems:
        for template_idx, template in enumerate(phase1_templates['cues']['none'][:2]):
            # Common random numbers across thinking/request/prefill interventions.
            seed_id = f"phase4:p{p['problem_idx']:04d}:t{template_idx}"
            for arm in ('main', 'prefill') if prefill else ('main',):
                for thinking in (False, True) if arm == 'main' else (True,):
                    for request_type in ('plain', 'explicit', 'boxed') if arm == 'main' else ('explicit', 'boxed'):
                        for prefill_name in ('none',) if arm == 'main' else templates['prefills']:
                            rid = f'{seed_id}:{arm}:thinking-{int(thinking)}:{request_type}:{prefill_name}'
                            rows.append({**p, 'problem_id': p['problem_idx'], 'phase': 'phase4',
                                'id': rid, 'template_idx': template_idx, 'template': template_idx,
                                'arm': arm, 'thinking': thinking, 'request_type': request_type,
                                'prefill': prefill_name,
                                'prefill_text': templates['prefills'][prefill_name] if arm == 'prefill' else '',
                                'seed': request_seed(seed_id), 'messages': [
                                    {'role': 'system', 'content': system},
                                    {'role': 'user', 'content': template.replace('{problem}', p['question']) + templates['requests'][request_type]}]})
    return rows


def build(output):
    from datasets import load_dataset
    problems = sample_problems(load_dataset('openai/gsm8k', 'main', split='test', revision=DATASET_REVISION))
    templates = json.loads((ROOT / 'prompts/templates_4.json').read_text())
    phase1 = json.loads((ROOT / 'prompts/templates.json').read_text())
    system = (ROOT / 'prompts/system_tutor.txt').read_text()
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
              ('DESIGN.md', 'prompts/templates_4.json', 'prompts/templates.json', 'prompts/system_tutor.txt')}
    manifest = {'phase': 'phase4', 'registration_commit': '7f2213b', 'dataset_revision': DATASET_REVISION,
                'sample': problems, 'sample_seed': 0, 'source_sha256': hashes,
                'models': {m: r for m, r in MODELS.items() if m.endswith(('14B', '32B'))},
                'max_new_tokens_both_modes': 4096, 'seed_pairing': 'SHA256 phase4:problem:template, first 4 bytes'}
    spec = {m: {'requests': expand_requests(problems, templates, phase1, system, m.endswith('14B')),
                'manifest': manifest} for m in manifest['models']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec))
    (output.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--model', choices=('14B', '32B'))
    parser.add_argument('--run-id', default='phase4-main')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.build:
        spec = build(args.build)
        print({m: len(s['requests']) for m,s in spec.items()})
        return
    from src.vast_generate import _install_modal_stub
    _install_modal_stub()
    import vllm
    original = vllm.LLM.generate
    vllm.LLM.generate = lambda self, *a, **k: original(self, *a, **{**k, 'use_tqdm': True})
    from src.modal_app import generate, render_prompt
    from transformers import AutoTokenizer
    model = 'Qwen/Qwen3-' + args.model
    spec = json.loads(args.requests.read_text())[model]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=MODELS[model])
    examples = {}
    for row in spec['requests']:
        if row['arm'] == 'prefill' and row['prefill'] not in examples:
            base = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompt = render_prompt(tokenizer, row)
            examples[row['prefill']] = {'native_assistant_tail': base.rsplit('<|im_start|>assistant\n',1)[-1],
                'rendered_prompt_tail': prompt[-500:],
                'prefill_tokens': len(tokenizer.encode(row['prefill_text'], add_special_tokens=False))}
    directory = Path('/raw') / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'Qwen3-{args.model}.prefill-rendering.json').write_text(json.dumps(examples,indent=2)+'\n')
    generate(model, spec['requests'], args.run_id, {**spec['manifest'], 'host': 'vast.ai'},
             {'gpu': 'RTX PRO 6000 WS', 'batch_size': args.batch_size, 'chunk_size': 256,
              'max_tokens_override': 4096, 'phase4_gate': True})


if __name__ == '__main__':
    main()
