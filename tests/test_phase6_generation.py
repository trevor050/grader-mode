"""CPU-only Phase 6 protocol tests; no model weights, GPU, or paid APIs."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from src import vast_generate_6 as worker
from src.models_6 import MODELS_6, SAMPLING, engine_kwargs
from src.vast_generate_4 import expand_requests as phase4_requests


def problems(n=200):
    return [{'problem_idx': i, 'question': f'Problem {i}?', 'gold': i + 1000}
            for i in range(n)]


def test_grid_reuses_exact_phase4_recipe_and_seeds():
    original = phase4_requests(problems(), *worker.recipe())
    by_id = {r['id']: r for r in original}
    for model, n in [('3.8', 2800), ('3.5', 1600)]:
        rows = worker.expand_requests(problems(), model)
        assert len(rows) == n
        assert len({r['id'] for r in rows}) == n
        assert set(Counter(worker.cell_id(r) for r in rows).values()) == {400}
        for row in rows:
            old = by_id[row['source_request_id']]
            for key in ('messages', 'seed', 'prefill_text', 'thinking', 'request_type'):
                assert row[key] == old[key]
            assert row['request_type'] in ('plain', 'explicit')
            assert row['arm'] != 'prefill' or row['prefill'] != 'empty'


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return 'system and user<|im_start|>assistant\n<think>\n' + (
            '' if kwargs['enable_thinking'] else '\n</think>\n\n')

    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return [10] * len(text)

    def decode(self, ids, **kwargs):
        assert kwargs['skip_special_tokens'] is False
        assert kwargs['clean_up_tokenization_spaces'] is False
        mapping = {248068: '<think>', 248069: '</think>', 248046: '<|im_end|>',
                   248044: '<|endoftext|>', 7: ' thinking ', 8: ' answer '}
        return ''.join(mapping.get(i, 'x') for i in ids)


def test_native_rendering_and_literal_prefill():
    tok = Tokenizer()
    rows = worker.expand_requests(problems(1), '3.8')
    for row in rows:
        rendered = worker.render_prompt(tok, row, '3.8')
        kwargs = tok.calls[-1]
        assert kwargs['preserve_thinking'] is False
        assert kwargs['reasoning_effort'] == 'medium'
        assert kwargs['enable_thinking'] == row['thinking']
        if row['arm'] == 'prefill':
            assert rendered.endswith('<think>\n' + row['prefill_text'])
            assert rendered.count('<think>') == 1
    worker.render_prompt(tok, rows[0], '3.5')
    assert 'reasoning_effort' not in tok.calls[-1]


def test_context_checked_before_engine_load():
    row = worker.expand_requests(problems(1), '3.8')[0]
    tok = Tokenizer()
    tok.encode = lambda *a, **kw: [1] * 4097
    with pytest.raises(ValueError, match='8192'):
        worker.prepare_requests(tok, [row], '3.8')


def test_pilots_balanced_and_disjoint_namespaces():
    for model, cells in [('3.8', 7), ('3.5', 4)]:
        rows = worker.expand_requests(problems(), model)
        smoke = worker.pilot_requests(rows, model, 'smoke')
        throughput = worker.pilot_requests(rows, model, 'throughput')
        assert len(smoke) == 8
        assert len(throughput) == 256
        counts = Counter(worker.cell_id(r) for r in throughput)
        assert len(counts) == cells and max(counts.values()) - min(counts.values()) <= 1
        assert {worker.cell_id(r) for r in smoke} == set(counts)
        assert not {r['id'] for r in smoke + throughput} & {r['id'] for r in rows}
        assert len({r['id'] for r in smoke + throughput}) == 264
        assert all(r['excluded_from_main'] for r in smoke + throughput)
        if model == '3.5':
            assert Counter(r['template_idx'] for r in smoke) == {0: 4, 1: 4}
        else:
            assert len({r['problem_idx'] for r in smoke}) == 2


def test_pins_engine_and_sampling():
    for model in ('3.8', '3.5'):
        opts = engine_kwargs(model, 256)
        assert opts['revision'] == MODELS_6[model]['revision']
        assert opts['tokenizer_revision'] == opts['revision']
        assert opts['language_model_only'] is True
        assert opts['enforce_eager'] is False
        assert opts['max_model_len'] == 8192
        assert opts['dtype'] == 'bfloat16' and opts['quantization'] == 'fp8'
        assert opts['generation_config'] == 'vllm'
    assert SAMPLING == {'temperature': .7, 'top_p': .95, 'top_k': -1, 'max_tokens': 4096}
    with pytest.raises(ValueError):
        engine_kwargs('3.8', 12)


@pytest.mark.parametrize('ids,thinking,reasoning,final', [
    ([7, 248069, 8, 248046], True, ' thinking ', ' answer '),
    ([7, 248044], True, ' thinking ', ''),
    ([8, 248046], False, '', ' answer '),
    ([248068, 7, 248069, 8, 248046], True, ' thinking ', ' answer '),
    ([248069, 248046], True, '', ''),
])
def test_raw_token_boundaries_and_whitespace(ids, thinking, reasoning, final):
    result = worker.token_fields(Tokenizer(), ids, thinking, 'literal ' if thinking else '')
    assert result['raw_text'] == Tokenizer().decode(ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
    assert result['reasoning_continuation'] == reasoning
    assert result['final_continuation'] == final
    assert result['reasoning_with_prefill'] == ('literal ' if thinking else '') + reasoning
    a, b = result['token_spans']['final']
    assert Tokenizer().decode(ids[a:b], skip_special_tokens=False,
                              clean_up_tokenization_spaces=False) == final


def test_permit_drop_malformed_binding_and_deadline(tmp_path):
    path = tmp_path / 'main.permit.json'
    with pytest.raises(TimeoutError):
        worker.wait_permit(path, deadline_unix=time.time() - 1, model='3.8', requests_sha256='abc')
    path.write_text('{')
    with pytest.raises(ValueError, match='permit'):
        worker.wait_permit(path, deadline_unix=time.time() + 10, model='3.8', requests_sha256='abc')
    path.write_text(json.dumps({'decision': 'run', 'model': '3.5'}))
    with pytest.raises(ValueError, match='model'):
        worker.wait_permit(path, deadline_unix=time.time() + 10, model='3.8', requests_sha256='abc')
    path.write_text(json.dumps({'decision': 'drop'}))
    assert worker.wait_permit(path, deadline_unix=time.time() + 10,
                              model='3.8', requests_sha256='abc') == 'drop'


def test_immutable_outputs_and_partial_failure(tmp_path):
    path = tmp_path / 'chunk.jsonl'
    worker.atomic_write(path, 'original', immutable=True)
    with pytest.raises(FileExistsError):
        worker.atomic_write(path, 'different', immutable=True)
    assert path.read_text() == 'original'
    with pytest.raises(FileExistsError):
        worker.claim_output(tmp_path)


def test_output_can_contain_only_current_shell_log(tmp_path):
    (tmp_path / 'worker.log').write_text('current shell log')
    worker.claim_output(tmp_path)
    assert (tmp_path / 'worker.log').read_text() == 'current shell log'
    with pytest.raises(FileExistsError):
        worker.claim_output(tmp_path)


class SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Engine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params))
        assert all(p.max_tokens == 4096 and not p.skip_special_tokens for p in sampling_params)
        return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'], outputs=[
            SimpleNamespace(token_ids=[7, 248069, 8, 248046],
                            text=' thinking </think> answer <|im_end|>',
                            finish_reason='stop', stop_reason=248046)]) for p in prompts]


@pytest.mark.parametrize('model,drop_stage,expected', [
    ('3.8', None, ['smoke', 'throughput', 'main', 'prefill']),
    ('3.5', None, ['smoke', 'throughput', 'main']),
    ('3.8', 'throughput', ['smoke']),
    ('3.8', 'main', ['smoke', 'throughput']),
    ('3.8', 'prefill', ['smoke', 'throughput', 'main']),
])
def test_one_engine_staged_lifecycle(monkeypatch, tmp_path, model, drop_stage, expected):
    rows = worker.expand_requests(problems(2), model)
    engine, tok = Engine(), Tokenizer()
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SamplingParams))
    permitted = []

    def permit(path, **kwargs):
        stage = path.name.split('.')[0]
        # A permit must be requested only after the preceding done file exists.
        previous = {'throughput': 'smoke', 'main': 'throughput', 'prefill': 'main'}[stage]
        assert (tmp_path / f'{previous}.done.json').is_file()
        permitted.append(stage)
        return 'drop' if stage == drop_stage else 'run'

    monkeypatch.setattr(worker, 'wait_permit', permit)
    manifest = {'source_sha256': {}, 'registration_commit': '8af0156'}
    worker.execute_stages(engine, tok, rows, model, tmp_path, manifest,
                          settings={'test': True}, requests_sha256='abc',
                          deadline_unix=time.time() + 30)
    chunks = sorted(tmp_path.glob('*/chunk-*.jsonl'))
    actual = {p.parent.name for p in chunks}
    assert actual == set(expected)
    all_rows = [json.loads(l) for p in chunks for l in p.read_text().splitlines()]
    assert len({r['id'] for r in all_rows}) == len(all_rows)
    for stage in expected:
        done = json.loads((tmp_path / f'{stage}.done.json').read_text())
        assert done['status'] == 'completed'
        assert sum(c['count'] for c in done['per_cell'].values()) == done['requests']
        assert done['output_tokens'] == done['requests'] * 4
        assert done['generation_seconds'] >= 0
        assert all((tmp_path / p['path']).is_file() for p in done['chunks'])
    if 'main' in expected:
        assert len([r for r in all_rows if r['stage'] == 'main']) == 16
    assert json.loads((tmp_path / 'worker.done.json').read_text())['status'] == (
        'dropped' if drop_stage else 'completed')


def test_failed_chunk_does_not_publish_completion(monkeypatch, tmp_path):
    rows = worker.expand_requests(problems(2), '3.8')
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SamplingParams))
    engine = Engine()
    engine.generate = lambda *a, **kw: []
    with pytest.raises(ValueError, match='count'):
        worker.execute_stages(engine, Tokenizer(), rows, '3.8', tmp_path, {},
                              settings={}, requests_sha256='abc', deadline_unix=time.time()+30)
    assert not list(tmp_path.glob('*/chunk-*.jsonl'))
    assert not (tmp_path / 'smoke.done.json').exists()


def test_metadata_hashes_checked_before_engine(monkeypatch, tmp_path):
    for filename in MODELS_6['3.8']['file_sha256']:
        (tmp_path / filename).write_text('changed')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kw: str(tmp_path)))
    with pytest.raises(ValueError, match='hash'):
        worker.checkpoint_metadata('3.8')


def test_supervisor_hard_kills_unresponsive_child(monkeypatch, tmp_path):
    """Exercise a real process-group timeout with a sleeping CPU process."""
    real_popen = worker.subprocess.Popen
    processes = []

    def sleeping_child(argv, **kwargs):
        assert kwargs['start_new_session'] is True
        process = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, 'Popen', sleeping_child)
    args = SimpleNamespace(deadline_unix=time.time() + .15, output_dir=tmp_path)
    start = time.monotonic()
    assert worker.supervise(args, []) == 124
    assert time.monotonic() - start < 5
    assert processes[0].poll() is not None
    assert json.loads((tmp_path / 'worker.timeout.json').read_text())['status'] == 'timeout'


def test_build_uses_fixed_sample_and_rejects_mutated_specs(tmp_path):
    # Use the already-generated controller artifact when available; reconstruct
    # from the original manifest without network access otherwise.
    if not worker.DEFAULT_PHASE4_MANIFEST.is_file():
        pytest.skip('original local Phase 4 manifest unavailable')
    spec = worker.build(tmp_path / 'requests.json')
    assert len(spec['models']['3.8']['requests']) == 2800
    assert len(spec['models']['3.5']['requests']) == 1600
    assert (tmp_path / 'requests.README.md').is_file()
    worker.validate_spec(spec, '3.8')
    spec['models']['3.8']['requests'][0]['seed'] += 1
    with pytest.raises(ValueError, match='mixed or incomplete'):
        worker.validate_spec(spec, '3.8')
    with pytest.raises(FileExistsError):
        worker.build(tmp_path / 'requests.json')


def test_saved_rows_match_parent_replay_and_analysis_contract(monkeypatch, tmp_path):
    """Synthetic integration only: worker runtime never invokes the scorer."""
    scorer = pytest.importorskip('src.score_6')
    analysis = pytest.importorskip('src.analyze_6')
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SamplingParams))
    monkeypatch.setattr(worker, 'wait_permit', lambda *a, **kw: 'run')
    raw = []
    for model in ('3.8', '3.5'):
        directory = tmp_path / model
        rows = worker.expand_requests(problems(2), model)
        worker.execute_stages(Engine(), Tokenizer(), rows, model, directory, {'source_sha256': worker.SOURCE_SHA256},
                              settings={}, requests_sha256='abc', deadline_unix=time.time() + 30)
        raw.extend(json.loads(line) for path in directory.glob('*/chunk-*.jsonl')
                   if path.parent.name in ('main', 'prefill')
                   for line in path.read_text().splitlines())
    scored = scorer.score_rows(raw)
    assert len(scored) == 44
    assert analysis.validate_grid(scored, expected_problems=2)['n_responses'] == 44
