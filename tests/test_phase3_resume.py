"""CPU-only contingency contracts. Never contact Modal or inspect real outcomes."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from src.models import MODELS
from src.phase3 import MODEL, CAP_SECONDS, RATE, build_requests, choose_neurons, score_generation


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture(scope='module')
def frozen():
    sample = [dict(problem_idx=i, question=f'Synthetic question {i}?', gold=42) for i in range(200)]
    manifest = dict(phase='phase3', preregistration_commit='0e6a330', cap_usd=7., sample=sample,
                    source_sha256={'DESIGN.md': 'a'*64})
    requests = build_requests(sample)
    numerical = {k: np.full((2, 50), v) for k, v in
                 [('d', 1.), ('delta', 1.), ('tutor_mean', 1.), ('neutral_mean', 0.),
                  ('tutor_sd', 1.), ('neutral_sd', 1.)]}
    numerical['n_prompts'] = 400
    neurons = choose_neurons(numerical)
    tracking = []
    for r in requests:
        if r['split'] == 'test':
            tracking.append({k: r[k] for k in ('id', 'problem_idx', 'selection_rank', 'template_idx', 'gold')}
                            | dict(leak=True, excluded_from_rates=False, closed_reasoning=False,
                                   n_reasoning_tokens=0, slopes=None, first_quarter=None, deciles=None,
                                   rendered_prompt_sha256=sha(json.dumps(r['messages'])),
                                   source_response_sha256=sha('<think>unfinished')))
    rows, batches = [], []
    dev = [r for r in requests if r['split'] == 'dev' and r['template_idx'] == 0]
    for cell, c in [('baseline', 0), ('dose_1', 1), ('dose_2', 2)]:
        for r in dev:
            text = '<think>work</think>' + ('Try another step.' if c == 2 and r['selection_rank'] <= 20 else '\\boxed{42}')
            rows.append({**r, 'id': f'{r["id"]}:{cell}:c{c:g}', 'generation_cell': cell,
                         'intervention': 'baseline' if c == 0 else 'tutor', 'c': c,
                         'run_id': 'source', 'model': MODEL, 'model_revision': MODELS[MODEL],
                         'response': text, 'finish_reason': 'stop', 'output_tokens': 30,
                         'final_reply_tokens': 5, 'end_token_id': 151645, 'batch_seed': dev[0]['seed'],
                         'budget_terminated': False, 'rendered_prompt_sha256': sha(json.dumps(r['messages'])),
                         **score_generation(text, r['gold'], 'stop')})
        batches.append(dict(split='dev', cell=cell, c=c, n=100, seed=dev[0]['seed'],
                            first_id=dev[0]['id'], last_id=dev[-1]['id'], seconds=100.,
                            budget_terminated=False, decode_steps=31))
    selection = dict(c_star=2, dose_threshold_met=True, dev_baseline_leak=1.,
                     dose_trials=[dict(c=1, n=100, n_valid=100, leak=1., reduction=0.),
                                  dict(c=2, n=100, n_valid=100, leak=.8, reduction=.2)],
                     test_templates=None, test_generation_started=False)
    metrics = dict(run_id='source', status='failed', stage='finished',
                   stop_reason='ValueError: Reduced test grid exceeds GPU budget',
                   model=MODEL, revision=MODELS[MODEL], gpu='L40S', manifest=deepcopy(manifest),
                   screen_prompts_per_system=[400, 400], tracking_rows=400, output_rows=len(rows),
                   eos_token_ids=[151645, 151643], batches=batches,
                   sampling=dict(enable_thinking=True, dtype='bfloat16', temperature=.7, top_p=.95,
                                 top_k=0, max_new_tokens=3072, left_padding=True))
    attempts = [dict(run_id='discarded', status='failed_or_cancelled', wall_seconds=348.528,
                     started_unix=100., ended_unix=448.528, zero_containers_verified_unix=448.528),
                dict(run_id='source', status='failed', wall_seconds=5000., started_unix=500.,
                     ended_unix=5500., zero_containers_verified_unix=5500.)]
    return dict(source_run_id='source', requests=requests, generation_rows=rows,
                tracking_rows=tracking, neurons=neurons, selection=selection, metrics=metrics,
                manifest=manifest, attempts=attempts)


@pytest.fixture
def case(frozen):
    return deepcopy(frozen)


def validate(case):
    from src.phase3_resume import validate_eligibility
    return validate_eligibility(**case)


def test_ordered_complete_dev_grid_is_eligible(case):
    result = validate(case)
    assert result['c_star'] == 2
    assert result['n_dev_rows'] == 300 and result['n_tracking_rows'] == 400
    assert result['previous_seconds'] == pytest.approx(5348.528)


def test_only_one_continuation_allocation_per_logical_source(case):
    case['attempts'].append(dict(run_id='previous-resume', source_run_id='source', status='failed',
                                wall_seconds=30., started_unix=5510., ended_unix=5540., zero_containers_verified_unix=5540.))
    with pytest.raises(ValueError, match='single-use'):
        validate(case)


@pytest.mark.parametrize('mutation', [
    lambda c: c['attempts'][-1].update(status='running'),
    lambda c: c['attempts'][0].update(status='submission_unresolved'),
    lambda c: c['attempts'][-1].pop('ended_unix'),
    lambda c: c['metrics'].update(status='running'),
    lambda c: c['metrics'].update(stop_reason='RuntimeError: CUDA failure'),
    lambda c: c['metrics'].update(stop_reason='TimeoutError: deadline'),
    lambda c: c['metrics'].update(tracking_rows=399),
    lambda c: c['metrics'].update(screen_prompts_per_system=[400, 399]),
    lambda c: c['metrics'].update(manifest={}),
    lambda c: c['selection'].update(test_generation_started=True),
    lambda c: c['selection'].update(test_templates=1),
    lambda c: c['selection'].update(c_star=4),
    lambda c: c['selection']['dose_trials'][0].update(leak=.7),
    lambda c: c['generation_rows'].pop(),
    lambda c: c['generation_rows'].append(deepcopy(c['generation_rows'][0])),
    lambda c: c['generation_rows'][0].update(split='test'),
    lambda c: c['generation_rows'][0].update(seed=3),
    lambda c: c['generation_rows'][0].update(leak=False),
    lambda c: c['generation_rows'][0].update(response='changed'),
    lambda c: c['tracking_rows'].pop(),
    lambda c: c['neurons']['control_signs'][0].__setitem__(0, 0),
    lambda c: c['metrics']['batches'][0].update(split='test'),
])
def test_ineligible_source_fails_closed(case, mutation):
    mutation(case)
    with pytest.raises(ValueError):
        validate(case)


def test_fallback_eight_requires_all_ordered_doses(case):
    for row in case['generation_rows']:
        if row['generation_cell'] == 'dose_2':
            row['response'] = '<think>work</think>\\boxed{42}'
            row.update(score_generation(row['response'], row['gold'], 'stop'))
    for c in (4, 8):
        for row in [deepcopy(r) for r in case['generation_rows'] if r['generation_cell'] == 'dose_1']:
            row.update(c=c, generation_cell=f'dose_{c}', id=f'{row["phase1b_id"]}:dose_{c}:c{c}')
            case['generation_rows'].append(row)
        batch = deepcopy(case['metrics']['batches'][1])
        batch.update(c=c, cell=f'dose_{c}')
        case['metrics']['batches'].append(batch)
    case['metrics']['output_rows'] = 500
    case['selection'].update(c_star=8, dose_threshold_met=False,
                             dose_trials=[dict(c=c, n=100, n_valid=100, leak=1., reduction=0.) for c in (1, 2, 4, 8)])
    assert validate(case)['c_star'] == 8
    case['selection']['dose_trials'].pop()
    with pytest.raises(ValueError):
        validate(case)


def test_exact_capacity_keeps_27_and_preserves_work_reserve():
    from src.phase3_resume import exact_batch_capacity
    kv, prompt, reserve = 10000, 123, 4*1024**3
    cost = kv*(prompt+3072)
    assert exact_batch_capacity(reserve+27*cost+cost//2, kv, prompt) == 27
    assert exact_batch_capacity(reserve+100*cost, kv, prompt) == 64
    with pytest.raises(ValueError):
        exact_batch_capacity(reserve+cost-1, kv, prompt)


def test_full_padded_batch_projection_and_two_then_one():
    from src.phase3_resume import project_test_plan
    # 200/27 => 8 full-cost batches/cell, 100/27 => 4, not per-request pricing.
    one = 4*7*100*1.2
    two = 8*7*100*1.2
    p = project_test_plan(100., 27, 1000.)
    assert p['test_templates'] == 2 and p['projected_test_seconds'] == two
    assert p['batches_per_cell'] == 8 and p['headroom'] == 1.2
    p = project_test_plan(100., 27, CAP_SECONDS-300-one)
    assert p['test_templates'] == 1 and p['projected_test_seconds'] == one
    assert p['projected_gpu_usd'] <= 7.+1e-12
    with pytest.raises(ValueError, match='Reduced test grid'):
        project_test_plan(100., 27, CAP_SECONDS-300-one+.01)


@pytest.mark.parametrize('seconds,batch,spent', [(0, 27, 1), (math.nan, 27, 0),
    (1, True, 0), (1, 0, 0), (1, 65, 0), (1, 27, -1), (1, 27, math.inf)])
def test_invalid_budget_inputs_rejected(seconds, batch, spent):
    from src.phase3_resume import project_test_plan
    with pytest.raises(ValueError):
        project_test_plan(seconds, batch, spent)


def test_lifetime_ledger_uncertainty_and_limits():
    from src.phase3_resume import allocation_limits, close_allocation, ledger_seconds
    a = [dict(run_id='old', status='failed', started_unix=100.,ended_unix=448.528,
              zero_containers_verified_unix=448.528, wall_seconds=348.528)]
    assert ledger_seconds(a) == 348.528
    limits = allocation_limits(348.528, 1000.)
    assert limits['timeout_seconds'] == int(CAP_SECONDS-348.528-120-180)
    assert limits['absolute_deadline_unix'] == pytest.approx(1000+CAP_SECONDS-348.528-180)
    uncertain = close_allocation(call_known=False, elapsed=1., previous_seconds=348.528,
                                 status='failed', zero_containers=False)
    assert uncertain['status'] == 'submission_unresolved'
    assert uncertain['wall_seconds'] == CAP_SECONDS-348.528
    a.append(dict(run_id='uncertain', **uncertain))
    with pytest.raises(ValueError):
        ledger_seconds(a)


def test_shutdown_exposure_is_reserved_until_zero_containers():
    from src.phase3_resume import close_allocation
    result = close_allocation(call_known=True, elapsed=200., previous_seconds=500.,
                              status='failed', zero_containers=False)
    assert result['status'] == 'shutdown_unresolved'
    assert result['wall_seconds'] == CAP_SECONDS-500.
    result = close_allocation(call_known=True, elapsed=220., previous_seconds=500.,
                              status='completed', zero_containers=True)
    assert result['wall_seconds'] == 220. and result['status'] == 'completed'


def test_zero_container_evidence_is_strict():
    from src.phase3_resume import validate_no_containers
    validate_no_containers([])
    validate_no_containers([{'app_name': 'unrelated-study', 'app_id': 'ap-other'}])
    for name in ('grader-mode-phase3', 'grader-mode-phase3-resume'):
        with pytest.raises(ValueError):
            validate_no_containers([{'app_name': name, 'app_id': 'ap-ours'}])
    with pytest.raises(ValueError):
        validate_no_containers({'containers': []})
    with pytest.raises(ValueError):
        validate_no_containers([{}])


def test_source_hashes_fail_closed_without_writes(tmp_path):
    from src.phase3_resume import verify_hashes
    path = tmp_path/'frozen.py'; path.write_text('original')
    verify_hashes(tmp_path, {'frozen.py': sha('original')})
    path.write_text('altered')
    with pytest.raises(ValueError):
        verify_hashes(tmp_path, {'frozen.py': sha('original')})
    with pytest.raises(ValueError):
        verify_hashes(tmp_path, {'../escape': 'a'*64})


def test_test_batches_are_fixed_across_seven_cells(case):
    from src.phase3_resume import make_test_batches
    rows = [r for r in case['requests'] if r['split'] == 'test' and r['template_idx'] < 2]
    batches = make_test_batches(rows, 2, 27)
    assert [len(b) for b in batches] == [27]*7+[11]
    assert [r['id'] for b in batches for r in b] == [r['id'] for r in rows]
    assert [len(b) for b in make_test_batches(rows, 1, 27)] == [27, 27, 27, 19]
    with pytest.raises(ValueError):
        make_test_batches(rows[:-1], 2, 27)


def test_new_row_keeps_logical_identity_and_separates_allocation(case):
    from src.phase3_resume import make_generation_row
    source = next(r for r in case['requests'] if r['split'] == 'test')
    row = make_generation_row(source, cell='control_3', c=2, source_run_id='source',
        allocation_run_id='fresh-resume', response='<think>work</think>\\boxed{42}',
        output_tokens=30, end_token_id=151645, batch_seed=source['seed'],
        rendered_prompt_sha256='a'*64, final_reply_tokens=5)
    for key in source.keys()-{'id'}:
        assert row[key] == source[key]
    assert row['id'] == f'{source["id"]}:control_3:c2'
    assert row['run_id'] == 'source' and row['allocation_run_id'] == 'fresh-resume'
    assert row['leak'] and row['correct_commit'] and row['final_reply'] == '\\boxed{42}'


def test_remote_payload_detects_mutation(case):
    from src.phase3_resume import digest, validate_payload
    package = dict(source_manifest=case['manifest'], source_selection=case['selection'],
        eligibility=validate(case), neurons=case['neurons'],
        dev_requests=[r for r in case['requests'] if r['split']=='dev' and r['template_idx']==0],
        test_requests=[r for r in case['requests'] if r['split']=='test' and r['template_idx']<2])
    package['payload_sha256'] = digest(package)
    validate_payload(package)
    package['dev_requests'][0]['seed'] += 1
    with pytest.raises(ValueError, match='payload changed'):
        validate_payload(package)


def test_modal_module_import_is_offline_and_function_is_single_use():
    # Importing definitions must not run main/spawn or load torch/transformers.
    import ast
    from pathlib import Path
    import src.modal_app_3_resume as runner
    assert runner.app.name == 'grader-mode-phase3-resume'
    tree = ast.parse(Path(runner.__file__).read_text())
    study = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resume_study')
    decorator = study.decorator_list[0]
    values = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords
              if kw.arg in ('gpu', 'startup_timeout', 'retries', 'single_use_containers',
                            'min_containers', 'buffer_containers', 'max_containers')}
    assert values == dict(gpu='L40S', startup_timeout=120, retries=0, single_use_containers=True,
                          min_containers=0, buffer_containers=0, max_containers=1)


def test_remote_directory_check_lists_root_and_rejects_empty_existing_directory(monkeypatch):
    from types import SimpleNamespace
    import src.modal_app_3_resume as runner
    class Volume:
        def listdir(self, path):
            assert path == '/'
            return [SimpleNamespace(path='/fresh-resume')]
    monkeypatch.setattr(runner, 'volume', Volume())
    with pytest.raises(ValueError):
        runner.require_remote_fresh('fresh-resume')
    runner.require_remote_fresh('other')


def test_recovery_preserves_partial_outputs_and_never_overwrites(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.modal_app_3_resume as runner
    class Volume:
        def listdir(self, path):
            assert path == 'resume'
            return [SimpleNamespace(path='resume/generations.jsonl'),
                    SimpleNamespace(path='resume/metrics.json'), SimpleNamespace(path='resume/metrics.json.tmp')]
        def read_file(self, path):
            yield b'partial'
    monkeypatch.setattr(runner, 'volume', Volume())
    assert runner.recover_artifacts(tmp_path, 'resume') == []
    assert (tmp_path/'generations.jsonl').read_bytes() == b'partial'
    assert not (tmp_path/'metrics.json.tmp').exists()
    assert len(runner.recover_artifacts(tmp_path, 'resume')) == 2


@pytest.mark.parametrize('mode', ['completed', 'lost_spawn', 'shutdown_unknown', 'remote_error', 'slow_shutdown'])
def test_local_lifecycle_is_offline_and_accounts_uncertainty(case, tmp_path, monkeypatch, mode):
    """Exercise the real controller with all remote boundaries replaced by fakes."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    import src.build_prompts
    import src.modal_app_3_resume as runner
    root = tmp_path/'project'; out = root/'results/phase3'; out.mkdir(parents=True)
    ledger = out/'gpu-attempts.json'; ledger.write_text(json.dumps(case['attempts']))
    original_ledger = deepcopy(case['attempts'])
    monkeypatch.setattr(src.build_prompts, 'ROOT', root)
    package = dict(source_run_id='source', source_manifest={'source_sha256': {}},
                   source_artifacts_sha256={}, payload_sha256='a'*64)
    monkeypatch.setattr(runner, 'load_source_package', lambda *a: package)
    monkeypatch.setattr(runner, 'validate_payload', lambda *a: None)
    monkeypatch.setattr(runner, 'file_hash', lambda *a: 'a'*64)
    monkeypatch.setattr(runner, 'verify_hashes', lambda *a: None)
    monkeypatch.setattr(runner, 'require_remote_fresh', lambda *a: None)
    monkeypatch.setattr(runner, 'recover_artifacts', lambda *a: [])
    monkeypatch.setattr(runner.time, 'sleep', lambda *a: None)
    checks = []
    def evidence():
        checks.append(1)
        if (mode == 'shutdown_unknown' and len(checks) > 1) or (mode == 'slow_shutdown' and 2 <= len(checks) <= 6):
            raise RuntimeError('cannot verify shutdown')
        return dict(checked_unix=100., containers=[])
    monkeypatch.setattr(runner, 'container_evidence', evidence)
    @contextmanager
    def ephemeral():
        yield {}
    monkeypatch.setattr(runner, 'modal', SimpleNamespace(Dict=SimpleNamespace(ephemeral=ephemeral)))
    cancelled = []
    class Call:
        object_id = 'fake-call'
        def get(self, timeout):
            if mode == 'remote_error':
                raise RuntimeError('remote failed')
            return dict(status='completed', function_seconds=123.)
        def cancel(self, terminate_containers):
            cancelled.append(terminate_containers)
    class Function:
        def with_options(self, **kwargs):
            assert kwargs['retries'] == 0
            assert kwargs['timeout'] == int(CAP_SECONDS-5348.528-120-180)
            return self
        def spawn(self, *args):
            if mode == 'lost_spawn':
                raise RuntimeError('submission response lost')
            return Call()
    monkeypatch.setattr(runner, 'resume_study', Function())
    if mode in ('completed','slow_shutdown'):
        runner.main.info.raw_f(source_run_id='source', run_id='fresh-resume')
    else:
        with pytest.raises((RuntimeError, ValueError)):
            runner.main.info.raw_f(source_run_id='source', run_id='fresh-resume')
    attempts = json.loads(ledger.read_text())
    assert attempts[:-1] == original_ledger
    attempt = attempts[-1]
    if mode in ('lost_spawn', 'shutdown_unknown'):
        assert attempt['status'] == ('submission_unresolved' if mode == 'lost_spawn' else 'shutdown_unresolved')
        assert attempt['wall_seconds'] == CAP_SECONDS-5348.528
    else:
        assert attempt['status'] == ('completed' if mode in ('completed','slow_shutdown') else 'failed_or_cancelled')
    assert cancelled == ([] if mode == 'lost_spawn' else [True])
    assert (root/'results/raw/fresh-resume/local-execution.json').exists()
    assert set(p.name for p in out.iterdir()) == {'gpu-attempts.json', 'run.lock'}


def test_load_source_package_verifies_frozen_bundle_and_never_writes(case, tmp_path):
    from src.phase3_resume import FROZEN_PATHS, file_hash, load_source_package, validate_payload
    root = tmp_path/'project'
    raw = root/'results/raw/source'; raw.mkdir(parents=True)
    out = root/'results/phase3'; out.mkdir(parents=True)
    for name in FROZEN_PATHS:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic frozen source')
    sources, labels = [], []
    for r in case['requests']:
        if r['split'] != 'test':
            continue
        response = '<think>unfinished'
        source = {**r, 'model': MODEL, 'model_revision': MODELS[MODEL], 'response': response,
                  'finish_reason': 'length', 'rendered_prompt_sha256': sha(json.dumps(r['messages']))}
        sources.append(source)
        labels.append({**source, **score_generation(response, r['gold'], 'length')})
    for name, rows in [('Qwen3-14B.jsonl', sources), ('scored.jsonl', labels)]:
        (root/'results/raw/phase1b-full'/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for row in case['tracking_rows']:
        row.update(leak=False, excluded_from_rates=True)
    case['manifest']['source_sha256'] = {name: file_hash(root/name) for name in FROZEN_PATHS}
    case['metrics']['manifest'] = deepcopy(case['manifest'])
    old = root/'results/raw/phase1-stage2/manifest.json'; old.parent.mkdir(parents=True)
    old.write_text(json.dumps(case['manifest']))
    numerical = {key: np.full((40, 17408), value) for key, value in
                 [('d', 1.), ('delta', 1.), ('tutor_mean', 1.), ('neutral_mean', 0.),
                  ('tutor_sd', 1.), ('neutral_sd', 1.)]}
    case['neurons'] = choose_neurons({**numerical, 'n_prompts': 400})
    np.savez_compressed(out/'screen-statistics.npz', **numerical)
    for name in ('manifest', 'metrics', 'selection', 'neurons'):
        (out/(name+'.json')).write_text(json.dumps(case[name]))
    for name, rows in [('generations.jsonl', case['generation_rows']), ('tracking.jsonl', case['tracking_rows'])]:
        (raw/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    np.savez_compressed(raw/'trajectories.npz', scores=np.empty((0, 6)), token_ids=np.empty(0, dtype=np.int32),
                        offsets=np.zeros(401, dtype=np.int64), ids=[r['id'] for r in case['tracking_rows']])
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*') if p.is_file()}
    package = load_source_package(root, 'source', case['attempts'])
    validate_payload(package)
    validate_payload(package, rebuild_requests=False)
    assert len(package['dev_requests']) == 100 and len(package['test_requests']) == 200
    assert package['source_selection'] == case['selection']
    assert before == {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*') if p.is_file()}
    (root/'DESIGN.md').write_text('changed')
    with pytest.raises(ValueError, match='frozen source changed'):
        load_source_package(root, 'source', case['attempts'])


def test_resume_rejects_prior_allocation_without_verified_shutdown():
    from src.phase3_resume import ledger_seconds
    attempts=[dict(run_id='prior',status='failed',started_unix=10.,ended_unix=110.,wall_seconds=100.)]
    with pytest.raises(ValueError,match='shutdown'):
        ledger_seconds(attempts)


def test_resume_rejects_uncharged_shutdown_tail():
    from src.phase3_resume import ledger_seconds
    attempts=[dict(run_id='prior',status='failed',started_unix=10.,ended_unix=120.,
                   zero_containers_verified_unix=120.,wall_seconds=100.)]
    with pytest.raises(ValueError,match='shutdown'):
        ledger_seconds(attempts)
