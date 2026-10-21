"""CPU-only, fail-closed integration of a completed optional Phase 3 continuation.

``validate`` returns verified source, pilot and test data without writing files.
``integrate`` additionally runs the registered 2,000-resample analysis and writes
four new derived artifacts to an unoccupied output directory. Neither function
changes raw artifacts, the lifetime ledger, original metadata, doses or labels.
The original validator deliberately remains unchanged. This entry point certifies
only the continuation schema emitted by phase3_resume/modal_app_3_resume.

CLI: uv run python -m src.phase3_resume_analysis --source-run-id ORIGINAL \
    --resume-run-id ALLOCATION --output-dir results/phase3/continued

No Modal imports, tokenizer/model loads, network requests or GPU work occur.
Prompt/token provenance uses the frozen runner and recorded hashes/counts; CPU
validation independently rescores text but does not retokenize model outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from src.analyze_3 import _validate_generations, analyze, render_summary, trajectory_table
from src.models import MODELS
from src.phase3 import CAP_SECONDS, CELLS, MODEL, RATE, score_generation
from src.phase3_resume import (CLOSED, FROZEN_PATHS, SAMPLING, TIMING_METHOD,
    WORK_RESERVE, allocation_limits, digest, exact_batch_capacity, file_hash,
    ledger_seconds, load_source_package, make_test_batches, number, project_test_plan,
    require, safe_run_id, validate_no_containers, validate_payload, verify_hashes)
from src.phase3_validation import SOURCE_FIELDS, _validate_batches

CONTINUATION_PATHS = ('src/phase3_resume.py', 'src/modal_app_3_resume.py',
                      'src/phase3_validation.py', 'src/analyze_3.py')
REMOTE_ARTIFACTS = {'resume-manifest.json', 'source-manifest.json',
    'source-selection.json', 'neurons.json', 'selection.json', 'metrics.json',
    'planning-pilot.jsonl', 'generations.jsonl'}
OUTPUT_NAMES = ('summary.json', 'summary.md', 'trajectories.csv', 'integration.json')


def _equal(actual, expected, context):
    # JSON equality retains boolean/numeric types, unlike Python's True == 1.
    require(digest(actual) == digest(expected), context)


def _close(actual, expected, context, *, tolerance=1e-6):
    require(number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance), context)


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _json(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'duplicate JSON key in {path}: {key}')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'nonfinite JSON in {path}: {value}')
    return json.loads(Path(path).read_text(), object_pairs_hook=object_pairs,
                      parse_constant=invalid_constant)


def _object(path):
    value = _json(path)
    require(isinstance(value, dict), f'{path}: JSON object required')
    return value


def _rows(path):
    # The scorer reader permits ordinary JSON; here require objects, no duplicate
    # keys or nonfinite constants, and no silent blank/incomplete records.
    rows = []
    for index, line in enumerate(Path(path).read_text().splitlines(), 1):
        require(bool(line.strip()), f'{path}:{index}: empty JSONL record')
        def pairs(values):
            result = {}
            for key, value in values:
                require(key not in result, f'{path}:{index}: duplicate JSON key')
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError(f'{path}:{index}: nonfinite JSON: {value}')
        row = json.loads(line, object_pairs_hook=pairs, parse_constant=invalid_constant)
        require(isinstance(row, dict), f'{path}:{index}: JSON object required')
        rows.append(row)
    require(bool(rows), f'{path}: nonempty JSONL required')
    return rows


def _lineage(attempts, source_id, resume_id):
    require(isinstance(attempts, list) and len(attempts) >= 2, 'full lifetime ledger required')
    seen = set()
    last_end = None
    for attempt in attempts:
        require(isinstance(attempt, dict) and attempt.get('status') in CLOSED,
                'running/unresolved lifetime allocation')
        rid = attempt.get('run_id')
        require(isinstance(rid, str) and rid not in seen, 'duplicate/invalid lifetime allocation ID')
        seen.add(rid)
        start, end, zero, wall = (attempt.get(k) for k in
            ('started_unix', 'ended_unix', 'zero_containers_verified_unix', 'wall_seconds'))
        require(number(start, positive=True) and number(end, positive=True)
                and number(zero, positive=True) and number(wall), 'closed ledger shutdown evidence required')
        require(start <= zero <= end+.001 and start <= end and wall+.001 >= zero-start,
                'allocation shutdown tail is not fully charged')
        require(last_end is None or start >= last_end, 'overlapping/out-of-order lifetime allocations')
        last_end = end
    linked = [a for a in attempts if a.get('source_run_id') == source_id]
    require(len(linked) == 1 and linked[0]['run_id'] == resume_id,
            'exactly one continuation for source required')
    require(attempts[-1] is linked[0], 'later allocations prevent continuation certification')
    attempt, prefix = attempts[-1], attempts[:-1]
    require(attempt['status'] == 'completed', 'continuation ledger must be completed')
    require(any(a['run_id'] == source_id for a in prefix) and source_id != resume_id,
            'source allocation absent from ledger prefix')
    previous = ledger_seconds(prefix)
    _close(attempt.get('previous_seconds'), previous, 'ledger previous_seconds differs from all retained prior charges')
    total = previous+attempt['wall_seconds']
    require(total <= CAP_SECONDS, 'lifetime GPU cap exceeds $7')
    require(attempt.get('gpu') == 'L40S', 'continuation allocation GPU mismatch')
    _close(attempt.get('rate_usd_per_hour'), RATE*3600, 'ledger GPU rate mismatch')
    require(isinstance(attempt.get('call_id'), str) and bool(attempt['call_id']), 'known continuation call required')
    require(attempt.get('accounting_basis') == 'allocation wall time through verified shutdown',
            'authoritative closed wall accounting required')
    _close(attempt.get('observed_wall_seconds'), attempt['wall_seconds'], 'reserved/unresolved allocation exposure')
    limits = allocation_limits(previous, attempt['started_unix'])
    for key, value in limits.items():
        _close(attempt.get(key), value, f'allocation limit mismatch: {key}')
    return prefix, attempt, previous, total


def _artifacts(root, raw, result):
    hashes = result.get('artifacts_sha256')
    require(isinstance(hashes, dict) and REMOTE_ARTIFACTS <= hashes.keys(), 'missing continuation artifact hashes')
    require('resume-result.json' not in hashes, 'self-referential artifact hash')
    for name in hashes:
        require(isinstance(name, str) and Path(name).name == name and name not in ('.','..'),
                'unsafe continuation artifact name')
    verify_hashes(raw, hashes)
    metadata = {name: _object(raw/(name+'.json')) for name in
        ('metrics','selection','source-manifest','source-selection','neurons','resume-manifest','launch-manifest',
         'local-execution','containers-after')}
    metrics = metadata['metrics']
    _equal(metrics, {k:v for k,v in result.items() if k not in ('artifacts_sha256','accounting')},
           'metrics.json and resume-result.json disagree')
    _equal(metadata['resume-manifest'], metadata['launch-manifest'], 'launch/resume manifest mismatch')
    _equal(result.get('resume_manifest'), metadata['resume-manifest'], 'result resume manifest mismatch')
    _equal(result.get('source_manifest'), metadata['source-manifest'], 'result source manifest mismatch')
    _equal(result.get('source_selection'), metadata['source-selection'], 'result source selection mismatch')
    _equal(result.get('selection'), metadata['selection'], 'result selection mismatch')
    manifest = metadata['resume-manifest']
    versions = manifest.get('continuation_sha256')
    require(isinstance(versions, dict) and set(versions) == set(CONTINUATION_PATHS),
            'all four continuation version hashes required')
    verify_hashes(root, versions)
    return metadata


def _provenance(root, package, metadata, result, source_id, resume_id):
    validate_payload(package)
    require(package['source_metrics'].get('status') == 'failed',
            'original source must have failed at the explicit pretest projection gate')
    require(set(FROZEN_PATHS) <= package['source_manifest']['source_sha256'].keys(), 'eleven frozen hashes required')
    verify_hashes(root, package['source_manifest']['source_sha256'])
    verify_hashes(root, package['source_artifacts_sha256'])
    _equal(metadata['source-manifest'], package['source_manifest'], 'original source manifest changed')
    _equal(metadata['source-selection'], package['source_selection'], 'original c_star/dose selection changed')
    _equal(metadata['neurons'], package['neurons'], 'original neurons changed')
    manifest = metadata['resume-manifest']
    for key, expected in dict(schema_version=1, source_run_id=source_id, allocation_run_id=resume_id,
            source_manifest=package['source_manifest'], source_artifacts_sha256=package['source_artifacts_sha256'],
            payload_sha256=package['payload_sha256'], timing_method=TIMING_METHOD,
            rate_usd_per_gpu_second=RATE, cap_usd=7., parent_validation_required=True).items():
        _equal(manifest.get(key), expected, f'rederived resume manifest mismatch: {key}')
    for key, expected in dict(schema_version=1, run_id=source_id, allocation_run_id=resume_id,
            status='completed', stage='finished', model=MODEL, revision=MODELS[MODEL], gpu='L40S',
            sampling=SAMPLING, timing_method=TIMING_METHOD, rate_usd_per_gpu_second=RATE, cap_usd=7.,
            test_generation_started=True, parent_validation_required=True, work_reserve_bytes=WORK_RESERVE,
            capacity_rounding='exact positive integer, capped at 64',
            oom_policy='fail closed, no split/rebatch after timing pilot',
            greedy_cache_parity_passed=True, hook_math_runtime_passed=True,
            original_rendered_prompt_hashes_verified=300).items():
        _equal(result.get(key), expected, f'continuation metadata mismatch: {key}')
    require(not result.get('incomplete_batches') and not result.get('budget_terminated')
            and not result.get('planning_refusal') and not result.get('stop_reason'), 'incomplete/budget-terminated continuation')
    eos = result.get('eos_token_ids')
    require(isinstance(eos,list) and len(eos)==2 and all(type(t) is int for t in eos)
            and set(eos)=={151643,151645}, 'both pinned EOS IDs required')
    versions = result.get('versions')
    require(isinstance(versions,dict) and {'torch','transformers','numpy','accelerate','modal'} <= versions.keys()
            and all(isinstance(v,str) and bool(v.strip()) for v in versions.values())
            and versions['modal']=='1.6.2', 'runtime versions required')
    require(isinstance(result.get('cuda_device'),str) and 'L40S' in result['cuda_device'], 'recorded device is not L40S')


def _row(row, source, package, source_id, resume_id, cell, c, *, pilot):
    require(isinstance(row,dict) and source is not None, 'unknown generation source request')
    for key in SOURCE_FIELDS:
        require(key in row and type(row[key]) is type(source[key]) and row[key]==source[key],
                f'generation source metadata mismatch: {key}')
    require((row.get('run_id'),row.get('allocation_run_id'),row.get('model'),row.get('model_revision'))
            == (source_id,resume_id,MODEL,MODELS[MODEL]), 'generation logical run/allocation/model mismatch')
    require(row.get('generation_cell')==cell and row.get('intervention')==cell
            and number(row.get('c')) and row['c']==c
            and row.get('id')==f'{source["id"]}:{cell}:c{c:g}', 'generation fixed cell/dose/id mismatch')
    require(row.get('rendered_prompt_sha256')==package['prompt_hashes'][source['id']],
            'generation source prompt hash mismatch')
    require(row.get('budget_terminated') is False and _integer(row.get('batch_seed')),
            'generation budget shortening/seed missing')
    require(_integer(row.get('output_tokens')) and row['output_tokens']<=3072
            and _integer(row.get('final_reply_tokens')) and row['final_reply_tokens']<=row['output_tokens'],
            'invalid generation token counts')
    finish, end = row.get('finish_reason'),row.get('end_token_id')
    require((finish=='length' and end is None and row['output_tokens']==3072)
            or (finish=='stop' and type(end) is int and end in (151643,151645)), 'invalid finish/EOS or shortened output')
    require(isinstance(row.get('response'),str), 'generation response must be text')
    for key,value in score_generation(row['response'],source['gold'],finish).items():
        require(key in row and type(row[key]) is type(value) and row[key]==value,
                f'independent generation score mismatch: {key}')
    if pilot:
        require(row.get('planning_only') is True and row.get('analysis_excluded') is True
                and row.get('pilot_kind')==TIMING_METHOD, 'planning pilot exclusion/provenance required')
    else:
        require(not {'planning_only','analysis_excluded','pilot_kind'} & row.keys(),
                'planning pilot/excluded data cannot enter test study')


def _batch_totals(record, rows, max_prompt):
    require(isinstance(record,dict) and record.get('budget_terminated') is False,
            'incomplete/budget-shortened batch')
    require(_integer(record.get('prompt_width'),positive=True) and record['prompt_width']<=max_prompt,
            'batch prompt exceeds frozen capacity width')
    for key,value in dict(output_tokens=sum(r['output_tokens'] for r in rows),
                          max_output_tokens=max(r['output_tokens'] for r in rows),
                          final_reply_tokens=sum(r['final_reply_tokens'] for r in rows)).items():
        require(type(record.get(key)) is int and record[key]==value, f'batch token total mismatch: {key}')


def _plan_and_rows(package, result, pilots, tests, source_id, resume_id, previous):
    require(result.get('kv_bytes_per_token')==163840 and type(result['kv_bytes_per_token']) is int,
            'pinned 14B KV bytes must be 163840')
    width = result.get('max_prompt_tokens')
    batch_size = exact_batch_capacity(result.get('capacity_free_bytes'),163840,width)
    require(type(result.get('batch_size')) is int and result['batch_size']==batch_size, 'exact safe batch capacity mismatch')
    require(_integer(result.get('capacity_total_bytes'),positive=True)
            and result['capacity_free_bytes']<=result['capacity_total_bytes'], 'invalid device capacity record')
    pilot_sources = package['dev_requests'][:batch_size]
    require(len(pilots)==batch_size, 'planning pilot must contain first B development template-0 requests')
    for row,source in zip(pilots,pilot_sources):
        _row(row,source,package,source_id,resume_id,'baseline',0.,pilot=True)
        require(row['batch_seed']==pilot_sources[0]['seed'], 'planning pilot first Phase 1b seed mismatch')
    pilot = result.get('planning_pilot')
    _validate_batches(pilots,[pilot],{r['id']:r for r in pilot_sources})
    _batch_totals(pilot,pilots,width)
    require(pilot.get('split')=='dev' and pilot.get('cell')=='baseline' and pilot.get('c')==0
            and pilot.get('n')==batch_size, 'pilot must be full-capacity baseline development only')
    selection = result.get('selection')
    require(isinstance(selection,dict), 'continuation selection required')
    spent = selection.get('spent_gpu_seconds_at_test_gate')
    setup,startup = result.get('setup_seconds'),result.get('startup_elapsed_seconds')
    require(number(setup) and number(startup) and number(spent)
            and spent+1e-6>=previous+startup+setup+pilot['seconds'],
            'test gate spent omits prior/startup/setup/pilot charges')
    plan = project_test_plan(pilot['seconds'],batch_size,spent)
    expected_selection = {**package['source_selection'],**plan,'test_generation_started':True,
                          'pilot_rows_excluded_from_study':batch_size}
    _equal(selection,expected_selection,'continuation c_star/dose/2-then-1 projection selection mismatch')
    expected = {r['id']:r for r in package['test_requests']}
    require(len(tests)==100*plan['test_templates']*7, 'incomplete seven-cell test grid')
    for row in tests:
        cell=row.get('generation_cell')
        require(cell in CELLS and row.get('split')=='test', 'only seven fixed test cells may enter study')
        _row(row,expected.get(row.get('phase1b_id')),package,source_id,resume_id,cell,
             0 if cell=='baseline' else selection['c_star'],pilot=False)
    _validate_generations(tests,plan['test_templates'])
    batches=result.get('batches')
    _validate_batches(tests,batches,expected)
    frozen=make_test_batches(package['test_requests'],plan['test_templates'],batch_size)
    frozen_ids=[[r['id'] for r in b] for b in frozen]
    _equal(result.get('test_batch_source_ids'),frozen_ids,'frozen test batch source IDs changed')
    require(len(batches)==7*len(frozen), 'missing/rebatched seven-cell batches')
    indexed={(r['generation_cell'],r['phase1b_id']):r for r in tests}
    for record,(cell,index) in zip(batches,((cell,index) for cell in CELLS for index in range(len(frozen)))):
        source_batch=frozen[index]
        require(record.get('cell')==cell and type(record.get('batch_index')) is int
                and record['batch_index']==index and record.get('split')=='test'
                and record.get('c')==(0 if cell=='baseline' else selection['c_star'])
                and record.get('n')==len(source_batch) and record.get('seed')==source_batch[0]['seed']
                and record.get('first_id')==source_batch[0]['id'] and record.get('last_id')==source_batch[-1]['id'],
                'test batch boundaries/index/seed differ from frozen seven-cell plan')
        _batch_totals(record,[indexed[cell,r['id']] for r in source_batch],width)
    require(not set(r['id'] for r in pilots) & set(r['id'] for r in tests), 'pilot leaked into inference')
    return batch_size,plan


def _accounting(result, metadata, attempt, previous, total):
    local = metadata['local-execution']
    _equal(local.get('attempt'),attempt,'local execution attempt differs from lifetime ledger')
    require(local.get('recovery_errors')==[] and local.get('parent_validation_required') is True,
            'local artifact recovery unresolved')
    _close(local.get('allocation_gpu_usd_upper'),total*RATE,'local lifetime GPU cost mismatch')
    evidence=metadata['containers-after']
    validate_no_containers(evidence.get('containers'))
    checked=evidence.get('checked_unix')
    require(number(checked,positive=True), 'shutdown evidence time missing')
    _close(checked,attempt['zero_containers_verified_unix'],'zero evidence does not match ledger')
    require(attempt['started_unix']<=checked<=attempt['ended_unix']+.001,
            'shutdown evidence precedes allocation startup or follows closure')
    for key,expected in dict(previous_seconds=previous,allocation_started_unix=attempt['started_unix'],
                             absolute_deadline_unix=attempt['absolute_deadline_unix']).items():
        _close(result.get(key),expected,f'continuation clock/accounting mismatch: {key}')
    accounting=result.get('accounting')
    require(isinstance(accounting,dict), 'remote return accounting required')
    for key,expected in dict(previous_seconds=previous, startup_elapsed_seconds=result['startup_elapsed_seconds'],
                             rate_usd_per_gpu_second=RATE,cap_usd=7.).items():
        _close(accounting.get(key),expected,f'remote accounting mismatch: {key}')
    require(accounting.get('final_allocation_wall_time')=='local lifetime ledger authoritative',
            'local shutdown ledger must remain authoritative')
    function=result.get('function_seconds')
    returned=accounting.get('function_seconds')
    require(number(function) and number(returned) and number(attempt.get('function_seconds'))
            and function<=returned<=attempt['function_seconds']+.001, 'remote function accounting order mismatch')
    _close(accounting.get('lifetime_gpu_seconds_at_return'),previous+result['startup_elapsed_seconds']+returned,
           'remote return lifetime accounting mismatch',tolerance=.001)
    _close(result.get('lifetime_gpu_seconds_at_checkpoint'),previous+result['startup_elapsed_seconds']+function,
           'remote checkpoint lifetime accounting mismatch',tolerance=.001)
    _close(result.get('estimated_gpu_usd'),result['lifetime_gpu_seconds_at_checkpoint']*RATE,
           'remote checkpoint cost mismatch')
    require(function+1e-6>=result['setup_seconds']+result['planning_pilot']['seconds']
            +sum(b['seconds'] for b in result['batches']), 'function omits measured setup/pilot/test time')
    require(result['selection']['spent_gpu_seconds_at_test_gate']<=result['lifetime_gpu_seconds_at_checkpoint']+.001
            and accounting['lifetime_gpu_seconds_at_return']<=total+.001
            and checked+.001>=attempt['started_unix']+result['startup_elapsed_seconds']+returned,
            'gate/return/shutdown elapsed accounting inconsistent')


def _validate(root, source_run_id, resume_run_id):
    root=Path(root).resolve()
    safe_run_id(source_run_id); safe_run_id(resume_run_id)
    raw=(root/'results/raw'/resume_run_id).resolve()
    require(raw.is_relative_to(root/'results/raw') and not raw.is_relative_to(root/'results/raw'/source_run_id),
            'continuation raw path must be separate and inside project')
    ledger=root/'results/phase3/gpu-attempts.json'
    attempts=_json(ledger)
    prefix,attempt,previous,total=_lineage(attempts,source_run_id,resume_run_id)
    result=_object(raw/'resume-result.json')
    metadata=_artifacts(root,raw,result)
    # Passing the *full* ledger to this loader would intentionally trigger its
    # single-use guard. Independently certify full lineage above, load the prefix.
    package=load_source_package(root,source_run_id,prefix)
    _provenance(root,package,metadata,result,source_run_id,resume_run_id)
    pilots=_rows(raw/'planning-pilot.jsonl'); tests=_rows(raw/'generations.jsonl')
    batch_size,plan=_plan_and_rows(package,result,pilots,tests,source_run_id,resume_run_id,previous)
    require(type(result.get('output_rows')) is int and result['output_rows']==len(tests), 'test output row count mismatch')
    _accounting(result,metadata,attempt,previous,total)
    source_files=package['source_artifacts_sha256']
    def source_path(name):
        matches=[root/p for p in source_files if Path(p).name==name]
        require(len(matches)==1,f'original {name} path ambiguous')
        return matches[0]
    tracks=_rows(source_path('tracking.jsonl')); dev=_rows(source_path('generations.jsonl'))
    require(not any({'planning_only','analysis_excluded','pilot_kind'} & r.keys() for r in dev),
            'original development generations contain planning pilot data')
    verification=dict(completed=True,h14_missing=False,n_tracking_rows=len(tracks),
        n_source_labels_verified=400,n_source_files_verified=len(package['source_manifest']['source_sha256']),
        n_trajectories_verified=400,n_dev_rows=len(dev),n_test_rows=len(tests),
        planning_pilot_rows_excluded=len(pilots),batch_size=batch_size,test_templates=plan['test_templates'],
        grid_counts={cell:sum(r['generation_cell']==cell for r in tests) for cell in CELLS},
        source_artifacts_sha256=source_files,source_sha256=package['source_manifest']['source_sha256'],
        previous_gpu_seconds=previous,lifetime_gpu_seconds=total,lifetime_gpu_usd=total*RATE,
        gpu_cap_usd=7.,zero_containers_verified_unix=attempt['zero_containers_verified_unix'],
        validated_allocation_lineage=[source_run_id,resume_run_id],n_lifetime_attempts=len(attempts),
        timing_method=TIMING_METHOD)
    # Hash local-only shutdown/launch/ledger evidence too, without altering it.
    evidence_paths=[ledger,raw/'resume-result.json',raw/'local-execution.json',raw/'containers-after.json',
                    raw/'launch-manifest.json']
    evidence={str(p.relative_to(root)):file_hash(p) for p in evidence_paths}
    evidence.update({str((raw/n).relative_to(root)):h for n,h in result['artifacts_sha256'].items()})
    return dict(source_package=package,tracking_rows=tracks,development_rows=dev,test_rows=tests,
        planning_rows=pilots,resume_result=result,selection=metadata['selection'],neurons=metadata['neurons'],
        execution_verification=verification,evidence_sha256=evidence)


def validate(*, source_run_id, resume_run_id, root=None):
    """Verify immutable artifacts and a closed continuation; no writes or inference."""
    if root is None:
        from src.build_prompts import ROOT
        root=ROOT
    try:
        return _validate(root,source_run_id,resume_run_id)
    except (KeyError,TypeError,IndexError,AttributeError,OSError) as error:
        raise ValueError(f'invalid/incomplete continuation artifacts: {error}') from error


def integrate(*, source_run_id, resume_run_id, output_dir='results/phase3', root=None):
    """Validate, analyze the joined logical study and write only new derived files.

    Existing output names are refused, including tracking-only summaries. Select
    a fresh output directory to preserve a previous partial report. Relative output
    paths are resolved under root. Raw directories are never output targets.
    """
    if root is None:
        from src.build_prompts import ROOT
        root=ROOT
    root=Path(root).resolve()
    output=Path(output_dir)
    output=(output if output.is_absolute() else root/output).resolve()
    require(not output.is_relative_to((root/'results/raw').resolve()), 'cannot write integration into raw artifacts')
    require(all(not (output/name).exists() for name in OUTPUT_NAMES), 'refuse to overwrite existing integration output')
    validated=validate(root=root,source_run_id=source_run_id,resume_run_id=resume_run_id)
    package=validated['source_package']
    protected={**package['source_artifacts_sha256'],**package['source_manifest']['source_sha256'],
               **validated['evidence_sha256'],
               **validated['resume_result']['resume_manifest']['continuation_sha256']}
    require(not {(output/n).resolve() for n in OUTPUT_NAMES} & {(root/n).resolve() for n in protected},
            'integration target overlaps immutable evidence')
    report=analyze(validated['tracking_rows'],validated['development_rows']+validated['test_rows'],
                   expected_templates=validated['selection']['test_templates'],resamples=2000,seed=0)
    report.update(study_kind='continued_logical_study',source_run_id=source_run_id,allocation_run_id=resume_run_id,
        source_manifest=package['source_manifest'],source_metrics=package['source_metrics'],
        source_selection=package['source_selection'],resume_result=validated['resume_result'],
        selection=validated['selection'],neurons=validated['neurons'],
        execution_verification=validated['execution_verification'])
    integration=dict(schema_version=1,study_kind='continued_logical_study',source_run_id=source_run_id,
        allocation_run_id=resume_run_id,source_manifest=package['source_manifest'],
        source_artifacts_sha256=package['source_artifacts_sha256'],payload_sha256=package['payload_sha256'],
        continuation_sha256=validated['resume_result']['resume_manifest']['continuation_sha256'],
        integration_module_sha256=file_hash(Path(__file__)),evidence_sha256=validated['evidence_sha256'],
        execution_verification=validated['execution_verification'],planning_pilot_inference_included=False,
        original_source_status=package['source_metrics']['status'],bootstrap=report['bootstrap'])
    # Recheck read evidence immediately before output, avoiding a changed ledger
    # being silently certified while CPU bootstrap was running.
    verify_hashes(root,protected)
    output.mkdir(parents=True,exist_ok=True)
    for name in OUTPUT_NAMES:
        require(not (output/name).exists(), 'integration output appeared during validation')
    with (output/'summary.json').open('x') as handle:
        handle.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'summary.md').open('x') as handle:
        handle.write(render_summary(report))
    trajectories=trajectory_table(validated['tracking_rows'])
    with (output/'trajectories.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(trajectories[0]))
        writer.writeheader(); writer.writerows(trajectories)
    integration['output_sha256']={n:file_hash(output/n) for n in OUTPUT_NAMES if n!='integration.json'}
    with (output/'integration.json').open('x') as handle:
        handle.write(json.dumps(integration,indent=2,allow_nan=False)+'\n')
    return report


def main():
    from src.build_prompts import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-id',required=True)
    parser.add_argument('--resume-run-id',required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results/phase3'))
    parser.add_argument('--root',type=Path,default=ROOT,help='Project root (primarily synthetic validation)')
    args=parser.parse_args()
    try:
        report=integrate(root=args.root,source_run_id=args.source_run_id,resume_run_id=args.resume_run_id,
                         output_dir=args.output_dir)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    print(f"Phase 3 continuation: {report['n_tracking_rows']} tracking rows; "
          f"{report['n_generation_rows']} test rows -> {args.output_dir}")


if __name__=='__main__':
    main()
