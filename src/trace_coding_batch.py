"""Durable, budget-reserved Batch transport for src.trace_coding.

No network calls occur on import. Ledger reservations are never refunded, even
for failed jobs, so retries cannot silently exceed the lifetime $6 cap.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from src.trace_coding import (CATEGORIES, HARD_CAP_USD, MAX_OUTPUT_TOKENS, MODEL,
    canonical, digest, judge_body, token_cost, token_counts)

QUEUE_TOKEN_LIMIT = 10_000_000  # Below documented Luna Build tier's 20M queue.
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


def open_client(env_file: Path):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(env_file.expanduser(), override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY missing from environment/approved env file')
    # Creation has no automatic retries: a timed-out POST may already be paid.
    return OpenAI(max_retries=0, timeout=60)


def _config_hash():
    from src.trace_coding import INPUT_PER_MILLION, OUTPUT_PER_MILLION
    return digest([judge_body({'blinded': {}}), INPUT_PER_MILLION, OUTPUT_PER_MILLION])


def _save(directory: Path, state: dict):
    temporary = directory / 'state.json.tmp'
    with temporary.open('w') as f:
        f.write(canonical(state) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(directory / 'state.json')
    # Persist the rename before a paid request can be attempted.
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _ledger(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'ledger.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another trace coding process holds the ledger lock') from None
        path = directory / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'version': 1, 'config_hash': _config_hash(), 'responses': {}, 'jobs': [], 'reserved_usd': 0.0}
        if state.get('version') != 1 or state.get('config_hash') != _config_hash():
            raise ValueError('incompatible model/codebook/pricing in existing ledger; preserve lifetime budget')
        # Recheck budget rather than trusting a manually reduced aggregate.
        if abs(sum(j['reserved_usd'] for j in state['jobs']) - state['reserved_usd']) > 1e-9:
            raise ValueError('budget ledger mismatch')
        if state['reserved_usd'] > HARD_CAP_USD:
            raise ValueError('lifetime budget exceeds hard cap')
        yield state


def _safe_read(call):
    """Bounded backoff for read-only status/content/list and nonbilling uploads."""
    from openai import APIConnectionError, APITimeoutError
    for attempt in range(5):
        try:
            return call()
        except Exception as error:
            status = getattr(error, 'status_code', None)
            if attempt == 4 or (status not in (408, 409, 429) and not (status and status >= 500)
                                and not isinstance(error, (TimeoutError, ConnectionError, APIConnectionError, APITimeoutError))):
                raise RuntimeError('OpenAI request failed; response body suppressed') from None
            time.sleep(min(2 ** attempt, 16))


def _recover(directory: Path, state: dict, client):
    for job in state['jobs']:
        if job['status'] != 'submitting':
            continue
        # SDK iterates paginated list so recovery is not restricted to 100 jobs.
        matches = [b for b in _safe_read(lambda: list(client.batches.list(limit=100)))
                   if (b.metadata or {}).get('trace_job') == job['key']]
        if len(matches) != 1:
            raise ValueError('ambiguous Batch create; no unique metadata match. Preserve ledger, reconcile on OpenAI dashboard before retrying')
        job.update(status='running', batch_id=matches[0].id)
        _save(directory, state)


def _launch(directory: Path, state: dict, job: dict, client):
    if job['status'] != 'prepared':
        return False
    if not job.get('input_file_id'):
        with (directory / job['payload_file']).open('rb') as payload:
            def upload():
                payload.seek(0)
                return client.files.create(file=payload, purpose='batch')
            uploaded = _safe_read(upload)
        job['input_file_id'] = uploaded.id
        _save(directory, state)
    job['status'] = 'submitting'
    _save(directory, state)
    # Never automatically retry an uncertain creation. Recovery searches the
    # durable random metadata key; absent/duplicate matches fail closed.
    try:
        batch = client.batches.create(input_file_id=job['input_file_id'],
            endpoint='/v1/chat/completions', completion_window='24h', metadata={'trace_job': job['key']})
    except Exception as error:
        if getattr(error, 'status_code', None) in (400, 401, 403, 404, 413, 422, 429):
            # A definitive rejection did not create a paid batch; resume later.
            job['status'] = 'prepared'
            _save(directory, state)
        raise RuntimeError('Batch create failed/ambiguous; resume with the same ledger; payload suppressed') from None
    job.update(status='running', batch_id=batch.id)
    _save(directory, state)
    return True


def submit(traces: list[dict], directory: Path, client, *, batch_size: int = 250) -> dict:
    if type(batch_size) is not int or batch_size < 1 or batch_size > 1000:
        raise ValueError('batch size must be 1..1000')
    if not traces or len({t['trace_id'] for t in traces}) != len(traces):
        raise ValueError('empty or duplicate trace identities')
    submitted = 0
    with _ledger(directory) as state:
        for t in traces:
            previous = state['responses'].get(t['trace_id'])
            if previous and previous['payload_hash'] != t['payload_hash']:
                raise ValueError('response payload changed for cached identity')
        active = {i for j in state['jobs'] if j['status'] != 'done' for i in j['ids']}
        pending = [t for t in traces if t['trace_id'] not in active
                   and not state['responses'].get(t['trace_id'], {}).get('category')]
        bounds = {t['trace_id']: token_counts(t)[1] for t in pending}
        reserve = token_cost(sum(bounds.values()), len(pending) * MAX_OUTPUT_TOKENS)
        if state['reserved_usd'] + reserve > HARD_CAP_USD:
            raise ValueError('new work plus lifetime reservations exceeds $6 hard cap')
        if any(v > QUEUE_TOKEN_LIMIT for v in bounds.values()):
            raise ValueError('one trace exceeds queue token limit')
        _recover(directory, state, client)
        for job in state['jobs']:
            submitted += _launch(directory, state, job, client)
        queued = sum(j['input_upper_tokens'] for j in state['jobs'] if j['status'] != 'done')
        while pending:
            chunk, used = [], 0
            while pending and len(chunk) < batch_size:
                t = pending[0]
                if queued + used + bounds[t['trace_id']] > QUEUE_TOKEN_LIMIT:
                    break
                chunk.append(pending.pop(0))
                used += bounds[t['trace_id']]
            if not chunk:
                break
            key = uuid.uuid4().hex
            payload_file = f'batch-{key}.jsonl'
            with (directory / payload_file).open('w') as f:
                for trace in chunk:
                    f.write(canonical({'custom_id': trace['trace_id'], 'method': 'POST',
                        'url': '/v1/chat/completions', 'body': judge_body(trace)}) + '\n')
                f.flush()
                os.fsync(f.fileno())
            if (directory / payload_file).stat().st_size >= 190_000_000:
                raise ValueError('batch file exceeds conservative 190 MB upload limit')
            reservation = token_cost(used, len(chunk) * MAX_OUTPUT_TOKENS)
            job = {'key': key, 'status': 'prepared', 'ids': [t['trace_id'] for t in chunk],
                'payload_file': payload_file, 'input_upper_tokens': used, 'reserved_usd': reservation}
            state['jobs'].append(job)
            state['reserved_usd'] += reservation
            for t in chunk:
                state['responses'][t['trace_id']] = {'payload_hash': t['payload_hash'],
                    'reasoning_unclosed': t['reasoning_unclosed'], 'truncated': t['truncated']}
            _save(directory, state)
            submitted += _launch(directory, state, job, client)
            queued += used
        return {'submitted_batches': submitted, 'remaining_unsubmitted': len(pending),
                'n_cached_labels': sum(bool(r.get('category')) for r in state['responses'].values()),
                'reserved_usd': state['reserved_usd'], 'hard_cap_usd': HARD_CAP_USD}


def _parse_result(result: dict) -> tuple[str, dict | None]:
    identity = result.get('custom_id')
    response = result.get('response') or {}
    if result.get('error') or response.get('status_code') != 200:
        return identity, None
    try:
        body = response['body']
        choices = body['choices']
        choice = choices[0]
        message = choice['message']
        label = json.loads(message['content'])
        if (len(choices) != 1 or choice['finish_reason'] != 'stop' or message.get('refusal')
            or set(label) != {'category'} or label['category'] not in CATEGORIES):
            return identity, None
        usage = body['usage']
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens')):
            return identity, None
        return identity, {'category': label['category'], 'judge_model': body.get('model', MODEL),
            'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens']}
    except (KeyError, TypeError, ValueError, IndexError):
        return identity, None


def collect_results(directory: Path, client, output: Path) -> dict:
    with _ledger(directory) as state:
        _recover(directory, state, client)
        for job in state['jobs']:
            if job['status'] != 'running':
                continue
            batch = _safe_read(lambda: client.batches.retrieve(job['batch_id']))
            if batch.status not in TERMINAL:
                continue
            seen, parsed = set(), {}
            for file_id in (batch.output_file_id, batch.error_file_id):
                if not file_id:
                    continue
                content = _safe_read(lambda: client.files.content(file_id)).text
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    identity, label = _parse_result(json.loads(line))
                    if identity not in job['ids'] or identity in seen:
                        raise ValueError('Batch output has unknown/duplicate response identity')
                    seen.add(identity)
                    if label:
                        parsed[identity] = label
            for identity, label in parsed.items():
                previous = state['responses'][identity]
                if previous.get('category') and previous['category'] != label['category']:
                    raise ValueError('conflicting cached labels')
                previous.update(label)
            job.update(status='done', remote_status=batch.status,
                       failed_responses=len(job['ids']) - len(parsed))
            _save(directory, state)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w') as f:
            for identity, response in sorted(state['responses'].items()):
                if response.get('category'):
                    f.write(canonical({'trace_id': identity, **response}) + '\n')
        return {'n_labels': sum(bool(r.get('category')) for r in state['responses'].values()),
            'active_batches': sum(j['status'] != 'done' for j in state['jobs']),
            'failed_responses': sum(j.get('failed_responses', 0) for j in state['jobs']),
            'reserved_usd': state['reserved_usd']}


def reconcile(directory: Path, client, job_key: str, *, confirm_no_remote_batch: bool = False) -> dict:
    """Operator escape hatch for a crash before POST, with no budget refund.

    Only use after confirming on the provider dashboard that no accepted batch
    exists. A new attempt reserves its full cost again, covering even a late
    orphan. Routine uncertain POSTs recover automatically from metadata instead.
    """
    if not confirm_no_remote_batch:
        raise ValueError('explicit no-remote-batch confirmation is required')
    with _ledger(directory) as state:
        matches = [j for j in state['jobs'] if j['key'] == job_key and j['status'] == 'submitting']
        if len(matches) != 1:
            raise ValueError('job key must identify one unresolved submitting job')
        if any((b.metadata or {}).get('trace_job') == job_key
               for b in _safe_read(lambda: list(client.batches.list(limit=100)))):
            raise ValueError('remote job exists; use submit/collect to recover it instead')
        matches[0].update(status='done', remote_status='operator_confirmed_no_remote_batch',
                          failed_responses=len(matches[0]['ids']))
        _save(directory, state)
        return {'reconciled_job': job_key, 'reserved_usd': state['reserved_usd'], 'reservation_refunded': False}
