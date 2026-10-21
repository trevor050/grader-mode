"""Fake Batch API transport; synthetic requests only."""
import importlib
import json
from types import SimpleNamespace

import pytest

from test_trace_coding import raw
from src.trace_coding import extract_trace


def module():
    return importlib.import_module('src.trace_coding_batch')


class FakeFiles:
    def __init__(self):
        self.uploads, self.outputs = [], {}

    def create(self, *, file, purpose):
        assert purpose == 'batch'
        self.uploads.append(file.read().decode())
        return SimpleNamespace(id=f'file-{len(self.uploads)}')

    def content(self, identity):
        return SimpleNamespace(text=self.outputs[identity])


class FakeBatches:
    def __init__(self, client):
        self.client, self.items, self.creates, self.raise_after_create = client, {}, 0, False

    def create(self, **kwargs):
        self.creates += 1
        batch = SimpleNamespace(id=f'batch-{self.creates}', status='in_progress', metadata=kwargs['metadata'],
            output_file_id=None, error_file_id=None)
        self.items[batch.id] = batch
        if self.raise_after_create:
            raise TimeoutError('ambiguous create result')
        return batch

    def list(self, **kwargs):
        return list(self.items.values())

    def retrieve(self, identity):
        return self.items[identity]


class FakeClient:
    def __init__(self):
        self.files = FakeFiles()
        self.batches = FakeBatches(self)


def complete(client, batch_id, *, invalid=False):
    batch = client.batches.items[batch_id]
    batch.status = 'completed'
    batch.output_file_id = 'out-' + batch_id
    offset = int(batch_id.split('-')[1])-1
    requests = [json.loads(x) for x in client.files.uploads[offset].splitlines()]
    client.files.outputs[batch.output_file_id] = '\n'.join(json.dumps({
        'custom_id': r['custom_id'], 'error': None, 'response': {'status_code': 200, 'body': {
        'model': 'gpt-6-luna', 'usage': {'prompt_tokens': 100, 'completion_tokens': 8},
        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'category': 'bad' if invalid else '(a)'})}}]}}}) for r in requests)


def test_submit_collect_resubmit_is_idempotent_and_payload_blinded(tmp_path):
    m, client = module(), FakeClient()
    traces = [extract_trace(raw(i)) for i in range(3)]
    result = m.submit(traces, tmp_path, client, batch_size=2)
    assert result['submitted_batches'] == 2
    assert client.batches.creates == 2
    for payload in client.files.uploads:
        for forbidden in ('FINAL SECRET', 'synthetic-', 'RLVR', 'Olmo'):
            assert forbidden not in payload
    assert m.submit(traces, tmp_path, client, batch_size=2)['submitted_batches'] == 0
    complete(client, 'batch-1'); complete(client, 'batch-2')
    output = tmp_path/'labels.jsonl'
    result = m.collect_results(tmp_path, client, output)
    assert result['n_labels'] == 3
    assert len(output.read_text().splitlines()) == 3
    assert m.submit(traces, tmp_path, client)['submitted_batches'] == 0
    first = output.read_bytes()
    m.collect_results(tmp_path, client, output)
    assert output.read_bytes() == first


def test_changed_payload_same_response_id_rejected(tmp_path):
    m, client = module(), FakeClient()
    m.submit([extract_trace(raw())], tmp_path, client)
    with pytest.raises(ValueError, match='changed'):
        m.submit([extract_trace(raw(response='Changed</think>final'))], tmp_path, client)
    assert client.batches.creates == 1


def test_lifetime_reservations_enforce_hard_cap_before_network(tmp_path, monkeypatch):
    m, client = module(), FakeClient()
    monkeypatch.setattr(m, 'HARD_CAP_USD', .000001)
    with pytest.raises(ValueError, match='cap'):
        m.submit([extract_trace(raw())], tmp_path, client)
    assert not client.files.uploads and client.batches.creates == 0


def test_ambiguous_batch_creation_recovers_by_metadata_without_duplicate(tmp_path):
    m, client = module(), FakeClient()
    client.batches.raise_after_create = True
    traces = [extract_trace(raw())]
    with pytest.raises(RuntimeError, match='ambiguous|failed'):
        m.submit(traces, tmp_path, client)
    client.batches.raise_after_create = False
    m.submit(traces, tmp_path, client)
    assert client.batches.creates == 1


def test_ambiguous_unfindable_creation_fails_closed(tmp_path):
    m, client = module(), FakeClient()
    client.batches.raise_after_create = True
    traces = [extract_trace(raw())]
    with pytest.raises(RuntimeError):
        m.submit(traces, tmp_path, client)
    client.batches.items.clear()
    client.batches.raise_after_create = False
    with pytest.raises(ValueError, match='ambiguous'):
        m.submit(traces, tmp_path, client)
    assert client.batches.creates == 1


def test_invalid_labels_not_cached_retry_consumes_new_reservation(tmp_path):
    m, client = module(), FakeClient()
    traces = [extract_trace(raw())]
    first = m.submit(traces, tmp_path, client)['reserved_usd']
    complete(client, 'batch-1', invalid=True)
    result = m.collect_results(tmp_path, client, tmp_path/'labels.jsonl')
    assert result['n_labels'] == 0 and result['failed_responses'] == 1
    second = m.submit(traces, tmp_path, client)['reserved_usd']
    assert second == pytest.approx(2*first)


def test_queue_limit_leaves_remaining_work_for_next_resume(tmp_path, monkeypatch):
    m, client = module(), FakeClient()
    monkeypatch.setattr(m, 'QUEUE_TOKEN_LIMIT', 1)
    with pytest.raises(ValueError, match='queue'):
        m.submit([extract_trace(raw())], tmp_path, client)
    assert client.batches.creates == 0


def test_upload_retry_rewinds_payload(tmp_path, monkeypatch):
    m, client = module(), FakeClient()
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    original = client.files.create
    attempts = 0
    def upload(*, file, purpose):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            file.read()
            error = RuntimeError('rate limit')
            error.status_code = 429
            raise error
        return original(file=file, purpose=purpose)
    client.files.create = upload
    m.submit([extract_trace(raw())], tmp_path, client)
    assert len(client.files.uploads[0].splitlines()) == 1


def test_sdk_connection_errors_receive_bounded_backoff(monkeypatch):
    import httpx
    import openai
    m = module()
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    for error in (openai.APIConnectionError(request=httpx.Request('GET', 'https://api.openai.com/v1/batches')),
                  openai.APITimeoutError(request=httpx.Request('GET', 'https://api.openai.com/v1/batches'))):
        calls = []
        def fetch():
            calls.append(1)
            if len(calls) < 3:
                raise error
            return 'ok'
        assert m._safe_read(fetch) == 'ok'
        assert len(calls) == 3


def test_explicit_reconciliation_keeps_reservation_and_allows_resume(tmp_path):
    m, client = module(), FakeClient()
    client.batches.raise_after_create = True
    traces = [extract_trace(raw())]
    with pytest.raises(RuntimeError):
        m.submit(traces, tmp_path, client)
    state = json.loads((tmp_path/'state.json').read_text())
    job_key, reserved = state['jobs'][0]['key'], state['reserved_usd']
    client.batches.items.clear()  # Simulate confirmed no accepted remote request.
    client.batches.raise_after_create = False
    m.reconcile(tmp_path, client, job_key, confirm_no_remote_batch=True)
    after = m.submit(traces, tmp_path, client)
    assert after['reserved_usd'] == pytest.approx(2*reserved)
    assert client.batches.creates == 2


def test_reconciliation_refuses_to_abandon_a_remote_job(tmp_path):
    m, client = module(), FakeClient()
    client.batches.raise_after_create = True
    with pytest.raises(RuntimeError):
        m.submit([extract_trace(raw())], tmp_path, client)
    state = json.loads((tmp_path/'state.json').read_text())
    with pytest.raises(ValueError, match='remote'):
        m.reconcile(tmp_path, client, state['jobs'][0]['key'], confirm_no_remote_batch=True)


def test_definitive_rejection_can_resume_without_new_reservation(tmp_path):
    m, client = module(), FakeClient()
    original = client.batches.create
    def rejected(**kwargs):
        error = RuntimeError('credentials rejected')
        error.status_code = 401
        raise error
    client.batches.create = rejected
    traces = [extract_trace(raw())]
    with pytest.raises(RuntimeError):
        m.submit(traces, tmp_path, client)
    reserved = json.loads((tmp_path/'state.json').read_text())['reserved_usd']
    client.batches.create = original
    result = m.submit(traces, tmp_path, client)
    assert result['submitted_batches'] == 1 and result['reserved_usd'] == reserved
