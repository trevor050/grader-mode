"""Exercise remote export logic without allocating a GPU."""
import importlib.metadata
import json
from pathlib import Path
import sys
import pytest
from types import SimpleNamespace

from src import modal_app


@pytest.mark.parametrize('model,thinking,max_tokens,context', [
    ('Qwen/Qwen3-0.6B',False,768,2048),
    ('Qwen/Qwen3-0.6B',True,3072,4096),
    ('allenai/Olmo-3-7B-Instruct',False,768,8192),
    ('allenai/Olmo-3-7B-Think',True,3072,8192),
    ('allenai/Olmo-3-7B-Think',True,8192,12288),
])
def test_export_survives_runtime_injected_modal_without_metadata(tmp_path, monkeypatch, model, thinking, max_tokens, context):
    def version(name):
        if name == 'modal':
            raise importlib.metadata.PackageNotFoundError(name)
        return 'test-version'
    monkeypatch.setattr(importlib.metadata, 'version', version)
    class Volume:
        def __init__(self):
            self.commits = 0
        def commit(self):
            self.commits += 1
    raw, cache = Volume(), Volume()
    monkeypatch.setattr(modal_app, 'raw_volume', raw)
    monkeypatch.setattr(modal_app, 'hf_cache', cache)
    monkeypatch.setattr(modal_app, 'Path', lambda p: tmp_path if p == '/raw' else Path(p))
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs.get('enable_thinking') is thinking if model.startswith('Qwen') else 'enable_thinking' not in kwargs
            return 'rendered prompt'
        def encode(self, text, **kwargs):
            return [1, 2]
    class Engine:
        def __init__(self, **kwargs):
            assert kwargs['dtype'] == 'bfloat16' and kwargs['quantization'] is None
            assert kwargs['max_model_len'] == context
            assert kwargs['max_num_seqs'] == 256
        def generate(self, prompts, params, **kwargs):
            assert params[0]['seed'] == 123
            assert params[0]['max_tokens'] == max_tokens
            return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                text='Final Answer: 42', token_ids=[1, 2, 3], finish_reason='stop')])]
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    request = {'thinking': thinking, 'id': 'fixture', 'seed': 123, 'messages': [{'role': 'user', 'content': 'test'}]}
    if model.startswith('allenai'):
        request.update(phase='phase1c',arm='tutor')
    progress={}
    config={'gpu':'L40S','batch_size':256,'progress':progress}
    if thinking:
        config['max_tokens_think']=max_tokens
    files = modal_app.generate(model, [request], 'unit-export', {}, config)
    rows = [json.loads(x) for x in (tmp_path / files['raw']).read_text().splitlines()]
    assert rows[0]['response'] == 'Final Answer: 42'
    stats = json.loads((tmp_path / files['metrics']).read_text())
    assert stats['requests'] == 1 and stats['output_tokens'] == 3
    assert rows[0]['finish_reason'] == 'stop'
    assert stats['by_thinking'][str(thinking).lower()]['requests'] == 1
    assert stats['engine']['chunk_size'] == 2048
    assert stats['versions']['modal']
    assert raw.commits == 2 and cache.commits == 1
    assert progress[model]['completed']==1
    assert progress[model]['output_tokens']==3
    if model.startswith('allenai'):
        assert stats['sampling']['max_tokens']==max_tokens
        assert stats['model_spec']['max_model_len']==context
