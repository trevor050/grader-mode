"""Exercise remote export logic without allocating a GPU."""
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from src import modal_app


def test_export_survives_runtime_injected_modal_without_metadata(tmp_path, monkeypatch):
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
            assert kwargs['enable_thinking'] is False
            return 'rendered prompt'
        def encode(self, text, **kwargs):
            return [1, 2]
    class Engine:
        def __init__(self, **kwargs):
            assert kwargs['dtype'] == 'bfloat16' and kwargs['quantization'] is None
        def generate(self, prompts, params, **kwargs):
            assert params[0]['seed'] == 123
            return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                text='Final Answer: 42', token_ids=[1, 2, 3], finish_reason='stop')])]
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    request = {'id': 'fixture', 'seed': 123, 'messages': [{'role': 'user', 'content': 'test'}]}
    files = modal_app.generate('Qwen/Qwen3-0.6B', [request], 'unit-export', {}, {'gpu': 'L40S', 'batch_size': 64})
    rows = [json.loads(x) for x in (tmp_path / files['raw']).read_text().splitlines()]
    assert rows[0]['response'] == 'Final Answer: 42'
    stats = json.loads((tmp_path / files['metrics']).read_text())
    assert stats['requests'] == 1 and stats['output_tokens'] == 3
    assert stats['versions']['modal']
    assert raw.commits == 2 and cache.commits == 1
