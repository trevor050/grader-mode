import pytest
from src.modal_app import generation_limits, render_prompt
from src.vast_generate_4 import expand_requests


class Tokenizer:
    def __init__(self, opening=''):
        self.opening = opening
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is True
        return '<|im_start|>assistant\n' + self.opening


def test_caps_preserve_old_phases_and_override_both_modes():
    assert generation_limits('Qwen/Qwen3-14B', {}, False) == (768, 2048)
    assert generation_limits('Qwen/Qwen3-14B', {}, True) == (3072, 4096)
    for thinking in (False, True):
        assert generation_limits('Qwen/Qwen3-14B', {'max_tokens_override': 4096}, thinking) == (4096, 8192)
    with pytest.raises(ValueError):
        generation_limits('Qwen/Qwen3-14B', {'max_tokens_override': True}, True)


@pytest.mark.parametrize('opening', ['', '<think>\n'])
def test_prefills_share_exactly_one_opening(opening):
    for text in ('', 'A reminder.', 'Neutral.', 'Irrelevant.'):
        row = {'phase':'phase4', 'arm':'prefill', 'thinking':True, 'prefill_text':text, 'messages':[]}
        assert render_prompt(Tokenizer(opening), row) == '<|im_start|>assistant\n<think>\n'+text


def test_grid_pairing():
    import json
    from pathlib import Path
    templates = json.loads(Path('prompts/templates_4.json').read_text())
    phase1 = json.loads(Path('prompts/templates.json').read_text())
    p = [{'problem_idx':1,'question':'question','gold':99}]
    rows = expand_requests(p, templates, phase1, 'system')
    assert len(rows) == 28
    assert len({r['id'] for r in rows}) == 28
    for t in (0,1):
        assert len({r['seed'] for r in rows if r['template_idx']==t}) == 1
    assert len(expand_requests(p, templates, phase1, 'system', False)) == 12


def test_measured_remaining_budget_fallback():
    from src.phase4_vast_control import projected_total
    full = projected_total(1, 1000, 200, 1200, 256000, 1.175)
    fallback = projected_total(1, 1000, 200, 1200, 256000, 1.175, fallback=True)
    assert fallback < full and full > 1.3
    with pytest.raises(ValueError):
        projected_total(0, 0, 200, 1200, 1, 1)


def test_shared_generator_budget_fallback_keeps_complete_matched_cells(monkeypatch, tmp_path):
    """Execute the real batching/output path with inert HF/vLLM dependencies."""
    import json
    import sys
    import types
    from pathlib import Path
    import src.modal_app as app
    from src.models import MODELS
    from src.vast_generate_4 import expand_requests
    template = json.loads(Path('prompts/templates_4.json').read_text())
    phase1 = json.loads(Path('prompts/templates.json').read_text())
    requests = expand_requests([{'problem_idx':i,'question':'Q?','gold':42} for i in range(3)],template,phase1,'tutor',False)
    calls = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            assert revision == MODELS[model]
            return cls()
        def apply_chat_template(self, messages, **kwargs):
            return '<|im_start|>assistant\n'
        def encode(self, text, **kwargs):
            return [1]*10

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs['max_model_len'] == 8192 and kwargs['dtype'] == 'bfloat16'
        def generate(self, prompts, params, **kwargs):
            calls.append(params)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[2,3],text='</think>Try a step.',finish_reason='stop')],prompt_token_ids=[1]*10) for _ in prompts]

    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(LLM=LLM,SamplingParams=SamplingParams))
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=Tokenizer))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**kw:None)
    monkeypatch.setattr(app,'Path',lambda p:tmp_path if p=='/raw' else Path(p))
    monkeypatch.setattr(app,'hf_cache',types.SimpleNamespace(commit=lambda:None))
    monkeypatch.setattr(app,'raw_volume',types.SimpleNamespace(commit=lambda:None))
    directory=tmp_path/'run';directory.mkdir()
    (directory/'Qwen3-32B.gate.json').write_text(json.dumps({'action':'continue','drop_request_types':['boxed']}))
    app.generate('Qwen/Qwen3-32B',requests,'run',{}, {'gpu':'fake','batch_size':8,'chunk_size':8,'max_tokens_override':4096,'phase4_gate':True})
    rows=[json.loads(l) for l in (directory/'Qwen3-32B.jsonl').read_text().splitlines()]
    expected=[r for r in requests if r['request_type']!='boxed']
    assert {r['id'] for r in rows} == {r['id'] for r in expected}
    assert len(rows)==24 and all(p.max_tokens==4096 for c in calls for p in c)
    stats=json.loads((directory/'Qwen3-32B.metrics.json').read_text())
    assert stats['requests']==24 and stats['sampling']['max_tokens']==4096
    assert stats['sampling']['thinking_on']['max_tokens']==4096
