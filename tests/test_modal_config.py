import ast
from pathlib import Path
import pytest
from src.models import select_models, gpu_for


def test_ladder_and_gpu_routing():
    models = select_models('all')
    assert len(models) == 6
    assert [gpu_for(m) for m in models] == ['L40S'] * 5 + ['RTX-PRO-6000']
    assert select_models('0.6B,Qwen3-32B') == [models[0], models[-1]]
    with pytest.raises(ValueError):
        select_models('0.6B,0.6B')
    with pytest.raises(ValueError):
        select_models('72B')


def test_no_warm_or_unbounded_gpu_functions():
    source = Path('src/modal_app.py').read_text()
    tree = ast.parse(source)
    decorators = [d for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  for d in node.decorator_list if isinstance(d, ast.Call)
                  and isinstance(d.func, ast.Attribute) and d.func.attr == 'function']
    assert len(decorators) == 1
    for d in decorators:
        kw = {k.arg: k.value for k in d.keywords}
        assert isinstance(kw['timeout'], ast.Constant) and kw['timeout'].value > 0
        assert ast.literal_eval(kw['single_use_containers']) is True
        assert not {'min_containers', 'keep_warm', 'buffer_containers'} & kw.keys()
    assert 'H100' not in source


def test_parallel_window_caps_writers_and_cancels_on_failure(tmp_path, monkeypatch):
    from src import modal_app, build_prompts
    from src.models import load_gpu_config
    config = load_gpu_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modal_app, 'load_gpu_config', lambda _: config)
    monkeypatch.setattr(build_prompts, 'build', lambda *args: ([{'id': 'fixture'}], {}))

    class FakeFunction:
        def __init__(self, fail=False):
            self.active, self.peak, self.cancelled, self.calls = set(), 0, [], []
            self.fail = fail
        def with_options(self, **kwargs):
            self.gpu = kwargs['gpu']
            return self
        def spawn(self, model, *args):
            owner = self
            class Call:
                def get(self):
                    if owner.fail:
                        raise RuntimeError('remote failed')
                    owner.active.discard(self)
                    return {}
                def cancel(self, terminate_containers):
                    assert terminate_containers
                    owner.cancelled.append(self)
                    owner.active.discard(self)
            call = Call()
            self.calls.append((model, self.gpu, call))
            self.active.add(call)
            self.peak = max(self.peak, len(self.active))
            return call

    fake = FakeFunction()
    monkeypatch.setattr(modal_app, 'run_model', fake)
    modal_app.main.info.raw_f(models='all', limit_problems=10, run_id='unit-only')
    assert fake.peak == 5
    assert len(fake.calls) == 6
    assert fake.calls[-1][1] == 'RTX-PRO-6000'
    assert not fake.active
    failed = FakeFunction(fail=True)
    monkeypatch.setattr(modal_app, 'run_model', failed)
    with pytest.raises(RuntimeError, match='remote failed'):
        modal_app.main.info.raw_f(models='all', limit_problems=10, run_id='unit-failure')
    assert len(failed.cancelled) == 5
    assert not failed.active


def test_phase1c_explicitly_disables_idle_buffers():
    tree=ast.parse(Path('src/modal_app_1c.py').read_text())
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef) or node.name not in ('preflight','run_model'):
            continue
        decorator=node.decorator_list[0]
        kw={k.arg:ast.literal_eval(k.value) for k in decorator.keywords if k.arg in
            ('min_containers','buffer_containers','scaledown_window','timeout')}
        assert kw['min_containers']==0
        assert kw['buffer_containers']==0
        assert kw['scaledown_window']==2
        assert kw['timeout']==(5400 if node.name=='run_model' else 600)
