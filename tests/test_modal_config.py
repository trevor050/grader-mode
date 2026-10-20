import ast
from pathlib import Path
import pytest
from src.models import select_models, gpu_for


def test_ladder_and_gpu_routing():
    models = select_models('all')
    assert len(models) == 6
    assert [gpu_for(m) for m in models] == ['L40S'] * 5 + ['A100-80GB']
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
    assert len(decorators) == 2
    for d in decorators:
        kw = {k.arg: k.value for k in d.keywords}
        assert isinstance(kw['timeout'], ast.Constant) and kw['timeout'].value > 0
        assert ast.literal_eval(kw['single_use_containers']) is True
        assert not {'min_containers', 'keep_warm', 'buffer_containers'} & kw.keys()
    assert 'H100' not in source
