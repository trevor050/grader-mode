"""Cross-module runtime constraints verified before any paid launch."""
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def test_vllm_032_transformers_constraint():
    # Primary source: vLLM 0.32.0 published wheel metadata, checked 2026-10-21.
    from src.models_5 import PACKAGE_VERSIONS
    assert Version(PACKAGE_VERSIONS['transformers']) in SpecifierSet('>=5.16.1,<5.20.0')


def test_modal_dependency_pins_match_generation_and_transformers_minimum():
    from src.modal_app_5 import PINS
    from src.models_5 import PACKAGE_VERSIONS
    pins = dict(item.split('==') for item in PINS)
    for key, version in PACKAGE_VERSIONS.items():
        assert pins[key] == version
    # Transformers 5.19.0 requires this, although vLLM itself has an older floor.
    assert Version(pins['safetensors']) >= Version('0.8.0')


def test_modal_mounts_are_relative_to_actual_python_source_root():
    from pathlib import Path
    # Modal add_local_python_source mounts src at /root/src. build_prompts.ROOT
    # is thus /root, and build() reads design/prompts there.
    source = Path('src/modal_app_5.py').read_text()
    assert "remote_path='/root/prompts'" in source
    assert "remote_path='/root/DESIGN.md'" in source


def test_empty_containers_without_app_stop_ack_never_clears_exposure(monkeypatch):
    from src import modal_app_5 as runner
    def cli(*args, **kwargs):
        if args[:2] == ('app', 'stop'):
            raise RuntimeError('provider stop rejected')
        return '[]'
    monkeypatch.setattr(runner, 'modal_cli', cli)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    zero, evidence = runner.stop_and_verify('ap-test')
    assert not zero
    assert not evidence['app_stop_confirmed']


def test_app_stop_retry_requires_ack_and_empty_containers(monkeypatch):
    from src import modal_app_5 as runner
    calls=[]
    def cli(*args, **kwargs):
        if args[:2] == ('app', 'stop'):
            calls.append(args)
            if len(calls)==1:
                raise RuntimeError('transient failure')
            return 'stopped'
        return '[]'
    monkeypatch.setattr(runner, 'modal_cli', cli)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    zero, evidence = runner.stop_and_verify('ap-test')
    assert zero and evidence['app_stop_confirmed'] and evidence['containers_after']==[]
    assert len(calls)==2 and evidence['stop_errors']


def test_full_glm_fallback_requires_actual_worker_start_timestamp():
    import pytest
    from test_phase5_analysis import grid, runtime_failure_proof, GLM
    from src.score_5 import score_rows
    from src.analyze_5 import validate_grid
    # The timestamp guard must run before any unrelated grid-size rejection.
    with pytest.raises(ValueError, match='generation_started_unix'):
        validate_grid(score_rows(grid(second_model=GLM)), expected_problems=200,
                      allow_glm_fallback=True, fallback_provenance=runtime_failure_proof())
