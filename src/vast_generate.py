"""Run the shared vLLM `generate` on a rented single-GPU host without Modal.

The Modal objects referenced by `src.modal_app` are replaced with inert stubs so
the generation code path is identical to the Modal runs. Outputs go to
/raw/<run_id>/ on the host, as on Modal.

Usage on the host: python -m src.vast_generate <requests.json> <checkpoint> <run_id> <max_tokens_think>
"""
import json
import sys
import types


class _Inert:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: self

    def __call__(self, *args, **kwargs):
        return self


def _install_modal_stub():
    stub = types.ModuleType('modal')
    stub.App = _Inert
    stub.Volume = _Inert()
    stub.Image = _Inert()
    stub.Dict = _Inert()
    stub.__version__ = 'not-used (vast.ai host)'
    sys.modules['modal'] = stub


def main():
    path, checkpoint, run_id, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    _install_modal_stub()
    # Progress bar only, for live cost tracking; sampling is unaffected.
    import vllm
    _generate = vllm.LLM.generate
    vllm.LLM.generate = lambda self, *a, **k: _generate(self, *a, **{**k, 'use_tqdm': True})
    from src.modal_app import generate
    from src.models_1c import MODELS_1C
    spec = json.load(open(path))[checkpoint]
    model = next(m for m, s in MODELS_1C.items() if s['ladder'] == 'think' and s['checkpoint'] == checkpoint)
    manifest = {**spec['manifest'], 'host': 'vast.ai', 'deviation': 10}
    generate(model, spec['requests'], run_id, manifest,
             {'gpu': sys.argv[5] if len(sys.argv) > 5 else 'vast', 'batch_size': 256,
              'max_tokens_think': max_tokens})


if __name__ == '__main__':
    main()
