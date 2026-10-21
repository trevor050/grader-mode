"""Phase 6 only: official FP8 pins and preregistered runtime policy.

Minimal-file SHA256 values were read from the immutable publisher revisions,
2026-10-21. quantization='fp8' loads the checkpoint's existing FP8 tensors;
dtype='bfloat16' selects activation/unquantized-module precision.
"""
from copy import deepcopy

REGISTRATION_COMMIT = '8af01567fd54252e599b1df7a13cb3c84af96c2c'
VLLM_VERSION = '0.32.0'
SAMPLING = {'temperature': .7, 'top_p': .95, 'top_k': -1, 'max_tokens': 4096}
MAX_MODEL_LEN = 8192
CHUNK_SIZE = 256
THINK_OPEN = 248068
THINK_CLOSE = 248069
EOS_TOKEN_IDS = (248046, 248044)
SOURCE_SHA256 = {
    'DESIGN.md': 'f1d2d05185c1593deba90a7b58601691d18b99ba047e02866d06abca10c90c84',
    'prompts/templates_4.json': '6cd003a8b8bf96456428084a1abeedb5c66ee5774443bcb6fdf31d7d65a8fd31',
    'prompts/templates.json': '637fb8aa1312f4485d410241fe6087cbbe9c6150a4267bfb8a71e418f7af0ed1',
    'prompts/system_tutor.txt': '6011cd11fc5a8ac78a4b90e35b7f5154db64467a2854a15f3cdee4ea8be88d11',
}
SAMPLE_SHA256 = '44adbea65a655f1fab0d2c09aa10cda3f0e8c077a185e13b085891227fea9464'
MODELS_6 = {
    '3.8': {
        'model': 'Qwen/Qwen3.8-27B-FP8',
        'revision': '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
        'reasoning_effort': 'medium',
        'file_sha256': {
            'config.json': '74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc',
            'generation_config.json': 'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e',
            'tokenizer_config.json': 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27',
            'chat_template.jinja': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041',
            'tokenizer.json': '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3',
        },
    },
    '3.5': {
        'model': 'Qwen/Qwen3.5-27B-FP8',
        'revision': '97f5941bf617e31c5e237364a8602ce3f03a551a',
        'reasoning_effort': None,
        'file_sha256': {
            'config.json': '53f7e06eebbfe64e70eb0b0979f3da31de9aff6870594e9cf731ba3961501adc',
            'generation_config.json': '303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd',
            'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
            'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
            'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
        },
    },
}


def model_config(model):
    if model not in MODELS_6:
        raise ValueError('Phase 6 model must be 3.8 or 3.5')
    return deepcopy(MODELS_6[model])


def engine_kwargs(model, max_num_seqs=256):
    if type(max_num_seqs) is not int or max_num_seqs not in (32, 64, 128, 256):
        raise ValueError('max_num_seqs must be 32, 64, 128, or 256')
    cfg = model_config(model)
    return dict(model=cfg['model'], revision=cfg['revision'],
                tokenizer_revision=cfg['revision'], dtype='bfloat16',
                quantization='fp8', language_model_only=True,
                max_model_len=MAX_MODEL_LEN, max_num_seqs=max_num_seqs,
                enforce_eager=False, tensor_parallel_size=1, seed=0,
                generation_config='vllm')
