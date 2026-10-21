"""Phase 5 only: publisher revisions and runtime/format policy frozen 2026-10-21.

Config/template hashes are from the files at the revisions below, not main.
Nemotron keeps its official mixed NVFP4/FP8/BF16 checkpoint configuration;
quantization=None means vLLM detects that configuration, never requantizes it.
No imports of the historical model allowlists or remote execution packages.
"""
from copy import deepcopy

GEMMA = 'google/gemma-4-12B-it'
NEMOTRON = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'
GLM = 'QuantTrio/GLM-4.7-Flash-AWQ'
PRIMARY_MODELS = (GEMMA, NEMOTRON)
ACTIVE_MODELS = PRIMARY_MODELS
FALLBACK_MODEL = GLM
REGISTRATION_COMMIT = 'c3e705068f0c3622c3b5d4cb9575530004ad68ae'
ORIGINAL_REGISTRATION_COMMIT = 'f15dcc41904f6fad858bdab6c8b7daac09238520'
RENDER_DATE = '2026-10-21'
VLLM_VERSION = '0.32.0'
PACKAGE_VERSIONS = {'vllm': VLLM_VERSION, 'transformers': '5.19.0',
                    'tokenizers': '0.23.3', 'jinja2': '3.1.6'}
SAMPLING = {'temperature': 0.7, 'top_p': 0.95, 'top_k': -1, 'max_tokens': 4096}
PREFILLS = ('empty', 'reminder', 'neutral', 'irrelevant')
MODELS_5 = {
    GEMMA: {
        'revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
        'tokenizer_revision': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
        'dtype': 'bfloat16', 'quantization': None, 'format': 'gemma4',
        'native_off': True, 'reasoning_effort': None, 'max_model_len': 8192,
        'stop_token_ids': [1, 106, 50],
        'special_tokens': {'<eos>': 1, '<|tool_response>': 50, '<|think|>': 98,
                           '<|channel>': 100, '<channel|>': 101,
                           '<|turn>': 105, '<turn|>': 106},
        'file_sha256': {
            'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
            'generation_config.json': 'a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e',
            'config.json': '478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9',
            'tokenizer_config.json': 'a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9',
            'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
        },
    },
    NEMOTRON: {'revision': 'bee7596271d1495f6992ae224aefde4410e816b8',
        'tokenizer_revision': 'bee7596271d1495f6992ae224aefde4410e816b8',
        'dtype': 'bfloat16',
        'quantization': None,
        'format': 'think_tags',
        'native_off': True,
        'reasoning_effort': None,
        'max_model_len': 8192,
        'quantization_policy': 'checkpoint_config',
        'checkpoint_quantization': 'modelopt',
        'native_on_tail': '<|im_start|>assistant\n<think>\n',
        'native_off_tail': '<|im_start|>assistant\n<think></think>',
        'stop_token_ids': [2, 11],
        'final_token_ids': [2, 11],
        'tool_token_ids': [5, 6, 7, 8, 9, 14, 15, 16, 17],
        'special_tokens': {'<unk>': 0,
                           '<s>': 1,
                           '</s>': 2,
                           '[INST]': 3,
                           '[/INST]': 4,
                           '[AVAILABLE_TOOLS]': 5,
                           '[/AVAILABLE_TOOLS]': 6,
                           '[TOOL_RESULTS]': 7,
                           '[/TOOL_RESULTS]': 8,
                           '[TOOL_CALLS]': 9,
                           '<|im_start|>': 10,
                           '<|im_end|>': 11,
                           '<think>': 12,
                           '</think>': 13,
                           '<tool_call>': 14,
                           '</tool_call>': 15,
                           '<tool_response>': 16,
                           '</tool_response>': 17},
        'file_sha256': {'chat_template.jinja': '58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9',
                        'generation_config.json': 'c67c90fcadf0e7ef78b663eb159664b36a35ffb9dd2523096e226e5c3d1b0e5f',
                        'config.json': 'f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933',
                        'tokenizer_config.json': '10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc',
                        'tokenizer.json': '623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7'}},
    GLM: {'revision': '88e3d3d913c0d97c8f505cdc03433c48226bedc3',
        'tokenizer_revision': '88e3d3d913c0d97c8f505cdc03433c48226bedc3',
        'dtype': 'bfloat16',
        'quantization': 'awq',
        'format': 'think_tags',
        'native_off': True,
        'reasoning_effort': None,
        'max_model_len': 8192,
        'quantization_policy': 'awq',
        'checkpoint_quantization': 'awq',
        'native_on_tail': '<|assistant|><think>',
        'native_off_tail': '<|assistant|></think>',
        'stop_token_ids': [154820, 154827, 154829],
        'final_token_ids': [154820, 154827],
        'tool_token_ids': [154829, 154843, 154844, 154845, 154846, 154847, 154848, 154849, 154850],
        'special_tokens': {'<|endoftext|>': 154820,
                           '<|system|>': 154826,
                           '<|user|>': 154827,
                           '<|assistant|>': 154828,
                           '<|observation|>': 154829,
                           '<think>': 154841,
                           '</think>': 154842,
                           '<tool_call>': 154843,
                           '</tool_call>': 154844,
                           '<tool_response>': 154845,
                           '</tool_response>': 154846,
                           '<arg_key>': 154847,
                           '</arg_key>': 154848,
                           '<arg_value>': 154849,
                           '</arg_value>': 154850},
        'file_sha256': {'chat_template.jinja': 'd63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375',
                        'generation_config.json': '9ddb5c6300acab6153583ec81620fbdecab4506ef68c6dda6fc179ca736fbecb',
                        'config.json': '2954dc4fdaa883d7ef9c5bd250d3d0c3e43be0d87b93f832fa343727b619ee5f',
                        'tokenizer_config.json': '31a173e2797ddc8b72ac996803513e627fc28d7aad02cfcce321a431d865c86d',
                        'tokenizer.json': '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'}},
}


def resolve_model(model_key):
    """Accept canonical HF names or the CLI keys gemma/nemotron/glm."""
    model = {'gemma': GEMMA, 'nemotron': NEMOTRON, 'glm': GLM}.get(model_key, model_key)
    if model not in MODELS_5:
        raise ValueError(f'Unregistered Phase 5 model: {model_key!r}')
    return model


def model_spec(model_key):
    return deepcopy(MODELS_5[resolve_model(model_key)])
