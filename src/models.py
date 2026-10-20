"""Verified original Qwen3 dense post-trained ladder, pinned 2026-10-20."""
MODELS = {
    'Qwen/Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca',
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
    'Qwen/Qwen3-4B': '1cfa9a7208912126459214e8b04321603b3df60c',
    'Qwen/Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
    'Qwen/Qwen3-14B': '40c069824f4251a91eefaf281ebe4c544efd3e18',
    'Qwen/Qwen3-32B': '9216db5781bf21249d130ec9da846c4624c16137',
}


def select_models(value):
    selected = list(MODELS) if value == 'all' else [x.strip() for x in value.split(',')]
    selected = [x if x.startswith('Qwen/') else f'Qwen/Qwen3-{x.removeprefix("Qwen3-")}' for x in selected]
    if len(set(selected)) != len(selected) or any(x not in MODELS for x in selected):
        raise ValueError(f'Choose distinct models from {list(MODELS)} or all')
    return selected


def gpu_for(model):
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    return 'A100-80GB' if model == 'Qwen/Qwen3-32B' else 'L40S'
