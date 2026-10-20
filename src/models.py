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



def load_gpu_config(path=None):
    import json
    from pathlib import Path
    source = Path(path) if path else Path(__file__).resolve().parents[1] / 'config/gpus.json'
    config = json.loads(source.read_text())
    if set(config['models']) != set(MODELS):
        raise ValueError('GPU config must cover exactly the registered ladder')
    for spec in config['models'].values():
        if spec['gpu'] not in {'L40S', 'A100-80GB', 'RTX-PRO-6000'}:
            raise ValueError('GPU must be L40S, A100-80GB, or RTX-PRO-6000')
        if not isinstance(spec['batch_size'], int) or not 1 <= spec['batch_size'] <= 512:
            raise ValueError('batch_size must be between 1 and 512')
    return config


def gpu_for(model, config=None):
    return (config or load_gpu_config())['models'][model]['gpu']
