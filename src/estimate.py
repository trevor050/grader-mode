"""Transparent planning scenarios from the 0.6B smoke; larger rungs are unmeasured."""
import argparse
import json
from pathlib import Path
from src.models import MODELS, load_gpu_config


def estimate(metrics, config):
    factor = 8000 / metrics['requests']
    rows = []
    for model in MODELS:
        size = float(model.removeprefix('Qwen/Qwen3-').removesuffix('B'))
        spec = config['models'][model]
        generation = metrics['generation_seconds'] * factor * size / .6
        function = generation + metrics['setup_seconds']
        rows.append({'model': model, 'gpu': spec['gpu'], 'parameters_billions': size,
                     'estimated_generation_seconds': generation,
                     'estimated_function_seconds': function,
                     'estimated_gpu_usd': function / 3600 * config['hourly_usd'][spec['gpu']]})
    equal_time = metrics['generation_seconds'] * factor + metrics['setup_seconds']
    return {'method': 'Planning heuristic: same response length; generation time proportional to model parameter count relative to 0.6B; same setup time per model. No hardware speedup assumed.',
            'limitations': 'Only 0.6B/L40S is measured. Larger models and RTX-PRO-6000 are unmeasured. Output lengths, GPU efficiency, larger batching, download/load time, and queueing can differ. These scenarios are not confidence bounds or a spending cap. CPU/RAM, storage and build charges are excluded.',
            'hourly_usd': config['hourly_usd'], 'models': rows,
            'total_gpu_seconds': sum(r['estimated_function_seconds'] for r in rows),
            'total_gpu_usd': sum(r['estimated_gpu_usd'] for r in rows),
            'equal_throughput_scenario': {'gpu_seconds': equal_time * len(rows),
                'gpu_usd': sum(equal_time / 3600 * config['hourly_usd'][r['gpu']] for r in rows)},
            'measured_smoke': metrics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', required=True, type=Path)
    p.add_argument('--output', default='results/cost-estimate.json', type=Path)
    p.add_argument('--gpu-config', default='config/gpus.json')
    args = p.parse_args()
    result = estimate(json.loads(args.metrics.read_text()), load_gpu_config(args.gpu_config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Parameter-scaled planning estimate: {result['total_gpu_seconds']/3600:.2f} GPU-hours, ${result['total_gpu_usd']:.2f} GPU-only")


if __name__ == '__main__':
    main()
