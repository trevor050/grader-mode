"""Transparent planning scenarios from the 0.6B smoke; larger rungs are unmeasured."""
import argparse
import json
from pathlib import Path
from src.models import MODELS, load_gpu_config


def estimate(metrics, config, phase='phase1'):
    phase1b = phase == 'phase1b'
    if phase not in ('phase1', 'phase1b'):
        raise ValueError('Unknown phase')
    if phase1b and metrics.get('model') != 'Qwen/Qwen3-4B':
        raise ValueError('Phase 1b projection requires the 4B smoke metrics')
    target_requests, base_size = (4800, 4.) if phase1b else (8000, .6)
    factor = target_requests / metrics['requests']
    rows = []
    for model in MODELS:
        size = float(model.removeprefix('Qwen/Qwen3-').removesuffix('B'))
        if phase1b and size < 4:
            continue
        spec = config['models'][model]
        generation = metrics['generation_seconds'] * factor * size / base_size
        function = generation + metrics['setup_seconds']
        rows.append({'model': model, 'gpu': spec['gpu'], 'parameters_billions': size,
                     'estimated_generation_seconds': generation,
                     'estimated_function_seconds': function,
                     'estimated_gpu_usd': function / 3600 * config['hourly_usd'][spec['gpu']]})
    equal_time = metrics['generation_seconds'] * factor + metrics['setup_seconds']
    result = {'phase': phase, 'projected_requests_per_model': target_requests,
            'method': f'Planning heuristic: same response length; generation time proportional to model parameter count relative to {base_size:g}B; same setup time per model. No hardware speedup assumed.',
            'limitations': f'Only {base_size:g}B/L40S is measured for this projection. Larger models and RTX-PRO-6000 are unmeasured. Output lengths, GPU efficiency, larger batching, download/load time, and queueing can differ. These scenarios are not confidence bounds or a spending cap. CPU/RAM, storage and build charges are excluded.',
            'hourly_usd': config['hourly_usd'], 'models': rows,
            'total_gpu_seconds': sum(r['estimated_function_seconds'] for r in rows),
            'total_gpu_usd': sum(r['estimated_gpu_usd'] for r in rows),
            'equal_throughput_scenario': {'gpu_seconds': equal_time * len(rows),
                'gpu_usd': sum(equal_time / 3600 * config['hourly_usd'][r['gpu']] for r in rows)},
            'measured_smoke': {key: value for key, value in metrics.items() if key != 'manifest'}}
    if phase1b:
        result['budget_gate'] = {'limit_usd': 15, 'launch_full': result['total_gpu_usd'] <= 15,
                                 'basis': 'parameter-scaled projection; equal-throughput scenario is diagnostic only'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', required=True, type=Path)
    p.add_argument('--output', default='results/cost-estimate.json', type=Path)
    p.add_argument('--gpu-config', default='config/gpus.json')
    p.add_argument('--phase', choices=('phase1', 'phase1b'), default='phase1')
    args = p.parse_args()
    result = estimate(json.loads(args.metrics.read_text()), load_gpu_config(args.gpu_config), args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Parameter-scaled planning estimate: {result['total_gpu_seconds']/3600:.2f} GPU-hours, ${result['total_gpu_usd']:.2f} GPU-only")


if __name__ == '__main__':
    main()
