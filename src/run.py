"""Run generation, Volume pull, scoring, and local analysis in an isolated run."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from src.models import select_models, load_gpu_config


def commands(models, limit, run_id, output_dir, gpu_config):
    raw = str(Path('results/raw') / run_id)
    scored = str(Path(raw) / 'scored.jsonl')
    return [
        [sys.executable, '-m', 'modal', 'run', '-m', 'src.modal_app', '--models', models,
         '--limit-problems', str(limit), '--run-id', run_id, '--gpu-config', gpu_config],
        [sys.executable, '-m', 'src.score', '--input', raw, '--output', scored],
        [sys.executable, '-m', 'src.analyze', '--input', scored, '--output-dir', output_dir],
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', default='0.6B')
    p.add_argument('--limit-problems', type=int, default=10)
    p.add_argument('--run-id', default='')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--gpu-config', default='config/gpus.json')
    args = p.parse_args()
    select_models(args.models)
    load_gpu_config(args.gpu_config)
    if not 1 <= args.limit_problems <= 200:
        p.error('limit-problems must be between 1 and 200')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not run_id.replace('-', '').replace('_', '').isalnum():
        p.error('run-id must contain only letters, digits, - or _')
    if (Path('results/raw') / run_id).exists():
        p.error('run-id already exists; use a fresh run-id')
    for command in commands(args.models, args.limit_problems, run_id, args.output_dir, args.gpu_config):
        print('Running: ' + ' '.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
