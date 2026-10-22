"""Transfer and run this smoke test through ssh pc, never inspect other PC paths."""
import argparse
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/looped-feasibility"


def ssh(command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "pc", command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upload", "fetch", "run", "status", "probe"])
    parser.add_argument("--model-index", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seconds", type=int, default=6500)
    parser.add_argument("--nanbeige", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume", action="store_true", help="Retain the initial model0 singleton prefix")
    parser.add_argument("--thinking-off", action="store_true")
    parser.add_argument("--check-singletons", action="store_true")
    parser.add_argument("--n-problems", type=int, choices=[3, 5], default=5)
    args = parser.parse_args()
    if args.action == "upload":
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for local, name in [(RAW/"manifest.json", "manifest.json"), (RAW/"scoring.py", "scoring.py"),
                                (RAW/"manifest-nanbeige.json", "manifest-nanbeige.json"),
                                (ROOT/"explore/looped/probe_format.py", "probe_format.py"),
                                (ROOT/"explore/looped/run_smoke.py", "run_smoke_next.py")]:
                tf.add(local, arcname=name)
        ssh('wsl --exec bash -c "cd ~/grader-looped && tar xzf -"', input=buf.getvalue())
    elif args.action == "run":
        interpreter = ".venv-nanbeige/bin/python" if args.nanbeige else ".venv/bin/python"
        manifest = "manifest-nanbeige.json" if args.nanbeige else "manifest.json"
        name = "nanbeige" if args.nanbeige else f"model{args.model_index}"
        if args.resume:
            assert args.model_index == 0
            if not args.nanbeige:
                name = "ouro"
        resume = ("--resume-from nanbeige-prefix.jsonl" if args.nanbeige else "--resume-from model0.jsonl") if args.resume else ""
        thinking = "--thinking-off" if args.thinking_off else ""
        check = "--check-singletons" if args.check_singletons else ""
        command = (f'wsl --exec bash -c "cd ~/grader-looped && '
                   f'{interpreter} -u run_smoke_next.py --model-index {0 if args.nanbeige else args.model_index} '
                   f'--manifest {manifest} --deadline-seconds {args.seconds} --output {name}.jsonl '
                   f'--batch-size {args.batch_size} {resume} {thinking} {check} '
                   f'> {name}.log 2>&1"')
        ssh(command)
    elif args.action == "probe":
        assert 0 < args.seconds <= 900
        ssh(f'wsl --exec bash -c "cd ~/grader-looped && .venv/bin/python -u probe_format.py '
            f'--deadline-seconds {args.seconds} --n-problems {args.n_problems} > format-probe.log 2>&1"')
    elif args.action == "status":
        name = "nanbeige.log" if args.nanbeige else "ouro.log"
        ssh(f'wsl --exec bash -c "cd ~/grader-looped && tail -n 3 {name}; '
            'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"')
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        archive = RAW/"pc-evidence.tar.gz"
        with archive.open("wb") as out:
            ssh('wsl --exec bash -c "cd ~/grader-looped && '
                'shopt -s nullglob && tar czf - --ignore-failed-read manifest*.json scoring.py run_smoke*.py '
                'requirements*.txt model*.jsonl model*.meta.json model*.log '
                'ouro*.jsonl ouro*.meta.json ouro*.log '
                'format-probe*.jsonl format-probe*.meta.json format-probe*.log probe_format.py '
                'nanbeige*.jsonl nanbeige*.meta.json nanbeige*.singleton-check.json nanbeige*.log"', stdout=out)
        with tarfile.open(archive) as tf:
            tf.extractall(RAW/"pc", filter="data")
        print(archive)


if __name__ == "__main__":
    main()
