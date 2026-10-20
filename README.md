# grader-mode: Phase 1

A reproducible Modal/vLLM runner for the committed [pre-registration](DESIGN.md). It expands the 3 cues × 3 formats × 4 matched templates × 200 filtered GSM8K test problems, plus four no-system ceiling requests per problem: **8,000 requests/model, 48,000 across six models**.

## Setup

```sh
uv sync --dev
uv run pytest
```

Modal must be authenticated to the intended workspace. No Hugging Face token is needed for these public models. Never commit credentials. Local dependencies are locked in `uv.lock`; GPU-only vLLM is pinned in the Modal image. Original model and dataset commit hashes are pinned in source.

## Smoke test

```sh
uv run python -m src.run --models 0.6B --limit-problems 10 --run-id smoke-fresh
```

This runs **400 generations** on one L40S, pulls committed Modal Volume files to `results/raw/smoke-fresh/`, scores them locally, and writes `results/summary.md`, `results/summary.json`, and a deterministic 100-response handcheck CSV. Use a fresh run ID; existing run directories are refused. Raw responses and manifests are gitignored. Handcheck labels start blank; agreement is unavailable until a person enters labels and reruns analysis.

## Full sweep (requires the lead's spending decision)

The full sweep has **not** been authorized or run. Exact launch command from the repository root:

```sh
uv run python -m src.run --models all --limit-problems 200 --run-id phase1-full --output-dir results/full
```

Up to five model calls run concurrently, subject to workspace GPU quotas and availability. This bounds shared-Volume writers to Modal’s recommended maximum; the sixth rung starts after a prior call finishes. GPU allocation and batch sizes come from [`config/gpus.json`](config/gpus.json), passed with `--gpu-config`. Defaults: L40S for 0.6B–14B, `RTX-PRO-6000` (96 GB) for 32B. A100-80GB is a supported config alternative; H100 execution is excluded. The supplied hourly rate assumptions are also in this config.

There are no deployed endpoints, schedules, warm pools, or minimum containers. The ephemeral app and single-use containers terminate after work. Every GPU invocation has a 14,400-second timeout and 1,200-second startup timeout. Interrupting/failing the local entrypoint cancels sibling calls and terminates their containers. Timeouts bound individual calls; they are not a dollar budget.

## Individual stages and recovery

```sh
uv run python -m src.build_prompts --limit-problems 10 --output-dir results/raw/prompts-only
uv run modal run -m src.modal_app --models 0.6B --limit-problems 10 --run-id another-fresh-smoke
uv run python -m src.score --input results/raw/another-fresh-smoke --output results/raw/another-fresh-smoke/scored.jsonl
uv run python -m src.analyze --input results/raw/another-fresh-smoke/scored.jsonl --output-dir results --resamples 2000
uv run python -m src.estimate --metrics results/raw/another-fresh-smoke/Qwen3-0.6B.metrics.json
uv run modal app list --json
```

HF weights persist in `grader-mode-hf-cache`; generations and metrics persist in `grader-mode-results`. The entrypoint pulls files with `Volume.read_file` after the remote commit. If local pulling fails after generation succeeds, recover without new GPU work:

```sh
uv run modal volume get grader-mode-results another-fresh-smoke/Qwen3-0.6B.jsonl results/raw/another-fresh-smoke/Qwen3-0.6B.jsonl
uv run modal volume get grader-mode-results another-fresh-smoke/Qwen3-0.6B.metrics.json results/raw/another-fresh-smoke/Qwen3-0.6B.metrics.json
```

The 10-problem smoke is a prefix of the fixed seed-0 sample of 200, not a new sample. Filtering uses whole normalized number tokens. Per-request seeds derive from SHA-256 of stable IDs encoding the original dataset index, cue, format, template index, and arm. Requests are deterministically shuffled for generation, with independent per-request seeds.

Sampling is temperature 0.7, top-p 0.95, maximum 768 new tokens, bf16, no quantization, and `enable_thinking=False`. Top-k is disabled explicitly and model-specific generation defaults are bypassed. No repetition penalty, retries of failed generations, or output cleanup changes the registered sampling. Engine batches are 64 on L40S and 128 for the default 32B GPU. Eager execution avoids compilation overhead; metrics distinguish engine generation time from setup/function elapsed time.

See [analysis methods](docs/analysis-methods.md) for ceiling extraction, H1/H2 contrasts, regression failures, bootstrap pairing, and the descriptive-only smoke caveat. See [API references](docs/api-references.md) for the verified current APIs. `DESIGN.md` and both registered prompt files remain unchanged.
