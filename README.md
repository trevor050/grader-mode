# grader-mode: Phase 1 and Phase 1b

A reproducible Modal/vLLM runner for the committed [pre-registration](DESIGN.md). It expands the 3 cues × 3 formats × 4 matched templates × 200 filtered GSM8K test problems, plus four no-system ceiling requests per problem: **8,000 requests/model, 48,000 across six models**.

## Phase 1b

The committed Phase 1b design uses the **same 200 GSM8K problems**, three cues
(`none`, `human_strict`, `env`), four matched templates, thinking off/on, and
no ceiling arm: **4,800 requests per model**. `--models all` selects 4B, 8B,
14B, and 32B for this phase. GPU routing and batch size use the existing config.

```sh
uv run python -m src.run --phase phase1b --models 4B --limit-problems 5 --run-id phase1b-smoke --output-dir results/phase1b-smoke
uv run python -m src.estimate --phase phase1b --metrics results/raw/phase1b-smoke/Qwen3-4B.metrics.json --output results/phase1b/budget-projection.json
# Launch only after the smoke-based full-run projection is <= $15:
uv run python -m src.run --phase phase1b --models all --limit-problems 200 --run-id phase1b-full --output-dir results/phase1b
```

Thinking off retains 768 new tokens; thinking on uses 3,072 with a 4,096-token
context. Raw rows retain `finish_reason`. Scored rows keep reasoning and final
reply separately, and score only the final reply. Empty thinking-on final
replies are excluded from rate denominators and counted in the summaries.
`src.analyze_1b` reports cells, H4/H5, baseline contrasts, reasoning-regex counts,
and the gold >= 10 sensitivity in `summary.md` and `summary.json`.

Recovery without GPU work:

```sh
uv run python -m src.score --input results/raw/phase1b-full --output results/raw/phase1b-full/scored.jsonl
uv run python -m src.analyze_1b --input results/raw/phase1b-full/scored.jsonl --output-dir results/phase1b
# Optional standalone sensitivity report (also included in the default report):
uv run python -m src.analyze_1b --input results/raw/phase1b-full/scored.jsonl --output-dir results/phase1b-gold10 --min-gold 10
```

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

## Phase 1 sweep

Phase 1 launch command from the repository root (requires explicit run authorization):

```sh
uv run python -m src.run --models all --limit-problems 200 --run-id phase1-full --output-dir results/full
```

Up to five model calls run concurrently, subject to workspace GPU quotas and availability. This bounds shared-Volume writers to Modal’s recommended maximum; the sixth rung starts after a prior call finishes. GPU allocation and batch sizes come from [`config/gpus.json`](config/gpus.json), passed with `--gpu-config`. Defaults: L40S for 0.6B–14B, `RTX-PRO-6000` (96 GB) for 32B. A100-80GB is a supported config alternative; H100 execution is excluded. The supplied hourly rate assumptions are also in this config.

There are no deployed endpoints, schedules, warm pools, or minimum containers. The ephemeral app and single-use containers terminate after work. Every GPU invocation has a 5,400-second timeout and 1,200-second startup timeout. Interrupting/failing the local entrypoint cancels sibling calls and terminates their containers. Timeouts bound individual calls; they are not a dollar budget.

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

Sampling is temperature 0.7, top-p 0.95, maximum 768 new tokens, bf16, no quantization, and `enable_thinking=False`. Top-k is disabled explicitly and model-specific generation defaults are bypassed. No repetition penalty, retries of failed generations, or output cleanup changes the registered sampling. Engine concurrency is 256 sequences, submitted in chunks of 2,048 with CUDA graphs enabled; metrics distinguish engine generation time from setup/function elapsed time.

See [analysis methods](docs/analysis-methods.md) for ceiling extraction, H1/H2 contrasts, regression failures, bootstrap pairing, and the descriptive-only smoke caveat. See [API references](docs/api-references.md) for the verified current APIs. `DESIGN.md` and both registered prompt files remain unchanged.

### Phase 1c

Pinned OLMo 3 Instruct/Think SFT → DPO → RLVR ladders use a separate ephemeral
`grader-mode-phase1c` Modal app. `DESIGN.md` and `prompts/` remain authoritative.

```sh
uv run modal run -m src.modal_app_1c --stage preflight
uv run python -m src.phase1c --stage smoke
uv run python -m src.phase1c --stage calibration
# Only launches when the measured smoke + calibration projection is <= $10:
uv run python -m src.phase1c --stage full
uv run pytest
```

Smoke uses three Phase 1 problems on each RLVR checkpoint. Calibration uses
20 disjoint heldout problems per RLVR checkpoint and records each ladder's
system-prompt decision in `results/phase1c/calibration.json`. Main results and
gold>=10 sensitivity summaries go to `results/phase1c/`; smoke summaries go
to `results/phase1c-smoke/`. Raw JSONL, prompts, and metrics remain gitignored
under `results/raw/` and the Modal results Volume. Run IDs refuse overwrites;
do not blindly rerun a generation stage after a partial failure.
