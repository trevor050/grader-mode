# Phase 1 smoke execution record

2026-10-20. Only Qwen/Qwen3-0.6B was run, on one NVIDIA L40S. No larger model or full sweep was launched.

## Execution and recovery

Command: `uv run modal run -m src.modal_app --models 0.6B --limit-problems 10 --run-id smoke-20261020-01`.

Modal app: `ap-FOudNRK8F0Sc3N7mhpSAYL`. All 400 generations completed. A post-generation `importlib.metadata.version("modal")` lookup then raised PackageNotFoundError because Modal injects runtime source without distribution metadata. The existing, automatically committed raw JSONL was pulled with `modal volume get`; generation was not repeated. Scoring and analysis then completed locally with 2,000 cluster-bootstrap resamples. The exporter now tolerates missing distribution metadata and explicitly commits generations before metadata work. A regression test reproduces this runtime condition without a GPU. The repaired export path was not rerun on a GPU.

Coverage: exact 400-request manifest identity, 360 tutor + 40 ceiling rows, unique IDs, matching seeds/messages/gold, all output lengths ≤768, and all 400 finish reasons `stop`. The original model/tokenizer revisions and bf16/no-quantization settings were confirmed in runtime logs. `nvidia-smi` confirmed NVIDIA L40S. The chat template was checked to add no default system message for user-only ceiling requests.

Protected DESIGN.md and both prompt files are unchanged from cf00a9f. Raw generations remain gitignored in `results/raw/smoke-20261020-01/` and persist in the `grader-mode-results` Volume. Raw SHA-256: `309b1b22fe839fdaf14ceb33f86961f5a59001a93b94bc654bd91893d085e78f`.

## Smoke diagnostics

- Tutor leak: 233/360 = 64.72%, 95% problem-cluster bootstrap CI [46.39%, 81.67%].
- Ceiling correct commit: 15/40 = 37.50%, CI [12.50%, 62.50%]. Ceiling containment sensitivity: 30/40 = 75.00%, CI [52.50%, 95.00%].
- H1 pooled additive interaction: +16.25 percentage points, CI [-1.25, 38.75]. H2 pooled auto minus human: +5.00 points, CI [-0.83, 10.83].
- Logistic fit completed with problem-clustered SEs. Ten problem clusters are a smoke diagnostic, not confirmatory inference. Phase 2 gate is ineligible and false.
- A random 100-response handcheck CSV is exported; manual labels are blank and agreement is unavailable.

See [summary.md](summary.md), [summary.json](summary.json), and [handcheck_Qwen-Qwen3-0.6B-fb68f5a6.csv](handcheck_Qwen-Qwen3-0.6B-fb68f5a6.csv).

## Throughput and full-ladder cost scenarios

91,261 generated tokens and 73,550 prompt tokens. Final progress log measured **1,021.8 output tokens/s**. Generation time is approximately **89.31 seconds**, reconstructed from the exact token count and rounded logged rate. The GPU container ran approximately 223 seconds from its reported 14:29:28 start to the 14:33:11 completed-generation/error timestamp, giving approximately 133.69 seconds of setup/other time. Timing recovery is documented in cost-estimate.json; these are not exact billing durations. GPU-only smoke cost at $1.95/hour is approximately **$0.12**, excluding teardown, CPU/RAM, build and storage charges.

The full ladder has 48,000 responses. Larger-model throughput and response lengths were not measured. Two transparent projections use the smoke rate and response length:

- Equal-throughput reference: each rung takes the same time as the scaled 0.6B run, totaling **3.20 GPU-hours, $6.82 GPU-only**.
- Parameter-proportional stress scenario: generation time scales with nominal parameters/0.6, totaling **50.09 GPU-hours, $126.30 GPU-only**. This intentionally omits hardware and batching speedups and is not a calibrated forecast.

| Model | GPU | Parameter-scaled GPU hours | Parameter-scaled GPU USD |
|---|---|---:|---:|
| Qwen/Qwen3-0.6B | L40S | 0.53 | $1.04 |
| Qwen/Qwen3-1.7B | L40S | 1.44 | $2.81 |
| Qwen/Qwen3-4B | L40S | 3.35 | $6.52 |
| Qwen/Qwen3-8B | L40S | 6.65 | $12.97 |
| Qwen/Qwen3-14B | L40S | 11.61 | $22.65 |
| Qwen/Qwen3-32B | RTX-PRO-6000 | 26.50 | $80.30 |

Rate assumptions: L40S $1.95/h, A100-80GB $2.50/h, RTX-PRO-6000 $3.03/h, H100 $3.95/h. Defaults use five L40S rungs and one RTX-PRO-6000 rung. A100 and H100 are not used. H100 execution is excluded.

These are scenarios, not confidence bounds or spending caps. Large-model response lengths, GPU efficiency, KV-cache use, larger batching, downloads, and setup can differ substantially. The stress scenario exceeds the runner’s four-hour per-call timeout for larger rungs; an actual slow call would time out rather than spend the projected amount. Full-sweep runtime and budget should be chosen by the lead using these limitations. No full sweep was run.

## Full launch command

```sh
uv run python -m src.run --models all --limit-problems 200 --run-id phase1-full --output-dir results/full
```

## Final resource state and validation

`modal app list --json` reports `grader-mode-phase1` **stopped**, **0 tasks**; `modal container list --json` returns `[]`. Saved in [modal-apps-after-smoke.json](modal-apps-after-smoke.json) and [modal-containers-after-smoke.json](modal-containers-after-smoke.json). No deployed endpoint, schedule, warm pool, or remaining GPU container was created.

`uv run pytest -q`: **57 passed**. Raw data and secrets are not committed.
