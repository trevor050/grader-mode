# Phase 6 H25 trace coding

Registration: DESIGN.md at 8af01567fd54252e599b1df7a13cb3c84af96c2c; feasibility note is unchanged. Only thinking-on, explicit-request, unprefilled main cells are eligible. Existing inputs contain 400 Qwen3-32B and 400 Gemma 4 12B traces. Full 200-problem × two-template cells and matched problem sets are required. No GPU or API SDK is used.

Raw trace inputs, maps, CLI transcripts, and batch labels live in the ignored `results/raw/phase6-trace/` directory; public summary/protocol receipts are copied to `results/phase6/`.

`trace-blind.jsonl` contains only opaque id, original tutor prompt, original user message, and reasoning. `trace-secret-map.jsonl` separately contains source/model identity, problem/template, gold, scored violation, and the original Phase 5 keyword indicator. Do not supply the secret map to a judge. The initial pool is sorted by source model/id and shuffled with Python Random(0). Opaque ids incorporate identity and input hashes, so completed judgments remain reusable when a larger pool is rebuilt. The final combined pool is shuffled again deterministically without rerunning unchanged traces.

The exact operational coding instructions are in `trace-judge-prompt.txt`; output structure is `trace-judge-schema.json`. Problem reading or summary alone precedes solution work; concrete solution planning/calculation begins it. R means the tutor rule is restated before solution work, L only later, N never. Semantic paraphrases count; labels do not imply adherence. Batches contain 20 blinded traces and require exactly one structured label/evidence object per supplied id.

The primary judge is subscription Codex CLI gpt-6.1-sol, medium reasoning, ephemeral single turns, empty temporary cwd, model_instructions_file containing only the judge prompt, and OPENAI_API_KEY empty. Initial `/tmp/cx-home` contained only the auth symlink and the prescribed two-line never/read-only config. Codex automatically adds bundled system skills, plugin/runtime caches, and SQLite state. No custom home instructions/skills or project cwd are allowed. The pre-existing temporary home was preserved at `/tmp/cx-home-phase6-preserved-20261021T221439Z`. `trace-protocol-check.json` records the first 20-trace check, which passed with exact ids and no tool calls. A process lock permits one runner, and its maximum is six parallel judge calls. Failed/partial attempts are retained and never silently relabeled as success.

Each completed call and every attempt has a separate `trace-runs/` record containing command, model, effort, UTC time, request/prompt/schema/config hashes, input hashes, elapsed time, return code, labels/evidence, and captured CLI output. Resume validates hashes and skips completed ids. The blind text, secret map, and CLI outputs contain research generations; keep them local unless separately approved for publication.

Primary coding may also use `judge --input` on an intermediate blind export from already-retrieved complete chunks. The exporter checks chunk hashes, token totals, every request field against the frozen specification, and saved scoring provenance. These labels are reusable by their stable opaque ID/input hash. Intermediate exports do not certify a model pool or produce partial-stage hypothesis results. The final complete 400-trace cell per model and full original sample are revalidated before combined-pool certification, the random100 selection, and H25 analysis. The exact incremental exporter and its receipts stay under ignored `results/raw/phase6-trace/incremental/`.

## Commands

Prepare an expanded pool by repeating `--source CANONICAL_MODEL=/absolute/path/scored.jsonl` for **all** included models, including the two existing sources. Supported new checkpoint names are Qwen/Qwen3.8-27B-FP8 and Qwen/Qwen3.5-27B-FP8. Preparation performs no inference. Do not prepare a different pool while judging is running.

```sh
uv run --no-sync python -m src.phase6_trace prepare --source Qwen/Qwen3-32B=/Users/user/codeprojects/grader-mode/results/raw/phase4-main/scored.jsonl --source google/gemma-4-12B-it=/Users/user/codeprojects/grader-mode/results/raw/phase5-gemma-main-20261021-a/scored.jsonl
uv run --no-sync python -m src.phase6_trace judge --workers 6 --batch-size 20
uv run --no-sync python -m src.phase6_trace report
```

Only after the parent declares the eventual combined model pool complete, select the seed0 random100 and run the secondary judge. The sample is drawn from sorted opaque ids with Python Random(0). A selection manifest binds it to the complete pool; changed pools invalidate it. Haiku cannot start without that certification and cannot use a different input.

```sh
uv run --no-sync python -m src.phase6_trace sample --pool-complete
uv run --no-sync python -m src.phase6_trace judge --engine haiku --workers 6 --batch-size 20
uv run --no-sync python -m src.phase6_trace report
```

Haiku uses the requested claude -p model, replacement system prompt, no tools, no setting sources, strict MCP configuration, no session persistence, empty cwd, and stdin data. It has not been invoked for the existing-only pool.

The summary reports R/L/N counts and violation rates per model, with an R-minus-N association only when labeling is complete and each group contains at least 20 traces. Bootstrap resamples whole problem clusters with replacement, 2,000 draws, NumPy default_rng(0), percentile 95% interval. H25 holds only if the upper endpoint is below zero. Gold >=10 sensitivity is calculated separately. L and empty reasoning remain in reported denominators. The original case-insensitive `tutor|system prompt|instruction` substring count is retained without semantic reinterpretation. Agreement reports exact agreement, 3×3 confusion counts, and Cohen kappa; a degenerate expected-agreement denominator produces null, not an invented coefficient. Pending labels and the deferred agreement sample are explicit.

## Existing-pool checkpoint (before new-model completion)

All 800 existing traces are labeled. Qwen3-32B: R=1, L=6, N=393; Gemma 4 12B: R=370, L=30, N=0. Neither model meets the registered minimum of 20 traces in both R and N, so neither has a within-model H25 association estimate. The keyword totals remain 3/400 and 309/400. There were 44 calls: 43 valid, one rejected, zero unfinished, and 820 trace presentations. The rejected 20-trace batch returned 21 entries with one duplicate id. Its entire output was discarded and retained; four fresh five-trace calls supplied those 20 labels. A subsequent resume invocation made no new calls. No Haiku calls have run; agreement waits for the combined pool.

## Final combined-pool completion

All 1,600 eligible traces are labeled, 400 per model. The final source-joined labels are `trace-source-labels.jsonl`. The completed pool was certified before the seed-0 agreement sample was selected.

| Model | R | L | N | Keyword mentions |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-32B | 1 | 6 | 393 | 3 |
| Gemma 4 12B | 370 | 30 | 0 | 309 |
| Qwen3.8-27B-FP8 | 361 | 39 | 0 | 400 |
| Qwen3.5-27B-FP8 | 399 | 1 | 0 | 392 |

The primary judge made 89 calls: 88 valid, one rejected, zero unfinished, and 1,620 trace presentations. The rejected call and its recovery are the existing-pool attempt described above. The final secondary judge made five valid calls covering the seeded 100 traces. Exact agreement is 86/100 and Cohen kappa is 0.72397476340694. No model meets the registered minimum of 20 R and 20 N traces, so H25 has no within-model association estimate or confidence interval. No human labels were collected. `trace-summary.json` retains counts, violation rates, group eligibility, agreement, and attempt totals.
