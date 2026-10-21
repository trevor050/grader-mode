# Phase 6 execution and analysis methods

The registration is DESIGN.md at `8af01567fd54252e599b1df7a13cb3c84af96c2c`. The lead explicitly applied the protocol deviation at `8219501182ce380c45115efd256938a2e372f088` to Phase 6. DESIGN.md and prompts were not edited by Phase 6. The deviation scores thinking-off replies after their last literal `</think>` and reports a second outcome counting all such replies as violations. Protocol diagnostics remain descriptive and do not block inference. Ordinary and special token decodings, spans, and generator helper fields are validated before the exception is applied.

## Generation

`src/models_6.py` pins the official FP8 revisions, tokenizer/config hashes, BF16 activation dtype, and vLLM 0.32.0. Qwen3.8 uses `reasoning_effort="medium"`; Qwen3.5 does not supply that template argument. Sampling is temperature 0.7, top-p 0.95, top-k -1, one continuation, and at most 4,096 generated tokens including reasoning. Native thinking-on prompts end in `<think>\n`; native thinking-off prompts contain the completed empty think block. Prefill bytes are appended literally. The empty prefill is the existing main/on/explicit condition.

The fixed 200-problem sample, both templates, prompt messages, and per-problem/template seeds are reused from Phase 4. Seed text is `phase4:pNNNN:tT`; no Phase 6 seed namespace is substituted. The original sample is also independently reconstructed from the pinned GSM8K dataset for analysis validation. Main stages have 1,600 replies per model; Qwen3.8 has three complete 400-reply prefill cells.

Each model first receives eight outcome-blind, unscored smoke requests covering its cells and a separate 256-request throughput measurement. Pilot IDs are disjoint from study IDs. CUDA graphs, 256 resident sequences, and chunks of 256 requests are used on the selected 96 GB GPUs. One engine remains loaded across stages for each model. Every next complete stage is permitted only after the saved throughput, per-cell output lengths, elapsed rental exposure, transfer envelope, reserve, and absolute deadline admit it. The projection uses a 20% generation-time margin. Numeric adherence outcomes are not read by the controller. Stage decisions and dollar shortfalls are retained in `vast-execution.json`.

The official vLLM container is pinned by digest. Before renting, PyPI metadata and a Python 3.12 Linux wheel dry run were checked; both instances then recorded live Python/PyTorch/CUDA/vLLM import receipts. The first local request-spec packaging error was repaired on the same rental before model loading. All elapsed time and transfer reservations remain charged. `bootstrap-repair.json`, `runtime-preflight.json`, and `runtime-receipts.json` preserve those checks.

Only Vast is used. CLI calls are serialized, spaced, and backed off. Unique instance labels, absolute deadlines, local teardown watchdogs, and remote transfer guards bound exposure. Completed chunks are retrieved incrementally and verified against immutable remote hashes before collection. Destruction uses `--yes`, followed by an independent instance listing. A separate `grader-explore` instance (57125990) started during collection; the user explicitly instructed that it be preserved. Teardown verification therefore covers all Phase 6 instances, while the account-wide nonzero listing is disclosed. The account credit delta is not attributed to Phase 6 during this concurrent usage. Conservative cost includes every reserved rental second and the full transfer envelope; provider credit deltas are separately reported as provisional.

## Scoring and uncertainty

`src/phase6_collect.py` accepts only completed whole stages whose chunk checksums, counts, token totals, request IDs, prompts, and factors match the frozen request specification. Pilots are excluded. `src/score_6.py` validates saved token IDs and their decoded partitions without loading a tokenizer. Only final-reply text is scored. The Phase 4 leak and literal numeric commitment detector is reused; violation is leak OR attempt. Cap hits, no-final replies, protocol diagnostics, correct/wrong attempts, and reply lengths retain all requests in their denominators.

`src/analyze_6.py` validates the full retained grid, original sample, source hashes, seeds, pins, and common implementation provenance. Confidence intervals use 2,000 paired problem-cluster resamples with NumPy seed 0 and percentile 95% intervals. Both templates and all cells remain paired within a problem. Gold >= 10 repeats the analysis on the prespecified subset. The off-closer violation sensitivity uses the same draws. H22/H24 use the registered thinking-by-conflict interaction; H23 requires all three comparator-minus-reminder lower bounds to exceed zero. H25 is N minus R and remains an association, with at least 20 R and 20 N traces required within each model. L remains descriptive. Undefined ratio bootstrap draws are reported.

## Blinded labels

`trace-methods.md` and the saved judge prompt describe H25 coding. The eligible pool is the complete main/on/explicit, unprefilled cell for each model, including empty reasoning and protocol diagnostics. The two existing source files are read only from the main checkout. `historical-h25-selection.json` binds the exact historical source files to the 800 selected records; no Qwen3-14B rows enter H25. Judges receive only opaque IDs, tutor prompts, user messages, and reasoning. Model metadata, final replies, gold, and outcomes stay in a separate private map.

The primary judge is `gpt-6.1-sol` through isolated, ephemeral subscription Codex CLI turns at medium reasoning. The temporary CODEX_HOME starts with only the auth symlink and never/read-only configuration. Judge cwd is empty, OPENAI_API_KEY is empty, and tool use is rejected. Six primary calls may run in parallel. The second judge is `claude-haiku-5-5` through the prescribed tool-free subscription CLI on a seed-0 random 100 selected only after the final combined pool is declared complete. Exact agreement, confusion counts, and Cohen kappa are reported. Attempts, including rejected structured outputs, are retained. No metered model API is used.

The parser audit selects 40 final replies per retained model with deterministic seed-0 round-robin allocation over nonempty factor/attempt strata. Its export contains only opaque ID, problem, gold, and final reply. Separate isolated Codex turns label attempts and numeric commitments without factors, model metadata, reasoning, or parser predictions. A complete model can be labeled while another generates; final aggregation verifies the exact combined deterministic sample and reuses unchanged labels. Two parser calls plus six trace calls respect the Mac's eight-call limit. All first valid labels and disagreements are retained. Labels are Codex self-labels; no human labels or independent human validation are claimed.

## Replay

Raw generations, request specifications, exact execution bundles, trace maps, judge transcripts, and audit exports stay under ignored `results/raw/phase6-*/`. They are copied with hash verification to the main checkout's `results/raw/` before worktree cleanup. Public receipts and summaries contain hashes and aggregate results. The raw bundles retain the exact generation source used on each instance; the subsequent source-check compatibility change accepts only the exact authorized DESIGN deviation and changes no generation recipe.

```sh
uv run --no-sync python -m src.phase6_collect --requests results/raw/phase6-worker-final/requests.json --source-dir 3.8=results/raw/phase6-3.8-a1-20261021 --source-dir 3.5=results/raw/phase6-3.5-a2-20261021 --output-dir results/raw/phase6-main
uv run --no-sync python -m src.score_6 --input results/raw/phase6-main/generations.jsonl --output results/raw/phase6-main/scored.jsonl
uv run --no-sync python -m src.analyze_6 --input results/raw/phase6-main/scored.jsonl --problem-manifest results/raw/phase6-main/manifest.json --output-dir results/phase6 --labels results/phase6/parser-self-labels.jsonl --trace-labels results/phase6/trace-source-labels.jsonl --historical-input results/raw/phase6-trace/historical-h25.jsonl --trace-report results/phase6/trace-summary.json --execution-ledger results/phase6/vast-execution.json --keyword-continuity results/phase6/keyword-continuity.json
```

Collection refuses to overwrite an existing generation file. Replay scoring and analysis require no paid resources. A fresh paid generation is not authorized by these replay commands.

## CPU verification environment

Phase 6 tests require no model weights or GPU runtime. The broad repository suite also runs older Phase 3 CPU Torch fixtures. This worktree installed `torch==2.13.0` as a local extra for those regressions, without changing the shared dependency files. Use `uv run --no-sync pytest` to preserve the extra; a fresh environment running the broad suite needs `uv pip install torch==2.13.0` after the standard development sync. Local CPU Torch is separate from the pinned CUDA Torch in the rented image.

The analysis CLI emits the deterministic parser export beside the summary. After verifying it against the canonical 80-reply sample, this run preserved that text under ignored `results/raw/phase6-parser-audit/analysis-audit-sample.jsonl`; it is not part of the published Git artifacts.
