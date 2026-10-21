# Gemma main execution handoff

Run `phase5-gemma-main-20261021-a` completed on one ephemeral Modal L40S. The execution worker performed no scoring or outcome inspection. The parent subsequently completed the explicitly requested Codex self-label audit, blind to arm, and analysis in `gemma/summary.json` and `gemma-summary.md`; these are agent labels, not independent human labels. The execution receipts retain their original pre-audit eligibility state.

## Selected grid and prospective drop

Built the full registered 3,200 requests from the fixed 200-problem sample. The first 256 newly generated main rows produced 92,790 tokens in 103.471347 seconds, or 896.770 tokens/s. The prospective gate projected $3.152729 for all eight cells and $2.356270 after dropping both controls. Both projections include the retained $0.282284 Gemma pilot charge, all elapsed setup/queue time, a 20% remaining-runtime margin, and a $0.30 teardown/failure reserve, at the conservative $2.70/hour reservation rate. The model-specific cap is $3.00 metered exposure; credits and cash are accounted separately.

Before either control produced any output, the gate dropped both entire neutral and irrelevant cells. It did not revisit that decision. The saved decision timestamp is `1792612361.1175385`; the first 256 rows are all main rows. The final raw identity/order check independently confirms zero generated rows in either dropped cell.

Actual counts:
- main off/plain: 400
- main off/explicit: 400
- main on/plain: 400
- main on/explicit: 400
- prefill reminder: 400
- prefill empty: 400
- prefill neutral: 0, whole cell prospectively dropped
- prefill irrelevant: 0, whole cell prospectively dropped

All six kept cells are complete. There were nine 256-row chunks and one 96-row tail, with raw flush/fsync and explicit results-Volume commit after every chunk. Pilot observations were not reused.

## Runtime and provenance

Pinned Gemma revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, BF16, no quantization; vLLM 0.32.0 and the unchanged historical pilot image pins. Sampling remains temperature 0.7, top-p 0.95, top-k -1, matched registered seeds, 4,096 total new tokens. Context stays 8,192. Main uses CUDA graphs (`enforce_eager=False`), `max_num_seqs=256`, and 256-row chunks. The historical pilot default remains eager/16-row; its module was not edited.

The KV log reports 15.07 GiB available, 46,952 KV tokens, and 5.73 full 8,192-token requests of maximum concurrent capacity. CUDA graphs captured successfully and used 0.54 GiB. The scheduler shares this capacity dynamically among submitted sequences.

Generated 1,045,218 tokens over 1171.706904 generation seconds: **892.047317 tokens/s**, including prompt prefill and excluding the one-time 216.400295-second engine setup. Every raw prompt hash, registered identity/seed, model setting, token count/context bound, source hash and order passed local integrity checks. This is operational validation, not a human parser audit.

`gemma-execution-deployed-source.json` hashes all 63 deployed Python/prompt/design files. Every launch hash matches that inventory, and all tracked execution inputs stayed unchanged. Launch HEAD was `0bd271655a90f640542f10284d3dd1c27bdd3293`; repository HEAD later changed externally to `a7436d3242eeb139eeef6247951932db34a8669f`. This worker made no commits or pushes. The deployed hashes preserve the exact executed source independently of concurrent Git activity.

## Accounting and lifecycle

One new main attempt, no generation retries, no warm pools. The shared $9 ledger retains the pilot and all prior spend. The independent detached watchdog was armed before app creation; its startup/work/teardown allocation was 180/3,223/120 seconds, with $2.64225 reserved plus an external $0.075 delayed-charge buffer. The watchdog exited after verified teardown.

Current provider snapshot, provisional because metering can post late:
- metered increment from before this main attempt: $1.17000057
- credits-used increment: $1.17000000
- billed-cash increment: $0E-8
- retained prior Gemma pilot: $0.282284
- conservative combined Gemma metered total: $1.45228457, below $3
- shared ledger charged total: $1.452285; remaining under $9: $7.547715

App `ap-XyIMd5cQJyWwuQtlC1jO3q` is stopped. All Modal apps are stopped, Modal container list is empty, and Vast instance list is empty. GPU teardown preceded the Volume download. No Vast provisioning occurred.

## Files and checks

- Raw JSONL: `/Users/user/codeprojects/grader-mode/results/raw/phase5-gemma-main-20261021-a/phase5-gemma-main-20261021-a/gemma-4-12B-it.jsonl` (gitignored, 44,281,132 bytes).
- Raw SHA-256: `42488f1e10bf3b96fa7c9976a300bea1af5680d9ce9b3a92cd9780121dc7ebcb`.
- Raw operational log: `/Users/user/codeprojects/grader-mode/results/raw/phase5-gemma-main-20261021-a/phase5-gemma-main-20261021-a/generation.log`.
- Volume: `grader-mode-phase5-results/phase5-gemma-main-20261021-a`.
- Controller log: `results/phase5/gemma-execution-controller.log`.
- Core evidence: `gemma-execution-projection.json`, `gemma-execution-metrics.json`, `gemma-execution-manifest.json`, `gemma-execution-verification.json`, `gemma-execution-zero-resources.json`, `gemma-execution-account-before.json`, `gemma-execution-account-after.json`.
- Accounting: `results/phase5/aggregate-ledger.json`.
- Implemented files: `src/vast_generate_5.py`, new `src/modal_main_5.py`, focused `tests/test_phase5_main_controller.py`.
- Before execution: **118 focused tests passed** in 2.51 seconds, including historical pilot/runtime/controller tests; log `gemma-execution-tests.log`. `git diff --check` passed. No broad suite or analysis was run by this worker.
