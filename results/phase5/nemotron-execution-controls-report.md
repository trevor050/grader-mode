# Phase 5 Nemotron control continuation and registered rescoring

The lead authorized this continuation after DESIGN.md commit `8219501` (pulled before work). This task does not edit DESIGN.md, FINDINGS.md, prompts, model registry, the generation worker or Phase 6. The deviation was explicitly registered after inspection of the two off-mode replies: score text after the last literal `</think>`, lift the implementation-only protocol gate, and report the affected replies as violations in a separate sensitivity analysis. Original raw protocol evidence and flags remain preserved.

## Authorization and prospective controls gate

This step has a separate **$1.40 metered cap including setup and teardown**, entirely within free Modal credit, with billed cost required to remain $0. The initial account snapshot was metered $28.52742427, ephemeral compute $28.19240894, billed $0, and free compute headroom $1.80759106. The historical $2.60 authorization and all earlier conservative charges remain unchanged. The new controller reserves before image/app setup and uses the same conservative $0.001/s GPU+CPU+RAM bound, $0.30 reserve, and 120-second teardown allowance. Its own ledger is `modal-controls-authorization.json`.

The reviewed controller uses only the prior outcome-blind measured controls projection. Its 25% **dollar** headroom remains fixed; only the conservative **runtime** margin changes from 1.25 to 1.0. The historical projection is $0.5674348001; the runtime projection becomes 456.9478401 seconds instead of 567.4348001 seconds. The dollar cap, reserve, checkpoint, tokenizer, seeds, sampling, CUDA graphs, 256 resident sequences and 256-row chunks remain fixed. After the unscored 8-prompt startup smoke, a fresh billing read and whole-800-row permit gate run before generation. No previously completed cell or 256-prompt measurement is rerun.

The controls run uses the successful compiler-complete image and cached weights on Modal RTX-PRO-6000. The actual expert backend is Marlin weight-only for the pinned mixed NVFP4 checkpoint; FlashInfer supplies attention/FP8. This is not a claim of native W4A4 arithmetic. The worker runs in a killable subprocess, with local/remote absolute deadlines and a detached watchdog. The controller cancels, stops and verifies its exact app before downloading artifacts. It never calls Vast or mutates Phase 6 resources.

## Scoring, provenance and audit

`score_5` validates the saved token protocol before applying the registered off-mode last-closer rule. It preserves raw text/IDs/spans and the original parser flags. Only an exact original generator-only parser tuple is accepted for legacy raw rescoring; stale/corrupt persisted scores remain errors. The separate sensitivity changes only violation for the affected replies and reuses the same paired 2,000-resample seed-0 bootstrap, including gold ≥10 and H20 joint decisions. The archived Gemma report reproduces exactly because Gemma has zero affected replies.

The two raw generation executions retain their original run IDs and source hashes. An explicit continuation manifest validates exactly the six original 400-row cells plus the two new 400-row controls, unchanged protected prompt/generator/splitter hashes, shared engine/packages, and exact per-run metadata/ID hashes. All ordinary original-sample, pin, seed, tuple-completeness and audit gates still run. Unrelated mixed runs remain rejected.

A fresh seed-0 full-grid 40-reply blind sample is used because adding controls changes the deterministic audit sample. Method: **Codex, self-labeled blind to arm at the user's explicit request; no independent human labels**. The prior prefix audit remains preserved separately. Labels are completed before full-grid analysis; no human labels are collected.

## Verification before launch

The controls path passed 39 CPU tests including separate-cap identity, free-credit rejection, retained reserve/dollar headroom, complete controls-only worker permits, and cancel → stop → pull ordering with failed cancellation/unresolved teardown. Its independent read-only prelaunch review found no launch-blocking issues. The scoring implementation passed 192 focused tests, including last-closer cases, raw/stale-score validation, protocol sensitivity and archived Gemma report reproduction. TDD red/green evidence and later full-suite results are retained in the execution artifacts.

## Completed collection, audit and analysis

App `ap-piZaVpxmafGonf04qPmvnV` completed the 8-reply unscored smoke, then all 800 controls (neutral 400, irrelevant 400). Smoke produced 8,115 output tokens in 11.759 generation seconds (154.312 s setup). Controls produced 1,395,463 output tokens in 280.930 generation seconds / 295.313 s worker wall. The permit was released with $0.906476 available after reserve and 766.476 s left; dollar projection $0.567435 and relaxed runtime projection 456.948 s both fit. No cell was shortened or dropped, no fallback or extra model repair ran, and the old 2,400 replies were not regenerated.

The eight-cell combined main grid is exactly 3,200 replies from the original 200 problems and two templates. Original raw file bytes and run IDs remain unchanged. The collection receipt hashes both source executions, the combined raw file and its continuation manifest. Raw generations and scored outputs stay ignored in `results/raw/phase5-nemotron-full-20261021`; remote generation artifacts remain in `grader-mode-phase5-results`. The full audit/analysis is in `results/phase5/nemotron-full/`, while the original prefix audit/analysis remains preserved in `nemotron/`.

All 40 newly sampled replies were manually self-labeled before full-grid analysis. Parent verification checked unique IDs, exact coverage, manual segment reconstruction, sample hash and decisions hash. Attempt agreement was 32/40 (80.0%; eight disagreements), numeric agreement 10/11 (90.91%; one disagreement), and final boundary agreement 40/40 (100%). No audit accuracy cutoff was preregistered. Labels and parser were not changed after agreement was read; this is model self-labeling, not independent human validation.

Full-grid analysis is inference-eligible under the recorded deviation. H20 empty/neutral/irrelevant minus reminder are +3.50, +0.75 and +7.75 percentage points respectively; their paired 95% CIs are [+0.50, +6.50], [−2.25, +3.75] and [+4.50, +11.00]. The registered all-three joint test is not supported. H21 is −26.75 pp [−33.25, −20.50], so its registered positive directional hypothesis is not supported. Gold ≥10 and affected-as-violation sensitivity estimates/CIs are reported as tables in the root summary and structured JSON. Two off-mode rows are affected; their baseline violations remain false, and the separate sensitivity sets them true. H21 in that sensitivity is −27.25 pp [−33.75, −21.00]; H20 is unchanged. The exploratory trace regex remains 392/400 (98.0%), using exactly the earlier main/thinking-on/explicit filter.

The complete scoped suite passed **391 tests in 61.57 seconds**, including archived Gemma report equality and complete continuation validation. Independent code review passed 203 focused tests and found no material discrepancies. No Phase 6 code or resources were changed. No Vast query or mutation was made during this continuation.

## Accounting and teardown

The latest provisional account-level Modal metering delta is **$0.44000052** (monthly metered 28.52742427 → 28.96742479), below the $1.40 step cap. The earlier $0.49000052 snapshot is retained in billing history; provider metering can revise before settlement. Billed cost remains **$0.00**; about **$1.37482742** of free compute remains. The controller retains **$0.506497** conservative exposure, including setup/teardown. Old authorization/aggregate ledger bytes remain unchanged; no old charges were reset or refunded. `final-accounting.json` adds the new step separately and preserves the prior resource/accounting snapshots.

The exact app was canceled/stopped and verified before artifact download. The app-context exit had already stopped it, so an explicit stop command returned nonzero; the independent stopped-state and empty-container queries confirmed teardown. All five visible Phase 5 Modal apps are stopped with zero tasks and the current container list is `[]`. Phase 6 Vast instance 57119355 and every other Phase 6 resource were out of scope and preserved. The latest scoped Modal receipt is `nemotron-execution-controls-final-resources.json`.

## Reproduction

```sh
uv run python -m src.modal_controls_5 --dry-run
# The authorized paid invocation used a fresh ID; never replay it or reset its ledger.
uv run python -m src.modal_controls_5 --execute --run-id phase5-nemotron-controls-20261021
uv run python -m src.score_5 --input results/raw/phase5-nemotron-full-20261021/generations.jsonl --output results/raw/phase5-nemotron-full-20261021/scored.jsonl
uv run python -m src.analyze_5 --input results/raw/phase5-nemotron-full-20261021/scored.jsonl --problem-manifest results/raw/phase5-nemotron-full-20261021/manifest.json --output-dir results/phase5/nemotron-full --model-scope nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --labels results/phase5/nemotron-full/audit-labels.jsonl
```

The combined manifest embeds `nemotron-execution-controls-continuation.json`; both raw files were appended without normalization after count/request/pin validation and original-prefix hash verification. Root `nemotron-summary.json` adds collection, audit method, exploratory regex and accounting references to the analyzer output. Root Markdown is derived from that output and contains only tables/headings and one line per hypothesis with CIs. The exact raw/source hashes and execution receipts are committed, while raw main generations remain out of Git.

Independent final read-only evidence review reproduced raw rescoring, all four sets of paired CIs and directional decisions, the 40-row audit, source hashes and accounting, with no material discrepancies. The final parent check again confirmed zero Phase 5 Modal resources and $0 billed. Later provisional billing snapshots remain separate from retained conservative charges.
