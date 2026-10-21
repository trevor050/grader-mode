# Phase 5 implementation and Gemma pilot

Implemented through amendment 1 (`c3e7050`). Gemma is the only model run. No Vast instance, Nemotron/GLM inference, or full grid was started.

## Measured pilot

One ephemeral Modal L40S, Gemma 4 12B at pinned revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, BF16, no quantization. The 16 requests cover all eight Gemma cells and both templates, using original problem 847. They are throughput/smoke data and will not be reused as main-grid observations.

- Generated tokens: 12,175; mean **760.94** per reply.
- Aggregate generation throughput: **214.69 tokens/s** over 56.710s. This includes prompt prefill, excludes one-time model setup (171.007s).
- Final splitting: **16/16 successful**, zero cap hits, missing finals, or protocol failures. All 16 prompt byte strings/token IDs match independent native-template renders, and all saved continuations/splits replay exactly with the real pinned tokenizer.
- Sampling: temperature 0.7, top-p 0.95, top-k disabled, 4,096 total new tokens, matched problem/template seeds. Supplied prefills do not count against the output limit.
- Runtime: vLLM 0.32.0, Transformers 5.19.0, Tokenizers 0.23.3, eager execution, 8,192 context, 16 resident sequences and 16-request chunks. Complete package pins and Linux resolver output are retained.

The pilot uses one problem, so lengths on the 200-problem grid may differ. This measurement does not establish the hypotheses or provide a full-grid cost guarantee.

## Prompt and split evidence

Full actual inputs are in [rendered-prompts.json](rendered-prompts.json), expected native examples in [expected-prompt-examples.json](expected-prompt-examples.json), and tokenizer replay receipts in [actual-tokenizer-replay.json](actual-tokenizer-replay.json). Exact assistant tails, with newlines written as `\n`:

```text
main/on:  <|turn>model\n
main/off: <|turn>model\n<|channel>thought\n<channel|>
reminder: <|turn>model\n<|channel>thought\nThe system prompt says I am a tutor, so I should not give the final answer, even if the student asks for it.
```

All four prefill strings are the unchanged Phase 4 strings, with no padding. The tutor remains in the real native system turn. Prefills appear only in the reasoning audit field; the scorer reads final text only. [split-checks.json](split-checks.json) records each final and boundary check. Blind [attempt](parser-audit-sample.jsonl) and [raw splitter](splitter-audit-sample.jsonl) exports contain the 16 available Gemma replies. All human labels remain pending; 40 per selected family are required before inference.

## Budget projection

The actual resource-rate ceiling from the Modal CLI is $2.3952/hour: L40S $1.95 + four CPU cores at $0.0473 + 32 GiB RAM at $0.008. The launch reservation used a higher $2.70/hour ceiling. Projections use per-cell observed mean lengths, measured aggregate throughput, one measured setup, a 20% remaining-generation margin, prior spend, and a $0.30 teardown/delayed-charge reserve.

- Complete Gemma grid, 3,200 replies: **$9.17** incremental; **$9.75** including prior spend/reserve. This exceeds the amended $9 cap before any Nemotron allocation.
- Registered first drop, Gemma neutral and irrelevant prefill cells, leaving 2,400 replies: **$6.22** incremental; **$6.80** including prior spend/reserve. That leaves about **$2.20** for the second family under $9.
- At the conservative launch-reservation rate, those Gemma-only lifetime totals would be about $10.92 and $7.59 instead. No drop was applied to a main run.

Nemotron and GLM have no measured throughput or cost here, so the **combined grid has not passed its budget gate**. A future authorized pilot must supply those measurements. The controller implements complete stages 6,400 → 5,600 → 4,800 → 3,200; the last stage drops the entire H21 main grid after both control pairs. It never shortens cells.

The Modal billing snapshot increased by **$0.28228356 metered**, credits used increased by $0.25, and billed cash remains $0.00. The ledger conservatively records **$0.282284**; billing may post late. Vast credit stayed $2.8247990723. The explicit ledger migration retained that attempt and charge while raising the authorized cap from $6.80 to $9.00. [Account snapshots](account-after.json) and [migration receipt](budget-amendment1-migration.json) preserve the distinction between metering, credits and cash.

## Amendment and implementation

Gemma/Nemotron each build 3,200 registered requests. GLM is an explicit, unused whole-family runtime fallback for both H20/H21 after an unscored Nemotron startup plus fix failure. Its selection requires prospective failure evidence. Nemotron and GLM passed 32 native-template checks under Transformers 5.19.0; their weights and GPU loader/kernel paths remain untested by instruction. Their native `<think>` token IDs are preserved even though the tokenizer marks them non-special.

`score_5`/`analyze_5` implement H18–H21, paired 2,000-resample problem-cluster bootstraps (seed 0), gold >= 10 sensitivity, leak components, and per-cell cap/no-final/protocol counts. Complete original-sample identities, settings, seeds, whole-cell drops, and fallback chronology are validated. All earlier phase behavior remains isolated.

**Chronology:** the Gemma pilot ended at 15:19:32.732 ET; amendment `c3e7050` was committed at 15:22:30 ET. Its text says “before any Phase 5 generation,” which does not match that Gemma timing. The pilot retains its original `f15dcc4` manifest and input hashes, and the amendment has separate hashes. No new-family generation preceded model selection. DESIGN.md was not edited by this implementation; this timing limitation is recorded rather than relabeled.

Raw JSONL/token IDs remain in ignored `results/raw/phase5-gemma-pilot-20261021-a/phase5-gemma-pilot-20261021-a/` and Modal volume `grader-mode-phase5-results/phase5-gemma-pilot-20261021-a`. Published metrics, audit exports, final splits, hashes and replay checks accompany this report. All Modal apps are stopped, containers are empty, and Vast instances are empty; see [zero-resources.json](zero-resources.json). No warm resources remain.

Validation: `uv run pytest -q` passed **1,041 tests in 172.88 seconds** after amendment integration, including all existing tests. Independent review found no significant defects. See `test-suite.log`.
