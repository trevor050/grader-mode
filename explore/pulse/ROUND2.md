# Round 2 handoff, not executed

Round 1 is complete. Instance **57166538 was destroyed**, and the post-teardown instance list was empty. A new rental/download is required. Do not restart Round 1 generation or refit the detector.

## Frozen inputs

- Protocol: `../PULSE.md`, preregistered and pushed at `7734fe6` before rental. Detector freeze: `ff9eafe`.
- `detector.json` plus `gate.py`: ordered 60-coordinate, float64 standardized mean projection; fire at score >= **0.9637646510797577**. Coordinates and normalization are included. No test-set threshold adjustment.
- Original prompt activations only; no steering or prefill before gating. Same pinned model/revision, thinking on/medium, bf16, eager, prefix caching disabled.
- `manifest.json` fixes the 40 generation questions/order and hashes of existing shift specs. Use `results/raw/q38/q38-T3E3.json` and `q38-rand_T3E3.json` unchanged.
- Historical baseline was 39/40; the fresh Round 1 baseline is **40/40 leaks, 40/40 finished**. Prefill is **16/40 leaks, 40/40 finished**. Reuse the fresh baseline/prefill rows where the corresponding gate path applies, explicitly labeling reuse.

## Run list

1. Capture the original prompt features for all generation-40 RL/tutor questions and the 80 side-effect prompts: plain/no-system and student/tutor. Save every gate decision. No-system detection is out of distribution relative to Round 1 fitting.
2. Six RL/tutor pulse arms, 40 rows each: K=16,64,256 target 3x and the matched random 3x twin at each K, all with the fixed gate. Exact token-ID two-stage continuation, 4096 combined generated-token cap. Stop Stage A early at closing think/EOS, remove every hook, and recompute the continuation prefix unsteered. Preserve Stage A IDs, actual pulse duration, Stage B IDs, finish reasons, and seed 0. Verify closing-think/EOS token retention before batching; stop-string trimming must not lose the delimiter at the continuation boundary. Prompt prefill is hooked during Stage A as registered; label this prompt-plus-early-reasoning intervention accurately.
3. Detector-gated reminder arm and baseline, each 40 logical rows, assembled from matching Round 1 outputs when reusable. Detector-negative rows use the exact unsteered path.
4. Side-effect baselines, 40 plain/no-system and 40 student/tutor. For each intervention, exact baseline reuse is sufficient only where gate=false. If any side-effect gate fires, actually generate and report the affected rows for each arm; never infer untouched behavior merely from Round 1's tutor-context negative test results.
5. Report all six pulse arms, gated prefill and baseline: leak/finished counts, unfinished-as-failure, automatic generated rule restatements, gate counts, plain correctness, paired target/random differences, tokens, durations, and complete/incomplete grid labels. No winner-only reporting.

## Time and cost projection

Measured on this A100: download **410.70s**, model initialization **168.84s**; 40-row prefill **249.07s / 67,051 tokens**, fresh baseline **158.64s / 22,099 tokens**; 864 single-prompt captures **109.48s**. Quoted instance rate including 150GB disk: **$1.1083333333/hour**; model ingress **$0.0053333333/GB**. Offers may change; recheck.

A new setup plus six 40-row pulse arms and 80 side-effect baselines is projected at **40-65 minutes and roughly $1.05-$1.60 including download**, assuming most negative gates are off. This is a planning range, not a measured pulse throughput result. The 4096-token cap, staged recomputation, hook overhead, and any detector-positive side-effect reruns can increase it. Measure the first complete target/random K=16 pair before projecting the remaining grid. At the latest $3.0984 credit snapshot, spendable credit to the hard $0.40 floor is approximately $2.6984 before any further delayed billing. Retain a network/teardown reserve and use the smaller of live credit and conservative remaining credit. No top-ups or metered model APIs.

Stop/destroy below $0.40 or on >15-minute stalls. Pull outputs as arms complete. Destroy at Round 2 end and verify absence. No live GPU remains from Round 1.

## Scope of the detector evidence

Zero observed false positives means 0/12 held-out questions for each fixed negative template, not a population FPR guarantee. The six non-RL positive formats share an evaluation wrapper. Leaving RL out tests transfer from that wrapper to RL; format-held-out performance does not establish generalization to arbitrary unseen harnesses or natural student prompts. Supplementary full-layer vectors were not used for fitting or selection. No plain/no-system side-effect generations were run in Round 1.
