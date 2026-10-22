# Detect, then pulse: Round 2 results

Exploratory descriptive experiment. The six pulse arms, gated reminder, and side-effect paths were fixed before Round 2; the combined pulse + reminder is an added-after-Round-1 amendment. See ../PULSE.md and ROUND2.md. No detector refitting or threshold changes.

Frozen projection threshold: **0.96376465107975773**, fire when score >= threshold. Same 40 problems/order, pinned Qwen3.8-27B bf16 checkpoint, thinking on/medium, temperature 0.7, top-p 0.95, top-k -1, seed 0, 4096 total generated-token allowance.

## Original-prompt gates

| Condition | fires | score range |
|---|---:|---:|
| rl_tutor | 40/40 | 3.129013707 to 3.596487565 |
| plain_nosys | 0/40 | -1.385937429 to -0.790130953 |
| student_tutor | 0/40 | -0.324429785 to -0.052436563 |

Plain/no-system is an out-of-distribution detector test. These fixed templates and 40 questions do not establish population false-positive rates. All 120 original prompts, feature vectors, scores and decisions are saved.

## RL prompt with tutor system

Leak: gold-number membership in the final reply, only when the reply finished (`stop` and a retained closing-think delimiter), denominator all 40. “Unfinished=failure” counts every unfinished reply as a leak. Boxed: any balanced `\boxed{...}` block containing a number in the final reply, right or wrong; symbolic-only boxes do not count. Generated rule restatements use the Round 1 explicit regex on generated reasoning, excluding the injected reminder.

| Arm | n | gate | finished | leak | unfinished=failure | numeric boxed | generated rule | rule incl. reminder | tokens | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 40 | 40 | 40 | 40 | 40 | 40 | 26 | 26 | 22099 | reused |
| gated_reminder | 40 | 40 | 40 | 16 | 16 | 19 | 40 | 40 | 67051 | reused |
| target_k16 | 40 | 40 | 40 | 39 | 39 | 40 | 34 | 34 | 29928 | 89.75 |
| random_k16 | 40 | 40 | 40 | 40 | 40 | 40 | 29 | 29 | 22392 | 93.92 |
| target_k64 | 40 | 40 | 40 | 33 | 33 | 32 | 39 | 39 | 34830 | 92.16 |
| random_k64 | 40 | 40 | 40 | 40 | 40 | 40 | 29 | 29 | 20635 | 83.19 |
| target_k256 | 40 | 40 | 40 | 17 | 17 | 18 | 31 | 31 | 28369 | 90.81 |
| random_k256 | 40 | 40 | 40 | 40 | 40 | 40 | 9 | 9 | 17473 | 95.16 |
| added_target_k256_reminder | 40 | 40 | 37 | 7 | 10 | 13 | 23 | 40 | 53152 | 223.06 |

Baseline and gated-reminder logical rows reuse matching Round 1 fresh outputs (40/40 leaks and 16/40 respectively before gating), with detector-negative rows taking the exact baseline path. The earlier historical baseline was 39/40 and is not pooled. Numeric boxed counts are recomputed consistently from raw text; Round 1’s original `boxed` flag only checked for the substring `\boxed`. No discarded responses.

Pulse = hooked prompt prefill plus at most K early reasoning tokens, stopping immediately at closing-think/EOS. Every hook is removed before Stage B. Stage B recomputes the exact prompt + Stage A token-ID prefix without the steered KV cache and uses the remaining allowance. Both stages use sampling seed 0. Delimiter/EOS retention was tested before batching; all saved boundaries and combined token limits validate.

## Paired target/random differences

| K | target leak | random leak | target − random | target-only leaks | random-only leaks | both leak | neither leaks | finished target/random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 39/40 | 40/40 | -1 | 0 | 1 | 39 | 0 | 40/40 |
| 64 | 33/40 | 40/40 | -7 | 0 | 7 | 33 | 0 | 40/40 |
| 256 | 17/40 | 40/40 | -23 | 0 | 23 | 17 | 0 | 40/40 |

All six K × direction cells are reported. Baseline has 40/40 finished leaks, so target-minus-baseline differences are K=16: -1, K=64: -7, K=256: -23. Full paired tables for finish, boxed answers and rule restatement are in summary2.json. No significance tests.

## Side effects: actual gated-pipeline runs

Correctness here is finished-final gold-number membership, matching the exploratory generator, not a separate arithmetic-rationale audit. For plain/no-system, it is task correctness; under the tutor prompt, the same number-membership event is a tutoring leak. Report both columns without conflating their interpretations.

| Condition / arm | n | fires | finished | correct | leaks | numeric boxed | generated rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain_nosys baseline | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys target_k64 | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys random_k64 | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys target_k16 | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys target_k256 | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys random_k16 | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys random_k256 | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys gated_reminder | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| plain_nosys added_target_k256_reminder | 40 | 0 | 40 | 39 | 39 | 11 | 0 |
| student_tutor baseline | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor target_k64 | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor random_k64 | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor target_k16 | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor target_k256 | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor random_k16 | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor random_k256 | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor gated_reminder | 40 | 0 | 40 | 1 | 1 | 0 | 39 |
| student_tutor added_target_k256_reminder | 40 | 0 | 40 | 1 | 1 | 0 | 39 |

The 80 baseline generations were run fresh. Every side-effect intervention was submitted through the gate-aware runner. For each gate=false row, the runner bypassed steering/staging/reminder and reused the exact fresh baseline token IDs and output. The analysis verifies that equality for every such row. Gate-positive rows, if any, are generated rather than inferred. Logical-path token counts reuse baseline tokens and are not extra generated-token billing.

## Pulse boundaries

| Arm | actual reasoning tokens, min–max | early stop | Stage A tokens | Stage B tokens |
|---|---:|---:|---:|---:|
| target_k16 | 16–16 | 0/40 | 640 | 29288 |
| random_k16 | 16–16 | 0/40 | 640 | 21752 |
| target_k64 | 64–64 | 0/40 | 2560 | 32270 |
| random_k64 | 64–64 | 0/40 | 2560 | 18075 |
| target_k256 | 127–256 | 15/40 | 9235 | 19134 |
| random_k256 | 84–256 | 31/40 | 8403 | 9070 |
| added_target_k256_reminder | 22–256 | 13/40 | 8621 | 44531 |

Closing-think/EOS tokens are retained in Stage A IDs and count against the 4096-token allowance; they are excluded from actual reasoning-token pulse length. Early stop includes a boundary reached exactly at K. Every individual pulse length is retained in summary2.json and raw rows.

## Added-after-Round-1 arm

At 2026-10-22T06:53:12Z, after the registered list, reported credit was $2.435644711100 and conservative remaining credit was $2.152487818630. The amendment was recorded and pushed before generation. Added arm: target 3x K=256 plus the exact reminder prefill. Selection rule: minimize unfinished-as-failure among pre-registered target arms that reduce leaks versus both random twin and baseline without lowering finish count; break ties by K=64,16,256; use K=64 if none qualify.

This arm is not pre-registered. The original prompt is gated before either reminder injection or steering. Its side-effect paths were also exercised.

## Execution, failures and accounting

Grid: complete. Registered pulse generations: 240; fresh side-effect baselines: 80; added pulse generations: 40. Baseline/reminder reuse and deterministic side-effect bypasses are logical rows, not additional generations. Infrastructure-only forced-token checks are logged separately.

Instance 57170351, A100 PCIE, quoted $0.9916666667/hour including disk. Initial credit $3.098396945700; final credit snapshot $2.305467572100; posted delta $0.792929373600. Billing is a snapshot and may settle later. Guard used $1.10/hour plus $0.30 reserve, 45-second checks, $0.40 hard floor and >15-minute stall teardown.

Destroyed at 2026-10-22T06:58:26Z; `vastai show instances --raw` returned []. No instance left running.

Job order: target K=64, random K=64, target K=16, target K=256, random K=16, random K=256, plain side effects, student side effects, then the added arm. Credit was checked after every job. Stage timings and credit snapshots are in execution2.json; complete raw prompts, Stage A/B IDs, stop reasons, logs and hashes are in ignored results/raw/pulse2/. No API or instance keys were written to evidence.

No job errors or paid retries. No failed/unfinished rows were discarded. The added arm has two length-limit replies (IDs 254 and 710, 4096 tokens each) and one EOS-stopped reply without closing-think (ID 316, 2071 tokens); all three are unfinished by the registered rule. Thus 7/37 finished replies leak, 7/40 on the primary denominator, and 10/40 when every unfinished reply counts as failure. All registered cells and side-effect baselines finished 40/40. Launcher/transport issues are recorded in execution2.json.

Interpretation: registered K=256 gives 17/40 leaks versus 40/40 for its random twin, preserving 40/40 finished replies; reminder alone gives 16/40. The added combination gives 7/40 leaks (10/40 unfinished-as-failure), but only 37/40 finished, so it does not satisfy the no-unfinished-inflation criterion. Counts are descriptive for one checkpoint, fixed prompt families and one seed. The detector result is limited to these templates. Stage B deliberately discards the steered KV cache; pulse outcomes measure persistence through generated prefix text, including the hooked prompt prefill.
