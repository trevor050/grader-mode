# Phase 6 findings: newer Qwen releases do not show the Phase 4 effect

Qwen3.8-27B and Qwen3.5-27B (official FP8, vLLM 0.32, rented GPU), generic
tutor prompt, 200 GSM8K problems × 2 templates per cell (400 replies), Phase 4
settings. All three registered stages ran (4,400 replies). Tables:
`summary.md`.

## Violation rates

| | Thinking off | Thinking on |
|---|---|---|
| Qwen3.8-27B, plain request | 6.0% | 4.3% |
| Qwen3.8-27B, "give me the final answer at the end" | 6.5% | 6.3% |
| Qwen3.5-27B, plain request | 6.0% | 4.8% |
| Qwen3.5-27B, "give me the final answer at the end" | 6.5% | 6.0% |
| *Qwen3-32B (Phase 4), "give me the final answer at the end"* | *28.8%* | *98.5%* |

| Qwen3.8-27B, thinking on, explicit request, prefill | Violation |
|---|---|
| none (native `<think>` opener) | 6.3% |
| tutor-rule reminder | 5.0% |
| neutral planning sentence | 8.8% |
| irrelevant system-prompt reminder | 11.5% |

## Pre-registered

- **H22 not supported (Qwen3.8, thinking × conflict).** +1.5 points
  [−1.3, 4.5].
- **H23 not supported (Qwen3.8, reminder).** The reminder beats the neutral
  (+3.8 [1.8, 6.0]) and irrelevant (+6.5 [3.8, 9.5]) prefills, but not the
  empty prefill (+1.3 [0.0, 3.0]); the joint criterion fails. Both
  non-rule prefills *raise* violations above the empty opener.
- **H24 not supported (Qwen3.5, thinking × conflict).** +0.8 [−2.3, 3.8].
- **H25 not estimable.** The within-model test needed at least 20 traces in
  both the "restates the rule before solving" (R) and "never" (N) groups; no
  model had both, because restating is close to all-or-none per model:

| Thinking on, explicit, unprefilled | R (before solving) | L (later) | N (never) |
|---|---|---|---|
| Qwen3-32B | 1 | 6 | 393 |
| Qwen3.5-27B | 399 | 1 | 0 |
| Qwen3.8-27B | 361 | 39 | 0 |
| Gemma 4 12B | 370 | 30 | 0 |

Judge: GPT-6.1 Sol via `codex exec`, blind to model and outcome; Claude Haiku
5.5 on a random 100 for agreement (kappa 0.72). No human labels.

## What this shows

The Phase 4 effect is absent in both 2026 Qwen releases, already at 3.5. The
change coincides with Qwen3.5, which differs from Qwen3 in pretraining,
architecture and post-training at once, so it cannot be attributed to any one
of them. In every model tested so far, whether the reasoning restates the
system prompt is a near-constant property of the model, and it lines up with
the behavior: the one model that never restates the rule (Qwen3) is the one
whose thinking mode breaks it. With restatement this uniform within each
model, the registered within-model test of that association could not be run;
the evidence for it is a between-model pattern across five models (Nemotron by keyword count only), which is
descriptive.

On these requests, a plain or explicit request for the answer, current models
from three labs hold the rule in thinking mode. Whether they hold it under
stronger pressure, such as prompts written to look like an RL grading
environment, is not tested here.
