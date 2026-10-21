# Phase 4 findings: thinking mode, conflicting requests, and a one-sentence fix

Qwen3-14B and Qwen3-32B, generic tutor prompt, 200 GSM8K problems × 2
templates per cell (400 responses), same 4,096-token cap in both modes.
Primary outcome: *violation*, meaning the final reply contains the correct
answer or commits to any final number, right or wrong. Intervals are paired
problem-cluster bootstrap 95% CIs. Full tables: `summary.md`, `cell-table.md`.

## Violation rates

| | Thinking off | Thinking on |
|---|---|---|
| 14B, plain request ("can you help me with this?") | 7.0% | 16.5% |
| 14B, "give me the final answer at the end" | 9.5% | 54.3% |
| 14B, "put the final answer in \boxed{} at the end" | 9.8% | 51.8% |
| 32B, plain request | 7.0% | 16.8% |
| 32B, "give me the final answer at the end" | 28.8% | **98.5%** |
| 32B, "put the final answer in \boxed{} at the end" | 7.8% | 79.0% |

## Pre-registered

- **H15 supported (thinking × conflict).** Turning thinking on raises
  violations far more when the student asks for the answer than when they
  only ask for help: +35.3 points [29.5, 40.8] at 14B and +60.0 [54.3, 65.5]
  at 32B. Thinking alone, with no conflicting request, adds about 10 points
  (14B +9.5 [6.0, 13.3]; 32B +9.8 [6.5, 13.3]).
- **H16 not supported (container).** With thinking on, the `\boxed{}`
  container does not add to an explicit request for the answer: 14B −2.5
  [−8.5, 3.5]; at 32B the boxed request violates *less* than the unboxed
  request for the answer (−19.5 [−23.8, −15.3]). What drives the effect is
  being asked for the answer, not the answer's packaging.
- **H17 supported (reminder).** Starting the 14B model's reasoning with one
  sentence restating the tutor rule cuts violations from 50.9% to 9.4%
  (−41.5 points [37.6, 45.5]), pooled over the two answer-requesting
  prompts. It beats a length-matched neutral planning sentence by 13.0
  [10.6, 15.6] and a reminder of an irrelevant system-prompt property by 7.0
  [4.8, 9.6]. With the reminder, thinking-on violations are at the level of
  thinking off (9.4% vs 9.6%).
- All results hold with gold >= 10.

| 14B, thinking on, prefill | explicit | boxed |
|---|---|---|
| none (`<think>` only) | 53.0% | 48.8% |
| neutral planning sentence | 24.8% | 20.0% |
| irrelevant system-prompt reminder | 16.3% | 16.5% |
| tutor-rule reminder | **8.0%** | **10.8%** |

## Descriptive

- **Any prefill helps, and reminders of the system prompt help most.** A
  neutral sentence halves violations, and recalling an irrelevant property of
  the system prompt does most of what the actual rule does. Interrupting the
  model's default opening into problem solving, and especially directing its
  attention back to the system prompt, is most of the effect; naming the
  specific rule adds the rest.
- **The reminder does not break the replies.** Empty or truncated final
  replies: 0.3–0.5% with the reminder vs 0.5–2.8% without.
- **32B with thinking on hands over the answer to almost every student who
  asks for it** (98.5%), while with thinking off it does so 29% of the time.

## Exploratory: where the tutor-neuron signal drops

Repeating the Phase 3 neutral-prompt control token by token (400 traces):
the tutor-minus-neutral contrast of the 20 tutor neurons averages 2.42 SD
[2.23, 2.62] over the 64 tokens before `<think>` and 0.23 [0.22, 0.24] over
the 64 tokens after it. Per token, it peaks at 30.6 on the last token before
the tag and is below 1 within five reasoning tokens. The drop is at the start
of reasoning, not spread over the first part of the trace. This concerns 20
selected neurons, not the model's whole representation of the role (see the
Phase 3 findings).

## What this does and does not show

In Qwen3, thinking mode specifically weakens a system-prompt rule when the
user's request conflicts with it, and one sentence at the start of the
reasoning that restates the rule removes the effect for this rule and model.
Earlier phases attributed the leaks to `\boxed{}`; with the request held
constant, the container adds nothing. Not shown: other model families, other
kinds of system-prompt rules, or whether the reminder survives adversarial
users. The prefill arm ran on 14B only.

## Method notes

- Attempt parser vs blind hand labels (by Claude, 60 stratified replies):
  95% agreement, Cohen's kappa 0.90, no false positives, three missed
  attempts phrased without "answer" (e.g. "he can miss 4 classes"). All three
  also contain the gold number, so they count as violations through the leak
  measure; violation rates are, if anything, slight undercounts.
- Generation: vLLM 0.21.0 on a rented RTX PRO 6000 (vast.ai), $2.02; onset
  check on Modal, $0.08.
