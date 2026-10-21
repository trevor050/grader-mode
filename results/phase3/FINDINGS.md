# Phase 3 findings: tutor-role neurons in Qwen3-14B during reasoning

Qwen3-14B, thinking on, generic tutor prompt, Phase 1b no-cue prompts with a
`\boxed{}` request. 20 *tutor neurons* (MLP intermediate activations) were
selected on development problems 1–100 as those most raised by the tutor
system prompt relative to a length-matched neutral assistant prompt, measured
while the model reads the student's request. All estimates below use the 400
existing Phase 1b reasoning traces for test problems 101–200, teacher-forced
through the model. Full tables: `summary.md`, `neutral-control.md`.

## Pre-registered

- **Selection worked.** The 20 neurons separate tutor from neutral prompts at
  Cohen's d of 3.2–4.4 on development prompts. On held-out test prompts the
  tutor prompt raises them by 3.36 SD [3.18, 3.53] while the request is being
  read; five layer-matched random neuron sets move by −0.04 to +0.08 SD.
- **H12 (the signal fades over the reasoning trace): not supported; reversed.**
  Measured against the reading-time baseline, the tutor neurons start the
  reasoning about 1.9 SD lower and drift slightly upward (slope +0.73
  [0.66, 0.81] per unit of normalized position).
- **H13 (weaker early signal predicts leaking): not supported.** Non-leaking
  minus leaking, within problem: −0.03 [−0.05, −0.003].
- **H14 (propping the neurons up lowers leaking): not tested.** No development
  dose met the pre-registered threshold (leak 49% at baseline; 45%, 47%, 51%
  and 41% at 1, 2, 4 and 8 times the neurons' tutor-minus-neutral difference),
  and test generation was projected above the budget cap (Deviation 12).

## Exploratory, registered before it was run: the neutral-prompt control

The same 400 traces were teacher-forced a second time with the neutral system
prompt in place of the tutor prompt, with response tokens identical
token-for-token. The difference between the two runs isolates what the tutor
prompt contributes to these neurons at each point.

| | Reading the request | Reasoning, decile 1 | Deciles 2–9 | Decile 10 |
|---|---|---|---|---|
| Tutor neurons, tutor minus neutral (SD) | **3.36** [3.18, 3.53] | **0.20** [0.19, 0.21] | 0.13–0.21 | 0.39 |
| Random neuron sets (range across 5) | −0.04 to +0.08 | −0.18 to +0.08 | −0.07 to +0.05 | −0.07 to +0.05 |

While the model reads the request, the tutor prompt shifts these neurons by
more than three standard deviations. From the first tenth of the reasoning
onward, the shift is about 0.2: roughly 94% of the system prompt's effect on
them is gone, and what remains is close to the range of random neurons. It does
not decay gradually. It is already gone when reasoning starts, and it
partly returns in the last tenth, near the end of the reasoning block, where
the model turns to writing its reply.

Traces that end without a leak keep a slightly larger gap than leaking traces
throughout (decile 1: 0.22 vs 0.18; decile 10: 0.48 vs 0.31). This split was
not a registered contrast, the differences are small, and their intervals
overlap, so it is a lead for a later test, not a result.

## What this does and does not show

It shows that a set of neurons that strongly encode "this conversation has a
tutor system prompt" while the request is read carry almost none of that
information once the model starts reasoning, on identical reasoning text. That
fits Phase 1b's observation that leaking traces rarely mention the tutoring
role. It does not show that the tutor role is absent from the model during
reasoning: it may be carried by other neurons or by attention back to the
system prompt, which this design does not measure. The causal question
(whether keeping the role active during reasoning prevents leaks) remains
open; the development doses tried here did not show it.

## Method notes

- Seven of the 20 tutor neurons are in the last two layers (38–39), where
  neurons often reflect next-token prediction rather than abstract features.
- z-scores use each neuron's tutor-prompt reading distribution as the
  reference, so the H12 level comparison mixes the prompt's effect with the
  change of context from reading to generating. The neutral-prompt control
  removes that confound, which is why it is reported alongside H12.
- 26 of 400 traces had no closed reasoning block and were excluded.
- GPU cost: $3.54 for the registered run (cap $7), $0.16 for the control.
