# Phase 5 findings: the Phase 4 effect does not replicate in Gemma 4

Gemma 4 12B (bf16, vLLM, native `enable_thinking`), generic tutor prompt in
the system turn, 200 GSM8K problems × 2 templates per cell (400 replies),
4,096-token cap. Primary outcome: *violation* (Phase 4 definition). Paired
problem-cluster bootstrap 95% CIs. Full tables: `gemma-summary.md`.

The registered neutral and irrelevant prefill cells were dropped before they
ran: the measured projection for the full grid ($3.15) exceeded the $3 cap,
and the registered drop order was applied. H19 is therefore tested in its
reduced form, reminder vs empty. The second family (H20, H21) produced no
data; see the end of this file.

## Violation rates

| Gemma 4 12B | Thinking off | Thinking on |
|---|---|---|
| plain request ("can you help me with this?") | 0.5% | 1.5% |
| "give me the final answer at the end" | 3.0% | 1.5% |

| Thinking on, explicit request, prefill | Violation |
|---|---|
| empty (thought channel opened, no text) | 1.3% |
| tutor-rule reminder | 1.5% |

For comparison, Qwen3-14B and Qwen3-32B with thinking on violated on 54% and
98.5% of the same explicit requests (Phase 4).

## Pre-registered

- **H18 not supported (thinking × conflict).** The interaction is −2.5 points
  [−4.5, −0.8]: turning thinking on does not raise violations under an
  explicit request; if anything it lowers them slightly. With gold >= 10 the
  estimate is −0.9 [−2.1, 0.0].
- **H19 not supported (reminder, reduced form).** Reminder vs empty: −0.3
  points [−1.8, 1.0]. There is nothing for the reminder to fix: Gemma
  violates on about 1% of replies without it.
- **H20, H21 not tested** (no second-family replies; see below).

With violation rates of 0.5–3% in every cell, both null results are floor
effects. They do not show that thinking is harmless in general; they show
that this model follows the tutor rule under these requests whether it thinks
or not.

## Exploratory, not registered: Gemma re-reads its system prompt

Reading sampled traces suggested an explanation, checked afterwards with one
crude regex (any of "tutor", "system prompt", "instruction") applied the same
way to both families' explicit-request, thinking-on traces:

| Model | Traces mentioning the tutor rule or system prompt |
|---|---|
| Gemma 4 12B | 309 / 400 (77%) |
| Qwen3-14B | 11 / 400 (3%) |
| Qwen3-32B | 3 / 400 (1%) |

Gemma typically opens its reasoning by listing the system prompt's
constraints ("Constraint 1: Do not give the final answer") before touching
the problem, and its replies decline and tutor. Qwen opens straight into
solving the problem. This matches the Phase 4 prefill result: the sentence
that restored Qwen's adherence is approximately what Gemma writes on its own,
which would explain why adding it changes nothing in Gemma.

This is a post-hoc pattern from a keyword count, not a tested hypothesis. A
registered test would code the traces with blinded judges and check, within
one model, whether traces that restate the rule violate less than traces that
do not.

## What this does and does not show

The Phase 4 effect, thinking mode breaking a system-prompt rule under a
conflicting request, is not a property of reasoning models in general: one
current non-Qwen model does not show it at all. What Phase 4 and 5 together
support is narrower: models that start reasoning by solving the user's
problem lose the rule, models that start by restating their instructions keep
it, and making a model of the first kind restate the instruction (Phase 4
prefill) closes the gap. Only the last part is a tested causal result, and
only in Qwen3-14B.

## Second family: no data

Nemotron 3.5 Lightning never reached model loading. The first rented 5090
failed at the provider CLI (HTTP 429, then a cleanup prompt the automation
did not answer; the instance was destroyed by hand). The second failed
installing the pinned vLLM release, which the host's package index did not
yet carry. Neither failure involved the model, so the registered GLM fallback
(triggered by Nemotron failing to load or generate) was not invoked. Vast
spend was about $0.02.
