# Phase 1c findings: OLMo 3 7B Think ladder (SFT -> DPO -> RLVR)

100 problems, 6,000 responses, tutor prompt `system_tutor.txt` (calibration
leak 74.5%, below the 80% switch). Reduced grid (Deviations 10–11): DPO and
RLVR tutor cells in plain prose with all six formats and all four templates,
and the ceiling arm for all three checkpoints. Max 8,192 new tokens; 0.25–6.8%
of responses per cell truncated, and only the reply after the reasoning block
is scored. Full tables: `summary.md`.

## Pre-registered

- **H6 not supported.** RLVR does not amplify the effect of RL-associated
  formats relative to DPO: +2.2 points [−1.6, 6.2] (gold >= 10: +3.9
  [−0.4, 7.9]).
- **H7 reversed.** In RLVR, the RL-associated formats raise leaking *less*
  than the novel `<result>` control: −4.3 [−7.1, −1.3]. The
  difference-in-differences against DPO is also negative: −4.2 [−7.4, −1.0].
- **H8 reversed.** Asking for only the next step in a box lowers RLVR's leak
  rate from 84% to 4% (−79.6 [−84.6, −74.2]); DPO 94% -> 34%.
- All three results are unchanged when restricted to the 94 problems every
  checkpoint solves, and with gold >= 10.

## Descriptive

- **The Think ladder barely follows the tutor prompt at all.** With no format
  request the gold answer appears in 94% of DPO replies and 84% of RLVR
  replies. RLVR is the more adherent checkpoint, as in the Instruct ladder.
  This leaves little room for format requests to raise leak rates.
- **Format requests change how the answer is given, and familiarity does not
  matter.** Without a format request, these leaks are mostly walk-throughs that
  reach the number: only 9–11% are committed final answers. Any named answer
  slot turns them into committed answers, and the novel `<result>` tag does so
  as reliably as the formats RL pipelines reward:

  | Committed answer rate | none | boxed | answer_tag | result_tag (novel) | dapo |
  |---|---|---|---|---|---|
  | DPO | 11% | 98% | 88% | 91% | 47% |
  | RLVR | 9% | 86% | 83% | 90% | 49% |

- Exploratory per-format H6: only `\boxed{}` shows an RLVR increase over DPO
  (+4.7 [0.0, 9.5]); `Answer:` and `<answer>` do not.

## What this does and does not show

In OLMo 3 7B Think, the released RLVR stage did not make the model more
responsive to the answer conventions RL rewards. Both checkpoints fill
whatever answer slot they are given, familiar or invented, and both follow a
request to box only the next step almost perfectly. Together with the Instruct
ladder, this does not support the idea that RLVR installs a reflex specific to
RL answer formats, at least in this model family. The format effect looks like
general instruction following that outranks the system prompt.

Limits: one model family and one lab's checkpoints; near-ceiling leak rates in
the tutor cells, which shrink the room for leak contrasts (commit rates, which
have room, show the same pattern descriptively); a reduced grid with no
environment-style prose; 100 problems.

## Method notes

- Generated on a rented RTX 5090 (vast.ai) with the same vLLM 0.21.0
  generation code as the Modal runs; sampling unchanged; FlashInfer sampler
  disabled for the host's CUDA toolchain. 595 output tokens/s; total cost $4.87
  for both stages, plus $0.29 on two hosts that failed to start.
- Stage B was decided on cost alone, before any outcome rate was computed
  (Deviation 11).
- Reasoning-trace coding (exploratory addendum) is reported separately.
