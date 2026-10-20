# Phase 1b findings (Qwen3 4B–32B, 200 problems, 19,200 responses)

Numbers from `results/phase1b/summary.md`. Every Phase 1b cell includes a
`\boxed{}` answer demand; effects below are measured under that demand.

## Pre-registered

- **H4 not supported.** With thinking off, RL-environment register vs an
  equally strict human teacher: 4B +6.3 points [2.6, 9.5]; 14B +1.6 [0.0, 3.4];
  8B +2.5 [-0.9, 5.8]; 32B +1.0 [-1.5, 3.5]. One of four models clears zero
  (two required).
- **H5 supported at 14B and 32B.** The env-minus-strict-teacher gap is larger
  with thinking on: 14B +11.8 points [7.5, 15.9], 32B +30.5 [26.0, 35.1].
  Null at 8B (+0.1 [-3.7, 3.9]) and reversed at 4B (-5.5 [-9.0, -1.6]), where
  thinking-on leak rates are already 94–97% in every cell.

## Descriptive (not hypothesis tests)

- **Thinking mode, under a boxed-answer request, overrides the tutor prompt.**
  No-cue leak, thinking off -> on: 32B 8% -> 78%, 14B 11% -> 52%, 8B 70% -> 96%,
  4B 74% -> 97%. Phase 1b has no no-format condition, so this is not shown for
  requests without a format demand.
- **With thinking on at 32B:** env 94%, no cue 78%, strict teacher 63%. The
  strict-teacher framing lowers leaking relative to no cue (also at 14B: 45%
  vs 52%).
- **The gold >= 10 sensitivity analysis gives the same pattern** (32B H5 +34.0
  [29.3, 38.6]; 14B H4 +1.8 [0.2, 3.6]).

## Exploratory reasoning-trace observation (post hoc, regex, by Claude)

Share of thinking-on reasoning traces that mention the tutoring role or
withholding the answer (regex: tutor / system prompt|message|instruction /
not give|provide|reveal|state the answer / guide the student / help the
student learn):

| | none, leak | none, held | strict, leak | strict, held | env, leak | env, held |
|---|---|---|---|---|---|---|
| 14B | 6.8% | 16.6% | 22.2% | 45.4% | 2.2% | 6.3% |
| 32B | 5.9% | 26.1% | 6.3% | 16.5% | 0.9% | 8.9% |

Under env prompts, 32B's leaking traces almost never mention the tutoring role
(7 of 743); they open directly into problem solving. Leaks go with the role not
being recalled, rather than with a recorded decision to override it. The
grading-vocabulary regex matches only 2.5–5.8% of traces in any cell. This is
a crude post-hoc measure; it motivates a pre-registered trace-coding test.

## Limits

One generic tutor prompt; Qwen3 only; GSM8K; truncated thinking-on rows with no
final reply are excluded (2–7% per cell, more in no-cue and strict-teacher
cells than env), which could bias those cells upward or downward.
