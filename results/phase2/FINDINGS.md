# Phase 2 findings: a boxed-answer direction in Qwen3-8B

Qwen3-8B, thinking off, generic tutor prompt (`system_tutor.txt`). Direction d =
difference of mean residuals (boxed minus no-format) at the last prompt token,
layer 13 (zero-based) of 36, selected on development problems 1–100. All
estimates below are on held-out test problems 101–200, 2 templates each
(Deviation 8), 4,000 responses. Intervals are problem-cluster bootstrap 95% CIs.
Full tables: `summary.md`.

## Registered verdict

All 15 steered cells exceed the pre-registered 10% degeneration threshold, so
under the rule as written every H9–H11 result is "not interpretable". That
threshold is tripped by a defect in the check itself (Deviation 9): it flags
the same 17% of responses in unsteered baseline cells, all on problems whose
statement has fewer than two numerals. On the remaining 83 problems the flag
fires on 0–2 responses per cell. The results below are reported with that
correction, which was made after viewing the data.

## Results (corrected degeneration check, post hoc)

- **H9, necessity: supported.** Ablating d cuts the boxed-format leak rate from
  65.0% to 31.5% (−33.5 points [25.5, 41.0]). Ablating any of five random unit
  directions at the same layer leaves it at 62–69%; d beats their mean by 32.9
  points [26.5, 39.4]. Same result with gold ≥ 10.
- **H11, generalization: supported.** The direction was estimated from one
  boxed phrasing. Ablating it also lowers leaks for two unseen boxed phrasings
  (−33.0 [26.0, 41.0] and −31.5 [25.0, 38.5]) and for the non-box format
  "Final Answer:" (−16.5 [9.0, 24.5]).
- **H10, sufficiency: not supported as specific.** Adding α·d to no-format
  prompts raises the leak rate from 35.0% to 88.5%, but adding random
  directions of the same norm raises it to 66.5–90.5%. The excess over random is
  4.9 points [0.6, 9.3], and 3.6 [−1.6, 8.7] with gold ≥ 10. At this magnitude
  a generic perturbation of the residual stream breaks the tutor instruction
  about as well as d does. One signal does survive: committed final answers
  rise to 32.5% under d, against 14.5–18.0% for four of the five random
  directions (the fifth reaches 53.5%), so the evidence that d specifically
  drives answering is weak.

## What ablation costs

Ablating d is not free. Among committed answers, accuracy falls from 90% to 75%
under boxed and from 65% to 41% under "Final Answer:"; random ablations leave it
at 88–91%. Under boxed, most of the leak reduction comes from the model no
longer committing an answer at all (110 → 48 commits, wrong commits 11 → 12).
Under "Final Answer:", the model still writes the final line about as often
(124 → 127) but with the wrong number far more often (43 → 75 wrong). So in the
"Final Answer:" format part of the "fewer leaks" effect is damaged arithmetic,
not restored tutoring. The boxed results are not explained this way.

## What this does and does not show

A single direction at one middle layer, estimated from one phrasing of
"put the final answer in \boxed{}", is causally needed for a large share of
format-triggered answer leaking in this model, and the same direction is
involved in other answer-format demands. It is not shown to be sufficient:
pushing along it is only marginally more effective than pushing in a random
direction. Nothing here shows the direction comes from RL training, that it is
the only mechanism, or that it carries over to other models or to thinking mode.

## Method notes

- Layer selection used a 1-problem screen of 22 layers, then a 100-problem
  confirmation of three finalists (Deviation 7). Layer 13 cut dev leaks by 30.8
  points; the other finalists (layers 20, 7) cut them by 7.5 and 6.0. Other
  layers that the noisy screen missed may work as well.
- α = 2 × the mean boxed projection norm, the smallest registered multiplier
  that met the development threshold.
- Generation used HF transformers with forward hooks at bf16, with batch-level
  seeding. Outputs are not comparable token-for-token with the Phase 1 vLLM
  runs; within Phase 2, every cell shares seeds and batching.
- The commit pattern undercounts markdown-styled final answers (Deviation 2);
  leak rates are unaffected.
- GPU cost: at most $3.82 (1.96 L40S hours, including discarded attempts).
