# Phase 1 findings (Qwen3 0.6B–32B, 200 GSM8K problems, 48,000 responses)

Numbers from `results/phase1/summary.md`. Leak = the response contains the
problem's correct final answer. Hand-check of 100 Qwen3-4B responses: 100/100
agreement (LLM labeler; see `results/handcheck-audit-4B.md`).

1. **Grading language does not trigger leaking (H2 not supported).** Saying the
   work is checked by an autograder versus graded by a teacher changes leak
   rates by at most a few points in either direction, at every size. No model
   shows a positive autograder-minus-teacher interval; Qwen3-8B shows a small
   negative one.

2. **No coherent cue x format interaction (H1 not supported).** Significant
   interaction contrasts are small (3–7 points) and split in sign (four
   positive, six negative of 42). See Deviation 5 on the Phase 2 gate.

3. **Format demands dominate below 14B.** Asking for the answer in `\boxed{}`
   raises leak rates from 41% to 73% (8B) and from 44% to 72% (4B), whatever
   the cue. "Final Answer:" moves them far less (+4 to +9 points). Not a
   pre-registered hypothesis; it motivates Phase 1c.

4. **Tutor-prompt adherence changes abruptly between 8B and 14B.** Baseline
   leak (no cue, no format): 41% at 8B, 7% at 14B, 6% at 32B. At 14B and 32B the
   `\boxed{}` effect shrinks to about 2 points; at 32B, "Final Answer:" still
   raises leaking from 6% to 20%. 0.6B and 1.7B leak at roughly the rate they
   can solve problems at all.

5. **H3 (scale) is moot.** The H2 contrast is near zero at every size, so its
   slope across the ladder is zero ([-0.01, 0.00] per log-parameter).

**Gold >= 10 sensitivity** (`results/phase1-gold10/`): all conclusions hold. At
14B and 32B most baseline "leaks" were small-number false positives: excluding
problems with answers below 10, baseline leak is 1.6% (14B) and 0.6% (32B), and
32B's "Final Answer:" effect remains (0.6% to 13.6%).

Limits: one generic tutor prompt (not a production prompt), one model family,
GSM8K only, thinking disabled.
