# Phase 1c findings: OLMo 3 7B Instruct ladder (SFT -> DPO -> RLVR)

200 problems, 31,200 responses, tutor prompt `system_tutor_strong.txt` (chosen by
the pre-registered calibration rule). Think ladder deferred (Deviation 6).

## Pre-registered

- **H6 not supported.** RLVR does not amplify the effect of RL-associated
  formats relative to DPO: -3.4 points [-7.7, 1.1]. In the RLVR checkpoint the
  pooled RL-format effect is negative (-6.3 [-9.7, -2.7]).
- **H7 not supported.** RL-format effect minus novel-format effect in RLVR:
  -1.5 [-4.4, 1.5]; difference-in-differences vs DPO: -2.3 [-6.6, 2.0].
- **H8 reversed.** Asking for only the next step in a box *lowers* leaking by
  56 points [51, 60] in RLVR (59% -> 3%), and similarly in SFT and DPO. The box
  token alone does not pull out the answer.
- Restricting to problems all three checkpoints solve gives the same results.

## Descriptive

- Post-training stages improve tutor adherence: no-format leak 81% (SFT),
  72% (DPO), 59% (RLVR), while solving ability rises (ceiling correct commit
  46% -> 34% -> 72%).
- OLMo's leaks are mostly walk-throughs that reach the number, not committed
  final answers: commit rates are 2–14% in DPO/RLVR tutor cells.

## Design limitation (lead's error, stated plainly)

`system_tutor_strong.txt`, written by the lead for the calibration rule,
explicitly forbids revealing the answer "in a box, a tag, a final line, a
check, or a suggestion". That names the very formats Phase 1c manipulates, so
the strong prompt directly targets the mechanism under test. H6–H8 results here
are therefore results under a format-aware tutor prompt, and are not a clean
test of whether RLVR creates format sensitivity. The deferred Think ladder and
any Instruct rerun should use a strong prompt that does not mention formats.

Also: 10–13% of DPO and RLVR responses hit the 768-token limit and were scored
as-is, which can undercount leaks in those checkpoints.
