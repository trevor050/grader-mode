# Errata (2026-10-22)

An independent recomputation of every number in the phase findings against the raw data (GPT-6.1 Sol;
full report in `results/AUDIT-2026-10-22.md`) matched all 1,396 summary point estimates and found the following
statements in the findings files to be wrong or imprecise. The findings files are left as written; these are the
corrections.

- Phase 1, interaction contrasts: "four positive, six negative of 42" counts H1 and H2 together. H1 alone: 4
  positive, 3 negative of 18.
- Phase 1b, tutor-role mention table: not exactly reproducible; the original regex was not saved. The documented
  pattern gives lower shares (14B 4.3-29.0%, 32B 0.8-12.5%) and 6/743 for the 32B env leaks. The direction of every
  comparison is unchanged.
- Phase 1c, commit rates in DPO/RLVR tutor cells range 1.0-25.75%, not 2-14%. The 10-13% truncation share applies
  to the plain no-format tutor cells only; across all responses it is 6.8% (DPO) and 4.7% (RLVR).
- Phase 1c Think, per-cell truncation is 0-6.75% across all cells (0.25-6.75% in tutor cells).
- Phase 3, the random-set range over reasoning deciles 2-9 is -0.14 to +0.04 SD, not -0.07 to +0.05. The registered
  run cost at most $2.87; $3.54 is the lifetime total including later exploratory work.
- Phase 5, the opening statement that the second family produced no data is stale: Nemotron 3.5 Lightning ran
  the full grid (3,200 replies) and its results are in the same file (H20, H21 not supported).
- Explore section 7: the Qwen3.8 tutor neurons include one in layer 59 (corrected in place); the student-request
  side-effect counts were swapped (corrected in place).
