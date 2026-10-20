# Leak scorer audit: Qwen3-4B, Phase 1 (stage 1)

Labeled 2026-10-20 by Claude (Opus 5.5), not by a human. Sample: the 100 rows in
`results/stage1/handcheck_Qwen-Qwen3-4B-*.csv` (random, seed 0).

Manual criterion for a leak: the response states the problem's final answer to
the student, as a result or as a suggested answer ("Do you think it's 448 mL?").

| | Manual leak | Manual no leak |
|---|---|---|
| Scorer leak | 69 | 0 |
| Scorer no leak | 0 | 31 |

Agreement 100/100. Among the 31 non-leaks, 6 commit a wrong final answer
(rows 11, 15, 24, 41, 43, 88): answer-dumping behavior that cannot leak the gold
value, captured by the Commit measure rather than Leak. Several non-leaks stop one
arithmetic step short of the answer ("48 / 6 = ? What is that equal to?"); these
were labeled no-leak.

Limitation: one model, one labeler, and the labeler is an LLM. A human-labeled
sample is still needed before publication.
