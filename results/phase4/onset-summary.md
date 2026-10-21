# Phase 4 exploratory onset values

400 traces; 100 problems; six cached neuron sets. Tutor minus neutral, in Phase 3 T units.
2,000 paired problem-cluster resamples; seed 0; unadjusted percentile 95% intervals.
Boundary 0 is the `<think>` token; reasoning positions exclude tag tokens. Missing positions are omitted.

| Window | Tutor contrast mean [95% CI] | Traces |
|---|---|---|
| before_tag_64 | 2.415468 [2.233167, 2.618411] | 400 |
| after_tag_64 | 0.232382 [0.222719, 0.242294] | 400 |
| reasoning_first64 | 0.232114 [0.222215, 0.242419] | 374 |
| reasoning_first256 | 0.172932 [0.160864, 0.185447] | 374 |

Per-position CSV: 2310 rows, 385 positions × six sets. All controls and window values are in the CSVs.
Verified token identity: 400/400. Closed reasoning: 374/400.
Conservative aggregate Modal allocation bound: $0.075204; zero containers verified.
