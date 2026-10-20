# Phase 2 results

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.
Rates: percent; contrasts: percentage points; intervals: unadjusted 95%.
Test ranks: 101..200; templates: 2; development responses excluded: 2488.
Steered cells above 10% degeneration: 15 / 15.

## All test problems

### Qwen/Qwen3-8B

Problems: 100; responses: 4000.

| Format | Intervention | Responses | leak n | leak [95% CI] | commit n | commit [95% CI] | correct_commit n | correct_commit [95% CI] |
|---|---|---|---|---|---|---|---|---|
| boxed | baseline | 200 | 130 | 65.00 [57.00, 73.00] | 110 | 55.00 [47.50, 62.50] | 99 | 49.50 [41.50, 57.50] |
| boxed | ablate_d | 200 | 63 | 31.50 [24.50, 39.00] | 48 | 24.00 [17.00, 31.00] | 36 | 18.00 [12.00, 24.00] |
| boxed | ablate_r0 | 200 | 129 | 64.50 [56.00, 72.50] | 104 | 52.00 [44.00, 60.50] | 93 | 46.50 [39.00, 54.50] |
| boxed | ablate_r1 | 200 | 126 | 63.00 [55.00, 71.00] | 108 | 54.00 [45.50, 62.01] | 95 | 47.50 [39.49, 56.00] |
| boxed | ablate_r2 | 200 | 124 | 62.00 [54.00, 69.51] | 107 | 53.50 [45.50, 61.00] | 97 | 48.50 [41.00, 56.00] |
| boxed | ablate_r3 | 200 | 127 | 63.50 [55.50, 71.00] | 108 | 54.00 [46.00, 61.50] | 98 | 49.00 [40.99, 57.00] |
| boxed | ablate_r4 | 200 | 138 | 69.00 [61.00, 76.50] | 121 | 60.50 [52.00, 69.00] | 107 | 53.50 [45.00, 62.00] |
| none | baseline | 200 | 70 | 35.00 [27.00, 43.50] | 7 | 3.50 [1.00, 6.50] | 5 | 2.50 [0.50, 5.50] |
| none | add_d | 200 | 177 | 88.50 [83.00, 94.00] | 65 | 32.50 [25.50, 40.00] | 56 | 28.00 [21.00, 35.50] |
| none | add_r0 | 200 | 167 | 83.50 [76.99, 89.51] | 36 | 18.00 [12.00, 24.00] | 28 | 14.00 [9.00, 19.50] |
| none | add_r1 | 200 | 133 | 66.50 [59.00, 74.00] | 29 | 14.50 [9.00, 20.50] | 22 | 11.00 [6.50, 16.00] |
| none | add_r2 | 200 | 177 | 88.50 [82.50, 94.00] | 107 | 53.50 [45.50, 61.50] | 95 | 47.50 [39.50, 56.00] |
| none | add_r3 | 200 | 178 | 89.00 [83.00, 94.50] | 35 | 17.50 [12.50, 23.00] | 32 | 16.00 [11.00, 21.50] |
| none | add_r4 | 200 | 181 | 90.50 [85.00, 95.50] | 33 | 16.50 [11.00, 22.50] | 32 | 16.00 [10.50, 22.00] |
| boxed_alt_1 | baseline | 200 | 135 | 67.50 [59.50, 75.50] | 125 | 62.50 [54.50, 71.00] | 111 | 55.50 [47.00, 64.00] |
| boxed_alt_1 | ablate_d | 200 | 69 | 34.50 [26.99, 42.50] | 47 | 23.50 [17.00, 30.50] | 38 | 19.00 [13.00, 25.50] |
| boxed_alt_2 | baseline | 200 | 137 | 68.50 [61.00, 76.00] | 144 | 72.00 [65.50, 78.50] | 131 | 65.50 [58.00, 73.00] |
| boxed_alt_2 | ablate_d | 200 | 74 | 37.00 [29.50, 44.50] | 74 | 37.00 [30.00, 44.00] | 64 | 32.00 [25.50, 38.50] |
| final | baseline | 200 | 89 | 44.50 [37.00, 52.50] | 124 | 62.00 [55.50, 68.50] | 81 | 40.50 [33.00, 48.00] |
| final | ablate_d | 200 | 56 | 28.00 [22.00, 34.50] | 127 | 63.50 [58.00, 69.00] | 52 | 26.00 [20.00, 32.00] |

| Hypothesis | Contrast | Problems | Estimate [95% CI] | not_interpretable |
|---|---|---|---|---|
| H9 | necessity | 100 | 33.50 [25.50, 41.00] | true |
| H9 | superiority | 100 | 32.90 [26.50, 39.40] | true |
| H10 | sufficiency | 100 | 53.50 [44.50, 62.50] | true |
| H10 | superiority | 100 | 4.90 [0.60, 9.30] | true |
| H11 | boxed_alt_1 | 100 | 33.00 [26.00, 41.00] | true |
| H11 | boxed_alt_2 | 100 | 31.50 [25.00, 38.50] | true |
| H11 | final | 100 | 16.50 [9.00, 24.50] | true |
| H11 | pooled | 100 | 27.00 [22.83, 31.50] | true |

Degeneration threshold: > 10%.

| Format | Intervention | Responses | Check | Count | Rate [95% CI] | not_interpretable |
|---|---|---|---|---|---|---|
| boxed | ablate_d | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_d | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_d | 200 | low_number_count | 35 | 17.50 [10.50, 25.00] | true |
| boxed | ablate_d | 200 | degenerate | 35 | 17.50 [10.50, 25.00] | true |
| boxed | ablate_r0 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r0 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r0 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r0 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r1 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r1 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r1 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r1 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r2 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r2 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r2 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r2 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r3 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r3 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r3 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r3 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r4 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r4 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r4 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| boxed | ablate_r4 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| none | add_d | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_d | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_d | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| none | add_d | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r0 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r0 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r0 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r0 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r1 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r1 | 200 | repetition | 1 | 0.50 [0.00, 1.50] | true |
| none | add_r1 | 200 | low_number_count | 36 | 18.00 [11.00, 25.50] | true |
| none | add_r1 | 200 | degenerate | 36 | 18.00 [11.00, 25.50] | true |
| none | add_r2 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r2 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r2 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r2 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r3 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r3 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r3 | 200 | low_number_count | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r3 | 200 | degenerate | 34 | 17.00 [10.00, 25.00] | true |
| none | add_r4 | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r4 | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r4 | 200 | low_number_count | 35 | 17.50 [10.50, 25.01] | true |
| none | add_r4 | 200 | degenerate | 35 | 17.50 [10.50, 25.01] | true |
| boxed_alt_1 | ablate_d | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_1 | ablate_d | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_1 | ablate_d | 200 | low_number_count | 35 | 17.50 [10.50, 25.00] | true |
| boxed_alt_1 | ablate_d | 200 | degenerate | 35 | 17.50 [10.50, 25.00] | true |
| boxed_alt_2 | ablate_d | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_2 | ablate_d | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_2 | ablate_d | 200 | low_number_count | 35 | 17.50 [10.50, 25.00] | true |
| boxed_alt_2 | ablate_d | 200 | degenerate | 35 | 17.50 [10.50, 25.00] | true |
| final | ablate_d | 200 | empty | 0 | 0.00 [0.00, 0.00] | true |
| final | ablate_d | 200 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| final | ablate_d | 200 | low_number_count | 36 | 18.00 [11.00, 25.50] | true |
| final | ablate_d | 200 | degenerate | 36 | 18.00 [11.00, 25.50] | true |

## gold >= 10

### Qwen/Qwen3-8B

Problems: 83; responses: 3320.

| Format | Intervention | Responses | leak n | leak [95% CI] | commit n | commit [95% CI] | correct_commit n | correct_commit [95% CI] |
|---|---|---|---|---|---|---|---|---|
| boxed | baseline | 166 | 105 | 63.25 [54.22, 72.29] | 89 | 53.61 [45.18, 62.05] | 82 | 49.40 [40.96, 57.83] |
| boxed | ablate_d | 166 | 50 | 30.12 [22.89, 37.35] | 43 | 25.90 [18.67, 33.73] | 34 | 20.48 [13.86, 27.71] |
| boxed | ablate_r0 | 166 | 103 | 62.05 [52.41, 71.08] | 88 | 53.01 [43.98, 62.05] | 80 | 48.19 [39.16, 57.23] |
| boxed | ablate_r1 | 166 | 103 | 62.05 [53.01, 70.48] | 91 | 54.82 [45.18, 64.46] | 81 | 48.80 [39.76, 58.43] |
| boxed | ablate_r2 | 166 | 101 | 60.84 [51.81, 69.88] | 88 | 53.01 [44.58, 61.45] | 82 | 49.40 [40.96, 57.83] |
| boxed | ablate_r3 | 166 | 104 | 62.65 [54.22, 71.08] | 90 | 54.22 [45.78, 62.65] | 82 | 49.40 [40.36, 58.43] |
| boxed | ablate_r4 | 166 | 111 | 66.87 [57.83, 75.90] | 99 | 59.64 [50.00, 69.28] | 90 | 54.22 [44.58, 63.25] |
| none | baseline | 166 | 53 | 31.93 [23.49, 40.96] | 6 | 3.61 [0.60, 7.23] | 5 | 3.01 [0.60, 6.63] |
| none | add_d | 166 | 145 | 87.35 [80.12, 93.37] | 56 | 33.73 [25.90, 42.17] | 49 | 29.52 [21.69, 37.35] |
| none | add_r0 | 166 | 140 | 84.34 [77.71, 90.96] | 34 | 20.48 [13.86, 27.71] | 26 | 15.66 [9.64, 22.29] |
| none | add_r1 | 166 | 109 | 65.66 [57.83, 73.49] | 24 | 14.46 [8.43, 21.08] | 18 | 10.84 [6.02, 16.27] |
| none | add_r2 | 166 | 146 | 87.95 [81.33, 93.98] | 89 | 53.61 [45.18, 62.05] | 79 | 47.59 [38.55, 56.02] |
| none | add_r3 | 166 | 148 | 89.16 [82.53, 95.18] | 32 | 19.28 [13.25, 25.90] | 30 | 18.07 [12.65, 24.10] |
| none | add_r4 | 166 | 152 | 91.57 [85.54, 96.39] | 31 | 18.67 [12.65, 25.30] | 30 | 18.07 [12.05, 24.70] |
| boxed_alt_1 | baseline | 166 | 112 | 67.47 [58.43, 75.90] | 105 | 63.25 [54.22, 71.08] | 95 | 57.23 [48.19, 65.68] |
| boxed_alt_1 | ablate_d | 166 | 50 | 30.12 [22.29, 38.55] | 40 | 24.10 [16.87, 31.93] | 33 | 19.88 [13.25, 27.11] |
| boxed_alt_2 | baseline | 166 | 112 | 67.47 [59.04, 75.30] | 119 | 71.69 [63.86, 78.92] | 109 | 65.66 [57.23, 73.49] |
| boxed_alt_2 | ablate_d | 166 | 62 | 37.35 [29.52, 45.18] | 63 | 37.95 [30.12, 45.18] | 56 | 33.73 [26.51, 40.36] |
| final | baseline | 166 | 73 | 43.98 [35.54, 51.81] | 106 | 63.86 [57.23, 70.48] | 72 | 43.37 [35.54, 51.20] |
| final | ablate_d | 166 | 42 | 25.30 [19.28, 31.93] | 103 | 62.05 [56.63, 68.07] | 41 | 24.70 [18.67, 31.33] |

| Hypothesis | Contrast | Problems | Estimate [95% CI] | not_interpretable |
|---|---|---|---|---|
| H9 | necessity | 83 | 33.13 [24.70, 41.57] | true |
| H9 | superiority | 83 | 32.77 [26.27, 39.64] | true |
| H10 | sufficiency | 83 | 55.42 [45.78, 65.06] | true |
| H10 | superiority | 83 | 3.61 [-1.57, 8.67] | true |
| H11 | boxed_alt_1 | 83 | 37.35 [28.92, 45.18] | true |
| H11 | boxed_alt_2 | 83 | 30.12 [22.89, 37.95] | true |
| H11 | final | 83 | 18.67 [10.84, 27.11] | true |
| H11 | pooled | 83 | 28.71 [24.10, 33.13] | true |

Degeneration threshold: > 10%.

| Format | Intervention | Responses | Check | Count | Rate [95% CI] | not_interpretable |
|---|---|---|---|---|---|---|
| boxed | ablate_d | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_d | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_d | 166 | low_number_count | 25 | 15.06 [7.83, 22.89] | true |
| boxed | ablate_d | 166 | degenerate | 25 | 15.06 [7.83, 22.89] | true |
| boxed | ablate_r0 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r0 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r0 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r0 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r1 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r1 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r1 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r1 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r2 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r2 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r2 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r2 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r3 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r3 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r3 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r3 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r4 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r4 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed | ablate_r4 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| boxed | ablate_r4 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| none | add_d | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_d | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_d | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| none | add_d | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r0 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r0 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r0 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r0 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r1 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r1 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r1 | 166 | low_number_count | 25 | 15.06 [7.83, 22.89] | true |
| none | add_r1 | 166 | degenerate | 25 | 15.06 [7.83, 22.89] | true |
| none | add_r2 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r2 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r2 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r2 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r3 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r3 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r3 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r3 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r4 | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r4 | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| none | add_r4 | 166 | low_number_count | 24 | 14.46 [7.23, 21.69] | true |
| none | add_r4 | 166 | degenerate | 24 | 14.46 [7.23, 21.69] | true |
| boxed_alt_1 | ablate_d | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_1 | ablate_d | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_1 | ablate_d | 166 | low_number_count | 25 | 15.06 [7.83, 22.89] | true |
| boxed_alt_1 | ablate_d | 166 | degenerate | 25 | 15.06 [7.83, 22.89] | true |
| boxed_alt_2 | ablate_d | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_2 | ablate_d | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| boxed_alt_2 | ablate_d | 166 | low_number_count | 25 | 15.06 [7.83, 22.89] | true |
| boxed_alt_2 | ablate_d | 166 | degenerate | 25 | 15.06 [7.83, 22.89] | true |
| final | ablate_d | 166 | empty | 0 | 0.00 [0.00, 0.00] | true |
| final | ablate_d | 166 | repetition | 0 | 0.00 [0.00, 0.00] | true |
| final | ablate_d | 166 | low_number_count | 26 | 15.66 [8.43, 23.49] | true |
| final | ablate_d | 166 | degenerate | 26 | 15.66 [8.43, 23.49] | true |

## Development selections

Layer (zero-based): 13; alpha: 30.849032; multiplier: 2; scale: 15.424516.

| Format | Intervention | Responses | Empty | Repetition | Low number count | Composite | not_interpretable |
|---|---|---|---|---|---|---|---|
| boxed | confirm_L13 | 400 | 0 | 0 | 49 | 49 | true |
| boxed | confirm_L20 | 400 | 0 | 0 | 45 | 45 | true |
| boxed | confirm_L7 | 400 | 0 | 0 | 45 | 45 | true |
| boxed | screen_L10 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L11 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L12 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L13 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L14 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L15 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L16 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L17 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L18 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L19 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L20 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L21 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L22 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L23 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L24 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L25 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L26 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L27 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L28 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L7 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L8 | 4 | 0 | 0 | 0 | 0 | false |
| boxed | screen_L9 | 4 | 0 | 0 | 0 | 0 | false |
| none | alpha_m2 | 400 | 0 | 0 | 44 | 44 | true |

## Execution

| Measure | Value |
|---|---:|
| Screen problems | 1 |
| Screen templates | 4 |
| Candidate layers | 22 |
| Confirmed layers | 3 |
| Confirmation problems/layer | 100 |
| Confirmation templates | 4 |
| Test templates | 2 |
| Registered test templates | 4 |
| Test responses | 4000 |
| Bootstrap resamples | 2000 |
| Tests passed | 355 |
| Independent contrast/CI checks | 16 |
| GPU allocation seconds, upper bound | 7049.314 |
| GPU allocation hours, upper bound | 1.958143 |
| GPU USD, upper bound | 3.818378 |
| GPU USD cap | 6.00 |
| Workspace metered USD | 21.59343289 |
| Workspace billed USD | 0E-8 |
| Phase 2 running containers | 0 |
