# Phase 1c Think results

Available templates: [0, 1, 2, 3]; Stage B: complete.
Tutor prompt: prompts/system_tutor.txt (registered calibration decision).
SFT has ceiling rows only; DPO and RLVR have plain tutor rows in all six formats.
Solvability: at least one valid correct commit per checkpoint among the available ceiling templates; membership fixed before bootstrap.
Exploratory env × format and SFT → DPO analyses omitted: env prose and SFT tutor cells were not run.
Per-format H6 breakdowns are exploratory. No hand-check agreement is claimed by this report.

## All problems

Gold minimum: None; 6000 responses.

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.

Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.

### think

Checkpoints: complete; available: SFT, DPO, RLVR; missing: none.
Inference eligible: True; source problems: 100 / 100.

#### All selected problems

Problems: 100; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | ceiling | plain | none | 400 / 400 | 0 (0) | 0 | 0 | 96.50 [92.75, 99.50] | 91.00 [87.50, 94.00] | 87.50 [83.00, 91.50] |
| DPO | tutor | plain | none | 398 / 400 | 2 (2) | 6 | 2 | 94.47 [90.95, 97.25] | 10.55 [7.04, 14.29] | 8.79 [5.57, 12.31] |
| DPO | tutor | plain | boxed | 400 / 400 | 0 (0) | 1 | 0 | 96.25 [92.50, 99.25] | 98.25 [96.50, 99.50] | 94.75 [90.75, 98.00] |
| DPO | tutor | plain | dapo | 400 / 400 | 0 (0) | 1 | 0 | 95.00 [91.24, 98.00] | 46.50 [41.74, 51.50] | 43.75 [38.75, 48.75] |
| DPO | tutor | plain | answer_tag | 400 / 400 | 0 (0) | 2 | 0 | 95.00 [91.00, 98.25] | 87.50 [82.00, 92.50] | 84.50 [78.75, 90.00] |
| DPO | tutor | plain | result_tag | 399 / 400 | 1 (1) | 5 | 1 | 95.49 [91.94, 98.26] | 91.23 [87.25, 94.75] | 88.22 [83.21, 92.76] |
| DPO | tutor | plain | boxed_step | 391 / 400 | 9 (9) | 13 | 9 | 33.76 [28.17, 39.29] | 44.50 [38.97, 49.87] | 20.20 [15.95, 24.36] |
| DPO | ceiling | plain | none | 398 / 400 | 2 (2) | 2 | 2 | 96.48 [92.96, 99.25] | 78.89 [73.18, 84.21] | 75.38 [69.02, 81.11] |
| RLVR | tutor | plain | none | 398 / 400 | 2 (2) | 8 | 2 | 83.92 [79.04, 88.16] | 9.30 [6.05, 12.75] | 8.04 [5.01, 11.37] |
| RLVR | tutor | plain | boxed | 397 / 400 | 3 (3) | 7 | 3 | 90.43 [86.55, 93.70] | 85.64 [82.12, 88.94] | 83.38 [79.29, 87.22] |
| RLVR | tutor | plain | dapo | 398 / 400 | 2 (2) | 11 | 2 | 84.42 [80.20, 88.25] | 49.25 [44.36, 54.02] | 46.98 [42.11, 51.63] |
| RLVR | tutor | plain | answer_tag | 399 / 400 | 1 (1) | 7 | 1 | 86.47 [81.91, 90.48] | 82.96 [78.20, 87.22] | 79.70 [74.19, 84.75] |
| RLVR | tutor | plain | result_tag | 396 / 400 | 4 (3) | 5 | 4 | 91.41 [87.09, 94.98] | 90.40 [87.44, 93.15] | 86.87 [82.53, 90.70] |
| RLVR | tutor | plain | boxed_step | 374 / 400 | 26 (26) | 27 | 26 | 4.28 [1.85, 7.47] | 45.45 [39.07, 51.73] | 1.07 [0.26, 2.17] |
| RLVR | ceiling | plain | none | 399 / 400 | 1 (1) | 2 | 1 | 97.24 [93.70, 100.00] | 96.74 [94.75, 98.50] | 93.98 [90.22, 97.00] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/DPO/boxed | 1.78 [0.02, 3.75] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | 0.53 [-1.48, 2.34] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | 0.53 [-1.67, 2.76] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | 1.02 [-0.74, 2.80] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -60.71 [-66.24, -54.80] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | 0.94 [-0.70, 2.76] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 6.51 [2.23, 11.01] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | 0.50 [-3.76, 4.87] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 2.55 [-1.68, 6.65] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 7.49 [3.71, 11.35] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -79.64 [-84.55, -74.16] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | 3.19 [-0.30, 6.78] | 2000 / 2000 | ok |
| H6 | 2.24 [-1.56, 6.24] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -4.31 [-7.09, -1.28] | 2000 / 2000 | ok |
| H7/difference_in_differences | -4.24 [-7.42, -0.95] | 2000 / 2000 | ok |
| H8 | -79.64 [-84.55, -74.16] | 2000 / 2000 | ok |
| per_format_H6/boxed | 4.73 [0.01, 9.53] | 2000 / 2000 | ok |
| per_format_H6/dapo | -0.03 [-4.51, 4.41] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | 2.02 [-2.47, 6.58] | 2000 / 2000 | ok |

#### All-three-checkpoint common ceiling solved problems

Problems: 94; status: ok.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | ceiling | plain | none | 376 / 376 | 0 (0) | 0 | 0 | 99.47 [98.67, 100.00] | 91.22 [87.50, 94.41] | 90.69 [86.97, 94.15] |
| DPO | tutor | plain | none | 375 / 376 | 1 (1) | 3 | 1 | 97.07 [95.47, 98.67] | 9.87 [6.40, 13.41] | 9.33 [5.87, 12.83] |
| DPO | tutor | plain | boxed | 376 / 376 | 0 (0) | 1 | 0 | 99.20 [98.14, 100.00] | 98.40 [96.81, 99.73] | 97.87 [96.28, 99.20] |
| DPO | tutor | plain | dapo | 376 / 376 | 0 (0) | 1 | 0 | 98.14 [96.54, 99.47] | 45.74 [40.69, 50.53] | 45.21 [40.16, 50.00] |
| DPO | tutor | plain | answer_tag | 376 / 376 | 0 (0) | 2 | 0 | 97.87 [96.01, 99.20] | 88.30 [82.98, 92.83] | 87.50 [82.18, 92.29] |
| DPO | tutor | plain | result_tag | 376 / 376 | 0 (0) | 3 | 0 | 98.40 [96.81, 99.73] | 92.55 [88.83, 96.02] | 91.76 [88.03, 95.49] |
| DPO | tutor | plain | boxed_step | 368 / 376 | 8 (8) | 10 | 8 | 34.78 [29.81, 40.44] | 45.38 [39.89, 51.08] | 21.20 [17.08, 25.62] |
| DPO | ceiling | plain | none | 375 / 376 | 1 (1) | 1 | 1 | 99.20 [98.13, 100.00] | 81.07 [75.80, 86.40] | 80.00 [74.93, 85.33] |
| RLVR | tutor | plain | none | 374 / 376 | 2 (2) | 6 | 2 | 86.63 [83.11, 90.11] | 8.82 [5.61, 12.33] | 8.29 [5.33, 11.97] |
| RLVR | tutor | plain | boxed | 375 / 376 | 1 (1) | 4 | 1 | 93.60 [90.93, 95.74] | 86.13 [82.67, 89.63] | 86.13 [82.67, 89.63] |
| RLVR | tutor | plain | dapo | 376 / 376 | 0 (0) | 9 | 0 | 87.23 [83.78, 90.43] | 48.67 [44.14, 53.46] | 48.40 [43.88, 53.46] |
| RLVR | tutor | plain | answer_tag | 376 / 376 | 0 (0) | 6 | 0 | 88.83 [85.11, 92.29] | 82.18 [77.13, 86.97] | 81.65 [76.60, 86.44] |
| RLVR | tutor | plain | result_tag | 373 / 376 | 3 (2) | 3 | 3 | 94.37 [91.60, 97.04] | 90.35 [87.37, 93.26] | 89.54 [86.40, 92.72] |
| RLVR | tutor | plain | boxed_step | 355 / 376 | 21 (21) | 22 | 21 | 4.51 [1.94, 7.80] | 45.63 [38.83, 52.39] | 1.13 [0.28, 2.28] |
| RLVR | ceiling | plain | none | 376 / 376 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 96.81 [94.68, 98.40] | 96.81 [94.68, 98.40] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/DPO/boxed | 2.14 [0.53, 4.00] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | 1.07 [-0.80, 2.94] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | 0.81 [-1.32, 2.93] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | 1.34 [-0.52, 3.19] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -62.28 [-67.44, -56.80] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | 1.34 [-0.26, 3.02] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 6.97 [2.67, 11.03] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | 0.60 [-3.69, 4.91] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 2.20 [-2.03, 6.29] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 7.74 [3.94, 11.69] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -82.12 [-86.55, -77.57] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | 3.26 [-0.25, 6.65] | 2000 / 2000 | ok |
| H6 | 1.92 [-1.61, 5.66] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -4.48 [-7.61, -1.52] | 2000 / 2000 | ok |
| H7/difference_in_differences | -4.48 [-7.71, -1.21] | 2000 / 2000 | ok |
| H8 | -82.12 [-86.55, -77.57] | 2000 / 2000 | ok |
| per_format_H6/boxed | 4.83 [0.46, 8.91] | 2000 / 2000 | ok |
| per_format_H6/dapo | -0.47 [-4.79, 3.99] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | 1.39 [-2.93, 5.70] | 2000 / 2000 | ok |

## Sensitivity: gold >= 10

Gold minimum: 10; 5160 responses.

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.

Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.

### think

Checkpoints: complete; available: SFT, DPO, RLVR; missing: none.
Inference eligible: True; source problems: 100 / 100.

#### All selected problems

Problems: 86; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | ceiling | plain | none | 344 / 344 | 0 (0) | 0 | 0 | 95.93 [91.57, 99.42] | 92.15 [88.66, 95.06] | 88.08 [83.14, 92.44] |
| DPO | tutor | plain | none | 342 / 344 | 2 (2) | 5 | 2 | 93.86 [90.00, 97.09] | 11.70 [7.87, 15.88] | 9.65 [6.21, 13.53] |
| DPO | tutor | plain | boxed | 344 / 344 | 0 (0) | 1 | 0 | 95.64 [91.28, 99.13] | 98.55 [97.09, 99.71] | 94.48 [90.12, 97.97] |
| DPO | tutor | plain | dapo | 344 / 344 | 0 (0) | 0 | 0 | 94.19 [89.53, 97.97] | 45.64 [40.99, 50.29] | 42.44 [37.79, 47.38] |
| DPO | tutor | plain | answer_tag | 344 / 344 | 0 (0) | 2 | 0 | 94.19 [89.83, 97.97] | 88.37 [82.85, 93.31] | 84.88 [78.49, 90.70] |
| DPO | tutor | plain | result_tag | 343 / 344 | 1 (1) | 4 | 1 | 94.75 [90.64, 98.26] | 92.71 [88.89, 96.21] | 89.50 [84.12, 94.19] |
| DPO | tutor | plain | boxed_step | 337 / 344 | 7 (7) | 10 | 7 | 30.27 [25.29, 35.50] | 45.10 [39.17, 51.32] | 19.29 [14.93, 23.97] |
| DPO | ceiling | plain | none | 342 / 344 | 2 (2) | 2 | 2 | 95.91 [91.79, 99.12] | 81.29 [75.22, 86.55] | 77.19 [70.71, 83.09] |
| RLVR | tutor | plain | none | 342 / 344 | 2 (2) | 8 | 2 | 81.29 [76.16, 86.01] | 9.36 [6.10, 13.16] | 7.89 [4.71, 11.63] |
| RLVR | tutor | plain | boxed | 341 / 344 | 3 (3) | 7 | 3 | 89.44 [85.29, 93.27] | 85.92 [82.22, 89.44] | 83.28 [78.76, 87.50] |
| RLVR | tutor | plain | dapo | 342 / 344 | 2 (2) | 10 | 2 | 83.04 [78.13, 87.50] | 48.54 [43.57, 53.64] | 45.91 [40.93, 51.17] |
| RLVR | tutor | plain | answer_tag | 343 / 344 | 1 (1) | 7 | 1 | 85.42 [80.41, 89.83] | 83.38 [78.13, 88.01] | 79.59 [73.55, 84.84] |
| RLVR | tutor | plain | result_tag | 341 / 344 | 3 (3) | 5 | 3 | 90.03 [85.29, 94.13] | 89.44 [86.13, 92.40] | 85.34 [80.35, 89.53] |
| RLVR | tutor | plain | boxed_step | 320 / 344 | 24 (24) | 25 | 24 | 2.19 [0.64, 3.82] | 46.88 [40.12, 53.73] | 1.25 [0.30, 2.54] |
| RLVR | ceiling | plain | none | 343 / 344 | 1 (1) | 2 | 1 | 96.79 [93.02, 100.00] | 96.79 [94.75, 98.54] | 93.59 [89.41, 97.38] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/DPO/boxed | 1.78 [-0.25, 3.82] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | 0.33 [-1.99, 2.62] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | 0.33 [-2.22, 2.68] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | 0.89 [-1.13, 2.92] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -63.59 [-69.30, -57.99] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | 0.81 [-1.07, 2.83] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 8.16 [3.29, 12.87] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | 1.75 [-2.86, 6.41] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 4.14 [-0.53, 8.65] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 8.74 [4.39, 12.90] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -79.10 [-84.11, -73.62] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | 4.68 [0.72, 8.60] | 2000 / 2000 | ok |
| H6 | 3.87 [-0.35, 7.94] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -4.06 [-7.22, -0.68] | 2000 / 2000 | ok |
| H7/difference_in_differences | -3.98 [-7.44, -0.22] | 2000 / 2000 | ok |
| H8 | -79.10 [-84.11, -73.62] | 2000 / 2000 | ok |
| per_format_H6/boxed | 6.38 [1.15, 11.61] | 2000 / 2000 | ok |
| per_format_H6/dapo | 1.43 [-3.27, 6.24] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | 3.81 [-0.84, 8.69] | 2000 / 2000 | ok |

#### All-three-checkpoint common ceiling solved problems

Problems: 81; status: ok.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | ceiling | plain | none | 324 / 324 | 0 (0) | 0 | 0 | 99.38 [98.46, 100.00] | 91.98 [88.27, 95.06] | 91.36 [87.65, 94.75] |
| DPO | tutor | plain | none | 323 / 324 | 1 (1) | 2 | 1 | 96.90 [95.05, 98.46] | 10.84 [6.85, 15.17] | 10.22 [6.48, 14.60] |
| DPO | tutor | plain | boxed | 324 / 324 | 0 (0) | 1 | 0 | 99.07 [97.84, 100.00] | 98.77 [97.53, 99.69] | 98.15 [96.60, 99.38] |
| DPO | tutor | plain | dapo | 324 / 324 | 0 (0) | 0 | 0 | 97.84 [95.99, 99.38] | 44.75 [39.81, 49.70] | 44.14 [39.20, 49.38] |
| DPO | tutor | plain | answer_tag | 324 / 324 | 0 (0) | 2 | 0 | 97.53 [95.37, 99.07] | 89.51 [84.26, 94.14] | 88.58 [83.33, 93.21] |
| DPO | tutor | plain | result_tag | 324 / 324 | 0 (0) | 2 | 0 | 98.15 [96.30, 99.69] | 94.14 [90.43, 97.22] | 93.52 [89.81, 96.91] |
| DPO | tutor | plain | boxed_step | 318 / 324 | 6 (6) | 7 | 6 | 31.13 [26.25, 36.60] | 45.91 [39.69, 52.07] | 20.44 [16.14, 25.24] |
| DPO | ceiling | plain | none | 323 / 324 | 1 (1) | 1 | 1 | 99.07 [97.82, 100.00] | 82.97 [77.47, 87.85] | 81.73 [76.16, 86.69] |
| RLVR | tutor | plain | none | 322 / 324 | 2 (2) | 6 | 2 | 84.47 [80.50, 88.24] | 9.01 [5.59, 12.77] | 8.39 [5.00, 12.11] |
| RLVR | tutor | plain | boxed | 323 / 324 | 1 (1) | 4 | 1 | 92.57 [89.81, 95.33] | 86.07 [82.55, 89.51] | 86.07 [82.55, 89.51] |
| RLVR | tutor | plain | dapo | 324 / 324 | 0 (0) | 8 | 0 | 86.11 [82.41, 89.81] | 47.53 [42.28, 52.47] | 47.22 [41.98, 52.16] |
| RLVR | tutor | plain | answer_tag | 324 / 324 | 0 (0) | 6 | 0 | 88.27 [84.26, 91.98] | 82.72 [77.16, 88.27] | 82.10 [76.54, 87.35] |
| RLVR | tutor | plain | result_tag | 322 / 324 | 2 (2) | 3 | 2 | 93.48 [90.31, 96.28] | 89.44 [86.25, 92.55] | 88.51 [85.09, 91.93] |
| RLVR | tutor | plain | boxed_step | 305 / 324 | 19 (19) | 20 | 19 | 2.30 [0.67, 4.14] | 47.21 [40.06, 54.75] | 1.31 [0.32, 2.65] |
| RLVR | ceiling | plain | none | 324 / 324 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 96.91 [94.75, 98.77] | 96.91 [94.75, 98.77] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/DPO/boxed | 2.17 [0.32, 4.32] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | 0.94 [-1.23, 3.09] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | 0.63 [-1.85, 3.09] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | 1.24 [-0.90, 3.41] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -65.77 [-70.73, -60.23] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | 1.24 [-0.60, 3.19] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 8.10 [3.48, 12.69] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | 1.64 [-3.18, 6.28] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 3.80 [-0.58, 8.41] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 9.01 [4.37, 13.56] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -82.18 [-86.27, -77.76] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | 4.51 [0.79, 8.43] | 2000 / 2000 | ok |
| H6 | 3.27 [-0.53, 7.30] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -4.49 [-7.60, -1.09] | 2000 / 2000 | ok |
| H7/difference_in_differences | -4.49 [-8.21, -0.67] | 2000 / 2000 | ok |
| H8 | -82.18 [-86.27, -77.76] | 2000 / 2000 | ok |
| per_format_H6/boxed | 5.93 [1.24, 10.60] | 2000 / 2000 | ok |
| per_format_H6/dapo | 0.70 [-3.98, 5.56] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | 3.17 [-1.23, 8.07] | 2000 / 2000 | ok |

