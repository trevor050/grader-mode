# Phase 1c results

Gold minimum: 10; 26364 responses.

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.

Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.

Think ladder: deferred (DESIGN.md Deviation 6).

## instruct

Checkpoints: complete; available: SFT, DPO, RLVR; missing: none.
Inference eligible: True; source problems: 200 / 200.

### All selected problems

Problems: 169; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | tutor | plain | none | 676 / 676 | 0 (0) | 5 | 0 | 80.03 [75.88, 83.88] | 4.44 [2.96, 6.21] | 3.40 [2.07, 4.88] |
| SFT | tutor | plain | boxed | 676 / 676 | 0 (0) | 1 | 0 | 85.50 [81.51, 89.20] | 73.82 [69.97, 77.66] | 64.05 [59.47, 68.49] |
| SFT | tutor | plain | dapo | 676 / 676 | 0 (0) | 3 | 0 | 81.21 [76.62, 85.36] | 48.37 [44.08, 52.37] | 42.46 [37.72, 46.89] |
| SFT | tutor | plain | answer_tag | 676 / 676 | 0 (0) | 4 | 0 | 76.48 [72.34, 80.77] | 38.46 [34.46, 42.60] | 32.99 [28.70, 37.13] |
| SFT | tutor | plain | result_tag | 676 / 676 | 0 (0) | 5 | 0 | 76.63 [72.19, 80.92] | 48.67 [44.67, 52.81] | 42.16 [37.87, 46.60] |
| SFT | tutor | plain | boxed_step | 676 / 676 | 0 (0) | 1 | 0 | 22.19 [19.08, 25.59] | 20.71 [17.31, 24.41] | 8.14 [6.07, 10.80] |
| SFT | tutor | env | none | 676 / 676 | 0 (0) | 5 | 0 | 70.41 [65.83, 75.00] | 7.69 [5.77, 9.62] | 6.80 [4.88, 8.58] |
| SFT | tutor | env | boxed | 676 / 676 | 0 (0) | 11 | 0 | 81.66 [77.37, 85.65] | 55.62 [51.78, 59.76] | 48.37 [44.08, 52.81] |
| SFT | tutor | env | dapo | 676 / 676 | 0 (0) | 10 | 0 | 71.30 [67.01, 75.15] | 60.21 [56.21, 64.35] | 47.93 [43.78, 52.22] |
| SFT | tutor | env | answer_tag | 676 / 676 | 0 (0) | 6 | 0 | 65.83 [61.24, 70.41] | 24.56 [21.45, 27.66] | 19.82 [16.72, 22.93] |
| SFT | tutor | env | result_tag | 676 / 676 | 0 (0) | 11 | 0 | 71.15 [66.42, 75.59] | 25.30 [21.60, 28.85] | 19.97 [16.57, 23.22] |
| SFT | tutor | env | boxed_step | 676 / 676 | 0 (0) | 1 | 0 | 14.35 [11.54, 17.31] | 21.60 [18.34, 25.00] | 4.59 [3.11, 6.21] |
| SFT | ceiling | plain | none | 676 / 676 | 0 (0) | 4 | 0 | 87.28 [83.14, 91.12] | 53.70 [49.85, 57.40] | 46.75 [42.46, 50.89] |
| DPO | tutor | plain | none | 676 / 676 | 0 (0) | 83 | 0 | 70.71 [66.42, 74.85] | 1.78 [0.74, 2.96] | 1.63 [0.59, 2.81] |
| DPO | tutor | plain | boxed | 676 / 676 | 0 (0) | 66 | 0 | 70.41 [65.83, 75.00] | 15.53 [12.43, 18.64] | 14.20 [11.39, 17.01] |
| DPO | tutor | plain | dapo | 676 / 676 | 0 (0) | 45 | 0 | 68.20 [63.31, 72.78] | 6.80 [4.88, 8.73] | 5.92 [4.29, 7.84] |
| DPO | tutor | plain | answer_tag | 676 / 676 | 0 (0) | 56 | 0 | 63.76 [58.88, 68.64] | 8.28 [6.36, 10.36] | 7.54 [5.77, 9.47] |
| DPO | tutor | plain | result_tag | 676 / 676 | 0 (0) | 43 | 0 | 65.53 [60.36, 70.27] | 26.33 [23.07, 29.88] | 22.04 [18.79, 25.59] |
| DPO | tutor | plain | boxed_step | 676 / 676 | 0 (0) | 11 | 0 | 3.85 [2.37, 5.62] | 14.35 [11.54, 17.46] | 1.18 [0.44, 1.92] |
| DPO | tutor | env | none | 676 / 676 | 0 (0) | 34 | 0 | 46.89 [41.72, 51.92] | 1.48 [0.74, 2.37] | 1.33 [0.59, 2.22] |
| DPO | tutor | env | boxed | 676 / 676 | 0 (0) | 45 | 0 | 60.06 [55.32, 64.94] | 6.66 [4.88, 8.58] | 5.62 [3.99, 7.40] |
| DPO | tutor | env | dapo | 676 / 676 | 0 (0) | 36 | 0 | 50.89 [46.01, 56.07] | 9.17 [7.25, 11.24] | 7.69 [5.77, 9.62] |
| DPO | tutor | env | answer_tag | 676 / 676 | 0 (0) | 26 | 0 | 48.82 [43.79, 53.70] | 7.25 [5.18, 9.32] | 6.21 [4.29, 8.14] |
| DPO | tutor | env | result_tag | 676 / 676 | 0 (0) | 28 | 0 | 52.07 [46.89, 57.25] | 15.83 [13.17, 18.64] | 11.98 [9.47, 14.64] |
| DPO | tutor | env | boxed_step | 676 / 676 | 0 (0) | 9 | 0 | 3.25 [1.92, 4.73] | 15.53 [12.28, 19.08] | 0.89 [0.30, 1.63] |
| DPO | ceiling | plain | none | 676 / 676 | 0 (0) | 65 | 0 | 90.09 [86.24, 93.64] | 37.57 [32.84, 42.31] | 35.80 [31.07, 40.68] |
| RLVR | tutor | plain | none | 676 / 676 | 0 (0) | 64 | 0 | 57.25 [52.51, 61.98] | 2.96 [1.63, 4.29] | 2.22 [1.18, 3.40] |
| RLVR | tutor | plain | boxed | 676 / 676 | 0 (0) | 47 | 0 | 58.73 [53.70, 63.61] | 8.43 [6.21, 10.80] | 7.54 [5.47, 9.62] |
| RLVR | tutor | plain | dapo | 676 / 676 | 0 (0) | 33 | 0 | 46.75 [41.72, 51.78] | 14.50 [11.54, 17.60] | 13.17 [10.06, 16.42] |
| RLVR | tutor | plain | answer_tag | 676 / 676 | 0 (0) | 33 | 0 | 45.56 [40.38, 50.74] | 5.92 [4.14, 7.84] | 5.03 [3.40, 6.80] |
| RLVR | tutor | plain | result_tag | 676 / 676 | 0 (0) | 36 | 0 | 50.89 [46.01, 55.92] | 12.87 [10.36, 15.53] | 10.80 [8.43, 13.02] |
| RLVR | tutor | plain | boxed_step | 676 / 676 | 0 (0) | 2 | 0 | 2.07 [1.04, 3.40] | 15.09 [12.28, 17.90] | 1.18 [0.44, 2.07] |
| RLVR | tutor | env | none | 676 / 676 | 0 (0) | 26 | 0 | 28.85 [24.11, 33.88] | 0.59 [0.15, 1.18] | 0.59 [0.15, 1.18] |
| RLVR | tutor | env | boxed | 676 / 676 | 0 (0) | 20 | 0 | 43.93 [38.31, 49.41] | 8.14 [5.77, 10.80] | 6.51 [4.44, 8.88] |
| RLVR | tutor | env | dapo | 676 / 676 | 0 (0) | 9 | 0 | 34.32 [29.29, 39.35] | 17.90 [15.09, 20.71] | 14.64 [11.98, 17.46] |
| RLVR | tutor | env | answer_tag | 676 / 676 | 0 (0) | 19 | 0 | 28.55 [23.82, 33.28] | 7.40 [5.62, 9.32] | 4.59 [2.96, 6.07] |
| RLVR | tutor | env | result_tag | 676 / 676 | 0 (0) | 21 | 0 | 35.50 [30.18, 40.53] | 10.06 [7.84, 12.28] | 6.95 [5.03, 9.02] |
| RLVR | tutor | env | boxed_step | 676 / 676 | 0 (0) | 1 | 0 | 0.44 [0.00, 1.04] | 11.83 [8.88, 15.09] | 0.30 [0.00, 0.74] |
| RLVR | ceiling | plain | none | 676 / 676 | 0 (0) | 62 | 0 | 91.12 [87.43, 94.38] | 77.96 [74.11, 81.80] | 73.67 [69.08, 77.96] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/SFT/boxed | 5.47 [2.37, 8.73] | 2000 / 2000 | ok |
| plain_format_effects/SFT/dapo | 1.18 [-2.66, 4.73] | 2000 / 2000 | ok |
| plain_format_effects/SFT/answer_tag | -3.55 [-7.40, 0.30] | 2000 / 2000 | ok |
| plain_format_effects/SFT/result_tag | -3.40 [-6.80, 0.00] | 2000 / 2000 | ok |
| plain_format_effects/SFT/boxed_step | -57.84 [-62.28, -53.11] | 2000 / 2000 | ok |
| plain_format_effects/SFT/pooled_rl | 1.04 [-1.68, 3.99] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed | -0.30 [-4.29, 3.85] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | -2.51 [-6.51, 1.33] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | -6.95 [-10.95, -3.11] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | -5.18 [-9.17, -1.18] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -66.86 [-70.86, -62.57] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | -3.25 [-6.46, -0.15] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 1.48 [-3.25, 6.07] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | -10.50 [-15.39, -5.47] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | -11.69 [-16.27, -7.10] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | -6.36 [-11.09, -1.77] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -55.18 [-59.91, -50.44] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | -6.90 [-11.10, -2.91] | 2000 / 2000 | ok |
| H6 | -3.65 [-8.88, 1.43] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -0.54 [-4.14, 3.11] | 2000 / 2000 | ok |
| H7/difference_in_differences | -2.47 [-7.69, 2.47] | 2000 / 2000 | ok |
| H8 | -55.18 [-59.91, -50.44] | 2000 / 2000 | ok |
| per_format_H6/boxed | 1.78 [-4.59, 7.84] | 2000 / 2000 | ok |
| per_format_H6/dapo | -7.99 [-14.20, -1.62] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | -4.73 [-10.65, 1.33] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed | 5.77 [0.74, 10.65] | 2000 / 2000 | ok |
| env_format_interactions/SFT/dapo | -0.30 [-6.21, 5.47] | 2000 / 2000 | ok |
| env_format_interactions/SFT/answer_tag | -1.04 [-6.51, 4.59] | 2000 / 2000 | ok |
| env_format_interactions/SFT/result_tag | 4.14 [-1.18, 9.32] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed_step | 1.78 [-4.14, 7.40] | 2000 / 2000 | ok |
| env_format_interactions/SFT/pooled_rl | 1.48 [-2.81, 5.87] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed | 13.46 [7.99, 19.23] | 2000 / 2000 | ok |
| env_format_interactions/DPO/dapo | 6.51 [0.15, 13.47] | 2000 / 2000 | ok |
| env_format_interactions/DPO/answer_tag | 8.88 [3.10, 15.38] | 2000 / 2000 | ok |
| env_format_interactions/DPO/result_tag | 10.36 [3.99, 17.01] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed_step | 23.22 [18.49, 28.25] | 2000 / 2000 | ok |
| env_format_interactions/DPO/pooled_rl | 9.62 [4.73, 14.94] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed | 13.61 [7.84, 19.82] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 15.98 [9.62, 22.19] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 11.39 [5.18, 17.60] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 13.02 [7.40, 18.64] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 26.78 [21.89, 31.80] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 13.66 [8.68, 18.89] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed | -5.77 [-10.80, -0.89] | 2000 / 2000 | ok |
| sft_dpo_format_changes/dapo | -3.70 [-8.73, 1.18] | 2000 / 2000 | ok |
| sft_dpo_format_changes/answer_tag | -3.40 [-8.58, 1.63] | 2000 / 2000 | ok |
| sft_dpo_format_changes/result_tag | -1.78 [-6.66, 3.25] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed_step | -9.02 [-14.05, -3.85] | 2000 / 2000 | ok |
| sft_dpo_format_changes/pooled_rl | -4.29 [-8.23, -0.59] | 2000 / 2000 | ok |

### All-three-checkpoint common ceiling solved problems

Problems: 105; status: ok.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | tutor | plain | none | 420 / 420 | 0 (0) | 1 | 0 | 85.48 [81.67, 89.05] | 4.76 [2.86, 6.90] | 3.81 [2.14, 5.71] |
| SFT | tutor | plain | boxed | 420 / 420 | 0 (0) | 0 | 0 | 92.14 [88.57, 95.48] | 75.71 [70.95, 79.77] | 71.19 [66.19, 75.95] |
| SFT | tutor | plain | dapo | 420 / 420 | 0 (0) | 2 | 0 | 87.86 [83.81, 91.67] | 52.86 [47.86, 58.10] | 50.00 [44.76, 55.48] |
| SFT | tutor | plain | answer_tag | 420 / 420 | 0 (0) | 2 | 0 | 81.90 [77.62, 86.19] | 40.71 [34.76, 46.20] | 38.33 [32.62, 44.05] |
| SFT | tutor | plain | result_tag | 420 / 420 | 0 (0) | 1 | 0 | 82.14 [77.38, 86.67] | 52.38 [47.62, 57.14] | 50.48 [45.71, 55.24] |
| SFT | tutor | plain | boxed_step | 420 / 420 | 0 (0) | 1 | 0 | 23.33 [19.05, 27.62] | 18.10 [13.57, 23.10] | 8.57 [5.48, 11.90] |
| SFT | tutor | env | none | 420 / 420 | 0 (0) | 3 | 0 | 74.05 [68.81, 79.05] | 8.10 [5.48, 10.95] | 8.10 [5.48, 10.95] |
| SFT | tutor | env | boxed | 420 / 420 | 0 (0) | 6 | 0 | 87.86 [83.81, 91.43] | 56.90 [52.14, 61.67] | 53.81 [48.81, 58.81] |
| SFT | tutor | env | dapo | 420 / 420 | 0 (0) | 5 | 0 | 77.62 [73.57, 81.67] | 61.90 [56.67, 67.14] | 54.29 [49.05, 59.52] |
| SFT | tutor | env | answer_tag | 420 / 420 | 0 (0) | 0 | 0 | 69.76 [64.29, 75.00] | 25.71 [21.67, 30.24] | 23.10 [19.29, 27.15] |
| SFT | tutor | env | result_tag | 420 / 420 | 0 (0) | 4 | 0 | 75.71 [70.48, 80.71] | 24.76 [19.99, 30.00] | 22.62 [18.10, 27.38] |
| SFT | tutor | env | boxed_step | 420 / 420 | 0 (0) | 1 | 0 | 15.48 [11.67, 19.53] | 20.71 [16.67, 24.76] | 5.00 [3.10, 7.14] |
| SFT | ceiling | plain | none | 420 / 420 | 0 (0) | 1 | 0 | 96.43 [94.52, 98.10] | 61.43 [57.14, 65.71] | 58.81 [54.29, 63.33] |
| DPO | tutor | plain | none | 420 / 420 | 0 (0) | 31 | 0 | 75.95 [71.43, 80.71] | 1.67 [0.48, 3.10] | 1.67 [0.48, 3.10] |
| DPO | tutor | plain | boxed | 420 / 420 | 0 (0) | 23 | 0 | 72.62 [67.14, 77.86] | 15.48 [11.67, 19.52] | 14.52 [10.71, 18.33] |
| DPO | tutor | plain | dapo | 420 / 420 | 0 (0) | 12 | 0 | 73.10 [67.86, 78.10] | 7.38 [5.23, 9.76] | 6.90 [4.52, 9.29] |
| DPO | tutor | plain | answer_tag | 420 / 420 | 0 (0) | 26 | 0 | 69.29 [63.57, 75.00] | 9.76 [7.14, 12.62] | 8.81 [6.43, 11.43] |
| DPO | tutor | plain | result_tag | 420 / 420 | 0 (0) | 11 | 0 | 69.52 [64.52, 74.76] | 27.62 [23.57, 32.14] | 24.76 [20.71, 29.05] |
| DPO | tutor | plain | boxed_step | 420 / 420 | 0 (0) | 5 | 0 | 3.81 [1.90, 6.19] | 16.43 [12.62, 20.24] | 1.19 [0.24, 2.38] |
| DPO | tutor | env | none | 420 / 420 | 0 (0) | 9 | 0 | 49.29 [42.86, 55.95] | 1.43 [0.48, 2.62] | 1.19 [0.24, 2.38] |
| DPO | tutor | env | boxed | 420 / 420 | 0 (0) | 15 | 0 | 63.57 [57.38, 69.29] | 6.43 [4.29, 8.81] | 5.95 [4.05, 8.10] |
| DPO | tutor | env | dapo | 420 / 420 | 0 (0) | 8 | 0 | 51.67 [45.24, 58.33] | 8.33 [5.71, 11.19] | 6.90 [4.52, 9.52] |
| DPO | tutor | env | answer_tag | 420 / 420 | 0 (0) | 5 | 0 | 50.71 [44.29, 57.38] | 6.67 [4.05, 9.52] | 5.24 [2.86, 7.86] |
| DPO | tutor | env | result_tag | 420 / 420 | 0 (0) | 5 | 0 | 53.10 [46.43, 59.53] | 15.24 [11.67, 18.81] | 12.38 [9.05, 15.48] |
| DPO | tutor | env | boxed_step | 420 / 420 | 0 (0) | 4 | 0 | 3.10 [1.67, 4.77] | 17.86 [13.57, 22.14] | 1.19 [0.24, 2.38] |
| DPO | ceiling | plain | none | 420 / 420 | 0 (0) | 22 | 0 | 97.86 [96.19, 99.29] | 55.00 [50.71, 59.76] | 54.76 [50.48, 59.52] |
| RLVR | tutor | plain | none | 420 / 420 | 0 (0) | 23 | 0 | 60.48 [54.52, 66.19] | 2.38 [0.95, 4.29] | 1.90 [0.71, 3.33] |
| RLVR | tutor | plain | boxed | 420 / 420 | 0 (0) | 12 | 0 | 62.86 [56.90, 69.05] | 7.86 [5.24, 10.71] | 7.86 [5.24, 10.71] |
| RLVR | tutor | plain | dapo | 420 / 420 | 0 (0) | 8 | 0 | 49.29 [42.86, 55.48] | 14.05 [10.48, 17.86] | 13.33 [9.52, 17.15] |
| RLVR | tutor | plain | answer_tag | 420 / 420 | 0 (0) | 12 | 0 | 47.86 [41.43, 54.05] | 5.71 [3.33, 8.10] | 5.48 [3.33, 7.86] |
| RLVR | tutor | plain | result_tag | 420 / 420 | 0 (0) | 10 | 0 | 52.62 [46.19, 58.81] | 11.90 [9.04, 15.00] | 11.19 [8.33, 14.29] |
| RLVR | tutor | plain | boxed_step | 420 / 420 | 0 (0) | 1 | 0 | 2.62 [0.95, 4.52] | 16.43 [12.38, 20.71] | 1.43 [0.48, 2.62] |
| RLVR | tutor | env | none | 420 / 420 | 0 (0) | 7 | 0 | 29.76 [24.04, 35.95] | 0.48 [0.00, 1.19] | 0.48 [0.00, 1.19] |
| RLVR | tutor | env | boxed | 420 / 420 | 0 (0) | 8 | 0 | 45.24 [38.10, 51.90] | 5.95 [3.57, 8.57] | 5.71 [3.57, 8.10] |
| RLVR | tutor | env | dapo | 420 / 420 | 0 (0) | 4 | 0 | 32.62 [26.43, 38.82] | 14.76 [11.19, 18.57] | 13.81 [10.48, 17.62] |
| RLVR | tutor | env | answer_tag | 420 / 420 | 0 (0) | 5 | 0 | 29.29 [23.10, 35.71] | 6.43 [4.29, 8.81] | 5.00 [3.10, 7.38] |
| RLVR | tutor | env | result_tag | 420 / 420 | 0 (0) | 8 | 0 | 32.38 [25.48, 39.05] | 7.62 [5.24, 10.24] | 5.48 [3.33, 7.86] |
| RLVR | tutor | env | boxed_step | 420 / 420 | 0 (0) | 1 | 0 | 0.48 [0.00, 1.19] | 10.71 [7.14, 15.01] | 0.24 [0.00, 0.71] |
| RLVR | ceiling | plain | none | 420 / 420 | 0 (0) | 21 | 0 | 97.86 [96.43, 99.05] | 86.19 [82.38, 89.76] | 86.19 [82.38, 89.76] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/SFT/boxed | 6.67 [3.33, 10.00] | 2000 / 2000 | ok |
| plain_format_effects/SFT/dapo | 2.38 [-1.43, 6.19] | 2000 / 2000 | ok |
| plain_format_effects/SFT/answer_tag | -3.57 [-7.86, 0.95] | 2000 / 2000 | ok |
| plain_format_effects/SFT/result_tag | -3.33 [-7.62, 0.71] | 2000 / 2000 | ok |
| plain_format_effects/SFT/boxed_step | -62.14 [-67.14, -57.14] | 2000 / 2000 | ok |
| plain_format_effects/SFT/pooled_rl | 1.83 [-1.03, 4.76] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed | -3.33 [-8.10, 1.67] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | -2.86 [-7.62, 1.90] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | -6.67 [-11.90, -1.67] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | -6.43 [-11.43, -1.67] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -72.14 [-76.90, -67.61] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | -4.29 [-8.41, -0.39] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 2.38 [-3.81, 9.05] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | -11.19 [-17.38, -4.76] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | -12.62 [-18.33, -6.43] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | -7.86 [-14.05, -1.67] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -57.86 [-63.33, -52.14] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | -7.14 [-12.38, -1.51] | 2000 / 2000 | ok |
| H6 | -2.86 [-9.52, 4.13] | 2000 / 2000 | ok |
| H7/rlvr_specificity | 0.71 [-3.65, 5.24] | 2000 / 2000 | ok |
| H7/difference_in_differences | -1.43 [-7.30, 4.68] | 2000 / 2000 | ok |
| H8 | -57.86 [-63.33, -52.14] | 2000 / 2000 | ok |
| per_format_H6/boxed | 5.71 [-2.62, 14.76] | 2000 / 2000 | ok |
| per_format_H6/dapo | -8.33 [-16.19, 0.24] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | -5.95 [-13.33, 1.67] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed | 7.14 [1.66, 13.10] | 2000 / 2000 | ok |
| env_format_interactions/SFT/dapo | 1.19 [-5.71, 8.10] | 2000 / 2000 | ok |
| env_format_interactions/SFT/answer_tag | -0.71 [-7.62, 6.19] | 2000 / 2000 | ok |
| env_format_interactions/SFT/result_tag | 5.00 [-1.90, 11.67] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed_step | 3.57 [-3.34, 10.95] | 2000 / 2000 | ok |
| env_format_interactions/SFT/pooled_rl | 2.54 [-2.54, 8.02] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed | 17.62 [10.71, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/DPO/dapo | 5.24 [-3.10, 14.05] | 2000 / 2000 | ok |
| env_format_interactions/DPO/answer_tag | 8.10 [-0.01, 16.19] | 2000 / 2000 | ok |
| env_format_interactions/DPO/result_tag | 10.24 [1.43, 19.05] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed_step | 25.95 [19.52, 32.62] | 2000 / 2000 | ok |
| env_format_interactions/DPO/pooled_rl | 10.32 [3.81, 17.06] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed | 13.10 [5.24, 20.95] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 14.05 [5.71, 22.38] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 12.14 [4.05, 19.76] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 10.48 [3.10, 17.62] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 28.57 [22.14, 35.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 13.10 [6.03, 19.76] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed | -10.00 [-15.48, -4.52] | 2000 / 2000 | ok |
| sft_dpo_format_changes/dapo | -5.24 [-11.43, 0.95] | 2000 / 2000 | ok |
| sft_dpo_format_changes/answer_tag | -3.10 [-10.00, 3.57] | 2000 / 2000 | ok |
| sft_dpo_format_changes/result_tag | -3.10 [-9.05, 2.62] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed_step | -10.00 [-16.67, -3.33] | 2000 / 2000 | ok |
| sft_dpo_format_changes/pooled_rl | -6.11 [-11.11, -1.35] | 2000 / 2000 | ok |

## Execution

GPU function seconds: 4564.392524. GPU function hours: 1.267887.

GPU function cost estimate USD: 2.472379. L40S USD/hour: 1.95. Budget USD: 5.00.

Allocation wall upper seconds: 4707.483813. Allocation wall upper GPU cost estimate USD: 2.549887.

Modal app reported GPU cost USD: 2.53607492. Modal app reported total cost USD: 3.16128112.

GPU container seconds: 4683.137594. GPU container hours: 1.300872. GPU container cost estimate USD: 2.536700.

Idle replacement containers: 2. Idle replacement GPU seconds: 101.753545. Idle replacement GPU cost estimate USD: 0.055117.

Phase 1c containers after completion: 0. GPU timeout seconds: 5400. Minimum/buffer containers after update: 0/0. Think calls: 0.

Tutor prompt: `prompts/system_tutor_strong.txt`. [Execution and billing](execution.json).
