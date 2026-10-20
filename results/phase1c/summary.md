# Phase 1c results

Gold minimum: None; 31200 responses.

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.

Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.

Think ladder: deferred (DESIGN.md Deviation 6).

## instruct

Checkpoints: complete; available: SFT, DPO, RLVR; missing: none.
Inference eligible: True; source problems: 200 / 200.

### All selected problems

Problems: 200; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | tutor | plain | none | 800 / 800 | 0 (0) | 6 | 0 | 80.88 [77.25, 84.50] | 4.50 [3.12, 6.00] | 3.25 [2.12, 4.50] |
| SFT | tutor | plain | boxed | 800 / 800 | 0 (0) | 2 | 0 | 86.62 [83.12, 89.88] | 72.88 [69.12, 76.75] | 62.88 [58.38, 67.25] |
| SFT | tutor | plain | dapo | 800 / 800 | 0 (0) | 6 | 0 | 81.88 [77.88, 85.75] | 46.88 [42.88, 50.75] | 40.88 [36.75, 45.00] |
| SFT | tutor | plain | answer_tag | 800 / 800 | 0 (0) | 7 | 0 | 77.12 [73.25, 81.00] | 39.12 [35.25, 43.12] | 33.50 [29.62, 37.25] |
| SFT | tutor | plain | result_tag | 800 / 800 | 0 (0) | 6 | 0 | 77.12 [73.12, 81.00] | 47.25 [43.37, 51.12] | 40.50 [36.62, 44.75] |
| SFT | tutor | plain | boxed_step | 800 / 800 | 0 (0) | 1 | 0 | 24.00 [20.75, 27.12] | 20.25 [17.12, 23.50] | 7.62 [5.50, 9.88] |
| SFT | tutor | env | none | 800 / 800 | 0 (0) | 9 | 0 | 71.88 [67.50, 76.13] | 7.88 [6.00, 9.75] | 6.62 [5.00, 8.38] |
| SFT | tutor | env | boxed | 800 / 800 | 0 (0) | 14 | 0 | 82.38 [78.62, 86.00] | 55.12 [51.12, 59.00] | 47.50 [43.25, 51.62] |
| SFT | tutor | env | dapo | 800 / 800 | 0 (0) | 12 | 0 | 72.25 [68.50, 75.88] | 60.62 [56.62, 64.50] | 48.00 [43.88, 52.00] |
| SFT | tutor | env | answer_tag | 800 / 800 | 0 (0) | 8 | 0 | 67.25 [63.12, 71.38] | 24.25 [21.00, 27.25] | 19.12 [16.25, 22.00] |
| SFT | tutor | env | result_tag | 800 / 800 | 0 (0) | 14 | 0 | 72.25 [68.12, 76.25] | 26.88 [23.25, 30.38] | 21.25 [18.12, 24.38] |
| SFT | tutor | env | boxed_step | 800 / 800 | 0 (0) | 1 | 0 | 16.88 [13.88, 20.00] | 21.75 [19.00, 24.62] | 5.00 [3.62, 6.50] |
| SFT | ceiling | plain | none | 800 / 800 | 0 (0) | 6 | 0 | 88.00 [84.25, 91.50] | 53.00 [49.50, 56.75] | 45.75 [41.62, 49.75] |
| DPO | tutor | plain | none | 800 / 800 | 0 (0) | 106 | 0 | 72.25 [68.38, 76.25] | 1.75 [0.75, 2.88] | 1.62 [0.75, 2.75] |
| DPO | tutor | plain | boxed | 800 / 800 | 0 (0) | 79 | 0 | 72.88 [68.62, 76.75] | 15.75 [12.88, 18.75] | 14.37 [11.75, 17.12] |
| DPO | tutor | plain | dapo | 800 / 800 | 0 (0) | 60 | 0 | 69.00 [64.75, 73.12] | 7.00 [5.25, 8.88] | 6.00 [4.25, 7.62] |
| DPO | tutor | plain | answer_tag | 800 / 800 | 0 (0) | 71 | 0 | 66.00 [61.62, 70.50] | 8.25 [6.38, 10.12] | 7.25 [5.62, 8.88] |
| DPO | tutor | plain | result_tag | 800 / 800 | 0 (0) | 57 | 0 | 68.50 [64.12, 72.75] | 25.75 [22.62, 29.00] | 21.88 [19.00, 25.12] |
| DPO | tutor | plain | boxed_step | 800 / 800 | 0 (0) | 15 | 0 | 7.38 [4.88, 10.00] | 14.75 [12.12, 17.62] | 1.62 [0.88, 2.50] |
| DPO | tutor | env | none | 800 / 800 | 0 (0) | 46 | 0 | 49.88 [45.12, 54.87] | 1.38 [0.62, 2.25] | 1.25 [0.62, 2.00] |
| DPO | tutor | env | boxed | 800 / 800 | 0 (0) | 56 | 0 | 61.88 [57.00, 66.50] | 7.12 [5.50, 9.00] | 6.00 [4.38, 7.75] |
| DPO | tutor | env | dapo | 800 / 800 | 0 (0) | 47 | 0 | 53.37 [48.62, 58.13] | 8.88 [7.12, 10.75] | 7.25 [5.62, 9.00] |
| DPO | tutor | env | answer_tag | 800 / 800 | 0 (0) | 35 | 0 | 51.88 [47.12, 56.88] | 6.88 [5.00, 8.75] | 5.88 [4.12, 7.75] |
| DPO | tutor | env | result_tag | 800 / 800 | 0 (0) | 37 | 0 | 55.00 [50.25, 59.88] | 15.12 [12.62, 17.62] | 11.62 [9.38, 14.00] |
| DPO | tutor | env | boxed_step | 800 / 800 | 0 (0) | 12 | 0 | 6.38 [4.00, 9.12] | 14.88 [11.75, 17.88] | 0.88 [0.25, 1.50] |
| DPO | ceiling | plain | none | 800 / 800 | 0 (0) | 83 | 0 | 90.25 [86.75, 93.62] | 36.25 [32.00, 40.38] | 34.25 [30.12, 38.50] |
| RLVR | tutor | plain | none | 800 / 800 | 0 (0) | 81 | 0 | 59.13 [54.75, 63.62] | 2.50 [1.38, 3.75] | 1.88 [0.88, 2.88] |
| RLVR | tutor | plain | boxed | 800 / 800 | 0 (0) | 58 | 0 | 61.62 [57.12, 66.38] | 8.50 [6.38, 10.62] | 7.50 [5.62, 9.50] |
| RLVR | tutor | plain | dapo | 800 / 800 | 0 (0) | 44 | 0 | 48.50 [43.62, 53.25] | 13.88 [11.25, 16.62] | 12.62 [10.00, 15.38] |
| RLVR | tutor | plain | answer_tag | 800 / 800 | 0 (0) | 43 | 0 | 48.25 [43.25, 53.25] | 5.88 [4.25, 7.62] | 5.12 [3.62, 6.75] |
| RLVR | tutor | plain | result_tag | 800 / 800 | 0 (0) | 46 | 0 | 54.25 [49.50, 58.88] | 13.75 [11.50, 16.12] | 11.62 [9.50, 13.88] |
| RLVR | tutor | plain | boxed_step | 800 / 800 | 0 (0) | 2 | 0 | 3.38 [1.88, 5.25] | 14.75 [12.12, 17.62] | 1.12 [0.50, 1.88] |
| RLVR | tutor | env | none | 800 / 800 | 0 (0) | 36 | 0 | 32.62 [27.75, 37.62] | 1.00 [0.38, 1.62] | 0.88 [0.37, 1.50] |
| RLVR | tutor | env | boxed | 800 / 800 | 0 (0) | 30 | 0 | 47.12 [41.75, 52.25] | 8.50 [6.12, 10.88] | 6.75 [4.75, 8.88] |
| RLVR | tutor | env | dapo | 800 / 800 | 0 (0) | 19 | 0 | 36.88 [32.00, 41.75] | 18.00 [15.12, 20.75] | 14.62 [11.88, 17.25] |
| RLVR | tutor | env | answer_tag | 800 / 800 | 0 (0) | 30 | 0 | 32.38 [27.75, 37.25] | 6.75 [5.25, 8.50] | 4.12 [2.88, 5.50] |
| RLVR | tutor | env | result_tag | 800 / 800 | 0 (0) | 27 | 0 | 38.88 [33.87, 43.75] | 10.38 [8.12, 12.50] | 7.25 [5.38, 9.12] |
| RLVR | tutor | env | boxed_step | 800 / 800 | 0 (0) | 1 | 0 | 1.38 [0.50, 2.50] | 11.62 [8.88, 14.50] | 0.38 [0.00, 0.88] |
| RLVR | ceiling | plain | none | 800 / 800 | 0 (0) | 74 | 0 | 91.75 [88.38, 94.75] | 76.62 [73.00, 80.25] | 72.25 [68.00, 76.38] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/SFT/boxed | 5.75 [2.88, 8.62] | 2000 / 2000 | ok |
| plain_format_effects/SFT/dapo | 1.00 [-2.25, 4.25] | 2000 / 2000 | ok |
| plain_format_effects/SFT/answer_tag | -3.75 [-7.00, -0.38] | 2000 / 2000 | ok |
| plain_format_effects/SFT/result_tag | -3.75 [-6.87, -0.62] | 2000 / 2000 | ok |
| plain_format_effects/SFT/boxed_step | -56.88 [-61.13, -52.75] | 2000 / 2000 | ok |
| plain_format_effects/SFT/pooled_rl | 1.00 [-1.46, 3.58] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed | 0.62 [-3.12, 4.25] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | -3.25 [-7.00, 0.25] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | -6.25 [-9.88, -2.88] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | -3.75 [-7.75, 0.12] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -64.88 [-69.00, -60.75] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | -2.96 [-5.96, -0.08] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 2.50 [-1.63, 6.87] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | -10.63 [-14.75, -6.25] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | -10.88 [-14.75, -6.75] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | -4.88 [-8.75, -0.75] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -55.75 [-60.38, -51.37] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | -6.33 [-9.71, -2.67] | 2000 / 2000 | ok |
| H6 | -3.38 [-7.67, 1.09] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -1.46 [-4.42, 1.50] | 2000 / 2000 | ok |
| H7/difference_in_differences | -2.25 [-6.62, 2.04] | 2000 / 2000 | ok |
| H8 | -55.75 [-60.38, -51.37] | 2000 / 2000 | ok |
| per_format_H6/boxed | 1.87 [-3.63, 7.75] | 2000 / 2000 | ok |
| per_format_H6/dapo | -7.37 [-13.00, -1.88] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | -4.63 [-9.50, 0.62] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed | 4.75 [0.38, 9.25] | 2000 / 2000 | ok |
| env_format_interactions/SFT/dapo | -0.62 [-5.63, 4.75] | 2000 / 2000 | ok |
| env_format_interactions/SFT/answer_tag | -0.88 [-6.00, 4.00] | 2000 / 2000 | ok |
| env_format_interactions/SFT/result_tag | 4.13 [-1.00, 8.88] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed_step | 1.87 [-3.25, 7.12] | 2000 / 2000 | ok |
| env_format_interactions/SFT/pooled_rl | 1.08 [-3.00, 5.04] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed | 11.38 [6.00, 16.88] | 2000 / 2000 | ok |
| env_format_interactions/DPO/dapo | 6.75 [1.00, 12.75] | 2000 / 2000 | ok |
| env_format_interactions/DPO/answer_tag | 8.25 [3.00, 13.63] | 2000 / 2000 | ok |
| env_format_interactions/DPO/result_tag | 8.88 [3.25, 14.50] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed_step | 21.38 [16.88, 26.12] | 2000 / 2000 | ok |
| env_format_interactions/DPO/pooled_rl | 8.79 [4.33, 13.37] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed | 12.00 [6.63, 17.37] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 14.88 [9.50, 20.75] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 10.63 [5.13, 16.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 11.13 [6.13, 15.88] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 24.50 [20.00, 29.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 12.50 [8.12, 16.92] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed | -5.13 [-9.63, -0.63] | 2000 / 2000 | ok |
| sft_dpo_format_changes/dapo | -4.25 [-8.63, 0.25] | 2000 / 2000 | ok |
| sft_dpo_format_changes/answer_tag | -2.50 [-7.38, 2.12] | 2000 / 2000 | ok |
| sft_dpo_format_changes/result_tag | 0.00 [-4.87, 4.87] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed_step | -8.00 [-12.87, -3.25] | 2000 / 2000 | ok |
| sft_dpo_format_changes/pooled_rl | -3.96 [-7.54, -0.29] | 2000 / 2000 | ok |

### All-three-checkpoint common ceiling solved problems

Problems: 120; status: ok.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT | tutor | plain | none | 480 / 480 | 0 (0) | 1 | 0 | 85.83 [82.29, 89.17] | 5.00 [3.12, 7.08] | 3.75 [2.29, 5.42] |
| SFT | tutor | plain | boxed | 480 / 480 | 0 (0) | 1 | 0 | 92.71 [89.38, 95.62] | 75.21 [70.83, 79.38] | 70.42 [65.62, 75.00] |
| SFT | tutor | plain | dapo | 480 / 480 | 0 (0) | 3 | 0 | 87.50 [83.54, 91.25] | 50.83 [45.83, 55.62] | 47.50 [42.50, 52.50] |
| SFT | tutor | plain | answer_tag | 480 / 480 | 0 (0) | 5 | 0 | 81.46 [77.08, 85.63] | 41.46 [36.04, 46.67] | 38.75 [33.54, 43.96] |
| SFT | tutor | plain | result_tag | 480 / 480 | 0 (0) | 2 | 0 | 82.50 [78.12, 86.67] | 51.25 [46.66, 55.42] | 48.96 [44.38, 53.54] |
| SFT | tutor | plain | boxed_step | 480 / 480 | 0 (0) | 1 | 0 | 25.83 [21.67, 30.00] | 18.12 [13.75, 22.71] | 8.54 [5.83, 11.46] |
| SFT | tutor | env | none | 480 / 480 | 0 (0) | 5 | 0 | 75.83 [70.83, 80.42] | 8.33 [5.83, 11.04] | 7.71 [5.42, 10.42] |
| SFT | tutor | env | boxed | 480 / 480 | 0 (0) | 8 | 0 | 88.12 [84.58, 91.46] | 56.25 [51.46, 60.83] | 53.12 [48.12, 57.92] |
| SFT | tutor | env | dapo | 480 / 480 | 0 (0) | 6 | 0 | 78.33 [74.37, 82.29] | 63.12 [58.54, 67.71] | 54.79 [50.21, 59.59] |
| SFT | tutor | env | answer_tag | 480 / 480 | 0 (0) | 1 | 0 | 71.25 [65.83, 76.46] | 25.62 [21.67, 29.79] | 22.50 [18.96, 26.25] |
| SFT | tutor | env | result_tag | 480 / 480 | 0 (0) | 6 | 0 | 77.50 [72.50, 82.08] | 26.04 [21.25, 30.84] | 23.75 [19.17, 28.33] |
| SFT | tutor | env | boxed_step | 480 / 480 | 0 (0) | 1 | 0 | 17.92 [13.75, 22.08] | 21.46 [17.92, 25.42] | 5.83 [3.75, 8.12] |
| SFT | ceiling | plain | none | 480 / 480 | 0 (0) | 2 | 0 | 96.67 [94.79, 98.33] | 61.67 [57.50, 65.83] | 58.96 [54.58, 63.33] |
| DPO | tutor | plain | none | 480 / 480 | 0 (0) | 42 | 0 | 77.71 [73.12, 81.88] | 1.67 [0.62, 2.92] | 1.67 [0.62, 2.92] |
| DPO | tutor | plain | boxed | 480 / 480 | 0 (0) | 32 | 0 | 74.58 [69.58, 79.58] | 15.42 [11.88, 19.17] | 14.58 [11.04, 18.33] |
| DPO | tutor | plain | dapo | 480 / 480 | 0 (0) | 22 | 0 | 73.54 [68.33, 78.33] | 7.29 [5.21, 9.58] | 6.67 [4.58, 8.96] |
| DPO | tutor | plain | answer_tag | 480 / 480 | 0 (0) | 34 | 0 | 71.88 [66.46, 77.08] | 10.00 [7.50, 12.50] | 8.96 [6.67, 11.25] |
| DPO | tutor | plain | result_tag | 480 / 480 | 0 (0) | 20 | 0 | 71.46 [66.67, 76.25] | 27.29 [23.33, 31.26] | 24.79 [20.83, 28.75] |
| DPO | tutor | plain | boxed_step | 480 / 480 | 0 (0) | 7 | 0 | 7.29 [4.17, 11.04] | 16.67 [12.92, 20.62] | 1.88 [0.83, 3.12] |
| DPO | tutor | env | none | 480 / 480 | 0 (0) | 17 | 0 | 52.29 [46.04, 58.75] | 1.46 [0.62, 2.50] | 1.25 [0.42, 2.29] |
| DPO | tutor | env | boxed | 480 / 480 | 0 (0) | 21 | 0 | 65.21 [59.38, 70.83] | 7.08 [4.79, 9.38] | 6.67 [4.58, 8.96] |
| DPO | tutor | env | dapo | 480 / 480 | 0 (0) | 14 | 0 | 54.17 [47.50, 60.42] | 8.33 [6.04, 10.83] | 6.88 [4.58, 9.38] |
| DPO | tutor | env | answer_tag | 480 / 480 | 0 (0) | 8 | 0 | 53.54 [47.08, 60.21] | 6.67 [4.17, 9.38] | 5.42 [3.12, 7.92] |
| DPO | tutor | env | result_tag | 480 / 480 | 0 (0) | 10 | 0 | 56.04 [49.79, 62.29] | 15.21 [11.88, 18.75] | 12.50 [9.38, 15.62] |
| DPO | tutor | env | boxed_step | 480 / 480 | 0 (0) | 5 | 0 | 5.62 [2.92, 8.75] | 16.46 [12.50, 20.62] | 1.04 [0.21, 2.08] |
| DPO | ceiling | plain | none | 480 / 480 | 0 (0) | 31 | 0 | 97.29 [95.62, 98.75] | 54.37 [50.21, 58.75] | 54.17 [50.00, 58.54] |
| RLVR | tutor | plain | none | 480 / 480 | 0 (0) | 32 | 0 | 62.29 [56.88, 68.12] | 2.08 [0.83, 3.54] | 1.67 [0.62, 2.92] |
| RLVR | tutor | plain | boxed | 480 / 480 | 0 (0) | 17 | 0 | 65.62 [59.58, 71.46] | 7.92 [5.42, 10.83] | 7.92 [5.42, 10.83] |
| RLVR | tutor | plain | dapo | 480 / 480 | 0 (0) | 15 | 0 | 51.25 [44.79, 57.30] | 13.33 [10.00, 17.08] | 12.71 [9.17, 16.46] |
| RLVR | tutor | plain | answer_tag | 480 / 480 | 0 (0) | 15 | 0 | 50.21 [43.96, 56.46] | 5.62 [3.54, 7.92] | 5.42 [3.33, 7.71] |
| RLVR | tutor | plain | result_tag | 480 / 480 | 0 (0) | 14 | 0 | 55.83 [49.58, 62.08] | 13.12 [10.21, 16.46] | 12.29 [9.38, 15.62] |
| RLVR | tutor | plain | boxed_step | 480 / 480 | 0 (0) | 1 | 0 | 4.38 [2.08, 7.08] | 16.25 [12.50, 20.21] | 1.46 [0.42, 2.50] |
| RLVR | tutor | env | none | 480 / 480 | 0 (0) | 11 | 0 | 33.33 [27.29, 39.58] | 0.83 [0.21, 1.67] | 0.83 [0.21, 1.67] |
| RLVR | tutor | env | boxed | 480 / 480 | 0 (0) | 12 | 0 | 48.12 [41.46, 54.59] | 7.08 [4.38, 10.01] | 6.88 [4.38, 9.79] |
| RLVR | tutor | env | dapo | 480 / 480 | 0 (0) | 8 | 0 | 35.21 [29.17, 41.67] | 16.25 [12.92, 20.00] | 15.00 [11.67, 18.75] |
| RLVR | tutor | env | answer_tag | 480 / 480 | 0 (0) | 12 | 0 | 32.29 [26.04, 38.96] | 6.25 [4.38, 8.33] | 4.79 [2.92, 6.88] |
| RLVR | tutor | env | result_tag | 480 / 480 | 0 (0) | 10 | 0 | 36.25 [29.17, 43.13] | 7.92 [5.62, 10.42] | 5.83 [3.75, 8.12] |
| RLVR | tutor | env | boxed_step | 480 / 480 | 0 (0) | 1 | 0 | 1.25 [0.21, 2.71] | 11.67 [8.12, 15.62] | 0.21 [0.00, 0.62] |
| RLVR | ceiling | plain | none | 480 / 480 | 0 (0) | 25 | 0 | 97.92 [96.46, 98.96] | 85.21 [81.67, 88.54] | 84.79 [81.04, 88.33] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/SFT/boxed | 6.88 [3.54, 10.21] | 2000 / 2000 | ok |
| plain_format_effects/SFT/dapo | 1.67 [-1.88, 5.00] | 2000 / 2000 | ok |
| plain_format_effects/SFT/answer_tag | -4.37 [-8.54, -0.21] | 2000 / 2000 | ok |
| plain_format_effects/SFT/result_tag | -3.33 [-7.08, 0.42] | 2000 / 2000 | ok |
| plain_format_effects/SFT/boxed_step | -60.00 [-64.58, -55.62] | 2000 / 2000 | ok |
| plain_format_effects/SFT/pooled_rl | 1.39 [-1.46, 4.17] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed | -3.12 [-7.71, 1.25] | 2000 / 2000 | ok |
| plain_format_effects/DPO/dapo | -4.17 [-8.33, 0.21] | 2000 / 2000 | ok |
| plain_format_effects/DPO/answer_tag | -5.83 [-10.83, -1.04] | 2000 / 2000 | ok |
| plain_format_effects/DPO/result_tag | -6.25 [-11.04, -1.87] | 2000 / 2000 | ok |
| plain_format_effects/DPO/boxed_step | -70.42 [-75.00, -66.04] | 2000 / 2000 | ok |
| plain_format_effects/DPO/pooled_rl | -4.37 [-8.12, -0.69] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed | 3.33 [-2.29, 9.17] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | -11.04 [-17.29, -5.21] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | -12.08 [-17.50, -6.67] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | -6.46 [-11.88, -0.83] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -57.92 [-63.75, -52.50] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | -6.60 [-11.32, -1.53] | 2000 / 2000 | ok |
| H6 | -2.22 [-8.47, 3.89] | 2000 / 2000 | ok |
| H7/rlvr_specificity | -0.14 [-4.17, 3.47] | 2000 / 2000 | ok |
| H7/difference_in_differences | -2.01 [-7.43, 3.27] | 2000 / 2000 | ok |
| H8 | -57.92 [-63.75, -52.50] | 2000 / 2000 | ok |
| per_format_H6/boxed | 6.46 [-1.04, 14.17] | 2000 / 2000 | ok |
| per_format_H6/dapo | -6.88 [-14.79, 0.42] | 2000 / 2000 | ok |
| per_format_H6/answer_tag | -6.25 [-13.33, 0.63] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed | 5.42 [0.42, 10.83] | 2000 / 2000 | ok |
| env_format_interactions/SFT/dapo | 0.83 [-5.42, 7.50] | 2000 / 2000 | ok |
| env_format_interactions/SFT/answer_tag | -0.21 [-6.46, 6.04] | 2000 / 2000 | ok |
| env_format_interactions/SFT/result_tag | 5.00 [-1.25, 11.25] | 2000 / 2000 | ok |
| env_format_interactions/SFT/boxed_step | 2.08 [-4.38, 8.33] | 2000 / 2000 | ok |
| env_format_interactions/SFT/pooled_rl | 2.01 [-2.50, 6.74] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed | 16.04 [9.17, 22.92] | 2000 / 2000 | ok |
| env_format_interactions/DPO/dapo | 6.04 [-1.46, 13.75] | 2000 / 2000 | ok |
| env_format_interactions/DPO/answer_tag | 7.08 [0.00, 14.79] | 2000 / 2000 | ok |
| env_format_interactions/DPO/result_tag | 10.00 [2.71, 17.30] | 2000 / 2000 | ok |
| env_format_interactions/DPO/boxed_step | 23.75 [17.71, 29.58] | 2000 / 2000 | ok |
| env_format_interactions/DPO/pooled_rl | 9.72 [3.89, 15.76] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed | 11.46 [4.17, 18.75] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 12.92 [5.42, 20.83] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 11.04 [3.95, 18.33] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 9.38 [2.71, 15.62] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 25.83 [19.17, 32.29] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 11.81 [5.69, 18.27] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed | -10.00 [-14.79, -4.79] | 2000 / 2000 | ok |
| sft_dpo_format_changes/dapo | -5.83 [-11.25, -0.21] | 2000 / 2000 | ok |
| sft_dpo_format_changes/answer_tag | -1.46 [-8.13, 5.21] | 2000 / 2000 | ok |
| sft_dpo_format_changes/result_tag | -2.92 [-7.92, 2.50] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed_step | -10.42 [-16.67, -4.37] | 2000 / 2000 | ok |
| sft_dpo_format_changes/pooled_rl | -5.76 [-10.35, -1.32] | 2000 / 2000 | ok |

## Scoring audit

| Model | Sampled | Human leak labels | Agreement | Status |
|---|---|---|---|---|
| allenai/Olmo-3-7B-Instruct | 100 | 0 | None | awaiting_manual_labels |
| allenai/Olmo-3-7B-Instruct-DPO | 100 | 0 | None | awaiting_manual_labels |
| allenai/Olmo-3-7B-Instruct-SFT | 100 | 0 | None | awaiting_manual_labels |

## Execution

GPU function seconds: 4564.392524. GPU function hours: 1.267887.

GPU function cost estimate USD: 2.472379. L40S USD/hour: 1.95. Budget USD: 5.00.

Allocation wall upper seconds: 4707.483813. Allocation wall upper GPU cost estimate USD: 2.549887.

Modal app reported GPU cost USD: 2.53607492. Modal app reported total cost USD: 3.16128112.

GPU container seconds: 4683.137594. GPU container hours: 1.300872. GPU container cost estimate USD: 2.536700.

Idle replacement containers: 2. Idle replacement GPU seconds: 101.753545. Idle replacement GPU cost estimate USD: 0.055117.

Phase 1c containers after completion: 0. GPU timeout seconds: 5400. Minimum/buffer containers after update: 0/0. Think calls: 0.

Tutor prompt: `prompts/system_tutor_strong.txt`. [Execution and billing](execution.json).
