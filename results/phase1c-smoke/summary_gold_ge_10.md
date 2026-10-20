# Phase 1c results

Gold minimum: 10; 208 responses.

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.

Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.

## instruct

Checkpoints: partial; available: RLVR; missing: SFT, DPO.
Inference eligible: False; source problems: 3 / 200.

### All selected problems

Problems: 2; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| RLVR | tutor | plain | none | 8 / 8 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | plain | boxed | 8 / 8 | 0 (0) | 1 | 0 | 100.00 [100.00, 100.00] | 50.00 [50.00, 50.00] | 50.00 [50.00, 50.00] |
| RLVR | tutor | plain | dapo | 8 / 8 | 0 (0) | 1 | 0 | 100.00 [100.00, 100.00] | 75.00 [75.00, 75.00] | 75.00 [75.00, 75.00] |
| RLVR | tutor | plain | answer_tag | 8 / 8 | 0 (0) | 1 | 0 | 100.00 [100.00, 100.00] | 37.50 [25.00, 50.00] | 37.50 [25.00, 50.00] |
| RLVR | tutor | plain | result_tag | 8 / 8 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 75.00 [50.00, 100.00] | 75.00 [50.00, 100.00] |
| RLVR | tutor | plain | boxed_step | 8 / 8 | 0 (0) | 0 | 0 | 0.00 [0.00, 0.00] | 12.50 [0.00, 25.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | none | 8 / 8 | 0 (0) | 2 | 0 | 87.50 [75.00, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | boxed | 8 / 8 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 37.50 [25.00, 50.00] | 37.50 [25.00, 50.00] |
| RLVR | tutor | env | dapo | 8 / 8 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 37.50 [0.00, 75.00] | 37.50 [0.00, 75.00] |
| RLVR | tutor | env | answer_tag | 8 / 8 | 0 (0) | 0 | 0 | 87.50 [75.00, 100.00] | 25.00 [0.00, 50.00] | 25.00 [0.00, 50.00] |
| RLVR | tutor | env | result_tag | 8 / 8 | 0 (0) | 0 | 0 | 100.00 [100.00, 100.00] | 25.00 [25.00, 25.00] | 25.00 [25.00, 25.00] |
| RLVR | tutor | env | boxed_step | 8 / 8 | 0 (0) | 0 | 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | ceiling | plain | none | 8 / 8 | 0 (0) | 1 | 0 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/SFT/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/RLVR/boxed | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -100.00 [-100.00, -100.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| H6 | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| H7/rlvr_specificity | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| H7/difference_in_differences | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| H8 | -100.00 [-100.00, -100.00] | 2000 / 2000 | ok |
| per_format_H6/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| per_format_H6/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| per_format_H6/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/RLVR/boxed | 12.50 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 12.50 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 12.50 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 12.50 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 8.33 [0.00, 16.67] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |

### All-three-checkpoint common ceiling solved problems

Problems: None; status: unavailable.

## think

Checkpoints: partial; available: RLVR; missing: SFT, DPO.
Inference eligible: False; source problems: 3 / 100.

### All selected problems

Problems: 2; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| RLVR | tutor | plain | none | 4 / 8 | 4 (4) | 6 | 4 | 75.00 [50.00, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | plain | boxed | 7 / 8 | 1 (1) | 3 | 1 | 71.43 [66.67, 75.00] | 71.43 [66.67, 75.00] | 71.43 [66.67, 75.00] |
| RLVR | tutor | plain | dapo | 6 / 8 | 2 (2) | 5 | 2 | 50.00 [0.00, 75.00] | 33.33 [0.00, 50.00] | 33.33 [0.00, 50.00] |
| RLVR | tutor | plain | answer_tag | 8 / 8 | 0 (0) | 1 | 0 | 75.00 [50.00, 100.00] | 75.00 [50.00, 100.00] | 75.00 [50.00, 100.00] |
| RLVR | tutor | plain | result_tag | 7 / 8 | 1 (1) | 2 | 1 | 85.71 [66.67, 100.00] | 85.71 [66.67, 100.00] | 85.71 [66.67, 100.00] |
| RLVR | tutor | plain | boxed_step | 4 / 8 | 4 (4) | 4 | 4 | 0.00 [0.00, 0.00] | 25.00 [0.00, 100.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | none | 6 / 8 | 2 (2) | 3 | 2 | 83.33 [66.67, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | boxed | 5 / 8 | 3 (3) | 3 | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| RLVR | tutor | env | dapo | 5 / 8 | 3 (3) | 3 | 3 | 100.00 [100.00, 100.00] | 60.00 [33.33, 100.00] | 60.00 [33.33, 100.00] |
| RLVR | tutor | env | answer_tag | 7 / 8 | 1 (1) | 2 | 1 | 85.71 [75.00, 100.00] | 71.43 [50.00, 100.00] | 71.43 [50.00, 100.00] |
| RLVR | tutor | env | result_tag | 7 / 8 | 1 (1) | 2 | 1 | 85.71 [75.00, 100.00] | 85.71 [75.00, 100.00] | 85.71 [75.00, 100.00] |
| RLVR | tutor | env | boxed_step | 3 / 8 | 5 (5) | 5 | 5 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | ceiling | plain | none | 6 / 8 | 2 (2) | 2 | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |

| Contrast | Estimate [95% CI], pp | Defined / total draws | Status |
|---|---|---|---|
| plain_format_effects/SFT/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/SFT/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/DPO/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| plain_format_effects/RLVR/boxed | -3.57 [-33.33, 25.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/dapo | -25.00 [-50.00, -25.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 10.71 [0.00, 16.67] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -75.00 [-100.00, -50.00] | 2000 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | -9.52 [-19.44, -8.33] | 2000 / 2000 | ok |
| H6 | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| H7/rlvr_specificity | -20.24 [-25.00, -19.44] | 2000 / 2000 | ok |
| H7/difference_in_differences | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| H8 | -75.00 [-100.00, -50.00] | 2000 / 2000 | ok |
| per_format_H6/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| per_format_H6/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| per_format_H6/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/SFT/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/DPO/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| env_format_interactions/RLVR/boxed | 20.24 [8.33, 33.33] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 41.67 [25.00, 83.33] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 2.38 [-25.00, 33.33] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | -8.33 [-8.33, 0.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | -8.33 [-8.33, 0.00] | 1502 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 21.43 [11.11, 41.67] | 2000 / 2000 | ok |
| sft_dpo_format_changes/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |

### All-three-checkpoint common ceiling solved problems

Problems: None; status: unavailable.
