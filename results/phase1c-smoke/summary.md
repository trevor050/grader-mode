# Phase 1c results

Gold minimum: None; 312 responses.

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Rates are percentages; contrasts are percentage points. Intervals are unadjusted for multiplicity and conditional on defined bootstrap replicates.

Empty thinking-on final replies are excluded; nonempty truncated replies remain scored. RL-format pools are equal-weight means of boxed, dapo, answer_tag effects.

## instruct

Checkpoints: partial; available: RLVR; missing: SFT, DPO.
Inference eligible: False; source problems: 3 / 200.

### All selected problems

Problems: 3; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| RLVR | tutor | plain | none | 12 / 12 | 0 (0) | 4 | 0 | 100.00 [100.00, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | plain | boxed | 12 / 12 | 0 (0) | 5 | 0 | 100.00 [100.00, 100.00] | 33.33 [0.00, 50.00] | 33.33 [0.00, 50.00] |
| RLVR | tutor | plain | dapo | 12 / 12 | 0 (0) | 3 | 0 | 100.00 [100.00, 100.00] | 66.67 [50.00, 75.00] | 66.67 [50.00, 75.00] |
| RLVR | tutor | plain | answer_tag | 12 / 12 | 0 (0) | 3 | 0 | 100.00 [100.00, 100.00] | 41.67 [25.00, 50.00] | 41.67 [25.00, 50.00] |
| RLVR | tutor | plain | result_tag | 12 / 12 | 0 (0) | 3 | 0 | 100.00 [100.00, 100.00] | 58.33 [25.00, 100.00] | 58.33 [25.00, 100.00] |
| RLVR | tutor | plain | boxed_step | 12 / 12 | 0 (0) | 0 | 0 | 0.00 [0.00, 0.00] | 8.33 [0.00, 25.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | none | 12 / 12 | 0 (0) | 5 | 0 | 91.67 [75.00, 100.00] | 16.67 [0.00, 50.00] | 8.33 [0.00, 25.00] |
| RLVR | tutor | env | boxed | 12 / 12 | 0 (0) | 2 | 0 | 100.00 [100.00, 100.00] | 41.67 [25.00, 50.00] | 41.67 [25.00, 50.00] |
| RLVR | tutor | env | dapo | 12 / 12 | 0 (0) | 4 | 0 | 100.00 [100.00, 100.00] | 25.00 [0.00, 75.00] | 25.00 [0.00, 75.00] |
| RLVR | tutor | env | answer_tag | 12 / 12 | 0 (0) | 1 | 0 | 91.67 [75.00, 100.00] | 41.67 [0.00, 75.00] | 41.67 [0.00, 75.00] |
| RLVR | tutor | env | result_tag | 12 / 12 | 0 (0) | 1 | 0 | 100.00 [100.00, 100.00] | 33.33 [25.00, 50.00] | 33.33 [25.00, 50.00] |
| RLVR | tutor | env | boxed_step | 12 / 12 | 0 (0) | 0 | 0 | 0.00 [0.00, 0.00] | 16.67 [0.00, 50.00] | 0.00 [0.00, 0.00] |
| RLVR | ceiling | plain | none | 12 / 12 | 0 (0) | 4 | 0 | 100.00 [100.00, 100.00] | 75.00 [25.00, 100.00] | 75.00 [25.00, 100.00] |

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
| env_format_interactions/RLVR/boxed | 8.33 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 8.33 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | 0.00 [0.00, 0.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 8.33 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 8.33 [0.00, 25.00] | 2000 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 5.56 [0.00, 16.67] | 2000 / 2000 | ok |
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

Problems: 3; status: available.

| Checkpoint | Arm | Prose | Format | Valid / total | Excluded (truncated) | Truncated | Empty final | Leak [95% CI] | Commit [95% CI] | Correct commit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| RLVR | tutor | plain | none | 4 / 12 | 8 (8) | 10 | 8 | 75.00 [50.00, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | plain | boxed | 9 / 12 | 3 (3) | 5 | 3 | 77.78 [66.67, 100.00] | 77.78 [66.67, 100.00] | 77.78 [66.67, 100.00] |
| RLVR | tutor | plain | dapo | 10 / 12 | 2 (2) | 6 | 2 | 60.00 [0.00, 75.00] | 40.00 [0.00, 50.00] | 40.00 [0.00, 50.00] |
| RLVR | tutor | plain | answer_tag | 10 / 12 | 2 (2) | 4 | 2 | 80.00 [50.00, 100.00] | 70.00 [50.00, 100.00] | 70.00 [50.00, 100.00] |
| RLVR | tutor | plain | result_tag | 9 / 12 | 3 (3) | 5 | 3 | 77.78 [50.00, 100.00] | 77.78 [50.00, 100.00] | 77.78 [50.00, 100.00] |
| RLVR | tutor | plain | boxed_step | 6 / 12 | 6 (6) | 6 | 6 | 0.00 [0.00, 0.00] | 50.00 [0.00, 100.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | none | 8 / 12 | 4 (4) | 5 | 4 | 87.50 [66.67, 100.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| RLVR | tutor | env | boxed | 8 / 12 | 4 (4) | 5 | 4 | 87.50 [66.67, 100.00] | 87.50 [66.67, 100.00] | 87.50 [66.67, 100.00] |
| RLVR | tutor | env | dapo | 8 / 12 | 4 (4) | 4 | 4 | 100.00 [100.00, 100.00] | 37.50 [0.00, 100.00] | 37.50 [0.00, 100.00] |
| RLVR | tutor | env | answer_tag | 8 / 12 | 4 (4) | 6 | 4 | 75.00 [0.00, 100.00] | 62.50 [0.00, 100.00] | 62.50 [0.00, 100.00] |
| RLVR | tutor | env | result_tag | 11 / 12 | 1 (1) | 2 | 1 | 90.91 [75.00, 100.00] | 90.91 [75.00, 100.00] | 90.91 [75.00, 100.00] |
| RLVR | tutor | env | boxed_step | 4 / 12 | 8 (8) | 8 | 8 | 25.00 [0.00, 100.00] | 25.00 [0.00, 100.00] | 25.00 [0.00, 100.00] |
| RLVR | ceiling | plain | none | 7 / 12 | 5 (5) | 5 | 5 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |

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
| plain_format_effects/RLVR/boxed | 2.78 [-33.33, 37.50] | 1937 / 2000 | ok |
| plain_format_effects/RLVR/dapo | -15.00 [-50.00, 10.00] | 1937 / 2000 | ok |
| plain_format_effects/RLVR/answer_tag | 5.00 [0.00, 25.00] | 1937 / 2000 | ok |
| plain_format_effects/RLVR/result_tag | 2.78 [-25.00, 16.67] | 1937 / 2000 | ok |
| plain_format_effects/RLVR/boxed_step | -75.00 [-100.00, -50.00] | 1937 / 2000 | ok |
| plain_format_effects/RLVR/pooled_rl | -2.41 [-19.44, 24.17] | 1937 / 2000 | ok |
| H6 | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| H7/rlvr_specificity | -5.19 [-25.00, 41.67] | 2000 / 2000 | ok |
| H7/difference_in_differences | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| H8 | -75.00 [-100.00, -50.00] | 1937 / 2000 | ok |
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
| env_format_interactions/RLVR/boxed | -2.78 [-51.79, 33.33] | 1937 / 2000 | ok |
| env_format_interactions/RLVR/dapo | 27.50 [4.29, 83.33] | 1937 / 2000 | ok |
| env_format_interactions/RLVR/answer_tag | -17.50 [-50.71, 33.33] | 1937 / 2000 | ok |
| env_format_interactions/RLVR/result_tag | 0.63 [-9.29, 25.00] | 1937 / 2000 | ok |
| env_format_interactions/RLVR/boxed_step | 12.50 [-11.11, 75.00] | 1853 / 2000 | ok |
| env_format_interactions/RLVR/pooled_rl | 2.41 [-32.74, 41.67] | 1937 / 2000 | ok |
| sft_dpo_format_changes/boxed | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/dapo | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/answer_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/result_tag | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/boxed_step | undefined [undefined, undefined] | 0 / 2000 | unavailable |
| sft_dpo_format_changes/pooled_rl | undefined [undefined, undefined] | 0 / 2000 | unavailable |

### All-three-checkpoint common ceiling solved problems

Problems: None; status: unavailable.

## Scoring audit

| Model | Sampled | Human leak labels | Agreement | Status |
|---|---|---|---|---|
| allenai/Olmo-3-7B-Instruct | 100 | 0 | None | awaiting_manual_labels |
| allenai/Olmo-3-7B-Think | 100 | 0 | None | awaiting_manual_labels |
