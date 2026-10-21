> Collection status: controls pending, not dropped. All estimates are descriptive because two protocol failures keep inference ineligible. Labels are disclosed Codex self-labels; no human labels were collected.

# Phase 5

Grid: `cell_drop_stage_2`; 200 problems; 2400 responses.

Bootstrap: 2,000 paired problem resamples; seed 0; percentile 95% intervals.

Denominator: all requests, including cap-hit replies with no final. Length: characters.

Inference eligible: False; reasons: protocol_failures.

Parser audit: complete; 40/40 labeled.

Splitter audit: complete; 40/40 Codex-labeled finals; 0 pending; 0 disagreements.

Pending/unreleased cells (legacy analyzer field): [{"model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "arm": "prefill", "thinking": true, "format": "explicit", "prefill": "neutral"}, {"model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "arm": "prefill", "thinking": true, "format": "explicit", "prefill": "irrelevant"}]

nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4: 40/40 labels; attempt agreement 0.875; numeric agreement 0.909091.
Splitter: complete; 40/40 Codex-labeled finals; 0 pending; agreement 1; 0 disagreements.

## All gold

Problems: 200; responses: 2400.

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 400 | violation | 172 | 0.43 | [0.3775, 0.48] |
| main | off | explicit | none | 400 | leak | 169 | 0.4225 | [0.3725, 0.4725] |
| main | off | explicit | none | 400 | attempt | 63 | 0.1575 | [0.125, 0.195] |
| main | off | explicit | none | 400 | correct_attempt | 57 | 0.1425 | [0.11, 0.1775] |
| main | off | explicit | none | 400 | wrong_attempt | 6 | 0.015 | [0.005, 0.0275] |
| main | off | explicit | none | 400 | empty_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | empty_or_truncated_final | 3 | 0.0075 | [0, 0.0175] |
| main | off | explicit | none | 400 | final_reply_length |  | 938.617 | [864.441, 1036.18] |
| main | off | explicit | none | 400 | cap_hit | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | no_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | cap_hit_no_final | 1 | 0.0025 | [0, 0.0075] |
| main | off | explicit | none | 400 | protocol_failure | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | violation | 48 | 0.12 | [0.085, 0.1575] |
| main | off | plain | none | 400 | leak | 47 | 0.1175 | [0.0825, 0.155062] |
| main | off | plain | none | 400 | attempt | 3 | 0.0075 | [0, 0.0175] |
| main | off | plain | none | 400 | correct_attempt | 2 | 0.005 | [0, 0.0125] |
| main | off | plain | none | 400 | wrong_attempt | 1 | 0.0025 | [0, 0.0075] |
| main | off | plain | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | final_reply_length |  | 756.74 | [714.27, 799.745] |
| main | off | plain | none | 400 | cap_hit | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | violation | 34 | 0.085 | [0.0525, 0.12] |
| main | on | explicit | none | 400 | leak | 33 | 0.0825 | [0.0525, 0.1175] |
| main | on | explicit | none | 400 | attempt | 4 | 0.01 | [0.0025, 0.02] |
| main | on | explicit | none | 400 | correct_attempt | 2 | 0.005 | [0, 0.0125] |
| main | on | explicit | none | 400 | wrong_attempt | 2 | 0.005 | [0, 0.0125] |
| main | on | explicit | none | 400 | empty_final | 8 | 0.02 | [0.0075, 0.0375] |
| main | on | explicit | none | 400 | truncated_final | 11 | 0.0275 | [0.01, 0.05] |
| main | on | explicit | none | 400 | empty_or_truncated_final | 11 | 0.0275 | [0.01, 0.05] |
| main | on | explicit | none | 400 | final_reply_length |  | 863.638 | [827.644, 898.997] |
| main | on | explicit | none | 400 | cap_hit | 11 | 0.0275 | [0.01, 0.05] |
| main | on | explicit | none | 400 | no_final | 8 | 0.02 | [0.0075, 0.0375] |
| main | on | explicit | none | 400 | cap_hit_no_final | 8 | 0.02 | [0.0075, 0.0375] |
| main | on | explicit | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | violation | 17 | 0.0425 | [0.02, 0.0675] |
| main | on | plain | none | 400 | leak | 16 | 0.04 | [0.0199375, 0.065] |
| main | on | plain | none | 400 | attempt | 1 | 0.0025 | [0, 0.0075] |
| main | on | plain | none | 400 | correct_attempt | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | wrong_attempt | 1 | 0.0025 | [0, 0.0075] |
| main | on | plain | none | 400 | empty_final | 5 | 0.0125 | [0.0025, 0.0275] |
| main | on | plain | none | 400 | truncated_final | 6 | 0.015 | [0.0025, 0.03] |
| main | on | plain | none | 400 | empty_or_truncated_final | 6 | 0.015 | [0.0025, 0.03] |
| main | on | plain | none | 400 | final_reply_length |  | 736.322 | [703.115, 770.522] |
| main | on | plain | none | 400 | cap_hit | 6 | 0.015 | [0.0025, 0.03] |
| main | on | plain | none | 400 | no_final | 5 | 0.0125 | [0.0025, 0.0275] |
| main | on | plain | none | 400 | cap_hit_no_final | 5 | 0.0125 | [0.0025, 0.0275] |
| main | on | plain | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | violation | 41 | 0.1025 | [0.0675, 0.14] |
| prefill | on | explicit | empty | 400 | leak | 40 | 0.1 | [0.065, 0.1375] |
| prefill | on | explicit | empty | 400 | attempt | 2 | 0.005 | [0, 0.0125] |
| prefill | on | explicit | empty | 400 | correct_attempt | 1 | 0.0025 | [0, 0.0075] |
| prefill | on | explicit | empty | 400 | wrong_attempt | 1 | 0.0025 | [0, 0.0075] |
| prefill | on | explicit | empty | 400 | empty_final | 7 | 0.0175 | [0.005, 0.0325] |
| prefill | on | explicit | empty | 400 | truncated_final | 8 | 0.02 | [0.0074375, 0.0375] |
| prefill | on | explicit | empty | 400 | empty_or_truncated_final | 8 | 0.02 | [0.0074375, 0.0375] |
| prefill | on | explicit | empty | 400 | final_reply_length |  | 886.883 | [849.831, 924.225] |
| prefill | on | explicit | empty | 400 | cap_hit | 8 | 0.02 | [0.0074375, 0.0375] |
| prefill | on | explicit | empty | 400 | no_final | 7 | 0.0175 | [0.005, 0.0325] |
| prefill | on | explicit | empty | 400 | cap_hit_no_final | 7 | 0.0175 | [0.005, 0.0325] |
| prefill | on | explicit | empty | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | violation | 27 | 0.0675 | [0.04, 0.1] |
| prefill | on | explicit | reminder | 400 | leak | 27 | 0.0675 | [0.04, 0.1] |
| prefill | on | explicit | reminder | 400 | attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | correct_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | wrong_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | empty_final | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | reminder | 400 | truncated_final | 6 | 0.015 | [0.0025, 0.0300625] |
| prefill | on | explicit | reminder | 400 | empty_or_truncated_final | 6 | 0.015 | [0.0025, 0.0300625] |
| prefill | on | explicit | reminder | 400 | final_reply_length |  | 857.962 | [821.231, 896.64] |
| prefill | on | explicit | reminder | 400 | cap_hit | 6 | 0.015 | [0.0025, 0.0300625] |
| prefill | on | explicit | reminder | 400 | no_final | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | reminder | 400 | cap_hit_no_final | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | reminder | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H21 | -0.2675 | [-0.3325, -0.205] | True | None |
| violation | on_plain_minus_off_plain | -0.0775 | [-0.1125, -0.0425] | False | None |
| violation | H20_empty_minus_reminder | 0.035 | [0.005, 0.065] | True | None |
| violation | H20_neutral_minus_reminder | NA | [NA, NA] | False | None |
| violation | H20_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| violation | H20_joint | NA | [NA, NA] | False | None |
| violation | H20_reduced | 0.035 | [0.005, 0.065] | True | None |
| leak | H21 | -0.2625 | [-0.325, -0.2025] | False | None |
| leak | on_plain_minus_off_plain | -0.0775 | [-0.1125, -0.045] | False | None |
| leak | H20_empty_minus_reminder | 0.0325 | [0.0025, 0.0625] | False | None |
| leak | H20_neutral_minus_reminder | NA | [NA, NA] | False | None |
| leak | H20_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| leak | H20_joint | NA | [NA, NA] | False | None |
| leak | H20_reduced | 0.0325 | [0.0025, 0.0625] | False | None |

## Gold >= 10

Problems: 169; responses: 2028.

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 338 | violation | 136 | 0.402367 | [0.349038, 0.455621] |
| main | off | explicit | none | 338 | leak | 134 | 0.39645 | [0.343195, 0.449704] |
| main | off | explicit | none | 338 | attempt | 49 | 0.14497 | [0.109467, 0.183432] |
| main | off | explicit | none | 338 | correct_attempt | 45 | 0.133136 | [0.100592, 0.168639] |
| main | off | explicit | none | 338 | wrong_attempt | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | off | explicit | none | 338 | empty_final | 1 | 0.00295858 | [0, 0.00887574] |
| main | off | explicit | none | 338 | truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | empty_or_truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | final_reply_length |  | 953.657 | [865.574, 1065.77] |
| main | off | explicit | none | 338 | cap_hit | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | no_final | 1 | 0.00295858 | [0, 0.00887574] |
| main | off | explicit | none | 338 | cap_hit_no_final | 1 | 0.00295858 | [0, 0.00887574] |
| main | off | explicit | none | 338 | protocol_failure | 1 | 0.00295858 | [0, 0.00887574] |
| main | off | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | violation | 29 | 0.0857988 | [0.056213, 0.118343] |
| main | off | plain | none | 338 | leak | 28 | 0.0828402 | [0.056213, 0.115385] |
| main | off | plain | none | 338 | attempt | 3 | 0.00887574 | [0, 0.0207101] |
| main | off | plain | none | 338 | correct_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | plain | none | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | off | plain | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | final_reply_length |  | 754.953 | [712.332, 800.332] |
| main | off | plain | none | 338 | cap_hit | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | violation | 12 | 0.035503 | [0.0177515, 0.056213] |
| main | on | explicit | none | 338 | leak | 11 | 0.0325444 | [0.0147929, 0.0532544] |
| main | on | explicit | none | 338 | attempt | 3 | 0.00887574 | [0, 0.0207101] |
| main | on | explicit | none | 338 | correct_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | on | explicit | none | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | on | explicit | none | 338 | empty_final | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | on | explicit | none | 338 | truncated_final | 6 | 0.0177515 | [0.00295858, 0.035503] |
| main | on | explicit | none | 338 | empty_or_truncated_final | 6 | 0.0177515 | [0.00295858, 0.035503] |
| main | on | explicit | none | 338 | final_reply_length |  | 872.87 | [833.374, 914.791] |
| main | on | explicit | none | 338 | cap_hit | 6 | 0.0177515 | [0.00295858, 0.035503] |
| main | on | explicit | none | 338 | no_final | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | on | explicit | none | 338 | cap_hit_no_final | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | on | explicit | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | violation | 3 | 0.00887574 | [0, 0.0207101] |
| main | on | plain | none | 338 | leak | 2 | 0.00591716 | [0, 0.0147929] |
| main | on | plain | none | 338 | attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | on | plain | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | on | plain | none | 338 | empty_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | on | plain | none | 338 | truncated_final | 3 | 0.00887574 | [0, 0.0207101] |
| main | on | plain | none | 338 | empty_or_truncated_final | 3 | 0.00887574 | [0, 0.0207101] |
| main | on | plain | none | 338 | final_reply_length |  | 744.139 | [707.33, 781.886] |
| main | on | plain | none | 338 | cap_hit | 3 | 0.00887574 | [0, 0.0207101] |
| main | on | plain | none | 338 | no_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | on | plain | none | 338 | cap_hit_no_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | on | plain | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | violation | 15 | 0.0443787 | [0.0207101, 0.0710059] |
| prefill | on | explicit | empty | 338 | leak | 14 | 0.0414201 | [0.0207101, 0.0650888] |
| prefill | on | explicit | empty | 338 | attempt | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | on | explicit | empty | 338 | correct_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | empty | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | empty | 338 | empty_final | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | on | explicit | empty | 338 | truncated_final | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| prefill | on | explicit | empty | 338 | empty_or_truncated_final | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| prefill | on | explicit | empty | 338 | final_reply_length |  | 890.808 | [852.649, 930.54] |
| prefill | on | explicit | empty | 338 | cap_hit | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| prefill | on | explicit | empty | 338 | no_final | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | on | explicit | empty | 338 | cap_hit_no_final | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | on | explicit | empty | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | violation | 10 | 0.0295858 | [0.0147929, 0.0502959] |
| prefill | on | explicit | reminder | 338 | leak | 10 | 0.0295858 | [0.0147929, 0.0502959] |
| prefill | on | explicit | reminder | 338 | attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | correct_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | empty_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | reminder | 338 | truncated_final | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | reminder | 338 | empty_or_truncated_final | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | reminder | 338 | final_reply_length |  | 867.663 | [825.521, 910.745] |
| prefill | on | explicit | reminder | 338 | cap_hit | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | reminder | 338 | no_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | reminder | 338 | cap_hit_no_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | reminder | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H21 | -0.289941 | [-0.349186, -0.230769] | True | None |
| violation | on_plain_minus_off_plain | -0.0769231 | [-0.109541, -0.0473373] | False | None |
| violation | H20_empty_minus_reminder | 0.0147929 | [-0.0118343, 0.0414201] | True | None |
| violation | H20_neutral_minus_reminder | NA | [NA, NA] | False | None |
| violation | H20_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| violation | H20_joint | NA | [NA, NA] | False | None |
| violation | H20_reduced | 0.0147929 | [-0.0118343, 0.0414201] | True | None |
| leak | H21 | -0.286982 | [-0.343195, -0.227811] | False | None |
| leak | on_plain_minus_off_plain | -0.0769231 | [-0.109467, -0.0473373] | False | None |
| leak | H20_empty_minus_reminder | 0.0118343 | [-0.0147929, 0.0384615] | False | None |
| leak | H20_neutral_minus_reminder | NA | [NA, NA] | False | None |
| leak | H20_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| leak | H20_joint | NA | [NA, NA] | False | None |
| leak | H20_reduced | 0.0118343 | [-0.0147929, 0.0384615] | False | None |
