> Audit source: Codex, self-labeled blind to arm at the user's explicit request; no independent human labels.

# Phase 5

Grid: `full`; 200 problems; 3200 responses.

Bootstrap: 2,000 paired problem resamples; seed 0; percentile 95% intervals.

Denominator: all requests, including cap-hit replies with no final. Length: characters.

Inference eligible: True; reasons: none.

Parser audit: complete; 40/40 labeled.

Splitter audit: complete; 40/40 Codex-labeled finals; 0 pending; 0 disagreements.

Dropped cells: []

nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4: 40/40 labels; attempt agreement 0.8; numeric agreement 0.909091.
Splitter: complete; 40/40 Codex-labeled finals; 0 pending; agreement 1; 0 disagreements.

Protocol sensitivity: 2 off-mode replies counted as violations; deviation `8219501182ce380c45115efd256938a2e372f088`. Baseline leak/attempt and all other outcomes are unchanged.
Affected replies: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/phase5:p0101:t0:NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4:main:thinking-0:explicit:none`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/phase5:p0303:t1:NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4:main:thinking-0:explicit:none`

## All gold

Problems: 200; responses: 3200.

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 400 | violation | 172 | 0.43 | [0.3775, 0.48] |
| main | off | explicit | none | 400 | leak | 169 | 0.4225 | [0.3725, 0.4725] |
| main | off | explicit | none | 400 | attempt | 63 | 0.1575 | [0.125, 0.195] |
| main | off | explicit | none | 400 | correct_attempt | 57 | 0.1425 | [0.11, 0.1775] |
| main | off | explicit | none | 400 | wrong_attempt | 6 | 0.015 | [0.005, 0.0275] |
| main | off | explicit | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | empty_or_truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | final_reply_length |  | 941.645 | [867.99, 1039.68] |
| main | off | explicit | none | 400 | cap_hit | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
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
| prefill | on | explicit | irrelevant | 400 | violation | 58 | 0.145 | [0.105, 0.185] |
| prefill | on | explicit | irrelevant | 400 | leak | 55 | 0.1375 | [0.1, 0.18] |
| prefill | on | explicit | irrelevant | 400 | attempt | 5 | 0.0125 | [0.0025, 0.025] |
| prefill | on | explicit | irrelevant | 400 | correct_attempt | 2 | 0.005 | [0, 0.0125] |
| prefill | on | explicit | irrelevant | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | irrelevant | 400 | empty_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | truncated_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | empty_or_truncated_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | final_reply_length |  | 1043.6 | [999, 1085.57] |
| prefill | on | explicit | irrelevant | 400 | cap_hit | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | no_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | cap_hit_no_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | irrelevant | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 400 | violation | 30 | 0.075 | [0.05, 0.105] |
| prefill | on | explicit | neutral | 400 | leak | 27 | 0.0675 | [0.0425, 0.0975] |
| prefill | on | explicit | neutral | 400 | attempt | 4 | 0.01 | [0.0025, 0.02] |
| prefill | on | explicit | neutral | 400 | correct_attempt | 1 | 0.0025 | [0, 0.0075] |
| prefill | on | explicit | neutral | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | neutral | 400 | empty_final | 34 | 0.085 | [0.0575, 0.1175] |
| prefill | on | explicit | neutral | 400 | truncated_final | 39 | 0.0975 | [0.0675, 0.1325] |
| prefill | on | explicit | neutral | 400 | empty_or_truncated_final | 39 | 0.0975 | [0.0675, 0.1325] |
| prefill | on | explicit | neutral | 400 | final_reply_length |  | 824.572 | [778.451, 868.797] |
| prefill | on | explicit | neutral | 400 | cap_hit | 39 | 0.0975 | [0.0675, 0.1325] |
| prefill | on | explicit | neutral | 400 | no_final | 34 | 0.085 | [0.0575, 0.1175] |
| prefill | on | explicit | neutral | 400 | cap_hit_no_final | 34 | 0.085 | [0.0575, 0.1175] |
| prefill | on | explicit | neutral | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
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
| violation | H21 | -0.2675 | [-0.3325, -0.205] | True | False |
| violation | on_plain_minus_off_plain | -0.0775 | [-0.1125, -0.0425] | False | None |
| violation | H20_empty_minus_reminder | 0.035 | [0.005, 0.065] | True | True |
| violation | H20_neutral_minus_reminder | 0.0075 | [-0.0225, 0.0375] | True | False |
| violation | H20_irrelevant_minus_reminder | 0.0775 | [0.045, 0.11] | True | True |
| violation | H20_joint | NA | [NA, NA] | True | False |
| leak | H21 | -0.2625 | [-0.325, -0.2025] | False | None |
| leak | on_plain_minus_off_plain | -0.0775 | [-0.1125, -0.045] | False | None |
| leak | H20_empty_minus_reminder | 0.0325 | [0.0025, 0.0625] | False | None |
| leak | H20_neutral_minus_reminder | 0 | [-0.0275, 0.0275] | False | None |
| leak | H20_irrelevant_minus_reminder | 0.07 | [0.0399375, 0.1] | False | None |
| leak | H20_joint | NA | [NA, NA] | False | None |

## Gold >= 10

Problems: 169; responses: 2704.

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 338 | violation | 136 | 0.402367 | [0.349038, 0.455621] |
| main | off | explicit | none | 338 | leak | 134 | 0.39645 | [0.343195, 0.449704] |
| main | off | explicit | none | 338 | attempt | 49 | 0.14497 | [0.109467, 0.183432] |
| main | off | explicit | none | 338 | correct_attempt | 45 | 0.133136 | [0.100592, 0.168639] |
| main | off | explicit | none | 338 | wrong_attempt | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | off | explicit | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | empty_or_truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | final_reply_length |  | 955.962 | [867.668, 1067.05] |
| main | off | explicit | none | 338 | cap_hit | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
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
| prefill | on | explicit | irrelevant | 338 | violation | 29 | 0.0857988 | [0.0591716, 0.115459] |
| prefill | on | explicit | irrelevant | 338 | leak | 26 | 0.0769231 | [0.0502959, 0.106509] |
| prefill | on | explicit | irrelevant | 338 | attempt | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | on | explicit | irrelevant | 338 | correct_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | on | explicit | irrelevant | 338 | wrong_attempt | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | irrelevant | 338 | empty_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | truncated_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | empty_or_truncated_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | final_reply_length |  | 1038.81 | [992.295, 1085.89] |
| prefill | on | explicit | irrelevant | 338 | cap_hit | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | no_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | cap_hit_no_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | irrelevant | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 338 | violation | 16 | 0.0473373 | [0.0266272, 0.0710059] |
| prefill | on | explicit | neutral | 338 | leak | 14 | 0.0414201 | [0.0236686, 0.0650888] |
| prefill | on | explicit | neutral | 338 | attempt | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | neutral | 338 | correct_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | neutral | 338 | wrong_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | on | explicit | neutral | 338 | empty_final | 26 | 0.0769231 | [0.0473373, 0.109467] |
| prefill | on | explicit | neutral | 338 | truncated_final | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | on | explicit | neutral | 338 | empty_or_truncated_final | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | on | explicit | neutral | 338 | final_reply_length |  | 837.899 | [790.9, 883.333] |
| prefill | on | explicit | neutral | 338 | cap_hit | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | on | explicit | neutral | 338 | no_final | 26 | 0.0769231 | [0.0473373, 0.109467] |
| prefill | on | explicit | neutral | 338 | cap_hit_no_final | 26 | 0.0769231 | [0.0473373, 0.109467] |
| prefill | on | explicit | neutral | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
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
| violation | H21 | -0.289941 | [-0.349186, -0.230769] | True | False |
| violation | on_plain_minus_off_plain | -0.0769231 | [-0.109541, -0.0473373] | False | None |
| violation | H20_empty_minus_reminder | 0.0147929 | [-0.0118343, 0.0414201] | True | False |
| violation | H20_neutral_minus_reminder | 0.0177515 | [-0.00887574, 0.0473373] | True | False |
| violation | H20_irrelevant_minus_reminder | 0.056213 | [0.0236686, 0.091716] | True | True |
| violation | H20_joint | NA | [NA, NA] | True | False |
| leak | H21 | -0.286982 | [-0.343195, -0.227811] | False | None |
| leak | on_plain_minus_off_plain | -0.0769231 | [-0.109467, -0.0473373] | False | None |
| leak | H20_empty_minus_reminder | 0.0118343 | [-0.0147929, 0.0384615] | False | None |
| leak | H20_neutral_minus_reminder | 0.0118343 | [-0.0147929, 0.0414201] | False | None |
| leak | H20_irrelevant_minus_reminder | 0.0473373 | [0.0177515, 0.0798817] | False | None |
| leak | H20_joint | NA | [NA, NA] | False | None |

## Protocol sensitivity: all gold

Problems: 200; responses: 3200.

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 400 | violation | 174 | 0.435 | [0.3825, 0.485] |
| main | off | explicit | none | 400 | leak | 169 | 0.4225 | [0.3725, 0.4725] |
| main | off | explicit | none | 400 | attempt | 63 | 0.1575 | [0.125, 0.195] |
| main | off | explicit | none | 400 | correct_attempt | 57 | 0.1425 | [0.11, 0.1775] |
| main | off | explicit | none | 400 | wrong_attempt | 6 | 0.015 | [0.005, 0.0275] |
| main | off | explicit | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | empty_or_truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | final_reply_length |  | 941.645 | [867.99, 1039.68] |
| main | off | explicit | none | 400 | cap_hit | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
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
| prefill | on | explicit | irrelevant | 400 | violation | 58 | 0.145 | [0.105, 0.185] |
| prefill | on | explicit | irrelevant | 400 | leak | 55 | 0.1375 | [0.1, 0.18] |
| prefill | on | explicit | irrelevant | 400 | attempt | 5 | 0.0125 | [0.0025, 0.025] |
| prefill | on | explicit | irrelevant | 400 | correct_attempt | 2 | 0.005 | [0, 0.0125] |
| prefill | on | explicit | irrelevant | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | irrelevant | 400 | empty_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | truncated_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | empty_or_truncated_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | final_reply_length |  | 1043.6 | [999, 1085.57] |
| prefill | on | explicit | irrelevant | 400 | cap_hit | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | no_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | cap_hit_no_final | 5 | 0.0125 | [0.0025, 0.0275] |
| prefill | on | explicit | irrelevant | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | irrelevant | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 400 | violation | 30 | 0.075 | [0.05, 0.105] |
| prefill | on | explicit | neutral | 400 | leak | 27 | 0.0675 | [0.0425, 0.0975] |
| prefill | on | explicit | neutral | 400 | attempt | 4 | 0.01 | [0.0025, 0.02] |
| prefill | on | explicit | neutral | 400 | correct_attempt | 1 | 0.0025 | [0, 0.0075] |
| prefill | on | explicit | neutral | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.0175] |
| prefill | on | explicit | neutral | 400 | empty_final | 34 | 0.085 | [0.0575, 0.1175] |
| prefill | on | explicit | neutral | 400 | truncated_final | 39 | 0.0975 | [0.0675, 0.1325] |
| prefill | on | explicit | neutral | 400 | empty_or_truncated_final | 39 | 0.0975 | [0.0675, 0.1325] |
| prefill | on | explicit | neutral | 400 | final_reply_length |  | 824.572 | [778.451, 868.797] |
| prefill | on | explicit | neutral | 400 | cap_hit | 39 | 0.0975 | [0.0675, 0.1325] |
| prefill | on | explicit | neutral | 400 | no_final | 34 | 0.085 | [0.0575, 0.1175] |
| prefill | on | explicit | neutral | 400 | cap_hit_no_final | 34 | 0.085 | [0.0575, 0.1175] |
| prefill | on | explicit | neutral | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
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
| violation | H21 | -0.2725 | [-0.3375, -0.21] | True | False |
| violation | on_plain_minus_off_plain | -0.0775 | [-0.1125, -0.0425] | False | None |
| violation | H20_empty_minus_reminder | 0.035 | [0.005, 0.065] | True | True |
| violation | H20_neutral_minus_reminder | 0.0075 | [-0.0225, 0.0375] | True | False |
| violation | H20_irrelevant_minus_reminder | 0.0775 | [0.045, 0.11] | True | True |
| violation | H20_joint | NA | [NA, NA] | True | False |
| leak | H21 | -0.2625 | [-0.325, -0.2025] | False | None |
| leak | on_plain_minus_off_plain | -0.0775 | [-0.1125, -0.045] | False | None |
| leak | H20_empty_minus_reminder | 0.0325 | [0.0025, 0.0625] | False | None |
| leak | H20_neutral_minus_reminder | 0 | [-0.0275, 0.0275] | False | None |
| leak | H20_irrelevant_minus_reminder | 0.07 | [0.0399375, 0.1] | False | None |
| leak | H20_joint | NA | [NA, NA] | False | None |

## Protocol sensitivity: gold >= 10

Problems: 169; responses: 2704.

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 338 | violation | 137 | 0.405325 | [0.352071, 0.45858] |
| main | off | explicit | none | 338 | leak | 134 | 0.39645 | [0.343195, 0.449704] |
| main | off | explicit | none | 338 | attempt | 49 | 0.14497 | [0.109467, 0.183432] |
| main | off | explicit | none | 338 | correct_attempt | 45 | 0.133136 | [0.100592, 0.168639] |
| main | off | explicit | none | 338 | wrong_attempt | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | off | explicit | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | empty_or_truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | final_reply_length |  | 955.962 | [867.668, 1067.05] |
| main | off | explicit | none | 338 | cap_hit | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
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
| prefill | on | explicit | irrelevant | 338 | violation | 29 | 0.0857988 | [0.0591716, 0.115459] |
| prefill | on | explicit | irrelevant | 338 | leak | 26 | 0.0769231 | [0.0502959, 0.106509] |
| prefill | on | explicit | irrelevant | 338 | attempt | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | on | explicit | irrelevant | 338 | correct_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | on | explicit | irrelevant | 338 | wrong_attempt | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | irrelevant | 338 | empty_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | truncated_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | empty_or_truncated_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | final_reply_length |  | 1038.81 | [992.295, 1085.89] |
| prefill | on | explicit | irrelevant | 338 | cap_hit | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | no_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | cap_hit_no_final | 4 | 0.0118343 | [0, 0.0266272] |
| prefill | on | explicit | irrelevant | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | irrelevant | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 338 | violation | 16 | 0.0473373 | [0.0266272, 0.0710059] |
| prefill | on | explicit | neutral | 338 | leak | 14 | 0.0414201 | [0.0236686, 0.0650888] |
| prefill | on | explicit | neutral | 338 | attempt | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | on | explicit | neutral | 338 | correct_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | neutral | 338 | wrong_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | on | explicit | neutral | 338 | empty_final | 26 | 0.0769231 | [0.0473373, 0.109467] |
| prefill | on | explicit | neutral | 338 | truncated_final | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | on | explicit | neutral | 338 | empty_or_truncated_final | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | on | explicit | neutral | 338 | final_reply_length |  | 837.899 | [790.9, 883.333] |
| prefill | on | explicit | neutral | 338 | cap_hit | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | on | explicit | neutral | 338 | no_final | 26 | 0.0769231 | [0.0473373, 0.109467] |
| prefill | on | explicit | neutral | 338 | cap_hit_no_final | 26 | 0.0769231 | [0.0473373, 0.109467] |
| prefill | on | explicit | neutral | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | neutral | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
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
| violation | H21 | -0.292899 | [-0.352071, -0.230769] | True | False |
| violation | on_plain_minus_off_plain | -0.0769231 | [-0.109541, -0.0473373] | False | None |
| violation | H20_empty_minus_reminder | 0.0147929 | [-0.0118343, 0.0414201] | True | False |
| violation | H20_neutral_minus_reminder | 0.0177515 | [-0.00887574, 0.0473373] | True | False |
| violation | H20_irrelevant_minus_reminder | 0.056213 | [0.0236686, 0.091716] | True | True |
| violation | H20_joint | NA | [NA, NA] | True | False |
| leak | H21 | -0.286982 | [-0.343195, -0.227811] | False | None |
| leak | on_plain_minus_off_plain | -0.0769231 | [-0.109467, -0.0473373] | False | None |
| leak | H20_empty_minus_reminder | 0.0118343 | [-0.0147929, 0.0384615] | False | None |
| leak | H20_neutral_minus_reminder | 0.0118343 | [-0.0147929, 0.0414201] | False | None |
| leak | H20_irrelevant_minus_reminder | 0.0473373 | [0.0177515, 0.0798817] | False | None |
| leak | H20_joint | NA | [NA, NA] | False | None |
