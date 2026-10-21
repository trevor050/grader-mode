# Phase 4

Grid: `full`; 200 problems; 8000 responses.

Bootstrap: 2,000 paired problem resamples; seed 0; percentile 95% intervals.

Denominator: all requests, including empty finals. Length: characters.

Parser agreement: `pending`; 0/60 labeled.

## All gold

Problems: 200; responses: 8000.

### Qwen/Qwen3-14B

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | False | boxed | none | 400 | violation | 39 | 0.0975 | [0.0625, 0.135062] |
| main | False | boxed | none | 400 | leak | 37 | 0.0925 | [0.0575, 0.1325] |
| main | False | boxed | none | 400 | attempt | 6 | 0.015 | [0.0025, 0.03] |
| main | False | boxed | none | 400 | correct_attempt | 4 | 0.01 | [0.0025, 0.0225] |
| main | False | boxed | none | 400 | wrong_attempt | 2 | 0.005 | [0, 0.015] |
| main | False | boxed | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 400 | final_reply_length |  | 994.303 | [945.033, 1046.84] |
| main | False | explicit | none | 400 | violation | 38 | 0.095 | [0.06, 0.135] |
| main | False | explicit | none | 400 | leak | 36 | 0.09 | [0.0575, 0.1275] |
| main | False | explicit | none | 400 | attempt | 8 | 0.02 | [0.0075, 0.0375] |
| main | False | explicit | none | 400 | correct_attempt | 5 | 0.0125 | [0.0025, 0.025] |
| main | False | explicit | none | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.02] |
| main | False | explicit | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 400 | final_reply_length |  | 944.915 | [902.338, 989.488] |
| main | False | plain | none | 400 | violation | 28 | 0.07 | [0.0375, 0.105] |
| main | False | plain | none | 400 | leak | 27 | 0.0675 | [0.035, 0.1025] |
| main | False | plain | none | 400 | attempt | 3 | 0.0075 | [0, 0.0175] |
| main | False | plain | none | 400 | correct_attempt | 2 | 0.005 | [0, 0.0125] |
| main | False | plain | none | 400 | wrong_attempt | 1 | 0.0025 | [0, 0.0075] |
| main | False | plain | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | final_reply_length |  | 926.145 | [877.921, 974.457] |
| main | True | boxed | none | 400 | violation | 207 | 0.5175 | [0.465, 0.5725] |
| main | True | boxed | none | 400 | leak | 200 | 0.5 | [0.444937, 0.555062] |
| main | True | boxed | none | 400 | attempt | 133 | 0.3325 | [0.28, 0.3875] |
| main | True | boxed | none | 400 | correct_attempt | 125 | 0.3125 | [0.2575, 0.3675] |
| main | True | boxed | none | 400 | wrong_attempt | 8 | 0.02 | [0.0075, 0.035] |
| main | True | boxed | none | 400 | empty_final | 8 | 0.02 | [0.0075, 0.035] |
| main | True | boxed | none | 400 | truncated_final | 12 | 0.03 | [0.0125, 0.0525] |
| main | True | boxed | none | 400 | empty_or_truncated_final | 12 | 0.03 | [0.0125, 0.0525] |
| main | True | boxed | none | 400 | final_reply_length |  | 1025.71 | [981.067, 1073.38] |
| main | True | explicit | none | 400 | violation | 217 | 0.5425 | [0.49, 0.5925] |
| main | True | explicit | none | 400 | leak | 211 | 0.5275 | [0.475, 0.58] |
| main | True | explicit | none | 400 | attempt | 143 | 0.3575 | [0.3075, 0.4075] |
| main | True | explicit | none | 400 | correct_attempt | 135 | 0.3375 | [0.2875, 0.385062] |
| main | True | explicit | none | 400 | wrong_attempt | 8 | 0.02 | [0.0075, 0.035] |
| main | True | explicit | none | 400 | empty_final | 3 | 0.0075 | [0, 0.015] |
| main | True | explicit | none | 400 | truncated_final | 5 | 0.0125 | [0.0025, 0.025] |
| main | True | explicit | none | 400 | empty_or_truncated_final | 5 | 0.0125 | [0.0025, 0.025] |
| main | True | explicit | none | 400 | final_reply_length |  | 757.702 | [732.952, 785.931] |
| main | True | plain | none | 400 | violation | 66 | 0.165 | [0.125, 0.2125] |
| main | True | plain | none | 400 | leak | 61 | 0.1525 | [0.1125, 0.1975] |
| main | True | plain | none | 400 | attempt | 12 | 0.03 | [0.0125, 0.05] |
| main | True | plain | none | 400 | correct_attempt | 5 | 0.0125 | [0.0025, 0.025] |
| main | True | plain | none | 400 | wrong_attempt | 7 | 0.0175 | [0.0075, 0.0325] |
| main | True | plain | none | 400 | empty_final | 6 | 0.015 | [0.0025, 0.0325] |
| main | True | plain | none | 400 | truncated_final | 8 | 0.02 | [0.005, 0.0375] |
| main | True | plain | none | 400 | empty_or_truncated_final | 8 | 0.02 | [0.005, 0.0375] |
| main | True | plain | none | 400 | final_reply_length |  | 901.61 | [863.669, 944.507] |
| prefill | True | boxed | empty | 400 | violation | 195 | 0.4875 | [0.435, 0.5425] |
| prefill | True | boxed | empty | 400 | leak | 188 | 0.47 | [0.4175, 0.525] |
| prefill | True | boxed | empty | 400 | attempt | 136 | 0.34 | [0.2925, 0.39] |
| prefill | True | boxed | empty | 400 | correct_attempt | 128 | 0.32 | [0.2725, 0.37] |
| prefill | True | boxed | empty | 400 | wrong_attempt | 8 | 0.02 | [0.0075, 0.0375] |
| prefill | True | boxed | empty | 400 | empty_final | 9 | 0.0225 | [0.0075, 0.0400625] |
| prefill | True | boxed | empty | 400 | truncated_final | 11 | 0.0275 | [0.01, 0.0475] |
| prefill | True | boxed | empty | 400 | empty_or_truncated_final | 11 | 0.0275 | [0.01, 0.0475] |
| prefill | True | boxed | empty | 400 | final_reply_length |  | 1025.13 | [985.055, 1067.28] |
| prefill | True | boxed | irrelevant | 400 | violation | 66 | 0.165 | [0.125, 0.207562] |
| prefill | True | boxed | irrelevant | 400 | leak | 61 | 0.1525 | [0.11, 0.195] |
| prefill | True | boxed | irrelevant | 400 | attempt | 29 | 0.0725 | [0.0475, 0.1] |
| prefill | True | boxed | irrelevant | 400 | correct_attempt | 22 | 0.055 | [0.0325, 0.08] |
| prefill | True | boxed | irrelevant | 400 | wrong_attempt | 7 | 0.0175 | [0.0075, 0.0325] |
| prefill | True | boxed | irrelevant | 400 | empty_final | 3 | 0.0075 | [0, 0.0175] |
| prefill | True | boxed | irrelevant | 400 | truncated_final | 5 | 0.0125 | [0.0025, 0.0225625] |
| prefill | True | boxed | irrelevant | 400 | empty_or_truncated_final | 5 | 0.0125 | [0.0025, 0.0225625] |
| prefill | True | boxed | irrelevant | 400 | final_reply_length |  | 923.393 | [881.906, 967.618] |
| prefill | True | boxed | neutral | 400 | violation | 80 | 0.2 | [0.155, 0.25] |
| prefill | True | boxed | neutral | 400 | leak | 78 | 0.195 | [0.15, 0.2425] |
| prefill | True | boxed | neutral | 400 | attempt | 43 | 0.1075 | [0.075, 0.1425] |
| prefill | True | boxed | neutral | 400 | correct_attempt | 39 | 0.0975 | [0.0675, 0.130062] |
| prefill | True | boxed | neutral | 400 | wrong_attempt | 4 | 0.01 | [0.0025, 0.02] |
| prefill | True | boxed | neutral | 400 | empty_final | 3 | 0.0075 | [0, 0.0175] |
| prefill | True | boxed | neutral | 400 | truncated_final | 6 | 0.015 | [0.0025, 0.03] |
| prefill | True | boxed | neutral | 400 | empty_or_truncated_final | 6 | 0.015 | [0.0025, 0.03] |
| prefill | True | boxed | neutral | 400 | final_reply_length |  | 982.232 | [939.494, 1026.17] |
| prefill | True | boxed | reminder | 400 | violation | 43 | 0.1075 | [0.0725, 0.15] |
| prefill | True | boxed | reminder | 400 | leak | 39 | 0.0975 | [0.0625, 0.1375] |
| prefill | True | boxed | reminder | 400 | attempt | 9 | 0.0225 | [0.0075, 0.04] |
| prefill | True | boxed | reminder | 400 | correct_attempt | 5 | 0.0125 | [0, 0.0275] |
| prefill | True | boxed | reminder | 400 | wrong_attempt | 4 | 0.01 | [0.0025, 0.0225] |
| prefill | True | boxed | reminder | 400 | empty_final | 1 | 0.0025 | [0, 0.0075] |
| prefill | True | boxed | reminder | 400 | truncated_final | 1 | 0.0025 | [0, 0.0075] |
| prefill | True | boxed | reminder | 400 | empty_or_truncated_final | 1 | 0.0025 | [0, 0.0075] |
| prefill | True | boxed | reminder | 400 | final_reply_length |  | 891.778 | [849.065, 936.303] |
| prefill | True | explicit | empty | 400 | violation | 212 | 0.53 | [0.4775, 0.5825] |
| prefill | True | explicit | empty | 400 | leak | 204 | 0.51 | [0.4575, 0.56] |
| prefill | True | explicit | empty | 400 | attempt | 148 | 0.37 | [0.3225, 0.42] |
| prefill | True | explicit | empty | 400 | correct_attempt | 138 | 0.345 | [0.2975, 0.395] |
| prefill | True | explicit | empty | 400 | wrong_attempt | 10 | 0.025 | [0.01, 0.0425] |
| prefill | True | explicit | empty | 400 | empty_final | 0 | 0 | [0, 0] |
| prefill | True | explicit | empty | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | empty | 400 | empty_or_truncated_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | empty | 400 | final_reply_length |  | 778.908 | [750.059, 808.195] |
| prefill | True | explicit | irrelevant | 400 | violation | 65 | 0.1625 | [0.1175, 0.2125] |
| prefill | True | explicit | irrelevant | 400 | leak | 62 | 0.155 | [0.1125, 0.2025] |
| prefill | True | explicit | irrelevant | 400 | attempt | 29 | 0.0725 | [0.045, 0.1025] |
| prefill | True | explicit | irrelevant | 400 | correct_attempt | 26 | 0.065 | [0.04, 0.095] |
| prefill | True | explicit | irrelevant | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.02] |
| prefill | True | explicit | irrelevant | 400 | empty_final | 4 | 0.01 | [0.0025, 0.02] |
| prefill | True | explicit | irrelevant | 400 | truncated_final | 4 | 0.01 | [0.0025, 0.02] |
| prefill | True | explicit | irrelevant | 400 | empty_or_truncated_final | 4 | 0.01 | [0.0025, 0.02] |
| prefill | True | explicit | irrelevant | 400 | final_reply_length |  | 775.122 | [744.129, 808.623] |
| prefill | True | explicit | neutral | 400 | violation | 99 | 0.2475 | [0.2025, 0.2975] |
| prefill | True | explicit | neutral | 400 | leak | 93 | 0.2325 | [0.1875, 0.280062] |
| prefill | True | explicit | neutral | 400 | attempt | 75 | 0.1875 | [0.15, 0.23] |
| prefill | True | explicit | neutral | 400 | correct_attempt | 68 | 0.17 | [0.1325, 0.2125] |
| prefill | True | explicit | neutral | 400 | wrong_attempt | 7 | 0.0175 | [0.005, 0.035] |
| prefill | True | explicit | neutral | 400 | empty_final | 1 | 0.0025 | [0, 0.0075] |
| prefill | True | explicit | neutral | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | neutral | 400 | empty_or_truncated_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | neutral | 400 | final_reply_length |  | 790.158 | [753.777, 829.411] |
| prefill | True | explicit | reminder | 400 | violation | 32 | 0.08 | [0.0475, 0.1175] |
| prefill | True | explicit | reminder | 400 | leak | 30 | 0.075 | [0.0425, 0.1125] |
| prefill | True | explicit | reminder | 400 | attempt | 6 | 0.015 | [0.005, 0.0275] |
| prefill | True | explicit | reminder | 400 | correct_attempt | 4 | 0.01 | [0.0025, 0.02] |
| prefill | True | explicit | reminder | 400 | wrong_attempt | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | reminder | 400 | empty_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | reminder | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | reminder | 400 | empty_or_truncated_final | 2 | 0.005 | [0, 0.0125] |
| prefill | True | explicit | reminder | 400 | final_reply_length |  | 731.942 | [698.707, 766.034] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H15 | 0.3525 | [0.295, 0.4075] | True | True |
| violation | on_plain_minus_off_plain | 0.095 | [0.06, 0.1325] | False | None |
| violation | H16 | -0.025 | [-0.085, 0.035] | True | False |
| violation | H17_empty_minus_reminder | 0.415 | [0.37625, 0.455] | True | True |
| violation | H17_neutral_minus_reminder | 0.13 | [0.10625, 0.15625] | True | True |
| violation | H17_irrelevant_minus_reminder | 0.07 | [0.0475, 0.09625] | True | True |
| violation | H17_joint | NA | [NA, NA] | True | True |
| leak | H15 | 0.3525 | [0.295, 0.4075] | False | None |
| leak | on_plain_minus_off_plain | 0.085 | [0.05, 0.120062] | False | None |
| leak | H16 | -0.0275 | [-0.0875, 0.035] | False | None |
| leak | H17_empty_minus_reminder | 0.40375 | [0.365, 0.44375] | False | None |
| leak | H17_neutral_minus_reminder | 0.1275 | [0.1025, 0.155] | False | None |
| leak | H17_irrelevant_minus_reminder | 0.0675 | [0.0475, 0.09] | False | None |
| leak | H17_joint | NA | [NA, NA] | False | None |

### Qwen/Qwen3-32B

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | False | boxed | none | 400 | violation | 31 | 0.0775 | [0.045, 0.115] |
| main | False | boxed | none | 400 | leak | 29 | 0.0725 | [0.0424375, 0.11] |
| main | False | boxed | none | 400 | attempt | 9 | 0.0225 | [0.01, 0.0375] |
| main | False | boxed | none | 400 | correct_attempt | 6 | 0.015 | [0.005, 0.0275] |
| main | False | boxed | none | 400 | wrong_attempt | 3 | 0.0075 | [0, 0.0175] |
| main | False | boxed | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 400 | final_reply_length |  | 1038.44 | [991.274, 1087.07] |
| main | False | explicit | none | 400 | violation | 115 | 0.2875 | [0.24, 0.3375] |
| main | False | explicit | none | 400 | leak | 102 | 0.255 | [0.205, 0.305] |
| main | False | explicit | none | 400 | attempt | 74 | 0.185 | [0.1475, 0.225] |
| main | False | explicit | none | 400 | correct_attempt | 60 | 0.15 | [0.114938, 0.1925] |
| main | False | explicit | none | 400 | wrong_attempt | 14 | 0.035 | [0.0175, 0.0525625] |
| main | False | explicit | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 400 | final_reply_length |  | 1053.38 | [1012.93, 1093.35] |
| main | False | plain | none | 400 | violation | 28 | 0.07 | [0.04, 0.105] |
| main | False | plain | none | 400 | leak | 28 | 0.07 | [0.04, 0.105] |
| main | False | plain | none | 400 | attempt | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | correct_attempt | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | wrong_attempt | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 400 | final_reply_length |  | 955.795 | [912.144, 998.463] |
| main | True | boxed | none | 400 | violation | 316 | 0.79 | [0.75, 0.83] |
| main | True | boxed | none | 400 | leak | 303 | 0.7575 | [0.7125, 0.8] |
| main | True | boxed | none | 400 | attempt | 313 | 0.7825 | [0.7425, 0.8225] |
| main | True | boxed | none | 400 | correct_attempt | 298 | 0.745 | [0.7, 0.7875] |
| main | True | boxed | none | 400 | wrong_attempt | 15 | 0.0375 | [0.0175, 0.0625] |
| main | True | boxed | none | 400 | empty_final | 2 | 0.005 | [0, 0.015] |
| main | True | boxed | none | 400 | truncated_final | 5 | 0.0125 | [0.0025, 0.0275] |
| main | True | boxed | none | 400 | empty_or_truncated_final | 5 | 0.0125 | [0.0025, 0.0275] |
| main | True | boxed | none | 400 | final_reply_length |  | 837.675 | [804.371, 871.833] |
| main | True | explicit | none | 400 | violation | 394 | 0.985 | [0.97, 0.9975] |
| main | True | explicit | none | 400 | leak | 383 | 0.9575 | [0.93, 0.9825] |
| main | True | explicit | none | 400 | attempt | 197 | 0.4925 | [0.4375, 0.547562] |
| main | True | explicit | none | 400 | correct_attempt | 180 | 0.45 | [0.395, 0.505] |
| main | True | explicit | none | 400 | wrong_attempt | 17 | 0.0425 | [0.0175, 0.0725] |
| main | True | explicit | none | 400 | empty_final | 1 | 0.0025 | [0, 0.0075] |
| main | True | explicit | none | 400 | truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | True | explicit | none | 400 | empty_or_truncated_final | 2 | 0.005 | [0, 0.0125] |
| main | True | explicit | none | 400 | final_reply_length |  | 573.163 | [550.186, 595.756] |
| main | True | plain | none | 400 | violation | 67 | 0.1675 | [0.125, 0.212562] |
| main | True | plain | none | 400 | leak | 67 | 0.1675 | [0.125, 0.212562] |
| main | True | plain | none | 400 | attempt | 7 | 0.0175 | [0.005, 0.0325] |
| main | True | plain | none | 400 | correct_attempt | 7 | 0.0175 | [0.005, 0.0325] |
| main | True | plain | none | 400 | wrong_attempt | 0 | 0 | [0, 0] |
| main | True | plain | none | 400 | empty_final | 2 | 0.005 | [0, 0.0125] |
| main | True | plain | none | 400 | truncated_final | 4 | 0.01 | [0, 0.0225] |
| main | True | plain | none | 400 | empty_or_truncated_final | 4 | 0.01 | [0, 0.0225] |
| main | True | plain | none | 400 | final_reply_length |  | 863.385 | [829.869, 898.815] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H15 | 0.6 | [0.5425, 0.655] | True | True |
| violation | on_plain_minus_off_plain | 0.0975 | [0.065, 0.1325] | False | None |
| violation | H16 | -0.195 | [-0.2375, -0.1525] | True | False |
| leak | H15 | 0.605 | [0.545, 0.665] | False | None |
| leak | on_plain_minus_off_plain | 0.0975 | [0.065, 0.1325] | False | None |
| leak | H16 | -0.2 | [-0.24, -0.16] | False | None |

## Gold >= 10

Problems: 169; responses: 6760.

### Qwen/Qwen3-14B

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | False | boxed | none | 338 | violation | 12 | 0.035503 | [0.0147929, 0.0591716] |
| main | False | boxed | none | 338 | leak | 10 | 0.0295858 | [0.0118343, 0.0502959] |
| main | False | boxed | none | 338 | attempt | 4 | 0.0118343 | [0, 0.0266272] |
| main | False | boxed | none | 338 | correct_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | False | boxed | none | 338 | wrong_attempt | 2 | 0.00591716 | [0, 0.0177515] |
| main | False | boxed | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | final_reply_length |  | 987.515 | [937.101, 1043.67] |
| main | False | explicit | none | 338 | violation | 12 | 0.035503 | [0.0177515, 0.056213] |
| main | False | explicit | none | 338 | leak | 12 | 0.035503 | [0.0177515, 0.056213] |
| main | False | explicit | none | 338 | attempt | 3 | 0.00887574 | [0, 0.0207101] |
| main | False | explicit | none | 338 | correct_attempt | 3 | 0.00887574 | [0, 0.0207101] |
| main | False | explicit | none | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | final_reply_length |  | 941.657 | [898.566, 989.666] |
| main | False | plain | none | 338 | violation | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | False | plain | none | 338 | leak | 3 | 0.00887574 | [0, 0.0207101] |
| main | False | plain | none | 338 | attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | False | plain | none | 338 | correct_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | False | plain | none | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | False | plain | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | final_reply_length |  | 922.503 | [875.659, 974.369] |
| main | True | boxed | none | 338 | violation | 168 | 0.497041 | [0.43787, 0.553254] |
| main | True | boxed | none | 338 | leak | 161 | 0.476331 | [0.41716, 0.535503] |
| main | True | boxed | none | 338 | attempt | 116 | 0.343195 | [0.286982, 0.399408] |
| main | True | boxed | none | 338 | correct_attempt | 108 | 0.319527 | [0.263314, 0.37574] |
| main | True | boxed | none | 338 | wrong_attempt | 8 | 0.0236686 | [0.00887574, 0.0414201] |
| main | True | boxed | none | 338 | empty_final | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| main | True | boxed | none | 338 | truncated_final | 9 | 0.0266272 | [0.00887574, 0.0473373] |
| main | True | boxed | none | 338 | empty_or_truncated_final | 9 | 0.0266272 | [0.00887574, 0.0473373] |
| main | True | boxed | none | 338 | final_reply_length |  | 1028.45 | [981.117, 1074.72] |
| main | True | explicit | none | 338 | violation | 170 | 0.502959 | [0.44963, 0.553254] |
| main | True | explicit | none | 338 | leak | 164 | 0.485207 | [0.428994, 0.535503] |
| main | True | explicit | none | 338 | attempt | 117 | 0.346154 | [0.292899, 0.399408] |
| main | True | explicit | none | 338 | correct_attempt | 111 | 0.328402 | [0.278107, 0.384615] |
| main | True | explicit | none | 338 | wrong_attempt | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| main | True | explicit | none | 338 | empty_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | True | explicit | none | 338 | truncated_final | 4 | 0.0118343 | [0, 0.0266272] |
| main | True | explicit | none | 338 | empty_or_truncated_final | 4 | 0.0118343 | [0, 0.0266272] |
| main | True | explicit | none | 338 | final_reply_length |  | 757.967 | [729.916, 788.273] |
| main | True | plain | none | 338 | violation | 37 | 0.109467 | [0.0739645, 0.150888] |
| main | True | plain | none | 338 | leak | 32 | 0.0946746 | [0.0591716, 0.133136] |
| main | True | plain | none | 338 | attempt | 11 | 0.0325444 | [0.0118343, 0.0532544] |
| main | True | plain | none | 338 | correct_attempt | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| main | True | plain | none | 338 | wrong_attempt | 7 | 0.0207101 | [0.00591716, 0.035503] |
| main | True | plain | none | 338 | empty_final | 4 | 0.0118343 | [0, 0.0266272] |
| main | True | plain | none | 338 | truncated_final | 5 | 0.0147929 | [0.00295858, 0.0325444] |
| main | True | plain | none | 338 | empty_or_truncated_final | 5 | 0.0147929 | [0.00295858, 0.0325444] |
| main | True | plain | none | 338 | final_reply_length |  | 889.562 | [850.254, 927.318] |
| prefill | True | boxed | empty | 338 | violation | 153 | 0.452663 | [0.39645, 0.511834] |
| prefill | True | boxed | empty | 338 | leak | 146 | 0.431953 | [0.37574, 0.488166] |
| prefill | True | boxed | empty | 338 | attempt | 110 | 0.325444 | [0.272189, 0.378698] |
| prefill | True | boxed | empty | 338 | correct_attempt | 102 | 0.301775 | [0.251479, 0.35503] |
| prefill | True | boxed | empty | 338 | wrong_attempt | 8 | 0.0236686 | [0.00880178, 0.0443787] |
| prefill | True | boxed | empty | 338 | empty_final | 8 | 0.0236686 | [0.00887574, 0.0444527] |
| prefill | True | boxed | empty | 338 | truncated_final | 10 | 0.0295858 | [0.0118343, 0.0532544] |
| prefill | True | boxed | empty | 338 | empty_or_truncated_final | 10 | 0.0295858 | [0.0118343, 0.0532544] |
| prefill | True | boxed | empty | 338 | final_reply_length |  | 1030.84 | [986.398, 1075.35] |
| prefill | True | boxed | irrelevant | 338 | violation | 39 | 0.115385 | [0.0798817, 0.150888] |
| prefill | True | boxed | irrelevant | 338 | leak | 34 | 0.100592 | [0.0650888, 0.136095] |
| prefill | True | boxed | irrelevant | 338 | attempt | 25 | 0.0739645 | [0.0443787, 0.106509] |
| prefill | True | boxed | irrelevant | 338 | correct_attempt | 19 | 0.056213 | [0.0295858, 0.0857988] |
| prefill | True | boxed | irrelevant | 338 | wrong_attempt | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| prefill | True | boxed | irrelevant | 338 | empty_final | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | True | boxed | irrelevant | 338 | truncated_final | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | True | boxed | irrelevant | 338 | empty_or_truncated_final | 5 | 0.0147929 | [0.00295858, 0.0295858] |
| prefill | True | boxed | irrelevant | 338 | final_reply_length |  | 906.861 | [866.292, 949.25] |
| prefill | True | boxed | neutral | 338 | violation | 48 | 0.142012 | [0.100592, 0.186391] |
| prefill | True | boxed | neutral | 338 | leak | 46 | 0.136095 | [0.0975592, 0.180473] |
| prefill | True | boxed | neutral | 338 | attempt | 34 | 0.100592 | [0.0680473, 0.136095] |
| prefill | True | boxed | neutral | 338 | correct_attempt | 30 | 0.0887574 | [0.056213, 0.121302] |
| prefill | True | boxed | neutral | 338 | wrong_attempt | 4 | 0.0118343 | [0.00295858, 0.0266272] |
| prefill | True | boxed | neutral | 338 | empty_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | boxed | neutral | 338 | truncated_final | 5 | 0.0147929 | [0, 0.0325444] |
| prefill | True | boxed | neutral | 338 | empty_or_truncated_final | 5 | 0.0147929 | [0, 0.0325444] |
| prefill | True | boxed | neutral | 338 | final_reply_length |  | 974.689 | [932.047, 1021.64] |
| prefill | True | boxed | reminder | 338 | violation | 13 | 0.0384615 | [0.0177515, 0.0621302] |
| prefill | True | boxed | reminder | 338 | leak | 9 | 0.0266272 | [0.00887574, 0.0473373] |
| prefill | True | boxed | reminder | 338 | attempt | 7 | 0.0207101 | [0.00591716, 0.0384615] |
| prefill | True | boxed | reminder | 338 | correct_attempt | 3 | 0.00887574 | [0, 0.0236686] |
| prefill | True | boxed | reminder | 338 | wrong_attempt | 4 | 0.0118343 | [0.00288462, 0.0236686] |
| prefill | True | boxed | reminder | 338 | empty_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | True | boxed | reminder | 338 | truncated_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | True | boxed | reminder | 338 | empty_or_truncated_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | True | boxed | reminder | 338 | final_reply_length |  | 877.003 | [833.893, 920.544] |
| prefill | True | explicit | empty | 338 | violation | 164 | 0.485207 | [0.428994, 0.544379] |
| prefill | True | explicit | empty | 338 | leak | 156 | 0.461538 | [0.405325, 0.52071] |
| prefill | True | explicit | empty | 338 | attempt | 122 | 0.360947 | [0.313609, 0.414275] |
| prefill | True | explicit | empty | 338 | correct_attempt | 113 | 0.33432 | [0.286982, 0.384615] |
| prefill | True | explicit | empty | 338 | wrong_attempt | 9 | 0.0266272 | [0.00887574, 0.0473373] |
| prefill | True | explicit | empty | 338 | empty_final | 0 | 0 | [0, 0] |
| prefill | True | explicit | empty | 338 | truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | empty | 338 | empty_or_truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | empty | 338 | final_reply_length |  | 779.879 | [750.813, 809.491] |
| prefill | True | explicit | irrelevant | 338 | violation | 31 | 0.091716 | [0.0591716, 0.127219] |
| prefill | True | explicit | irrelevant | 338 | leak | 28 | 0.0828402 | [0.0532544, 0.115385] |
| prefill | True | explicit | irrelevant | 338 | attempt | 20 | 0.0591716 | [0.0325444, 0.0888314] |
| prefill | True | explicit | irrelevant | 338 | correct_attempt | 17 | 0.0502959 | [0.0266272, 0.0769231] |
| prefill | True | explicit | irrelevant | 338 | wrong_attempt | 3 | 0.00887574 | [0, 0.0236686] |
| prefill | True | explicit | irrelevant | 338 | empty_final | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| prefill | True | explicit | irrelevant | 338 | truncated_final | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| prefill | True | explicit | irrelevant | 338 | empty_or_truncated_final | 4 | 0.0118343 | [0.00295858, 0.0236686] |
| prefill | True | explicit | irrelevant | 338 | final_reply_length |  | 771.686 | [734.819, 810.41] |
| prefill | True | explicit | neutral | 338 | violation | 64 | 0.189349 | [0.147929, 0.233802] |
| prefill | True | explicit | neutral | 338 | leak | 58 | 0.171598 | [0.130178, 0.215976] |
| prefill | True | explicit | neutral | 338 | attempt | 59 | 0.174556 | [0.133136, 0.218935] |
| prefill | True | explicit | neutral | 338 | correct_attempt | 53 | 0.156805 | [0.118343, 0.198225] |
| prefill | True | explicit | neutral | 338 | wrong_attempt | 6 | 0.0177515 | [0.00295858, 0.035503] |
| prefill | True | explicit | neutral | 338 | empty_final | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | True | explicit | neutral | 338 | truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | neutral | 338 | empty_or_truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | neutral | 338 | final_reply_length |  | 778.891 | [742.118, 822.746] |
| prefill | True | explicit | reminder | 338 | violation | 7 | 0.0207101 | [0.00591716, 0.035503] |
| prefill | True | explicit | reminder | 338 | leak | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| prefill | True | explicit | reminder | 338 | attempt | 3 | 0.00887574 | [0, 0.0207101] |
| prefill | True | explicit | reminder | 338 | correct_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | reminder | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | True | explicit | reminder | 338 | empty_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | reminder | 338 | truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | reminder | 338 | empty_or_truncated_final | 2 | 0.00591716 | [0, 0.0147929] |
| prefill | True | explicit | reminder | 338 | final_reply_length |  | 721.334 | [686.466, 757.068] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H15 | 0.369822 | [0.307692, 0.428994] | True | True |
| violation | on_plain_minus_off_plain | 0.0976331 | [0.0621302, 0.142012] | False | None |
| violation | H16 | -0.00591716 | [-0.0739645, 0.0621302] | True | False |
| violation | H17_empty_minus_reminder | 0.439349 | [0.399408, 0.480806] | True | True |
| violation | H17_neutral_minus_reminder | 0.136095 | [0.107988, 0.16716] | True | True |
| violation | H17_irrelevant_minus_reminder | 0.0739645 | [0.0502589, 0.100629] | True | True |
| violation | H17_joint | NA | [NA, NA] | True | True |
| leak | H15 | 0.363905 | [0.301775, 0.423077] | False | None |
| leak | on_plain_minus_off_plain | 0.0857988 | [0.0502959, 0.12426] | False | None |
| leak | H16 | -0.00887574 | [-0.0769231, 0.0591716] | False | None |
| leak | H17_empty_minus_reminder | 0.424556 | [0.383136, 0.465976] | False | None |
| leak | H17_neutral_minus_reminder | 0.131657 | [0.10355, 0.162722] | False | None |
| leak | H17_irrelevant_minus_reminder | 0.0695266 | [0.0473373, 0.0931953] | False | None |
| leak | H17_joint | NA | [NA, NA] | False | None |

### Qwen/Qwen3-32B

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | False | boxed | none | 338 | violation | 5 | 0.0147929 | [0.00295858, 0.0325444] |
| main | False | boxed | none | 338 | leak | 3 | 0.00887574 | [0, 0.0236686] |
| main | False | boxed | none | 338 | attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | False | boxed | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | wrong_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | False | boxed | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | boxed | none | 338 | final_reply_length |  | 1037.27 | [989.634, 1092.47] |
| main | False | explicit | none | 338 | violation | 80 | 0.236686 | [0.192308, 0.284024] |
| main | False | explicit | none | 338 | leak | 68 | 0.201183 | [0.156805, 0.248521] |
| main | False | explicit | none | 338 | attempt | 61 | 0.180473 | [0.136095, 0.227811] |
| main | False | explicit | none | 338 | correct_attempt | 49 | 0.14497 | [0.10355, 0.189349] |
| main | False | explicit | none | 338 | wrong_attempt | 12 | 0.035503 | [0.0177515, 0.056213] |
| main | False | explicit | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | explicit | none | 338 | final_reply_length |  | 1051.51 | [1012.59, 1093.41] |
| main | False | plain | none | 338 | violation | 3 | 0.00887574 | [0, 0.0236686] |
| main | False | plain | none | 338 | leak | 3 | 0.00887574 | [0, 0.0236686] |
| main | False | plain | none | 338 | attempt | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | False | plain | none | 338 | final_reply_length |  | 958.337 | [914.427, 1006.67] |
| main | True | boxed | none | 338 | violation | 263 | 0.778107 | [0.733728, 0.822485] |
| main | True | boxed | none | 338 | leak | 250 | 0.739645 | [0.689349, 0.786982] |
| main | True | boxed | none | 338 | attempt | 263 | 0.778107 | [0.733728, 0.822485] |
| main | True | boxed | none | 338 | correct_attempt | 250 | 0.739645 | [0.689349, 0.786982] |
| main | True | boxed | none | 338 | wrong_attempt | 13 | 0.0384615 | [0.0147929, 0.0650888] |
| main | True | boxed | none | 338 | empty_final | 2 | 0.00591716 | [0, 0.0177515] |
| main | True | boxed | none | 338 | truncated_final | 3 | 0.00887574 | [0, 0.0236686] |
| main | True | boxed | none | 338 | empty_or_truncated_final | 3 | 0.00887574 | [0, 0.0236686] |
| main | True | boxed | none | 338 | final_reply_length |  | 838.05 | [802.309, 875.681] |
| main | True | explicit | none | 338 | violation | 334 | 0.988166 | [0.970414, 1] |
| main | True | explicit | none | 338 | leak | 323 | 0.955621 | [0.920118, 0.982249] |
| main | True | explicit | none | 338 | attempt | 167 | 0.494083 | [0.431953, 0.556213] |
| main | True | explicit | none | 338 | correct_attempt | 152 | 0.449704 | [0.390533, 0.514793] |
| main | True | explicit | none | 338 | wrong_attempt | 15 | 0.0443787 | [0.0177515, 0.0769231] |
| main | True | explicit | none | 338 | empty_final | 1 | 0.00295858 | [0, 0.0089497] |
| main | True | explicit | none | 338 | truncated_final | 1 | 0.00295858 | [0, 0.0089497] |
| main | True | explicit | none | 338 | empty_or_truncated_final | 1 | 0.00295858 | [0, 0.0089497] |
| main | True | explicit | none | 338 | final_reply_length |  | 572.929 | [548.483, 598.474] |
| main | True | plain | none | 338 | violation | 35 | 0.10355 | [0.0680473, 0.14497] |
| main | True | plain | none | 338 | leak | 35 | 0.10355 | [0.0680473, 0.14497] |
| main | True | plain | none | 338 | attempt | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| main | True | plain | none | 338 | correct_attempt | 6 | 0.0177515 | [0.00591716, 0.0325444] |
| main | True | plain | none | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| main | True | plain | none | 338 | empty_final | 2 | 0.00591716 | [0, 0.0147929] |
| main | True | plain | none | 338 | truncated_final | 3 | 0.00887574 | [0, 0.0236686] |
| main | True | plain | none | 338 | empty_or_truncated_final | 3 | 0.00887574 | [0, 0.0236686] |
| main | True | plain | none | 338 | final_reply_length |  | 866.18 | [831.346, 901.102] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H15 | 0.656805 | [0.597633, 0.713018] | True | True |
| violation | on_plain_minus_off_plain | 0.0946746 | [0.0591716, 0.133136] | False | None |
| violation | H16 | -0.210059 | [-0.257396, -0.162722] | True | False |
| leak | H15 | 0.659763 | [0.597633, 0.718935] | False | None |
| leak | on_plain_minus_off_plain | 0.0946746 | [0.0591716, 0.133136] | False | None |
| leak | H16 | -0.215976 | [-0.263314, -0.174482] | False | None |
