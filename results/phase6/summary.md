# Phase 6

| Setting | Value |
| --- | --- |
| Responses | 4400 |
| Problems | 200 |
| Bootstrap | 2,000 paired problem clusters; seed 0; percentile 95% |
| Denominator | All requests, including empty finals |
| Empty prefill | Reuses Qwen3.8 main / thinking on / explicit |
| Dropped stages | none |
| Inference eligible | True |
| Pending gates | none |
| Parser audit | complete; 80/80 |
| Audit labels | Codex self-labels; no independent human labels |
| Attempt agreement | 0.95 |
| Numeric agreement | 1 |
| Protocol deviation | Authorized deviation from DESIGN.md at 8219501: no analysis-blocking protocol-failure gate. Thinking-off replies containing literal </think> in saved-ID-validated decoded text are scored only after the last closer; violation_sensitivity counts those replies as violations. Generator spans/helpers retain their original validation; other scoring rules are unchanged. |
| Sensitivity outcome | violation_sensitivity; H25 uses only thinking-on traces, so its outcomes are unchanged |
| Malformed records | Saved-ID/helper mismatches are rejected; tokens after EOS retain protocol diagnostics and an empty scored final |

H22: holds=False; estimate=0.015; CI95=[-0.0125, 0.045].
H24: holds=False; estimate=0.0075; CI95=[-0.0225, 0.03756249999999966].
H23: holds=False; all three directional contrasts required.
H25: Qwen/Qwen3.8-27B-FP8: insufficient_R_or_N, holds=None; Qwen/Qwen3.5-27B-FP8: insufficient_R_or_N, holds=None; Qwen/Qwen3-32B: insufficient_R_or_N, holds=None; google/gemma-4-12B-it: insufficient_R_or_N, holds=None.

## All gold

| Aggregate | Metric | Count | Estimate | CI95 |
| --- | --- | ---: | ---: | --- |
| All models | violation | 286 | 0.065 | [0.04022159090909092, 0.09522727272727277] |
| All models | violation_sensitivity | 286 | 0.065 | [0.04022159090909092, 0.09522727272727277] |
| All models | off_think_close | 0 | 0 | [0.0, 0.0] |
| All models | protocol_failure | 6 | 0.00136364 | [0.00045454545454545455, 0.0025000000000000005] |
| Qwen/Qwen3.8-27B-FP8 | violation | 193 | 0.0689286 | [0.04357142857142855, 0.09964285714285713] |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | 193 | 0.0689286 | [0.04357142857142855, 0.09964285714285713] |
| Qwen/Qwen3.8-27B-FP8 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | protocol_failure | 6 | 0.00214286 | [0.0007142857142857143, 0.003928571428571427] |
| Qwen/Qwen3.5-27B-FP8 | violation | 93 | 0.058125 | [0.0325, 0.0881406249999999] |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | 93 | 0.058125 | [0.0325, 0.0881406249999999] |
| Qwen/Qwen3.5-27B-FP8 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | protocol_failure | 0 | 0 | [0.0, 0.0] |

| Model | Experiment | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI95 |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | violation | 26 | 0.065 | [0.035, 0.1] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | leak | 26 | 0.065 | [0.035, 0.1] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | final_reply_length |  | 1022.85 | [968.644, 1081.9090625] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 400 | violation_sensitivity | 26 | 0.065 | [0.035, 0.1] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | violation | 24 | 0.06 | [0.03493750000000001, 0.09] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | leak | 19 | 0.0475 | [0.0225, 0.0775] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | attempt | 7 | 0.0175 | [0.005, 0.03] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | wrong_attempt | 7 | 0.0175 | [0.005, 0.03] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | final_reply_length |  | 879.035 | [834.9472499999999, 925.5043125] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 400 | violation_sensitivity | 24 | 0.06 | [0.03493750000000001, 0.09] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | violation | 25 | 0.0625 | [0.03, 0.0975] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | leak | 23 | 0.0575 | [0.0275, 0.0925] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | attempt | 2 | 0.005 | [0.0, 0.015] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | wrong_attempt | 2 | 0.005 | [0.0, 0.015] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | final_reply_length |  | 936.477 | [912.5749375, 961.2986875] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 400 | violation_sensitivity | 25 | 0.0625 | [0.03, 0.0975] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | violation | 17 | 0.0425 | [0.02, 0.07] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | leak | 17 | 0.0425 | [0.02, 0.07] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | final_reply_length |  | 809.11 | [785.3460625, 832.8053124999999] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 400 | violation_sensitivity | 17 | 0.0425 | [0.02, 0.07] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | violation | 46 | 0.115 | [0.0825, 0.1525] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | leak | 45 | 0.1125 | [0.08, 0.1475] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | attempt | 11 | 0.0275 | [0.0125, 0.045] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | correct_attempt | 10 | 0.025 | [0.0125, 0.0425] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | wrong_attempt | 1 | 0.0025 | [0.0, 0.0075] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | empty_final | 13 | 0.0325 | [0.015, 0.05006249999999966] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | truncated_final | 7 | 0.0175 | [0.005, 0.0325] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | empty_or_truncated_final | 13 | 0.0325 | [0.015, 0.05006249999999966] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | final_reply_length |  | 852.365 | [817.8863125, 885.22575] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | cap_hit | 7 | 0.0175 | [0.005, 0.0325] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | no_final | 13 | 0.0325 | [0.015, 0.05006249999999966] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | cap_hit_no_final | 7 | 0.0175 | [0.005, 0.0325] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | protocol_failure | 6 | 0.015 | [0.005, 0.0275] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 400 | violation_sensitivity | 46 | 0.115 | [0.0825, 0.1525] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | violation | 35 | 0.0875 | [0.0525, 0.125] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | leak | 33 | 0.0825 | [0.05, 0.12] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | attempt | 3 | 0.0075 | [0.0, 0.02] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | correct_attempt | 1 | 0.0025 | [0.0, 0.0075] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | wrong_attempt | 2 | 0.005 | [0.0, 0.015] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | final_reply_length |  | 1081.73 | [1051.87, 1110.7045625] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 400 | violation_sensitivity | 35 | 0.0875 | [0.0525, 0.125] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | violation | 20 | 0.05 | [0.0225, 0.0825] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | leak | 19 | 0.0475 | [0.02, 0.08] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | attempt | 1 | 0.0025 | [0.0, 0.0075] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | wrong_attempt | 1 | 0.0025 | [0.0, 0.0075] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | final_reply_length |  | 918.79 | [895.378125, 943.2440624999999] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 400 | violation_sensitivity | 20 | 0.05 | [0.0225, 0.0825] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | violation | 26 | 0.065 | [0.035, 0.1] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | leak | 23 | 0.0575 | [0.03, 0.0925] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | attempt | 3 | 0.0075 | [0.0, 0.0175] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | wrong_attempt | 3 | 0.0075 | [0.0, 0.0175] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | final_reply_length |  | 1247.94 | [1202.3415625, 1295.0435] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 400 | violation_sensitivity | 26 | 0.065 | [0.035, 0.1] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | violation | 24 | 0.06 | [0.0325, 0.0925] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | leak | 23 | 0.0575 | [0.03, 0.09] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | attempt | 1 | 0.0025 | [0.0, 0.0075] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | wrong_attempt | 1 | 0.0025 | [0.0, 0.0075] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | final_reply_length |  | 1049.72 | [1002.2515625, 1101.6504375] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 400 | violation_sensitivity | 24 | 0.06 | [0.0325, 0.0925] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | violation | 24 | 0.06 | [0.0325, 0.0925] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | leak | 21 | 0.0525 | [0.025, 0.085] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | attempt | 3 | 0.0075 | [0.0, 0.02] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | wrong_attempt | 3 | 0.0075 | [0.0, 0.02] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | final_reply_length |  | 1064.74 | [1036.522375, 1093.7484375] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 400 | violation_sensitivity | 24 | 0.06 | [0.0325, 0.0925] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | violation | 19 | 0.0475 | [0.0225, 0.0775] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | leak | 19 | 0.0475 | [0.0225, 0.0775] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | final_reply_length |  | 907.018 | [877.0121250000001, 937.794875] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 400 | violation_sensitivity | 19 | 0.0475 | [0.0225, 0.0775] |

| Model | Outcome | Contrast | Estimate | CI95 | Holds |
| --- | --- | --- | ---: | --- | --- |
| Qwen/Qwen3.8-27B-FP8 | violation | H22 | 0.015 | [-0.0125, 0.045] | False |
| Qwen/Qwen3.8-27B-FP8 | violation | on_plain_minus_off_plain | -0.0175 | [-0.0375, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_empty_minus_reminder | 0.0125 | [0.0, 0.03] | False |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_neutral_minus_reminder | 0.0375 | [0.0175, 0.06] | True |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_irrelevant_minus_reminder | 0.065 | [0.0375, 0.095] | True |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_joint | NA |  | False |
| Qwen/Qwen3.8-27B-FP8 | leak | H22 | -0.0025 | [-0.025, 0.0225] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | on_plain_minus_off_plain | -0.005 | [-0.02, 0.01] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_empty_minus_reminder | 0.01 | [-0.0025, 0.0275] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_neutral_minus_reminder | 0.035 | [0.017437500000000005, 0.0575] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_irrelevant_minus_reminder | 0.065 | [0.0375, 0.095] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_joint | NA |  | None |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H22 | 0.015 | [-0.0125, 0.045] | False |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | on_plain_minus_off_plain | -0.0175 | [-0.0375, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_empty_minus_reminder | 0.0125 | [0.0, 0.03] | False |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_neutral_minus_reminder | 0.0375 | [0.0175, 0.06] | True |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_irrelevant_minus_reminder | 0.065 | [0.0375, 0.095] | True |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_joint | NA |  | False |
| Qwen/Qwen3.5-27B-FP8 | violation | H24 | 0.0075 | [-0.0225, 0.03756249999999966] | False |
| Qwen/Qwen3.5-27B-FP8 | violation | on_plain_minus_off_plain | -0.0125 | [-0.0325, 0.005] | None |
| Qwen/Qwen3.5-27B-FP8 | leak | H24 | 0.005 | [-0.02, 0.03] | None |
| Qwen/Qwen3.5-27B-FP8 | leak | on_plain_minus_off_plain | -0.01 | [-0.03, 0.005] | None |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | H24 | 0.0075 | [-0.0225, 0.03756249999999966] | False |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | on_plain_minus_off_plain | -0.0125 | [-0.0325, 0.005] | None |

| H25 model | Status | R | L | N | Keyword mentions | N minus R | CI95 | Undefined resamples | Holds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Qwen/Qwen3.8-27B-FP8 | insufficient_R_or_N | 361 | 39 | 0 | 400 | NA | [None, None] | None | None |
| Qwen/Qwen3.5-27B-FP8 | insufficient_R_or_N | 399 | 1 | 0 | 392 | NA | [None, None] | None | None |
| Qwen/Qwen3-32B | insufficient_R_or_N | 1 | 6 | 393 | 3 | NA | [None, None] | None | None |
| google/gemma-4-12B-it | insufficient_R_or_N | 370 | 30 | 0 | 309 | NA | [None, None] | None | None |

| H25 sensitivity model | Outcome | Estimate | CI95 | Holds |
| --- | --- | ---: | --- | --- |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | NA | [None, None] | None |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | NA | [None, None] | None |
| Qwen/Qwen3-32B | violation_sensitivity | NA | [None, None] | None |
| google/gemma-4-12B-it | violation_sensitivity | NA | [None, None] | None |

## Gold >= 10

| Aggregate | Metric | Count | Estimate | CI95 |
| --- | --- | ---: | ---: | --- |
| All models | violation | 43 | 0.0115654 | [0.007530930607853682, 0.01694459386767078] |
| All models | violation_sensitivity | 43 | 0.0115654 | [0.007530930607853682, 0.01694459386767078] |
| All models | off_think_close | 0 | 0 | [0.0, 0.0] |
| All models | protocol_failure | 6 | 0.00161377 | [0.0005379236148466918, 0.0029585798816568055] |
| Qwen/Qwen3.8-27B-FP8 | violation | 36 | 0.0152156 | [0.009721048182586641, 0.021978021978021983] |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | 36 | 0.0152156 | [0.009721048182586641, 0.021978021978021983] |
| Qwen/Qwen3.8-27B-FP8 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | protocol_failure | 6 | 0.00253593 | [0.0008453085376162299, 0.0046491969568892635] |
| Qwen/Qwen3.5-27B-FP8 | violation | 7 | 0.00517751 | [0.0014792899408284023, 0.009615384615384616] |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | 7 | 0.00517751 | [0.0014792899408284023, 0.009615384615384616] |
| Qwen/Qwen3.5-27B-FP8 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | protocol_failure | 0 | 0 | [0.0, 0.0] |

| Model | Experiment | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI95 |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | violation | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | leak | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | final_reply_length |  | 1029.58 | [971.5377958579882, 1101.9988165680472] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | explicit | none | 338 | violation_sensitivity | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | violation | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | attempt | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | wrong_attempt | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | final_reply_length |  | 873.979 | [827.31974852071, 926.5281065088757] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | False | plain | none | 338 | violation_sensitivity | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | violation | 2 | 0.00591716 | [0.0, 0.01775147928994083] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | attempt | 2 | 0.00591716 | [0.0, 0.01775147928994083] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | wrong_attempt | 2 | 0.00591716 | [0.0, 0.01775147928994083] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | final_reply_length |  | 935.976 | [911.0852071005917, 961.4733727810651] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | explicit | none | 338 | violation_sensitivity | 2 | 0.00591716 | [0.0, 0.01775147928994083] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | violation | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | final_reply_length |  | 811.66 | [787.4451183431953, 837.5431952662723] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | main | True | plain | none | 338 | violation_sensitivity | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | violation | 23 | 0.0680473 | [0.04437869822485207, 0.09467455621301775] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | leak | 22 | 0.0650888 | [0.04142011834319527, 0.09171597633136094] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | attempt | 9 | 0.0266272 | [0.011834319526627219, 0.04437869822485207] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | correct_attempt | 8 | 0.0236686 | [0.008875739644970414, 0.038461538461538464] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | wrong_attempt | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | empty_final | 12 | 0.035503 | [0.01775147928994083, 0.05621301775147929] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | truncated_final | 6 | 0.0177515 | [0.005917159763313609, 0.03254437869822485] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | empty_or_truncated_final | 12 | 0.035503 | [0.01775147928994083, 0.05621301775147929] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | final_reply_length |  | 852.849 | [815.4961538461538, 889.2239644970414] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | cap_hit | 6 | 0.0177515 | [0.005917159763313609, 0.03254437869822485] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | no_final | 12 | 0.035503 | [0.01775147928994083, 0.05621301775147929] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | cap_hit_no_final | 6 | 0.0177515 | [0.005917159763313609, 0.03254437869822485] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | protocol_failure | 6 | 0.0177515 | [0.005917159763313609, 0.03254437869822485] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | irrelevant | 338 | violation_sensitivity | 23 | 0.0680473 | [0.04437869822485207, 0.09467455621301775] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | violation | 6 | 0.0177515 | [0.0029585798816568047, 0.03550295857988166] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | leak | 4 | 0.0118343 | [0.0029585798816568047, 0.023668639053254437] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | attempt | 3 | 0.00887574 | [0.0, 0.023668639053254437] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | correct_attempt | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | wrong_attempt | 2 | 0.00591716 | [0.0, 0.01775147928994083] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | final_reply_length |  | 1078.96 | [1047.9334319526627, 1110.8013313609467] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | neutral | 338 | violation_sensitivity | 6 | 0.0177515 | [0.0029585798816568047, 0.03550295857988166] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | violation | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | attempt | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | wrong_attempt | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | final_reply_length |  | 914.441 | [890.532100591716, 941.2551775147929] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.8-27B-FP8 | prefill | True | explicit | reminder | 338 | violation_sensitivity | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | violation | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | attempt | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | wrong_attempt | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | final_reply_length |  | 1239.04 | [1192.9015532544379, 1289.9350591715977] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | explicit | none | 338 | violation_sensitivity | 3 | 0.00887574 | [0.0, 0.020710059171597635] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | violation | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | leak | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | final_reply_length |  | 1052.84 | [998.7270710059172, 1110.844452662722] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | False | plain | none | 338 | violation_sensitivity | 1 | 0.00295858 | [0.0, 0.008875739644970414] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | violation | 3 | 0.00887574 | [0.0, 0.023668639053254437] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | attempt | 3 | 0.00887574 | [0.0, 0.023668639053254437] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | wrong_attempt | 3 | 0.00887574 | [0.0, 0.023668639053254437] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | final_reply_length |  | 1059.55 | [1026.362056213018, 1094.196449704142] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | explicit | none | 338 | violation_sensitivity | 3 | 0.00887574 | [0.0, 0.023668639053254437] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | violation | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | leak | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | correct_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | wrong_attempt | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | empty_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | final_reply_length |  | 900.257 | [871.4468934911242, 931.0780325443786] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | cap_hit | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | protocol_failure | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | off_think_close | 0 | 0 | [0.0, 0.0] |
| Qwen/Qwen3.5-27B-FP8 | main | True | plain | none | 338 | violation_sensitivity | 0 | 0 | [0.0, 0.0] |

| Model | Outcome | Contrast | Estimate | CI95 | Holds |
| --- | --- | --- | ---: | --- | --- |
| Qwen/Qwen3.8-27B-FP8 | violation | H22 | 0.0118343 | [-0.0029585798816568047, 0.029585798816568046] | False |
| Qwen/Qwen3.8-27B-FP8 | violation | on_plain_minus_off_plain | -0.00887574 | [-0.020710059171597635, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_empty_minus_reminder | 0.00295858 | [0.0, 0.008875739644970414] | False |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_neutral_minus_reminder | 0.0147929 | [0.0029585798816568047, 0.026627218934911243] | True |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_irrelevant_minus_reminder | 0.0650888 | [0.038461538461538464, 0.09467455621301775] | True |
| Qwen/Qwen3.8-27B-FP8 | violation | H23_joint | NA |  | False |
| Qwen/Qwen3.8-27B-FP8 | leak | H22 | -0.00295858 | [-0.008875739644970414, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | on_plain_minus_off_plain | 0 | [0.0, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_empty_minus_reminder | 0 | [0.0, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_neutral_minus_reminder | 0.0118343 | [0.0029585798816568047, 0.023668639053254437] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_irrelevant_minus_reminder | 0.0650888 | [0.04142011834319527, 0.09171597633136094] | None |
| Qwen/Qwen3.8-27B-FP8 | leak | H23_joint | NA |  | None |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H22 | 0.0118343 | [-0.0029585798816568047, 0.029585798816568046] | False |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | on_plain_minus_off_plain | -0.00887574 | [-0.020710059171597635, 0.0] | None |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_empty_minus_reminder | 0.00295858 | [0.0, 0.008875739644970414] | False |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_neutral_minus_reminder | 0.0147929 | [0.0029585798816568047, 0.026627218934911243] | True |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_irrelevant_minus_reminder | 0.0650888 | [0.038461538461538464, 0.09467455621301775] | True |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | H23_joint | NA |  | False |
| Qwen/Qwen3.5-27B-FP8 | violation | H24 | 0.00295858 | [-0.014792899408284023, 0.020710059171597635] | False |
| Qwen/Qwen3.5-27B-FP8 | violation | on_plain_minus_off_plain | -0.00295858 | [-0.008875739644970414, 0.0] | None |
| Qwen/Qwen3.5-27B-FP8 | leak | H24 | 0.00295858 | [0.0, 0.008875739644970414] | None |
| Qwen/Qwen3.5-27B-FP8 | leak | on_plain_minus_off_plain | -0.00295858 | [-0.008875739644970414, 0.0] | None |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | H24 | 0.00295858 | [-0.014792899408284023, 0.020710059171597635] | False |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | on_plain_minus_off_plain | -0.00295858 | [-0.008875739644970414, 0.0] | None |

| H25 model | Status | R | L | N | Keyword mentions | N minus R | CI95 | Undefined resamples | Holds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Qwen/Qwen3.8-27B-FP8 | insufficient_R_or_N | 311 | 27 | 0 | 338 | NA | [None, None] | None | None |
| Qwen/Qwen3.5-27B-FP8 | insufficient_R_or_N | 338 | 0 | 0 | 330 | NA | [None, None] | None | None |
| Qwen/Qwen3-32B | insufficient_R_or_N | 1 | 5 | 332 | 2 | NA | [None, None] | None | None |
| google/gemma-4-12B-it | insufficient_R_or_N | 311 | 27 | 0 | 265 | NA | [None, None] | None | None |

| H25 sensitivity model | Outcome | Estimate | CI95 | Holds |
| --- | --- | ---: | --- | --- |
| Qwen/Qwen3.8-27B-FP8 | violation_sensitivity | NA | [None, None] | None |
| Qwen/Qwen3.5-27B-FP8 | violation_sensitivity | NA | [None, None] | None |
| Qwen/Qwen3-32B | violation_sensitivity | NA | [None, None] | None |
| google/gemma-4-12B-it | violation_sensitivity | NA | [None, None] | None |

| Trace coding | Value |
| --- | --- |
| Judges | gpt-6.1-sol / claude-haiku-5-5; no human labels |
| Seeded agreement sample | 100/100; seed 0 |
| Agreement | 0.86 |
| Cohen kappa | 0.723975 |

| Execution | Value |
| --- | --- |
| Cap USD | 5.0 |
| Conservative spend USD | 2.34983 |
| Provider credit delta USD, provisional | NA |
| Zero Phase 6 instances verified | True |
| Account-wide zero Vast instances verified | False |
| Other instances listed at resource check | [{"id": 57125990, "label": "grader-explore", "actual_status": "running"}] |
| Resource check Unix time | 1792625826.920087 |
| Account credit delta attributable to Phase 6 | False |

| Phase 5 keyword continuity | Mentions | N |
| --- | ---: | ---: |
| Qwen/Qwen3-14B | 11 | 400 |
| Qwen/Qwen3-32B | 3 | 400 |
| google/gemma-4-12B-it | 309 | 400 |

## Provenance

| Field | Value |
| --- | --- |
| run_id | "phase6-main" |
| phase | "phase6" |
| registration_commit | "8af01567fd54252e599b1df7a13cb3c84af96c2c" |
| source_sha256 | {"DESIGN.md": "f1d2d05185c1593deba90a7b58601691d18b99ba047e02866d06abca10c90c84", "prompts/system_tutor.txt": "6011cd11fc5a8ac78a4b90e35b7f5154db64467a2854a15f3cdee4ea8be88d11", "prompts/templates.json": "637fb8aa1312f4485d410241fe6087cbbe9c6150a4267bfb8a71e418f7af0ed1", "prompts/templates_4.json": "6cd003a8b8bf96456428084a1abeedb5c66ee5774443bcb6fdf31d7d65a8fd31"} |
| implementation_sha256 | {"src/build_prompts.py": "20b371f02a4efb3645501760bf1152b1adc87d8fe04a0b25dece5164a58a9943", "src/models_6.py": "8f4773628b797a8cc74ac9666d41640ef61b9c39e53210aa31483a1350200f36", "src/vast_generate_4.py": "96c6313c21b4193405b6d209623f1e3d129204cae64281e246ee5f77f0988377", "src/vast_generate_6.py": "24606a2cc03e2fc91d555969e3c3b0792c89bbac49b953836d39f608d354e8a3"} |
| models | {"Qwen/Qwen3.5-27B-FP8": {"dtype": "bfloat16", "max_tokens": 4096, "model_revision": "97f5941bf617e31c5e237364a8602ce3f03a551a", "package_versions": {"jinja2": "3.1.6", "tokenizers": "0.23.3", "torch": "2.13.0+cu130", "transformers": "5.19.0", "vllm": "0.32.0"}, "pins": {"file_sha256": {"chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "config.json": "53f7e06eebbfe64e70eb0b0979f3da31de9aff6870594e9cf731ba3961501adc", "generation_config.json": "303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd", "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42", "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"}, "model": "Qwen/Qwen3.5-27B-FP8", "reasoning_effort": null, "revision": "97f5941bf617e31c5e237364a8602ce3f03a551a"}, "quantization": "fp8", "reasoning_effort": null, "sampling": {"max_tokens": 4096, "n": 1, "skip_special_tokens": false, "stop_token_ids": [248046, 248044], "temperature": 0.7, "top_k": -1, "top_p": 0.95}, "settings": {"chunk_size": 256, "engine": {"dtype": "bfloat16", "enforce_eager": false, "generation_config": "vllm", "language_model_only": true, "max_model_len": 8192, "max_num_seqs": 256, "model": "Qwen/Qwen3.5-27B-FP8", "quantization": "fp8", "revision": "97f5941bf617e31c5e237364a8602ce3f03a551a", "seed": 0, "tensor_parallel_size": 1, "tokenizer_revision": "97f5941bf617e31c5e237364a8602ce3f03a551a"}, "native_template": {"preserve_thinking": false, "reasoning_effort": null}, "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]", "sampling": {"max_tokens": 4096, "temperature": 0.7, "top_k": -1, "top_p": 0.95}, "setup_seconds": 686.283938987006, "versions": {"jinja2": "3.1.6", "tokenizers": "0.23.3", "torch": "2.13.0+cu130", "transformers": "5.19.0", "vllm": "0.32.0"}}, "temperature": 0.7, "tokenizer_revision": "97f5941bf617e31c5e237364a8602ce3f03a551a", "top_k": -1, "top_p": 0.95, "vllm_version": "0.32.0"}, "Qwen/Qwen3.8-27B-FP8": {"dtype": "bfloat16", "max_tokens": 4096, "model_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", "package_versions": {"jinja2": "3.1.6", "tokenizers": "0.23.3", "torch": "2.13.0+cu130", "transformers": "5.19.0", "vllm": "0.32.0"}, "pins": {"file_sha256": {"chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "config.json": "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc", "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e", "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3", "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"}, "model": "Qwen/Qwen3.8-27B-FP8", "reasoning_effort": "medium", "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"}, "quantization": "fp8", "reasoning_effort": "medium", "sampling": {"max_tokens": 4096, "n": 1, "skip_special_tokens": false, "stop_token_ids": [248046, 248044], "temperature": 0.7, "top_k": -1, "top_p": 0.95}, "settings": {"chunk_size": 256, "engine": {"dtype": "bfloat16", "enforce_eager": false, "generation_config": "vllm", "language_model_only": true, "max_model_len": 8192, "max_num_seqs": 256, "model": "Qwen/Qwen3.8-27B-FP8", "quantization": "fp8", "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", "seed": 0, "tensor_parallel_size": 1, "tokenizer_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"}, "native_template": {"preserve_thinking": false, "reasoning_effort": "medium"}, "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]", "sampling": {"max_tokens": 4096, "temperature": 0.7, "top_k": -1, "top_p": 0.95}, "setup_seconds": 616.5063963840075, "versions": {"jinja2": "3.1.6", "tokenizers": "0.23.3", "torch": "2.13.0+cu130", "transformers": "5.19.0", "vllm": "0.32.0"}}, "temperature": 0.7, "tokenizer_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", "top_k": -1, "top_p": 0.95, "vllm_version": "0.32.0"}} |
| seed_pairing | "SHA256 phase4:pNNNN:tT, first 4 bytes" |
