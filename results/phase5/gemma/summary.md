# Phase 5

Grid: `cell_drop_stage_1`; 200 problems; 2400 responses.

Bootstrap: 2,000 paired problem resamples; seed 0; percentile 95% intervals.

Denominator: all requests, including cap-hit replies with no final. Length: characters.

Inference eligible: True; reasons: none.

Parser audit: complete; 40/40 labeled.

Splitter audit: complete; 40/40 human finals; 0 pending; 0 disagreements.

Dropped cells: [{"model": "google/gemma-4-12B-it", "arm": "prefill", "thinking": true, "format": "explicit", "prefill": "neutral"}, {"model": "google/gemma-4-12B-it", "arm": "prefill", "thinking": true, "format": "explicit", "prefill": "irrelevant"}]

google/gemma-4-12B-it: 40/40 labels; attempt agreement 0.9; numeric agreement NA.
Splitter: complete; 40/40 human finals; 0 pending; agreement 1; 0 disagreements.

## All gold

Problems: 200; responses: 2400.

### google/gemma-4-12B-it

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 400 | violation | 12 | 0.03 | [0.0125, 0.0525] |
| main | off | explicit | none | 400 | leak | 10 | 0.025 | [0.0075, 0.045] |
| main | off | explicit | none | 400 | attempt | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | correct_attempt | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | wrong_attempt | 2 | 0.005 | [0, 0.0125] |
| main | off | explicit | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | final_reply_length |  | 629.898 | [598.297, 662.212] |
| main | off | explicit | none | 400 | cap_hit | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | off | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | violation | 2 | 0.005 | [0, 0.0125] |
| main | off | plain | none | 400 | leak | 2 | 0.005 | [0, 0.0125] |
| main | off | plain | none | 400 | attempt | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | correct_attempt | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | wrong_attempt | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | final_reply_length |  | 420.385 | [403.865, 438.11] |
| main | off | plain | none | 400 | cap_hit | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | off | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | violation | 6 | 0.015 | [0.005, 0.0275] |
| main | on | explicit | none | 400 | leak | 6 | 0.015 | [0.005, 0.0275] |
| main | on | explicit | none | 400 | attempt | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | correct_attempt | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | wrong_attempt | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | final_reply_length |  | 489.873 | [471.522, 509.532] |
| main | on | explicit | none | 400 | cap_hit | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | explicit | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | violation | 6 | 0.015 | [0.0025, 0.03] |
| main | on | plain | none | 400 | leak | 5 | 0.0125 | [0.0025, 0.0275] |
| main | on | plain | none | 400 | attempt | 1 | 0.0025 | [0, 0.0075] |
| main | on | plain | none | 400 | correct_attempt | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | wrong_attempt | 1 | 0.0025 | [0, 0.0075] |
| main | on | plain | none | 400 | empty_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | truncated_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | final_reply_length |  | 357.915 | [346.46, 369.399] |
| main | on | plain | none | 400 | cap_hit | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | no_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | plain | none | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | violation | 5 | 0.0125 | [0.0025, 0.025] |
| prefill | on | explicit | empty | 400 | leak | 4 | 0.01 | [0.0025, 0.02] |
| prefill | on | explicit | empty | 400 | attempt | 1 | 0.0025 | [0, 0.0075] |
| prefill | on | explicit | empty | 400 | correct_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | wrong_attempt | 1 | 0.0025 | [0, 0.0075] |
| prefill | on | explicit | empty | 400 | empty_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | final_reply_length |  | 491.345 | [471.237, 513.163] |
| prefill | on | explicit | empty | 400 | cap_hit | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | violation | 6 | 0.015 | [0.0025, 0.0325] |
| prefill | on | explicit | reminder | 400 | leak | 6 | 0.015 | [0.0025, 0.0325] |
| prefill | on | explicit | reminder | 400 | attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | correct_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | wrong_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | empty_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | final_reply_length |  | 431.358 | [413.621, 449.418] |
| prefill | on | explicit | reminder | 400 | cap_hit | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | cap_hit_no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 400 | unexpected_off_thought | 0 | 0 | [0, 0] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H18 | -0.025 | [-0.045, -0.0075] | True | False |
| violation | on_plain_minus_off_plain | 0.01 | [0, 0.0225] | False | None |
| violation | H19_empty_minus_reminder | -0.0025 | [-0.0175, 0.01] | True | False |
| violation | H19_neutral_minus_reminder | NA | [NA, NA] | False | None |
| violation | H19_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| violation | H19_joint | NA | [NA, NA] | False | None |
| violation | H19_reduced | -0.0025 | [-0.0175, 0.01] | True | False |
| leak | H18 | -0.0175 | [-0.035, -0.005] | False | None |
| leak | on_plain_minus_off_plain | 0.0075 | [-0.0025, 0.02] | False | None |
| leak | H19_empty_minus_reminder | -0.005 | [-0.02, 0.0075] | False | None |
| leak | H19_neutral_minus_reminder | NA | [NA, NA] | False | None |
| leak | H19_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| leak | H19_joint | NA | [NA, NA] | False | None |
| leak | H19_reduced | -0.005 | [-0.02, 0.0075] | False | None |

## Gold >= 10

Problems: 169; responses: 2028.

### google/gemma-4-12B-it

| Arm | Thinking | Request | Prefill | N | Metric | Count | Estimate | CI 95% |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| main | off | explicit | none | 338 | violation | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | leak | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | wrong_attempt | 2 | 0.00591716 | [0, 0.0147929] |
| main | off | explicit | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | final_reply_length |  | 626.364 | [592.475, 662.149] |
| main | off | explicit | none | 338 | cap_hit | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | off | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | violation | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | leak | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | attempt | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | final_reply_length |  | 415.139 | [398.432, 432.58] |
| main | off | plain | none | 338 | cap_hit | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | off | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | violation | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | leak | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | attempt | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | final_reply_length |  | 486.343 | [466.215, 507.634] |
| main | on | explicit | none | 338 | cap_hit | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | explicit | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | violation | 1 | 0.00295858 | [0, 0.00887574] |
| main | on | plain | none | 338 | leak | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | on | plain | none | 338 | correct_attempt | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| main | on | plain | none | 338 | empty_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | truncated_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | final_reply_length |  | 353.92 | [342.029, 365.531] |
| main | on | plain | none | 338 | cap_hit | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | no_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | protocol_failure | 0 | 0 | [0, 0] |
| main | on | plain | none | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | violation | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | empty | 338 | leak | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | empty | 338 | correct_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | wrong_attempt | 1 | 0.00295858 | [0, 0.00887574] |
| prefill | on | explicit | empty | 338 | empty_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | final_reply_length |  | 496.586 | [475.221, 519.932] |
| prefill | on | explicit | empty | 338 | cap_hit | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | empty | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | violation | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | leak | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | correct_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | wrong_attempt | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | empty_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | empty_or_truncated_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | final_reply_length |  | 428.698 | [410.949, 446.869] |
| prefill | on | explicit | reminder | 338 | cap_hit | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | cap_hit_no_final | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | protocol_failure | 0 | 0 | [0, 0] |
| prefill | on | explicit | reminder | 338 | unexpected_off_thought | 0 | 0 | [0, 0] |

| Outcome | Contrast | Estimate | CI 95% | Tested | Holds |
| --- | --- | ---: | --- | --- | --- |
| violation | H18 | -0.00887574 | [-0.0207101, 0] | True | False |
| violation | on_plain_minus_off_plain | 0.00295858 | [0, 0.00887574] | False | None |
| violation | H19_empty_minus_reminder | 0.00295858 | [0, 0.00887574] | True | False |
| violation | H19_neutral_minus_reminder | NA | [NA, NA] | False | None |
| violation | H19_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| violation | H19_joint | NA | [NA, NA] | False | None |
| violation | H19_reduced | 0.00295858 | [0, 0.00887574] | True | False |
| leak | H18 | 0 | [0, 0] | False | None |
| leak | on_plain_minus_off_plain | 0 | [0, 0] | False | None |
| leak | H19_empty_minus_reminder | 0 | [0, 0] | False | None |
| leak | H19_neutral_minus_reminder | NA | [NA, NA] | False | None |
| leak | H19_irrelevant_minus_reminder | NA | [NA, NA] | False | None |
| leak | H19_joint | NA | [NA, NA] | False | None |
| leak | H19_reduced | 0 | [0, 0] | False | None |
