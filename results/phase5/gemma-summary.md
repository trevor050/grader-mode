# Phase 5: google/gemma-4-12B-it

| Collection | Value |
| --- | --- |
| Responses | 2400 |
| Problems | 200 |
| Grid | cell_drop_stage_1 |
| Inference eligible | True |
| Pending reasons | none |
| Bootstrap | 2,000 paired problem clusters; seed 0; percentile 95% CI |
| Denominator | All requests, including cap-hit and empty finals |
| Audit author | Codex, self-labeled blind to arm at the user's explicit request; no independent human labels |

| Audit | Labeled / required | Agreement | Disagreements |
| --- | ---: | ---: | ---: |
| Attempt | 40 / 40 | 0.9 | 4 |
| Committed number | 0 / 0 | not applicable (0 eligible) | 0 |
| Final boundary | 40 / 40 | 1.0 | 0 |

| Dropped cell |
| --- |
| prefill / on / explicit / neutral |
| prefill / on / explicit / irrelevant |

## All gold

| Problems | Replies |
| ---: | ---: |
| 200 | 2400 |

| Arm | Thinking | Request | Prefill | N | Component | Count | Estimate [95% CI] |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| main | off | explicit | none | 400 | violation | 12 | 0.0300 [0.0125, 0.0525] |
| main | off | explicit | none | 400 | leak | 10 | 0.0250 [0.0075, 0.0450] |
| main | off | explicit | none | 400 | attempt | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | wrong_attempt | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | final_reply_length |  | 629.8975 [598.2974, 662.2124] |
| main | off | explicit | none | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | violation | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | plain | none | 400 | leak | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | plain | none | 400 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | final_reply_length |  | 420.3850 [403.8646, 438.1101] |
| main | off | plain | none | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | violation | 6 | 0.0150 [0.0050, 0.0275] |
| main | on | explicit | none | 400 | leak | 6 | 0.0150 [0.0050, 0.0275] |
| main | on | explicit | none | 400 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | final_reply_length |  | 489.8725 [471.5219, 509.5317] |
| main | on | explicit | none | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | violation | 6 | 0.0150 [0.0025, 0.0300] |
| main | on | plain | none | 400 | leak | 5 | 0.0125 [0.0025, 0.0275] |
| main | on | plain | none | 400 | attempt | 1 | 0.0025 [0.0000, 0.0075] |
| main | on | plain | none | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | wrong_attempt | 1 | 0.0025 [0.0000, 0.0075] |
| main | on | plain | none | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | final_reply_length |  | 357.9150 [346.4597, 369.3991] |
| main | on | plain | none | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | violation | 5 | 0.0125 [0.0025, 0.0250] |
| prefill | on | explicit | empty | 400 | leak | 4 | 0.0100 [0.0025, 0.0200] |
| prefill | on | explicit | empty | 400 | attempt | 1 | 0.0025 [0.0000, 0.0075] |
| prefill | on | explicit | empty | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | wrong_attempt | 1 | 0.0025 [0.0000, 0.0075] |
| prefill | on | explicit | empty | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | final_reply_length |  | 491.3450 [471.2371, 513.1635] |
| prefill | on | explicit | empty | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | violation | 6 | 0.0150 [0.0025, 0.0325] |
| prefill | on | explicit | reminder | 400 | leak | 6 | 0.0150 [0.0025, 0.0325] |
| prefill | on | explicit | reminder | 400 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | final_reply_length |  | 431.3575 [413.6212, 449.4184] |
| prefill | on | explicit | reminder | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |

| Outcome | Contrast | Estimate [95% CI] | Tested | Supported |
| --- | --- | --- | --- | --- |
| violation | H18 | -0.0250 [-0.0450, -0.0075] | True | False |
| violation | on_plain_minus_off_plain | 0.0100 [0.0000, 0.0225] | False | None |
| violation | H19_empty_minus_reminder | -0.0025 [-0.0175, 0.0100] | True | False |
| violation | H19_neutral_minus_reminder | unavailable | False | None |
| violation | H19_irrelevant_minus_reminder | unavailable | False | None |
| violation | H19_joint | unavailable | False | None |
| violation | H19_reduced | -0.0025 [-0.0175, 0.0100] | True | False |
| leak | H18 | -0.0175 [-0.0350, -0.0050] | False | None |
| leak | on_plain_minus_off_plain | 0.0075 [-0.0025, 0.0200] | False | None |
| leak | H19_empty_minus_reminder | -0.0050 [-0.0200, 0.0075] | False | None |
| leak | H19_neutral_minus_reminder | unavailable | False | None |
| leak | H19_irrelevant_minus_reminder | unavailable | False | None |
| leak | H19_joint | unavailable | False | None |
| leak | H19_reduced | -0.0050 [-0.0200, 0.0075] | False | None |

## Gold >= 10

| Problems | Replies |
| ---: | ---: |
| 169 | 2028 |

| Arm | Thinking | Request | Prefill | N | Component | Count | Estimate [95% CI] |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| main | off | explicit | none | 338 | violation | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | explicit | none | 338 | leak | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | attempt | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | explicit | none | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | wrong_attempt | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | explicit | none | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | final_reply_length |  | 626.3639 [592.4748, 662.1486] |
| main | off | explicit | none | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | explicit | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | violation | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | leak | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | final_reply_length |  | 415.1391 [398.4316, 432.5800] |
| main | off | plain | none | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | violation | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | leak | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | final_reply_length |  | 486.3432 [466.2146, 507.6339] |
| main | on | explicit | none | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | violation | 1 | 0.0030 [0.0000, 0.0089] |
| main | on | plain | none | 338 | leak | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | attempt | 1 | 0.0030 [0.0000, 0.0089] |
| main | on | plain | none | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | wrong_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| main | on | plain | none | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | final_reply_length |  | 353.9201 [342.0289, 365.5309] |
| main | on | plain | none | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | violation | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | empty | 338 | leak | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | attempt | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | empty | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | wrong_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | empty | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | final_reply_length |  | 496.5858 [475.2212, 519.9321] |
| prefill | on | explicit | empty | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | violation | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | leak | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | final_reply_length |  | 428.6982 [410.9490, 446.8688] |
| prefill | on | explicit | reminder | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |

| Outcome | Contrast | Estimate [95% CI] | Tested | Supported |
| --- | --- | --- | --- | --- |
| violation | H18 | -0.0089 [-0.0207, 0.0000] | True | False |
| violation | on_plain_minus_off_plain | 0.0030 [0.0000, 0.0089] | False | None |
| violation | H19_empty_minus_reminder | 0.0030 [0.0000, 0.0089] | True | False |
| violation | H19_neutral_minus_reminder | unavailable | False | None |
| violation | H19_irrelevant_minus_reminder | unavailable | False | None |
| violation | H19_joint | unavailable | False | None |
| violation | H19_reduced | 0.0030 [0.0000, 0.0089] | True | False |
| leak | H18 | 0.0000 [0.0000, 0.0000] | False | None |
| leak | on_plain_minus_off_plain | 0.0000 [0.0000, 0.0000] | False | None |
| leak | H19_empty_minus_reminder | 0.0000 [0.0000, 0.0000] | False | None |
| leak | H19_neutral_minus_reminder | unavailable | False | None |
| leak | H19_irrelevant_minus_reminder | unavailable | False | None |
| leak | H19_joint | unavailable | False | None |
| leak | H19_reduced | 0.0000 [0.0000, 0.0000] | False | None |

H18: not supported, -0.0250 [-0.0450, -0.0075].

H19: not supported, empty minus reminder -0.0025 [-0.0175, 0.0100]; registered reduced contrast.
