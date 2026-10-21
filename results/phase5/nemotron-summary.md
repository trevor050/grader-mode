# Phase 5 Nemotron: full grid

| Collection / method | Value |
| --- | --- |
| Model | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 |
| Main grid | 3,200 replies; 200 problems × two templates × eight complete cells |
| Inference eligible | True |
| Deferred / dropped cells | 0 / 0 |
| Main generation | Pinned NVFP4; Modal RTX-PRO-6000; CUDA graphs; 256 sequences/chunks; max_tokens 4,096 |
| Excluded unscored prompts | Prior smoke 8 + throughput 256; continuation smoke 8 |
| Denominator | All requests, including cap-hit and empty finals |
| Bootstrap | 2,000 paired problem clusters; seed 0; percentile 95% CI; no multiplicity adjustment |
| Protocol deviation | 8219501; registered after inspecting two off-mode replies, before hypothesis decisions |
| Off-mode scoring | Final text after last literal `</think>`; original protocol flags retained; gate lifted |
| Protocol sensitivity | Both affected off-mode replies counted as violations; other outcomes unchanged |
| Audit author | Codex, self-labeled blind to arm at the user's explicit request; no independent human labels |
| Audit method | Fresh seed-0 full-grid sample, manually labeled before full-grid analysis |
| Detailed analysis | [JSON](nemotron-full/summary.json) |
| Execution / accounting | [Execution](nemotron-execution-controls-report.md); [accounting](final-accounting.json) |

| Audit | Labeled / required | Agreements | Agreement | Disagreements |
| --- | --- | --- | --- | --- |
| Attempt | 40 / 40 | 32 | 80.00% | 8 |
| Committed number | 11 / 11 | 10 | 90.91% | 1 |
| Final boundary | 40 / 40 | 40 | 100.00% | 0 |

## All gold: baseline components

| Cell | N | Component | Count | Estimate [95% CI] |
| --- | --- | --- | --- | --- |
| main/off/explicit | 400 | violation | 172 | 43.00% [37.75, 48.00]% |
| main/off/explicit | 400 | leak | 169 | 42.25% [37.25, 47.25]% |
| main/off/explicit | 400 | attempt | 63 | 15.75% [12.50, 19.50]% |
| main/off/explicit | 400 | correct_attempt | 57 | 14.25% [11.00, 17.75]% |
| main/off/explicit | 400 | wrong_attempt | 6 | 1.50% [0.50, 2.75]% |
| main/off/explicit | 400 | empty_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/explicit | 400 | truncated_final | 2 | 0.50% [0.00, 1.25]% |
| main/off/explicit | 400 | empty_or_truncated_final | 2 | 0.50% [0.00, 1.25]% |
| main/off/explicit | 400 | final_reply_length (characters) | — | 941.64 [867.99, 1039.68] |
| main/off/explicit | 400 | cap_hit | 2 | 0.50% [0.00, 1.25]% |
| main/off/explicit | 400 | no_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/explicit | 400 | cap_hit_no_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/explicit | 400 | protocol_failure | 2 | 0.50% [0.00, 1.25]% |
| main/off/explicit | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | violation | 48 | 12.00% [8.50, 15.75]% |
| main/off/plain | 400 | leak | 47 | 11.75% [8.25, 15.51]% |
| main/off/plain | 400 | attempt | 3 | 0.75% [0.00, 1.75]% |
| main/off/plain | 400 | correct_attempt | 2 | 0.50% [0.00, 1.25]% |
| main/off/plain | 400 | wrong_attempt | 1 | 0.25% [0.00, 0.75]% |
| main/off/plain | 400 | empty_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | truncated_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | empty_or_truncated_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | final_reply_length (characters) | — | 756.74 [714.27, 799.75] |
| main/off/plain | 400 | cap_hit | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | no_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | cap_hit_no_final | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| main/off/plain | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| main/on/explicit | 400 | violation | 34 | 8.50% [5.25, 12.00]% |
| main/on/explicit | 400 | leak | 33 | 8.25% [5.25, 11.75]% |
| main/on/explicit | 400 | attempt | 4 | 1.00% [0.25, 2.00]% |
| main/on/explicit | 400 | correct_attempt | 2 | 0.50% [0.00, 1.25]% |
| main/on/explicit | 400 | wrong_attempt | 2 | 0.50% [0.00, 1.25]% |
| main/on/explicit | 400 | empty_final | 8 | 2.00% [0.75, 3.75]% |
| main/on/explicit | 400 | truncated_final | 11 | 2.75% [1.00, 5.00]% |
| main/on/explicit | 400 | empty_or_truncated_final | 11 | 2.75% [1.00, 5.00]% |
| main/on/explicit | 400 | final_reply_length (characters) | — | 863.64 [827.64, 899.00] |
| main/on/explicit | 400 | cap_hit | 11 | 2.75% [1.00, 5.00]% |
| main/on/explicit | 400 | no_final | 8 | 2.00% [0.75, 3.75]% |
| main/on/explicit | 400 | cap_hit_no_final | 8 | 2.00% [0.75, 3.75]% |
| main/on/explicit | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| main/on/explicit | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| main/on/plain | 400 | violation | 17 | 4.25% [2.00, 6.75]% |
| main/on/plain | 400 | leak | 16 | 4.00% [1.99, 6.50]% |
| main/on/plain | 400 | attempt | 1 | 0.25% [0.00, 0.75]% |
| main/on/plain | 400 | correct_attempt | 0 | 0.00% [0.00, 0.00]% |
| main/on/plain | 400 | wrong_attempt | 1 | 0.25% [0.00, 0.75]% |
| main/on/plain | 400 | empty_final | 5 | 1.25% [0.25, 2.75]% |
| main/on/plain | 400 | truncated_final | 6 | 1.50% [0.25, 3.00]% |
| main/on/plain | 400 | empty_or_truncated_final | 6 | 1.50% [0.25, 3.00]% |
| main/on/plain | 400 | final_reply_length (characters) | — | 736.32 [703.11, 770.52] |
| main/on/plain | 400 | cap_hit | 6 | 1.50% [0.25, 3.00]% |
| main/on/plain | 400 | no_final | 5 | 1.25% [0.25, 2.75]% |
| main/on/plain | 400 | cap_hit_no_final | 5 | 1.25% [0.25, 2.75]% |
| main/on/plain | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| main/on/plain | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| prefill/empty | 400 | violation | 41 | 10.25% [6.75, 14.00]% |
| prefill/empty | 400 | leak | 40 | 10.00% [6.50, 13.75]% |
| prefill/empty | 400 | attempt | 2 | 0.50% [0.00, 1.25]% |
| prefill/empty | 400 | correct_attempt | 1 | 0.25% [0.00, 0.75]% |
| prefill/empty | 400 | wrong_attempt | 1 | 0.25% [0.00, 0.75]% |
| prefill/empty | 400 | empty_final | 7 | 1.75% [0.50, 3.25]% |
| prefill/empty | 400 | truncated_final | 8 | 2.00% [0.74, 3.75]% |
| prefill/empty | 400 | empty_or_truncated_final | 8 | 2.00% [0.74, 3.75]% |
| prefill/empty | 400 | final_reply_length (characters) | — | 886.88 [849.83, 924.22] |
| prefill/empty | 400 | cap_hit | 8 | 2.00% [0.74, 3.75]% |
| prefill/empty | 400 | no_final | 7 | 1.75% [0.50, 3.25]% |
| prefill/empty | 400 | cap_hit_no_final | 7 | 1.75% [0.50, 3.25]% |
| prefill/empty | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| prefill/empty | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| prefill/irrelevant | 400 | violation | 58 | 14.50% [10.50, 18.50]% |
| prefill/irrelevant | 400 | leak | 55 | 13.75% [10.00, 18.00]% |
| prefill/irrelevant | 400 | attempt | 5 | 1.25% [0.25, 2.50]% |
| prefill/irrelevant | 400 | correct_attempt | 2 | 0.50% [0.00, 1.25]% |
| prefill/irrelevant | 400 | wrong_attempt | 3 | 0.75% [0.00, 1.75]% |
| prefill/irrelevant | 400 | empty_final | 5 | 1.25% [0.25, 2.75]% |
| prefill/irrelevant | 400 | truncated_final | 5 | 1.25% [0.25, 2.75]% |
| prefill/irrelevant | 400 | empty_or_truncated_final | 5 | 1.25% [0.25, 2.75]% |
| prefill/irrelevant | 400 | final_reply_length (characters) | — | 1043.60 [999.00, 1085.57] |
| prefill/irrelevant | 400 | cap_hit | 5 | 1.25% [0.25, 2.75]% |
| prefill/irrelevant | 400 | no_final | 5 | 1.25% [0.25, 2.75]% |
| prefill/irrelevant | 400 | cap_hit_no_final | 5 | 1.25% [0.25, 2.75]% |
| prefill/irrelevant | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| prefill/irrelevant | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| prefill/neutral | 400 | violation | 30 | 7.50% [5.00, 10.50]% |
| prefill/neutral | 400 | leak | 27 | 6.75% [4.25, 9.75]% |
| prefill/neutral | 400 | attempt | 4 | 1.00% [0.25, 2.00]% |
| prefill/neutral | 400 | correct_attempt | 1 | 0.25% [0.00, 0.75]% |
| prefill/neutral | 400 | wrong_attempt | 3 | 0.75% [0.00, 1.75]% |
| prefill/neutral | 400 | empty_final | 34 | 8.50% [5.75, 11.75]% |
| prefill/neutral | 400 | truncated_final | 39 | 9.75% [6.75, 13.25]% |
| prefill/neutral | 400 | empty_or_truncated_final | 39 | 9.75% [6.75, 13.25]% |
| prefill/neutral | 400 | final_reply_length (characters) | — | 824.57 [778.45, 868.80] |
| prefill/neutral | 400 | cap_hit | 39 | 9.75% [6.75, 13.25]% |
| prefill/neutral | 400 | no_final | 34 | 8.50% [5.75, 11.75]% |
| prefill/neutral | 400 | cap_hit_no_final | 34 | 8.50% [5.75, 11.75]% |
| prefill/neutral | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| prefill/neutral | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |
| prefill/reminder | 400 | violation | 27 | 6.75% [4.00, 10.00]% |
| prefill/reminder | 400 | leak | 27 | 6.75% [4.00, 10.00]% |
| prefill/reminder | 400 | attempt | 0 | 0.00% [0.00, 0.00]% |
| prefill/reminder | 400 | correct_attempt | 0 | 0.00% [0.00, 0.00]% |
| prefill/reminder | 400 | wrong_attempt | 0 | 0.00% [0.00, 0.00]% |
| prefill/reminder | 400 | empty_final | 3 | 0.75% [0.00, 1.75]% |
| prefill/reminder | 400 | truncated_final | 6 | 1.50% [0.25, 3.01]% |
| prefill/reminder | 400 | empty_or_truncated_final | 6 | 1.50% [0.25, 3.01]% |
| prefill/reminder | 400 | final_reply_length (characters) | — | 857.96 [821.23, 896.64] |
| prefill/reminder | 400 | cap_hit | 6 | 1.50% [0.25, 3.01]% |
| prefill/reminder | 400 | no_final | 3 | 0.75% [0.00, 1.75]% |
| prefill/reminder | 400 | cap_hit_no_final | 3 | 0.75% [0.00, 1.75]% |
| prefill/reminder | 400 | protocol_failure | 0 | 0.00% [0.00, 0.00]% |
| prefill/reminder | 400 | unexpected_off_thought | 0 | 0.00% [0.00, 0.00]% |

## Sensitivity cell rates

| Scope | Problems | Cell | N | Violation count | Violation [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Baseline: gold ≥10 | 169 | main/off/explicit | 338 | 136 | 40.24% [34.90, 45.56]% |
| Baseline: gold ≥10 | 169 | main/off/plain | 338 | 29 | 8.58% [5.62, 11.83]% |
| Baseline: gold ≥10 | 169 | main/on/explicit | 338 | 12 | 3.55% [1.78, 5.62]% |
| Baseline: gold ≥10 | 169 | main/on/plain | 338 | 3 | 0.89% [0.00, 2.07]% |
| Baseline: gold ≥10 | 169 | prefill/empty | 338 | 15 | 4.44% [2.07, 7.10]% |
| Baseline: gold ≥10 | 169 | prefill/irrelevant | 338 | 29 | 8.58% [5.92, 11.55]% |
| Baseline: gold ≥10 | 169 | prefill/neutral | 338 | 16 | 4.73% [2.66, 7.10]% |
| Baseline: gold ≥10 | 169 | prefill/reminder | 338 | 10 | 2.96% [1.48, 5.03]% |
| Protocol sensitivity: all gold | 200 | main/off/explicit | 400 | 174 | 43.50% [38.25, 48.50]% |
| Protocol sensitivity: all gold | 200 | main/off/plain | 400 | 48 | 12.00% [8.50, 15.75]% |
| Protocol sensitivity: all gold | 200 | main/on/explicit | 400 | 34 | 8.50% [5.25, 12.00]% |
| Protocol sensitivity: all gold | 200 | main/on/plain | 400 | 17 | 4.25% [2.00, 6.75]% |
| Protocol sensitivity: all gold | 200 | prefill/empty | 400 | 41 | 10.25% [6.75, 14.00]% |
| Protocol sensitivity: all gold | 200 | prefill/irrelevant | 400 | 58 | 14.50% [10.50, 18.50]% |
| Protocol sensitivity: all gold | 200 | prefill/neutral | 400 | 30 | 7.50% [5.00, 10.50]% |
| Protocol sensitivity: all gold | 200 | prefill/reminder | 400 | 27 | 6.75% [4.00, 10.00]% |
| Protocol sensitivity: gold ≥10 | 169 | main/off/explicit | 338 | 137 | 40.53% [35.21, 45.86]% |
| Protocol sensitivity: gold ≥10 | 169 | main/off/plain | 338 | 29 | 8.58% [5.62, 11.83]% |
| Protocol sensitivity: gold ≥10 | 169 | main/on/explicit | 338 | 12 | 3.55% [1.78, 5.62]% |
| Protocol sensitivity: gold ≥10 | 169 | main/on/plain | 338 | 3 | 0.89% [0.00, 2.07]% |
| Protocol sensitivity: gold ≥10 | 169 | prefill/empty | 338 | 15 | 4.44% [2.07, 7.10]% |
| Protocol sensitivity: gold ≥10 | 169 | prefill/irrelevant | 338 | 29 | 8.58% [5.92, 11.55]% |
| Protocol sensitivity: gold ≥10 | 169 | prefill/neutral | 338 | 16 | 4.73% [2.66, 7.10]% |
| Protocol sensitivity: gold ≥10 | 169 | prefill/reminder | 338 | 10 | 2.96% [1.48, 5.03]% |

## Contrasts

| Scope | Outcome | Contrast | Estimate [95% CI], pp | Directional support |
| --- | --- | --- | --- | --- |
| Baseline: all gold | violation | H21 | -26.75 [-33.25, -20.50] | no |
| Baseline: all gold | violation | on_plain_minus_off_plain | -7.75 [-11.25, -4.25] | — |
| Baseline: all gold | violation | H20_empty_minus_reminder | +3.50 [+0.50, +6.50] | yes |
| Baseline: all gold | violation | H20_neutral_minus_reminder | +0.75 [-2.25, +3.75] | no |
| Baseline: all gold | violation | H20_irrelevant_minus_reminder | +7.75 [+4.50, +11.00] | yes |
| Baseline: all gold | leak | H21 | -26.25 [-32.50, -20.25] | — |
| Baseline: all gold | leak | on_plain_minus_off_plain | -7.75 [-11.25, -4.50] | — |
| Baseline: all gold | leak | H20_empty_minus_reminder | +3.25 [+0.25, +6.25] | — |
| Baseline: all gold | leak | H20_neutral_minus_reminder | +0.00 [-2.75, +2.75] | — |
| Baseline: all gold | leak | H20_irrelevant_minus_reminder | +7.00 [+3.99, +10.00] | — |
| Baseline: gold ≥10 | violation | H21 | -28.99 [-34.92, -23.08] | no |
| Baseline: gold ≥10 | violation | on_plain_minus_off_plain | -7.69 [-10.95, -4.73] | — |
| Baseline: gold ≥10 | violation | H20_empty_minus_reminder | +1.48 [-1.18, +4.14] | no |
| Baseline: gold ≥10 | violation | H20_neutral_minus_reminder | +1.78 [-0.89, +4.73] | no |
| Baseline: gold ≥10 | violation | H20_irrelevant_minus_reminder | +5.62 [+2.37, +9.17] | yes |
| Baseline: gold ≥10 | leak | H21 | -28.70 [-34.32, -22.78] | — |
| Baseline: gold ≥10 | leak | on_plain_minus_off_plain | -7.69 [-10.95, -4.73] | — |
| Baseline: gold ≥10 | leak | H20_empty_minus_reminder | +1.18 [-1.48, +3.85] | — |
| Baseline: gold ≥10 | leak | H20_neutral_minus_reminder | +1.18 [-1.48, +4.14] | — |
| Baseline: gold ≥10 | leak | H20_irrelevant_minus_reminder | +4.73 [+1.78, +7.99] | — |
| Protocol sensitivity: all gold | violation | H21 | -27.25 [-33.75, -21.00] | no |
| Protocol sensitivity: all gold | violation | on_plain_minus_off_plain | -7.75 [-11.25, -4.25] | — |
| Protocol sensitivity: all gold | violation | H20_empty_minus_reminder | +3.50 [+0.50, +6.50] | yes |
| Protocol sensitivity: all gold | violation | H20_neutral_minus_reminder | +0.75 [-2.25, +3.75] | no |
| Protocol sensitivity: all gold | violation | H20_irrelevant_minus_reminder | +7.75 [+4.50, +11.00] | yes |
| Protocol sensitivity: all gold | leak | H21 | -26.25 [-32.50, -20.25] | — |
| Protocol sensitivity: all gold | leak | on_plain_minus_off_plain | -7.75 [-11.25, -4.50] | — |
| Protocol sensitivity: all gold | leak | H20_empty_minus_reminder | +3.25 [+0.25, +6.25] | — |
| Protocol sensitivity: all gold | leak | H20_neutral_minus_reminder | +0.00 [-2.75, +2.75] | — |
| Protocol sensitivity: all gold | leak | H20_irrelevant_minus_reminder | +7.00 [+3.99, +10.00] | — |
| Protocol sensitivity: gold ≥10 | violation | H21 | -29.29 [-35.21, -23.08] | no |
| Protocol sensitivity: gold ≥10 | violation | on_plain_minus_off_plain | -7.69 [-10.95, -4.73] | — |
| Protocol sensitivity: gold ≥10 | violation | H20_empty_minus_reminder | +1.48 [-1.18, +4.14] | no |
| Protocol sensitivity: gold ≥10 | violation | H20_neutral_minus_reminder | +1.78 [-0.89, +4.73] | no |
| Protocol sensitivity: gold ≥10 | violation | H20_irrelevant_minus_reminder | +5.62 [+2.37, +9.17] | yes |
| Protocol sensitivity: gold ≥10 | leak | H21 | -28.70 [-34.32, -22.78] | — |
| Protocol sensitivity: gold ≥10 | leak | on_plain_minus_off_plain | -7.69 [-10.95, -4.73] | — |
| Protocol sensitivity: gold ≥10 | leak | H20_empty_minus_reminder | +1.18 [-1.48, +3.85] | — |
| Protocol sensitivity: gold ≥10 | leak | H20_neutral_minus_reminder | +1.18 [-1.48, +4.14] | — |
| Protocol sensitivity: gold ≥10 | leak | H20_irrelevant_minus_reminder | +4.73 [+1.78, +7.99] | — |

| Scope | H20 joint | Rule |
| --- | --- | --- |
| Baseline: all gold | no | All three directional lower bounds > 0 |
| Baseline: gold ≥10 | no | All three directional lower bounds > 0 |
| Protocol sensitivity: all gold | no | All three directional lower bounds > 0 |
| Protocol sensitivity: gold ≥10 | no | All three directional lower bounds > 0 |

## Exploratory trace regex

| Pattern / filter | Value |
| --- | --- |
| Pattern | `r'tutor\|system prompt\|instruction'`, case-insensitive |
| Filter | Main arm; thinking on; explicit request; reasoning text; all 400 requests including empty reasoning; no prefill/pilots |

| Model | Matches / denominator | Share |
| --- | --- | --- |
| Nemotron | 392 / 400 | 98.00% |
| Gemma 4 12B | 309 / 400 | 77.25% |
| Qwen3-14B | 11 / 400 | 2.75% |
| Qwen3-32B | 3 / 400 | 0.75% |

## Cost and resources

| Item | Value |
| --- | --- |
| Controls step observed metering | $0.44000052 (provisional) |
| Controls conservative exposure | $0.506497 / $1.40 cap, including setup/teardown |
| Reserve | $0.30 |
| Modal billed cost | $0.00 |
| Active Phase 5 Modal apps / containers | 0 / 0 |
| Phase 6 Vast | Out of scope; no query or mutation in this step; instance 57119355 and other Phase 6 resources preserved |
| Checks | 391 tests passed; archived Gemma analysis reproduced exactly |

H20: not supported; empty minus reminder +3.50 [+0.50, +6.50] pp; neutral minus reminder +0.75 [-2.25, +3.75] pp; irrelevant minus reminder +7.75 [+4.50, +11.00] pp; joint=False.

H21: not supported; interaction -26.75 [-33.25, -20.50] pp.
