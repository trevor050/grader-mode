# Phase 5: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

Collected **2,400 main replies** from the original 200 problems × two templates: H20 reminder/empty (800), then the complete H21 grid (1,600). The unscored 8-prompt smoke and balanced 256-prompt throughput measurement are separate and excluded. Model load and generation succeeded; no model-level fix or GLM fallback was used.

**Formal inference remains pending.** Two thinking-off explicit replies emitted an unexpected `</think>` boundary, so the unchanged `analyze_5` protocol-failure gate withholds H20/H21 decisions. Those requests remain in the denominator with empty final replies. Estimates and CIs below are descriptive; no parser or audit labels were changed after seeing outcomes. Full H20 is also unavailable until its controls run.

| Main result (violation) | Estimate | Paired problem-cluster 95% CI |
| --- | ---: | ---: |
| Explicit, thinking off | 43.00% (172/400) | 37.75%–48.00% |
| Explicit, thinking on | 8.50% (34/400) | 5.25%–12.00% |
| Plain, thinking off | 12.00% (48/400) | 8.50%–15.75% |
| Plain, thinking on | 4.25% (17/400) | 2.00%–6.75% |
| H21: (on−off) explicit − (on−off) plain | −26.75 pp | −33.25 to −20.50 pp |
| H20: empty minus reminder | +3.50 pp | +0.50 to +6.50 pp |

Reminder violations were 27/400 (6.75%, CI 4.00%–10.00%) versus empty 41/400 (10.25%, CI 6.75%–14.00%). In the registered gold ≥10 sensitivity (169 problems), H21 is −28.99 pp (CI −34.92 to −23.08); empty minus reminder is +1.48 pp (CI −1.18 to +4.14). These also remain descriptive.

**Pending stage:** neutral + irrelevant controls (800 replies). Its prospective projection was $0.567435 / 567.435 s, with $0.657895 available after the $0.30 reserve but only 517.557 s left before the conservative work deadline. The runtime allowance shortfall was **49.877 s, equivalent to $0.049877** at the $0.001/s bound. This was a deadline gate, not a credit shortage. Collection stopped and the app was destroyed. No cells were dropped. The unchanged analyzer's legacy `cell_drop_stage_2`/`dropped_cells` fields describe the same complete six-cell prefix; they do not change these controls' pending status.

**Audit:** Codex, self-labeled blind to arm at the user's explicit request; no independent human labels. All 40 were labeled before analysis. Attempt agreement is 35/40 (87.5%, κ=0.711), numeric agreement 10/11 (90.9%), final-boundary agreement 40/40 (100%). Five attempt disagreements and one numeric disagreement remain recorded; no accuracy cutoff was preregistered. See [method](nemotron/audit-method.json) and [disagreements](nemotron/parser-disagreements.json).

**Exploratory trace comparison:** case-insensitive `r'tutor|system prompt|instruction'` matched **392/400 (98.0%)** thinking-on explicit main traces. This uses exactly the FINDINGS.md filter and all-request denominator, including empty reasoning; excludes prefill and pilot replies. References there: Gemma 309/400 (77.25%), Qwen3-14B 11/400 (2.75%), Qwen3-32B 3/400 (0.75%). This lexical count does not establish a mechanism. [Calculation](nemotron/exploratory-trace-regex.json).

**Execution and cost:** Modal RTX-PRO-6000 Blackwell; pinned mixed NVFP4 checkpoint unchanged, actual expert kernel Marlin weight-only, FlashInfer attention/FP8. CUDA graphs on, max_num_seqs 256, chunks 256. Balanced throughput was 3,854.75 output tok/s including control/commit time. New Modal observed metering: **$1.29170640**, provisional, all within free credit ($0 billed); conservative whole-authorization exposure **$1.656979**, below $2.60, including three canceled toolchain attempts. Vast retry observed spend before the provider switch: **$0.09196761**. Current total observed Phase 5 spend including Gemma, earlier pilot, and all Vast is **$2.65387126**, provisional. Retained conservative reservations are not provider billing.

**Resources:** zero active Phase 5 Modal apps/containers and no Phase 5-labelled Vast instances. Concurrent Phase 6 Vast instance **57119355** (`phase6-3.8-a1-20261021`) is out of scope and was preserved. [Resource receipt](nemotron-execution-final-resources.json), [accounting](final-accounting.json), [execution/failure history](nemotron-execution-retry-report.md).

The detailed tables below retain the unmodified scoring estimates. “Tested=True” identifies a registered contrast computed by the analyzer, not an eligible hypothesis decision; Supported=None remains pending.

| Collection | Value |
| --- | --- |
| Responses | 2400 |
| Problems | 200 |
| Grid | six complete cells; controls pending |
| Inference eligible | False |
| Pending reasons | protocol_failures |
| Bootstrap | 2,000 paired problem clusters; seed 0; percentile 95% CI |
| Denominator | All requests, including cap-hit and empty finals |
| Audit author | Codex, self-labeled blind to arm at the user's explicit request; no independent human labels |

| Audit | Labeled / required | Agreement | Disagreements |
| --- | ---: | ---: | ---: |
| Attempt | 40 / 40 | 0.875 | 5 |
| Committed number | 11 / 11 | 0.9090909090909091 | 1 |
| Final boundary | 40 / 40 | 1.0 | 0 |

| Pending/unreleased cell |
| --- |
| prefill / on / explicit / neutral |
| prefill / on / explicit / irrelevant |

## All gold

| Problems | Replies |
| ---: | ---: |
| 200 | 2400 |

| Arm | Thinking | Request | Prefill | N | Component | Count | Estimate [95% CI] |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| main | off | explicit | none | 400 | violation | 172 | 0.4300 [0.3775, 0.4800] |
| main | off | explicit | none | 400 | leak | 169 | 0.4225 [0.3725, 0.4725] |
| main | off | explicit | none | 400 | attempt | 63 | 0.1575 [0.1250, 0.1950] |
| main | off | explicit | none | 400 | correct_attempt | 57 | 0.1425 [0.1100, 0.1775] |
| main | off | explicit | none | 400 | wrong_attempt | 6 | 0.0150 [0.0050, 0.0275] |
| main | off | explicit | none | 400 | empty_final | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | truncated_final | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | empty_or_truncated_final | 3 | 0.0075 [0.0000, 0.0175] |
| main | off | explicit | none | 400 | final_reply_length |  | 938.6175 [864.4406, 1036.1765] |
| main | off | explicit | none | 400 | cap_hit | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | no_final | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | cap_hit_no_final | 1 | 0.0025 [0.0000, 0.0075] |
| main | off | explicit | none | 400 | protocol_failure | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | explicit | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | violation | 48 | 0.1200 [0.0850, 0.1575] |
| main | off | plain | none | 400 | leak | 47 | 0.1175 [0.0825, 0.1551] |
| main | off | plain | none | 400 | attempt | 3 | 0.0075 [0.0000, 0.0175] |
| main | off | plain | none | 400 | correct_attempt | 2 | 0.0050 [0.0000, 0.0125] |
| main | off | plain | none | 400 | wrong_attempt | 1 | 0.0025 [0.0000, 0.0075] |
| main | off | plain | none | 400 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | final_reply_length |  | 756.7400 [714.2701, 799.7452] |
| main | off | plain | none | 400 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | violation | 34 | 0.0850 [0.0525, 0.1200] |
| main | on | explicit | none | 400 | leak | 33 | 0.0825 [0.0525, 0.1175] |
| main | on | explicit | none | 400 | attempt | 4 | 0.0100 [0.0025, 0.0200] |
| main | on | explicit | none | 400 | correct_attempt | 2 | 0.0050 [0.0000, 0.0125] |
| main | on | explicit | none | 400 | wrong_attempt | 2 | 0.0050 [0.0000, 0.0125] |
| main | on | explicit | none | 400 | empty_final | 8 | 0.0200 [0.0075, 0.0375] |
| main | on | explicit | none | 400 | truncated_final | 11 | 0.0275 [0.0100, 0.0500] |
| main | on | explicit | none | 400 | empty_or_truncated_final | 11 | 0.0275 [0.0100, 0.0500] |
| main | on | explicit | none | 400 | final_reply_length |  | 863.6375 [827.6444, 898.9968] |
| main | on | explicit | none | 400 | cap_hit | 11 | 0.0275 [0.0100, 0.0500] |
| main | on | explicit | none | 400 | no_final | 8 | 0.0200 [0.0075, 0.0375] |
| main | on | explicit | none | 400 | cap_hit_no_final | 8 | 0.0200 [0.0075, 0.0375] |
| main | on | explicit | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | violation | 17 | 0.0425 [0.0200, 0.0675] |
| main | on | plain | none | 400 | leak | 16 | 0.0400 [0.0199, 0.0650] |
| main | on | plain | none | 400 | attempt | 1 | 0.0025 [0.0000, 0.0075] |
| main | on | plain | none | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | wrong_attempt | 1 | 0.0025 [0.0000, 0.0075] |
| main | on | plain | none | 400 | empty_final | 5 | 0.0125 [0.0025, 0.0275] |
| main | on | plain | none | 400 | truncated_final | 6 | 0.0150 [0.0025, 0.0300] |
| main | on | plain | none | 400 | empty_or_truncated_final | 6 | 0.0150 [0.0025, 0.0300] |
| main | on | plain | none | 400 | final_reply_length |  | 736.3225 [703.1147, 770.5221] |
| main | on | plain | none | 400 | cap_hit | 6 | 0.0150 [0.0025, 0.0300] |
| main | on | plain | none | 400 | no_final | 5 | 0.0125 [0.0025, 0.0275] |
| main | on | plain | none | 400 | cap_hit_no_final | 5 | 0.0125 [0.0025, 0.0275] |
| main | on | plain | none | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | violation | 41 | 0.1025 [0.0675, 0.1400] |
| prefill | on | explicit | empty | 400 | leak | 40 | 0.1000 [0.0650, 0.1375] |
| prefill | on | explicit | empty | 400 | attempt | 2 | 0.0050 [0.0000, 0.0125] |
| prefill | on | explicit | empty | 400 | correct_attempt | 1 | 0.0025 [0.0000, 0.0075] |
| prefill | on | explicit | empty | 400 | wrong_attempt | 1 | 0.0025 [0.0000, 0.0075] |
| prefill | on | explicit | empty | 400 | empty_final | 7 | 0.0175 [0.0050, 0.0325] |
| prefill | on | explicit | empty | 400 | truncated_final | 8 | 0.0200 [0.0074, 0.0375] |
| prefill | on | explicit | empty | 400 | empty_or_truncated_final | 8 | 0.0200 [0.0074, 0.0375] |
| prefill | on | explicit | empty | 400 | final_reply_length |  | 886.8825 [849.8311, 924.2248] |
| prefill | on | explicit | empty | 400 | cap_hit | 8 | 0.0200 [0.0074, 0.0375] |
| prefill | on | explicit | empty | 400 | no_final | 7 | 0.0175 [0.0050, 0.0325] |
| prefill | on | explicit | empty | 400 | cap_hit_no_final | 7 | 0.0175 [0.0050, 0.0325] |
| prefill | on | explicit | empty | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | violation | 27 | 0.0675 [0.0400, 0.1000] |
| prefill | on | explicit | reminder | 400 | leak | 27 | 0.0675 [0.0400, 0.1000] |
| prefill | on | explicit | reminder | 400 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | empty_final | 3 | 0.0075 [0.0000, 0.0175] |
| prefill | on | explicit | reminder | 400 | truncated_final | 6 | 0.0150 [0.0025, 0.0301] |
| prefill | on | explicit | reminder | 400 | empty_or_truncated_final | 6 | 0.0150 [0.0025, 0.0301] |
| prefill | on | explicit | reminder | 400 | final_reply_length |  | 857.9625 [821.2314, 896.6404] |
| prefill | on | explicit | reminder | 400 | cap_hit | 6 | 0.0150 [0.0025, 0.0301] |
| prefill | on | explicit | reminder | 400 | no_final | 3 | 0.0075 [0.0000, 0.0175] |
| prefill | on | explicit | reminder | 400 | cap_hit_no_final | 3 | 0.0075 [0.0000, 0.0175] |
| prefill | on | explicit | reminder | 400 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 400 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |

| Outcome | Contrast | Estimate [95% CI] | Tested | Supported |
| --- | --- | --- | --- | --- |
| violation | H21 | -0.2675 [-0.3325, -0.2050] | True | None |
| violation | on_plain_minus_off_plain | -0.0775 [-0.1125, -0.0425] | False | None |
| violation | H20_empty_minus_reminder | 0.0350 [0.0050, 0.0650] | True | None |
| violation | H20_neutral_minus_reminder | unavailable | False | None |
| violation | H20_irrelevant_minus_reminder | unavailable | False | None |
| violation | H20_joint | unavailable | False | None |
| violation | H20_reduced | 0.0350 [0.0050, 0.0650] | True | None |
| leak | H21 | -0.2625 [-0.3250, -0.2025] | False | None |
| leak | on_plain_minus_off_plain | -0.0775 [-0.1125, -0.0450] | False | None |
| leak | H20_empty_minus_reminder | 0.0325 [0.0025, 0.0625] | False | None |
| leak | H20_neutral_minus_reminder | unavailable | False | None |
| leak | H20_irrelevant_minus_reminder | unavailable | False | None |
| leak | H20_joint | unavailable | False | None |
| leak | H20_reduced | 0.0325 [0.0025, 0.0625] | False | None |

## Gold >= 10

| Problems | Replies |
| ---: | ---: |
| 169 | 2028 |

| Arm | Thinking | Request | Prefill | N | Component | Count | Estimate [95% CI] |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| main | off | explicit | none | 338 | violation | 136 | 0.4024 [0.3490, 0.4556] |
| main | off | explicit | none | 338 | leak | 134 | 0.3964 [0.3432, 0.4497] |
| main | off | explicit | none | 338 | attempt | 49 | 0.1450 [0.1095, 0.1834] |
| main | off | explicit | none | 338 | correct_attempt | 45 | 0.1331 [0.1006, 0.1686] |
| main | off | explicit | none | 338 | wrong_attempt | 4 | 0.0118 [0.0030, 0.0237] |
| main | off | explicit | none | 338 | empty_final | 1 | 0.0030 [0.0000, 0.0089] |
| main | off | explicit | none | 338 | truncated_final | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | explicit | none | 338 | empty_or_truncated_final | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | explicit | none | 338 | final_reply_length |  | 953.6568 [865.5738, 1065.7686] |
| main | off | explicit | none | 338 | cap_hit | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | explicit | none | 338 | no_final | 1 | 0.0030 [0.0000, 0.0089] |
| main | off | explicit | none | 338 | cap_hit_no_final | 1 | 0.0030 [0.0000, 0.0089] |
| main | off | explicit | none | 338 | protocol_failure | 1 | 0.0030 [0.0000, 0.0089] |
| main | off | explicit | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | violation | 29 | 0.0858 [0.0562, 0.1183] |
| main | off | plain | none | 338 | leak | 28 | 0.0828 [0.0562, 0.1154] |
| main | off | plain | none | 338 | attempt | 3 | 0.0089 [0.0000, 0.0207] |
| main | off | plain | none | 338 | correct_attempt | 2 | 0.0059 [0.0000, 0.0148] |
| main | off | plain | none | 338 | wrong_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| main | off | plain | none | 338 | empty_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | empty_or_truncated_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | final_reply_length |  | 754.9527 [712.3320, 800.3321] |
| main | off | plain | none | 338 | cap_hit | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | cap_hit_no_final | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | off | plain | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | violation | 12 | 0.0355 [0.0178, 0.0562] |
| main | on | explicit | none | 338 | leak | 11 | 0.0325 [0.0148, 0.0533] |
| main | on | explicit | none | 338 | attempt | 3 | 0.0089 [0.0000, 0.0207] |
| main | on | explicit | none | 338 | correct_attempt | 2 | 0.0059 [0.0000, 0.0148] |
| main | on | explicit | none | 338 | wrong_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| main | on | explicit | none | 338 | empty_final | 4 | 0.0118 [0.0030, 0.0237] |
| main | on | explicit | none | 338 | truncated_final | 6 | 0.0178 [0.0030, 0.0355] |
| main | on | explicit | none | 338 | empty_or_truncated_final | 6 | 0.0178 [0.0030, 0.0355] |
| main | on | explicit | none | 338 | final_reply_length |  | 872.8698 [833.3738, 914.7915] |
| main | on | explicit | none | 338 | cap_hit | 6 | 0.0178 [0.0030, 0.0355] |
| main | on | explicit | none | 338 | no_final | 4 | 0.0118 [0.0030, 0.0237] |
| main | on | explicit | none | 338 | cap_hit_no_final | 4 | 0.0118 [0.0030, 0.0237] |
| main | on | explicit | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | explicit | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | violation | 3 | 0.0089 [0.0000, 0.0207] |
| main | on | plain | none | 338 | leak | 2 | 0.0059 [0.0000, 0.0148] |
| main | on | plain | none | 338 | attempt | 1 | 0.0030 [0.0000, 0.0089] |
| main | on | plain | none | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | wrong_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| main | on | plain | none | 338 | empty_final | 2 | 0.0059 [0.0000, 0.0148] |
| main | on | plain | none | 338 | truncated_final | 3 | 0.0089 [0.0000, 0.0207] |
| main | on | plain | none | 338 | empty_or_truncated_final | 3 | 0.0089 [0.0000, 0.0207] |
| main | on | plain | none | 338 | final_reply_length |  | 744.1391 [707.3305, 781.8856] |
| main | on | plain | none | 338 | cap_hit | 3 | 0.0089 [0.0000, 0.0207] |
| main | on | plain | none | 338 | no_final | 2 | 0.0059 [0.0000, 0.0148] |
| main | on | plain | none | 338 | cap_hit_no_final | 2 | 0.0059 [0.0000, 0.0148] |
| main | on | plain | none | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| main | on | plain | none | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | violation | 15 | 0.0444 [0.0207, 0.0710] |
| prefill | on | explicit | empty | 338 | leak | 14 | 0.0414 [0.0207, 0.0651] |
| prefill | on | explicit | empty | 338 | attempt | 2 | 0.0059 [0.0000, 0.0148] |
| prefill | on | explicit | empty | 338 | correct_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | empty | 338 | wrong_attempt | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | empty | 338 | empty_final | 5 | 0.0148 [0.0030, 0.0296] |
| prefill | on | explicit | empty | 338 | truncated_final | 6 | 0.0178 [0.0059, 0.0325] |
| prefill | on | explicit | empty | 338 | empty_or_truncated_final | 6 | 0.0178 [0.0059, 0.0325] |
| prefill | on | explicit | empty | 338 | final_reply_length |  | 890.8077 [852.6492, 930.5401] |
| prefill | on | explicit | empty | 338 | cap_hit | 6 | 0.0178 [0.0059, 0.0325] |
| prefill | on | explicit | empty | 338 | no_final | 5 | 0.0148 [0.0030, 0.0296] |
| prefill | on | explicit | empty | 338 | cap_hit_no_final | 5 | 0.0148 [0.0030, 0.0296] |
| prefill | on | explicit | empty | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | empty | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | violation | 10 | 0.0296 [0.0148, 0.0503] |
| prefill | on | explicit | reminder | 338 | leak | 10 | 0.0296 [0.0148, 0.0503] |
| prefill | on | explicit | reminder | 338 | attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | correct_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | wrong_attempt | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | empty_final | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | reminder | 338 | truncated_final | 3 | 0.0089 [0.0000, 0.0207] |
| prefill | on | explicit | reminder | 338 | empty_or_truncated_final | 3 | 0.0089 [0.0000, 0.0207] |
| prefill | on | explicit | reminder | 338 | final_reply_length |  | 867.6627 [825.5208, 910.7449] |
| prefill | on | explicit | reminder | 338 | cap_hit | 3 | 0.0089 [0.0000, 0.0207] |
| prefill | on | explicit | reminder | 338 | no_final | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | reminder | 338 | cap_hit_no_final | 1 | 0.0030 [0.0000, 0.0089] |
| prefill | on | explicit | reminder | 338 | protocol_failure | 0 | 0.0000 [0.0000, 0.0000] |
| prefill | on | explicit | reminder | 338 | unexpected_off_thought | 0 | 0.0000 [0.0000, 0.0000] |

| Outcome | Contrast | Estimate [95% CI] | Tested | Supported |
| --- | --- | --- | --- | --- |
| violation | H21 | -0.2899 [-0.3492, -0.2308] | True | None |
| violation | on_plain_minus_off_plain | -0.0769 [-0.1095, -0.0473] | False | None |
| violation | H20_empty_minus_reminder | 0.0148 [-0.0118, 0.0414] | True | None |
| violation | H20_neutral_minus_reminder | unavailable | False | None |
| violation | H20_irrelevant_minus_reminder | unavailable | False | None |
| violation | H20_joint | unavailable | False | None |
| violation | H20_reduced | 0.0148 [-0.0118, 0.0414] | True | None |
| leak | H21 | -0.2870 [-0.3432, -0.2278] | False | None |
| leak | on_plain_minus_off_plain | -0.0769 [-0.1095, -0.0473] | False | None |
| leak | H20_empty_minus_reminder | 0.0118 [-0.0148, 0.0385] | False | None |
| leak | H20_neutral_minus_reminder | unavailable | False | None |
| leak | H20_irrelevant_minus_reminder | unavailable | False | None |
| leak | H20_joint | unavailable | False | None |
| leak | H20_reduced | 0.0118 [-0.0148, 0.0385] | False | None |

H21: unavailable/pending, -0.2675 [-0.3325, -0.2050].

H20: unavailable/pending, empty minus reminder 0.0350 [0.0050, 0.0650]; registered reduced contrast.
