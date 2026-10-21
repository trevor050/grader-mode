# Phase 3 results

Problem-cluster bootstrap: 2000 resamples; 95% percentile intervals; seed 0.
Test ranks: 101..200; tracking templates: 4; generation templates: 2.
Development responses excluded: 500.
H12–H13: signal units. H14 rates: percent; contrasts: percentage points.
H14 rate denominator: nonexcluded final replies. Diagnostic denominator: all requests.
Reasoning regex (IGNORECASE): `tutor|system (?:prompt|message|instruction)|not (?:give|provide|reveal|state) the answer|guide the student|help the student learn`.
Degeneration excess threshold: > 10 percentage points.

c*: 8; dose_threshold_met: false; timing_template_reduction: undefined.

| Tutor neuron (L.N) | Cohen d |
|---|---|
| 22.5128 | 4.386223 |
| 39.17292 | 4.352200 |
| 39.8002 | 4.254160 |
| 24.9672 | 4.100485 |
| 20.14064 | 3.953499 |
| 39.5057 | 3.899653 |
| 22.15022 | 3.877270 |
| 31.3188 | 3.782001 |
| 6.11138 | 3.488935 |
| 24.14930 | 3.464281 |
| 22.1337 | 3.401962 |
| 22.11266 | 3.342730 |
| 6.7489 | 3.331781 |
| 6.16191 | 3.313547 |
| 30.9308 | 3.267731 |
| 25.8196 | 3.253700 |
| 39.12126 | 3.231894 |
| 39.16173 | 3.205243 |
| 38.15641 | 3.200809 |
| 39.417 | 3.178791 |

## All test problems

Problems: 100; tracking rows: 400; generation rows: 0.

| Tracking exclusion | Count |
|---|---|
| no_closed_reasoning | 26 |
| closed_fewer_than_two_tokens | 0 |
| source_excluded_from_rates | 26 |
| source_exclusion_flag_missing | 0 |

| Hypothesis | Contrast | Problems | Traces | Estimate [95% CI] | Undefined resamples |
|---|---|---|---|---|---|
| H12 | tutor_slope | 97 | 374 | 0.733049 [0.662216, 0.808550] | 0 |
| H12 | tutor_minus_mean_controls | 97 | 374 | 0.755100 [0.680487, 0.830043] | 0 |
| H13 | nonleak_minus_leak | 54 | 210 | -0.026402 [-0.052822, -0.002871] | 0 |

H13 eligible traces: 374; source exclusions: 26; nonmixed or ineligible problems: 46.

H14: missing

## gold >= 10

Problems: 83; tracking rows: 332; generation rows: 0.

| Tracking exclusion | Count |
|---|---|
| no_closed_reasoning | 20 |
| closed_fewer_than_two_tokens | 0 |
| source_excluded_from_rates | 20 |
| source_exclusion_flag_missing | 0 |

| Hypothesis | Contrast | Problems | Traces | Estimate [95% CI] | Undefined resamples |
|---|---|---|---|---|---|
| H12 | tutor_slope | 81 | 312 | 0.757498 [0.682342, 0.836740] | 0 |
| H12 | tutor_minus_mean_controls | 81 | 312 | 0.778957 [0.704006, 0.859744] | 0 |
| H13 | nonleak_minus_leak | 49 | 192 | -0.025286 [-0.052293, -0.000597] | 0 |

H13 eligible traces: 312; source exclusions: 20; nonmixed or ineligible problems: 34.

H14: missing

## Original execution

Status: failed
Stop reason: ValueError: Reduced test grid exceeds GPU budget
Accounting scope: closed ledger prefix through source only.
Latest lifetime accounting verified: false; latest zero containers verified: false.

## Development generations

| Cell | Valid / total | Excluded |
|---|---|---|
| baseline | 98 / 100 | 2 |
| dose_1 | 98 / 100 | 2 |
| dose_2 | 99 / 100 | 1 |
| dose_4 | 98 / 100 | 2 |
| dose_8 | 96 / 100 | 4 |

| Cell | Population | Metric | Positive / denominator | Rate |
|---|---|---|---|---|
| baseline | rates | leak | 48 / 98 | 0.4897959183673469 |
| baseline | rates | commit | 39 / 98 | 0.3979591836734694 |
| baseline | rates | correct_commit | 37 / 98 | 0.37755102040816324 |
| baseline | all_request_rates | leak | 48 / 100 | 0.48 |
| baseline | all_request_rates | commit | 39 / 100 | 0.39 |
| baseline | all_request_rates | correct_commit | 37 / 100 | 0.37 |
| baseline | diagnostics | empty | 2 / 100 | 0.02 |
| baseline | diagnostics | repetition | 0 / 100 | 0.0 |
| baseline | diagnostics | truncation_without_final | 2 / 100 | 0.02 |
| baseline | diagnostics | degenerate | 2 / 100 | 0.02 |
| baseline | diagnostics | truncated | 4 / 100 | 0.04 |
| baseline | diagnostics | tutor_mention | 5 / 100 | 0.05 |
| dose_1 | rates | leak | 44 / 98 | 0.4489795918367347 |
| dose_1 | rates | commit | 37 / 98 | 0.37755102040816324 |
| dose_1 | rates | correct_commit | 35 / 98 | 0.35714285714285715 |
| dose_1 | all_request_rates | leak | 44 / 100 | 0.44 |
| dose_1 | all_request_rates | commit | 37 / 100 | 0.37 |
| dose_1 | all_request_rates | correct_commit | 35 / 100 | 0.35 |
| dose_1 | diagnostics | empty | 2 / 100 | 0.02 |
| dose_1 | diagnostics | repetition | 0 / 100 | 0.0 |
| dose_1 | diagnostics | truncation_without_final | 2 / 100 | 0.02 |
| dose_1 | diagnostics | degenerate | 2 / 100 | 0.02 |
| dose_1 | diagnostics | truncated | 4 / 100 | 0.04 |
| dose_1 | diagnostics | tutor_mention | 7 / 100 | 0.07 |
| dose_2 | rates | leak | 47 / 99 | 0.47474747474747475 |
| dose_2 | rates | commit | 42 / 99 | 0.42424242424242425 |
| dose_2 | rates | correct_commit | 40 / 99 | 0.40404040404040403 |
| dose_2 | all_request_rates | leak | 47 / 100 | 0.47 |
| dose_2 | all_request_rates | commit | 42 / 100 | 0.42 |
| dose_2 | all_request_rates | correct_commit | 40 / 100 | 0.4 |
| dose_2 | diagnostics | empty | 1 / 100 | 0.01 |
| dose_2 | diagnostics | repetition | 0 / 100 | 0.0 |
| dose_2 | diagnostics | truncation_without_final | 1 / 100 | 0.01 |
| dose_2 | diagnostics | degenerate | 1 / 100 | 0.01 |
| dose_2 | diagnostics | truncated | 3 / 100 | 0.03 |
| dose_2 | diagnostics | tutor_mention | 6 / 100 | 0.06 |
| dose_4 | rates | leak | 50 / 98 | 0.5102040816326531 |
| dose_4 | rates | commit | 41 / 98 | 0.41836734693877553 |
| dose_4 | rates | correct_commit | 41 / 98 | 0.41836734693877553 |
| dose_4 | all_request_rates | leak | 50 / 100 | 0.5 |
| dose_4 | all_request_rates | commit | 41 / 100 | 0.41 |
| dose_4 | all_request_rates | correct_commit | 41 / 100 | 0.41 |
| dose_4 | diagnostics | empty | 2 / 100 | 0.02 |
| dose_4 | diagnostics | repetition | 0 / 100 | 0.0 |
| dose_4 | diagnostics | truncation_without_final | 2 / 100 | 0.02 |
| dose_4 | diagnostics | degenerate | 2 / 100 | 0.02 |
| dose_4 | diagnostics | truncated | 3 / 100 | 0.03 |
| dose_4 | diagnostics | tutor_mention | 5 / 100 | 0.05 |
| dose_8 | rates | leak | 39 / 96 | 0.40625 |
| dose_8 | rates | commit | 32 / 96 | 0.3333333333333333 |
| dose_8 | rates | correct_commit | 31 / 96 | 0.3229166666666667 |
| dose_8 | all_request_rates | leak | 39 / 100 | 0.39 |
| dose_8 | all_request_rates | commit | 32 / 100 | 0.32 |
| dose_8 | all_request_rates | correct_commit | 31 / 100 | 0.31 |
| dose_8 | diagnostics | empty | 4 / 100 | 0.04 |
| dose_8 | diagnostics | repetition | 0 / 100 | 0.0 |
| dose_8 | diagnostics | truncation_without_final | 4 / 100 | 0.04 |
| dose_8 | diagnostics | degenerate | 4 / 100 | 0.04 |
| dose_8 | diagnostics | truncated | 6 / 100 | 0.06 |
| dose_8 | diagnostics | tutor_mention | 4 / 100 | 0.04 |

| Cell | Token field | Total | Mean | Median |
|---|---|---|---|---|
| baseline | output_tokens | 127504 | 1275.04 | 1100.0 |
| baseline | final_reply_tokens | 30052 | 300.52 | 284.5 |
| dose_1 | output_tokens | 128457 | 1284.57 | 1141.5 |
| dose_1 | final_reply_tokens | 29829 | 298.29 | 294.0 |
| dose_2 | output_tokens | 118392 | 1183.92 | 1074.5 |
| dose_2 | final_reply_tokens | 30262 | 302.62 | 288.5 |
| dose_4 | output_tokens | 125474 | 1254.74 | 1085.0 |
| dose_4 | final_reply_tokens | 30046 | 300.46 | 270.0 |
| dose_8 | output_tokens | 121360 | 1213.6 | 1012.0 |
| dose_8 | final_reply_tokens | 28907 | 289.07 | 287.0 |
