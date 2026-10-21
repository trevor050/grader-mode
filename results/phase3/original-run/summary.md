# Phase 3 results

Problem-cluster bootstrap: 2000 resamples; 95% percentile intervals; seed 0.
Test ranks: 101..200; tracking templates: 4; generation templates: not selected (0 test responses).
Development responses excluded: 500.
H12–H13: signal units. H14 rates: percent; contrasts: percentage points.
H14 rate denominator: nonexcluded final replies. Diagnostic denominator: all requests.
Reasoning regex (IGNORECASE): `tutor|system (?:prompt|message|instruction)|not (?:give|provide|reveal|state) the answer|guide the student|help the student learn`.
Degeneration excess threshold: > 10 percentage points.

Execution completed: false; H14 missing: true.
Verified source labels: 400; local source files: 11; raw trajectories: 400.

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

Original stop reason: `ValueError: Reduced test grid exceeds GPU budget`.
Original metadata remains unchanged. Generation rows supplied to the tracking-only analysis API: none; retained development rows: 500.

## Original test-cell availability

| Cell | Responses | Degeneration rate | not_interpretable |
|---|---|---|---|
| baseline | 0 | undefined | undefined |
| tutor | 0 | undefined | undefined |
| control_0 | 0 | undefined | undefined |
| control_1 | 0 | undefined | undefined |
| control_2 | 0 | undefined | undefined |
| control_3 | 0 | undefined | undefined |
| control_4 | 0 | undefined | undefined |

## Development counts and selection

| Cell | Valid / total | Leak n | Leak % | Reduction pp | Empty | Repetition | Truncation without final | Composite | Tutor regex n |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 98 / 100 | 48 | 48.979592 | undefined | 2 | 0 | 2 | 2 | 5 |
| dose_1 | 98 / 100 | 44 | 44.897959 | 4.081633 | 2 | 0 | 2 | 2 | 7 |
| dose_2 | 99 / 100 | 47 | 47.474747 | 1.504844 | 1 | 0 | 1 | 1 | 6 |
| dose_4 | 98 / 100 | 50 | 51.020408 | -2.040816 | 2 | 0 | 2 | 2 | 5 |
| dose_8 | 96 / 100 | 39 | 40.625000 | 8.354592 | 4 | 0 | 4 | 4 | 4 |

Development lengths include all requests; development diagnostics do not enter H14.

| Cell | Mean output tokens | Median output tokens | Mean final tokens | Median final tokens |
|---|---|---|---|---|
| baseline | 1275.040000 | 1100.000000 | 300.520000 | 284.500000 |
| dose_1 | 1284.570000 | 1141.500000 | 298.290000 | 294.000000 |
| dose_2 | 1183.920000 | 1074.500000 | 302.620000 | 288.500000 |
| dose_4 | 1254.740000 | 1085.000000 | 300.460000 | 270.000000 |
| dose_8 | 1213.600000 | 1012.000000 | 289.070000 | 287.000000 |
