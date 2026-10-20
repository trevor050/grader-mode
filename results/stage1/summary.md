# Phase 1 results

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Inference eligibility: False.
- Incomplete six-model Qwen3 ladder: ineligible for inference.

## Qwen/Qwen3-0.6B

200 problems; 8000 responses.

Tutor leak: 56.53% [51.37%, 61.38%].
Ceiling solve (correct commit): 33.38% [28.75%, 37.88%].
Ceiling containment sensitivity: 60.00% [54.37%, 65.12%].
Tutor leak / ceiling solve: 1.69 [1.51, 1.92].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 58.38% [52.75%, 63.75%] | 22.00% [18.38%, 25.62%] | 12.12% [9.38%, 14.88%] | 1.75 [1.54, 2.01] |
| none | boxed | 56.12% [50.25%, 61.75%] | 89.38% [86.62%, 92.00%] | 48.00% [42.12%, 53.62%] | 1.68 [1.49, 1.92] |
| none | final | 54.00% [48.37%, 59.25%] | 96.25% [94.62%, 97.62%] | 50.00% [44.50%, 55.25%] | 1.62 [1.44, 1.85] |
| human | none | 57.25% [51.62%, 62.62%] | 19.62% [16.38%, 23.00%] | 10.38% [8.12%, 12.75%] | 1.72 [1.52, 1.96] |
| human | boxed | 57.50% [52.00%, 63.00%] | 85.12% [82.00%, 88.25%] | 46.75% [41.38%, 52.00%] | 1.72 [1.53, 1.96] |
| human | final | 54.62% [49.25%, 60.12%] | 94.62% [92.62%, 96.50%] | 49.62% [44.25%, 55.12%] | 1.64 [1.45, 1.87] |
| auto | none | 56.12% [50.25%, 61.75%] | 26.62% [22.75%, 30.75%] | 11.62% [9.12%, 14.25%] | 1.68 [1.49, 1.91] |
| auto | boxed | 59.38% [53.87%, 64.88%] | 91.38% [88.75%, 93.75%] | 50.75% [45.38%, 56.13%] | 1.78 [1.58, 2.03] |
| auto | final | 55.38% [49.75%, 60.88%] | 97.25% [95.88%, 98.50%] | 50.50% [45.00%, 56.12%] | 1.66 [1.46, 1.89] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 5.50 [0.87, 10.12] |
| H1 | final | 3.62 [-1.13, 8.38] |
| H1 | pooled_demand | 4.56 [0.38, 8.69] |
| H2 | none | -1.12 [-4.00, 1.75] |
| H2 | boxed | 1.88 [-1.25, 4.88] |
| H2 | final | 0.75 [-2.87, 4.25] |
| H2 | pooled | 0.50 [-1.25, 2.21] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | 0.3382 | 0.1182 | [0.1066, 0.5698] |
| C(cue, Treatment(reference='none'))[T.auto] | -0.0920 | 0.0692 | [-0.2275, 0.0436] |
| C(cue, Treatment(reference='none'))[T.human] | -0.0461 | 0.0688 | [-0.1809, 0.0886] |
| C(format, Treatment(reference='none'))[T.boxed] | -0.0920 | 0.0767 | [-0.2423, 0.0584] |
| C(format, Treatment(reference='none'))[T.final] | -0.1778 | 0.0774 | [-0.3295, -0.0262] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 0.2252 | 0.0977 | [0.0337, 0.4168] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.1022 | 0.1025 | [-0.0987, 0.3030] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | 0.1474 | 0.0981 | [-0.0449, 0.3398] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | 0.0713 | 0.0967 | [-0.1182, 0.2608] |

Manual handcheck: 100 sampled, 0 leak labels entered; awaiting_manual_labels.
CSV: `results/stage1/handcheck_Qwen-Qwen3-0.6B-fb68f5a6.csv`.
- leak: 0 manual labels; agreement unavailable.
- commit: 0 manual labels; agreement unavailable.
- correct_commit: 0 manual labels; agreement unavailable.

## Qwen/Qwen3-1.7B

200 problems; 8000 responses.

Tutor leak: 82.62% [78.50%, 86.49%].
Ceiling solve (correct commit): 63.88% [59.13%, 68.75%].
Ceiling containment sensitivity: 81.25% [76.62%, 85.62%].
Tutor leak / ceiling solve: 1.29 [1.22, 1.38].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 82.25% [77.75%, 86.50%] | 32.75% [28.00%, 37.50%] | 26.00% [21.75%, 30.25%] | 1.29 [1.20, 1.38] |
| none | boxed | 81.62% [76.88%, 86.00%] | 97.62% [96.00%, 99.00%] | 79.00% [74.00%, 83.75%] | 1.28 [1.19, 1.37] |
| none | final | 82.88% [78.50%, 87.00%] | 98.75% [97.75%, 99.62%] | 80.25% [75.88%, 84.62%] | 1.30 [1.22, 1.39] |
| human | none | 83.62% [79.25%, 87.88%] | 33.88% [28.88%, 38.88%] | 28.38% [23.50%, 33.12%] | 1.31 [1.23, 1.40] |
| human | boxed | 83.50% [79.12%, 87.75%] | 96.75% [94.62%, 98.62%] | 79.62% [74.88%, 84.25%] | 1.31 [1.22, 1.40] |
| human | final | 83.38% [79.00%, 87.50%] | 98.62% [97.62%, 99.50%] | 80.88% [76.12%, 85.38%] | 1.31 [1.22, 1.40] |
| auto | none | 82.75% [78.12%, 87.25%] | 50.25% [45.25%, 55.25%] | 42.00% [36.87%, 47.12%] | 1.30 [1.21, 1.38] |
| auto | boxed | 83.38% [79.38%, 87.25%] | 97.12% [95.25%, 98.62%] | 80.00% [75.50%, 84.38%] | 1.31 [1.23, 1.39] |
| auto | final | 80.25% [75.62%, 84.88%] | 98.75% [97.88%, 99.50%] | 77.38% [72.50%, 82.25%] | 1.26 [1.17, 1.35] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 1.25 [-2.25, 4.75] |
| H1 | final | -3.12 [-6.75, 0.38] |
| H1 | pooled_demand | -0.94 [-4.00, 2.06] |
| H2 | none | -0.88 [-3.75, 1.75] |
| H2 | boxed | -0.12 [-2.50, 2.00] |
| H2 | final | -3.12 [-6.00, -0.25] |
| H2 | pooled | -1.38 [-3.08, 0.17] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | 1.5334 | 0.1558 | [1.2280, 1.8388] |
| C(cue, Treatment(reference='none'))[T.auto] | 0.0346 | 0.0862 | [-0.1343, 0.2035] |
| C(cue, Treatment(reference='none'))[T.human] | 0.0972 | 0.0992 | [-0.0972, 0.2916] |
| C(format, Treatment(reference='none'))[T.boxed] | -0.0422 | 0.0922 | [-0.2229, 0.1385] |
| C(format, Treatment(reference='none'))[T.final] | 0.0434 | 0.0955 | [-0.1437, 0.2306] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 0.0867 | 0.1216 | [-0.1516, 0.3249] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.0331 | 0.1283 | [-0.2184, 0.2846] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | -0.2094 | 0.1226 | [-0.4497, 0.0309] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -0.0616 | 0.1364 | [-0.3289, 0.2057] |

Manual handcheck: 100 sampled, 0 leak labels entered; awaiting_manual_labels.
CSV: `results/stage1/handcheck_Qwen-Qwen3-1.7B-64739e94.csv`.
- leak: 0 manual labels; agreement unavailable.
- commit: 0 manual labels; agreement unavailable.
- correct_commit: 0 manual labels; agreement unavailable.

## Qwen/Qwen3-4B

200 problems; 8000 responses.

Tutor leak: 56.44% [52.39%, 60.38%].
Ceiling solve (correct commit): 51.75% [46.88%, 56.38%].
Ceiling containment sensitivity: 92.00% [88.62%, 95.12%].
Tutor leak / ceiling solve: 1.09 [0.99, 1.22].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 44.25% [39.62%, 49.00%] | 5.25% [3.62%, 7.25%] | 4.50% [3.00%, 6.25%] | 0.86 [0.75, 0.98] |
| none | boxed | 72.00% [67.50%, 76.38%] | 67.88% [63.00%, 72.38%] | 61.25% [56.12%, 66.00%] | 1.39 [1.26, 1.54] |
| none | final | 47.50% [42.50%, 52.75%] | 53.87% [49.00%, 58.75%] | 45.12% [40.12%, 50.13%] | 0.92 [0.80, 1.05] |
| human | none | 49.88% [44.88%, 54.75%] | 6.88% [4.88%, 8.88%] | 6.38% [4.50%, 8.38%] | 0.96 [0.85, 1.09] |
| human | boxed | 76.00% [71.38%, 80.12%] | 74.62% [70.62%, 78.50%] | 67.12% [62.62%, 71.25%] | 1.47 [1.33, 1.62] |
| human | final | 48.00% [43.00%, 53.00%] | 55.75% [51.00%, 60.75%] | 46.00% [41.12%, 50.88%] | 0.93 [0.81, 1.06] |
| auto | none | 47.75% [43.00%, 52.62%] | 10.12% [7.75%, 12.75%] | 7.75% [5.75%, 9.88%] | 0.92 [0.81, 1.05] |
| auto | boxed | 74.38% [70.12%, 78.38%] | 74.88% [70.88%, 78.50%] | 68.00% [63.38%, 72.12%] | 1.44 [1.30, 1.60] |
| auto | final | 48.25% [43.25%, 53.25%] | 56.12% [51.38%, 61.00%] | 45.88% [41.00%, 50.75%] | 0.93 [0.82, 1.07] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | -1.12 [-5.50, 3.75] |
| H1 | final | -2.75 [-7.50, 1.75] |
| H1 | pooled_demand | -1.94 [-5.75, 2.06] |
| H2 | none | -2.13 [-5.38, 1.37] |
| H2 | boxed | -1.62 [-4.75, 1.38] |
| H2 | final | 0.25 [-3.13, 3.50] |
| H2 | pooled | -1.17 [-3.04, 0.79] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | -0.2310 | 0.0982 | [-0.4235, -0.0385] |
| C(cue, Treatment(reference='none'))[T.auto] | 0.1410 | 0.0652 | [0.0132, 0.2687] |
| C(cue, Treatment(reference='none'))[T.human] | 0.2260 | 0.0744 | [0.0803, 0.3718] |
| C(format, Treatment(reference='none'))[T.boxed] | 1.1755 | 0.0965 | [0.9864, 1.3645] |
| C(format, Treatment(reference='none'))[T.final] | 0.1309 | 0.0907 | [-0.0468, 0.3087] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | -0.0199 | 0.1104 | [-0.2363, 0.1966] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | -0.0178 | 0.1101 | [-0.2337, 0.1981] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | -0.1109 | 0.0966 | [-0.3002, 0.0784] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -0.2060 | 0.1055 | [-0.4128, 0.0008] |

Manual handcheck: 100 sampled, 0 leak labels entered; awaiting_manual_labels.
CSV: `results/stage1/handcheck_Qwen-Qwen3-4B-92e31154.csv`.
- leak: 0 manual labels; agreement unavailable.
- commit: 0 manual labels; agreement unavailable.
- correct_commit: 0 manual labels; agreement unavailable.

## Across-model diagnostics

H3: descriptive_only.
Phase 2 gate passed: False.

Smoke results are descriptive pipeline checks. Handcheck agreement requires real manual labels. All contrasts are unadjusted for multiplicity. See docs/analysis-methods.md for estimands, ceiling sensitivity, and undefined-ratio handling. Full diagnostics and counts are in summary.json.
H3 pooled slope per natural-log parameter count: -0.01 [-0.02, 0.00].
