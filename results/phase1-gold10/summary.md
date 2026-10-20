# Phase 1 results

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Inference eligibility: False.
- Fewer than 200 problems: smoke runs are ineligible for inference.

## Qwen/Qwen3-0.6B

169 problems; 6760 responses.

Tutor leak: 53.94% [48.34%, 59.35%].
Ceiling solve (correct commit): 34.17% [29.29%, 39.20%].
Ceiling containment sensitivity: 57.84% [52.36%, 63.46%].
Tutor leak / ceiling solve: 1.58 [1.40, 1.80].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 55.33% [49.11%, 61.39%] | 23.22% [19.23%, 27.51%] | 12.43% [9.17%, 15.68%] | 1.62 [1.41, 1.86] |
| none | boxed | 52.96% [46.89%, 59.17%] | 89.94% [86.69%, 92.90%] | 47.93% [42.16%, 53.99%] | 1.55 [1.36, 1.78] |
| none | final | 52.96% [46.89%, 58.88%] | 96.75% [95.11%, 98.08%] | 51.04% [44.82%, 56.51%] | 1.55 [1.37, 1.78] |
| human | none | 55.03% [49.11%, 60.80%] | 21.89% [18.34%, 25.89%] | 11.69% [9.17%, 14.50%] | 1.61 [1.41, 1.85] |
| human | boxed | 54.29% [48.52%, 60.06%] | 85.80% [81.95%, 89.05%] | 46.89% [41.12%, 52.22%] | 1.59 [1.40, 1.81] |
| human | final | 52.22% [46.01%, 58.14%] | 95.27% [93.05%, 97.19%] | 49.56% [43.34%, 55.47%] | 1.53 [1.33, 1.76] |
| auto | none | 52.51% [46.30%, 58.88%] | 28.40% [24.11%, 32.69%] | 12.28% [9.47%, 15.24%] | 1.54 [1.34, 1.77] |
| auto | boxed | 57.10% [51.03%, 62.87%] | 91.57% [88.76%, 94.08%] | 51.63% [45.71%, 57.25%] | 1.67 [1.46, 1.91] |
| auto | final | 53.11% [46.89%, 58.88%] | 97.63% [96.30%, 98.82%] | 51.04% [44.82%, 56.80%] | 1.55 [1.35, 1.79] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 6.95 [1.63, 12.28] |
| H1 | final | 2.96 [-2.22, 8.14] |
| H1 | pooled_demand | 4.96 [0.22, 9.76] |
| H2 | none | -2.51 [-5.77, 0.74] |
| H2 | boxed | 2.81 [-0.59, 6.22] |
| H2 | final | 0.89 [-3.11, 4.89] |
| H2 | pooled | 0.39 [-1.43, 2.47] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | 0.2138 | 0.1272 | [-0.0355, 0.4631] |
| C(cue, Treatment(reference='none'))[T.auto] | -0.1132 | 0.0753 | [-0.2608, 0.0345] |
| C(cue, Treatment(reference='none'))[T.human] | -0.0120 | 0.0745 | [-0.1580, 0.1340] |
| C(format, Treatment(reference='none'))[T.boxed] | -0.0953 | 0.0851 | [-0.2621, 0.0714] |
| C(format, Treatment(reference='none'))[T.final] | -0.0953 | 0.0800 | [-0.2521, 0.0614] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 0.2806 | 0.1069 | [0.0710, 0.4902] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.0655 | 0.1133 | [-0.1565, 0.2875] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | 0.1191 | 0.1054 | [-0.0875, 0.3257] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -0.0177 | 0.0966 | [-0.2071, 0.1717] |

## Qwen/Qwen3-1.7B

169 problems; 6760 responses.

Tutor leak: 80.90% [76.10%, 85.16%].
Ceiling solve (correct commit): 64.64% [59.17%, 69.97%].
Ceiling containment sensitivity: 78.99% [73.67%, 84.02%].
Tutor leak / ceiling solve: 1.25 [1.18, 1.34].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 80.92% [75.89%, 85.36%] | 34.62% [29.44%, 40.38%] | 28.25% [23.52%, 33.58%] | 1.25 [1.17, 1.35] |
| none | boxed | 79.73% [74.26%, 84.76%] | 97.78% [96.00%, 99.26%] | 78.70% [73.22%, 83.88%] | 1.23 [1.15, 1.32] |
| none | final | 81.07% [76.18%, 85.50%] | 99.26% [98.52%, 99.85%] | 80.77% [75.89%, 85.36%] | 1.25 [1.17, 1.34] |
| human | none | 82.40% [77.37%, 86.99%] | 36.09% [30.47%, 41.86%] | 30.92% [25.59%, 36.54%] | 1.27 [1.19, 1.37] |
| human | boxed | 82.10% [77.22%, 86.54%] | 97.49% [95.27%, 99.26%] | 80.77% [75.74%, 85.36%] | 1.27 [1.19, 1.36] |
| human | final | 81.51% [76.48%, 86.24%] | 98.96% [97.93%, 99.70%] | 80.77% [75.74%, 85.51%] | 1.26 [1.18, 1.35] |
| auto | none | 80.62% [75.44%, 85.36%] | 53.25% [47.93%, 58.88%] | 44.08% [38.61%, 49.85%] | 1.25 [1.17, 1.34] |
| auto | boxed | 81.21% [76.33%, 85.50%] | 97.93% [96.30%, 99.26%] | 80.33% [75.44%, 84.62%] | 1.26 [1.18, 1.34] |
| auto | final | 78.55% [72.93%, 83.58%] | 99.26% [98.52%, 99.85%] | 77.96% [72.19%, 82.99%] | 1.22 [1.13, 1.31] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 1.78 [-2.37, 5.62] |
| H1 | final | -2.22 [-6.21, 1.63] |
| H1 | pooled_demand | -0.22 [-3.40, 2.88] |
| H2 | none | -1.78 [-4.88, 1.33] |
| H2 | boxed | -0.89 [-3.55, 1.63] |
| H2 | final | -2.96 [-6.51, 0.15] |
| H2 | pooled | -1.87 [-3.80, -0.10] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | 1.4446 | 0.1644 | [1.1223, 1.7670] |
| C(cue, Treatment(reference='none'))[T.auto] | -0.0190 | 0.0875 | [-0.1905, 0.1524] |
| C(cue, Treatment(reference='none'))[T.human] | 0.0988 | 0.1071 | [-0.1111, 0.3088] |
| C(format, Treatment(reference='none'))[T.boxed] | -0.0749 | 0.0989 | [-0.2688, 0.1190] |
| C(format, Treatment(reference='none'))[T.final] | 0.0096 | 0.1015 | [-0.1894, 0.2086] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 0.1132 | 0.1269 | [-0.1355, 0.3620] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.0546 | 0.1379 | [-0.2156, 0.3249] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | -0.1372 | 0.1236 | [-0.3794, 0.1051] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -0.0696 | 0.1458 | [-0.3554, 0.2162] |

## Qwen/Qwen3-14B

169 problems; 6760 responses.

Tutor leak: 4.37% [3.14%, 5.72%].
Ceiling solve (correct commit): 57.99% [52.51%, 63.17%].
Ceiling containment sensitivity: 95.56% [92.75%, 97.93%].
Tutor leak / ceiling solve: 0.08 [0.05, 0.10].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 1.63% [0.59%, 2.81%] | 0.15% [0.00%, 0.44%] | 0.15% [0.00%, 0.44%] | 0.03 [0.01, 0.05] |
| none | boxed | 3.55% [2.07%, 5.33%] | 2.07% [1.04%, 3.25%] | 1.63% [0.74%, 2.66%] | 0.06 [0.03, 0.09] |
| none | final | 1.78% [0.89%, 2.81%] | 1.92% [0.89%, 3.11%] | 1.48% [0.59%, 2.37%] | 0.03 [0.01, 0.05] |
| human | none | 3.40% [1.63%, 5.47%] | 0.44% [0.00%, 1.04%] | 0.44% [0.00%, 1.04%] | 0.06 [0.03, 0.10] |
| human | boxed | 5.77% [3.70%, 8.14%] | 2.51% [1.33%, 3.85%] | 2.07% [0.89%, 3.40%] | 0.10 [0.06, 0.14] |
| human | final | 5.92% [3.99%, 7.84%] | 7.10% [5.18%, 9.02%] | 5.33% [3.55%, 7.10%] | 0.10 [0.07, 0.14] |
| auto | none | 2.96% [1.63%, 4.44%] | 0.15% [0.00%, 0.44%] | 0.15% [0.00%, 0.44%] | 0.05 [0.03, 0.08] |
| auto | boxed | 6.51% [4.29%, 9.02%] | 4.44% [2.66%, 6.36%] | 3.99% [2.37%, 5.92%] | 0.11 [0.07, 0.15] |
| auto | final | 7.84% [5.62%, 10.36%] | 7.54% [5.47%, 9.76%] | 6.95% [4.88%, 9.17%] | 0.14 [0.10, 0.18] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 1.63 [-0.89, 3.99] |
| H1 | final | 4.73 [2.51, 7.10] |
| H1 | pooled_demand | 3.18 [1.18, 5.25] |
| H2 | none | -0.44 [-2.07, 1.04] |
| H2 | boxed | 0.74 [-1.18, 2.81] |
| H2 | final | 1.92 [-0.59, 4.44] |
| H2 | pooled | 0.74 [-0.54, 2.07] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | -4.1019 | 0.3505 | [-4.7889, -3.4149] |
| C(cue, Treatment(reference='none'))[T.auto] | 0.6115 | 0.3137 | [-0.0033, 1.2263] |
| C(cue, Treatment(reference='none'))[T.human] | 0.7558 | 0.3147 | [0.1390, 1.3726] |
| C(format, Treatment(reference='none'))[T.boxed] | 0.7999 | 0.3290 | [0.1551, 1.4447] |
| C(format, Treatment(reference='none'))[T.final] | 0.0885 | 0.3482 | [-0.5940, 0.7710] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 0.0258 | 0.4046 | [-0.7672, 0.8188] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | -0.2470 | 0.3676 | [-0.9674, 0.4734] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | 0.9377 | 0.3760 | [0.2007, 1.6746] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | 0.4912 | 0.3964 | [-0.2857, 1.2682] |

## Qwen/Qwen3-32B

169 problems; 6760 responses.

Tutor leak: 5.03% [4.22%, 5.85%].
Ceiling solve (correct commit): 61.83% [56.51%, 67.01%].
Ceiling containment sensitivity: 95.86% [93.05%, 98.22%].
Tutor leak / ceiling solve: 0.08 [0.07, 0.10].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 0.59% [0.00%, 1.33%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.01 [0.00, 0.02] |
| none | boxed | 0.89% [0.30%, 1.63%] | 0.74% [0.15%, 1.48%] | 0.44% [0.00%, 1.04%] | 0.01 [0.00, 0.03] |
| none | final | 13.61% [10.80%, 16.57%] | 19.53% [16.57%, 22.78%] | 13.17% [10.36%, 16.12%] | 0.22 [0.17, 0.27] |
| human | none | 2.66% [1.48%, 3.99%] | 0.59% [0.14%, 1.18%] | 0.59% [0.14%, 1.18%] | 0.04 [0.02, 0.06] |
| human | boxed | 4.14% [2.66%, 5.77%] | 2.37% [1.18%, 3.85%] | 1.78% [0.74%, 2.96%] | 0.07 [0.04, 0.10] |
| human | final | 9.91% [7.69%, 12.13%] | 13.61% [11.24%, 16.12%] | 9.62% [7.54%, 11.83%] | 0.16 [0.12, 0.20] |
| auto | none | 0.59% [0.15%, 1.18%] | 0.15% [0.00%, 0.44%] | 0.15% [0.00%, 0.44%] | 0.01 [0.00, 0.02] |
| auto | boxed | 5.33% [3.55%, 7.25%] | 4.44% [2.81%, 6.21%] | 3.55% [2.07%, 5.18%] | 0.09 [0.06, 0.12] |
| auto | final | 7.54% [5.62%, 9.62%] | 8.73% [6.66%, 10.95%] | 7.40% [5.33%, 9.47%] | 0.12 [0.09, 0.16] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 4.44 [2.81, 6.21] |
| H1 | final | -6.07 [-9.47, -2.81] |
| H1 | pooled_demand | -0.81 [-2.88, 1.26] |
| H2 | none | -2.07 [-3.40, -0.89] |
| H2 | boxed | 1.18 [-0.89, 3.40] |
| H2 | final | -2.37 [-5.03, 0.30] |
| H2 | pooled | -1.08 [-2.37, 0.25] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | -5.1240 | 0.6134 | [-6.3261, -3.9218] |
| C(cue, Treatment(reference='none'))[T.auto] | -0.0000 | 0.7982 | [-1.5644, 1.5644] |
| C(cue, Treatment(reference='none'))[T.human] | 1.5251 | 0.6245 | [0.3011, 2.7492] |
| C(format, Treatment(reference='none'))[T.boxed] | 0.4084 | 0.6187 | [-0.8041, 1.6210] |
| C(format, Treatment(reference='none'))[T.final] | 3.2759 | 0.6064 | [2.0872, 4.4645] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 1.8376 | 0.7888 | [0.2915, 3.3836] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.0487 | 0.7046 | [-1.3323, 1.4297] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | -0.6578 | 0.8001 | [-2.2259, 0.9103] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -1.8841 | 0.6462 | [-3.1506, -0.6177] |

## Qwen/Qwen3-4B

169 problems; 6760 responses.

Tutor leak: 54.93% [50.99%, 59.04%].
Ceiling solve (correct commit): 55.33% [50.00%, 60.36%].
Ceiling containment sensitivity: 91.27% [87.57%, 94.67%].
Tutor leak / ceiling solve: 0.99 [0.90, 1.11].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 42.90% [38.17%, 47.78%] | 5.62% [3.70%, 7.84%] | 4.88% [3.11%, 6.81%] | 0.78 [0.68, 0.88] |
| none | boxed | 71.30% [66.42%, 76.19%] | 68.64% [63.61%, 73.37%] | 61.98% [56.80%, 67.31%] | 1.29 [1.17, 1.43] |
| none | final | 45.86% [40.68%, 51.34%] | 53.55% [48.37%, 58.88%] | 45.56% [40.38%, 51.18%] | 0.83 [0.72, 0.95] |
| human | none | 48.22% [43.20%, 53.55%] | 7.69% [5.47%, 10.21%] | 7.10% [5.03%, 9.62%] | 0.87 [0.77, 0.98] |
| human | boxed | 75.30% [70.71%, 79.88%] | 76.33% [72.19%, 80.47%] | 68.64% [64.05%, 73.22%] | 1.36 [1.24, 1.51] |
| human | final | 45.41% [40.24%, 50.89%] | 54.44% [49.11%, 59.76%] | 44.82% [39.64%, 50.15%] | 0.82 [0.71, 0.94] |
| auto | none | 45.56% [40.68%, 50.59%] | 10.65% [8.14%, 13.17%] | 8.28% [6.21%, 10.65%] | 0.82 [0.73, 0.94] |
| auto | boxed | 73.22% [68.79%, 77.81%] | 75.44% [71.45%, 79.44%] | 68.93% [64.35%, 73.52%] | 1.32 [1.20, 1.47] |
| auto | final | 46.60% [41.27%, 51.92%] | 55.18% [50.00%, 60.36%] | 46.01% [40.82%, 51.33%] | 0.84 [0.73, 0.97] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | -0.74 [-5.77, 4.73] |
| H1 | final | -1.92 [-7.10, 3.25] |
| H1 | pooled_demand | -1.33 [-5.62, 3.11] |
| H2 | none | -2.66 [-6.36, 1.18] |
| H2 | boxed | -2.07 [-5.47, 1.18] |
| H2 | final | 1.18 [-2.52, 4.88] |
| H2 | pooled | -1.18 [-3.50, 1.09] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | -0.2860 | 0.1029 | [-0.4877, -0.0842] |
| C(cue, Treatment(reference='none'))[T.auto] | 0.1080 | 0.0713 | [-0.0317, 0.2476] |
| C(cue, Treatment(reference='none'))[T.human] | 0.2149 | 0.0837 | [0.0510, 0.3789] |
| C(format, Treatment(reference='none'))[T.boxed] | 1.1960 | 0.1051 | [0.9900, 1.4021] |
| C(format, Treatment(reference='none'))[T.final] | 0.1199 | 0.0968 | [-0.0699, 0.3097] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | -0.0120 | 0.1223 | [-0.2516, 0.2276] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | -0.0106 | 0.1180 | [-0.2418, 0.2207] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | -0.0782 | 0.1079 | [-0.2897, 0.1332] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -0.2328 | 0.1157 | [-0.4596, -0.0060] |

## Qwen/Qwen3-8B

169 problems; 6760 responses.

Tutor leak: 55.00% [51.30%, 58.88%].
Ceiling solve (correct commit): 62.13% [56.95%, 67.46%].
Ceiling containment sensitivity: 93.20% [89.79%, 96.15%].
Tutor leak / ceiling solve: 0.89 [0.81, 0.98].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 37.57% [32.54%, 42.75%] | 6.80% [4.73%, 9.17%] | 6.07% [4.14%, 8.28%] | 0.60 [0.52, 0.70] |
| none | boxed | 71.30% [66.27%, 76.04%] | 66.57% [61.68%, 71.30%] | 62.28% [57.25%, 67.16%] | 1.15 [1.05, 1.27] |
| none | final | 48.67% [43.93%, 53.55%] | 60.06% [56.07%, 64.35%] | 47.93% [43.20%, 52.66%] | 0.78 [0.70, 0.89] |
| human | none | 43.93% [39.05%, 48.96%] | 6.21% [4.29%, 8.43%] | 5.18% [3.40%, 7.25%] | 0.71 [0.62, 0.81] |
| human | boxed | 77.07% [72.78%, 81.36%] | 73.08% [69.08%, 77.22%] | 68.79% [64.20%, 73.22%] | 1.24 [1.15, 1.36] |
| human | final | 51.48% [46.60%, 56.21%] | 62.43% [58.14%, 66.42%] | 50.59% [45.56%, 55.62%] | 0.83 [0.74, 0.93] |
| auto | none | 42.60% [37.72%, 47.78%] | 7.40% [5.33%, 9.76%] | 6.66% [4.59%, 9.02%] | 0.69 [0.60, 0.78] |
| auto | boxed | 74.41% [70.12%, 78.85%] | 74.70% [70.71%, 78.55%] | 69.23% [64.64%, 73.82%] | 1.20 [1.10, 1.31] |
| auto | final | 47.93% [43.20%, 52.67%] | 59.02% [54.73%, 63.17%] | 47.19% [42.46%, 52.07%] | 0.77 [0.68, 0.87] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | -1.92 [-6.80, 2.96] |
| H1 | final | -5.77 [-11.54, 0.00] |
| H1 | pooled_demand | -3.85 [-8.58, 0.81] |
| H2 | none | -1.33 [-5.18, 2.66] |
| H2 | boxed | -2.66 [-5.92, 0.59] |
| H2 | final | -3.55 [-7.70, 0.74] |
| H2 | pooled | -2.51 [-4.49, -0.44] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | -0.5077 | 0.1147 | [-0.7324, -0.2829] |
| C(cue, Treatment(reference='none'))[T.auto] | 0.2096 | 0.0851 | [0.0428, 0.3764] |
| C(cue, Treatment(reference='none'))[T.human] | 0.2639 | 0.0781 | [0.1108, 0.4170] |
| C(format, Treatment(reference='none'))[T.boxed] | 1.4178 | 0.1093 | [1.2036, 1.6319] |
| C(format, Treatment(reference='none'))[T.final] | 0.4544 | 0.0959 | [0.2664, 0.6424] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | -0.0524 | 0.1168 | [-0.2814, 0.1766] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.0384 | 0.1150 | [-0.1870, 0.2638] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | -0.2392 | 0.1234 | [-0.4811, 0.0026] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | -0.1514 | 0.1107 | [-0.3684, 0.0656] |

## Across-model diagnostics

H3: descriptive_only.
Phase 2 gate passed: False.

Smoke results are descriptive pipeline checks. Handcheck agreement requires real manual labels. All contrasts are unadjusted for multiplicity. See docs/analysis-methods.md for estimands, ceiling sensitivity, and undefined-ratio handling. Full diagnostics and counts are in summary.json.
H3 pooled slope per natural-log parameter count: -0.00 [-0.01, 0.00].
