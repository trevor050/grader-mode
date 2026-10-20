# Phase 1 results

Problem-cluster bootstrap: 2,000 resamples; 95% percentile intervals; seed 0.

Inference eligibility: False.
- Fewer than 200 problems: smoke runs are ineligible for inference.
- Incomplete six-model Qwen3 ladder: ineligible for inference.

## Qwen/Qwen3-0.6B

10 problems; 400 responses.

Tutor leak: 64.72% [46.39%, 81.67%].
Ceiling solve (correct commit): 37.50% [12.50%, 62.50%].
Ceiling containment sensitivity: 75.00% [52.50%, 95.00%].
Tutor leak / ceiling solve: 1.73 [1.13, 4.22].

| Cue | Format | Leak, 95% CI | Commit, 95% CI | Correct commit, 95% CI | Leak / ceiling, 95% CI |
|---|---|---|---|---|---|
| none | none | 77.50% [57.50%, 95.00%] | 20.00% [7.50%, 35.00%] | 12.50% [0.00%, 27.50%] | 2.07 [1.32, 5.40] |
| none | boxed | 65.00% [45.00%, 82.50%] | 90.00% [77.50%, 97.56%] | 45.00% [25.00%, 67.50%] | 1.73 [1.05, 4.20] |
| none | final | 55.00% [30.00%, 80.00%] | 92.50% [80.00%, 100.00%] | 40.00% [15.00%, 65.00%] | 1.47 [0.93, 3.22] |
| human | none | 65.00% [42.50%, 85.00%] | 25.00% [12.50%, 40.00%] | 12.50% [2.50%, 22.50%] | 1.73 [1.00, 5.00] |
| human | boxed | 60.00% [37.50%, 80.00%] | 82.50% [60.00%, 97.56%] | 35.00% [12.50%, 60.00%] | 1.60 [1.00, 3.80] |
| human | final | 60.00% [37.50%, 80.00%] | 82.50% [60.00%, 97.50%] | 42.50% [20.00%, 65.00%] | 1.60 [1.08, 3.67] |
| auto | none | 67.50% [47.50%, 85.00%] | 27.50% [12.50%, 47.50%] | 12.50% [5.00%, 20.00%] | 1.80 [1.21, 4.25] |
| auto | boxed | 67.50% [45.00%, 87.50%] | 92.50% [77.50%, 100.00%] | 50.00% [25.00%, 77.50%] | 1.80 [1.17, 4.17] |
| auto | final | 65.00% [47.44%, 82.50%] | 95.00% [87.50%, 100.00%] | 57.50% [37.50%, 77.50%] | 1.73 [1.07, 4.67] |

Risk contrasts (percentage-point differences; 95% CI):

| Hypothesis | Contrast | Estimate [95% CI], percentage points |
|---|---|---|
| H1 | boxed | 12.50 [-7.50, 32.50] |
| H1 | final | 20.00 [-2.50, 50.00] |
| H1 | pooled_demand | 16.25 [-1.25, 38.75] |
| H2 | none | 2.50 [-22.50, 25.00] |
| H2 | boxed | 7.50 [-5.00, 20.00] |
| H2 | final | 5.00 [-10.00, 22.50] |
| H2 | pooled | 5.00 [-0.83, 10.83] |

Logistic regression: **ok**; problem-clustered standard errors.

| Term | Log-odds estimate | Cluster SE | 95% CI |
|---|---|---|---|
| Intercept | 1.2368 | 0.5900 | [0.0805, 2.3931] |
| C(cue, Treatment(reference='none'))[T.auto] | -0.5059 | 0.4591 | [-1.4057, 0.3940] |
| C(cue, Treatment(reference='none'))[T.human] | -0.6177 | 0.3475 | [-1.2987, 0.0633] |
| C(format, Treatment(reference='none'))[T.boxed] | -0.6177 | 0.3593 | [-1.3219, 0.0865] |
| C(format, Treatment(reference='none'))[T.final] | -1.0361 | 0.3393 | [-1.7010, -0.3711] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.boxed] | 0.6177 | 0.6128 | [-0.5834, 1.8189] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.boxed] | 0.4041 | 0.4975 | [-0.5710, 1.3793] |
| C(cue, Treatment(reference='none'))[T.auto]:C(format, Treatment(reference='none'))[T.final] | 0.9242 | 0.6322 | [-0.3148, 2.1633] |
| C(cue, Treatment(reference='none'))[T.human]:C(format, Treatment(reference='none'))[T.final] | 0.8225 | 0.4158 | [0.0076, 1.6374] |

Manual handcheck: 100 sampled, 0 leak labels entered; awaiting_manual_labels.
CSV: `results/handcheck_Qwen-Qwen3-0.6B-fb68f5a6.csv`.
- leak: 0 manual labels; agreement unavailable.
- commit: 0 manual labels; agreement unavailable.
- correct_commit: 0 manual labels; agreement unavailable.

## Across-model diagnostics

H3: unavailable.
Phase 2 gate passed: False.

Smoke results are descriptive pipeline checks. Handcheck agreement requires real manual labels. All contrasts are unadjusted for multiplicity. See docs/analysis-methods.md for estimands, ceiling sensitivity, and undefined-ratio handling. Full diagnostics and counts are in summary.json.
