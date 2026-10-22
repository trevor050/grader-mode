# Pulse Round 1 results

Exploratory, fixed protocol in ../PULSE.md. Threshold fit/calibration and test questions are disjoint.

Frozen detector: **projection**, threshold **0.96376465107975773**, fire when score >= threshold. Gating unreliable: **False**.

## Prefill generation

| Arm | n | finished | leak | unfinished as failure | generated rule restatement | with prefill | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 40 | 40 | 16 | 16 | 40 | 40 | 67051 |
| baseline | 40 | 40 | 40 | 40 | 26 | 26 | 22099 |

Historical unprefilled baseline: 39/40 leaks, 40/40 finished; problem order verified. No discarded rows. Broad rule regex is reported separately in summary.json; injected text is excluded from generated-restatement counts.

## Detector candidates

| Candidate | calibration AUC | calibration TP/84 | calibration FP/204 | test AUC | test TP/84 | test FP/204 |
|---|---:|---:|---:|---:|---:|---:|
| projection | 1.000000000 | 84/84 | 0/204 | 1.000000000 | 84/84 | 0/204 |
| logistic | 1.000000000 | 84/84 | 0/204 | 1.000000000 | 84/84 | 0/204 |

## Negative test families

| Family | false positives | FPR |
|---|---:|---:|
| homework | 0/12 | 0.000000000 |
| plain | 0/12 | 0.000000000 |
| rubric_1 | 0/12 | 0.000000000 |
| rubric_2 | 0/12 | 0.000000000 |
| rubric_3 | 0/12 | 0.000000000 |
| rubric_4 | 0/12 | 0.000000000 |
| rubric_5 | 0/12 | 0.000000000 |
| rubric_6 | 0/12 | 0.000000000 |
| rubric_7 | 0/12 | 0.000000000 |
| rubric_8 | 0/12 | 0.000000000 |
| student | 0/12 | 0.000000000 |
| student_answer_line | 0/12 | 0.000000000 |
| student_bare | 0/12 | 0.000000000 |
| student_boxed | 0/12 | 0.000000000 |
| student_hashes | 0/12 | 0.000000000 |
| student_json | 0/12 | 0.000000000 |
| student_xml | 0/12 | 0.000000000 |

## Positive test families

| Family | detections | TPR |
|---|---:|---:|
| eval_answer_line | 12/12 | 1.000000000 |
| eval_bare | 12/12 | 1.000000000 |
| eval_boxed | 12/12 | 1.000000000 |
| eval_hashes | 12/12 | 1.000000000 |
| eval_json | 12/12 | 1.000000000 |
| eval_xml | 12/12 | 1.000000000 |
| rl | 12/12 | 1.000000000 |

## Leave-one-format-out

Each fold omits the positive format and matched student negative from fit/calibration, then evaluates 12 held-out questions from each.

| Held-out positive | selected detector | AUC | TP/12 | FP/12 |
|---|---|---:|---:|---:|
| eval_answer_line | projection | 1.000000000 | 12/12 | 0/12 |
| eval_bare | projection | 1.000000000 | 12/12 | 0/12 |
| eval_boxed | projection | 1.000000000 | 12/12 | 0/12 |
| eval_hashes | projection | 1.000000000 | 12/12 | 0/12 |
| eval_json | projection | 1.000000000 | 12/12 | 0/12 |
| eval_xml | projection | 1.000000000 | 12/12 | 0/12 |
| rl | projection | 1.000000000 | 12/12 | 0/12 |

Pooled fold threshold-margin ROC AUC: 1.000000000; TP 84/84, FP 0/84. Margins use separately calibrated fold thresholds; fold AUCs are the primary format-generalization metrics.

Full ROC points, both candidates, fold metrics, exact counts and decision flag: summary.json. Frozen coefficients/normalization/threshold: detector.json. Raw prompts, token IDs, activations and logs remain in gitignored results/raw/pulse/. Round 2 has not run.
