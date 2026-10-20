# Phase 1c execution summary

Status: budget_stop_after_calibration.
Gold minimum: 10.

Projected total GPU cost: $42.9680; approved limit: $10.00.
Main-run responses: 0. Main-run problems: 0.
Measured GPU function time: 1904.663 seconds; estimated GPU cost: $1.0317.
Budget-accounted time: 1922.022 seconds; cost estimate: $1.0411 (historical function clocks plus new attempt wall clocks).
H6–H8, common-ceiling contrasts, and exploratory main-run estimates: unavailable.
Calibration observations in analysis: 0. Smoke observations in main analysis: 0.

| Ladder | Main tutor prompt decision |
|---|---|
| instruct | prompts/system_tutor_strong.txt |
| think | prompts/system_tutor.txt |

| Diagnostic stage | Ladder | Responses | Truncated | Empty finals | Truncated empty |
|---|---|---|---|---|---|
| smoke | instruct | 156 | 35 | 0 | 0 |
| smoke | think | 156 | 71 | 54 | 54 |
| calibration | instruct | 80 | 12 | 0 | 0 |
| calibration | think | 80 | 40 | 29 | 29 |

Calibration: [calibration.json](calibration.json).
Projection: [budget-projection.json](budget-projection.json).
Smoke diagnostics: [separate summary](../phase1c-smoke/summary.md).

Sequential GPU app lifetime upper estimate: 1969 seconds; $1.0665. CPU/RAM/storage excluded; costs are estimates, not a Modal invoice.

Phase 1c live containers: 0. Phase 1c nonstopped apps: 0. Tests passed: 187.

[Execution and truncation metrics](execution-metrics.json); [container cleanup](modal-containers-after.json).
