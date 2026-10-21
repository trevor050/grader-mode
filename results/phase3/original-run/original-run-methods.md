# Original v2 result packaging, 2026-10-20

The original v2 runner returned `status: failed` with `ValueError: Reduced test grid exceeds GPU budget` after 500 completed development responses and before any test generation. Its manifest, metrics, selection, raw responses and tracking data are unchanged. H12/H13 use the existing `analyze(tracks, None, resamples=2000, seed=0)` API. The full-generation CLI did not certify this incomplete run. Development responses remain retained and excluded from held-out inference. H14 and all seven test-cell rates/CIs/degeneration flags are missing. No source edits were made for partial-run analysis.

Exact elapsed time at the failed gate was not persisted. `budget-plan.json` brackets it between the final completed-development checkpoint and final function metrics, using the original slowest complete development cell, 20% headroom and 300-second reserve. Both one- and two-template projections exceed $7 across that bracket. No test-template count was selected.

The ledger extends wall time through the first observed zero-container timestamp. `controller_ended_unix` preserves the controller return time; reconciled `ended_unix` is the charged-allocation end at the zero-container observation. Later CPU analysis does not extend the bound. The controller log's terminal cost precedes this shutdown reconciliation. Returned function seconds are separate from the experiment completion flag. Valid corrected tracking/development data remain retained despite the pre-test budget-gate status.

The 540-pass full-suite checkpoint predates parent-added resumer modules. The parent owns continuation and merged validation. This worker launched no continuation.
