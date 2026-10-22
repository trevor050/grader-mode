# Closed-loop drift steering: Round 3

**INCOMPLETE.** Calibration completed; a controller bug destroyed the instance during the target arm. No complete 40-row behavioral cell exists. The saved checkpoint contains 33 completed finals and seven interrupted sequences. Random, side effects and reminder were not run.

Prospectively fixed exploratory amendment `01e53c6`, pushed before rental. No additional behavioral arms. Operational projection changes are labeled below. Calibration precedes every steered run; Round-1 detector and shifts unchanged. Same 40 problems/order, model revision, bf16, thinking on/medium and retained 4096-token cap as Round 2.

| Arm | finished | leak | unfinished=failure | numeric boxed | generated rule | pushed tokens | on chunks | switches | retriggered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round-2 fixed target K=256 | 40/40 | 17/40 | 17/40 | 18/40 | 31/40 | 9235 | — | — | — |
| closed_loop_target, saved partial checkpoint | 33 observed; 7 interrupted | 27 observed / 33 completed | not a model finish endpoint | 27 observed | 35 observed prefixes | 15295 at checkpoint | 489 | 568 | 40 observed |

For complete cells, leak is gold-number membership in a finished final, denominator all 40. The partial row reports observed completed replies only; seven missing finals are not scored as non-leaks, and no full-cell rate is estimated. Boxes may contain wrong numbers. Generated restatements exclude injected reminder. All retained incomplete/failed records are preserved.

## Calibration and signal evolution

Teacher-forced saved unsteered Round-1 RL baseline and Round-2 student baseline IDs. RL chunks **499**, student chunks **260**; median scores **-1.251343858 / -1.430876200**. Frozen midpoint threshold **-1.3411100290153271**. In-sample ROC AUC **0.604424233**, TP **276/499**, FP **115/260**. High-score orientation valid: **True**. Chunk labels denote prompt condition, not annotated drift moments. Full ROC in calibration3.json; correlated chunks and reused evaluation questions do not constitute held-out monitor validation.

| reasoning chunk (0-based) | RL n | RL median | student n | student median |
|---|---:|---:|---:|---:|
| 0 | 40 | -1.174427 | 40 | -2.165174 |
| 1 | 40 | -1.387438 | 40 | -1.316627 |
| 2 | 40 | -1.279000 | 40 | -1.319203 |
| 3 | 40 | -1.351528 | 39 | -1.622392 |
| 4 | 40 | -1.330715 | 31 | -1.391796 |
| 5 | 40 | -1.219100 | 20 | -1.156803 |
| 6 | 39 | -1.081657 | 9 | -1.104828 |
| 7 | 37 | -1.073902 | 5 | -1.145815 |
| 8 | 29 | -1.208389 | 4 | -1.499690 |
| 9 | 22 | -1.107751 | 3 | -1.123929 |
| 10 | 19 | -1.064926 | 3 | -1.448776 |
| 11 | 14 | -1.424821 | 3 | -1.085232 |
| 12 | 10 | -1.234346 | 3 | -1.443066 |
| 13 | 8 | -1.254121 | 3 | -1.849316 |
| 14 | 8 | -1.241372 | 2 | -1.548528 |
| 15 | 8 | -1.265281 | 2 | -1.271059 |
| 16 | 6 | -1.145491 | 2 | -1.433595 |
| 17 | 4 | -1.591667 | 2 | -1.248871 |
| 18 | 3 | -1.470046 | 2 | -0.793954 |
| 19 | 3 | -1.524747 | 2 | -0.891793 |
| 20 | 1 | -1.518198 | 2 | -1.231890 |
| 21 | 1 | -1.456009 | 2 | -1.295175 |
| 22 | 1 | -2.069981 | 1 | -1.228006 |
| 23 | 1 | -0.969316 | 0 | — |
| 24 | 1 | -1.342521 | 0 | — |
| 25 | 1 | -1.351437 | 0 | — |
| 26 | 1 | -1.025730 | 0 | — |
| 27 | 1 | -1.506788 | 0 | — |
| 28 | 1 | -1.258373 | 0 | — |
| 29 | 1 | -1.135727 | 0 | — |
| 30 | 1 | -1.435127 | 0 | — |
| 31 | 1 | -1.101823 | 0 | — |
| 32 | 1 | -1.867742 | 0 | — |
| 33 | 1 | -1.557961 | 0 | — |
| 34 | 1 | -1.754991 | 0 | — |
| 35 | 1 | -1.350939 | 0 | — |
| 36 | 1 | -1.370174 | 0 | — |
| 37 | 1 | -1.233291 | 0 | — |
| 38 | 1 | -1.847505 | 0 | — |
| 39 | 1 | -0.561201 | 0 | — |
| 40 | 1 | -1.949245 | 0 | — |
| 41 | 1 | -1.275789 | 0 | — |
| 42 | 1 | -1.151800 | 0 | — |
| 43 | 1 | -2.068798 | 0 | — |
| 44 | 1 | -1.168558 | 0 | — |
| 45 | 1 | -1.176533 | 0 | — |
| 46 | 1 | -2.184256 | 0 | — |
| 47 | 1 | -1.168441 | 0 | — |
| 48 | 1 | -1.407652 | 0 | — |
| 49 | 1 | -1.195575 | 0 | — |
| 50 | 1 | -1.560614 | 0 | — |
| 51 | 1 | -1.144415 | 0 | — |
| 52 | 1 | -1.497329 | 0 | — |
| 53 | 1 | -1.106134 | 0 | — |
| 54 | 1 | -3.191748 | 0 | — |
| 55 | 1 | -1.763855 | 0 | — |
| 56 | 1 | -1.591069 | 0 | — |
| 57 | 1 | -0.893481 | 0 | — |
| 58 | 1 | -2.122301 | 0 | — |
| 59 | 1 | -0.769215 | 0 | — |
| 60 | 1 | -1.727243 | 0 | — |
| 61 | 1 | -1.191099 | 0 | — |
| 62 | 1 | -1.588974 | 0 | — |
| 63 | 1 | -1.197679 | 0 | — |
| 64 | 1 | -1.938385 | 0 | — |
| 65 | 1 | -1.569725 | 0 | — |
| 66 | 1 | -0.761492 | 0 | — |
| 67 | 1 | -1.585640 | 0 | — |
| 68 | 1 | -1.409044 | 0 | — |

RL chunks mentioning grader/rules: **139** across **39** traces; **67/138** have a higher score than the preceding chunk. Mean adjacent change **-0.109507**. This phrase-based description was not used to tune threshold. Chunk text and per-trace trajectories are retained in ignored calibration-chunks.jsonl; indexed score evolution/mention summaries are in calibration3.json.

Grader-only subset: **63/132** adjacent scores rise; mean change **-0.130102**. The two original-prompt mapping projection errors were **0.009408 / 0.029478**, within the prospectively coded 0.03 tolerance, not exact activation equality.

## Interrupted target evidence

The saved checkpoint has **33** EOS-completed finals, **27** observed leaks and **27** numeric boxes. Generated rule restatement: **29/33** completed replies and **35/40** saved reasoning prefixes. Seven prefixes remain at 1344 reasoning tokens each. These are infrastructure interruptions, not model length-limit failures; no full-cell leak/finish rate or unfinished-as-model-failure statistic is estimated.

Validated checkpoint: **989** reasoning chunks, **489** on chunks, **15295** pushed reasoning tokens, **568** switches, re-triggers in **40/40** sequences and **392023** hooked prefix tokens. Per-sequence counts are in summary3.json.

Seven newer chunk records contain 96 additional pushed reasoning tokens. They are preserved separately rather than silently merged with the older checkpoint. At least 15,391 pushed tokens are recorded overall; 15,295 is the consistent-checkpoint total. No exact complete-run exposure is claimed.

The retained partial target already has 27 finished leaks and more pushed tokens than Round-2 K=256 (17/40 leaks; 9235 pushed tokens). This is descriptive across different chunk/seed-reset mechanics, not a completed paired target/random efficacy test.

## Feedback mechanics and limits

Before each C=32 retained-token reasoning chunk, use the original prompt gate initially and the previous chunk’s last up-to-8 reasoning-token mean thereafter. No hysteresis; negative prompt gates still monitor. Target/random share threshold and policy, while random follows its own trajectory. Global hooks require separate on/off batched calls. One discarded lookahead sample makes the final retained token observable; it is never continued or scored. Closing-think/EOS stops steering before finals, which continue unsteered.

Each call recomputes the full prefix with seed reset to 0 and prefix caching off. Push-on also shifts that recomputed prefix, so retained pushed-token totals exclude potentially much larger repeated prefill exposure. Readings are before each local shift, but earlier-layer shifts contaminate later-layer measurements. These mechanics differ from fixed K=256; token-budget and leak comparisons are descriptive.

| Arm | retained tokens | sampled tokens | discarded lookahead | hooked prefix tokens | seconds |
|---|---:|---:|---:|---:|---:|
| target, partial checkpoint | 39180 | 40136 | 956 | 392023 | 650.20 |

## Side effects and missing arms

student_closed_loop_target: not launched because the controller failed during target. Baseline equality and benign-path preservation are unmeasured.
plain_closed_loop_target: not launched because the controller failed during target. Baseline equality and benign-path preservation are unmeasured.
closed_loop_target: not completed.
closed_loop_random: not completed.
closed_loop_target_reminder: not completed.
Partial arms: closed_loop_target. Checkpoints and chunk records retained; these are not complete 40-row cells.

## Execution and accounting

Instance **57184025**; initial credit **$2.305467572100**; final posted credit **$0.936804900600**, provisional snapshot. Hard floor $0.40, independent 45-second watchdog, >15-minute stall destruction. End reason: `controller_error`.

Destroyed **2026-10-22T08:58:36Z**; verified **2026-10-22T08:58:37Z** with `vastai show instances --raw`: `[]`. Credit was checked after calibration and repeatedly during target. Exact snapshots, projections, failures, hashes and teardown are in execution3.json. Retained IDs/features/events, partial decoded replies and earlier startup logs remain in ignored results/raw/pulse3/. The final instance's full setup/run logs were not retrieved because the fatal fetch check aborted that retrieval block before teardown.

Failures: initial V2 metadata mismatch; a failed create left a stopped contract that was destroyed; internal/external request-ID mismatch; then a resumed local dispatcher treated a changing chunk-file hash as fatal and destroyed the active target. All setup costs are retained in lifetime accounting. The last error was avoidable controller behavior, not a model or budget-floor failure.

Added operational changes: runner/ID compatibility repairs before calibration; measured target forecast before calibration; random work forecast revised while target was active, before reading target leak counts or launching random. The threshold, steering policy, sample, token cap and arms were unchanged. The random forecast was never exercised because the controller failed first.

The post-failure fetch repair separates immutable completed-job artifacts from changing progress snapshots. Eighteen CPU contracts pass; this final repair has not been GPU re-run. No further rental was launched because a fresh measured setup plus at least the observed 650s target work, 600s teardown margin and $0.40 floor exceeds remaining credit.

Verification: CPU boundary/calibration/monitor-position/control-stat contracts; original-prompt activation mapping check against the frozen capture; analysis recomputes scores, chunk control transitions, token cap, order, stats and final-only measures. Any failures or deviations are listed in execution3.json.
