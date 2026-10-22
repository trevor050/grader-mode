# Pulse Round 1 status

- Before rental: credit $3.845357334199992; no live Vast instances. Existing unrelated edits: explore/attrib2.py, explore/rlprompt3/, notes/jpp-lens.md; preserved.
- Protocol: explore/PULSE.md, to be committed and pushed before any rental.
- 2026-10-22 05:44:16 UTC: rented A100 SXM4 80GB instance 57166538, 150GB disk; quoted total $1.1083333333/hour, ingress $0.0053333333/GB, egress $0.008/GB, reliability 0.992706. Protocol already pushed at 7734fe6. Persistent local guard uses $1.20/hour plus $0.45 network reserve; destroys below $0.40 or >900s without progress. No top-ups configured.
- 05:47:24 UTC: container running; setup launched via nohup. Container reached running approximately 3 minutes after rental. Scripts preserve all 864 planned capture conditions and exact generation token IDs.
- 05:48:51 UTC: Python 3.12 environment installed vLLM 0.32.0, Transformers 5.19.0, Torch 2.13.0+cu130; model download started. Last 05:50:33 UTC credit snapshot $3.7597377856999934, conservative remaining $3.26963499155. Local manifest and full synthetic analysis/gate checks passed; synthetic data are isolated under results/raw/pulse/analysis-smoke and are not experiment results.
- 05:54:35 UTC (619s after rental): credit $3.533081525699995, conservative remaining $3.18903866321. Download still progressing (42GB cache observed). Prompt rendering ends exactly in `<think>\n`. An early local-only tokenizer check before tokenizer assets arrived returned an empty encoding; added an explicit nonempty-tokenization assertion before model loading. No generations or activations collected yet. Analysis/helper use deterministic float64 row reductions; scalar and batch scores agree bitwise in the isolated synthetic fixture.
- 05:55:42 UTC (686s after rental): model snapshot download completed in 410.6953437328s; cache 52GiB. Tokenizer assets now encode nonempty token IDs, expected thinking opener verified. vLLM model load started 05:55:54 UTC.
- 05:58:42 UTC: vLLM initialized after 168.8424465656s including checkpoint multimodal warmup; 40-prompt prefill generation began. Most recent guard credit snapshot (2026-10-22T05:58:36Z): $3.3300874156999924; conservative remaining $3.108754830185023.
- 06:02:51 UTC (1115s after rental): prefill completed 40/40, finished 40/40, leaks 16/40; 67,051 generated tokens in 249.0749955177s. Fresh baseline permitted by fixed cost gate (projected $0.0830, conservative credit >$1.50) and started immediately. Prefill raw outputs pulled locally.
- 06:05:30 UTC (1274s after rental): fresh unprefilled baseline completed 40/40, finished 40/40, leaks 40/40; 22,099 generated tokens in 158.6397280693s (historical comparator was 39/40). Capture hooks installed successfully at 18 layers; 60 primary coordinates plus full vectors at layers 16/24/32. Initial capture progress: 108/864 in 14.0s, all shape/token/finite-value checks passing.
- 06:07:19 UTC (1383s after rental): all 864 planned captures and three full-layer supplements completed in 109.4841473103s. All hooks removed.
- 06:08:38 UTC: all remote output SHA256 hashes match local copies; primary activation matrix 864x60, supplements each 864x17408. Instance 57166538 destroyed; show instances returns []. Credit snapshot $3.1466717856999935, initial-minus-current delta $0.6986855485 (billing snapshot, not a final invoice). Round 2 cannot start in this assignment, so the instance was destroyed despite credit >$1.50.
- Local analysis: projection selected by the fixed calibration rule. Frozen threshold 0.9637646510797577; calibration and test each AUC 1, TP 84/84, FP 0/204. Each of seven leave-one-format-out folds: AUC 1, TP 12/12, FP 0/12. Each negative test family: 0/12 false positives. No threshold tuning on test data.

## Stage credit snapshots

Credit is the nearest 45-second guard snapshot to each boundary, not an instantaneous invoice. Exact guard history is in results/raw/pulse/budget.jsonl.

| Boundary UTC | Stage | Credit snapshot UTC | Credit | Elapsed from rental (s) |
|---|---|---|---:|---:|
| 05:44:16 | rental | 2026-10-22T05:44:18Z | 3.845357334200 | 0.0 |
| 05:48:51 | install complete | 2026-10-22T05:48:57Z | 3.808088085700 | 274.7 |
| 05:55:42 | download complete | 2026-10-22T05:55:23Z | 3.533081525700 | 685.7 |
| 05:58:42 | model_loaded | 2026-10-22T05:58:36Z | 3.330087415700 | 865.7 |
| 06:02:51 | generation_complete prefill | 2026-10-22T06:02:37Z | 3.254612105700 | 1114.7 |
| 06:05:30 | generation_complete baseline | 2026-10-22T06:05:50Z | 3.207818545700 | 1273.7 |
| 06:07:19 | capture_complete | 2026-10-22T06:07:26Z | 3.171031905700 | 1382.7 |

Teardown snapshot: 2026-10-22T06:08:38Z, credit $3.146671785700, no instances. Total wall time through this snapshot: 1461.7s. See execution.json for exact stage events and ROUND2.md for the prospective run list/cost range.

- Final verification 2026-10-22T06:11:22Z: credit $3.098396945700; no instances. Posted credit delta $0.746960388500, still a billing snapshot. Guard exited after recording instance absent. All original source/baseline hashes preserved; unrelated checkout edits untouched. Full analysis reproduces summary/detector/report byte for byte; scalar/batch saved-gate scores agree on all 864 real captures. Round 2 not executed.

## Round 2

- Authorized 2026-10-22. Frozen detector and Round 1 raw outputs reused; no refitting. Priority override: target/random K=64 first, target K=16/256, random K=16/256, then side effects.
- 2026-10-22 06:17:58 UTC: rented instance 57170351, A100 PCIE 80GB, 150GB disk, quoted $0.9916666667/hour including disk; ingress $0.0026041667/GB, egress $0.00390625/GB. Initial credit $3.098396945700. Independent 45-second guard uses $1.10/hour plus $0.30 reserve; hard stop $0.40; destroy on >900s stall. No other instances at rental.
- Implementation controls pushed before rental at a701c91. Setup in progress; no generation yet. Raw Round 2 evidence: results/raw/pulse2/.
- 2026-10-22 06:35:27 UTC: gate capture 120/120 complete: RL/tutor 40/40 fire, plain/no-system 0/40, student/tutor 0/40. Closing-think and both EOS IDs retained in forced-token infrastructure checks. Target K=64 completed and random K=64 running; selection rule fixed before comparing the first complete target/random pair. Operationalization for the optional added arm: among targets that reduce finished leaks versus random+baseline without reducing finish count, choose minimum unfinished-as-failure; ties K=64,16,256; if none qualify use K=64. This follows the protocol's promising-pulse criterion; all K counts will be reported.
- 2026-10-22 06:47:21 UTC: all six RL pulse cells complete and locally validated, no discarded rows. Target leak K=16/64/256: 39/33/17 of 40; random: 40/40/40 of 40; every cell finished 40/40. Plain/student fresh side-effect baselines and deterministic gate-bypass paths still in progress. First K=64 pair projected remaining registered work at $0.535634 under the conservative guard; credit snapshots are recorded after each job.
- 2026-10-22T06:53:12Z: registered list complete and validated: all six RL cells, 80 fresh side-effect baselines, 14 gate-aware side-effect paths. Plain correct 39/40; student leak 1/40; both finished 40/40 and gates 0/40. Added-after-Round-1 K=256+reminder amendment recorded in PULSE.md before generation; credit $2.435644711100.
- 2026-10-22T07:01:39Z: Round 2 complete. Added K=256+reminder: finished 37/40, leaks 7/40 (7/37 finished), numeric boxed 13/40, generated rule restatement 23/40; unfinished-as-failure 10/40. IDs 254 and 710 hit 4096 tokens; ID 316 stopped on EOS without closing-think, 2071 tokens. All rows retained. All 16 side-effect intervention paths (including added combination) exactly reuse the fresh baselines under 0/40 gates per condition.
- Teardown: instance 57170351 destroyed 2026-10-22T06:58:26Z; verified with vastai show instances = []. All 61 remote output hashes and log hashes match local copies. No job failures or paid retries; transport issues are explicit in execution2.json. Latest posted credit snapshot $2.336406649100 at 2026-10-22T06:59:10Z, provisional billing. Full analysis complete; final reports and section 12 prepared.
- Final account/resource refresh 2026-10-22T07:03:38Z: credit $2.305467572100; vastai show instances = []. Full raw hashes, unique-job ledger (25 commands, 9 GPU behavior jobs), CPU contracts and deterministic full analysis all verified.

## Round 3, 2026-10-22
- Prospective closed-loop amendment pushed at `01e53c6` before rental; runner at `950ec47`. Saved unsteered baseline replay, median-midpoint monitor threshold, C=32 retained-token chunks, global on/off group calls, no hysteresis, random's own trajectory, active monitoring of prompt-negative side effects.
- Rented instance 57180052, offer 47546409, A100 PCIE 80GB, 150GB disk, quoted $0.9916666667/hour including disk. Initial credit $2.3054675721. Independent 45-second guard uses $1.10/hour + $0.30 reserve; hard floor $0.40 and >900s progress stall destruction. Setup in progress; no calibration/behavioral results yet.
