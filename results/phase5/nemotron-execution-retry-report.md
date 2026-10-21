# Phase 5 second-family retry: execution record

Status: collected 2,400 Nemotron main replies on Modal RTX-PRO-6000 under the fresh $2.60 whole-run metered cap. H20 controls remain pending after a prospective runtime gate; no cells were dropped. Scoring, the disclosed blind Codex audit and analysis are complete, but formal inference is pending because two replies fail the existing protocol gate. Zero Phase 5 resources confirmed. Concurrent Phase 6 instance 57119355 is out of scope and preserved. Prior execution reports remain unchanged.

The checkout was pulled to `2f7ac93` before work. This work does not edit DESIGN.md, FINDINGS.md, prompts, model registry or generation worker. The lead concurrently committed Phase 6 notes and an appended registration (`58411eb`, `8af0156`) before Modal launch; the Phase 5 registration text remains unchanged. The second-family note excludes provider/package failures from the Nemotron model-failure allowance. No infrastructure cancellation selects GLM.

## Provisioning diagnosis and repairs

Public PyPI requires Python >=3.11,<3.15 for vLLM 0.32.0. The current `vastai/pytorch:latest` registry configuration advertises Python 3.10. Local cross-platform resolution reproduces rejection under Python 3.10 and succeeds under Linux x86-64/Python 3.12. The destroyed earlier instance's interpreter and index were not recorded, so the current-image evidence explains a reproducible incompatibility without proving that historical host's exact configuration.

The official `vllm/vllm-openai:v0.32.0` amd64 image was inspected through Docker Registry metadata and selected by digest `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9`. Its build uses Python 3.12 and CUDA 13.0. Actual deployment on instance 57111833 confirmed Python 3.12.3, torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, tokenizers 0.23.3 and Jinja2 3.1.6. The three non-vLLM pins were already satisfied. No torch/CUDA/vLLM reinstall or generation-pin change was needed. Vast's SSH launch mode replaces the image's serving entrypoint; onstart records image Python, and the guarded worker is launched separately.

All CLI subprocesses disable internal HTTP retries. Shared, deadline-aware locks space repeated operations by five seconds; bounded read-only backoff handles malformed/empty replies and transport failures. Different endpoints have separate pacing, allowing create immediately after a fresh search. Creates remain single-shot. Explicit API refusal is distinguished from an uncertain/lost create acknowledgement; only the latter requires a full startup-window quarantine. Private mode-0600 diagnostics preserve exit-zero stderr API errors, without copying raw provider payloads to committed logs. Destroy remains noninteractive (`--yes`) and absence is independently queried.

The old `cd ... && nohup ... &` launch left the background AND-list shell holding SSH pipes open. This was reproduced locally. The replacement uses a foreground Python launcher with `subprocess.Popen`, redirected stdin/stdout/stderr, `close_fds=True`, and a new session. A local regression test confirms the launcher exits while its child remains alive.

The user increased the hourly ceiling from $0.70 to $0.76 while retaining the same balance gate and $0.30 reserve. Host verification was an implementation constraint beyond the user's reliability requirement and DESIGN. An explicit optional route also permits `unverified` hosts with reliability >=0.99, never `deverified` hosts. That route requires an on-device check for exactly one RTX 5090, SM 12.0 and >=32,000 MiB VRAM before model download. Fatal container/CDI startup errors now stop immediately; failed machine 58482 is excluded from later scans.

## Retained failure history

- Prelaunch exact-ID scans lost offers 56695170 and 55558458 before reservation. No rental or ledger ticket was created.
- `retry-c01-startup`: create returned no usable acknowledgement. No resource ID appeared, repeated provider queries returned zero, and credit was unchanged. The detached watchdog retained the complete startup quarantine and independently confirmed zero at expiry. Its full $2.506445 reservation remains permanently charged in the conservative ledger.
- `retry-d01-startup`: retained stderr proved HTTP 400 with `no_such_ask`, an unavailable offer rather than an image error. No instance was created. Its earlier full-envelope conservative charge is preserved. Future explicit refusals without a contract count elapsed reservation exposure but no invented model transfer.
- `retry-f02-startup`, instance 57111833: official image and SSH worked; the unscored worker entered Nemotron initialization with CUDA graphs and 256 resident sequences. The SSH launch-shell timeout canceled collection while loading. Artifacts were pulled and zero instances confirmed. There was no Nemotron error or completed smoke; this transport cancellation does not select GLM. The report retains the actual runtime evidence rather than treating it as an independent checkpoint failure.
- `retry-h01-startup`, instance 57113309, machine 58482: the provider failed NVIDIA CDI device injection before SSH/container startup. No worker/model ran. The instance was destroyed immediately after diagnosis; a separate host-error receipt preserves the provider message. Artifact pull failed because the container never became accessible; no generation existed to recover. Zero was confirmed.
- `retry-i01-startup`, instance 57113988, machine 149008: the provider spent the startup window preparing the SSH image; its container log showed DNS resolution failures for `ppa.launchpadcontent.net`. Proxy SSH refused connections and direct SSH timed out. No on-device hardware check or model worker ran. The instance was destroyed; the detached watchdog independently confirmed zero and retained its full $0.866777 reservation. The lead then stopped all Vast iteration.

These failures and charges are retained in `aggregate-ledger.json` and `nemotron-execution-retry-history.json`; no prior charge was reset or refunded. The data path remains smoke (8, unscored), full-batch throughput (256, unscored), H20 core (800), H21 main (1,600), then H20 controls (800), with whole-stage prospective gates and a 20% runtime margin. No compliance outcome is used to release a stage.

## Audit and exploratory comparison

The user authorized the same disclosed Codex self-labeling method used for Gemma, blind to arm, and specified that no human labels will be collected. A completed 40-reply attempt/number/boundary audit is required before formal analysis.

The exploratory count uses `re.search(r'tutor|system prompt|instruction', reasoning, re.I)` on all main-arm, thinking-on, explicit-request rows, including empty reasoning; prefill and pilot rows are excluded. This exact filter reproduces FINDINGS.md: Gemma 309/400, Qwen3-14B 11/400, Qwen3-32B 3/400. Nemotron is 392/400 (98.0%), computed after collection and blind audit. It is exploratory and is not a causal test.

Sources: [PyPI release metadata](https://pypi.org/pypi/vllm/0.32.0/json), [vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), [Vast Docker launch modes](https://docs.vast.ai/guides/instances/docker-environment), [Vast create API](https://docs.vast.ai/api-reference/instances/create-instance). Exact registry receipts and local resolution logs remain in ignored raw storage; compact preflight and regression evidence is committed alongside this report.

## Lead-authorized provider switch

After repeated infrastructure failures, the lead instructed the study to use Modal RTX-PRO-6000 (Blackwell), through the same ephemeral Modal image/Volume path as Gemma. Host/provider is outside the registered model/input/sampling contrast. The original pinned checkpoint, tokenizer, native templates, 4,096-token output cap, seeds, sampling, CUDA graphs and 256 resident sequences remain fixed. The fresh authorization caps **all second-family Modal setup, smoke, measurement, model-level fixes/fallback and main stages together at $2.60 metered**; it forbids exceeding free credit. A $0.30 reserve remains outside prospective generation release. Unaffordable stages are pending, with a dollar shortfall, rather than cell drops.

The previous aggregate ledger is retained unchanged after settlement: $8.699651 conservative exposure, no outstanding reservations. This includes permanently retained uncertain/failed reservations and is not observed provider billing. The new explicit Modal allowance is recorded separately in `nemotron-execution-modal-authorization.json` and the Modal controller ledger; it does not reset or refund prior charges. The provider-switch receipt confirms zero Vast instances and zero unresolved tickets.

Modal pricing checked before launch: RTX-PRO-6000 $0.000842/s, CPU $0.0000131/core/s, memory $0.00000222/GiB/s. The controller uses a conservative bound that includes fixed CPU and RAM. [Modal pricing](https://modal.com/pricing), [supported GPU identifiers](https://modal.com/docs/guide/gpu).

### First Modal infrastructure cancellation

App `ap-nBvdhWqgVQHejlTBZLqAYM` ran for 73.647 seconds at the conservative wall-time bound ($0.073647), then was stopped before any smoke completion. Its log showed that the Gemma image lacked matching FlashInfer cubin/JIT kernels and a CUDA compiler toolchain, so vLLM selected Marlin. The controller's new quantization guard also needed to accept vLLM's correct `modelopt_mixed` name for the publisher's pinned mixed NVFP4/FP8/BF16 checkpoint. The app was canceled to fix those infrastructure issues before generation; it is not a Nemotron checkpoint failure. Artifacts were recovered and zero containers verified. The next image retains Gemma's base and exact package pins, then runs FlashInfer's installed-version-matching cubin installer and the minimal CUDA 13 / SM120f JIT-cache installer before source mounts. This is the architecture-specific equivalent of vLLM's supported `download-kernels` route, avoiding unrelated architecture bundles. All generation pins and prior charges remain fixed.

### Compiler-cache miss before smoke

App `ap-IVAoHS43aaVGabYmPTRrZe` installed the correct FlashInfer 0.7.0.post1 cubin and SM120f/CUDA13 cache packages. The pinned checkpoint loaded (17.82 GiB), but graph profiling requested an XQA wrapper absent from the precompiled cache. FlashInfer attempted JIT compilation and failed because `nvcc`/CUDA_HOME were absent. The app stopped automatically, artifacts were pulled, and zero containers were confirmed. No smoke or main replies completed. This is an infrastructure/toolchain failure, not checkpoint-failure evidence. Its conservative $0.270676 charge remains retained, bringing this Modal authorization's prior exposure to $0.344323.

The compiler-complete image retains all Gemma generation pins and adds Torch's exact `cuda-toolkit==13.0.3` compiler/NVVM/CRT/CCCL extras, Ninja and system C++ build tools. A local Linux/Python3.12 dry resolution rejected toolkit13.0.2 as conflicting with Torch2.13, then accepted13.0.3. The NVIDIA compiler wheel was inspected to verify its `nvidia/cu13` filesystem layout. Image build sets CUDA_HOME, creates the expected lib64 alias and compiles an sm_120 PTX probe before GPU spawn. This guards the newly observed JIT requirement while leaving CUDA graphs and 256 resident sequences unchanged.

The official pinned checkpoint uses weight-only NVFP4 experts and FP8 mixer layers; the actual backend selections (including Marlin for weight-only experts) are recorded rather than described as native W4A4 arithmetic. No activation requantization or checkpoint change was introduced.

### CUDA runtime link alias

The compiler-complete app `ap-4G0MBg715cuSf1TsCDkg1r` passed its CPU SM120 PTX probe, loaded the pinned weights, and successfully compiled the XQA objects. Linking then failed with `cannot find -lcudart`: the official pip runtime wheel ships `libcudart.so.13` but not its unversioned linker alias. The app automatically stopped, committed artifacts were recovered, and zero containers were confirmed. No smoke completion or main generation occurred. Its conservative $0.240339 remains charged (cumulative $0.584662). The image repair adds only the runtime linker alias and extends the CPU preflight to compile/link a shared CUDA library, before GPU startup. This is retained as infrastructure failure, without consuming a checkpoint-failure attempt.

### Successful Modal collection and prospective stop

App `ap-2oiNoayC7HjPSL5cQEahoE` passed the compiler/shared-link preflight, loaded Nemotron and completed the 8-prompt unscored smoke. The same cached engine then completed a balanced 256-prompt throughput measurement, H20 reminder/empty (800), and all four H21 cells (1,600). Main collection has the original 200 problems and two templates in each of six complete 400-row cells. No smoke/measurement reply enters scoring. No model-level failure, fix attempt or GLM fallback was used.

The physical device was NVIDIA RTX PRO 6000 Blackwell Server Edition, SM 12.0. The unchanged NVFP4 checkpoint uses vLLM `modelopt_mixed`; the actual expert backend was Marlin weight-only, with FlashInfer FP8/attention. This is not a claim of native NVFP4 W4A4 execution. CUDA graphs remained enabled (`enforce_eager=False`), max_num_seqs and batch/chunk size 256, model context 8,192 and output cap 4,096. Runtime pins and the frozen source hashes are in `nemotron-execution-modal-runtime.json`, the manifest and authorization ledger.

| Stage | Replies | Output tokens | Generation seconds | Release |
| --- | ---: | ---: | ---: | --- |
| Unscored smoke | 8 | 9,833 | 12.894 | Passed protocol smoke |
| Unscored balanced throughput | 256 | 317,530 | 73.500 | 3,854.75 tok/s using 82.373 s end-to-end control/commit wall |
| H20 reminder + empty | 800 | 1,122,670 | 237.041 | Projected $0.379405; available after reserve $1.315569 |
| H21 complete grid | 1,600 | 1,351,279 | 356.878 | Projected $0.658545; available after reserve $1.062127 |
| H20 neutral + irrelevant | 800 pending | — | — | Unreleased at prospective gate |

The controller refreshes lengths and the minimum observed throughput after each completed stage, without consulting compliance outcomes. Modal projections use 25% headroom plus 15 seconds, fixed $0.001/s cost upper bound (GPU + four CPU cores + 32 GiB RAM), $0.30 reserve and a separate 120-second teardown allowance. H21's later effective speed was 3,342.93 tok/s. Controls projected 567.435 s / $0.567435; monetary availability after reserve was $0.657895, but the conservative work deadline had only 517.557 s left. Thus the runtime allowance shortfall was 49.877 s, equivalent to **$0.049877** at the cost bound. The monetary gate itself passed. This is not a request for account credit: the deadline came from the cap, prior retained setup exposure, reserve and teardown envelope. Gates and all unreleased cells remain recorded in `nemotron-execution-modal-gates.json`. No partial controls or replacement cells ran.

The lead requires unaffordable stages to remain pending, so the collection manifest says `deferred_stages=[h20_controls]` and `dropped_cells=[]`. To analyze the completed prefix, the existing analyzer requires `--allow-cell-drops` and emits its older `cell_drop_stage_2`/`dropped_cells` vocabulary. These fields validate the six complete cells, not a decision to abandon controls. Full H20's three-comparison joint criterion remains unavailable.

### Scoring and audit

`score_5` scored 2,400 main requests. A separate Codex gpt-6.1-sol agent manually read only the seed-0 blind raw-protocol sample and chose final segments, attempt labels and committed numbers before the parent ran outcome analysis. The parent verified all 40 IDs, hashes and segment reconstruction. Method: **Codex, self-labeled blind to arm at the user's explicit request; no independent human labels**, exactly as disclosed for Gemma. No human labels will be collected. The task's authorization, author, method and hashes are retained in `nemotron/audit-method.json`.

Attempt agreement: 35/40 (87.5%; κ=0.710983), four false positives and one false negative. Numeric agreement: 10/11 (90.91%). Final-boundary agreement: 40/40 (100%). Five attempt and one number disagreements are retained without changing labels/parser after comparison. Audit completion records coverage, not perfect parser validity; no accuracy threshold was registered. The CLI's default two-family 80-label headline was replaced by the same library's explicit single-model-scope agreement, preserving the 40-row Nemotron target.

Two thinking-off explicit requests (`p0101:t0`, `p0303:t1`) emitted `</think>` despite initial channel `final`; the latter also hit the output cap. Both produce `unexpected_final_boundary`, remain in the all-request denominator and have empty parsed final replies. Their metadata is in `nemotron/protocol-failures.json`. **The unchanged protocol-failure inference gate remains closed.** The completed smoke did not exhibit this behavior; successful load/generation does not select the startup fallback merely because later requests violate the protocol. No rerun or parser change was selected after seeing outcomes.

### Descriptive results

All CIs use the existing 2,000 paired problem-cluster bootstrap, seed 0, percentile 95%, pairing both templates and cells. Violation is leak or numeric attempt in the final reply. The primary all-request denominator retains cap-hit, no-final and protocol-failure requests.

- Explicit violations: thinking off 172/400, 43.00% [37.75%, 48.00%]; thinking on 34/400, 8.50% [5.25%, 12.00%].
- Plain violations: off 48/400, 12.00% [8.50%, 15.75%]; on 17/400, 4.25% [2.00%, 6.75%].
- H21 interaction `(on−off) explicit − (on−off) plain`: −26.75 pp [−33.25, −20.50]. Formal decision pending.
- H20 empty 41/400, 10.25% [6.75%, 14.00%]; reminder 27/400, 6.75% [4.00%, 10.00%]. Empty minus reminder +3.50 pp [+0.50, +6.50]. Formal decision pending; full joint unavailable without controls.
- Gold ≥10 sensitivity (169 problems): H21 −28.99 pp [−34.92, −23.08]; empty minus reminder +1.48 pp [−1.18, +4.14]. Also descriptive.

Exploratory thinking-on explicit trace regex: Nemotron 392/400 (98.0%), versus Gemma 77.25%, Qwen3-14B 2.75%, Qwen3-32B 0.75% in FINDINGS.md. The exact filter/pattern/denominator are recorded in `nemotron/exploratory-trace-regex.json`. Counts reflect lexical mentions, not a demonstrated reasoning mechanism. DESIGN.md and FINDINGS.md were not edited.

### Spend and scoped teardown

Final provider snapshots remain provisional. The new Modal authorization's monthly metering delta is **$1.29170640**, including setup and canceled attempts; billed cost remains **$0** with free compute credit left. Its conservative ledger retains **$1.656979**, below $2.60. Attempt charges are $0.073647, $0.270676, $0.240339 and $1.072317. No charge was reset or refunded. Prior legacy conservative exposure is $8.699651; the combined $10.356630 is conservative accounting including retained reservations, not actual provider billing.

Vast Phase 5 accounting freezes at the pre-Phase-6 provider-switch receipt: credit $2.7149187927. Retry delta is about **$0.09196761**, total Phase 5 Vast delta about **$0.10988028**. Later external credit/top-up and concurrent Phase 6 usage cannot be attributed to Phase 5. Total observed Phase 5 provider cost, including the earlier Gemma/pilot run and all Phase 5 Vast, is **$2.65387126**, provisional. Full before/after snapshots and retained prior accounting are in `final-accounting.json`.

All four second-family Modal apps are stopped with zero tasks, and `modal container list --json` returns `[]`. No Phase 5-labelled Vast instances remain. **Phase 6 instance 57119355 (`phase6-3.8-a1-20261021`) is running out of scope and was never touched or destroyed.** The user explicitly requested scoped zero confirmation while Phase 6 continues. `nemotron-execution-final-resources.json` preserves that receipt; no global-zero claim is made.

### Reproduction and checks

Raw main generations, scored rows and complete remote artifacts remain in ignored `results/raw/phase5-nemotron-modal-20261021-d` and the Modal results Volume. Hashes are in the compact collection receipt. The commands were:

```sh
uv run python -m src.score_5 --input results/raw/phase5-nemotron-modal-20261021-d/phase5-nemotron-modal-20261021-d-startup/phase5-nemotron-modal-20261021-d-startup/main/generations.jsonl --output results/raw/phase5-nemotron-modal-20261021-d/scored.jsonl
uv run python -m src.analyze_5 --input results/raw/phase5-nemotron-modal-20261021-d/scored.jsonl --problem-manifest results/raw/phase5-nemotron-modal-20261021-d/phase5-nemotron-modal-20261021-d-startup/phase5-nemotron-modal-20261021-d-startup/main/manifest.json --output-dir results/phase5/nemotron --model-scope nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --allow-cell-drops --labels results/phase5/nemotron/audit-labels.jsonl
```

The renderer supplies detailed cells/components, then collection/accounting/audit-method notes are added to `nemotron-summary.md`. The analyzer JSON remains unchanged in `nemotron/summary.json`; root `nemotron-summary.json` adds provenance and pending-stage semantics. Parser agreement uses the same deterministic exports with `model_scope` set to the Nemotron canonical ID. The audit's manual labels remain separate from scorer predictions.

Validation before collection: 329 Phase 5/Modal tests passed. After the final image-only CUDA runtime link alias, all 33 Modal controller tests passed, CPU image PTX/shared-link probes passed, and the live 8-prompt smoke passed. Collection/audit verification independently checked the original 200-problem manifest, six balanced 400-reply cells, no pilot/control contamination, saved source hashes and 40 manual boundaries. Prior execution reports and registered inputs remain preserved.

Final pre-commit verification reran the full scoped Phase 5 + Modal suite: **329 passed in 63.37 s**, saved in `nemotron-execution-final-tests.log`. `git diff --check`, protected-input comparisons and the staged secret-pattern scan passed.

Independent final evidence review (Codex gpt-6.1-sol, high, read-only) found no material discrepancies after recomputing the 2,400 rows, counts/CIs, audit, regex and accounting. A fresh `score_5` replay was byte-identical. CLI log trailing whitespace was normalized before commit; raw provider artifacts remain unchanged in ignored storage.
