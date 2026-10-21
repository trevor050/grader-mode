# Phase 5 second-family execution, authorization R4

Stopped before model load. Selected model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at `bee7596271d1495f6992ae224aefde4410e816b8`. No scoring, compliance reading, GLM selection, main generation, commit, or push occurred.

## Attempts and fixes

1. Startup: fresh offer 51954044, reliability 0.9940899, one RTX 5090, `vastai/pytorch:latest`, 80 GB, $0.651851852/hour. Instance 57106643. Failed before worker deployment with provider CLI JSONDecodeError. A later read reproduced HTTP 429, API requests too frequent. Automatic cleanup also failed because the CLI requires `--yes` despite `--raw`. Manual `destroy --yes` confirmed absence. Wall to manual zero: 110.238s. No runtime smoke entered.
2. One fix attempt: added bounded read-only JSON retries, kept create single-shot, and made destroy noninteractive with absence verification. The fresh cheapest qualified replacement was offer 56695172, reliability 0.9935535, $0.688888889/hour with storage. Instance 57107006. SSH became ready after approximately 392s. Remote pip immediately rejected exact `vllm==0.32.0`, listing versions only through 0.31.0. Logs were pulled and instance destroyed; zero confirmed at 416.717s from reservation. No runtime smoke entered.

The exact release exists in the [primary registry](https://pypi.org/pypi/vllm/0.32.0/json), requiring Python >=3.11 and <3.15. The remote Python version and index configuration were not captured, so interpreter compatibility versus registry/index configuration is unconfirmed. No package downgrade or pin change was attempted. The registry receipt is in the ignored fix directory.

Neither failure is an unscored Nemotron load/generate failure. The required two runtime failures for prospective GLM selection do not exist; GLM was not run.

## Data and stage decisions

Smoke: 0/8. Throughput: 0/256. All eight registered main cells: **0/400 complete**. H20 core 800, H21 main 1,600, and H20 controls 800 were all unreleased because no smoke or throughput measurement passed. No stage projection or registered budget drop was possible. Model setup time and generation throughput are unavailable.

There is **no main raw file or main manifest**. Ignored operational artifacts: `/Users/user/codeprojects/grader-mode/results/raw/phase5-second-family-20261021`. Prospective request grids/manifests: `bundle.tar.requests.json`; original and repaired source bundles retain separate hashes. First-attempt evidence is in `startup/`, second in `fix/`. No raw reply exists.

## Accounting and teardown

- Initial credit: $2.8247990723; latest credit: $2.8068864015; provisional provider credit delta: **$0.0179126708**. Billing can post late.
- Compute/storage elapsed estimates plus both complete reserved network envelopes: $0.683115615. This deliberately overstates transfers, including envelopes for downloads that never occurred.
- Conservative ledger: first failed cleanup retains its complete $2.524260 reservation; fix charged $0.371489; Vast total **$2.895749**. No reservation refund/reset was made.
- Prior Gemma remains **$1.452285**; aggregate ledger **$4.348034 / $9**, outstanding **$0**. The $0.30 provider reserve remains intact.
- Latest `vastai show instances --raw`: **[]**. Both detached watchdog processes exited. Modal zero was supplied by the parent and was not independently requeried during this Vast-only task.

Focused verification: **59 tests passed in 16.41s**. Exact fixtures cover empty successful CLI stdout, no create retry, and `destroy --yes` with empty acknowledgement. R3 preflight evidence is preserved under `*-preflight-r3.*`.
