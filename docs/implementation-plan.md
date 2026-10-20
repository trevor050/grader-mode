# Phase 1 runner implementation plan

The committed DESIGN.md is the approved specification. Work directly on clean main as requested.

1. Add uv project/lock, compact project instructions, tests for exact numeric matching and deterministic 40-request/problem grids. Preserve all three registered files byte-for-byte.
2. Build the fixed seed-0 filtered 200-problem GSM8K manifest; smoke uses its first 10. Retain dataset/model commit hashes and prompt hashes.
3. Implement an explicit-timeout Modal GPU function with per-model GPU config (L40S and RTX-PRO-6000 defaults), offline bf16 vLLM, per-request sampling seeds, cached weights and durable generations. Submit selected models concurrently; pull results before local scoring.
4. Implement scoring and problem-cluster bootstrap/regression/contrasts with synthetic tests; report unidentified fits and unlabeled handchecks honestly.
5. Run only Qwen3-0.6B on 10 problems (400 generations); verify all grid rows, seeds, settings, output files, and stopped Modal app state.
6. Save smoke metrics and clearly qualified full-ladder cost extrapolation. Run local tests, verify protected file hashes, commit in coherent pieces, push main.
