# Think analysis and trace coding implementation plan

**Goal:** Prepare the reduced Think ladder analysis and blinded exploratory trace coding without viewing Think outcomes during generation.

**Architecture:** Keep the existing Instruct analysis untouched. A separate Think module validates the reduced A/B grids and reuses the registered scoring and clustered estimands. Trace coding separates blinded extraction and cost-only projection from explicit later submission/report commands.

**Stack:** Python, NumPy/Pandas, pytest, OpenAI Batch API, local tokenizer estimates, persistent local cache and budget ledger.

- [x] Add synthetic tests and reduced-grid analysis in `src/analyze_1c_think.py` and `tests/test_analyze_1c_think.py`. Require exact cells, reject duplicates/partial stages, support absent B explicitly, and preserve paired cluster draws and exclusions.
- [x] Add synthetic tests for blinded extraction, verbatim codebook, deterministic samples, budget/resume behavior, agreement, and deferred outcome reports in `tests/test_trace_coding.py`.
- [x] Implement `src/trace_coding.py` with cost-only dry run, local cache, Batch submission/collection, blinded 100-trace sample, pairwise agreement, and an explicit generation-complete outcome-report gate.
- [x] Record model/pricing sources and runnable commands in `docs/trace-coding.md`. Keep raw traces, batch payloads, labels, keys, and state outside tracked files.
- [x] Run only schema/token operations on existing RLVR Stage A; project 6,000 traces. No judges or outcome analysis on real Think data.
- [x] Run `uv run pytest -q`; rerun Instruct analysis from its scored input and byte-compare all four `summary*` artifacts against saved originals; inspect diff and protected paths.
