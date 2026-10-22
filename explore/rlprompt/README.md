# RL grading-environment framing probe

Exploratory behavioral comparison, not preregistered. `run.py freeze` extracts the three literal prompt templates from the shared exploratory source via AST and reproduces `load_problems(25, 0)`. It never imports the shared exploratory module or alters its files.

Run from the repository root:

```sh
uv run pytest explore/rlprompt/test_run.py -q
uv run python -m explore.rlprompt.run freeze
uv run python -m explore.rlprompt.control execute
uv run python -m explore.rlprompt.run report
uv run python -m explore.rlprompt.verify
uv run python -m explore.rlprompt.visual
```

The controller refuses to reset an existing execution ledger. Do not rerun paid execution without a new explicit authorization and a new isolated ledger/label. It reuses the Phase 6 pinned vLLM image, official FP8 checkpoints, native-template renderer, saved-token decoder, and final-only scorer. Gemma uses the Phase 5 pinned BF16 checkpoint and native channel parser with the same numeric scoring rules. The Phase 6 validation-only row adapter does not turn this study into Phase 6 data.

One uniquely labeled rental loads models sequentially. The first 40 replies are a fixed, balanced subset of the final grid and supply only timing to the gate. No outcomes guide continuation or model selection. An independent local watchdog enforces the rental deadline; every cleanup checks the label and excludes the lead's instance `57125990`. The $1.80 cap includes setup and a conservative transfer allowance, with a $0.15 reserve. Qwen3-32B was dropped from the initial schedule, then reconsidered after measured primary costs. `control optional32` uses a separate ledger with all prior spend carried forward and requires the primary rental to be destroyed first. The first 32B attempt stopped before weight loading on a native-template assertion; `execution-32.json` retains its charge. `execution-32-retry.json` carries that cumulative cost into the retry. The official FP8 fallback pin is in `results/optional32-pin.json`. Gemma is conditional on the remaining timing projection.

All outputs are under `results/`. `summary.md` contains counts and the first three `rl_tutor` reasoning openings per completed model. `inputs.json` freezes prompts, tutor text, sample, and hashes of protected inputs. `implementation.json` records reused source hashes; `execution.json` records pricing, gates, accounting, and final resource state. Full raw/scored JSONL files and the transfer bundle remain local and gitignored. They preserve output IDs and decoded spans for final-only replay. Raw integrity hashes are retained in `verification.json`. `primary-worker.py` is an evidentiary, byte-for-byte snapshot of the primary worker source, not a standalone entry point. The local ignored primary bundle retains its original runnable layout. `verify.py` replays all scores and checks the full grids, protected hashes, pins, and aggregate budget. `visual.py` builds the filterable HTML table from the verified summary JSON.
