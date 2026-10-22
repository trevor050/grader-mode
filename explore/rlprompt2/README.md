# RL prompt component ablations

All new code and evidence for this follow-up live here. `templates.json` and `results/inputs.json` were frozen before generation. `run.py freeze` reconstructs the exact source literals via AST, and the CPU check executes only the original loader function to verify the 40-problem order.

- `python3 -m explore.rlprompt2.run freeze`
- `python3 -m explore.rlprompt2.control execute` (paid Vast run; refuses an existing ledger)
- `python3 -m explore.rlprompt2.report`

Qwen: A–I × 40, thinking on, prior Phase 6 publisher FP8 pin, medium effort. Optional Gemma: A/E/F/H × 40, prior Phase 5 BF16 pin, native thinking on. All sampling settings, tutor text, model pins, exact problem rows, and source hashes are in `results/inputs.json`.

C substitutes longest-first `the automated grader|the grader|grader`, case-insensitively, with literal `my teacher`, including uppercase references, without grammatical cleanup. B retains the blank lines after the removed header. D/F remove exactly the specified bullet lines and their newlines. G uses only the original header and generated-problem block plus the requested minimal clause. I appends the requested sentence after a blank line.

Budget: one uniquely labelled owned instance, $1.50 including setup/storage/transfer, $0.15 teardown reserve, independent wall-clock destruction guard. Only that label is eligible for destruction; 57125990 and 57136341 are explicitly excluded. The first 36 rows provide an outcome-blind throughput gate; Gemma gets a separate remaining-budget gate. Raw token IDs, decoded segments, prompts, and generations remain locally in ignored JSONL files; committed hashes identify them.

Scoring reuses the previous final-only scorer. The conflict code is exactly the requested regex, not a human judgment. The report requires complete unique grids, immutable request/model/sampling identity, correct token counts, and verified teardown before producing a table. No truncated rows are excluded.
