# grader-mode

- Phase 1/1b pre-registration: `DESIGN.md`; never change it or anything in `prompts/` during implementation/data collection.
- Local Python: `uv sync --dev`, `uv run pytest`. GPU dependencies exist only in the Modal image.
- Pipeline: build prompts → `modal run -m src.modal_app` → score → analyze. Use `uv run` so CLI and package versions match.
- Only explicitly approved GPU runs. Phase 1b authorization (2026-10-20): 4B/5-problem smoke, then 4B/8B/14B/32B × 200 under run ID `phase1b-full` only if smoke-based projection is <= $15. No warm pools; 5,400-second GPU timeout.
- No warm containers, servers, or deployed services. GPU functions need explicit timeouts. Raw data stays in gitignored `results/raw/` plus Modal results Volume.
- Raw identity: `(model, id)`; request ID encodes original GSM8K row, cue, format, template, arm. Gold and original question stay attached. Smoke is a prefix of the fixed 200-problem sample.
- `docs/analysis-methods.md` records implementation choices left open by the pre-registration; do not silently change estimands after seeing data.
- Phase 1b: `uv run python -m src.run --phase phase1b --models all --limit-problems 200 --run-id phase1b-full --output-dir results/phase1b`; analysis in `src/analyze_1b.py` includes gold >= 10. Thinking-on output may omit the opening think tag; never score its reasoning.
