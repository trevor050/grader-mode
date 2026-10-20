# grader-mode

- Phase 1 pre-registration: `DESIGN.md`; never change it or either file in `prompts/` during implementation/data collection.
- Local Python: `uv sync --dev`, `uv run pytest`. GPU dependencies exist only in the Modal image.
- Pipeline: build prompts → `modal run -m src.modal_app` → score → analyze. Use `uv run` so CLI and package versions match.
- Only explicitly approved GPU runs. Current authorization: Qwen3-0.6B, 10 problems, no full sweep or larger models.
- No warm containers, servers, or deployed services. GPU functions need explicit timeouts. Raw data stays in gitignored `results/raw/` plus Modal results Volume.
- Raw identity: `(model, id)`; request ID encodes original GSM8K row, cue, format, template, arm. Gold and original question stay attached. Smoke is a prefix of the fixed 200-problem sample.
- `docs/analysis-methods.md` records implementation choices left open by the pre-registration; do not silently change estimands after seeing data.
