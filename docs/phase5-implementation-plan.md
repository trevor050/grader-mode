# Phase 5 implementation and pilot plan

Original registration: DESIGN.md at f15dcc4. Amendment 1: c3e7050. Model formats and pinned revisions: notes/phase5-feasibility.md and notes/phase5-model-options.md. Registration overrides exploratory recommendations.

- Preserve design and prompt bytes. The original Gemma pilot manifest retains f15dcc4; separate hash evidence records the authorized amendment pulled after that pilot.
- Isolate model registries, native rendering and token-ID final splitting from prior phases. Gemma uses native channel syntax; Nemotron and the unused GLM runtime fallback use their exact native thinking openers.
- Implement original 200 problems and templates 0/1, matched seeds, 4096 total new tokens, temperature .7, top-p .95 and disabled top-k. Each selected family has 1600 main and 1600 prefill requests.
- Score final replies only. Implement H18/H19/H20/H21 with paired 2000-resample problem bootstrap, seed0, gold>=10 sensitivity and per-cell cap/no-final/protocol counts. Blind attempt, number and final-boundary labels remain pending until humans supply 40 per family.
- Keep one durable aggregate ledger. Preserve all original $6.80 accounting when explicitly migrating to amendment 1's $9 cap. Retain startup/fix failures, reservations, scoped watchdogs and confirmed app-stop plus empty-container teardown.
- Run only the already-authorized Gemma L40S BF16 pilot: 16 balanced requests, exact native prompt and saved-ID replay checks, timing and spending evidence. No Vast rental, Nemotron/GLM inference or full grid in this task.
- Nemotron runtime fallback is prospective: one startup attempt plus one fix attempt, judged without violation scoring, then GLM may replace both H20 and H21. Whole-cell budget drops preserve the registered priority; never shorten cells.
- Publish Gemma measurements and conditional projections, keeping second-family costs pending its future authorized pilot. Commit and push code and publishable pilot evidence, then stop.
