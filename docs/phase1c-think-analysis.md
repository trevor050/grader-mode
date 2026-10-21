# Reduced-grid Phase 1c Think analysis

This implements DESIGN.md Deviations 10–11 in a separate module. It does not
change the Instruct analyzer, scorer, preregistration, prompts, or existing
results. Analysis verification uses synthetic data only. Real Think data
access was limited to schema checks and the separate trace-cost dry run; no
response excerpts or outcome aggregates were viewed or computed.

## Invocation after the analysis embargo is lifted

The lead invokes `uv run python -m src.analyze_1c_think` only after generation
ends and outcome analysis is authorized. Importing the module does no I/O.
Defaults read the three named checkpoint JSONL files in each of
`results/raw/phase1c-think-A` and `results/raw/phase1c-think-B`:

- `Olmo-3-7B-Think-SFT.jsonl`
- `Olmo-3-7B-Think-DPO.jsonl`
- `Olmo-3-7B-Think.jsonl`

The only default outputs are `results/phase1c-think/summary.json` and
`results/phase1c-think/summary.md`. Both contain the all-problem and gold >= 10
results, each with its own common-ceiling-solved restriction. No scored JSONL,
response excerpts, hand-check CSVs, trace coding, or extra sensitivity files
are emitted. `--stage-a`, `--stage-b`, and `--output-dir` permit alternate paths
for synthetic verification or explicitly selected inputs. Output paths within
raw inputs or `results/phase1c` are rejected.

`--allow-missing-b` permits the entire B stage to be absent, including an empty
directory with no checkpoint files. It never permits partial B: the presence
of any B checkpoint file requires all three nonempty files and every B cell.
The report explicitly records available templates and missing-B authorization.

## Integrity checks

A contains templates 0–1; B contains 2–3. DPO and RLVR each require all six
plain tutor formats and plain/none ceiling rows. SFT requires only plain/none
ceiling rows. Every checkpoint/cell/template requires exactly the same 100
original problem indices, with consistent integer gold and original question
text across stages, templates, formats, and checkpoints. Thus A has 3,000
rows, and A+B has 6,000 rows. Missing or extra rows, duplicate model/ID pairs,
calibration rows, unknown factors, invalid types, filename/model/checkpoint/run
mismatches, and IDs inconsistent with their factor metadata are rejected.

Raw request IDs follow the existing builder exactly:
`phase1c:think:CHECKPOINT:pNNNN:plain:FORMAT:tT:ARM`. The `run_id` must be
`phase1c-think-A` or `phase1c-think-B`, with matching template indices.
Required raw fields include `question`, `response`, `finish_reason`, and the
existing request/model metadata. Non-generation JSON metadata and
`requests.jsonl` are not ingested; unexpected generation JSONL files fail.
Structural validation precedes scoring and all subset operations.

These checks establish cardinality and internal consistency, not that the 100
indices are the registered first 100 sampled problems or that generation used
the pinned revisions, tutor prompt, sampling settings, and token budget. The
upstream generation manifests and lead's schema/provenance validation own those
checks; the analyzer does not open manifests or reconstruct the dataset.

## Scoring and estimands

The analyzer recomputes scores with `src.score.score_row`, ignoring any stale
score columns. Only the final reply after the balanced reasoning close is
scored, including the implicit opening-tag convention. An unclosed reasoning
block has no final reply. Empty finals are excluded from all three rate
numerators and denominators, even for stop finishes. Nonempty length-limited
finals remain eligible. Every cell reports total, valid, excluded, truncated,
truncated-empty, and empty-final counts, plus leak, commit, and correct-commit
rates. Raw text and extracted reasoning are discarded before aggregation.

Rates and contrasts use the existing Instruct estimands: valid successes /
valid replies, recomputed in every problem-cluster draw; equal-weight mean of
the three format-specific effects for pooled RL formats. In particular:

- H6: pooled RL-format effect at RLVR minus that at DPO.
- H7, RLVR specificity: mean leak in boxed/dapo/answer_tag minus result_tag.
- H7, difference-in-differences: RLVR specificity minus DPO specificity.
- H8: boxed_step minus none at RLVR.

H7 cancels the shared none baseline algebraically, including when that cell's
finals are all empty. Plain format effects and per-format H6 breakdowns are
reported; the latter remain exploratory. Env × format and SFT → DPO
exploratory analyses are explicitly omitted because their tutor cells were
not collected. SFT contributes to ceiling estimates and the solvability
restriction only.

There are exactly 2,000 bootstrap draws with NumPy `default_rng(0)`, common
across all checkpoints, cells, templates, arms, and outcomes within each subset.
Whole original problems are sampled with replacement, retaining their templates
and cells together. Subsets restart the same seeded procedure using their own
sorted problem indices. Percentile 95% intervals use defined replicates only;
undefined denominator draws are counted and produce null estimates/intervals
when none are defined. There is no row/template bootstrap or independent
checkpoint resampling.

Solvability requires at least one valid correct-commit ceiling reply in every
one of SFT, DPO, and RLVR among the templates that actually ran. Membership is
fixed before resampling. All numerical results are repeated for gold >= 10,
and the all-three ceiling intersection is also applied within that sensitivity
set. Empty restrictions produce null outcomes, not zero rates.

## Methods risks

Intervals are conditional on defined replicates and unadjusted for
multiplicity. Excluding empty finals can select shorter reasoners differently
across checkpoints/formats; truncation and exclusion counts must accompany
any interpretation. Nonempty truncated finals are still partial completions.
Ceiling membership is an observed correctness-based selection and changes when
additional templates run; restricted results are a sensitivity analysis rather
than an independent randomized comparison. The implementation does not decide
whether an interval establishes a hypothesis or assess causal specificity of
RLVR training data/rewards. It does not perform the required human scoring
agreement audit or exploratory reasoning-trace coding.
