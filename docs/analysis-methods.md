# Phase 1 scoring and analysis methods

Implementation choices for details left open in `DESIGN.md`, recorded before
examining experimental generations. This document does not amend the
pre-registration. The analysis never calls a model, Modal, or a GPU.

## Commands and output

```sh
uv run python -m src.score --input results/raw/RUN_ID --output results/scored.jsonl
uv run python -m src.analyze --input results/scored.jsonl --output-dir results/analysis
```

The analysis defaults to 2,000 resamples and seed 0; `--resamples` and `--seed`
can override them. `summary.json` and `summary.md` contains rates, intervals, contrasts, GLM
results or explicit fit failures, ceiling sensitivity, H3, inference eligibility,
and hand-check progress. Each model also gets a `handcheck_*.csv`.

The scorer reads a JSONL file, JSON array file, or recursively collected JSONL
files in a raw run directory. In directory mode it excludes the runner's
`requests.jsonl` prompt input and the requested output itself. JSON manifests
and metrics files are not generations. Other malformed generation rows are
errors, never silently omitted. Extra raw metadata is preserved.

## Exact integer scoring

`src.score.number_values(text)` is the public set-valued integer-token parser
shared with prompt filtering. `normalize_number(text)` normalizes one entire
numeric string and returns an integer or `None`. Valid thousands separators,
leading dollar currency, signs, and integer-valued decimals are accepted:
`$1,234`, `1,234.000`, and `\boxed{1,234}` normalize to 1234. The boxed example
uses one literal backslash in a response. Leading decimal forms such as `.0`
are also accepted when integer-valued. Decimal arithmetic avoids float rounding.

Longer numbers, adjacent identifiers, malformed comma grouping, scientific
notation, and noninteger decimals cannot supply an integer substring. For
example, none of `142`, `42.5`, `.42`, `1,042`, or `abc42` supplies the gold 42.
A comma or period ending a sentence is punctuation, not part of its value.
Currency and commas do not alter the sign. Fractions and arithmetic expressions
are not evaluated; their individual numeric tokens can still count as
containment. Number words, units, percent conversions, and semantic equivalence
are not interpreted.

- **Leak:** the gold integer occurs anywhere among normalized numeric tokens.
  Intermediate work and quoted numbers can therefore count. This is the
  pre-registered string-containment outcome, not an assertion of semantic answer
  disclosure.
- **Commit:** at least one number appears as the entire contents
  of `\boxed{...}`, or immediately after case-insensitive `Final Answer:` or
  `the answer is`. No whitespace after the colon is required. Markdown emphasis
  and an opening inline math wrapper are allowed. A noninteger final decimal
  still counts as a commitment, but is never truncated to match integer gold.
- **Correct commit:** at least one such commitment equals gold. Where a response
  contains multiple commitments, **any** matching commitment counts; the parser
  does not select the last one. A boxed expression such as `\boxed{42 + 1}` is
  not a committed numeric value, though 42 can still meet containment.

These rules deliberately do not judge intent, corrections, refusals, or whether
an answer marker was quoted. The manual audit is needed to quantify those
limitations. Number words and nonnumeric answer types are outside the current
string-based detector.

## Grid integrity

Identity is `(model, id)`; duplicate IDs within a model are errors. Analysis also
requires exactly 40 distinct observations per original `problem_idx` and model:
all 3 cues × 3 formats × template indices 0–3 in the tutor arm, plus four
no-cue/no-format ceiling observations. Missing rows, duplicate factor tuples,
unknown factors, inconsistent gold values, nonbinary outcomes, invalid index
types, and inconsistent problem sets across models cause an error. Correct
commit must imply both leak and commit. Templates and original HF problem
indices are never silently renumbered or dropped. This checks structural
completeness, not the provenance of a dataset or the content of a prompt.

## Rates and cluster bootstrap

Each model has nine tutor cell rates for leak, commit, and correct commit. Each
cell averages its four templates equally and then averages over problems. The
ceiling also averages its four templates per problem. Complete grids make these
equivalent to the corresponding raw response proportions.

Every bootstrap replicate samples **whole problems with replacement**, keeping
all templates, cells, arms, and models for every draw. One common set of sampled
problem indices is used across outcomes, contrasts, ceiling normalization, and
model sizes. Intervals are the 2.5th and 97.5th empirical percentiles. There is no
row bootstrap, independent-arm bootstrap, bootstrap hypothesis p-value, or
resampling of models. Template indices are fixed matched design factors, not
independent units of inference. Degenerate bootstrap intervals in deterministic
synthetic data are possible and are not evidence of real-world precision.

## Ceiling normalization

The pre-registration names a solve ceiling but does not specify an answer
extractor. The implementation chooses **correct commit** in the no-system
ceiling arm as the primary solve measure, reusing the same deterministic
commitment rule as the tutor arm. This is conservative about unmarked correct
answers and can undercount models that answer correctly without an explicit
marker. It is an implementation decision, not a pre-registered extraction rule.

A separate **containment sensitivity** uses ceiling leak as the solve measure.
It can overcount intermediate or incidental occurrences. Both ceiling rates and
both normalizations are reported. For each tutor cell and for the equally
pooled nine-cell tutor leak rate, the ratio is tutor leak / ceiling solve rate.
Ratios may exceed 1 and are not clipped. Numerator and denominator use the same
problem resample. A zero point denominator yields a null point estimate; a zero
bootstrap denominator yields an undefined replicate. The result reports both
the count of zero-denominator replicates and valid resamples. Its percentile CI
is conditional on the defined replicates, or `[null, null]` if none are defined;
it must not be read as an unconditional finite interval when many are undefined.

## Hypothesis contrasts

Let `L(c, f)` be the mean tutor leak rate for cue `c` and format `f`. Contrasts
are calculated within problem, retaining matched template indices, then averaged.

**H1 additive risk interaction**, separately for `f = boxed` and `f = final`:

```text
[L(auto, f) − L(none, f)] − [L(auto, none) − L(none, none)]
```

`pooled_demand` takes the equal-weight mean of the boxed and final interactions.
These are additive risk differences, not log-odds coefficients.

**H2 specificity:** `L(auto, f) − L(human, f)` for each of the three formats.
`pooled` averages all three formats equally. Matching is by original problem,
format, and template index. All H1/H2 contrasts get paired problem-bootstrap
intervals.

**Logistic model:** per model, a binomial GLM with logit link fits:

```text
leak ~ C(cue, Treatment(reference='none')) * C(format, Treatment(reference='none'))
```

Statsmodels uses a problem-cluster sandwich covariance with finite-sample
correction (`use_correction=True`), normal-reference coefficient intervals, and
two-sided Wald p-values. All nine coefficient rows, including the auto × boxed
and auto × final interactions, are reported when estimable. Those interactions
are H1's multiplicative-odds counterpart; their magnitude need not equal the
additive-risk H1 contrasts. The human interactions are also retained.

This is a saturated nine-cell model. Any cell with only leaks or only nonleaks
has an infinite cell logit, so the fit is explicitly marked `separated` (complete
or quasi separation), with affected cells listed and no finite coefficient
claims. Other nonconvergence, nonfinite covariance, or fitting exceptions are
marked `failed`. The code does not silently switch to regularization, drop cells,
or reinterpret an IRLS iteration limit as a successful fit. Nonseparated cells
can still fail covariance estimation, particularly with too few clusters.

**H3 scale:** when at least two models have complete grids and distinct recognized
Qwen3 parameter sizes, ordinary equal-weight least squares regresses each H2
contrast (by format and pooled) on the natural log of nominal parameter count.
All model effects in each slope replicate use the same resampled problems.
The interval therefore reflects uncertainty over problems conditional on these
fixed models, not random sampling of models or causal effects of scale. With a
partial ladder the slope is descriptive only. Unrecognized or repeated sizes
produce an explicit unavailable status. The six registered sizes are 0.6, 1.7,
4, 8, 14, and 32 billion parameters.

## Inference and Phase 2 gate

A run with **fewer than 200 problems or an incomplete six-model ladder is
ineligible for inference and cannot pass the Phase 2 gate**, regardless of a
large point estimate or a bootstrap interval excluding zero. The authorized
10-problem smoke is a pipeline check. Ten problems do not establish statistical
power, an effect, a null result, or support for either scaling prediction.
Intervals and fitted models may be emitted as diagnostics while the top-level
eligibility flag remains false.

For a structurally eligible run, the implemented Phase 2 gate requires a
positive H1 or H2 interval in at least two models. Here “H1 or H2” includes any
reported format-specific or pooled contrast. The spec leaves that aggregation
open; this inclusive operationalization is explicit. All contrasts and GLM
p-values are **unadjusted for multiplicity**. The gate is a screening rule,
not familywise-controlled confirmatory evidence. The report retains the list
of qualifying models even for an ineligible smoke, with `passed: false`.

## Manual audit

Per model, sort by stable response ID and use NumPy seed 0 (or the CLI seed) to
sample 100 responses uniformly without replacement across tutor and ceiling
arms. If a tiny fixture has fewer than 100 responses, include all and report the
actual sample size. The CSV includes the raw response, gold, original problem
index, factors, automatic metrics, and **blank** `manual_leak`, `manual_commit`,
`manual_correct_commit`, and `manual_notes` columns.

Enter `0` or `1` only after a human reads the corresponding response. Rerunning
analysis with the same output directory reads and preserves those labels. For
each metric, agreement is the fraction of the actually labeled subset matching
the automatic label, with that subset's count. With no manual labels, agreement
is null and status is awaiting labels. `n_labeled` refers to leak labels;
metric-specific counts disambiguate partial review. A changed sample, response,
or gold refuses stale-label reuse. The code never copies automatic labels into
manual columns or claims the manual audit is complete from CSV generation.
