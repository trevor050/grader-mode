# Exploratory Think trace coding

The outcome embargo remains in force until generation finishes. No real judge,
sample export, or Think outcome report was run during implementation. The only
real-data operations were schema inspection and local token counting on the
existing RLVR Stage A file. DESIGN.md and prompts/ are unchanged.

## Model and cost

Official sources checked 2026-10-20:

- [Model catalog](https://developers.openai.com/api/docs/models)
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna): efficient focused/high-volume model, 1,050,000-token context, structured outputs and Batch supported; reasoning `none` supported.
- [Pricing](https://developers.openai.com/api/docs/pricing): Batch input **$0.05/M**, output **$0.25/M**. No cache discount assumed.
- [Batch guide](https://developers.openai.com/api/docs/guides/batch): 50% discount, 24-hour completion window, separate queue limits.

`gpt-6-luna` with reasoning disabled is the cheapest current nondeprecated
text model in the fetched catalog adequate as a candidate for this bounded
three-way classification. Adequacy is a design judgment, not a measured
accuracy claim. The second-family/hand agreement check remains required.
Older nano pages are marked deprecated. Requests use a strict JSON schema,
category enum `(a)`/`(b)`/`(c)`, 64 maximum completion tokens, and `store: false`.
Definitions are copied verbatim, including line breaks, from the addendum; a
synthetic test checks that copy against DESIGN.md.

The local `o200k_base` count is an estimate because Luna's precise local
encoding is not established by the retrieved docs. It includes the full judge
request (instructions, blinded fields, schema), plus framing allowance. A
separate UTF-8-byte bound plus 2,048 tokens per request is used for budget
reservations. Overlong requests are rejected before long-context pricing can
apply. Output cost reserves the full configured 512-token maximum (reasoning effort low), including any
billable reasoning tokens. No actual judging or token-count API call is made
by the dry run.

```sh
uv run python -m src.trace_coding dry-run \
  --input results/raw/phase1c-think-A/Olmo-3-7B-Think.jsonl \
  --project-total 6000 \
  --output results/phase1c-think/trace-cost-projection.json
```

Measured on the 1,200 tutor rows in the existing 1,400-row RLVR Stage A file:
3,719,304 estimated input tokens. Scaling to **6,000 tutor traces** projects
**$1.025826**, including **$0.096** maximum output reservation. Conservative
budget projection: **$4.266515**, below the **$6 lifetime cap**. This extrapolates
RLVR Stage A trace lengths; DPO and Stage B can differ. Submission recomputes
costs for the actual inputs and includes all prior reservations.

The reduced grid actually contains **4,800 tutor traces** when both stages are
complete, plus 1,200 ceiling responses. The requested 6,000-trace projection
therefore includes headroom. Ceiling rows are never judged.

## Deferred workflow, after generation finishes

Use the same source population for submission, sampling, and reporting. Input
directories must contain only generation JSONL files plus optional
`requests.jsonl` (ignored); pass individual generation files for partial/local
projections. Duplicate response identities fail rather than silently deduplicate.
Do not include scored JSONL files or prior exports in an input directory.

1. Submit judge 1 using the approved local env file. Keys are loaded into the
   environment by `python-dotenv`; no key is written to payloads, state, or logs.

```sh
uv run python -m src.trace_coding submit --generation-complete \
  --input results/raw/phase1c-think-A results/raw/phase1c-think-B
```

2. Collect completed batches, then repeat the same submit/collect commands until
   `remaining_unsubmitted` and `active_batches` are zero. Commands return without
   polling or sleeping until completion. Collection exports cached labels only.
   Failed/malformed responses remain unlabeled; the next submit retries only
   those responses and reserves another full attempt cost.

```sh
uv run python -m src.trace_coding collect \
  --output results/raw/trace-coding/judge1.jsonl
```

3. Export the random 100-trace sample (seed 0). Sampling sorts opaque IDs first,
   so file order does not affect the sample. This writes exactly the opaque ID,
   tutor system prompt, unchanged full user message, reasoning, and blank label.

```sh
uv run python -m src.trace_coding sample --generation-complete \
  --input results/raw/phase1c-think-A results/raw/phase1c-think-B \
  --output results/raw/trace-coding/sample-100.jsonl
```

Give a **different-family judge** the classification instructions in
`src.trace_coding.JUDGE_INSTRUCTIONS` verbatim and only the three blinded fields
from each sample row. The opaque ID is for matching, not classification. Fill
`category` in a copy as `judge2-100.jsonl`; independently hand-label another copy
as `hand-100.jsonl`. Never show judge 1 labels or final replies to either rater.
No automated second-judge API transport is required or provided: the exported
sample is the interchange file. The family/model provenance below records the
operator's declaration, not independent verification of which service ran.

```sh
uv run python -m src.trace_coding agreement \
  --judge1 results/raw/trace-coding/judge1.jsonl \
  --judge2 results/raw/trace-coding/judge2-100.jsonl \
  --hand results/raw/trace-coding/hand-100.jsonl \
  --sample results/raw/trace-coding/sample-100.jsonl \
  --judge2-family anthropic --judge2-model <actual-model-used> \
  --output results/phase1c-think/trace-agreement.json
```

Judge 1 may contain the full population; only sample IDs enter agreement.
Second-judge and hand labels must match exactly those 100 IDs, with no blanks,
duplicates, or invalid labels. Same-family validation is rejected. All three
pairwise raw agreement and Cohen's kappa values are reported. Kappa is null
with an explicit status when constant marginals make its denominator zero.

4. Generate the exploratory outcome report only after the embargo is lifted.

```sh
uv run python -m src.trace_coding report --generation-complete \
  --input results/raw/phase1c-think-A results/raw/phase1c-think-B \
  --labels results/raw/trace-coding/judge1.jsonl \
  --output results/phase1c-think/trace-report.json
```

Shares use all tutor traces, including unclosed/truncated reasoning. Leak rates
within each checkpoint × format × category use only nonempty final replies,
scored by the existing Phase 1c scorer; exclusions are counted. Empty categories
have zero share and null leak rate. All tutor responses must have a label.
This is exploratory and supplies no confirmatory tests.

## Blinding, persistence, and recovery

The judge payload contains only the codebook plus tutor system prompt, full
user message, and extracted reasoning. The user message intentionally retains
format wording. Metadata labels, original response IDs, checkpoint, run, gold,
final reply, and scored outcomes are absent. `custom_id` is an opaque SHA-256 of
the raw `(model, id)` identity. No closing `</think>` means the entire response
is reasoning and is flagged in local state. Balanced-block extraction also
handles implicit opening tags. Unclosed empty traces are still included.

Default state is **`results/raw/trace-coding/`**, already gitignored. Keep this
same directory across all primary-judge attempts; do not reset/copy the ledger
or use multiple state directories to bypass the lifetime budget. State includes
blinded payload files, labels keyed by opaque response identity, extraction flags,
content hashes, request configuration fingerprint, batch IDs, and cumulative
reservations. Changed content for an existing response ID is rejected.

The ledger is locked against concurrent processes, atomically replaced and
fsynced **before** paid creation. Reservations are never refunded, including
failed requests, uncertain POSTs, or retries. New work plus all previous reserved
attempts must stay <= $6 before any upload/submission. Each batch defaults to
250 requests; a conservative 10M queued-token cap leaves work for later resume
rather than overfilling the documented Luna Build-tier 20M queue. Other work on
the same account can still cause 429s; retry the same command later. Read-only
calls and nonbilling uploads use bounded exponential backoff, including SDK
connection/timeout errors; uploads rewind the file on retry.

Paid Batch creation is never blindly retried. Definitive 4xx rejections retain
the prepared job for later resume. Ambiguous creation is recovered by matching
the persisted random job key against paginated provider Batch metadata. Zero or
multiple matches stop instead of risking an automatic duplicate paid job.

A process killed after persisting its intent but before sending the POST has no
remote job to find. After checking the provider dashboard and confirming no
accepted batch exists, reconcile that one unresolved key, then resume normally:

```sh
uv run python -m src.trace_coding reconcile \
  --job-key <key-from-state.json> --confirm-no-remote-batch
```

Reconciliation checks remote metadata again and refuses if a job exists. It
retains the original reservation; the new attempt reserves full cost again,
covering even an unexpectedly late orphan. Preserve the original ledger for
investigation. Normal collection is idempotent and refuses unknown/duplicate
output IDs or conflicting cached labels.
