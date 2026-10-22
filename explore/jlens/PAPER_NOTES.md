# J-lens: implementation research notes

Research-only, 2026-10-21. No model execution or tests. Primary paper: Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/), July 6, 2026. Methods and methodological appendices read. Paper quotation below totals two words.

## Exact released estimator

Inspected [Anthropic reference release](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e), commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`; live GitHub HEAD matched. Local `.reference/jlens/fitting.py` matched the pinned remote byte-for-byte.

For prompt \(p\), token length \(T_p\), width \(d\), source block output \(h_{\ell,t}\), target block output \(z_{b,u}\), define

\[
A_{\ell,p}(t,u)=\frac{\partial z_{b,u}}{\partial h_{\ell,t}}\in\mathbb R^{d\times d},\quad
I_p=\{16,\ldots,T_p-2\},\quad M_p=|I_p|.
\]
\[
\boxed{\widehat J_\ell=
\frac1N\sum_{p=1}^N\frac1{M_p}
\sum_{t\in I_p}\sum_{\substack{u\in I_p\\u\ge t}}
A_{\ell,p}(t,u)}.
\]

This is derived directly from [`jacobian_for_prompt`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py#L100): one output-coordinate cotangent at **every valid target position**, then mean source-position gradients; [`fit`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py#L225) averages accepted prompts equally. Earlier targets contribute zero by causality. There is **no target-count divisor**, pair-count divisor, per-source future-count divisor, or prompt-length weighting across prompts. Uniform causal-pair averaging gives a different scale; averaging targets separately for each source changes relative source weights. Variable sequence lengths therefore matter, even with equal prompt weights.

- Defaults: `max_seq_len=128`, `skip_first=16`, final target block `b=n_layers-1`, every source block strictly before `b`. Both source and target masks exclude the first 16 positions and final position. At length 128: 111 valid positions, maximum target-source separation 110. This is the horizon of a teacher-forced input window, not sampled future continuations or an infinite-horizon rollout. Target residual at position `u` feeds prediction of token `u+1`.
- Jacobian is residual-to-residual, **not** a log-probability, loss, output-token probability, or vocabulary-logit Jacobian. Shape is `d × d` per source block. It contains no learned intercept, centering, whitening, SVD truncation, or PCA.
- [`ActivationRecorder`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hooks.py#L13) hooks **post-block outputs**, starts the differentiable graph at the earliest source, and retains activation gradients. Indices are zero-based block indices, not embedding-inclusive hidden-state indices.
- One forward and `ceil(d/dim_batch)` backwards per prompt; `dim_batch=8` default replicates the prompt to parallelize output-coordinate rows. Accumulation is fp32 on CPU; [`save`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py#L54) defaults to fp16 storage. These are not low-rank approximations.

## Readout, normalization, freezing

[`transport` / `apply`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py#L140) implement, with column-vector notation,

\[
q=J_\ell h,\qquad r=\operatorname{LMHead}(\operatorname{Norm}_{final}(q)),\qquad
P_\ell=\operatorname{softmax}(r).
\]

`apply` returns logits, not probabilities. Row-vector code uses `residual @ J.T`. [`HFLensModel.unembed`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py#L166) applies the model's actual final norm, LM-head bias if present, then model-specific logit softcapping when configured. Do not replace it with a bare unembedding product.

Mathematical implementation consequence: for RMSNorm with gain \(\gamma\), ignoring LM-head bias and softcap only for this derivation,

\[
r_i=\frac{w_i^\top D_\gamma J_\ell h}
{\sqrt{\|J_\ell h\|^2/d+\epsilon}},\quad
v^{effective}_{\ell,i}=J_\ell^\top D_\gamma w_i.
\]

The denominator is common across vocabulary entries at one activation, but learned gain is not. A raw `W_U @ J` dictionary and a gain-folded readout dictionary need an explicit convention. LayerNorm centering/bias requires its actual formula; unit-normalizing every vocabulary direction also changes ranking versus raw dot products. The release provides readout, not a token-vector intervention API.

[`from_hf`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py#L102) calls `eval()`, freezes **parameter gradients**, disables KV cache, and attempts to enable BOS. Activation derivatives still propagate through attention Q/K and internal normalization. There is no QK detachment, denominator stop-gradient, median aggregation, outlier rejection, or Jacobian-norm rescaling in this fitter; norm calculations are diagnostics. BOS enabling can be ineffective for some tokenizer implementations. Final norm is outside the differentiated block-output target, then reapplied during readout.

## Paper-specific qualifications and J-space

The [default Sonnet recipe](https://transformer-circuits.pub/2026/workspace/#app-method-details) targets the **penultimate** block, uses 1,000 × 128-token pretraining-like sequences; exact corpus is unspecified. Final/penultimate targets, frozen-QK, self/future-only gradients, mean/median aggregation and outlier filters are alternatives. [Attribution graphs](https://transformer-circuits.pub/2026/workspace/#app-interp-circuit) freeze attention patterns and normalization denominators, a separate procedure.

[J-space](https://transformer-circuits.pub/2026/workspace/#methods-jspace) is a “sparse subframe,” not PCA:

\[
V_\ell=(W_UJ_\ell)^\top\in\mathbb R^{d\times |\mathcal V|},\qquad
\mathcal F_{\ell,k}=\{V_\ell a:a\ge0,\ \|a\|_0\le k\},
\quad a^*\approx\arg\min_{a\ge0,\|a\|_0\le k}\|h-V_\ell a\|_2^2.
\]

Gradient pursuit approximates the nearest component. Local cones have dimension ≤k; the dictionary can span all d dimensions. Typical k≈25; concept-vector experiments also use 16. [Formal appendix](https://transformer-circuits.pub/2026/workspace/#app-sparse-frame) inconsistently writes unrestricted spans/orthogonal projections while describing nonnegative cones; those are not equivalent.

[Occupancy](https://transformer-circuits.pub/2026/workspace/#struct-bottleneck) is the marginal-reconstruction crossover against equal-size random dictionaries, median ≈25. Reported activation variance is **excess over random**, ≤10%, at layerwise median occupancy. Separate concept/probe-vector reconstructions report 6–7% / 10–15%; these are different denominators, not PCA explained variance.

## Five functional properties: released experimental support

The following maps the five properties to the **released protocols**, not independently reproduced results. Source: [`data/experiments/README.md`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/experiments/README.md), with accompanying JSONs. Each set contains prompts, not trial outputs.

| Property | Released experiment and discriminating measurement |
|---|---|
| Verbal report | `verbal-report`: 14 categories, chosen-item→candidate coordinate swaps, candidate next-token rank. `verbal-introspection`: inject a unit token direction, scaled by layer mean residual norm and strength, throughout user tokens; measure reported-token reciprocal rank at the later open quote. |
| Directed modulation | `directed-modulation`: focus/suppress/control instructions during an unrelated teacher-forced carrier sentence; score target concepts or arithmetic answers. `top-down-summoning`: next-word versus explicit-property questions on the same stimulus, with property-label/foil swaps. |
| Internal reasoning | `probe-swap`: 90 two-hop prompts, replace unspoken bridge-entity representations across workspace layers and prompt positions; measure the counterfactual downstream answer. The concept is an intermediate argument, not simply the token being requested. |
| Flexible generalization | `flexible-generalization`: four arguments × four function templates per category; swap argument representations across contexts and grade the answer required by the receiving function. Inspect per-function results rather than treating all categories as equally successful. |
| Selectivity | `selectivity-language`: explicit language-dependent question versus neutral continuation on identical multilingual passages. `selectivity-linecount`: direct counting/letter/continuation conditions on wrapped prose. Distinguish a label entering the lens from automatic behavior that already uses the underlying property. |

[Reported findings](https://transformer-circuits.pub/2026/workspace/#workspace): perturbations redirect reports; instructions modulate content; intermediate swaps redirect answers; transplanted concepts follow receiving operations; ablation selectively impairs flexible inference. [Generalization caveat](https://transformer-circuits.pub/2026/workspace/#app-flex-gen): country swaps succeed 42/48, number relations 0/48 at α=1.

Additional released structural protocols: `ignition` (interpolated token embeddings and transition sharpness), `capacity` (80-word blocked lists; band-min ranks), `dual-task` (single/dual covert task reachability). None defines a PCA basis or a universal workspace dimension.

Lens-quality [`evaluation protocols`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md) measure known intermediate recovery, usually at one prescribed token position with minimum rank across layers. Order-of-operations accepts single-token synonym sets. Counts observed in release JSONs: multihop 93, multilingual 107, order-ops 55, poetry 98, typo 96, association 102. These are release inventories, not necessarily the figure-specific scored subsets. A dataset's generic top-1 convention must not override its experiment-specific pass@k or next-token metric.

## Released assets and public gaps

- [Reference code](https://github.com/anthropics/jacobian-lens), [walkthrough](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/walkthrough.ipynb), [viewer](https://www.neuronpedia.org/jlens), and [paper slice viewer](https://transformer-circuits.pub/2026/workspace/public/slice-stack/index.html). Code and synthetic replication/evaluation prompts are Apache-2.0; the repository labels itself unmaintained. No Claude model weights or fitting corpus are bundled.
- Notebook points to [Neuronpedia lens artifacts](https://huggingface.co/neuronpedia/jacobian-lens/tree/16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a), revision `qwen-n1000` resolving to `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`, and filenames `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt` and corresponding `qwen3.6-27b/.../Qwen3.6-27B_jacobian_lens_n1000.pt`. Existence verified through Hub API; tensor contents were not downloaded or validated.
- Provenance warning: the pinned [4B config](https://huggingface.co/neuronpedia/jacobian-lens/blob/16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a/qwen3.5-4b/jlens/Salesforce-wikitext/config.yaml) reports `prompts_fitted: 417` with early stopping, while the notebook selects an `_n1000.pt` artifact. The colocated config cannot establish that artifact's fit count. Worse, the [27B-directory config](https://huggingface.co/neuronpedia/jacobian-lens/blob/16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a/qwen3.6-27b/jlens/Salesforce-wikitext/config.yaml) identifies **`openai/gpt-oss-20b`**, not Qwen. Do not infer tensor provenance from the directory name.
- [`load_wikitext_prompts`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/examples.py#L41) takes the first N WikiText-103 training records with ≥600 stripped characters, without shuffling; the fitter token-truncates them. This is a concrete public fitting distribution, not evidence of the private Claude corpus. The notebook uses 100 prompts and `dim_batch=32` for its optional fit example.
- Repository inspection found fitting, inference, visualization, adapters, prompts, and small code tests, but **no sparse gradient-pursuit implementation, workspace-component ablation/swap runner, experimental result files, occupancy/FVE runner, or template/oracle training pipeline**. Those require a separately documented reproduction. The directed-modulation README explicitly withholds its underlying prose corpus. Figure subset selection, seeds, intervention details, and exact private-model norms cannot be reconstructed from prompt JSONs alone.
- Saved lens files contain J matrices, source layers, width, and prompt count, but not target layer, model revision, corpus identity, masks, or normalization convention. Merge checks only width/source-layer compatibility. Parent implementation must preserve these missing provenance fields externally and reject incompatible merges; equal shapes do not establish equal recipes.

Research boundary: only this note was authored; no implementation, tests, commits, GPU use, or artifact downloads.
