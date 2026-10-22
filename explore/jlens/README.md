# Jacobian lens for the grader direction

CPU-tested HF/PyTorch implementation of the [primary paper](https://transformer-circuits.pub/2026/workspace/) and [released estimator](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py). See [PAPER_NOTES.md](PAPER_NOTES.md) for equations, experimental evidence, exact source conventions, and public gaps. No GPU execution was performed during development.

Everything in this directory is independent of `explore.py`, `select_neurons.py`, `track.py`, and `analyze_acts.py`. No imports from those scripts. Outputs and the isolated environment are ignored by Git.

## Setup and CPU verification

From the repository root:

```bash
uv venv explore/jlens/.venv
uv pip install --python explore/jlens/.venv/bin/python -r explore/jlens/requirements.txt
OMP_NUM_THREADS=1 explore/jlens/.venv/bin/python -m pytest explore/jlens/test_jlens.py -q --basetemp=explore/jlens/.pytest_tmp
```

On the GPU box, use an existing compatible CUDA PyTorch installation or install a matching CUDA wheel, then install the remaining requirements in a dedicated environment. The tested versions are PyTorch 2.9.0, transformers 4.57.1, Python 3.12. Tiny random Qwen3 and GPT-2 configurations and a locally constructed tokenizer require no downloads. Tests compare against a full position-by-position Jacobian, check batched fitting, deterministic readout, lazy vocabulary rows, sparse nonnegative reconstruction, neuron write vectors, checkpoint continuation, and both CLI commands.

## Layer conventions: read this before using direction files

All indices here are **residual boundaries**:

- `L=0`: input to block 0, the embedding residual.
- `L=16`: input to block 16, exactly HF `hidden_states[16]`.
- A block `b` writes its output at boundary `b+1`.
- `L=N`: output of the final block **before final normalization**. HF `hidden_states[N]` is already normalized and is not interchangeable with this boundary.

Existing `dirs-False.pt`/`dirs-True.pt` files from `analyze_acts.py` use HF hidden-state indices, so pass their key directly for `L<N`. A file keyed by post-block hook indices requires `L=key+1` and rekeying the tensor dictionary. Anthropic's released fitter indexes post-block outputs. This tool's `L` equals that source index plus one.

Default target boundary is `N-1`, the penultimate block's output, matching the paper's main Sonnet recipe. Set `--target-layer N` for the final-block target used by the released code's default. For Qwen3-14B, `N=40`, so these targets are 39 and 40. Source must be at or before target. Identity transport at the target boundary is supported. The final model norm is applied after transport in either case.

## Calibration corpus

`prompts.txt` contains one **raw, already rendered** calibration context per nonempty line. Literal newlines separate records; use single-line records. No chat template is applied. Tokenizer special tokens follow the checkpoint's defaults; no synthetic BOS is inserted. Exact token IDs are hashed into output provenance. Match BOS handling explicitly when comparing another implementation.

Use broad, independent text contexts to estimate a general disposition to verbalize. A corpus containing only RL-framed prompts changes the estimand to those contexts. Keep calibration separate from the prompts used to discover/evaluate the grader direction. Default truncation is 128 tokens. Positions `[16,T-1)` are used for both source and target, so each context must have at least 18 tokens. Short inputs fail; they are never silently skipped. Actual lengths are recorded. For closest reproduction, prepare 1,000 contexts of at least 128 tokens each; start with a small pilot.

The exact released reduction is **sum over current/future target positions, mean over source positions, then equal mean over prompts**. No target-horizon division, sampled continuations, log-probability gradients, or mean of per-context normalized readouts. Q/K and normalization derivatives remain active. The transported vector is corpus-averaged before the actual model final norm/head are applied.

## GPU commands

Run these on the lead's GPU box, from this directory, after activating its environment. Pin `--revision` to the actual model commit for a repeatable run. These commands do not rent or launch compute.

Direction-only readout, inexpensive relative to fitting a reusable matrix:

```bash
python jlens.py readout \
  --model Qwen/Qwen3-14B --device cuda --dtype bfloat16 \
  --acts-dir-file /path/to/dirs-False.pt --layer 20 \
  --prompts /path/to/calibration-prompts.txt --out readout-L20.pt
```

Accepts a tensor `[5120]`, a tensor `[n_directions,5120]`, `{layer: tensor}`, or `{"directions": {layer: tensor}}`. Keys may be integers or strings. A positive direction is RL-framing minus homework-framing. Readout includes transported directions, full vocabulary logits, logit-lens logits using the same final norm/head, and both highest/lowest ranked token IDs/text/scores. Apply `.softmax(-1)` for the paper's normalized readout distribution; those scores are not calibrated probabilities of actual behavior. To inspect the opposite pole, supply `-direction` too, especially for models with biases or LayerNorm.

Reusable J-space dictionary estimation at several layers:

```bash
python jlens.py jspace \
  --model Qwen/Qwen3-14B --device cuda --dtype bfloat16 \
  --prompts /path/to/calibration-prompts.txt --layers 16,20,24 \
  --dim-batch 1 --out jspace.pt
```

`jspace.pt` contains `J[layer]` in CPU fp32 plus model/corpus/token/mask/target provenance. This specifies the entire vocabulary frame through `W_U J`, without storing it. It is not a learned low-rank basis. Each completed prompt is atomically checkpointed to `jspace.pt.partial.pt` as a running sum; rerun the same command with `--resume` after interruption. Resume requires identical recorded metadata/layers; changing precision, corpus, target, or model is rejected. A partial file is not a usable completed lens. Existing completed outputs are never overwritten.

Reuse it for sparse direction reconstruction:

```bash
python jlens.py readout \
  --model Qwen/Qwen3-14B --device cuda --dtype bfloat16 \
  --acts-dir-file /path/to/dirs-False.pt --layer 20 \
  --prompts /path/to/calibration-prompts.txt --lens-file jspace.pt \
  --projection-k 25 --out grader-projection-L20.pt
```

Model revision, tokenized corpus, target, dtype, masks, and boundary conventions must match. Local weights/config are fingerprinted by streaming their bytes through SHA-256, so changing files at the same path invalidates reuse. This adds one sequential read of the checkpoint. Hub models use their immutable resolved commit. The fit also records the requested revision. Keep files immutable during a running command. Save actual source direction provenance separately.

An optional `jspace --sample-positions 1 --projection-k 25` reconstructs evenly sampled valid activations from the calibration corpus. Its `activation_reconstruction` reports **uncentered, held-in reconstruction energy**, not the paper's excess-over-random activation variance. Disabled by default because decomposition across a large corpus can dominate runtime. This implementation does not automatically reproduce the paper's occupancy crossover or choose k from data. Run k/control sensitivity before interpreting empirical fractions as exceptional.

## Python: projection and neuron groups

```python
import torch
from transformers import AutoModelForCausalLM
from jlens import HFLens, LensDictionary, project_direction, neuron_directions

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    attn_implementation="eager").to("cuda")
adapter = HFLens(model)
fitted = torch.load("jspace.pt", map_location="cpu", weights_only=True)
dictionary = LensDictionary(adapter, fitted["J"][20])
directions = torch.load("dirs-False.pt", map_location="cpu", weights_only=True)
result = project_direction(directions[20], dictionary, k=25)
print(result["explained_energy"], result["token_ids"], result["coefficients"])

# Neurons at down_proj INPUT in block 19 write at boundary 20.
# Supply measured signed activations to analyze the group's actual contribution.
writes = neuron_directions(adapter, 19, [42, 99], activations=[0.8, -0.2])
group_result = project_direction(writes.sum(0), dictionary, k=25)
individual_results = [project_direction(w, dictionary, k=25) for w in writes]
```

Absent `activations`, neuron write directions are columns of `mlp.down_proj.weight`. A list of neuron IDs or their scalar activation differences cannot be projected directly into residual space. Keep groups separated by layer; use the dictionary at boundary `block+1`. The adapter supports common HF decoder layouts; explicit `down_proj` neuron extraction is for gated-MLP families such as Qwen/Llama. It freezes model parameters and selects eager attention. Do not share its temporary hooks with a concurrent generation thread.

Sparse projection uses unit-normalized rows of **raw `W_U J`**, as named in the paper. Unit scaling preserves the available cones. The readout applies the real final norm, including its learned gain; that gain is not folded into this raw dictionary. Thus sparse coordinates and readout ranks need not match. The released code has no sparse reconstruction implementation, so this is a documented nonnegative gradient-pursuit reproduction: greedy support growth plus projected gradient refinement, not a guaranteed global solution to the combinatorial nearest-cone problem. Returned coefficients refer to unit atoms. At most k atoms are selected.

## Cost and memory on a 48GB GPU

The [Qwen3-14B configuration](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) has width 5,120, 40 blocks, intermediate width 17,408, vocabulary 151,936. Budget roughly **30GB decimal for bf16 weights**, plus CUDA/runtime overhead, decoder activations and workspaces. With 128 tokens, `--dim-batch 1`, frozen parameters, no KV cache or weight gradients, expect an approximate **32–40GB** working envelope. This is a planning estimate, not a measured CUDA peak or guarantee. The command records CUDA peak allocated memory; compare with `nvidia-smi` for total process memory. Model loading first stages weights in host RAM, so allow roughly 35GB+ host headroom. Longer sequences or larger dimension batches increase memory; retain batch 1 until measured.

Three fp32 transport matrices occupy **300 MiB host RAM/disk** (100 MiB each), with another 300 MiB for checkpoint sums while loading/resuming. Lazy dictionary norms require only vocabulary scalars; a 512-row temporary `W_chunk J` is 10 MiB. A full fp32 vocabulary frame would be about 2.9 GiB **per layer** and is never retained. Dictionary row-norm preparation does dense `W_U J` arithmetic in chunks, trading compute for bounded memory.

Runtime is dominated by the chosen operation:

- Direct readout: one prefix capture and one forward-mode directional pass per prompt/direction through the chosen remainder; roughly linear in prompt count and number of directions.
- Reusable fitting: one forward and **5,120 single-replica backwards per prompt**, sharing the graph across requested layers. At dimension batch b there are `ceil(5120/b)` backward calls but each has b replicas, so FLOPs remain comparable. Three requested layers share work instead of multiplying the full model cost by three.
- Sparse projection: dictionary norm preparation plus up to k vocabulary scans per vector; repeated directions can share the prepared `LensDictionary`.

Do not assume the 1,000-context full fit is a quick job. Benchmark one complete representative context first. For example, **if** one backward takes 10–100ms on the lead's hardware, the arithmetic estimate alone is 51–512 seconds/context, about 14–142 hours for 1,000 contexts, before other overhead. These are conditional examples, not device benchmarks. Use the printed per-prompt timing to project the actual runtime, keep partial checkpoints, and consider direction-only JVP readout first. No GPU benchmark, rental, remote process, or paid run is authorized by this package.

## Interpreting the grader question

At each boundary, ask how well the positive **and negative** RL-minus-homework direction reconstruct with k=5/10/25 atoms. `explained_energy = 1 - ||direction-component||² / ||direction||²`; zero vectors report zero. Because coefficients are nonnegative, the two poles can have different fractions. This measures greedy sparse cone reconstruction, not membership in a proper linear subspace. An unrestricted linear projection would often be trivially 100%.

Compare with same-dimensional random dictionary controls of the **same vocabulary size and k**, matched-norm random directions and shuffled framing labels. A large raw fraction alone is weak evidence in an overcomplete dictionary. Held-out prompts, corpus/seed sensitivity, k curves, and uncertainty across problem samples matter. This script estimates the lens; it does not turn a discovered direction into a preregistered causal result. The paper's ≤10% activation figure has a different denominator and random correction; never compare it directly with a steering direction's reconstruction fraction.

For semantics, compare highest/lowest J-lens tokens with the saved logit-lens readout at 16/20/24 and nearby boundaries. Stable evaluation/reward/assessment concepts in the RL pole across independently calibrated lenses would support a verbalizable grading-related component. Homework/tutoring concepts at the opposite pole would help interpret the contrast. Fragmentary tokens, broad topic words, or layer-specific flips may reflect tokenizer granularity, model vocabulary geometry, or distribution mismatch. The readout labels a direction's average verbalization disposition; it does not show that a particular prompt activates it or that the model is consciously aware of evaluation. Validate those separately with held-out activations and interventions.
