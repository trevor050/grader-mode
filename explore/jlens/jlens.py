#!/usr/bin/env python3
"""Corpus-averaged Jacobian lens and sparse nonnegative J-space analysis.

See PAPER_NOTES.md for primary sources and README.md for layer conventions.
No vocabulary-by-hidden Jacobian is constructed. Direct readout uses a JVP;
fitting stores hidden-by-hidden transports via bounded-batch reverse AD.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


class _ReachedTarget(Exception):
    def __init__(self, value):
        self.value = value


class HFLens:
    """Frozen, eager decoder adapter. Indices are residual *boundaries*.

    Boundary l is the INPUT of block l; boundary n is the final pre-norm
    residual. Consequently HF hidden_states[l] matches for l < n. HF's last
    hidden state is post-norm and must NOT be used as boundary n.
    Supports the standard HF Qwen/Llama/Mistral, GPT-2 and GPT-NeoX layouts.
    Hooks are temporary; do not use this adapter concurrently on one model.
    """
    def __init__(self, model):
        self.model = model.eval()
        model.requires_grad_(False)
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        layouts = [("model", "layers", "norm"),
                   ("transformer", "h", "ln_f"),
                   ("gpt_neox", "layers", "final_layer_norm")]
        for name, layers, norm in layouts:
            decoder = getattr(model, name, None)
            if decoder is not None and hasattr(decoder, layers) and hasattr(decoder, norm):
                self.decoder = decoder
                self.layers = getattr(decoder, layers)
                self.norm = getattr(decoder, norm)
                break
        else:
            raise ValueError("unsupported decoder layout; add an explicit layout to HFLens")
        self.head = model.get_output_embeddings()
        self.embedding = model.get_input_embeddings()
        self.width = self.embedding.weight.shape[1]
        self.n_layers = len(self.layers)
        if self.head.weight.shape[1] != self.width:
            raise ValueError("projected residual/unembedding widths are unsupported")
        # Eager attention supports forward AD; optimized SDPA/Flash kernels may not.
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"

    @property
    def device(self):
        return self.embedding.weight.device

    @property
    def dtype(self):
        return self.embedding.weight.dtype

    def resolve(self, layers, target_layer):
        target = self.n_layers - 1 if target_layer is None else target_layer
        if not layers or not 0 <= target <= self.n_layers:
            raise ValueError("empty layers or target outside residual boundaries")
        if any(not 0 <= layer <= target for layer in layers):
            raise ValueError("source must be a boundary at or before target")
        return target

    def _run(self, ids, layers, target, replacement=None, root=False):
        values, handles = {}, []
        first = min(layers)

        def make_hook(index):
            def hook(module, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                if replacement is not None and index == first:
                    h = replacement
                elif root and index == first:
                    h = h.detach().requires_grad_(True)
                values[index] = h
                if index == target:
                    raise _ReachedTarget(h)
                if args:
                    return (h, *args[1:]), kwargs
                return args, {**kwargs, "hidden_states": h}
            return hook

        try:
            for index in sorted(set([*layers, target])):
                module = self.norm if index == self.n_layers else self.layers[index]
                handles.append(module.register_forward_pre_hook(make_hook(index), with_kwargs=True))
            try:
                self.decoder(input_ids=ids.to(self.device), use_cache=False)
            except _ReachedTarget as end:
                return end.value, values
            raise RuntimeError("target hook never ran")
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def capture(self, ids, layers):
        _, values = self._run(ids, layers, self.n_layers)
        return {layer: values[layer].detach() for layer in layers}

    def forward_from_residual(self, ids, layer, residual, target):
        self.resolve([layer], target)
        return self._run(ids, [layer], target, replacement=residual)[0]

    @torch.no_grad()
    def unembed(self, residual):
        value = residual.to(device=self.head.weight.device, dtype=self.head.weight.dtype)
        logits = self.head(self.norm(value))
        cap = getattr(self.model.config, "final_logit_softcapping", None)
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits


def valid_mask(ids, skip_first=16):
    if ids.ndim != 2 or ids.shape[0] != 1 or skip_first < 0:
        raise ValueError("prompts must be unpadded [1,T] input ids; skip_first >= 0")
    mask = torch.zeros(ids.shape[1], dtype=torch.bool, device=ids.device)
    mask[skip_first:-1] = True
    if not mask.any():
        raise ValueError(f"prompt has {ids.shape[1]} tokens; need > skip_first+1 ({skip_first+1})")
    return mask


def _directions(value, adapter):
    value = torch.as_tensor(value)
    if value.ndim not in (1, 2) or value.shape[-1] != adapter.width or not torch.isfinite(value).all():
        raise ValueError(f"expected finite [d] or [n,d] directions, d={adapter.width}")
    return value.to(adapter.device, adapter.dtype)


def mean_transport(adapter, prompts, directions, layer, *, target_layer=None, skip_first=16):
    """Compute J_l d without forming J_l, using one JVP per prompt/direction.

    For valid positions M (same source/target mask as released code), define
    F_p(u) = sum_{t' in M} z[t'; h_l + 1_M u] / |M|.
    D F_p(0) d = (1/|M|) sum_{t,t' in M} (dz[t']/dh_l[t]) d.
    Causality zeroes t'<t. Average transports BEFORE normalization/unembedding.
    No division by the number of future positions: matches released estimator.
    """
    target = adapter.resolve([layer], target_layer)
    directions = _directions(directions, adapter)
    single = directions.ndim == 1
    directions = directions.reshape(-1, adapter.width)
    if not len(prompts):
        raise ValueError("empty corpus")
    total = torch.zeros_like(directions, dtype=torch.float64, device="cpu")
    for ids in prompts:
        ids = ids.to(adapter.device)
        mask = valid_mask(ids, skip_first)
        base = adapter.capture(ids, [layer])[layer]
        scale = mask[None, :, None].to(base.dtype)

        def f(delta):
            z = adapter.forward_from_residual(ids, layer, base + scale * delta, target)
            return z[:, mask].sum(dim=(0, 1)) / mask.sum()

        for i, direction in enumerate(directions):
            # Forward AD doesn't build a backwards graph or weight gradients.
            with torch.no_grad():
                _, transported = torch.func.jvp(f, (torch.zeros_like(direction),), (direction,))
            total[i] += transported.detach().double().cpu()
    result = (total / len(prompts)).to(adapter.dtype).to(adapter.device)
    return result[0] if single else result


def estimate_jacobians(adapter, prompts, layers, *, target_layer=None, skip_first=16,
                       dim_batch=1, progress=None, initial_state=None, checkpoint=None):
    """Fit d x d J by bounded-batch VJPs, sharing each graph across all layers.

    Batch replicates one prompt, one output basis cotangent per replica.
    Parameters are frozen, graph starts at earliest source; results/accumulators
    live on CPU fp32. No vocabulary tensors or weight gradients are retained.
    """
    layers = sorted(set(layers))
    target = adapter.resolve(layers, target_layer)
    if dim_batch < 1 or not len(prompts):
        raise ValueError("dim_batch and corpus size must be positive")
    result = {l: torch.zeros(adapter.width, adapter.width, dtype=torch.float32) for l in layers}
    n_done = 0
    if initial_state is not None:
        n_done = initial_state["n_done"]
        if not 0 <= n_done <= len(prompts) or set(initial_state["jacobian_sum"]) != set(layers):
            raise ValueError("invalid checkpoint count/layers")
        for layer in layers:
            value = initial_state["jacobian_sum"][layer]
            if value.shape != result[layer].shape or not torch.isfinite(value).all():
                raise ValueError("invalid checkpoint Jacobian")
            result[layer] = value.float().cpu().clone()
    for prompt_index in range(n_done, len(prompts)):
        ids = prompts[prompt_index]
        start = time.perf_counter()
        ids = ids.to(adapter.device)
        mask = valid_mask(ids, skip_first)
        with torch.enable_grad():
            z, sources = adapter._run(ids.expand(dim_batch, -1), layers, target, root=True)
            source_values = [sources[l] for l in layers]
            for offset in range(0, adapter.width, dim_batch):
                count = min(dim_batch, adapter.width - offset)
                cotangent = torch.zeros_like(z)
                for b in range(count):
                    cotangent[b, mask, offset + b] = 1
                grads = torch.autograd.grad(z, source_values, cotangent,
                                           retain_graph=offset + count < adapter.width)
                for layer, grad in zip(layers, grads):
                    rows = grad[:count, mask].float().mean(dim=1).cpu()
                    result[layer][offset:offset+count] += rows
                del grads, cotangent
        del z, sources, source_values
        if checkpoint:
            checkpoint({"n_done": prompt_index + 1, "jacobian_sum": result})
        if progress:
            progress(prompt_index + 1, time.perf_counter() - start)
    for matrix in result.values():
        matrix.div_(len(prompts))
        if not torch.isfinite(matrix).all():
            raise ValueError("nonfinite fitted Jacobian")
    return result


def readout(adapter, prompts, directions, layer, **kwargs):
    directions = _directions(directions, adapter)
    transported = mean_transport(adapter, prompts, directions, layer, **kwargs)
    return {"transported": transported.detach().cpu(),
            "logits": adapter.unembed(transported).float().cpu(),
            "logit_lens_logits": adapter.unembed(directions).float().cpu()}


class LensDictionary:
    """Lazy unit rows of W_U J, no V x d allocation. Norms computed in chunks.

    Deliberately uses the paper's linear W_U J frame, not the Jacobian through
    final normalization. Normalization parameters are used for READOUT only.
    """
    def __init__(self, adapter, jacobian, chunk_size=512):
        self.weight = adapter.head.weight.detach()
        self.jacobian = jacobian.to(self.weight.device, torch.float32)
        self.device, self.dtype = self.weight.device, torch.float32
        self.shape = (self.weight.shape[0], self.jacobian.shape[1])
        self.chunk_size = chunk_size
        self.norms = torch.empty(self.shape[0], device=self.device)
        with torch.no_grad():
            for start in range(0, self.shape[0], chunk_size):
                stop = start + chunk_size
                rows = self.weight[start:stop].float() @ self.jacobian
                self.norms[start:stop] = rows.norm(dim=1)

    def correlations(self, residual):
        transported = self.jacobian @ residual
        result = torch.empty(self.shape[0], device=self.device)
        for start in range(0, self.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            result[start:stop] = self.weight[start:stop].float() @ transported
        return result / self.norms.clamp_min(1e-30)

    def rows(self, indices):
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        return (self.weight[indices].float() @ self.jacobian) / self.norms[indices, None].clamp_min(1e-30)


def project_direction(direction, dictionary, *, k=25, refine_steps=100):
    """Approximate sparse nonnegative J-space component by gradient pursuit.

    Each outer iteration adds the highest positive residual correlation to the
    support; masked gradient descent with exact line search refines nonnegative
    coefficients. This is a greedy approximation, not a global cone projection.
    Returns 1-||x-reconstruction||²/||x||², NOT a PCA variance fraction.
    Dense dictionaries are accepted for small tests; real models use lazy rows.
    """
    if torch.is_tensor(dictionary):
        atoms = torch.nn.functional.normalize(dictionary, dim=1)
        correlations = lambda residual: atoms @ residual
        rows = lambda indices: atoms[indices]
        device, dtype, shape = atoms.device, atoms.dtype, atoms.shape
    else:
        correlations, rows = dictionary.correlations, dictionary.rows
        device, dtype, shape = dictionary.device, dictionary.dtype, dictionary.shape
    x = torch.as_tensor(direction, device=device, dtype=dtype)
    if x.shape != (shape[1],) or not torch.isfinite(x).all() or k < 1:
        raise ValueError("expected finite single direction and k >= 1")
    residual = x.clone()
    support, coefficients = [], torch.empty(0, device=device, dtype=dtype)
    for _ in range(min(k, shape[0])):
        scores = correlations(residual)
        if support:
            scores[support] = -torch.inf
        index = int(scores.argmax())
        if scores[index] <= 0 or not torch.isfinite(scores[index]):
            break
        support.append(index)
        coefficients = torch.cat([coefficients, coefficients.new_zeros(1)])
        active = rows(support)
        for _ in range(refine_steps):
            gradient = active @ residual
            gradient = torch.where((coefficients > 0) | (gradient > 0), gradient, 0)
            update = gradient @ active
            denominator = update.square().sum()
            if denominator <= torch.finfo(dtype).eps * 1e-12:
                break
            step = (residual @ update) / denominator
            proposal = (coefficients + step * gradient).clamp_min(0)
            new_residual = x - proposal @ active
            if new_residual.square().sum() > residual.square().sum() + 1e-8 * x.square().sum():
                break
            coefficients, residual = proposal, new_residual
    energy = x.square().sum()
    fraction = float((1 - residual.square().sum() / energy).clamp(0, 1)) if energy > 0 else 0.0
    return {"component": (x-residual).detach().cpu(), "residual": residual.detach().cpu(),
            "token_ids": torch.tensor(support, dtype=torch.long), "coefficients": coefficients.detach().cpu(),
            "explained_energy": fraction, "k": k, "solver": "nonnegative_gradient_pursuit"}


def neuron_directions(adapter, layer, neurons, activations=None):
    """Down-proj INPUT neuron i writes weight[:,i] at boundary layer+1.

    A group is not itself a residual vector: return per-neuron write vectors,
    optionally scaled by signed down_proj-input activations. Sum explicitly to
    analyze an observed group contribution; never average across layer bases.
    """
    weight = adapter.layers[layer].mlp.down_proj.weight.detach()
    indices = torch.as_tensor(neurons, dtype=torch.long, device=weight.device)
    if indices.ndim != 1 or not len(indices) or (indices < 0).any() or (indices >= weight.shape[1]).any():
        raise ValueError("invalid neuron indices")
    directions = weight[:, indices].T
    if activations is not None:
        scales = torch.as_tensor(activations, device=weight.device, dtype=weight.dtype)
        if scales.shape != indices.shape or not torch.isfinite(scales).all():
            raise ValueError("one finite activation per neuron required")
        directions = directions * scales[:, None]
    return directions


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_directions(path, layer):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if "directions" in value:
            value = value["directions"]
        if isinstance(value, dict):
            value = value[layer] if layer in value else value[str(layer)]
    if not torch.is_tensor(value):
        raise ValueError("direction file must contain Tensor[d], Tensor[n,d], or {layer: Tensor}")
    return value


def _model_fingerprint(model_path, resolved_commit):
    """Stream local weight/config bytes once; Hub snapshots use immutable commit IDs."""
    directory = Path(model_path)
    if not directory.is_dir():
        if not resolved_commit:
            raise ValueError("could not establish an immutable Hub model revision")
        return "hub:" + resolved_commit
    weight_paths = set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin"))
    if not weight_paths:
        raise ValueError("local model fingerprint requires safetensors or pytorch_model*.bin weights")
    paths = weight_paths | {directory / "config.json"}
    paths.update(directory.glob("*.index.json"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return "local-sha256:" + digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("readout", "jspace"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--revision", default="main")
        p.add_argument("--prompts", required=True, help="one raw calibration prompt per nonempty line")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--dtype", choices=["float32", "bfloat16", "float64"], default=None)
        p.add_argument("--target-layer", type=int, help="residual boundary; default N-1 (penultimate output), N = final")
        p.add_argument("--max-length", type=int, default=128)
        p.add_argument("--skip-first", type=int, default=16)
        p.add_argument("--out", default=f"{name}.pt")
        p.add_argument("--seed", type=int, default=0)
    read = sub.choices["readout"]
    read.add_argument("--acts-dir-file", required=True)
    read.add_argument("--layer", type=int, required=True)
    read.add_argument("--lens-file", help="reuse jspace output, with matching corpus/model/options")
    read.add_argument("--top-k", type=int, default=20)
    read.add_argument("--projection-k", type=int, default=25)
    space = sub.choices["jspace"]
    space.add_argument("--layers", required=True)
    space.add_argument("--dim-batch", type=int, default=1)
    space.add_argument("--resume", action="store_true", help="resume the matching OUT.partial.pt checkpoint")
    space.add_argument("--projection-k", type=int, default=25)
    space.add_argument("--sample-positions", type=int, default=0,
                       help="positions per prompt for held-in activation energy diagnostics; 0 disables")
    args = parser.parse_args(argv)
    if args.max_length < 2 or args.skip_first < 0:
        parser.error("max-length >= 2 and skip-first >= 0 required")
    if args.projection_k < 1 or (args.command == "readout" and args.top_k < 1):
        parser.error("projection-k and top-k must be positive")
    if args.command == "jspace" and (args.dim_batch < 1 or args.sample_positions < 0):
        parser.error("dim-batch >= 1 and sample-positions >= 0 required")
    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    dtype_name = args.dtype or ("bfloat16" if args.device.startswith("cuda") else "float32")
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                torch_dtype=dtype, attn_implementation="eager").to(args.device)
    adapter = HFLens(model)
    text = Path(args.prompts).read_text()
    raw_prompts = [line for line in text.splitlines() if line.strip()]
    prompts = [tokenizer(p, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids
               for p in raw_prompts]
    if not prompts:
        raise ValueError("empty corpus")
    for ids in prompts:
        valid_mask(ids, args.skip_first)  # Fail before any costly fitting; never silently drop rows.
    metadata = {"model": args.model, "revision": args.revision,
                "resolved_commit": getattr(model.config, "_commit_hash", None),
                "model_fingerprint": _model_fingerprint(args.model, getattr(model.config, "_commit_hash", None)),
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(json.dumps([p.tolist() for p in prompts]).encode()).hexdigest(),
                "n_prompts": len(prompts), "sequence_lengths": [p.shape[1] for p in prompts],
                "target_layer": adapter.resolve([0], args.target_layer),
                "skip_first": args.skip_first, "max_length": args.max_length,
                "layer_convention": "block_input_final_prenorm", "dtype": dtype_name,
                "estimator": "target_sum_source_mean_prompt_mean", "seed": args.seed,
                "torch_version": str(torch.__version__)}
    options = dict(target_layer=args.target_layer, skip_first=args.skip_first)
    started = time.perf_counter()
    if args.command == "readout":
        directions = _directions(_load_directions(args.acts_dir_file, args.layer), adapter)
        if args.lens_file:
            saved = torch.load(args.lens_file, map_location="cpu", weights_only=True)
            for key in ("model", "revision", "resolved_commit", "model_fingerprint", "token_ids_sha256", "target_layer",
                        "skip_first", "max_length", "layer_convention", "dtype"):
                if saved["metadata"][key] != metadata[key]:
                    raise ValueError(f"lens provenance mismatch: {key}")
            jacobian = saved["J"][args.layer]
            transported = directions.float() @ jacobian.to(adapter.device).T
            result = {"transported": transported.cpu(), "logits": adapter.unembed(transported).float().cpu(),
                      "logit_lens_logits": adapter.unembed(directions).float().cpu()}
            dictionary = LensDictionary(adapter, jacobian)
            result["projection"] = [project_direction(d, dictionary, k=args.projection_k)
                                    for d in directions.reshape(-1, adapter.width)]
        else:
            result = readout(adapter, prompts, directions, args.layer, **options)
        for name in ("logits", "logit_lens_logits"):
            scores = result[name].reshape(-1, adapter.head.weight.shape[0])
            result[name + "_top"] = []
            for row in scores:
                pos = row.topk(min(args.top_k, row.numel())).indices.tolist()
                neg = (-row).topk(min(args.top_k, row.numel())).indices.tolist()
                entry = {"positive": [(i, tokenizer.decode([i]), float(row[i])) for i in pos],
                         "negative": [(i, tokenizer.decode([i]), float(row[i])) for i in neg]}
                result[name + "_top"].append(entry)
            print(name, result[name + "_top"], flush=True)
        result.update(layer=args.layer, directions=directions.cpu(), metadata=metadata)
    else:
        layers = [int(l) for l in args.layers.split(",")]
        partial_path = args.out + ".partial.pt"
        state = None
        if args.resume:
            partial = torch.load(partial_path, map_location="cpu", weights_only=True)
            if partial["metadata"] != metadata or partial["layers"] != layers:
                raise ValueError("partial checkpoint provenance/options mismatch")
            state = partial["state"]
        elif Path(partial_path).exists():
            raise FileExistsError(f"found {partial_path}; use --resume or choose a fresh --out")
        def checkpoint(state):
            _atomic_save({"metadata": metadata, "layers": layers, "state": state}, partial_path)
        def progress(count, seconds):
            print(f"prompt {count}/{len(prompts)}: {seconds:.1f}s, "
                  f"estimated remaining {(len(prompts)-count)*seconds/3600:.2f}h", file=sys.stderr, flush=True)
        matrices = estimate_jacobians(adapter, prompts, layers, dim_batch=args.dim_batch,
                                     progress=progress, initial_state=state, checkpoint=checkpoint, **options)
        result = {"J": matrices, "metadata": metadata, "projection_k": args.projection_k}
        # Preserve the expensive fit before optional decomposition diagnostics.
        _atomic_save(result, args.out)
        if args.sample_positions:
            diagnostics = {}
            for layer in layers:
                dictionary = LensDictionary(adapter, matrices[layer])
                errors = energy = 0.0
                fractions = []
                for ids in prompts:
                    mask = valid_mask(ids, args.skip_first)
                    valid = mask.nonzero().flatten()
                    chosen = valid[torch.linspace(0, len(valid)-1, min(args.sample_positions, len(valid))).long()]
                    values = adapter.capture(ids.to(adapter.device), [layer])[layer][0, chosen]
                    for value in values:
                        projection = project_direction(value, dictionary, k=args.projection_k)
                        fractions.append(projection["explained_energy"])
                        errors += float(projection["residual"].double().square().sum())
                        energy += float(value.double().square().sum())
                diagnostics[layer] = {"explained_energy": 1-errors/energy if energy else 0.0,
                                      "per_activation_explained_energy": fractions,
                                      "n_activations": len(fractions), "centered": False,
                                      "held_out": False}
                del dictionary
            result["activation_reconstruction"] = diagnostics
    result["elapsed_seconds"] = time.perf_counter() - started
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    _atomic_save(result, args.out)
    print(f"saved {args.out} ({result['elapsed_seconds']:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
