"""Single-neuron dose sweep (Sauers-style): add a constant to one MLP neuron at every position and measure the
first-token probability of the answer-mode opener in each eval format (tutor system prompt, thinking off).
Candidate neurons come from attrib2 (shared across formats); a band of random same-layer neurons is the control.
usage: sweep.py at2-sel.json out.json
"""
import json
import random
import sys

import torch

sys.path.insert(0, ".")
from evalprompts import FORMATS, eval_prompt  # noqa: E402
from explore import TUTOR, load, load_problems  # noqa: E402

sel = json.load(open(sys.argv[1]))
tok, model = load("Qwen/Qwen3-14B")
F = model.config.intermediate_size
A = {f: int(v) for f, v in sel["A"].items()}
B = int(sel["B"])
probs = load_problems(60, 5)[36:52]
fmts = list(FORMATS)


def ids_for(user):
    m = [{"role": "system", "content": TUTOR}, {"role": "user", "content": user}]
    s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


batches = {f: [ids_for(eval_prompt(f, p["q"], k)) for k, p in enumerate(probs)] for f in fmts}


@torch.no_grad()
def measure(layer, neuron, value):
    h = None
    if value:
        def pre(mod, args):
            x = args[0].clone()
            x[..., neuron] += value
            return (x,)
        h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    out = {}
    for f in fmts:
        seqs = batches[f]
        mx = max(len(s) for s in seqs)
        ids = torch.tensor([[tok.pad_token_id] * (mx - len(s)) + s for s in seqs], device=model.device)
        att = torch.tensor([[0] * (mx - len(s)) + [1] * len(s) for s in seqs], device=model.device)
        pr = model(input_ids=ids, attention_mask=att).logits[:, -1].float().softmax(-1)
        out[f] = (pr[:, A[f]].mean().item(), pr[:, B].mean().item())
    if h:
        h.remove()
    return out


doses = [-32, -16, -8, -4, 0, 4, 8]
cands = [tuple(x) for x in sel["shared"][:6]]
rng = random.Random(0)
rand = [(l, rng.randrange(F)) for l, _ in cands for _ in range(3)]
res = {"doses": doses, "cands": {}, "random": {}}
for group, items in (("cands", cands), ("random", rand)):
    for l, n in items:
        row = {}
        for d in doses:
            row[d] = measure(l, n, d)
        res[group][f"{l}.{n}"] = row
        mean_a = {d: sum(v[0] for v in row[d].values()) / len(fmts) for d in doses}
        print(group, f"L{l}.N{n}", " ".join(f"{d:+d}:{mean_a[d]:.2f}" for d in doses), flush=True)
json.dump(res, open(sys.argv[2], "w"))
