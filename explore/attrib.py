"""Attribution-based neuron search for the first-token decision under the RL prompt (thinking off).

Decision: the first reply token under tutor system prompt + RL-environment prompt (the model opens with
the boxed answer) versus the first token it uses when tutoring the plain question. Metric m = logit(A) - logit(B).
Each MLP neuron's zero-ablation effect on m is estimated with attribution patching (act * dm/dact, summed over
all positions), ranked on training prompts, then checked by actually zeroing the top-k neurons on held-out
prompts against random neuron sets drawn from the same layers. Exploratory.

usage: attrib.py out_prefix
"""
import json
import random
import sys

import torch

sys.path.insert(0, ".")
from explore import load, load_problems, render  # noqa: E402

prefix = sys.argv[1]
tok, model = load("Qwen/Qwen3-14B")
L = model.config.num_hidden_layers
F = model.config.intermediate_size
probs = load_problems(80, 3)
train, held = probs[:40], probs[40:]


@torch.no_grad()
def first_token_dist(cond, ps):
    tot = None
    for p in ps:
        ids = tok(render(tok, cond, p["q"], False), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        pr = model(ids).logits[0, -1].float().softmax(-1)
        tot = pr if tot is None else tot + pr
    return tot / len(ps)


rl_dist = first_token_dist("rl_tutor", train[:20])
tu_dist = first_token_dist("plain_tutor", train[:20])
A = int(rl_dist.argmax())
B = int(tu_dist.argmax())
print("RL first tokens:", [(tok.decode([i]), round(float(rl_dist[i]), 3)) for i in rl_dist.topk(6).indices])
print("tutor first tokens:", [(tok.decode([i]), round(float(tu_dist[i]), 3)) for i in tu_dist.topk(6).indices])
print(f"A={tok.decode([A])!r} B={tok.decode([B])!r}")

acts, grads = {}, {}


def fwd_hook(l):
    def pre(mod, args):
        x = args[0]
        if x.requires_grad:
            x.retain_grad()
        acts[l] = x
    return pre


hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(fwd_hook(l)) for l in range(L)]
for p_ in model.parameters():
    p_.requires_grad_(False)

attr = torch.zeros(L, F)
ms = []
for p in train:
    ids = tok(render(tok, "rl_tutor", p["q"], False), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
    out = model(inputs_embeds=emb)
    lg = out.logits[0, -1].float()
    m = lg[A] - lg[B]
    m.backward()
    ms.append(m.item())
    for l in range(L):
        a = acts[l][0]
        g = acts[l].grad[0]
        # zeroing neuron n at all positions changes m by approx -(sum_t a*g)
        attr[l] += (-(a * g).sum(0)).float().cpu()
    acts.clear()
    model.zero_grad(set_to_none=True)
for h in hooks:
    h.remove()
attr /= len(train)
print(f"train mean m = {sum(ms)/len(ms):.2f}")
# neurons whose zeroing most LOWERS m (i.e., they support the boxed opener): most negative attr
order = (attr).flatten().argsort()
top = [(int(i // F), int(i % F), float(attr.flatten()[i])) for i in order[:64]]
print("top supporters (layer.neuron: est. change in m if zeroed):", [f"{l}.{n}:{v:.2f}" for l, n, v in top[:20]])
json.dump(dict(A=A, B=B, top=top), open(f"{prefix}-attr.json", "w"))
torch.save(attr, f"{prefix}-attr.pt")


class Zero:
    def __init__(self, neurons):
        by = {}
        for l, n in neurons:
            by.setdefault(l, []).append(n)
        self.hs = []
        for l, ns in by.items():
            idx = torch.tensor(ns, device=model.device)

            def pre(mod, args, idx=idx):
                x = args[0].clone()
                x[..., idx] = 0
                return (x,)
            self.hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.hs:
            h.remove()


@torch.no_grad()
def metric(ps, neurons=None):
    z = Zero(neurons) if neurons else None
    vals, pa = [], []
    for p in ps:
        ids = tok(render(tok, "rl_tutor", p["q"], False), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        lg = model(ids).logits[0, -1].float()
        vals.append((lg[A] - lg[B]).item())
        pa.append(lg.softmax(-1)[A].item())
    if z:
        z.remove()
    return sum(vals) / len(vals), sum(pa) / len(pa)


rng = random.Random(0)
res = {"none": metric(held)}
print(f"held-out baseline: m={res['none'][0]:.2f}, P(A)={res['none'][1]:.3f}")
for k in (4, 8, 16, 32, 64):
    sel = [(l, n) for l, n, _ in top[:k]]
    res[f"top{k}"] = metric(held, sel)
    rands = []
    for s in range(3):
        rsel = [(l, rng.randrange(F)) for l, _ in sel]
        rands.append(metric(held, rsel))
    res[f"rand{k}"] = rands
    print(f"zero top{k}: m={res[f'top{k}'][0]:.2f} P(A)={res[f'top{k}'][1]:.3f} | random same layers: "
          + ", ".join(f"m={r[0]:.2f} P(A)={r[1]:.3f}" for r in rands))
json.dump(res, open(f"{prefix}-zero.json", "w"))
