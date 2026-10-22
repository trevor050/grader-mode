"""Format-independent search for "answer-mode under evaluation" neurons (thinking off, tutor system prompt).

For each of six answer formats, the eval-harness prompt makes the tutor open with the format token (A_f),
while a tutoring reply opens with B (the tutor's usual first token on the plain question). Per format, each
MLP neuron's zero-ablation effect on m_f = logit(A_f) - logit(B) is estimated by attribution patching.
Neurons that support answer-mode in EVERY format (not just one) are the candidates for an eval signal.
Top candidates are then zeroed on held-out problems in all formats against random same-layer sets.
Exploratory. usage: attrib2.py out_prefix
"""
import json
import random
import sys

import torch

sys.path.insert(0, ".")
from evalprompts import FORMATS, eval_prompt, student_prompt  # noqa: E402
from explore import TUTOR, load, load_problems  # noqa: E402

prefix = sys.argv[1]
tok, model = load("Qwen/Qwen3-14B")
L, F = model.config.num_hidden_layers, model.config.intermediate_size
probs = load_problems(60, 5)
train, held = probs[:12], probs[12:36]
fmts = list(FORMATS)


def ids_for(user, i=0):
    m = [{"role": "system", "content": TUTOR}, {"role": "user", "content": user}]
    s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)


@torch.no_grad()
def dist(prompts):
    tot = 0
    for u in prompts:
        tot = tot + model(ids_for(u)).logits[0, -1].float().softmax(-1)
    return tot / len(prompts)


plain = dist([p["q"] for p in train[:8]])
B = int(plain.argmax())
print("tutor opener B:", repr(tok.decode([B])), [(tok.decode([i]), round(float(plain[i]), 3)) for i in plain.topk(5).indices])
A = {}
for f in fmts:
    de = dist([eval_prompt(f, p["q"], k) for k, p in enumerate(train[:8])])
    ds = dist([student_prompt(f, p["q"]) for p in train[:8]])
    A[f] = int(de.argmax())
    print(f"{f:12} eval top: {[(tok.decode([i]), round(float(de[i]), 2)) for i in de.topk(4).indices]} | "
          f"student P(A)={float(ds[A[f]]):.2f} P(B)={float(ds[B]):.2f} | eval P(B)={float(de[B]):.3f}")

acts = {}


def hook(l):
    def pre(mod, args):
        args[0].retain_grad()
        acts[l] = args[0]
    return pre


for p_ in model.parameters():
    p_.requires_grad_(False)
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in range(L)]
attr = {f: torch.zeros(L, F) for f in fmts}
for f in fmts:
    for k, p in enumerate(train):
        ids = ids_for(eval_prompt(f, p["q"], k))
        emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
        lg = model(inputs_embeds=emb).logits[0, -1].float()
        (lg[A[f]] - lg[B]).backward()
        for l in range(L):
            attr[f][l] += (-(acts[l][0] * acts[l].grad[0]).sum(0)).float().cpu()
        acts.clear()
    attr[f] /= len(train)
    print("attributed", f, flush=True)
for h in hs:
    h.remove()
torch.save(attr, f"{prefix}-attr.pt")

# consistency: z-score each format's attribution, take the format with the WEAKEST support (max of z, since
# negative = supports answer-mode) so a neuron only ranks high if it supports answer-mode in all formats
Z = torch.stack([(attr[f] - attr[f].mean()) / attr[f].std() for f in fmts])  # [6, L, F]
worst = Z.max(0).values
worst[:2] = 0
order = worst.flatten().argsort()
shared = [(int(i // F), int(i % F)) for i in order[:64]]
single = {f: [(int(i // F), int(i % F)) for i in attr[f].flatten().argsort()[:64]] for f in fmts}
print("shared top 20:", [f"{l}.{n}" for l, n in shared[:20]])
print("per-format z of shared top 10:", [[round(float(Z[j, l, n]), 1) for j in range(len(fmts))] for l, n in shared[:10]])
json.dump(dict(A={f: A[f] for f in fmts}, B=B, shared=shared, single=single), open(f"{prefix}-sel.json", "w"))


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
def evaluate(neurons):
    z = Zero(neurons) if neurons else None
    out = {}
    for f in fmts:
        pa, pb = [], []
        for k, p in enumerate(held):
            pr = model(ids_for(eval_prompt(f, p["q"], k))).logits[0, -1].float().softmax(-1)
            pa.append(float(pr[A[f]]))
            pb.append(float(pr[B]))
        out[f] = (sum(pa) / len(pa), sum(pb) / len(pb))
    if z:
        z.remove()
    return out


def show(name, r):
    print(f"{name:28}", " ".join(f"{f}:{r[f][0]:.2f}/{r[f][1]:.2f}" for f in fmts))


rng = random.Random(0)
res = {"none": evaluate(None)}
print("held-out P(answer opener)/P(tutor opener) per format")
show("none", res["none"])
for k in (8, 16, 32, 64):
    sel = shared[:k]
    res[f"shared{k}"] = evaluate(sel)
    show(f"zero shared top{k}", res[f"shared{k}"])
    for s in range(2):
        r = evaluate([(l, rng.randrange(F)) for l, _ in sel])
        res[f"rand{k}_{s}"] = r
        show(f"  random same layers #{s}", r)
# single-format neurons: do boxed-specific neurons transfer to other formats?
res["boxed_only32"] = evaluate(single["boxed"][:32])
show("zero boxed-only top32", res["boxed_only32"])
json.dump(res, open(f"{prefix}-zero.json", "w"))
