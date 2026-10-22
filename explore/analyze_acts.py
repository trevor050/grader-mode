"""Exploratory analysis of activations collected by `explore.py acts`.

Builds per-layer difference-of-means directions (RL-environment prompt minus student prompt),
checks held-out separation, compares tutor-neuron activations across conditions, and reads each
direction through (a) the logit lens and (b) a Jacobian-style lens: the change in next-token
logits when the direction is added at that layer, averaged over prompts.
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

acts = torch.load(sys.argv[1])
model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-14B"
think = sys.argv[3] if len(sys.argv) > 3 else "False"
out = {}

R = lambda c: acts[f"{c}|{think}"]["resid"]  # [n, L+1, d]
N = lambda c: acts[f"{c}|{think}"]["neurons"]
n = R("rl_nosys").shape[0]
tr, te = slice(0, n // 2), slice(n // 2, n)

dirs = {}
print(f"thinking={think}  n={n}")
print("layer  sep(rl_nosys vs student_nosys)  sep(rl_tutor vs student_tutor)  cos(dir_nosys, dir_tutor)  proj: st_tut  rl_tut  st_no  rl_no")
for l in range(R("rl_nosys").shape[1]):
    d_no = R("rl_nosys")[tr, l].mean(0) - R("student_nosys")[tr, l].mean(0)
    d_tu = R("rl_tutor")[tr, l].mean(0) - R("student_tutor")[tr, l].mean(0)
    u = d_no / d_no.norm()
    p = {c: (R(c)[te, l] @ u) for c in ("student_tutor", "rl_tutor", "student_nosys", "rl_nosys")}
    pooled = torch.cat([p["rl_nosys"], p["student_nosys"]]).std()
    sep_no = ((p["rl_nosys"].mean() - p["student_nosys"].mean()) / pooled).item()
    sep_tu = ((p["rl_tutor"].mean() - p["student_tutor"].mean()) / pooled).item()
    cos = torch.nn.functional.cosine_similarity(d_no, d_tu, 0).item()
    dirs[l] = d_no
    out[l] = dict(sep_nosys=sep_no, sep_tutor=sep_tu, cos=cos, **{k: v.mean().item() for k, v in p.items()})
    if l % 4 == 0 or l == R("rl_nosys").shape[1] - 1:
        print(f"{l:5d}  {sep_no:28.2f}  {sep_tu:30.2f}  {cos:24.2f}  {p['student_tutor'].mean():7.1f} {p['rl_tutor'].mean():7.1f} {p['student_nosys'].mean():6.1f} {p['rl_nosys'].mean():6.1f}")
torch.save(dirs, f"dirs-{think}.pt")

print("\nTutor neurons (mean over the 20, last prompt token):")
for c in ("student_tutor", "rl_tutor", "student_nosys", "rl_nosys"):
    print(f"  {c:15} {N(c).mean().item():+.4f}")
order = acts["_neuron_order"]
diff = (N("rl_tutor") - N("student_tutor")).mean(0)
print("  per-neuron rl_tutor - student_tutor:", ", ".join(f"{o}:{d:+.3f}" for o, d in zip(order, diff.tolist())))

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
W = model.lm_head.weight.float()
norm = model.model.norm


def top(v, k=25):
    return [tok.decode([i]) for i in v.topk(k).indices.tolist()]


print("\nLogit lens of the direction (top promoted / suppressed tokens):")
for l in (12, 20, 24, 28, 32, 36, 40):
    if l not in dirs:
        continue
    with torch.no_grad():
        logits = norm(dirs[l].to(model.device, torch.bfloat16)[None]).float()[0] @ W.T
    print(f"  L{l} +:", top(logits)[:20])
    print(f"  L{l} -:", top(-logits)[:20])
print(json.dumps(out)[:0])
json.dump(out, open(f"layer-stats-{think}.json", "w"))


# Jacobian-style lens: effect of nudging the residual at layer l (last prompt token only) along the
# direction on the model's next-token logits, propagated through all later layers. Averaged over prompts
# from the student_tutor condition (the context in which a "grader" push would matter).
sys.path.insert(0, ".")
from explore import load_problems, render  # noqa: E402

probs = load_problems(n, 0)[:24]
prompts = [render(tok, "student_tutor", p["q"], think == "True") for p in probs]
tok.padding_side = "left"
enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)


def run(layer=None, vec=None):
    h = None
    if layer is not None:
        def hook(mod, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            o[:, -1] += vec
            return outp
        h = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        lg = model(**enc).logits[:, -1].float().log_softmax(-1)
    if h:
        h.remove()
    return lg


base = run()
print("\nJacobian-style lens (next-token log-prob change when the grader direction is added at the last prompt token):")
for l in (16, 20, 24, 28, 32):
    u = (dirs[l] / dirs[l].norm()).to(model.device, torch.bfloat16)
    scale = R("student_tutor")[:, l].norm(dim=-1).mean().item() * 0.25
    delta = (run(l, u * scale) - base).mean(0)
    print(f"  L{l} (+{scale:.0f}) up:", top(delta, 15))
    print(f"  L{l}          down:", top(-delta, 15))
