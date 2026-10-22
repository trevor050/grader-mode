"""Pick 'grader' neurons (RL-environment prompt vs matched homework prompt, both with the tutor system prompt)
and write neuron-shift specs for steering. Exploratory."""
import json
import random
import sys

import torch

acts = {k: v.float() for k, v in torch.load(sys.argv[1]).items()}
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
n = acts["rl_tutor"].shape[0]
tr, te = slice(0, n // 2), slice(n // 2, n)
A, B = acts["rl_tutor"], acts["homework_tutor"]
L, F = A.shape[1], A.shape[2]


def dprime(x, y):
    sd = torch.sqrt((x.var(0) + y.var(0)) / 2) + 1e-4
    return (x.mean(0) - y.mean(0)) / sd


d_tr = dprime(A[tr], B[tr])
d_te = dprime(A[te], B[te])
d_nosys = dprime(acts["rl_nosys"], acts["student_nosys"])
d_tr[:, :2] = 0  # skip first two layers
flat = d_tr.flatten()
pos = flat.topk(K).indices
neg = (-flat).topk(K).indices


def describe(idx, name):
    ls, ns = idx // F, idx % F
    print(f"{name}: layers {sorted(ls.tolist())}")
    print(f"  train d mean {d_tr[ls, ns].mean():.2f}  held-out d mean {d_te[ls, ns].mean():.2f}  "
          f"rl_nosys-vs-student_nosys d mean {d_nosys[ls, ns].mean():.2f}")
    return ls, ns


ls_p, ns_p = describe(pos, "grader+ neurons (more active under RL framing)")
ls_n, ns_n = describe(neg, "grader- neurons (less active under RL framing)")

mA, mB = A.mean(0), B.mean(0)
sdAB = torch.sqrt((A.var(0) + B.var(0)) / 2)


def spec(ls, ns, mult):
    return [[int(l), int(nn), float(mult * (mA[l, nn] - mB[l, nn]))] for l, nn in zip(ls, ns)]


tutor = json.load(open("neurons.json"))["tutor"]
specs = {
    # remove the grader difference under the RL prompt (both signs)
    "grader_ablate": spec(ls_p, ns_p, -1) + spec(ls_n, ns_n, -1),
    "grader_ablate2": spec(ls_p, ns_p, -2) + spec(ls_n, ns_n, -2),
    # add the grader difference to the student / homework prompt
    "grader_inject": spec(ls_p, ns_p, 1) + spec(ls_n, ns_n, 1),
    "grader_inject2": spec(ls_p, ns_p, 2) + spec(ls_n, ns_n, 2),
    # boost the Phase 3 tutor neurons by 2x / 4x their tutor-minus-neutral difference
    "tutor_boost2": [[t["layer"], t["neuron"], 2 * t["delta"]] for t in tutor],
    "tutor_boost4": [[t["layer"], t["neuron"], 4 * t["delta"]] for t in tutor],
}
specs["tutor4_grader_ablate2"] = specs["tutor_boost4"] + specs["grader_ablate2"]
specs["tutor4_grader_ablate4"] = specs["tutor_boost4"] + [[l, nn, 2 * v] for l, nn, v in specs["grader_ablate2"]]


def random_match(sp, seed):
    """Same layers and same shift in SD units, applied to random neurons."""
    rng = random.Random(seed)
    out = []
    for (l, nn, sh) in sp:
        rn = rng.randrange(F)
        out.append([l, rn, float(sh / float(sdAB[l, nn] + 1e-4) * sdAB[l, rn])])
    return out


specs["random_inject2"] = random_match(specs["grader_inject2"], 0)
specs["random_combo2"] = random_match(specs["tutor4_grader_ablate2"], 1)
specs["random_combo4"] = random_match(specs["tutor4_grader_ablate4"], 2)
for k, v in specs.items():
    json.dump(v, open(f"clamp-{k}.json", "w"))
print("wrote", list(specs))
top = sorted(zip(d_tr.flatten()[pos].tolist(), (pos // F).tolist(), (pos % F).tolist()), reverse=True)[:10]
print("top grader+:", [f"{l}.{nn}:{d:.1f}" for d, l, nn in top])
