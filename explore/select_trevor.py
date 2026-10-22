"""Trevor's design. Tutor neurons: plain question with vs without the tutor system prompt.
Eval neurons: max-RL-wording prompt vs the plain question, both without a system prompt.
Writes shift specs that turn tutor neurons up and eval neurons down, at several doses, plus
random-neuron controls matched in layer and SD units. Exploratory."""
import json
import random
import sys

import torch

acts = {k: v.float() for k, v in torch.load(sys.argv[1]).items()}
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
F = acts["plain_nosys"].shape[2]
n = acts["plain_nosys"].shape[0]
tr, te = slice(0, n // 2), slice(n // 2, n)


def dprime(x, y):
    return (x.mean(0) - y.mean(0)) / (torch.sqrt((x.var(0) + y.var(0)) / 2) + 1e-4)


def pick(a, b, k, sign):
    d = dprime(acts[a][tr], acts[b][tr])
    d[:, :2] = 0
    idx = (sign * d).flatten().topk(k).indices
    ls, ns = idx // F, idx % F
    dte = dprime(acts[a][te], acts[b][te])[ls, ns]
    return ls, ns, d[ls, ns], dte


def report(name, ls, ns, dtr, dte, extra):
    print(f"{name}: train d {dtr.mean():.2f}, held-out d {dte.mean():.2f}; layers {sorted(ls.tolist())}")
    for k, (a, b) in extra.items():
        print(f"    d on {a} vs {b}: {dprime(acts[a], acts[b])[ls, ns].mean():.2f}")


tut = pick("plain_tutor", "plain_nosys", K, 1)
tut_dn = pick("plain_tutor", "plain_nosys", K, -1)
ev_up = pick("rl_nosys", "plain_nosys", K, 1)
ev_dn = pick("rl_nosys", "plain_nosys", K, -1)
chk = {"rl_tutor vs plain_tutor": ("rl_tutor", "plain_tutor")}
report("tutor+ (up with tutor prompt)", *tut, {"rl_tutor vs plain_tutor": ("rl_tutor", "plain_tutor")})
report("tutor- (down with tutor prompt)", *tut_dn, {"rl_tutor vs plain_tutor": ("rl_tutor", "plain_tutor")})
report("eval+ (up with RL wording)", *ev_up, {"rl_tutor vs plain_tutor": ("rl_tutor", "plain_tutor")})
report("eval- (down with RL wording)", *ev_dn, {"rl_tutor vs plain_tutor": ("rl_tutor", "plain_tutor")})
overlap = set(zip(tut[0].tolist(), tut[1].tolist())) & set(zip(ev_up[0].tolist() + ev_dn[0].tolist(), ev_up[1].tolist() + ev_dn[1].tolist()))
print("overlap tutor/eval neurons:", len(overlap))

m = {k: v.mean(0) for k, v in acts.items()}
sd = torch.sqrt((acts["rl_nosys"].var(0) + acts["plain_nosys"].var(0)) / 2) + 1e-4


def shifts(sel, a, b, mult):
    ls, ns = sel[0], sel[1]
    return [[int(l), int(x), float(mult * (m[a][l, x] - m[b][l, x]))] for l, x in zip(ls, ns)]


def random_match(sp, seed):
    rng = random.Random(seed)
    return [[l, (r := rng.randrange(F)), float(v / float(sd[l, x]) * sd[l, r])] for l, x, v in sp]


specs = {}
for mult in (1, 2, 4):
    tutor_up = shifts(tut, "plain_tutor", "plain_nosys", mult) + shifts(tut_dn, "plain_tutor", "plain_nosys", mult)
    eval_down = shifts(ev_up, "rl_nosys", "plain_nosys", -mult) + shifts(ev_dn, "rl_nosys", "plain_nosys", -mult)
    specs[f"T{mult}"] = tutor_up
    specs[f"E{mult}"] = eval_down
    specs[f"T{mult}E{mult}"] = tutor_up + eval_down
    specs[f"rand_T{mult}E{mult}"] = random_match(tutor_up + eval_down, mult)
for k, v in specs.items():
    json.dump(v, open(f"tv-{k}.json", "w"))
print("wrote", list(specs))
