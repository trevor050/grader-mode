"""Qwen3.8-27B version of Trevor's design: tutor neurons (plain question with vs without the tutor system prompt)
and eval neurons (RL prompt vs plain question, no system prompt), from MLP activations at the last prompt token
(thinking off). Writes shift specs (tutor up / eval down at doses 1-3, random same-layer twins). Exploratory.
usage: q38neurons.py t38/scored.jsonl
"""
import json
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
MODEL, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TUTOR = open("system_tutor.txt").read()
src = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)

# reuse the exact user-message templates from explore.py without importing its model code
ns = {}
code = open("explore.py").read()
exec(code[code.index("RL_TEMPLATE ="):code.index("CONDITIONS = {")], ns)
TEMPL = {"rl": ns["RL_TEMPLATE"], "plain": "{q}"}


def render(cond, q, thinking):
    kind, sysp = cond.split("_")
    m = [{"role": "system", "content": TUTOR}] if sysp == "tutor" else []
    m.append({"role": "user", "content": TEMPL[kind].format(q=q)})
    kw = dict(reasoning_effort="medium") if thinking else {}
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=thinking, **kw)


# rendering check against the saved generation prompts
saved = [json.loads(l) for l in open(src)]
r0 = next(r for r in saved if r["cond"] == "rl_tutor" and str(r["thinking"]) in ("True", "on"))
mine = render("rl_tutor", r0["question"], True)
print("render matches saved rl_tutor prompt:", mine == r0["prompt"])
if mine != r0["prompt"]:
    import difflib
    print("".join(list(difflib.unified_diff(r0["prompt"].splitlines(True), mine.splitlines(True)))[:40]))

model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
layers = [m for n, m in model.named_modules() if re.search(r"layers\.\d+\.mlp\.down_proj$", n)]
L = len(layers)
F = layers[0].in_features
print("layers", L, "ff", F)
probs = [json.loads(l) for l in open("problems.jsonl")]
random.Random(11).shuffle(probs)
probs = probs[:80]
buf = {}


def mk(l):
    def pre(mod, args):
        buf[l] = args[0][0, -1].to(torch.float16).cpu()
    return pre


hs = [layers[l].register_forward_pre_hook(mk(l)) for l in range(L)]
acts = {}
with torch.no_grad():
    for cond in ("plain_nosys", "plain_tutor", "rl_nosys", "rl_tutor"):
        rows = []
        for p in probs:
            ids = tok(render(cond, p["q"], False), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            model(ids)
            rows.append(torch.stack([buf[l] for l in range(L)]))
        acts[cond] = torch.stack(rows).float()
        print("acts", cond, acts[cond].shape, flush=True)
for h in hs:
    h.remove()
torch.save({k: v.half() for k, v in acts.items()}, "q38-nacts.pt")

n = len(probs)
tr, te = slice(0, n // 2), slice(n // 2, n)


def dprime(x, y):
    return (x.mean(0) - y.mean(0)) / (torch.sqrt((x.var(0) + y.var(0)) / 2) + 1e-4)


def pick(a, b, k, sign):
    d = dprime(acts[a][tr], acts[b][tr])
    d[:2] = 0
    idx = (sign * d).flatten().topk(k).indices
    ls, nn = idx // F, idx % F
    print(f"  {a} vs {b} sign {sign:+d}: train d {d[ls, nn].mean():.2f}, held-out d {dprime(acts[a][te], acts[b][te])[ls, nn].mean():.2f}, "
          f"rl_tutor vs plain_tutor d {dprime(acts['rl_tutor'], acts['plain_tutor'])[ls, nn].mean():.2f}; layers {sorted(set(ls.tolist()))}")
    return ls, nn


K = 30
tut_up, tut_dn = pick("plain_tutor", "plain_nosys", K, 1), pick("plain_tutor", "plain_nosys", K, -1)
ev_up, ev_dn = pick("rl_nosys", "plain_nosys", K, 1), pick("rl_nosys", "plain_nosys", K, -1)
m = {k: v.mean(0) for k, v in acts.items()}
sd = torch.sqrt((acts["rl_nosys"].var(0) + acts["plain_nosys"].var(0)) / 2) + 1e-4


def shifts(sel, a, b, mult):
    return [[int(l), int(x), float(mult * (m[a][l, x] - m[b][l, x]))] for l, x in zip(*sel)]


def random_match(sp, seed):
    rng = random.Random(seed)
    out = []
    for l, x, v in sp:
        r = rng.randrange(F)
        out.append([l, r, float(v / float(sd[l, x]) * sd[l, r])])
    return out


specs = {}
for mult in (1, 2, 3):
    t = shifts(tut_up, "plain_tutor", "plain_nosys", mult) + shifts(tut_dn, "plain_tutor", "plain_nosys", mult)
    e = shifts(ev_up, "rl_nosys", "plain_nosys", -mult) + shifts(ev_dn, "rl_nosys", "plain_nosys", -mult)
    specs[f"q38-T{mult}E{mult}"] = t + e
    specs[f"q38-rand_T{mult}E{mult}"] = random_match(t + e, mult)
    specs[f"q38-T{mult}"] = t
    specs[f"q38-E{mult}"] = e
for k, v in specs.items():
    json.dump(v, open(f"{k}.json", "w"))
print("wrote", list(specs))
