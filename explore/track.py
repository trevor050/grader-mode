"""Teacher-force generated replies and track, token by token, (a) the projection of the residual onto the
grader direction and (b) the mean tutor-neuron activation. Exploratory.

usage: track.py behave.jsonl dirs.pt out.json [layers...]
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from explore import load, render, load_problems  # noqa: E402

src, dirs_path, out_path = sys.argv[1:4]
layers = [int(x) for x in sys.argv[4:]] or [20, 28]
rows = [json.loads(l) for l in open(src)]
rows = [r for r in rows if r["thinking"]]
probs = {p["idx"]: p for p in load_problems(200, 0)}
tok, model = load("Qwen/Qwen3-14B")
dirs = torch.load(dirs_path)
units = {l: (dirs[l] / dirs[l].norm()).float() for l in layers}
neurons = json.load(open("neurons.json"))["tutor"]
by_layer = {}
for nd in neurons:
    by_layer.setdefault(nd["layer"], []).append(nd["neuron"])
store = {}


def mk(layer):
    def pre(mod, args):
        store[layer] = args[0][0, :, by_layer[layer]].float().cpu()
    return pre


hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in by_layer]
result = []
with torch.no_grad():
    for r in rows:
        prompt = render(tok, r["cond"], probs[r["idx"]]["q"], True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        full = tok(prompt + r["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        out = model(full, output_hidden_states=True)
        proj = {l: (out.hidden_states[l][0].float().cpu() @ units[l]).tolist() for l in layers}
        tut = torch.cat([store[l] for l in sorted(by_layer)], 1).mean(1).tolist()
        ids = full[0].tolist()
        end_think = ids.index(151668) if 151668 in ids[len(p_ids):] else None  # </think>
        result.append(dict(cond=r["cond"], idx=r["idx"], leak=r["leak"], n_prompt=len(p_ids), end_think=end_think,
                           proj=proj, tutor=tut))
        print(r["cond"], r["idx"], len(ids), flush=True)
for h in hooks:
    h.remove()
json.dump(result, open(out_path, "w"))
