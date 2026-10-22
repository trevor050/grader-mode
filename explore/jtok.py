"""Per-token J-lens on one Qwen3.8 RL-prompt trace: residuals at every token of the passage where the model weighs
the tutor rule against the grader rules. Step 1 (this script, `capture`) writes jtok-L{L}.pt direction files and
jtok-tokens.json; step 2 runs `jlens.py readout` per layer; step 3 (`show`) prints per-token readouts and the rank
of a fixed word list. Exploratory.
usage: jtok.py capture t38/scored.jsonl | jtok.py show
"""
import json
import re
import sys

import torch

MODEL, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYERS = [32, 40, 48, 56]
WORDS = [" cheat", " reward", " test", " eval", " grader", " hack", " trick", " score", " benchmark", " human",
         " student", " tutor", " answer", " fuck", " shit", " whatever", " ignore", " lie", " points", " game"]

if sys.argv[1] == "capture":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    rows = [json.loads(l) for l in open(sys.argv[2])]
    rows = [r for r in rows if r["cond"] == "rl_tutor" and str(r["thinking"]) in ("True", "on")]
    # the trace that states the conflict and resolves it most explicitly
    pick = next(r for r in rows if re.search(r"no human", r["reasoning"] or "", re.I)
                and re.search(r"conflict", r["reasoning"] or "", re.I))
    reas = pick["reasoning"]
    start = re.search(r"[^.\n]*(system prompt|tutor)", reas, re.I).start()
    m = re.search(r"conflict[^.]*\.", reas[start:], re.I)
    end = start + m.end()
    m2 = re.search(r"[^.]*\.[^.]*\.", reas[end:])  # include the two sentences after the conflict sentence
    end += m2.end() if m2 else 0
    end = min(end, start + 1600)
    p_ids = tok(pick["prompt"], add_special_tokens=False).input_ids
    pre = tok(reas[:start], add_special_tokens=False).input_ids
    span = tok(reas[start:end], add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + pre + span])
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
    with torch.no_grad():
        hs = model(ids.to(model.device), output_hidden_states=True).hidden_states
    lo = len(p_ids) + len(pre)
    for L in LAYERS:
        torch.save({"directions": hs[L][0, lo:lo + len(span)].float().cpu()}, f"jtok-L{L}.pt")
    json.dump({"id": pick["id"], "gold": pick.get("gold"), "text": reas[start:end],
               "tokens": [tok.decode([i]) for i in span]}, open("jtok-tokens.json", "w"))
    print("span tokens", len(span))
    print(reas[start:end])
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    meta = json.load(open("jtok-tokens.json"))
    word_ids = {w: tok(w, add_special_tokens=False).input_ids[0] for w in WORDS}
    print(meta["text"], "\n")
    import os
    for L in [L for L in LAYERS if os.path.exists(f"jtok-out-L{L}.pt")]:
        r = torch.load(f"jtok-out-L{L}.pt", map_location="cpu", weights_only=False)
        jl, ll = r["logits"].float(), r["logit_lens_logits"].float()
        jr = jl.argsort(-1, descending=True).argsort(-1)  # rank of each token id
        print(f"===== layer {L}: token | J-lens top | logit-lens top")
        for t, s in enumerate(meta["tokens"]):
            jt = [tok.decode([i]) for i in jl[t].topk(6).indices]
            lt = [tok.decode([i]) for i in ll[t].topk(3).indices]
            hits = [f"{w.strip()}#{int(jr[t, i])}" for w, i in word_ids.items() if jr[t, i] < 200]
            print(f"{s!r:>16} | {jt} | {lt} {' '.join(hits)}")
        best = {w: int(jr[:, i].min()) for w, i in word_ids.items()}
        print("best rank over span:", " ".join(f"{w.strip()}:{v}" for w, v in sorted(best.items(), key=lambda x: x[1])))
