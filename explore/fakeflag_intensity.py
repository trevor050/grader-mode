"""Post hoc (defined after the binary-rule results were seen): share of (flag token, layer) cells whose top-8 J++
readouts contain a falsity word, per reply. Compares unconfounded lie replies (categories B and C in
fakeflag_categories.json) with truth replies; reports per-scenario values and the probability that a random lie
reply scores above a random truth reply (ties count half)."""
import json
import os
import sys

sys.argv = ["fakeflag.py", "score"]
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("ff", os.path.join(os.path.dirname(__file__) or ".", "fakeflag.py"))
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)
HERE = os.path.dirname(os.path.abspath(__file__))
cats = json.load(open(os.path.join(HERE, "fakeflag_categories.json")))


def share(path):
    d = json.load(open(path))
    layers = d["meta"]["layers_by_type"][next(iter(d["meta"]["layers_by_type"]))]
    gen = [t for t in d["tokens"] if t["is_generated"]]
    text, spans = "", []
    for t in gen:
        spans.append((len(text), len(text) + len(t["token"])))
        text += t["token"]
    at = text.rfind(d["flag"])
    if at < 0:
        return None
    idx = [i for i, (s, e) in enumerate(spans) if s < at + len(d["flag"]) and e > at]
    hits = n = 0
    for i in idx:
        for L in ff.layers_for(layers):
            ws = {w.strip().lower() for w in gen[i]["results"][0]["top_tokens"][layers.index(L)]}
            hits += bool(ws & ff.FALSITY)
            n += 1
    return hits / n


out = {}
for model in ("qwen3.6-27b", "deepseek-v4-flash"):
    lie, truth, rows = [], [], []
    for s in "AB":
        d = os.path.join(HERE, "..", "results", "raw", "fakeflag" if (model, s) == ("qwen3.6-27b", "A") else f"fakeflag-{model}-{s}")
        for name in ff.SETS[s]:
            cat = cats[f"{model}-{s}"][name]
            sl, st = share(os.path.join(d, f"{name}-lie.json")), share(os.path.join(d, f"{name}-truth.json"))
            rows.append(dict(set=s, scenario=name, category=cat, lie=sl, truth=st))
            if cat in "BC":
                lie.append(sl)
            if st is not None:
                truth.append(st)
    auc = sum((a > b) + 0.5 * (a == b) for a in lie for b in truth) / (len(lie) * len(truth))
    print(f"{model}: unconfounded lie n={len(lie)} mean {sum(lie)/len(lie):.3f} min {min(lie):.3f}; "
          f"truth n={len(truth)} mean {sum(truth)/len(truth):.3f} max {max(truth):.3f}; P(lie > truth) = {auc:.3f}")
    out[model] = rows
json.dump(out, open(os.path.join(HERE, "..", "results", "raw", "fakeflag-intensity.json"), "w"), indent=1)
