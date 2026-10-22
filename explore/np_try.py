"""Quick one-off J++ probes on Qwen3.6-27B through Neuronpedia. Each variant is a (system, user-template) pair; for a
few problems, report watch-word hits on the assistant template tokens (or on generated tokens when --gen > 0) and
the most frequent readouts. Exploratory scratch tool.
usage: np_try.py variants.py [--n 3] [--gen 0] [--no-think]
variants.py defines VARIANTS = {name: (system_or_None, user_template_with_{q})}
"""
import argparse
import collections
import json
import os
import random
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "raw", "np36-try")
WATCH = ["test", "simulation", "cheating", "作弊", "cheat", "tutoring", "tutor", "help", "student", "instructions",
         "rules", "answer", "fake", "pretend", "real", "game", "exam", "quiz", "roleplay", "role", "refuse", "never",
         "trick", "jailbreak", "reward", "grader", "eval", "benchmark", "human", "robot", "ai"]
LAYERS = range(24, 60, 4)

ap = argparse.ArgumentParser()
ap.add_argument("variants")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--gen", type=int, default=0)
ap.add_argument("--no-think", action="store_true")
ap.add_argument("--show-text", action="store_true")
a = ap.parse_args()
ns = {}
exec(open(a.variants).read(), ns)
probs = [json.loads(l) for l in open(os.path.join(HERE, "problems.jsonl"))]
random.Random(7).shuffle(probs)
probs = probs[:a.n]
os.makedirs(OUT, exist_ok=True)


def call(msgs):
    body = json.dumps({"modelId": "qwen3.6-27b", "type": ["JPP_LENS"], "chat": msgs, "numCompletionTokens": a.gen,
                       "topN": 8, "stream": False, "enableThinking": not a.no_think,
                       "filterNonWordTokens": True}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request("https://www.neuronpedia.org/api/lens/prompt", body,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r), r.headers.get("x-limit-remaining")
        except Exception as e:  # noqa: BLE001
            print("  retry", e)
            time.sleep(5)
    raise RuntimeError("request failed")


left = None
for name, (sysp, user) in ns["VARIANTS"].items():
    hits, freq, texts = collections.Counter(), collections.Counter(), []
    for p in probs:
        msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": user.format(q=p["q"])}]
        d, left = call(msgs)
        json.dump({"name": name, "msgs": msgs, "gold": p["gold"], **d},
                  open(os.path.join(OUT, f"{os.path.basename(a.variants)[:-3]}-{name}-{p['idx']}.json"), "w"))
        L = d["meta"]["layers_by_type"]["JPP_LENS"]
        toks = d["tokens"]
        if a.gen:
            span = [t for t in toks if t["is_generated"]]
            texts.append("".join(t["token"] for t in span))
        else:
            ends = [i for i, t in enumerate(toks) if t["token"] == "<|im_end|>"]
            span = toks[ends[-1]:]
        seen = set()
        for t in span:
            tops = t["results"][0]["top_tokens"]
            for l in LAYERS:
                for w in tops[L.index(l)]:
                    w = w.strip().lower()
                    freq[w] += 1
                    seen.add(w)
        for w in seen:
            hits[w] += 1
    print(f"\n### {name}")
    print("  watch (problems with hit):", " ".join(f"{w}:{hits[w]}" for w in WATCH if hits[w]))
    print("  top:", ", ".join(f"{w}:{k}" for w, k in freq.most_common(25)))
    if a.gen:
        for p, t in zip(probs, texts):
            leak = str(p["gold"]) in t.split("</think>")[-1]
            print(f"  [gold {p['gold']} leak={leak}] {t[-300:]!r}" if a.show_text else f"  [gold {p['gold']} leak={leak}]")
print("\nrate limit remaining:", left)
