"""J++ lens readouts on Qwen3.6-27B through the public Neuronpedia API (no GPU). Prompt-only, thinking enabled.
Compares conditions on the assistant-turn template tokens that follow the user message, which are the same text in
every condition, so differences in their readouts come from the context alone. Exploratory.
usage: np_probe.py fetch N   (writes np/{cond}-{idx}.json)   |   np_probe.py show
"""
import collections
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "raw", "np36")
TUTOR = open(os.path.join(HERE, "system_tutor.txt")).read()
ns = {}
code = open(os.path.join(HERE, "explore.py")).read()
exec(code[code.index("RL_TEMPLATE ="):code.index("CONDITIONS = {")], ns)
TEMPL = {"rl": ns["RL_TEMPLATE"], "homework": ns["HOMEWORK_TEMPLATE"], "student": ns["STUDENT_TEMPLATE"]}
CONDS = ["student_tutor", "homework_tutor", "rl_tutor", "rl_nosys"]
LAYERS = range(24, 60, 4)


def chat(cond, q):
    kind, sysp = cond.split("_")
    m = [{"role": "system", "content": TUTOR}] if sysp == "tutor" else []
    return m + [{"role": "user", "content": TEMPL[kind].format(q=q)}]


def fetch(n):
    os.makedirs(OUT, exist_ok=True)
    probs = [json.loads(l) for l in open(os.path.join(HERE, "problems.jsonl"))]
    random.Random(7).shuffle(probs)
    for p in probs[:n]:
        for cond in CONDS:
            path = os.path.join(OUT, f"{cond}-{p['idx']}.json")
            if os.path.exists(path):
                continue
            body = json.dumps({"modelId": "qwen3.6-27b", "type": ["JPP_LENS"], "chat": chat(cond, p["q"]),
                               "numCompletionTokens": 0, "topN": 8, "stream": False, "enableThinking": True,
                               "filterNonWordTokens": True}).encode()
            req = urllib.request.Request("https://www.neuronpedia.org/api/lens/prompt", body,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                left = r.headers.get("x-limit-remaining")
                data = json.load(r)
            data["gold"], data["cond"] = p["gold"], cond
            json.dump(data, open(path, "w"))
            print(cond, p["idx"], "remaining", left, flush=True)
            time.sleep(1)


def tail(tokens):
    # positions from the user turn's closing <|im_end|> to the end of the prompt
    ends = [i for i, t in enumerate(tokens) if t["token"] == "<|im_end|>"]
    return tokens[ends[-1]:]


def show():
    counts = {c: collections.Counter() for c in CONDS}
    n = collections.Counter()
    example = {}
    for f in sorted(os.listdir(OUT)):
        d = json.load(open(os.path.join(OUT, f)))
        layers = d["meta"]["layers_by_type"]["JPP_LENS"]
        t = tail(d["tokens"])
        example.setdefault(d["cond"], [x["token"] for x in t])
        n[d["cond"]] += 1
        for x in t:
            tops = x["results"][0]["top_tokens"]
            for L in LAYERS:
                for w in set(tops[layers.index(L)]):
                    counts[d["cond"]][w.strip().lower()] += 1
    print("tail tokens:", example)
    print("files per condition:", dict(n))
    for c in CONDS:
        print(f"\n== {c}: most frequent readouts on the template tokens (layers {LAYERS.start}-{LAYERS.stop - 4})")
        print(", ".join(f"{w}:{k}" for w, k in counts[c].most_common(40)))
    for a, b in (("rl_tutor", "student_tutor"), ("rl_tutor", "homework_tutor"), ("homework_tutor", "student_tutor")):
        diff = {w: counts[a][w] / n[a] - counts[b][w] / n[b] for w in set(counts[a]) | set(counts[b])}
        top = sorted(diff.items(), key=lambda x: -x[1])
        print(f"\n== {a} minus {b} (per prompt)")
        print(" more:", ", ".join(f"{w}:{v:+.1f}" for w, v in top[:30]))
        print(" less:", ", ".join(f"{w}:{v:+.1f}" for w, v in top[-30:][::-1]))


if __name__ == "__main__":
    fetch(int(sys.argv[2])) if sys.argv[1] == "fetch" else show()
