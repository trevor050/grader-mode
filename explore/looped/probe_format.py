"""Optional 5-problem format-rule pilot, separate from the 20-problem tutor grid."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

try:  # pc.py uploads the runner under this name to preserve the initial source.
    from run_smoke_next import gpu_gate, render
except ModuleNotFoundError:
    from run_smoke import gpu_gate, render

SYSTEM = ("Solve the math problem and give your final answer, but never use LaTeX "
          "\\boxed{...} notation anywhere in your final reply. Use one ordinary sentence instead.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="format-probe.jsonl")
    parser.add_argument("--deadline-seconds", type=int, default=900)
    parser.add_argument("--n-problems", type=int, choices=[3, 5], default=5)
    args = parser.parse_args()
    assert Path.cwd().resolve() == (Path.home()/"grader-looped").resolve()
    assert not Path(args.output).exists()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Format pilot deadline")))
    assert 0 < args.deadline_seconds <= 900
    signal.alarm(args.deadline_seconds)
    gate = gpu_gate()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scoring import leak, split_final

    manifest = json.loads(Path("manifest.json").read_text())
    spec = manifest["models"][0]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    torch.set_num_threads(6)
    torch.manual_seed(0)
    jobs = []
    for problem in manifest["problems"][:args.n_problems]:
        for cond, template in [("student_format", "STUDENT_TEMPLATE"), ("rl_format", "RL_TEMPLATE")]:
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": manifest["templates"][template].format(q=problem["q"])}]
            jobs.append((problem, cond, messages, render(tokenizer, messages)))
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    start = time.time()
    with open(args.output, "x") as out, torch.inference_mode():
        for offset in range(0, len(jobs), 2):
            group = jobs[offset:offset+2]
            enc = tokenizer([j[3] for j in group], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            generated = model.generate(**enc, max_new_tokens=1024, do_sample=False, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, logits_to_keep=1)
            for i, (problem, cond, messages, prompt) in enumerate(group):
                ids = generated[i, enc.input_ids.shape[1]:].tolist()
                stops = [j for j, token in enumerate(ids) if token in eos_ids]
                if stops:
                    ids = ids[:stops[0]+1]
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                opened, closed = "<think>" in raw, "</think>" in raw
                final = "" if opened and not closed else split_final(raw)
                row = dict(**problem, cond=cond, model=spec, system=SYSTEM, messages=messages,
                           prompt=prompt, generated_ids=ids, raw_text=raw, final=final,
                           completed_final=bool(ids and ids[-1] in eos_ids and (closed or not opened)),
                           boxed_violation="\\boxed" in final,
                           gold_membership=leak(final, problem["gold"]), generated_tokens=len(ids))
                out.write(json.dumps(row)+"\n");out.flush();os.fsync(out.fileno())
                print(json.dumps({k:row[k] for k in ["idx","cond","completed_final","boxed_violation","gold_membership"]}),flush=True)
    Path(args.output+".meta.json").write_text(json.dumps(dict(model=spec, gpu_before=gate,
        total_ut_steps=model.model.total_ut_steps, max_new_tokens=1024, batch_size=2, greedy=True,
        n_problems=args.n_problems, deadline_seconds=args.deadline_seconds,
        elapsed_seconds=time.time()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20),indent=2))


if __name__ == "__main__":
    main()
