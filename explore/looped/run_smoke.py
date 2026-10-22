"""Bounded BF16 looped-model feasibility run; raw evidence is never tracked in Git.

Run from ~/grader-looped after copying manifest.json and extracted scoring.py.
The native template's unsolicited system message is removed for rl_nosys.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time

os.environ.setdefault("HF_HOME", str(Path.cwd() / "hf-cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / "hf-modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def gpu_gate():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"
    ], text=True).strip()
    util, used = [int(x.strip()) for x in text.split(",")]
    if util > 20 or used > 2048:
        raise RuntimeError(f"GPU busy, refusing to load: {text}")
    return {"utilization_percent": util, "memory_used_mib": used}


def render(tokenizer, messages, thinking_off=False):
    options = {"enable_thinking": False} if thinking_off else {}
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
    if messages[0]["role"] != "system":
        defaults = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                    "<|im_start|>system\n你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。<|im_end|>\n"]
        found = [prefix for prefix in defaults if prompt.startswith(prefix)]
        assert len(found) == 1, "Unrecognized implicit system prompt"
        prompt = prompt[len(found[0]):]
        assert not prompt.startswith("<|im_start|>system")
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--deadline-seconds", type=int, default=6500)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", default="generations.jsonl")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume-from")
    parser.add_argument("--thinking-off", action="store_true", help="Use Nanbeige's native non-thinking chat mode")
    parser.add_argument("--check-singletons", action="store_true", help="Check first four retained rows against singleton decoding")
    args = parser.parse_args()
    assert Path.cwd().resolve() == (Path.home() / "grader-looped").resolve()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    start = time.time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Task deadline")))
    signal.alarm(args.deadline_seconds)
    gate = gpu_gate()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scoring import leak, split_final

    torch.manual_seed(0)
    torch.set_num_threads(6)
    manifest = json.loads(Path(args.manifest).read_text())
    spec = manifest["models"][args.model_index]
    assert not args.thinking_off or spec["id"].startswith("Nanbeige/")
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=True,
                                             use_fast=not spec["id"].startswith("Nanbeige/"))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    ).eval()
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point())
    meta = dict(model=spec, gpu_before=gate, torch=torch.__version__, transformers=transformers.__version__,
                load_seconds=time.time()-start,
                total_ut_steps=getattr(model.model, "total_ut_steps", getattr(model.config, "num_loops", None)),
                early_exit_threshold=getattr(model, "early_exit_threshold", None), generation_config=model.generation_config.to_dict(),
                input_manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                max_new_tokens=1024, do_sample=False, batch_size=args.batch_size,
                resume_from=args.resume_from,
                thinking_mode="off" if args.thinking_off else "native-default",
                parameters=sum(p.numel() for p in model.parameters()),
                loaded_allocated_mib=torch.cuda.memory_allocated()/2**20)
    Path(args.output + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("LOADED", json.dumps(meta), flush=True)
    conditions = [("student_tutor", True, "STUDENT_TEMPLATE"),
                  ("homework_tutor", True, "HOMEWORK_TEMPLATE"),
                  ("rl_tutor", True, "RL_TEMPLATE"), ("rl_nosys", False, "RL_TEMPLATE")]
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    completed = 0
    generation_extra = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    previous = []
    if args.resume_from:
        previous = [json.loads(line) for line in Path(args.resume_from).read_text().splitlines()]
        assert len({(r["cond"], r["idx"]) for r in previous}) == len(previous)
        assert all(r["model"] == spec["id"] and r["revision"] == spec["revision"] for r in previous)
        meta["prefix_sha256"] = hashlib.sha256(Path(args.resume_from).read_bytes()).hexdigest()
    done = {(r["cond"], r["idx"]) for r in previous}
    if args.check_singletons:
        assert len(previous) >= 4
        checks = []
        with torch.inference_mode():
            for prior in previous[:4]:
                check_enc = tokenizer(prior["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda")
                check_gen = model.generate(**check_enc, max_new_tokens=16, do_sample=False, use_cache=True,
                                           pad_token_id=tokenizer.pad_token_id, **generation_extra)
                actual = check_gen[0, check_enc.input_ids.shape[1]:].tolist()
                expected = prior["generated_ids"][:16]
                checks.append(dict(cond=prior["cond"], idx=prior["idx"], match=actual==expected,
                                   expected_ids=expected, singleton_ids=actual,
                                   singleton_text=tokenizer.decode(actual, skip_special_tokens=False)))
        Path(args.output+".singleton-check.json").write_text(json.dumps(checks, indent=2))
        print("SINGLETON_CHECK", json.dumps(checks), flush=True)
        assert all(c["match"] for c in checks), "Real-checkpoint batch/singleton mismatch"
        meta["singleton_check"] = checks
    jobs = []
    for problem in manifest["problems"]:
        for cond, use_system, template in conditions:
            messages = ([{"role": "system", "content": manifest["tutor"]}] if use_system else [])
            messages.append({"role": "user", "content": manifest["templates"][template].format(q=problem["q"])})
            prompt = render(tokenizer, messages, args.thinking_off)
            if (cond, problem["idx"]) in done:
                prior = next(r for r in previous if (r["cond"], r["idx"]) == (cond, problem["idx"]))
                assert prior["prompt"] == prompt and prior["gold"] == problem["gold"]
            else:
                jobs.append((problem, cond, messages, prompt))
    assert len(jobs) + len(previous) == 80
    # Validate left-padding/cache behavior against the retained singleton prefix.
    if args.batch_size > 1 and len(previous) >= 2:
        test = previous[:2]
        test_enc = tokenizer([r["prompt"] for r in test], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            test_gen = model.generate(**test_enc, max_new_tokens=16, do_sample=False, use_cache=True,
                                      pad_token_id=tokenizer.pad_token_id, **generation_extra)
        actual = test_gen[:, test_enc.input_ids.shape[1]:].tolist()
        assert all(ids == r["generated_ids"][:16] for ids, r in zip(actual, test)), "Batched/singleton prefix differs"
        meta["batch_prefix_check"] = "two unequal-length prompts, first 16 tokens identical"
        del test_gen, test_enc
        print("BATCH_PREFIX_CHECK_PASSED", flush=True)
    with open(args.output, "x") as out, torch.inference_mode():
        for prior in previous:
            out.write(json.dumps(prior) + "\n")
        out.flush()
        completed = len(previous)
        for offset in range(0, len(jobs), args.batch_size):
            if completed >= args.limit:
                break
            group = jobs[offset:offset + min(args.batch_size, args.limit-completed)]
            enc = tokenizer([j[3] for j in group], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            t = time.time()
            generated = model.generate(**enc, max_new_tokens=1024, do_sample=False, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, **generation_extra)
            batch_seconds = time.time()-t
            for index, (problem, cond, messages, prompt) in enumerate(group):
                ids = generated[index, enc.input_ids.shape[1]:].tolist()
                stops = [i for i, token in enumerate(ids) if token in eos_ids]
                if stops:
                    ids = ids[:stops[0]+1]
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                final = split_final(raw)
                row = dict(model=spec["id"], revision=spec["revision"], cond=cond, **problem,
                           messages=messages, prompt=prompt,
                           prompt_ids=enc.input_ids[index][enc.attention_mask[index].bool()].tolist(),
                           generated_ids=ids, raw_text=raw, text=text, final=final,
                           leak=leak(final, problem["gold"]),
                           has_close_think="</think>" in raw,
                           has_open_think="<think>" in raw or prompt.endswith("<think>\n"),
                           generated_tokens=len(ids), eos=bool(ids and ids[-1] in eos_ids),
                           truncated=bool(ids and ids[-1] not in eos_ids and len(ids)>=1024),
                           seconds=batch_seconds/len(group), batch_seconds=batch_seconds,
                           batch_size=len(group), batch_index=offset//args.batch_size,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                out.write(json.dumps(row) + "\n"); out.flush(); os.fsync(out.fileno())
                completed += 1
                print(json.dumps({k: row[k] for k in ["cond", "idx", "gold", "leak", "eos", "truncated", "generated_tokens", "seconds"]}), flush=True)
    meta.update(completed=completed, elapsed_seconds=time.time()-start,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    Path(args.output + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
