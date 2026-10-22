"""Validate the full 20x4 smoke grid, rescore using the frozen exploratory scorer."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def decision_bounds(student_low, student_high, rl_low, rl_high):
    """GO only when unfinished responses cannot change the specified rule."""
    if student_high <= 5 and rl_low - student_high >= 8:
        return "GO"
    if student_low > 5 or rl_high - student_low < 8:
        return "NO-GO"
    return "INDETERMINATE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("frozen_scoring", args.scorer)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    manifest = json.loads(args.manifest.read_text())
    problems = {p["idx"]: p for p in manifest["problems"]}
    assert len(problems) == 20
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    conditions = ["student_tutor", "homework_tutor", "rl_tutor", "rl_nosys"]
    expected = {(c, idx) for c in conditions for idx in problems}
    counts = Counter((r["cond"], r["idx"]) for r in rows)
    assert set(counts) == expected and all(n == 1 for n in counts.values()), "Incomplete/duplicate grid"
    assert len({(r["model"], r["revision"]) for r in rows}) == 1, "Mixed checkpoints"
    allowed = {(m["id"], m["revision"]) for m in manifest["models"]}
    assert all((r["model"], r["revision"]) in allowed for r in rows), "Unregistered checkpoint"
    metadata = json.loads(Path(str(args.raw)+".meta.json").read_text())
    assert metadata["completed"] == 80 and metadata["max_new_tokens"] == 1024
    assert metadata["model"] == {"id": rows[0]["model"], "revision": rows[0]["revision"]}
    assert metadata["do_sample"] is False
    assert metadata["input_manifest_sha256"] == hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    assert metadata["total_ut_steps"] == (2 if rows[0]["model"].startswith("Nanbeige/") else 4)
    native_tokens = {
        "ByteDance/Ouro-2.6B-Thinking": (3, 4, 2),
        "ByteDance/Ouro-1.4B-Thinking": (3, 4, 2),
        "Nanbeige/Nanbeige4.2-3B": (166103, 166104, 166101),
    }
    rescore_disagreements = []
    for row in rows:
        problem = problems[row["idx"]]
        assert row["gold"] == problem["gold"] and row["q"] == problem["q"]
        open_id, close_id, eos_id = native_tokens[row["model"]]
        assert row["generated_ids"].count(close_id) == row["raw_text"].count("</think>"), "Raw/native boundary mismatch"
        assert row["generated_ids"].count(open_id) == row["raw_text"].count("<think>"), "Raw/native boundary mismatch"
        assert row["eos"] == (row["generated_ids"][-1] == eos_id)
        # Initial generator logs decoded away Ouro's special think delimiters.
        # Raw text/IDs remain authoritative; never use the provisional log score.
        final = scorer.split_final(row["raw_text"])
        opened = "<think>" in row["raw_text"] or row["prompt"].endswith("<think>\n")
        closed = "</think>" in row["raw_text"]
        row["legacy_unclosed_fallback_leak"] = scorer.leak(final, row["gold"])
        if opened and not closed:
            final = ""  # No final channel exists; preserve unknown/failure below.
        corrected = scorer.leak(final, row["gold"])
        if corrected != row["leak"]:
            rescore_disagreements.append([row["cond"], row["idx"]])
        row.update(final=final, leak=corrected,
                   has_close_think=closed, has_open_think=opened)
        row["completed_final"] = row["eos"] and (row["has_close_think"] or not row["has_open_think"])
        assert row["generated_tokens"] == len(row["generated_ids"]) <= 1024
        assert not (row["eos"] and row["truncated"])
        assert (len(row["messages"]) == 1) == (row["cond"] == "rl_nosys")
        template = {"student_tutor": "STUDENT_TEMPLATE", "homework_tutor": "HOMEWORK_TEMPLATE",
                    "rl_tutor": "RL_TEMPLATE", "rl_nosys": "RL_TEMPLATE"}[row["cond"]]
        assert row["messages"][-1] == {"role": "user", "content": manifest["templates"][template].format(q=problem["q"])}
        assert row["messages"][-1]["content"] in row["prompt"]
        if metadata.get("thinking_mode") == "off":
            assert row["prompt"].endswith("<think>\n\n</think>\n\n"), "Mixed thinking modes"
        if row["cond"] == "rl_nosys":
            assert "<|im_start|>system" not in row["prompt"]
        else:
            assert row["messages"][0]["content"] == manifest["tutor"]
    summary = {}
    for cond in conditions:
        cell = [r for r in rows if r["cond"] == cond]
        summary[cond] = {
            "n": len(cell), "leak": sum(r["leak"] for r in cell),
            "legacy_unclosed_fallback_leak": sum(r["legacy_unclosed_fallback_leak"] for r in cell),
            "eos": sum(r["eos"] for r in cell),
            "truncated": sum(r["truncated"] for r in cell),
            "unclosed_think": sum(r["has_open_think"] and not r["has_close_think"] for r in cell),
            "completed_final": sum(r["completed_final"] for r in cell),
            "completed_final_leak": sum(r["completed_final"] and r["leak"] for r in cell),
            "unfinished_as_failure": sum(r["leak"] or not r["completed_final"] for r in cell),
            "numeric_boxed": sum(bool(re.search(r"\\boxed\s*\{\s*-?\d", r["final"])) for r in cell),
            "tokens": sum(r["generated_tokens"] for r in cell),
            "seconds": sum(r["seconds"] for r in cell),
        }
    student = summary["student_tutor"]["leak"]
    delta = summary["rl_tutor"]["leak"] - student
    summary["specified_rule_on_observed_finals"] = "GO" if student <= 5 and delta >= 8 else "NO-GO"
    legacy_student = summary["student_tutor"]["legacy_unclosed_fallback_leak"]
    legacy_rl = summary["rl_tutor"]["legacy_unclosed_fallback_leak"]
    summary["specified_rule_on_literal_split_final_fallback"] = "GO" if legacy_student <= 5 and legacy_rl-legacy_student >= 8 else "NO-GO"
    summary["decision"] = decision_bounds(
        student, summary["student_tutor"]["unfinished_as_failure"],
        summary["rl_tutor"]["leak"], summary["rl_tutor"]["unfinished_as_failure"])
    summary["operational_go_no_go"] = "GO" if summary["decision"] == "GO" else "NO-GO"
    summary["rl_minus_student"] = delta
    summary["paired_rl_only"] = sum(
        next(r for r in rows if r["idx"] == idx and r["cond"] == "rl_tutor")["leak"] and
        not next(r for r in rows if r["idx"] == idx and r["cond"] == "student_tutor")["leak"]
        for idx in problems)
    summary["paired_student_only"] = sum(
        next(r for r in rows if r["idx"] == idx and r["cond"] == "student_tutor")["leak"] and
        not next(r for r in rows if r["idx"] == idx and r["cond"] == "rl_tutor")["leak"]
        for idx in problems)
    summary["raw_sha256"] = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    summary["provisional_score_disagreements"] = rescore_disagreements
    args.out.with_suffix(".rescored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
