"""Freeze only existing exploratory prompts/scorer and the requested seed-0 sample."""
import ast
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/raw/looped-feasibility"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "explore/explore.py").read_text()
    wanted = {"RL_TEMPLATE", "HOMEWORK_TEMPLATE", "STUDENT_TEMPLATE"}
    templates = {}
    funcs = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    templates[target.id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name in {"leak", "split_final"}:
            funcs.append(ast.get_source_segment(source, node))
    assert set(templates) == wanted and len(funcs) == 2
    (OUT / "scoring.py").write_text("import re\n\n" + "\n\n".join(funcs) + "\n")
    problems = [json.loads(line) for line in (ROOT / "explore/problems.jsonl").read_text().splitlines()]
    random.Random(0).shuffle(problems)
    manifest = {
        "sample_seed": 0,
        "problems": problems[:20],
        "tutor": (ROOT / "explore/system_tutor.txt").read_text(),
        "templates": templates,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["explore/explore.py", "explore/problems.jsonl", "explore/system_tutor.txt"]
        },
        "models": [
            {"id": "ByteDance/Ouro-2.6B-Thinking", "revision": "f1edd81e7ac41355db670500ceaf204e0f73af68"},
            {"id": "ByteDance/Ouro-1.4B-Thinking", "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"},
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    nanbeige = dict(manifest, models=[{"id": "Nanbeige/Nanbeige4.2-3B",
                                     "revision": "b82e54bd609793562a75cbf9337970a93369eab5"}])
    (OUT / "manifest-nanbeige.json").write_text(json.dumps(nanbeige, indent=2) + "\n")
    print(json.dumps({"ids": [p["idx"] for p in problems[:20]], "hashes": manifest["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
