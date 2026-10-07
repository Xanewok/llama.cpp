"""Render the gpuq item of the arena-fix depth A/B (build window). Usage: bw-make-item.py [OUT_DIR]

The experiment is the spike's depth experiment (codex/depth-experiment.toml at the spike commit) with the patch arm moved to
the frozen graph-capture-fix build. harness.mkitem owns the queue manifest; this script never submits.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
FROZEN = Path.home() / "repos/qwen38-eval-worktrees/frozen/graph-capture-fix"
EVAL = Path.home() / "repos/qwen38-eval"
NAME = "graph-capture-fix-depth-j5q"


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "codex/bw-results").resolve()
    commit = (FROZEN / "SOURCE").read_text().splitlines()[0]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert commit == head, f"frozen SOURCE {commit} != HEAD {head}"
    evidence = out_dir / "cpu-evidence.json"
    assert evidence.is_file(), f"missing {evidence}"

    sub = {
        "": {
            "id": f'"{NAME}"',
            "title": '"stack6 vs in-graph AllReduce node with the arena fix (GGML_META_AR_NODE=1), both kv-pad 4096, '
                     'Swift 1.5 J5q TP2 k=5, ABBA, cross-arm identity; the 32k depth segfaulted in the unfixed spike"',
            "out": f'"{ROOT}/codex/gpu-results/depth"',
            "cpu_evidence": '{ path = "%s", command = "bw-graph-capture-fix.sh stage 1: test-meta-plan-arena (Meta(CPU,CPU), '
                            'GGML_META_AR_NODE 0 and 1) passes; the same test on a tree with 346505678 reverted dies on a signal" }' % evidence,
        },
        "arms.up": {"repo": f'"{ROOT}"'},
        "arms.patch": {"label": '"in-graph AllReduce node + arena fix"', "commit": f'"{commit}"', "root": f'"{FROZEN}"',
                       "repo": f'"{ROOT}"'},
    }
    extra_pins = [ROOT / "codex/bw-graph-capture-fix.sh", ROOT / "tests/test-meta-plan-arena.cpp"]
    section, seen, lines = "", set(), []
    for line in (ROOT / "codex/depth-experiment.toml").read_text().splitlines():
        m = re.match(r"\[([\w.]+)\]$", line)
        if m:
            section = m.group(1)
        key = re.match(r"(\w+) = ", line)
        if key and key.group(1) in sub.get(section, {}):
            seen.add((section, key.group(1)))
            line = f"{key.group(1)} = {sub[section][key.group(1)]}"
            if (section, key.group(1)) == ("", "cpu_evidence"):
                line += "\nextra_pins = [" + ", ".join(f'"{p}"' for p in extra_pins) + "]"
        lines.append(line)
    want = {(s, k) for s, d in sub.items() for k in d}
    assert seen == want, f"template lines not found: {sorted(want - seen)}"

    items = ROOT / "codex/items"
    items.mkdir(exist_ok=True)
    experiment = items / f"{NAME}.toml"
    experiment.write_text("\n".join(lines) + "\n")
    x = tomllib.loads(experiment.read_text())
    assert x["id"] == NAME and x["arms"]["patch"]["commit"] == commit and x["arms"]["up"]["root"].endswith("frozen/stack6")
    item = experiment.with_suffix(".gpuq.toml")
    subprocess.run([sys.executable, "-m", "harness.mkitem", str(experiment), "--out", str(item)], cwd=EVAL,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)
    print(f"item {item}")


if __name__ == "__main__":
    main()
