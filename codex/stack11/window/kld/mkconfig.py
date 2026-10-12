"""Write GEN/config-MODEL.json for the paired stack10/stack11 KLD item (runner.py). Arms: base = frozen/stack10, cand = frozen/stack11-base.

  mkconfig.py --gen DIR --model j5q|iq4 [--s10 DIR] [--s11 DIR] [--binary-s10 FILE --binary-s11 FILE]

--binary-* replace GEN/bin/kld_dump-{s10,s11} only for a dry render against a stand-in (the dry configs are never runnable: the runner
accepts GPU configs from kld/gen only).
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WT = Path("/home/xanewok/repos/qwen38-eval-worktrees")
DATA = WT / "work/kld-s9-gdn4/data"
MODELS = {
    "j5q": ("/home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf", "f2c850ea3cd818b9d669fbd1e18767ef669ce012e614d819cce5ab6809e3a774"),
    "iq4": ("/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf", "e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df"),
}
TAIL = 384
# D: gdn4 (#30087, in stack11, not in stack10) alone measured mean KL vs stack9 3.5e-4 / 5.6e-4 on the assistant tail and 0.044 /
# 0.058 on all tail rows at 32k / 64k (kld-s9-gdn4 a1, J5q); the master rebase and the rest are expected to add little (G).
# Falsifier: twice that, i.e. the other stack11 changes move the distribution as much again as gdn4 did.
VERDICT = dict(assistant_tail=dict(expected=6e-4, falsifier=1.2e-3, ratio=10.0), all_tail=dict(expected=0.06, falsifier=0.12, ratio=10.0))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(root, binary):
    root = Path(root)
    return dict(root=str(root), bin=str(root / "build/bin"), commit=(root / "SOURCE").read_text().splitlines()[0],
                binary=str(binary), binary_sha256=sha(binary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--s10", type=Path, default=WT / "frozen/stack10")
    ap.add_argument("--s11", type=Path, default=WT / "frozen/stack11-base")
    ap.add_argument("--binary-s10", type=Path)
    ap.add_argument("--binary-s11", type=Path)
    args = ap.parse_args()
    model, model_sha = MODELS[args.model]
    arms = ["base", "cand", "cand_repeat", "repeat"]
    config = dict(
        root=str(ROOT), scratch=f"/mnt/scratch/work/kld-s11/out-{args.model}",
        model=model, model_sha=model_sha,
        tokens=str(DATA / "tokens.bin"), canary=str(DATA / "canary.bin"), canary_source=3,
        builds=dict(base=build(args.s10, args.binary_s10 or args.gen / "bin/kld_dump-s10"),
                    cand=build(args.s11, args.binary_s11 or args.gen / "bin/kld_dump-s11")),
        labels=dict(base="frozen/stack10", cand="frozen/stack11-base"),
        verdict=VERDICT,
        depths=[dict(name="d32k", tokens=str(DATA / "d32k.bin"), selection=[0, 1, 3], prefix=32768 + TAIL, tail=TAIL, arms=arms),
                dict(name="d64k", tokens=str(DATA / "d64k.bin"), selection=[0, 1, 3], prefix=None, tail=TAIL, arms=arms + ["f16"])],
        sha256={str(DATA / f"{n}.bin"): sha(DATA / f"{n}.bin") for n in ("tokens", "d32k", "d64k", "canary")},
    )
    args.gen.mkdir(parents=True, exist_ok=True)
    out = args.gen / f"config-{args.model}.json"
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out}: base {config['builds']['base']['commit'][:9]}, cand {config['builds']['cand']['commit'][:9]}")


if __name__ == "__main__":
    main()
