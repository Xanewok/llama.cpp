#!/usr/bin/env python3
"""Check the q8_0 instance preprocessor boundary without HIP/MUSA toolchains."""
import os
from pathlib import Path
import subprocess

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
env = dict(os.environ, GIT_DIR=str(root / "codex/git"), GIT_WORK_TREE=str(root), GIT_NO_LAZY_FETCH="1")
old = subprocess.check_output(["git", "-c", "protocol.allow=never", "show", "7a8525093:ggml/src/ggml-cuda/fattn-mma-f16.cuh"], env=env, text=True)
new = (root / "ggml/src/ggml-cuda/fattn-mma-f16.cuh").read_text()
out = root / "build-arch-check"
out.mkdir(exist_ok=True)


def snippets(text, guarded):
    start = text.index("#define DECL_FATTN_MMA_Q8_0_CASE")
    if guarded:
        start = text.rfind("#if", 0, start)
    end = text.index("#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2", start)
    macro = text[start:end]
    start = text.index("extern DECL_FATTN_MMA_Q8_0_CASE(256, 256, 8, 1)")
    if guarded:
        start = text.rfind("#if", 0, start)
    end = text.index("// The number of viable configurations", start)
    return macro + "\n" + text[start:end]


before, after = out / "guard-before.cpp", out / "guard-after.cpp"
before.write_text(snippets(old, False))
after.write_text(snippets(new, True))
run = lambda path, flags=(): subprocess.check_output(["cpp", "-P", *flags, str(path)], text=True)
assert run(before).split() == run(after).split()
for backend in ("GGML_USE_HIP", "GGML_USE_MUSA"):
    assert not run(after, ("-D" + backend,)).strip()
print("M: NVIDIA instance tokens match; HIP/MUSA remove the instance macro, declarations and predicate.")
print("M: hipcc/mcc availability must be checked separately; this is not a backend syntax compile.")
