import json
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent.parent
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", TMPDIR=str(root / "build-cpu/tmp"))
(root / "build-cpu/tmp").mkdir(exist_ok=True)
model = "/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad/models/stories15M-q4_0.gguf"
source = (root / "tests/test-backend-sampler.cpp").read_text().split("BACKEND_TESTS[] = {")[1].split("};")[0]
results = {}
for case in re.findall(r'{ "([a-z_]+)"', source):
    with (root / "codex" / ("case-" + case + ".log")).open("w") as log:
        result = subprocess.run(["nice", "-n", "19", str(root / "build-cpu/bin/test-backend-sampler"), "--device", "cpu", "--test", case, "--model", model], stdout=log, stderr=subprocess.STDOUT)
    results[case] = result.returncode
    print(case, result.returncode, flush=True)
(root / "codex/sampler-cases.json").write_text(json.dumps(results, indent=4) + "\n")
