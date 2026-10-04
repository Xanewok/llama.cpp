"""CPU-only source mutations; restore sources and rebuild the correct binaries on every exit."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", CCACHE_DIR=str(ROOT / "build-cpu/ccache"), TMPDIR=str(ROOT / "build-cpu/tmp"))
(ROOT / "build-cpu/tmp").mkdir(exist_ok=True)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FILES = ["common/sampling.cpp", "ggml/src/ggml-backend-meta.cpp", "src/llama-context.cpp", "src/llama-graph.cpp", "src/llama-sampler.cpp", "src/llama-sampler.h"]
original = {name: Path(name).read_bytes() for name in FILES}
MODEL = "/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad/models/stories15M-q4_0.gguf"
results = {}


def build(tag):
    with Path("codex/controls-build-" + tag + ".log").open("w") as log:
        subprocess.run(["nice", "-n", "19", "cmake", "--build", "build-cpu", "--target", "test-backend-sampler", "test-llama-archs", "-j8"], stdout=log, stderr=subprocess.STDOUT, check=True)


def run(tag, case):
    cmd = ["nice", "-n", "19", "build-cpu/bin/test-backend-sampler", "--device", "cpu", "--test", case, "--model", MODEL]
    start = time.monotonic()
    with Path("codex/controls-" + tag + "-" + case + ".log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    results[tag + ":" + case] = {"returncode": rc, "M_seconds": time.monotonic() - start}
    print(tag, case, rc, flush=True)
    return rc


def restore():
    for name, content in original.items():
        Path(name).write_bytes(content)


try:
    for name in FILES:
        Path(name).write_bytes(subprocess.check_output(["git", "show", "98c695111:" + name]))
    build("base")
    assert run("base", "logit_bias") != 0
    assert run("base", "penalties") != 0
    assert run("base", "leading_bias") != 0
    restore()
    path = Path("common/sampling.cpp")
    source = path.read_text()
    assert source.count("if (!cur_biased) {") == 1
    path.write_text(source.replace("if (!cur_biased) {", "if (true) {"))
    build("double")
    assert run("double", "leading_bias") != 0
    restore()
    source = path.read_text()
    bias_block = """        if (!result->prefilter_bias.empty()) {
            llama_sampler_chain_add(result->backend_candidates, llama_sampler_init_logit_bias(
                    llama_vocab_n_tokens(vocab), result->prefilter_bias.size(), result->prefilter_bias.data()));
        }
"""
    assert source.count(bias_block) == 1
    source = source.replace(bias_block, "")
    # Rank raw logits on the backend, then bias only the retained candidates on the host.
    source = source.replace("cur_biased = !prefilter_bias.empty();", "cur_biased = false;")
    path.write_text(source)
    build("after-top-k")
    assert run("after-top-k", "leading_bias") != 0
finally:
    restore()
    build("restored")
    Path("codex/controls.json").write_text(json.dumps(results, indent=4) + "\n")

assert run("restored", "leading_bias") == 0
start = time.monotonic()
with Path("codex/bias-tp.log").open("w") as log:
    rc = subprocess.run(["nice", "-n", "19", "build-cpu/bin/test-llama-archs", "--tp-sampling", "-s", "42"], stdout=log, stderr=subprocess.STDOUT).returncode
results["restored:tp"] = {"returncode": rc, "M_seconds": time.monotonic() - start}
Path("codex/controls.json").write_text(json.dumps(results, indent=4) + "\n")
assert rc == 0
