"""Render np2 depth experiments with the existing read-only harness."""
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
UP = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack6')
PATCH = HERE / 'frozen/mmvq-np2'


def render(quant):
    j5 = quant == 'j5q'
    model = '/home/xanewok/models/Swift-1.5-Qwen3.8-27B-' + ('J5q-bf16' if j5 else 'IQ4_XS') + '.gguf'
    size = 19996051744 if j5 else 15475951296
    sha = 'f2c850ea3cd818b9d669fbd1e18767ef669ce012e614d819cce5ab6809e3a774' if j5 else 'e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df'
    proof = 'q5_K' if j5 else 'iq4_xs'
    stamp = (PATCH / 'SOURCE').read_text().splitlines()[0]
    text = f'''id = "mmvq-np2-depth-{quant}"
kind = "depth"
title = "Opt-in wide MMVQ vs stack6: {quant} TP2 np2 k5; submit only after memcheck, perf and numerical review"
plan = ["up", "patch", "patch", "up"]
out = "{HERE}/gpu-results/depth-{quant}"
devices = ["GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", "GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f"]
protected = ["identity", "steady_minimum", "frozen", "min_overlap", "wide_route"]
extra_pins = ["{HERE}/REPORT-mmvq-np2.md"]

[model]
id = "swift15-{quant}"
path = "{model}"
bytes = {size}
sha = "{sha}"

[draft_vocab]
path = "/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
sha = "e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6"

[arms.up]
label = "stack6 default routing"
commit = "ac7465885dc9f13de396ccd91b4be5eaffde3730"
root = "{UP}"
bin = "build/bin"
exe = "build/bin/round-trace"
env = {{ GGML_SCHED_INPUT_STAGING = "1", GGML_CUDA_MMVQ_WIDE = "off", GGML_CUDA_MMVQ_WIDE_SPLIT = "0", LD_LIBRARY_PATH = "{{root}}/build/bin" }}

[arms.patch]
label = "wide w2r2 MMVQ, five quant types, 9-16 columns"
commit = "{stamp}"
root = "{PATCH}"
bin = "build/bin"
exe = "build/bin/round-trace"
env = {{ GGML_SCHED_INPUT_STAGING = "1", GGML_CUDA_MMVQ_WIDE = "all", GGML_CUDA_MMVQ_WIDE_SPLIT = "0", GGML_CUDA_DEBUG_MMVQ_WIDE = "1", LD_LIBRARY_PATH = "{{root}}/build/bin" }}
canary_log = ["mmvq_wide: type={proof} columns=12 mode=wide"]

[server]
args = ["-m", "{{model}}", "-c", "{{ctx}}", "-np", "{{np}}", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
        "-sm", "tensor", "--tensor-split", "50,50", "-ngl", "99", "-b", "2048", "-ub", "512", "--kv-pad", "4096",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
        "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "--spec-draft-sampling", "probabilistic",
        "--spec-draft-backend-sampling", "--backend-sampling", "--spec-draft-vocab", "{{draft_vocab}}", "--cache-ram", "0", "--metrics",
        "-t", "8", "-tb", "8", "--host", "127.0.0.1", "--port", "{{port}}"]
knobs = ["GGML_SCHED_INPUT_STAGING", "GGML_CUDA_MMVQ_WIDE", "GGML_CUDA_MMVQ_WIDE_SPLIT", "GGML_CUDA_DEBUG_MMVQ_WIDE", "LLAMA_ROUND_TRACE", "GGML_SCHED_DEBUG"]
boot_cap_s = 300
request_cap_s = 900

[depth]
depths = [8192, 32768, 90112]
k = 5
streams = 2
identity = "off"
identity_off_reason = "np2 batch composition is timing-dependent; changed GEMM rounding requires separate paired KLD evidence before adoption"
distinct_builds = true
min_steady_intervals = 40
min_predicted = 256
min_overlap = 0.8
slot_ctx = 102400
fixtures = "/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures"
request = {{ seed = 1729, n_predict = 512, temperature = 1, top_k = 20, top_p = 0.95, min_p = 0.05 }}
wall_s = 1100

[canary]
plan = ["patch"]
depths = [8192]
proofs = []

[budget]
canary_s = 600
margin = 1.5
cleanup_s = 150
est_s = 5000
startup_grace_s = 300
preflight_s = 300

[outputs]
run = "full-a{{attempt}}"
canary = "canary-a{{attempt}}"
preflight = "preflight-plan"
'''
    experiment = HERE / 'items' / f'depth-{quant}.toml'
    experiment.write_text(text)
    subprocess.run([sys.executable, '-m', 'harness.mkitem', str(experiment), '--out', str(HERE / 'items' / f'depth-{quant}.gpuq.toml')],
                   cwd=EVAL, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))

if __name__ == '__main__':
    render('iq4')
    render('j5q')
