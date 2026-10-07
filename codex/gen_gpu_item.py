#!/usr/bin/env python3
"""Write manifests locally; never submit or launch a payload."""
import hashlib
import json
from pathlib import Path

from gpu_payload import ROOT, FROZEN, BASE, MODELS, DATA, VOCAB, COV99


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    recorded = {}
    for run in ('2026-10-06-nh6-nextn-lowbit-j5q/a2', '2026-10-06-nh6-nextn-q8-j5q/a1'):
        recorded.update(json.loads((Path('/home/xanewok/repos/qwen38-eval/results') / run / 'pins.json').read_text())['pins'])
    paths = [BASE, VOCAB, COV99]
    for model in MODELS[1:]:
        paths += sorted(model.parent.glob('*.gguf'))
    paths += [DATA / f'depth-{depth}.json' for depth in range(1500, 19001, 500)]
    inputs = {}
    for path in paths:
        # Large GGUF hashes already belong to successful, pinned real-model runs.
        inputs[str(path)] = recorded.get(str(path)) or hash_file(path)
    (ROOT / 'codex/gpu_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    pins = list(inputs) + [str(p.resolve()) for p in (ROOT / 'codex').glob('*.py')]
    pins += [str(p.resolve()) for p in (ROOT / 'codex').glob('*.sh')]
    pins += [str(ROOT / 'codex/gpu_inputs.json'), str(FROZEN / 'SOURCE')]
    if (FROZEN / 'build/bin').exists():
        pins += [str(p) for p in (FROZEN / 'build/bin').iterdir() if p.is_file()]
    else:
        pins += [str(FROZEN / 'build/bin' / name) for name in ('SHA256SUMS', 'round-trace', 'libggml-cuda.so')]
    def quoted(value):
        return json.dumps(str(value))
    prefix = f'''id = "spec-tf-j5q"
title = "Teacher-forced J5q MTP k5 acceptance: base, Q3_K, IQ3_S, Q8_0 and offline vocab variants"
kind = "measure"
ab = true
retry = "fresh"
budget_s = 22200
est_s = 15000
startup_grace_s = 300
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [{{ class = "integrity", regex = "frozen bytes changed|inputs changed|stamp mismatch" }}]

[run]
cmd = ["bash", {quoted(ROOT / 'codex/gpu_ab.sh')}, "--mode", "run", "--out", "/mnt/scratch/work/spec-tf/full-a{{attempt}}", "--budget", "21600"]
cwd = {quoted(ROOT)}
env = {{ CUDA_HOME = "/usr/local/cuda-13.2", CUDA_VISIBLE_DEVICES = "0,1", OMP_NUM_THREADS = "8", OPENBLAS_NUM_THREADS = "1" }}
ports = []

[preflight]
cmd = ["bash", {quoted(ROOT / 'codex/gpu_ab.sh')}, "--mode", "preflight"]
timeout_s = 600

[canary]
cmd = ["bash", {quoted(ROOT / 'codex/gpu_ab.sh')}, "--mode", "canary", "--out", "/mnt/scratch/work/spec-tf/canary-a{{attempt}}", "--budget", "540"]
timeout_s = 600

[outputs]
paths = ["/mnt/scratch/work/spec-tf/full-a{{attempt}}/complete.json"]

[fix]
editable = []
protected = ["frozen", "budget", "control", "identity"]

[pins]
files = [
'''
    manifest = prefix + ''.join('    ' + quoted(p) + ',\n' for p in sorted(set(pins))) + ']\n'
    (ROOT / 'codex/spec-tf-j5q.gpuq.toml').write_text(manifest)
    print(f'inputs_M={len(inputs)} pins_M={len(set(pins))}')


if __name__ == '__main__':
    main()
