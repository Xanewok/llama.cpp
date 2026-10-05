#!/usr/bin/env python3
"""Generate fixed identities and manifests without launching or querying GPUs."""
import json
import os
import subprocess
from pathlib import Path

from sweep import ROOT, sha, write

HOME = Path('/home/xanewok')
TEMPLATE = HOME / 'repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja'
VOCAB = HOME / 'models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'
BASE = HOME / 'repos/qwen38-weekend/frozen/stack4'
PATCH = ROOT / 'codex/frozen/draft-vocab'
MODELS = dict(j5q=('Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf',
                  'f2c850ea3cd818b9d669fbd1e18767ef669ce012e614d819cce5ab6809e3a774'),
              iq4=('Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf',
                   'e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df'))
DATA = Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    freezes = {'base': BASE, 'subset': PATCH}
    if (BASE / 'SOURCE').read_text().splitlines()[0] != '590912a1de2097523db6c8efe185387bb5c7d4fc':
        raise RuntimeError('unexpected baseline source')
    files = []
    binaries = {}
    for arm, frozen in freezes.items():
        directory = frozen / 'build/bin'
        subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=directory, check=True)
        files += [p for p in directory.iterdir() if p.is_file()]
        files.append(frozen / 'SOURCE')
        binaries[arm] = str(directory / 'round-trace')
    files += [TEMPLATE, VOCAB, ROOT / 'codex/gpu_ab.sh']
    files += [p for p in DATA.iterdir() if p.suffix in ('.py', '.sh')]
    files += [DATA / 'prefixes.json', DATA / 'prefix_bounds.json']
    hashes_common = {str(p): sha(p) for p in sorted(files)}
    if hashes_common[str(TEMPLATE)] != '4af3dcf0e98bf87adf684a4824683ab15fffd64e93ed1170061ef86a15827ba8':
        raise RuntimeError('served template changed')
    if hashes_common[str(VOCAB)] != 'e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6':
        raise RuntimeError('draft vocabulary changed')
    for name, (filename, model_sha) in MODELS.items():
        model = HOME / 'models' / filename
        if sha(model) != model_sha:
            raise RuntimeError('model identity changed')
        hashes = dict(hashes_common, **{str(model): model_sha})
        args = ['-m', str(model), '-c', '102400', '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0',
                '-ngl', '99', '-fa', 'on', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '5',
                '--spec-draft-n-min', '0', '--spec-draft-p-min', '0', '--spec-draft-type-k', 'f16',
                '--spec-draft-type-v', 'f16', '-sm', 'tensor', '--tensor-split', '50,50', '-b', '2048',
                '-ub', '512', '--jinja', '--chat-template-file', str(TEMPLATE), '--temp', '1', '--top-k', '20',
                '--top-p', '0.95', '--min-p', '0.05', '--host', '127.0.0.1', '--port', '8080', '--metrics',
                '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling', '--backend-sampling',
                '-t', '8', '-tb', '8']
        config = dict(binaries=binaries, vocab=str(VOCAB), argv=args, hashes=hashes,
                      source={arm: (frozen / 'SOURCE').read_text() for arm, frozen in freezes.items()},
                      profile_source='/home/xanewok/.claude/skills/local-qwen-server/bin/serve.sh',
                      profile_note='Stack4 already contains compact-head support. Base omits its subset flag; subset adds it. Only binary, vocabulary and k differ.')
        write(DATA / f'{name}.json', config)
        ident = f'draft-vocab-{name}'
        entry = str(ROOT / 'codex/gpu_ab.sh')
        common = ['bash', entry, '--model', name]
        run = common + ['--execute', '--budget-s', '3000']
        preflight = common + ['--preflight']
        canary = common + ['--execute', '--canary', '--budget-s', '475']
        pins = sorted(set(hashes) | {str(DATA / 'j5q.json'), str(DATA / 'iq4.json')})
        native = '/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack4-depth-j5q/full-a1/2-patch/depth-8192.json'
        pins += [native]
        recorded = '/home/xanewok/repos/k-sweep/runs/k-sweep-j5q-4/full-a1/'
        pins += [recorded + filename for filename in ('p02.tokens.json', 'b1-p02.response.json')]
        fixtures = HOME / 'repos/qwen38-eval/orchestration/gpuq/fixtures'
        pins += [str(p) for p in (fixtures / 'llama-sse').glob('*.json')]
        pins += [str(fixtures / p) for p in ('xid-lines.txt', 'nvidia-smi-query.csv')]
        out = str(ROOT / 'runs' / ident)
        toml = f'''id = "{ident}"
title = "Full vs reduced draft vocabulary: {name}, MTP k=5/6/7, ABBA replay"
priority = 42
kind = "measure"
budget_s = 3600
est_s = 2160
ab = true
retry = "fresh"
startup_grace_s = 240
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = []

[run]
cmd = {json.dumps(run)}
cwd = "{ROOT}"
env = {{ PORT = "{{port}}", OUT = "{out}/full-a{{attempt}}" }}
ports = [8092]

[preflight]
cmd = {json.dumps(preflight)}
timeout_s = 100
env = {{ CUDA_VISIBLE_DEVICES = "", PYTHONDONTWRITEBYTECODE = "1" }}

[canary]
cmd = {json.dumps(canary)}
cwd = "{ROOT}"
timeout_s = 480
env = {{ PORT = "{{port}}", OUT = "{out}/canary-a{{attempt}}" }}

[outputs]
paths = ["{out}"]

[pins]
files = {json.dumps(sorted(set(pins)), indent=2)}

[fix]
editable = []
protected = ["REFUSE", "frozen", "budget"]
'''
        (ROOT / f'codex/{ident}.gpuq.toml').write_text(toml)
        print(ident, 'pinned_files', len(pins), 'model_expected_sha', model_sha)


if __name__ == '__main__':
    main()
