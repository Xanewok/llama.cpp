#!/usr/bin/env python3
"""Read local source/logs; write a bounded provenance inventory in this worktree."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SP = Path('/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad')
V = Path('/home/xanewok/repos/HyperQwen/venv/lib/python3.12/site-packages/vllm')
S = Path('/home/xanewok/repos/qwen38-weekend/llama.cpp-stack2')
C = Path('/home/xanewok/repos/llama.cpp-draft-chain')
FILES = [Path('/home/xanewok/repos/draft-anatomy/REPORT.md'),
         SP / 'gpu-sampling/REPORT.md', SP / 'gpu-sampling/items/96b-attrib.md',
         SP / 'patchA/REPORT.md', SP / 'codex-reviews/patchA.txt',
         SP / 'codex/chain-ab.log', SP / 'queue/logs/99-patchA-graph-cache-ab.log', C / 'codex/REPORT.md']
FILES += [V / p for p in ['v1/worker/gpu/spec_decode/mtp/speculator.py', 'v1/worker/gpu/spec_decode/autoregressive/speculator.py',
    'v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py', 'v1/worker/gpu/spec_decode/speculator.py',
    'v1/worker/gpu/spec_decode/rejection_sampler.py', 'v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
    'v1/worker/gpu/sample/gumbel.py', 'v1/worker/gpu/async_utils.py', 'v1/worker/gpu/model_runner.py',
    'v1/worker/gpu/block_table.py', 'v1/engine/core.py', 'v1/attention/backends/flash_attn.py',
    'model_executor/models/qwen3_5_mtp.py']]
FILES += [S / p for p in ['common/speculative.cpp', 'common/sampling.cpp', 'common/log.cpp', 'common/log.h',
    'src/llama-context.cpp', 'src/llama-graph.cpp', 'src/models/qwen35.cpp', 'ggml/src/ggml-backend-meta.cpp',
    'ggml/src/ggml-cuda/allreduce.cu', 'ggml/src/ggml-cuda/ggml-cuda.cu', 'tools/server/server-context.cpp']]
FILES += [C / p for p in ['common/speculative.cpp', 'src/llama-context.cpp', 'src/llama-graph.cpp', 'src/models/qwen35.cpp', 'src/llama-sampler.cpp']]


def main():
    inventory = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'lines_M': len(p.read_text().splitlines())} for p in FILES]
    chain = (SP / 'codex/chain-ab.log').read_text()
    deltas = [{'k': int(k), 'request': kind, 'delta_ms_round_D': float(delta)}
              for k, kind, delta in re.findall(r'D: k=(\d+) (\w+) chain - plain: ([\d.]+) ms/round', chain)]
    assert len(deltas) == 6
    assert 'long k=3: completion identity failed' in chain
    cache = (SP / 'queue/logs/99-patchA-graph-cache-ab.log').read_text()
    means = {(arm, kind): float(value) for arm, kind, value in re.findall(
             r'^(base|patch|up)\s+(think|tool)\s+ms/round mean ([\d.]+)', cache, re.M)}
    assert len(means) == 6
    cache_deltas = {kind: round(means['patch', kind] - means['base', kind], 2) for kind in ('think', 'tool')}
    assert cache_deltas == {'think': -0.45, 'tool': -0.18}
    result = {'inventory': inventory, 'chain_deltas': deltas, 'chain_long_identity_M': 'failed',
              'patchA_vs_disabled_delta_ms_round_D': cache_deltas,
              'limits': 'Printed means are rounded; no post-subset nsys attribution or new GPU execution.'}
    (ROOT / 'codex/evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}, indent=2))
    print(f'M: {len(inventory)} source/report/log hashes recorded')


if __name__ == '__main__':
    main()
