"""Shared constants and placeholder filling of the stack10 A/B item templates (render_stack10_items.py and gpu/gpu_s10.py fixtures).

Contract: a template is valid only if every wave-1 served flag, env and wave-1 canary pattern of its shape survives unchanged; the
candidate arm differs from the base arm by the frozen build alone (no env switch), and only canary boots carry log envs.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engagement

ROOT = Path(__file__).resolve().parents[2]
W = ROOT.parent.parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
CCX01 = W / 'work/aa-load/ccx01.sh'
COMPONENTS = 'p55 k7 k4 k6a k6b k2 k10 k9 k8 k11a k11b'.split()

# Wave-1 canary patterns of the base arm per shape (items s9p55-*), the part of the candidate's canary that is not a component line.
WAVE1 = {
    'np1': ["MTP chain: k=5\\b", "mtp_fold_decode: engaged n_seq=1\\b",
            "set_logits: using backend candidates: n_cand = [1-9]\\d*, n_ban = [1-9]\\d*, top_k = 20\\b", "(?s)^(?!.*set_sampler: [^\\n]*using CPU)"],
    'np2': ["gdn_cache_fusion: graph device=0 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b",
            "gdn_cache_fusion: graph device=1 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b",
            "mtp_fold_decode: engaged n_seq=2\\b", "MTP chain disabled: needs one sequence \\(n_seq=2\\)", "(?s)^(?!.*MTP chain: k=)"],
    'duo': ["MTP chain disabled: needs one sequence \\(n_seq=2\\)", "(?s)^(?!.*MTP chain: k=)", "n_kv_pad += 4096\\b", "mtp_fold_decode: engaged n_seq=1\\b"],
}
# Component switches and mutants: scrubbed from the inherited environment of every boot (server.knobs).
SWITCH_ENVS = ['GGML_CUDA_ADD_RMS_NORM', 'GGML_CUDA_ADD_RMS_NORM_MUTANT', 'GGML_CUDA_ALPHA_ALIAS_MUTANT', 'GGML_CUDA_ALPHA_GATE', 'GGML_CUDA_ALPHA_GATE_MUTANT',
               'GGML_CUDA_FATTN_KV_MAX_MUTANT', 'GGML_CUDA_FATTN_KV_MAX_SHARE', 'GGML_CUDA_INPUT_STAGE_MUTANT', 'GGML_CUDA_INPUT_STAGE_ORDER',
               'GGML_CUDA_MMQ_PDL', 'GGML_CUDA_MMQ_PDL_LOG', 'GGML_CUDA_MMQ_PDL_MUTANT', 'GGML_CUDA_MMQ_SK_MUTANT', 'GGML_CUDA_MMQ_TILES',
               'GGML_CUDA_MMQ_TILES_LOG', 'GGML_CUDA_PDL', 'GGML_CUDA_Q8_DECODE', 'GGML_CUDA_Q8_REUSE', 'GGML_CUDA_Q8_REUSE_STALE_MUTANT',
               'GGML_CUDA_RMS_NORM_SCALE_PAIR', 'GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT', 'GGML_CUDA_TOPK_BATCHED', 'GGML_CUDA_TOPK_LOG',
               'GGML_CUDA_TOPK_MUTANT', 'GGML_META_INPUT_RANK_ORDER', 'GGML_META_INPUT_STAGE', 'GGML_META_INPUT_STAGE_MUTANT',
               'GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT', 'GGML_SSM_CONV_RS_LOG', 'GGML_SSM_CONV_RS_MUTANT', 'GGML_SSM_CONV_RS_OFF',
               'LLAMA_KV_ROT_ALLOC_FAIL', 'LLAMA_KV_ROT_MUTANT', 'LLAMA_KV_ROT_PERSIST', 'LLAMA_SYNC_ELIDE', 'LLAMA_SYNC_ELIDE_DEBUG',
               'LLAMA_SYNC_ELIDE_MUTANT']
# Recorded results and fixtures the gpu runner's --fixtures replays (same files the wave-1 items pin).
DATA_PINS = [
    '/home/xanewok/repos/qwen38-eval/recipes/_common/kld/tokens64k.bin',
    '/home/xanewok/repos/qwen38-eval/corpora/templates/swift-leading-instructions-v1.jinja',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/nvidia-smi-apps.csv',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/nvidia-smi-query.csv',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/xid-lines.txt',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/summary.md',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-server/serve-bench-tail.log',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1-input.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/cold-tool-1.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/warmup-0.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/server.log',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/fit.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/conc-94000.log',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/conc-94000.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/single-94000.log',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/errors.txt',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/conc-32768.json',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/steps.txt',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/conc-32768.log',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/serve.log',
    '/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/hq-two-stream/boot1/single-94000.json',
    '/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/depth-8192.json',
    '/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/server.log',
    '/home/xanewok/repos/qwen38-eval/results/2026-10-07-gdn-getter-gate-np2-busy-j5q/canary-a1/0-up/depth-8192.json',
    '/home/xanewok/repos/qwen38-eval/results/2026-10-07-gdn-getter-gate-np1-j5q/canary-a1/0-patch/server.log',
    '/home/xanewok/repos/qwen38-eval/results/2026-10-06-draft-chain-m1-depth-j5q/canary-a2/0-chain/server.log',
    '/home/xanewok/repos/qwen38-eval/results/2026-10-07-cand-neginf-depth-j5q/canary-a1/1-cn/server.log',
]
# Counter kinds of the round-trace writer used by the np2 J5q item (all exist in stack9-p55 and stack10 libllama).
TRACE_COUNTERS = ['graph_hit', 'graph_miss', 'llama_build', 'compute_submit', 'set_inputs', 'sync', 'sampling']


def shape_of(text):
    m = re.search(r'^# shape: (np1|np2|duo)$', text, re.M)
    assert m, 'template lacks its "# shape:" line'
    return m.group(1)


def extra_pins(tree, template_name):
    t = Path(tree)
    codex = t / 'codex'
    own = ['gpu_stack9.py', 'gpu_stack10.py', 'mac-evidence/np2-on.log', 'mac-evidence/np2-old-on.log', 'stack9-evidence/recorded-getter.log',
           'stack10/gpu/gpu_ab.sh', 'stack10/gpu/gpu_s10.py', 'stack10/render_stack10_items.py', 'stack10/s10items.py', 'stack10/engagement.py',
           f'stack10/items/{template_name}']
    return [str(codex / n) for n in own] + [str(CCX01), *DATA_PINS]


def fill(text, *, sha, base_sha, components, tree, evidence, template_name):
    shape = shape_of(text)
    env = engagement.canary_env(components, shape)
    patterns = WAVE1[shape] + engagement.canary_patterns(components, shape)
    values = {'@STACK10_SHA@': sha, '@BASE_SHA@': base_sha, '@TREE@': str(tree), '@EVIDENCE@': str(evidence),
              '@EXTRA_PINS@': json.dumps(extra_pins(tree, template_name)), '@CANARY_LOG@': json.dumps(patterns),
              '@CANARY_ENV@': ''.join(f'{k} = {json.dumps(v)}\n' for k, v in sorted(env.items())),
              '@COMPONENTS@': ' '.join(components)}
    for key, value in values.items():
        text = text.replace(key, value)
    assert not re.search(r'@[A-Z_0-9]+@', text), 'unfilled placeholder'
    return text
