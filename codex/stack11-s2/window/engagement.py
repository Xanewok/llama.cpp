"""Per-component engagement proof of the stack11 A/B canaries and health boots: single source for render_items.py, health.sh and
the preflight (stack11-base's engagement plus p90, anchors checked in this tree).

Contract: each entry names the log line a component prints when its kernel/path is taken, the source file and literal that print
it (preflight fails when the literal is gone), the canary-only env that makes the line appear, the regexes the canary's server.log
must match, and a sample line built from the format string that the regexes must match (so a format/regex drift fails on the CPU).
A component without an entry has no observable engagement line.
  shapes: where the canary can show it. np1 = depth item, one stream; np2 = depth item, two streams joint start (12-column verify);
          duo = campaign canary (np2 server, sequential requests, 6-column verify).

usage: engagement.py selftest     every anchor exists in the tree, every regex compiles and matches its sample
       engagement.py list         component, shapes, env, regexes
       engagement.py health PROFILE   JSON {env, patterns, recorded} of a served-flag health boot (health.sh)
"""
import json
import os
import re
import sys
from pathlib import Path

NOK = Path(__file__).resolve().parents[3]  # topic/stack11-s2: K7 and K10 removed
ROOT = NOK.parent / 'stack11-s2-k'  # the candidate with K7 and K10 (frozen/stack11-s2): anchors are checked here
ALL = ('np1', 'np2', 'duo')
CU = 'ggml/src/ggml-cuda/ggml-cuda.cu'

ENGAGEMENT = {
    'base': dict(shapes=ALL, env={}, where='server.log, -lv 5 (debug) / boot', patterns=[
        r'persistent rotations backend=',
        r'alpha_gate_fusion: device=0 width=[1-9]\d* rows=[1-9]\d*',
        r'alpha_gate_fusion: device=1 width=[1-9]\d* rows=[1-9]\d*'],
        anchors=[('src/llama-kv-cache.cpp', 'persistent rotations backend=%s'),
                 (CU, 'alpha_gate_fusion: device=%d width=%lld rows=%lld')],
        samples=['llama_kv_cache: persistent rotations backend=CUDA0 matrices=16',
                 'alpha_gate_fusion: device=0 width=48 rows=6', 'alpha_gate_fusion: device=1 width=48 rows=6']),
    'k7': dict(shapes=ALL, env={}, where='server.log, -lv 5 (debug)', patterns=[
        r'rms_norm_scale_pair_fusion: device=[01] ncols=[1-9]\d* rows=[1-9]\d*'],
        anchors=[(CU, 'rms_norm_scale_pair_fusion: device=%d ncols=%lld rows=%lld')],
        samples=['rms_norm_scale_pair_fusion: device=1 ncols=128 rows=96']),
    'k4': dict(shapes=ALL, env={'GGML_SSM_CONV_RS_LOG': '1'}, where='server.log, INFO, only with GGML_SSM_CONV_RS_LOG', patterns=[
        r'ssm_conv_rs engaged: d_conv=\d+ n_t=[1-9]\d* n_s=[1-9]\d* slots=[1-9]\d* silu=[01] mutant=0\b'],
        anchors=[('ggml/src/ggml-cuda/ssm-conv.cu', 'ssm_conv_rs engaged: d_conv=%d n_t=%d n_s=%d slots=%d silu=%d mutant=%d')],
        samples=['ggml_cuda_op_ssm_conv_rs: ssm_conv_rs engaged: d_conv=4 n_t=6 n_s=1 slots=7 silu=1 mutant=0']),
    'k6a': dict(shapes=ALL, env={}, where='server.log, -lv 5 (debug)', patterns=[
        r'q8_reuse: device=[01] graph=\d+ rows=[1-9]\d* skipped=[1-9]\d* quantized=\d+'],
        anchors=[(CU, 'q8_reuse: device=%d graph=%" PRIu64 " rows=%d skipped=%d quantized=%d bytes=%zu')],
        samples=['q8_reuse: device=0 graph=12 rows=6 skipped=3 quantized=2 bytes=4096']),
    # 10-09 canaries (M): engages in the duo canary only (112 lines); 0 in the np1 depth and np2 joint-start canaries
    'k6b': dict(shapes=('duo',), env={}, where='server.log, -lv 5 (debug)', patterns=[
        r'add_rms_norm: device=[01] rows=[1-9]\d* columns=[1-9]\d*'],
        anchors=[(CU, 'add_rms_norm: device=%d rows=%" PRId64 " columns=%" PRId64 "')],
        samples=['add_rms_norm: device=0 rows=6 columns=5120']),
    'k10': dict(shapes=ALL, env={'GGML_CUDA_TOPK_LOG': '1'}, where='server.log (stderr), only with GGML_CUDA_TOPK_LOG', patterns=[
        r'top_k_batched: engaged ncols=\d+ nrows=[1-9]\d* k=\d+'],
        anchors=[('ggml/src/ggml-cuda/top-k.cu', 'top_k_batched: engaged ncols=%d nrows=%d k=%d')],
        samples=['top_k_batched: engaged ncols=248320 nrows=6 k=20']),
    # pkg 90: the token table is a pinned host buffer read in place; the load log names the buffer type
    'p90': dict(shapes=ALL, env={}, where='server.log, boot INFO', patterns=[
        r'load_tensors: +CUDA_Host_Mapped model buffer size = +[1-9][0-9.]* MiB'],
        anchors=[('src/llama-model.cpp', '%12s model buffer size = %8.2f MiB')],
        samples=['load_tensors:   CUDA_Host_Mapped model buffer size =   644.00 MiB']),
    # small-I tiles are taken for 16 and 24 columns only: the 12-column np2 verify, never the 6-column np1 or sequential-duo one.
    'k8': dict(shapes=('np2',), env={'GGML_CUDA_MMQ_TILES_LOG': '1'}, where='server.log (stderr), only with GGML_CUDA_MMQ_TILES_LOG', patterns=[
        r'mmq_tiles: type=\w+ m=\d+ cols=(?:9|1\d|2[0-4]) z=\d+ J=(?:16|24) ntiles=\d+ nsm=\d+ small_i=1 '],
        anchors=[('ggml/src/ggml-cuda/mmq.cuh', 'mmq_tiles: type=%s m=%lld cols=%lld z=%lld J=%d ntiles=%d nsm=%d small_i=%d units=%d parts=%d blocks=%d')],
        samples=['mmq_tiles: type=q6_K m=8704 cols=12 z=1 J=16 ntiles=136 nsm=36 small_i=1 units=68 parts=68 blocks=136']),
    # no line exists without the debug check, which queries the backends at every skipped sync (busy = missed enqueue) and then
    # syncs for real: the canary proves the path is taken, query-based, and that no skipped sync had queued work; the measured
    # arms run with the default (elision, no check).
    'k11a': dict(shapes=ALL, env={'LLAMA_SYNC_ELIDE_DEBUG': '1'}, where='server.log, WARN/ERROR, only with LLAMA_SYNC_ELIDE_DEBUG', patterns=[
        r'sync_elide: debug check engaged \(backend query exact\)',
        r'(?s)^(?!.*sync_elide: missed:)'],
        anchors=[('src/llama-context.cpp', 'sync_elide: debug check engaged (backend query %s)'), ('src/llama-context.cpp', 'sync_elide: missed: backend busy at a skipped sync')],
        samples=['synchronize: sync_elide: debug check engaged (backend query exact)\nother lines']),
    'k11b': dict(shapes=ALL, env={}, where='server.log, boot INFO (needs GGML_SCHED_INPUT_STAGING=1, set in every arm)', patterns=[
        r'one pinned copy per input, uploads slowest host link first: CUDA[01] \('],
        anchors=[(CU, 'one pinned copy per input, uploads slowest host link first: %s')],
        samples=['ggml_backend_cuda_input_stage_init: one pinned copy per input, uploads slowest host link first: CUDA1 (812 us per 4 MiB), CUDA0 (790 us per 4 MiB)']),
}
# stack11-base master has token_embd in CPU_Mapped; the negative control of p90 (the stack11-base arm must not show the mapped buffer)
BASE_P90 = [r'load_tensors: +CPU_Mapped model buffer size = +[1-9][0-9.]* MiB', r'(?s)^(?!.*CUDA_Host_Mapped)']
NO_LINE = ('p55', 'p87', 'p88', 'p91')  # 87/88/91 print nothing: the Mac gates and the A/B counters are their evidence  # per-type MMVQ launch parameters: no log line, the engagement is the full-suite MUL_MAT gate


def selected(components, shape):
    return [c for c in components if c in ENGAGEMENT and shape in ENGAGEMENT[c]['shapes']]


def canary_env(components, shape):
    env = {}
    for c in selected(components, shape):
        env |= ENGAGEMENT[c]['env']
    return env


def canary_patterns(components, shape):
    return [p for c in selected(components, shape) for p in ENGAGEMENT[c]['patterns']]


# Served-flag health boot (health.sh): one short chat completion per slot. Log-only envs are added to the served env; LLAMA_SYNC_ELIDE_DEBUG
# also syncs for real after each skipped sync (a check, not the served path). Single-slot profiles take the np1 lines; on -duo the two
# requests may or may not share a verify batch, so only lines of components that engage in both np2 shapes (joint and sequential) gate.
HEALTH_ENV = {'LLAMA_ARG_LOG_VERBOSITY': '5', 'GGML_SSM_CONV_RS_LOG': '1', 'GGML_CUDA_TOPK_LOG': '1', 'LLAMA_SYNC_ELIDE_DEBUG': '1',
              'GGML_CUDA_MMQ_TILES_LOG': '1'}
HEALTH_EXTRA = {
    'np1': [r'MTP chain: k=5\b', r'mtp_fold_decode: engaged n_seq=1\b', r'n_kv_pad += 4096\b'],
    'duo': [r'MTP chain disabled: needs one sequence \(n_seq=2\)', r'(?s)^(?!.*MTP chain: k=)', r'mtp_fold_decode: engaged n_seq=[12]\b',
            r'n_kv_pad += 4096\b'],
}
HEALTH_FATAL = r'CUDA error|GGML_ASSERT|out of memory|Segmentation fault|terminate called|sync_elide: missed:'


def health(profile):
    duo = profile.endswith('-duo')
    absent = os.environ.get('ENGAGEMENT_DROP', '').split()  # components compiled out of the tree under test (the nok build)
    shapes = ('np2', 'duo') if duo else ('np1',)
    gating = [c for c in ENGAGEMENT if c not in absent and all(s in ENGAGEMENT[c]['shapes'] for s in shapes)]
    recorded = [c for c in ENGAGEMENT if c not in gating and c not in absent]
    patterns = [p for c in gating for p in ENGAGEMENT[c]['patterns']] + HEALTH_EXTRA['duo' if duo else 'np1']
    return dict(env=HEALTH_ENV, patterns=patterns, gating=gating,
                recorded={c: ENGAGEMENT[c]['patterns'] for c in recorded}, fatal=HEALTH_FATAL)


def selftest():
    for comp, e in ENGAGEMENT.items():
        for path, literal in e['anchors']:
            assert literal in (ROOT / path).read_text(errors='replace'), f'{comp}: {literal!r} not in {path}'
        text = '\n'.join(e['samples'])
        for pattern in e['patterns']:
            assert re.search(pattern, text), f'{comp}: {pattern!r} does not match its sample'
        assert set(e['shapes']) <= set(ALL)
    # negative controls: a mutated line and a missing line must fail the regex
    assert not re.search(ENGAGEMENT['k8']['patterns'][0], ENGAGEMENT['k8']['samples'][0].replace('small_i=1', 'small_i=0'))
    assert not re.search(ENGAGEMENT['k8']['patterns'][0], ENGAGEMENT['k8']['samples'][0].replace('cols=12', 'cols=6'))
    assert not re.search(ENGAGEMENT['k4']['patterns'][0], ENGAGEMENT['k4']['samples'][0].replace('mutant=0', 'mutant=1'))
    assert not re.search(ENGAGEMENT['k11a']['patterns'][1], 'x\nsynchronize: sync_elide: missed: backend busy at a skipped sync (a device enqueue was not tracked), waited 31 us, caller f\n')
    assert not re.search(ENGAGEMENT['k11a']['patterns'][0], 'sync_elide: debug check engaged (backend query unavailable, timing fallback)')
    assert not re.search(ENGAGEMENT['k11a']['patterns'][0], 'a boot without the check')
    assert canary_env(['k8', 'k4'], 'np1') == {'GGML_SSM_CONV_RS_LOG': '1'}
    assert canary_env(['k8', 'k4'], 'np2') == {'GGML_CUDA_MMQ_TILES_LOG': '1', 'GGML_SSM_CONV_RS_LOG': '1'}
    assert 'k2' not in ENGAGEMENT
    for comp in ('k7', 'k10'):  # the nok tree must not print them: the dropped-variant canary relies on their absence
        for path, literal in ENGAGEMENT[comp]['anchors']:
            assert literal not in (NOK / path).read_text(errors='replace'), f'{comp}: {literal!r} still in {NOK / path}'
    assert all(re.search(p, 'load_tensors:   CPU_Mapped model buffer size =   644.00 MiB') for p in BASE_P90)
    assert not all(re.search(p, ENGAGEMENT['p90']['samples'][0]) for p in BASE_P90)
    assert not re.search(ENGAGEMENT['p90']['patterns'][0], 'load_tensors:   CPU_Mapped model buffer size =   644.00 MiB')
    assert not re.search(ENGAGEMENT['p90']['patterns'][0], 'load_tensors:   CUDA_Host_Mapped model buffer size =     0.00 MiB')
    h1, h2 = health('swift15-j5q'), health('swift15-iq4-duo')
    assert 'k8' not in h1['gating'] and 'k8' not in h2['gating'] and 'k6b' not in h2['gating'] and 'k6b' not in h1['gating']
    assert {'base', 'k7', 'k4', 'k6a', 'k10', 'k11a', 'k11b', 'p90'} <= set(h1['gating']) & set(h2['gating'])
    for pattern in h1['patterns'] + h2['patterns']:
        re.compile(pattern)
    assert re.search(h2['patterns'][-1], 'llama_context: n_kv_pad              = 4096')
    assert not re.search(h2['patterns'][-1], 'llama_context: n_kv_pad              = 256')
    assert re.search(HEALTH_FATAL, 'x\nsynchronize: sync_elide: missed: backend busy at a skipped sync\n')
    os.environ['ENGAGEMENT_DROP'] = 'k7 k10'
    assert not {'k7', 'k10'} & set(health('swift15-j5q')['gating']) and 'k4' in health('swift15-j5q')['gating']
    del os.environ['ENGAGEMENT_DROP']
    print(f'engagement: {len(ENGAGEMENT)} components, anchors in source, regexes match samples and reject mutants PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    elif sys.argv[1:2] == ['health'] and len(sys.argv) == 3:
        print(json.dumps(health(sys.argv[2])))
    elif sys.argv[1:] == ['list']:
        for comp, e in ENGAGEMENT.items():
            print(comp, ','.join(e['shapes']), e['env'], e['patterns'], sep='\t')
    else:
        raise SystemExit(__doc__)
