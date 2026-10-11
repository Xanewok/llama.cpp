"""Render the h2d-contig GPU items (frozen/stack10 vs frozen/h2d-contig) and contract-check them; never submits.

h2d-contig-np2-real-{iq4,j5q}: served duo shapes (as chain-m2-s10-np2-real-*), np2 real prompts 8k/32k/90k, ABBA, CCX0+1.
  np2 tokens do not repeat across boots (identity off), so exactness rests on the window's CUDA byte-identity test and on
h2d-contig-np1-identity-j5q: served J5q np1, cross-arm token identity at 8k/32k (a wrong packing offset changes the inputs,
  so tokens and draft counters diverge, D), ABBA, also the np1 timing.
Proof of engagement: patch boots log `input_scatter: engaged device=N ... mutant=0` at WARN (every boot, default verbosity), up
boots never; the patterns are traced against real stack10 logs at render time. --draft: the build is not frozen yet.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
WT = Path('/home/xanewok/repos/qwen38-eval-worktrees')
EVAL = Path('/home/xanewok/repos/qwen38-eval')
SNAPSHOT = EVAL / 'orchestration/gpuq/snapshots/harness-4aee1dfcaccd836b42938c5047d553a0e794d2dfa6654f549021a4a33762a07f'
NP2_SOURCE = EVAL / 'harness/experiments/duo-bub-np2-real-iq4.toml'
NP1_SOURCE = EVAL / 'harness/experiments/aa-cpuload-j5q.toml'
J5Q = EVAL / 'harness/experiments/acc-audit-np2-j5q.toml'
CCX = WT / 'work/aa-load/ccx01.sh'
POSTRUN = HERE / 'postrun_proofs.sh'
FIXED, STOCK = WT / 'frozen/h2d-contig', WT / 'frozen/stack10'
STOCK_SHA = 'a588f2a44382a361df3e2a456be0c80ca10dd1d6'
MAC = WT / 'work/build-window-1007/h2d-contig-mac-a3'
# real stack10 canary logs (verbosity 5): up patterns must match, patch-only patterns must miss
TRACED_NP2 = [EVAL / 'results/2026-10-11-duo-bub-np2-real-iq4/canary-a1/0-b2048/server.log',
              EVAL / 'results/2026-10-09-s10-np2-real-j5q/canary-a2/0-s10/server.log']
TRACED_NP1 = [EVAL / 'results/2026-10-10-aa-cpuload-j5q/canary-a2/0-l/server.log']
sys.path.insert(0, str(SNAPSHOT))
from harness import mkitem

SCATTER = [f'input_scatter: engaged device={d} entries=[1-9]\\d* bytes=[1-9]\\d* mutant=0\\b' for d in (0, 1)]
NO_SCATTER = '(?s)^(?!.*input_scatter:)'
STAGE = 'one pinned copy per input, uploads slowest host link first: CUDA[01] \\('
FUSION = ['gdn_cache_fusion: graph device=0 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b',
          'gdn_cache_fusion: graph device=1 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b']
NP2_COMMON = ['(?s)n_kv_pad += 4096\\b.*n_kv_pad += 4096\\b', 'mtp_fold_decode: engaged n_seq=2\\b', *FUSION, STAGE,
              'MTP chain disabled: needs one sequence \\(n_seq=2\\)']
NP1_COMMON = ['MTP chain: k=5\\b', 'mtp_fold_decode: engaged n_seq=1\\b', STAGE]

NP2 = {
    # served swift15-iq4-duo: -b 2048 -ub 512; boots 301-308 s M
    'iq4': dict(model=None, b='2048', ub='512', est_s=1400),
    # served swift15-j5q-duo: -b 2048 -ub 128; boots 356-387 s M
    'j5q': dict(model=J5Q, b='2048', ub='128', est_s=1700),
}
WHY = ('why: STEP fix 3; stack10 np2 boundary input staging 0.32 ms/round, 14 cudaMemcpyBatchAsync/round at ~17 us API each '
       '(M, attrib-s10 np2); the patch issues one cudaMemcpyAsync (~3.4 us M) + one scatter launch (~4 us M) per device upload')
NOTE_NP2 = (WHY + ' | expect: -0.1..-0.2 ms/round (D: 14 x ~10 us host API) at 8k/32k/90k, acceptance equal '
            '| falsifier: delta above -0.05 ms/round at 2 depths or any above +0.1; postrun_proofs FAIL '
            '| answers: is STEP fix 3 worth a package and a stack slot.')
NOTE_NP1 = (WHY + ' | expect: cross-arm token identity at 8k/32k; -0.03..-0.08 ms/round np1 (D: 6 calls/round) '
            '| falsifier: identity failure (integrity) = packing bug; postrun_proofs FAIL '
            '| answers: model-level exactness of the packed upload.')


def experiment_text(x):
    def value(v):
        if isinstance(v, dict):
            return '{ ' + ', '.join(k + ' = ' + value(y) for k, y in v.items()) + ' }'
        return json.dumps(v)
    def emit(name, data):
        text = f'[{name}]\n' + ''.join(f'{k} = {value(v)}\n' for k, v in data.items() if not isinstance(v, dict)) + '\n'
        return text + ''.join(emit(name + '.' + k, v) for k, v in data.items() if isinstance(v, dict))
    head = ''.join(f'{k} = {json.dumps(v)}\n' for k, v in x.items() if not isinstance(v, dict)) + '\n'
    return head + ''.join(emit(k, v) for k, v in x.items() if isinstance(v, dict))


def frozen_files(root, draft):
    if root.exists():
        stamps = [root / 'STAMPS.txt'] if (root / 'STAMPS.txt').exists() else []
        return [root / 'SOURCE', root / 'RECIPE.txt', *stamps, *sorted((root / 'build/bin').iterdir())]
    assert draft, f'{root} is missing; render after the window or pass --draft'
    names = sorted(p.name for p in (STOCK / 'build/bin').iterdir())
    return [root / 'SOURCE', root / 'RECIPE.txt', root / 'STAMPS.txt', *(root / 'build/bin' / n for n in names)]


def fixed_commit(draft):
    if FIXED.exists():
        return (FIXED / 'SOURCE').read_text().splitlines()[0]
    assert draft
    return subprocess.check_output(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], text=True,
                                   env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin'}).strip()


def traced(logs, up, patch):
    """Up patterns hold on real stack10 logs; each patch-only pattern misses there (else it proves nothing)."""
    for log in logs:
        text = log.read_text(errors='replace')
        missing = [p for p in up if not re.search(p, text)]
        assert not missing, f'{log}: up patterns miss {missing}'
        hit = [p for p in patch if p not in up and re.search(p, text)]
        assert not hit, f'{log}: patch-only patterns match the stock log {hit}'


def arms(stock, up_log, patch_log, draft):
    up = copy.deepcopy(stock) | dict(label='up', flags=[], canary_log=up_log)
    up.pop('host_load', None)
    patch = copy.deepcopy(up) | dict(label='patch', root=str(FIXED), commit=fixed_commit(draft), repo=str(TREE),
                                     canary_log=patch_log)
    return {'up': up, 'patch': patch}


def pins(x, source, traced_logs, draft, extra=()):
    own = [HERE / n for n in ('render_h2d.py', 'postrun_proofs.sh', 'bw-h2d-contig.sh')]
    p = x['extra_pins'] + [str(f) for f in [*own, source, *extra, *traced_logs]]
    p += [str(f) for f in frozen_files(FIXED, draft) + frozen_files(STOCK, draft)]
    return list(dict.fromkeys(p))


def cpu_evidence():
    return dict(path=str(MAC / 'stages.tsv'), command=(
        'codex/h2d/bw-h2d-mac.sh (Mac M4): CPU build, test-meta-input-stage (CUDA parts skip), ggml-cuda.cu staging + scatter '
        'code on a host CUDA stand-in (per-rank/shared, lazy/random stream progress, >32 entries, every copy width, region '
        'regrowth, fallback), 9 code mutants and the env mutant caught. GPU: bw-h2d-contig stage 42 byte identity on both GPUs, '
        'GGML_CUDA_INPUT_SCATTER_MUTANT=1 must fail'))


def np2_experiment(shape, draft):
    s = NP2[shape]
    x = tomllib.loads(NP2_SOURCE.read_text())
    stock = x['arms']['b2048']
    assert stock['root'] == str(STOCK) and stock['commit'] == STOCK_SHA and stock['flags'] == ['-b', '2048']
    assert x['depth']['request']['ignore_eos'] is True
    if s['model']:
        x['model'] = tomllib.loads(s['model'].read_text())['model']
    name = f'h2d-contig-np2-real-{shape}'
    x |= dict(id=name, priority=20, out=f'{EVAL}/results/2026-10-11-{name}', plan=['up', 'patch', 'patch', 'up'],
              title=f'STEP fix 3 one H2D + device scatter per upload vs frozen/stack10: {x["model"]["id"]} served duo flags '
                    f'(np2, -b {s["b"]} -ub {s["ub"]}, chain + fold C env both arms, cram 0, joint start), real prompts '
                    '8k/32k/90k, untraced, ABBA, CCX0+1')
    args = x['server']['args']
    i = args.index('{arm_flags}')
    args[i:i + 3] = ['-b', s['b'], '-ub', s['ub']]
    assert args[i + 4] == '--jinja' and '-ub' not in args[i + 4:]
    up_log, patch_log = [*NP2_COMMON, NO_SCATTER], [*NP2_COMMON, *SCATTER]
    traced(TRACED_NP2, up_log, patch_log)
    x['arms'] = arms(stock | dict(flags=[]), up_log, patch_log, draft)
    x['depth'] |= dict(wall_s=450, identity_off_reason=(
        'Concurrent GPU ubatch arrival varies across boots, so np2 tokens do not repeat across boots. Exactness: the window '
        'byte-identity test on both GPUs (per rank and shared, mutant must fail) and h2d-contig-np1-identity-j5q '
        '(cross-arm tokens). Same graphs both arms: this is a timing item.'))
    x.pop('question', None)
    x['canary'] = dict(plan=['up', 'patch'], depths=[8192], proofs=[],
                       env={'LLAMA_ARG_LOG_VERBOSITY': '5', 'GGML_CUDA_DEBUG_GDN_STATE': '1'})
    x['budget'] |= dict(canary_s=600, est_s=s['est_s'])
    x['cpu_evidence'] = cpu_evidence()
    x['extra_pins'] = pins(x, NP2_SOURCE, TRACED_NP2, draft, [J5Q])
    return x, NOTE_NP2


def np1_experiment(draft):
    x = tomllib.loads(NP1_SOURCE.read_text())
    stock = x['arms']['q']
    assert stock['root'] == str(STOCK) and stock['commit'] == STOCK_SHA and stock['flags'] == []
    assert x['depth']['streams'] == 1 and x['depth']['identity'] == 'cross-arm'
    name = 'h2d-contig-np1-identity-j5q'
    x.pop('hostload', None)
    x |= dict(id=name, priority=20, out=f'{EVAL}/results/2026-10-11-{name}', plan=['up', 'patch', 'patch', 'up'],
              title='STEP fix 3 one H2D + device scatter per upload vs frozen/stack10: swift15-j5q served np1 flags, '
                    'cross-arm token identity at 8k/32k (exactness gate of the packed upload) + np1 timing, ABBA, CCX0+1')
    x['protected'] = [p for p in x['protected'] if p != 'hostload']
    up_log, patch_log = [*NP1_COMMON, NO_SCATTER], [*NP1_COMMON, *SCATTER]
    traced(TRACED_NP1, up_log, patch_log)
    x['arms'] = arms(stock, up_log, patch_log, draft)
    x['depth'] |= dict(wall_s=450)
    x.pop('question', None)
    x['canary'] = dict(plan=['up', 'patch'], depths=[8192], proofs=[], env={'LLAMA_ARG_LOG_VERBOSITY': '5'})
    x['budget'] |= dict(canary_s=600, est_s=1300)
    x['cpu_evidence'] = cpu_evidence()
    x['extra_pins'] = pins(x, NP1_SOURCE, TRACED_NP1, draft)
    return x, NOTE_NP1


def render(x, note, draft):
    path = HERE / f'{x["id"]}.toml'
    path.write_text(experiment_text(x).rstrip() + '\n')
    e = mkitem.lib.read_experiment(path)
    mkitem.harness_snapshot = lambda _: SNAPSHOT
    text = mkitem.render(e).replace(str(SNAPSHOT / 'experiment.toml'), str(path))
    lines = []
    for line in text.splitlines():
        if line.startswith('cmd = ') and ('--execute' in line or '--canary' in line):
            line = 'cmd = ' + json.dumps(['bash', str(CCX), *json.loads(line.removeprefix('cmd = '))])
        if line == '[outputs]':
            run_dir = f'{x["out"]}/full-a{{attempt}}'
            lines += ['[analysis]', 'cmd = ' + json.dumps(['bash', str(POSTRUN), run_dir]), 'timeout_s = 60', '']
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    text = text.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
    title = next(line for line in text.splitlines() if line.startswith('title = '))
    text = text.replace(title, title + '\nnote = ' + json.dumps(note), 1)
    item = tomllib.loads(text)
    assert str(path) in item['pins']['files'] and len(item['note']) <= 600 and item['canary']['timeout_s'] <= 600
    assert len(x['plan'])*mkitem.boot_wall(e) + x['budget']['canary_s'] <= item['budget_s'] <= 3600
    dest = path.with_suffix('.gpuq.toml')
    dest.write_text(text)
    proc = subprocess.run(['gpuq', 'submit', '--check', str(dest)], capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    dest.with_suffix('.contract.log').write_text(log)
    ok = proc.returncode == 0 and 'contract violated' not in log.lower()
    print(f'checked, NOT submitted: {dest}; rc={proc.returncode}; est_s_G={item["est_s"]}; budget_s_D={item["budget_s"]}')
    assert ok or draft, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true', help='build not frozen yet: expected pins, check may fail')
    args = parser.parse_args()
    if not args.draft:
        assert FIXED.exists() and (FIXED / 'STAMPS.txt').exists()
    for shape in NP2:
        render(*np2_experiment(shape, args.draft), args.draft)
    render(*np1_experiment(args.draft), args.draft)


if __name__ == '__main__':
    main()
