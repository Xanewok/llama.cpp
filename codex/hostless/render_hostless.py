"""Render the hostless spike GPU items (frozen/hostless-spike, LLAMA_HOSTLESS=0..3 in one build) and contract-check them; never
submits.

hostless-np2-real-iq4: the chain-m2-s10-np2-real-iq4 A/B shape (served swift15-iq4-duo flags, np2, real prompts 8k/32k/90k,
  untraced, ABBA, CCX0+1) with arms L0 (= M2) and L3 (resident table + device-fed V + one-round output lag) of one build.
hostless-trace-iq4: trace_hostless.sh (nsys, np2 8k + 90k, arms L0 L1 L2 L3, one boot each, proof boot per arm with
  LLAMA_HOSTLESS_VERIFY=1, VRAM per GPU in <arm>/vram.csv).
Proof of engagement: WARN lines `hostless: ...` (default verbosity) per level; L0 logs none. The A/B patterns are traced against
the real chain-m2-s10 canary log at render time (L0 patterns match it, L3-only patterns miss it).
--draft: the build is not frozen yet (expected pins; the check fails on the missing files by design).
--standin DIR: renders both items on frozen/chain-m2-s10 into DIR, so the check covers everything but the missing build.
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
M2_AB = WT / 'wt/chain-m2-s10/codex/m2s10/chain-m2-s10-np2-real-iq4.toml'
M2_TRACE = WT / 'work/m2s10-trace/m2s10-trace-iq4.toml'
CCX = WT / 'work/aa-load/ccx01.sh'
POSTRUN = HERE / 'postrun_proofs.sh'
TRACE_SH = HERE / 'trace_hostless.sh'
FIXED, M2 = WT / 'frozen/hostless-spike', WT / 'frozen/chain-m2-s10'
M2_SHA = 'eedee42c50f928e48a28a030a73238e4e25efa64'
MAC = [WT / 'work/build-window-1007/hostless-spike-mac-a7', WT / 'work/build-window-1007/hostless-spike-mac-a8']
TRACED = [EVAL / 'results/2026-10-11-chain-m2-s10-np2-real-iq4/canary-a1/1-patch/server.log']
DATE = '2026-10-12'
sys.path.insert(0, str(SNAPSHOT))
from harness import mkitem

KNOBS = ['LLAMA_HOSTLESS', 'LLAMA_HOSTLESS_MUTANT', 'LLAMA_HOSTLESS_VERIFY']
FUSION = ['gdn_cache_fusion: graph device=0 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b',
          'gdn_cache_fusion: graph device=1 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b']
COMMON = ['(?s)n_kv_pad += 4096\\b.*n_kv_pad += 4096\\b', 'mtp_fold_decode: engaged n_seq=2\\b', *FUSION,
          'one pinned copy per input, uploads slowest host link first: CUDA[01] \\(', 'MTP chain: k=5 seqs=2\\b',
          '(?s)^(?!.*MTP chain disabled)', '(?s)^(?!.*drafting step by step)', '(?s)^(?!.*the batch does not fit the MTP chain)']
L0_LOG = [*COMMON, '(?s)^(?!.*hostless)']
L3_LOG = [*COMMON, 'hostless: level=3 mutant=none\\b', 'hostless: tok_embd buffer=Meta\\S* type=iq4_xs bytes=675430400\\b',
          'hostless: draft id slots=64 buffer=Meta', 'hostless: chain drafts deferred k=5 seqs=[12]\\b',
          'hostless: target batch reads device draft ids rows=\\d+ slot_rows=[1-9]', 'hostless: device order engaged backends=[1-9]',
          'hostless: output lag engaged', 'hostless verify: batches=[1-9]\\d* rows=\\d+ mismatches=0\\b',
          '(?s)^(?!.*hostless verify: MISMATCH)']
NOTE_AB = ('why: host-less round spike; M2 np2 rounds lose A = 1.39/1.51 ms (8k/90k, M, sync-map) to host boundaries S1/S2/S3 '
           '| expect (D): L3 - L0 -0.6..-0.9 ms/round at 8k/32k/90k, tokens/round equal within noise '
           '| falsifier: delta above -0.3 ms/round at 2 depths, or any depth above +0.1; postrun_proofs FAIL; VRAM free < 300 MiB '
           '| answers: the upper bound of removing S1+S3 and the process_token tail, before any device-acceptance work.')
NOTE_TRACE = ('why: host-less spike mechanism per level | expect (D): L1 drops the ? split (S1), L2 the draft->V boundary (S3, V '
              'queued behind C), L3 the process_token tail; S2 (V -> acceptance -> C) stays | falsifier: an A bucket that does '
              'not shrink at its level, B/C growth > 0.1 ms, a hostless proof FAIL | answers: gapcause.py per arm (shapes '
              'L0 V?C, L1+ VC), twin ms/round in runs.json, VRAM per GPU in <arm>/vram.csv')


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
    names = sorted(p.name for p in (M2 / 'build/bin').iterdir())
    return [root / 'SOURCE', root / 'RECIPE.txt', root / 'STAMPS.txt', *(root / 'build/bin' / n for n in names)]


def commit_of(root, draft):
    if root.exists():
        return (root / 'SOURCE').read_text().splitlines()[0]
    assert draft
    return subprocess.check_output(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], text=True,
                                   env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin'}).strip()


def traced(logs, l0, l3):
    """L0 patterns hold on the real M2 canary log; each L3-only pattern misses there (else it proves nothing)."""
    for log in logs:
        text = log.read_text(errors='replace')
        missing = [p for p in l0 if not re.search(p, text)]
        assert not missing, f'{log}: L0 patterns miss {missing}'
        hit = [p for p in l3 if p not in l0 and not p.startswith('(?s)^(?!') and re.search(p, text)]
        assert not hit, f'{log}: L3-only patterns match the M2 log {hit}'


def cpu_evidence():
    return dict(path=str(MAC[1] / 'judge.log'), command=(
        'codex/hostless/mac-{a,b}.sh (Mac M4, bw-np2-driver np2 k5 s1729, chain + fold C): L1/L2/L3 vs L0 bitwise tokens/'
        'draft counters/chain dump ids, plain CPU, HOST_DEFER (CPU as stream) and Meta(CPU,CPU) (L1 engages there); floor '
        'repeat equal; slotshift mutant differs and trips LLAMA_HOSTLESS_VERIFY (judge.py; part a: ' + str(MAC[0]) + '). '
        'x86: bw-hostless-spike stage 40'))


def ab_experiment(root, draft):
    x = tomllib.loads(M2_AB.read_text())
    m2 = x['arms']['patch']
    assert m2['root'] == str(M2) and m2['commit'] == M2_SHA and m2['flags'] == []
    assert m2['env']['LLAMA_MTP_CHAIN'] == '1' and m2['env']['LLAMA_MTP_FOLD_C'] == '1'
    name = 'hostless-np2-real-iq4'
    x |= dict(id=name, priority=20, out=f'{EVAL}/results/{DATE}-{name}', plan=['L0', 'L3', 'L3', 'L0'],
              title=f'hostless spike L3 vs L0 in one build ({root.name}, LLAMA_HOSTLESS env): {x["model"]["id"]} served duo '
                    'flags (np2, -b 2048 -ub 512, chain + fold C env both arms, cram 0, joint start), real prompts 8k/32k/90k, '
                    'untraced, ABBA, CCX0+1; L3 = token-embedding table on the GPUs + V batch from device draft ids + '
                    'one-round output lag')
    traced(TRACED, L0_LOG, L3_LOG)
    arm = dict(commit=commit_of(root, draft), root=str(root), repo=str(TREE), bin='build/bin', exe='build/bin/round-trace',
               flags=[])
    x['arms'] = {'L0': arm | dict(label='L0', canary_log=L0_LOG, env=m2['env'] | {'LLAMA_HOSTLESS': '0'}),
                 'L3': arm | dict(label='L3', canary_log=L3_LOG, env=m2['env'] | {'LLAMA_HOSTLESS': '3'})}
    x['server']['knobs'] = list(dict.fromkeys([*x['server']['knobs'], *KNOBS]))
    x['depth'] |= dict(identity_off_reason=(
        'Concurrent GPU ubatch arrival varies across boots, so np2 tokens do not repeat across arms or boots. Exactness: Mac '
        'np2 CPU identity of every level vs L0 (bw-np2-driver: plain, HOST_DEFER, Meta(CPU,CPU); slotshift mutant caught) and '
        'the GPU canary with LLAMA_HOSTLESS_VERIFY=1 (the ids V gathered on the device equal the host-read draft ids, '
        'mismatches=0); this is a timing item.'))
    x['canary'] = dict(plan=['L0', 'L3'], depths=[8192], proofs=[],
                       env={'LLAMA_ARG_LOG_VERBOSITY': '5', 'GGML_CUDA_DEBUG_GDN_STATE': '1', 'LLAMA_HOSTLESS_VERIFY': '1'})
    x['budget'] |= dict(canary_s=600, est_s=1400)
    x['cpu_evidence'] = cpu_evidence()
    own = [HERE / n for n in ('render_hostless.py', 'postrun_proofs.sh', 'bw-hostless-spike.sh', 'judge.py', 'cells.sh')]
    keep = [p for p in x['extra_pins'] if '/frozen/' not in p and '/wt/chain-m2-s10/' not in p]
    pins = keep + [str(p) for p in [*own, M2_AB, *TRACED, *(m / 'judge.log' for m in MAC)]]
    pins += [str(p) for p in frozen_files(root, draft)]
    x['extra_pins'] = list(dict.fromkeys(pins))
    return x


def render_ab(root, draft, dest_dir):
    x = ab_experiment(root, draft)
    path = dest_dir / f'{x["id"]}.toml'
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
    text = text.replace(title, title + '\nnote = ' + json.dumps(NOTE_AB), 1)
    item = tomllib.loads(text)
    assert str(path) in item['pins']['files'] and len(item['note']) <= 600 and item['canary']['timeout_s'] <= 600
    assert len(x['plan'])*mkitem.boot_wall(e) + x['budget']['canary_s'] <= item['budget_s'] <= 3600
    return check(path.with_suffix('.gpuq.toml'), text, item, draft)


def render_trace(root, draft, dest_dir):
    base = tomllib.loads(M2_TRACE.read_text())
    name = 'hostless-trace-iq4'
    out = f'/mnt/scratch/results/{DATE}-{name}'
    commit = commit_of(root, draft)
    common = dict(XID_SINCE=base['run']['env']['XID_SINCE'], ITEM=name, FROZEN=str(root), FROZEN_COMMIT=commit)
    keep = [p for p in base['pins']['files'] if '/frozen/' not in p and not p.endswith('trace_m2s10.sh')]
    pins = [str(TRACE_SH), *keep, str(HERE / 'render_hostless.py'), *(str(p) for p in frozen_files(root, draft))]
    sig = copy.deepcopy(base['signatures'])
    sig[0]['regex'] = sig[0]['regex'].replace('unknown arm', 'unknown arm|hostless proof failed')
    item = {
        'id': name, 'priority': 25, 'host_sensitive': False,
        'title': (f'hostless spike levels, np2 nsys trace: {root.name} with LLAMA_HOSTLESS=0..3 (arms L0..L3, one boot each), '
                  'served swift15-iq4-duo argv/env of the chain-m2-s10-np2-real-iq4 A/B (-np 2, -c 204800, -b 2048 -ub 512, '
                  '--cache-ram 0, fold C + chain env, k=5), both slots decoding at 8k and ~90k under nsys CUDA trace (no CPU '
                  'sampling) plus the unprofiled twin per window; proof boot per arm at verbosity 5 with LLAMA_HOSTLESS_VERIFY=1; '
                  'VRAM per GPU per arm; mechanism capture, not a timing A/B'),
        'note': NOTE_TRACE, 'kind': 'measure', 'budget_s': 3600, 'est_s': 1400, 'ab': False, 'retry': 'fresh',
        'startup_grace_s': 600, 'exit_codes': base['exit_codes'], 'signatures': sig,
        'run': {'cmd': ['bash', str(TRACE_SH)], 'cwd': base['run']['cwd'],
                'env': dict(DRY_RUN='0', CANARY='0', **common, BUDGET_S='3000', OUT=f'{out}/full-a{{attempt}}', PORT='{port}'),
                'ports': 'dynamic'},
        'preflight': {'cmd': ['bash', str(TRACE_SH), '--preflight'], 'timeout_s': 180,
                      'env': dict(CUDA_VISIBLE_DEVICES='', **common, OUT=f'{out}/preflight')},
        'canary': {'cmd': ['bash', str(TRACE_SH)], 'cwd': base['canary']['cwd'], 'timeout_s': 860,
                   'env': dict(DRY_RUN='0', CANARY='1', **common, BUDGET_S='800', RESERVE_S_CANARY='150',
                               OUT=f'{out}/canary-a{{attempt}}', PORT='{port}')},
        'outputs': {'paths': [out]},
        'fix': base['fix'],
        'pins': {'files': list(dict.fromkeys(pins))},
    }
    assert len(item['note']) <= 600
    return check(dest_dir / f'{name}.gpuq.toml', toml_text(item), item, draft)


def toml_text(item):
    def value(v):
        if isinstance(v, dict):
            return '{ ' + ', '.join(f'{k} = {value(y)}' for k, y in v.items()) + ' }'
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return '[\n' + ''.join(f'  {value(y)},\n' for y in v) + ']'
        if isinstance(v, list) and len(v) > 4:
            return '[\n' + ''.join(f'  {json.dumps(y)},\n' for y in v) + ']'
        return json.dumps(v)
    lines = [f'{k} = {value(v)}' for k, v in item.items() if not isinstance(v, dict) or k in ('exit_codes',)]
    for k, v in item.items():
        if isinstance(v, dict) and k != 'exit_codes':
            lines += ['', f'[{k}]', *(f'{a} = {value(b)}' for a, b in v.items())]
    return '\n'.join(lines) + '\n'


def check(dest, text, item, draft):
    dest.write_text(text)
    proc = subprocess.run(['gpuq', 'submit', '--check', str(dest)], capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    dest.with_suffix('.contract.log').write_text(log)
    ok = proc.returncode == 0 and 'contract violated' not in log.lower()
    print(f'checked, NOT submitted: {dest}; rc={proc.returncode}; est_s_G={item["est_s"]}; budget_s_D={item["budget_s"]}')
    assert ok or draft, log
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true', help='build not frozen yet: expected pins, check may fail')
    parser.add_argument('--standin', type=Path, help='scratch dir: both items on frozen/chain-m2-s10, check must pass')
    args = parser.parse_args()
    root, dest = (M2, args.standin) if args.standin else (FIXED, HERE)
    if not args.draft and not args.standin:
        assert FIXED.exists() and (FIXED / 'STAMPS.txt').exists()
    if args.standin:
        args.standin.mkdir(parents=True, exist_ok=True)
    render_ab(root, args.draft, dest)
    render_trace(root, args.draft, dest)


if __name__ == '__main__':
    main()
