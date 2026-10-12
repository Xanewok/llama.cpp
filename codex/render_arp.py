"""Render the ar-readfirst A/B items (frozen/pkg89-base = master abee0c847 vs frozen/ar-readfirst) and contract-check them;
never submits. Adapted from work/zc-pr-window/render_zc.py (same templates, fixtures, pins and render path).

arp-np1-identity-j5q  J5q np1 (6-row verify AllReduces), cross-arm token identity at 8k/32k + np1 timing, ABBA.
arp-np2-real-j5q      served J5q duo shape (np2, 12-row AllReduces), real prompts 8k/32k/90k, ABBA. Timing (np2 tokens do not repeat).
Arm proof: canary boots run at verbosity 4, where ggml_cuda_ar_pipeline_init's INFO lines print: the rf build logs its kernel
configuration line (`chunked kernel round_vecs=1 store_first=0 ...`, local commit), master logs only `initialized AllReduce pipeline`.
--perf PERF_TSV: the window's micro-benchmark; the note's expectation is 128 AllReduce calls per round x the per-call delta (D).
--dry: labelled stand-in for frozen/ar-readfirst under dry/ (ids dry-*), for `gpuq submit --check` before the window.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
WT = Path('/home/xanewok/repos/qwen38-eval-worktrees')
EVAL = Path('/home/xanewok/repos/qwen38-eval')
SNAPSHOT = EVAL / 'orchestration/gpuq/snapshots/harness-4aee1dfcaccd836b42938c5047d553a0e794d2dfa6654f549021a4a33762a07f'
CCX = WT / 'work/aa-load/ccx01.sh'
POSTRUN = HERE / 'postrun_proofs.sh'
BASE_TREE, BASE_SHA = WT / 'wt/embd-master-base', 'abee0c8476b585b0d081d580b5ce21d0addc22c4'
FROZEN = 'ar-readfirst'
VANILLA = WT / 'wt/vanilla-master/codex'
TEMPLATES = {'np2-j5q': VANILLA / 'items/vanilla-np2-real-j5q.toml.in',
             'np1-j5q': VANILLA / 'items/vanilla-depth-j5q.toml.in'}
FIXTURE_PINS = EVAL / 'harness/experiments/duo-bub-np2-real-iq4.toml'
TRACED = {'np2-j5q': VANILLA / 'results/vanilla-np2-real-j5q/canary-a5/0-vanilla/server.log',
          'np1-j5q': VANILLA / 'results/vanilla-depth-j5q/canary-a1/0-vanilla/server.log'}
TRACED_VERBOSE = WT / 'work/pkg89-window/window/a1/placement/base/server.log'
sys.path.insert(0, str(SNAPSHOT))
from harness import mkitem  # noqa: E402

CANARY_ENV = {'LLAMA_ARG_LOG_VERBOSITY': '4'}
AR_INIT = 'ggml_cuda_ar_pipeline_init: initialized AllReduce pipeline: 2 GPUs'
RF_CFG = 'ggml_cuda_ar_pipeline_init: chunked kernel round_vecs=1 store_first=0 mutant=0 stall_cycles=0'
NOT_RF = '(?s)^(?!.*chunked kernel round_vecs=)'
COMMON = ['common_speculative_init_result: creating MTP draft context against the target model',
          'backend sampling .*SPLIT_MODE_TENSOR.*using CPU',
          'NCCL not compiled in; falling back to internal AllReduce']
SLOTS = {'np2': 'load_model: initializing, n_slots = 2, n_ctx_slot = 102400\\b',
         'np1': 'load_model: initializing, n_slots = 1, n_ctx_slot = 102400\\b'}
AR_CALLS_PER_ROUND = 128   # V pass, 2 x 64 layers (round-floor REPORT, M)


def expectation(perf, cell, fallback_ms):
    """ms/round delta = calls per round x per-call delta (batch 8) of arm rf vs base; D when read from the window, G otherwise."""
    if perf is None:
        return f'{fallback_ms:+.2f} ms/round (G, no perf table)'
    for line in Path(perf).read_text().splitlines():
        f = line.split('\t')
        if len(f) > 8 and f[0] == '8' and f[1] == cell and f[2] == 'rf':
            return f'{AR_CALLS_PER_ROUND * float(f[7]) / 1000:+.2f} ms/round (D: {AR_CALLS_PER_ROUND} calls x {f[7]} us, {cell} batch 8, {f[8]} %)'
    raise SystemExit(f'render_arp: {perf} has no rf row for {cell}')


def notes(perf):
    why = ('why: AllReduce chunked kernel, read-first pipeline (loads of round r-1 issued before the stores of round r) on master '
           'abee0c847; byte-identical sums (window identity vs base, mutants fail)')
    return {
        'np2': why + (f' | expect: rf - base {expectation(perf, "rows12", -0.8)} at 8k/32k/90k, acceptance equal | falsifier: '
                      'delta above -0.1 ms/round at 2 depths or any above +0.1; postrun_proofs FAIL | answers: serve/package '
                      'the read-first kernel.'),
        'np1': why + (f' | expect: cross-arm token identity at 8k/32k; rf - base {expectation(perf, "rows6", -0.3)} | falsifier: '
                      'any token difference (integrity); postrun_proofs FAIL | answers: model-level exactness + np1 gain.'),
    }


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


class Roots:
    """Frozen roots: the real ones after the window's freeze, or a labelled stand-in for frozen/ar-readfirst (--dry)."""

    def __init__(self, dry, sha):
        self.dry = dry
        self.sha = sha
        self.dir = HERE / 'dry' if dry else HERE / 'items/rendered'
        self.base = WT / 'frozen/pkg89-base'
        self.rf = self.dir / f'frozen/{FROZEN}' if dry else WT / f'frozen/{FROZEN}'
        if dry:
            self.stand_in(self.rf, sha)
        for root, s in ((self.base, BASE_SHA), (self.rf, sha)):
            assert (root / 'SOURCE').read_text().split()[0] == s, f'{root}: SOURCE is not {s}'
            assert (root / 'build/bin/round-trace').is_file()
            assert (dry and root == self.rf) or (root / 'STAMPS.txt').is_file(), f'{root} lacks STAMPS.txt'
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def stand_in(root, sha):
        """Labelled and unrunnable: only the harness contract check may read it."""
        binary = root / 'build/bin'
        binary.mkdir(parents=True, exist_ok=True)
        (root / 'SOURCE').write_text(sha + '\nDRY STAND-IN, never a measurement\n')
        (root / 'RECIPE.txt').write_text('DRY STAND-IN\n')
        (root / 'DRY-STAND-IN').write_text('DRY STAND-IN\n')
        exe = binary / 'round-trace'
        exe.write_text('#!/bin/false\n# DRY STAND-IN\n')
        exe.chmod(0o755)
        (binary / 'SHA256SUMS').write_text(f'{hashlib.sha256(exe.read_bytes()).hexdigest()}  round-trace\n')


def traced(shape_key, common):
    """Common patterns hold on a default-verbosity master log; on the verbose master boot the base AllReduce line holds and
    the rf configuration line misses (else it proves nothing)."""
    text = TRACED[shape_key].read_text(errors='replace')
    missing = [p for p in common if not re.search(p, text)]
    assert not missing, f'{TRACED[shape_key]}: patterns miss {missing}'
    verbose = TRACED_VERBOSE.read_text(errors='replace')
    assert re.search(AR_INIT, verbose) and re.search(NOT_RF, verbose), 'base AllReduce patterns miss the verbose master log'
    assert not re.search(RF_CFG, verbose) and not re.search(RF_CFG, text), 'rf pattern matches a master log'


def arms(template_arm, roots, shape_key):
    assert template_arm['flags'] == [] and set(template_arm['env']) == {'LD_LIBRARY_PATH'}, template_arm
    common = [*COMMON, SLOTS[shape_key[:3]]]
    traced(shape_key, common)
    base = copy.deepcopy(template_arm) | dict(label='base', commit=BASE_SHA, root=str(roots.base), repo=str(BASE_TREE),
                                              canary_log=[*common, AR_INIT, NOT_RF])
    rf = copy.deepcopy(base) | dict(label='rf', commit=roots.sha, root=str(roots.rf), repo=str(TREE),
                                    canary_log=[*common, AR_INIT, RF_CFG])
    return {'base': base, 'rf': rf}


def template(path):
    text = path.read_text()
    for placeholder in ('@VANILLA_ROOT@', '@VANILLA_SHA@', '@STACK9_P55_SHA@', '@STACK10_SHA@'):
        text = text.replace(placeholder, '0' * 40 if placeholder.endswith('SHA@') else '/nonexistent')
    return tomllib.loads(text)


def pins(shape_key):
    fixtures = tomllib.loads(FIXTURE_PINS.read_text())['extra_pins']
    own = [HERE / n for n in ('render_arp.py', 'postrun_proofs.sh', 'bw-ar-readfirst.sh', 'ar_summary.py')]
    p = [*fixtures, str(CCX), *map(str, own), str(TEMPLATES[shape_key]), str(TRACED[shape_key]), str(TRACED_VERBOSE)]
    return list(dict.fromkeys(str(q) for q in p if Path(q).exists()))


def cpu_evidence(window):
    return dict(path=str(window or HERE / 'mac-gate.sh'), command=(
        'Mac CPU gate (bin/bw-mac wt/ar-readfirst codex/mac-gate.sh): protocol model of the kernel text, read-first and store-first, '
        'wait-removed mutant fails. GPU: window bw-ar-readfirst: driver byte identity vs master on both GPUs (F32/F16/BF16, BF16 '
        'and F32 wire, copy engine off, tails, NaN/Inf), stalled rank 1, no-wait mutants fail, compute-sanitizer, test-llama-archs '
        'Meta rows equal to master, per-call micro-benchmark gate (rf vs base >= 10 % at 6 and 12 rows, no decode regression).'))


def experiment(roots, name, shape_key, title, window, plan=('base', 'rf', 'rf', 'base')):
    x = template(TEMPLATES[shape_key])
    x['arms'] = arms(x['arms']['vanilla'], roots, shape_key)
    prefix = 'dry-' if roots.dry else ''
    x |= dict(id=prefix + name, priority=25, plan=list(plan),
              out=str(EVAL / f'results/{name}') if not roots.dry else str(roots.dir / 'results' / name),
              title=('DRY STAND-IN, DO NOT SUBMIT: ' if roots.dry else '') + title)
    x['server'] |= dict(boot_cap_s=300, request_cap_s=600)
    x['canary'] = dict(plan=['base', 'rf'], depths=[8192], proofs=[], env=dict(CANARY_ENV))
    x['cpu_evidence'] = cpu_evidence(window)
    for key in ('boot_timeout_s', 'boot_wall_s'):
        x['depth'].pop(key, None)
    x['depth']['fixtures'] = str(EVAL / 'orchestration/gpuq/fixtures')
    x['extra_pins'] = pins(shape_key)
    return x


NP2_IDENTITY_OFF = ('Concurrent GPU ubatch arrival varies across boots, so np2 tokens do not repeat across boots. Exactness: the '
                    'window driver byte identity vs master (both GPUs, every type and wire), test-llama-archs Meta rows equal to '
                    'master, and arp-np1-identity-j5q (cross-arm tokens). This is a timing item.')


def np2_experiment(roots, window, note):
    x = experiment(roots, 'arp-np2-real-j5q', 'np2-j5q',
                   'ar-readfirst AllReduce chunked kernel (loads of round r-1 before stores of round r) vs upstream master abee0c847: '
                   'swift15-j5q served duo shape (np2, -b 2048 -ub 128, CPU sampling under -sm tensor, no chain, no fold, cram 0, '
                   'joint start), real prompts 8k/32k/90k, untraced, ABBA, CCX0+1', window)
    assert x['depth']['streams'] == 2 and x['depth']['request']['ignore_eos'] is True
    x['depth'] |= dict(wall_s=450, identity_off_reason=NP2_IDENTITY_OFF)
    x['budget'] |= dict(canary_s=480, cleanup_s=150, est_s=2100, startup_grace_s=300, preflight_s=600)
    return x, note


def np1_experiment(roots, window, note):
    x = experiment(roots, 'arp-np1-identity-j5q', 'np1-j5q',
                   'ar-readfirst AllReduce chunked kernel (loads of round r-1 before stores of round r) vs upstream master abee0c847: '
                   'swift15-j5q served np1 flags, cross-arm token identity at 8k/32k (exactness of the reduction) + np1 timing, '
                   'ABBA, CCX0+1', window)
    assert x['depth']['streams'] == 1
    x['depth'] |= dict(depths=[8192, 32768], identity='cross-arm', wall_s=240)
    x['depth'].pop('identity_off_reason', None)
    x['budget'] |= dict(canary_s=360, cleanup_s=150, est_s=900, startup_grace_s=300, preflight_s=600)
    return x, note


def render(x, note, roots):
    path = roots.dir / f'{x["id"]}.toml'
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
            lines += ['[analysis]', 'cmd = ' + json.dumps(['bash', str(POSTRUN), run_dir, str(len(x['plan']))]), 'timeout_s = 60', '']
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    text = text.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
    title = next(line for line in text.splitlines() if line.startswith('title = '))
    text = text.replace(title, title + '\nnote = ' + json.dumps(note), 1)
    item = tomllib.loads(text)
    assert str(path) in item['pins']['files'] and len(item['note']) <= 600 and item['canary']['timeout_s'] <= 600
    assert len(x['plan'])*mkitem.boot_wall(e) + x['budget']['canary_s'] <= item['budget_s'] <= 3600
    assert item['canary']['cmd'].count('--canary') == 1
    assert set(item['canary'].get('env', {})) <= {'GPUQ_ACTIVE'} and x['canary']['env'] == CANARY_ENV
    dest = path.with_suffix('.gpuq.toml')
    dest.write_text(text)
    proc = subprocess.run(['gpuq', 'submit', '--check', str(dest)], capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    dest.with_suffix('.contract.log').write_text(log)
    ok = proc.returncode == 0 and 'contract violated' not in log.lower()
    print(f'checked, NOT submitted: {dest}; rc={proc.returncode}; est_s_G={item["est_s"]}; budget_s_D={item["budget_s"]}')
    assert ok, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true', help='stand-in frozen root under dry/ (never submittable)')
    parser.add_argument('--perf', help="the window's perf.tsv (expectations in the notes)")
    parser.add_argument('--window', help="the window's output dir (cpu_evidence path)")
    args = parser.parse_args()
    sha = subprocess.run(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    roots = Roots(args.dry, sha)
    n = notes(args.perf)
    window = Path(args.window) / 'stages.tsv' if args.window else None
    render(*np1_experiment(roots, window, n['np1']), roots)
    render(*np2_experiment(roots, window, n['np2']), roots)


if __name__ == '__main__':
    main()
