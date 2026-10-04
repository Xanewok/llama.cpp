#!/usr/bin/env python3
"""Host-function attribution of llama.cpp STEP time: nsys CPU backtrace samples x GPU0 idle intervals of steady decode rounds.

STEP (attrib_pack axes) = GPU0 idle gaps + sampling kernels + memcpy. A CPU sample of a launching (decode) thread whose
timestamp falls inside such an interval is charged to its top-of-stack function and to the first llama/ggml frame on its
stack. ms/round = samples * sample_ms / steady rounds, with sample_ms the measured spacing of one thread's running samples.

Contract:
- Rounds, STEP intervals and the segment split (readback / cpu_pre / inputs / launch_lat) are the pack's own
  (attrib_pack.llama_rounds / sweep / gap_segment_intervals), so STEP ms/round here equals budget.md exactly (checked).
- Only threads that issue CUDA launches inside the steady rounds are attributed; other threads are reported as a diagnostic.
- Samples exist only while a thread runs: blocked time inside STEP is "unsampled", never charged to a function.
- A trace without sample tables, or with fewer than --min-samples STEP samples, is rejected, never summarised.
"""
import argparse
import bisect
import collections
import json
import statistics
import sys
from pathlib import Path

import attrib_pack as ap
from attrib_pack import Reject

STEP_CLASSES = ('IDLE', 'SAMP', 'COPY')
GAP_MS = 5.0       # consecutive samples of one thread further apart than this are not one running stretch
LAUNCH_SHARE = 0.05
NO_LLAMA = '(no llama/ggml frame)'
BROKEN = '[Broken backtraces]'


def is_llama(module):
    return Path(module).name.startswith(('libllama', 'libggml'))


def frame_label(sym, module, unresolved):
    base = Path(module).name
    return (f'[{base}]' if unresolved or sym.startswith('0x') else sym), base


def load_samples(path, tids, a, b):
    """Samples with a timestamp in [a, b): (events [(id, t, tid)] of `tids`, {id: [(symbol, module, unresolved)] leaf first},
    {tid: n samples} of the other threads, {tid: thread name}, capture options). Callchains are fetched for `tids` only;
    an event without callchain rows (samples-per-backtrace > 1) is kept with an empty stack."""
    db = ap.open_ro(path)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    miss = [t for t in ('COMPOSITE_EVENTS', 'SAMPLING_CALLCHAINS', 'ThreadNames') if t not in tables]
    if miss:
        raise Reject('no CPU samples in the trace (missing tables: ' + ', '.join(miss) + '); nsys start needs --sample=process-tree')
    names = {t: v for t, v in db.execute('select n.globalTid, s.value from ThreadNames n join StringIds s on s.id = n.nameId')}
    cfg = dict(db.execute("select name, value from META_DATA_CAPTURE where name like 'LINUX_PERF_OPTIONS:%' or name in "
                          "('USE_DWARF', 'USE_FRAME_POINTERS', 'USE_LBRS', 'USE_UNWIND_TABLES')").fetchall()) \
        if 'META_DATA_CAPTURE' in tables else {}
    marks = ','.join('?' * len(tids))
    events = db.execute(f'select id, start, globalTid from COMPOSITE_EVENTS where globalTid in ({marks}) and start >= ? and start < ? '
                        'order by start', (*tids, a, b)).fetchall()
    other = collections.Counter()
    for t, c in db.execute('select globalTid, count(*) from COMPOSITE_EVENTS where start >= ? and start < ? group by 1', (a, b)):
        if t not in tids:
            other[t] = c
    chains = collections.defaultdict(list)
    for eid, sym, mod, unres in db.execute(
            f'select e.id, s.value, m.value, c.unresolved from COMPOSITE_EVENTS e join SAMPLING_CALLCHAINS c on c.id = e.id '
            f'join StringIds s on s.id = c.symbol join StringIds m on m.id = c.module '
            f'where e.globalTid in ({marks}) and e.start >= ? and e.start < ? order by e.id, c.stackDepth', (*tids, a, b)):
        chains[eid].append((sym, mod, unres))
    db.close()
    return events, chains, other, names, cfg


def decode_threads(tr, a, b):
    n = collections.Counter(tid for s, e, nm, cid, tid in tr.runtime if nm in ap.LAUNCH and a <= s < b)
    tot = sum(n.values())
    if not tot:
        raise Reject('no CUDA launch calls inside the steady rounds')
    return {t: c for t, c in n.items() if c >= LAUNCH_SHARE * tot}


def sample_spacing_ms(ts_by_tid):
    gaps = []
    for ts in ts_by_tid.values():
        ts.sort()
        gaps += [(y - x) / 1e6 for x, y in zip(ts, ts[1:]) if 0 < (y - x) / 1e6 <= GAP_MS]
    if not gaps:
        raise Reject('decode thread has no consecutive running samples to calibrate the sample spacing')
    return statistics.median(gaps)


def classify_chain(frames):
    """frames = [(sym, module, unresolved)] leaf first -> (top label, first llama/ggml label, inclusive llama/ggml labels, broken)."""
    broken = any(m == BROKEN for _, m, _ in frames)
    fr = [(frame_label(s, m, u), m) for s, m, u in frames if m != BROKEN]
    if not fr:
        return ('(no stack)', ''), (NO_LLAMA, ''), set(), broken
    top = fr[0][0]
    ll = [lab for lab, m in fr if is_llama(m)]
    return top, (ll[0] if ll else (NO_LLAMA, '')), set(ll), broken


def host_attribution(tr, model, spec, defaults, classes=STEP_CLASSES, min_samples=200, min_rounds=None):
    engine = spec.get('engine') or defaults['engine']
    if engine != 'llama':
        raise Reject('host sampling attribution is implemented for llama.cpp traces only')
    k = spec.get('k', defaults.get('k'))
    min_rounds = min_rounds if min_rounds is not None else spec.get('min_rounds', defaults.get('min_rounds', 30))
    pack = ap.decode_budget(tr, model, spec['depth'], 'llama', k, min_rounds, spec.get('trim'), spec.get('tok_per_round'),
                            spec.get('draft_head_bytes'), spec.get('skip_ms', 750.0), spec.get('unprofiled'))
    trim = spec.get('trim') if spec.get('trim') is not None else 0.03
    rounds, _, info = ap.llama_rounds(tr, model.get('head_type'), k, trim, spec.get('skip_ms', 750.0))
    n = len(rounds)
    d0 = tr.devs[0]
    rt = ap.runtime_index(tr)
    steps, segs, wall = [], [], collections.Counter()
    lo, hi = None, None
    for r in rounds:
        w = ap.phase_windows(tr, d0, r, 'llama')
        a, b = w[0][1], w[-1][2]
        lo, hi = (a if lo is None else min(lo, a)), (b if hi is None else max(hi, b))
        for s, e, c in ap.sweep(tr, d0, a, b)[0]:
            wall[c] += e - s
            if c in classes:
                steps.append((s, e, c))
        segs += [(x, y, nm) for nm, x, y in ap.gap_segment_intervals(tr, r, rt)]
    steps.sort()
    segs.sort()
    step_ms = sum(wall[c] for c in classes) / n / 1e6
    if tuple(classes) == STEP_CLASSES and abs(step_ms - pack['axes']['STEP']['measured_ms']) > 1e-6:
        raise Reject(f"STEP wall {step_ms:.6f} ms/round disagrees with the pack's {pack['axes']['STEP']['measured_ms']:.6f}")
    tids = decode_threads(tr, lo, hi)
    ev, chains, other, names, cfg = load_samples(tr.path, list(tids), lo, hi)
    if not ev:
        raise Reject('only 0 decode-thread CPU samples in the steady rounds (launching thread(s) '
                     + ', '.join(f'{names.get(t, "?")} {t}' for t in tids) + ' never sampled)')
    ts = collections.defaultdict(list)
    for eid, t, tid in ev:
        ts[tid].append(t)
    sample_ms = sample_spacing_ms({t: list(v) for t, v in ts.items()})
    starts = [s for s, _, _ in steps]
    sstarts = [x for x, _, _ in segs]
    top, first, incl = collections.Counter(), collections.Counter(), collections.Counter()
    by_seg = collections.defaultdict(lambda: {'top': collections.Counter(), 'first': collections.Counter(), 'n': 0})
    by_cls = collections.Counter()
    n_in = n_broken = n_nollama = 0
    for eid, t, tid in ev:
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t >= steps[i][1]:
            continue
        n_in += 1
        by_cls[steps[i][2]] += 1
        tl, fl, inc, broken = classify_chain(chains.get(eid, []))
        top[tl] += 1
        first[fl] += 1
        for lab in inc:
            incl[lab] += 1
        n_broken += broken
        n_nollama += fl[0] == NO_LLAMA
        j = bisect.bisect_right(sstarts, t) - 1
        sg = segs[j][2] if j >= 0 and t < segs[j][1] else 'other'
        by_seg[sg]['n'] += 1
        by_seg[sg]['top'][tl] += 1
        by_seg[sg]['first'][fl] += 1
    if n_in < min_samples:
        raise Reject(f'only {n_in} decode-thread CPU samples inside STEP (< {min_samples}); sampling did not run or the window is too short')
    ms = lambda c: c * sample_ms / n
    rows = lambda cnt: [{'fn': f, 'module': m, 'samples': c, 'ms_per_round': ms(c), 'pct_of_sampled': 100 * c / n_in}
                        for (f, m), c in cnt.most_common()]
    seg_wall = collections.Counter()
    for x, y, nm in segs:
        seg_wall[nm] += y - x
    other_rows = [{'thread': names.get(t, '?'), 'tid': t, 'samples_in_window': c} for t, c in other.most_common(6)]
    warn = []
    cover = n_in * sample_ms / (step_ms * n)
    if cover < 0.5:
        warn.append(f'decode-thread samples cover {100 * cover:.0f}% of STEP wall: the rest is blocked/off-CPU and not attributed')
    if n_nollama > 0.2 * n_in:
        warn.append(f'{100 * n_nollama / n_in:.0f}% of STEP samples have no llama/ggml frame (broken or truncated stacks: {n_broken})')
    if cover > 1.2 * len(tids):
        warn.append(f'samples cover {cover:.2f}x of STEP wall on {len(tids)} thread(s): the sample spacing calibration is off')
    return {'id': spec['id'], 'depth': spec['depth'], 'n_rounds': n, 'step_classes': list(classes),
            'step_ms_per_round': step_ms, 'step_ms_by_class': {c: wall[c] / n / 1e6 for c in classes},
            'round_ms_gpu0': pack['round_ms_gpu0'], 'sample_ms': sample_ms, 'step_samples': n_in,
            'step_samples_by_class': dict(by_cls), 'sampled_cpu_ms_per_round': n_in * sample_ms / n,
            'coverage_of_step_wall': cover, 'truncated_stacks_pct': 100 * n_broken / n_in, 'unsampled_ms_per_round': step_ms - n_in * sample_ms / n,
            'decode_threads': [{'tid': t, 'name': names.get(t, '?'), 'launches': c, 'samples_in_window': len(ts[t])}
                               for t, c in tids.items()],
            'other_threads_samples_in_window': other_rows, 'capture': cfg,
            'top_of_stack': rows(top), 'first_llama_frame': rows(first),
            'inclusive_llama_frames': [{'fn': f, 'module': m, 'samples': c, 'ms_per_round': ms(c)} for (f, m), c in incl.most_common()],
            'segments': {nm: {'wall_ms_per_round': seg_wall[nm] / n / 1e6, 'samples': v['n'], 'sampled_ms_per_round': ms(v['n']),
                              'top_of_stack': rows(v['top'])[:12], 'first_llama_frame': rows(v['first'])[:12]}
                         for nm, v in by_seg.items()},
            'segment_wall_ms_per_round': {nm: v / n / 1e6 for nm, v in seg_wall.items()},
            'profiler_overhead': pack.get('profiler_overhead'), 'steady_selection': info, 'warnings': warn + pack['warnings']}


# ------------------------------------------------------------------------------------------------------- reporting
def short(f, n=88):
    return f if len(f) <= n else f[:n - 1] + '~'


def table(rows, key, n):
    L = ['| function | module | ms/round | % of sampled | samples |', '|---|---|---:|---:|---:|']
    for r in rows[:n]:
        L.append(f"| `{short(r['fn'])}` | {r['module']} | {r['ms_per_round']:.3f} | {r['pct_of_sampled']:.1f} | {r['samples']} |")
    return L


def trace_md(t, n=14):
    c = t['capture']
    L = [f"### {t['id']}", '',
         f"{t['n_rounds']} steady rounds, GPU0 round {t['round_ms_gpu0']:.2f} ms; STEP ({'+'.join(t['step_classes'])}) "
         f"{t['step_ms_per_round']:.2f} ms/round (" + ', '.join(f'{k} {v:.2f}' for k, v in sorted(t['step_ms_by_class'].items())) + '). '
         f"{t['step_samples']} CPU samples of the decode thread(s) fall inside STEP; sample spacing {t['sample_ms']:.3f} ms (measured) -> "
         f"{t['sampled_cpu_ms_per_round']:.2f} ms/round sampled = {100 * t['coverage_of_step_wall']:.0f}% of STEP wall; "
         f"{t['unsampled_ms_per_round']:.2f} ms/round unsampled (blocked, or between samples).",
         f"Decode threads: " + ', '.join(f"{d['name']} (tid {d['tid']}, {d['launches']} launches, {d['samples_in_window']} samples)"
                                       for d in t['decode_threads']) + '. Other threads, samples in window: ' +
         (', '.join(f"{o['thread']} {o['samples_in_window']}" for o in t['other_threads_samples_in_window']) or 'none') + '.',
         f"{t['truncated_stacks_pct']:.0f}% of the STEP samples have a truncated stack (nsys dwarf copies a fixed stack slice: outer frames are lost, "
         f"names near the leaf are kept, so 'first llama/ggml frame' is reliable and 'inclusive' frames high in the stack may be absent).",
         f"Capture: sampling period {c.get('LINUX_PERF_OPTIONS:SAMPLING_PERIOD', '?')} cycles, samples per backtrace "
         f"{c.get('LINUX_PERF_OPTIONS:SAMPLES_PER_BACKTRACE', '?')}, dwarf={c.get('USE_DWARF', '?')}, frame-pointer={c.get('USE_FRAME_POINTERS', '?')}.",
         '', 'First llama/ggml frame on the stack (ms per round of STEP):', ''] + table(t['first_llama_frame'], 'first', n) + [
         '', 'Top-of-stack function:', ''] + table(t['top_of_stack'], 'top', n) + [
         '', 'Inclusive (any llama/ggml frame anywhere on the stack; frames overlap, do not sum): ' + '; '.join(
             f"`{short(r['fn'], 60)}` {r['ms_per_round']:.2f}" for r in t['inclusive_llama_frames'][:10]), '']
    segs = t['segments']
    L.append('By gap segment (the pack\'s readback / cpu_pre / inputs / launch_lat split of the inter-block gaps; "other" = STEP time '
             'outside those segments, e.g. sampling kernels): wall ms/round, sampled ms/round, top first-llama frames.')
    L.append('')
    for nm in sorted(segs, key=lambda s: -segs[s]['sampled_ms_per_round']):
        v = segs[nm]
        wall = t['segment_wall_ms_per_round'].get(nm)
        L.append(f"- {nm}: wall {'-' if wall is None else f'{wall:.2f}'}, sampled {v['sampled_ms_per_round']:.2f}: " +
                 '; '.join(f"`{short(r['fn'], 50)}` {r['ms_per_round']:.2f}" for r in v['first_llama_frame'][:4]))
    L.append('')
    L += [f'WARNING: {w}' for w in t['warnings']] + ([''] if t['warnings'] else [])
    return L


def delta_rows(a, b, field, n):
    fa = {(r['fn'], r['module']): r['ms_per_round'] for r in a[field]}
    fb = {(r['fn'], r['module']): r['ms_per_round'] for r in b[field]}
    keys = sorted(set(fa) | set(fb), key=lambda k: -abs(fb.get(k, 0) - fa.get(k, 0)))[:n]
    return [(k, fa.get(k, 0.0), fb.get(k, 0.0)) for k in keys]


def delta_md(a, b, n=14):
    L = [f"### Depth delta {a['depth']} -> {b['depth']}", '',
         f"STEP {a['step_ms_per_round']:.2f} -> {b['step_ms_per_round']:.2f} ms/round ({b['step_ms_per_round'] - a['step_ms_per_round']:+.2f}); "
         f"sampled {a['sampled_cpu_ms_per_round']:.2f} -> {b['sampled_cpu_ms_per_round']:.2f} "
         f"({b['sampled_cpu_ms_per_round'] - a['sampled_cpu_ms_per_round']:+.2f}); "
         f"unsampled {a['unsampled_ms_per_round']:.2f} -> {b['unsampled_ms_per_round']:.2f} "
         f"({b['unsampled_ms_per_round'] - a['unsampled_ms_per_round']:+.2f}). Rows sorted by |delta|.", '']
    for field, title in (('first_llama_frame', 'First llama/ggml frame'), ('top_of_stack', 'Top-of-stack function')):
        L += [f'{title} (ms/round):', '', f"| function | module | {a['depth']} | {b['depth']} | delta |", '|---|---|---:|---:|---:|']
        for (f, m), x, y in delta_rows(a, b, field, n):
            L.append(f'| `{short(f)}` | {m} | {x:.3f} | {y:.3f} | {y - x:+.3f} |')
        L.append('')
    sa, sb = a['segment_wall_ms_per_round'], b['segment_wall_ms_per_round']
    L.append('Segment wall ms/round: ' + '; '.join(f"{k} {sa.get(k, 0):.2f} -> {sb.get(k, 0):.2f} ({sb.get(k, 0) - sa.get(k, 0):+.2f})"
                                                    for k in sorted(set(sa) | set(sb))))
    L.append('')
    return L


def baseline_round(path, depth):
    """Profiled GPU0 round (CUDA-trace-only run) at this depth from an attrib-llama budget.json, or None."""
    try:
        for tr in json.loads(Path(path).read_text())['traces']:
            r = tr.get('result') or {}
            if tr['status'] == 'ok' and r.get('kind') == 'decode' and r.get('depth') == depth:
                po = r.get('profiler_overhead') or {}
                return r['round_ms_gpu0'], po.get('unprofiled_ms_per_round')
    except (OSError, ValueError, KeyError):
        pass
    return None


def overhead_md(results, baseline):
    L = ['### Sampling overhead', '',
         '| window | profiled round ms | unprofiled twin ms | total profiler overhead ms | CUDA-trace-only profiled round ms (baseline run) | '
         'baseline twin ms | sampling-only delta ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for t in results:
        po = t.get('profiler_overhead')
        bl = baseline_round(baseline, t['depth']) if baseline else None
        L.append(f"| {t['id']} | {t['round_ms_gpu0']:.2f} | {po['unprofiled_ms_per_round']:.2f} | {po['overhead_ms']:+.2f} | "
                 if po else f"| {t['id']} | {t['round_ms_gpu0']:.2f} | - | - | ")
        L[-1] += (f"{bl[0]:.2f} | {'-' if bl[1] is None else f'{bl[1]:.2f}'} | {t['round_ms_gpu0'] - bl[0]:+.2f} |" if bl else '- | - | - |')
    L += ['', 'Total overhead = profiled GPU0 round minus the same-boot unprofiled twin (inferred rounds from server draft counters; '
          'includes CUPTI tracing + CPU sampling + dwarf unwinding). Sampling-only delta = this run minus the earlier CUDA-trace-only '
          'run (different boot: compare the two twins for drift; the delta is not a controlled A/B).', '']
    return L


def write_report(results, rejects, out_dir, title, baseline=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'host.json').write_text(json.dumps({'schema': 1, 'title': title, 'traces': results,
                                               'rejected': rejects}, indent=1))
    L = [f'# Host-function attribution of STEP: {title}', '',
         'Labels: ms/round = M (CPU samples x measured sample spacing / steady rounds); STEP intervals = M (pack partition, GPU0); '
         'attribution is of CPU time while a thread runs, so blocked time is unsampled; function names are as resolved by nsys '
         '(libcuda is stripped: its frames show as `[libcuda.so.1]`).', '']
    for r in rejects:
        L += [f"### {r['id']}: REJECTED", '', f"Reason: {r['reason']}", '']
    for t in results:
        L += trace_md(t)
    dec = sorted(results, key=lambda t: t['depth'])
    if len(dec) >= 2:
        L += delta_md(dec[0], dec[-1])
    if results:
        L += overhead_md(results, baseline)
    (out / 'host.md').write_text('\n'.join(L) + '\n')


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap_.add_argument('manifest', help='pack manifest.json (gpu/make_manifest.py); decode traces only')
    ap_.add_argument('--out', required=True)
    ap_.add_argument('--baseline', help='attrib-llama budget.json (CUDA-trace-only run) for the sampling-only delta')
    ap_.add_argument('--min-samples', type=int, default=200)
    ap_.add_argument('--classes', default=','.join(STEP_CLASSES), help='GPU0 classes that make up STEP (default = the pack axis)')
    a = ap_.parse_args(argv)
    man = json.loads(Path(a.manifest).read_text())
    model = ap.load_model(man['model'])
    defaults = {'engine': man.get('engine') or model.get('engine'), 'k': man.get('k'), 'min_rounds': man.get('min_rounds', 30)}
    results, rejects = [], []
    for spec in man['traces']:
        if spec.get('kind', 'decode') != 'decode':
            continue
        try:
            tr = ap.Trace(spec['sqlite'], need_memcpy=True, need_runtime=True)
            results.append(host_attribution(tr, model, spec, defaults, tuple(a.classes.split(',')), a.min_samples))
        except Reject as e:
            rejects.append({'id': spec['id'], 'reason': str(e)})
    write_report(results, rejects, a.out, man.get('title', ''), a.baseline)
    for r in rejects:
        print(f"REJECTED {r['id']}: {r['reason']}", file=sys.stderr)
    print(f'{len(results)}/{len(results) + len(rejects)} traces attributed -> {a.out}/host.md')
    return 1 if rejects or not results else 0


if __name__ == '__main__':
    sys.exit(main())
